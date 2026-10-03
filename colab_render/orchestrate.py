#!/usr/bin/env python3
"""
Server-side driver for rendering audiobooks on Colab through the official
Colab CLI (https://github.com/googlecolab/google-colab-cli), or on Kaggle
through the official Kaggle CLI (--backend kaggle).

Runs on your own Linux server (the Colab CLI has no Windows build), needs only
the Python standard library, and talks to Colab exclusively through the
`colab` command. One GPU session is opened per book, every chapter is
rendered in it (the model loads once), each finished chapter is downloaded
immediately, and the session is always stopped at the end, even on errors.

Commands (all print JSON on the last line, so an agent can parse them):

  plan      BOOK --voices V           check the script and voices, no GPU
  render    BOOK --voices V --out DIR render every chapter on a Colab GPU
            ... --backend kaggle          or as one background Kaggle notebook run
  assemble  CHAPTER_DIR               rebuild chapter.wav + stems from lines/
  import-qwen DIR --out SCRIPT        convert old Audio_Ready_Qwen scene files
  usage                               compute-unit balance and burn rate
  watchdog  [--stop]                  find sessions nobody is using; stop them

Book markup (the same tags VoiceStudio uses):

  # Розділ 1. Назва             -> starts a chapter
  [voice:Диктор] Текст ...       -> switches the voice until the next tag
  [voice:Марта] — Репліка.
  [pause 1.5s]                   -> extra pause after the previous line
  (empty line)                   -> paragraph break (longer pause)
  прив+ет                        -> stress mark, see --stress

Voices file: either {"Name": {"ref_audio": "a.wav", "ref_text": "...",
"seed": 7, "speed": 1.0}} or the old Drive library voice_library.json
({"entries": {...}} with reference_audio_path / reference_text).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
import time
import wave
import zipfile
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
RENDER_SCRIPT = os.path.join(HERE, "render_chapter.py")
STATE_DIR = os.path.expanduser(os.environ.get("AUDIOBOOK_STATE_DIR", "~/.cache/audiobook-colab"))
LOCK_PATH = os.path.join(STATE_DIR, "render.lock.json")
REMOTE_JOB = "audiobook_job.zip"  # Colab file API paths are relative to /content
REMOTE_OUT = "audiobook_out.zip"
MAX_LINE_CHARS = 280
DONE_MARKER = "AUDIOBOOK_RENDER_OK"  # printed by render_chapter.py on success

VOICE_TAG = re.compile(r"\[voice:\s*([^\]]+?)\s*\]")
PAUSE_TAG = re.compile(r"\[pause[\s:]*([\d.]+)\s*(ms|s)?\s*\]", re.IGNORECASE)
# Per-line delivery, applies to the text on the same source line only.
SPEED_TAG = re.compile(r"\[speed[\s:]*([\d.]+)\s*\]", re.IGNORECASE)
VOLUME_TAG = re.compile(r"\[volume[\s:]*([\d.]+)\s*\]", re.IGNORECASE)
SENTENCE_END = re.compile(r"(?<=[.!?…])[\"»”)]*\s+")


def emit(obj: Dict[str, Any]) -> None:
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def note(msg: str) -> None:
    print(f"[audiobook] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Script parsing
# ---------------------------------------------------------------------------

def split_long(text: str, limit: int = MAX_LINE_CHARS) -> List[str]:
    """Split on sentence ends, then pack sentences up to `limit` characters."""
    if len(text) <= limit:
        return [text]
    parts = [p.strip() for p in SENTENCE_END.split(text) if p.strip()]
    chunks: List[str] = []
    cur = ""
    for p in parts:
        if cur and len(cur) + 1 + len(p) > limit:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur} {p}".strip()
    if cur:
        chunks.append(cur)
    return chunks


def parse_book(text: str, default_voice: Optional[str] = None) -> List[Dict[str, Any]]:
    chapters: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    voice = default_voice

    def chapter() -> Dict[str, Any]:
        nonlocal current
        if current is None:
            current = {"title": "", "lines": []}
            chapters.append(current)
        return current

    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("# "):
            current = {"title": line[2:].strip(), "lines": []}
            chapters.append(current)
            continue
        if not line:
            if current and current["lines"]:
                current["lines"][-1]["paragraph_end"] = True
            continue
        delivery: Dict[str, float] = {}
        for tag, key in ((SPEED_TAG, "speed"), (VOLUME_TAG, "volume")):
            m = tag.search(line)
            if m:
                delivery[key] = float(m.group(1))
                line = tag.sub("", line).strip()
        # A line may hold several [voice:] / [pause] tags; walk them in order.
        pos = 0
        tokens = sorted(
            [(m.start(), m.end(), "voice", m.group(1)) for m in VOICE_TAG.finditer(line)]
            + [(m.start(), m.end(), "pause", m) for m in PAUSE_TAG.finditer(line)]
        )
        segments: List[tuple] = []
        for start, end, kind, val in tokens:
            if start > pos:
                segments.append(("text", line[pos:start]))
            segments.append((kind, val))
            pos = end
        if pos < len(line):
            segments.append(("text", line[pos:]))
        for kind, val in segments:
            ch = chapter()
            if kind == "voice":
                voice = val.strip()
            elif kind == "pause":
                amount = float(val.group(1))
                ms = int(amount if (val.group(2) or "s").lower() == "ms" else amount * 1000)
                if ch["lines"]:
                    ch["lines"][-1]["pause_after_ms"] = ms
            else:
                body = " ".join(val.split())
                if not body:
                    continue
                if not voice:
                    raise ValueError(f"text before any [voice:...] tag: {body[:60]!r}")
                for piece in split_long(body):
                    ch["lines"].append({"voice": voice, "text": piece, **delivery})
    chapters = [c for c in chapters if c["lines"]]
    for n, c in enumerate(chapters, 1):
        c["index"] = n
        for i, ln in enumerate(c["lines"], 1):
            ln["id"] = f"{i:04d}"
    return chapters


# ---------------------------------------------------------------------------
# Voices
# ---------------------------------------------------------------------------

def load_voices(path: str) -> Dict[str, Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    base = os.path.dirname(os.path.abspath(path))
    voices: Dict[str, Dict[str, Any]] = {}
    if isinstance(data, dict) and isinstance(data.get("entries"), dict):
        # Old Drive library from voice_library_manager.py.
        for key, e in data["entries"].items():
            name = e.get("character_name") or key
            meta = e.get("metadata") or {}
            voices[name] = {
                "ref_audio": e["reference_audio_path"],
                "ref_text": e["reference_text"],
                "seed": int(meta.get("seed", 1234)),
                "speed": float(meta.get("speed", 1.0) or 1.0),
            }
    else:
        for name, v in data.items():
            voices[name] = {
                "ref_audio": v["ref_audio"],
                "ref_text": v["ref_text"],
                "seed": int(v.get("seed", 1234)),
                "speed": float(v.get("speed", 1.0) or 1.0),
            }
    for v in voices.values():
        if not os.path.isabs(v["ref_audio"]):
            v["ref_audio"] = os.path.join(base, v["ref_audio"])
    return voices


def check_voices(chapters: List[Dict[str, Any]], voices: Dict[str, Dict[str, Any]]) -> List[str]:
    problems = []
    used = sorted({ln["voice"] for c in chapters for ln in c["lines"]})
    for name in used:
        v = voices.get(name)
        if v is None:
            problems.append(f"voice '{name}' is used in the script but missing from the voices file")
            continue
        if not os.path.exists(v["ref_audio"]):
            problems.append(f"voice '{name}': reference file not found: {v['ref_audio']}")
            continue
        if not v["ref_text"].strip():
            problems.append(f"voice '{name}': empty reference transcript")
        try:
            with wave.open(v["ref_audio"], "rb") as w:
                secs = w.getnframes() / float(w.getframerate())
                if w.getsampwidth() != 2:
                    problems.append(f"voice '{name}': reference must be 16-bit PCM wav")
            if secs > 12:
                note(f"voice '{name}': reference is {secs:.1f}s; 3-10s clips clone more steadily")
        except wave.Error as e:
            problems.append(f"voice '{name}': not a readable wav ({e})")
    return problems


# ---------------------------------------------------------------------------
# Jobs and output assembly
# ---------------------------------------------------------------------------

def slug(text: str) -> str:
    s = re.sub(r"[^\w-]+", "_", text, flags=re.UNICODE).strip("_")
    return s[:40] or "chapter"


def chapter_dir_name(ch: Dict[str, Any]) -> str:
    return f"{ch['index']:02d}_{slug(ch['title'])}"


# ---------------------------------------------------------------------------
# Pauses between lines
# ---------------------------------------------------------------------------

GAPS_MS = {
    "continue": 220,    # the next line finishes the same sentence
    "line": 550,        # after a full stop
    "question": 650,    # after ? or !
    "ellipsis": 800,    # after …
    "dialogue": 1000,   # into and out of a character's speech
    "speaker_change": 650,
    "paragraph": 1200,
}

_CLOSERS = "\"»”)' "
_SPEECH_OPEN = ("—", "–", "«", "\"", "„")
# "…огнем! — прохрипел Андрей": a dash followed by a verb of speaking.
SPEECH_VERB = re.compile(
    r"[,!?.…»\"]\s*[—–-]\s*(?:\w+\s+){0,2}?"
    r"(?:про|вы|за|от|пере|до|при|вс|у)?"
    r"(?:хрип|шепт|шепн|крикн|крич|сказ|ответ|спрос|бросил|выдохн|выдох|рявкн|рыч|произн|"
    r"повтор|добав|воскликн|ор|заор|буркн|бормот|прохрип|стон|прошипел|шипел|рявк|окликн|позвал)",
    re.IGNORECASE)
# A capitalised gerund or "который" after a full stop still continues the
# sentence ("…в костях." / "Превратившись в корку…"). Gerunds in -я/-а are
# left out: they look too much like nouns and adjectives ("Земля", "Тяжелая").
_GERUND_START = re.compile(r"^(?:[А-ЯЁ][а-яё]+(?:вшись|вши|вшие|ясь|аясь)|Котор(?:ый|ая|ое|ые|ого|ой|ую|ым|ых))\b")


def speech_flags(texts: List[str]) -> List[bool]:
    """True for lines that are (part of) a character's direct speech."""
    flags = [t.lstrip().startswith(_SPEECH_OPEN) or bool(SPEECH_VERB.search(t)) for t in texts]
    # "Пятьдесят третий, я Скиф!" / "Квадрат накрыт! — прохрипел Андрей":
    # exclamations right before an attributed line belong to the same speech.
    for i in range(len(texts) - 1, 0, -1):
        if flags[i] and SPEECH_VERB.search(texts[i]):
            j = i - 1
            while j >= 0 and j >= i - 2 and texts[j].rstrip(_CLOSERS)[-1:] in "!?" and not flags[j]:
                flags[j] = True
                j -= 1
    return flags


def gap_between(cur: str, nxt: str, cur_speech: bool, nxt_speech: bool, gaps: Dict[str, int]) -> int:
    end = cur.rstrip(_CLOSERS)[-1:] if cur.strip() else ""
    first = nxt.lstrip(" —–-«\"„")[:1]
    if cur_speech != nxt_speech:
        return gaps["dialogue"]
    runs_on = end in (",", ";", ":", "—", "–", "-") or end.isalnum()
    if runs_on or (first.islower() and end not in ".!?") or (end in ".…" and _GERUND_START.match(nxt.lstrip())):
        return gaps["continue"]
    if end == "…" or cur.rstrip(_CLOSERS).endswith("..."):
        return gaps["ellipsis"]
    if end in "?!":
        return gaps["question"]
    return gaps["line"]


def assign_gaps(lines: List[Dict[str, Any]], gaps: Optional[Dict[str, int]] = None) -> None:
    """Fill pause_after_ms for every line that has no explicit [pause]."""
    g = dict(GAPS_MS, **(gaps or {}))
    texts = [ln["text"] for ln in lines]
    speech = speech_flags(texts)
    for i, ln in enumerate(lines):
        if ln.get("pause_after_ms") is not None:
            continue
        if i == len(lines) - 1:
            ln["pause_after_ms"] = 0
        elif ln.get("paragraph_end"):
            ln["pause_after_ms"] = g["paragraph"]
        else:
            gap = gap_between(texts[i], texts[i + 1], speech[i], speech[i + 1], g)
            if lines[i + 1]["voice"] != ln["voice"]:
                gap = max(gap, g["speaker_change"])
            ln["pause_after_ms"] = gap


def build_job(ch: Dict[str, Any], voices: Dict[str, Dict[str, Any]], settings: Dict[str, Any], zip_path: str) -> str:
    lines = [dict(ln) for ln in ch["lines"]]
    assign_gaps(lines, settings.get("gap_ms"))
    used = sorted({ln["voice"] for ln in lines})
    job_voices = {}
    files = {}
    for i, name in enumerate(used):
        v = voices[name]
        with open(v["ref_audio"], "rb") as f:
            data = f.read()
        rel = f"voices/{i:02d}.wav"
        files[rel] = data
        job_voices[name] = {
            "ref_audio": rel,
            "ref_text": v["ref_text"],
            "ref_sha256": hashlib.sha256(data).hexdigest(),
            "seed": v["seed"],
            "speed": v["speed"],
        }
    job = {
        "title": ch["title"],
        "engine": settings["engine"],
        "engine_options": settings.get("engine_options", {}),
        "language": settings["language"],
        "stress_mode": settings["stress"],
        "lexicon": settings.get("lexicon", {}),
        "gap_ms": settings.get("gap_ms", {}),
        "voices": job_voices,
        "lines": lines,
    }
    job["job_id"] = hashlib.sha256(json.dumps(job, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("job.json", json.dumps(job, ensure_ascii=False, indent=2))
        for rel, data in files.items():
            z.writestr(rel, data)
    return job["job_id"]


def assemble(chapter_dir: str) -> Dict[str, Any]:
    """Join lines/ into chapter.wav and one aligned stem per voice (stdlib only)."""
    with open(os.path.join(chapter_dir, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    sr = int(manifest["sample_rate"])
    voices = sorted({m["voice"] for m in manifest["lines"]})
    chapter = bytearray()
    stems = {v: bytearray() for v in voices}
    for m in manifest["lines"]:
        with wave.open(os.path.join(chapter_dir, "lines", f"{m['id']}.wav"), "rb") as w:
            pcm = w.readframes(w.getnframes())
        silence_same = b"\x00" * len(pcm)
        gap = b"\x00\x00" * int(sr * int(m.get("gap_after_ms", 0)) / 1000)
        chapter += pcm + gap
        for v in voices:
            stems[v] += (pcm if v == m["voice"] else silence_same) + gap

    def write(path: str, data: bytes) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(bytes(data))

    write(os.path.join(chapter_dir, "chapter.wav"), chapter)
    for v, data in stems.items():
        write(os.path.join(chapter_dir, "stems", f"{slug(v)}.wav"), data)
    mp3 = None
    if shutil.which("ffmpeg"):
        mp3 = os.path.join(chapter_dir, "chapter.mp3")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", os.path.join(chapter_dir, "chapter.wav"),
             "-codec:a", "libmp3lame", "-b:a", "128k", mp3],
            check=False,
        )
    return {
        "chapter_dir": chapter_dir,
        "seconds": len(chapter) / 2 / sr,
        "stems": sorted(os.listdir(os.path.join(chapter_dir, "stems"))),
        "mp3": mp3 if mp3 and os.path.exists(mp3) else None,
    }


# ---------------------------------------------------------------------------
# Colab CLI wrapper
# ---------------------------------------------------------------------------

class ColabError(RuntimeError):
    """A `colab` or `kaggle` command failed."""


class Cli:
    env_var = ""
    default = ""

    def __init__(self, binary: Optional[str] = None):
        self.bin = binary or os.environ.get(self.env_var, self.default)

    def run(self, *args: str, timeout: Optional[float] = None, stream: bool = False) -> str:
        cmd = [self.bin, *args]
        note("$ " + " ".join(cmd))
        if stream:
            out_lines = []
            with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
                assert proc.stdout is not None
                for line in proc.stdout:
                    out_lines.append(line)
                    print(line.rstrip(), file=sys.stderr, flush=True)
                code = proc.wait(timeout=timeout)
            out = "".join(out_lines)
        else:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            code, out = res.returncode, res.stdout + res.stderr
        if code != 0:
            raise ColabError(f"{' '.join(cmd)} failed ({code}):\n{out[-2000:]}")
        return out


class Colab(Cli):
    env_var = "COLAB_BIN"
    default = "colab"

    def usage(self) -> Dict[str, Any]:
        out = self.run("usage")
        info: Dict[str, Any] = {"raw": out.strip()}
        for key, pattern in (
            ("balance_cu", r"balance:\s*([\d.]+)"),
            ("rate_cu_per_hour", r"rate:\s*([\d.]+)"),
            ("active_assignments", r"assignments:\s*([\d.]+)"),
        ):
            m = re.search(pattern, out, re.IGNORECASE)
            if m:
                info[key] = float(m.group(1))
        return info

    def sessions(self) -> List[Dict[str, str]]:
        out = self.run("sessions")
        found = []
        for line in out.splitlines():
            m = re.match(r"\s*\[([^\]]+)\]\s+(\S+)\s*\|\s*Hardware:\s*([^|]+)", line)
            if m:
                found.append({"name": m.group(1), "endpoint": m.group(2), "hardware": m.group(3).strip()})
        return found


def read_text(path: str) -> Optional[str]:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return None


class Kaggle(Cli):
    """Kaggle has no live session: inputs go up as a private dataset, the
    render runs as a background notebook version, outputs come back after."""

    env_var = "KAGGLE_BIN"
    default = "kaggle"

    @staticmethod
    def username(explicit: Optional[str] = None, binary: Optional[str] = None) -> Optional[str]:
        if explicit:
            return explicit
        if os.environ.get("KAGGLE_USERNAME"):
            return os.environ["KAGGLE_USERNAME"]
        cfg_dir = os.environ.get("KAGGLE_CONFIG_DIR", os.path.expanduser("~/.kaggle"))
        try:
            with open(os.path.join(cfg_dir, "kaggle.json"), encoding="utf-8") as f:
                return json.load(f).get("username")
        except (OSError, ValueError):
            pass
        # `kaggle auth login` (OAuth) and access_token logins leave no kaggle.json;
        # the CLI still reports the account in `config view` ("- username: NAME").
        try:
            res = subprocess.run([binary or os.environ.get("KAGGLE_BIN", "kaggle"), "config", "view"],
                                 capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            return None
        m = re.search(r"username:\s*(\S+)", res.stdout)
        return m.group(1) if m and m.group(1).lower() != "none" else None

    def status_word(self, *args: str) -> str:
        """Last word of `... status`, lowercased: complete, running, ready, error, ..."""
        out = self.run(*args, timeout=120).strip().lower()
        m = re.findall(r"[a-z_]+", out.splitlines()[0] if out else "")
        return m[-1] if m else out

    def wait(self, args: List[str], done: set, failed: set, poll: float, max_seconds: float,
             retry_errors: bool = False) -> str:
        deadline = time.time() + max_seconds
        while True:
            try:
                word = self.status_word(*args)
            except ColabError as e:
                # A just-created dataset answers 403/404 for a little while.
                if not retry_errors or time.time() > deadline:
                    raise
                word = f"error: {e}"
            if word in done or word in failed:
                return word
            if time.time() > deadline:
                raise ColabError(f"timed out waiting for {' '.join(args)} (last status: {word})")
            time.sleep(poll)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def read_lock() -> Optional[Dict[str, Any]]:
    try:
        with open(LOCK_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def load_settings(args: argparse.Namespace) -> Dict[str, Any]:
    lexicon = {}
    if args.lexicon:
        with open(args.lexicon, encoding="utf-8") as f:
            lexicon = json.load(f)
    engine_options = json.loads(args.engine_options) if args.engine_options else {}
    return {
        "engine": args.engine,
        "engine_options": engine_options,
        "language": args.language,
        "stress": args.stress,
        "lexicon": lexicon,
    }


def load_book(args: argparse.Namespace):
    with open(args.book, encoding="utf-8") as f:
        chapters = parse_book(f.read(), args.default_voice)
    if args.chapters:
        wanted = {int(x) for x in args.chapters.split(",")}
        chapters = [c for c in chapters if c["index"] in wanted]
    voices = load_voices(args.voices)
    return chapters, voices


QWEN_SCENE_FILE = re.compile(r"chapter_(\d+)_scene(\d+)\.md$")
QWEN_BLOCK = re.compile(r"^Text:\s*(.*?)\s*$(?:\s*^System_Prompt:\s*(.*?)\s*$)?", re.M)

# System_Prompt words -> what the cloned voice can still change: tempo and
# loudness. OmniVoice copies the reference's emotion and ignores the words.
QWEN_SPEED = [
    (re.compile(r"\b(incredibly|extremely|very) slow", re.I), 0.82),
    (re.compile(r"\b(slow|slowly|slower|drawn[- ]out|deliberate|fading|sad|sorrow\w*|grief|"
                r"mournful|crying|tearful|exhausted|weary|tired|dying|weak|solemn|reflective)\b", re.I), 0.9),
    (re.compile(r"\b(very fast|rapid|rushed|frantic|panick?ed|breathless|terrified|hysterical)\b", re.I), 1.15),
    (re.compile(r"\b(fast|quick|quickly|urgent|hurried|tense|shout\w*|yell\w*|scream\w*|"
                r"commanding|barked|aggressive|angry|furious|excited|alarmed|intense|action)\b", re.I), 1.08),
]
QWEN_VOLUME = [
    (re.compile(r"\b(whisper|whispering|whispered|barely audible)\b", re.I), 0.55),
    (re.compile(r"\b(quiet|quietly|soft|softly|hushed|muted)\b", re.I), 0.72),
]


def qwen_delivery(prompt: str) -> str:
    tags = []
    for table, name in ((QWEN_SPEED, "speed"), (QWEN_VOLUME, "volume")):
        for rx, value in table:
            if rx.search(prompt or ""):
                tags.append(f"[{name} {value}]")
                break
    return " ".join(tags)


def _letters(text: str) -> Tuple[str, List[int]]:
    """Lowercase letters only (no stress marks, ё as е) and, for each, its
    index in `text`. Spacing, punctuation and digits differ between the
    prose and the TTS lines; the letters mostly do not."""
    out, idx = [], []
    for i, c in enumerate(text):
        for d in unicodedata.normalize("NFD", c.lower()):
            if d.isalpha():
                out.append("е" if d == "ё" else d)
                idx.append(i)
    return "".join(out), idx


def _speech_mask(prose: str) -> List[bool]:
    """Per character: inside a character's direct speech? A paragraph that
    opens with a dash alternates speech / author's words at each ' — '."""
    mask = [False] * len(prose)
    for m in re.finditer(r"[^\n]+", prose):
        para = m.group(0)
        if not para.lstrip().startswith(("—", "–")):
            continue
        speech = True
        start = base = m.start() + len(para) - len(para.lstrip()) + 1
        for d in re.finditer(r"\s[—–]\s", para[base - m.start():]):
            cut = base + d.start()
            for k in range(start, cut):
                mask[k] = speech
            speech = not speech
            start = cut + 1
        for k in range(start, m.end()):
            mask[k] = speech
    return mask


def prose_pauses(lines: List[str], prose: str, gaps: Optional[Dict[str, int]] = None) -> List[Optional[int]]:
    """Pause after each TTS line, read from the original prose the lines were
    cut from: same sentence -> short, sentence end -> by its punctuation,
    paragraph -> long, into or out of direct speech -> dialogue. None where a
    line could not be found in the prose."""
    return align_to_prose(lines, prose, gaps)[1]


def align_to_prose(lines: List[str], prose: str, gaps: Optional[Dict[str, int]] = None
                   ) -> Tuple[List[Optional[Tuple[int, int]]], List[Optional[int]]]:
    """(span of each line in `prose` or None, pause after each line)."""
    g = dict(GAPS_MS, **(gaps or {}))
    letters, where = _letters(prose)
    mask = _speech_mask(prose)
    spans: List[Optional[Tuple[int, int]]] = []
    pos = 0
    for line in lines:
        key, _ = _letters(line)
        if len(key) < 4:
            spans.append(None)
            continue
        window = letters[pos:pos + max(4000, 3 * len(key))]
        head = window.find(key[:16])
        if head < 0:
            mid = len(key) // 2
            hit = window.find(key[mid:mid + 12])
            head = hit - mid if hit >= 0 else -1
        if head < 0:
            spans.append(None)
            continue
        start = pos + max(head, 0)
        tail_key = key[-12:]
        tail = letters.find(tail_key, start + max(0, len(key) - len(tail_key) - 40), start + len(key) + 60)
        end = tail + len(tail_key) - 1 if tail >= 0 else min(start + len(key) - 1, len(letters) - 1)
        spans.append((where[start], where[end]))
        pos = end + 1
    pauses: List[Optional[int]] = []
    for i, span in enumerate(spans):
        nxt = spans[i + 1] if i + 1 < len(spans) else None
        if span is None or nxt is None or nxt[0] <= span[1]:
            pauses.append(None if i + 1 < len(spans) else 0)
            continue
        between = prose[span[1] + 1:nxt[0]]
        tail = prose[max(span[0], span[1] - 3):span[1] + 1] + between
        if "\n" in between:
            gap = g["paragraph"]
        elif mask[span[1]] != mask[nxt[0]]:
            gap = g["dialogue"]
        elif not re.search(r"[.!?…]", between):
            gap = g["continue"]
        elif "…" in between or "..." in tail:
            gap = g["ellipsis"]
        elif re.search(r"[!?]", between):
            gap = g["question"]
        else:
            gap = g["line"]
        pauses.append(gap)
    return spans, pauses


# One TTS call per "breath": OmniVoice reads a line as a finished utterance,
# so fragments of one sentence, or several short sentences of one paragraph,
# sound choppy when synthesised one by one.
MAX_BREATH_CHARS = 240
_STRESSED_WORD = re.compile(r"\w*\u0301\w*(?:\u0301\w*)*")


def _carry_stress(text: str, prose_part: str) -> str:
    """Copy the author's stress marks (U+0301) from the prose onto the same
    words in the TTS text."""
    for word in set(_STRESSED_WORD.findall(prose_part)):
        plain = word.replace("\u0301", "")
        if not plain:
            continue
        rx = re.compile(r"(?<!\w)" + re.escape(plain) + r"(?!\w)", re.IGNORECASE)
        text = rx.sub(lambda m: m.group(0)[0] + word[1:] if m.group(0)[0] != word[0] else word, text)
    return text


def merge_breaths(items: List[Dict[str, Any]], prose: str, limit: int = MAX_BREATH_CHARS) -> List[Dict[str, Any]]:
    """items: {text, tags, pause, span}. Join a line onto the previous one when
    they are one sentence, or short sentences of one paragraph and one side of
    the dialogue with the same delivery tags, up to `limit` characters."""
    soft = {GAPS_MS["line"], GAPS_MS["question"], GAPS_MS["ellipsis"]}
    out: List[Dict[str, Any]] = []
    for it in items:
        prev = out[-1] if out else None
        if (prev and prev["span"] and it["span"] and len(prev["text"]) + 1 + len(it["text"]) <= limit
                and (prev["pause"] == GAPS_MS["continue"]
                     or (prev["pause"] in soft and prev["tags"] == it["tags"]))):
            text = it["text"]
            if prev["pause"] == GAPS_MS["continue"]:
                # "…в костях." + "Выбивая…" was cut from "…в костях, выбивая…"
                prev["text"] = re.sub(r"(\.\.\.|…|\.)$", ",", prev["text"].rstrip())
                if prose[it["span"][0]].islower():
                    text = text[0].lower() + text[1:]
            prev["text"] = f"{prev['text']} {text}"
            prev["pause"] = it["pause"]
            prev["span"] = (prev["span"][0], it["span"][1])
        else:
            out.append(dict(it))
    for it in out:
        if it["span"]:
            it["text"] = _carry_stress(it["text"], prose[it["span"][0]:it["span"][1] + 1])
    return out


def convert_qwen_scenes(src_dir: str, voice: str, prose_dir: Optional[str] = None) -> str:
    """Old Audio_Ready_Qwen exports: one file per scene, blocks of ID / Text /
    System_Prompt separated by '---'. Each scene becomes one chapter of the
    script, so a test can render a single scene. System_Prompt becomes
    [speed]/[volume] tags; the emotion itself comes from the reference."""
    files = []
    for name in os.listdir(src_dir):
        m = QWEN_SCENE_FILE.search(name)
        if m:
            files.append((int(m.group(1)), int(m.group(2)), name))
    if not files:
        raise FileNotFoundError(f"no *chapter_NN_sceneK.md files in {src_dir}")
    out = []
    for ch, sc, name in sorted(files):
        blocks = [(t.strip(), p) for t, p in QWEN_BLOCK.findall(read_text(os.path.join(src_dir, name)) or "")
                  if t.strip()]
        if not blocks:
            continue
        items = [{"text": t, "tags": qwen_delivery(p), "pause": None, "span": None} for t, p in blocks]
        prose = read_text(os.path.join(prose_dir, name)) if prose_dir else None
        if prose:
            spans, pauses = align_to_prose([t for t, _ in blocks], prose)
            for it, span, pause in zip(items, spans, pauses):
                it["span"], it["pause"] = span, pause
            items = merge_breaths(items, prose)
        out.append(f"# Глава {ch:02d}, сцена {sc}")
        out.append(f"[voice:{voice}]")
        for it in items:
            line = f"{it['tags']} {it['text']}" if it["tags"] else it["text"]
            out.append(line if it["pause"] is None else f"{line} [pause {it['pause']}ms]")
        out.append("")
    return "\n".join(out)


def cmd_import_qwen(args: argparse.Namespace) -> int:
    script = convert_qwen_scenes(args.src, args.voice, args.prose)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(script)
    chapters = parse_book(script)
    emit({"ok": True, "out": args.out, "chapters": len(chapters),
          "lines": sum(len(c["lines"]) for c in chapters),
          "chars": sum(len(ln["text"]) for c in chapters for ln in c["lines"])})
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    chapters, voices = load_book(args)
    problems = check_voices(chapters, voices)
    chars = sum(len(ln["text"]) for c in chapters for ln in c["lines"])
    if args.gaps:
        for c in chapters:
            lines = [dict(ln) for ln in c["lines"]]
            assign_gaps(lines)
            for ln in lines:
                print(f"{c['index']:>3} {ln['id']} {ln['pause_after_ms']:>5}  {ln['text'][:70]}")
    emit({
        "ok": not problems,
        "problems": problems,
        "chapters": [
            {"index": c["index"], "title": c["title"], "lines": len(c["lines"]),
             "chars": sum(len(ln["text"]) for ln in c["lines"]),
             "voices": sorted({ln["voice"] for ln in c["lines"]})}
            for c in chapters
        ],
        "total_chars": chars,
        # ~11 characters per second (measured with OmniVoice on Russian prose),
        # plus the pauses; a rough figure for planning only.
        "estimated_audio_minutes": round(chars / 11 / 60, 1),
    })
    return 0 if not problems else 2


def finish_chapter(ch: Dict[str, Any], ch_dir: str, out_zip: str, job_id: str, started: float) -> Dict[str, Any]:
    if os.path.isdir(ch_dir):
        shutil.rmtree(ch_dir)
    with zipfile.ZipFile(out_zip) as z:
        z.extractall(ch_dir)
    info = assemble(ch_dir)
    with open(os.path.join(ch_dir, ".job_id"), "w") as f:
        f.write(job_id)
    info.update({"index": ch["index"], "title": ch["title"], "wall_seconds": round(time.time() - started, 1)})
    note(f"chapter {ch['index']} done: {info['seconds'] / 60:.1f} min of audio")
    return info


def cmd_render(args: argparse.Namespace) -> int:
    chapters, voices = load_book(args)
    problems = check_voices(chapters, voices)
    if args.backend == "kaggle" and not Kaggle.username(args.kaggle_user, args.kaggle_bin):
        problems.append("Kaggle account unknown: run `kaggle auth login` or pass --kaggle-user")
    if problems:
        emit({"ok": False, "problems": problems})
        return 2
    settings = load_settings(args)
    os.makedirs(args.out, exist_ok=True)

    lock = read_lock()
    if lock and _pid_alive(int(lock.get("pid", -1))):
        emit({"ok": False, "problems": [f"another render is running (pid {lock['pid']}, {lock.get('session')})"]})
        return 3

    todo = []
    for ch in chapters:
        ch_dir = os.path.join(args.out, chapter_dir_name(ch))
        tmp_zip = os.path.join(tempfile.gettempdir(), f"audiobook_job_{os.getpid()}_{ch['index']}.zip")
        job_id = build_job(ch, voices, settings, tmp_zip)
        done_marker = os.path.join(ch_dir, ".job_id")
        if not args.force and read_text(done_marker) == job_id:
            note(f"chapter {ch['index']} already rendered, skipping")
            os.remove(tmp_zip)
            continue
        todo.append((ch, ch_dir, tmp_zip, job_id))

    if not todo:
        emit({"ok": True, "rendered": [], "note": "nothing to do"})
        return 0

    os.makedirs(STATE_DIR, exist_ok=True)
    try:
        if args.backend == "kaggle":
            return _render_kaggle(args, todo)
        return _render_colab(args, todo)
    finally:
        for _, _, tmp_zip, _ in todo:
            if os.path.exists(tmp_zip):
                os.remove(tmp_zip)
        try:
            os.remove(LOCK_PATH)
        except OSError:
            pass


def _write_lock(session: str, args: argparse.Namespace) -> None:
    with open(LOCK_PATH, "w", encoding="utf-8") as f:
        json.dump({"pid": os.getpid(), "session": session, "backend": args.backend,
                   "started": time.time(), "book": args.book}, f)


def _render_colab(args: argparse.Namespace, todo: List[tuple]) -> int:
    colab = Colab(args.colab_bin)
    results = []
    session = args.session or f"audiobook-{time.strftime('%m%d-%H%M%S')}"
    _write_lock(session, args)
    created = False
    try:
        new_args = ["new", "-s", session]
        if args.gpu:
            new_args += ["--gpu", args.gpu]
        colab.run(*new_args, timeout=600)
        created = True
        for ch, ch_dir, tmp_zip, job_id in todo:
            note(f"chapter {ch['index']}: {ch['title'] or '(untitled)'} ({len(ch['lines'])} lines)")
            colab.run("upload", "-s", session, tmp_zip, REMOTE_JOB, timeout=600)
            started = time.time()
            exec_out = colab.run(
                "exec", "-s", session, "-f", RENDER_SCRIPT,
                "--timeout", str(args.chapter_timeout),
                "--env", f"AUDIOBOOK_JOB=/content/{REMOTE_JOB}",
                "--env", f"AUDIOBOOK_OUT=/content/{REMOTE_OUT}",
                timeout=args.chapter_timeout + 120, stream=True,
            )
            if f"{DONE_MARKER} {job_id}" not in exec_out:
                # `colab exec` returns 0 even when the script raised.
                raise ColabError(f"chapter {ch['index']} did not finish on the VM:\n{exec_out[-2000:]}")
            out_zip = tmp_zip.replace("_job_", "_out_")
            colab.run("download", "-s", session, REMOTE_OUT, out_zip, timeout=1800)
            results.append(finish_chapter(ch, ch_dir, out_zip, job_id, started))
            os.remove(out_zip)
    except (ColabError, subprocess.TimeoutExpired, zipfile.BadZipFile, OSError) as e:
        emit({"ok": False, "rendered": results, "error": str(e)[-1500:]})
        return 1
    finally:
        if created and not args.keep:
            try:
                colab.run("stop", "-s", session, timeout=300)
            except (ColabError, subprocess.TimeoutExpired) as e:
                note(f"WARNING: could not stop session {session}: {e}. Run `orchestrate.py watchdog --stop`.")
    emit({"ok": True, "backend": "colab", "session": session, "kept": bool(args.keep), "rendered": results})
    return 0


KAGGLE_HEADER = """# Generated by orchestrate.py: render_chapter.py in batch mode for Kaggle.
import os
os.environ.setdefault("AUDIOBOOK_BATCH_GLOB", "/kaggle/input/**/job_*")
os.environ.setdefault("AUDIOBOOK_BATCH_OUT", "/kaggle/working")
"""


def _render_kaggle(args: argparse.Namespace, todo: List[tuple]) -> int:
    kaggle = Kaggle(args.kaggle_bin)
    user = Kaggle.username(args.kaggle_user, args.kaggle_bin)
    dataset_id = f"{user}/{args.kaggle_dataset}"
    kernel_id = f"{user}/{args.kaggle_kernel}"
    _write_lock(kernel_id, args)
    work = tempfile.mkdtemp(prefix="audiobook_kaggle_")
    results: List[Dict[str, Any]] = []
    started = time.time()
    try:
        # 1. Inputs: every chapter's job archive in one private dataset version.
        ds_dir = os.path.join(work, "dataset")
        os.makedirs(ds_dir)
        for ch, _, tmp_zip, _ in todo:
            # Not .zip: Kaggle unpacks zips, and a lone one lands in the dataset root.
            shutil.copy(tmp_zip, os.path.join(ds_dir, f"job_{ch['index']:03d}.job"))
        with open(os.path.join(ds_dir, "dataset-metadata.json"), "w", encoding="utf-8") as f:
            json.dump({"title": args.kaggle_dataset, "id": dataset_id, "licenses": [{"name": "CC0-1.0"}]}, f)
        try:
            kaggle.run("datasets", "status", dataset_id, timeout=120)
            exists = True
        except ColabError:
            exists = False
        created_ds = False
        if exists:
            try:
                kaggle.run("datasets", "version", "-p", ds_dir, "-m",
                           f"audiobook jobs {time.strftime('%Y-%m-%d %H:%M')}", "-q", timeout=1800)
                created_ds = True
            except ColabError as e:
                note(f"dataset version failed, trying create: {e}")
        if not created_ds:
            kaggle.run("datasets", "create", "-p", ds_dir, "-q", timeout=1800)
        kaggle.wait(["datasets", "status", dataset_id], {"ready"}, {"failed", "deleted"},
                    min(args.poll_seconds, 15.0), 30 * 60, retry_errors=True)

        # 2. The render: a private script notebook with GPU and internet (for pip + model download).
        k_dir = os.path.join(work, "kernel")
        os.makedirs(k_dir)
        with open(RENDER_SCRIPT, encoding="utf-8") as f:
            source = f.read()
        with open(os.path.join(k_dir, "render.py"), "w", encoding="utf-8") as f:
            f.write(KAGGLE_HEADER + source)
        with open(os.path.join(k_dir, "kernel-metadata.json"), "w", encoding="utf-8") as f:
            json.dump({
                "id": kernel_id, "title": args.kaggle_kernel, "code_file": "render.py",
                "language": "python", "kernel_type": "script", "is_private": True,
                "enable_gpu": True, "enable_internet": True, "dataset_sources": [dataset_id],
            }, f)
        kaggle.run("kernels", "push", "-p", k_dir, "--accelerator", args.kaggle_accelerator,
                   "-t", str(args.kaggle_max_hours * 3600), timeout=600)
        note(f"rendering on Kaggle: https://www.kaggle.com/code/{kernel_id}")
        final = kaggle.wait(["kernels", "status", kernel_id], {"complete"},
                            {"error", "cancel_requested", "cancel_acknowledged"},
                            args.poll_seconds, args.kaggle_max_hours * 3600 + 1800)

        # 3. Outputs (also after an error: finished chapters are kept).
        out_dir = os.path.join(work, "output")
        os.makedirs(out_dir)
        kaggle.run("kernels", "output", kernel_id, "-p", out_dir, "-o", timeout=3600)
        # Surface the notebook's reference check (what Whisper heard) here too.
        for log_name in sorted(n for n in os.listdir(out_dir) if n.endswith(".log")):
            for line in (read_text(os.path.join(out_dir, log_name)) or "").splitlines():
                if "reference '" in line:
                    note(line.strip())
        try:
            with open(os.path.join(out_dir, "batch_report.json"), encoding="utf-8") as f:
                report = {r["job"]: r for r in json.load(f)}
        except (OSError, ValueError):
            report = {}
        failures = []
        # A run that stopped early (e.g. no internet) leaves later jobs without
        # an entry; give them the same reason instead of "no output".
        fatal = next((r["error"] for r in report.values()
                      if str(r.get("error", "")).startswith("NoInternet")), None)
        for ch, ch_dir, _, job_id in todo:
            name = f"job_{ch['index']:03d}.zip"
            entry = report.get(name)
            out_zip = os.path.join(out_dir, f"out_{ch['index']:03d}.zip")
            if not entry or not entry.get("ok") or not os.path.exists(out_zip):
                failures.append({"index": ch["index"], "error": (entry or {}).get("error", fatal or "no output")})
            elif entry.get("job_id") != job_id:
                # The notebook saw an older dataset version.
                failures.append({"index": ch["index"], "error": "stale input on Kaggle, run again"})
            else:
                results.append(finish_chapter(ch, ch_dir, out_zip, job_id, started))
        if final != "complete" or failures:
            log_tail = ""
            for name in os.listdir(out_dir):
                if name.endswith(".log"):
                    log_tail = read_text(os.path.join(out_dir, name)) or ""
            emit({"ok": False, "backend": "kaggle", "kernel": kernel_id, "status": final,
                  "rendered": results, "failed": failures, "log_tail": log_tail[-1500:]})
            return 1
    except (ColabError, subprocess.TimeoutExpired, zipfile.BadZipFile, OSError) as e:
        emit({"ok": False, "backend": "kaggle", "rendered": results, "error": str(e)[-1500:]})
        return 1
    finally:
        shutil.rmtree(work, ignore_errors=True)
    emit({"ok": True, "backend": "kaggle", "kernel": kernel_id, "rendered": results})
    return 0


def cmd_assemble(args: argparse.Namespace) -> int:
    emit({"ok": True, **assemble(args.chapter_dir)})
    return 0


def cmd_usage(args: argparse.Namespace) -> int:
    emit({"ok": True, **Colab(args.colab_bin).usage()})
    return 0


def cmd_watchdog(args: argparse.Namespace) -> int:
    colab = Colab(args.colab_bin)
    lock = read_lock()
    busy = bool(lock and _pid_alive(int(lock.get("pid", -1))))
    if lock and not busy:
        os.remove(LOCK_PATH)
    sessions = colab.sessions()
    active = lock.get("session") if busy and lock else None
    idle = [s for s in sessions if s["name"] != active]
    overdue = bool(busy and lock and time.time() - float(lock.get("started", 0)) > args.max_hours * 3600)
    stopped, failed = [], []
    if args.stop:
        for s in idle:
            if s["name"] == "?":
                failed.append({**s, "reason": "orphan with no local name; stop it from the Colab UI"})
                continue
            try:
                colab.run("stop", "-s", s["name"], timeout=300)
                stopped.append(s["name"])
            except (ColabError, subprocess.TimeoutExpired) as e:
                failed.append({**s, "reason": str(e)[-300:]})
    try:
        usage = colab.usage()
    except ColabError as e:
        usage = {"error": str(e)[-300:]}
    emit({
        "ok": not failed,
        "render_running": busy,
        "render_session": active,
        "render_overdue": overdue,
        "idle_sessions": idle,
        "stopped": stopped,
        "failed": failed,
        "usage": usage,
    })
    return 0 if not failed else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--colab-bin", default=None, help="path to the colab CLI (default: $COLAB_BIN or 'colab')")
    sub = p.add_subparsers(dest="cmd", required=True)

    def book_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("book", help="book script with # chapters and [voice:NAME] tags")
        sp.add_argument("--voices", required=True, help="voices.json or old voice_library.json")
        sp.add_argument("--default-voice", default=None, help="voice for text before the first tag")
        sp.add_argument("--chapters", default=None, help="only these chapter numbers, e.g. 1,2,5")

    sp = sub.add_parser("plan", help="validate script and voices, no GPU")
    book_args(sp)
    sp.add_argument("--gaps", action="store_true", help="also list each line with its pause after it")
    sp.set_defaults(func=cmd_plan)

    sp = sub.add_parser("render", help="render chapters on a Colab GPU")
    book_args(sp)
    sp.add_argument("--out", required=True, help="output directory")
    sp.add_argument("--engine", default="omnivoice", choices=["omnivoice", "voxcpm2", "dummy"])
    sp.add_argument("--engine-options", default=None, help='JSON passed to the engine, e.g. {"num_step": 32}')
    sp.add_argument("--language", default="uk", help="uk, ru, ... (OmniVoice uses it; VoxCPM2 ignores it)")
    sp.add_argument("--stress", default="strip", choices=["strip", "acute", "keep"],
                    help="what to do with '+' stress marks: strip, turn into U+0301, or keep")
    sp.add_argument("--lexicon", default=None, help="JSON {word: respelling}")
    sp.add_argument("--gpu", default="L4", help="T4, L4, A100, H100 or '' for CPU")
    sp.add_argument("--session", default=None, help="Colab session name")
    sp.add_argument("--keep", action="store_true", help="leave the session running afterwards (it keeps billing)")
    sp.add_argument("--force", action="store_true", help="re-render chapters that are already done")
    sp.add_argument("--chapter-timeout", type=int, default=6 * 3600, help="seconds per chapter (Colab)")
    sp.add_argument("--backend", default="colab", choices=["colab", "kaggle"],
                    help="colab: live session via the Colab CLI; kaggle: background notebook via the Kaggle API")
    sp.add_argument("--kaggle-bin", default=None, help="path to the kaggle CLI (default: $KAGGLE_BIN or 'kaggle')")
    sp.add_argument("--kaggle-user", default=None, help="Kaggle username (default: from the kaggle CLI login)")
    sp.add_argument("--kaggle-dataset", default="audiobook-jobs", help="private dataset slug for the inputs")
    sp.add_argument("--kaggle-kernel", default="audiobook-render", help="private notebook slug for the render")
    sp.add_argument("--kaggle-accelerator", default="NvidiaTeslaT4",
                    help="NvidiaTeslaT4 (T4 x2) or NvidiaL4; avoid P100, current PyTorch no longer supports it")
    sp.add_argument("--kaggle-max-hours", type=int, default=11, help="Kaggle caps a run at 12 hours")
    sp.add_argument("--poll-seconds", type=float, default=60.0, help="how often to check Kaggle status")
    sp.set_defaults(func=cmd_render)

    sp = sub.add_parser("assemble", help="rebuild chapter.wav and stems from lines/")
    sp.add_argument("chapter_dir")
    sp.set_defaults(func=cmd_assemble)

    sp = sub.add_parser("import-qwen", help="convert Audio_Ready_Qwen scene files into a script")
    sp.add_argument("src", help="folder with *_chapter_NN_sceneK.md files")
    sp.add_argument("--out", required=True, help="script file to write")
    sp.add_argument("--voice", default="Диктор", help="voice for every line")
    sp.add_argument("--prose", default=None,
                    help="folder with the original prose under the same file names; pauses follow its "
                         "sentences, paragraphs and dialogue")
    sp.set_defaults(func=cmd_import_qwen)

    sp = sub.add_parser("usage", help="compute units balance and burn rate")
    sp.set_defaults(func=cmd_usage)

    sp = sub.add_parser("watchdog", help="report (and with --stop, end) sessions nobody is using")
    sp.add_argument("--stop", action="store_true")
    sp.add_argument("--max-hours", type=float, default=8.0, help="flag a render running longer than this")
    sp.set_defaults(func=cmd_watchdog)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, KeyError, FileNotFoundError, json.JSONDecodeError, ColabError) as e:
        emit({"ok": False, "problems": [f"{type(e).__name__}: {e}"]})
        return 2


if __name__ == "__main__":
    sys.exit(main())
