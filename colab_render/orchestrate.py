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
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
EMOTION_TAG = re.compile(r"\[emotion[\s:]*([\w-]+)\s*\]", re.IGNORECASE)
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
        m = EMOTION_TAG.search(line)
        if m:
            delivery["emotion"] = m.group(1).lower()
            line = EMOTION_TAG.sub("", line).strip()
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


def resolve_emotions(chapters: List[Dict[str, Any]], voices: Dict[str, Dict[str, Any]]) -> None:
    """[emotion:X] picks the voice "Name:X" from the cast when it exists. That
    reference already carries the loudness and pace, so the line's volume and
    speed tags go."""
    for c in chapters:
        for ln in c["lines"]:
            emo = ln.pop("emotion", None)
            if emo and f"{ln['voice']}:{emo}" in voices:
                ln["voice"] = f"{ln['voice']}:{emo}"
                # Its pace too: a [speed] on top stacked up to 1.33 for the narrator.
                ln.pop("volume", None)
                ln.pop("speed", None)


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
    "attribution": 350,  # "— К бою! | — заорал Андрей": speech, then the author's words
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
    vc_lines = settings.get("vc_lines", "none")
    if vc_lines != "none":
        # Emotion lines get converted to the calm voice's timbre, so ship it.
        used = sorted(set(used) | {v.split(":")[0] for v in used if ":" in v and v.split(":")[0] in voices})
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
        **({"vc_lines": vc_lines} if vc_lines != "none" else {}),
        **({"max_synth_chars": settings["max_synth_chars"]} if settings.get("max_synth_chars") is not None else {}),
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
    # One stem per character: "Андрей:shout" goes on Андрей's track.
    for m in manifest["lines"]:
        m["voice"] = m["voice"].split(":")[0]
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
            except subprocess.TimeoutExpired:
                # The Kaggle API sometimes hangs for minutes, then answers in a
                # second; a hung status call is "not known yet", not a failure.
                if time.time() > deadline:
                    raise
                word = "timeout"
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
        "vc_lines": getattr(args, "vc_lines", "none"),
        "max_synth_chars": getattr(args, "max_synth_chars", None),
    }


def load_book(args: argparse.Namespace):
    with open(args.book, encoding="utf-8") as f:
        chapters = parse_book(f.read(), args.default_voice)
    if args.chapters:
        wanted = {int(x) for x in args.chapters.split(",")}
        chapters = [c for c in chapters if c["index"] in wanted]
    voices = load_voices(args.voices)
    resolve_emotions(chapters, voices)
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


# System_Prompt -> which of the character's emotion references to use
# (see `cast`). First match wins; no match means the calm voice.
QWEN_EMOTION = [
    (re.compile(r"\b(whisper\w*|barely audible|hushed)\b", re.I), "whisper"),
    (re.compile(r"\b(shout\w*|scream\w*|yell\w*|roar\w*|bellow\w*|battle cry|very loud|commanding)\b", re.I),
     "shout"),
    (re.compile(r"\b(sad\w*|sorrow\w*|grief|griev\w*|mournful|crying|tearful|despair\w*|broken)\b", re.I), "sad"),
    (re.compile(r"\b(tense|tension|anxious|nervous|fear\w*|afraid|panic\w*|terrified|urgent|alarm\w*|"
                r"angry|anger|furious|rage|menacing|threatening)\b", re.I), "tense"),
]


def qwen_delivery(prompt: str) -> str:
    tags = []
    for rx, emo in QWEN_EMOTION:
        if rx.search(prompt or ""):
            tags.append(f"[emotion {emo}]")
            break
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
                and prev.get("voice") == it.get("voice")
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


# ---------------------------------------------------------------------------
# Who speaks: direct speech in the prose, attributed to characters
# ---------------------------------------------------------------------------

def speech_runs(prose: str) -> List[Tuple[int, int]]:
    """(start, end) of every stretch of direct speech, in order. Their
    numbers (1, 2, ...) are what speakers.json refers to."""
    mask = _speech_mask(prose)
    runs: List[Tuple[int, int]] = []
    i = 0
    while i < len(prose):
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < len(prose) and mask[j]:
            j += 1
        a, b = i, j - 1
        while a <= b and not prose[a].isalnum():
            a += 1
        while b >= a and prose[b] in " \t—–-":
            b -= 1
        if a <= b and any(c.isalpha() for c in prose[a:b + 1]):
            runs.append((a, b))
        i = j
    return runs


def _name_forms(cast: Dict[str, List[str]]) -> List[Tuple[re.Pattern, str]]:
    """'Андрей' also matches 'Андрея', 'Андрею'... (a name plus up to two letters)."""
    out = []
    for name, aliases in cast.items():
        for alias in [name, *aliases]:
            stem = alias[:-1] if len(alias) > 4 and alias[-1] in "аяйьео" else alias
            out.append((re.compile(r"(?<!\w)" + re.escape(stem) + r"\w{0,2}(?!\w)", re.IGNORECASE), name))
    return out


_PRONOUN = re.compile(r"(?i)^(он|она|парень|мужчина|старик|воин|женщина|девушка|старуха|знахарка|целительница|"
                      r"травница|хозяйка)\b")
_FEMALE_WORDS = {"она", "женщина", "девушка", "старуха", "знахарка", "целительница", "травница", "хозяйка"}


def guess_speakers(prose: str, cast: Dict[str, List[str]],
                   genders: Optional[Dict[str, str]] = None) -> List[Optional[str]]:
    """Without an LLM, in this order:
    1. a speaking verb after the speech, then a cast name ("— заорал Андрей");
    2. the verb, then "он"/"она"/"женщина"...: the character of that gender
       named most recently before the speech;
    3. the rest of a paragraph goes to whoever spoke in it already;
    4. two characters taking turns: A, B, ? -> A.
    Whatever is still unknown stays with the narrator."""
    names = _name_forms(cast)
    genders = genders or {}
    runs = speech_runs(prose)
    out: List[Optional[str]] = [None] * len(runs)

    def mentions(upto: int) -> List[Tuple[int, str]]:
        found = []
        text = prose[:upto]
        for rx, name in names:
            for m in rx.finditer(text):
                found.append((m.start(), name))
        return sorted(found)

    for i, (a, b) in enumerate(runs):
        after = prose[b + 1:b + 80].split("\n")[0]
        if not SPEECH_VERB.search(prose[max(0, b - 2):b + 1] + after):
            continue
        tail = re.sub(r"^[\s,!?.…»\"—–-]*", "", after)
        verb_and_more = re.match(r"(\w+)[\s,]+(.*)", tail)
        words = re.match(r"((?:\w+[\s,]+){0,4}\w+)", tail)
        for rx, name in names:
            if words and rx.search(words.group(1)):
                out[i] = name
                break
        if out[i] or not verb_and_more:
            continue
        rest = verb_and_more.group(2)
        pron = _PRONOUN.match(rest) or _PRONOUN.match(re.sub(r"^\w+\s+", "", rest, count=1))  # "тихо сказала она"
        if not pron:
            continue
        want = "f" if pron.group(1).lower() in _FEMALE_WORDS else "m"
        para_start = prose.rfind("\n", 0, a) + 1
        for _, name in reversed(mentions(para_start)):
            if genders.get(name, "m") == want:
                out[i] = name
                break

    line_of = [prose.count("\n", 0, a) for a, _ in runs]
    for _ in range(2):
        for i in range(len(runs)):
            if out[i]:
                continue
            same = [out[j] for j in range(len(runs)) if line_of[j] == line_of[i] and out[j]]
            if same:
                out[i] = same[0]
    for i in range(2, len(runs)):
        if not out[i] and out[i - 1] and out[i - 2] and out[i - 1] != out[i - 2]:
            out[i] = out[i - 2]
    return out


def cast_genders(path: str) -> Dict[str, str]:
    """'f' or 'm' per character: an explicit "gender", else read from the description."""
    with open(path, encoding="utf-8") as f:
        spec = json.load(f)
    out = {}
    for n, c in spec["characters"].items():
        g = (c.get("gender") or "").lower()[:1]
        if g not in ("f", "m"):
            g = "f" if re.search(r"(?i)\b(female|woman|girl|lady)\b", c.get("description", "")) else "m"
        out[n] = g
    return out


def llm_speakers(prose: str, cast: Dict[str, List[str]], url: str, model: str, key: Optional[str],
                 timeout: float = 300.0) -> List[Optional[str]]:
    """Ask an OpenAI-compatible chat endpoint (e.g. CLIProxy) who says each
    numbered stretch of speech. Only cast names come back; anything else is None."""
    import urllib.request

    runs = speech_runs(prose)
    if not runs:
        return []
    marked, pos = [], 0
    for n, (a, b) in enumerate(runs, 1):
        marked.append(prose[pos:a] + f"[S{n}]" + prose[a:b + 1] + f"[/S{n}]")
        pos = b + 1
    marked.append(prose[pos:])
    cast_lines = "\n".join(f"- {n}" + (f" (also: {', '.join(a)})" if a else "") for n, a in cast.items())
    prompt = (
        "Below is a scene from a Russian novel. Every stretch of direct speech is wrapped in "
        "[Sn]...[/Sn]. For each n, decide which character says it, using the author's words, "
        "pronouns and the conversation flow. Characters:\n" + cast_lines +
        "\nAnswer with JSON only, like {\"S1\": \"Андрей\", \"S2\": \"other\"}. Use exactly a name from "
        "the list, or \"other\" for anyone else (a minor or unnamed character).\n\n" + "".join(marked))
    body = json.dumps({"model": model, "temperature": 0,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {key}"} if key else {})})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        answer = json.load(resp)["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", answer, re.S)
    data = json.loads(m.group(0)) if m else {}
    return [data.get(f"S{n}") if data.get(f"S{n}") in cast else None for n in range(1, len(runs) + 1)]


def load_cast_names(path: str) -> Dict[str, List[str]]:
    """{name: [aliases]} from a cast file; the narrator ("voice" entries) is not a speaker."""
    with open(path, encoding="utf-8") as f:
        spec = json.load(f)
    return {n: list(c.get("aliases", [])) for n, c in spec["characters"].items() if not c.get("narrator")}


def cmd_attribute(args: argparse.Namespace) -> int:
    cast = load_cast_names(args.cast)
    genders = cast_genders(args.cast)
    key = os.environ.get(args.llm_key_env) if args.llm_key_env else None
    result: Dict[str, Any] = {}
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            result = json.load(f)
    names = sorted(n for n in os.listdir(args.prose) if QWEN_SCENE_FILE.search(n))
    if args.only:
        names = [n for n in names if re.search(args.only, n)]
    stats = {"runs": 0, "named": 0, "llm_errors": 0}
    for name in names:
        prose = read_text(os.path.join(args.prose, name)) or ""
        runs = speech_runs(prose)
        guessed = guess_speakers(prose, cast, genders)
        asked: List[Optional[str]] = [None] * len(runs)
        if args.llm_url and runs:
            try:
                asked = llm_speakers(prose, cast, args.llm_url, args.llm_model, key)
            except Exception as e:  # keep going: the guesses still help
                stats["llm_errors"] += 1
                note(f"{name}: LLM failed ({e}); using the name-after-verb guesses only")
        entries = []
        for n, ((a, b), g, l) in enumerate(zip(runs, guessed, asked), 1):
            who = l or g
            entries.append({"n": n, "text": prose[a:b + 1][:120], "speaker": who,
                            "how": "llm" if l else ("verb+name" if g else None)})
            stats["runs"] += 1
            stats["named"] += bool(who)
        result[name] = entries
        note(f"{name}: {sum(1 for e in entries if e['speaker'])}/{len(entries)} speech stretches attributed")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    emit({"ok": True, "out": args.out, "scenes": len(names), **stats})
    return 0


def split_speech(items: List[Dict[str, Any]], prose: str) -> List[Dict[str, Any]]:
    """A TTS line can hold speech and the author's words ("— К бою! — заорал
    Андрей."). Cut it where the prose switches, at the dash, so each side can
    get its own voice. The author's side drops the line's delivery tags: the
    System_Prompt described the speech."""
    mask = _speech_mask(prose)
    out: List[Dict[str, Any]] = []
    for it in items:
        span = it["span"]
        if not span:
            out.append(it)
            continue
        cuts = [k for k in range(span[0] + 1, span[1] + 1) if mask[k] != mask[k - 1] and prose[k - 1:k + 1].strip()]
        # A cut is where the prose mask flips; find the matching dash in the TTS text.
        dashes = [m.start() for m in re.finditer(r"\s[—–]\s", it["text"])]
        if not cuts or not dashes:
            it["speech"] = mask[span[0]]
            out.append(it)
            continue
        pieces, start = [], 0
        for d in dashes[:len(cuts)]:
            pieces.append(it["text"][start:d])
            start = d + 1
        pieces.append(it["text"][start:])
        bounds = [span[0], *cuts, span[1] + 1]
        speech = mask[span[0]]
        for i, piece in enumerate(pieces):
            piece = piece.strip().lstrip("—– ").strip()
            if not piece:
                speech = not speech
                continue
            a = bounds[min(i, len(bounds) - 2)]
            b = bounds[min(i + 1, len(bounds) - 1)] - 1
            last = i == len(pieces) - 1
            out.append({"text": piece, "tags": it["tags"] if speech else "", "span": (a, b), "speech": speech,
                        "pause": it["pause"] if last else GAPS_MS["attribution"]})
            speech = not speech
    return out


def assign_speakers(items: List[Dict[str, Any]], prose: str, entries: List[Dict[str, Any]],
                    narrator: str, voices: Optional[Dict[str, Any]]) -> Dict[str, int]:
    runs = speech_runs(prose)
    counts: Dict[str, int] = {}
    for it in items:
        it["voice"] = narrator
        if not it.get("speech") or not it["span"]:
            continue
        mid = (it["span"][0] + it["span"][1]) // 2
        n = next((k for k, (a, b) in enumerate(runs, 1) if a - 2 <= mid <= b + 2), None)
        who = next((e.get("speaker") for e in entries if e.get("n") == n), None) if n else None
        if who and (voices is None or who in voices):
            it["voice"] = who
            counts[who] = counts.get(who, 0) + 1
        else:
            counts["(narrator)"] = counts.get("(narrator)", 0) + 1
    return counts


def convert_qwen_scenes(src_dir: str, voice: str, prose_dir: Optional[str] = None,
                        speakers: Optional[Dict[str, Any]] = None,
                        voices: Optional[Dict[str, Any]] = None) -> str:
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
            if items and items[0]["pause"] is None and re.match(r"(?i)глава|розділ|часть|частина", items[0]["text"]):
                items[0]["pause"] = GAPS_MS["paragraph"]  # the spoken heading
            if speakers is not None:
                items = split_speech(items, prose)
                counts = assign_speakers(items, prose, speakers.get(name, []), voice, voices)
                note(f"{name}: speech lines by voice {counts}")
            items = merge_breaths(items, prose)
        out.append(f"# Глава {ch:02d}, сцена {sc}")
        current = None
        for it in items:
            if it.get("voice", voice) != current:
                current = it.get("voice", voice)
                out.append(f"[voice:{current}]")
            line = f"{it['tags']} {it['text']}" if it["tags"] else it["text"]
            out.append(line if it["pause"] is None else f"{line} [pause {it['pause']}ms]")
        out.append("")
    return "\n".join(out)


def cmd_import_qwen(args: argparse.Namespace) -> int:
    speakers = None
    if args.speakers:
        if not args.prose:
            raise SystemExit("--speakers needs --prose (speech is found in the original prose)")
        with open(args.speakers, encoding="utf-8") as f:
            speakers = json.load(f)
    voices = load_voices(args.voices) if args.voices else None
    script = convert_qwen_scenes(args.src, args.voice, args.prose, speakers, voices)
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


def publish(local_dir: str, remote: str, includes: List[str], binary: Optional[str] = None) -> Optional[str]:
    """Copy finished files to cloud storage with rclone (e.g. remote 'gdrive:Audiobook/echo'),
    so the user's PC does not have to hold them. Returns an error text, or None."""
    rclone = binary or os.environ.get("RCLONE_BIN", "rclone")
    cmd = [rclone, "copy", local_dir, remote]
    for pattern in includes:
        cmd += ["--include", pattern]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"rclone failed: {e}"
    if res.returncode != 0:
        return f"rclone failed: {(res.stderr or res.stdout)[-500:]}"
    note(f"uploaded to {remote}")
    return None


def cmd_lexicon(args: argparse.Namespace) -> int:
    data: Dict[str, str] = {}
    if os.path.exists(args.lexicon):
        with open(args.lexicon, encoding="utf-8") as f:
            data = json.load(f)
    if not args.word:
        emit({"ok": True, "lexicon": args.lexicon, "entries": len(data), "words": data})
        return 0
    key = next((k for k in data if k.lower() == args.word.lower()), args.word)
    old = data.get(key)
    if args.remove:
        data.pop(key, None)
    elif args.spelling:
        data[key] = args.spelling
    else:
        emit({"ok": True, "word": key, "spelling": old})
        return 0
    tmp = args.lexicon + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
        f.write("\n")
    os.replace(tmp, args.lexicon)
    emit({"ok": True, "word": key, "was": old, "now": data.get(key),
          "note": "lines with this word are re-rendered on the next render; a stress mark on a word-initial "
                  "е is read as ё, respell instead (йе́дкой)"})
    return 0


def publish_file(path: str, remote_path: str, binary: Optional[str] = None) -> Optional[str]:
    rclone = binary or os.environ.get("RCLONE_BIN", "rclone")
    try:
        res = subprocess.run([rclone, "copyto", path, remote_path], capture_output=True, text=True, timeout=3600)
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"rclone failed: {e}"
    if res.returncode != 0:
        return f"rclone failed: {(res.stderr or res.stdout)[-500:]}"
    note(f"uploaded {remote_path}")
    return None


def cmd_render(args: argparse.Namespace) -> int:
    chapters, voices = load_book(args)
    problems = check_voices(chapters, voices)
    if args.backend == "kaggle" and not Kaggle.username(args.kaggle_user, args.kaggle_bin):
        binary = args.kaggle_bin or os.environ.get("KAGGLE_BIN", "kaggle")
        if not shutil.which(binary):
            problems.append(f"kaggle CLI not found on PATH ({binary}); add ~/.local/bin to PATH or pass --kaggle-bin")
        else:
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
        def finish_and_publish(ch, ch_dir, out_zip, job_id, started):
            info = finish_chapter(ch, ch_dir, out_zip, job_id, started)
            err = publish(ch_dir, f"{args.publish.rstrip('/')}/{os.path.basename(ch_dir)}",
                          ["*.mp3", "manifest.json"])
            mp3 = os.path.join(ch_dir, "chapter.mp3")
            if not err and getattr(args, "tag", None) and os.path.exists(mp3):
                name = f"{os.path.basename(ch_dir)}_{args.tag}.mp3"
                err = publish_file(mp3, f"{args.publish.rstrip('/')}/{name}")
                info["published_as"] = name
            if err:
                info["publish_error"] = err
                note(f"WARNING: {err}")
            return info

        finish = finish_and_publish if args.publish else None
        if args.backend == "kaggle":
            return _render_kaggle(args, todo, finish)
        return _render_colab(args, todo, finish)
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


def _render_colab(args: argparse.Namespace, todo: List[tuple], finish=None) -> int:
    finish = finish or finish_chapter
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
            results.append(finish(ch, ch_dir, out_zip, job_id, started))
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


def _render_kaggle(args: argparse.Namespace, todo: List[tuple], finish=None) -> int:
    finish = finish or finish_chapter
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
        exists = False
        for attempt in range(3):
            try:
                kaggle.run("datasets", "status", dataset_id, timeout=120)
                exists = True
                break
            except ColabError:
                break
            except subprocess.TimeoutExpired:
                if attempt == 2:
                    raise
                time.sleep(10)
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
                if "reference '" in line or "cast '" in line:
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
                results.append(finish(ch, ch_dir, out_zip, job_id, started))
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


# ---------------------------------------------------------------------------
# Casting: a voice per character, with its emotions, before rendering
# ---------------------------------------------------------------------------

# Delivery styles are English (Qwen3-TTS VoiceDesign follows English or
# Chinese instructions best); the sample sentences are what each reference
# says, so they carry the emotion in their words too.
DEFAULT_EMOTIONS = {
    "calm": {"style": "calm, even, unhurried delivery",
             "text": "Я помню тот день до мелочей: серое небо, тишина и запах мокрой земли."},
    "tense": {"style": "tense and anxious, low urgent voice, restrained fear",
              "text": "Тихо. Слышишь? Они где-то рядом, не двигайся и держи оружие наготове."},
    "shout": {"style": "shouting at full voice, commanding, furious",
              "text": "Все в укрытие! Быстро, я сказал! Огонь по левому флангу, не жалеть патронов!"},
    "whisper": {"style": "whispering very quietly, breathy, almost silent",
                "text": "Не шуми. Подожди здесь, я проверю, что там за поворотом, и сразу вернусь."},
    "sad": {"style": "sad and grieving, slow, voice breaking with pain",
            "text": "Его больше нет. Я не успел, понимаешь? Просто не успел, а он так ждал."},
}


def load_cast(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        spec = json.load(f)
    emotions = {k: dict(v) for k, v in DEFAULT_EMOTIONS.items()}
    for name, e in (spec.get("emotions") or {}).items():
        emotions.setdefault(name, {}).update(e)
    spec["emotions"] = {k: v for k, v in emotions.items() if v.get("text")}
    if not spec.get("characters"):
        raise ValueError("cast file has no characters")
    return spec


def build_cast_job(spec: Dict[str, Any], voices: Dict[str, Dict[str, Any]], settings: Dict[str, Any],
                   zip_path: str, only: Optional[List[str]] = None, existing: Optional[str] = None) -> str:
    """existing: a cast folder from an earlier run (…/cast/<name>/<emotion>.wav).
    Its calm voices and chosen emotion takes are reused and only converted to
    each character's own timbre; nothing is designed again."""
    files: Dict[str, bytes] = {}
    characters: Dict[str, Any] = {}
    for i, (name, ch) in enumerate(spec["characters"].items()):
        if only and name not in only:
            continue
        ch = dict(ch)
        given = ch.pop("voice", None)
        if given:
            # Emotions for a voice that already exists, e.g. the narrator.
            v = voices[given]
            with open(v["ref_audio"], "rb") as f:
                files[f"voices/{i:02d}.wav"] = f.read()
            ch.update(ref_audio=f"voices/{i:02d}.wav", ref_text=v["ref_text"])
        wanted = ch.pop("emotions", None)
        if wanted:
            ch["only_emotions"] = wanted
        if existing:
            folder = os.path.join(existing, slug(name).rstrip("_") or name)
            folder = folder if os.path.isdir(folder) else os.path.join(existing, name)
            calm = os.path.join(folder, "calm.wav")
            if not given and os.path.exists(calm):
                with open(calm, "rb") as f:
                    files[f"voices/{i:02d}.wav"] = f.read()
                ch.update(ref_audio=f"voices/{i:02d}.wav", ref_text=spec["emotions"]["calm"]["text"])
            sources = {}
            for emo in spec["emotions"]:
                path = os.path.join(folder, f"{emo}.wav")
                if emo != "calm" and os.path.exists(path):
                    with open(path, "rb") as f:
                        files[f"sources/{i:02d}_{emo}.wav"] = f.read()
                    sources[emo] = f"sources/{i:02d}_{emo}.wav"
            if not sources:
                continue
            ch["emotion_sources"] = sources
        characters[name] = ch
    if not characters:
        raise ValueError("no characters selected")
    job = {
        "kind": "cast",
        "language": settings["language"],
        "takes": settings["takes"],
        "design_engine": settings["design_engine"],
        "design_options": settings.get("design_options", {}),
        "vc": settings.get("vc", "seed-vc"),
        "emotions": spec["emotions"],
        "characters": characters,
    }
    job["job_id"] = hashlib.sha256(json.dumps(job, ensure_ascii=False, sort_keys=True).encode()
                                   + b"".join(files[k] for k in sorted(files))).hexdigest()[:16]
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("job.json", json.dumps(job, ensure_ascii=False, indent=2))
        for rel, data in files.items():
            z.writestr(rel, data)
    return job["job_id"]


def finish_cast(ch: Dict[str, Any], out_dir: str, out_zip: str, job_id: str, started: float) -> Dict[str, Any]:
    with zipfile.ZipFile(out_zip) as z:
        z.extractall(out_dir)
    with open(os.path.join(out_dir, "voices.json"), encoding="utf-8") as f:
        cast_voices = json.load(f)
    with open(os.path.join(out_dir, "cast_report.json"), encoding="utf-8") as f:
        report = json.load(f)
    for v in cast_voices.values():
        if not os.path.isabs(v["ref_audio"]):
            v["ref_audio"] = os.path.join(os.path.abspath(out_dir), v["ref_audio"])
    summary = {}
    for name, entry in report.items():
        if name.startswith("_"):
            continue
        for emo, e in entry["emotions"].items():
            chosen = e.get("chosen") or {}
            summary[f"{name}:{emo}"] = {"file": os.path.join(out_dir, e["file"]) if e.get("file") else None,
                                        "similarity": chosen.get("similarity"), "heard": chosen.get("heard"),
                                        "warning": e.get("warning")}
    return {"voices": cast_voices, "summary": summary, "between_characters": report.get("_between_characters", {}),
            "wall_seconds": round(time.time() - started, 1)}


def merge_voices(path: str, new: Dict[str, Dict[str, Any]], remove: Iterable[str] = ()) -> None:
    data: Dict[str, Any] = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    for key in remove:
        data.pop(key, None)
    data.update(new)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def cmd_cast(args: argparse.Namespace) -> int:
    spec = load_cast(args.cast)
    voices = load_voices(args.voices) if args.voices and os.path.exists(args.voices) else {}
    problems = []
    for name, ch in spec["characters"].items():
        if ch.get("voice") and ch["voice"] not in voices:
            problems.append(f"character '{name}': voice '{ch['voice']}' is not in {args.voices}")
        if not (ch.get("description") or "").strip():
            # Also for an existing voice: its emotion takes are designed from
            # the description and the closest timbre is kept.
            problems.append(f"character '{name}' needs a description of the voice")
    if problems:
        emit({"ok": False, "problems": problems})
        return 2
    settings = {
        "language": args.language or spec.get("language", "Russian"),
        "takes": args.takes or int(spec.get("takes", 3)),
        "design_engine": args.design_engine,
        "design_options": json.loads(args.design_options) if args.design_options else {},
        "vc": args.vc or ("dummy" if args.design_engine == "dummy" else "seed-vc"),
    }
    only = [x.strip() for x in args.characters.split(",")] if args.characters else None
    os.makedirs(args.out, exist_ok=True)
    tmp_zip = os.path.join(tempfile.gettempdir(), f"audiobook_cast_{os.getpid()}.zip")
    job_id = build_cast_job(spec, voices, settings, tmp_zip, only, args.convert_existing)
    collected: List[Dict[str, Any]] = []

    def finish(ch, out_dir, out_zip, jid, started):
        info = finish_cast(ch, out_dir, out_zip, jid, started)
        for vname, v in info["voices"].items():
            # "Диктор:tense" reads at the narrator's speed and seed.
            base = voices.get(vname.split(":")[0])
            if base:
                v["speed"], v["seed"] = base["speed"], base["seed"]
        # An emotion this run dropped must not keep an older file of another timbre.
        dropped = [k for k, v in info["summary"].items() if not v.get("file") and k not in info["voices"]]
        if args.voices:
            merge_voices(args.voices, info["voices"], dropped)
            note(f"added {len(info['voices'])} voice(s) to {args.voices}"
                 + (f", removed dropped {', '.join(dropped)}" if dropped else ""))
        collected.append(info)
        if args.publish:
            # cast/<name>/... lands at <remote>/<name>/..., not <remote>/cast/<name>/.
            err = publish(os.path.join(out_dir, "cast"), args.publish,
                          ["/*/*.wav"] + (["/*/takes/*.wav"] if args.publish_takes else []))
            err = err or publish(out_dir, args.publish, ["/cast_report.json"])
            if err:
                note(f"WARNING: {err}")
        return {"summary": info["summary"], "between_characters": info["between_characters"],
                "wall_seconds": info["wall_seconds"]}

    todo = [({"index": 1, "title": "cast", "lines": []}, args.out, tmp_zip, job_id)]
    os.makedirs(STATE_DIR, exist_ok=True)
    try:
        if args.backend == "local":
            import importlib.util

            spec_mod = importlib.util.spec_from_file_location("render_chapter", RENDER_SCRIPT)
            rc = importlib.util.module_from_spec(spec_mod)
            spec_mod.loader.exec_module(rc)
            out_zip = tmp_zip.replace("_cast_", "_castout_")
            started = time.time()
            rc.run_one(tmp_zip, out_zip, tempfile.mkdtemp(prefix="audiobook_castwork_"))
            result = finish(todo[0][0], args.out, out_zip, job_id, started)
            os.remove(out_zip)
            emit({"ok": True, "backend": "local", "rendered": [result]})
            return 0
        if args.backend == "kaggle":
            return _render_kaggle(args, todo, finish)
        return _render_colab(args, todo, finish)
    finally:
        if os.path.exists(tmp_zip):
            os.remove(tmp_zip)
        try:
            os.remove(LOCK_PATH)
        except OSError:
            pass


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


def _subparsers(parser: argparse.ArgumentParser) -> Dict[str, argparse.ArgumentParser]:
    for a in parser._actions:
        if isinstance(a, argparse._SubParsersAction):
            return dict(a.choices)
    return {}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--colab-bin", default=None, help="path to the colab CLI (default: $COLAB_BIN or 'colab')")
    sub = p.add_subparsers(dest="cmd", required=True)

    def book_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("book", nargs="?", default=None,
                        help="book script with # chapters and [voice:NAME] tags (or 'book' in --project)")
        sp.add_argument("--voices", default=None, help="voices.json or old voice_library.json")
        sp.add_argument("--default-voice", default=None, help="voice for text before the first tag")
        sp.add_argument("--chapters", default=None, help="only these chapter numbers, e.g. 1,2,5")

    sp = sub.add_parser("plan", help="validate script and voices, no GPU")
    book_args(sp)
    sp.add_argument("--gaps", action="store_true", help="also list each line with its pause after it")
    sp.set_defaults(func=cmd_plan)

    sp = sub.add_parser("render", help="render chapters on a Colab GPU")
    book_args(sp)
    sp.add_argument("--out", default=None, help="output directory")
    sp.add_argument("--engine", default="omnivoice", choices=["omnivoice", "voxcpm2", "dummy"])
    sp.add_argument("--engine-options", default=None, help='JSON passed to the engine, e.g. {"num_step": 32}')
    sp.add_argument("--language", default="uk", help="uk, ru, ... (OmniVoice uses it; VoxCPM2 ignores it)")
    sp.add_argument("--stress", default="strip", choices=["strip", "acute", "keep"],
                    help="what to do with '+' stress marks: strip, turn into U+0301, or keep")
    sp.add_argument("--lexicon", default=None, help="JSON {word: respelling}")
    sp.add_argument("--vc-lines", default="none", choices=["none", "seed-vc", "dummy"],
                    help="convert every emotion line ('Name:tense', ...) to the calm voice's timbre after "
                         "synthesis, so a character never sounds like a second person")
    sp.add_argument("--max-synth-chars", type=int, default=None,
                    help="synthesise lines longer than this in pieces (default 170; 0 = never split)")
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
    sp.add_argument("--publish", default=None,
                    help="rclone destination for each chapter's mp3, e.g. gdrive:Audiobook/echo")
    sp.add_argument("--tag", default=None,
                    help="with --publish, also upload <chapter>_<tag>.mp3 to the destination root, e.g. v16")
    sp.set_defaults(func=cmd_render)

    sp = sub.add_parser("cast", help="design each character's voice and emotions (Qwen3-TTS VoiceDesign)")
    sp.add_argument("cast", nargs="?", default=None,
                    help="cast JSON: characters (description or existing voice) and optional emotions")
    sp.add_argument("--out", default=None, help="folder for cast/<name>/<emotion>.wav, takes and the report")
    sp.add_argument("--voices", default=None, help="voices.json to read existing voices from and add the cast to")
    sp.add_argument("--characters", default=None, help="only these characters, comma separated")
    sp.add_argument("--takes", type=int, default=None, help="takes per emotion to choose from (default 3)")
    sp.add_argument("--language", default=None, help="Qwen language name, e.g. Russian (default from the cast file)")
    sp.add_argument("--design-engine", default="qwen", choices=["qwen", "dummy"])
    sp.add_argument("--design-options", default=None, help='JSON, e.g. {"temperature": 0.8, "vc_steps": 30}')
    sp.add_argument("--vc", default=None, choices=["seed-vc", "none", "dummy"],
                    help="convert each emotion take to the character's own voice (default seed-vc)")
    sp.add_argument("--convert-existing", default=None,
                    help="cast folder of an earlier run: keep its takes, only convert them to each voice")
    sp.add_argument("--backend", default="kaggle", choices=["kaggle", "colab", "local"])
    sp.add_argument("--gpu", default="L4", help="Colab GPU")
    sp.add_argument("--session", default=None, help="Colab session name")
    sp.add_argument("--keep", action="store_true", help="leave the Colab session running afterwards")
    sp.add_argument("--chapter-timeout", type=int, default=3 * 3600, help="seconds for the cast job (Colab)")
    sp.add_argument("--kaggle-bin", default=None)
    sp.add_argument("--kaggle-user", default=None)
    sp.add_argument("--kaggle-dataset", default="audiobook-cast")
    # A new kernel slug got 409 Conflict on push; reuse the render notebook.
    sp.add_argument("--kaggle-kernel", default="audiobook-render")
    sp.add_argument("--kaggle-accelerator", default="NvidiaTeslaT4")
    sp.add_argument("--kaggle-max-hours", type=int, default=4)
    sp.add_argument("--poll-seconds", type=float, default=60.0)
    sp.add_argument("--publish", default=None, help="rclone destination for the voices, e.g. gdrive:Audiobook/echo/cast")
    sp.add_argument("--publish-takes", action="store_true", help="also upload every take, not just the chosen ones")
    sp.set_defaults(func=cmd_cast, book=None)

    sp = sub.add_parser("lexicon", help="add, change or remove a pronunciation (stress) fix")
    sp.add_argument("word", nargs="?", default=None, help="the word as written in the book")
    sp.add_argument("spelling", nargs="?", default=None,
                    help="how the model should read it, e.g. кабуры́; omit to show the current entry")
    sp.add_argument("--lexicon", default=None, help="lexicon JSON (or 'lexicon' in --project)")
    sp.add_argument("--remove", action="store_true", help="delete the entry for the word")
    sp.set_defaults(func=cmd_lexicon)

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
    sp.add_argument("--speakers", default=None,
                    help="speakers.json from `attribute`: character lines get the character's voice")
    sp.add_argument("--voices", default=None,
                    help="with --speakers: characters missing from this voices file stay with the narrator")
    sp.set_defaults(func=cmd_import_qwen)

    sp = sub.add_parser("attribute", help="find who says each line of direct speech (LLM and/or name after verb)")
    sp.add_argument("prose", help="folder with the original prose scene files")
    sp.add_argument("--cast", required=True, help="cast JSON (character names and aliases)")
    sp.add_argument("--out", required=True, help="speakers.json to write (existing scenes are replaced)")
    sp.add_argument("--only", default=None, help="regex on file names, e.g. chapter_01_")
    sp.add_argument("--llm-url", default=None, help="OpenAI-compatible base URL, e.g. http://127.0.0.1:8317/v1")
    sp.add_argument("--llm-model", default="qoder-qwen3.8-flash")
    sp.add_argument("--llm-key-env", default=None, help="env var holding the API key, if the endpoint needs one")
    sp.set_defaults(func=cmd_attribute)

    sp = sub.add_parser("usage", help="compute units balance and burn rate")
    sp.set_defaults(func=cmd_usage)

    sp = sub.add_parser("watchdog", help="report (and with --stop, end) sessions nobody is using")
    sp.add_argument("--stop", action="store_true")
    sp.add_argument("--max-hours", type=float, default=8.0, help="flag a render running longer than this")
    sp.set_defaults(func=cmd_watchdog)
    return p


# Settings that hold file paths; in a project file they are relative to it.
PROJECT_PATHS = {"book", "voices", "lexicon", "out", "cast", "convert_existing"}
REQUIRED = {"plan": ["book", "voices"], "render": ["book", "voices", "out"], "cast": ["cast", "out"],
            "lexicon": ["lexicon"]}


def load_project(path: str, command: str) -> Dict[str, Any]:
    """A book's settings in one JSON file, so a whole render is one short
    command. Top-level keys apply to every command, "commands": {"render": {...},
    "cast": {...}} to one; keys are the long option names (with _ or -)."""
    with open(os.path.expanduser(path), encoding="utf-8") as f:
        data = json.load(f)
    base = os.path.dirname(os.path.abspath(os.path.expanduser(path)))
    merged = {k: v for k, v in data.items() if k not in ("commands", "_comment")}
    merged.update((data.get("commands") or {}).get(command) or {})
    out: Dict[str, Any] = {}
    for key, value in merged.items():
        key = key.replace("-", "_")
        if key in PROJECT_PATHS and isinstance(value, str) and ":" not in value[:3]:
            value = os.path.join(base, os.path.expanduser(value))
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)  # e.g. engine_options
        out[key] = value
    return out


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    project = None
    if "--project" in argv:
        i = argv.index("--project")
        if i + 1 >= len(argv):
            emit({"ok": False, "problems": ["--project needs a file"]})
            return 2
        project = argv[i + 1]
        del argv[i:i + 2]
    parser = build_parser()
    if project:
        command = next((a for a in argv if not a.startswith("-") and a in _subparsers(parser)), None)
        if command:
            sp = _subparsers(parser)[command]
            known = {a.dest for a in sp._actions}
            try:
                settings = load_project(project, command)
            except (OSError, json.JSONDecodeError) as e:
                emit({"ok": False, "problems": [f"project file {project}: {e}"]})
                return 2
            sp.set_defaults(**{k: v for k, v in settings.items() if k in known})
    args = parser.parse_args(argv)
    missing = [f"{'' if k in ('book', 'cast') else '--'}{k.replace('_', '-')}"
               for k in REQUIRED.get(args.cmd, []) if not getattr(args, k, None)]
    if missing:
        emit({"ok": False, "problems": [f"missing {', '.join(missing)} (pass it or put it in --project)"]})
        return 2
    try:
        return args.func(args)
    except (ValueError, KeyError, FileNotFoundError, json.JSONDecodeError, ColabError) as e:
        emit({"ok": False, "problems": [f"{type(e).__name__}: {e}"]})
        return 2


if __name__ == "__main__":
    sys.exit(main())
