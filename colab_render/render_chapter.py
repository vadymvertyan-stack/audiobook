"""
Render one audiobook chapter on a GPU (Colab VM or any CUDA box).

This file is self-contained on purpose: `colab exec -f render_chapter.py`
sends only this file's text to the remote kernel, so it must not import
anything else from this repository.

Input is a job archive built by `orchestrate.py`:

    job.zip
      job.json        # engine, language, voices, lines (see orchestrate.py)
      voices/*.wav    # one reference clip per voice

Output is an archive with:

    out.zip
      lines/<id>.wav       # every line on its own
      manifest.json        # timing, pauses, seed and settings for every line

The server (orchestrate.py) joins lines/ into chapter.wav and per-voice
stems, so the audio crosses the Colab file API only once.

Consistency rules this script enforces (the reasons the old Qwen script
drifted):
  * every voice MUST have a reference clip + its transcript; there is no
    "design a voice from a description" fallback per line;
  * the clone prompt is built once per voice and reused for every line;
  * each voice has one fixed seed used for all of its lines;
  * speed is passed to the model, never done by resampling (resampling
    shifts pitch).

Configuration (env vars win, so it works under `colab exec --env`):
  AUDIOBOOK_JOB      path to job.zip           (default /content/audiobook_job.zip)
  AUDIOBOOK_OUT      path to write out.zip     (default /content/audiobook_out.zip)
  AUDIOBOOK_WORKDIR  scratch dir               (default /content/audiobook_job)
  AUDIOBOOK_ENGINE   override job.json engine  (omnivoice | voxcpm2 | dummy)
  AUDIOBOOK_BATCH_GLOB  Kaggle mode: render every job_NN.zip matching this
                        glob into AUDIOBOOK_BATCH_OUT/out_NN.zip
                        (+ batch_report.json)
"""

import builtins
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
import wave
import zipfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

COMBINING_ACUTE = "́"
CYRILLIC_VOWELS = set("аеєиіїоуюяыэёАЕЄИІЇОУЮЯЫЭЁ")
REF_WARN_SECONDS = 12.0
DONE_MARKER = "AUDIOBOOK_RENDER_OK"

# Survives between `colab exec` calls in the same kernel, so a book rendered
# chapter by chapter in one session loads the model only once.
_CACHE_ATTR = "_audiobook_engine_cache"


def _read_text(path: str) -> Optional[str]:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return None


def log(msg: str) -> None:
    # Regular output also keeps `colab exec` from hitting its quiet-period timeout.
    print(f"[render {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Audio helpers (stdlib wave + numpy, no soundfile dependency)
# ---------------------------------------------------------------------------

def to_mono_float(audio: Any) -> np.ndarray:
    if hasattr(audio, "detach"):
        audio = audio.detach().float().cpu().numpy()
    arr = np.asarray(audio, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim > 1:
        arr = arr.mean(axis=0 if arr.shape[0] < arr.shape[-1] else -1)
    return np.nan_to_num(arr.astype(np.float32))


def resample_linear(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Only used to bring an engine's output to the chapter sample rate."""
    if src_sr == dst_sr or audio.size == 0:
        return audio
    n_out = int(round(audio.size * dst_sr / src_sr))
    x_old = np.linspace(0.0, 1.0, num=audio.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def write_wav(path: str, audio: np.ndarray, sr: int) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def read_wav(path: str) -> Tuple[np.ndarray, int]:
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        width = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f"{path}: only 16-bit PCM wav is supported, got {width * 8}-bit")
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        audio = audio.reshape(-1, ch).mean(axis=1)
    return audio, sr


def wav_seconds(path: str) -> Optional[float]:
    try:
        with wave.open(path, "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return None


def trim_silence(audio: np.ndarray, sr: int, threshold: float = 0.01, keep_ms: int = 40) -> np.ndarray:
    if audio.size == 0:
        return audio
    loud = np.flatnonzero(np.abs(audio) > threshold)
    if loud.size == 0:
        return audio
    pad = int(sr * keep_ms / 1000)
    start = max(0, loud[0] - pad)
    end = min(audio.size, loud[-1] + pad + 1)
    return audio[start:end]


# ---------------------------------------------------------------------------
# Text: stress marks and lexicon
# ---------------------------------------------------------------------------

def apply_stress_mode(text: str, mode: str) -> str:
    """`+` before a stressed vowel (RUAccent / ESpeech style: "прив+ет").

    keep  -> leave "+" as is (for engines trained on it)
    acute -> "приве́т" (combining U+0301 after the vowel)
    strip -> "привет" (drop "+" and any U+0301 already in the text)
    """
    if mode == "keep":
        return text
    out: List[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "+" and i + 1 < len(text) and text[i + 1] in CYRILLIC_VOWELS:
            if mode == "acute":
                out.append(text[i + 1] + COMBINING_ACUTE)
            else:
                out.append(text[i + 1])
            i += 2
            continue
        out.append(ch)
        i += 1
    result = "".join(out)
    if mode == "strip":
        result = unicodedata.normalize("NFC", result.replace(COMBINING_ACUTE, ""))
    return result


def apply_lexicon(text: str, lexicon: Dict[str, str]) -> str:
    """Whole-word respelling, longest key first."""
    if not lexicon:
        return text
    import re

    keys = sorted((k for k in lexicon if k.strip()), key=len, reverse=True)
    if not keys:
        return text
    pattern = re.compile(r"(?<!\w)(" + "|".join(re.escape(k) for k in keys) + r")(?!\w)", re.IGNORECASE)
    lowered = {k.lower(): v for k, v in lexicon.items()}
    return pattern.sub(lambda m: lowered.get(m.group(0).lower(), m.group(0)), text)


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------

class NoInternet(RuntimeError):
    pass


def _pip_install(*packages: str) -> None:
    import socket

    try:
        socket.getaddrinfo("pypi.org", 443)
    except OSError as e:
        raise NoInternet(
            f"no internet on this machine ({e}); on Kaggle, verify your phone number "
            "at kaggle.com/settings, otherwise notebooks run offline") from e
    log(f"pip install {' '.join(packages)}")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *packages], check=True)


def _seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


class DummyEngine:
    """No-GPU stand-in: a tone whose pitch depends on the voice. For pipeline tests."""

    name = "dummy"
    sample_rate = 24000

    def prepare_voice(self, voice_name: str, voice: Dict[str, Any]) -> Any:
        return {"freq": 140.0 + (int(hashlib.md5(voice_name.encode()).hexdigest(), 16) % 160)}

    def synth(self, text: str, prompt: Any, voice: Dict[str, Any], language: str, speed: float, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed + len(text))
        seconds = max(0.4, len(text) * 0.06 / max(speed, 0.1))
        t = np.arange(int(seconds * self.sample_rate)) / self.sample_rate
        tone = 0.2 * np.sin(2 * np.pi * prompt["freq"] * t)
        return (tone + 0.005 * rng.standard_normal(t.size)).astype(np.float32)


def _words(text: str) -> List[str]:
    text = unicodedata.normalize("NFD", text.lower().replace("ё", "е"))
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.findall(r"\w+", text)


def reference_mismatch(ref_text: str, heard: str) -> Optional[str]:
    """None when the transcript of the reference matches ref_text closely
    enough, otherwise a short reason. The first and last words matter most:
    an extra or missing word at an edge leaks into every generated line."""
    import difflib

    want, got = _words(ref_text), _words(heard)
    if not want or not got:
        return "empty text or transcript"
    for edge, w, g in (("first", want[0], got[0]), ("last", want[-1], got[-1])):
        if difflib.SequenceMatcher(None, w, g).ratio() < 0.6:
            return f"{edge} word differs: text '{w}', audio '{g}'"
    ratio = difflib.SequenceMatcher(None, want, got).ratio()
    if ratio < 0.8:
        return f"only {ratio:.0%} of the words match"
    return None


class OmniVoiceEngine:
    """k2-fsa/OmniVoice: non-autoregressive, deterministic token choice by default."""

    name = "omnivoice"
    sample_rate = 24000

    def __init__(self, model_id: str = "k2-fsa/OmniVoice", options: Optional[Dict[str, Any]] = None):
        try:
            from omnivoice import OmniVoice  # noqa: F401
        except ImportError:
            _pip_install("omnivoice")
        import torch
        from omnivoice import OmniVoice

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        log(f"loading {model_id} on {device}")
        self.model = OmniVoice.from_pretrained(model_id, device_map=device, dtype=dtype)
        self.options = dict(options or {})
        self.check_ref = bool(self.options.pop("check_ref", True))

    def prepare_voice(self, voice_name: str, voice: Dict[str, Any]) -> Any:
        if self.check_ref:
            self._check_reference(voice_name, voice)
        return self.model.create_voice_clone_prompt(ref_audio=voice["ref_audio"], ref_text=voice["ref_text"])

    def _check_reference(self, voice_name: str, voice: Dict[str, Any]) -> None:
        """Whisper the reference and compare it with ref_text. A word that is in
        the text but not in the audio gets spoken at the start of every line."""
        if getattr(self.model, "_asr_pipe", None) is None:
            self.model.load_asr_model()
        heard = self.model.transcribe(voice["ref_audio"])
        problem = reference_mismatch(voice["ref_text"], heard)
        log(f"reference '{voice_name}': whisper heard: {heard}")
        if problem:
            raise SystemExit(
                f"reference for '{voice_name}' does not match its ref_text ({problem}). "
                f"Whisper heard: \"{heard}\". Fix ref_text or re-cut the wav; "
                'pass --engine-options \'{"check_ref": false}\' to skip this check.')

    def synth(self, text: str, prompt: Any, voice: Dict[str, Any], language: str, speed: float, seed: int) -> np.ndarray:
        _seed_everything(seed)
        kwargs = dict(self.options)
        if language:
            kwargs["language"] = language
        if abs(speed - 1.0) > 1e-3:
            kwargs["speed"] = speed
        audio = self.model.generate(text=text, voice_clone_prompt=prompt, **kwargs)
        return to_mono_float(audio[0] if isinstance(audio, (list, tuple)) else audio)


class VoxCPM2Engine:
    """openbmb/VoxCPM2: 48 kHz, 'ultimate cloning' with reference + transcript."""

    name = "voxcpm2"

    def __init__(self, model_id: str = "openbmb/VoxCPM2", options: Optional[Dict[str, Any]] = None):
        try:
            from voxcpm import VoxCPM  # noqa: F401
        except ImportError:
            _pip_install("voxcpm")
        from voxcpm import VoxCPM

        log(f"loading {model_id}")
        self.model = VoxCPM.from_pretrained(model_id, load_denoiser=False)
        self.sample_rate = int(self.model.tts_model.sample_rate)
        self.options = {"cfg_value": 2.0, "inference_timesteps": 10}
        self.options.update(options or {})
        self._warned_speed = False

    def prepare_voice(self, voice_name: str, voice: Dict[str, Any]) -> Any:
        return {"wav": voice["ref_audio"], "text": voice["ref_text"]}

    def synth(self, text: str, prompt: Any, voice: Dict[str, Any], language: str, speed: float, seed: int) -> np.ndarray:
        if abs(speed - 1.0) > 1e-3 and not self._warned_speed:
            log("voxcpm2 has no speed control; speed values are ignored")
            self._warned_speed = True
        wav = self.model.generate(
            text=text,
            prompt_wav_path=prompt["wav"],
            prompt_text=prompt["text"],
            reference_wav_path=prompt["wav"],
            seed=seed,
            **self.options,
        )
        return to_mono_float(wav)


def get_engine(name: str, options: Dict[str, Any]) -> Any:
    cache = getattr(builtins, _CACHE_ATTR, None)
    if cache is None:
        cache = {}
        setattr(builtins, _CACHE_ATTR, cache)
    key = name + json.dumps(options, sort_keys=True)
    if key in cache:
        log(f"reusing loaded engine '{name}'")
        return cache[key]
    model_id = options.pop("model_id", None)
    if name == "dummy":
        engine = DummyEngine()
    elif name == "omnivoice":
        engine = OmniVoiceEngine(model_id or "k2-fsa/OmniVoice", options)
    elif name == "voxcpm2":
        engine = VoxCPM2Engine(model_id or "openbmb/VoxCPM2", options)
    else:
        raise ValueError(f"unknown engine '{name}' (use omnivoice, voxcpm2 or dummy)")
    cache[key] = engine
    return engine


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def line_cache_key(engine: str, language: str, voice: Dict[str, Any], text: str, speed: float, seed: int) -> str:
    payload = json.dumps(
        [engine, language, voice.get("ref_text"), voice.get("ref_sha256"), text, round(speed, 4), seed],
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def render_job(job_dir: str, out_dir: str, engine_override: Optional[str] = None) -> Dict[str, Any]:
    with open(os.path.join(job_dir, "job.json"), encoding="utf-8") as f:
        job = json.load(f)

    engine_name = engine_override or job.get("engine", "omnivoice")
    language = job.get("language", "")
    stress_mode = job.get("stress_mode", "strip")
    lexicon = job.get("lexicon", {}) or {}
    gaps = {"line": 250, "speaker_change": 400, "paragraph": 700}
    gaps.update(job.get("gap_ms", {}) or {})
    voices: Dict[str, Dict[str, Any]] = job["voices"]
    lines: List[Dict[str, Any]] = job["lines"]

    # Fail before loading any model if a line has no usable reference voice.
    missing = sorted({ln["voice"] for ln in lines if ln["voice"] not in voices})
    if missing:
        raise SystemExit(f"no reference voice for: {', '.join(missing)}")
    for name, v in voices.items():
        v["ref_audio"] = os.path.join(job_dir, v["ref_audio"])
        if not os.path.exists(v["ref_audio"]) or not (v.get("ref_text") or "").strip():
            raise SystemExit(f"voice '{name}' needs both a reference wav and its exact transcript")
        secs = wav_seconds(v["ref_audio"])
        if secs and secs > REF_WARN_SECONDS:
            log(f"warning: reference for '{name}' is {secs:.1f}s; 3-10s clips clone more steadily")

    engine = get_engine(engine_name, dict(job.get("engine_options", {}) or {}))
    sr = int(engine.sample_rate)

    prompts: Dict[str, Any] = {}
    used_voices = sorted({ln["voice"] for ln in lines})
    for name in used_voices:
        log(f"preparing voice '{name}'")
        prompts[name] = engine.prepare_voice(name, voices[name])

    lines_dir = os.path.join(out_dir, "lines")
    os.makedirs(lines_dir, exist_ok=True)
    manifest_lines: List[Dict[str, Any]] = []
    started = time.time()
    for idx, ln in enumerate(lines, 1):
        voice = voices[ln["voice"]]
        speed = float(ln.get("speed") or voice.get("speed") or 1.0)
        seed = int(voice.get("seed", 1234))
        text = apply_stress_mode(apply_lexicon(ln["text"], lexicon), stress_mode)
        key = line_cache_key(engine_name, language, voice, text, speed, seed)
        path = os.path.join(lines_dir, f"{ln['id']}.wav")
        key_path = path + ".key"
        reused = False
        if os.path.exists(path) and _read_text(key_path) == key:
            audio, _ = read_wav(path)
            reused = True
        else:
            audio = engine.synth(text, prompts[ln["voice"]], voice, language, speed, seed)
            native_sr = int(getattr(engine, "sample_rate", sr))
            audio = trim_silence(resample_linear(audio, native_sr, sr), sr)
            peak = float(np.max(np.abs(audio))) if audio.size else 0.0
            if peak > 0.99:
                audio = audio * (0.99 / peak)
            write_wav(path, audio, sr)
            with open(key_path, "w") as f:
                f.write(key)
        manifest_lines.append(
            {
                "id": ln["id"],
                "voice": ln["voice"],
                "text": ln["text"],
                "tts_text": text,
                "speed": speed,
                "seed": seed,
                "samples": int(audio.size),
                "seconds": round(audio.size / sr, 3),
                "paragraph_end": bool(ln.get("paragraph_end")),
                "pause_after_ms": ln.get("pause_after_ms"),
                "reused": reused,
            }
        )
        elapsed = time.time() - started
        log(f"{idx}/{len(lines)} {ln['voice']}: {audio.size / sr:.1f}s audio ({elapsed:.0f}s elapsed)")

    # Timing only: the chapter and the per-voice stems are assembled on the
    # server from lines/ (orchestrate.py), so only one copy of the audio
    # travels back through the Colab file API.
    cursor = 0
    for i, m in enumerate(manifest_lines):
        m["start"] = round(cursor / sr, 3)
        cursor += m["samples"]
        if i == len(manifest_lines) - 1:
            m["gap_after_ms"] = 0
            break
        if m["pause_after_ms"] is not None:
            gap = int(m["pause_after_ms"])
        elif m["paragraph_end"]:
            gap = gaps["paragraph"]
        elif manifest_lines[i + 1]["voice"] != m["voice"]:
            gap = gaps["speaker_change"]
        else:
            gap = gaps["line"]
        m["gap_after_ms"] = gap
        cursor += int(sr * gap / 1000)
    total = cursor

    manifest = {
        "title": job.get("title", ""),
        "engine": engine_name,
        "language": language,
        "stress_mode": stress_mode,
        "sample_rate": sr,
        "seconds": round(total / sr, 3),
        "render_seconds": round(time.time() - started, 1),
        "voices": {n: {"seed": voices[n].get("seed"), "ref_text": voices[n]["ref_text"]} for n in used_voices},
        "lines": manifest_lines,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    log(f"chapter done: {total / sr / 60:.1f} min of audio in {manifest['render_seconds']:.0f}s")
    return manifest


def zip_dir(src: str, dst_zip: str) -> None:
    with zipfile.ZipFile(dst_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(src):
            for name in files:
                if name.endswith(".key") or name.startswith("."):
                    continue
                full = os.path.join(root, name)
                z.write(full, os.path.relpath(full, src))


def run_one(job_zip: str, out_zip: str, workdir: str, engine_override: Optional[str] = None) -> str:
    """Render one job archive into one output archive. Returns the job id."""
    # Never let the server download a previous chapter's archive by mistake.
    if os.path.exists(out_zip):
        os.remove(out_zip)
    job_dir = os.path.join(workdir, "job")
    out_dir = os.path.join(workdir, "out")
    # Keep earlier line renders in out_dir: a re-run after a crash skips them.
    if os.path.isdir(job_dir):
        shutil.rmtree(job_dir)
    if os.path.isdir(job_zip):  # already unpacked (Kaggle datasets)
        shutil.copytree(job_zip, job_dir)
    else:
        os.makedirs(job_dir)
        with zipfile.ZipFile(job_zip) as z:
            z.extractall(job_dir)
    with open(os.path.join(job_dir, "job.json"), encoding="utf-8") as f:
        job_id = json.load(f).get("job_id", "")
    marker = os.path.join(out_dir, ".job_id")
    if os.path.isdir(out_dir) and _read_text(marker) != job_id:
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    with open(marker, "w") as f:
        f.write(job_id)

    render_job(job_dir, out_dir, engine_override)
    zip_dir(out_dir, out_zip)
    log(f"wrote {out_zip}")
    return job_id


def run_batch(pattern: str, out_root: str, workdir: str, engine_override: Optional[str] = None) -> List[Dict[str, Any]]:
    """Kaggle mode: render every job_NN.zip found, one out_NN.zip each.

    One failing chapter does not stop the others; batch_report.json says
    which ones finished, and is rewritten after every chapter.
    """
    import glob
    import traceback

    # Kaggle unpacks .zip files in a dataset, so job_001.zip arrives as a
    # job_001/ folder. Take both forms; a zip wins if both are present.
    found = {}
    for path in glob.glob(pattern, recursive=True):
        found[os.path.basename(path)[:-4] if path.endswith(".zip") else os.path.basename(path)] = path
    dir_pattern = pattern[:-4] if pattern.endswith(".zip") else pattern
    for path in glob.glob(dir_pattern, recursive=True):
        if os.path.isfile(os.path.join(path, "job.json")):
            found.setdefault(os.path.basename(path.rstrip("/")), path)
    jobs = [found[k] for k in sorted(found)]
    log(f"batch: {len(jobs)} job(s) matching {pattern}")
    os.makedirs(out_root, exist_ok=True)
    report: List[Dict[str, Any]] = []
    for job in jobs:
        name = os.path.basename(job.rstrip("/"))
        if not name.endswith(".zip"):
            name += ".zip"
        out_name = "out_" + name[len("job_"):] if name.startswith("job_") else "out_" + name
        entry: Dict[str, Any] = {"job": name, "out": out_name}
        try:
            entry["job_id"] = run_one(job, os.path.join(out_root, out_name), workdir, engine_override)
            entry["ok"] = True
        except (Exception, SystemExit) as e:  # SystemExit: render_job's input checks
            entry["ok"] = False
            entry["error"] = f"{type(e).__name__}: {e}"
            entry["traceback"] = traceback.format_exc()[-3000:]
            log(f"FAILED {name}: {entry['error']}")
        report.append(entry)
        with open(os.path.join(out_root, "batch_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        if entry.get("error", "").startswith("NoInternet"):
            log("no internet: skipping the remaining jobs")
            break
    return report


def main() -> None:
    engine_override = os.environ.get("AUDIOBOOK_ENGINE") or None
    batch_glob = os.environ.get("AUDIOBOOK_BATCH_GLOB")
    if batch_glob:
        import tempfile

        run_batch(
            batch_glob,
            os.environ.get("AUDIOBOOK_BATCH_OUT", "/kaggle/working"),
            os.environ.get("AUDIOBOOK_WORKDIR", os.path.join(tempfile.gettempdir(), "audiobook_job")),
            engine_override,
        )
        return

    job_id = run_one(
        os.environ.get("AUDIOBOOK_JOB", "/content/audiobook_job.zip"),
        os.environ.get("AUDIOBOOK_OUT", "/content/audiobook_out.zip"),
        os.environ.get("AUDIOBOOK_WORKDIR", "/content/audiobook_job"),
        engine_override,
    )
    # `colab exec` exits 0 even when the kernel raised, so the server looks
    # for this exact line to know the chapter really finished.
    print(f"{DONE_MARKER} {job_id}", flush=True)


if __name__ == "__main__":
    main()
