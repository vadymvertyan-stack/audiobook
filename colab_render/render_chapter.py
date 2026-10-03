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


def trim_silence(audio: np.ndarray, sr: int, threshold: float = 0.002,
                 lead_ms: int = 60, tail_ms: int = 150) -> np.ndarray:
    # -54 dBFS and a generous tail: a final "с", "т" or breathy vowel is
    # quiet, and cutting at -40 dB swallowed it.
    if audio.size == 0:
        return audio
    loud = np.flatnonzero(np.abs(audio) > threshold)
    if loud.size == 0:
        return audio
    start = max(0, loud[0] - int(sr * lead_ms / 1000))
    end = min(audio.size, loud[-1] + int(sr * tail_ms / 1000) + 1)
    return audio[start:end]


def finish_tail(audio: np.ndarray, sr: int, fade_ms: int = 90, pad_ms: int = 150,
                floor: float = 0.002, loud: float = 0.0056) -> np.ndarray:
    """Let a line end softly. OmniVoice often stops while the voice is still
    at -25..-40 dB and drops to silence within one codec frame, which sounds
    clipped. Find where the voice really stops (last sample above -54 dB);
    if the 20 ms before it are still loud (above -45 dB), fade the last
    `fade_ms` before that point down to silence, then pad real silence."""
    if audio.size == 0:
        return audio
    audio = audio.astype(np.float32, copy=True)
    voiced = np.flatnonzero(np.abs(audio) > floor)
    if voiced.size:
        stop = int(voiced[-1]) + 1
        before = audio[max(0, stop - int(sr * 0.02)):stop]
        if before.size and float(np.max(np.abs(before))) > loud:
            n = min(stop, int(sr * fade_ms / 1000))
            # Slow at first, so the final consonant survives, then down to zero.
            curve = np.cos(np.linspace(0.0, np.pi / 2, n)) ** 2
            audio[stop - n:stop] *= curve.astype(np.float32)
            audio[stop:] = 0.0
    return np.concatenate([audio, np.zeros(int(sr * pad_ms / 1000), dtype=np.float32)])


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

    def repl(m: "re.Match[str]") -> str:
        word = m.group(0)
        new = lowered.get(word.lower(), word)
        # keep a sentence-initial capital: "Кобуры" -> "Кобуры́", not "кобуры́"
        if word[:1].isupper() and new[:1].islower() and not word.isupper():
            new = new[:1].upper() + new[1:]
        return new

    return pattern.sub(repl, text)


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
    if not packages:
        return
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


# Fallback only: orchestrate.py normally sends every line's pause.
DEFAULT_GAPS_MS = {"line": 550, "speaker_change": 650, "paragraph": 1100}


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
        # OmniVoice fades the last 100 ms of every line, which eats a soft final
        # consonant; keep only a click-guard fade. Edge silence is ours to set.
        self.options = {"fade_duration": 0.02, "pad_duration": 0.0}
        self.options.update(options or {})
        self.check_ref = bool(self.options.pop("check_ref", True))
        # OmniVoice sizes each line from the reference's chars/second and the
        # model stops when that budget runs out, sometimes inside the last
        # word. Stretching the whole budget (headroom) only slowed the speech
        # down; instead the line gets a short extra tail of time on top of
        # the estimate (6% of it, 0.15-0.4 s), and the trim removes what is left.
        self.headroom = float(self.options.pop("headroom", 1.0))
        self.tail = float(self.options.pop("tail", 0.06))

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
        speed = speed / self.headroom
        duration = self._duration_with_tail(text, prompt, speed)
        if duration:
            kwargs["duration"] = duration
        elif abs(speed - 1.0) > 1e-3:
            kwargs["speed"] = speed
        audio = self.model.generate(text=text, voice_clone_prompt=prompt, **kwargs)
        return to_mono_float(audio[0] if isinstance(audio, (list, tuple)) else audio)


    def _duration_with_tail(self, text: str, prompt: Any, speed: float) -> Optional[float]:
        if self.tail <= 0:
            return None
        try:
            frames = self.model._estimate_target_tokens(text, prompt.ref_text, prompt.ref_audio_tokens.size(-1),
                                                        speed=speed)
            seconds = frames / float(self.model.audio_tokenizer.config.frame_rate)
        except Exception as e:  # an OmniVoice version without these internals
            log(f"duration estimate unavailable ({e}); using speed only")
            self.tail = 0.0
            return None
        return seconds + min(0.4, max(0.15, seconds * self.tail))


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

def line_cache_key(engine: str, language: str, voice: Dict[str, Any], text: str, speed: float, seed: int,
                   options: Optional[Dict[str, Any]] = None) -> str:
    # Engine options (tail, num_step, ...) change the audio too.
    payload = json.dumps(
        [engine, language, voice.get("ref_text"), voice.get("ref_sha256"), text, round(speed, 4), seed]
        + ([options] if options else []),
        ensure_ascii=False, sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def render_job(job_dir: str, out_dir: str, engine_override: Optional[str] = None) -> Dict[str, Any]:
    with open(os.path.join(job_dir, "job.json"), encoding="utf-8") as f:
        job = json.load(f)

    engine_name = engine_override or job.get("engine", "omnivoice")
    language = job.get("language", "")
    stress_mode = job.get("stress_mode", "strip")
    lexicon = job.get("lexicon", {}) or {}
    gaps = dict(DEFAULT_GAPS_MS)
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
        speed = float(voice.get("speed") or 1.0) * float(ln.get("speed") or 1.0)
        volume = float(ln.get("volume") or 1.0)
        seed = int(voice.get("seed", 1234))
        text = apply_stress_mode(apply_lexicon(ln["text"], lexicon), stress_mode)
        key = line_cache_key(engine_name, language, voice, text, speed, seed, job.get("engine_options") or None) + (f"|v{volume}" if volume != 1.0 else "")
        path = os.path.join(lines_dir, f"{ln['id']}.wav")
        key_path = path + ".key"
        reused = False
        if os.path.exists(path) and _read_text(key_path) == key:
            audio, _ = read_wav(path)
            reused = True
        else:
            audio = engine.synth(text, prompts[ln["voice"]], voice, language, speed, seed)
            native_sr = int(getattr(engine, "sample_rate", sr))
            audio = finish_tail(trim_silence(resample_linear(audio, native_sr, sr), sr), sr)
            audio = audio * volume
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
                "volume": volume,
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


# ---------------------------------------------------------------------------
# Casting: design each character's voice and its emotions before rendering
# ---------------------------------------------------------------------------
#
# A cast job (job.json with "kind": "cast") holds characters (a voice
# description, or an existing reference wav) and emotions (a delivery style
# plus a sample sentence that fits it). For every character the GPU:
#   1. designs the calm voice from the description, N takes, and keeps the
#      most typical one (closest to the others): that is the character's timbre;
#   2. designs every other emotion N times with "description + style" and
#      keeps the take whose speaker embedding is closest to that timbre,
#      because VoiceDesign drifts a little between generations;
#   3. checks every take with Whisper, so a clip that does not say its text
#      never becomes a reference (a mismatch leaks into every rendered line).
# The result is cast/<character>/<emotion>.wav plus voices.json entries
# "Name" (calm) and "Name:emotion", ready for the renderer.

DESIGN_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
EMBED_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"  # only its speaker encoder is used
ASR_MODEL = "openai/whisper-large-v3-turbo"
LOW_SIMILARITY = 0.75
TOO_ALIKE = 0.9  # two different characters this close in timbre are hard to tell apart


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b) / den if den else 0.0


def _clean_pool(takes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Takes Whisper heard nearly word for word, else ones that pass the
    reference check, else all. A slurred word that passes here can still fail
    the stricter check in the render."""
    return ([t for t in takes if t["ok"] and t.get("match", 1.0) >= 0.95]
            or [t for t in takes if t["ok"]] or takes)


def pick_typical(takes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The take closest on average to the other takes (clean ones first): an
    outlier timbre loses even if it came out first."""
    pool = _clean_pool(takes)
    if len(pool) == 1:
        return pool[0]
    best, best_score = pool[0], -2.0
    for t in pool:
        score = float(np.mean([cosine(t["emb"], o["emb"]) for o in pool if o is not t]))
        t["typicality"] = round(score, 4)
        if score > best_score:
            best, best_score = t, score
    return best


def pick_closest(takes: List[Dict[str, Any]], timbre: np.ndarray) -> Dict[str, Any]:
    for t in takes:
        t["similarity"] = round(cosine(t["emb"], timbre), 4)
    return max(_clean_pool(takes), key=lambda t: t["similarity"])


class DummyDesigner:
    """For tests: a tone whose pitch depends on the seed; 'Whisper' hears the text."""

    sample_rate = 24000

    def __init__(self, options: Optional[Dict[str, Any]] = None):
        self._last = ""

    def design(self, text: str, instruct: str, language: str, seed: int) -> np.ndarray:
        self._last = text
        freq = 120.0 + (seed % 5) * 15.0
        t = np.arange(int(self.sample_rate * 2.0)) / self.sample_rate
        return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    def embed(self, audio: np.ndarray) -> np.ndarray:
        spec = np.abs(np.fft.rfft(audio[: self.sample_rate]))
        peak = int(np.argmax(spec))
        return np.exp(-0.5 * ((np.arange(400) - peak) / 20.0) ** 2)

    def transcribe(self, audio: np.ndarray, language: str) -> str:
        return self._last


class QwenDesigner:
    """Qwen3-TTS VoiceDesign for the takes, the Qwen3-TTS Base speaker encoder
    to compare timbres, Whisper to check what each take actually says."""

    sample_rate = 24000

    def __init__(self, options: Optional[Dict[str, Any]] = None):
        options = options or {}
        try:
            import qwen_tts  # noqa: F401
        except ImportError:
            _pip_install("qwen-tts")
        import torch
        from qwen_tts import Qwen3TTSModel

        self.torch = torch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        log(f"loading {options.get('design_model', DESIGN_MODEL)} on {device}")
        self.designer = Qwen3TTSModel.from_pretrained(options.get("design_model", DESIGN_MODEL),
                                                      device_map=device, dtype=dtype)
        log(f"loading speaker encoder from {options.get('embed_model', EMBED_MODEL)}")
        self.encoder = Qwen3TTSModel.from_pretrained(options.get("embed_model", EMBED_MODEL),
                                                     device_map=device, dtype=dtype)
        from transformers import pipeline

        self.asr = pipeline("automatic-speech-recognition", model=options.get("asr_model", ASR_MODEL),
                            torch_dtype=dtype, device=device)
        self.gen = {k: options[k] for k in ("temperature", "top_p", "top_k") if k in options}

    def design(self, text: str, instruct: str, language: str, seed: int) -> np.ndarray:
        _seed_everything(seed)
        wavs, sr = self.designer.generate_voice_design(text=text, instruct=instruct, language=language, **self.gen)
        return resample_linear(to_mono_float(wavs[0]), int(sr), self.sample_rate)

    def embed(self, audio: np.ndarray) -> np.ndarray:
        with self.torch.inference_mode():
            emb = self.encoder.model.extract_speaker_embedding(audio=audio.astype(np.float32), sr=self.sample_rate)
        return emb.float().cpu().numpy()

    def transcribe(self, audio: np.ndarray, language: str) -> str:
        res = self.asr({"raw": audio.astype(np.float32), "sampling_rate": self.sample_rate},
                       generate_kwargs={"language": language.lower(), "task": "transcribe"})
        return str(res.get("text", "")).strip()


class DummyVC:
    """For tests: 'converts' by returning the source unchanged."""

    def convert(self, source: str, target: str, workdir: str) -> Tuple[np.ndarray, int]:
        return read_wav(source)


class SeedVC:
    """Seed-VC (github.com/Plachtaa/seed-vc) v1, zero-shot timbre conversion:
    words, rhythm and emotion come from the source take, the voice itself from
    the character's calm reference. It pins transformers 4.46, so it lives in
    its own venv on top of the system torch and runs as a subprocess."""

    REPO = "https://github.com/Plachtaa/seed-vc"
    SKIP = re.compile(r"^(-|torch|torchvision|torchaudio)|gradio|FreeSimpleGUI|sounddevice", re.I)

    def __init__(self, options: Optional[Dict[str, Any]] = None):
        options = options or {}
        import tempfile

        self.steps = int(options.get("vc_steps", 30))
        self.cfg = float(options.get("vc_cfg", 0.7))
        self.repo = options.get("vc_dir") or os.path.join(tempfile.gettempdir(), "seed-vc")
        self.python = os.path.join(self.repo, ".venv", "bin", "python")
        if os.path.exists(self.python):
            return
        _pip_install()  # only checks for internet
        if not os.path.isdir(os.path.join(self.repo, ".git")):
            log("cloning Seed-VC")
            subprocess.run(["git", "clone", "--depth", "1", self.REPO, self.repo], check=True)
        subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", os.path.join(self.repo, ".venv")],
                       check=True)
        with open(os.path.join(self.repo, "requirements.txt"), encoding="utf-8") as f:
            reqs = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#") and not self.SKIP.search(ln)]
        req = os.path.join(self.repo, "req-kaggle.txt")
        with open(req, "w", encoding="utf-8") as f:
            f.write("\n".join(reqs) + "\n")
        log(f"installing Seed-VC requirements ({len(reqs)} packages, system torch kept)")
        subprocess.run([self.python, "-m", "pip", "install", "-q", "-r", req], check=True)

    def convert(self, source: str, target: str, workdir: str) -> Tuple[np.ndarray, int]:
        import glob

        out = os.path.join(workdir, "vc_out")
        shutil.rmtree(out, ignore_errors=True)
        os.makedirs(out)
        cmd = [self.python, "inference.py", "--source", os.path.abspath(source), "--target", os.path.abspath(target),
               "--output", out, "--diffusion-steps", str(self.steps), "--length-adjust", "1.0",
               "--inference-cfg-rate", str(self.cfg), "--f0-condition", "False", "--auto-f0-adjust", "False",
               "--semi-tone-shift", "0", "--fp16", "True"]
        res = subprocess.run(cmd, cwd=self.repo, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Seed-VC failed: {(res.stderr or res.stdout)[-1500:]}")
        files = glob.glob(os.path.join(out, "*.wav"))
        if not files:
            raise RuntimeError(f"Seed-VC wrote no file: {res.stdout[-800:]}")
        import soundfile

        audio, sr = soundfile.read(files[0], dtype="float32")
        return to_mono_float(audio), int(sr)


def get_vc(name: str, options: Dict[str, Any]) -> Any:
    if name in ("", "none", None):
        return None
    if name == "dummy":
        return DummyVC()
    if name == "seed-vc":
        return SeedVC(options)
    raise SystemExit(f"unknown voice conversion: {name}")


def get_designer(name: str, options: Dict[str, Any]) -> Any:
    if name == "dummy":
        return DummyDesigner(options)
    if name == "qwen":
        return QwenDesigner(options)
    raise SystemExit(f"unknown design engine: {name}")


def cast_voices(job_dir: str, out_dir: str) -> Dict[str, Any]:
    with open(os.path.join(job_dir, "job.json"), encoding="utf-8") as f:
        job = json.load(f)
    language = job.get("language", "Russian")
    takes_n = max(1, int(job.get("takes", 3)))
    emotions: Dict[str, Dict[str, str]] = job["emotions"]
    designer = get_designer(job.get("design_engine", "qwen"), dict(job.get("design_options") or {}))
    vc = get_vc(job.get("vc", "seed-vc"), dict(job.get("design_options") or {}))
    sr = designer.sample_rate
    vc_work = os.path.join(out_dir, ".vc_work")

    def finish_take(audio: np.ndarray, text: str, path: str, seed: Optional[int]) -> Dict[str, Any]:
        audio = trim_silence(audio, sr)
        audio = np.concatenate([np.zeros(int(sr * 0.1), np.float32), audio.astype(np.float32),
                                np.zeros(int(sr * 0.2), np.float32)])
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 0:
            audio = audio * (0.9 / peak)
        heard = designer.transcribe(audio, language)
        problem = reference_mismatch(text, heard)
        import difflib

        match = round(difflib.SequenceMatcher(None, _words(text), _words(heard)).ratio(), 3)
        write_wav(path, audio, sr)
        return {"file": os.path.relpath(path, out_dir), "seed": seed, "heard": heard, "match": match,
                "ok": problem is None, "problem": problem, "seconds": round(audio.size / sr, 2),
                "emb": designer.embed(audio)}

    def take(text: str, instruct: str, seed: int, path: str) -> Dict[str, Any]:
        return finish_take(designer.design(text, instruct, language, seed), text, path, seed)

    def given(src: str, text: str, path: str) -> Dict[str, Any]:
        audio, a_sr = read_wav(src)
        return finish_take(resample_linear(audio, a_sr, sr), text, path, None)

    def to_timbre(takes: List[Dict[str, Any]], timbre: np.ndarray, timbre_wav: str, text: str,
                  path: str) -> Optional[Dict[str, Any]]:
        """Convert the best clean takes to the character's own voice until one
        still says its text; None if none does (the line then uses calm)."""
        pick_closest(takes, timbre)
        pool = sorted(_clean_pool(takes), key=lambda t: -t["similarity"])
        for t in pool[:2]:
            os.makedirs(vc_work, exist_ok=True)
            audio, a_sr = vc.convert(os.path.join(out_dir, t["file"]), timbre_wav, vc_work)
            conv = finish_take(resample_linear(audio, a_sr, sr), text, path, t["seed"])
            conv["similarity"] = round(cosine(conv["emb"], timbre), 4)
            conv["converted_from"] = t["file"]
            if conv["ok"]:
                return conv
            log(f"converted {t['file']} no longer says its text ({conv['problem']}); trying the next take")
        return None

    report: Dict[str, Any] = {}
    voices: Dict[str, Dict[str, Any]] = {}
    for c_idx, (name, ch) in enumerate(job["characters"].items()):
        folder = os.path.join(out_dir, "cast", re.sub(r"[^\w-]+", "_", name).strip("_") or f"voice{c_idx}")
        cand_dir = os.path.join(folder, "takes")
        os.makedirs(cand_dir, exist_ok=True)
        description = (ch.get("description") or "").strip()
        base_seed = int(ch.get("seed", 1000 + 97 * c_idx))
        entry: Dict[str, Any] = {"description": description, "emotions": {}}
        if not description:
            raise SystemExit(f"character '{name}' needs a description of the voice")
        if ch.get("ref_audio"):
            # An existing voice (e.g. the narrator): its emotions must match it.
            timbre_wav = os.path.join(job_dir, ch["ref_audio"])
            ref, ref_sr = read_wav(timbre_wav)
            timbre = designer.embed(resample_linear(ref, ref_sr, sr))
            entry["timbre"] = "given reference"
            wanted = [e for e in emotions if e != "calm"]
        else:
            calm = emotions.get("calm") or next(iter(emotions.values()))
            takes = [take(calm["text"], f"{description}. {calm.get('style', '')}".strip(". "),
                          base_seed + k, os.path.join(cand_dir, f"calm_{k + 1}.wav")) for k in range(takes_n)]
            best = pick_typical(takes)
            timbre = best["emb"]
            final = os.path.join(folder, "calm.wav")
            shutil.copy(os.path.join(out_dir, best["file"]), final)
            timbre_wav = final
            entry["emotions"]["calm"] = _cast_summary(best, takes, final, out_dir, calm["text"])
            voices[name] = {"ref_audio": os.path.relpath(final, out_dir), "ref_text": calm["text"], "seed": base_seed}
            log(f"cast '{name}' calm: take {takes.index(best) + 1}/{takes_n}, heard: {best['heard']}")
            wanted = [e for e in emotions if e != "calm"]
        if ch.get("only_emotions"):
            wanted = [e for e in wanted if e in ch["only_emotions"]]
        sources = ch.get("emotion_sources") or {}
        if sources:
            wanted = [e for e in wanted if e in sources]
        for e_idx, emo in enumerate(wanted, 1):
            spec = emotions[emo]
            if emo in sources:
                # An emotion take chosen earlier: only convert it to the voice.
                takes = [given(os.path.join(job_dir, sources[emo]), spec["text"],
                               os.path.join(cand_dir, f"{emo}_given.wav"))]
            else:
                instruct = f"{description}. {spec.get('style', emo)}" if description else spec.get("style", emo)
                takes = [take(spec["text"], instruct, base_seed + 100 * e_idx + k,
                              os.path.join(cand_dir, f"{emo}_{k + 1}.wav")) for k in range(takes_n)]
            final = os.path.join(folder, f"{emo}.wav")
            if vc is not None:
                best = to_timbre(takes, timbre, timbre_wav, spec["text"], os.path.join(cand_dir, f"{emo}_vc.wav"))
                if best is None:
                    log(f"cast '{name}' {emo}: no take survived conversion; lines with it use the calm voice")
                    entry["emotions"][emo] = {"warning": "dropped: no take said its text after conversion",
                                              "takes": [{k: v for k, v in t.items() if k != "emb"} for t in takes]}
                    continue
            else:
                best = pick_closest(takes, timbre)
            shutil.copy(os.path.join(out_dir, best["file"]), final)
            entry["emotions"][emo] = _cast_summary(best, takes, final, out_dir, spec["text"])
            voices[f"{name}:{emo}"] = {"ref_audio": os.path.relpath(final, out_dir), "ref_text": spec["text"],
                                       "seed": base_seed}
            flag = "" if best["similarity"] >= LOW_SIMILARITY else "  <- timbre drifted, listen"
            log(f"cast '{name}' {emo}: similarity {best['similarity']:.2f}{flag}, heard: {best['heard']}")
        entry["calm_emb"] = np.asarray(timbre, dtype=np.float64).ravel().tolist()
        report[name] = entry

    # Characters must also sound different from each other.
    names = [n for n in report if "calm_emb" in report[n]]
    pairs = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            sim = round(cosine(report[a]["calm_emb"], report[b]["calm_emb"]), 4)
            pairs[f"{a} / {b}"] = sim
            if sim >= TOO_ALIKE:
                log(f"cast '{a}' and '{b}' sound alike (similarity {sim:.2f}); make one description more distinct")
    for n in names:
        report[n].pop("calm_emb")
    report["_between_characters"] = dict(sorted(pairs.items(), key=lambda kv: -kv[1]))
    shutil.rmtree(vc_work, ignore_errors=True)
    with open(os.path.join(out_dir, "cast_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "voices.json"), "w", encoding="utf-8") as f:
        json.dump(voices, f, ensure_ascii=False, indent=2)
    return report


def _cast_summary(best: Dict[str, Any], takes: List[Dict[str, Any]], final: str, out_dir: str,
                  text: str) -> Dict[str, Any]:
    def clean(t: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in t.items() if k != "emb"}

    return {"file": os.path.relpath(final, out_dir), "text": text, "chosen": clean(best),
            "takes": [clean(t) for t in takes],
            "warning": None if best["ok"] else "no take said its text cleanly; listen before using"}


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
        job_meta = json.load(f)
    job_id = job_meta.get("job_id", "")
    marker = os.path.join(out_dir, ".job_id")
    if os.path.isdir(out_dir) and _read_text(marker) != job_id:
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    with open(marker, "w") as f:
        f.write(job_id)

    if job_meta.get("kind") == "cast":
        cast_voices(job_dir, out_dir)
    else:
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

    # Jobs arrive as job_NNN.job (a zip under another name, so Kaggle leaves
    # it alone), as job_NNN.zip, or as a job_NNN/ folder that Kaggle unpacked.
    # A zip or .job wins over a folder with the same name.
    if pattern.endswith(".zip"):
        pattern = pattern[:-4]
    found: Dict[str, str] = {}
    for path in glob.glob(pattern + "*", recursive=True):
        base = os.path.basename(path.rstrip("/"))
        stem, ext = os.path.splitext(base)
        if os.path.isfile(path) and ext in (".zip", ".job"):
            found[stem] = path
        elif os.path.isdir(path) and os.path.isfile(os.path.join(path, "job.json")):
            found.setdefault(base, path)
    jobs = [found[k] for k in sorted(found)]
    log(f"batch: {len(jobs)} job(s) matching {pattern}")
    os.makedirs(out_root, exist_ok=True)
    report: List[Dict[str, Any]] = []
    for job in jobs:
        name = os.path.splitext(os.path.basename(job.rstrip("/")))[0] + ".zip"
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
