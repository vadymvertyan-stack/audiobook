"""
Voice Library All-in-One - Single file for Google Colab.

Combines:
- voicebox_adapter_colab.py (VoiceClone/VoiceDesign engines)
- voice_library_manager.py (VoiceLibraryManager)
- voice_library_colab_setup.py (Colab integration)

Features:
- Auto-installation of all dependencies
- Telegram bot integration for audio delivery
- Voice cloning and design with Qwen3-TTS

Setup:
1. Add secrets in Colab (🔑 icon in left sidebar):
   - TELEGRAM_BOT_TOKEN - get from @BotFather
   - TELEGRAM_CHAT_ID - get from @userinfobot

2. Upload this file to Colab and run:
   from voice_library_all_in_one import init_voice_library_colab, get_voice_library
   library = init_voice_library_colab()

3. Or just run the cell - it will auto-initialize and send test audio to Telegram

Usage:
    library = get_voice_library()
    voice_prompt, is_new = library.get_voice("narrator", "male_deep")
    
    # Generate audio
    result = generate_audio_with_library(
        text="Привіт! Це тестовий голос.",
        character_name='narrator',
        voice_preset='male_deep'
    )

Version: 1.3.1
Created: 2026-02-21
Author: VIBEMODLY Team

Changelog:
- v1.3.1: Fixed ALL_ATTENTION_FUNCTIONS import error - transformers 4.46.0 was too old
          Now uses transformers 4.47.0 which has ALL_ATTENTION_FUNCTIONS
          Required version range: 4.45.0 - 4.99.x (must have ALL_ATTENTION_FUNCTIONS, must NOT be 5.x)
- v1.3.0: Fixed transformers version conflict - now installs 4.46.0 BEFORE qwen-tts
          with --no-deps flag to prevent qwen-tts from overriding transformers
- v1.2.0: Added Telegram bot integration for audio delivery
- v1.1.0: Fixed chat_template parameter error in model loading
- v1.0.0: Initial release
"""

# =============================================================================
# VERSION COMPATIBILITY CHECK (MUST BE FIRST!)
# =============================================================================
# Ця перевірка має бути на самому початку, ДО будь-яких імпортів transformers
# щоб уникнути кешування модулів

# qwen-tts вимагає ALL_ATTENTION_FUNCTIONS яка з'явилася в transformers 4.45.0+
# Але transformers 5.0.0+ несумісна з qwen-tts (помилка chat_template)
# Тому потрібна версія в межах 4.45.0 <= version < 5.0.0
# Використовуємо 4.47.0 як безпечну версію

COMPATIBLE_TRANSFORMERS_VERSION = "4.47.0"
MIN_TRANSFORMERS_VERSION = "4.45.0"  # Мінімальна версія з ALL_ATTENTION_FUNCTIONS
MAX_TRANSFORMERS_VERSION = "4.99.0"  # Максимум до 5.0.0

def _check_transformers_version() -> tuple:
    """
    Перевіряє версію transformers через importlib.metadata (БЕЗ імпорту модуля).
    Повертає (current_version, is_compatible, version_tuple).
    """
    try:
        import importlib.metadata
        try:
            current_version = importlib.metadata.version('transformers')
        except importlib.metadata.PackageNotFoundError:
            return None, True, (0, 0, 0)  # Не встановлено - сумісно (буде встановлено)
    except ImportError:
        try:
            import pkg_resources
            current_version = pkg_resources.get_distribution('transformers').version
        except:
            return None, True, (0, 0, 0)
    
    def parse_version(v):
        try:
            return tuple(int(x) for x in v.split('.')[:3])
        except:
            return (0, 0, 0)
    
    installed = parse_version(current_version)
    min_ver = parse_version(MIN_TRANSFORMERS_VERSION)
    max_ver = parse_version(MAX_TRANSFORMERS_VERSION)
    
    # Сумісна якщо: min_ver <= installed < max_ver (тобто 4.45.x - 4.99.x)
    is_compatible = installed >= min_ver and installed < (5, 0, 0)
    
    return current_version, is_compatible, installed


def _install_compatible_transformers() -> bool:
    """
    Встановлює сумісну версію transformers ДО встановлення qwen-tts.
    Це запобігає конфліктам версій.
    """
    import subprocess
    import sys
    
    current_version, is_compatible, installed_tuple = _check_transformers_version()
    
    if is_compatible and current_version:
        print(f"[VoiceLibrary] ✓ transformers {current_version} is compatible (4.45+ required, <5.0)", flush=True)
        return True
    
    if current_version:
        if installed_tuple[0] >= 5:
            print(f"[VoiceLibrary] ⚠ transformers {current_version} is 5.x+ (incompatible with qwen-tts)", flush=True)
        else:
            print(f"[VoiceLibrary] ⚠ transformers {current_version} is too old (need 4.45+)", flush=True)
    else:
        print(f"[VoiceLibrary] transformers not installed", flush=True)
    
    print(f"[VoiceLibrary] Installing transformers {COMPATIBLE_TRANSFORMERS_VERSION}...", flush=True)
    
    try:
        # Видаляємо стару версію
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'uninstall', '-y', 'transformers'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # Встановлюємо сумісну версію
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '-q', 
             f'transformers=={COMPATIBLE_TRANSFORMERS_VERSION}'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # Перевіряємо результат
        new_version, _, _ = _check_transformers_version()
        print(f"[VoiceLibrary] ✓ transformers {new_version} installed", flush=True)
        return True
        
    except Exception as e:
        print(f"[VoiceLibrary] ✗ Failed to install transformers: {e}", flush=True)
        return False


def _verify_transformers_version() -> bool:
    """
    Перевіряє версію transformers після всіх встановлень.
    Якщо версія несумісна - виводить повідомлення про необхідність перезавантаження.
    """
    current_version, is_compatible, installed_tuple = _check_transformers_version()
    
    if is_compatible:
        if current_version:
            print(f"[VoiceLibrary] ✓ transformers version verified: {current_version}", flush=True)
        return True
    
    # Версія несумісна - потрібне перезавантаження kernel
    print("\n" + "="*70, flush=True)
    print("⚠️  KERNEL RESTART REQUIRED!", flush=True)
    print("="*70, flush=True)
    print(f"[VoiceLibrary] Current transformers: {current_version}", flush=True)
    print(f"[VoiceLibrary] Required: 4.45.0 - 4.99.x (got {current_version})", flush=True)
    print("", flush=True)
    print("The transformers version was changed but Python has cached imports.", flush=True)
    print("Please restart the Colab kernel and run this cell again:", flush=True)
    print("", flush=True)
    print("  1. Click 'Runtime' → 'Restart session' (or press Ctrl+M .)", flush=True)
    print("  2. Run this cell again", flush=True)
    print("="*70 + "\n", flush=True)
    
    return False


# Встановлюємо сумісну версію transformers ДО будь-яких інших дій
_install_compatible_transformers()


# =============================================================================
# AUTO-INSTALL DEPENDENCIES
# =============================================================================

def _install_dependencies() -> bool:
    """
    Автоматичне встановлення всіх необхідних залежностей.
    Працює в Google Colab та локальному середовищі.
    """
    import subprocess
    import sys
    
    # Сумісна версія transformers визначена на початку файлу
    # COMPATIBLE_TRANSFORMERS_VERSION = "4.46.0"
    
    dependencies = [
        'torch',
        'torchaudio',
        'numpy',
        'scipy',
        'librosa',
        'soundfile',
        'pydub',
        'accelerate',
        'pyTelegramBotAPI',  # Для відправки аудіо в Telegram
    ]
    
    # keras_nlp потрібен для transformers 4.44.0 (виправляє помилку BACKENDS_MAPPING)
    keras_packages = ['keras-nlp', 'keras']
    
    # Qwen-TTS встановлюється окремо (git install)
    qwen_tts_package = 'git+https://github.com/QwenLM/Qwen3-TTS.git'
    
    print("\n" + "="*60, flush=True)
    print("[VoiceLibrary] AUTO-INSTALLING DEPENDENCIES", flush=True)
    print("="*60, flush=True)
    
    installed = []
    failed = []
    
    for package in dependencies:
        try:
            print(f"[VoiceLibrary] Checking {package}...", flush=True)
            __import__(package.replace('-', '_'))
            print(f"[VoiceLibrary]   ✓ {package} already installed", flush=True)
        except ImportError:
            try:
                print(f"[VoiceLibrary]   Installing {package}...", flush=True)
                subprocess.check_call(
                    [sys.executable, '-m', 'pip', 'install', '-q', package],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                print(f"[VoiceLibrary]   ✓ {package} installed", flush=True)
                installed.append(package)
            except Exception as e:
                print(f"[VoiceLibrary]   ✗ Failed to install {package}: {e}", flush=True)
                failed.append(package)
    
    # Перевірка transformers (версія вже встановлена на початку скрипта)
    print(f"[VoiceLibrary] Checking transformers...", flush=True)
    current_version, is_compatible, _ = _check_transformers_version()
    if current_version:
        print(f"[VoiceLibrary]   Current transformers version: {current_version}", flush=True)
        if is_compatible:
            print(f"[VoiceLibrary]   ✓ transformers is compatible", flush=True)
        else:
            print(f"[VoiceLibrary]   ⚠ transformers version mismatch (will be verified later)", flush=True)
    else:
        print(f"[VoiceLibrary]   transformers not found", flush=True)
    
    # Встановлення keras_nlp для виправлення помилки BACKENDS_MAPPING
    # Це потрібно для transformers 4.44.0, які перевіряють keras_nlp backend
    print(f"[VoiceLibrary] Checking keras-nlp...", flush=True)
    try:
        import keras_nlp
        print(f"[VoiceLibrary]   ✓ keras-nlp already installed", flush=True)
    except ImportError:
        try:
            print(f"[VoiceLibrary]   Installing keras-nlp (required for transformers backend)...", flush=True)
            subprocess.check_call(
                [sys.executable, '-m', 'pip', 'install', '-q', 'keras-nlp'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print(f"[VoiceLibrary]   ✓ keras-nlp installed", flush=True)
            installed.append('keras-nlp')
        except Exception as e:
            print(f"[VoiceLibrary]   ⚠ Failed to install keras-nlp: {e}", flush=True)
            print(f"[VoiceLibrary]   Continuing without keras-nlp (may cause backend errors)", flush=True)
    
    # Встановлення Qwen-TTS
    # transformers вже встановлено правильно на початку скрипта (4.47.0)
    # qwen-tts може спробувати змінити версію, але ми перевіримо це пізніше
    try:
        print(f"[VoiceLibrary] Checking qwen-tts...", flush=True)
        from qwen_tts import Qwen3TTSModel
        print(f"[VoiceLibrary]   ✓ qwen-tts already installed", flush=True)
    except ImportError:
        try:
            print(f"[VoiceLibrary]   Installing qwen-tts from git...", flush=True)
            # Встановлюємо qwen-tts (він може спробувати змінити transformers)
            subprocess.check_call(
                [sys.executable, '-m', 'pip', 'install', '-q', qwen_tts_package],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print(f"[VoiceLibrary]   ✓ qwen-tts installed", flush=True)
            installed.append('qwen-tts')
        except Exception as e:
            print(f"[VoiceLibrary]   ✗ Failed to install qwen-tts: {e}", flush=True)
            failed.append('qwen-tts')
    except ValueError as e:
        # Якщо помилка BACKENDS_MAPPING все ще виникає
        if "keras_nlp" in str(e):
            print(f"[VoiceLibrary]   ⚠ keras_nlp backend error detected", flush=True)
            print(f"[VoiceLibrary]   Attempting to fix by installing keras-nlp...", flush=True)
            try:
                subprocess.check_call(
                    [sys.executable, '-m', 'pip', 'install', '-q', 'keras-nlp'],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                print(f"[VoiceLibrary]   ✓ keras-nlp installed, retrying qwen-tts import...", flush=True)
                installed.append('keras-nlp')
                # Повторна перевірка qwen-tts
                try:
                    from qwen_tts import Qwen3TTSModel
                    print(f"[VoiceLibrary]   ✓ qwen-tts now working!", flush=True)
                except Exception as e2:
                    print(f"[VoiceLibrary]   ✗ qwen-tts still failing: {e2}", flush=True)
                    failed.append('qwen-tts')
            except Exception as e2:
                print(f"[VoiceLibrary]   ✗ Failed to install keras-nlp: {e2}", flush=True)
                failed.append('qwen-tts')
        else:
            print(f"[VoiceLibrary]   ✗ qwen-tts import error: {e}", flush=True)
            failed.append('qwen-tts')
    
    print("\n[VoiceLibrary] Installation Summary:", flush=True)
    if installed:
        print(f"  Installed: {', '.join(installed)}", flush=True)
    if failed:
        print(f"  Failed: {', '.join(failed)}", flush=True)
    if not failed:
        print("  All dependencies installed successfully!", flush=True)
    print("="*60 + "\n", flush=True)
    
    # Перевіряємо версію transformers після всіх встановлень
    print("[VoiceLibrary] Verifying transformers version...", flush=True)
    _verify_transformers_version()
    
    return len(failed) == 0


# Автоматичне встановлення залежностей перед імпортом
_install_dependencies()

# =============================================================================
# PATCH: Виправлення помилки BACKENDS_MAPPING для відсутніх backend'ів
# =============================================================================
# Цей патч потрібен для transformers 4.44.0, які не мають записів для деяких
# backend'ів (keras_nlp, tensorflow_text, тощо) у BACKENDS_MAPPING,
# що викликає помилку при імпорті qwen_tts
try:
    from transformers.utils.import_utils import BACKENDS_MAPPING
    print("[DIAGNOSTIC] Applying BACKENDS_MAPPING patch for missing backends...", flush=True)
    
    # Список всіх можливих backend'ів, які можуть бути відсутніми
    missing_backends = [
        'keras_nlp',
        'tensorflow_text',
        'tensorflow',
        'keras',
        'jax',
        'flax',
        'tf',
        'torch',
    ]
    
    for backend_name in missing_backends:
        if backend_name not in BACKENDS_MAPPING:
            # Додаємо заглушку - вказуємо, що backend недоступний
            BACKENDS_MAPPING[backend_name] = (lambda: False, f"{backend_name} is not available")
            print(f"[DIAGNOSTIC]   Added patch for: {backend_name}", flush=True)
    
    print("[DIAGNOSTIC] BACKENDS_MAPPING patch applied successfully", flush=True)
except Exception as e:
    print(f"[DIAGNOSTIC] Warning: Could not apply BACKENDS_MAPPING patch: {e}", flush=True)


# =============================================================================
# DIAGNOSTIC: Early logging to catch import errors
# =============================================================================
print("[DIAGNOSTIC] Starting voice_library_all_in_one import...", flush=True)

# =============================================================================
# SECTION 1: IMPORTS
# =============================================================================

print("[DIAGNOSTIC] Importing abc...", flush=True)
from abc import ABC, abstractmethod

print("[DIAGNOSTIC] Importing os...", flush=True)
import os

print("[DIAGNOSTIC] Importing json...", flush=True)
import json

print("[DIAGNOSTIC] Importing hashlib...", flush=True)
import hashlib

print("[DIAGNOSTIC] Importing logging...", flush=True)
import logging

print("[DIAGNOSTIC] Importing datetime...", flush=True)
from datetime import datetime

print("[DIAGNOSTIC] Importing dataclasses...", flush=True)
from dataclasses import dataclass, field, asdict

print("[DIAGNOSTIC] Importing typing...", flush=True)
from typing import Optional, Dict, List, Any, Tuple

print("[DIAGNOSTIC] Importing pathlib...", flush=True)
from pathlib import Path

print("[DIAGNOSTIC] Importing enum...", flush=True)
from enum import Enum

print("[DIAGNOSTIC] Importing numpy...", flush=True)
try:
    import numpy as np
    print("[DIAGNOSTIC] numpy imported successfully", flush=True)
except ImportError as e:
    print(f"[DIAGNOSTIC] ERROR: numpy import failed: {e}", flush=True)
    print("[DIAGNOSTIC] Install numpy with: pip install numpy", flush=True)
    raise

print("[DIAGNOSTIC] All basic imports completed successfully!", flush=True)

logging.basicConfig(level=logging.INFO, format='[%(name)s] %(message)s')
logger = logging.getLogger('VoiceLibrary')


# =============================================================================
# SECTION 2: ENUMS AND CONSTANTS
# =============================================================================

class VoiceMode(Enum):
    VOICE_DESIGN = "voicedesign"
    VOICE_CLONE = "clone"
    
    @classmethod
    def from_string(cls, value: str) -> 'VoiceMode':
        value_lower = value.lower()
        if value_lower in ("voicedesign", "voice_design", "design"):
            return cls.VOICE_DESIGN
        elif value_lower in ("clone", "voice_clone"):
            return cls.VOICE_CLONE
        else:
            raise ValueError(f"Unknown voice mode: {value}")


class ValidationLevel(Enum):
    BASIC = 'basic'
    STANDARD = 'standard'
    STRICT = 'strict'


class ReferenceQualityMetrics:
    DURATION_OPTIMAL_MIN: float = 15.0
    DURATION_OPTIMAL_MAX: float = 20.0
    DURATION_ABSOLUTE_MIN: float = 10.0
    DURATION_ABSOLUTE_MAX: float = 30.0
    RMS_MIN: float = 0.02
    RMS_MAX: float = 0.30
    RMS_OPTIMAL: float = 0.10
    CLIP_THRESHOLD: float = 0.98
    CLIP_MAX_SAMPLES: int = 10
    DYNAMIC_RANGE_MIN: float = 0.1
    SILENCE_RATIO_MAX: float = 0.15


print("[DIAGNOSTIC] SECTION 2: ENUMS AND CONSTANTS - OK", flush=True)


# =============================================================================
# SECTION 3: HELPER FUNCTIONS
# =============================================================================

def validate_reference_audio(audio_path: str, min_duration: float = 2.0, max_duration: float = 30.0, min_rms: float = 0.01) -> Tuple[bool, Optional[str]]:
    try:
        if not os.path.exists(audio_path):
            return False, f"File not found: {audio_path}"
        audio, sr = load_audio_for_voice_clone(audio_path)
        duration = len(audio) / sr
        if duration < min_duration:
            return False, f"Audio too short ({duration:.1f}s). Min: {min_duration}s"
        if duration > max_duration:
            return False, f"Audio too long ({duration:.1f}s). Max: {max_duration}s"
        rms = np.sqrt(np.mean(audio**2))
        if rms < min_rms:
            return False, f"Audio too quiet (RMS: {rms:.4f}). Min: {min_rms}"
        if np.abs(audio).max() > 0.99:
            return False, "Audio has clipping"
        return True, None
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def load_audio_for_voice_clone(audio_path: str, sample_rate: int = 24000, mono: bool = True, normalize: bool = True) -> Tuple[np.ndarray, int]:
    try:
        import librosa
        audio, sr = librosa.load(audio_path, sr=sample_rate, mono=mono)
        if normalize:
            audio = _normalize_audio(audio)
        return audio.astype(np.float32), sr
    except ImportError:
        try:
            import soundfile as sf
            audio, sr = sf.read(audio_path)
            if mono and len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            if sr != sample_rate:
                import scipy.signal
                audio = scipy.signal.resample_poly(audio, sample_rate, sr)
                sr = sample_rate
            if normalize:
                audio = _normalize_audio(audio)
            return audio.astype(np.float32), sr
        except ImportError:
            raise ImportError("Install librosa or soundfile: pip install librosa soundfile")


def _normalize_audio(audio: np.ndarray, target_db: float = -20.0, peak_limit: float = 0.85) -> np.ndarray:
    audio = audio.astype(np.float32)
    rms = np.sqrt(np.mean(audio**2))
    target_rms = 10**(target_db / 20)
    if rms > 0:
        gain = target_rms / rms
        audio = audio * gain
    audio = np.clip(audio, -peak_limit, peak_limit)
    return audio


def get_cache_key(audio_path: str, reference_text: str) -> str:
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    combined = audio_bytes + reference_text.encode("utf-8")
    return hashlib.md5(combined).hexdigest()


print("[DIAGNOSTIC] SECTION 3: HELPER FUNCTIONS - OK", flush=True)


# =============================================================================
# SECTION 4: DATACLASSES
# =============================================================================

@dataclass
class VoiceProfile:
    id: str
    name: str
    mode: VoiceMode = VoiceMode.VOICE_DESIGN
    seed: Optional[int] = None
    prompt: Optional[str] = None
    reference_audio_path: Optional[str] = None
    reference_text: Optional[str] = None
    gender: str = "male"
    language: str = "russian"
    speed: float = 1.0
    pitch: str = "medium"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.seed is None and self.mode == VoiceMode.VOICE_DESIGN:
            self.seed = self._generate_seed_from_id()
    
    def _generate_seed_from_id(self) -> int:
        hash_obj = hashlib.md5(self.id.encode('utf-8'))
        return int(hash_obj.hexdigest(), 16) % (2**32)
    
    def to_engine_config(self) -> Dict[str, Any]:
        config = {"mode": self.mode.value, "speed": self.speed, "language": self.language}
        if self.mode == VoiceMode.VOICE_DESIGN:
            config["seed"] = self.seed
            config["prompt"] = self.prompt or self._build_default_prompt()
        else:
            config["reference_audio"] = self.reference_audio_path
            config["reference_text"] = self.reference_text
        return config
    
    def _build_default_prompt(self) -> str:
        gender_prefix = "Male voice" if self.gender == "male" else "Female voice"
        pitch_desc = {"low": "low-pitched", "medium": "medium-pitched", "high": "high-pitched"}
        return f"{gender_prefix}, {pitch_desc.get(self.pitch, 'medium')}. {self.description}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "name": self.name, "mode": self.mode.value, "seed": self.seed,
                "prompt": self.prompt, "reference_audio_path": self.reference_audio_path,
                "reference_text": self.reference_text, "gender": self.gender, "language": self.language,
                "speed": self.speed, "pitch": self.pitch, "description": self.description, "tags": self.tags}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VoiceProfile':
        mode = VoiceMode.from_string(data.get("mode", "voicedesign"))
        return cls(id=data["id"], name=data["name"], mode=mode, seed=data.get("seed"),
                   prompt=data.get("prompt"), reference_audio_path=data.get("reference_audio_path"),
                   reference_text=data.get("reference_text"), gender=data.get("gender", "male"),
                   language=data.get("language", "russian"), speed=data.get("speed", 1.0),
                   pitch=data.get("pitch", "medium"), description=data.get("description", ""), tags=data.get("tags", []))


@dataclass
class VoiceLibraryEntry:
    character_name: str
    voice_preset: str
    reference_audio_path: str
    reference_text: str
    duration: float
    quality_score: float
    created_at: str
    updated_at: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.quality_score <= 1.0:
            raise ValueError(f'quality_score must be in [0.0, 1.0], got: {self.quality_score}')
        if self.duration < 0:
            raise ValueError(f'duration must be positive, got: {self.duration}')
        if not self.character_name or not self.character_name.strip():
            raise ValueError('character_name cannot be empty')

    def to_dict(self) -> Dict[str, Any]:
        return {'character_name': self.character_name, 'voice_preset': self.voice_preset,
                'reference_audio_path': self.reference_audio_path, 'reference_text': self.reference_text,
                'duration': self.duration, 'quality_score': self.quality_score,
                'created_at': self.created_at, 'updated_at': self.updated_at, 'metadata': self.metadata}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VoiceLibraryEntry':
        return cls(character_name=data['character_name'], voice_preset=data['voice_preset'],
                   reference_audio_path=data['reference_audio_path'], reference_text=data['reference_text'],
                   duration=data['duration'], quality_score=data['quality_score'],
                   created_at=data['created_at'], updated_at=data['updated_at'], metadata=data.get('metadata', {}))


@dataclass
class ColabVoiceConfig:
    character_name: str
    voice_preset: str = 'male_deep'
    gender: str = 'male'
    language: str = 'russian'
    use_clone: bool = True
    auto_create_reference: bool = True
    custom_reference_text: Optional[str] = None
    def to_dict(self) -> Dict[str, Any]: return asdict(self)


@dataclass
class VoiceLibraryStats:
    total_voices: int = 0
    total_duration_sec: float = 0.0
    average_quality: float = 0.0
    drive_synced: bool = False
    last_sync_time: Optional[str] = None
    def to_dict(self) -> Dict[str, Any]: return asdict(self)


@dataclass
class GenerationResult:
    success: bool
    audio: Optional[np.ndarray] = None
    sample_rate: int = 24000
    duration: float = 0.0
    voice_id: Optional[str] = None
    is_new_voice: bool = False
    error_message: Optional[str] = None
    def to_dict(self) -> Dict[str, Any]:
        return {'success': self.success, 'sample_rate': self.sample_rate, 'duration': self.duration,
                'voice_id': self.voice_id, 'is_new_voice': self.is_new_voice, 'error_message': self.error_message}


print("[DIAGNOSTIC] SECTION 4: DATACLASSES - OK", flush=True)


# =============================================================================
# SECTION 5: EXCEPTIONS
# =============================================================================

class DriveSyncError(Exception): pass
class ColabInitError(Exception): pass
class VoiceNotFoundError(Exception): pass


print("[DIAGNOSTIC] SECTION 5: EXCEPTIONS - OK", flush=True)


# =============================================================================
# SECTION 6: VOICE PROMPT CACHE
# =============================================================================

class VoicePromptCache:
    def __init__(self, cache_dir: str = "/content/cache", max_memory_items: int = 100):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: Dict[str, Any] = {}
        self._max_memory_items = max_memory_items
        self._hits = 0
        self._misses = 0
        print(f"[VoicePromptCache] Initialized: {cache_dir}", flush=True)
    
    def get_cache_key(self, *args) -> str:
        content_parts = []
        for arg in args:
            if isinstance(arg, dict): content_parts.append(json.dumps(arg, sort_keys=True))
            elif isinstance(arg, (list, tuple)): content_parts.append(str(tuple(arg)))
            else: content_parts.append(str(arg))
        return hashlib.md5(":".join(content_parts).encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        if key in self._memory_cache:
            self._hits += 1
            return self._memory_cache[key]
        cache_file = self.cache_dir / f"{key}.pt"
        if cache_file.exists():
            try:
                import torch
                data = torch.load(cache_file, map_location='cpu')
                self._set_memory_cache(key, data)
                self._hits += 1
                return data
            except Exception: pass
        self._misses += 1
        return None
    
    def set(self, key: str, value: Any, save_to_disk: bool = True) -> None:
        self._set_memory_cache(key, value)
        if save_to_disk:
            try:
                import torch
                cache_file = self.cache_dir / f"{key}.pt"
                if hasattr(value, 'numpy'): torch.save(value, cache_file)
                elif hasattr(value, '__array__'): torch.save(torch.from_numpy(value) if hasattr(value, 'dtype') else torch.tensor(value), cache_file)
                else: torch.save(value, cache_file)
            except Exception: pass
    
    def _set_memory_cache(self, key: str, value: Any) -> None:
        if len(self._memory_cache) >= self._max_memory_items:
            for k in list(self._memory_cache.keys())[:self._max_memory_items // 2]: del self._memory_cache[k]
        self._memory_cache[key] = value
    
    def has(self, key: str) -> bool: return key in self._memory_cache or (self.cache_dir / f"{key}.pt").exists()
    def delete(self, key: str) -> bool:
        deleted = False
        if key in self._memory_cache: del self._memory_cache[key]; deleted = True
        cache_file = self.cache_dir / f"{key}.pt"
        if cache_file.exists():
            try: cache_file.unlink(); deleted = True
            except Exception: pass
        return deleted
    
    def clear_memory(self) -> None: self._memory_cache.clear()
    def clear_disk(self) -> None:
        try:
            for f in self.cache_dir.glob("*.pt"): f.unlink()
        except Exception: pass
    def clear(self) -> None: self.clear_memory(); self.clear_disk()
    
    def get_stats(self) -> Dict[str, Any]:
        total = self._hits + self._misses
        return {"hits": self._hits, "misses": self._misses, "hit_rate": round(self._hits / total, 2) if total > 0 else 0,
                "memory_items": len(self._memory_cache), "disk_items": len(list(self.cache_dir.glob("*.pt")))}
    
    def get_preset_config(self, voice_preset: str) -> Optional[Dict[str, Any]]: return self.get(f"preset_{voice_preset}")
    def set_preset_config(self, voice_preset: str, config: Dict[str, Any]) -> None: self.set(f"preset_{voice_preset}", config, save_to_disk=True)


print("[DIAGNOSTIC] SECTION 6: VOICE PROMPT CACHE - OK", flush=True)


# =============================================================================
# SECTION 7: TTS ENGINE (ABSTRACT)
# =============================================================================

class TTSEngine(ABC):
    @abstractmethod
    def generate(self, text: str, voice_config: Dict[str, Any], language: str = "russian") -> Tuple[Optional[np.ndarray], int]: pass
    @abstractmethod
    def is_loaded(self) -> bool: pass
    @property
    @abstractmethod
    def engine_type(self) -> str: pass


print("[DIAGNOSTIC] SECTION 7: TTS ENGINE (ABSTRACT) - OK", flush=True)


# =============================================================================
# SECTION 8: VOICE DESIGN ENGINE
# =============================================================================

class VoiceDesignEngine(TTSEngine):
    def __init__(self, model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"):
        self._model = None
        self._model_name = model_name
        self._sample_rate = 24000
    
    def load_model(self, device: str = "cuda") -> bool:
        try:
            import torch
            import transformers
            print(f"[DIAGNOSTIC] transformers version: {transformers.__version__}", flush=True)
            print(f"[DIAGNOSTIC] torch version: {torch.__version__}", flush=True)
            print(f"[DIAGNOSTIC] torch.cuda.is_available(): {torch.cuda.is_available()}", flush=True)
            
            # Виправлення помилки BACKENDS_MAPPING для keras_nlp
            # Потрібно додати keras_nlp до BACKENDS_MAPPING перед імпортом qwen_tts
            try:
                from transformers.utils.import_utils import BACKENDS_MAPPING
                if 'keras_nlp' not in BACKENDS_MAPPING:
                    print(f"[DIAGNOSTIC] Patching BACKENDS_MAPPING for keras_nlp...", flush=True)
                    from transformers.utils.import_utils import _LazyModule
                    # Додаємо заглушку для keras_nlp
                    BACKENDS_MAPPING['keras_nlp'] = (lambda: False, "keras_nlp is not available")
                    print(f"[DIAGNOSTIC] BACKENDS_MAPPING patched successfully", flush=True)
            except Exception as patch_error:
                print(f"[DIAGNOSTIC] Warning: Could not patch BACKENDS_MAPPING: {patch_error}", flush=True)
            
            from qwen_tts import Qwen3TTSModel
            print(f"[VoiceDesign] Loading model {self._model_name}...", flush=True)
            
            # DIAGNOSTIC: Check if keras_nlp is installed
            try:
                import keras_nlp
                print(f"[DIAGNOSTIC] keras_nlp is installed: {keras_nlp.__version__}", flush=True)
            except ImportError:
                print(f"[DIAGNOSTIC] keras_nlp is NOT installed", flush=True)
            
            # DIAGNOSTIC: Try different loading strategies
            if device == "cuda":
                dtype = torch.float16
                print(f"[DIAGNOSTIC] Using dtype: {dtype}", flush=True)
                
                # Strategy 1: Try with explicit device instead of device_map
                try:
                    print(f"[DIAGNOSTIC] Strategy 1: Loading with device='cuda:0'...", flush=True)
                    self._model = Qwen3TTSModel.from_pretrained(
                        self._model_name, 
                        device_map="cuda:0", 
                        torch_dtype=dtype,
                        trust_remote_code=True
                    )
                    print(f"[VoiceDesign] Model loaded with strategy 1!", flush=True)
                    return True
                except Exception as e1:
                    print(f"[DIAGNOSTIC] Strategy 1 failed: {e1}", flush=True)
                    
                    # Strategy 2: Load on CPU first, then move to CUDA
                    try:
                        print(f"[DIAGNOSTIC] Strategy 2: Loading on CPU first...", flush=True)
                        self._model = Qwen3TTSModel.from_pretrained(
                            self._model_name, 
                            device_map="cpu", 
                            torch_dtype=torch.float32,
                            trust_remote_code=True
                        )
                        print(f"[DIAGNOSTIC] Moving model to CUDA...", flush=True)
                        self._model = self._model.to("cuda")
                        print(f"[VoiceDesign] Model loaded with strategy 2!", flush=True)
                        return True
                    except Exception as e2:
                        print(f"[DIAGNOSTIC] Strategy 2 failed: {e2}", flush=True)
                        raise e1  # Re-raise original error
            else:
                print(f"[DIAGNOSTIC] Loading on CPU...", flush=True)
                self._model = Qwen3TTSModel.from_pretrained(
                    self._model_name, 
                    device_map="cpu", 
                    torch_dtype=torch.float32,
                    trust_remote_code=True
                )
                print(f"[VoiceDesign] Model loaded!", flush=True)
                return True
        except Exception as e:
            print(f"[VoiceDesign] Load error: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return False
    
    def set_model(self, model) -> None: self._model = model
    def is_loaded(self) -> bool: return self._model is not None
    @property
    def engine_type(self) -> str: return "voicedesign"
    
    def generate(self, text: str, voice_config: Dict[str, Any], language: str = "russian") -> Tuple[Optional[np.ndarray], int]:
        if not self.is_loaded():
            print("[VoiceDesign] Model not loaded!", flush=True)
            return None, 0
        try:
            import torch
            import random
            seed = voice_config.get("seed", 42)
            prompt = voice_config.get("prompt", "A natural voice")
            speed = voice_config.get("speed", 1.0)
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            
            result = None
            if hasattr(self._model, 'generate_voice_design'): result = self._model.generate_voice_design(text, prompt, language)
            elif hasattr(self._model, 'generate'):
                try: result = self._model.generate(text=text, prompt=prompt, language=language)
                except TypeError: result = self._model.generate(text, prompt, language)
            elif hasattr(self._model, 'synthesize'): result = self._model.synthesize(text, prompt, language)
            else: print(f"[VoiceDesign] No generate method found!", flush=True); return None, 0
            
            if isinstance(result, tuple): audio, sr = result
            else: audio = result; sr = self._sample_rate
            if hasattr(audio, 'cpu'): audio = audio.cpu().numpy()
            elif hasattr(audio, 'numpy'): audio = audio.numpy()
            audio = np.array(audio).flatten()
            if speed != 1.0: audio = self._change_speed(audio, speed)
            print(f"[VoiceDesign] Generated: {len(audio)} samples", flush=True)
            return audio, sr
        except Exception as e:
            print(f"[VoiceDesign] Generation error: {e}", flush=True)
            import traceback; traceback.print_exc()
            return None, 0
    
    def _change_speed(self, audio: np.ndarray, speed: float) -> np.ndarray:
        try:
            from pydub import AudioSegment
            audio_int16 = (audio * 32767).astype(np.int16)
            segment = AudioSegment(audio_int16.tobytes(), frame_rate=self._sample_rate, sample_width=2, channels=1)
            if speed != 1.0: segment = segment.speedup(playback_speed=speed)
            return np.array(segment.get_array_of_samples()).astype(np.float32) / 32767.0
        except Exception as e:
            print(f"[VoiceDesign] Speed change error: {e}", flush=True)
            return audio


print("[DIAGNOSTIC] SECTION 8: VOICE DESIGN ENGINE - OK", flush=True)


# =============================================================================
# SECTION 9: VOICE CLONE ENGINE
# =============================================================================

class VoiceCloneEngine(TTSEngine):
    LANGUAGE_MAP = {"russian": "ru", "ukrainian": "uk", "english": "en", "chinese": "zh", "japanese": "ja",
                    "korean": "ko", "german": "de", "french": "fr", "spanish": "es", "italian": "it", "portuguese": "pt"}
    
    def __init__(self, model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base", cache_dir: Optional[str] = None):
        self._model = None
        self._model_name = model_name
        self._sample_rate = 24000
        self._cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".voicebox_cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._prompt_cache: Dict[str, Any] = {}
        self._fallback_engine: Optional[VoiceDesignEngine] = None
    
    def _get_device(self) -> str:
        try:
            import torch
            if torch.cuda.is_available():
                if 'COLAB_GPU' in os.environ or 'COLAB_TPU_ADDR' in os.environ: print("[VoiceClone] Detected Google Colab with CUDA", flush=True)
                return "cuda"
            return "cpu"
        except ImportError: return "cpu"
    
    def load_model(self, device: Optional[str] = None) -> bool:
        try:
            import torch
            import transformers
            print(f"[DIAGNOSTIC] transformers version: {transformers.__version__}", flush=True)
            print(f"[DIAGNOSTIC] torch version: {torch.__version__}", flush=True)
            print(f"[DIAGNOSTIC] torch.cuda.is_available(): {torch.cuda.is_available()}", flush=True)
            
            # Виправлення помилки BACKENDS_MAPPING для keras_nlp
            # Потрібно додати keras_nlp до BACKENDS_MAPPING перед імпортом qwen_tts
            try:
                from transformers.utils.import_utils import BACKENDS_MAPPING
                if 'keras_nlp' not in BACKENDS_MAPPING:
                    print(f"[DIAGNOSTIC] Patching BACKENDS_MAPPING for keras_nlp...", flush=True)
                    from transformers.utils.import_utils import _LazyModule
                    # Додаємо заглушку для keras_nlp
                    BACKENDS_MAPPING['keras_nlp'] = (lambda: False, "keras_nlp is not available")
                    print(f"[DIAGNOSTIC] BACKENDS_MAPPING patched successfully", flush=True)
            except Exception as patch_error:
                print(f"[DIAGNOSTIC] Warning: Could not patch BACKENDS_MAPPING: {patch_error}", flush=True)
            
            from qwen_tts import Qwen3TTSModel
            if device is None: device = self._get_device()
            print(f"[VoiceClone] Loading model {self._model_name}...", flush=True)
            print(f"[VoiceClone] Device: {device}", flush=True)
            
            # DIAGNOSTIC: Check if keras_nlp is installed
            try:
                import keras_nlp
                print(f"[DIAGNOSTIC] keras_nlp is installed: {keras_nlp.__version__}", flush=True)
            except ImportError:
                print(f"[DIAGNOSTIC] keras_nlp is NOT installed", flush=True)
            
            # DIAGNOSTIC: Try different loading strategies
            if device == "cuda":
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                print(f"[DIAGNOSTIC] Using dtype: {dtype}", flush=True)
                
                # Strategy 1: Try with explicit device instead of "auto"
                try:
                    print(f"[DIAGNOSTIC] Strategy 1: Loading with device_map='cuda:0'...", flush=True)
                    self._model = Qwen3TTSModel.from_pretrained(
                        self._model_name, 
                        device_map="cuda:0", 
                        torch_dtype=dtype,
                        trust_remote_code=True
                    )
                    print(f"[VoiceClone] Model loaded with strategy 1!", flush=True)
                    return True
                except Exception as e1:
                    print(f"[DIAGNOSTIC] Strategy 1 failed: {e1}", flush=True)
                    
                    # Strategy 2: Load on CPU first, then move to CUDA
                    try:
                        print(f"[DIAGNOSTIC] Strategy 2: Loading on CPU first...", flush=True)
                        self._model = Qwen3TTSModel.from_pretrained(
                            self._model_name, 
                            device_map="cpu", 
                            torch_dtype=torch.float32,
                            trust_remote_code=True
                        )
                        print(f"[DIAGNOSTIC] Moving model to CUDA...", flush=True)
                        self._model = self._model.to("cuda")
                        print(f"[VoiceClone] Model loaded with strategy 2!", flush=True)
                        return True
                    except Exception as e2:
                        print(f"[DIAGNOSTIC] Strategy 2 failed: {e2}", flush=True)
                        raise e1  # Re-raise original error
            else:
                dtype = torch.float32
                print(f"[DIAGNOSTIC] Loading on CPU with dtype: {dtype}", flush=True)
                self._model = Qwen3TTSModel.from_pretrained(
                    self._model_name, 
                    device_map="cpu", 
                    torch_dtype=dtype,
                    trust_remote_code=True
                )
                print(f"[VoiceClone] Model loaded!", flush=True)
                return True
        except ImportError as e:
            print(f"[VoiceClone] Import error: {e}", flush=True)
            print("[VoiceClone] Install: pip install git+https://github.com/QwenLM/Qwen3-TTS.git", flush=True)
            return False
        except Exception as e:
            print(f"[VoiceClone] Load error: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return False
    
    def set_model(self, model) -> None: self._model = model; print("[VoiceClone] Model set externally", flush=True)
    def is_loaded(self) -> bool: return self._model is not None
    @property
    def engine_type(self) -> str: return "clone"
    
    def create_voice_prompt(self, reference_audio_path: str, reference_text: str, use_cache: bool = True, validate: bool = True) -> Tuple[Optional[Dict[str, Any]], bool]:
        if not self.is_loaded(): print("[VoiceClone] Model not loaded!", flush=True); return None, False
        if validate:
            is_valid, error_msg = validate_reference_audio(reference_audio_path)
            if not is_valid: print(f"[VoiceClone] Validation failed: {error_msg}", flush=True); return None, False
        cache_key = get_cache_key(reference_audio_path, reference_text)
        if use_cache:
            if cache_key in self._prompt_cache: print(f"[VoiceClone] Voice prompt from memory cache", flush=True); return self._prompt_cache[cache_key], True
            cached_prompt = self._load_prompt_from_disk(cache_key)
            if cached_prompt is not None: self._prompt_cache[cache_key] = cached_prompt; print(f"[VoiceClone] Voice prompt from disk cache", flush=True); return cached_prompt, True
        try:
            print(f"[VoiceClone] Creating voice prompt...", flush=True)
            voice_prompt = self._model.create_voice_clone_prompt(ref_audio=reference_audio_path, ref_text=reference_text, x_vector_only_mode=False)
            if use_cache: self._prompt_cache[cache_key] = voice_prompt; self._save_prompt_to_disk(cache_key, voice_prompt)
            print(f"[VoiceClone] Voice prompt created", flush=True)
            return voice_prompt, False
        except Exception as e:
            print(f"[VoiceClone] Voice prompt error: {e}", flush=True)
            import traceback; traceback.print_exc()
            return None, False
    
    def generate(self, text: str, voice_config: Dict[str, Any], language: str = "russian") -> Tuple[Optional[np.ndarray], int]:
        if not self.is_loaded(): print("[VoiceClone] Model not loaded!", flush=True); return None, 0
        try:
            import torch
            reference_audio = voice_config.get("reference_audio")
            reference_text = voice_config.get("reference_text", "")
            speed = voice_config.get("speed", 1.0)
            instruct = voice_config.get("instruct")
            use_cache = voice_config.get("use_cache", True)
            voice_prompt = voice_config.get("voice_prompt")
            was_cached = False
            if voice_prompt is None:
                if not reference_audio: print("[VoiceClone] No reference_audio or voice_prompt!", flush=True); return None, 0
                voice_prompt, was_cached = self.create_voice_prompt(reference_audio, reference_text, use_cache=use_cache)
                if voice_prompt is None: print("[VoiceClone] Failed to create voice prompt", flush=True); return self._fallback_to_voicedesign(text, voice_config, language)
            lang_code = self.LANGUAGE_MAP.get(language.lower(), language.lower())
            print(f"[VoiceClone] Generating audio...", flush=True)
            result = self._generate_with_model(text=text, voice_prompt=voice_prompt, language=lang_code, instruct=instruct)
            if result is None: return None, 0
            audio, sr = result
            if hasattr(audio, 'cpu'): audio = audio.cpu().numpy()
            elif hasattr(audio, 'numpy'): audio = audio.numpy()
            audio = np.array(audio).flatten()
            if speed != 1.0: audio = self._change_speed(audio, speed)
            print(f"[VoiceClone] Generated: {len(audio)} samples ({len(audio)/sr:.2f}s)", flush=True)
            return audio, sr
        except Exception as e:
            print(f"[VoiceClone] Generation error: {e}", flush=True)
            import traceback; traceback.print_exc()
            return None, 0
    
    def _generate_with_model(self, text: str, voice_prompt: Dict[str, Any], language: str, instruct: Optional[str] = None) -> Optional[Tuple[np.ndarray, int]]:
        try:
            if hasattr(self._model, 'generate_voice_clone'):
                wavs, sample_rate = self._model.generate_voice_clone(text=text, voice_clone_prompt=voice_prompt, instruct=instruct)
                return wavs[0] if isinstance(wavs, list) else wavs, sample_rate
            elif hasattr(self._model, 'generate'):
                result = self._model.generate(text=text, voice_prompt=voice_prompt, language=language)
                if isinstance(result, tuple): return result
                return result, self._sample_rate
            else: print("[VoiceClone] No generate method found!", flush=True); return None
        except Exception as e: print(f"[VoiceClone] Model generation error: {e}", flush=True); return None
    
    def _fallback_to_voicedesign(self, text: str, voice_config: Dict[str, Any], language: str) -> Tuple[Optional[np.ndarray], int]:
        print("[VoiceClone] Fallback to VoiceDesign...", flush=True)
        try:
            if self._fallback_engine is None: self._fallback_engine = VoiceDesignEngine()
            if self.is_loaded(): self._fallback_engine.set_model(self._model)
            fallback_config = {"seed": voice_config.get("seed", 42), "prompt": voice_config.get("prompt", "A natural voice"), "speed": voice_config.get("speed", 1.0)}
            return self._fallback_engine.generate(text, fallback_config, language)
        except Exception as e: print(f"[VoiceClone] Fallback error: {e}", flush=True); return None, 0
    
    def _save_prompt_to_disk(self, cache_key: str, voice_prompt: Any) -> None:
        try:
            import torch
            torch.save(voice_prompt, self._cache_dir / f"{cache_key}.prompt")
        except Exception as e: print(f"[VoiceClone] Cache save error: {e}", flush=True)
    
    def _load_prompt_from_disk(self, cache_key: str) -> Optional[Any]:
        try:
            import torch
            cache_file = self._cache_dir / f"{cache_key}.prompt"
            if cache_file.exists(): return torch.load(cache_file)
        except Exception: pass
        return None
    
    def clear_cache(self) -> int:
        self._prompt_cache.clear()
        deleted = 0
        for cache_file in self._cache_dir.glob("*.prompt"):
            try: cache_file.unlink(); deleted += 1
            except Exception: pass
        print(f"[VoiceClone] Cleared {deleted} cache files", flush=True)
        return deleted
    
    def _change_speed(self, audio: np.ndarray, speed: float) -> np.ndarray:
        try:
            from pydub import AudioSegment
            audio_int16 = (audio * 32767).astype(np.int16)
            segment = AudioSegment(audio_int16.tobytes(), frame_rate=self._sample_rate, sample_width=2, channels=1)
            if speed != 1.0: segment = segment.speedup(playback_speed=speed)
            return np.array(segment.get_array_of_samples()).astype(np.float32) / 32767.0
        except Exception as e: print(f"[VoiceClone] Speed change error: {e}", flush=True); return audio


print("[DIAGNOSTIC] SECTION 9: VOICE CLONE ENGINE - OK", flush=True)


# =============================================================================
# SECTION 10: VOICE LIBRARY MANAGER
# =============================================================================

class VoiceLibraryManager:
    OPTIMAL_DURATION_MIN = 15.0
    OPTIMAL_DURATION_MAX = 20.0
    MIN_DURATION = 10.0
    MAX_DURATION = 30.0
    DRIVE_BASE_PATH = '/content/drive/MyDrive/vibemodly_voices'
    LIBRARY_FILE = 'voice_library.json'
    REFERENCE_DIR = 'references'

    def __init__(self, voice_clone_engine, voice_design_engine, cache=None, drive_path: Optional[str] = None):
        self._engine_clone = voice_clone_engine
        self._engine_design = voice_design_engine
        self._cache = cache
        self._library: Dict[str, VoiceLibraryEntry] = {}
        self._drive_path = drive_path or self.DRIVE_BASE_PATH
        self._created_at = datetime.now().isoformat()

    def get_voice(self, character_name: str, voice_preset: str, gender: str = 'male', language: str = 'russian') -> Tuple[Optional[Dict[str, Any]], bool]:
        if not character_name or not character_name.strip(): raise ValueError('character_name cannot be empty')
        if not voice_preset or not voice_preset.strip(): raise ValueError('voice_preset cannot be empty')
        entry = self._library.get(character_name)
        if entry:
            ref_path = os.path.join(self._drive_path, entry.reference_audio_path)
            if os.path.exists(ref_path):
                voice_prompt, _ = self._engine_clone.create_voice_prompt(reference_audio_path=ref_path, reference_text=entry.reference_text, use_cache=True)
                if voice_prompt: return voice_prompt, False
        audio_path, ref_text = self.create_canonical_reference(character_name=character_name, voice_preset=voice_preset, gender=gender, language=language)
        if audio_path is None: print(f'[VoiceLibrary] Failed to create reference for {character_name}'); return None, False
        is_valid, error_msg, quality_score = self.validate_reference(audio_path, strict=False)
        if not is_valid: print(f'[VoiceLibrary] Reference validation failed: {error_msg}')
        voice_prompt, _ = self._engine_clone.create_voice_prompt(reference_audio_path=audio_path, reference_text=ref_text, use_cache=True)
        if voice_prompt is None: return None, False
        duration = self._calculate_duration(audio_path)
        entry = VoiceLibraryEntry(character_name=character_name, voice_preset=voice_preset,
                                  reference_audio_path=os.path.relpath(audio_path, self._drive_path),
                                  reference_text=ref_text, duration=duration, quality_score=quality_score,
                                  created_at=datetime.now().isoformat(), updated_at=datetime.now().isoformat(),
                                  metadata={'gender': gender, 'language': language})
        self._library[character_name] = entry
        self.save_library()
        return voice_prompt, True

    def create_canonical_reference(self, character_name: str, voice_preset: str, gender: str = 'male', language: str = 'russian', custom_text: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
        if not character_name or not character_name.strip(): raise ValueError('character_name cannot be empty')
        if not voice_preset or not voice_preset.strip(): raise ValueError('voice_preset cannot be empty')
        if self._engine_design is None: print('[VoiceLibrary] VoiceDesignEngine not initialized'); return None, None
        ref_text = custom_text or self._generate_reference_text(language, target_duration=17.5)
        seed = None; prompt = None
        if self._cache is not None:
            cached_config = self._cache.get_preset_config(voice_preset)
            if cached_config: seed = cached_config.get('seed'); prompt = cached_config.get('prompt')
        self._ensure_directory_structure()
        audio_path = self._get_reference_path(character_name)
        try:
            print(f'[VoiceLibrary] Creating canonical reference for {character_name}...')
            audio, sr = self._engine_design.generate(text=ref_text, voice_config={'seed': seed, 'prompt': prompt, 'gender': gender}, language=language)
            if audio is None: print('[VoiceLibrary] Failed to generate audio'); return None, None
            if not self._save_audio_file(audio, sr, audio_path): print('[VoiceLibrary] Failed to save audio file'); return None, None
            print(f'[VoiceLibrary] Canonical reference created: {audio_path}')
            return audio_path, ref_text
        except Exception as e: print(f'[VoiceLibrary] Reference creation error: {e}'); return None, None

    def list_voices(self) -> List[VoiceLibraryEntry]: return list(self._library.values())
    def get_entry(self, character_name: str) -> Optional[VoiceLibraryEntry]: return self._library.get(character_name)
    def delete_voice(self, character_name: str) -> bool:
        if character_name in self._library: del self._library[character_name]; return True
        return False

    def _generate_reference_text(self, language: str = 'russian', target_duration: float = 17.5) -> str:
        texts = {
            'russian': 'Привет! Это тестовый фрагмент голоса для определения его характеристик. Я могу озвучивать ваши истории с разными эмоциями и интонациями. Этот текст специально создан для формирования качественного reference аудио.',
            'ukrainian': 'Привіт! Це тестовий фрагмент голосу для визначення його характеристик. Я можу озвучувати ваші історії з різними емоціями та інтонаціями. Цей текст спеціально створено для формування якісного reference аудіо.',
            'english': 'Hello! This is a test voice sample for determining its characteristics. I can voice your stories with different emotions and intonations. This text is specially created for forming a high-quality reference audio.'
        }
        return texts.get(language, texts['russian'])

    def _get_reference_path(self, character_name: str) -> str:
        safe_name = character_name.replace(' ', '_').lower()
        return os.path.join(self._drive_path, self.REFERENCE_DIR, f'{safe_name}_ref.wav')

    def _ensure_drive_mounted(self) -> bool:
        if not os.path.exists('/content/drive/MyDrive'):
            try:
                from google.colab import drive
                drive.mount('/content/drive')
                return True
            except Exception as e: print(f'[VoiceLibrary] Failed to mount Drive: {e}'); return False
        return True

    def _ensure_directory_structure(self) -> bool:
        for dir_path in [self._drive_path, os.path.join(self._drive_path, self.REFERENCE_DIR),
                         os.path.join(self._drive_path, 'cache'), os.path.join(self._drive_path, 'backup')]:
            os.makedirs(dir_path, exist_ok=True)
        return True

    def save_library(self) -> bool:
        if not self._ensure_drive_mounted(): return False
        self._ensure_directory_structure()
        library_path = os.path.join(self._drive_path, self.LIBRARY_FILE)
        if os.path.exists(library_path): self._create_backup(library_path)
        data = {'version': '1.0.0', 'created_at': self._created_at, 'updated_at': datetime.now().isoformat(),
                'entries': {name: entry.to_dict() for name, entry in self._library.items()}, 'statistics': self._calculate_statistics()}
        try:
            with open(library_path, 'w', encoding='utf-8') as f: json.dump(data, f, ensure_ascii=False, indent=2)
            print(f'[VoiceLibrary] Saved {len(self._library)} voices')
            return True
        except Exception as e: print(f'[VoiceLibrary] Save error: {e}'); return False

    def load_library(self) -> bool:
        if not self._ensure_drive_mounted(): return False
        library_path = os.path.join(self._drive_path, self.LIBRARY_FILE)
        if not os.path.exists(library_path): print('[VoiceLibrary] Library not found, creating new'); self._library = {}; return True
        try:
            with open(library_path, 'r', encoding='utf-8') as f: data = json.load(f)
            version = data.get('version', '0.0.0')
            if version != '1.0.0': print(f'[VoiceLibrary] Version {version}, may need migration')
            self._library = {}
            for name, entry_data in data.get('entries', {}).items(): self._library[name] = VoiceLibraryEntry.from_dict(entry_data)
            print(f'[VoiceLibrary] Loaded {len(self._library)} voices')
            return True
        except Exception as e: print(f'[VoiceLibrary] Load error: {e}'); return False

    def _create_backup(self, library_path: str) -> None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = os.path.join(self._drive_path, 'backup', f'voice_library_{timestamp}.json')
        try:
            import shutil
            shutil.copy2(library_path, backup_path)
            print(f'[VoiceLibrary] Backup created: voice_library_{timestamp}.json')
        except Exception as e: print(f'[VoiceLibrary] Backup failed: {e}')

    def _calculate_statistics(self) -> Dict[str, Any]:
        if not self._library: return {'total_voices': 0, 'total_duration_sec': 0.0, 'average_quality': 0.0}
        total_duration = sum(e.duration for e in self._library.values())
        avg_quality = sum(e.quality_score for e in self._library.values()) / len(self._library)
        return {'total_voices': len(self._library), 'total_duration_sec': round(total_duration, 2), 'average_quality': round(avg_quality, 2)}

    def validate_reference(self, audio_path: str, strict: bool = False) -> Tuple[bool, Optional[str], float]:
        try:
            audio, sr = self._load_audio(audio_path)
            duration = len(audio) / sr
            duration_ok, duration_msg = self._validate_duration(duration, strict)
            if not duration_ok: return False, duration_msg, 0.0
            rms_ok, rms_msg, rms_score = self._validate_rms(audio)
            if not rms_ok: return False, rms_msg, 0.0
            clip_ok, clip_msg, clip_score = self._validate_clipping(audio)
            if not clip_ok: return False, clip_msg, 0.0
            dyn_ok, dyn_msg, dyn_score = self._validate_dynamic_range(audio)
            silence_ok, silence_msg, silence_score = self._validate_silence(audio, sr)
            quality_score = self._calculate_quality_score(duration=duration, rms_score=rms_score, clip_score=clip_score,
                                                          dyn_score=dyn_score, silence_score=silence_score, strict=strict)
            return True, None, quality_score
        except Exception as e: return False, f'Validation error: {e}', 0.0

    def _load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        try:
            from scipy.io import wavfile
            sr, audio = wavfile.read(audio_path)
            if audio.dtype == np.int16: audio = audio.astype(np.float32) / 32768.0
            elif audio.dtype == np.int32: audio = audio.astype(np.float32) / 2147483648.0
            elif audio.dtype == np.float32: audio = audio.astype(np.float32)
            else: audio = audio.astype(np.float32)
            if np.max(np.abs(audio)) > 1.0: audio = audio / np.max(np.abs(audio))
            return audio, sr
        except ImportError: raise ImportError('scipy not installed. Install: pip install scipy')

    def _validate_duration(self, duration: float, strict: bool) -> Tuple[bool, Optional[str]]:
        metrics = ReferenceQualityMetrics
        min_dur = metrics.DURATION_OPTIMAL_MIN if strict else metrics.DURATION_ABSOLUTE_MIN
        max_dur = metrics.DURATION_OPTIMAL_MAX if strict else metrics.DURATION_ABSOLUTE_MAX
        if duration < min_dur: return False, f'Duration {duration:.1f}s too short. Min: {min_dur}s'
        if duration > max_dur: return False, f'Duration {duration:.1f}s too long. Max: {max_dur}s'
        return True, None

    def _validate_rms(self, audio: np.ndarray) -> Tuple[bool, Optional[str], float]:
        metrics = ReferenceQualityMetrics
        rms = np.sqrt(np.mean(audio**2))
        if rms < metrics.RMS_MIN: return False, f'Audio too quiet (RMS: {rms:.4f})', 0.0
        if rms > metrics.RMS_MAX: return False, f'Audio too loud (RMS: {rms:.4f})', 0.0
        score = 1.0 - abs(rms - metrics.RMS_OPTIMAL) / metrics.RMS_OPTIMAL
        return True, None, max(0.0, min(1.0, score))

    def _validate_clipping(self, audio: np.ndarray) -> Tuple[bool, Optional[str], float]:
        metrics = ReferenceQualityMetrics
        clip_count = np.sum(np.abs(audio) >= metrics.CLIP_THRESHOLD)
        if clip_count > metrics.CLIP_MAX_SAMPLES: return False, f'Clipping detected ({clip_count} samples)', 0.0
        score = 1.0 - (clip_count / metrics.CLIP_MAX_SAMPLES) if metrics.CLIP_MAX_SAMPLES > 0 else 1.0
        return True, None, max(0.5, score)

    def _validate_dynamic_range(self, audio: np.ndarray) -> Tuple[bool, Optional[str], float]:
        metrics = ReferenceQualityMetrics
        peak = np.max(np.abs(audio))
        rms = np.sqrt(np.mean(audio**2))
        dynamic_range = peak / rms if rms > 0 else 0
        if dynamic_range < metrics.DYNAMIC_RANGE_MIN: return False, f'Dynamic range too low: {dynamic_range:.2f}', 0.0
        score = min(1.0, dynamic_range / 5.0)
        return True, None, score

    def _validate_silence(self, audio: np.ndarray, sr: int) -> Tuple[bool, Optional[str], float]:
        metrics = ReferenceQualityMetrics
        silence_threshold = 0.01
        silence_samples = np.sum(np.abs(audio) < silence_threshold)
        silence_ratio = silence_samples / len(audio)
        if silence_ratio > metrics.SILENCE_RATIO_MAX: return False, f'Too much silence: {silence_ratio*100:.1f}%', 0.0
        score = 1.0 - silence_ratio
        return True, None, score

    def _calculate_quality_score(self, duration: float, rms_score: float, clip_score: float, dyn_score: float, silence_score: float, strict: bool) -> float:
        metrics = ReferenceQualityMetrics
        optimal_min = metrics.DURATION_OPTIMAL_MIN if strict else metrics.DURATION_ABSOLUTE_MIN
        optimal_max = metrics.DURATION_OPTIMAL_MAX if strict else metrics.DURATION_ABSOLUTE_MAX
        optimal_mid = (optimal_min + optimal_max) / 2
        if duration < optimal_min: duration_score = duration / optimal_min
        elif duration > optimal_max: duration_score = optimal_max / duration
        else: duration_score = 1.0 - abs(duration - optimal_mid) / optimal_mid
        weights = {'duration': 0.30, 'rms': 0.25, 'clipping': 0.20, 'dynamic': 0.15, 'silence': 0.10}
        return round(weights['duration'] * duration_score + weights['rms'] * rms_score + weights['clipping'] * clip_score +
                    weights['dynamic'] * dyn_score + weights['silence'] * silence_score, 2)

    def _save_audio_file(self, audio: np.ndarray, sr: int, path: str) -> bool:
        try:
            from scipy.io import wavfile
            if np.max(np.abs(audio)) > 1.0: audio = audio / np.max(np.abs(audio))
            audio_int16 = (audio * 32767).astype(np.int16)
            wavfile.write(path, sr, audio_int16)
            return True
        except ImportError: print('[VoiceLibrary] scipy not installed. Install: pip install scipy'); return False
        except Exception as e: print(f'[VoiceLibrary] Audio save error: {e}'); return False

    def _calculate_duration(self, audio_path: str) -> float:
        try:
            audio, sr = self._load_audio(audio_path)
            return len(audio) / sr
        except Exception: return 0.0


print("[DIAGNOSTIC] SECTION 10: VOICE LIBRARY MANAGER - OK", flush=True)


# =============================================================================
# SECTION 11: COLAB INTEGRATION
# =============================================================================

_voice_library_instance: Optional[VoiceLibraryManager] = None
_clone_engine_instance: Optional[VoiceCloneEngine] = None
_design_engine_instance: Optional[VoiceDesignEngine] = None
_cache_instance: Optional[VoicePromptCache] = None
_is_colab_initialized: bool = False


def is_colab_environment() -> bool:
    try:
        import google.colab
        return True
    except ImportError: return False


def mount_google_drive() -> bool:
    if not is_colab_environment():
        logger.warning('Not in Colab environment - Drive mount skipped')
        return False
    if os.path.exists('/content/drive/MyDrive'):
        logger.info('Google Drive already mounted')
        return True
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        logger.info('Google Drive mounted')
        return True
    except Exception as e:
        logger.error(f'Drive mount error: {e}')
        return False


def init_voice_library_colab(drive_path: str = '/content/drive/MyDrive/vibemodly_voices',
                             cache_dir: str = '/content/cache/voice_library',
                             auto_mount_drive: bool = True, load_models: bool = True) -> VoiceLibraryManager:
    global _voice_library_instance, _clone_engine_instance, _design_engine_instance, _cache_instance, _is_colab_initialized
    if _voice_library_instance is not None and _is_colab_initialized:
        logger.info('VoiceLibrary already initialized')
        return _voice_library_instance
    logger.info('=== Initializing VoiceLibrary for Colab ===')
    if auto_mount_drive and is_colab_environment():
        if not mount_google_drive(): raise ColabInitError('Failed to mount Google Drive')
    os.makedirs(drive_path, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(os.path.join(drive_path, 'references'), exist_ok=True)
    os.makedirs(os.path.join(drive_path, 'backup'), exist_ok=True)
    logger.info(f'Directories created: {drive_path}')
    _cache_instance = VoicePromptCache(cache_dir)
    logger.info('Cache initialized')
    if load_models:
        try:
            # Створення двигунів
            _clone_engine_instance = VoiceCloneEngine(cache_dir=cache_dir)
            logger.info('VoiceCloneEngine created')
            _design_engine_instance = VoiceDesignEngine()
            logger.info('VoiceDesignEngine created')
            
            # Завантаження моделей (це може зайняти час)
            logger.info('Loading VoiceClone model...')
            clone_loaded = _clone_engine_instance.load_model()
            if clone_loaded:
                logger.info('VoiceClone model loaded successfully!')
            else:
                logger.warning('VoiceClone model failed to load')
            
            logger.info('Loading VoiceDesign model...')
            design_loaded = _design_engine_instance.load_model()
            if design_loaded:
                logger.info('VoiceDesign model loaded successfully!')
            else:
                logger.warning('VoiceDesign model failed to load')
            
            # Очищаємо GPU кеш після завантаження моделей
            try:
                import torch
                torch.cuda.empty_cache()
                logger.info('GPU cache cleared')
                
                # Показуємо використання GPU пам'яті
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    logger.info(f'GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved')
            except Exception:
                pass
                
        except Exception as e:
            logger.warning(f'Models not loaded: {e}')
            logger.info('Models can be loaded later via set_models() or engine.load_model()')
    _voice_library_instance = VoiceLibraryManager(voice_clone_engine=_clone_engine_instance,
                                                   voice_design_engine=_design_engine_instance,
                                                   cache=_cache_instance, drive_path=drive_path)
    _voice_library_instance.load_library()
    _is_colab_initialized = True
    logger.info('VoiceLibrary for Colab initialized!')
    return _voice_library_instance


def set_models(clone_model=None, design_model=None) -> None:
    global _clone_engine_instance, _design_engine_instance
    if clone_model is not None:
        if _clone_engine_instance is None: _clone_engine_instance = VoiceCloneEngine()
        _clone_engine_instance.set_model(clone_model)
        logger.info('VoiceClone model set')
    if design_model is not None:
        if _design_engine_instance is None: _design_engine_instance = VoiceDesignEngine()
        _design_engine_instance.set_model(design_model)
        logger.info('VoiceDesign model set')


def get_colab_stats() -> VoiceLibraryStats:
    global _voice_library_instance, _is_colab_initialized
    if _voice_library_instance is None: return VoiceLibraryStats()
    voices = _voice_library_instance.list_voices()
    return VoiceLibraryStats(total_voices=len(voices), total_duration_sec=sum(v.duration for v in voices),
                             average_quality=sum(v.quality_score for v in voices) / len(voices) if voices else 0.0,
                             drive_synced=_is_colab_initialized, last_sync_time=datetime.now().isoformat() if _is_colab_initialized else None)


def get_voice_library() -> VoiceLibraryManager:
    global _voice_library_instance
    if _voice_library_instance is None: return init_voice_library_colab()
    return _voice_library_instance


def get_voice_prompt(character_name: str, voice_preset: str = 'male_deep', gender: str = 'male', language: str = 'russian') -> Tuple[Optional[Dict[str, Any]], bool]:
    library = get_voice_library()
    return library.get_voice(character_name=character_name, voice_preset=voice_preset, gender=gender, language=language)


def create_voice_from_reference(character_name: str, reference_audio_path: str, reference_text: str,
                                voice_preset: str = 'custom', gender: str = 'male', language: str = 'russian') -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    global _voice_library_instance, _clone_engine_instance
    if _voice_library_instance is None: return None, 'Library not initialized. Call init_voice_library_colab()'
    if _clone_engine_instance is None: return None, 'VoiceClone engine not initialized'
    try:
        is_valid, error_msg = validate_reference_audio(reference_audio_path)
        if not is_valid: return None, f'Validation failed: {error_msg}'
        voice_prompt, _ = _clone_engine_instance.create_voice_prompt(reference_audio_path=reference_audio_path,
                                                                      reference_text=reference_text, use_cache=True, validate=True)
        if voice_prompt is None: return None, 'Failed to create voice prompt'
        audio, sr = load_audio_for_voice_clone(reference_audio_path)
        duration = len(audio) / sr
        entry = VoiceLibraryEntry(character_name=character_name, voice_preset=voice_preset,
                                  reference_audio_path=reference_audio_path, reference_text=reference_text,
                                  duration=duration, quality_score=0.85, created_at=datetime.now().isoformat(),
                                  updated_at=datetime.now().isoformat(),
                                  metadata={'gender': gender, 'language': language, 'source': 'custom_reference'})
        _voice_library_instance._library[character_name] = entry
        _voice_library_instance.save_library()
        logger.info(f'Voice "{character_name}" created from reference')
        return voice_prompt, None
    except Exception as e: return None, f'Error: {str(e)}'


def clone_voice_from_entry(character_name: str, text: str, speed: float = 1.0) -> GenerationResult:
    global _voice_library_instance, _clone_engine_instance
    if _voice_library_instance is None: return GenerationResult(success=False, error_message='Library not initialized')
    if _clone_engine_instance is None: return GenerationResult(success=False, error_message='VoiceClone engine not initialized')
    try:
        entry = _voice_library_instance.get_entry(character_name)
        if entry is None: return GenerationResult(success=False, error_message=f'Voice "{character_name}" not found in library')
        voice_prompt, is_new = _voice_library_instance.get_voice(character_name=character_name, voice_preset=entry.voice_preset,
                                                                  gender=entry.metadata.get('gender', 'male'), language=entry.metadata.get('language', 'russian'))
        if voice_prompt is None: return GenerationResult(success=False, error_message='Failed to get voice_prompt')
        audio, sr = _clone_engine_instance.generate(text=text, voice_config={'voice_prompt': voice_prompt, 'speed': speed},
                                                    language=entry.metadata.get('language', 'russian'))
        if audio is None: return GenerationResult(success=False, error_message='Generation failed')
        duration = len(audio) / sr
        return GenerationResult(success=True, audio=audio, sample_rate=sr, duration=duration, voice_id=character_name, is_new_voice=is_new)
    except Exception as e: return GenerationResult(success=False, error_message=f'Error: {str(e)}')


def list_available_voices() -> List[Dict[str, Any]]:
    library = get_voice_library()
    voices = library.list_voices()
    return [{'name': v.character_name, 'preset': v.voice_preset, 'duration': v.duration, 'quality': v.quality_score,
             'gender': v.metadata.get('gender', 'unknown'), 'language': v.metadata.get('language', 'unknown'), 'created': v.created_at} for v in voices]


def delete_voice_from_library(character_name: str) -> bool:
    library = get_voice_library()
    result = library.delete_voice(character_name)
    if result: library.save_library(); logger.info(f'Voice "{character_name}" deleted')
    return result


def generate_audio_with_library(text: str, character_name: str = 'narrator', voice_preset: str = 'male_deep',
                                gender: str = 'male', language: str = 'russian', speed: float = 1.0,
                                use_library: bool = True, fallback_to_design: bool = True) -> GenerationResult:
    global _voice_library_instance, _clone_engine_instance, _design_engine_instance
    if not use_library: return _generate_without_library(text=text, voice_preset=voice_preset, gender=gender, language=language, speed=speed)
    if _voice_library_instance is None:
        logger.warning('Library not initialized, attempting initialization...')
        try: init_voice_library_colab(load_models=False)
        except Exception as e: return GenerationResult(success=False, error_message=f'Failed to initialize library: {e}')
    try:
        voice_prompt, is_new = _voice_library_instance.get_voice(character_name=character_name, voice_preset=voice_preset, gender=gender, language=language)
        if voice_prompt is None:
            if fallback_to_design:
                logger.warning(f'Voice prompt not obtained, fallback to VoiceDesign')
                return _generate_without_library(text=text, voice_preset=voice_preset, gender=gender, language=language, speed=speed)
            else: return GenerationResult(success=False, error_message='Failed to get voice_prompt')
        if _clone_engine_instance is None: return GenerationResult(success=False, error_message='VoiceClone engine not initialized')
        logger.info(f'Generating via VoiceClone for "{character_name}"')
        audio, sr = _clone_engine_instance.generate(text=text, voice_config={'voice_prompt': voice_prompt, 'speed': speed}, language=language)
        if audio is None:
            if fallback_to_design:
                logger.warning('VoiceClone generation failed, fallback to VoiceDesign')
                return _generate_without_library(text=text, voice_preset=voice_preset, gender=gender, language=language, speed=speed)
            else: return GenerationResult(success=False, error_message='VoiceClone generation failed')
        duration = len(audio) / sr
        logger.info(f'Audio generated: {duration:.2f}s')
        return GenerationResult(success=True, audio=audio, sample_rate=sr, duration=duration, voice_id=character_name, is_new_voice=is_new)
    except Exception as e:
        logger.error(f'Generation error: {e}')
        return GenerationResult(success=False, error_message=str(e))


def _generate_without_library(text: str, voice_preset: str, gender: str, language: str, speed: float) -> GenerationResult:
    global _design_engine_instance
    if _design_engine_instance is None: return GenerationResult(success=False, error_message='VoiceDesign engine not initialized')
    try:
        seed = hash(voice_preset) % (2**32)
        gender_prefix = 'Male voice' if gender == 'male' else 'Female voice'
        prompt = f'{gender_prefix}, natural and clear.'
        audio, sr = _design_engine_instance.generate(text=text, voice_config={'seed': seed, 'prompt': prompt, 'speed': speed}, language=language)
        if audio is None: return GenerationResult(success=False, error_message='VoiceDesign generation failed')
        duration = len(audio) / sr
        return GenerationResult(success=True, audio=audio, sample_rate=sr, duration=duration, voice_id=f'design_{voice_preset}', is_new_voice=True)
    except Exception as e: return GenerationResult(success=False, error_message=str(e))


def generate_audio_batch(texts: List[str], character_name: str = 'narrator', voice_preset: str = 'male_deep',
                         gender: str = 'male', language: str = 'russian', speed: float = 1.0) -> List[GenerationResult]:
    results = []
    voice_prompt, is_new = get_voice_prompt(character_name=character_name, voice_preset=voice_preset, gender=gender, language=language)
    if voice_prompt is None: return [GenerationResult(success=False, error_message='Failed to get voice_prompt') for _ in texts]
    for text in texts:
        result = generate_audio_with_library(text=text, character_name=character_name, voice_preset=voice_preset,
                                             gender=gender, language=language, speed=speed, use_library=True)
        results.append(result)
    return results


def demo_voice_library(test_text: str = 'Hello! This is a test text for voice demonstration.',
                       character_name: str = 'demo_narrator', voice_preset: str = 'male_deep',
                       gender: str = 'male', language: str = 'russian') -> None:
    print('\n' + '='*60)
    print('=== Voice Library Demo ===')
    print('='*60 + '\n')
    print('Step 1: Initializing library...')
    try:
        library = init_voice_library_colab(load_models=False)
        print('Library initialized')
    except Exception as e:
        print(f'Initialization error: {e}')
        return
    print('\nStep 2: Library stats (before)...')
    stats = get_colab_stats()
    print(f'   * Voices: {stats.total_voices}')
    print(f'   * Average quality: {stats.average_quality:.2f}')
    print(f'\nStep 3: Getting voice "{character_name}"...')
    voice_prompt, is_new = get_voice_prompt(character_name=character_name, voice_preset=voice_preset, gender=gender, language=language)
    if voice_prompt is not None:
        status = 'new' if is_new else 'from library'
        print(f'Voice obtained ({status})')
    else:
        print('Failed to get voice')
        return
    print(f'\nStep 4: Generating audio...')
    print(f'   Text: "{test_text[:50]}..."')
    result = generate_audio_with_library(text=test_text, character_name=character_name, voice_preset=voice_preset, gender=gender, language=language)
    if result.success:
        print(f'Audio generated:')
        print(f'   * Duration: {result.duration:.2f}s')
        print(f'   * Sample rate: {result.sample_rate} Hz')
        print(f'   * New voice: {result.is_new_voice}')
    else:
        print(f'Generation error: {result.error_message}')
    print('\nStep 5: Library stats (after)...')
    stats = get_colab_stats()
    print(f'   * Voices: {stats.total_voices}')
    print(f'   * Total duration: {stats.total_duration_sec:.2f}s')
    print(f'   * Average quality: {stats.average_quality:.2f}')
    print('\nStep 6: List of voices in library...')
    voices = list_available_voices()
    for v in voices: print(f'   * {v["name"]}: {v["duration"]:.1f}s, quality {v["quality"]:.0%}')
    print('\n' + '='*60)
    print('=== Demo completed ===')
    print('='*60 + '\n')


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'VoiceMode', 'ValidationLevel', 'VoiceProfile', 'VoiceLibraryEntry', 'ColabVoiceConfig',
    'VoiceLibraryStats', 'GenerationResult', 'DriveSyncError', 'ColabInitError', 'VoiceNotFoundError',
    'VoicePromptCache', 'TTSEngine', 'VoiceDesignEngine', 'VoiceCloneEngine', 'VoiceLibraryManager',
    'validate_reference_audio', 'load_audio_for_voice_clone', 'get_cache_key',
    'is_colab_environment', 'mount_google_drive', 'init_voice_library_colab', 'set_models', 'get_colab_stats',
    'get_voice_library', 'get_voice_prompt', 'create_voice_from_reference', 'clone_voice_from_entry',
    'list_available_voices', 'delete_voice_from_library', 'generate_audio_with_library', 'generate_audio_batch',
    'demo_voice_library',
]

__version__ = '1.3.1'
__author__ = 'VIBEMODLY Team'

print("[DIAGNOSTIC] SECTION 11: COLAB INTEGRATION - OK", flush=True)
print("[DIAGNOSTIC] ========================================", flush=True)
print("[DIAGNOSTIC] voice_library_all_in_one LOADED SUCCESSFULLY!", flush=True)
print("[DIAGNOSTIC] ========================================", flush=True)


# =============================================================================
# AUTO-INITIALIZATION FOR COLAB
# =============================================================================

def _auto_init_for_colab() -> None:
    """
    Автоматична ініціалізація при імпорті в Google Colab.
    Монтує Google Drive, створює необхідні директорії та завантажує моделі.
    Після завантаження генерує тестове аудіо та відправляє в Telegram бот.
    """
    global _voice_library_instance, _is_colab_initialized
    
    if not is_colab_environment():
        print("[VoiceLibrary] Not in Colab environment - auto-init skipped", flush=True)
        return
    
    if _voice_library_instance is not None and _is_colab_initialized:
        print("[VoiceLibrary] Already initialized", flush=True)
        return
    
    print("\n" + "="*60, flush=True)
    print("[VoiceLibrary] AUTO-INITIALIZATION FOR COLAB", flush=True)
    print("="*60, flush=True)
    
    # Отримуємо Telegram credentials з Colab secrets
    TELEGRAM_TOKEN = None
    TELEGRAM_CHAT_ID = None
    
    try:
        from google.colab import userdata
        try:
            TELEGRAM_TOKEN = userdata.get("TELEGRAM_BOT_TOKEN")
            print("[VoiceLibrary] ✓ TELEGRAM_BOT_TOKEN found", flush=True)
        except:
            print("[VoiceLibrary] ⚠ TELEGRAM_BOT_TOKEN not found in secrets", flush=True)
        
        try:
            TELEGRAM_CHAT_ID = userdata.get("TELEGRAM_CHAT_ID")
            print("[VoiceLibrary] ✓ TELEGRAM_CHAT_ID found", flush=True)
        except:
            print("[VoiceLibrary] ⚠ TELEGRAM_CHAT_ID not found in secrets", flush=True)
    except Exception as e:
        print(f"[VoiceLibrary] ⚠ Could not access Colab secrets: {e}", flush=True)
    
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("\n" + "="*60, flush=True)
        print("⚠️  TELEGRAM CONFIGURATION REQUIRED", flush=True)
        print("="*60, flush=True)
        print("Add these secrets in Colab (🔑 icon in left sidebar):", flush=True)
        print("  1. TELEGRAM_BOT_TOKEN - your bot token from @BotFather", flush=True)
        print("  2. TELEGRAM_CHAT_ID - your chat ID (send /start to @userinfobot)", flush=True)
        print("="*60 + "\n", flush=True)
    
    try:
        # Ініціалізація з завантаженням моделей
        init_voice_library_colab(load_models=True)
        print("[VoiceLibrary] Auto-initialization completed!", flush=True)
        print("[VoiceLibrary] Models loaded and ready to use!", flush=True)
        print("\n" + "="*60, flush=True)
        print("✓ VoiceLibrary is ready!", flush=True)
        print("="*60 + "\n", flush=True)
        
        # Автоматична демонстрація - генеруємо тестове аудіо
        print("\n" + "="*60, flush=True)
        print("🎤 AUTO-DEMO: Generating test audio...", flush=True)
        print("="*60 + "\n", flush=True)
        
        test_text = "Привіт! Це тестовий голос для перевірки системи Voice Library."
        print(f"[Demo] Text: '{test_text}'", flush=True)
        print(f"[Demo] Generating audio...", flush=True)
        
        result = generate_audio_with_library(
            text=test_text,
            character_name='demo_narrator',
            voice_preset='male_deep',
            gender='male',
            language='russian'
        )
        
        if result.success:
            print(f"\n✅ SUCCESS! Audio generated:", flush=True)
            print(f"   Duration: {result.duration:.2f} seconds", flush=True)
            print(f"   Sample rate: {result.sample_rate} Hz", flush=True)
            print(f"   Voice ID: {result.voice_id}", flush=True)
            print(f"   Is new voice: {result.is_new_voice}", flush=True)
            
            # Зберігаємо аудіо у файл
            try:
                from scipy.io import wavfile
                from io import BytesIO
                
                output_path = '/content/demo_output.wav'
                
                # Нормалізація аудіо
                audio = result.audio
                max_val = np.max(np.abs(audio))
                if max_val > 0:
                    audio = audio / max_val * 0.9
                
                audio_int16 = (audio * 32767).astype(np.int16)
                wavfile.write(output_path, result.sample_rate, audio_int16)
                print(f"\n💾 Audio saved to: {output_path}", flush=True)
                
                # Відправляємо в Telegram якщо є credentials
                if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
                    print(f"\n📤 Sending audio to Telegram...", flush=True)
                    try:
                        import telebot
                        from pydub import AudioSegment
                        
                        # Створюємо бота
                        bot = telebot.TeleBot(TELEGRAM_TOKEN)
                        
                        # Конвертуємо в MP3 для меншого розміру
                        segment = AudioSegment(
                            audio_int16.tobytes(),
                            frame_rate=result.sample_rate,
                            sample_width=2,
                            channels=1
                        )
                        
                        buf = BytesIO()
                        segment.export(buf, format="mp3")
                        buf.seek(0)
                        
                        # Відправляємо аудіо
                        caption = f"""🎙 Voice Library Test Audio

📝 Text: "{test_text[:50]}..."
⏱ Duration: {result.duration:.2f}s
🎵 Sample rate: {result.sample_rate} Hz
🎭 Voice: male_deep
🆔 Voice ID: {result.voice_id}

✅ VoiceLibrary is working correctly!"""
                        
                        bot.send_audio(
                            int(TELEGRAM_CHAT_ID),
                            buf,
                            caption=caption,
                            parse_mode="HTML"
                        )
                        
                        print(f"✅ Audio sent to Telegram successfully!", flush=True)
                        print(f"   Chat ID: {TELEGRAM_CHAT_ID}", flush=True)
                        
                    except Exception as e:
                        print(f"❌ Failed to send to Telegram: {e}", flush=True)
                        import traceback
                        traceback.print_exc()
                else:
                    # Якщо немає Telegram credentials - відтворюємо в Colab
                    try:
                        from IPython.display import Audio, display
                        print(f"\n🔊 Playing audio in Colab...", flush=True)
                        display(Audio(output_path))
                    except Exception:
                        print(f"[Demo] Could not play audio (IPython not available)", flush=True)
                        
            except Exception as e:
                print(f"[Demo] Could not save audio: {e}", flush=True)
        else:
            print(f"\n❌ FAILED: {result.error_message}", flush=True)
            
            # Відправляємо повідомлення про помилку в Telegram
            if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
                try:
                    import telebot
                    bot = telebot.TeleBot(TELEGRAM_TOKEN)
                    bot.send_message(
                        int(TELEGRAM_CHAT_ID),
                        f"❌ Voice Library Test Failed\n\nError: {result.error_message}"
                    )
                except Exception:
                    pass
        
        print("\n" + "="*60, flush=True)
        print("✓ Demo completed! VoiceLibrary is ready to use.", flush=True)
        print("="*60 + "\n", flush=True)
        
    except Exception as e:
        print(f"[VoiceLibrary] Auto-initialization failed: {e}", flush=True)
        print("[VoiceLibrary] You can manually call init_voice_library_colab() later", flush=True)
        
        # Відправляємо повідомлення про помилку в Telegram
        if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
            try:
                import telebot
                bot = telebot.TeleBot(TELEGRAM_TOKEN)
                bot.send_message(
                    int(TELEGRAM_CHAT_ID),
                    f"❌ Voice Library Initialization Failed\n\nError: {str(e)}"
                )
            except Exception:
                pass
        
        print("="*60 + "\n", flush=True)


# Автоматична ініціалізація при імпорті
_auto_init_for_colab()
