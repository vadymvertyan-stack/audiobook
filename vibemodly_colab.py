# =============================================================================
# VIBEMODLY v69.0 - VOICE CLONE CONSISTENCY
# =============================================================================
# Qwen3-TTS 1.7B VoiceDesign + Base - консистентність голосу через клонування
# Перший виклик: VoiceDesign (seed + prompt) -> зберігаємо як reference
# Наступні виклики: VoiceClone з reference аудіо для консистентності
# =============================================================================

import subprocess
import sys
import os

# Детермінізм для консистентності голосу - МАЄ БУТИ ДО імпорту torch!
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import threading

print("=" * 60, flush=True)
print("  VIBEMODLY v68.0 - VOICEDESIGN + SEED", flush=True)
print("=" * 60, flush=True)

# --- 1. ВСТАНОВЛЕННЯ ЗАЛЕЖНОСТЕЙ ---
print("[INSTALL] Встановлення бібліотек...", flush=True)

# Google Genai
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "google-genai"], shell=False)

# PyTorch з CUDA
subprocess.run(f"{sys.executable} -m pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cu118", shell=True)

# Transformers та qwen-tts
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers"], shell=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "accelerate"], shell=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "soundfile"], shell=False)

# Qwen-TTS - головна бібліотека!
print("[INSTALL] qwen-tts (зачекайте...)...", flush=True)
result = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "qwen-tts"], shell=False, capture_output=True)
print(f"[INSTALL] qwen-tts: {'OK' if result.returncode == 0 else 'ERROR'}", flush=True)

# Інші
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyTelegramBotAPI"], shell=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pydub"], shell=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ebooklib", "PyPDF2", "pdfplumber", "beautifulsoup4", "lxml", "python-docx"], shell=False)
subprocess.run("apt-get install -y ffmpeg > /dev/null 2>&1", shell=True)

print("[INSTALL] Готово!", flush=True)

# =============================================================================
# ДІАГНОСТИКА GEMINI API
# =============================================================================

print("\n" + "=" * 60, flush=True)
print("  ДІАГНОСТИКА GEMINI API", flush=True)
print("=" * 60 + "\n", flush=True)

from google import genai
from google.colab import userdata

GEMINI_KEY = None
for name in ["GEMINI_API_KEY", "GOOGLE_API_KEY"]:
    try:
        key = userdata.get(name)
        if key:
            GEMINI_KEY = key.strip()
            print(f"[OK] Ключ: {name}", flush=True)
            break
    except:
        pass

if not GEMINI_KEY:
    print("[ERROR] Додайте GEMINI_API_KEY в Secrets!", flush=True)
    sys.exit(1)

client = genai.Client(api_key=GEMINI_KEY)

# Тест моделі
WORKING_MODEL = None
for model_name in ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash"]:
    try:
        print(f"[TEST] {model_name}...", end=" ", flush=True)
        client.models.generate_content(model=model_name, contents="Hi")
        print("✅", flush=True)
        WORKING_MODEL = model_name
        break
    except Exception as e:
        if "429" in str(e):
            print("⚠️ (квота)", flush=True)
            WORKING_MODEL = model_name
            break
        print("❌", flush=True)

if not WORKING_MODEL:
    print("[ERROR] Gemini не працює!", flush=True)
    sys.exit(1)

print(f"[OK] Модель: {WORKING_MODEL}", flush=True)

# =============================================================================
# ЗАВАНТАЖЕННЯ QWEN-TTS
# =============================================================================

print("\n" + "=" * 60, flush=True)
print("  ЗАВАНТАЖЕННЯ QWEN-TTS (4GB)", flush=True)
print("=" * 60 + "\n", flush=True)

import torch
import numpy as np
from io import BytesIO
from pydub import AudioSegment
import json
import time
import requests
import pickle
from datetime import datetime

# Детерміновані алгоритми PyTorch для консистентності голосу
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =============================================================================
# ФУНКЦІЇ ЗБЕРЕЖЕННЯ/ВІДНОВЛЕННЯ СТАНУ ГЕНЕРАЦІЇ
# =============================================================================

# Файл для збереження стану
STATE_FILE = "vibemodly_state.pkl"

def save_generation_state(state_data):
    """Зберігає стан генерації у файл
    
    Args:
        state_data: Словник з даними стану генерації
        
    Returns:
        bool: True якщо збереження успішне, False інакше
    """
    try:
        state_data['saved_at'] = datetime.now().isoformat()
        with open(STATE_FILE, 'wb') as f:
            pickle.dump(state_data, f)
        print(f"[STATE] Стан збережено: {state_data.get('current_step', 'unknown')}", flush=True)
        return True
    except Exception as e:
        print(f"[STATE] Помилка збереження: {e}", flush=True)
        return False

def load_generation_state():
    """Завантажує збережений стан генерації
    
    Returns:
        dict or None: Словник з даними стану або None якщо файл не існує/помилка
    """
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'rb') as f:
                state = pickle.load(f)
            print(f"[STATE] Завантажено стан від: {state.get('saved_at', 'unknown')}", flush=True)
            return state
    except Exception as e:
        print(f"[STATE] Помилка завантаження: {e}", flush=True)
    return None

def clear_generation_state():
    """Видаляє файл збереженого стану"""
    try:
        if os.path.exists(STATE_FILE):
            os.remove(STATE_FILE)
            print("[STATE] Файл стану видалено", flush=True)
    except Exception as e:
        print(f"[STATE] Помилка видалення: {e}", flush=True)

def has_saved_state():
    """Перевіряє наявність збереженого стану
    
    Returns:
        bool: True якщо файл стану існує, False інакше
    """
    return os.path.exists(STATE_FILE)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INIT] Пристрій: {DEVICE}", flush=True)

# Глобальні змінні
tts_model_voicedesign = None  # Для створення унікального голосу через prompt + seed
tts_model_base = None  # Для клонування голосу (VoiceClone)
SAMPLE_RATE = 24000
TTS_MODEL_VOICEDESIGN = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
TTS_MODEL_BASE = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"  # Модель для клонування

# Кеш voice_prompt для кожного персонажа (для консистентності голосу)
# Формат: {character_name: {"voice_prompt": dict, "reference_audio": np.ndarray, "reference_text": str}}
character_voice_prompts = {}

def init_tts():
    """Завантаження моделей Qwen3-TTS (VoiceDesign + Base для клонування)"""
    global tts_model_voicedesign, tts_model_base
    
    print("[TTS] Завантаження моделей Qwen3-TTS...", flush=True)
    print("[TTS] Це займе 2-3 хвилини. Не переривайте!", flush=True)
    
    try:
        from qwen_tts import Qwen3TTSModel
        
        # VoiceDesign модель - для створення унікальних голосів через prompt + seed
        print("[TTS] Завантаження VoiceDesign моделі...", flush=True)
        tts_model_voicedesign = Qwen3TTSModel.from_pretrained(
            TTS_MODEL_VOICEDESIGN,
            device_map="cuda:0" if DEVICE == "cuda" else "cpu",
            dtype=torch.float16 if DEVICE == "cuda" else torch.float32
        )
        print("[TTS] ✅ VoiceDesign готова!", flush=True)
        
        # Base модель - для клонування голосу (консистентність між репліками)
        print("[TTS] Завантаження Base моделі для клонування...", flush=True)
        try:
            tts_model_base = Qwen3TTSModel.from_pretrained(
                TTS_MODEL_BASE,
                device_map="auto" if DEVICE == "cuda" else "cpu",
                torch_dtype=torch.bfloat16 if DEVICE == "cuda" and torch.cuda.is_bf16_supported() else torch.float16 if DEVICE == "cuda" else torch.float32
            )
            print("[TTS] ✅ Base модель готова! Підтримка клонування голосу активована.", flush=True)
        except Exception as e:
            print(f"[TTS] ⚠️ Base модель не завантажена: {e}", flush=True)
            print("[TTS] ⚠️ Клонування голосу недоступне, використовуємо тільки VoiceDesign.", flush=True)
            tts_model_base = None
        
        print("[TTS] ✅ Всі моделі завантажено!", flush=True)
        return True
    except Exception as e:
        print(f"[TTS] ❌ Помилка: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return False

# Завантажуємо модель одразу
init_tts()

# =============================================================================
# ГЕНЕРАЦІЯ АУДІО
# =============================================================================

import hashlib

# =============================================================================
# VOICEBOX ADAPTER - ВБУДОВАНИЙ КОД
# =============================================================================
# Код адаптера вбудовано напряму для роботи без зовнішніх залежностей
# Джерело: voicebox_adapter_colab.py

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any, List, Union
from pathlib import Path
from enum import Enum

# Адаптер завжди доступний, оскільки код вбудовано
VOICEBOX_ADAPTER_AVAILABLE = True
print("[ADAPTER] ✅ Voicebox Adapter вбудовано (без зовнішніх імпортів)", flush=True)

# =============================================================================
# VOICE MODE
# =============================================================================

class VoiceMode(Enum):
    """Режим генерації голосу."""
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
            raise ValueError(f"Невідомий режим голосу: {value}")


# =============================================================================
# ДОПОМІЖНІ ФУНКЦІЇ
# =============================================================================

def validate_reference_audio(audio_path: str, min_duration: float = 2.0, max_duration: float = 30.0, min_rms: float = 0.01) -> Tuple[bool, Optional[str]]:
    """Валідація reference аудіо для клонування голосу."""
    try:
        if not os.path.exists(audio_path):
            return False, f"Файл не знайдено: {audio_path}"
        audio, sr = load_audio_for_voice_clone(audio_path)
        duration = len(audio) / sr
        if duration < min_duration:
            return False, f"Аудіо занадто коротке ({duration:.1f}с). Мінімум: {min_duration}с"
        if duration > max_duration:
            return False, f"Аудіо занадто довге ({duration:.1f}с). Максимум: {max_duration}с"
        rms = np.sqrt(np.mean(audio**2))
        if rms < min_rms:
            return False, f"Аудіо занадто тихе (RMS: {rms:.4f}). Мінімум: {min_rms}"
        if np.abs(audio).max() > 0.99:
            return False, "Аудіо має кліпінг (перевищення амплітуди). Зменшіть гучність"
        return True, None
    except Exception as e:
        return False, f"Помилка валідації аудіо: {str(e)}"


def load_audio_for_voice_clone(audio_path: str, sample_rate: int = 24000, mono: bool = True, normalize: bool = True) -> Tuple[np.ndarray, int]:
    """Завантаження та підготовка аудіо для клонування голосу."""
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
            raise ImportError("Потрібно встановити librosa або soundfile: pip install librosa soundfile")


def _normalize_audio(audio: np.ndarray, target_db: float = -20.0, peak_limit: float = 0.85) -> np.ndarray:
    """Нормалізація аудіо до цільового рівня гучності."""
    audio = audio.astype(np.float32)
    rms = np.sqrt(np.mean(audio**2))
    target_rms = 10**(target_db / 20)
    if rms > 0:
        gain = target_rms / rms
        audio = audio * gain
    audio = np.clip(audio, -peak_limit, peak_limit)
    return audio


def get_cache_key(audio_path: str, reference_text: str) -> str:
    """Генерація ключа кешу на основі аудіо файлу та тексту."""
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    combined = audio_bytes + reference_text.encode("utf-8")
    return hashlib.md5(combined).hexdigest()


# =============================================================================
# VOICE PROMPT CACHE
# =============================================================================

class VoicePromptCache:
    """Кешування voice prompts для пришвидшення генерації."""
    
    def __init__(self, cache_dir: str = "/content/cache", max_memory_items: int = 100):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: Dict[str, Any] = {}
        self._max_memory_items = max_memory_items
        self._hits = 0
        self._misses = 0
        print(f"[VoicePromptCache] Ініціалізовано: {cache_dir}", flush=True)
    
    def get_cache_key(self, *args) -> str:
        content_parts = []
        for arg in args:
            if isinstance(arg, dict):
                content_parts.append(json.dumps(arg, sort_keys=True))
            elif isinstance(arg, (list, tuple)):
                content_parts.append(str(tuple(arg)))
            else:
                content_parts.append(str(arg))
        content = ":".join(content_parts)
        return hashlib.md5(content.encode()).hexdigest()
    
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
            except Exception as e:
                print(f"[CACHE] Помилка читання: {e}", flush=True)
        self._misses += 1
        return None
    
    def set(self, key: str, value: Any, save_to_disk: bool = True) -> None:
        self._set_memory_cache(key, value)
        if save_to_disk:
            try:
                import torch
                cache_file = self.cache_dir / f"{key}.pt"
                if hasattr(value, 'numpy'):
                    torch.save(value, cache_file)
                elif hasattr(value, '__array__'):
                    tensor = torch.from_numpy(value) if hasattr(value, 'dtype') else torch.tensor(value)
                    torch.save(tensor, cache_file)
                else:
                    torch.save(value, cache_file)
            except Exception as e:
                print(f"[CACHE] Помилка збереження: {e}", flush=True)
    
    def _set_memory_cache(self, key: str, value: Any) -> None:
        if len(self._memory_cache) >= self._max_memory_items:
            keys_to_remove = list(self._memory_cache.keys())[:self._max_memory_items // 2]
            for k in keys_to_remove:
                del self._memory_cache[k]
        self._memory_cache[key] = value
    
    def has(self, key: str) -> bool:
        return key in self._memory_cache or (self.cache_dir / f"{key}.pt").exists()
    
    def delete(self, key: str) -> bool:
        deleted = False
        if key in self._memory_cache:
            del self._memory_cache[key]
            deleted = True
        cache_file = self.cache_dir / f"{key}.pt"
        if cache_file.exists():
            try:
                cache_file.unlink()
                deleted = True
            except Exception:
                pass
        return deleted
    
    def clear(self) -> None:
        self._memory_cache.clear()
        try:
            for f in self.cache_dir.glob("*.pt"):
                f.unlink()
        except Exception:
            pass
    
    def get_stats(self) -> Dict[str, Any]:
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0
        return {"hits": self._hits, "misses": self._misses, "hit_rate": round(hit_rate, 2)}
    
    def save_audio_cache(self, text: str, voice_config: Dict[str, Any], audio_data: Any) -> str:
        key = self.get_cache_key(text, voice_config)
        self.set(key, audio_data)
        return key
    
    def get_audio_cache(self, text: str, voice_config: Dict[str, Any]) -> Optional[Any]:
        key = self.get_cache_key(text, voice_config)
        return self.get(key)


# =============================================================================
# VOICE PROFILE
# =============================================================================

@dataclass
class VoiceProfile:
    """Уніфікований голосовий профіль."""
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
        pitch_str = pitch_desc.get(self.pitch, "medium-pitched")
        return f"{gender_prefix}, {pitch_str}. {self.description}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "mode": self.mode.value,
            "seed": self.seed, "prompt": self.prompt,
            "reference_audio_path": self.reference_audio_path,
            "reference_text": self.reference_text,
            "gender": self.gender, "language": self.language,
            "speed": self.speed, "pitch": self.pitch,
            "description": self.description, "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VoiceProfile':
        mode = VoiceMode.from_string(data.get("mode", "voicedesign"))
        return cls(
            id=data["id"], name=data["name"], mode=mode,
            seed=data.get("seed"), prompt=data.get("prompt"),
            reference_audio_path=data.get("reference_audio_path"),
            reference_text=data.get("reference_text"),
            gender=data.get("gender", "male"), language=data.get("language", "russian"),
            speed=data.get("speed", 1.0), pitch=data.get("pitch", "medium"),
            description=data.get("description", ""), tags=data.get("tags", [])
        )
    
    @classmethod
    def from_preset(cls, key: str, preset: Dict[str, Any]) -> 'VoiceProfile':
        gender = preset.get("gender", "male")
        pitch = preset.get("pitch", "medium")
        prompt_suffix = preset.get("prompt_suffix", "")
        gender_prefix = "Male voice" if gender == "male" else "Female voice"
        pitch_desc = {"low": "low-pitched", "medium": "medium-pitched", "high": "high-pitched"}
        prompt = f"{gender_prefix}, {pitch_desc.get(pitch, 'medium')}. {prompt_suffix}"
        return cls(
            id=key, name=preset.get("name", key), mode=VoiceMode.VOICE_DESIGN,
            seed=preset.get("seed"), prompt=prompt, gender=gender,
            language="russian", speed=preset.get("speed", 1.0), pitch=pitch,
            description=preset.get("description", ""), tags=[gender, pitch]
        )


# =============================================================================
# VOICE PROFILE MANAGER
# =============================================================================

class VoiceProfileManager:
    """Менеджер для управління голосовими профілями."""
    
    def __init__(self, voice_presets: Optional[Dict[str, Dict]] = None):
        self._profiles: Dict[str, VoiceProfile] = {}
        self._presets: Dict[str, Dict] = voice_presets or {}
    
    def set_presets(self, voice_presets: Dict[str, Dict]) -> None:
        self._presets = voice_presets
    
    def get_profile(self, character_name: str, voice_preset: Optional[str] = None) -> VoiceProfile:
        if voice_preset and voice_preset in self._presets:
            preset = self._presets[voice_preset]
            return VoiceProfile.from_preset(voice_preset, preset)
        profile_key = f"char_{character_name}"
        if profile_key in self._profiles:
            return self._profiles[profile_key]
        default_profile = VoiceProfile(
            id=profile_key, name=character_name, mode=VoiceMode.VOICE_DESIGN,
            gender="male", language="russian"
        )
        self._profiles[profile_key] = default_profile
        return default_profile
    
    def create_profile(self, profile: VoiceProfile) -> None:
        self._profiles[profile.id] = profile
    
    def update_profile(self, profile: VoiceProfile) -> None:
        self._profiles[profile.id] = profile
    
    def delete_profile(self, profile_id: str) -> bool:
        if profile_id in self._profiles:
            del self._profiles[profile_id]
            return True
        return False
    
    def get_all_profiles(self) -> List[VoiceProfile]:
        return list(self._profiles.values())
    
    def get_profiles_by_gender(self, gender: str) -> List[VoiceProfile]:
        return [p for p in self._profiles.values() if p.gender == gender]
    
    def get_profiles_by_mode(self, mode: VoiceMode) -> List[VoiceProfile]:
        return [p for p in self._profiles.values() if p.mode == mode]
    
    def get_available_presets(self, gender: Optional[str] = None) -> List[Dict[str, Any]]:
        presets = []
        for key, preset in self._presets.items():
            if gender is None or preset.get("gender") == gender:
                presets.append({
                    "key": key, "name": preset.get("name", key),
                    "description": preset.get("description", ""),
                    "gender": preset.get("gender", "male")
                })
        return presets
    
    def save_profiles(self, filepath: str) -> bool:
        try:
            data = {"profiles": [p.to_dict() for p in self._profiles.values()], "presets": self._presets}
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"[VoiceProfileManager] Помилка збереження: {e}", flush=True)
            return False
    
    def load_profiles(self, filepath: str) -> bool:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            for profile_data in data.get("profiles", []):
                profile = VoiceProfile.from_dict(profile_data)
                self._profiles[profile.id] = profile
            if "presets" in data:
                self._presets.update(data["presets"])
            return True
        except Exception as e:
            print(f"[VoiceProfileManager] Помилка завантаження: {e}", flush=True)
            return False
    
    def create_clone_profile(self, profile_id: str, name: str, reference_audio_path: str,
                            reference_text: str, gender: str = "male", language: str = "russian",
                            speed: float = 1.0) -> VoiceProfile:
        profile = VoiceProfile(
            id=profile_id, name=name, mode=VoiceMode.VOICE_CLONE,
            reference_audio_path=reference_audio_path, reference_text=reference_text,
            gender=gender, language=language, speed=speed, description=f"Клонований голос: {name}"
        )
        self._profiles[profile_id] = profile
        return profile


# =============================================================================
# TTS ENGINE (ABSTRACT)
# =============================================================================

class TTSEngine(ABC):
    """Абстрактний інтерфейс TTS двигуна."""
    
    @abstractmethod
    def generate(self, text: str, voice_config: Dict[str, Any], language: str = "russian") -> Tuple[Optional[np.ndarray], int]:
        pass
    
    @abstractmethod
    def is_loaded(self) -> bool:
        pass
    
    @property
    @abstractmethod
    def engine_type(self) -> str:
        pass


# =============================================================================
# VOICE DESIGN ENGINE
# =============================================================================

class VoiceDesignEngine(TTSEngine):
    """VoiceDesign двигун - створення унікального голосу через prompt + seed."""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"):
        self._model = None
        self._model_name = model_name
        self._sample_rate = 24000
    
    def load_model(self, device: str = "cuda") -> bool:
        try:
            import torch
            from qwen_tts import Qwen3TTSModel
            print(f"[VoiceDesign] Завантаження моделі {self._model_name}...", flush=True)
            self._model = Qwen3TTSModel.from_pretrained(
                self._model_name,
                device_map=f"{device}:0" if device == "cuda" else "cpu",
                dtype=torch.float16 if device == "cuda" else torch.float32
            )
            print(f"[VoiceDesign] ✅ Модель завантажено!", flush=True)
            return True
        except Exception as e:
            print(f"[VoiceDesign] ❌ Помилка завантаження: {e}", flush=True)
            return False
    
    def set_model(self, model) -> None:
        self._model = model
    
    def is_loaded(self) -> bool:
        return self._model is not None
    
    @property
    def engine_type(self) -> str:
        return "voicedesign"
    
    def generate(self, text: str, voice_config: Dict[str, Any], language: str = "russian") -> Tuple[Optional[np.ndarray], int]:
        if not self.is_loaded():
            print("[VoiceDesign] ❌ Модель не завантажена!", flush=True)
            return None, 0
        try:
            import torch
            import random
            seed = voice_config.get("seed", 42)
            prompt = voice_config.get("prompt", "A natural voice")
            speed = voice_config.get("speed", 1.0)
            print(f"[VoiceDesign] Seed: {seed}", flush=True)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            result = None
            if hasattr(self._model, 'generate_voice_design'):
                result = self._model.generate_voice_design(text, prompt, language)
            elif hasattr(self._model, 'generate'):
                try:
                    result = self._model.generate(text=text, prompt=prompt, language=language)
                except TypeError:
                    result = self._model.generate(text, prompt, language)
            elif hasattr(self._model, 'synthesize'):
                result = self._model.synthesize(text, prompt, language)
            else:
                print(f"[VoiceDesign] ❌ Не знайдено метод генерації!", flush=True)
                return None, 0
            if isinstance(result, tuple):
                audio, sr = result
            else:
                audio = result
                sr = self._sample_rate
            if hasattr(audio, 'cpu'):
                audio = audio.cpu().numpy()
            elif hasattr(audio, 'numpy'):
                audio = audio.numpy()
            audio = np.array(audio).flatten()
            if speed != 1.0:
                audio = self._change_speed(audio, speed)
            print(f"[VoiceDesign] ✅ Згенеровано: {len(audio)} семплів", flush=True)
            return audio, sr
        except Exception as e:
            print(f"[VoiceDesign] ❌ Помилка генерації: {e}", flush=True)
            return None, 0
    
    def _change_speed(self, audio: np.ndarray, speed: float) -> np.ndarray:
        try:
            from pydub import AudioSegment
            audio_int16 = (audio * 32767).astype(np.int16)
            segment = AudioSegment(audio_int16.tobytes(), frame_rate=self._sample_rate, sample_width=2, channels=1)
            if speed > 1.0:
                segment = segment.speedup(playback_speed=speed)
            elif speed < 1.0:
                segment = segment.speedup(playback_speed=speed)
            samples = np.array(segment.get_array_of_samples())
            return samples.astype(np.float32) / 32767.0
        except Exception as e:
            print(f"[VoiceDesign] Помилка зміни швидкості: {e}", flush=True)
            return audio


# =============================================================================
# VOICE CLONE ENGINE
# =============================================================================

class VoiceCloneEngine(TTSEngine):
    """Voice Clone двигун - клонування голосу з reference audio."""
    
    LANGUAGE_MAP = {
        "russian": "ru", "ukrainian": "uk", "english": "en", "chinese": "zh",
        "japanese": "ja", "korean": "ko", "german": "de", "french": "fr",
        "spanish": "es", "italian": "it", "portuguese": "pt",
    }
    
    def __init__(self, model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base", cache_dir: Optional[str] = None):
        self._model = None
        self._model_name = model_name
        self._sample_rate = 24000
        self._cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".voicebox_cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._prompt_cache: Dict[str, Any] = {}
        self._fallback_engine: Optional['VoiceDesignEngine'] = None
    
    def _get_device(self) -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            return "cpu"
        except ImportError:
            return "cpu"
    
    def load_model(self, device: Optional[str] = None) -> bool:
        try:
            import torch
            from qwen_tts import Qwen3TTSModel
            if device is None:
                device = self._get_device()
            print(f"[VoiceClone] Завантаження моделі {self._model_name}...", flush=True)
            if device == "cuda":
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                device_map = "auto"
            else:
                dtype = torch.float32
                device_map = "cpu"
            self._model = Qwen3TTSModel.from_pretrained(self._model_name, device_map=device_map, torch_dtype=dtype)
            print(f"[VoiceClone] ✅ Модель завантажено!", flush=True)
            return True
        except Exception as e:
            print(f"[VoiceClone] ❌ Помилка завантаження: {e}", flush=True)
            return False
    
    def set_model(self, model) -> None:
        self._model = model
    
    def is_loaded(self) -> bool:
        return self._model is not None
    
    @property
    def engine_type(self) -> str:
        return "clone"
    
    def create_voice_prompt(self, reference_audio_path: str, reference_text: str, use_cache: bool = True, validate: bool = True) -> Tuple[Optional[Dict[str, Any]], bool]:
        if not self.is_loaded():
            return None, False
        if validate:
            is_valid, error_msg = validate_reference_audio(reference_audio_path)
            if not is_valid:
                return None, False
        cache_key = get_cache_key(reference_audio_path, reference_text)
        if use_cache:
            if cache_key in self._prompt_cache:
                return self._prompt_cache[cache_key], True
            cached_prompt = self._load_prompt_from_disk(cache_key)
            if cached_prompt is not None:
                self._prompt_cache[cache_key] = cached_prompt
                return cached_prompt, True
        try:
            voice_prompt = self._model.create_voice_clone_prompt(ref_audio=reference_audio_path, ref_text=reference_text, x_vector_only_mode=False)
            if use_cache:
                self._prompt_cache[cache_key] = voice_prompt
                self._save_prompt_to_disk(cache_key, voice_prompt)
            return voice_prompt, False
        except Exception as e:
            print(f"[VoiceClone] ❌ Помилка: {e}", flush=True)
            return None, False
    
    def generate(self, text: str, voice_config: Dict[str, Any], language: str = "russian") -> Tuple[Optional[np.ndarray], int]:
        if not self.is_loaded():
            return None, 0
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
                if not reference_audio:
                    return None, 0
                voice_prompt, was_cached = self.create_voice_prompt(reference_audio, reference_text, use_cache=use_cache)
                if voice_prompt is None:
                    return self._fallback_to_voicedesign(text, voice_config, language)
            lang_code = self.LANGUAGE_MAP.get(language.lower(), language.lower())
            result = self._generate_with_model(text=text, voice_prompt=voice_prompt, language=lang_code, instruct=instruct)
            if result is None:
                return None, 0
            audio, sr = result
            if hasattr(audio, 'cpu'):
                audio = audio.cpu().numpy()
            elif hasattr(audio, 'numpy'):
                audio = audio.numpy()
            audio = np.array(audio).flatten()
            if speed != 1.0:
                audio = self._change_speed(audio, speed)
            return audio, sr
        except Exception as e:
            print(f"[VoiceClone] ❌ Помилка: {e}", flush=True)
            return None, 0
    
    def _generate_with_model(self, text: str, voice_prompt: Dict[str, Any], language: str, instruct: Optional[str] = None) -> Optional[Tuple[np.ndarray, int]]:
        try:
            if hasattr(self._model, 'generate_voice_clone'):
                wavs, sample_rate = self._model.generate_voice_clone(text=text, voice_clone_prompt=voice_prompt, instruct=instruct)
                return wavs[0] if isinstance(wavs, list) else wavs, sample_rate
            elif hasattr(self._model, 'generate'):
                result = self._model.generate(text=text, voice_prompt=voice_prompt, language=language)
                if isinstance(result, tuple):
                    return result
                return result, self._sample_rate
            return None
        except Exception as e:
            print(f"[VoiceClone] ❌ Помилка: {e}", flush=True)
            return None
    
    def _fallback_to_voicedesign(self, text: str, voice_config: Dict[str, Any], language: str) -> Tuple[Optional[np.ndarray], int]:
        print("[VoiceClone] ⚠️ Fallback на VoiceDesign...", flush=True)
        try:
            if self._fallback_engine is None:
                self._fallback_engine = VoiceDesignEngine()
                if self.is_loaded():
                    self._fallback_engine.set_model(self._model)
            fallback_config = {"seed": voice_config.get("seed", 42), "prompt": voice_config.get("prompt", "A natural voice"), "speed": voice_config.get("speed", 1.0)}
            return self._fallback_engine.generate(text, fallback_config, language)
        except Exception as e:
            print(f"[VoiceClone] ❌ Fallback помилка: {e}", flush=True)
            return None, 0
    
    def _save_prompt_to_disk(self, cache_key: str, voice_prompt: Any) -> None:
        try:
            import torch
            cache_file = self._cache_dir / f"{cache_key}.prompt"
            torch.save(voice_prompt, cache_file)
        except Exception:
            pass
    
    def _load_prompt_from_disk(self, cache_key: str) -> Optional[Any]:
        try:
            import torch
            cache_file = self._cache_dir / f"{cache_key}.prompt"
            if cache_file.exists():
                return torch.load(cache_file)
        except Exception:
            pass
        return None
    
    def clear_cache(self) -> int:
        self._prompt_cache.clear()
        deleted = 0
        for cache_file in self._cache_dir.glob("*.prompt"):
            try:
                cache_file.unlink()
                deleted += 1
            except Exception:
                pass
        return deleted
    
    def _change_speed(self, audio: np.ndarray, speed: float) -> np.ndarray:
        try:
            from pydub import AudioSegment
            audio_int16 = (audio * 32767).astype(np.int16)
            segment = AudioSegment(audio_int16.tobytes(), frame_rate=self._sample_rate, sample_width=2, channels=1)
            if speed > 1.0:
                segment = segment.speedup(playback_speed=speed)
            elif speed < 1.0:
                segment = segment.speedup(playback_speed=speed)
            samples = np.array(segment.get_array_of_samples())
            return samples.astype(np.float32) / 32767.0
        except Exception:
            return audio


# =============================================================================
# TTS ENGINE MANAGER
# =============================================================================

class TTSEngineManager:
    """Менеджер для управління TTS двигунами."""
    
    def __init__(self, shared_model: bool = True):
        self._engines: Dict[str, TTSEngine] = {}
        self._default_engine_type = "voicedesign"
        self._shared_model = shared_model
        self._shared_model_instance = None
    
    def register_engine(self, engine: TTSEngine) -> None:
        self._engines[engine.engine_type] = engine
        print(f"[TTSEngineManager] Зареєстровано двигун: {engine.engine_type}", flush=True)
    
    def get_engine(self, engine_type: Optional[str] = None) -> TTSEngine:
        if engine_type is None:
            engine_type = self._default_engine_type
        if engine_type not in self._engines:
            if engine_type == "voicedesign":
                engine = VoiceDesignEngine()
            elif engine_type == "clone":
                engine = VoiceCloneEngine()
            else:
                raise ValueError(f"Невідомий тип двигуна: {engine_type}")
            if self._shared_model and self._shared_model_instance is not None:
                engine.set_model(self._shared_model_instance)
            self._engines[engine_type] = engine
        return self._engines[engine_type]
    
    def set_default_engine(self, engine_type: str) -> None:
        if engine_type not in ["voicedesign", "clone"]:
            raise ValueError(f"Невірний тип двигуна: {engine_type}")
        self._default_engine_type = engine_type
    
    def has_engine(self, engine_type: str) -> bool:
        return engine_type in self._engines
    
    def is_engine_loaded(self, engine_type: str) -> bool:
        if engine_type in self._engines:
            return self._engines[engine_type].is_loaded()
        return False
    
    def load_all_engines(self, device: Optional[str] = None) -> Dict[str, bool]:
        results = {}
        if device is None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        for engine_type, engine in self._engines.items():
            if hasattr(engine, 'load_model'):
                success = engine.load_model(device)
                results[engine_type] = success
                if self._shared_model and success and self._shared_model_instance is None:
                    self._shared_model_instance = engine._model
            else:
                results[engine_type] = engine.is_loaded()
        return results
    
    def load_engine(self, engine_type: str, device: Optional[str] = None) -> bool:
        engine = self.get_engine(engine_type)
        if self._shared_model and self._shared_model_instance is not None:
            engine.set_model(self._shared_model_instance)
            return True
        if device is None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        if hasattr(engine, 'load_model'):
            success = engine.load_model(device)
            if self._shared_model and success:
                self._shared_model_instance = engine._model
            return success
        return engine.is_loaded()
    
    def unload_all(self) -> None:
        for engine_type, engine in self._engines.items():
            if hasattr(engine, '_model') and engine._model is not None:
                del engine._model
                engine._model = None
        self._shared_model_instance = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        print("[TTSEngineManager] Всі двигуни вивантажено", flush=True)
    
    def get_voice_clone_engine(self) -> VoiceCloneEngine:
        return self.get_engine("clone")
    
    def get_voicedesign_engine(self) -> VoiceDesignEngine:
        return self.get_engine("voicedesign")


# =============================================================================
# КІНЕЦЬ ВБУДОВАНОГО КОДУ АДАПТЕРА
# =============================================================================

# Глобальні змінні для адаптера
tts_engine_manager = None
voice_profile_manager = None
voice_cache = None

def init_voicebox_adapter():
    """Ініціалізація Voicebox Adapter"""
    global tts_engine_manager, voice_profile_manager, voice_cache
    
    if not VOICEBOX_ADAPTER_AVAILABLE:
        return False
    
    try:
        # Ініціалізація менеджера двигунів
        tts_engine_manager = TTSEngineManager()
        
        # Реєстрація VoiceDesign двигуна з існуючою моделлю
        if tts_model_voicedesign is not None:
            engine = VoiceDesignEngine()
            engine.set_model(tts_model_voicedesign)
            tts_engine_manager.register_engine(engine)
            print("[ADAPTER] ✅ VoiceDesign двигун зареєстровано", flush=True)
        
        # Ініціалізація менеджера профілів (VOICE_PRESETS визначається пізніше)
        # Буде викликано повторно після визначення VOICE_PRESETS
        print("[ADAPTER] ⏳ Менеджер профілів буде ініціалізовано пізніше", flush=True)
        
        # Ініціалізація кешу
        cache_dir = "/content/drive/MyDrive/vibemodly_cache" if os.path.exists("/content/drive/MyDrive") else "/content/cache"
        voice_cache = VoicePromptCache(cache_dir)
        
        print("[ADAPTER] ✅ Ініціалізація завершена", flush=True)
        return True
        
    except Exception as e:
        print(f"[ADAPTER] ❌ Помилка ініціалізації: {e}", flush=True)
        return False

# Ініціалізуємо адаптер після завантаження TTS моделі
adapter_initialized = False
if VOICEBOX_ADAPTER_AVAILABLE:
    adapter_initialized = init_voicebox_adapter()
    if adapter_initialized:
        print("[ADAPTER] ✅ Voicebox Adapter активовано", flush=True)
    else:
        print("[ADAPTER] ⚠️ Voicebox Adapter не ініціалізовано, використовуємо стандартний підхід", flush=True)

def get_character_seed(character_name):
    """Генерація стабільного seed на основі імені персонажа"""
    hash_obj = hashlib.md5(character_name.encode('utf-8'))
    return int(hash_obj.hexdigest(), 16) % (2**32)

def generate_audio(text, voice_desc="", lang="russian", character_name="narrator", gender="male", voice_preset=None, speed=1.0, custom_seed=None, use_voice_clone=True):
    """Генерація аудіо через Qwen3-TTS з підтримкою клонування голосу.
    
    АЛГОРИТМ КОНСИСТЕНТНОСТІ ГОЛОСУ:
    1. Перший виклик для персонажа: VoiceDesign (seed + prompt) → зберігаємо аудіо як reference
    2. Наступні виклики: VoiceClone з reference аудіо (якщо доступна модель Base)
    
    Це забезпечує справжню консистентність голосу між різними текстами.
    
    Args:
        text: Текст для озвучки
        voice_desc: Опис голосу
        lang: Мова (russian, ukrainian)
        character_name: Ім'я персонажа (для seed)
        gender: Стать (male, female)
        voice_preset: Ключ пресету голосу
        speed: Швидкість мовлення
        custom_seed: Власний seed (пріоритет над іншими)
        use_voice_clone: Використовувати клонування для консистентності (за замовчуванням True)
    """
    global tts_model_voicedesign, tts_model_base, character_voice_params, character_voice_prompts
    
    print(f"[TTS] === ПОЧАТОК ГЕНЕРАЦІЇ ===", flush=True)
    print(f"[TTS] Персонаж: {character_name}", flush=True)
    print(f"[TTS] Пресет: {voice_preset}", flush=True)
    print(f"[TTS] Стать: {gender}", flush=True)
    print(f"[TTS] Текст: {text[:50]}...", flush=True)
    
    if tts_model_voicedesign is None:
        print("[TTS] ❌ Модель VoiceDesign не завантажена!", flush=True)
        return None
    
    try:
        # Застосування словника наголосів
        text = apply_stress_dictionary(text)
        
        # Перевіряємо, чи є збережені параметри для цього персонажа
        saved_params = character_voice_params.get(character_name, {})
        
        # Визначення seed (пріоритет: custom_seed -> saved -> preset seed -> character seed)
        if custom_seed is not None:
            seed = custom_seed
            print(f"[TTS] Seed (custom): {seed}", flush=True)
        elif saved_params.get("seed") is not None:
            seed = saved_params["seed"]
            print(f"[TTS] Seed (saved): {seed}", flush=True)
        elif voice_preset and voice_preset in VOICE_PRESETS:
            preset = VOICE_PRESETS[voice_preset]
            seed = preset.get("seed", get_character_seed(character_name))
            print(f"[TTS] Seed (preset): {seed}", flush=True)
        else:
            seed = get_character_seed(character_name)
            print(f"[TTS] Seed (character): {seed}", flush=True)
        
        # Визначення prompt (пріоритет: saved -> preset -> build_voice_prompt)
        if saved_params.get("prompt"):
            prompt = saved_params["prompt"]
            print(f"[TTS] Prompt (saved): {prompt[:60]}...", flush=True)
        elif voice_preset and voice_preset in VOICE_PRESETS:
            preset = VOICE_PRESETS[voice_preset]
            gender = preset.get("gender", gender)
            speed = preset.get("speed", speed)
            prompt_suffix = preset.get("prompt_suffix", "")
            pitch = preset.get("pitch", "medium")
            
            # Формування повного prompt
            gender_prefix = "Male voice" if gender == "male" else "Female voice"
            pitch_desc = {"low": "low-pitched", "medium": "medium-pitched", "high": "high-pitched"}
            prompt = f"{gender_prefix}, {pitch_desc.get(pitch, 'medium')}. {prompt_suffix}"
            
            print(f"[TTS] ✅ Пресет: {voice_preset} ({preset['name']})", flush=True)
            print(f"[TTS] Prompt: {prompt[:80]}...", flush=True)
        else:
            # Використання функції build_voice_prompt
            prompt = build_voice_prompt(character_name, voice_preset, gender)
            if voice_desc:
                prompt = f"{prompt}. {voice_desc}"
            print(f"[TTS] ⚠️ Пресет не знайдено: {voice_preset}", flush=True)
            print(f"[TTS] Prompt: {prompt[:80]}...", flush=True)
        
        # === ГОЛОВНА ЛОГІКА - VOICE CLONE АБО VOICE DESIGN ===
        
        # Перевіряємо, чи можемо використати клонування
        can_use_clone = (
            use_voice_clone and 
            tts_model_base is not None and 
            character_name in character_voice_prompts and
            character_voice_prompts[character_name].get("voice_prompt") is not None
        )
        
        if can_use_clone:
            # === VOICE CLONE - використовуємо збережений voice_prompt ===
            print(f"[TTS] === VOICE CLONE ===", flush=True)
            print(f"[TTS] Персонаж: {character_name}", flush=True)
            print(f"[TTS] Використовуємо збережений voice_prompt", flush=True)
            print(f"[TTS] Текст: {text[:50]}...", flush=True)
            
            voice_prompt = character_voice_prompts[character_name]["voice_prompt"]
            
            # Мапінг мов для Qwen3-TTS
            lang_map = {
                "russian": "ru", "ukrainian": "uk", "english": "en",
                "chinese": "zh", "japanese": "ja", "korean": "ko"
            }
            lang_code = lang_map.get(lang.lower(), lang.lower())
            
            # Генерація через VoiceClone
            if hasattr(tts_model_base, 'generate_voice_clone'):
                result = tts_model_base.generate_voice_clone(
                    text=text,
                    voice_clone_prompt=voice_prompt,
                    instruct=None
                )
                # generate_voice_clone повертає список wav файлів
                if isinstance(result, tuple):
                    wavs, sr = result
                    audio = wavs[0] if isinstance(wavs, list) else wavs
                else:
                    audio = result
                    sr = SAMPLE_RATE
            elif hasattr(tts_model_base, 'generate'):
                result = tts_model_base.generate(
                    text=text,
                    voice_prompt=voice_prompt,
                    language=lang_code
                )
                if isinstance(result, tuple):
                    audio, sr = result
                else:
                    audio = result
                    sr = SAMPLE_RATE
            else:
                print(f"[TTS] ⚠️ Метод клонування не знайдено, fallback на VoiceDesign", flush=True)
                can_use_clone = False
        
        if not can_use_clone:
            # === VOICE DESIGN - створюємо новий голос ===
            print(f"[TTS] === VOICE DESIGN ===", flush=True)
            print(f"[TTS] Персонаж: {character_name}", flush=True)
            print(f"[TTS] Seed: {seed}", flush=True)
            print(f"[TTS] Gender: {gender}, Speed: {speed}", flush=True)
            print(f"[TTS] Prompt: {prompt[:80]}...", flush=True)
            print(f"[TTS] Текст: {text[:50]}...", flush=True)
            
            # Встановлення seed для відтворюваності
            import random
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            os.environ['PYTHONHASHSEED'] = str(seed)
            
            # Генерація через VoiceDesign
            if hasattr(tts_model_voicedesign, 'generate_voice_design'):
                print(f"[TTS] Використовуємо generate_voice_design()", flush=True)
                result = tts_model_voicedesign.generate_voice_design(text, prompt, lang)
            elif hasattr(tts_model_voicedesign, 'generate'):
                print(f"[TTS] Використовуємо generate()", flush=True)
                try:
                    result = tts_model_voicedesign.generate(text=text, prompt=prompt, language=lang)
                except TypeError:
                    try:
                        result = tts_model_voicedesign.generate(text, prompt, lang)
                    except:
                        result = tts_model_voicedesign.generate(text)
            elif hasattr(tts_model_voicedesign, 'synthesize'):
                print(f"[TTS] Використовуємо synthesize()", flush=True)
                result = tts_model_voicedesign.synthesize(text, prompt, lang)
            else:
                print(f"[TTS] ❌ Не знайдено метод генерації!", flush=True)
                print(f"[TTS] Доступні методи: {dir(tts_model_voicedesign)}", flush=True)
                return None
            
            # Обробка результату
            if isinstance(result, tuple):
                audio, sr = result
            else:
                audio = result
                sr = SAMPLE_RATE
        
        # Конвертація в numpy
        if hasattr(audio, 'cpu'):
            audio = audio.cpu().numpy()
        elif hasattr(audio, 'numpy'):
            audio = audio.numpy()
        
        # Flatten
        audio = np.array(audio).flatten()
        
        # Перевірка валідності аудіо
        if len(audio) < SAMPLE_RATE * 0.5:  # менше 0.5 секунди
            print(f"[TTS] ⚠️ Аудіо занадто коротке: {len(audio)} семплів", flush=True)
            return None
        
        # Зміна швидкості якщо потрібно
        if speed != 1.0:
            audio = change_speed(audio, speed)
        
        print(f"[TTS] ✅ Згенеровано: {len(audio)} семплів ({len(audio)/SAMPLE_RATE:.1f} сек)", flush=True)
        
        # === ЗБЕРЕЖЕННЯ ПАРАМЕТРІВ ГОЛОСУ ===
        # Зберігаємо seed + prompt для цього персонажа
        if character_name not in character_voice_params:
            character_voice_params[character_name] = {
                "seed": seed,
                "prompt": prompt
            }
            print(f"[VOICE] Збережено параметри для '{character_name}': seed={seed}", flush=True)
        
        # === СТВОРЕННЯ VOICE_PROMPT ДЛЯ КЛОНУВАННЯ ===
        # Якщо це перший виклик і модель Base доступна - створюємо voice_prompt
        if (use_voice_clone and 
            tts_model_base is not None and 
            character_name not in character_voice_prompts and
            len(audio) >= SAMPLE_RATE * 2):  # Мінімум 2 секунди для reference
            
            print(f"[VOICE] Створення voice_prompt для '{character_name}'...", flush=True)
            
            try:
                # Зберігаємо reference аудіо у тимчасовий файл
                import tempfile
                import soundfile as sf
                
                # Нормалізація аудіо
                max_val = np.max(np.abs(audio))
                if max_val > 0:
                    ref_audio = audio / max_val * 0.9
                else:
                    ref_audio = audio
                
                # Збереження у тимчасовий файл
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    tmp_path = tmp.name
                    sf.write(tmp_path, ref_audio, SAMPLE_RATE)
                
                # Створення voice_prompt через модель Base
                if hasattr(tts_model_base, 'create_voice_clone_prompt'):
                    voice_prompt = tts_model_base.create_voice_clone_prompt(
                        ref_audio=tmp_path,
                        ref_text=text[:200],  # Перші 200 символів як reference текст
                        x_vector_only_mode=False
                    )
                    
                    # Зберігаємо voice_prompt для наступних викликів
                    character_voice_prompts[character_name] = {
                        "voice_prompt": voice_prompt,
                        "reference_text": text[:200],
                        "seed": seed,
                        "prompt": prompt
                    }
                    print(f"[VOICE] ✅ voice_prompt створено для '{character_name}'", flush=True)
                
                # Видалення тимчасового файлу
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                    
            except Exception as e:
                print(f"[VOICE] ⚠️ Не вдалося створити voice_prompt: {e}", flush=True)
        
        return audio
        
    except Exception as e:
        print(f"[TTS] ❌ Помилка: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return None

def change_speed(audio, speed):
    """Зміна швидкості аудіо"""
    try:
        # Конвертація в AudioSegment
        audio_int16 = (audio * 32767).astype(np.int16)
        segment = AudioSegment(
            audio_int16.tobytes(),
            frame_rate=SAMPLE_RATE,
            sample_width=2,
            channels=1
        )
        
        # Зміна швидкості
        if speed > 1.0:
            segment = segment.speedup(playback_speed=speed)
        elif speed < 1.0:
            segment = segment.speedup(playback_speed=speed)  # pydub використовує speedup для обох напрямків
        
        # Конвертація назад
        samples = np.array(segment.get_array_of_samples())
        return samples.astype(np.float32) / 32767.0
    except Exception as e:
        print(f"[TTS] Помилка зміни швидкості: {e}", flush=True)
        return audio

# =============================================================================
# СИСТЕМА ПРЕСЕТІВ ГОЛОСІВ
# =============================================================================

# Базові пресети голосів з фіксованим seed
VOICE_PRESETS = {
    # Чоловічі голоси
    "male_deep": {
        "name": "Глибокий чоловічий",
        "gender": "male",
        "description": "Глибокий, низький, спокійний чоловічий голос",
        "pitch": "low",
        "speed": 1.0,
        "seed": 1001,  # Фіксований seed для консистентності
        "prompt_suffix": "Deep, low-pitched, calm male voice. Mature and authoritative."
    },
    "male_young": {
        "name": "Молодий чоловічий",
        "gender": "male",
        "description": "Енергійний, середній чоловічий голос",
        "pitch": "medium",
        "speed": 1.1,
        "seed": 1002,
        "prompt_suffix": "Young, energetic male voice. Clear and bright."
    },
    "male_elderly": {
        "name": "Літній чоловічий",
        "gender": "male",
        "description": "Повільний, мудрий чоловічий голос",
        "pitch": "low",
        "speed": 0.85,
        "seed": 1003,
        "prompt_suffix": "Elderly, wise male voice. Slow and thoughtful."
    },
    "male_rough": {
        "name": "Грубий чоловічий",
        "gender": "male",
        "description": "Грубий, хрипкий чоловічий голос",
        "pitch": "low",
        "speed": 0.95,
        "seed": 1004,
        "prompt_suffix": "Rough, raspy male voice. Gritty and weathered."
    },
    
    # Жіночі голоси
    "female_soft": {
        "name": "М'який жіночий",
        "gender": "female",
        "description": "М'який, ніжний жіночий голос",
        "pitch": "medium",
        "speed": 1.0,
        "seed": 2001,
        "prompt_suffix": "Soft, gentle female voice. Warm and caring."
    },
    "female_strong": {
        "name": "Сильний жіночий",
        "gender": "female",
        "description": "Сильний, впевнений жіночий голос",
        "pitch": "medium",
        "speed": 1.05,
        "seed": 2002,
        "prompt_suffix": "Strong, confident female voice. Clear and assertive."
    },
    "female_young": {
        "name": "Молода жінка",
        "gender": "female",
        "description": "Молодий, дзвінкий жіночий голос",
        "pitch": "high",
        "speed": 1.1,
        "seed": 2003,
        "prompt_suffix": "Young, bright female voice. Energetic and cheerful."
    },
    "female_elderly": {
        "name": "Літня жінка",
        "gender": "female",
        "description": "Повільний, теплий жіночий голос",
        "pitch": "medium",
        "speed": 0.9,
        "seed": 2004,
        "prompt_suffix": "Elderly female voice. Warm and grandmotherly."
    },
    
    # Диктор (закадровий голос)
    "narrator_neutral": {
        "name": "Диктор нейтральний",
        "gender": "male",
        "description": "Нейтральний голос диктора",
        "pitch": "medium",
        "speed": 0.95,
        "seed": 3001,
        "prompt_suffix": "Professional narrator voice. Clear and engaging storytelling."
    },
    "narrator_dramatic": {
        "name": "Диктор драматичний",
        "gender": "male",
        "description": "Драматичний голос диктора",
        "pitch": "low",
        "speed": 0.85,
        "seed": 3002,
        "prompt_suffix": "Dramatic narrator voice. Deep and atmospheric storytelling."
    },
    "narrator_female": {
        "name": "Диктор жіночий",
        "gender": "female",
        "description": "Жіночий голос диктора",
        "pitch": "medium",
        "speed": 0.95,
        "seed": 3003,
        "prompt_suffix": "Professional female narrator voice. Clear and warm storytelling."
    }
}

# =============================================================================
# VOICEBOX ADAPTER - ІНІЦІАЛІЗАЦІЯ МЕНЕДЖЕРА ПРОФІЛІВ
# =============================================================================

# Ініціалізуємо менеджер профілів після визначення VOICE_PRESETS
if VOICEBOX_ADAPTER_AVAILABLE and voice_profile_manager is None:
    try:
        voice_profile_manager = VoiceProfileManager(VOICE_PRESETS)
        print("[ADAPTER] ✅ Менеджер профілів ініціалізовано", flush=True)
    except Exception as e:
        print(f"[ADAPTER] ❌ Помилка ініціалізації менеджера профілів: {e}", flush=True)

def build_voice_prompt(character_name, voice_preset=None, gender=None):
    """
    Будує текстовий опис голосу для VoiceDesign моделі.
    Qwen3-TTS використовує текстовий prompt для створення голосу.
    
    Args:
        character_name: Ім'я персонажа
        voice_preset: Ключ пресету голосу
        gender: Стать (male, female, neutral)
    
    Returns:
        str: Текстовий опис голосу для генерації
    """
    # Базовий prompt з пресета
    if voice_preset and voice_preset in VOICE_PRESETS:
        preset = VOICE_PRESETS[voice_preset]
        gender_prefix = "Male voice" if preset.get("gender") == "male" else "Female voice"
        pitch = preset.get("pitch", "medium")
        pitch_desc = {"low": "low-pitched", "medium": "medium-pitched", "high": "high-pitched"}
        prompt_suffix = preset.get("prompt_suffix", "")
        return f"{gender_prefix}, {pitch_desc.get(pitch, 'medium')}. {prompt_suffix}"
    
    # Prompt на основі персонажа
    character_prompts = {
        "narrator": "A professional narrator voice, clear and articulate, neutral tone, suitable for audiobooks",
        "narrator_test": "A professional narrator voice, clear and articulate, neutral tone, suitable for audiobooks",
    }
    
    if character_name in character_prompts:
        return character_prompts[character_name]
    
    # За замовчуванням - на основі статі
    gender_prompt = {
        "male": "A male voice, natural and expressive",
        "female": "A female voice, natural and expressive",
        "neutral": "A neutral voice, clear and natural"
    }
    return gender_prompt.get(gender, "A natural voice")

# =============================================================================
# СЛОВНИК НАГОЛОСІВ
# =============================================================================

# Словник для виправлення наголосів та вимови
# Формат: "слово": "правильна_вимова_з_наголосом"
STRESS_DICTIONARY = {
    # Російські слова
    "звонит": "звонИт",
    "звонят": "звонЯт",
    "понял": "пОнял",
    "поняла": "понялА",
    "поняли": "пОняли",
    "включит": "включИт",
    "включат": "включАт",
    "красивее": "красИвее",
    "свекла": "свёкла",
    "щавель": "щавЕль",
    "торты": "тОрты",
    "шофёр": "шофёр",
    "договор": "договОр",
    "каталог": "каталОг",
    "квартал": "квартАл",
    "новости": "нОвости",
    "средства": "срЕдства",
    "статуя": "стАтуя",
    "столяр": "столЯр",
    "цемент": "цемЕнт",
    "шерстя": "шЕрстя",
    "щемит": "щемИт",
    "язык": "язЫк",
    "прибыл": "прИбыл",
    "прибыли": "прИбыли",
    "прибыло": "прИбыло",
    "жалюзи": "жалюзИ",
    "каучук": "каучУк",
    "коклюш": "коклЮш",
    "корысть": "кОрысть",
    "кремень": "крЕмень",
    "лыжня": "лыжнЯ",
    "мальчиковый": "мальчикОвый",
    "менеджмент": "менЕджмент",
    "металлургия": "металлУргия",
    "мизерный": "мИзерный",
    "молитва": "молИтва",
    "намерение": "намЕрение",
    "насморк": "нАсморк",
    "начать": "начАть",
    "начала": "началА",
    "начали": "нАчали",
    "облегчить": "облегчИть",
    "обеспечение": "обеспЕчение",
    "одновременно": "одноврЕменно",
    "отрочество": "Отрочество",
    "партер": "партЕр",
    "пицца": "пИцца",
    "премирование": "премировАние",
    "прибыть": "прИбыть",
    "принудить": "принудИть",
    "пурпур": "пУрпур",
    "рапорт": "рапОрт",
    "рассредоточение": "рассредоточЕние",
    "ремень": "ремЕнь",
    "созыв": "сОзыв",
    "созыва": "сОзыва",
    "среда": "средА",
    "танцовщица": "танцОвщица",
    "творог": "твОрог",
    "тезис": "тЕзис",
    "тенденция": "тендЕнция",
    "толика": "тОлика",
    "тонус": "тОнуc",
    "удобнее": "удОбнее",
    "украинский": "укрАинский",
    "умерший": "Умерший",
    "упрочение": "упрОчение",
    "феномен": "фенОмен",
    "ходатайство": "ходАтайство",
    "христианин": "христИанин",
    "цыган": "цыгАн",
    "черпать": "чЕрпать",
    "шасси": "шассИ",
    "эксперт": "экспЕрт",
    
    # Українські слова
    "україна": "укрАїна",
    "український": "укрАїнський",
    "киянин": "кИянин",
    "кияни": "кИяни",
    "одинадцять": "одинАдцять",
    "чотирнадцять": "чотирнАдцять",
    "п'ятнадцять": "п'ятнАдцять",
    "шістнадцять": "шістнАдцять",
    "сімнадцять": "сімнАдцять",
    "вісімнадцять": "вісімнАдцять",
    "дев'ятнадцять": "дев'ятнАдцять",
    "двадцять": "двАдцять",
    "тридцять": "трИдцять",
    "сорок": "сОрок",
    "дев'яносто": "дев'янОсто",
    "сімдесят": "сІмдесят",
    "вісімдесят": "вІсімдесят",
    "завдання": "завдАння",
    "випадок": "вИпадок",
    "документ": "докУмент",
    "магазин": "магаЗин",
    "програма": "прогрАма",
    "програміст": "програмИст",
    "читання": "читАння",
    "писання": "писАння",
    "питання": "питАння",
    "відповідь": "відпОвідь",
    "діяльність": "дІяльність",
    "можливість": "мОжливість",
    "необхідність": "необхІдність",
    "відповідальність": "відповідальнІсть",
    "приватний": "привАтний",
    "публічний": "публІчний",
    "загальний": "загАльний",
    "особливий": "осОбливий",
    "звичайний": "звичАйний",
    "незвичайний": "незвичАйний",
    "прекрасний": "прекрАсний",
    "важливий": "вАжливий",
    "цікавий": "цІкавий",
    "корисний": "корИсний",
    "необхідний": "необхІдний",
    "головний": "голОвний",
    "останній": "остАнній",
    "перший": "пЕрший",
    "другий": "дрУгий",
    "третій": "трЕтій",
    "четвертий": "четвЕртий",
    "п'ятий": "п'Ятий",
    "шостий": "шОстий",
    "сьомий": "сьОмий",
    "восьмий": "восьмИй",
    "дев'ятий": "дев'Ятий",
    "десятий": "десЯтий",
}

def apply_stress_dictionary(text):
    """Застосування словника наголосів до тексту"""
    words = text.split()
    result = []
    
    for word in words:
        # Видаляємо пунктуацію для пошуку
        clean_word = word.lower().strip('.,!?;:"\'-()[]{}')
        
        if clean_word in STRESS_DICTIONARY:
            # Замінюємо слово з правильним наголосом
            stressed = STRESS_DICTIONARY[clean_word]
            # Зберігаємо регістр першої літери
            if word[0].isupper():
                stressed = stressed[0].upper() + stressed[1:]
            # Додаємо пунктуацію назад
            for punct in '.,!?;:"\'-()[]{}':
                if word.endswith(punct):
                    stressed += punct
            result.append(stressed)
        else:
            result.append(word)
    
    return ' '.join(result)

# =============================================================================
# ВИЗНАЧЕННЯ МОВИ ТЕКСТУ
# =============================================================================

def detect_language(text):
    """Визначення мови тексту (russian, ukrainian)"""
    # Українські специфічні літери
    ukrainian_chars = set('іїєґІЇЄҐ')
    # Російські специфічні літери  
    russian_chars = set('ыэъёЫЭЪЁ')
    
    ukr_count = sum(1 for c in text if c in ukrainian_chars)
    rus_count = sum(1 for c in text if c in russian_chars)
    
    # Також перевіряємо типові слова
    ukrainian_words = ['і', 'та', 'але', 'це', 'що', 'як', 'на', 'з', 'до', 'не', 'за', 'від']
    russian_words = ['и', 'но', 'это', 'что', 'как', 'на', 'с', 'до', 'не', 'за', 'от', 'в']
    
    text_lower = text.lower()
    words = set(text_lower.split())
    
    ukr_word_matches = len(words & set(ukrainian_words))
    rus_word_matches = len(words & set(russian_words))
    
    # Підсумок
    ukr_score = ukr_count + ukr_word_matches * 2
    rus_score = rus_count + rus_word_matches * 2
    
    if ukr_score > rus_score:
        return "ukrainian"
    else:
        return "russian"

# Кеш тестових голосів
test_voice_cache = {}

# Параметри голосу для кожного персонажа (seed + prompt для консистентності)
# Замість reference_audio використовуємо seed + prompt
character_voice_params = {}  # {character_name: {"seed": int, "prompt": str}}

def get_demo_text(lang="russian", long=False):
    """Отримання демо-тексту для тестування голосу
    
    Args:
        lang: Мова (russian, ukrainian)
        long: Якщо True, повертає довгий текст для кращого захоплення характеристик голосу
    """
    if lang == "ukrainian":
        if long:
            return """Привіт! Це тестовий фрагмент голосу для визначення його характеристик. 
            Я можу озвучувати ваші історії з різними емоціями та інтонаціями. 
            Цей довгий текст допоможе краще захопити унікальні особливості голосу.
            Слухайте уважно, адже кожен голос має свій неповторний тембр та ритм."""
        else:
            return "Привіт! Це тестовий фрагмент голосу. Я можу озвучувати ваші історії."
    else:
        if long:
            return """Привет! Это тестовый фрагмент голоса для определения его характеристик. 
            Я могу озвучивать ваши истории с разными эмоциями и интонациями. 
            Этот длинный текст поможет лучше захватить уникальные особенности голоса.
            Слушайте внимательно, ведь каждый голос имеет свой неповторимый тембр и ритм."""
        else:
            return "Привет! Это тестовый фрагмент голоса. Я могу озвучивать ваши истории."

# Шлях до збереження голосів на Google Drive
VOICE_CONFIG_PATH = "/content/drive/MyDrive/vibemodly_voices.json"

def mount_google_drive():
    """Монтування Google Drive"""
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("[DRIVE] Google Drive змонтовано", flush=True)
        return True
    except Exception as e:
        print(f"[DRIVE] Помилка монтування: {e}", flush=True)
        return False

def load_voice_configs():
    """Завантаження збережених голосів з Google Drive"""
    global VOICE_PRESETS
    
    try:
        if os.path.exists(VOICE_CONFIG_PATH):
            with open(VOICE_CONFIG_PATH, "r", encoding="utf-8") as f:
                saved_voices = json.load(f)
                VOICE_PRESETS.update(saved_voices)
                print(f"[VOICES] Завантажено {len(saved_voices)} кастомних голосів", flush=True)
    except Exception as e:
        print(f"[VOICES] Помилка завантаження: {e}", flush=True)

def save_voice_configs():
    """Збереження голосів на Google Drive"""
    try:
        os.makedirs(os.path.dirname(VOICE_CONFIG_PATH), exist_ok=True)
        with open(VOICE_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(VOICE_PRESETS, f, ensure_ascii=False, indent=2)
        print(f"[VOICES] Збережено {len(VOICE_PRESETS)} голосів", flush=True)
        return True
    except Exception as e:
        print(f"[VOICES] Помилка збереження: {e}", flush=True)
        return False

def get_voice_preset_list(gender=None, exclude_narrator=True):
    """Отримання списку доступних пресетів
    
    Args:
        gender: Фільтр за статтю (male, female, None - всі)
        exclude_narrator: Чи виключати narrator_ голоси зі списку
    
    Returns:
        list: Список словників з ключами key, name, description, gender
    """
    presets = []
    for key, preset in VOICE_PRESETS.items():
        # Пропускаємо narrator_ голоси якщо exclude_narrator=True
        if exclude_narrator and key.startswith("narrator_"):
            continue
        
        # Фільтруємо за статтю
        if gender is None or preset.get("gender") == gender:
            presets.append({
                "key": key,
                "name": preset["name"],
                "description": preset["description"],
                "gender": preset["gender"]
            })
    
    print(f"[PRESETS] Знайдено {len(presets)} голосів для gender={gender}, exclude_narrator={exclude_narrator}", flush=True)
    return presets

def create_custom_voice(name, gender, description, pitch="medium", speed=1.0, prompt_suffix=""):
    """Створення кастомного голосу"""
    key = f"custom_{name.lower().replace(' ', '_')}"
    VOICE_PRESETS[key] = {
        "name": name,
        "gender": gender,
        "description": description,
        "pitch": pitch,
        "speed": speed,
        "prompt_suffix": prompt_suffix
    }
    save_voice_configs()
    return key

# =============================================================================
# PEXELS API - ФОНОВІ ЗВУКИ
# =============================================================================

PEXELS_API_KEY = None
try:
    PEXELS_API_KEY = userdata.get("PEXELS_API_KEY")
    if PEXELS_API_KEY:
        print("[PEXELS] API ключ знайдено", flush=True)
except:
    print("[PEXELS] API ключ не знайдено - фонові шумі вимкнено", flush=True)

# Кеш завантажених звуків
ambient_cache = {}

def get_ambient_sound(sound_description, duration_sec=10):
    """Завантаження фонового звуку з Pexels"""
    global PEXELS_API_KEY, ambient_cache
    
    if not PEXELS_API_KEY:
        return None
    
    # Перевірка кешу
    cache_key = f"{sound_description}_{duration_sec}"
    if cache_key in ambient_cache:
        return ambient_cache[cache_key]
    
    try:
        # Пошук відео з аудіо на Pexels
        search_url = "https://api.pexels.com/videos/search"
        headers = {"Authorization": PEXELS_API_KEY}
        
        # Мапінг описів на ключові слова для пошуку
        keyword_map = {
            "гудіння": "humming machine",
            "систем": "machine ambient",
            "кроки": "footsteps walking",
            "тремтіння": "rumble vibration",
            "сигналізація": "alarm siren",
            "вітер": "wind ambient",
            "дихання": "breathing",
            "тиша": "quiet room",
            "космос": "space ambient",
            "корабель": "ship engine"
        }
        
        # Визначення ключових слів
        search_query = "ambient sound"
        for ua_key, en_value in keyword_map.items():
            if ua_key.lower() in sound_description.lower():
                search_query = en_value
                break
        
        print(f"[PEXELS] Пошук: {search_query}", flush=True)
        
        params = {
            "query": search_query,
            "per_page": 5,
            "orientation": "landscape"
        }
        
        response = requests.get(search_url, headers=headers, params=params)
        
        if response.status_code != 200:
            print(f"[PEXELS] Помилка API: {response.status_code}", flush=True)
            return None
        
        data = response.json()
        videos = data.get("videos", [])
        
        if not videos:
            print("[PEXELS] Відео не знайдено", flush=True)
            return None
        
        # Пошук відео з аудіо
        for video in videos:
            video_files = video.get("video_files", [])
            for vf in video_files:
                if vf.get("file_type") == "video/mp4" and vf.get("quality") in ["sd", "hd"]:
                    video_url = vf.get("link")
                    if video_url:
                        # Завантаження відео та витягування аудіо
                        print(f"[PEXELS] Завантаження: {video_url[:50]}...", flush=True)
                        
                        video_response = requests.get(video_url, stream=True)
                        if video_response.status_code == 200:
                            # Збереження тимчасового файлу
                            temp_video = f"/content/temp_ambient_{hash(sound_description)}.mp4"
                            with open(temp_video, "wb") as f:
                                for chunk in video_response.iter_content(chunk_size=8192):
                                    f.write(chunk)
                            
                            # Витягування аудіо через pydub
                            try:
                                audio_segment = AudioSegment.from_file(temp_video)
                                # Нормалізація та зациклення до потрібної тривалості
                                target_duration = duration_sec * 1000  # ms
                                if len(audio_segment) < target_duration:
                                    # Зациклення
                                    loops = int(target_duration / len(audio_segment)) + 1
                                    audio_segment = audio_segment * loops
                                audio_segment = audio_segment[:target_duration]
                                
                                # Зменшення гучності для фону
                                audio_segment = audio_segment - 15  # dB
                                
                                # Конвертація в numpy
                                samples = np.array(audio_segment.get_array_of_samples())
                                
                                # Видалення тимчасового файлу
                                os.remove(temp_video)
                                
                                # Збереження в кеш
                                ambient_cache[cache_key] = samples
                                print(f"[PEXELS] Аудіо готово: {len(samples)} семплів", flush=True)
                                return samples
                                
                            except Exception as e:
                                print(f"[PEXELS] Помилка обробки: {e}", flush=True)
                                if os.path.exists(temp_video):
                                    os.remove(temp_video)
        
        print("[PEXELS] Підходящого відео не знайдено", flush=True)
        return None
        
    except Exception as e:
        print(f"[PEXELS] Помилка: {e}", flush=True)
        return None

def generate_silence(duration_sec):
    """Генерація тиші"""
    return np.zeros(int(SAMPLE_RATE * duration_sec))

# =============================================================================
# ДОПОМІЖНІ ФУНКЦІЇ
# =============================================================================

import telebot
from telebot import types as tb_types

# Імпорти для файлів
try:
    import ebooklib
    from ebooklib import epub
    EPUB_OK = True
except:
    EPUB_OK = False

try:
    import PyPDF2
    from PyPDF2 import PdfReader
    PDF_OK = True
except:
    PDF_OK = False

try:
    import pdfplumber
    PDFPLUMBER_OK = True
except:
    PDFPLUMBER_OK = False

try:
    from bs4 import BeautifulSoup
    BS4_OK = True
except:
    BS4_OK = False

try:
    from docx import Document
    DOCX_OK = True
except:
    DOCX_OK = False

try:
    from lxml import etree
    LXML_OK = True
except:
    LXML_OK = False

def read_file(path, ext):
    """Читання файлу"""
    ext = ext.lower().strip(".")
    
    try:
        if ext == "txt":
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        
        elif ext == "epub" and EPUB_OK:
            book = epub.read_epub(path)
            parts = []
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    soup = BeautifulSoup(item.get_content(), "html.parser")
                    parts.append(soup.get_text())
            return "\n".join(parts)
        
        elif ext == "pdf":
            parts = []
            if PDFPLUMBER_OK:
                with pdfplumber.open(path) as pdf:
                    for page in pdf.pages:
                        t = page.extract_text()
                        if t:
                            parts.append(t)
            elif PDF_OK:
                reader = PdfReader(path)
                for page in reader.pages:
                    t = page.extract_text()
                    if t:
                        parts.append(t)
            return "\n".join(parts)
        
        elif ext == "html" and BS4_OK:
            with open(path, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f.read(), "html.parser")
                return soup.get_text()
        
        elif ext == "docx" and DOCX_OK:
            doc = Document(path)
            return "\n".join([p.text for p in doc.paragraphs if p.text])
        
        elif ext == "fb2" and LXML_OK:
            with open(path, "rb") as f:
                tree = etree.parse(f)
            ns = {"fb": "http://www.gribuser.ru/xml/fictionbook/2.0"}
            paragraphs = tree.xpath("//fb:p", namespaces=ns)
            return "\n".join([p.text_content() for p in paragraphs if p.text_content()])
        
        else:
            return None
    except Exception as e:
        print(f"[FILE] Помилка: {e}", flush=True)
        return None

def extract_json(text):
    """Витягування JSON"""
    try:
        start = text.find("{")
        if start == -1:
            return None
        
        depth = 0
        end = start
        for i, char in enumerate(text[start:], start):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            if depth == 0:
                end = i + 1
                break
        
        if end <= start:
            return None
        
        return json.loads(text[start:end])
    except:
        return None

def analyze_text(text):
    """Аналіз через Gemini з retry логікою"""
    global client, WORKING_MODEL
    print("[GEMINI] Аналіз...", flush=True)
    
    prompt = f"""Ти - сценарист аудіо-книг. Створи JSON-сценарій для озвучки.

Текст: {text[:6000]}

ВАЖЛИВО:
1. Розділи текст на МАЛЕНЬКІ сцени - кожна сцена це одна дія або момент
2. Наратив (закадровий текст) повинен бути КОРОТКИМ і розміщуватися МІЖ діалогами
3. Кожна сцена містить: ОДИН короткий нарратив (або порожній), фонові звуки, і діалоги
4. Визнач стать кожного персонажа для підбору голосу
5. НЕ ПЕРЕКЛАДАЙ текст! Залиш narrative та dialogues мовою оригіналу
6. Опис голосу (voice_description) можна писати українською

ПРИКЛАД правильної структури сцен:
- Сцена 1: нарратив "Тишу палуби порушувало гудіння систем" + діалоги (якщо є)
- Сцена 2: наратив "Капітан підійшов до ілюмінатора" + діалоги (якщо є)
- Сцена 3: наратив "" (порожній) + діалоги персонажів

Формат (ТІЛЬКИ JSON):
{{
  "title": "Назва",
  "narrator": {{
    "voice_description": "Глибокий, спокійний чоловічий голос для закадрового тексту",
    "gender": "male"
  }},
  "characters": [
    {{
      "name": "Ім'я",
      "gender": "male або female",
      "voice_description": "Детальний опис голосу українською (тембр, вік, емоції)"
    }}
  ],
  "scenes": [
    {{
      "narrative": "КОРОТКИЙ текст опису дії (1-2 речення) мовою оригіналу",
      "ambient_sounds": ["гудіння систем", "кроки"],
      "dialogues": [
        {{
          "character": "Ім'я персонажа",
          "text": "Текст діалогу мовою оригіналу",
          "emotion": "емоційний стан"
        }}
      ]
    }}
  ]
}}

Фонові звуки для прикладу: "гудіння систем", "кроки", "тремтіння", "сигналізація", "вітер", "дихання", "двигун", "тиша", "шум дощу"
"""
    
    # Список моделей для спроби
    models_to_try = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
    if WORKING_MODEL and WORKING_MODEL in models_to_try:
        models_to_try.remove(WORKING_MODEL)
        models_to_try.insert(0, WORKING_MODEL)
    
    max_retries = 3
    
    for model_name in models_to_try:
        for attempt in range(max_retries):
            try:
                print(f"[GEMINI] Спроба {attempt + 1}/{max_retries} з моделлю {model_name}...", flush=True)
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                result = extract_json(response.text)
                if result and "scenes" in result:
                    print(f"[GEMINI] ✅ Сцен: {len(result['scenes'])}", flush=True)
                    if "narrator" in result:
                        print(f"[GEMINI] Наратор: {result['narrator'].get('voice_description', 'default')}", flush=True)
                    chars = result.get("characters", [])
                    for c in chars:
                        print(f"[GEMINI] Персонаж: {c.get('name')} ({c.get('gender', 'unknown')})", flush=True)
                    return result
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    # Витягуємо час очікування з помилки
                    import re
                    retry_match = re.search(r'retry in ([\d.]+)s', error_str)
                    if retry_match:
                        wait_time = float(retry_match.group(1)) + 2  # Додаємо 2 секунди запасу
                    else:
                        wait_time = 40  # За замовчуванням
                    
                    if attempt < max_retries - 1:
                        print(f"[GEMINI] ⏳ Квота вичерпана. Очікування {wait_time:.0f} сек...", flush=True)
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"[GEMINI] ⚠️ Квота вичерпана для {model_name}, пробуємо іншу модель...", flush=True)
                        break
                else:
                    print(f"[GEMINI] ❌ Помилка: {e}", flush=True)
                    break
    
    print("[GEMINI] ❌ Всі спроби вичерпано", flush=True)
    return None

# Примітка: стара функція build_audio видалена, використовується _build_audio_with_voices

def create_srt(subtitles):
    """Створення SRT"""
    lines = []
    for s in subtitles:
        h1 = int(s["start"] // 3600)
        m1 = int((s["start"] % 3600) // 60)
        s1 = int(s["start"] % 60)
        ms1 = int((s["start"] % 1) * 1000)
        
        h2 = int(s["end"] // 3600)
        m2 = int((s["end"] % 3600) // 60)
        s2 = int(s["end"] % 60)
        ms2 = int((s["end"] % 1) * 1000)
        
        t1 = f"{h1:02d}:{m1:02d}:{s1:02d},{ms1:03d}"
        t2 = f"{h2:02d}:{m2:02d}:{s2:02d},{ms2:03d}"
        
        lines.append(f"{s['index']}")
        lines.append(f"{t1} --> {t2}")
        lines.append(f"{s['char']}: {s['text']}")
        lines.append("")
    
    return "\n".join(lines)

def estimate_audio_duration(scenario):
    """Попередня оцінка тривалості аудіофайлу
    
    Повертає словник з оцінками:
    - min_duration: мінімальна тривалість (хвилини)
    - max_duration: максимальна тривалість (хвилини)
    - total_words: загальна кількість слів
    - total_dialogues: кількість діалогів
    - total_narrative: кількість нарративних фрагментів
    """
    scenes = scenario.get("scenes", [])
    
    total_words = 0
    total_dialogues = 0
    total_narrative = 0
    
    for scene in scenes:
        # Наратив
        narrative = scene.get("narrative", "")
        if narrative:
            total_words += len(narrative.split())
            total_narrative += 1
        
        # Діалоги
        for dialogue in scene.get("dialogues", []):
            text = dialogue.get("text", "")
            total_words += len(text.split())
            total_dialogues += 1
    
    # Середня швидкість мовлення: 130-150 слів за хвилину для російської/української
    # З урахуванням пауз та емоцій - 120-140 слів/хв
    words_per_minute_min = 120
    words_per_minute_max = 150
    
    # Базова тривалість
    min_duration = total_words / words_per_minute_max
    max_duration = total_words / words_per_minute_min
    
    # Додаємо час на паузи:
    # - 0.3 сек після кожного нарративу
    # - 0.5 сек після кожного діалогу
    pause_time = (total_narrative * 0.3 + total_dialogues * 0.5) / 60  # в хвилинах
    min_duration += pause_time
    max_duration += pause_time
    
    # Додаємо 10% запас на можливі емоційні паузи
    min_duration *= 1.05
    max_duration *= 1.15
    
    return {
        "min_duration": round(min_duration, 1),
        "max_duration": round(max_duration, 1),
        "total_words": total_words,
        "total_dialogues": total_dialogues,
        "total_narrative": total_narrative,
        "estimated_time": round((min_duration + max_duration) / 2, 1)  # середня оцінка
    }

# =============================================================================
# TELEGRAM BOT
# =============================================================================

TELEGRAM_TOKEN = None
try:
    TELEGRAM_TOKEN = userdata.get("TELEGRAM_BOT_TOKEN")
except:
    pass

if not TELEGRAM_TOKEN:
    print("[ERROR] TELEGRAM_BOT_TOKEN не знайдено!", flush=True)
    sys.exit(1)

# Збереження стану користувачів
user_states = {}
user_voice_selections = {}

class Bot:
    def __init__(self, token):
        self.bot = telebot.TeleBot(token)
        self.setup()
        print("[BOT] Готово!", flush=True)
    
    def setup(self):
        @self.bot.message_handler(commands=['start'])
        def start(m):
            # Монтування Google Drive при старті
            mount_google_drive()
            load_voice_configs()
            
            self.bot.reply_to(m, f"""🎙 VIBEMODLY v69.0

 Модель: {WORKING_MODEL}
 TTS: Qwen3-TTS 1.7B VoiceDesign + Base
 
 ✨ Консистентність голосу: VoiceClone з reference аудіо
 🔄 Retry логіка: автоматичне очікування при квоті Gemini
 💾 Автозбереження: стан зберігається при лімітах Colab
 
 Команди:
 /voices - Список голосів
 /preview <голос> - Прослухати голос
 /newvoice - Створити голос
 /edit - Редагувати голос
 /save_state - Переглянути збережений стан
 /clear_state - Видалити збережений стан
 
 Надішли файл або текст для створення аудіо-книги.
 
 Підтримувані формати: EPUB, PDF, TXT, HTML, DOCX, FB2""")
        
        @self.bot.message_handler(commands=['voices'])
        def voices_cmd(m):
            """Список доступних голосів"""
            text = "🎙 Доступні голоси:\n\n"
            
            # Чоловічі
            text += "👨 ЧОЛОВІЧІ:\n"
            for key, preset in VOICE_PRESETS.items():
                if preset.get("gender") == "male" and not key.startswith("narrator"):
                    text += f"  /preview_{key} - {preset['name']}\n"
            
            # Жіночі
            text += "\n👩 ЖІНОЧІ:\n"
            for key, preset in VOICE_PRESETS.items():
                if preset.get("gender") == "female":
                    text += f"  /preview_{key} - {preset['name']}\n"
            
            # Диктори
            text += "\n📖 ДИКТОРИ:\n"
            for key, preset in VOICE_PRESETS.items():
                if key.startswith("narrator"):
                    text += f"  /preview_{key} - {preset['name']}\n"
            
            text += "\n💡 Натисніть на команду для прослуховування"
            
            self.bot.reply_to(m, text)
        
        @self.bot.message_handler(commands=['save_state'])
        def save_state_cmd(m):
            """Ручне збереження поточного стану"""
            if has_saved_state():
                state = load_generation_state()
                if state:
                    text = f"📋 Збережений стан:\n\n"
                    text += f"• Сценарій: {state.get('scenario_title', 'Невідомо')}\n"
                    text += f"• Сцена: {state.get('scene_idx', 0)}\n"
                    text += f"• Елемент: {state.get('current', 0)}\n"
                    text += f"• Час: {state.get('current_time', 0):.1f} сек\n"
                    text += f"• Збережено: {state.get('saved_at', 'Невідомо')}\n"
                    if state.get('colab_limit_reached'):
                        text += "\n⚠️ Зупинено через ліміт Colab\n"
                    text += "\n/clear_state - Видалити стан"
                    self.bot.reply_to(m, text)
            else:
                self.bot.reply_to(m, "📭 Збереженого стану немає.\n\nСтан автоматично створюється при досягненні лімітів Colab.")
        
        @self.bot.message_handler(commands=['clear_state'])
        def clear_state_cmd(m):
            """Видалення збереженого стану"""
            if has_saved_state():
                clear_generation_state()
                self.bot.reply_to(m, "🗑 Збережений стан видалено.\n\nГенерація почнеться з початку при наступному запуску.")
            else:
                self.bot.reply_to(m, "📭 Збереженого стану немає.")
        
        @self.bot.message_handler(func=lambda m: m.text.startswith('/preview_'))
        def preview_voice(m):
            """Прослуховування голосу"""
            voice_key = m.text.replace('/preview_', '')
            
            if voice_key not in VOICE_PRESETS:
                self.bot.reply_to(m, "Голос не знайдено. Використайте /voices")
                return
            
            preset = VOICE_PRESETS[voice_key]
            status = self.bot.reply_to(m, f"🔊 Генерація демо для '{preset['name']}'...")
            
            # Тестовий текст
            demo_text = "Привіт! Це демонстрація голосу. Я можу озвучувати ваші історії."
            
            audio = generate_audio(
                demo_text,
                character_name=f"demo_{voice_key}",
                gender=preset.get("gender", "male"),
                voice_preset=voice_key
            )
            
            if audio is not None:
                # Конвертація
                audio_int16 = (audio * 32767).astype(np.int16) if audio.max() <= 1 else audio.astype(np.int16)
                segment = AudioSegment(
                    audio_int16.tobytes(),
                    frame_rate=SAMPLE_RATE,
                    sample_width=2,
                    channels=1
                )
                
                buf = BytesIO()
                segment.export(buf, format="mp3")
                buf.seek(0)
                
                self.bot.delete_message(m.chat.id, status.message_id)
                self.bot.send_audio(m.chat.id, buf, caption=f"🎙 {preset['name']}\n{preset['description']}")
            else:
                self.bot.edit_message_text("Помилка генерації", m.chat.id, status.message_id)
        
        @self.bot.message_handler(commands=['newvoice'])
        def new_voice_cmd(m):
            """Створення нового голосу"""
            user_states[m.chat.id] = {"step": "name"}
            self.bot.reply_to(m, """🎙 Створення нового голосу

Введіть назву голосу:
(наприклад: "Марк - головний герой")""")
        
        @self.bot.message_handler(commands=['edit'])
        def edit_voice_cmd(m):
            """Редагування голосу"""
            text = "✏️ Редагування голосу\n\n"
            text += "Виберіть голос для редагування:\n\n"
            
            # Виводимо всі голоси з командами для редагування
            for key, preset in VOICE_PRESETS.items():
                text += f"/edit_{key} - {preset['name']}\n"
            
            text += "\n💡 Натисніть на команду для редагування"
            
            self.bot.reply_to(m, text)
        
        @self.bot.message_handler(func=lambda m: m.text.startswith('/edit_'))
        def edit_voice_select(m):
            """Вибір голосу для редагування"""
            voice_key = m.text.replace('/edit_', '')
            
            if voice_key not in VOICE_PRESETS:
                self.bot.reply_to(m, "Голос не знайдено. Використайте /edit")
                return
            
            preset = VOICE_PRESETS[voice_key]
            
            # Зберігаємо стан редагування
            user_states[m.chat.id] = {
                "step": "edit_field",
                "voice_key": voice_key,
                "preset": preset.copy()
            }
            
            # Показуємо поточні налаштування
            text = f"""✏️ Редагування: {preset['name']}

Поточні налаштування:
• Стать: {preset.get('gender', 'male')}
• Висота: {preset.get('pitch', 'medium')}
• Швидкість: {preset.get('speed', 1.0)}
• Опис: {preset.get('description', '')}

Що бажаєте змінити?
/gender - Стать
/pitch - Висоту
/speed - Швидкість
/desc - Опис
/save - Зберегти зміни
/cancel - Скасувати"""
            
            self.bot.reply_to(m, text)
        
        @self.bot.message_handler(commands=['gender', 'pitch', 'speed', 'desc', 'save', 'cancel'])
        def edit_voice_field(m):
            """Зміна поля голосу"""
            if m.chat.id not in user_states:
                self.bot.reply_to(m, "Спочатку виберіть голос: /edit")
                return
            
            state = user_states[m.chat.id]
            if state.get("step") != "edit_field":
                self.bot.reply_to(m, "Спочатку виберіть голос: /edit")
                return
            
            cmd = m.text.split('@')[0]  # Видаляємо @bot_name якщо є
            
            if cmd == "/cancel":
                del user_states[m.chat.id]
                self.bot.reply_to(m, "❌ Редагування скасовано")
                return
            
            elif cmd == "/save":
                # Зберігаємо зміни
                voice_key = state["voice_key"]
                VOICE_PRESETS[voice_key] = state["preset"]
                save_voice_configs()
                del user_states[m.chat.id]
                self.bot.reply_to(m, f"✅ Голос '{state['preset']['name']}' збережено!")
                return
            
            elif cmd == "/gender":
                state["step"] = "edit_gender"
                self.bot.reply_to(m, "Оберіть стать:\n/male - Чоловічий\n/female - Жіночий")
                return
            
            elif cmd == "/pitch":
                state["step"] = "edit_pitch"
                self.bot.reply_to(m, "Оберіть висоту:\n/low - Низький\n/medium - Середній\n/high - Високий")
                return
            
            elif cmd == "/speed":
                state["step"] = "edit_speed"
                self.bot.reply_to(m, f"Поточна швидкість: {state['preset'].get('speed', 1.0)}\nВведіть нову (0.5 - 2.0):")
                return
            
            elif cmd == "/desc":
                state["step"] = "edit_desc"
                self.bot.reply_to(m, f"Поточний опис: {state['preset'].get('description', '')}\nВведіть новий опис:")
                return
        
        @self.bot.callback_query_handler(func=lambda call: call.data.startswith("approve_") or call.data.startswith("change_"))
        def voice_approval_callback(call):
            """Обробка затвердження голосів"""
            chat_id = int(call.data.split("_")[1])
            
            if chat_id not in user_states:
                self.bot.answer_callback_query(call.id, "Сесія закінчилась")
                return
            
            state = user_states[chat_id]
            
            if call.data.startswith("approve_"):
                # Перевіряємо стан - якщо вже пройшли тест диктора, починаємо генерацію
                if state.get("step") == "ready_to_generate":
                    self.bot.answer_callback_query(call.id, "🚀 Починаємо генерацію!")
                    self.bot.edit_message_reply_markup(call.message.chat.id, call.message.message_id, reply_markup=None)
                    scenario = state.get("scenario")
                    voice_selections = state.get("voice_selections", {})
                    duration_est = state.get("duration_est", {})
                    status = self.bot.send_message(chat_id, "🔊 Генерація аудіо...")
                    self._generate_audio_impl(chat_id, scenario, status, voice_selections, duration_est)
                else:
                    # Перший раз - показуємо тест наративу перед генерацією
                    self.bot.answer_callback_query(call.id, "✅ Голоси затверджені!")
                    self.bot.edit_message_reply_markup(call.message.chat.id, call.message.message_id, reply_markup=None)
                    
                    # ВИКЛИКАЄМО ТЕСТ НАРАТИВУ перед генерацією
                    self._finalize_voice_approval(chat_id)
                
            elif call.data.startswith("change_"):
                # Змінити голоси - починаємо заново
                self.bot.answer_callback_query(call.id, "🔄 Виберіть нові голоси")
                self.bot.edit_message_reply_markup(call.message.chat.id, call.message.message_id, reply_markup=None)
                
                state["current_char_idx"] = 0
                state["voice_selections"] = {}
                state["step"] = "voice_approval"
                self._auto_assign_voices(chat_id)
        
        @self.bot.callback_query_handler(func=lambda call: call.data.startswith("prevvoice_") or call.data.startswith("nextvoice_"))
        def navigate_voice_callback(call):
            """Навігація між голосами"""
            # Формат: prevvoice_{chat_id} або nextvoice_{chat_id}
            try:
                chat_id = int(call.data.split("_")[1])
            except (IndexError, ValueError):
                self.bot.answer_callback_query(call.id, "Помилка даних")
                return
            
            if chat_id not in user_states:
                self.bot.answer_callback_query(call.id, "Сесія закінчилась")
                return
            
            state = user_states[chat_id]
            
            if call.data.startswith("prevvoice_"):
                state["current_voice_idx"] = max(0, state.get("current_voice_idx", 0) - 1)
            else:
                voices = state.get("available_voices", [])
                state["current_voice_idx"] = min(len(voices) - 1, state.get("current_voice_idx", 0) + 1)
            
            self.bot.answer_callback_query(call.id)
            self.bot.delete_message(call.message.chat.id, call.message.message_id)
            self._preview_voice_for_char(chat_id)
        
        @self.bot.callback_query_handler(func=lambda call: call.data.startswith("selectvoice_"))
        def select_voice_callback(call):
            """Вибір голосу для персонажа"""
            # Формат: selectvoice_{chat_id}_{voice_idx}
            try:
                parts = call.data.split("_")
                chat_id = int(parts[1])
                voice_idx = int(parts[2])
            except (IndexError, ValueError):
                self.bot.answer_callback_query(call.id, "Помилка даних")
                return
            
            if chat_id not in user_states:
                self.bot.answer_callback_query(call.id, "Сесія закінчилась")
                return
            
            state = user_states[chat_id]
            chars = state["chars"]
            idx = state["current_char_idx"]
            
            if idx >= len(chars):
                return
            
            # Отримуємо voice_key за індексом
            voices = state.get("available_voices", [])
            if voice_idx < 0 or voice_idx >= len(voices):
                self.bot.answer_callback_query(call.id, "Помилка: голос не знайдено")
                return
            
            voice_key = voices[voice_idx]
            char = chars[idx]
            char_name = char["name"]
            
            # Зберігаємо вибір
            state["voice_selections"][char_name] = voice_key
            preset = VOICE_PRESETS.get(voice_key, {})
            
            self.bot.answer_callback_query(call.id, f"✅ {char_name}: {preset.get('name', voice_key)}")
            self.bot.delete_message(call.message.chat.id, call.message.message_id)
            
            # Переходимо до наступного персонажа або показуємо підсумок
            state["current_char_idx"] = idx + 1
            if state["current_char_idx"] >= len(chars):
                # Всі персонажі вибрані
                self._show_voice_summary(chat_id)
            else:
                # Продовжуємо вибір для наступного персонажа
                self._preview_voice_for_char(chat_id)
        
        @self.bot.callback_query_handler(func=lambda call: call.data.startswith("regenvoice_"))
        def regen_voice_callback(call):
            """Перегенерація голосу (показати наступний)"""
            # Формат: regenvoice_{chat_id}
            try:
                chat_id = int(call.data.split("_")[1])
            except (IndexError, ValueError):
                self.bot.answer_callback_query(call.id, "Помилка даних")
                return
            
            if chat_id not in user_states:
                self.bot.answer_callback_query(call.id, "Сесія закінчилась")
                return
            
            state = user_states[chat_id]
            voices = state.get("available_voices", [])
            current_idx = state.get("current_voice_idx", 0)
            
            # Переходимо до наступного голосу
            if current_idx < len(voices) - 1:
                state["current_voice_idx"] = current_idx + 1
            else:
                state["current_voice_idx"] = 0  # Починаємо спочатку
            
            self.bot.answer_callback_query(call.id)
            self.bot.delete_message(call.message.chat.id, call.message.message_id)
            self._preview_voice_for_char(chat_id)
        
        @self.bot.callback_query_handler(func=lambda call: call.data.startswith("cancel_"))
        def cancel_voice_callback(call):
            """Скасування вибору голосу"""
            try:
                chat_id = int(call.data.split("_")[1])
            except (IndexError, ValueError):
                self.bot.answer_callback_query(call.id, "Помилка даних")
                return
            
            if chat_id in user_states:
                del user_states[chat_id]
            
            self.bot.answer_callback_query(call.id, "❌ Скасовано")
            self.bot.edit_message_reply_markup(call.message.chat.id, call.message.message_id, reply_markup=None)
        
        @self.bot.callback_query_handler(func=lambda call: call.data.startswith("changevoice|"))
        def change_single_voice_callback(call):
            """Зміна голосу для окремого персонажа"""
            # Формат: changevoice|{chat_id}|{char_idx}
            print(f"[CALLBACK] changevoice: {call.data}", flush=True)
            
            try:
                parts = call.data.split("|")
                chat_id = int(parts[1])
                char_idx = int(parts[2])
                print(f"[CALLBACK] chat_id={chat_id}, char_idx={char_idx}", flush=True)
            except (IndexError, ValueError) as e:
                print(f"[CALLBACK] ❌ Помилка парсингу: {e}", flush=True)
                self.bot.answer_callback_query(call.id, "Помилка даних")
                return
            
            if chat_id not in user_states:
                print(f"[CALLBACK] ❌ Сесія не знайдена: {chat_id}", flush=True)
                self.bot.answer_callback_query(call.id, "Сесія закінчилась")
                return
            
            state = user_states[chat_id]
            chars = state.get("chars", [])
            print(f"[CALLBACK] chars count: {len(chars)}", flush=True)
            
            if char_idx >= len(chars):
                print(f"[CALLBACK] ❌ char_idx {char_idx} >= {len(chars)}", flush=True)
                self.bot.answer_callback_query(call.id, "Помилка: персонаж не знайдено")
                return
            
            char = chars[char_idx]
            char_name = char.get("name", "Невідомо")
            char_gender = char.get("gender", "male")
            
            print(f"[CALLBACK] Персонаж: {char_name}, стать: {char_gender}", flush=True)
            
            # Отримуємо список голосів відповідної статі
            # Функція get_voice_preset_list вже виключає narrator_ голоси за замовчуванням
            gender_filter = "female" if char_gender == "female" else "male"
            voices = get_voice_preset_list(gender_filter, exclude_narrator=True)
            
            print(f"[CALLBACK] Голосів знайдено: {len(voices)} для статі {gender_filter}", flush=True)
            for v in voices:
                print(f"[CALLBACK]   - {v['key']}: {v['name']}", flush=True)
            
            if not voices:
                # Якщо голосів немає - показуємо всі голоси (без narrator)
                voices = get_voice_preset_list(None, exclude_narrator=True)
                print(f"[CALLBACK] Fallback: використовуємо всі голоси ({len(voices)})", flush=True)
            
            # Зберігаємо стан для вибору голосу
            state["changing_voice_for_char"] = char_idx
            state["available_voices"] = [v['key'] for v in voices[:8]]  # Збільшено до 8 голосів
            state["current_voice_idx"] = 0
            state["step"] = "selecting_single_voice"
            
            print(f"[CALLBACK] available_voices: {state['available_voices']}", flush=True)
            
            self.bot.answer_callback_query(call.id, f"🔄 Зміна голосу: {char_name}")
            self.bot.edit_message_reply_markup(call.message.chat.id, call.message.message_id, reply_markup=None)
            
            # Показуємо голоси для вибору
            self._show_voice_options_for_char(chat_id, char_idx)
        
        @self.bot.callback_query_handler(func=lambda call: call.data.startswith("selectsinglevoice_"))
        def select_single_voice_callback(call):
            """Вибір нового голосу для персонажа"""
            # Формат: selectsinglevoice_{chat_id}_{voice_idx}
            try:
                parts = call.data.split("_")
                chat_id = int(parts[1])
                voice_idx = int(parts[2])
            except (IndexError, ValueError):
                self.bot.answer_callback_query(call.id, "Помилка даних")
                return
            
            if chat_id not in user_states:
                self.bot.answer_callback_query(call.id, "Сесія закінчилась")
                return
            
            state = user_states[chat_id]
            voices = state.get("available_voices", [])
            
            if voice_idx < 0 or voice_idx >= len(voices):
                self.bot.answer_callback_query(call.id, "Помилка: голос не знайдено")
                return
            
            voice_key = voices[voice_idx]
            char_idx = state.get("changing_voice_for_char", 0)
            chars = state["chars"]
            
            if char_idx >= len(chars):
                self.bot.answer_callback_query(call.id, "Помилка: персонаж не знайдено")
                return
            
            char = chars[char_idx]
            char_name = char["name"]
            
            # Оновлюємо вибір голосу
            state["voice_selections"][char_name] = voice_key
            # ОЧИЩЕННЯ КЕШУ: видаляємо старі тестові голоси при зміні голосу персонажа,
            # щоб гарантувати перегенерацію з новим голосом
            cache_key = f"{char_name}_{voice_key}"
            if cache_key in test_voice_cache:
                del test_voice_cache[cache_key]
            generated_test_voices = state.get("generated_test_voices", {})
            if char_name in generated_test_voices:
                del generated_test_voices[char_name]
            preset = VOICE_PRESETS.get(voice_key, {})
            
            self.bot.answer_callback_query(call.id, f"✅ {char_name}: {preset.get('name', voice_key)}")
            self.bot.delete_message(call.message.chat.id, call.message.message_id)
            
            # Повертаємося до підсумку
            state["step"] = "waiting_approval"
            self._show_voice_summary(chat_id)
        
        @self.bot.callback_query_handler(func=lambda call: call.data.startswith("prevsinglevoice_") or call.data.startswith("nextsinglevoice_"))
        def navigate_single_voice_callback(call):
            """Навігація між голосами при зміні голосу персонажа"""
            try:
                chat_id = int(call.data.split("_")[1])
            except (IndexError, ValueError):
                self.bot.answer_callback_query(call.id, "Помилка даних")
                return
            
            if chat_id not in user_states:
                self.bot.answer_callback_query(call.id, "Сесія закінчилась")
                return
            
            state = user_states[chat_id]
            voices = state.get("available_voices", [])
            
            if call.data.startswith("prevsinglevoice_"):
                state["current_voice_idx"] = max(0, state.get("current_voice_idx", 0) - 1)
            else:
                state["current_voice_idx"] = min(len(voices) - 1, state.get("current_voice_idx", 0) + 1)
            
            self.bot.answer_callback_query(call.id)
            self.bot.delete_message(call.message.chat.id, call.message.message_id)
            
            char_idx = state.get("changing_voice_for_char", 0)
            self._show_voice_options_for_char(chat_id, char_idx)
        
        @self.bot.callback_query_handler(func=lambda call: call.data.startswith("cancelvoicechange_"))
        def cancel_voice_change_callback(call):
            """Скасування зміни голосу"""
            try:
                chat_id = int(call.data.split("_")[1])
            except (IndexError, ValueError):
                self.bot.answer_callback_query(call.id, "Помилка даних")
                return
            
            if chat_id not in user_states:
                self.bot.answer_callback_query(call.id, "Сесія закінчилась")
                return
            
            state = user_states[chat_id]
            state["step"] = "waiting_approval"
            
            self.bot.answer_callback_query(call.id, "❌ Скасовано")
            self.bot.delete_message(call.message.chat.id, call.message.message_id)
            
            # Повертаємося до підсумку
            self._show_voice_summary(chat_id)
        
        @self.bot.callback_query_handler(func=lambda call: call.data.startswith("continue_") or call.data.startswith("stop_"))
        def checkpoint_callback_handler(call):
            """Обробка кнопок контрольної точки - сигналізує через Event"""
            try:
                parts = call.data.split("_")
                action = parts[0]
                chat_id = int(parts[1])
            except (IndexError, ValueError):
                self.bot.answer_callback_query(call.id, "Помилка даних")
                return
            
            if chat_id not in user_states:
                self.bot.answer_callback_query(call.id, "Сесія закінчилась")
                self.bot.edit_message_reply_markup(call.message.chat.id, call.message.message_id, reply_markup=None)
                return
            
            state = user_states[chat_id]
            
            if action == "stop":
                # Зупиняємо генерацію
                state["generation_stopped"] = True
                state["checkpoint_approved"] = False
                
                # Сигналізуємо через Event
                checkpoint_event = state.get("checkpoint_event")
                if checkpoint_event:
                    checkpoint_event.set()
                
                self.bot.answer_callback_query(call.id, "⏹ Генерацію зупинено")
                self.bot.edit_message_reply_markup(call.message.chat.id, call.message.message_id, reply_markup=None)
                print(f"[CHECKPOINT] ⏹ Користувач натиснув 'Зупинити'", flush=True)
            else:
                # Продовжуємо генерацію
                state["generation_stopped"] = False
                state["checkpoint_approved"] = True
                
                # Сигналізуємо через Event
                checkpoint_event = state.get("checkpoint_event")
                if checkpoint_event:
                    checkpoint_event.set()
                
                self.bot.answer_callback_query(call.id, "✅ Продовжуємо")
                self.bot.edit_message_reply_markup(call.message.chat.id, call.message.message_id, reply_markup=None)
                print(f"[CHECKPOINT] ✅ Користувач натиснув 'Продовжити'", flush=True)
        
        @self.bot.message_handler(content_types=['document'])
        def doc(m):
            try:
                name = m.document.file_name
                ext = name.split(".")[-1].lower()
                
                if ext not in ["epub", "pdf", "txt", "html", "docx", "fb2"]:
                    self.bot.reply_to(m, "Непідтримуваний формат")
                    return
                
                status = self.bot.reply_to(m, "Завантаження...")
                
                info = self.bot.get_file(m.document.file_id)
                data = self.bot.download_file(info.file_path)
                
                path = f"/content/{name}"
                with open(path, "wb") as f:
                    f.write(data)
                
                content = read_file(path, ext)
                os.remove(path)
                
                if not content:
                    self.bot.edit_message_text("Помилка читання", m.chat.id, status.message_id)
                    return
                
                self.process(m, content, status)
            except Exception as e:
                self.bot.reply_to(m, f"Помилка: {e}")
        
        @self.bot.message_handler(content_types=['text'])
        def txt(m):
            # Перевірка стану створення голосу
            if m.chat.id in user_states:
                state = user_states[m.chat.id]
                
                # === СТВОРЕННЯ НОВОГО ГОЛОСУ ===
                if state.get("step") == "name":
                    state["name"] = m.text
                    state["step"] = "gender"
                    self.bot.reply_to(m, "Оберіть стать:\n/male - Чоловічий\n/female - Жіночий")
                    return
                
                elif state.get("step") == "gender":
                    if m.text.startswith("/male"):
                        state["gender"] = "male"
                    elif m.text.startswith("/female"):
                        state["gender"] = "female"
                    else:
                        self.bot.reply_to(m, "Оберіть стать:\n/male - Чоловічий\n/female - Жіночий")
                        return
                    
                    state["step"] = "description"
                    self.bot.reply_to(m, "Введіть опис голосу:\n(наприклад: 'Глибокий, спокійний, впевнений')")
                    return
                
                elif state.get("step") == "description":
                    state["description"] = m.text
                    state["step"] = "pitch"
                    self.bot.reply_to(m, "Оберіть висоту голосу:\n/low - Низький\n/medium - Середній\n/high - Високий")
                    return
                
                elif state.get("step") == "pitch":
                    if m.text.startswith("/low"):
                        state["pitch"] = "low"
                    elif m.text.startswith("/medium"):
                        state["pitch"] = "medium"
                    elif m.text.startswith("/high"):
                        state["pitch"] = "high"
                    else:
                        self.bot.reply_to(m, "Оберіть висоту:\n/low - Низький\n/medium - Середній\n/high - Високий")
                        return
                    
                    state["step"] = "speed"
                    self.bot.reply_to(m, "Введіть швидкість (0.5 - 2.0):\n(1.0 - нормальна, 0.8 - повільно, 1.2 - швидко)")
                    return
                
                elif state.get("step") == "speed":
                    try:
                        speed = float(m.text)
                        if 0.5 <= speed <= 2.0:
                            state["speed"] = speed
                        else:
                            raise ValueError()
                    except:
                        self.bot.reply_to(m, "Введіть число від 0.5 до 2.0")
                        return
                    
                    # Створення голосу
                    key = create_custom_voice(
                        name=state["name"],
                        gender=state["gender"],
                        description=state["description"],
                        pitch=state["pitch"],
                        speed=state["speed"],
                        prompt_suffix=f"{state['description']} voice."
                    )
                    
                    del user_states[m.chat.id]
                    self.bot.reply_to(m, f"✅ Голос '{state['name']}' створено!\nКлюч: {key}\n\nВикористайте /voices для перегляду")
                    return
                
                # === РЕДАГУВАННЯ ГОЛОСУ ===
                elif state.get("step") == "edit_gender":
                    if m.text.startswith("/male"):
                        state["preset"]["gender"] = "male"
                    elif m.text.startswith("/female"):
                        state["preset"]["gender"] = "female"
                    else:
                        self.bot.reply_to(m, "Оберіть стать:\n/male - Чоловічий\n/female - Жіночий")
                        return
                    
                    state["step"] = "edit_field"
                    self.bot.reply_to(m, f"✅ Стать змінено на {state['preset']['gender']}\n\n/gender /pitch /speed /desc /save /cancel")
                    return
                
                elif state.get("step") == "edit_pitch":
                    if m.text.startswith("/low"):
                        state["preset"]["pitch"] = "low"
                    elif m.text.startswith("/medium"):
                        state["preset"]["pitch"] = "medium"
                    elif m.text.startswith("/high"):
                        state["preset"]["pitch"] = "high"
                    else:
                        self.bot.reply_to(m, "Оберіть висоту:\n/low - Низький\n/medium - Середній\n/high - Високий")
                        return
                    
                    state["step"] = "edit_field"
                    self.bot.reply_to(m, f"✅ Висоту змінено на {state['preset']['pitch']}\n\n/gender /pitch /speed /desc /save /cancel")
                    return
                
                elif state.get("step") == "edit_speed":
                    try:
                        speed = float(m.text)
                        if 0.5 <= speed <= 2.0:
                            state["preset"]["speed"] = speed
                            state["step"] = "edit_field"
                            self.bot.reply_to(m, f"✅ Швидкість змінено на {speed}\n\n/gender /pitch /speed /desc /save /cancel")
                        else:
                            raise ValueError()
                    except:
                        self.bot.reply_to(m, "Введіть число від 0.5 до 2.0")
                    return
                
                elif state.get("step") == "edit_desc":
                    state["preset"]["description"] = m.text
                    state["preset"]["prompt_suffix"] = f"{m.text} voice."
                    state["step"] = "edit_field"
                    self.bot.reply_to(m, f"✅ Опис змінено\n\n/gender /pitch /speed /desc /save /cancel")
                    return
            
            # Звичайна обробка тексту
            status = self.bot.reply_to(m, "Аналіз...")
            self.process(m, m.text, status)
    
    def process(self, m, content, status):
        try:
            self.bot.edit_message_text("Аналіз сюжету...", m.chat.id, status.message_id)
            
            scenario = analyze_text(content)
            if not scenario:
                self.bot.edit_message_text("Помилка сценарію", m.chat.id, status.message_id)
                return
            
            chars = scenario.get("characters", [])
            scenes = len(scenario.get("scenes", []))
            
            # Попередній аналіз тривалості
            duration_est = estimate_audio_duration(scenario)
            
            # Формування повідомлення з оцінкою
            est = duration_est["estimated_time"]
            min_est = duration_est["min_duration"]
            max_est = duration_est["max_duration"]
            words = duration_est["total_words"]
            dialogues = duration_est["total_dialogues"]
            narrative = duration_est["total_narrative"]
            
            # === СИСТЕМА ЗАТВЕРДЖЕННЯ ГОЛОСІВ ===
            if chars:
                # Зберігаємо сценарій та стан для подальшої обробки
                user_states[m.chat.id] = {
                    "step": "voice_approval",
                    "scenario": scenario,
                    "chars": chars,
                    "current_char_idx": 0,
                    "voice_selections": {},  # Вибрані голоси для персонажів
                    "duration_est": duration_est
                }
                
                # Показуємо інформацію та починаємо затвердження голосів
                char_list = "\n".join([f"  • {c['name']} ({c.get('gender', 'unknown')})" for c in chars])
                self.bot.edit_message_text(
                    f"📊 Аналіз завершено:\n\n"
                    f"👤 Персонажів: {len(chars)}\n"
                    f"🎬 Сцен: {scenes}\n"
                    f"💬 Діалогів: {dialogues}\n"
                    f"📖 Наративів: {narrative}\n"
                    f"📝 Слів: {words}\n\n"
                    f"⏱ Очікувана тривалість: {est} хв\n"
                    f"   (від {min_est} до {max_est} хв)\n\n"
                    f"Персонажі:\n{char_list}\n\n"
                    f"🎙 Починаємо вибір голосів...",
                    m.chat.id, status.message_id
                )
                
                # Починаємо автоматичне призначення голосів
                self._auto_assign_voices(m.chat.id)
                return
            
            # Якщо персонажів немає - одразу генеруємо
            self.bot.edit_message_text(
                f"📊 Аналіз завершено:\n\n"
                f"🎬 Сцен: {scenes}\n"
                f"💬 Діалогів: {dialogues}\n"
                f"📖 Наративів: {narrative}\n"
                f"📝 Слів: {words}\n\n"
                f"⏱ Очікувана тривалість: {est} хв\n"
                f"   (від {min_est} до {max_est} хв)\n\n"
                f"🔊 Генерація аудіо...",
                m.chat.id, status.message_id
            )
            
            self._generate_audio(m, scenario, status, {})
            
        except Exception as e:
            self.bot.reply_to(m, f"Помилка: {e}")
    
    def _auto_assign_voices(self, chat_id):
        """Автоматичне призначення голосів за статтю та пакетна генерація тестів"""
        global test_voice_cache, character_voice_params
        
        state = user_states.get(chat_id)
        if not state or state.get("step") != "voice_approval":
            return
        
        chars = state["chars"]
        scenario = state["scenario"]
        
        # Визначаємо мову тексту
        scenes = scenario.get("scenes", [])
        text_lang = "russian"
        if scenes:
            for scene in scenes:
                narrative = scene.get("narrative", "")
                if narrative:
                    text_lang = detect_language(narrative)
                    break
                dialogues = scene.get("dialogues", [])
                if dialogues:
                    sample_text = dialogues[0].get("text", "")
                    if sample_text:
                        text_lang = detect_language(sample_text)
                        break
        
        state["text_lang"] = text_lang
        
        # Автоматично призначаємо голоси за статтю
        voice_selections = {}
        voices_to_generate = []
        
        for char in chars:
            char_name = char["name"]
            char_gender = char.get("gender", "male")
            
            # Вибираємо голос за замовчуванням для статі
            if char_gender == "female":
                default_voice = "female_soft"
            else:
                default_voice = "male_deep"
            
            voice_selections[char_name] = default_voice
            voices_to_generate.append((char_name, default_voice, char_gender))
        
        state["voice_selections"] = voice_selections
        
        # Показуємо повідомлення про генерацію
        status = self.bot.send_message(
            chat_id, 
            f"🎙 Автоматичне призначення голосів...\n"
            f"📊 Персонажів: {len(chars)}\n"
            f"🔊 Генерація тестових голосів..."
        )
        
        # Пакетна генерація тестових голосів
        demo_text = get_demo_text(text_lang, long=False)
        generated_voices = {}
        
        for char_name, voice_key, gender in voices_to_generate:
            cache_key = f"{voice_key}_{text_lang}"
            
            # Перевіряємо кеш
            if cache_key in test_voice_cache:
                print(f"[CACHE] Використано кеш: {cache_key}", flush=True)
                generated_voices[char_name] = test_voice_cache[cache_key]
                continue
            
            # Генеруємо тестовий голос
            print(f"[TTS] Генерація тесту для {char_name}: {voice_key}", flush=True)
            audio = generate_audio(
                demo_text,
                character_name=f"test_{voice_key}",
                gender=gender,
                voice_preset=voice_key,
                lang=text_lang
            )
            
            # Повторна спроба при помилці
            if audio is None or len(audio) < SAMPLE_RATE:
                print(f"[TTS] Повторна спроба для {voice_key}...", flush=True)
                audio = generate_audio(
                    demo_text,
                    character_name=f"test_{voice_key}_retry",
                    gender=gender,
                    voice_preset=voice_key,
                    lang=text_lang,
                    custom_seed=hash(voice_key) % 10000 + 500
                )
            
            if audio is not None and len(audio) >= SAMPLE_RATE:
                test_voice_cache[cache_key] = audio
                generated_voices[char_name] = audio
                print(f"[TTS] ✅ {char_name}: {len(audio)/SAMPLE_RATE:.1f} сек", flush=True)
            else:
                print(f"[TTS] ❌ {char_name}: помилка генерації", flush=True)
                generated_voices[char_name] = None
        
        state["generated_test_voices"] = generated_voices
        
        # Видаляємо повідомлення про генерацію
        self.bot.delete_message(chat_id, status.message_id)
        
        # Показуємо підсумок з кнопками
        self._show_voice_summary(chat_id)
    
    def _show_voice_summary(self, chat_id):
        """Показ підсумку призначених голосів з можливістю зміни"""
        state = user_states.get(chat_id)
        if not state:
            return
        
        chars = state["chars"]
        voice_selections = state.get("voice_selections", {})
        generated_voices = state.get("generated_test_voices", {})
        text_lang = state.get("text_lang", "russian")
        
        # Відправляємо аудіо для кожного персонажа
        for i, char in enumerate(chars):
            char_name = char["name"]
            char_gender = char.get("gender", "male")
            voice_key = voice_selections.get(char_name, "default")
            preset = VOICE_PRESETS.get(voice_key, {})
            
            # Отримуємо згенерований тестовий голос
            audio = generated_voices.get(char_name)
            
            if audio is not None and len(audio) >= SAMPLE_RATE:
                # Конвертація
                max_val = np.max(np.abs(audio)) if len(audio) > 0 else 0
                if max_val > 0:
                    audio_normalized = audio / max_val * 0.9
                else:
                    audio_normalized = audio
                
                audio_int16 = (audio_normalized * 32767).astype(np.int16)
                
                segment = AudioSegment(
                    audio_int16.tobytes(),
                    frame_rate=SAMPLE_RATE,
                    sample_width=2,
                    channels=1
                )
                
                buf = BytesIO()
                segment.export(buf, format="mp3")
                buf.seek(0)
                
                gender_icon = "👨" if char_gender == "male" else "👩"
                duration_sec = len(audio) / SAMPLE_RATE
                
                # Створюємо клавіатуру з кнопкою зміни
                markup = tb_types.InlineKeyboardMarkup()
                markup.add(
                    tb_types.InlineKeyboardButton(
                        f"🔄 Змінити голос",
                        callback_data=f"changevoice|{chat_id}|{i}"
                    )
                )
                
                self.bot.send_audio(
                    chat_id, 
                    buf, 
                    caption=f"{gender_icon} **{char_name}**\n"
                           f"🎙 {preset.get('name', voice_key)}\n"
                           f"⏱ {duration_sec:.1f} сек",
                    parse_mode="Markdown",
                    reply_markup=markup
                )
            else:
                # Якщо аудіо не згенеровано - показуємо тільки текст
                gender_icon = "👨" if char_gender == "male" else "👩"
                text = f"{gender_icon} **{char_name}**\n"
                text += f"🎙 {preset.get('name', voice_key)}\n"
                text += f"⚠️ Тестовий голос не згенеровано\n"
                
                markup = tb_types.InlineKeyboardMarkup()
                markup.add(
                    tb_types.InlineKeyboardButton(
                        f"🔄 Змінити голос",
                        callback_data=f"changevoice|{chat_id}|{i}"
                    )
                )

                self.bot.send_message(chat_id, text, parse_mode="Markdown", reply_markup=markup)
        
        # Показуємо фінальне повідомлення з кнопками затвердження
        text = "🎙 **Голоси призначені!**\n\n"
        text += "💡 Прослухайте голоси вище та змініть якщо потрібно.\n"
        text += "Натисніть 'Затвердити' коли готові."
        
        # Кнопки затвердження та скасування
        markup = tb_types.InlineKeyboardMarkup(row_width=2)
        markup.row(
            tb_types.InlineKeyboardButton("✅ Затвердити всі", callback_data=f"approve_{chat_id}"),
            tb_types.InlineKeyboardButton("⏹ Скасувати", callback_data=f"cancel_{chat_id}")
        )
        
        self.bot.send_message(chat_id, text, parse_mode="Markdown", reply_markup=markup)
        
        # Змінюємо стан на очікування затвердження
        state["step"] = "waiting_approval"
    
    def _show_voice_options_for_char(self, chat_id, char_idx):
        """Показ варіантів голосів для вибору з генерацією аудіо"""
        global test_voice_cache, character_voice_params
        
        print(f"[VOICE] _show_voice_options_for_char: chat_id={chat_id}, char_idx={char_idx}", flush=True)
        
        state = user_states.get(chat_id)
        if not state:
            print(f"[VOICE] ❌ Стан не знайдено для chat_id={chat_id}", flush=True)
            return
        
        chars = state["chars"]
        if char_idx >= len(chars):
            print(f"[VOICE] ❌ char_idx {char_idx} >= len(chars) {len(chars)}", flush=True)
            return
        
        char = chars[char_idx]
        char_name = char["name"]
        char_gender = char.get("gender", "male")
        
        voices = state.get("available_voices", [])
        voice_idx = state.get("current_voice_idx", 0)
        
        print(f"[VOICE] Персонаж: {char_name}, стать: {char_gender}, голосів: {len(voices)}, поточний: {voice_idx}", flush=True)
        
        if not voices:
            # Якщо немає голосів - показуємо помилку
            self.bot.send_message(
                chat_id, 
                f"⚠️ Не знайдено голосів для статі: {char_gender}\n"
                f"Спробуйте змінити стать персонажа або додайте нові голоси."
            )
            return
        
        if voice_idx >= len(voices):
            voice_idx = 0
            state["current_voice_idx"] = 0
        
        voice_key = voices[voice_idx]
        preset = VOICE_PRESETS.get(voice_key, {})
        text_lang = state.get("text_lang", "russian")
        
        # Генеруємо тестове аудіо для поточного голосу
        demo_text = get_demo_text(text_lang, long=False)
        cache_key = f"{voice_key}_{text_lang}"
        
        status = self.bot.send_message(chat_id, f"🔊 Генерація тесту: {preset.get('name', voice_key)}...")
        
        # Перевіряємо кеш
        audio = test_voice_cache.get(cache_key)
        
        if audio is None:
            # Генеруємо тестове аудіо
            audio = generate_audio(
                demo_text,
                character_name=f"test_{voice_key}",
                gender=char_gender,
                voice_preset=voice_key,
                lang=text_lang
            )
            
            # Якщо не вдалося - пробуємо ще раз з іншим seed
            if audio is None or len(audio) < SAMPLE_RATE:
                print(f"[TTS] Повторна спроба для {voice_key}...", flush=True)
                audio = generate_audio(
                    demo_text,
                    character_name=f"test_{voice_key}_retry",
                    gender=char_gender,
                    voice_preset=voice_key,
                    lang=text_lang,
                    custom_seed=hash(voice_key) % 10000 + 500
                )
            
            if audio is not None and len(audio) >= SAMPLE_RATE:
                test_voice_cache[cache_key] = audio
                print(f"[CACHE] Збережено тестовий голос: {cache_key}", flush=True)
        
        # Перевіряємо чи аудіо валідне
        if audio is None or len(audio) < SAMPLE_RATE:
            # Показуємо помилку з можливістю спробувати інший голос
            markup = tb_types.InlineKeyboardMarkup(row_width=2)
            nav_buttons = []
            if voice_idx > 0:
                nav_buttons.append(tb_types.InlineKeyboardButton("⬅️", callback_data=f"prevsinglevoice_{chat_id}"))
            if voice_idx < len(voices) - 1:
                nav_buttons.append(tb_types.InlineKeyboardButton("➡️", callback_data=f"nextsinglevoice_{chat_id}"))
            if nav_buttons:
                markup.row(*nav_buttons)
            markup.row(
                tb_types.InlineKeyboardButton("❌ Скасувати", callback_data=f"cancelvoicechange_{chat_id}")
            )
            
            self.bot.edit_message_text(
                f"⚠️ Помилка генерації для {preset.get('name', voice_key)}\nОберіть інший голос:",
                chat_id, status.message_id, reply_markup=markup
            )
            return
        
        # Конвертація
        max_val = np.max(np.abs(audio)) if len(audio) > 0 else 0
        if max_val > 0:
            audio_normalized = audio / max_val * 0.9
        else:
            audio_normalized = audio
        
        audio_int16 = (audio_normalized * 32767).astype(np.int16)
        
        segment = AudioSegment(
            audio_int16.tobytes(),
            frame_rate=SAMPLE_RATE,
            sample_width=2,
            channels=1
        )
        
        buf = BytesIO()
        segment.export(buf, format="mp3")
        buf.seek(0)
        
        # Створюємо клавіатуру
        markup = tb_types.InlineKeyboardMarkup(row_width=3)
        
        # Кнопки навігації
        nav_buttons = []
        if voice_idx > 0:
            nav_buttons.append(tb_types.InlineKeyboardButton("⬅️ Попередній", callback_data=f"prevsinglevoice_{chat_id}"))
        if voice_idx < len(voices) - 1:
            nav_buttons.append(tb_types.InlineKeyboardButton("➡️ Наступний", callback_data=f"nextsinglevoice_{chat_id}"))
        
        if nav_buttons:
            markup.row(*nav_buttons)
        
        # Кнопки для кожного голосу
        voice_buttons = []
        for i, vkey in enumerate(voices):
            v = VOICE_PRESETS.get(vkey, {})
            marker = "✓ " if i == voice_idx else ""
            voice_buttons.append(
                tb_types.InlineKeyboardButton(
                    f"{marker}{i+1}",
                    callback_data=f"selectsinglevoice_{chat_id}_{i}"
                )
            )
        
        markup.row(*voice_buttons)
        
        # Кнопка скасування
        markup.row(
            tb_types.InlineKeyboardButton("❌ Скасувати", callback_data=f"cancelvoicechange_{chat_id}")
        )
        
        duration_sec = len(audio) / SAMPLE_RATE
        caption = f"🎭 **Зміна голосу для: {char_name}**\n"
        caption += f"🎙 Голос: **{preset.get('name', voice_key)}**\n"
        caption += f"📝 {preset.get('description', '')}\n"
        caption += f"⏱ {duration_sec:.1f} сек\n"
        caption += f"\n📊 Голос {voice_idx + 1} з {len(voices)}"
        
        self.bot.delete_message(chat_id, status.message_id)
        self.bot.send_audio(
            chat_id, 
            buf, 
            caption=caption,
            parse_mode="Markdown",
            reply_markup=markup
        )
    
    def _preview_voice_for_char(self, chat_id):
        """Показ голосу з тестовим аудіо для поточного персонажа"""
        global test_voice_cache, character_voice_params
        
        state = user_states.get(chat_id)
        if not state:
            return
        
        chars = state["chars"]
        idx = state["current_char_idx"]
        char = chars[idx]
        char_name = char["name"]
        char_gender = char.get("gender", "male")
        
        voices = state.get("available_voices", [])
        voice_idx = state.get("current_voice_idx", 0)
        
        if not voices or voice_idx >= len(voices):
            return
        
        voice_key = voices[voice_idx]
        preset = VOICE_PRESETS.get(voice_key, {})
        
        # Визначаємо мову тексту з сценарію
        scenario = state.get("scenario", {})
        scenes = scenario.get("scenes", [])
        text_lang = "russian"  # За замовчуванням
        
        if scenes:
            # Беремо текст першого діалогу для визначення мови
            for scene in scenes:
                dialogues = scene.get("dialogues", [])
                if dialogues:
                    sample_text = dialogues[0].get("text", "")
                    if sample_text:
                        text_lang = detect_language(sample_text)
                        break
                narrative = scene.get("narrative", "")
                if narrative:
                    text_lang = detect_language(narrative)
                    break
        
        # Спочатку пробуємо короткий текст (більш надійно)
        demo_text = get_demo_text(text_lang, long=False)
        
        # Перевіряємо кеш (короткий текст)
        cache_key = f"{voice_key}_{text_lang}"
        audio = test_voice_cache.get(cache_key)
        
        status = self.bot.send_message(chat_id, f"🔊 Генерація тесту: {preset.get('name', voice_key)}...")
        
        if audio is None:
            # Генеруємо тестове аудіо
            audio = generate_audio(
                demo_text,
                character_name=f"test_{voice_key}",
                gender=char_gender,
                voice_preset=voice_key,
                lang=text_lang
            )
            
            # Якщо не вдалося - пробуємо ще раз з іншим seed
            if audio is None or len(audio) < SAMPLE_RATE:  # менше 1 секунди
                print(f"[TTS] Повторна спроба для {voice_key}...", flush=True)
                audio = generate_audio(
                    demo_text,
                    character_name=f"test_{voice_key}_retry",
                    gender=char_gender,
                    voice_preset=voice_key,
                    lang=text_lang,
                    custom_seed=hash(voice_key) % 10000 + 500  # інший seed
                )
            
            if audio is not None and len(audio) >= SAMPLE_RATE:
                # Зберігаємо в кеш
                test_voice_cache[cache_key] = audio
                print(f"[CACHE] Збережено тестовий голос: {cache_key} ({len(audio)} семплів)", flush=True)
        else:
            print(f"[CACHE] Використано кешований голос: {cache_key}", flush=True)
        
        # Перевіряємо чи аудіо валідне
        if audio is None or len(audio) < SAMPLE_RATE:  # менше 1 секунди = помилка
            error_msg = "Помилка генерації тесту"
            if audio is not None:
                error_msg += f" (тільки {len(audio)/SAMPLE_RATE:.1f} сек)"
            
            # Створюємо клавіатуру з можливістю спробувати інший голос
            markup = tb_types.InlineKeyboardMarkup(row_width=2)
            markup.row(
                tb_types.InlineKeyboardButton("🔄 Спробувати інший", callback_data=f"regenvoice_{chat_id}"),
                tb_types.InlineKeyboardButton("⏹ Скасувати", callback_data=f"cancel_{chat_id}")
            )
            
            self.bot.edit_message_text(error_msg, chat_id, status.message_id, reply_markup=markup)
            return
        
        # Конвертація
        max_val = np.max(np.abs(audio)) if len(audio) > 0 else 0
        if max_val > 0:
            audio_normalized = audio / max_val * 0.9
        else:
            audio_normalized = audio
        
        audio_int16 = (audio_normalized * 32767).astype(np.int16)
        
        segment = AudioSegment(
            audio_int16.tobytes(),
            frame_rate=SAMPLE_RATE,
            sample_width=2,
            channels=1
        )
        
        buf = BytesIO()
        segment.export(buf, format="mp3")
        buf.seek(0)
        
        # Створюємо інлайн-клавіатуру
        markup = tb_types.InlineKeyboardMarkup(row_width=2)
        
        # Кнопки навігації
        nav_buttons = []
        if voice_idx > 0:
            nav_buttons.append(tb_types.InlineKeyboardButton("⬅️ Попередній", callback_data=f"prevvoice_{chat_id}"))
        if voice_idx < len(voices) - 1:
            nav_buttons.append(tb_types.InlineKeyboardButton("➡️ Наступний", callback_data=f"nextvoice_{chat_id}"))
        
        if nav_buttons:
            markup.row(*nav_buttons)
        
        # Кнопки дій
        markup.row(
            tb_types.InlineKeyboardButton("✅ Вибрати цей голос", callback_data=f"selectvoice_{chat_id}_{voice_idx}"),
            tb_types.InlineKeyboardButton("🔄 Інший голос", callback_data=f"regenvoice_{chat_id}")
        )
        
        duration_sec = len(audio) / SAMPLE_RATE
        caption = f"🎭 Персонаж: **{char_name}**\n"
        caption += f"🎙 Голос: **{preset.get('name', voice_key)}**\n"
        caption += f"📝 Опис: {preset.get('description', '')}\n"
        caption += f"🌐 Мова: {text_lang}\n"
        caption += f"⏱ Тривалість: {duration_sec:.1f} сек\n"
        caption += f"\n📊 Голос {voice_idx + 1} з {len(voices)}"
        
        self.bot.delete_message(chat_id, status.message_id)
        self.bot.send_audio(
            chat_id, 
            buf, 
            caption=caption,
            parse_mode="Markdown",
            reply_markup=markup
        )
    
    def _finalize_voice_approval(self, chat_id):
        """Завершення затвердження голосів - показ підсумку та тест диктора"""
        global test_voice_cache, character_voice_params
        
        print(f"[DICTOR] === ПОЧАТОК ТЕСТУ ДИКТОРА ===", flush=True)
        
        state = user_states.get(chat_id)
        if not state:
            print(f"[DICTOR] ❌ Стан не знайдено!", flush=True)
            return
        
        # Якщо параметри диктора вже збережені - використовуємо їх замість генерації нових
        # Це забезпечує консистентність голосу диктора між сесіями
        if "narrator" in character_voice_params and character_voice_params["narrator"]:
            print(f"[DICTOR] ✅ Використовуємо збережені параметри голосу диктора: seed={character_voice_params['narrator'].get('seed')}", flush=True)
        
        scenario = state["scenario"]
        voice_selections = state["voice_selections"]
        chars = state["chars"]
        
        # Зберігаємо вибрані голоси в сценарії
        for char in chars:
            char_name = char["name"]
            if char_name in voice_selections:
                char["voice_preset"] = voice_selections[char_name]
        
        # Оновлюємо сценарій
        scenario["characters"] = chars
        
        # Показуємо підсумок
        text = "✅ **Всі голоси персонажів вибрані!**\n\n"
        text += "📋 Підсумок:\n"
        for char in chars:
            preset_key = char.get("voice_preset") or voice_selections.get(char["name"], "default")
            preset = VOICE_PRESETS.get(preset_key, {})
            text += f"• {char['name']}: {preset.get('name', 'За замовчуванням')}\n"
        
        # Додаємо інформацію про диктора
        narrator = scenario.get("narrator", {})
        narrator_gender = narrator.get("gender", "male")
        text += f"\n📖 **Голос диктора:**\n"
        text += f"• Стать: {narrator_gender}\n"
        text += f"• Опис: {narrator.get('voice_description', 'Стандартний')}\n"
        
        text += "\n🔊 Тестування голосу диктора..."
        
        self.bot.send_message(chat_id, text, parse_mode="Markdown")
        
        # Визначаємо мову тексту
        scenes = scenario.get("scenes", [])
        text_lang = "russian"
        if scenes:
            for scene in scenes:
                narrative = scene.get("narrative", "")
                if narrative:
                    text_lang = detect_language(narrative)
                    break
                dialogues = scene.get("dialogues", [])
                if dialogues:
                    sample_text = dialogues[0].get("text", "")
                    if sample_text:
                        text_lang = detect_language(sample_text)
                        break
        
        print(f"[DICTOR] Мова: {text_lang}", flush=True)
        print(f"[DICTOR] Стать диктора: {narrator_gender}", flush=True)
        
        # Вибираємо пресет диктора за статтю
        if narrator_gender == "female":
            narrator_preset = "narrator_female"
        else:
            narrator_preset = "narrator_neutral"
        
        print(f"[DICTOR] Пресет: {narrator_preset}", flush=True)
        
        # Перевіряємо, що пресет існує
        if narrator_preset not in VOICE_PRESETS:
            print(f"[DICTOR] ⚠️ Пресет {narrator_preset} не знайдено! Використовуємо male_deep", flush=True)
            narrator_preset = "male_deep"
        
        demo_text = get_demo_text(text_lang, long=False)
        print(f"[DICTOR] Демо текст: {demo_text[:50]}...", flush=True)
        
        # Перевіряємо кеш
        cache_key = f"{narrator_preset}_{text_lang}"
        narrator_audio = test_voice_cache.get(cache_key)
        
        status = self.bot.send_message(chat_id, "⏳ Генерація тесту диктора...")
        
        if narrator_audio is None:
            print(f"[DICTOR] Генерація голосу диктора: {narrator_preset}", flush=True)
            
            # Перевіряємо, що модель завантажена
            if tts_model_voicedesign is None:
                print(f"[DICTOR] ❌ Модель VoiceDesign не завантажена!", flush=True)
                self.bot.edit_message_text("❌ Модель TTS не завантажена. Перевірте логи.", chat_id, status.message_id)
                return
            
            # ВАЖЛИВО: character_name="narrator" для консистентності seed + prompt
            # Це забезпечує збереження параметрів голосу під правильним ключем
            # 
            # КОНСИСТЕНТНІСТЬ ГОЛОСУ ДИКТОРА:
            # - НЕ використовуємо custom_seed при повторних спробах
            # - Seed береться з пресету (наприклад, 3001 для narrator_neutral)
            # - Параметри зберігаються в character_voice_params["narrator"]
            # - При генерації аудіо використовується збережений seed
            narrator_audio = generate_audio(
                demo_text,
                character_name="narrator",
                gender=narrator_gender,
                voice_preset=narrator_preset,
                lang=text_lang
            )
            
            print(f"[DICTOR] Результат першої спроби: {type(narrator_audio)}", flush=True)
            if narrator_audio is not None:
                print(f"[DICTOR] Довжина: {len(narrator_audio)} семплів ({len(narrator_audio)/SAMPLE_RATE:.1f} сек)", flush=True)
            
            # Якщо не вдалося - пробуємо ще раз БЕЗ зміни seed
            # (використовуємо той самий seed з пресету для консистентності)
            if narrator_audio is None or len(narrator_audio) < SAMPLE_RATE:
                print(f"[DICTOR] Повторна спроба для диктора (той самий seed)...", flush=True)
                narrator_audio = generate_audio(
                    demo_text,
                    character_name="narrator",
                    gender=narrator_gender,
                    voice_preset=narrator_preset,
                    lang=text_lang
                    # НЕ передаємо custom_seed - використовуємо seed з пресету
                )
                if narrator_audio is not None:
                    print(f"[DICTOR] Результат другої спроби: {len(narrator_audio)} семплів", flush=True)
            
            # Якщо все ще не вдалося - пробуємо ще раз
            # ВАЖЛИВО: НЕ змінюємо seed, щоб зберегти консистентність голосу
            if narrator_audio is None or len(narrator_audio) < SAMPLE_RATE:
                print(f"[DICTOR] Третя спроба для диктора (той самий seed)...", flush=True)
                narrator_audio = generate_audio(
                    demo_text,
                    character_name="narrator",
                    gender=narrator_gender,
                    voice_preset=narrator_preset,
                    lang=text_lang
                    # НЕ передаємо custom_seed - використовуємо seed з пресету
                )
                if narrator_audio is not None:
                    print(f"[DICTOR] Результат третьої спроби: {len(narrator_audio)} семплів", flush=True)
            
            if narrator_audio is not None and len(narrator_audio) >= SAMPLE_RATE:
                test_voice_cache[cache_key] = narrator_audio
                # ЗБЕРЕЖЕННЯ ПАРАМЕТРІВ ГОЛОСУ ДИКТОРА: seed + prompt для консистентності
                # Параметри вже збережені в character_voice_params через generate_audio
                print(f"[DICTOR] ✅ Збережено тест диктора: {cache_key}", flush=True)
            else:
                print(f"[DICTOR] ❌ Всі спроби невдалі!", flush=True)
        else:
            print(f"[DICTOR] ✅ Використано кешований голос диктора: {cache_key}", flush=True)
        
        # Перевіряємо чи аудіо валідне
        if narrator_audio is None or len(narrator_audio) < SAMPLE_RATE:
            # Пропускаємо тест диктора і починаємо генерацію
            self.bot.edit_message_text("⚠️ Тест диктора не вдалося. Починаємо генерацію...", chat_id, status.message_id)
            scenario["narrator"]["voice_preset"] = narrator_preset
            self._start_generation(chat_id, scenario)
            return
        
        # Конвертація
        max_val = np.max(np.abs(narrator_audio)) if len(narrator_audio) > 0 else 0
        if max_val > 0:
            audio_normalized = narrator_audio / max_val * 0.9
        else:
            audio_normalized = narrator_audio
        
        audio_int16 = (audio_normalized * 32767).astype(np.int16)
        segment = AudioSegment(
            audio_int16.tobytes(),
            frame_rate=SAMPLE_RATE,
            sample_width=2,
            channels=1
        )
        
        buf = BytesIO()
        segment.export(buf, format="mp3")
        buf.seek(0)
        
        # Створюємо інлайн-клавіатуру
        markup = tb_types.InlineKeyboardMarkup()
        markup.add(
            tb_types.InlineKeyboardButton("🚀 Почати генерацію", callback_data=f"approve_{chat_id}"),
            tb_types.InlineKeyboardButton("🔄 Змінити голоси", callback_data=f"change_{chat_id}")
        )
        
        duration_sec = len(narrator_audio) / SAMPLE_RATE
        self.bot.delete_message(chat_id, status.message_id)
        self.bot.send_audio(
            chat_id, 
            buf, 
            caption=f"📖 Тест голосу диктора\n\n🌐 Мова: {text_lang}\n⏱ Тривалість: {duration_sec:.1f} сек\n\n_{demo_text[:80]}..._",
            parse_mode="Markdown",
            reply_markup=markup
        )
        
        # Зберігаємо пресет диктора у сценарії
        scenario["narrator"]["voice_preset"] = narrator_preset
        
        # Зберігаємо стан - ГОТОВИЙ ДО ГЕНЕРАЦІЇ
        state["step"] = "ready_to_generate"
    
    def _start_generation(self, chat_id, scenario):
        """Початок генерації аудіо"""
        state = user_states.get(chat_id)
        if not state:
            return
        
        duration_est = state.get("duration_est", {})
        voice_selections = state.get("voice_selections", {})
        
        status = self.bot.send_message(chat_id, "🔊 Генерація аудіо...")
        
        self._generate_audio_impl(chat_id, scenario, status, voice_selections, duration_est)
    
    def _generate_audio(self, m, scenario, status, voice_selections):
        """Генерація аудіо (старий метод для сумісності)"""
        self._generate_audio_impl(m.chat.id, scenario, status, voice_selections, {})
    
    def _generate_audio_impl(self, chat_id, scenario, status, voice_selections, duration_est):
        """Реалізація генерації аудіо"""
        try:
            def update(text):
                try:
                    self.bot.edit_message_text(text, chat_id, status.message_id)
                except:
                    pass
            
            # Ініціалізуємо флаг зупинки та Event для контрольної точки
            if chat_id in user_states:
                user_states[chat_id]["generation_stopped"] = False
                user_states[chat_id]["checkpoint_event"] = threading.Event()
                user_states[chat_id]["checkpoint_approved"] = False
            else:
                user_states[chat_id] = {
                    "generation_stopped": False,
                    "checkpoint_event": threading.Event(),
                    "checkpoint_approved": False
                }
            
            def checkpoint_callback(audio_segment, current_time, total_items, current_item):
                """Callback для контрольної точки - БЛОКУЄ до натискання кнопки"""
                state = user_states.get(chat_id, {})
                
                # Перевіряємо флаг зупинки
                if state.get("generation_stopped", False):
                    return False
                
                try:
                    buf = BytesIO()
                    audio_segment.export(buf, format="mp3")
                    buf.seek(0)
                    
                    minutes = current_time / 60
                    progress = f"{current_item}/{total_items}"
                    
                    # Скидаємо Event перед відправкою
                    checkpoint_event = state.get("checkpoint_event")
                    if checkpoint_event:
                        checkpoint_event.clear()
                    state["checkpoint_approved"] = False
                    
                    markup = tb_types.InlineKeyboardMarkup()
                    markup.add(
                        tb_types.InlineKeyboardButton("✅ Продовжити", callback_data=f"continue_{chat_id}"),
                        tb_types.InlineKeyboardButton("⏹ Зупинити", callback_data=f"stop_{chat_id}")
                    )
                    
                    self.bot.send_audio(
                        chat_id, 
                        buf, 
                        caption=f"🔍 Контрольна точка: {minutes:.1f} хв\nПрогрес: {progress}\n\n⏳ Очікування підтвердження...",
                        reply_markup=markup
                    )
                    
                    print(f"[CHECKPOINT] ⏳ Очікування підтвердження від користувача...", flush=True)
                    
                    # === БЛОКУЄМО ДО НАТИСКАННЯ КНОПКИ ===
                    if checkpoint_event:
                        # Чекаємо до 5 хвилин
                        approved = checkpoint_event.wait(timeout=300)
                        
                        if not approved:
                            # Таймаут - зупиняємо генерацію
                            print("[CHECKPOINT] ⏱️ Таймаут очікування", flush=True)
                            return False
                        
                        # Перевіряємо чи схвалили
                        if state.get("generation_stopped", False):
                            print("[CHECKPOINT] ⏹ Зупинено користувачем", flush=True)
                            return False
                        
                        if state.get("checkpoint_approved", False):
                            print("[CHECKPOINT] ✅ Продовжуємо генерацію", flush=True)
                            return True
                        else:
                            print("[CHECKPOINT] ❌ Не схвалено", flush=True)
                            return False
                    
                    return True
                    
                except Exception as e:
                    print(f"[CHECKPOINT] ❌ Помилка: {e}", flush=True)
                    return True
            
            # Оновлюємо персонажів з вибраними голосами
            chars = scenario.get("characters", [])
            char_map = {}
            for c in chars:
                char_map[c["name"]] = {
                    "voice": c.get("voice_description", ""),
                    "gender": c.get("gender", "male"),
                    "voice_preset": voice_selections.get(c["name"])
                }
            
            # Генерація з урахуванням вибраних голосів
            audio, subs = self._build_audio_with_voices(scenario, char_map, update, checkpoint_callback)
            
            # Перевіряємо флаг зупинки
            state = user_states.get(chat_id, {})
            generation_stopped = state.get("generation_stopped", False)
            
            if not audio:
                if generation_stopped:
                    self.bot.edit_message_text("⏹ Генерацію зупинено користувачем", chat_id, status.message_id)
                else:
                    self.bot.edit_message_text("Помилка аудіо", chat_id, status.message_id)
                return
            
            # Збереження
            buf = BytesIO()
            audio.export(buf, format="mp3")
            buf.seek(0)
            
            srt = create_srt(subs)
            srt_buf = BytesIO(srt.encode("utf-8"))
            srt_buf.seek(0)
            
            title = scenario.get("title", "audiobook")
            
            self.bot.send_audio(chat_id, buf, caption=f"{title}\n{len(audio)/1000:.0f} сек")
            self.bot.send_document(chat_id, srt_buf, caption="Субтитри")
            
            self.bot.delete_message(chat_id, status.message_id)
            
            # Очищаємо стан
            if chat_id in user_states:
                del user_states[chat_id]
                
        except Exception as e:
            self.bot.send_message(chat_id, f"Помилка: {e}")
    
    def _build_audio_with_voices(self, scenario, char_map, status_cb, checkpoint_cb):
        """Збірка аудіо з урахуванням вибраних голосів
        
        Підтримує збереження та відновлення стану при досягненні лімітів Colab.
        """
        global character_voice_params
        print("[AUDIO] Збірка з голосами...", flush=True)
        
        checkpoint_times = [60, 1800]  # 1 хв, 30 хв
        checkpoints_passed = set()
        
        scenes = scenario.get("scenes", [])
        narrator = scenario.get("narrator", {})
        
        if not scenes:
            return None, None
        
        # Визначаємо мову тексту
        text_lang = "russian"
        for scene in scenes:
            narrative = scene.get("narrative", "")
            if narrative:
                text_lang = detect_language(narrative)
                break
            dialogues = scene.get("dialogues", [])
            if dialogues:
                sample_text = dialogues[0].get("text", "")
                if sample_text:
                    text_lang = detect_language(sample_text)
                    break
        
        print(f"[AUDIO] Мова тексту: {text_lang}", flush=True)
        
        # Наратор - використовуємо seed + prompt для консистентності голосу
        # Параметри голосу зберігаються в character_voice_params["narrator"]
        narrator_voice = narrator.get("voice_description", "Спокійний глибокий голос")
        narrator_gender = narrator.get("gender", "male")
        narrator_preset = narrator.get("voice_preset", "narrator_neutral")
        
        # Якщо пресет не встановлено, використовуємо narrator_neutral за замовчуванням
        if narrator_preset is None or narrator_preset not in VOICE_PRESETS:
            narrator_preset = "narrator_neutral" if narrator_gender == "male" else "narrator_female"
            print(f"[AUDIO] Диктор без пресета, використовуємо: {narrator_preset}", flush=True)
        
        # === ПЕРЕВІРКА ЗБЕРЕЖЕНОГО СТАНУ ===
        saved_state = load_generation_state()
        start_scene_idx = 0
        start_dial_idx = 0
        clips = []
        subtitles = []
        current_time = 0.0
        current = 0
        
        if saved_state and saved_state.get("scenario_title") == scenario.get("title"):
            print(f"[STATE] Знайдено збережений стан: сцена {saved_state.get('scene_idx', 0)}, елемент {saved_state.get('current', 0)}", flush=True)
            start_scene_idx = saved_state.get("scene_idx", 0)
            start_dial_idx = saved_state.get("dial_idx", 0)
            current = saved_state.get("current", 0)
            current_time = saved_state.get("current_time", 0.0)
            
            # Відновлюємо clips та subtitles
            saved_clips = saved_state.get("clips", [])
            if saved_clips:
                clips = [np.array(c) for c in saved_clips]
                print(f"[STATE] Відновлено {len(clips)} аудіо кліпів", flush=True)
            
            saved_subtitles = saved_state.get("subtitles", [])
            if saved_subtitles:
                subtitles = saved_subtitles.copy()
                print(f"[STATE] Відновлено {len(subtitles)} субтитрів", flush=True)
            
            # Відновлюємо character_voice_params
            saved_voice_params = saved_state.get("character_voice_params", {})
            if saved_voice_params:
                character_voice_params.update(saved_voice_params)
                print(f"[STATE] Відновлено voice params для {len(saved_voice_params)} персонажів", flush=True)
            
            # Відновлюємо character_voice_prompts (для клонування голосу)
            saved_voice_prompts = saved_state.get("character_voice_prompts", {})
            if saved_voice_prompts:
                character_voice_prompts.update(saved_voice_prompts)
                print(f"[STATE] Відновлено voice prompts для {len(saved_voice_prompts)} персонажів", flush=True)
        else:
            print("[STATE] Збережений стан не знайдено або не відповідає поточному сценарію", flush=True)
        
        # Підрахунок елементів
        total_items = 0
        for s in scenes:
            if s.get("narrative"):
                total_items += 1
            total_items += len(s.get("dialogues", []))
        
        for scene_idx, scene in enumerate(scenes):
            # ПРОПУСК ВЖЕ ОБРОБЛЕНИХ СЦЕН при відновленні
            if scene_idx < start_scene_idx:
                continue
            
            try:
                # 1. Наратив
                narrative = scene.get("narrative", "")
                if narrative:
                    current += 1
                    if status_cb:
                        status_cb(f"Наратив: {current}/{total_items}")
                    
                    print(f"[AUDIO] Наратив сцена {scene_idx + 1}", flush=True)
                    # Генерація голосу диктора з використанням seed + prompt для консистентності
                    # Параметри голосу зберігаються в character_voice_params
                    # custom_seed видалено - використовуємо збережений seed з character_voice_params
                    audio = generate_audio(
                        narrative, 
                        narrator_voice, 
                        character_name="narrator",
                        gender=narrator_gender,
                        voice_preset=narrator_preset,
                        lang=text_lang
                    )
                    if audio is not None:
                        clips.append(audio)
                        duration = len(audio) / SAMPLE_RATE
                        subtitles.append({
                            "index": current,
                            "start": current_time,
                            "end": current_time + duration,
                            "char": "Диктор",
                            "text": narrative
                        })
                        current_time += duration
                        
                        silence = np.zeros(int(SAMPLE_RATE * 0.3))
                        clips.append(silence)
                        current_time += 0.3
                        
                        # === ЗБЕРЕЖЕННЯ СТАНУ ПІСЛЯ НАРАТИВУ ===
                        save_generation_state({
                            'scenario_title': scenario.get("title"),
                            'scene_idx': scene_idx,
                            'dial_idx': 0,
                            'current': current,
                            'current_time': current_time,
                            'clips': [c.tolist() if isinstance(c, np.ndarray) else c for c in clips],
                            'subtitles': subtitles.copy(),
                            'character_voice_params': character_voice_params.copy(),
                            'character_voice_prompts': character_voice_prompts.copy(),
                            'current_step': f'narrative_scene_{scene_idx}'
                        })
                
                # 2. Діалоги
                for dial_idx, dial in enumerate(scene.get("dialogues", [])):
                    # ПРОПУСК ВЖЕ ОБРОБЛЕНИХ ДІАЛОГІВ при відновленні
                    if scene_idx == start_scene_idx and dial_idx < start_dial_idx:
                        continue
                    
                    current += 1
                    if status_cb:
                        status_cb(f"Озвучка: {current}/{total_items}")
                    
                    char = dial.get("character", "")
                    text = dial.get("text", "")
                    emotion = dial.get("emotion", "спокій")
                    
                    char_info = char_map.get(char, {"voice": "", "gender": "male"})
                    voice = char_info["voice"]
                    gender = char_info["gender"]
                    voice_preset = char_info.get("voice_preset")
                    
                    voice_with_emotion = f"{voice}. {emotion}"
                    
                    # Генерація голосу персонажа з використанням seed + prompt для консистентності
                    # Параметри голосу зберігаються в character_voice_params
                    print(f"[AUDIO] {char}: {text[:40]}...", flush=True)
                    
                    audio = generate_audio(
                        text, 
                        voice_with_emotion, 
                        character_name=char,
                        gender=gender,
                        voice_preset=voice_preset,
                        lang=text_lang
                    )
                    
                    if audio is not None:
                        clips.append(audio)
                        duration = len(audio) / SAMPLE_RATE
                        subtitles.append({
                            "index": current,
                            "start": current_time,
                            "end": current_time + duration,
                            "char": char,
                            "text": text
                        })
                        current_time += duration
                        
                        silence = np.zeros(int(SAMPLE_RATE * 0.5))
                        clips.append(silence)
                        current_time += 0.5
                        
                        # === ЗБЕРЕЖЕННЯ СТАНУ ПІСЛЯ ДІАЛОГУ ===
                        save_generation_state({
                            'scenario_title': scenario.get("title"),
                            'scene_idx': scene_idx,
                            'dial_idx': dial_idx + 1,
                            'current': current,
                            'current_time': current_time,
                            'clips': [c.tolist() if isinstance(c, np.ndarray) else c for c in clips],
                            'subtitles': subtitles.copy(),
                            'character_voice_params': character_voice_params.copy(),
                            'character_voice_prompts': character_voice_prompts.copy(),
                            'current_step': f'dialogue_scene_{scene_idx}_dial_{dial_idx}'
                        })
                    
                    # Контрольні точки
                    if checkpoint_cb:
                        for checkpoint_time in checkpoint_times:
                            if current_time >= checkpoint_time and checkpoint_time not in checkpoints_passed:
                                print(f"[CHECKPOINT] {checkpoint_time/60:.0f} хв", flush=True)
                                
                                if clips:
                                    checkpoint_audio = np.concatenate(clips)
                                    max_val = np.max(np.abs(checkpoint_audio))
                                    if max_val > 0:
                                        checkpoint_audio = checkpoint_audio / max_val * 0.9
                                    checkpoint_audio = (checkpoint_audio * 32767).astype(np.int16)
                                    
                                    checkpoint_segment = AudioSegment(
                                        checkpoint_audio.tobytes(),
                                        frame_rate=SAMPLE_RATE,
                                        sample_width=2,
                                        channels=1
                                    )
                                    
                                    should_continue = checkpoint_cb(checkpoint_segment, current_time, total_items, current)
                                    checkpoints_passed.add(checkpoint_time)
                                    
                                    if should_continue is False:
                                        print("[CHECKPOINT] Генерацію зупинено користувачем", flush=True)
                                        # Зберігаємо стан перед виходом
                                        save_generation_state({
                                            'scenario_title': scenario.get("title"),
                                            'scene_idx': scene_idx,
                                            'dial_idx': dial_idx + 1,
                                            'current': current,
                                            'current_time': current_time,
                                            'clips': [c.tolist() if isinstance(c, np.ndarray) else c for c in clips],
                                            'subtitles': subtitles.copy(),
                                            'character_voice_params': character_voice_params.copy(),
                                            'character_voice_prompts': character_voice_prompts.copy(),
                                            'current_step': f'checkpoint_stop_{checkpoint_time}',
                                            'stopped_by_user': True
                                        })
                                        return None, None
                
                # 3. Фонові звуки сцени - змішування з основним аудіо
                ambient_sounds = scene.get("ambient_sounds", [])
                if ambient_sounds and PEXELS_API_KEY:
                    print(f"[AUDIO] Фонові звуки: {ambient_sounds}", flush=True)
                    for sound_desc in ambient_sounds:
                        ambient = get_ambient_sound(sound_desc, duration_sec=max(5, current_time))
                        if ambient is not None and clips:
                            # Нормалізація фонового звуку
                            ambient_max = np.max(np.abs(ambient))
                            if ambient_max > 0:
                                ambient = ambient / ambient_max * 0.12  # 12% гучності для фону
                            
                            # Змішуємо з усіма кліпами сцени
                            all_clips = np.concatenate(clips) if clips else np.array([])
                            if len(all_clips) > 0:
                                mix_length = min(len(ambient), len(all_clips))
                                if mix_length > 0:
                                    # Створюємо змішане аудіо
                                    all_clips[:mix_length] += ambient[:mix_length]
                                    # Оновлюємо кліпи
                                    clips = [all_clips]
                                    print(f"[AUDIO] Змішано фоновий звук: {sound_desc}", flush=True)
                                    
            except Exception as e:
                error_msg = str(e).lower()
                # Перевірка на помилки лімітів Colab (GPU/TPU)
                if any(keyword in error_msg for keyword in ['resource exhausted', 'quota', 'limit', 'cuda out of memory', 'gpu memory', 'runtime error']):
                    print(f"[STATE] ⚠️ Досягнуто ліміт Colab: {e}", flush=True)
                    print(f"[STATE] Зберігаємо стан для відновлення...", flush=True)
                    
                    # Зберігаємо стан при помилці ліміту
                    save_generation_state({
                        'scenario_title': scenario.get("title"),
                        'scene_idx': scene_idx,
                        'dial_idx': 0,
                        'current': current,
                        'current_time': current_time,
                        'clips': [c.tolist() if isinstance(c, np.ndarray) else c for c in clips],
                        'subtitles': subtitles.copy(),
                        'character_voice_params': character_voice_params.copy(),
                        'character_voice_prompts': character_voice_prompts.copy(),
                        'current_step': f'colab_limit_scene_{scene_idx}',
                        'error': str(e),
                        'colab_limit_reached': True
                    })
                    
                    print("[STATE] ✅ Стан збережено. Запустіть знову для продовження.", flush=True)
                    raise
                else:
                    # Інші помилки - просто перкидаємо
                    raise
        
        if not clips:
            return None, None
        
        # Конкатенація
        final = np.concatenate(clips)
        
        # Нормалізація
        max_val = np.max(np.abs(final))
        if max_val > 0:
            final = final / max_val * 0.9
        
        # 16-bit PCM
        final = (final * 32767).astype(np.int16)
        
        segment = AudioSegment(
            final.tobytes(),
            frame_rate=SAMPLE_RATE,
            sample_width=2,
            channels=1
        )
        
        # === ОЧИЩЕННЯ СТАНУ ПІСЛЯ УСПІШНОГО ЗАВЕРШЕННЯ ===
        clear_generation_state()
        print("[STATE] ✅ Генерація завершена успішно, стан очищено", flush=True)
        
        return segment, subtitles
    
    def run(self):
        print("[BOT] Запуск...", flush=True)
        print("[BOT] Надішліть текст або файл боту!", flush=True)
        while True:
            try:
                self.bot.infinity_polling(timeout=10, long_polling_timeout=5)
            except Exception as e:
                print(f"[BOT] Помилка: {e}", flush=True)
                time.sleep(5)

# =============================================================================
# ГОЛОВНА ФУНКЦІЯ
# =============================================================================

def main():
    print("", flush=True)
    print("=" * 60, flush=True)
    print("  VIBEMODLY v69.0 - VOICE CLONE CONSISTENCY", flush=True)
    print(f"  Gemini: {WORKING_MODEL}", flush=True)
    print("  TTS: Qwen3-TTS 1.7B VoiceDesign + Base", flush=True)
    print(f"  Pexels: {'OK' if PEXELS_API_KEY else 'NOT FOUND'}", flush=True)
    print("=" * 60, flush=True)
    print("  Консистентність голосу:", flush=True)
    print("  • Перший виклик: VoiceDesign (seed + prompt)", flush=True)
    print("  • Наступні: VoiceClone з reference аудіо", flush=True)
    print("  • voice_prompt кешується для кожного персонажа", flush=True)
    print("=" * 60, flush=True)
    print("  Retry логіка:", flush=True)
    print("  • Автоматичне очікування при 429 помилці", flush=True)
    print("  • Fallback на інші моделі Gemini", flush=True)
    print("=" * 60, flush=True)
    print("", flush=True)
    
    # Монтування Google Drive
    mount_google_drive()
    load_voice_configs()
    
    bot = Bot(TELEGRAM_TOKEN)
    bot.run()

if __name__ == "__main__":
    main()
