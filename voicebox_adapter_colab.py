"""
Voicebox Adapter for Google Colab - Єдиний файл для легкого імпорту

Цей модуль надає абстракцію для роботи з різними TTS двигунами:
- VoiceDesign (seed + prompt) - створення унікального голосу
- VoiceClone (reference audio) - клонування для консистентності

АЛГОРИТМ КОНСИСТЕНТНОСТІ ГОЛОСУ:
1. Перший виклик для персонажа: VoiceDesign (seed + prompt) → зберігаємо аудіо як reference
2. Наступні виклики: VoiceClone з reference аудіо (якщо доступна модель Base)

Використання в Colab:
    # Скопіюйте цей файл в notebook або завантажте як .py файл
    from voicebox_adapter_colab import TTSEngineManager, VoiceProfileManager
    
    # Ініціалізація
    tts_manager = TTSEngineManager()
    profile_manager = VoiceProfileManager()
    
    # Отримання двигуна для профілю
    profile = profile_manager.get_profile("narrator")
    engine = tts_manager.get_engine(profile.mode)
    
    # Генерація аудіо
    audio, sample_rate = engine.generate("Текст", profile.to_engine_config())

Версія: 1.1.0
Автор: VIBEMODLY Team
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any, List, Union
from pathlib import Path
from enum import Enum
import numpy as np
import hashlib
import json
import asyncio
import os


# =============================================================================
# VOICE MODE
# =============================================================================

class VoiceMode(Enum):
    """
    Режим генерації голосу.
    
    VOICE_DESIGN: Створення голосу через seed + prompt (VIBEMODLY підхід)
    VOICE_CLONE: Клонування голосу з reference audio (Voicebox підхід)
    """
    VOICE_DESIGN = "voicedesign"
    VOICE_CLONE = "clone"
    
    @classmethod
    def from_string(cls, value: str) -> 'VoiceMode':
        """
        Конвертація рядка у VoiceMode.
        
        Args:
            value: Рядкове значення ('voicedesign', 'clone')
            
        Returns:
            VoiceMode: Відповідний режим
            
        Raises:
            ValueError: Якщо значення невідоме
        """
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

def validate_reference_audio(
    audio_path: str,
    min_duration: float = 2.0,
    max_duration: float = 30.0,
    min_rms: float = 0.01
) -> Tuple[bool, Optional[str]]:
    """
    Валідація reference аудіо для клонування голосу.
    
    Перевіряє:
    - Існування файлу
    - Тривалість (2-30 секунд)
    - Рівень гучності (не занадто тихий)
    - Відсутність кліпінгу
    
    Args:
        audio_path: Шлях до аудіо файлу
        min_duration: Мінімальна тривалість у секундах (за замовчуванням 2.0)
        max_duration: Максимальна тривалість у секундах (за замовчуванням 30.0)
        min_rms: Мінімальний RMS рівень (за замовчуванням 0.01)
        
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
        
    Example:
        >>> is_valid, error = validate_reference_audio("voice_sample.wav")
        >>> if not is_valid:
        ...     print(f"Помилка: {error}")
    """
    try:
        # Перевірка існування файлу
        if not os.path.exists(audio_path):
            return False, f"Файл не знайдено: {audio_path}"
        
        # Завантаження аудіо
        audio, sr = load_audio_for_voice_clone(audio_path)
        
        # Розрахунок тривалості
        duration = len(audio) / sr
        
        if duration < min_duration:
            return False, f"Аудіо занадто коротке ({duration:.1f}с). Мінімум: {min_duration}с"
        
        if duration > max_duration:
            return False, f"Аудіо занадто довге ({duration:.1f}с). Максимум: {max_duration}с"
        
        # Перевірка рівня гучності
        rms = np.sqrt(np.mean(audio**2))
        if rms < min_rms:
            return False, f"Аудіо занадто тихе (RMS: {rms:.4f}). Мінімум: {min_rms}"
        
        # Перевірка на кліпінг
        if np.abs(audio).max() > 0.99:
            return False, "Аудіо має кліпінг (перевищення амплітуди). Зменшіть гучність"
        
        return True, None
        
    except Exception as e:
        return False, f"Помилка валідації аудіо: {str(e)}"


def load_audio_for_voice_clone(
    audio_path: str,
    sample_rate: int = 24000,
    mono: bool = True,
    normalize: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Завантаження та підготовка аудіо для клонування голосу.
    
    Виконує:
    - Завантаження аудіо файлу
    - Конвертацію в моно (опціонально)
    - Ресемплинг до цільової частоти
    - Нормалізацію (опціонально)
    
    Args:
        audio_path: Шлях до аудіо файлу
        sample_rate: Цільова частота дискретизації (за замовчуванням 24000 Hz)
        mono: Конвертувати в моно (за замовчуванням True)
        normalize: Нормалізувати аудіо (за замовчуванням True)
        
    Returns:
        Tuple[np.ndarray, int]: (аудіо масив, sample_rate)
        
    Example:
        >>> audio, sr = load_audio_for_voice_clone("voice.wav")
        >>> print(f"Завантажено {len(audio)/sr:.1f} секунд")
    """
    try:
        import librosa
        
        # Завантаження з librosa (підтримує багато форматів)
        audio, sr = librosa.load(audio_path, sr=sample_rate, mono=mono)
        
        # Нормалізація якщо потрібно
        if normalize:
            audio = _normalize_audio(audio)
        
        return audio.astype(np.float32), sr
        
    except ImportError:
        # Fallback на soundfile якщо librosa недоступний
        try:
            import soundfile as sf
            
            audio, sr = sf.read(audio_path)
            
            # Конвертація в моно
            if mono and len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            # Ресемплинг якщо потрібно
            if sr != sample_rate:
                import scipy.signal
                audio = scipy.signal.resample_poly(audio, sample_rate, sr)
                sr = sample_rate
            
            if normalize:
                audio = _normalize_audio(audio)
            
            return audio.astype(np.float32), sr
            
        except ImportError:
            raise ImportError(
                "Потрібно встановити librosa або soundfile: "
                "pip install librosa soundfile"
            )


def _normalize_audio(
    audio: np.ndarray,
    target_db: float = -20.0,
    peak_limit: float = 0.85
) -> np.ndarray:
    """
    Нормалізація аудіо до цільового рівня гучності.
    
    Args:
        audio: Вхідний аудіо масив
        target_db: Цільовий RMS рівень в dB
        peak_limit: Пікове обмеження (0.0-1.0)
        
    Returns:
        np.ndarray: Нормалізований аудіо масив
    """
    audio = audio.astype(np.float32)
    
    # Розрахунок поточного RMS
    rms = np.sqrt(np.mean(audio**2))
    
    # Розрахунок цільового RMS
    target_rms = 10**(target_db / 20)
    
    # Застосування gain
    if rms > 0:
        gain = target_rms / rms
        audio = audio * gain
    
    # Пікове обмеження
    audio = np.clip(audio, -peak_limit, peak_limit)
    
    return audio


def get_cache_key(audio_path: str, reference_text: str) -> str:
    """
    Генерація ключа кешу на основі аудіо файлу та тексту.
    
    Args:
        audio_path: Шлях до аудіо файлу
        reference_text: Reference текст
        
    Returns:
        str: MD5 хеш як ключ кешу
    """
    # Читання аудіо файлу
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    
    # Комбінування байтів аудіо та тексту
    combined = audio_bytes + reference_text.encode("utf-8")
    
    # Генерація хешу
    return hashlib.md5(combined).hexdigest()


# =============================================================================
# VOICE PROMPT CACHE
# =============================================================================

class VoicePromptCache:
    """
    Кешування voice prompts для пришвидшення генерації.
    
    Підтримує двохрівневе кешування:
    1. Memory cache - швидкий доступ до часто використовуваних prompts
    2. Disk cache - збереження між сесіями
    
    Використання:
        cache = VoicePromptCache("/content/cache")
        
        # Отримання з кешу
        cached = cache.get("some_key")
        
        # Збереження в кеш
        cache.set("some_key", audio_data)
        
        # Очищення кешу
        cache.clear()
    """
    
    def __init__(
        self, 
        cache_dir: str = "/content/cache",
        max_memory_items: int = 100
    ):
        """
        Ініціалізація кешу.
        
        Args:
            cache_dir: Директорія для disk кешу
            max_memory_items: Максимальна кількість елементів в memory кеші
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._memory_cache: Dict[str, Any] = {}
        self._max_memory_items = max_memory_items
        
        # Статистика
        self._hits = 0
        self._misses = 0
        
        print(f"[VoicePromptCache] Ініціалізовано: {cache_dir}", flush=True)
    
    def get_cache_key(self, *args) -> str:
        """
        Генерація ключа кешу з аргументів.
        
        Args:
            *args: Аргументи для генерації ключа
            
        Returns:
            str: MD5 хеш ключа
        """
        # Конвертація аргументів у рядок
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
        """
        Отримання з кешу (memory + disk).
        
        Args:
            key: Ключ кешу
            
        Returns:
            Any: Збережені дані або None якщо не знайдено
        """
        # Спочатку перевіряємо memory cache
        if key in self._memory_cache:
            self._hits += 1
            print(f"[CACHE] Memory hit: {key[:8]}...", flush=True)
            return self._memory_cache[key]
        
        # Потім disk cache
        cache_file = self.cache_dir / f"{key}.pt"
        if cache_file.exists():
            try:
                import torch
                data = torch.load(cache_file, map_location='cpu')
                
                # Збереження в memory cache для наступних звернень
                self._set_memory_cache(key, data)
                
                self._hits += 1
                print(f"[CACHE] Disk hit: {key[:8]}...", flush=True)
                return data
                
            except Exception as e:
                print(f"[CACHE] Помилка читання disk cache: {e}", flush=True)
        
        self._misses += 1
        return None
    
    def set(self, key: str, value: Any, save_to_disk: bool = True) -> None:
        """
        Збереження в кеш.
        
        Args:
            key: Ключ кешу
            value: Дані для збереження
            save_to_disk: Чи зберігати на диск
        """
        # Збереження в memory cache
        self._set_memory_cache(key, value)
        
        # Збереження на диск
        if save_to_disk:
            try:
                import torch
                
                cache_file = self.cache_dir / f"{key}.pt"
                
                # Конвертація numpy в tensor якщо потрібно
                if hasattr(value, 'numpy'):  # torch tensor
                    torch.save(value, cache_file)
                elif hasattr(value, '__array__'):  # numpy array
                    tensor = torch.from_numpy(value) if hasattr(value, 'dtype') else torch.tensor(value)
                    torch.save(tensor, cache_file)
                else:
                    torch.save(value, cache_file)
                
                print(f"[CACHE] Збережено на диск: {key[:8]}...", flush=True)
                
            except Exception as e:
                print(f"[CACHE] Помилка збереження на диск: {e}", flush=True)
    
    def _set_memory_cache(self, key: str, value: Any) -> None:
        """
        Збереження в memory cache з лімітом розміру.
        
        Args:
            key: Ключ кешу
            value: Дані для збереження
        """
        # Якщо перевищено ліміт - видаляємо найстаріші елементи
        if len(self._memory_cache) >= self._max_memory_items:
            # Видаляємо половину елементів (FIFO)
            keys_to_remove = list(self._memory_cache.keys())[:self._max_memory_items // 2]
            for k in keys_to_remove:
                del self._memory_cache[k]
        
        self._memory_cache[key] = value
    
    def has(self, key: str) -> bool:
        """
        Перевірка наявності в кеші.
        
        Args:
            key: Ключ кешу
            
        Returns:
            bool: True якщо є в кеші
        """
        return key in self._memory_cache or (self.cache_dir / f"{key}.pt").exists()
    
    def delete(self, key: str) -> bool:
        """
        Видалення з кешу.
        
        Args:
            key: Ключ кешу
            
        Returns:
            bool: True якщо видалено
        """
        deleted = False
        
        # Видалення з memory cache
        if key in self._memory_cache:
            del self._memory_cache[key]
            deleted = True
        
        # Видалення з disk cache
        cache_file = self.cache_dir / f"{key}.pt"
        if cache_file.exists():
            try:
                cache_file.unlink()
                deleted = True
            except Exception as e:
                print(f"[CACHE] Помилка видалення файлу: {e}", flush=True)
        
        return deleted
    
    def clear_memory(self) -> None:
        """Очищення memory cache."""
        self._memory_cache.clear()
        print("[CACHE] Memory cache очищено", flush=True)
    
    def clear_disk(self) -> None:
        """Очищення disk cache."""
        try:
            for f in self.cache_dir.glob("*.pt"):
                f.unlink()
            print("[CACHE] Disk cache очищено", flush=True)
        except Exception as e:
            print(f"[CACHE] Помилка очищення disk cache: {e}", flush=True)
    
    def clear(self) -> None:
        """Очищення всього кешу (memory + disk)."""
        self.clear_memory()
        self.clear_disk()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Отримання статистики кешу.
        
        Returns:
            Dict: Статистика з ключами hits, misses, hit_rate, memory_items, disk_items
        """
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0
        
        disk_items = len(list(self.cache_dir.glob("*.pt")))
        
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 2),
            "memory_items": len(self._memory_cache),
            "disk_items": disk_items,
            "max_memory_items": self._max_memory_items
        }
    
    def get_cache_size(self) -> Dict[str, int]:
        """
        Отримання розміру кешу.
        
        Returns:
            Dict: Розміри memory та disk кешу в байтах
        """
        # Розмір disk cache
        disk_size = 0
        for f in self.cache_dir.glob("*.pt"):
            disk_size += f.stat().st_size
        
        # Розмір memory cache (приблизний)
        memory_size = 0
        for key, value in self._memory_cache.items():
            try:
                if hasattr(value, 'nbytes'):
                    memory_size += value.nbytes
                elif hasattr(value, 'element_size') and hasattr(value, 'nelement'):
                    memory_size += value.element_size() * value.nelement()
                else:
                    memory_size += len(str(value))
            except:
                pass
        
        return {
            "memory_bytes": memory_size,
            "disk_bytes": disk_size,
            "memory_mb": round(memory_size / (1024 * 1024), 2),
            "disk_mb": round(disk_size / (1024 * 1024), 2)
        }
    
    def save_audio_cache(
        self, 
        text: str, 
        voice_config: Dict[str, Any], 
        audio_data: Any
    ) -> str:
        """
        Збереження аудіо в кеш з автоматичним ключем.
        
        Args:
            text: Текст
            voice_config: Конфігурація голосу
            audio_data: Аудіо дані
            
        Returns:
            str: Ключ кешу
        """
        key = self.get_cache_key(text, voice_config)
        self.set(key, audio_data)
        return key
    
    def get_audio_cache(
        self, 
        text: str, 
        voice_config: Dict[str, Any]
    ) -> Optional[Any]:
        """
        Отримання аудіо з кешу з автоматичним ключем.
        
        Args:
            text: Текст
            voice_config: Конфігурація голосу
            
        Returns:
            Any: Аудіо дані або None
        """
        key = self.get_cache_key(text, voice_config)
        return self.get(key)


# =============================================================================
# VOICE PROFILE
# =============================================================================

@dataclass
class VoiceProfile:
    """
    Уніфікований голосовий профіль.
    
    Підтримує два режими:
    1. VoiceDesign: seed + prompt для створення унікального голосу
    2. VoiceClone: reference audio для клонування існуючого голосу
    
    Attributes:
        id: Унікальний ідентифікатор профілю
        name: Назва профілю
        mode: Режим генерації (VOICE_DESIGN або VOICE_CLONE)
        seed: Seed для VoiceDesign режиму
        prompt: Текстовий опис голосу для VoiceDesign
        reference_audio_path: Шлях до reference аудіо для VoiceClone
        reference_text: Текст reference аудіо для VoiceClone
        gender: Стать (male, female, neutral)
        language: Мова за замовчуванням
        speed: Швидкість мовлення
        pitch: Висота голосу (low, medium, high)
        description: Опис профілю
        tags: Теги для категоризації
    """
    id: str
    name: str
    mode: VoiceMode = VoiceMode.VOICE_DESIGN
    
    # VoiceDesign параметри
    seed: Optional[int] = None
    prompt: Optional[str] = None
    
    # Voice Clone параметри
    reference_audio_path: Optional[str] = None
    reference_text: Optional[str] = None
    
    # Загальні параметри
    gender: str = "male"
    language: str = "russian"
    speed: float = 1.0
    pitch: str = "medium"
    
    # Метадані
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Ініціалізація після створення."""
        # Якщо seed не вказано, генеруємо з id
        if self.seed is None and self.mode == VoiceMode.VOICE_DESIGN:
            self.seed = self._generate_seed_from_id()
    
    def _generate_seed_from_id(self) -> int:
        """Генерація стабільного seed на основі id."""
        hash_obj = hashlib.md5(self.id.encode('utf-8'))
        return int(hash_obj.hexdigest(), 16) % (2**32)
    
    def to_engine_config(self) -> Dict[str, Any]:
        """
        Конвертація у конфігурацію для TTS Engine.
        
        Returns:
            Dict[str, Any]: Конфігурація для передачі в TTSEngine.generate()
        """
        config = {
            "mode": self.mode.value,
            "speed": self.speed,
            "language": self.language
        }
        
        if self.mode == VoiceMode.VOICE_DESIGN:
            config["seed"] = self.seed
            config["prompt"] = self.prompt or self._build_default_prompt()
        else:  # VOICE_CLONE
            config["reference_audio"] = self.reference_audio_path
            config["reference_text"] = self.reference_text
        
        return config
    
    def _build_default_prompt(self) -> str:
        """
        Побудова prompt за замовчуванням на основі параметрів.
        
        Returns:
            str: Текстовий опис голосу
        """
        gender_prefix = "Male voice" if self.gender == "male" else "Female voice"
        pitch_desc = {
            "low": "low-pitched",
            "medium": "medium-pitched", 
            "high": "high-pitched"
        }
        pitch_str = pitch_desc.get(self.pitch, "medium-pitched")
        
        return f"{gender_prefix}, {pitch_str}. {self.description}"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Серіалізація профілю у словник.
        
        Returns:
            Dict[str, Any]: Словник з даними профілю
        """
        return {
            "id": self.id,
            "name": self.name,
            "mode": self.mode.value,
            "seed": self.seed,
            "prompt": self.prompt,
            "reference_audio_path": self.reference_audio_path,
            "reference_text": self.reference_text,
            "gender": self.gender,
            "language": self.language,
            "speed": self.speed,
            "pitch": self.pitch,
            "description": self.description,
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VoiceProfile':
        """
        Десеріалізація профілю зі словника.
        
        Args:
            data: Словник з даними профілю
            
        Returns:
            VoiceProfile: Екземпляр профілю
        """
        mode = VoiceMode.from_string(data.get("mode", "voicedesign"))
        
        return cls(
            id=data["id"],
            name=data["name"],
            mode=mode,
            seed=data.get("seed"),
            prompt=data.get("prompt"),
            reference_audio_path=data.get("reference_audio_path"),
            reference_text=data.get("reference_text"),
            gender=data.get("gender", "male"),
            language=data.get("language", "russian"),
            speed=data.get("speed", 1.0),
            pitch=data.get("pitch", "medium"),
            description=data.get("description", ""),
            tags=data.get("tags", [])
        )
    
    @classmethod
    def from_preset(cls, key: str, preset: Dict[str, Any]) -> 'VoiceProfile':
        """
        Створення профілю з пресету VIBEMODLY.
        
        Args:
            key: Ключ пресету
            preset: Словник з даними пресету (формат VOICE_PRESETS)
            
        Returns:
            VoiceProfile: Екземпляр профілю
        """
        # Формування prompt з пресету
        gender = preset.get("gender", "male")
        pitch = preset.get("pitch", "medium")
        prompt_suffix = preset.get("prompt_suffix", "")
        
        gender_prefix = "Male voice" if gender == "male" else "Female voice"
        pitch_desc = {"low": "low-pitched", "medium": "medium-pitched", "high": "high-pitched"}
        prompt = f"{gender_prefix}, {pitch_desc.get(pitch, 'medium')}. {prompt_suffix}"
        
        return cls(
            id=key,
            name=preset.get("name", key),
            mode=VoiceMode.VOICE_DESIGN,
            seed=preset.get("seed"),
            prompt=prompt,
            gender=gender,
            language="russian",
            speed=preset.get("speed", 1.0),
            pitch=pitch,
            description=preset.get("description", ""),
            tags=[gender, pitch]
        )


# =============================================================================
# VOICE PROFILE MANAGER
# =============================================================================

class VoiceProfileManager:
    """
    Менеджер для управління голосовими профілями.
    
    Інтегрується з існуючою системою VOICE_PRESETS з VIBEMODLY
    та надає розширені можливості для управління профілями.
    """
    
    def __init__(self, voice_presets: Optional[Dict[str, Dict]] = None):
        """
        Ініціалізація менеджера.
        
        Args:
            voice_presets: Словник з пресетами (формат VOICE_PRESETS)
        """
        self._profiles: Dict[str, VoiceProfile] = {}
        self._presets: Dict[str, Dict] = voice_presets or {}
        
        # Імпортуємо VOICE_PRESETS з VIBEMODLY якщо не передано
        if not self._presets:
            try:
                # Спроба імпорту з глобального контексту
                import builtins
                if hasattr(builtins, 'VOICE_PRESETS'):
                    self._presets = builtins.VOICE_PRESETS
            except:
                pass
    
    def set_presets(self, voice_presets: Dict[str, Dict]) -> None:
        """
        Встановлення словника пресетів.
        
        Args:
            voice_presets: Словник з пресетами
        """
        self._presets = voice_presets
    
    def get_profile(
        self, 
        character_name: str, 
        voice_preset: Optional[str] = None
    ) -> VoiceProfile:
        """
        Отримання профілю для персонажа.
        
        Args:
            character_name: Ім'я персонажа
            voice_preset: Ключ пресету (опціонально)
            
        Returns:
            VoiceProfile: Профіль голосу
        """
        # Якщо вказано пресет і він існує
        if voice_preset and voice_preset in self._presets:
            preset = self._presets[voice_preset]
            return VoiceProfile.from_preset(voice_preset, preset)
        
        # Якщо профіль вже створено для цього персонажа
        profile_key = f"char_{character_name}"
        if profile_key in self._profiles:
            return self._profiles[profile_key]
        
        # Створення нового профілю за замовчуванням
        default_profile = VoiceProfile(
            id=profile_key,
            name=character_name,
            mode=VoiceMode.VOICE_DESIGN,
            gender="male",
            language="russian"
        )
        
        self._profiles[profile_key] = default_profile
        return default_profile
    
    def create_profile(self, profile: VoiceProfile) -> None:
        """
        Створення нового профілю.
        
        Args:
            profile: Екземпляр VoiceProfile
        """
        self._profiles[profile.id] = profile
        print(f"[VoiceProfileManager] Створено профіль: {profile.name}", flush=True)
    
    def update_profile(self, profile: VoiceProfile) -> None:
        """
        Оновлення існуючого профілю.
        
        Args:
            profile: Екземпляр VoiceProfile
        """
        self._profiles[profile.id] = profile
        print(f"[VoiceProfileManager] Оновлено профіль: {profile.name}", flush=True)
    
    def delete_profile(self, profile_id: str) -> bool:
        """
        Видалення профілю.
        
        Args:
            profile_id: ID профілю
            
        Returns:
            bool: True якщо видалено успішно
        """
        if profile_id in self._profiles:
            del self._profiles[profile_id]
            print(f"[VoiceProfileManager] Видалено профіль: {profile_id}", flush=True)
            return True
        return False
    
    def get_all_profiles(self) -> List[VoiceProfile]:
        """
        Отримання всіх профілів.
        
        Returns:
            List[VoiceProfile]: Список всіх профілів
        """
        return list(self._profiles.values())
    
    def get_profiles_by_gender(self, gender: str) -> List[VoiceProfile]:
        """
        Отримання профілів за статтю.
        
        Args:
            gender: Стать (male, female)
            
        Returns:
            List[VoiceProfile]: Список профілів
        """
        return [p for p in self._profiles.values() if p.gender == gender]
    
    def get_profiles_by_mode(self, mode: VoiceMode) -> List[VoiceProfile]:
        """
        Отримання профілів за режимом.
        
        Args:
            mode: Режим генерації
            
        Returns:
            List[VoiceProfile]: Список профілів
        """
        return [p for p in self._profiles.values() if p.mode == mode]
    
    def get_available_presets(self, gender: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Отримання списку доступних пресетів.
        
        Args:
            gender: Фільтр за статтю (опціонально)
            
        Returns:
            List[Dict]: Список пресетів з ключами key, name, description, gender
        """
        presets = []
        for key, preset in self._presets.items():
            if gender is None or preset.get("gender") == gender:
                presets.append({
                    "key": key,
                    "name": preset.get("name", key),
                    "description": preset.get("description", ""),
                    "gender": preset.get("gender", "male")
                })
        return presets
    
    def save_profiles(self, filepath: str) -> bool:
        """
        Збереження профілів у файл.
        
        Args:
            filepath: Шлях до файлу
            
        Returns:
            bool: True якщо збережено успішно
        """
        try:
            data = {
                "profiles": [p.to_dict() for p in self._profiles.values()],
                "presets": self._presets
            }
            
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"[VoiceProfileManager] Збережено {len(self._profiles)} профілів", flush=True)
            return True
            
        except Exception as e:
            print(f"[VoiceProfileManager] Помилка збереження: {e}", flush=True)
            return False
    
    def load_profiles(self, filepath: str) -> bool:
        """
        Завантаження профілів з файлу.
        
        Args:
            filepath: Шлях до файлу
            
        Returns:
            bool: True якщо завантажено успішно
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Завантаження профілів
            for profile_data in data.get("profiles", []):
                profile = VoiceProfile.from_dict(profile_data)
                self._profiles[profile.id] = profile
            
            # Завантаження пресетів
            if "presets" in data:
                self._presets.update(data["presets"])
            
            print(f"[VoiceProfileManager] Завантажено {len(self._profiles)} профілів", flush=True)
            return True
            
        except FileNotFoundError:
            print(f"[VoiceProfileManager] Файл не знайдено: {filepath}", flush=True)
            return False
        except Exception as e:
            print(f"[VoiceProfileManager] Помилка завантаження: {e}", flush=True)
            return False
    
    def create_clone_profile(
        self,
        profile_id: str,
        name: str,
        reference_audio_path: str,
        reference_text: str,
        gender: str = "male",
        language: str = "russian",
        speed: float = 1.0
    ) -> VoiceProfile:
        """
        Створення профілю для клонування голосу.
        
        Args:
            profile_id: Унікальний ID профілю
            name: Назва профілю
            reference_audio_path: Шлях до reference аудіо
            reference_text: Текст reference аудіо
            gender: Стать
            language: Мова
            speed: Швидкість
            
        Returns:
            VoiceProfile: Створений профіль
        """
        profile = VoiceProfile(
            id=profile_id,
            name=name,
            mode=VoiceMode.VOICE_CLONE,
            reference_audio_path=reference_audio_path,
            reference_text=reference_text,
            gender=gender,
            language=language,
            speed=speed,
            description=f"Клонований голос: {name}"
        )
        
        self._profiles[profile_id] = profile
        print(f"[VoiceProfileManager] Створено clone профіль: {name}", flush=True)
        
        return profile


# =============================================================================
# TTS ENGINE (ABSTRACT)
# =============================================================================

class TTSEngine(ABC):
    """
    Абстрактний інтерфейс TTS двигуна.
    
    Всі TTS двигуни повинні успадковувати цей клас та реалізувати
    методи generate() та is_loaded().
    """
    
    @abstractmethod
    def generate(
        self, 
        text: str, 
        voice_config: Dict[str, Any],
        language: str = "russian"
    ) -> Tuple[Optional[np.ndarray], int]:
        """
        Генерація аудіо з голосовою конфігурацією.
        
        Args:
            text: Текст для озвучення
            voice_config: Конфігурація голосу (залежить від типу двигуна)
            language: Мова (russian, ukrainian)
            
        Returns:
            Tuple[np.ndarray, int]: (аудіо масив, sample_rate) або (None, 0) при помилці
        """
        pass
    
    @abstractmethod
    def is_loaded(self) -> bool:
        """
        Перевірка чи завантажено модель.
        
        Returns:
            bool: True якщо модель готова до роботи
        """
        pass
    
    @property
    @abstractmethod
    def engine_type(self) -> str:
        """
        Тип двигуна.
        
        Returns:
            str: 'voicedesign' або 'clone'
        """
        pass


# =============================================================================
# VOICE DESIGN ENGINE
# =============================================================================

class VoiceDesignEngine(TTSEngine):
    """
    VoiceDesign двигун - створення унікального голосу через prompt + seed.
    
    Це поточний підхід VIBEMODLY, який використовує модель Qwen3-TTS VoiceDesign
    для створення голосів на основі текстового опису та фіксованого seed.
    
    voice_config має містити:
        - seed: int - фіксований seed для детермінізму
        - prompt: str - текстовий опис голосу
        - speed: float - швидкість мовлення (опціонально)
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"):
        """
        Ініціалізація VoiceDesign двигуна.
        
        Args:
            model_name: Назва моделі HuggingFace
        """
        self._model = None
        self._model_name = model_name
        self._sample_rate = 24000
    
    def load_model(self, device: str = "cuda") -> bool:
        """
        Завантаження моделі.
        
        Args:
            device: Пристрій (cuda, cpu)
            
        Returns:
            bool: True якщо завантаження успішне
        """
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
        """
        Встановлення вже завантаженої моделі.
        
        Використовується для інтеграції з існуючим кодом VIBEMODLY,
        де модель завантажується глобально.
        
        Args:
            model: Екземпляр Qwen3TTSModel
        """
        self._model = model
    
    def is_loaded(self) -> bool:
        return self._model is not None
    
    @property
    def engine_type(self) -> str:
        return "voicedesign"
    
    def generate(
        self, 
        text: str, 
        voice_config: Dict[str, Any],
        language: str = "russian"
    ) -> Tuple[Optional[np.ndarray], int]:
        """
        Генерація аудіо через VoiceDesign.
        
        Args:
            text: Текст для озвучення
            voice_config: Конфігурація з ключами:
                - seed: int - seed для детермінізму
                - prompt: str - опис голосу
                - speed: float - швидкість (опціонально, за замовчуванням 1.0)
            language: Мова (russian, ukrainian)
            
        Returns:
            Tuple[np.ndarray, int]: (аудіо, sample_rate) або (None, 0)
        """
        if not self.is_loaded():
            print("[VoiceDesign] ❌ Модель не завантажена!", flush=True)
            return None, 0
        
        try:
            import torch
            import random
            
            # Отримання параметрів
            seed = voice_config.get("seed", 42)
            prompt = voice_config.get("prompt", "A natural voice")
            speed = voice_config.get("speed", 1.0)
            
            print(f"[VoiceDesign] Seed: {seed}", flush=True)
            print(f"[VoiceDesign] Prompt: {prompt[:60]}...", flush=True)
            
            # Встановлення seed для детермінізму
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            
            # Генерація через модель
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
            
            # Обробка результату
            if isinstance(result, tuple):
                audio, sr = result
            else:
                audio = result
                sr = self._sample_rate
            
            # Конвертація в numpy
            if hasattr(audio, 'cpu'):
                audio = audio.cpu().numpy()
            elif hasattr(audio, 'numpy'):
                audio = audio.numpy()
            
            audio = np.array(audio).flatten()
            
            # Зміна швидкості якщо потрібно
            if speed != 1.0:
                audio = self._change_speed(audio, speed)
            
            print(f"[VoiceDesign] ✅ Згенеровано: {len(audio)} семплів", flush=True)
            
            return audio, sr
            
        except Exception as e:
            print(f"[VoiceDesign] ❌ Помилка генерації: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return None, 0
    
    def _change_speed(self, audio: np.ndarray, speed: float) -> np.ndarray:
        """
        Зміна швидкості аудіо.
        
        Args:
            audio: Аудіо масив
            speed: Коефіцієнт швидкості
            
        Returns:
            np.ndarray: Аудіо зі зміненою швидкістю
        """
        try:
            from pydub import AudioSegment
            
            # Конвертація в AudioSegment
            audio_int16 = (audio * 32767).astype(np.int16)
            segment = AudioSegment(
                audio_int16.tobytes(),
                frame_rate=self._sample_rate,
                sample_width=2,
                channels=1
            )
            
            # Зміна швидкості
            if speed > 1.0:
                segment = segment.speedup(playback_speed=speed)
            elif speed < 1.0:
                segment = segment.speedup(playback_speed=speed)
            
            # Конвертація назад
            samples = np.array(segment.get_array_of_samples())
            return samples.astype(np.float32) / 32767.0
            
        except Exception as e:
            print(f"[VoiceDesign] Помилка зміни швидкості: {e}", flush=True)
            return audio


# =============================================================================
# VOICE CLONE ENGINE
# =============================================================================

class VoiceCloneEngine(TTSEngine):
    """
    Voice Clone двигун - клонування голосу з reference audio.
    
    Це підхід з Voicebox, який використовує модель Qwen3-TTS Base
    для клонування голосу на основі зразка аудіо.
    
    Особливості:
    - Кешування voice prompts для швидшої генерації
    - Валідація reference аудіо (2-30 секунд)
    - Підтримка CUDA та CPU
    - Fallback на VoiceDesign при помилках
    - Сумісність з Google Colab
    
    voice_config має містити:
        - reference_audio: str - шлях до reference аудіо
        - reference_text: str - текст reference аудіо
        - speed: float - швидкість мовлення (опціонально)
        - instruct: str - інструкції для мовлення (опціонально)
    """
    
    # Мапінг мов для Qwen3-TTS
    LANGUAGE_MAP = {
        "russian": "ru",
        "ukrainian": "uk", 
        "english": "en",
        "chinese": "zh",
        "japanese": "ja",
        "korean": "ko",
        "german": "de",
        "french": "fr",
        "spanish": "es",
        "italian": "it",
        "portuguese": "pt",
    }
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        cache_dir: Optional[str] = None
    ):
        """
        Ініціалізація VoiceClone двигуна.
        
        Args:
            model_name: Назва моделі HuggingFace
            cache_dir: Директорія для кешування voice prompts
        """
        self._model = None
        self._model_name = model_name
        self._sample_rate = 24000
        self._cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".voicebox_cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory кеш для voice prompts
        self._prompt_cache: Dict[str, Any] = {}
        
        # VoiceDesign fallback engine
        self._fallback_engine: Optional['VoiceDesignEngine'] = None
    
    def _get_device(self) -> str:
        """
        Визначення найкращого доступного пристрою.
        
        Returns:
            str: 'cuda' або 'cpu'
        """
        try:
            import torch
            
            if torch.cuda.is_available():
                # Перевірка для Google Colab
                if 'COLAB_GPU' in os.environ or 'COLAB_TPU_ADDR' in os.environ:
                    print("[VoiceClone] Виявлено Google Colab з CUDA", flush=True)
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                # MPS може мати проблеми, використовуємо CPU для стабільності
                return "cpu"
            return "cpu"
        except ImportError:
            return "cpu"
    
    def load_model(self, device: Optional[str] = None) -> bool:
        """
        Завантаження моделі.
        
        Args:
            device: Пристрій (cuda, cpu). Якщо None, визначається автоматично.
            
        Returns:
            bool: True якщо завантаження успішне
        """
        try:
            import torch
            from qwen_tts import Qwen3TTSModel
            
            if device is None:
                device = self._get_device()
            
            print(f"[VoiceClone] Завантаження моделі {self._model_name}...", flush=True)
            print(f"[VoiceClone] Пристрій: {device}", flush=True)
            
            # Визначення типу даних залежно від пристрою
            if device == "cuda":
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                device_map = "auto"
            else:
                dtype = torch.float32
                device_map = "cpu"
            
            self._model = Qwen3TTSModel.from_pretrained(
                self._model_name,
                device_map=device_map,
                torch_dtype=dtype
            )
            
            print(f"[VoiceClone] ✅ Модель завантажено!", flush=True)
            return True
            
        except ImportError as e:
            print(f"[VoiceClone] ❌ Помилка імпорту: {e}", flush=True)
            print("[VoiceClone] Встановіть: pip install git+https://github.com/QwenLM/Qwen3-TTS.git", flush=True)
            return False
        except Exception as e:
            print(f"[VoiceClone] ❌ Помилка завантаження: {e}", flush=True)
            return False
    
    def set_model(self, model) -> None:
        """
        Встановлення вже завантаженої моделі.
        
        Використовується для інтеграції з існуючим кодом VIBEMODLY,
        де модель завантажується глобально.
        
        Args:
            model: Екземпляр Qwen3TTSModel
        """
        self._model = model
        print("[VoiceClone] ✅ Модель встановлено зовні", flush=True)
    
    def is_loaded(self) -> bool:
        return self._model is not None
    
    @property
    def engine_type(self) -> str:
        return "clone"
    
    def create_voice_prompt(
        self,
        reference_audio_path: str,
        reference_text: str,
        use_cache: bool = True,
        validate: bool = True
    ) -> Tuple[Optional[Dict[str, Any]], bool]:
        """
        Створення voice prompt з reference аудіо.
        
        Це основний метод для підготовки голосу до клонування.
        Voice prompt кешується для повторного використання.
        
        Args:
            reference_audio_path: Шлях до reference аудіо файлу
            reference_text: Текст, який озвучено в reference аудіо
            use_cache: Використовувати кеш якщо доступний
            validate: Валідувати reference аудіо перед створенням
            
        Returns:
            Tuple[Optional[Dict], bool]: (voice_prompt, was_cached)
                - voice_prompt: Словник з даними для клонування або None при помилці
                - was_cached: True якщо prompt взято з кешу
                
        Example:
            >>> engine = VoiceCloneEngine()
            >>> engine.load_model()
            >>> prompt, cached = engine.create_voice_prompt(
            ...     "voice_sample.wav",
            ...     "Привіт, це зразок мого голосу."
            ... )
        """
        if not self.is_loaded():
            print("[VoiceClone] ❌ Модель не завантажена!", flush=True)
            return None, False
        
        # Валідація reference аудіо
        if validate:
            is_valid, error_msg = validate_reference_audio(reference_audio_path)
            if not is_valid:
                print(f"[VoiceClone] ❌ Валідація не пройшла: {error_msg}", flush=True)
                return None, False
        
        # Генерація ключа кешу
        cache_key = get_cache_key(reference_audio_path, reference_text)
        
        # Перевірка кешу
        if use_cache:
            # Спочатку перевіряємо in-memory кеш
            if cache_key in self._prompt_cache:
                print(f"[VoiceClone] ✅ Voice prompt з memory cache", flush=True)
                return self._prompt_cache[cache_key], True
            
            # Потім перевіряємо disk cache
            cached_prompt = self._load_prompt_from_disk(cache_key)
            if cached_prompt is not None:
                self._prompt_cache[cache_key] = cached_prompt
                print(f"[VoiceClone] ✅ Voice prompt з disk cache", flush=True)
                return cached_prompt, True
        
        # Створення нового voice prompt
        try:
            print(f"[VoiceClone] Створення voice prompt...", flush=True)
            print(f"[VoiceClone] Audio: {reference_audio_path}", flush=True)
            print(f"[VoiceClone] Text: {reference_text[:50]}...", flush=True)
            
            # Виклик методу моделі для створення voice prompt
            voice_prompt = self._model.create_voice_clone_prompt(
                ref_audio=reference_audio_path,
                ref_text=reference_text,
                x_vector_only_mode=False
            )
            
            # Кешування
            if use_cache:
                self._prompt_cache[cache_key] = voice_prompt
                self._save_prompt_to_disk(cache_key, voice_prompt)
            
            print(f"[VoiceClone] ✅ Voice prompt створено", flush=True)
            return voice_prompt, False
            
        except Exception as e:
            print(f"[VoiceClone] ❌ Помилка створення voice prompt: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return None, False
    
    def generate(
        self,
        text: str,
        voice_config: Dict[str, Any],
        language: str = "russian"
    ) -> Tuple[Optional[np.ndarray], int]:
        """
        Генерація аудіо через клонування голосу.
        
        Args:
            text: Текст для озвучення
            voice_config: Конфігурація з ключами:
                - reference_audio: str - шлях до reference аудіо
                - reference_text: str - текст reference аудіо
                - speed: float - швидкість (опціонально, за замовчуванням 1.0)
                - instruct: str - інструкції для мовлення (опціонально)
                - voice_prompt: dict - готовий voice prompt (опціонально)
                - use_cache: bool - використовувати кеш (опціонально, за замовчуванням True)
            language: Мова (russian, ukrainian, english, etc.)
            
        Returns:
            Tuple[np.ndarray, int]: (аудіо, sample_rate) або (None, 0)
        """
        if not self.is_loaded():
            print("[VoiceClone] ❌ Модель не завантажена!", flush=True)
            return None, 0
        
        try:
            import torch
            
            # Отримання параметрів
            reference_audio = voice_config.get("reference_audio")
            reference_text = voice_config.get("reference_text", "")
            speed = voice_config.get("speed", 1.0)
            instruct = voice_config.get("instruct")
            use_cache = voice_config.get("use_cache", True)
            
            # Готовий voice prompt або створення нового
            voice_prompt = voice_config.get("voice_prompt")
            was_cached = False
            
            if voice_prompt is None:
                if not reference_audio:
                    print("[VoiceClone] ❌ Не вказано reference_audio або voice_prompt!", flush=True)
                    return None, 0
                
                # Створення voice prompt
                voice_prompt, was_cached = self.create_voice_prompt(
                    reference_audio,
                    reference_text,
                    use_cache=use_cache
                )
                
                if voice_prompt is None:
                    print("[VoiceClone] ❌ Не вдалося створити voice prompt", flush=True)
                    # Спроба fallback на VoiceDesign
                    return self._fallback_to_voicedesign(text, voice_config, language)
            
            # Конвертація мови
            lang_code = self.LANGUAGE_MAP.get(language.lower(), language.lower())
            
            print(f"[VoiceClone] Генерація аудіо...", flush=True)
            print(f"[VoiceClone] Text: {text[:50]}...", flush=True)
            print(f"[VoiceClone] Language: {lang_code}", flush=True)
            print(f"[VoiceClone] Cached: {was_cached}", flush=True)
            
            # Генерація через модель
            result = self._generate_with_model(
                text=text,
                voice_prompt=voice_prompt,
                language=lang_code,
                instruct=instruct
            )
            
            if result is None:
                return None, 0
            
            audio, sr = result
            
            # Конвертація в numpy
            if hasattr(audio, 'cpu'):
                audio = audio.cpu().numpy()
            elif hasattr(audio, 'numpy'):
                audio = audio.numpy()
            
            audio = np.array(audio).flatten()
            
            # Зміна швидкості якщо потрібно
            if speed != 1.0:
                audio = self._change_speed(audio, speed)
            
            print(f"[VoiceClone] ✅ Згенеровано: {len(audio)} семплів ({len(audio)/sr:.2f}с)", flush=True)
            
            return audio, sr
            
        except Exception as e:
            print(f"[VoiceClone] ❌ Помилка генерації: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return None, 0
    
    def _generate_with_model(
        self,
        text: str,
        voice_prompt: Dict[str, Any],
        language: str,
        instruct: Optional[str] = None
    ) -> Optional[Tuple[np.ndarray, int]]:
        """
        Внутрішній метод генерації через модель.
        
        Args:
            text: Текст для озвучення
            voice_prompt: Voice prompt словник
            language: Код мови
            instruct: Інструкції для мовлення
            
        Returns:
            Optional[Tuple[np.ndarray, int]]: (аудіо, sample_rate) або None
        """
        try:
            # Основний метод генерації
            if hasattr(self._model, 'generate_voice_clone'):
                wavs, sample_rate = self._model.generate_voice_clone(
                    text=text,
                    voice_clone_prompt=voice_prompt,
                    instruct=instruct
                )
                # generate_voice_clone повертає список wav файлів
                return wavs[0] if isinstance(wavs, list) else wavs, sample_rate
            
            elif hasattr(self._model, 'generate'):
                # Альтернативний метод
                result = self._model.generate(
                    text=text,
                    voice_prompt=voice_prompt,
                    language=language
                )
                if isinstance(result, tuple):
                    return result
                return result, self._sample_rate
            
            else:
                print("[VoiceClone] ❌ Не знайдено метод генерації!", flush=True)
                return None
                
        except Exception as e:
            print(f"[VoiceClone] ❌ Помилка генерації моделі: {e}", flush=True)
            return None
    
    def _fallback_to_voicedesign(
        self,
        text: str,
        voice_config: Dict[str, Any],
        language: str
    ) -> Tuple[Optional[np.ndarray], int]:
        """
        Fallback на VoiceDesign при помилці клонування.
        
        Args:
            text: Текст для озвучення
            voice_config: Конфігурація голосу
            language: Мова
            
        Returns:
            Tuple[np.ndarray, int]: (аудіо, sample_rate) або (None, 0)
        """
        print("[VoiceClone] ⚠️ Fallback на VoiceDesign...", flush=True)
        
        try:
            # Створення fallback engine якщо потрібно
            if self._fallback_engine is None:
                self._fallback_engine = VoiceDesignEngine()
                
                # Спроба використати ту саму модель якщо це можливо
                if self.is_loaded():
                    self._fallback_engine.set_model(self._model)
            
            # Підготовка конфігурації для VoiceDesign
            fallback_config = {
                "seed": voice_config.get("seed", 42),
                "prompt": voice_config.get("prompt", "A natural voice"),
                "speed": voice_config.get("speed", 1.0)
            }
            
            return self._fallback_engine.generate(text, fallback_config, language)
            
        except Exception as e:
            print(f"[VoiceClone] ❌ Fallback помилка: {e}", flush=True)
            return None, 0
    
    def combine_voice_prompts(
        self,
        audio_paths: List[str],
        reference_texts: List[str]
    ) -> Tuple[np.ndarray, str]:
        """
        Комбінування кількох reference аудіо для кращої якості клонування.
        
        Корисно коли є кілька зразків голосу одного диктора.
        
        Args:
            audio_paths: Список шляхів до аудіо файлів
            reference_texts: Список текстів для кожного аудіо
            
        Returns:
            Tuple[np.ndarray, str]: (combined_audio, combined_text)
        """
        if len(audio_paths) != len(reference_texts):
            raise ValueError("Кількість аудіо файлів та текстів повинна співпадати")
        
        if not audio_paths:
            raise ValueError("Список аудіо файлів не може бути порожнім")
        
        print(f"[VoiceClone] Комбінування {len(audio_paths)} аудіо файлів...", flush=True)
        
        combined_audio = []
        
        for i, audio_path in enumerate(audio_paths):
            audio, sr = load_audio_for_voice_clone(audio_path)
            audio = _normalize_audio(audio)
            combined_audio.append(audio)
            print(f"[VoiceClone] [{i+1}/{len(audio_paths)}] {audio_path}", flush=True)
        
        # Конкатенація аудіо
        mixed = np.concatenate(combined_audio)
        mixed = _normalize_audio(mixed)
        
        # Комбінування текстів
        combined_text = " ".join(reference_texts)
        
        print(f"[VoiceClone] ✅ Комбіновано: {len(mixed)/sr:.2f}с", flush=True)
        
        return mixed, combined_text
    
    def _save_prompt_to_disk(self, cache_key: str, voice_prompt: Any) -> None:
        """
        Збереження voice prompt на диск.
        
        Args:
            cache_key: Ключ кешу
            voice_prompt: Voice prompt для збереження
        """
        try:
            import torch
            
            cache_file = self._cache_dir / f"{cache_key}.prompt"
            torch.save(voice_prompt, cache_file)
        except Exception as e:
            print(f"[VoiceClone] ⚠️ Не вдалося зберегти кеш: {e}", flush=True)
    
    def _load_prompt_from_disk(self, cache_key: str) -> Optional[Any]:
        """
        Завантаження voice prompt з диска.
        
        Args:
            cache_key: Ключ кешу
            
        Returns:
            Optional[Any]: Voice prompt або None
        """
        try:
            import torch
            
            cache_file = self._cache_dir / f"{cache_key}.prompt"
            if cache_file.exists():
                return torch.load(cache_file)
        except Exception as e:
            print(f"[VoiceClone] ⚠️ Не вдалося завантажити кеш: {e}", flush=True)
        return None
    
    def clear_cache(self) -> int:
        """
        Очищення кешу voice prompts.
        
        Returns:
            int: Кількість видалених файлів
        """
        # Очищення memory cache
        self._prompt_cache.clear()
        
        # Очищення disk cache
        deleted = 0
        for cache_file in self._cache_dir.glob("*.prompt"):
            try:
                cache_file.unlink()
                deleted += 1
            except Exception:
                pass
        
        print(f"[VoiceClone] ✅ Очищено {deleted} файлів кешу", flush=True)
        return deleted
    
    def _change_speed(self, audio: np.ndarray, speed: float) -> np.ndarray:
        """
        Зміна швидкості аудіо.
        
        Args:
            audio: Аудіо масив
            speed: Коефіцієнт швидкості
            
        Returns:
            np.ndarray: Аудіо зі зміненою швидкістю
        """
        try:
            from pydub import AudioSegment
            
            # Конвертація в AudioSegment
            audio_int16 = (audio * 32767).astype(np.int16)
            segment = AudioSegment(
                audio_int16.tobytes(),
                frame_rate=self._sample_rate,
                sample_width=2,
                channels=1
            )
            
            # Зміна швидкості
            if speed > 1.0:
                segment = segment.speedup(playback_speed=speed)
            elif speed < 1.0:
                segment = segment.speedup(playback_speed=speed)
            
            # Конвертація назад
            samples = np.array(segment.get_array_of_samples())
            return samples.astype(np.float32) / 32767.0
            
        except Exception as e:
            print(f"[VoiceClone] ⚠️ Помилка зміни швидкості: {e}", flush=True)
            return audio


# =============================================================================
# TTS ENGINE MANAGER
# =============================================================================

class TTSEngineManager:
    """
    Менеджер для управління TTS двигунами.
    
    Дозволяє отримувати відповідний двигун за типом профілю,
    а також керувати завантаженням моделей.
    
    Особливості:
    - Автоматичне створення двигунів за потребою
    - Спільна модель між двигунами для економії пам'яті
    - Підтримка VoiceDesign та VoiceClone двигунів
    - Управління пристроями CUDA/CPU
    """
    
    def __init__(self, shared_model: bool = True):
        """
        Ініціалізація менеджера.
        
        Args:
            shared_model: Використовувати спільну модель між двигунами (економить пам'ять)
        """
        self._engines: Dict[str, TTSEngine] = {}
        self._default_engine_type = "voicedesign"
        self._shared_model = shared_model
        self._shared_model_instance = None
    
    def register_engine(self, engine: TTSEngine) -> None:
        """
        Реєстрація двигуна.
        
        Args:
            engine: Екземпляр TTSEngine
        """
        self._engines[engine.engine_type] = engine
        print(f"[TTSEngineManager] Зареєстровано двигун: {engine.engine_type}", flush=True)
    
    def get_engine(self, engine_type: Optional[str] = None) -> TTSEngine:
        """
        Отримання двигуна за типом.
        
        Args:
            engine_type: Тип двигуна ('voicedesign', 'clone')
                        Якщо None, повертає двигун за замовчуванням
                        
        Returns:
            TTSEngine: Екземпляр двигуна
            
        Raises:
            ValueError: Якщо двигун не знайдено
        """
        if engine_type is None:
            engine_type = self._default_engine_type
        
        if engine_type not in self._engines:
            # Створення двигуна якщо не існує
            if engine_type == "voicedesign":
                engine = VoiceDesignEngine()
            elif engine_type == "clone":
                engine = VoiceCloneEngine()
            else:
                raise ValueError(f"Невідомий тип двигуна: {engine_type}")
            
            # Спільна модель якщо включено і модель вже завантажена
            if self._shared_model and self._shared_model_instance is not None:
                engine.set_model(self._shared_model_instance)
            
            self._engines[engine_type] = engine
        
        return self._engines[engine_type]
    
    def set_default_engine(self, engine_type: str) -> None:
        """
        Встановлення двигуна за замовчуванням.
        
        Args:
            engine_type: Тип двигуна
        """
        if engine_type not in ["voicedesign", "clone"]:
            raise ValueError(f"Невірний тип двигуна: {engine_type}")
        self._default_engine_type = engine_type
    
    def has_engine(self, engine_type: str) -> bool:
        """
        Перевірка наявності двигуна.
        
        Args:
            engine_type: Тип двигуна
            
        Returns:
            bool: True якщо двигун зареєстровано
        """
        return engine_type in self._engines
    
    def is_engine_loaded(self, engine_type: str) -> bool:
        """
        Перевірка чи завантажено двигун.
        
        Args:
            engine_type: Тип двигуна
            
        Returns:
            bool: True якщо двигун завантажено
        """
        if engine_type in self._engines:
            return self._engines[engine_type].is_loaded()
        return False
    
    def load_all_engines(self, device: Optional[str] = None) -> Dict[str, bool]:
        """
        Завантаження всіх зареєстрованих двигунів.
        
        Args:
            device: Пристрій (cuda, cpu). Якщо None, автовизначення.
            
        Returns:
            Dict[str, bool]: Результати завантаження
        """
        results = {}
        
        # Визначення пристрою
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
                
                # Збереження спільної моделі
                if self._shared_model and success and self._shared_model_instance is None:
                    self._shared_model_instance = engine._model
            else:
                results[engine_type] = engine.is_loaded()
        
        return results
    
    def load_engine(self, engine_type: str, device: Optional[str] = None) -> bool:
        """
        Завантаження конкретного двигуна.
        
        Args:
            engine_type: Тип двигуна ('voicedesign', 'clone')
            device: Пристрій (cuda, cpu). Якщо None, автовизначення.
            
        Returns:
            bool: True якщо завантажено успішно
        """
        engine = self.get_engine(engine_type)
        
        # Якщо спільна модель вже завантажена, використовуємо її
        if self._shared_model and self._shared_model_instance is not None:
            engine.set_model(self._shared_model_instance)
            return True
        
        # Визначення пристрою
        if device is None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        
        # Завантаження моделі
        if hasattr(engine, 'load_model'):
            success = engine.load_model(device)
            
            # Збереження спільної моделі
            if self._shared_model and success:
                self._shared_model_instance = engine._model
            
            return success
        
        return engine.is_loaded()
    
    def unload_all(self) -> None:
        """
        Вивантаження всіх двигунів та звільнення пам'яті.
        """
        for engine_type, engine in self._engines.items():
            if hasattr(engine, '_model') and engine._model is not None:
                del engine._model
                engine._model = None
        
        self._shared_model_instance = None
        
        # Очищення CUDA кешу
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        print("[TTSEngineManager] Всі двигуни вивантажено", flush=True)
    
    def get_voice_clone_engine(self) -> VoiceCloneEngine:
        """
        Отримання VoiceClone двигуна (зручний метод).
        
        Returns:
            VoiceCloneEngine: Екземпляр VoiceClone двигуна
        """
        return self.get_engine("clone")
    
    def get_voicedesign_engine(self) -> VoiceDesignEngine:
        """
        Отримання VoiceDesign двигуна (зручний метод).
        
        Returns:
            VoiceDesignEngine: Екземпляр VoiceDesign двигуна
        """
        return self.get_engine("voicedesign")


# =============================================================================
# ЕКСПОРТ
# =============================================================================

__all__ = [
    # Enums
    'VoiceMode',
    
    # Classes
    'TTSEngine',
    'VoiceDesignEngine',
    'VoiceCloneEngine',
    'TTSEngineManager',
    'VoiceProfile',
    'VoiceProfileManager',
    'VoicePromptCache',
    
    # Functions
    'validate_reference_audio',
    'load_audio_for_voice_clone',
    'get_cache_key',
]

__version__ = '1.1.0'
__author__ = 'VIBEMODLY Team'
