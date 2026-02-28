"""
Voice Profile - Уніфікована система голосових профілів

Цей модуль надає:
- VoiceMode: Enum для режимів генерації голосу
- VoiceProfile: Dataclass для уніфікованого голосового профілю
- VoiceProfileManager: Менеджер для управління профілями
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum
import hashlib
import json


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
