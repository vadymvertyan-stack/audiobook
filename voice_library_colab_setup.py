"""
Voice Library Colab Setup - Інтеграція VoiceLibraryManager з Google Colab.

Цей модуль надає повну інтеграцію системи управління голосами з:
- Ініціалізацією в Google Colab середовищі
- Синхронізацією з Google Drive
- Адаптерами для VoiceClone та VoiceDesign двигунів
- Глобальними функціями для зручного використання

Використання в Colab:
    from voice_library_colab_setup import init_voice_library_colab, get_voice_library
    
    # Ініціалізація
    library = init_voice_library_colab()
    
    # Отримання голосу
    voice_prompt, is_new = library.get_voice("narrator", "male_deep")

Версія: 1.0.0
Дата створення: 2026-02-21
Автор: VIBEMODLY Team
"""

# =============================================================================
# SECTION 1: IMPORTS
# =============================================================================

# Standard library
import os
import json
import hashlib
import logging
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Any, Tuple, Union
from pathlib import Path
from enum import Enum

# Third-party
import numpy as np

# Local imports - VoiceLibraryManager
from voice_library_manager import (
    VoiceLibraryManager,
    VoiceLibraryEntry,
    DriveSyncError,
    ValidationLevel,
    ReferenceQualityMetrics
)

# Local imports - Voicebox Adapter
from voicebox_adapter_colab import (
    VoiceCloneEngine,
    VoiceDesignEngine,
    VoicePromptCache,
    VoiceProfile,
    VoiceMode,
    validate_reference_audio,
    load_audio_for_voice_clone,
    get_cache_key
)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='[%(name)s] %(message)s'
)
logger = logging.getLogger('VoiceLibraryColab')


# =============================================================================
# SECTION 2: DATACLASSES
# =============================================================================

@dataclass
class ColabVoiceConfig:
    """
    Конфігурація голосу для Google Colab середовища.
    
    Розширює базову конфігурацію з додатковими параметрами
    для роботи в Colab середовищі.
    
    Attributes:
        character_name: Ім'я персонажа
        voice_preset: Ключ пресету з VOICE_PRESETS
        gender: Стать голосу ('male' або 'female')
        language: Мова ('russian', 'ukrainian', 'english')
        use_clone: Використовувати клонування голосу
        auto_create_reference: Автоматично створювати reference
        custom_reference_text: Власний текст для reference
    """
    character_name: str
    voice_preset: str = 'male_deep'
    gender: str = 'male'
    language: str = 'russian'
    use_clone: bool = True
    auto_create_reference: bool = True
    custom_reference_text: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертація в словник."""
        return asdict(self)


@dataclass
class VoiceLibraryStats:
    """
    Статистика бібліотеки голосів.
    
    Attributes:
        total_voices: Загальна кількість голосів
        total_duration_sec: Загальна тривалість reference аудіо
        average_quality: Середня оцінка якості
        drive_synced: Чи синхронізовано з Drive
        last_sync_time: Час останньої синхронізації
    """
    total_voices: int = 0
    total_duration_sec: float = 0.0
    average_quality: float = 0.0
    drive_synced: bool = False
    last_sync_time: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертація в словник."""
        return asdict(self)


@dataclass
class GenerationResult:
    """
    Результат генерації аудіо.
    
    Attributes:
        success: Чи успішна генерація
        audio: Аудіо дані (numpy array) або None
        sample_rate: Частота дискретизації
        duration: Тривалість в секундах
        voice_id: ID використаного голосу
        is_new_voice: Чи щойно створено голос
        error_message: Повідомлення про помилку
    """
    success: bool
    audio: Optional[np.ndarray] = None
    sample_rate: int = 24000
    duration: float = 0.0
    voice_id: Optional[str] = None
    is_new_voice: bool = False
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертація в словник (без аудіо даних)."""
        return {
            'success': self.success,
            'sample_rate': self.sample_rate,
            'duration': self.duration,
            'voice_id': self.voice_id,
            'is_new_voice': self.is_new_voice,
            'error_message': self.error_message
        }


class ColabInitError(Exception):
    """
    Помилка ініціалізації в Google Colab.
    
    Виникає при проблемах з:
    - Монтуванням Google Drive
    - Завантаженням моделей TTS
    - Створенням директорій
    """
    pass


class VoiceNotFoundError(Exception):
    """
    Помилка - голос не знайдено.
    
    Виникає при спробі отримати неіснуючий голос
    з бібліотеки.
    """
    pass


# =============================================================================
# SECTION 3: COLAB ADAPTERS
# =============================================================================

# Глобальні змінні для singleton
_voice_library_instance: Optional[VoiceLibraryManager] = None
_clone_engine_instance: Optional[VoiceCloneEngine] = None
_design_engine_instance: Optional[VoiceDesignEngine] = None
_cache_instance: Optional[VoicePromptCache] = None
_is_colab_initialized: bool = False


def is_colab_environment() -> bool:
    """
    Перевірка чи запущено в Google Colab.
    
    Returns:
        bool: True якщо в Colab середовищі
    """
    try:
        import google.colab
        return True
    except ImportError:
        return False


def mount_google_drive() -> bool:
    """
    Монтування Google Drive в Colab.
    
    Returns:
        bool: True якщо успішно змонтовано
    """
    if not is_colab_environment():
        logger.warning('Не в Colab середовищі - Drive монтування пропущено')
        return False
    
    if os.path.exists('/content/drive/MyDrive'):
        logger.info('Google Drive вже змонтовано')
        return True
    
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        logger.info('✅ Google Drive змонтовано')
        return True
    except Exception as e:
        logger.error(f'❌ Помилка монтування Drive: {e}')
        return False


def init_voice_library_colab(
    drive_path: str = '/content/drive/MyDrive/vibemodly_voices',
    cache_dir: str = '/content/cache/voice_library',
    auto_mount_drive: bool = True,
    load_models: bool = True
) -> VoiceLibraryManager:
    """
    Ініціалізація VoiceLibraryManager для Google Colab.
    
    Це головна функція для налаштування бібліотеки голосів в Colab.
    Автоматично:
    - Монтує Google Drive
    - Створює необхідні директорії
    - Ініціалізує TTS двигуни
    - Завантажує існуючу бібліотеку
    
    Args:
        drive_path: Шлях до директорії бібліотеки на Drive
        cache_dir: Директорія для кешу
        auto_mount_drive: Автоматично монтувати Drive
        load_models: Завантажувати моделі TTS
        
    Returns:
        VoiceLibraryManager: Ініціалізований менеджер бібліотеки
        
    Raises:
        ColabInitError: При помилці ініціалізації
        
    Example:
        >>> library = init_voice_library_colab()
        >>> voice, is_new = library.get_voice('narrator', 'male_deep')
    """
    global _voice_library_instance, _clone_engine_instance
    global _design_engine_instance, _cache_instance, _is_colab_initialized
    
    # Повертаємо існуючий інстанс якщо вже ініціалізовано
    if _voice_library_instance is not None and _is_colab_initialized:
        logger.info('VoiceLibrary вже ініціалізовано')
        return _voice_library_instance
    
    logger.info('=== Ініціалізація VoiceLibrary для Colab ===')
    
    # 1. Монтування Drive
    if auto_mount_drive and is_colab_environment():
        if not mount_google_drive():
            raise ColabInitError('Не вдалося змонтувати Google Drive')
    
    # 2. Створення директорій
    os.makedirs(drive_path, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(os.path.join(drive_path, 'references'), exist_ok=True)
    os.makedirs(os.path.join(drive_path, 'backup'), exist_ok=True)
    logger.info(f'📁 Директорії створено: {drive_path}')
    
    # 3. Ініціалізація кешу
    _cache_instance = VoicePromptCache(cache_dir)
    logger.info('💾 Кеш ініціалізовано')
    
    # 4. Ініціалізація двигунів
    if load_models:
        try:
            # VoiceClone двигун
            _clone_engine_instance = VoiceCloneEngine(cache_dir=cache_dir)
            logger.info('🎙️ VoiceCloneEngine створено')
            
            # VoiceDesign двигун
            _design_engine_instance = VoiceDesignEngine()
            logger.info('🎨 VoiceDesignEngine створено')
            
        except Exception as e:
            logger.warning(f'⚠️ Моделі не завантажено: {e}')
            logger.info('Моделі можна завантажити пізніше через set_models()')
    
    # 5. Створення менеджера бібліотеки
    _voice_library_instance = VoiceLibraryManager(
        voice_clone_engine=_clone_engine_instance,
        voice_design_engine=_design_engine_instance,
        cache=_cache_instance,
        drive_path=drive_path
    )
    
    # 6. Завантаження існуючої бібліотеки
    _voice_library_instance.load_library()
    
    _is_colab_initialized = True
    logger.info('✅ VoiceLibrary для Colab ініціалізовано!')
    
    return _voice_library_instance


def set_models(
    clone_model=None,
    design_model=None
) -> None:
    """
    Встановлення моделей TTS після їх завантаження.
    
    Використовується коли моделі завантажуються зовні
    (наприклад, в vibemodly_colab.py).
    
    Args:
        clone_model: Модель для VoiceClone
        design_model: Модель для VoiceDesign
    """
    global _clone_engine_instance, _design_engine_instance
    
    if clone_model is not None:
        if _clone_engine_instance is None:
            _clone_engine_instance = VoiceCloneEngine()
        _clone_engine_instance.set_model(clone_model)
        logger.info('✅ VoiceClone модель встановлено')
    
    if design_model is not None:
        if _design_engine_instance is None:
            _design_engine_instance = VoiceDesignEngine()
        _design_engine_instance.set_model(design_model)
        logger.info('✅ VoiceDesign модель встановлено')


def get_colab_stats() -> VoiceLibraryStats:
    """
    Отримання статистики бібліотеки для Colab.
    
    Returns:
        VoiceLibraryStats: Статистика бібліотеки
    """
    global _voice_library_instance, _is_colab_initialized
    
    if _voice_library_instance is None:
        return VoiceLibraryStats()
    
    voices = _voice_library_instance.list_voices()
    
    return VoiceLibraryStats(
        total_voices=len(voices),
        total_duration_sec=sum(v.duration for v in voices),
        average_quality=sum(v.quality_score for v in voices) / len(voices) if voices else 0.0,
        drive_synced=_is_colab_initialized,
        last_sync_time=datetime.now().isoformat() if _is_colab_initialized else None
    )


# =============================================================================
# SECTION 4: GLOBAL FUNCTIONS
# =============================================================================

def get_voice_library() -> VoiceLibraryManager:
    """
    Отримання singleton екземпляру VoiceLibraryManager.
    
    Якщо бібліотека не ініціалізована - ініціалізує з налаштуваннями за замовчуванням.
    
    Returns:
        VoiceLibraryManager: Екземпляр менеджера бібліотеки
        
    Raises:
        ColabInitError: Якщо не вдалося ініціалізувати
        
    Example:
        >>> library = get_voice_library()
        >>> voices = library.list_voices()
    """
    global _voice_library_instance
    
    if _voice_library_instance is None:
        return init_voice_library_colab()
    
    return _voice_library_instance


def get_voice_prompt(
    character_name: str,
    voice_preset: str = 'male_deep',
    gender: str = 'male',
    language: str = 'russian'
) -> Tuple[Optional[Dict[str, Any]], bool]:
    """
    Швидке отримання voice_prompt для персонажа.
    
    Це зручна функція-обгортка для отримання голосу без
    прямої роботи з VoiceLibraryManager.
    
    Args:
        character_name: Ім'я персонажа
        voice_preset: Ключ пресету з VOICE_PRESETS
        gender: Стать голосу
        language: Мова голосу
        
    Returns:
        Tuple[voice_prompt, is_new]: Голос та прапорець нового голосу
        
    Example:
        >>> prompt, is_new = get_voice_prompt('narrator', 'male_deep')
        >>> if is_new:
        ...     print('Створено новий голос')
    """
    library = get_voice_library()
    return library.get_voice(
        character_name=character_name,
        voice_preset=voice_preset,
        gender=gender,
        language=language
    )


def create_voice_from_reference(
    character_name: str,
    reference_audio_path: str,
    reference_text: str,
    voice_preset: str = 'custom',
    gender: str = 'male',
    language: str = 'russian'
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Створення голосу з існуючого reference аудіо.
    
    Використовується коли вже є якісний зразок голосу
    для клонування.
    
    Args:
        character_name: Ім'я персонажа
        reference_audio_path: Шлях до reference аудіо
        reference_text: Текст reference аудіо
        voice_preset: Ключ пресету
        gender: Стать голосу
        language: Мова
        
    Returns:
        Tuple[voice_prompt, error_message]: Голос або помилка
        
    Example:
        >>> prompt, error = create_voice_from_reference(
        ...     'narrator',
        ...     '/content/my_voice.wav',
        ...     'Привіт, це тестовий запис голосу.'
        ... )
    """
    global _voice_library_instance, _clone_engine_instance
    
    if _voice_library_instance is None:
        return None, 'Бібліотека не ініціалізована. Викличте init_voice_library_colab()'
    
    if _clone_engine_instance is None:
        return None, 'VoiceClone двигун не ініціалізовано'
    
    try:
        # Валідація reference
        is_valid, error_msg = validate_reference_audio(reference_audio_path)
        if not is_valid:
            return None, f'Валідація не пройшла: {error_msg}'
        
        # Створення voice prompt
        voice_prompt, _ = _clone_engine_instance.create_voice_prompt(
            reference_audio_path=reference_audio_path,
            reference_text=reference_text,
            use_cache=True,
            validate=True
        )
        
        if voice_prompt is None:
            return None, 'Не вдалося створити voice prompt'
        
        # Додавання до бібліотеки
        # Створюємо запис для бібліотеки
        from voice_library_manager import VoiceLibraryEntry
        
        # Розрахунок тривалості
        audio, sr = load_audio_for_voice_clone(reference_audio_path)
        duration = len(audio) / sr
        
        entry = VoiceLibraryEntry(
            character_name=character_name,
            voice_preset=voice_preset,
            reference_audio_path=reference_audio_path,
            reference_text=reference_text,
            duration=duration,
            quality_score=0.85,  # Базова оцінка
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            metadata={'gender': gender, 'language': language, 'source': 'custom_reference'}
        )
        
        # Збереження в бібліотеці
        _voice_library_instance._library[character_name] = entry
        _voice_library_instance.save_library()
        
        logger.info(f'✅ Голос "{character_name}" створено з reference')
        return voice_prompt, None
        
    except Exception as e:
        return None, f'Помилка: {str(e)}'


def clone_voice_from_entry(
    character_name: str,
    text: str,
    speed: float = 1.0
) -> GenerationResult:
    """
    Генерація аудіо з клонуванням голосу з бібліотеки.
    
    Args:
        character_name: Ім'я персонажа з бібліотеки
        text: Текст для озвучення
        speed: Швидкість мовлення
        
    Returns:
        GenerationResult: Результат генерації
        
    Example:
        >>> result = clone_voice_from_entry('narrator', 'Привіт, світ!')
        >>> if result.success:
        ...     audio = result.audio
    """
    global _voice_library_instance, _clone_engine_instance
    
    if _voice_library_instance is None:
        return GenerationResult(
            success=False,
            error_message='Бібліотека не ініціалізована'
        )
    
    if _clone_engine_instance is None:
        return GenerationResult(
            success=False,
            error_message='VoiceClone двигун не ініціалізовано'
        )
    
    try:
        # Отримання запису з бібліотеки
        entry = _voice_library_instance.get_entry(character_name)
        if entry is None:
            return GenerationResult(
                success=False,
                error_message=f'Голос "{character_name}" не знайдено в бібліотеці'
            )
        
        # Отримання voice_prompt
        voice_prompt, is_new = _voice_library_instance.get_voice(
            character_name=character_name,
            voice_preset=entry.voice_preset,
            gender=entry.metadata.get('gender', 'male'),
            language=entry.metadata.get('language', 'russian')
        )
        
        if voice_prompt is None:
            return GenerationResult(
                success=False,
                error_message='Не вдалося отримати voice_prompt'
            )
        
        # Генерація аудіо
        audio, sr = _clone_engine_instance.generate(
            text=text,
            voice_config={
                'voice_prompt': voice_prompt,
                'speed': speed
            },
            language=entry.metadata.get('language', 'russian')
        )
        
        if audio is None:
            return GenerationResult(
                success=False,
                error_message='Генерація не вдалася'
            )
        
        duration = len(audio) / sr
        
        return GenerationResult(
            success=True,
            audio=audio,
            sample_rate=sr,
            duration=duration,
            voice_id=character_name,
            is_new_voice=is_new
        )
        
    except Exception as e:
        return GenerationResult(
            success=False,
            error_message=f'Помилка: {str(e)}'
        )


def list_available_voices() -> List[Dict[str, Any]]:
    """
    Список всіх доступних голосів у бібліотеці.
    
    Returns:
        List[Dict]: Список голосів з метаданими
        
    Example:
        >>> voices = list_available_voices()
        >>> for v in voices:
        ...     print(f"{v['name']}: {v['duration']:.1f}s")
    """
    library = get_voice_library()
    voices = library.list_voices()
    
    return [
        {
            'name': v.character_name,
            'preset': v.voice_preset,
            'duration': v.duration,
            'quality': v.quality_score,
            'gender': v.metadata.get('gender', 'unknown'),
            'language': v.metadata.get('language', 'unknown'),
            'created': v.created_at
        }
        for v in voices
    ]


def delete_voice_from_library(character_name: str) -> bool:
    """
    Видалення голосу з бібліотеки.
    
    Args:
        character_name: Ім'я персонажа для видалення
        
    Returns:
        bool: True якщо видалено успішно
    """
    library = get_voice_library()
    result = library.delete_voice(character_name)
    
    if result:
        library.save_library()
        logger.info(f'🗑️ Голос "{character_name}" видалено')
    
    return result


# =============================================================================
# SECTION 5: GENERATE_AUDIO INTEGRATION
# =============================================================================

def generate_audio_with_library(
    text: str,
    character_name: str = 'narrator',
    voice_preset: str = 'male_deep',
    gender: str = 'male',
    language: str = 'russian',
    speed: float = 1.0,
    use_library: bool = True,
    fallback_to_design: bool = True
) -> GenerationResult:
    """
    Генерація аудіо з використанням VoiceLibrary.
    
    Це головна функція для інтеграції з існуючим кодом VIBEMODLY.
    Підтримує voice_id параметр для ідентифікації голосу.
    
    Args:
        text: Текст для озвучення
        character_name: Ім'я персонажа (використовується як voice_id)
        voice_preset: Ключ пресету з VOICE_PRESETS
        gender: Стать голосу
        language: Мова
        speed: Швидкість мовлення
        use_library: Використовувати бібліотеку голосів
        fallback_to_design: Fallback на VoiceDesign при помилці
        
    Returns:
        GenerationResult: Результат генерації з аудіо даними
        
    Example:
        >>> result = generate_audio_with_library(
        ...     'Привіт, це тестовий текст.',
        ...     character_name='narrator',
        ...     voice_preset='male_deep'
        ... )
        >>> if result.success:
        ...     audio = result.audio
        ...     print(f'Тривалість: {result.duration:.2f}с')
    """
    global _voice_library_instance, _clone_engine_instance, _design_engine_instance
    
    # Якщо бібліотека не використовується - fallback на просту генерацію
    if not use_library:
        return _generate_without_library(
            text=text,
            voice_preset=voice_preset,
            gender=gender,
            language=language,
            speed=speed
        )
    
    # Перевірка ініціалізації бібліотеки
    if _voice_library_instance is None:
        logger.warning('Бібліотека не ініціалізована, спроба ініціалізації...')
        try:
            init_voice_library_colab(load_models=False)
        except Exception as e:
            return GenerationResult(
                success=False,
                error_message=f'Не вдалося ініціалізувати бібліотеку: {e}'
            )
    
    try:
        # Отримання voice_prompt з бібліотеки
        voice_prompt, is_new = _voice_library_instance.get_voice(
            character_name=character_name,
            voice_preset=voice_preset,
            gender=gender,
            language=language
        )
        
        if voice_prompt is None:
            if fallback_to_design:
                logger.warning(f'Voice prompt не отримано, fallback на VoiceDesign')
                return _generate_without_library(
                    text=text,
                    voice_preset=voice_preset,
                    gender=gender,
                    language=language,
                    speed=speed
                )
            else:
                return GenerationResult(
                    success=False,
                    error_message='Не вдалося отримати voice_prompt'
                )
        
        # Перевірка Clone двигуна
        if _clone_engine_instance is None:
            return GenerationResult(
                success=False,
                error_message='VoiceClone двигун не ініціалізовано'
            )
        
        # Генерація через VoiceClone
        logger.info(f'🎙️ Генерація через VoiceClone для "{character_name}"')
        
        audio, sr = _clone_engine_instance.generate(
            text=text,
            voice_config={
                'voice_prompt': voice_prompt,
                'speed': speed
            },
            language=language
        )
        
        if audio is None:
            if fallback_to_design:
                logger.warning('Генерація через Clone не вдалася, fallback на VoiceDesign')
                return _generate_without_library(
                    text=text,
                    voice_preset=voice_preset,
                    gender=gender,
                    language=language,
                    speed=speed
                )
            else:
                return GenerationResult(
                    success=False,
                    error_message='Генерація через VoiceClone не вдалася'
                )
        
        duration = len(audio) / sr
        
        logger.info(f'✅ Аудіо згенеровано: {duration:.2f}с')
        
        return GenerationResult(
            success=True,
            audio=audio,
            sample_rate=sr,
            duration=duration,
            voice_id=character_name,
            is_new_voice=is_new
        )
        
    except Exception as e:
        logger.error(f'Помилка генерації: {e}')
        return GenerationResult(
            success=False,
            error_message=str(e)
        )


def _generate_without_library(
    text: str,
    voice_preset: str,
    gender: str,
    language: str,
    speed: float
) -> GenerationResult:
    """
    Fallback генерація через VoiceDesign без бібліотеки.
    
    Args:
        text: Текст для озвучення
        voice_preset: Ключ пресету
        gender: Стать голосу
        language: Мова
        speed: Швидкість
        
    Returns:
        GenerationResult: Результат генерації
    """
    global _design_engine_instance
    
    if _design_engine_instance is None:
        return GenerationResult(
            success=False,
            error_message='VoiceDesign двигун не ініціалізовано'
        )
    
    try:
        # Отримання seed з voice_preset (спрощено)
        seed = hash(voice_preset) % (2**32)
        
        # Формування prompt
        gender_prefix = 'Male voice' if gender == 'male' else 'Female voice'
        prompt = f'{gender_prefix}, natural and clear.'
        
        audio, sr = _design_engine_instance.generate(
            text=text,
            voice_config={
                'seed': seed,
                'prompt': prompt,
                'speed': speed
            },
            language=language
        )
        
        if audio is None:
            return GenerationResult(
                success=False,
                error_message='VoiceDesign генерація не вдалася'
            )
        
        duration = len(audio) / sr
        
        return GenerationResult(
            success=True,
            audio=audio,
            sample_rate=sr,
            duration=duration,
            voice_id=f'design_{voice_preset}',
            is_new_voice=True
        )
        
    except Exception as e:
        return GenerationResult(
            success=False,
            error_message=str(e)
        )


def generate_audio_batch(
    texts: List[str],
    character_name: str = 'narrator',
    voice_preset: str = 'male_deep',
    gender: str = 'male',
    language: str = 'russian',
    speed: float = 1.0
) -> List[GenerationResult]:
    """
    Пакетна генерація аудіо для списку текстів.
    
    Оптимізовано для генерації кількох реплік одного персонажа.
    Voice_prompt кешується між викликами.
    
    Args:
        texts: Список текстів для озвучення
        character_name: Ім'я персонажа
        voice_preset: Ключ пресету
        gender: Стать голосу
        language: Мова
        speed: Швидкість
        
    Returns:
        List[GenerationResult]: Список результатів генерації
        
    Example:
        >>> texts = ['Привіт!', 'Як справи?', 'До побачення!']
        >>> results = generate_audio_batch(texts, 'narrator')
        >>> for r in results:
        ...     if r.success:
        ...         print(f'{r.duration:.2f}с')
    """
    results = []
    
    # Отримуємо voice_prompt один раз для всіх текстів
    voice_prompt, is_new = get_voice_prompt(
        character_name=character_name,
        voice_preset=voice_preset,
        gender=gender,
        language=language
    )
    
    if voice_prompt is None:
        return [GenerationResult(
            success=False,
            error_message='Не вдалося отримати voice_prompt'
        ) for _ in texts]
    
    for text in texts:
        result = generate_audio_with_library(
            text=text,
            character_name=character_name,
            voice_preset=voice_preset,
            gender=gender,
            language=language,
            speed=speed,
            use_library=True
        )
        results.append(result)
    
    return results


# =============================================================================
# SECTION 6: DEMO FUNCTION
# =============================================================================

def demo_voice_library(
    test_text: str = 'Привіт! Це тестовий текст для демонстрації голосу.',
    character_name: str = 'demo_narrator',
    voice_preset: str = 'male_deep',
    gender: str = 'male',
    language: str = 'russian'
) -> None:
    """
    Демонстрація можливостей VoiceLibrary.
    
    Показує повний цикл роботи з бібліотекою:
    1. Ініціалізація
    2. Створення/отримання голосу
    3. Генерація аудіо
    4. Відображення статистики
    
    Args:
        test_text: Текст для демонстрації
        character_name: Ім'я демо-персонажа
        voice_preset: Пресет голосу
        gender: Стать голосу
        language: Мова
        
    Example:
        >>> demo_voice_library()
        === Voice Library Demo ===
        ✅ Бібліотека ініціалізована
        🎙️ Голос 'demo_narrator' отримано
        ...
    """
    print('\n' + '='*60)
    print('=== Voice Library Demo ===')
    print('='*60 + '\n')
    
    # 1. Ініціалізація
    print('📦 Крок 1: Ініціалізація бібліотеки...')
    try:
        library = init_voice_library_colab(load_models=False)
        print('✅ Бібліотека ініціалізована')
    except Exception as e:
        print(f'❌ Помилка ініціалізації: {e}')
        return
    
    # 2. Статистика до
    print('\n📊 Крок 2: Статистика бібліотеки (до)...')
    stats = get_colab_stats()
    print(f'   • Голосів: {stats.total_voices}')
    print(f'   • Середня якість: {stats.average_quality:.2f}')
    
    # 3. Отримання голосу
    print(f'\n🎙️ Крок 3: Отримання голосу "{character_name}"...')
    voice_prompt, is_new = get_voice_prompt(
        character_name=character_name,
        voice_preset=voice_preset,
        gender=gender,
        language=language
    )
    
    if voice_prompt is not None:
        status = 'новий' if is_new else 'з бібліотеки'
        print(f'✅ Голос отримано ({status})')
    else:
        print('❌ Не вдалося отримати голос')
        return
    
    # 4. Генерація аудіо
    print(f'\n🔊 Крок 4: Генерація аудіо...')
    print(f'   Текст: "{test_text[:50]}..."')
    
    result = generate_audio_with_library(
        text=test_text,
        character_name=character_name,
        voice_preset=voice_preset,
        gender=gender,
        language=language
    )
    
    if result.success:
        print(f'✅ Аудіо згенеровано:')
        print(f'   • Тривалість: {result.duration:.2f}с')
        print(f'   • Sample rate: {result.sample_rate} Hz')
        print(f'   • Новий голос: {result.is_new_voice}')
    else:
        print(f'❌ Помилка генерації: {result.error_message}')
    
    # 5. Статистика після
    print('\n📊 Крок 5: Статистика бібліотеки (після)...')
    stats = get_colab_stats()
    print(f'   • Голосів: {stats.total_voices}')
    print(f'   • Загальна тривалість: {stats.total_duration_sec:.2f}с')
    print(f'   • Середня якість: {stats.average_quality:.2f}')
    
    # 6. Список голосів
    print('\n📚 Крок 6: Список голосів у бібліотеці...')
    voices = list_available_voices()
    for v in voices:
        print(f'   • {v["name"]}: {v["duration"]:.1f}с, якість {v["quality"]:.0%}')
    
    print('\n' + '='*60)
    print('=== Demo завершено ===')
    print('='*60 + '\n')


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Dataclasses
    'ColabVoiceConfig',
    'VoiceLibraryStats',
    'GenerationResult',
    
    # Exceptions
    'ColabInitError',
    'VoiceNotFoundError',
    
    # Colab adapters
    'is_colab_environment',
    'mount_google_drive',
    'init_voice_library_colab',
    'set_models',
    'get_colab_stats',
    
    # Global functions
    'get_voice_library',
    'get_voice_prompt',
    'create_voice_from_reference',
    'clone_voice_from_entry',
    'list_available_voices',
    'delete_voice_from_library',
    
    # Generate audio integration
    'generate_audio_with_library',
    'generate_audio_batch',
    
    # Demo
    'demo_voice_library',
]

__version__ = '1.0.0'
__author__ = 'VIBEMODLY Team'
