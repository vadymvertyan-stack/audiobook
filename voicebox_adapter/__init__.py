"""
Voicebox Adapter - Інтеграція Voicebox TTS з VIBEMODLY

Цей модуль надає абстракцію для роботи з різними TTS двигунами:
- VoiceDesign (seed + prompt) - поточний підхід VIBEMODLY
- VoiceClone (reference audio) - підхід з Voicebox

Використання:
    from voicebox_adapter import TTSEngineManager, VoiceProfileManager
    
    # Ініціалізація
    tts_manager = TTSEngineManager()
    profile_manager = VoiceProfileManager()
    
    # Отримання двигуна для профілю
    profile = profile_manager.get_profile("narrator")
    engine = tts_manager.get_engine(profile.mode)
    
    # Генерація аудіо
    audio, sample_rate = engine.generate("Текст", profile.to_engine_config())
"""

from .engine import TTSEngine, VoiceDesignEngine, VoiceCloneEngine, TTSEngineManager
from .profile import VoiceProfile, VoiceMode, VoiceProfileManager
from .cache import VoicePromptCache

__all__ = [
    'TTSEngine',
    'VoiceDesignEngine', 
    'VoiceCloneEngine',
    'TTSEngineManager',
    'VoiceProfile',
    'VoiceMode',
    'VoiceProfileManager',
    'VoicePromptCache'
]

__version__ = '1.0.0'
__author__ = 'VIBEMODLY Team'
