"""
TTS Engine - Абстракція для різних TTS двигунів

Цей модуль надає:
- TTSEngine: Абстрактний базовий клас для TTS двигунів
- VoiceDesignEngine: Двигун для створення голосу через seed + prompt
- VoiceCloneEngine: Двигун для клонування голосу з reference audio
- TTSEngineManager: Менеджер для управління двигунами
- Допоміжні функції для валідації та обробки аудіо

Сумісність:
- Google Colab (CUDA/T4 GPU)
- Локальне середовище з CUDA
- CPU fallback
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any, List, Union
from pathlib import Path
import numpy as np
import hashlib
import asyncio
import os

from .cache import VoicePromptCache


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
