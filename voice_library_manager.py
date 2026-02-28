"""
Voice Library Manager - Персистентна бібліотека голосів для VIBEMODLY.

Цей модуль надає систему управління голосами персонажів з:
- Створення canonical reference аудіо (15-20 сек)
- Збереження на Google Drive
- Валідація якості reference
- Інтеграція з VoiceCloneEngine

Версія: 1.0.0
Дата створення: 2026-02-21
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import numpy as np
import os
import shutil
from typing import Any, Dict, List, Optional, Tuple


class DriveSyncError(Exception):
	"""
	Помилка синхронізації з Google Drive.

	Виникає при проблемах з монтуванням, читанням або записом на Google Drive.
	"""
	pass


class ValidationLevel(Enum):
	"""
	Рівні валідації reference аудіо.

	Attributes:
		BASIC: Тільки критичні перевірки (тривалість, наявність аудіо)
		STANDARD: Стандартний набір перевірок (RMS, кліпінг)
		STRICT: Суворі оптимальні значення (оптимальна тривалість 15-20 сек)
	"""

	BASIC = 'basic'
	STANDARD = 'standard'
	STRICT = 'strict'


class ReferenceQualityMetrics:
	"""
	Метрики якості reference аудіо.

	Цей клас містить порогові значення для валідації reference аудіо
	перед використанням для клонування голосу.

	Attributes:
		DURATION_OPTIMAL_MIN: Оптимальна мінімальна тривалість (15 сек)
		DURATION_OPTIMAL_MAX: Оптимальна максимальна тривалість (20 сек)
		DURATION_ABSOLUTE_MIN: Абсолютний мінімум тривалості (10 сек)
		DURATION_ABSOLUTE_MAX: Абсолютний максимум тривалості (30 сек)
		RMS_MIN: Мінімальний RMS рівень (не занадто тихий)
		RMS_MAX: Максимальний RMS рівень (не занадто гучний)
		RMS_OPTIMAL: Оптимальний RMS рівень
		CLIP_THRESHOLD: Поріг виявлення кліпінгу
		CLIP_MAX_SAMPLES: Максимальна кількість семплів на порозі кліпінгу
		DYNAMIC_RANGE_MIN: Мінімальний динамічний діапазон
		SILENCE_RATIO_MAX: Максимальне співвідношення тиші
	"""

	# Порогові значення тривалості
	DURATION_OPTIMAL_MIN: float = 15.0  # Оптимальний мінімум
	DURATION_OPTIMAL_MAX: float = 20.0  # Оптимальний максимум
	DURATION_ABSOLUTE_MIN: float = 10.0  # Абсолютний мінімум
	DURATION_ABSOLUTE_MAX: float = 30.0  # Абсолютний максимум

	# RMS рівні
	RMS_MIN: float = 0.02  # Мінімальний RMS (не занадто тихий)
	RMS_MAX: float = 0.30  # Максимальний RMS (не занадто гучний)
	RMS_OPTIMAL: float = 0.10  # Оптимальний RMS

	# Кліпінг
	CLIP_THRESHOLD: float = 0.98  # Поріг кліпінгу
	CLIP_MAX_SAMPLES: int = 10  # Макс. семплів на порозі

	# Динамічний діапазон
	DYNAMIC_RANGE_MIN: float = 0.1  # Мінімальний динамічний діапазон
	SILENCE_RATIO_MAX: float = 0.15  # Макс. співвідношення тиші


@dataclass
class VoiceLibraryEntry:
	"""
	Запис у бібліотеці голосів.

	Представляє один голос персонажа з усіма необхідними метаданими
	для клонування голосу та управління бібліотекою.

	Attributes:
		character_name: Унікальний ідентифікатор персонажа
		voice_preset: Ключ пресету з VOICE_PRESETS
		reference_audio_path: Відносний шлях до reference аудіо
		reference_text: Текст reference аудіо
		duration: Тривалість в секундах
		quality_score: Оцінка якості (0.0-1.0)
		created_at: Timestamp створення в форматі ISO 8601
		updated_at: Timestamp останнього оновлення в форматі ISO 8601
		metadata: Додаткові метадані (gender, language, seed, prompt, etc.)

	Example:
		>>> entry = VoiceLibraryEntry(
		...     character_name="narrator",
		...     voice_preset="narrator_neutral",
		...     reference_audio_path="references/narrator_ref.wav",
		...     reference_text="Привіт! Це тестовий фрагмент...",
		...     duration=17.5,
		...     quality_score=0.92,
		...     created_at="2026-02-21T10:00:00Z",
		...     updated_at="2026-02-21T10:00:00Z"
		... )
	"""

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
		"""
		Пост-ініціалізація для валідації даних.

		Перевіряє:
		- quality_score в діапазоні [0.0, 1.0]
		- duration позитивне число
		- character_name не порожній
		"""
		# Валідація quality_score
		if not 0.0 <= self.quality_score <= 1.0:
			raise ValueError(
				f'quality_score має бути в діапазоні [0.0, 1.0], '
				f'отримано: {self.quality_score}'
			)

		# Валідація duration
		if self.duration < 0:
			raise ValueError(
				f'duration має бути позитивним числом, '
				f'отримано: {self.duration}'
			)

		# Валідація character_name
		if not self.character_name or not self.character_name.strip():
			raise ValueError('character_name не може бути порожнім')

	def to_dict(self) -> Dict[str, Any]:
		"""
		Конвертація запису в словник.

		Returns:
			Словник з усіма полями запису
		"""
		return {
			'character_name': self.character_name,
			'voice_preset': self.voice_preset,
			'reference_audio_path': self.reference_audio_path,
			'reference_text': self.reference_text,
			'duration': self.duration,
			'quality_score': self.quality_score,
			'created_at': self.created_at,
			'updated_at': self.updated_at,
			'metadata': self.metadata,
		}

	@classmethod
	def from_dict(cls, data: Dict[str, Any]) -> 'VoiceLibraryEntry':
		"""
		Створення запису зі словника.

		Args:
			data: Словник з даними запису

		Returns:
			Новий екземпляр VoiceLibraryEntry

		Raises:
			KeyError: Якщо відсутні обов'язкові поля
		"""
		return cls(
			character_name=data['character_name'],
			voice_preset=data['voice_preset'],
			reference_audio_path=data['reference_audio_path'],
			reference_text=data['reference_text'],
			duration=data['duration'],
			quality_score=data['quality_score'],
			created_at=data['created_at'],
			updated_at=data['updated_at'],
			metadata=data.get('metadata', {}),
		)


class VoiceLibraryManager:
	"""
	Менеджер бібліотеки голосів з персистентністю на Google Drive.

	Керування голосами персонажів включає:
	- Створення canonical reference аудіо (15-20 сек)
	- Збереження на Google Drive
	- Валідацію якості reference
	- Інтеграцію з VoiceCloneEngine та VoiceDesignEngine
	"""

	# Константи тривалості
	OPTIMAL_DURATION_MIN = 15.0
	OPTIMAL_DURATION_MAX = 20.0
	MIN_DURATION = 10.0
	MAX_DURATION = 30.0

	# Шляхи
	DRIVE_BASE_PATH = '/content/drive/MyDrive/vibemodly_voices'
	LIBRARY_FILE = 'voice_library.json'
	REFERENCE_DIR = 'references'

	def __init__(
		self,
		voice_clone_engine,  # VoiceCloneEngine
		voice_design_engine,  # VoiceDesignEngine
		cache=None,  # Optional[VoicePromptCache]
		drive_path: Optional[str] = None
	):
		"""
		Ініціалізація менеджера бібліотеки голосів.

		Args:
			voice_clone_engine: Екземпляр VoiceCloneEngine для клонування
			voice_design_engine: Екземпляр VoiceDesignEngine для дизайну голосів
			cache: Опціональний кеш для voice prompts
			drive_path: Шлях до Google Drive для збереження
		"""
		self._engine_clone = voice_clone_engine
		self._engine_design = voice_design_engine
		self._cache = cache
		self._library: Dict[str, VoiceLibraryEntry] = {}
		self._drive_path = drive_path or self.DRIVE_BASE_PATH
		self._created_at = datetime.now().isoformat()

	def get_voice(
		self,
		character_name: str,
		voice_preset: str,
		gender: str = 'male',
		language: str = 'russian'
	) -> Tuple[Optional[Dict[str, Any]], bool]:
		"""
		Отримання voice_prompt для персонажа.

		Якщо reference ще не існує - створює canonical reference.

		Args:
			character_name: Унікальний ідентифікатор персонажа
			voice_preset: Ключ пресету з VOICE_PRESETS
			gender: Стать голосу ('male' або 'female')
			language: Мова голосу ('russian', 'ukrainian', 'english')

		Returns:
			Кортеж (voice_prompt_dict, is_new_reference):
			- voice_prompt_dict: Словник з параметрами голосу або None
			- is_new_reference: True якщо reference було щойно створено

		Raises:
			ValueError: Якщо параметри некоректні
		"""
		# Валідація параметрів
		if not character_name or not character_name.strip():
			raise ValueError('character_name не може бути порожнім')
		if not voice_preset or not voice_preset.strip():
			raise ValueError('voice_preset не може бути порожнім')

		# 1. Перевірити чи є в бібліотеці
		entry = self._library.get(character_name)
		if entry:
			# Перевірити чи існує файл reference
			ref_path = os.path.join(self._drive_path, entry.reference_audio_path)
			if os.path.exists(ref_path):
				# Створити voice_prompt з існуючого reference
				voice_prompt, _ = self._engine_clone.create_voice_prompt(
					reference_audio_path=ref_path,
					reference_text=entry.reference_text,
					use_cache=True
				)
				if voice_prompt:
					return voice_prompt, False

		# 2. Створити новий canonical reference
		audio_path, ref_text = self.create_canonical_reference(
			character_name=character_name,
			voice_preset=voice_preset,
			gender=gender,
			language=language
		)

		if audio_path is None:
			print(f'[VoiceLibrary] ❌ Не вдалося створити reference для {character_name}')
			return None, False

		# 3. Валідація reference
		is_valid, error_msg, quality_score = self.validate_reference(audio_path, strict=False)
		if not is_valid:
			print(f'[VoiceLibrary] ⚠️ Reference не пройшов валідацію: {error_msg}')

		# 4. Створити voice_prompt
		voice_prompt, _ = self._engine_clone.create_voice_prompt(
			reference_audio_path=audio_path,
			reference_text=ref_text,
			use_cache=True
		)

		if voice_prompt is None:
			return None, False

		# 5. Розрахувати тривалість
		duration = self._calculate_duration(audio_path)

		# 6. Зберегти в бібліотеку
		entry = VoiceLibraryEntry(
			character_name=character_name,
			voice_preset=voice_preset,
			reference_audio_path=os.path.relpath(audio_path, self._drive_path),
			reference_text=ref_text,
			duration=duration,
			quality_score=quality_score,
			created_at=datetime.now().isoformat(),
			updated_at=datetime.now().isoformat(),
			metadata={'gender': gender, 'language': language}
		)
		self._library[character_name] = entry

		# 7. Зберегти бібліотеку
		self.save_library()

		return voice_prompt, True

	def create_canonical_reference(
		self,
		character_name: str,
		voice_preset: str,
		gender: str = 'male',
		language: str = 'russian',
		custom_text: Optional[str] = None
	) -> Tuple[Optional[str], Optional[str]]:
		"""
		Створення canonical reference аудіо (15-20 сек).

		Генерує reference аудіо через VoiceDesignEngine для подальшого
		використання при клонуванні голосу.

		Args:
			character_name: Унікальний ідентифікатор персонажа
			voice_preset: Ключ пресету з VOICE_PRESETS
			gender: Стать голосу ('male' або 'female')
			language: Мова голосу ('russian', 'ukrainian', 'english')
			custom_text: Опціональний текст для reference

		Returns:
			Кортеж (reference_path, reference_text) або (None, None) при помилці

		Raises:
			ValueError: Якщо параметри некоректні
		"""
		# Валідація параметрів
		if not character_name or not character_name.strip():
			raise ValueError('character_name не може бути порожнім')
		if not voice_preset or not voice_preset.strip():
			raise ValueError('voice_preset не може бути порожнім')

		# 1. Перевірити що двигуни доступні
		if self._engine_design is None:
			print('[VoiceLibrary] ❌ VoiceDesignEngine не ініціалізовано')
			return None, None

		# 2. Генерація тексту для reference
		ref_text = custom_text or self._generate_reference_text(language, target_duration=17.5)

		# 3. Отримати seed та prompt з voice_preset
		# (це буде отримано з VOICE_PRESETS у vibemodly_colab.py)
		seed = None
		prompt = None

		# Спроба отримати конфігурацію з кешу або метаданих
		if self._cache is not None:
			cached_config = self._cache.get_preset_config(voice_preset)
			if cached_config:
				seed = cached_config.get('seed')
				prompt = cached_config.get('prompt')

		# 4. Створення директорії для references
		self._ensure_directory_structure()

		# 5. Шлях для збереження
		audio_path = self._get_reference_path(character_name)

		# 6. Генерація через VoiceDesignEngine
		try:
			print(f'[VoiceLibrary] 🎙️ Створення canonical reference для {character_name}...')

			# Виклик generate через VoiceDesign
			audio, sr = self._engine_design.generate(
				text=ref_text,
				voice_config={
					'seed': seed,
					'prompt': prompt,
					'gender': gender
				},
				language=language
			)

			if audio is None:
				print('[VoiceLibrary] ❌ Не вдалося згенерувати аудіо')
				return None, None

			# 7. Збереження аудіо файлу
			if not self._save_audio_file(audio, sr, audio_path):
				print('[VoiceLibrary] ❌ Не вдалося зберегти аудіо файл')
				return None, None

			print(f'[VoiceLibrary] ✅ Canonical reference створено: {audio_path}')
			return audio_path, ref_text

		except Exception as e:
			print(f'[VoiceLibrary] ❌ Помилка створення reference: {e}')
			return None, None

	def list_voices(self) -> List[VoiceLibraryEntry]:
		"""
		Отримання списку всіх голосів у бібліотеці.

		Returns:
			Список всіх записів VoiceLibraryEntry
		"""
		return list(self._library.values())

	def get_entry(self, character_name: str) -> Optional[VoiceLibraryEntry]:
		"""
		Отримання запису для персонажа.

		Args:
			character_name: Унікальний ідентифікатор персонажа

		Returns:
			VoiceLibraryEntry або None якщо не знайдено
		"""
		return self._library.get(character_name)

	def delete_voice(self, character_name: str) -> bool:
		"""
		Видалення голосу з бібліотеки.

		Args:
			character_name: Унікальний ідентифікатор персонажа

		Returns:
			True якщо голос було видалено, False якщо не знайдено
		"""
		if character_name in self._library:
			del self._library[character_name]
			return True
		return False

	def update_voice(
		self,
		character_name: str,
		voice_preset: Optional[str] = None,
		custom_text: Optional[str] = None
	) -> bool:
		"""
		Оновлення голосу персонажа.

		Дозволяє змінити пресет голосу та/або перегенерувати reference
		з новим текстом.

		Args:
			character_name: Унікальний ідентифікатор персонажа
			voice_preset: Новий пресет голосу (опціонально)
			custom_text: Новий текст для reference (опціонально)

		Returns:
			True якщо оновлення успішне, False якщо голос не знайдено

		Raises:
			ValueError: Якщо character_name порожній
		"""
		# Валідація параметрів
		if not character_name or not character_name.strip():
			raise ValueError('character_name не може бути порожнім')

		# Перевірити чи існує запис
		entry = self._library.get(character_name)
		if not entry:
			print(f'[VoiceLibrary] ⚠️ Голос "{character_name}" не знайдено')
			return False

		# Оновити пресет якщо задано
		if voice_preset and voice_preset.strip():
			entry.voice_preset = voice_preset

		# Перегенерувати reference якщо задано новий текст
		if custom_text and custom_text.strip():
			audio_path, ref_text = self.create_canonical_reference(
				character_name=character_name,
				voice_preset=entry.voice_preset,
				gender=entry.metadata.get('gender', 'male'),
				language=entry.metadata.get('language', 'russian'),
				custom_text=custom_text
			)
			if audio_path:
				entry.reference_audio_path = os.path.relpath(audio_path, self._drive_path)
				entry.reference_text = ref_text
				# Оновити тривалість
				entry.duration = self._calculate_duration(audio_path)
				# Валідація та оцінка якості
				is_valid, error_msg, quality_score = self.validate_reference(audio_path, strict=False)
				entry.quality_score = quality_score
			else:
				print(f'[VoiceLibrary] ⚠️ Не вдалося перегенерувати reference')

		# Оновити timestamp
		entry.updated_at = datetime.now().isoformat()

		# Зберегти бібліотеку
		self.save_library()

		print(f'[VoiceLibrary] ✅ Голос "{character_name}" оновлено')
		return True

	def _generate_reference_text(
		self,
		language: str = 'russian',
		target_duration: float = 17.5
	) -> str:
		"""
		Генерація тексту для canonical reference.

		Текст розрахований на ~17.5 секунд озвучення.

		Args:
			language: Мова тексту ('russian', 'ukrainian', 'english')
			target_duration: Цільова тривалість в секундах

		Returns:
			Текст для озвучення (~250-300 символів для 17.5 сек)
		"""
		# Тексти для різних мов (~250-300 символів для 17.5 сек)
		texts = {
			'russian': (
				'Привет! Это тестовый фрагмент голоса для определения его '
				'характеристик. Я могу озвучивать ваши истории с разными '
				'эмоциями и интонациями. Этот текст специально создан для '
				'формирования качественного reference аудио.'
			),
			'ukrainian': (
				'Привіт! Це тестовий фрагмент голосу для визначення його '
				'характеристик. Я можу озвучувати ваші історії з різними '
				'емоціями та інтонаціями. Цей текст спеціально створено для '
				'формування якісного reference аудіо.'
			),
			'english': (
				'Hello! This is a test voice sample for determining its '
				'characteristics. I can voice your stories with different '
				'emotions and intonations. This text is specially created '
				'for forming a high-quality reference audio.'
			)
		}
		return texts.get(language, texts['russian'])

	def _get_reference_path(self, character_name: str) -> str:
		"""
		Отримання шляху для збереження reference аудіо.

		Args:
			character_name: Унікальний ідентифікатор персонажа

		Returns:
			Повний шлях до файлу reference аудіо
		"""
		safe_name = character_name.replace(' ', '_').lower()
		return os.path.join(
			self._drive_path,
			self.REFERENCE_DIR,
			f'{safe_name}_ref.wav'
		)

	# ========================================================================
	# Google Drive Persistence Methods
	# ========================================================================

	def _ensure_drive_mounted(self) -> bool:
		"""
		Перевірка чи змонтовано Google Drive.

		Якщо Drive не змонтовано, намагається змонтувати його.

		Returns:
			bool: True якщо Drive доступний, False інакше
		"""
		if not os.path.exists('/content/drive/MyDrive'):
			try:
				from google.colab import drive
				drive.mount('/content/drive')
				return True
			except Exception as e:
				print(f'[VoiceLibrary] ❌ Не вдалося змонтувати Drive: {e}')
				return False
		return True

	def _ensure_directory_structure(self) -> bool:
		"""
		Створення необхідної структури директорій.

		Створює наступні директорії:
		- Базова директорія бібліотеки
		- Директорія для reference аудіо
		- Директорія для кешу
		- Директорія для backup

		Returns:
			bool: True якщо структура створена/існує
		"""
		dirs = [
			self._drive_path,
			os.path.join(self._drive_path, self.REFERENCE_DIR),
			os.path.join(self._drive_path, 'cache'),
			os.path.join(self._drive_path, 'backup')
		]

		for dir_path in dirs:
			os.makedirs(dir_path, exist_ok=True)

		return True

	def save_library(self) -> bool:
		"""
		Збереження бібліотеки на Google Drive.

		Зберігає всі голоси у JSON файл на Google Drive.
		Перед збереженням створює backup існуючого файлу.

		Returns:
			bool: True якщо збережено успішно, False при помилці

		Raises:
			DriveSyncError: Якщо не вдалося змонтувати Drive
		"""
		if not self._ensure_drive_mounted():
			return False

		self._ensure_directory_structure()

		library_path = os.path.join(self._drive_path, self.LIBRARY_FILE)

		# Створення backup перед збереженням
		if os.path.exists(library_path):
			self._create_backup(library_path)

		# Підготовка даних
		data = {
			'version': '1.0.0',
			'created_at': self._created_at,
			'updated_at': datetime.now().isoformat(),
			'entries': {
				name: entry.to_dict()
				for name, entry in self._library.items()
			},
			'statistics': self._calculate_statistics()
		}

		# Збереження
		try:
			with open(library_path, 'w', encoding='utf-8') as f:
				json.dump(data, f, ensure_ascii=False, indent=2)

			print(f'[VoiceLibrary] ✅ Збережено {len(self._library)} голосів')
			return True

		except Exception as e:
			print(f'[VoiceLibrary] ❌ Помилка збереження: {e}')
			return False

	def load_library(self) -> bool:
		"""
		Завантаження бібліотеки з Google Drive.

		Завантажує голоси з JSON файлу на Google Drive.
		Якщо файл не існує, створює порожню бібліотеку.

		Returns:
			bool: True якщо завантажено успішно, False при помилці
		"""
		if not self._ensure_drive_mounted():
			return False

		library_path = os.path.join(self._drive_path, self.LIBRARY_FILE)

		if not os.path.exists(library_path):
			print('[VoiceLibrary] 📭 Бібліотека не знайдена, створюємо нову')
			self._library = {}
			return True

		try:
			with open(library_path, 'r', encoding='utf-8') as f:
				data = json.load(f)

			# Валідація версії
			version = data.get('version', '0.0.0')
			if version != '1.0.0':
				print(f'[VoiceLibrary] ⚠️ Версія {version}, може знадобитися міграція')

			# Завантаження записів
			self._library = {}
			for name, entry_data in data.get('entries', {}).items():
				self._library[name] = VoiceLibraryEntry.from_dict(entry_data)

			print(f'[VoiceLibrary] ✅ Завантажено {len(self._library)} голосів')
			return True

		except Exception as e:
			print(f'[VoiceLibrary] ❌ Помилка завантаження: {e}')
			return False

	def _create_backup(self, library_path: str) -> None:
		"""
		Створення резервної копії бібліотеки.

		Створює копію файлу бібліотеки з timestamp у назві.

		Args:
			library_path: Шлях до файлу бібліотеки для backup
		"""
		timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
		backup_name = f'voice_library_{timestamp}.json'
		backup_path = os.path.join(self._drive_path, 'backup', backup_name)

		try:
			shutil.copy2(library_path, backup_path)
			print(f'[VoiceLibrary] 💾 Backup створено: {backup_name}')
		except Exception as e:
			print(f'[VoiceLibrary] ⚠️ Не вдалося створити backup: {e}')

	def _calculate_statistics(self) -> Dict[str, Any]:
		"""
		Розрахунок статистики бібліотеки.

		Обчислює:
		- Загальну кількість голосів
		- Загальну тривалість reference аудіо
		- Середню оцінку якості

		Returns:
			Dict[str, Any]: Словник зі статистикою бібліотеки
		"""
		if not self._library:
			return {
				'total_voices': 0,
				'total_duration_sec': 0.0,
				'average_quality': 0.0
			}

		total_duration = sum(e.duration for e in self._library.values())
		avg_quality = sum(e.quality_score for e in self._library.values()) / len(self._library)

		return {
			'total_voices': len(self._library),
			'total_duration_sec': round(total_duration, 2),
			'average_quality': round(avg_quality, 2)
		}

	def sync_with_drive(self) -> Dict[str, int]:
		"""
		Двостороння синхронізація з Google Drive.

		Синхронізує локальну бібліотеку з віддаленою на Google Drive.

		Returns:
			Dict[str, int]: Статистика синхронізації:
				- added: кількість доданих голосів
				- downloaded: кількість завантажених голосів
				- updated: кількість оновлених голосів
				- deleted: кількість видалених голосів
		"""
		# TODO: Реалізувати повну синхронізацію
		return {'added': 0, 'downloaded': 0, 'updated': 0, 'deleted': 0}

	# ========================================================================
	# Reference Audio Validation Methods
	# ========================================================================

	def validate_reference(
		self,
		audio_path: str,
		strict: bool = False
	) -> Tuple[bool, Optional[str], float]:
		"""
		Валідація reference аудіо.

		Виконує комплексну перевірку якості аудіо файлу перед використанням
		для клонування голосу. Перевіряє тривалість, гучність, кліпінг,
		динамічний діапазон та співвідношення тиші.

		Args:
			audio_path: Шлях до аудіо файлу
			strict: Сувора валідація (використовує оптимальні значення)

		Returns:
			Tuple[is_valid, error_message, quality_score]:
			- is_valid: True якщо аудіо пройшло валідацію
			- error_message: Повідомлення про помилку або None
			- quality_score: Оцінка якості від 0.0 до 1.0

		Example:
			>>> is_valid, error, score = manager.validate_reference('voice.wav')
			>>> if is_valid:
			...     print(f'Якісне аудіо! Оцінка: {score}')
			... else:
			...     print(f'Помилка: {error}')
		"""
		try:
			# Завантаження аудіо
			audio, sr = self._load_audio(audio_path)

			# 1. Перевірка тривалості
			duration = len(audio) / sr
			duration_ok, duration_msg = self._validate_duration(duration, strict)
			if not duration_ok:
				return False, duration_msg, 0.0

			# 2. Перевірка RMS рівня
			rms_ok, rms_msg, rms_score = self._validate_rms(audio)
			if not rms_ok:
				return False, rms_msg, 0.0

			# 3. Перевірка кліпінгу
			clip_ok, clip_msg, clip_score = self._validate_clipping(audio)
			if not clip_ok:
				return False, clip_msg, 0.0

			# 4. Перевірка динамічного діапазону
			dyn_ok, dyn_msg, dyn_score = self._validate_dynamic_range(audio)

			# 5. Перевірка тиші
			silence_ok, silence_msg, silence_score = self._validate_silence(audio, sr)

			# Розрахунок загальної оцінки
			quality_score = self._calculate_quality_score(
				duration=duration,
				rms_score=rms_score,
				clip_score=clip_score,
				dyn_score=dyn_score,
				silence_score=silence_score,
				strict=strict
			)

			return True, None, quality_score

		except Exception as e:
			return False, f'Помилка валідації: {e}', 0.0

	def _load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
		"""
		Завантаження аудіо файлу.

		Завантажує WAV файл та нормалізує до float32.
		Підтримує 16-бітні та 32-бітні WAV файли.

		Args:
			audio_path: Шлях до аудіо файлу

		Returns:
			Tuple[np.ndarray, int]: Кортеж (audio_data, sample_rate):
			- audio_data: Масив numpy з нормалізованими семплами
			- sample_rate: Частота дискретизації

		Raises:
			ImportError: Якщо scipy не встановлено
			FileNotFoundError: Якщо файл не знайдено
			ValueError: Якщо формат файлу не підтримується
		"""
		# Спробувати використати scipy.io.wavfile
		try:
			from scipy.io import wavfile
			sr, audio = wavfile.read(audio_path)
			# Нормалізація до float32
			if audio.dtype == np.int16:
				audio = audio.astype(np.float32) / 32768.0
			elif audio.dtype == np.int32:
				audio = audio.astype(np.float32) / 2147483648.0
			elif audio.dtype == np.float32:
				audio = audio.astype(np.float32)
			else:
				# Спроба конвертації інших типів
				audio = audio.astype(np.float32)
				# Нормалізація якщо потрібно
				if np.max(np.abs(audio)) > 1.0:
					audio = audio / np.max(np.abs(audio))
			return audio, sr
		except ImportError:
			# Fallback - заглушка
			raise ImportError('scipy не встановлено. Встановіть: pip install scipy')

	def _validate_duration(
		self,
		duration: float,
		strict: bool
	) -> Tuple[bool, Optional[str]]:
		"""
		Валідація тривалості аудіо.

		Перевіряє чи тривалість аудіо знаходиться в допустимих межах.
		У строгому режимі використовуються оптимальні значення (15-20 сек),
		у звичайному - абсолютні межі (10-30 сек).

		Args:
			duration: Тривалість аудіо в секундах
			strict: Чи використовувати строгі (оптимальні) межі

		Returns:
			Tuple[bool, Optional[str]]:
			- bool: True якщо тривалість в межах норми
			- Optional[str]: Повідомлення про помилку або None
		"""
		metrics = ReferenceQualityMetrics

		if strict:
			min_dur = metrics.DURATION_OPTIMAL_MIN
			max_dur = metrics.DURATION_OPTIMAL_MAX
		else:
			min_dur = metrics.DURATION_ABSOLUTE_MIN
			max_dur = metrics.DURATION_ABSOLUTE_MAX

		if duration < min_dur:
			return False, f'Тривалість {duration:.1f}с занадто коротка. Мінімум: {min_dur}с'

		if duration > max_dur:
			return False, f'Тривалість {duration:.1f}с занадто довга. Максимум: {max_dur}с'

		return True, None

	def _validate_rms(
		self,
		audio: np.ndarray
	) -> Tuple[bool, Optional[str], float]:
		"""
		Валідація RMS рівня гучності.

		RMS (Root Mean Square) показує середній рівень гучності аудіо.
		Занадто тихе аудіо погано для клонування, занадто гучне -
		ознака можливих проблем.

		Args:
			audio: Масив numpy з аудіо даними

		Returns:
			Tuple[bool, Optional[str], float]:
			- bool: True якщо RMS в допустимих межах
			- Optional[str]: Повідомлення про помилку або None
			- float: Оцінка близькості до оптимального значення (0.0-1.0)
		"""
		metrics = ReferenceQualityMetrics
		rms = np.sqrt(np.mean(audio**2))

		if rms < metrics.RMS_MIN:
			return False, f'Аудіо занадто тихе (RMS: {rms:.4f})', 0.0

		if rms > metrics.RMS_MAX:
			return False, f'Аудіо занадто гучне (RMS: {rms:.4f})', 0.0

		# Оцінка близькості до оптимального значення
		optimal = metrics.RMS_OPTIMAL
		score = 1.0 - abs(rms - optimal) / optimal

		return True, None, max(0.0, min(1.0, score))

	def _validate_clipping(
		self,
		audio: np.ndarray
	) -> Tuple[bool, Optional[str], float]:
		"""
		Валідація кліпінгу (обрізання сигналу).

		Кліпінг виникає коли амплітуда сигналу перевищує максимально
		допустиме значення, що призводить до спотворень.

		Args:
			audio: Масив numpy з аудіо даними

		Returns:
			Tuple[bool, Optional[str], float]:
			- bool: True якщо кліпінг в допустимих межах
			- Optional[str]: Повідомлення про помилку або None
			- float: Оцінка (менше кліпінгу = краще, 0.5-1.0)
		"""
		metrics = ReferenceQualityMetrics

		# Підрахунок семплів на порозі кліпінгу
		clip_count = np.sum(np.abs(audio) >= metrics.CLIP_THRESHOLD)

		if clip_count > metrics.CLIP_MAX_SAMPLES:
			return False, f'Виявлено кліпінг ({clip_count} семплів)', 0.0

		# Оцінка (менше кліпінгу = краще)
		score = 1.0 - (clip_count / metrics.CLIP_MAX_SAMPLES) if metrics.CLIP_MAX_SAMPLES > 0 else 1.0

		return True, None, max(0.5, score)

	def _validate_dynamic_range(
		self,
		audio: np.ndarray
	) -> Tuple[bool, Optional[str], float]:
		"""
		Валідація динамічного діапазону.

		Динамічний діапазон показує співвідношення між піковим
		та середнім рівнем сигналу. Занадто малий діапазон
		вказує на компресію або монотонність.

		Args:
			audio: Масив numpy з аудіо даними

		Returns:
			Tuple[bool, Optional[str], float]:
			- bool: True якщо динамічний діапазон достатній
			- Optional[str]: Повідомлення про помилку або None
			- float: Оцінка динамічного діапазону (0.0-1.0)
		"""
		metrics = ReferenceQualityMetrics

		# Розрахунок динамічного діапазону
		peak = np.max(np.abs(audio))
		rms = np.sqrt(np.mean(audio**2))

		if rms > 0:
			dynamic_range = peak / rms
		else:
			dynamic_range = 0

		if dynamic_range < metrics.DYNAMIC_RANGE_MIN:
			return False, f'Динамічний діапазон занадто малий: {dynamic_range:.2f}', 0.0

		# Оцінка (більший діапазон = краще, але не занадто)
		score = min(1.0, dynamic_range / 5.0)

		return True, None, score

	def _validate_silence(
		self,
		audio: np.ndarray,
		sr: int
	) -> Tuple[bool, Optional[str], float]:
		"""
		Валідація співвідношення тиші.

		Перевіряє відсоток тиші в аудіо. Занадто багато тиші
		знижує якість reference для клонування.

		Args:
			audio: Масив numpy з аудіо даними
			sr: Частота дискретизації

		Returns:
			Tuple[bool, Optional[str], float]:
			- bool: True якщо співвідношення тиші допустиме
			- Optional[str]: Повідомлення про помилку або None
			- float: Оцінка (менше тиші = краще, 0.0-1.0)
		"""
		metrics = ReferenceQualityMetrics

		# Визначення тиші (семпли нижче порогу)
		silence_threshold = 0.01
		silence_samples = np.sum(np.abs(audio) < silence_threshold)
		silence_ratio = silence_samples / len(audio)

		if silence_ratio > metrics.SILENCE_RATIO_MAX:
			return False, f'Занадто багато тиші: {silence_ratio*100:.1f}%', 0.0

		# Оцінка (менше тиші = краще)
		score = 1.0 - silence_ratio

		return True, None, score

	def _calculate_quality_score(
		self,
		duration: float,
		rms_score: float,
		clip_score: float,
		dyn_score: float,
		silence_score: float,
		strict: bool
	) -> float:
		"""
		Розрахунок загальної оцінки якості.

		Обчислює зважену суму оцінок за різними критеріями:
		- Тривалість: 30%
		- RMS рівень: 25%
		- Відсутність кліпінгу: 20%
		- Динамічний діапазон: 15%
		- Відсутність тиші: 10%

		Args:
			duration: Тривалість аудіо в секундах
			rms_score: Оцінка RMS рівня (0.0-1.0)
			clip_score: Оцінка відсутності кліпінгу (0.0-1.0)
			dyn_score: Оцінка динамічного діапазону (0.0-1.0)
			silence_score: Оцінка відсутності тиші (0.0-1.0)
			strict: Чи використовувати строгі межі для оцінки тривалості

		Returns:
			float: Загальна оцінка якості (0.0-1.0)
		"""
		metrics = ReferenceQualityMetrics

		# Оцінка тривалості
		if strict:
			optimal_min = metrics.DURATION_OPTIMAL_MIN
			optimal_max = metrics.DURATION_OPTIMAL_MAX
		else:
			optimal_min = metrics.DURATION_ABSOLUTE_MIN
			optimal_max = metrics.DURATION_ABSOLUTE_MAX

		optimal_mid = (optimal_min + optimal_max) / 2

		if duration < optimal_min:
			duration_score = duration / optimal_min
		elif duration > optimal_max:
			duration_score = optimal_max / duration
		else:
			# Чим ближче до середини оптимального діапазону, тим краще
			duration_score = 1.0 - abs(duration - optimal_mid) / optimal_mid

		# Ваги критеріїв
		weights = {
			'duration': 0.30,
			'rms': 0.25,
			'clipping': 0.20,
			'dynamic': 0.15,
			'silence': 0.10
		}

		# Зважена сума
		total_score = (
			weights['duration'] * duration_score +
			weights['rms'] * rms_score +
			weights['clipping'] * clip_score +
			weights['dynamic'] * dyn_score +
			weights['silence'] * silence_score
		)

		return round(total_score, 2)

	# ========================================================================
	# Audio File Helper Methods
	# ========================================================================

	def _save_audio_file(
		self,
		audio: np.ndarray,
		sr: int,
		path: str
	) -> bool:
		"""
		Збереження аудіо у WAV файл.

		Конвертує float32 аудіо в int16 формат та зберігає у WAV файл.

		Args:
			audio: Масив numpy з аудіо даними (float32, діапазон [-1, 1])
			sr: Частота дискретизації
			path: Шлях для збереження файлу

		Returns:
			True якщо файл збережено успішно, False при помилці

		Raises:
			ImportError: Якщо scipy не встановлено
		"""
		try:
			from scipy.io import wavfile

			# Нормалізація якщо потрібно
			if np.max(np.abs(audio)) > 1.0:
				audio = audio / np.max(np.abs(audio))

			# Конвертація в int16
			audio_int16 = (audio * 32767).astype(np.int16)

			# Збереження
			wavfile.write(path, sr, audio_int16)

			return True

		except ImportError:
			print('[VoiceLibrary] ❌ scipy не встановлено. Встановіть: pip install scipy')
			return False
		except Exception as e:
			print(f'[VoiceLibrary] ❌ Помилка збереження аудіо: {e}')
			return False

	def _calculate_duration(self, audio_path: str) -> float:
		"""
		Розрахунок тривалості аудіо файлу.

		Args:
			audio_path: Шлях до аудіо файлу

		Returns:
			Тривалість у секундах або 0.0 при помилці
		"""
		try:
			audio, sr = self._load_audio(audio_path)
			return len(audio) / sr
		except Exception:
			return 0.0
