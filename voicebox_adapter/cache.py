"""
Voice Prompt Cache - Кешування voice prompts для пришвидшення генерації

Цей модуль надає:
- VoicePromptCache: Клас для кешування voice prompts з підтримкою memory та disk кешу
"""

import hashlib
import json
from pathlib import Path
from typing import Optional, Any, Dict
import os


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
    
    def preload_from_disk(self, keys: Optional[list] = None) -> int:
        """
        Попереднє завантаження з disk в memory cache.
        
        Args:
            keys: Список ключів для завантаження (None = всі)
            
        Returns:
            int: Кількість завантажених елементів
        """
        loaded = 0
        
        try:
            import torch
            
            if keys is None:
                # Завантажуємо всі файли з disk cache
                for cache_file in self.cache_dir.glob("*.pt"):
                    key = cache_file.stem
                    if key not in self._memory_cache:
                        try:
                            data = torch.load(cache_file, map_location='cpu')
                            self._set_memory_cache(key, data)
                            loaded += 1
                        except Exception as e:
                            print(f"[CACHE] Помилка завантаження {key}: {e}", flush=True)
            else:
                # Завантажуємо тільки вказані ключі
                for key in keys:
                    if key not in self._memory_cache:
                        cache_file = self.cache_dir / f"{key}.pt"
                        if cache_file.exists():
                            try:
                                data = torch.load(cache_file, map_location='cpu')
                                self._set_memory_cache(key, data)
                                loaded += 1
                            except Exception as e:
                                print(f"[CACHE] Помилка завантаження {key}: {e}", flush=True)
            
            if loaded > 0:
                print(f"[CACHE] Попередньо завантажено: {loaded} елементів", flush=True)
                
        except ImportError:
            print("[CACHE] torch не доступний для завантаження", flush=True)
        
        return loaded
    
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


class VoicePromptCacheManager:
    """
    Менеджер для управління множинними кешами.
    
    Дозволяє створювати окремі кеші для різних цілей:
    - voice prompts
    - generated audio
    - reference audio
    """
    
    def __init__(self, base_dir: str = "/content/cache"):
        """
        Ініціалізація менеджера.
        
        Args:
            base_dir: Базова директорія для всіх кешів
        """
        self.base_dir = Path(base_dir)
        self._caches: Dict[str, VoicePromptCache] = {}
    
    def get_cache(self, name: str) -> VoicePromptCache:
        """
        Отримання або створення кешу за ім'ям.
        
        Args:
            name: Ім'я кешу
            
        Returns:
            VoicePromptCache: Екземпляр кешу
        """
        if name not in self._caches:
            cache_dir = self.base_dir / name
            self._caches[name] = VoicePromptCache(str(cache_dir))
        
        return self._caches[name]
    
    def clear_all(self) -> None:
        """Очищення всіх кешів."""
        for cache in self._caches.values():
            cache.clear()
    
    def get_all_stats(self) -> Dict[str, Dict]:
        """
        Отримання статистики всіх кешів.
        
        Returns:
            Dict: Статистика по кожному кешу
        """
        return {name: cache.get_stats() for name, cache in self._caches.items()}
