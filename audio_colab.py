# =============================================================================
# VIBEMODLY v58.0 - GOOGLE-GENAI BUILD
# =============================================================================
# Головна зміна: Перехід на нову бібліотеку google-genai
# Модель: gemini-2.5-flash (працює з новими ключами)
# =============================================================================

import subprocess
import sys
import os

print("=" * 60, flush=True)
print("  VIBEMODLY v58.0 - GOOGLE-GENAI BUILD", flush=True)
print("=" * 60, flush=True)

# --- 1. ВСТАНОВЛЕННЯ ЗАЛЕЖНОСТЕЙ ---
print("[INSTALL] Встановлення бібліотек...", flush=True)

# НОВА бібліотека google-genai (замість застарілої google-generativeai)
subprocess.run(f"{sys.executable} -m pip install -q google-genai", shell=True)
subprocess.run(f"{sys.executable} -m pip install -q pyTelegramBotAPI", shell=True)

# PyTorch та TTS
subprocess.run(f"{sys.executable} -m pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cu118", shell=True)
subprocess.run(f"{sys.executable} -m pip install -q transformers accelerate", shell=True)
subprocess.run(f"{sys.executable} -m pip install -q soundfile pydub", shell=True)

# Для читання файлів
subprocess.run(f"{sys.executable} -m pip install -q ebooklib PyPDF2 pdfplumber beautifulsoup4 lxml python-docx", shell=True)

# FFmpeg
subprocess.run("apt-get install -y ffmpeg > /dev/null 2>&1", shell=True)

print("[INSTALL] Готово!", flush=True)

# =============================================================================
# ОСНОВНИЙ КОД
# =============================================================================

import json
import time
import torch
import numpy as np
import requests
from io import BytesIO
from pydub import AudioSegment

print("[INIT] Імпорт модулів...", flush=True)

# НОВА бібліотека
from google import genai
from google.genai import types

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

# Google Colab
from google.colab import userdata

# Константи
SAMPLE_RATE = 24000
TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
GEMINI_MODEL = "gemini-2.5-flash"  # Працююча модель!
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Глобальні змінні
tts_model = None
tts_processor = None
gemini_client = None

print(f"[INIT] Пристрій: {DEVICE}", flush=True)

# =============================================================================
# ОТРИМАННЯ КЛЮЧІВ
# =============================================================================

def get_api_keys():
    """Отримання ключів з Secrets"""
    print("[KEYS] Читання ключів...", flush=True)
    
    GEMINI_KEY = None
    TELEGRAM_TOKEN = None
    
    # Пробуємо GEMINI_API_KEY
    try:
        GEMINI_KEY = userdata.get("GEMINI_API_KEY")
        if GEMINI_KEY:
            print("[KEYS] GEMINI_API_KEY: знайдено", flush=True)
    except Exception as e:
        print(f"[KEYS] GEMINI_API_KEY: {e}", flush=True)
    
    # Пробуємо GOOGLE_API_KEY
    if not GEMINI_KEY:
        try:
            GEMINI_KEY = userdata.get("GOOGLE_API_KEY")
            if GEMINI_KEY:
                print("[KEYS] GOOGLE_API_KEY: знайдено", flush=True)
        except Exception as e:
            print(f"[KEYS] GOOGLE_API_KEY: {e}", flush=True)
    
    # Виправлення дублювання AIza
    if GEMINI_KEY:
        GEMINI_KEY = GEMINI_KEY.strip()
        if GEMINI_KEY.startswith("AIzaAIza"):
            print("[KEYS] Виправлення дублювання AIza...", flush=True)
            GEMINI_KEY = GEMINI_KEY[4:]
    
    # TELEGRAM_BOT_TOKEN
    try:
        TELEGRAM_TOKEN = userdata.get("TELEGRAM_BOT_TOKEN")
        if TELEGRAM_TOKEN:
            print("[KEYS] TELEGRAM_BOT_TOKEN: знайдено", flush=True)
    except Exception as e:
        print(f"[KEYS] TELEGRAM_BOT_TOKEN: {e}", flush=True)
    
    # Вивід статусу
    print("", flush=True)
    print("=" * 40, flush=True)
    print("СТАТУС КЛЮЧІВ:", flush=True)
    print(f"  Gemini: {'OK' if GEMINI_KEY else 'NOT FOUND'}", flush=True)
    print(f"  Telegram: {'OK' if TELEGRAM_TOKEN else 'NOT FOUND'}", flush=True)
    print("=" * 40, flush=True)
    print("", flush=True)
    
    if not GEMINI_KEY:
        raise ValueError("Додайте GEMINI_API_KEY або GOOGLE_API_KEY в Secrets!")
    if not TELEGRAM_TOKEN:
        raise ValueError("Додайте TELEGRAM_BOT_TOKEN в Secrets!")
    
    return GEMINI_KEY, TELEGRAM_TOKEN

# =============================================================================
# ІНІЦІАЛІЗАЦІЯ GEMINI (НОВА БІБЛІОТЕКА)
# =============================================================================

def init_gemini(api_key):
    """Ініціалізація Gemini з новою бібліотекою google-genai"""
    global gemini_client
    print("[GEMINI] Ініціалізація...", flush=True)
    
    try:
        gemini_client = genai.Client(api_key=api_key)
        
        # Тест моделі
        print(f"[GEMINI] Тест моделі {GEMINI_MODEL}...", flush=True)
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents="Hi"
        )
        print(f"[GEMINI] Підключено: {GEMINI_MODEL}", flush=True)
        return True
    except Exception as e:
        print(f"[GEMINI] Помилка: {e}", flush=True)
        
        # Пробуємо альтернативні моделі
        alt_models = ["gemini-2.5-flash-lite", "gemini-2.0-flash", "gemini-3-flash-preview"]
        for model_name in alt_models:
            try:
                print(f"[GEMINI] Спроба: {model_name}...", flush=True)
                response = gemini_client.models.generate_content(
                    model=model_name,
                    contents="Hi"
                )
                global GEMINI_MODEL
                GEMINI_MODEL = model_name
                print(f"[GEMINI] Підключено: {model_name}", flush=True)
                return True
            except Exception as e2:
                print(f"[GEMINI] {model_name}: помилка", flush=True)
        
        raise Exception("Gemini API недоступний! Перевірте ключ та квоту.")

# =============================================================================
# ІНІЦІАЛІЗАЦІЯ TTS
# =============================================================================

def init_tts():
    """Ініціалізація Qwen3-TTS"""
    global tts_model, tts_processor
    print(f"[TTS] Завантаження моделі {TTS_MODEL}...", flush=True)
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print("[TTS] Токенізатор...", flush=True)
        tts_processor = AutoTokenizer.from_pretrained(TTS_MODEL, trust_remote_code=True)
        
        print("[TTS] Модель (зачекайте 1-2 хв)...", flush=True)
        tts_model = AutoModelForCausalLM.from_pretrained(
            TTS_MODEL,
            trust_remote_code=True,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            device_map="auto"
        )
        tts_model.eval()
        print("[TTS] Готово!", flush=True)
        return True
    except Exception as e:
        print(f"[TTS] Помилка: {e}", flush=True)
        return False

# =============================================================================
# ГЕНЕРАЦІЯ АУДІО
# =============================================================================

def generate_audio(text, voice_desc=""):
    """Генерація аудіо"""
    global tts_model, tts_processor
    
    if tts_model is None:
        return None
    
    try:
        prompt = f"<|text|>{text}<|voice|>{voice_desc}<|start|>"
        inputs = tts_processor(prompt, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            output = tts_model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        
        if hasattr(output, "audio"):
            return output.audio
        else:
            return output[0].cpu().numpy() if hasattr(output[0], "cpu") else output[0]
    except Exception as e:
        print(f"[TTS] Помилка: {e}", flush=True)
        return None

# =============================================================================
# ЧИТАННЯ ФАЙЛІВ
# =============================================================================

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

# =============================================================================
# JSON ПАРСИНГ
# =============================================================================

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

# =============================================================================
# АНАЛІЗ ТЕКСТУ (НОВА БІБЛІОТЕКА)
# =============================================================================

def analyze_text(text):
    """Аналіз через Gemini"""
    global gemini_client, GEMINI_MODEL
    print("[GEMINI] Аналіз...", flush=True)
    
    prompt = f"""Ти - сценарист аудіо-книг. Створи JSON-сценарій.
Текст: {text[:8000]}

Формат (ТІЛЬКИ JSON):
{{
  "title": "Назва",
  "characters": [
    {{"name": "Ім'я", "voice_description": "Опис голосу українською"}}
  ],
  "scenes": [
    {{
      "dialogues": [
        {{"character": "Ім'я", "text": "Текст російською"}}
      ]
    }}
  ]
}}
"""
    
    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        result = extract_json(response.text)
        if result and "scenes" in result:
            print(f"[GEMINI] Сцен: {len(result['scenes'])}", flush=True)
            return result
        return None
    except Exception as e:
        print(f"[GEMINI] Помилка: {e}", flush=True)
        return None

# =============================================================================
# ЗБІРКА АУДІО
# =============================================================================

def build_audio(scenario, status_cb=None):
    """Збірка аудіо"""
    print("[AUDIO] Збірка...", flush=True)
    
    scenes = scenario.get("scenes", [])
    chars = scenario.get("characters", [])
    
    if not scenes:
        return None, None
    
    voice_map = {}
    for c in chars:
        voice_map[c["name"]] = c.get("voice_description", "")
    
    clips = []
    subtitles = []
    current_time = 0.0
    
    total = sum(len(s.get("dialogues", [])) for s in scenes)
    current = 0
    
    for scene in scenes:
        for dial in scene.get("dialogues", []):
            current += 1
            if status_cb:
                status_cb(f"Озвучка: {current}/{total}")
            
            char = dial.get("character", "")
            text = dial.get("text", "")
            voice = voice_map.get(char, "")
            
            audio = generate_audio(text, voice)
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
                current_time += duration + 0.5
    
    if not clips:
        return None, None
    
    final = np.concatenate(clips)
    segment = AudioSegment(
        final.tobytes(),
        frame_rate=SAMPLE_RATE,
        sample_width=2,
        channels=1
    )
    
    return segment, subtitles

# =============================================================================
# SRT
# =============================================================================

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

# =============================================================================
# TELEGRAM BOT
# =============================================================================

class Bot:
    def __init__(self, token):
        self.bot = telebot.TeleBot(token)
        self.setup()
        print("[BOT] Готово!", flush=True)
    
    def setup(self):
        @self.bot.message_handler(commands=['start'])
        def start(m):
            self.bot.reply_to(m, """VIBEMODLY v58.0

Надішли файл або текст для створення аудіо-книги.

Підтримувані формати: EPUB, PDF, TXT, HTML, DOCX, FB2""")
        
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
            status = self.bot.reply_to(m, "Аналіз...")
            self.process(m, m.text, status)
    
    def process(self, m, content, status):
        try:
            self.bot.edit_message_text("Аналіз сюжету...", m.chat.id, status.message_id)
            
            scenario = analyze_text(content)
            if not scenario:
                self.bot.edit_message_text("Помилка сценарію", m.chat.id, status.message_id)
                return
            
            chars = len(scenario.get("characters", []))
            scenes = len(scenario.get("scenes", []))
            
            self.bot.edit_message_text(
                f"Сценарій: {chars} персонажів, {scenes} сцен\nГенерація аудіо...",
                m.chat.id, status.message_id
            )
            
            def update(text):
                try:
                    self.bot.edit_message_text(text, m.chat.id, status.message_id)
                except:
                    pass
            
            audio, subs = build_audio(scenario, update)
            
            if not audio:
                self.bot.edit_message_text("Помилка аудіо", m.chat.id, status.message_id)
                return
            
            # Збереження
            buf = BytesIO()
            audio.export(buf, format="mp3")
            buf.seek(0)
            
            srt = create_srt(subs)
            srt_buf = BytesIO(srt.encode("utf-8"))
            srt_buf.seek(0)
            
            title = scenario.get("title", "audiobook")
            
            self.bot.send_audio(m.chat.id, buf, caption=f"{title}\n{len(audio)/1000:.0f} сек")
            self.bot.send_document(m.chat.id, srt_buf, caption="Субтитри")
            
            self.bot.delete_message(m.chat.id, status.message_id)
        except Exception as e:
            self.bot.reply_to(m, f"Помилка: {e}")
    
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
    print("  VIBEMODLY v58.0", flush=True)
    print("  Google-Genai Build + gemini-2.5-flash", flush=True)
    print("=" * 60, flush=True)
    print("", flush=True)
    
    # Ключі
    GEMINI_KEY, TELEGRAM_TOKEN = get_api_keys()
    
    # Gemini
    init_gemini(GEMINI_KEY)
    
    # TTS
    init_tts()
    
    # Бот
    bot = Bot(TELEGRAM_TOKEN)
    bot.run()

if __name__ == "__main__":
    main()
