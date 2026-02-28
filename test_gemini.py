# Тест Gemini API ключа з новою бібліотекою google-genai
import os
from dotenv import load_dotenv

# Завантажуємо .env
load_dotenv()

print("=" * 50)
print("ТЕСТ GEMINI API КЛЮЧА (google-genai)")
print("=" * 50)

# Читаємо ключ
api_key = os.getenv("GEMINI_API_KEY")
print(f"Ключ з .env: {api_key[:10]}...{api_key[-5:]}" if api_key else "Ключ не знайдено!")

if not api_key:
    print("ПОМИЛКА: Ключ не знайдено в .env")
    exit(1)

# Тестуємо з новою бібліотекою
print("\nТестування з google-genai...")

try:
    from google import genai
    from google.genai import types
    
    client = genai.Client(api_key=api_key)
    
    # Спочатку отримаємо список моделей
    print("\nДоступні моделі:")
    try:
        for model in client.models.list():
            print(f"  - {model.name}")
    except Exception as e:
        print(f"  Не вдалося отримати список: {e}")
    
    # Пробуємо різні моделі (зі списку доступних)
    models_to_try = [
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite", 
        "gemini-2.0-flash-lite",
        "gemini-3-flash-preview",
        "gemini-2.0-flash"
    ]
    
    for model_name in models_to_try:
        try:
            print(f"\nСпроба: {model_name}...")
            response = client.models.generate_content(
                model=model_name,
                contents="Say hello in Ukrainian"
            )
            print(f"ВІДПОВІДЬ: {response.text}")
            print(f"✅ МОДЕЛЬ {model_name} ПРАЦЮЄ!")
            break
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg:
                print(f"⚠️ {model_name}: Квота вичерпана (ключ працює!)")
            elif "404" in error_msg:
                print(f"❌ {model_name}: Не знайдено")
            else:
                print(f"❌ {model_name}: {error_msg[:100]}")
    
except ImportError:
    print("Встановлюємо google-genai...")
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "google-genai"])
    print("Перезапустіть скрипт")
except Exception as e:
    print(f"ПОМИЛКА: {e}")
    print("\nМожливі причини:")
    print("1. Ключ недійсний або прострочений")
    print("2. Generative Language API не увімкнено")
    print("3. Ключ створений для іншого проекту")
    print("\nПерейдіть на: https://aistudio.google.com/app/apikey")
