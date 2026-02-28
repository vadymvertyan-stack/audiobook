# Kilo Code Sound Notifications

Автоматичні звукові сповіщення при завершенні задач Kilo Code.

## Встановлення

### Варіант 1: Локальне встановлення

1. Скопіюйте папку `kilo-sound-extension` до:
   - Windows: `%USERPROFILE%\.vscode\extensions\`
   - macOS/Linux: `~/.vscode/extensions/`

2. Перезапустіть VS Code

### Варіант 2: Упаковка та встановлення

```bash
# Встановіть vsce (VS Code Extension Manager)
npm install -g @vscode/vsce

# Упакуйте розширення
cd kilo-sound-extension
vsce package

# Встановіть розширення
code --install-extension kilo-code-sound-1.0.0.vsix
```

## Налаштування

Відкрийте налаштування VS Code (`Ctrl+,`) і знайдіть "Kilo Code Sound":

| Налаштування | Опис | За замовчуванням |
|--------------|------|------------------|
| `kiloCodeSound.enabled` | Увімкнути звукові сповіщення | `true` |
| `kiloCodeSound.successSound` | Шлях до звуку успіху | `C:\Windows\Media\Windows Notify System Generic.wav` |
| `kiloCodeSound.errorSound` | Шлях до звуку помилки | `C:\Windows\Media\Windows Background.wav` |
| `kiloCodeSound.volume` | Гучність (0.0-1.0) | `1.0` |

## Команди

- `Kilo Code Sound: Test Success` - тест звуку успіху
- `Kilo Code Sound: Test Error` - тест звуку помилки

## Як це працює

Розширення відстежує:

1. **Output панель VS Code** - шукає патерни завершення задачі:
   - "Task completed", "Done", "Success", "Finished"
   - "Task failed", "Error", "Помилка"

2. **Зміни у visible editors** - для визначення сповіщень

3. **Фокус вікна** - для перевірки стану при поверненні до VS Code

## Обмеження

- VS Code не надає прямого API для відстеження сповіщень інших розширень
- Розширення використовує евристичні методи для визначення завершення задачі
- Можуть бути false positives (хибні спрацьовування)

## Альтернативний підхід

Якщо це розширення не працює належним чином, спробуйте:

1. **Гарячі клавіші** - натисніть `Ctrl+Shift+Alt+S` після завершення задачі
2. **Термінал** - виконайте `kilo_sound_success.bat`
3. **Feature request** - попросіть розробників Kilo Code додати підтримку звуків

## Ліцензія

MIT
