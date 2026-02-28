/**
 * Kilo Code Sound Notifications Extension
 * 
 * Автоматично відтворює звукові сповіщення при завершенні задач Kilo Code.
 * Відстежує сповіщення VS Code та певні патерни у Output панелі.
 */

const vscode = require('vscode');
const { exec } = require('child_process');
const path = require('path');
const fs = require('fs');

let outputListener = null;
let notificationListener = null;

/**
 * Активація розширення
 * @param {vscode.ExtensionContext} context
 */
function activate(context) {
    console.log('Kilo Code Sound Notifications: Extension activated');
    
    const config = vscode.workspace.getConfiguration('kiloCodeSound');
    
    if (config.get('enabled')) {
        setupListeners(context);
    }
    
    // Слухаємо зміни налаштувань
    context.subscriptions.push(
        vscode.workspace.onDidChangeConfiguration(e => {
            if (e.affectsConfiguration('kiloCodeSound')) {
                const newConfig = vscode.workspace.getConfiguration('kiloCodeSound');
                if (newConfig.get('enabled')) {
                    setupListeners(context);
                } else {
                    disposeListeners();
                }
            }
        })
    );
    
    // Команда для тестування звуку
    context.subscriptions.push(
        vscode.commands.registerCommand('kiloCodeSound.testSuccess', () => {
            playSound('success');
        })
    );
    
    context.subscriptions.push(
        vscode.commands.registerCommand('kiloCodeSound.testError', () => {
            playSound('error');
        })
    );
}

/**
 * Налаштування слухачів подій
 */
function setupListeners(context) {
    // Слухаємо сповіщення VS Code
    notificationListener = vscode.window.onDidChangeWindowState(e => {
        // Вікно отримало фокус - можливо, задача завершилась
        if (e.focused) {
            checkForCompletion();
        }
    });
    
    // Слухаємо повідомлення в Output панелі
    const outputChannels = vscode.window.visibleTextEditors;
    
    // Слухаємо зміни в активному редакторі
    context.subscriptions.push(
        vscode.window.onDidChangeActiveTextEditor(editor => {
            if (editor && editor.document.uri.scheme === 'output') {
                monitorOutputChannel(editor);
            }
        })
    );
    
    // Періодична перевірка стану Kilo Code
    const interval = setInterval(() => {
        checkKiloCodeStatus();
    }, 5000);
    
    context.subscriptions.push({ dispose: () => clearInterval(interval) });
    
    // Слухаємо сповіщення (notifications)
    context.subscriptions.push(
        vscode.window.onDidChangeVisibleTextEditors(editors => {
            // Перевіряємо, чи з'явилось нове сповіщення
            checkForNotifications();
        })
    );
    
    console.log('Kilo Code Sound Notifications: Listeners setup complete');
}

/**
 * Відстеження Output каналу
 */
function monitorOutputChannel(editor) {
    const document = editor.document;
    const text = document.getText();
    
    // Патерни для визначення завершення задачі Kilo Code
    const completionPatterns = [
        /Task completed/i,
        /Завершено/i,
        /Done/i,
        /Success/i,
        /Finished/i,
        /attempt_completion/i,
        /Task failed/i,
        /Error/i,
        /Помилка/i
    ];
    
    for (const pattern of completionPatterns) {
        if (pattern.test(text)) {
            if (pattern.source.includes('Error') || pattern.source.includes('Помилка') || pattern.source.includes('failed')) {
                playSound('error');
            } else {
                playSound('success');
            }
            break;
        }
    }
}

/**
 * Перевірка завершення задачі
 */
let lastCompletionTime = 0;
function checkForCompletion() {
    const now = Date.now();
    // Захист від повторних спрацьовувань
    if (now - lastCompletionTime < 3000) {
        return;
    }
    lastCompletionTime = now;
}

/**
 * Перевірка сповіщень
 */
function checkForNotifications() {
    // VS Code не надає прямого доступу до сповіщень
    // Але ми можемо відстежувати зміни у visible editors
}

/**
 * Перевірка статусу Kilo Code
 */
function checkKiloCodeStatus() {
    // Тут можна додати логіку для перевірки статусу Kilo Code
    // Наприклад, через API розширення, якщо воно його надає
}

/**
 * Відтворення звуку
 * @param {string} type - 'success' або 'error'
 */
function playSound(type) {
    const config = vscode.workspace.getConfiguration('kiloCodeSound');
    
    if (!config.get('enabled')) {
        return;
    }
    
    const soundPath = type === 'error' 
        ? config.get('errorSound')
        : config.get('successSound');
    
    // Перевіряємо, чи існує файл
    if (!fs.existsSync(soundPath)) {
        console.warn(`Sound file not found: ${soundPath}`);
        // Використовуємо системний звук
        playSystemSound(type);
        return;
    }
    
    // Відтворюємо звук через PowerShell
    const command = `powershell -Command "(New-Object Media.SoundPlayer '${soundPath}').PlaySync()"`;
    
    exec(command, (error, stdout, stderr) => {
        if (error) {
            console.error(`Error playing sound: ${error.message}`);
            playSystemSound(type);
        }
    });
}

/**
 * Відтворення системного звуку
 * @param {string} type
 */
function playSystemSound(type) {
    const command = type === 'error'
        ? 'powershell -Command "[System.Media.SystemSounds]::Hand.Play()"'
        : 'powershell -Command "[System.Media.SystemSounds]::Asterisk.Play()"';
    
    exec(command, (error) => {
        if (error) {
            console.error(`Error playing system sound: ${error.message}`);
        }
    });
}

/**
 * Видалення слухачів
 */
function disposeListeners() {
    if (outputListener) {
        outputListener.dispose();
        outputListener = null;
    }
    if (notificationListener) {
        notificationListener.dispose();
        notificationListener = null;
    }
}

/**
 * Деактивація розширення
 */
function deactivate() {
    disposeListeners();
    console.log('Kilo Code Sound Notifications: Extension deactivated');
}

module.exports = {
    activate,
    deactivate
}
