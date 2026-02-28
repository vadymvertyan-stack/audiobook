"""
Скрипт для відтворення звукового сповіщення про завершення задачі.
Використовується після завершення роботи AI-асистента Kilo Code.

Використання:
    python play_completion_sound.py [success|error|notify]
"""

import sys
import platform
import subprocess
from pathlib import Path


def play_sound_windows(sound_type: str = 'success') -> None:
	"""Відтворює звук на Windows через PowerShell."""
	sounds = {
		'success': 'C:\\Windows\\Media\\Windows Notify System Generic.wav',
		'error': 'C:\\Windows\\Media\\Windows Background.wav',
		'notify': 'C:\\Windows\\Media\\notify.wav',
		'complete': 'C:\\Windows\\Media\\Windows Notify Calendar.wav',
	}
	
	sound_file = sounds.get(sound_type, sounds['success'])
	
	ps_command = f"(New-Object Media.SoundPlayer '{sound_file}').PlaySync()"
	subprocess.run(
		['powershell', '-Command', ps_command],
		shell=True,
		capture_output=True
	)


def play_sound_macos(sound_type: str = 'success') -> None:
	"""Відтворює звук на macOS через afplay."""
	sounds = {
		'success': '/System/Library/Sounds/Glass.aiff',
		'error': '/System/Library/Sounds/Basso.aiff',
		'notify': '/System/Library/Sounds/Ping.aiff',
		'complete': '/System/Library/Sounds/Hero.aiff',
	}
	
	sound_file = sounds.get(sound_type, sounds['success'])
	subprocess.run(['afplay', sound_file], capture_output=True)


def play_sound_linux(sound_type: str = 'success') -> None:
	"""Відтворює звук на Linux через paplay або aplay."""
	# Спробуємо paplay (PulseAudio)
	try:
		subprocess.run(['paplay', '/usr/share/sounds/freedesktop/stereo/complete.oga'], 
		              capture_output=True)
		return
	except FileNotFoundError:
		pass
	
	# Спробуємо aplay (ALSA)
	try:
		subprocess.run(['aplay', '-q', '/usr/share/sounds/alsa/Front_Center.wav'], 
		              capture_output=True)
		return
	except FileNotFoundError:
		pass
	
	# Спробуємо spd-say (speech dispatcher)
	try:
		messages = {
			'success': 'Task completed',
			'error': 'Task failed',
			'notify': 'Notification',
			'complete': 'Done',
		}
		message = messages.get(sound_type, 'Done')
		subprocess.run(['spd-say', message], capture_output=True)
	except FileNotFoundError:
		print("No sound system available on Linux")


def play_completion_sound(sound_type: str = 'success') -> None:
	"""
	Відтворює звукове сповіщення про завершення задачі.
	
	Args:
		sound_type: Тип звуку ('success', 'error', 'notify', 'complete')
	"""
	system = platform.system()
	
	if system == 'Windows':
		play_sound_windows(sound_type)
	elif system == 'Darwin':
		play_sound_macos(sound_type)
	elif system == 'Linux':
		play_sound_linux(sound_type)
	else:
		print(f"Unsupported platform: {system}")


if __name__ == '__main__':
	sound_type = sys.argv[1] if len(sys.argv) > 1 else 'success'
	
	if sound_type not in ['success', 'error', 'notify', 'complete']:
		print(f"Unknown sound type: {sound_type}")
		print("Available types: success, error, notify, complete")
		sys.exit(1)
	
	play_completion_sound(sound_type)
