---
name: audiobook-play
description: Render a Russian or Ukrainian audio play (several voices, emotions) on a free Kaggle GPU with orchestrate.py, fix word stresses, and put the mp3 on Google Drive. Use when the user asks to voice a scene, chapter or book, to fix how a word is pronounced, or asks what is rendering.
---

# Audio play renderer

Everything goes through one script. Each book has a settings file, so every
command is short. You only run commands and read their JSON answer; you never
edit Python files, voices.json or Kaggle settings.

```
A=/home/vadym/audiobook/colab_render/audiobook.sh
P=/home/vadym/audiobook-data/books/<slug>/book.json      # e.g. books/echo/book.json
```

`audiobook.sh` runs the renderer as the user who owns it (vadym), where the
Kaggle login and the Google Drive remote are, even when you run as root. Write
commands below as `$A ...` .

Every command prints one JSON line at the end. `"ok": true` means it worked.
On `"ok": false`, tell the user the `problems` or `error` text in one or two
plain sentences and stop. Do not retry more than once.

## 1. Check before spending GPU (free, seconds)

```
$A plan --project $P --chapters 3
```

Tell the user: chapters, voices used, `estimated_audio_minutes`. If `ok` is
false, the usual cause is a speaker name that is not in voices.json: tell the
user which name.

## 2. Render

```
$A render --project $P --chapters 3 --tag v1
```

- `--chapters` takes numbers like `1` or `1,2,5`. Without it the whole book is
  rendered (hours; ask the user first if the book has more than 5 chapters).
- `--tag` names the uploaded file `<chapter>_<tag>.mp3` on Google Drive. Use
  the next free version (v1, v2, ...) for a re-render of the same chapter.
- The command waits until Kaggle finishes: about 5 minutes per 6 minutes of
  audio, plus 3-4 minutes start-up. Run it in the background and tell the user
  you will report when it is done.
- Running it again skips chapters that are done and unchanged. Add `--force`
  only when the user asks to redo a chapter that has not changed.
- Only one render at a time. If the answer says another render is running,
  tell the user and wait.
- Kaggle gives about 30 GPU hours a week. A whole volume is 6-7 hours.

When it finishes, tell the user the file name from `published_as` (in
`rendered`) and its length (`seconds`, as minutes:seconds). The file is on
Google Drive in the folder from the settings file.

Do not add `--vc-lines`: it converts every emotional line again and the
narrator then sounds doubled with an accent (tested, rejected by the user).

## 3. Fix a word's stress or pronunciation

The user says e.g. «кобурЫ» (capital letter = stressed vowel).

```
$A lexicon --project $P кобуры кобуры́
```

- Write the stress as the vowel plus U+0301 (the combining acute, «ы́»), the
  word in lower case.
- Never put the mark on «е» at the start of a word: the model reads «е́» as «ё».
  Respell instead: «едкой» -> «йе́дкой».
- If a mark is ignored (the user says it is still wrong), respell the sound:
  unstressed «о» -> «а» («кобуры» -> «кабуры́»), split letters with spaces or
  hyphens («АК-» -> «а-ка́ »).
- Show the current entry: `$A lexicon --project $P кобуры`.
  Remove one: add `--remove`.
- Then re-render the chapter with the next `--tag`; only lines with that
  word change, but the whole chapter is re-uploaded.

## 4. Character voices (rare, ask the user first)

Voices are already designed for the book's main characters. Only when the user
asks for a new character or a new voice:

```
$A cast --project $P --characters Имя
```

This takes 15-30 minutes on Kaggle. The character must be described in the
cast file (an English voice description). If it is not, ask the user to
describe the voice and do not run the command.

## What to tell the user

Short, in the user's language: what you started, then the result (file name,
length) or the error in plain words. No JSON, no paths unless asked.
