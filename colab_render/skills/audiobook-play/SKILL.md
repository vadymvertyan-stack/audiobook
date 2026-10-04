---
name: audiobook-play
description: Voice (озвучить / озвучити) a scene, chapter or book of an audio play such as «Эхо последнего рубежа» on a free Kaggle GPU, fix a word's stress (наголос / ударение), and put the mp3 on Google Drive. Use for any request to voice, render or re-render a chapter, fix a stress, or ask what is rendering.
---

# Audio play renderer

Everything goes through one script. Each book has a settings file, so every
command is short. You only run commands and read their JSON answer; you never
edit Python files, voices.json, the script or Kaggle settings by hand: every
change goes through a command below, so it is kept and can be undone.

```
A=/home/vadym/audiobook/colab_render/audiobook.sh
P=/home/vadym/audiobook-data/books/<slug>/book.json      # e.g. books/echo/book.json
```

`audiobook.sh` runs the renderer as the user who owns it (vadym), where the
Kaggle login and the Google Drive remote are, even when you run as root. Write
commands below as `$A ...`.

Every command prints one JSON line at the end. `"ok": true` means it worked.
On `"ok": false`, tell the user the `problems` or `error` text in one or two
plain sentences and stop. Do not retry more than once.

Facts you need:
- The GPU is on Kaggle, not on this server. You never need a GPU here and
  never look for the book text yourself: the script is already prepared.
- Book slugs: `echo` = «Эхо последнего рубежа», volume 1 (Том 1, 16 chapters,
  47 scenes). Volume 2 is not prepared; say so if asked.
- The script's parts are scenes, titled «Глава 02, сцена 1». Pass words from
  the titles to `--chapters`: `--chapters "глава 2"` (all its scenes),
  `--chapters "глава 2, сцена 1"`, several with `;`. Numbers also work
  (`--chapters 4-6`).
- Version names like "v15a" are only file names (`--tag`). The approved sound
  settings are in the book file; never try to recreate a "version".

## 1. Check before spending GPU (free, seconds)

```
$A plan --project $P --chapters "глава 2"
```

Tell the user: scenes, voices used, `estimated_audio_minutes`. If `ok` is
false, the usual cause is a speaker name that is not in voices.json: tell the
user which name. If a scene has `"narrator_only": true`, nobody has assigned
its dialogue to characters yet, so the narrator would read every line: tell the
user and ask whether to render anyway.

## 2. Render

```
$A render --project $P --chapters "глава 2" --tag v1
```

- Without `--chapters` the whole book is rendered (6-7 hours): ask first.
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

## 4. Prepare a chapter: who says each line

In chapters that are not prepared yet (`narrator_only` in `plan`), the
narrator would read the dialogue. You are the one who decides who speaks:

```
$A speech --project $P --chapters "глава 2"
```

For every scene you get `stretches`: `n`, the `speech`, the author's words
`before` and `after` it, and `speaker` (a guess, or null). Read the context
(«— сказал Добрыня», who was addressed, turn-taking) and set the speakers of
ONE scene at a time:

```
$A speech --project $P --chapters "глава 2, сцена 1" --set 1=Андрей 2=Добрыня 5=Добрыня
```

- Use only names from `cast` in the answer; `N=` (empty) gives it back to the
  narrator. A character not in the cast must be added first (section 6).
- Unsure? Leave it to the narrator rather than guess.
- When the chapter is done, rebuild the script (all scenes, seconds):

```
$A prepare --project $P
```

`narrator_only_scenes` lists scenes still without any character.

## 5. Fine-tune single lines (emotion, tempo, pause, wording)

```
$A line --project $P --chapters "глава 2, сцена 1"
```

lists the scene's lines with `id`, `voice`, `emotion`, `text`. Change one:

```
$A line --project $P --chapters "глава 2, сцена 1" --id 0012 --emotion shout
```

- `--emotion` tense | shout | whisper | sad | none. It only works if the
  voice has that emotion (voices list `Name:shout`); otherwise the calm voice
  is used.
- `--speed 0.9` (slower) .. `1.2` (faster), `--pause 800` (ms after the line),
  `--voice Name` (this line only), `--text "..."` (this line only, e.g. to
  respell one place or fix a typo).
- Edits are kept when `prepare` runs again. Re-render the scene to hear them.

## 6. New characters and their voices (ask the user before spending GPU)

Add or change a character (the description is English, about the voice:
age, pitch, manner):

```
$A character --project $P Добрыня --description "Old warrior about sixty, deep calm bass, slow and weighty" --gender male
```

Without a name it lists the cast. Then make the voice and its emotions
(15-30 minutes on Kaggle, runs in the background):

```
$A cast --project $P --characters Добрыня
```

Afterwards: set that character's lines (section 4), `prepare`, render.

## 7. Character voices for the whole cast (rare, ask the user first)

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
