---
name: audiobook-colab
description: Turn a book into a multi-voice audio play on a Colab GPU via the Colab CLI, keep each character's voice fixed, and make sure no Colab session is left burning compute units.
---

# Audiobook on Colab

Use this skill when the user sends a book or chapter (txt, fb2, epub, docx, pdf)
and asks for an audiobook or audio play, asks how many compute units are left,
or asks to stop Colab.

All GPU work goes through `orchestrate.py` in the audiobook repository, which
drives the official `colab` CLI. Never call `colab new` yourself for rendering,
and never leave a session running: an idle VM still bills compute units.

Paths below assume the repo is at `~/audiobook` and voices live in
`~/audiobook-data/voices/voices.json`. Adjust them to the real install.

## 1. Turn the text into a script

Rewrite the book into this markup, keeping the author's words exactly:

```
# Розділ 1. Назва
[voice:Диктор] Авторський текст.
[voice:Марта] — Пряма мова Марти.
[pause 1.5s]
[voice:Диктор] Знову автор.
```

Rules:
- Narration goes to `Диктор` (or the narrator name the user picked).
- Only direct speech changes the voice. Attributions like "сказала Марта" stay
  with the narrator, on their own line.
- An empty line marks a paragraph break.
- Put `+` before the stressed vowel in ambiguous words (`з+амок` / `зам+ок`),
  names and rare words. For Russian you can run RUAccent; for Ukrainian, the
  `ukrainian-word-stress` package.
- Every speaker name in the script must exist in voices.json. If a character
  has no voice yet, ask the user which existing voice to use, or ask them for
  a clean 5 to 10 second recording with its exact transcript.

Save the script as `~/audiobook-data/books/<slug>/script.txt`.

## 2. Check it (free, no GPU)

```
python3 ~/audiobook/colab_render/orchestrate.py plan ~/audiobook-data/books/<slug>/script.txt \
  --voices ~/audiobook-data/voices/voices.json
```

Read the JSON on the last line. If `ok` is false, fix `problems` before going
on. Tell the user the number of chapters, the voices used and
`estimated_audio_minutes`.

## 3. Check the budget

```
python3 ~/audiobook/colab_render/orchestrate.py usage
```

Report `balance_cu`. Before a long book, test one chapter first
(`--chapters 1`) and use its `wall_seconds` to estimate the rest.

## 4. Render

```
python3 ~/audiobook/colab_render/orchestrate.py render ~/audiobook-data/books/<slug>/script.txt \
  --voices ~/audiobook-data/voices/voices.json \
  --out ~/audiobook-data/books/<slug>/out \
  --engine omnivoice --language uk --gpu L4
```

- Use `--language ru` for Russian text.
- Use `--engine voxcpm2` only if the user asks for it.
- Use `--stress acute` only if the user asks to test stress marks. The
  default, `strip`, removes the `+` signs.
- Re-running the same command skips chapters that are already done. Use it
  after a failure instead of `--force`.
- Never pass `--keep` unless the user explicitly wants the VM left running.

Every chapter folder contains `chapter.mp3` (or `chapter.wav`), `stems/` (one
track per voice, for adding background sound) and `manifest.json`. Send the
user `chapter.mp3` for each chapter in `rendered`.

If the result has `ok: false`, send the user the `error` text in one or two
sentences, and say whether the session was stopped (it always is unless
`--keep` was used).

## 5. Watchdog (cron every 15 minutes)

```
python3 ~/audiobook/colab_render/orchestrate.py watchdog --stop
```

- `stopped` not empty: tell the user which sessions were stopped.
- `failed` not empty: tell the user. An orphan named `?` has to be stopped from
  the Colab web UI (Runtime, Manage sessions).
- `render_overdue: true`: a render has been running longer than 8 hours. Ask
  the user whether to stop it.
- Otherwise stay silent.
