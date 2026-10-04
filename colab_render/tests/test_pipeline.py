"""End-to-end tests with a fake `colab` CLI and the dummy engine (no GPU, no Google).

Run: python3 -m unittest discover -s colab_render/tests
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import orchestrate  # noqa: E402
import render_chapter  # noqa: E402

BOOK = """# Розділ 1. Ліс
[voice:Диктор] Була темна ніч. Вітер гнув сосни до самої землі.
[voice:Марта] — Ти чуєш? Хтось іде.
[pause 1.5s]
[voice:Диктор] Вона прислухалася.

[voice:Олег] — Це лише вітер.

# Розділ 2. Ранок
[voice:Диктор] Зранку все стихло.
"""


def write_ref(path, seconds=4.0, sr=24000):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"\x01\x00" * int(seconds * sr))


class BookFixture(unittest.TestCase):
    """A three-voice, two-chapter book with reference clips, and fake CLIs."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.state = os.path.join(self.tmp, "state")
        os.environ["FAKE_COLAB_ROOT"] = os.path.join(self.tmp, "vm")
        os.makedirs(os.environ["FAKE_COLAB_ROOT"])
        os.environ.pop("FAKE_COLAB_FAIL", None)
        orchestrate.STATE_DIR = self.state
        orchestrate.LOCK_PATH = os.path.join(self.state, "render.lock.json")
        voices = {}
        for i, name in enumerate(["Диктор", "Марта", "Олег"]):
            wav = os.path.join(self.tmp, f"v{i}.wav")
            write_ref(wav)
            voices[name] = {"ref_audio": os.path.basename(wav), "ref_text": f"Еталонний текст {i}.", "seed": 100 + i}
        self.voices = os.path.join(self.tmp, "voices.json")
        with open(self.voices, "w", encoding="utf-8") as f:
            json.dump(voices, f, ensure_ascii=False)
        self.book = os.path.join(self.tmp, "book.txt")
        with open(self.book, "w", encoding="utf-8") as f:
            f.write(BOOK)
        self.out = os.path.join(self.tmp, "out")
        self.fake = [sys.executable, os.path.join(HERE, "fake_colab.py")]
        fake_bin = os.path.join(self.tmp, "colab")
        with open(fake_bin, "w") as f:
            f.write(f"#!/bin/sh\nexec {sys.executable} {os.path.join(HERE, 'fake_colab.py')} \"$@\"\n")
        os.chmod(fake_bin, 0o755)
        self.colab_bin = fake_bin

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def calls(self):
        with open(os.path.join(os.environ["FAKE_COLAB_ROOT"], "calls.log")) as f:
            return [line.split()[0] for line in f]

    def render(self, *extra):
        return orchestrate.main(
            ["--colab-bin", self.colab_bin, "render", self.book, "--voices", self.voices,
             "--out", self.out, "--engine", "dummy", *extra]
        )


class PipelineTest(BookFixture):
    def test_parse_book(self):
        chapters = orchestrate.parse_book(BOOK)
        self.assertEqual([c["title"] for c in chapters], ["Розділ 1. Ліс", "Розділ 2. Ранок"])
        lines = chapters[0]["lines"]
        self.assertEqual([ln["voice"] for ln in lines], ["Диктор", "Марта", "Диктор", "Олег"])
        self.assertEqual(lines[1]["pause_after_ms"], 1500)
        self.assertTrue(lines[2]["paragraph_end"])

    def test_text_before_voice_tag_is_an_error(self):
        with self.assertRaises(ValueError):
            orchestrate.parse_book("Просто текст")
        self.assertEqual(orchestrate.parse_book("Текст", default_voice="Диктор")[0]["lines"][0]["voice"], "Диктор")

    def test_long_paragraph_is_split_on_sentences(self):
        text = "[voice:Диктор] " + " ".join(["Це речення має кілька слів."] * 30)
        lines = orchestrate.parse_book(text)[0]["lines"]
        self.assertGreater(len(lines), 1)
        self.assertTrue(all(len(ln["text"]) <= orchestrate.MAX_LINE_CHARS for ln in lines))

    def test_stress_modes(self):
        self.assertEqual(render_chapter.apply_stress_mode("прив+ет з+амок", "strip"), "привет замок")
        self.assertEqual(render_chapter.apply_stress_mode("прив+ет", "acute"), "приве́т")
        self.assertEqual(render_chapter.apply_stress_mode("прив+ет", "keep"), "прив+ет")
        self.assertEqual(render_chapter.apply_stress_mode("2+2", "strip"), "2+2")

    def test_lexicon_whole_words(self):
        lex = {"ТзОВ": "тe-зе-о-ве", "Кот": "Кіт"}
        self.assertEqual(render_chapter.apply_lexicon("Кот і Котляревський", lex), "Кіт і Котляревський")

    def test_import_qwen_scenes(self):
        src = os.path.join(self.tmp, "qwen")
        os.makedirs(src)
        for name, body in [
            ("vol1_chapter_10_scene1.md", "ID: 0001\nText: Десятая.\nSystem_Prompt: x\n"),
            ("vol1_chapter_02_scene1.md", "ID: 0001\nText: Глава вторая.\nSystem_Prompt: Calm.\n---\n"
                                          "ID: 0002\nText: — Ты слышишь?\nSystem_Prompt: Tense.\n"),
            ("notes.md", "Text: ignored"),
        ]:
            with open(os.path.join(src, name), "w", encoding="utf-8") as f:
                f.write(body)
        chapters = orchestrate.parse_book(orchestrate.convert_qwen_scenes(src, "Диктор"))
        self.assertEqual([c["title"] for c in chapters], ["Глава 02, сцена 1", "Глава 10, сцена 1"])
        self.assertEqual([(ln["voice"], ln["text"]) for ln in chapters[0]["lines"]],
                         [("Диктор", "Глава вторая."), ("Диктор", "— Ты слышишь?")])

    def test_plan_reports_missing_voice(self):
        with open(self.voices, encoding="utf-8") as f:
            v = json.load(f)
        del v["Олег"]
        with open(self.voices, "w", encoding="utf-8") as f:
            json.dump(v, f, ensure_ascii=False)
        code = orchestrate.main(["plan", self.book, "--voices", self.voices])
        self.assertEqual(code, 2)

    def test_old_voice_library_format(self):
        lib = {"entries": {"narrator": {
            "character_name": "Диктор", "reference_audio_path": "v0.wav",
            "reference_text": "Текст.", "metadata": {"seed": 42}}}}
        path = os.path.join(self.tmp, "voice_library.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(lib, f, ensure_ascii=False)
        voices = orchestrate.load_voices(path)
        self.assertEqual(voices["Диктор"]["seed"], 42)
        self.assertTrue(os.path.isabs(voices["Диктор"]["ref_audio"]))

    def test_render_end_to_end(self):
        self.assertEqual(self.render(), 0)
        ch1 = os.path.join(self.out, "01_Розділ_1_Ліс")
        with open(os.path.join(ch1, "manifest.json"), encoding="utf-8") as f:
            manifest = json.load(f)
        self.assertEqual(len(manifest["lines"]), 4)
        self.assertEqual(manifest["lines"][1]["gap_after_ms"], 1500)
        gaps = orchestrate.GAPS_MS
        self.assertEqual(manifest["lines"][2]["gap_after_ms"], gaps["paragraph"])
        self.assertEqual(manifest["lines"][0]["gap_after_ms"], gaps["dialogue"])  # narrator -> speech
        self.assertEqual({ln["seed"] for ln in manifest["lines"] if ln["voice"] == "Диктор"}, {100})
        with wave.open(os.path.join(ch1, "chapter.wav")) as w:
            chapter_frames = w.getnframes()
        self.assertAlmostEqual(chapter_frames / 24000, manifest["seconds"], places=2)
        stems = sorted(os.listdir(os.path.join(ch1, "stems")))
        self.assertEqual(len(stems), 3)
        for s in stems:
            with wave.open(os.path.join(ch1, "stems", s)) as w:
                self.assertEqual(w.getnframes(), chapter_frames)
        self.assertTrue(os.path.isdir(os.path.join(self.out, "02_Розділ_2_Ранок")))
        # One session for the whole book, and it was stopped.
        calls = self.calls()
        self.assertEqual(calls.count("new"), 1)
        self.assertEqual(calls.count("exec"), 2)
        self.assertEqual(calls[-1], "stop")
        self.assertFalse(os.path.exists(orchestrate.LOCK_PATH))

    def test_rerun_skips_finished_chapters(self):
        self.assertEqual(self.render(), 0)
        os.remove(os.path.join(os.environ["FAKE_COLAB_ROOT"], "calls.log"))
        self.assertEqual(self.render(), 0)
        self.assertFalse(os.path.exists(os.path.join(os.environ["FAKE_COLAB_ROOT"], "calls.log")))

    def test_session_is_stopped_when_render_fails(self):
        os.environ["FAKE_COLAB_FAIL"] = "exec"
        self.assertEqual(self.render(), 1)
        self.assertEqual(self.calls()[-1], "stop")
        self.assertFalse(os.path.exists(orchestrate.LOCK_PATH))

    def test_watchdog_stops_forgotten_session(self):
        self.assertEqual(self.render("--keep", "--session", "left-on"), 0)
        out = subprocess.run(
            [sys.executable, os.path.join(os.path.dirname(HERE), "orchestrate.py"),
             "--colab-bin", self.colab_bin, "watchdog", "--stop"],
            capture_output=True, text=True, env={**os.environ, "AUDIOBOOK_STATE_DIR": self.state},
        )
        report = json.loads(out.stdout.strip().splitlines()[-1])
        self.assertEqual(report["stopped"], ["left-on"])
        self.assertEqual(report["usage"]["active_assignments"], 0.0)


class KaggleTest(BookFixture):
    """Same book and voices, rendered through the fake `kaggle` CLI."""

    def setUp(self):
        super().setUp()
        os.environ["FAKE_KAGGLE_ROOT"] = os.path.join(self.tmp, "kaggle_state")
        os.environ.pop("FAKE_KAGGLE_FAIL", None)
        os.environ["KAGGLE_CONFIG_DIR"] = os.path.join(self.tmp, "no-kaggle-config")
        kaggle_bin = os.path.join(self.tmp, "kaggle")
        with open(kaggle_bin, "w") as f:
            f.write(f"#!/bin/sh\nexec {sys.executable} {os.path.join(HERE, 'fake_kaggle.py')} \"$@\"\n")
        os.chmod(kaggle_bin, 0o755)
        self.kaggle_bin = kaggle_bin

    def kaggle_calls(self):
        with open(os.path.join(os.environ["FAKE_KAGGLE_ROOT"], "calls.log")) as f:
            return [line.strip() for line in f]

    def render_kaggle(self, *extra, user="tester"):
        args = ["render", self.book, "--voices", self.voices, "--out", self.out, "--engine", "dummy",
                "--backend", "kaggle", "--kaggle-bin", self.kaggle_bin, "--poll-seconds", "0", *extra]
        if user:
            args += ["--kaggle-user", user]
        return orchestrate.main(args)

    def test_kaggle_end_to_end(self):
        self.assertEqual(self.render_kaggle(), 0)
        for name in ("01_Розділ_1_Ліс", "02_Розділ_2_Ранок"):
            self.assertTrue(os.path.exists(os.path.join(self.out, name, "chapter.wav")))
        calls = self.kaggle_calls()
        self.assertIn("datasets create", calls)
        self.assertEqual(calls.count("kernels push"), 1)  # whole book in one run
        self.assertEqual(calls[-1], "kernels output")
        # A second book version updates the same dataset instead of creating a new one.
        self.assertEqual(self.render_kaggle("--force"), 0)
        self.assertIn("datasets version", self.kaggle_calls())
        self.assertFalse(os.path.exists(orchestrate.LOCK_PATH))

    def test_kaggle_keeps_finished_chapters_when_one_fails(self):
        os.environ["FAKE_KAGGLE_FAIL"] = "chapter2"
        self.assertEqual(self.render_kaggle(), 1)
        self.assertTrue(os.path.exists(os.path.join(self.out, "01_Розділ_1_Ліс", "chapter.wav")))
        self.assertFalse(os.path.exists(os.path.join(self.out, "02_Розділ_2_Ранок")))
        # Re-run renders only the missing chapter.
        os.environ.pop("FAKE_KAGGLE_FAIL")
        os.remove(os.path.join(os.environ["FAKE_KAGGLE_ROOT"], "calls.log"))
        self.assertEqual(self.render_kaggle(), 0)
        jobs = os.listdir(os.path.join(os.environ["FAKE_KAGGLE_ROOT"], "datasets", "tester", "audiobook-jobs"))
        self.assertEqual(sorted(j for j in jobs if j.startswith("job_")), ["job_002.job"])

    def test_gaps_follow_punctuation_and_speech(self):
        # Real lines from the first scene of the test play.
        lines = [{"voice": "Д", "text": t} for t in [
            "Земля содрогалась, словно в предсмертных судорогах.",            # 0
            "Удар пришелся совсем рядом, отдаваясь тупой болью в костях.",   # 1
            "Превратившись в сплошную корку из грязи.",                      # 2 continues 1
            "Его камуфляж был покрыт пылью, стирая рисунок...",              # 3
            "Пятьдесят третий, я Скиф!",                                      # 4 speech
            "Квадрат семь ноль два накрыт плотным огнем! — прохрипел Андрей в тангенту.",  # 5 speech
            "Ответа не было.",                                               # 6
            "Ты слышишь?",                                                    # 7
            "— Ложись!",                                                      # 8 speech
        ]]
        orchestrate.assign_gaps(lines)
        g = orchestrate.GAPS_MS
        got = [ln["pause_after_ms"] for ln in lines]
        self.assertEqual(got, [g["line"], g["continue"], g["line"], g["dialogue"], g["question"],
                               g["dialogue"], g["line"], g["dialogue"], 0])

    PROSE = """# Глава 1: Пробуждение


Земля содрогалась, словно в предсмертных судорогах. Каждое попадание ста пятидесяти двух миллиметрового снаряда отдавалось глубоко в костях, выбивая из легких остатки воздуха.

Андре́й вжался в осыпающийся бруствер окопа. Его камуфляж давно потерял первоначальный рисунок, превратившись в сплошную корку из засохшей глины.

— Пятьдесят третий, я Скиф! Квадрат семь-ноль-два накрыт плотным огнем! — прохрипел Андре́й в тангенту радиостанции, прижимая гарнитуру к уху грязной перчаткой. — Противник лезет по лесополке! Дайте огня, братики, нас тут сейчас размотают!

Эфир ответил лишь шипением статических помех. РЭБ противника глушил связь намертво.

— Они нас здесь похоронят, Скиф... — прошептал он срывающимся голосом. — Мы отсюда не выйдем...

Андре́й не успел ответить. Не нашего калибра. Кто здесь?
"""

    def test_pauses_from_prose(self):
        lines = [
            "Глава первая: Пробуждение.",                                                   # 0
            "Земля содрогалась, словно в предсмертных судорогах.",                          # 1
            "Каждое попадание ста пятидесяти двух миллиметрового снаряда отдавалось глубоко в костях.",  # 2
            "Выбивая из легких остатки воздуха.",                                           # 3
            "Андрей вжался в осыпающийся бруствер окопа.",                                  # 4
            "Его камуфляж давно потерял первоначальный рисунок...",                         # 5
            "Превратившись в сплошную корку из засохшей глины.",                           # 6
            "Пятьдесят третий, я Скиф!",                                                    # 7
            "Квадрат семь ноль два накрыт плотным огнем! — прохрипел Андрей в тангенту радиостанции.",  # 8
            "Прижимая гарнитуру к уху грязной перчаткой.",                                  # 9
            "Противник лезет по лесополке! Дайте огня, братики, нас тут сейчас размотают!",  # 10
            "Эфир ответил лишь шипением статических помех. РЭБ противника глушил связь намертво.",  # 11
            "Они нас здесь похоронят, Скиф... — прошептал он срывающимся голосом.",         # 12
            "Мы отсюда не выйдем...",                                                       # 13
            "Андрей не успел ответить.",                                                    # 14
            "Не нашего калибра.",                                                           # 15
            "Кто здесь?",                                                                   # 16
        ]
        g = orchestrate.GAPS_MS
        got = orchestrate.prose_pauses(lines, self.PROSE)
        self.assertEqual(got, [
            None,                                     # spoken heading differs from "# Глава 1"
            g["line"], g["continue"], g["paragraph"],  # "…в костях," + "выбивая…" is one sentence
            g["line"], g["continue"], g["paragraph"],  # "рисунок," + "превратившись…"
            g["question"],                            # 7 -> 8: the same speech goes on
            g["continue"],                            # 8 ends in the author's words, 9 goes on with them
            g["dialogue"],                            # 9 author -> 10 speech
            g["paragraph"], g["paragraph"],
            g["dialogue"],                            # 12 author -> 13 speech
            g["paragraph"],
            g["line"], g["line"], 0,
        ])

    SCENE_LINES = [
        "Андрей вжался в осыпающийся бруствер окопа.",
        "Пятьдесят третий, я Скиф!",
        "Квадрат семь ноль два накрыт плотным огнем! — прохрипел Андрей в тангенту радиостанции.",
        "Прижимая гарнитуру к уху грязной перчаткой.",
        "Противник лезет по лесополке! Дайте огня, братики, нас тут сейчас размотают!",
        "Эфир ответил лишь шипением статических помех. РЭБ противника глушил связь намертво.",
        "Они нас здесь похоронят, Скиф... — прошептал он срывающимся голосом.",
        "Мы отсюда не выйдем...",
    ]

    def test_speech_runs_and_name_guesses(self):
        runs = orchestrate.speech_runs(self.PROSE)
        texts = [self.PROSE[a:b + 1] for a, b in runs]
        self.assertEqual(len(texts), 4)
        self.assertTrue(texts[0].startswith("Пятьдесят третий") and texts[0].endswith("огнем!"))
        self.assertTrue(texts[2].startswith("Они нас здесь похоронят"))
        cast = {"Андрей": ["Скиф"], "Лис": []}
        # 1: verb + name; 2: same paragraph; 3: "прошептал он" -> last man named; 4: same paragraph
        self.assertEqual(orchestrate.guess_speakers(self.PROSE, cast), ["Андрей"] * 4)
        with_lis = self.PROSE.replace("Эфир ответил", "Лис дрожал. Эфир ответил")
        self.assertEqual(orchestrate.guess_speakers(with_lis, cast), ["Андрей", "Андрей", "Лис", "Лис"])

    def test_guess_speakers_women_and_turn_taking(self):
        prose = ("Милонега склонилась над ним.\n\n"
                 "— Лежи. Не рвись, — тихо сказала женщина.\n\n"
                 "— Где я? — прохрипел Андрей.\n\n"
                 "— В моей избе.\n\n"
                 "— Кто ты?\n")
        cast = {"Андрей": [], "Милонега": [], "Лис": []}
        got = orchestrate.guess_speakers(prose, cast, {"Милонега": "f"})
        self.assertEqual(got, ["Милонега", "Андрей", "Милонега", "Андрей"])

    def test_attribute_asks_the_llm_and_keeps_name_guesses(self):
        import http.server
        import threading
        seen = {}

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                seen["prompt"] = body["messages"][0]["content"]
                seen["auth"] = self.headers.get("Authorization")
                reply = {"choices": [{"message": {"content": 'Sure: {"S1": "Андрей", "S2": "Андрей", '
                                                             '"S3": "Лис", "S4": "Лис", "S5": "Ромео"}'}}]}
                data = json.dumps(reply, ensure_ascii=False).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        srv = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        prose_dir = os.path.join(self.tmp, "prose")
        os.makedirs(prose_dir)
        with open(os.path.join(prose_dir, "vol1_chapter_01_scene1.md"), "w", encoding="utf-8") as f:
            f.write(self.PROSE)
        cast = os.path.join(self.tmp, "cast.json")
        with open(cast, "w", encoding="utf-8") as f:
            json.dump({"characters": {"Диктор": {"narrator": True, "voice": "Диктор", "description": "x"},
                                      "Андрей": {"aliases": ["Скиф"], "description": "x"},
                                      "Лис": {"description": "x"}}}, f, ensure_ascii=False)
        out = os.path.join(self.tmp, "speakers.json")
        os.environ["TEST_LLM_KEY"] = "secret"
        try:
            rc = orchestrate.main(["attribute", prose_dir, "--cast", cast, "--out", out,
                                   "--llm-url", f"http://127.0.0.1:{srv.server_port}/v1",
                                   "--llm-key-env", "TEST_LLM_KEY"])
        finally:
            srv.shutdown()
            os.environ.pop("TEST_LLM_KEY")
        self.assertEqual(rc, 0)
        self.assertIn("[S3]Они нас здесь похоронят", seen["prompt"])
        self.assertNotIn("- Диктор", seen["prompt"])
        self.assertEqual(seen["auth"], "Bearer secret")
        with open(out, encoding="utf-8") as f:
            entries = json.load(f)["vol1_chapter_01_scene1.md"]
        self.assertEqual([e["speaker"] for e in entries], ["Андрей", "Андрей", "Лис", "Лис"])

    def test_import_with_speakers_gives_characters_their_voices(self):
        src, prose_dir = os.path.join(self.tmp, "qwen"), os.path.join(self.tmp, "prose")
        os.makedirs(src)
        os.makedirs(prose_dir)
        name = "vol1_chapter_01_scene1.md"
        with open(os.path.join(src, name), "w", encoding="utf-8") as f:
            f.write("\n---\n".join(f"ID: {i:04d}\nText: {t}\nSystem_Prompt: "
                                    + ("Hoarse shout" if i == 3 else "Calm")
                                    for i, t in enumerate(self.SCENE_LINES, 1)))
        with open(os.path.join(prose_dir, name), "w", encoding="utf-8") as f:
            f.write(self.PROSE)
        speakers = {name: [{"n": 1, "speaker": "Андрей"}, {"n": 2, "speaker": "Андрей"},
                           {"n": 3, "speaker": "Лис"}, {"n": 4, "speaker": None}]}
        script = orchestrate.convert_qwen_scenes(src, "Диктор", prose_dir, speakers)
        lines = orchestrate.parse_book(script)[0]["lines"]
        got = [(ln["voice"], ln["text"][:24]) for ln in lines]
        self.assertIn(("Андрей", "Пятьдесят третий, я Скиф"), [(v, t[:24]) for v, t in got])
        voice_of = {t[:12]: v for v, t in got}
        self.assertEqual(voice_of["Квадрат семь"], "Андрей")
        self.assertEqual(voice_of["прохрипел Ан"], "Диктор")  # the author's words stay with the narrator
        self.assertEqual(voice_of["Противник ле"], "Андрей")
        self.assertEqual(voice_of["Они нас здес"], "Лис")
        self.assertEqual(voice_of["прошептал он"], "Диктор")
        self.assertEqual(voice_of["Мы отсюда не"], "Диктор")  # unattributed speech: narrator
        shout = next(ln for ln in lines if ln["text"].startswith("Квадрат"))
        self.assertEqual(shout.get("emotion"), "shout")
        author = next(ln for ln in lines if ln["text"].startswith("прохрипел"))
        self.assertNotIn("emotion", author)  # System_Prompt described the speech, not the narrator
        before = next(ln for ln in lines if ln["text"].startswith("Квадрат"))
        self.assertEqual(before["pause_after_ms"], orchestrate.GAPS_MS["attribution"])

    def test_agent_workflow_speech_line_prepare_character(self):
        src, prose_dir = os.path.join(self.tmp, "qwen"), os.path.join(self.tmp, "prose")
        os.makedirs(src)
        os.makedirs(prose_dir)
        name = "vol1_chapter_01_scene1.md"
        with open(os.path.join(src, name), "w", encoding="utf-8") as f:
            f.write("\n---\n".join(f"ID: {i:04d}\nText: {t}\nSystem_Prompt: Calm"
                                    for i, t in enumerate(self.SCENE_LINES, 1)))
        with open(os.path.join(prose_dir, name), "w", encoding="utf-8") as f:
            f.write(self.PROSE)
        cast = os.path.join(self.tmp, "cast.json")
        with open(cast, "w", encoding="utf-8") as f:
            json.dump({"characters": {"Диктор": {"narrator": True, "voice": "Диктор", "description": "x"},
                                      "Андрей": {"aliases": ["Скиф"], "description": "x", "seed": 1101}}},
                      f, ensure_ascii=False)
        with open(self.voices, encoding="utf-8") as f:
            vs = json.load(f)
        vs["Андрей"] = dict(vs["Диктор"])
        vs["Лис"] = dict(vs["Диктор"])
        with open(self.voices, "w", encoding="utf-8") as f:
            json.dump(vs, f, ensure_ascii=False)
        proj = os.path.join(self.tmp, "book.json")
        with open(proj, "w", encoding="utf-8") as f:
            json.dump({"book": "script.txt", "voices": self.voices, "src": "qwen", "prose": "prose",
                       "speakers": "speakers.json", "edits": "edits.json", "cast": "cast.json"}, f)
        speakers = os.path.join(self.tmp, "speakers.json")
        captured = []
        real_emit = orchestrate.emit
        orchestrate.emit = lambda obj: captured.append(obj)
        try:
            # A new character, then the agent names the speaker of stretch 3.
            self.assertEqual(orchestrate.main(["character", "--project", proj, "Лис", "--description",
                                               "young male soldier", "--gender", "male"]), 0)
            self.assertEqual(orchestrate.main(["speech", "--project", proj, "--chapters", "глава 1"]), 0)
            rows = captured[-1]["scenes"][0]["stretches"]
            self.assertEqual(len(rows), 4)
            self.assertIn("прошептал", rows[2]["after"])
            self.assertEqual(orchestrate.main(["speech", "--project", proj, "--chapters", "глава 1, сцена 1",
                                               "--set", "3=Лис", "4=Лис"]), 0)
            self.assertEqual(orchestrate.main(["speech", "--project", proj, "--chapters", "глава 1",
                                               "--set", "1=Ромео"]), 2)  # not in the cast
            self.assertEqual(orchestrate.main(["prepare", "--project", proj]), 0)
            self.assertEqual(orchestrate.main(["line", "--project", proj, "--chapters", "глава 1"]), 0)
            lines = captured[-1]["scenes"][0]["lines"]
            they = next(ln for ln in lines if ln["text"].startswith("Они нас"))
            self.assertEqual(they["voice"], "Лис")
            self.assertEqual(orchestrate.main(["line", "--project", proj, "--chapters", "глава 1, сцена 1",
                                               "--id", they["id"], "--emotion", "whisper", "--pause", "900"]), 0)
            # A re-import keeps the edit.
            self.assertEqual(orchestrate.main(["prepare", "--project", proj]), 0)
            self.assertEqual(captured[-1]["stale_edits"], [])
            self.assertEqual(orchestrate.main(["line", "--project", proj, "--chapters", "глава 1"]), 0)
            they = next(ln for ln in captured[-1]["scenes"][0]["lines"] if ln["text"].startswith("Они нас"))
            self.assertEqual((they["emotion"], they["pause_after_ms"]), ("whisper", 900))
        finally:
            orchestrate.emit = real_emit
        with open(speakers, encoding="utf-8") as f:
            self.assertEqual(json.load(f)[name][2]["how"], "agent")
        with open(cast, encoding="utf-8") as f:
            self.assertEqual(json.load(f)["characters"]["Лис"]["seed"], 1201)

    def test_merge_breaths_from_prose(self):
        lines = [
            "Земля содрогалась, словно в предсмертных судорогах.",
            "Каждое попадание ста пятидесяти двух миллиметрового снаряда отдавалось глубоко в костях.",
            "Выбивая из легких остатки воздуха.",
            "Андрей вжался в осыпающийся бруствер окопа.",
            "Его камуфляж давно потерял первоначальный рисунок...",
            "Превратившись в сплошную корку из засохшей глины.",
            "Противник лезет по лесополке! Дайте огня, братики, нас тут сейчас размотают!",
        ]
        spans, pauses = orchestrate.align_to_prose(lines, self.PROSE)
        items = [{"text": t, "tags": "", "pause": p, "span": sp} for t, p, sp in zip(lines, pauses, spans)]
        got = [it["text"] for it in orchestrate.merge_breaths(items, self.PROSE)]
        self.assertEqual(got, [
            "Земля содрогалась, словно в предсмертных судорогах. Каждое попадание ста пятидесяти двух "
            "миллиметрового снаряда отдавалось глубоко в костях, выбивая из легких остатки воздуха.",
            # The author's stress mark comes along from the prose.
            "Андре\u0301й вжался в осыпающийся бруствер окопа. Его камуфляж давно потерял первоначальный "
            "рисунок, превратившись в сплошную корку из засохшей глины.",
            "Противник лезет по лесополке! Дайте огня, братики, нас тут сейчас размотают!",
        ])

    def test_explicit_pause_and_paragraph_win(self):
        lines = [{"voice": "Д", "text": "Раз.", "pause_after_ms": 2000},
                 {"voice": "Д", "text": "Два.", "paragraph_end": True},
                 {"voice": "М", "text": "Три."}]
        orchestrate.assign_gaps(lines)
        self.assertEqual([ln["pause_after_ms"] for ln in lines], [2000, orchestrate.GAPS_MS["paragraph"], 0])

    def test_import_qwen_keeps_tempo_and_volume(self):
        src = os.path.join(self.tmp, "qwen2")
        os.makedirs(src)
        with open(os.path.join(src, "vol1_chapter_01_scene1.md"), "w", encoding="utf-8") as f:
            f.write("ID: 0001\nText: Тихо.\nSystem_Prompt: Whispering, incredibly slow, fading to silence\n---\n"
                    "ID: 0002\nText: — Огонь!\nSystem_Prompt: Very loud commanding shout\n---\n"
                    "ID: 0003\nText: Бежим.\nSystem_Prompt: Fast, urgent\n")
        lines = orchestrate.parse_book(orchestrate.convert_qwen_scenes(src, "Диктор"))[0]["lines"]
        self.assertEqual([(ln["text"], ln.get("speed"), ln.get("volume")) for ln in lines],
                         [("Тихо.", 0.82, 0.55), ("— Огонь!", 1.08, None), ("Бежим.", 1.08, None)])

    def test_finish_tail_fades_a_cut_ending_and_pads_silence(self):
        import numpy as np
        sr = 24000
        cut = np.full(sr // 2, 0.3, dtype=np.float32)  # model stopped while loud
        out = render_chapter.finish_tail(cut, sr)
        self.assertEqual(out.size, cut.size + int(sr * 0.15))
        self.assertTrue(np.all(out[cut.size:] == 0))
        self.assertLess(abs(float(out[cut.size - 1])), 0.001)
        self.assertGreater(float(out[cut.size - int(sr * 0.03)]), 0.05)  # gentle start of the fade
        # Loud, then an instant drop to silence inside the array: fade before the drop.
        dropped = np.concatenate([cut, np.zeros(sr // 10, dtype=np.float32)])
        out = render_chapter.finish_tail(dropped, sr)
        self.assertLess(abs(float(out[cut.size - 1])), 0.001)
        # A natural decay is left alone.
        decay = np.concatenate([cut, 0.3 * np.exp(-np.arange(sr // 5) / (sr * 0.02))]).astype(np.float32)
        out = render_chapter.finish_tail(decay, sr)
        self.assertTrue(np.allclose(out[:decay.size], decay))

    def test_lexicon_keeps_sentence_capital(self):
        lex = {"кобуры": "кобуры\u0301", "АК-74": "а-ка семьдесят четыре"}
        self.assertEqual(render_chapter.apply_lexicon("Кобуры и АК-74, кобуры.", lex),
                         "Кобуры\u0301 и а-ка семьдесят четыре, кобуры\u0301.")

    def test_cast_designs_characters_and_emotions(self):
        cast = os.path.join(self.tmp, "cast.json")
        with open(cast, "w", encoding="utf-8") as f:
            json.dump({"characters": {
                "Андрей": {"description": "male, 30, low husky voice"},
                "Милонега": {"description": "female, calm healer", "emotions": ["whisper"]},
                "Диктор": {"voice": "Диктор", "description": "mature male narrator", "emotions": ["tense"]},
            }}, f, ensure_ascii=False)
        with open(self.voices, encoding="utf-8") as f:
            data = json.load(f)
        data["Диктор"]["speed"] = 1.25
        with open(self.voices, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        out = os.path.join(self.tmp, "cast")
        rc = orchestrate.main(["cast", cast, "--out", out, "--voices", self.voices,
                               "--design-engine", "dummy", "--backend", "local", "--takes", "2"])
        self.assertEqual(rc, 0)
        voices = orchestrate.load_voices(self.voices)
        for name in ["Андрей", "Андрей:tense", "Андрей:shout", "Андрей:whisper", "Андрей:sad",
                     "Милонега", "Милонега:whisper", "Диктор:tense"]:
            self.assertIn(name, voices)
            self.assertTrue(os.path.exists(voices[name]["ref_audio"]), name)
        self.assertNotIn("Милонега:shout", voices)
        self.assertNotIn("Диктор:shout", voices)
        self.assertEqual(voices["Диктор:tense"]["speed"], 1.25)  # inherits the narrator's speed
        self.assertEqual(voices["Диктор"]["ref_audio"], os.path.join(self.tmp, "v0.wav"))  # untouched
        self.assertEqual(voices["Андрей:shout"]["ref_text"], orchestrate.DEFAULT_EMOTIONS["shout"]["text"])
        with open(os.path.join(out, "cast_report.json"), encoding="utf-8") as f:
            report = json.load(f)
        self.assertEqual(len(report["Андрей"]["emotions"]["shout"]["takes"]), 2)
        self.assertTrue(os.path.exists(os.path.join(out, "cast", "Андрей", "takes", "shout_2.wav")))
        self.assertTrue(report["Андрей"]["emotions"]["shout"]["chosen"]["converted_from"].endswith(".wav"))

        # Re-convert the chosen takes only: nothing is designed again.
        out2 = os.path.join(self.tmp, "cast2")
        rc = orchestrate.main(["cast", cast, "--out", out2, "--voices", self.voices, "--design-engine", "dummy",
                               "--backend", "local", "--convert-existing", os.path.join(out, "cast")])
        self.assertEqual(rc, 0)
        with open(os.path.join(out2, "cast_report.json"), encoding="utf-8") as f:
            report2 = json.load(f)
        self.assertEqual(sorted(report2["Андрей"]["emotions"]), ["sad", "shout", "tense", "whisper"])
        self.assertEqual(len(report2["Андрей"]["emotions"]["shout"]["takes"]), 1)
        self.assertNotIn("calm", report2["Андрей"]["emotions"])  # calm is reused, not redesigned
        self.assertEqual(sorted(report2["Диктор"]["emotions"]), ["tense"])

    def test_merge_voices_removes_dropped(self):
        path = os.path.join(self.tmp, "mv.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"Милонега": {"a": 1}, "Милонега:tense": {"old": 1}}, f)
        orchestrate.merge_voices(path, {"Милонега:sad": {"b": 2}}, ["Милонега:tense"])
        with open(path, encoding="utf-8") as f:
            self.assertEqual(sorted(json.load(f)), ["Милонега", "Милонега:sad"])

    def test_long_lines_are_split_at_pauses(self):
        text = ("Командный пункт затих, и только где-то далеко, за рекой, глухо ухала арта; ветер нёс над окопами "
                "едкий дым — горький, тяжёлый, бесконечный, и Андрей, прижавшись к брустверу, смотрел, как над "
                "лесом медленно поднимается серое, мутное утро.")
        pieces = render_chapter.split_for_synthesis(text, 170)
        self.assertEqual(len(pieces), 2)
        self.assertTrue(pieces[0].endswith("дым"))
        self.assertEqual(" ".join(pieces), text)
        self.assertEqual(render_chapter.split_for_synthesis("Коротко.", 170), ["Коротко."])

    def test_emotion_lines_levelled_to_calm(self):
        import numpy as np
        sr = 24000
        d = os.path.join(self.tmp, "lvl")
        os.makedirs(d)
        t = np.arange(sr) / sr
        tone = np.sin(2 * np.pi * 220 * t).astype(np.float32)
        lines = []
        for i, (voice, amp) in enumerate([("Диктор", 0.1), ("Диктор:tense", 0.3), ("Андрей:shout", 0.05)]):
            render_chapter.write_wav(os.path.join(d, f"{i}.wav"), tone * amp, sr)
            lines.append({"id": str(i), "voice": voice})
        for _ in range(2):  # idempotent
            render_chapter.level_emotion_lines(lines, d, sr)
        db = {m["id"]: render_chapter.active_rms_db(render_chapter.read_wav(os.path.join(d, f"{m['id']}.wav"))[0], sr)
              for m in lines}
        self.assertAlmostEqual(db["1"] - db["0"], 1.0, delta=0.4)   # tense: calm + 1 dB
        self.assertAlmostEqual(db["2"] - db["0"], 4.0, delta=0.4)   # shout of a voice with no calm line

    def test_soft_limit_keeps_the_gain(self):
        import numpy as np
        x = np.sin(np.linspace(0, 60, 24000)).astype(np.float32) * 1.6
        y = render_chapter.soft_limit(x)
        self.assertLessEqual(float(np.max(np.abs(y))), 0.98)
        quiet = np.abs(x) < 0.6
        self.assertTrue(np.allclose(x[quiet], y[quiet]))
        self.assertGreater(float(np.sqrt(np.mean(y ** 2))), 0.6)  # not scaled down as a whole

    def test_vc_lines_ships_calm_voice(self):
        script = "# Глава\n[voice:Диктор] [emotion tense] [speed 1.15] Тихо.\n"
        chapters = orchestrate.parse_book(script)
        voices = orchestrate.load_voices(self.voices)
        voices["Диктор:tense"] = dict(voices["Диктор"])
        orchestrate.resolve_emotions(chapters, voices)
        self.assertNotIn("speed", chapters[0]["lines"][0])  # no stacked tempo
        z = os.path.join(self.tmp, "job.zip")
        settings = {"engine": "dummy", "language": "ru", "stress": "strip", "vc_lines": "dummy"}
        orchestrate.build_job(chapters[0], voices, settings, z)
        import zipfile
        with zipfile.ZipFile(z) as f:
            job = json.loads(f.read("job.json"))
        self.assertEqual(sorted(job["voices"]), ["Диктор", "Диктор:tense"])
        self.assertEqual(job["vc_lines"], "dummy")

    def test_lexicon_key_ending_in_hyphen(self):
        lx = {"АК-": "а-ка́ ", "АК": "а-ка́"}
        self.assertEqual(render_chapter.apply_lexicon("из АК-семьдесят и АК.", lx), "из а-ка́ семьдесят и а-ка́.")
        self.assertEqual(render_chapter.apply_lexicon("ПАК-семь", lx), "ПАК-семь")

    def test_pick_closest_prefers_clean_takes(self):
        import numpy as np
        timbre = np.array([1.0, 0.0])
        takes = [{"emb": np.array([1.0, 0.0]), "ok": False}, {"emb": np.array([0.7, 0.7]), "ok": True}]
        self.assertIs(render_chapter.pick_closest(takes, timbre), takes[1])
        takes = [{"emb": np.array([1.0, 0.0]), "ok": True}, {"emb": np.array([0.9, 0.1]), "ok": True},
                 {"emb": np.array([0.0, 1.0]), "ok": True}]
        self.assertIs(render_chapter.pick_typical(takes), takes[1])

    def test_emotion_tag_picks_cast_voice(self):
        script = "# Глава\n[voice:Андрей] [emotion shout] [volume 0.55] — Огонь!\n[emotion whisper] Тихо.\n"
        chapters = orchestrate.parse_book(script)
        voices = {"Андрей": {}, "Андрей:shout": {}}
        orchestrate.resolve_emotions(chapters, voices)
        lines = chapters[0]["lines"]
        self.assertEqual(lines[0]["voice"], "Андрей:shout")
        self.assertNotIn("volume", lines[0])  # the shout reference carries its loudness
        self.assertEqual(lines[1]["voice"], "Андрей")  # no whisper reference: calm voice
        self.assertEqual(orchestrate.qwen_delivery("Very loud commanding shout"), "[emotion shout] [speed 1.08]")
        self.assertTrue(orchestrate.qwen_delivery("Whispering, incredibly slow").startswith("[emotion whisper]"))

    def test_project_file_drives_render(self):
        log = os.path.join(self.tmp, "rclone.log")
        fake = os.path.join(self.tmp, "rclone")
        with open(fake, "w") as f:
            f.write(f'#!/bin/sh\necho "$@" >> {log}\n')
        os.chmod(fake, 0o755)
        proj = os.path.join(self.tmp, "book.json")
        with open(proj, "w", encoding="utf-8") as f:
            json.dump({"book": os.path.basename(self.book), "voices": os.path.relpath(self.voices, self.tmp),
                       "commands": {"render": {"out": "proj_out", "engine": "dummy", "publish": "gdrive:B",
                                               "engine_options": {"x": 1}}}}, f)
        os.environ["RCLONE_BIN"] = fake
        try:
            rc = orchestrate.main(["--colab-bin", self.colab_bin, "render", "--project", proj, "--tag", "v2",
                                   "--chapters", "1"])
        finally:
            os.environ.pop("RCLONE_BIN")
        self.assertEqual(rc, 0)
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "proj_out", "01_Розділ_1_Ліс", "chapter.wav")))
        with open(log) as f:
            calls = f.read()
        self.assertIn("copyto", calls)
        self.assertIn("gdrive:B/01_Розділ_1_Ліс_v2.mp3", calls)
        # Nothing to render from: a clear message, not a traceback.
        self.assertEqual(orchestrate.main(["render", "--voices", self.voices]), 2)

    def test_lexicon_command(self):
        lx = os.path.join(self.tmp, "lx.json")
        proj = os.path.join(self.tmp, "p.json")
        with open(proj, "w") as f:
            json.dump({"lexicon": "lx.json"}, f)
        self.assertEqual(orchestrate.main(["lexicon", "--project", proj, "кобуры", "кабуры́"]), 0)
        self.assertEqual(orchestrate.main(["lexicon", "--project", proj, "Кобуры", "кобуры́"]), 0)
        with open(lx, encoding="utf-8") as f:
            self.assertEqual(json.load(f), {"кобуры": "кобуры́"})
        self.assertEqual(orchestrate.main(["lexicon", "--project", proj, "кобуры", "--remove"]), 0)
        with open(lx, encoding="utf-8") as f:
            self.assertEqual(json.load(f), {})

    def test_select_chapters_by_number_or_title(self):
        chs = [{"index": i, "title": t} for i, t in enumerate(
            ["Глава 01, сцена 1", "Глава 01, сцена 2", "Глава 02, сцена 1", "Глава 12, сцена 1"], 1)]
        pick = lambda spec: [c["index"] for c in orchestrate.select_chapters(chs, spec)]
        self.assertEqual(pick("1,3-4"), [1, 3, 4])
        self.assertEqual(pick("глава 2"), [3])
        self.assertEqual(pick("Глава 1, сцена 2"), [2])
        self.assertEqual(pick("глава_01_сцена_2"), [2])
        self.assertEqual(pick("глава 2; глава 12"), [3, 4])
        with self.assertRaises(ValueError):
            pick("глава 7")

    def test_render_publishes_mp3_with_rclone(self):
        log = os.path.join(self.tmp, "rclone.log")
        fake = os.path.join(self.tmp, "rclone")
        with open(fake, "w") as f:
            f.write(f'#!/bin/sh\necho "$@" >> {log}\n')
        os.chmod(fake, 0o755)
        os.environ["RCLONE_BIN"] = fake
        try:
            self.assertEqual(self.render("--publish", "gdrive:Audiobook/test/"), 0)
        finally:
            os.environ.pop("RCLONE_BIN")
        with open(log) as f:
            calls = f.read().splitlines()
        self.assertEqual(len(calls), 2)
        self.assertIn("gdrive:Audiobook/test/01_Розділ_1_Ліс --include *.mp3", calls[0])

    def test_omnivoice_gets_a_short_tail_of_extra_time(self):
        from types import SimpleNamespace
        eng = render_chapter.OmniVoiceEngine.__new__(render_chapter.OmniVoiceEngine)
        eng.tail = 0.06
        eng.model = SimpleNamespace(
            _estimate_target_tokens=lambda text, ref, n, speed=1.0: int(len(text) * 2.5 / speed),
            audio_tokenizer=SimpleNamespace(config=SimpleNamespace(frame_rate=25)))
        prompt = SimpleNamespace(ref_text="x", ref_audio_tokens=SimpleNamespace(size=lambda dim: 100))
        self.assertAlmostEqual(eng._duration_with_tail("a" * 20, prompt, 1.0), 2.0 + 0.15)   # short: 0.15 s floor
        self.assertAlmostEqual(eng._duration_with_tail("a" * 100, prompt, 1.0), 10.0 + 0.4)  # long: 0.4 s cap
        self.assertAlmostEqual(eng._duration_with_tail("a" * 50, prompt, 1.25), 4.0 + 0.24)

    def test_reference_mismatch(self):
        text = "Мой голос мужской, уверенный и ровный. Добро пожаловать в мир аудиопье́сы."
        self.assertIsNone(render_chapter.reference_mismatch(
            text, " Мой голос мужской, уверенный и ровный. Добро пожаловать в мир аудиопьесы."))
        # The real failure: the cut lost the last word.
        self.assertIn("last word", render_chapter.reference_mismatch(
            text, "Мой голос мужской, уверенный и ровный, Добро пожаловать в мир."))
        self.assertIn("first word", render_chapter.reference_mismatch(
            "Привет. " + text, text))

    def test_render_stops_early_without_internet(self):
        import socket
        real = socket.getaddrinfo
        socket.getaddrinfo = lambda *a, **k: (_ for _ in ()).throw(socket.gaierror(-3, "Temporary failure"))
        try:
            with self.assertRaises(render_chapter.NoInternet):
                render_chapter._pip_install("omnivoice")
        finally:
            socket.getaddrinfo = real

    def test_kaggle_single_chapter(self):
        # A lone .zip would be unpacked into the dataset root and missed.
        self.assertEqual(self.render_kaggle("--chapters", "2"), 0)
        self.assertTrue(os.path.exists(os.path.join(self.out, "02_Розділ_2_Ранок", "chapter.wav")))

    def test_kaggle_run_error_is_reported(self):
        os.environ["FAKE_KAGGLE_FAIL"] = "kernel"
        self.assertEqual(self.render_kaggle(), 1)

    def test_kaggle_needs_a_username(self):
        self.assertEqual(self.render_kaggle(user=None), 2)

    def test_kaggle_username_from_cli_login(self):
        os.environ["FAKE_KAGGLE_USER"] = "oauthuser"
        try:
            self.assertEqual(orchestrate.Kaggle.username(None, self.kaggle_bin), "oauthuser")
        finally:
            del os.environ["FAKE_KAGGLE_USER"]
        self.assertIsNone(orchestrate.Kaggle.username(None, self.kaggle_bin))

    def test_kaggle_username_from_config(self):
        cfg = os.environ["KAGGLE_CONFIG_DIR"]
        os.makedirs(cfg)
        with open(os.path.join(cfg, "kaggle.json"), "w") as f:
            json.dump({"username": "fromfile", "key": "x"}, f)
        self.assertEqual(orchestrate.Kaggle.username(), "fromfile")



class KaggleWaitTimeoutTest(unittest.TestCase):
    def test_hung_status_call_keeps_polling(self):
        import subprocess
        k = orchestrate.Kaggle.__new__(orchestrate.Kaggle)
        answers = [subprocess.TimeoutExpired("kaggle", 120), "running", "complete"]

        def status_word(*args):
            a = answers.pop(0)
            if isinstance(a, Exception):
                raise a
            return a
        k.status_word = status_word
        self.assertEqual(k.wait(["kernels", "status", "x"], {"complete"}, {"error"}, 0, 60), "complete")

if __name__ == "__main__":
    unittest.main()
