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
        self.assertEqual(manifest["lines"][2]["gap_after_ms"], 700)  # paragraph
        self.assertEqual(manifest["lines"][0]["gap_after_ms"], 400)  # speaker change
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
        self.assertEqual(sorted(j for j in jobs if j.startswith("job_")), ["job_002"])

    def test_render_stops_early_without_internet(self):
        import socket
        real = socket.getaddrinfo
        socket.getaddrinfo = lambda *a, **k: (_ for _ in ()).throw(socket.gaierror(-3, "Temporary failure"))
        try:
            with self.assertRaises(render_chapter.NoInternet):
                render_chapter._pip_install("omnivoice")
        finally:
            socket.getaddrinfo = real

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


if __name__ == "__main__":
    unittest.main()
