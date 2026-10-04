#!/usr/bin/env python3
"""Stand-in for the `colab` CLI so the pipeline can be tested without Google.

A local directory ($FAKE_COLAB_ROOT) plays the VM's /content. `exec` runs the
script with the dummy engine. Every call is appended to calls.log.
"""

import json
import os
import shutil
import subprocess
import sys

ROOT = os.environ["FAKE_COLAB_ROOT"]
CONTENT = os.path.join(ROOT, "content")
STATE = os.path.join(ROOT, "sessions.json")


def load():
    try:
        with open(STATE) as f:
            return json.load(f)
    except OSError:
        return {}


def save(s):
    with open(STATE, "w") as f:
        json.dump(s, f)


def opt(args, name, default=None):
    return args[args.index(name) + 1] if name in args else default


def main():
    args = sys.argv[1:]
    os.makedirs(CONTENT, exist_ok=True)
    with open(os.path.join(ROOT, "calls.log"), "a") as f:
        f.write(" ".join(args) + "\n")
    cmd = args[0]
    sessions = load()
    if cmd == "new":
        if os.environ.get("FAKE_COLAB_FAIL") == "new":
            print("[colab] no GPU available", file=sys.stderr)
            return 1
        sessions[opt(args, "-s")] = {"gpu": opt(args, "--gpu", "CPU")}
        save(sessions)
        print("[colab] Session READY.")
    elif cmd == "upload":
        shutil.copy(args[-2], os.path.join(CONTENT, args[-1]))
    elif cmd == "download":
        shutil.copy(os.path.join(CONTENT, args[-2]), args[-1])
    elif cmd == "exec":
        # Like the real CLI, a failing script still exits 0.
        if os.environ.get("FAKE_COLAB_FAIL") == "exec":
            print("Traceback: CUDA out of memory", file=sys.stderr)
            return 0
        env = dict(os.environ)
        i = 0
        while i < len(args):
            if args[i] == "--env":
                k, v = args[i + 1].split("=", 1)
                env[k] = v.replace("/content", CONTENT, 1)
                i += 2
            else:
                i += 1
        env["AUDIOBOOK_ENGINE"] = "dummy"
        env["AUDIOBOOK_WORKDIR"] = os.path.join(CONTENT, "audiobook_job")
        subprocess.run([sys.executable, opt(args, "-f")], env=env)
        return 0
    elif cmd == "stop":
        sessions.pop(opt(args, "-s"), None)
        save(sessions)
        print("[colab] Session terminated.")
    elif cmd == "sessions":
        if not sessions:
            print("[colab] No active sessions found on server.")
        for name, s in sessions.items():
            print(f"[{name}] gpu-{name} | Hardware: {s['gpu']} | Shape: Standard | Variant: GPU")
    elif cmd == "usage":
        print(f"Current balance: 187.50 compute units\nUsage rate: {3.0 * len(sessions):.2f}/hr\nActive assignments: {len(sessions)}")
    else:
        print(f"fake colab: unsupported command {cmd}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
