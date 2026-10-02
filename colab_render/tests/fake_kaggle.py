#!/usr/bin/env python3
"""Stand-in for the `kaggle` CLI: datasets live under $FAKE_KAGGLE_ROOT, and
`kernels push` runs the notebook right away with the dummy engine.

FAKE_KAGGLE_FAIL=chapter2  makes the job for chapter 2 fail inside the run.
FAKE_KAGGLE_FAIL=kernel    makes the whole run end with status ERROR.
"""

import json
import os
import shutil
import subprocess
import sys

ROOT = os.environ["FAKE_KAGGLE_ROOT"]


def opt(args, name, default=None):
    return args[args.index(name) + 1] if name in args else default


def main():
    args = sys.argv[1:]
    os.makedirs(ROOT, exist_ok=True)
    with open(os.path.join(ROOT, "calls.log"), "a") as f:
        f.write(" ".join(args[:2]) + "\n")
    group, cmd = args[0], args[1]
    if group == "datasets":
        if cmd == "status":
            if not os.path.isdir(os.path.join(ROOT, "datasets", args[2])):
                print("404 Client Error: Not Found", file=sys.stderr)
                return 1
            print("ready")
        elif cmd in ("create", "version"):
            src = opt(args, "-p")
            with open(os.path.join(src, "dataset-metadata.json")) as f:
                ds_id = json.load(f)["id"]
            dst = os.path.join(ROOT, "datasets", ds_id)
            if cmd == "version" and not os.path.isdir(dst):
                return 1
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
    elif group == "kernels":
        state_path = os.path.join(ROOT, "kernel_state.json")
        if cmd == "push":
            src = opt(args, "-p")
            with open(os.path.join(src, "kernel-metadata.json")) as f:
                meta = json.load(f)
            assert meta["enable_gpu"] and meta["enable_internet"]
            working = os.path.join(ROOT, "working")
            shutil.rmtree(working, ignore_errors=True)
            os.makedirs(working)
            inputs = os.path.join(ROOT, "input")
            shutil.rmtree(inputs, ignore_errors=True)
            for ds in meta["dataset_sources"]:
                shutil.copytree(os.path.join(ROOT, "datasets", ds), os.path.join(inputs, ds.split("/")[1]))
            if os.environ.get("FAKE_KAGGLE_FAIL") == "chapter2":
                os.remove(os.path.join(inputs, ds.split("/")[1], "job_002.zip"))
                with open(os.path.join(inputs, ds.split("/")[1], "job_002.zip"), "w") as f:
                    f.write("not a zip")
            env = dict(os.environ)
            env["AUDIOBOOK_BATCH_GLOB"] = os.path.join(inputs, "**", "job_*.zip")
            env["AUDIOBOOK_BATCH_OUT"] = working
            env["AUDIOBOOK_WORKDIR"] = os.path.join(ROOT, "tmp")
            env["AUDIOBOOK_ENGINE"] = "dummy"
            with open(os.path.join(working, meta["id"].split("/")[1] + ".log"), "w") as log:
                subprocess.run([sys.executable, os.path.join(src, meta["code_file"])], env=env,
                               stdout=log, stderr=subprocess.STDOUT)
            status = "ERROR" if os.environ.get("FAKE_KAGGLE_FAIL") == "kernel" else "COMPLETE"
            with open(state_path, "w") as f:
                json.dump({"status": status, "polls": 0}, f)
            print(f"Kernel version 1 successfully pushed.  Please check progress at https://www.kaggle.com/code/{meta['id']}")
        elif cmd == "status":
            with open(state_path) as f:
                state = json.load(f)
            # First poll says RUNNING, like a real run that takes a while.
            status = "RUNNING" if state["polls"] == 0 else state["status"]
            state["polls"] += 1
            with open(state_path, "w") as f:
                json.dump(state, f)
            print(f'{args[2]} has status "KernelWorkerStatus.{status}"')
        elif cmd == "output":
            dst = opt(args, "-p")
            for name in os.listdir(os.path.join(ROOT, "working")):
                shutil.copy(os.path.join(ROOT, "working", name), dst)
    else:
        print(f"fake kaggle: unsupported {group} {cmd}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
