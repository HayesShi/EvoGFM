"""Refresh workspace results after all official artifact preparation ends."""

import argparse
from pathlib import Path
import subprocess
import sys
import time

import psutil

from experiments.db_a_native import WORKSPACE
from experiments.db_a_queue import read


def watch(root, previous):
    if previous is not None:
        try:
            psutil.Process(previous).wait()
        except psutil.NoSuchProcess:
            pass
    output = WORKSPACE / "runs/a_full_gpu_20261118"
    outcomes = root / "outcomes.json"
    last = -1
    while True:
        stamp = outcomes.stat().st_mtime_ns
        state = read(root / "status.json", {})
        if stamp != last or state.get("phase") == "complete":
            try:
                subprocess.run(
                    [
                        sys.executable,
                        "-s",
                        "-m",
                        "experiments.db_a_collect",
                        "--root",
                        str(root),
                        "--output",
                        str(output),
                    ],
                    check=True,
                )
                last = stamp
            except (ValueError, OSError, subprocess.CalledProcessError) as error:
                print(f"Index refresh will retry: {error}", flush=True)
        if state.get("phase") == "complete":
            break
        time.sleep(60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=lambda value: Path(value).resolve(), required=True
    )
    parser.add_argument("--after-pid", type=int)
    args = parser.parse_args()
    watch(args.root, args.after_pid)


if __name__ == "__main__":
    main()
