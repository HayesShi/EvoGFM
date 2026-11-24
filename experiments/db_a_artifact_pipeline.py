"""When official release bytes arrive, prepare their complete GPU workloads."""

import argparse
from pathlib import Path
import subprocess
import sys
import time

from experiments.db_a_causal_trace import prepare
from experiments.db_a_download import SOURCES
from experiments.db_a_execution_suite import write_json
from experiments.db_a_matrix import SYSTEM_METHODS, controlled, replay_jobs
from experiments.db_a_native import DEFAULT_READOUT, WORKSPACE


RELEASES = {
    "reddit-body": "reddit-body.tsv",
    "reddit-title": "reddit-title.tsv",
    "stack-overflow": "stack-overflow.txt.gz",
}


def command(name, arguments, output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "preparation.log").open("a", encoding="utf-8") as log:
        subprocess.run(
            [sys.executable, "-s", "-u", "-m", name, *map(str, arguments)],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def submit(dataset, root, state):
    source_file = root / "data" / "raw" / RELEASES[dataset]
    expected = SOURCES[source_file.name][1]
    if not source_file.is_file() or source_file.stat().st_size != expected:
        return False
    destination = root / "data" / dataset
    manifest = destination / "manifest.json"
    state[dataset] = "preparing official raw release"
    write_json(root / "data" / "pipeline_status.json", state)
    if not manifest.is_file():
        command(
            "experiments.db_a_data",
            ["--dataset", dataset, "--source", source_file, "--output", destination],
            destination,
        )
    folder = root / "traces" / dataset / "causal_test"
    if not (folder / "trace.json").is_file():
        prepare(dataset, folder, 256, 256, DEFAULT_READOUT, manifest)
    jobs = replay_jobs(
        folder / "trace.json",
        root / "system" / dataset,
        SYSTEM_METHODS,
        "2/3/7/8 formal system replay",
    )
    write_json(root / "queue" / f"037_{dataset}_first_system_round.json", jobs[:6])
    write_json(root / "queue" / f"056_{dataset}_system_repeats.json", jobs[6:])
    if dataset == "stack-overflow":
        controlled(dataset, root, (6, 7, 8, 4, 5))
    return True


def watch(root, wait_limit):

    state = dict.fromkeys(RELEASES, "awaiting complete verified byte ranges")
    attempts = dict.fromkeys(RELEASES, 0)
    since = time.monotonic()
    while time.monotonic() - since < wait_limit:
        for dataset in RELEASES:
            if state[dataset] == "queued" or attempts[dataset] >= 2:
                continue
            try:
                if submit(dataset, root, state):
                    state[dataset] = "queued"
            except (ValueError, OSError, subprocess.SubprocessError) as error:
                attempts[dataset] += 1
                state[dataset] = f"failed attempt {attempts[dataset]}: {error}"
            write_json(root / "data" / "pipeline_status.json", state)
        subprocess.run(
            [
                sys.executable,
                "-s",
                "-m",
                "experiments.db_a_collect",
                "--root",
                str(root),
                "--output",
                str(WORKSPACE / "runs/a_full_gpu_20261118"),
            ],
            check=False,
        )
        if all(value == "queued" or attempts[key] >= 2 for key, value in state.items()):
            break
        time.sleep(60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=lambda x: Path(x).resolve(), required=True)
    parser.add_argument("--wait-limit", type=int, default=14400)
    args = parser.parse_args()
    watch(args.root, args.wait_limit)


if __name__ == "__main__":
    main()
