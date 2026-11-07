"""Run/resume evidence groups 3, 4 and 5 using seed42 and fixed protocols."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
from evogfm.data import download

STREAM = ["cora", "citeseer", "wisconsin", "cornell", "actor", "amazon-ratings"]
CONTROLS = ["static", "cumulative", "reset", "reset-curvature"]
NEW_FUTURE = ["artnet-exp", "city-reviews", "tolokers-2"]
OLD_FUTURE = ["minesweeper", "tolokers", "chameleon", "squirrel"]


def completed(path, count=None, updates=None):
    if not path.exists():
        return False
    data = json.loads(path.read_text(encoding="utf-8"))
    if count is not None:
        return len(data.get("rows", [])) == count
    trajectories = data.get("trajectories", {})
    return all(
        len(trajectories.get(kind, [])) == 7
        and (
            updates is None
            or kind == "static"
            or all(len(row["curve"]) == updates for row in trajectories[kind][1:])
        )
        for kind in CONTROLS
    )


def run(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)

    def command(script, *options):
        cmd = [
            sys.executable,
            "-u",
            script,
            "--threads",
            str(args.threads),
            *map(str, options),
        ]
        print("RUN", subprocess.list2cmdline(cmd), flush=True)
        subprocess.run(cmd, check=True)

    if "3" in args.groups:
        for order, stream in [("forward", STREAM), ("reverse4", STREAM[::-1])]:
            states = root / order
            if not all(
                completed(states / f"{task}_validation.json", updates=4)
                for task in ("nc", "lp")
            ):
                command(
                    "continual.py",
                    "--checkpoint",
                    args.checkpoint,
                    "--output",
                    states,
                    "--nc-metric",
                    "auto",
                    "--stream",
                    *stream,
                    "--future",
                    "minesweeper",
                    "tolokers",
                    "--controls",
                    *CONTROLS,
                    *(
                        ["--source-states", args.source_states]
                        if args.source_states
                        else []
                    ),
                )
            if not (states / "retention/retention.json").exists():
                command(
                    "evaluate_retention.py",
                    "--checkpoint",
                    args.checkpoint,
                    "--states",
                    states,
                    "--output",
                    states / "retention",
                )
    if "4" in args.groups or "5" in args.groups:
        datasets = NEW_FUTURE if "4" in args.groups else []
        if "5" in args.groups:
            datasets = [*datasets, "city-roads-M"]
        for name in datasets:
            download(
                f"https://zenodo.org/api/records/16895532/files/{name}.zip/content",
                Path("data/raw/graphland") / f"{name}.zip",
            )
    if "4" in args.groups:
        for order in ("forward", "reverse4"):
            states = root / order
            if not completed(states / "nc_validation.json", updates=4):
                raise ValueError("Complete group3 before evaluating group4")
            target = states / "graphland_future"
            if not completed(target / "results.json", 84):
                command(
                    "evaluate_graphland.py",
                    "--checkpoint",
                    args.checkpoint,
                    "--states",
                    states,
                    "--datasets",
                    *NEW_FUTURE,
                    "--output",
                    target,
                )
            target = states / "old_future"
            if not all(
                completed(target / f"{task}_validation.json") for task in ("nc", "lp")
            ):
                command(
                    "evaluate_future.py",
                    "--checkpoint",
                    args.checkpoint,
                    "--states",
                    states,
                    "--datasets",
                    *OLD_FUTURE,
                    "--output",
                    target,
                    "--nc-metric",
                    "auto",
                )
    if "5" in args.groups:
        for variant, folder in [
            ("none", "nc_full"),
            ("zero-inverse", "nc_zero"),
            ("propagation", "nc_propagation"),
        ]:
            target = root / folder
            if not completed(target / f"test_{variant}.json", 2):
                command(
                    "evaluate.py",
                    "--checkpoint",
                    args.checkpoint,
                    "--nc",
                    "cora",
                    "roman-empire",
                    "--lp",
                    "--node-readout",
                    "advanced",
                    "--split",
                    "test",
                    "--ablation",
                    variant,
                    "--output",
                    target,
                )
        if not completed(root / "regression/results.json", 3):
            command(
                "evaluate_graphland.py",
                "--checkpoint",
                args.checkpoint,
                "--datasets",
                "city-roads-M",
                "--output",
                root / "regression",
            )
        if not completed(root / "mutag/results.json", 4):
            command(
                "classify_fewshot.py",
                "--checkpoint",
                args.checkpoint,
                "--output",
                root / "mutag",
            )
        if not completed(root / "ppa/results.json", 4):
            command(
                "ablate_ppa.py",
                "--checkpoint",
                args.checkpoint,
                "--output",
                root / "ppa",
                "--heart-archive",
                args.heart_archive,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--groups", nargs="+", choices=["3", "4", "5"], default=["3", "4", "5"]
    )
    parser.add_argument("--checkpoint", default="runs/reference/best.pt")
    parser.add_argument(
        "--source-states", help="Optional matching source curvature cache"
    )
    parser.add_argument("--heart-archive", default="../work/heart_archive_v7")
    parser.add_argument("--output", default="runs/round345")
    parser.add_argument("--threads", type=int, default=3)
    run(parser.parse_args())
