"""Run a complete paper protocol with portable paths."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from . import settings


def commands(args):
    python = sys.executable
    module = lambda name, *extra: [python, "-m", "experiments." + name, *extra]
    if args.task == "nc":
        return [module("study", "--phase", "all")]
    if args.task == "lp":
        return [module("link_prediction")]
    if args.task == "continual":
        return [
            module("cross_task", "prepare"),
            module("prepare_targets"),
            module("cross_task", "train"),
            module("evaluate_stream"),
        ]
    if args.task == "mechanisms":
        return [
            module("interventions", "--phase", phase)
            for phase in ("power", "permutation")
        ] + [module("label_shuffle")]
    # Keep graph classification's original estimator and native nested-CV readout.
    return [
        [
            python,
            str(settings.REPO / "classify_graphs.py"),
            "--checkpoint",
            str(args.checkpoint.resolve()),
            "--data-root",
            str(args.data_root.resolve()),
            "--dataset",
            dataset,
            "--shots",
            str(shot),
            "--balanced-readout",
            "--resume",
            "--output",
            str(
                args.output.resolve() / "graph_classification" / f"{dataset}_{shot}shot"
            ),
        ]
        for dataset in ("MUTAG", "PROTEINS", "NCI1", "DD", "IMDB-BINARY")
        for shot in (5, 20)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=["nc", "lp", "gc", "continual", "mechanisms"])
    parser.add_argument("--data-root", type=Path, default=settings.DATA)
    parser.add_argument("--checkpoint", type=Path, default=settings.CHECKPOINT)
    parser.add_argument("--source-splits", type=Path, default=settings.SOURCE_SPLITS)
    parser.add_argument("--output", type=Path, default=settings.ROOT)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=settings.DEVICE)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without starting experiments",
    )
    args = parser.parse_args()
    plan = commands(args)
    print(json.dumps(dict(task=args.task, commands=plan), indent=2), flush=True)
    if args.dry_run:
        return
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    env = os.environ.copy()
    env.update(
        EVOGFM_RESULTS=str(args.output.resolve()),
        EVOGFM_DATA=str(args.data_root.resolve()),
        EVOGFM_CHECKPOINT=str(args.checkpoint.resolve()),
        EVOGFM_DEVICE=args.device,
        EVOGFM_SOURCE_SPLITS=str(args.source_splits.resolve()),
    )
    for command in plan:
        subprocess.run(command, cwd=settings.REPO, env=env, check=True)


if __name__ == "__main__":
    main()
