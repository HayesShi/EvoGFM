"""Five exact strategies on one frozen predictor and one native event trace.

Run from EvoGFM. The service worker is the unchanged full-predictor replay used
for the paired sharing runs; this module supplies its method factories and
isolates each strategy in a fresh child. Reference verification is offline.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

from experiments import db_a_shared_fast as replay
from experiments.db_a_native_trace import identity, write_json as _write_json
from experiments.db_a_native import WORKSPACE, THREAD_ENV

VERSION = "exact-execution-suite-v1"
METHODS = ("full_refresh", "demand_only", "head_fused", "shared_lru", "ours")
DISPLAY = dict(
    full_refresh="FullGraph-Refresh",
    demand_only="Demand-Only",
    head_fused="Head-Fused",
    shared_lru="Shared-LRU",
    ours="Ours",
)


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    for attempt in range(40):
        try:
            return _write_json(path, value)
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(0.05)


def source_record():
    paths = [r["path"] for r in replay._execution_original_sources()]
    paths += [
        str(WORKSPACE / p)
        for p in (
            "evogfm/execution_baselines.py",
            "evogfm/incremental.py",
            "experiments/db_a_native.py",
            "experiments/db_a_execution_suite.py",
        )
    ]
    return [identity(p) for p in dict.fromkeys(paths)]


# Keep the original source resolver separate from the injected worker globals.
replay._execution_original_sources = replay.source_record


def preload_method(method):
    from evogfm.execution_baselines import (
        FullGraphRefreshLinkScorer,
        DemandOnlyLinkScorer,
        SharedLRULinkScorer,
    )
    from evogfm.shared_attention import SharedAttentionLinkScorer
    from evogfm.shared_attention_fast import SharedAttentionFastLinkScorer

    return dict(
        full_refresh=FullGraphRefreshLinkScorer,
        demand_only=DemandOnlyLinkScorer,
        head_fused=SharedAttentionLinkScorer,
        shared_lru=SharedLRULinkScorer,
        ours=SharedAttentionFastLinkScorer,
    )[method]


def scorer_for(
    method, implementation, model, head, raw, edges, encoded, budget_mib, tile_edges
):
    budget = int(budget_mib * 2**20) if method in ("shared_lru", "ours") else 0
    return implementation(
        model,
        head,
        raw,
        edges,
        encoded=encoded,
        shared_budget_bytes=budget,
        tile_edges=tile_edges,
        policy="lazy",
    )


def worker(args):
    # Factory injection is confined to this isolated process. No source file,
    # timing boundary, feature channel, readout or verification tolerance changes.
    replay.preload_method = preload_method
    replay.scorer_for = scorer_for
    replay.source_record = source_record
    replay.VERSION = VERSION
    replay.write_json = write_json
    return replay.worker(args)


def fidelity(actual, reference):
    result = replay.verify(actual, reference)
    queries = json.loads((actual / "queries.json").read_text(encoding="utf8"))
    changed, positions, rel_max = 0, 0, 0.0
    violations = 0
    for row in queries:
        name = row["prediction_file"]
        with np.load(actual / name) as a, np.load(reference / name) as b:
            av, bv = a["probability"].ravel(), b["probability"].ravel()
            ar, br = np.argsort(-av, kind="stable"), np.argsort(-bv, kind="stable")
            changed += int((ar != br).sum())
            positions += len(ar)
            if len(av):
                rel_max = max(
                    rel_max,
                    float(np.max(np.abs(av - bv) / np.maximum(np.abs(bv), 1e-12))),
                )
            violations += int((~np.isclose(av, bv, atol=1e-6, rtol=1e-6)).sum())
    result.update(
        rank_position_disagreement=changed / max(1, positions),
        probability_max_relative_error=rel_max,
        probability_violation_count=violations,
    )
    write_json(actual / "verification.json", result)
    return result


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    reference = args.reference.resolve()
    rng = np.random.default_rng(42)
    schedule = [
        dict(repeat=r, methods=rng.permutation(args.methods).tolist())
        for r in range(args.repeats)
    ]
    sources = source_record()
    settings = dict(
        seed=42,
        methods=args.methods,
        schedule=schedule,
        trace=identity(args.trace),
        reference=str(reference),
        threads=args.threads,
        budget_mib=args.budget_mib,
        tile_edges=args.tile_edges,
        source_record=sources,
        scope="Complete E0 systems replay, 50% initialization prefix; same full frozen predictor",
    )
    schedule_path = output / "schedule.json"
    if (
        schedule_path.exists()
        and json.loads(schedule_path.read_text(encoding="utf8")) != settings
    ):
        raise ValueError("Configuration or source changed; use a new output directory")
    write_json(schedule_path, settings)
    env = os.environ.copy()
    env.update({key: str(args.threads) for key in THREAD_ENV})
    results = []
    for block in schedule:
        for method in block["methods"]:
            if source_record() != sources:
                raise RuntimeError("Frozen implementation changed during execution")
            dest = output / f"r{block['repeat']:02d}_{method}"
            dest.mkdir(exist_ok=True)
            if not (dest / "result.json").exists():
                command = [
                    sys.executable,
                    "-u",
                    "-m",
                    "experiments.db_a_execution_suite",
                    "worker",
                    "--trace",
                    str(args.trace.resolve()),
                    "--output",
                    str(dest),
                    "--method",
                    method,
                    "--repeat",
                    str(block["repeat"]),
                    "--role",
                    "timed",
                    "--threads",
                    str(args.threads),
                    "--budget-mib",
                    str(args.budget_mib),
                    "--tile-edges",
                    str(args.tile_edges),
                ]
                write_json(dest / "command.json", command)
                with (dest / "worker.log").open("a", encoding="utf8") as log:
                    started = utc()
                    child = subprocess.Popen(
                        command,
                        cwd=WORKSPACE / "EvoGFM",
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                    write_json(
                        output / "active.json",
                        dict(
                            pid=os.getpid(),
                            worker_pid=child.pid,
                            phase="running",
                            method=method,
                            repeat=block["repeat"],
                            directory=str(dest),
                            updated_unix=time.time(),
                        ),
                    )
                    code = child.wait()
                write_json(
                    dest / "process.json",
                    dict(start_utc=started, end_utc=utc(), exit_code=code),
                )
                if code:
                    write_json(
                        output / "active.json",
                        dict(
                            phase="failed",
                            method=method,
                            repeat=block["repeat"],
                            directory=str(dest),
                            exit_code=code,
                        ),
                    )
                    raise RuntimeError(f"Worker failed ({code}): {dest/'worker.log'}")
            try:
                verification = fidelity(dest, reference)
            except Exception as error:
                write_json(
                    output / "active.json",
                    dict(
                        phase="validation_failed",
                        method=method,
                        repeat=block["repeat"],
                        directory=str(dest),
                        error=str(error),
                    ),
                )
                raise
            result = json.loads((dest / "result.json").read_text(encoding="utf8"))
            if result["source_record"] != sources:
                raise RuntimeError(
                    "Completed worker source differs from locked configuration"
                )
            result.update(
                valid=verification["valid"],
                max_abs=verification["max_abs"],
                rank_position_disagreement=verification["rank_position_disagreement"],
                probability_max_relative_error=verification[
                    "probability_max_relative_error"
                ],
                probability_violation_count=verification["probability_violation_count"],
                candidate_throughput=result["scored_pairs"] / result["total_s"],
            )
            results.append(result)
            write_json(
                output / "progress.json", dict(results=results, schedule=schedule)
            )
            print(
                json.dumps(
                    dict(
                        method=method,
                        repeat=block["repeat"],
                        total_s=result["total_s"],
                        valid=result["valid"],
                        max_abs=result["max_abs"],
                        rank_position_disagreement=result["rank_position_disagreement"],
                    )
                ),
                flush=True,
            )
    write_json(output / "summary.json", dict(results=results, schedule=schedule))
    write_json(output / "active.json", dict(phase="complete", updated_unix=time.time()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "worker"):
        child = sub.add_parser(name)
        child.add_argument("--trace", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)
        child.add_argument("--threads", type=int, default=1)
        child.add_argument("--budget-mib", type=float, default=64.0)
        child.add_argument("--tile-edges", type=int, default=1024)
        if name == "run":
            child.add_argument("--reference", type=Path, required=True)
            child.add_argument("--repeats", type=int, default=5)
            child.add_argument(
                "--methods", nargs="+", choices=METHODS, default=list(METHODS)
            )
        else:
            child.add_argument("--method", choices=METHODS, required=True)
            child.add_argument("--repeat", type=int, default=0)
            child.add_argument("--role", choices=("timed", "pilot"), default="pilot")
    args = parser.parse_args()
    if args.threads < 1 or args.tile_edges < 1 or args.budget_mib < 0:
        parser.error("positive threads/tile and nonnegative budget required")
    if args.command == "worker":
        worker(args)
    else:
        if args.repeats < 1 or len(args.methods) != len(set(args.methods)):
            parser.error("positive repeats and distinct methods required")
        run(args)


if __name__ == "__main__":
    main()
