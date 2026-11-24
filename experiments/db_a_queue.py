"""Resumable serial supervision for the A GPU experiment matrix."""

import argparse
from datetime import datetime, timedelta, timezone
from fnmatch import fnmatchcase
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.db_a_execution_suite import write_json


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def read(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def gpu():
    try:
        command = [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        fields = (
            subprocess.check_output(command, text=True, timeout=8).strip().split(", ")
        )
        return dict(
            zip(
                ("device", "total_mib", "used_mib", "free_mib", "utilization_pct"),
                fields,
            )
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"error": str(error)}


def plan(queue):
    jobs = [
        case
        for path in sorted(queue.glob("[0-9][0-9][0-9]_*.json"))
        for case in read(path, [])
    ]
    if len(jobs) != len({job["id"] for job in jobs}):
        raise ValueError("Duplicate case IDs in the queue")
    return jobs


def supervise(root, idle_seconds, priority=()):
    root.mkdir(parents=True, exist_ok=True)
    queue = root / "queue"
    queue.mkdir(exist_ok=True)
    snapshots = root / "monitor.jsonl"
    results = read(root / "outcomes.json", {})
    next_snapshot = 0

    def status(jobs, case=None, process=None, peak_rss=0, phase="running", force=False):
        nonlocal next_snapshot
        counts = {
            key: sum(results.get(job["id"], {}).get("status") == key for job in jobs)
            for key in ("complete", "failed")
        }
        active = None
        if case:
            directory = Path(case["output"])
            active = dict(
                id=case["id"],
                group=case["group"],
                method=case.get("method"),
                dataset=case.get("dataset"),
                pid=process.pid if process else None,
                peak_sampled_rss_bytes=peak_rss,
                progress=read(directory / "status.json", {}),
            )
        sample = {
            "timestamp": now(),
            "timezone": "Asia/Shanghai",
            "supervisor_pid": os.getpid(),
            "phase": phase,
            "active": active,
            "completed_cases": counts["complete"],
            "failed_cases": counts["failed"],
            "queue_size": len(jobs),
            "errors": {
                key: value["log"]
                for key, value in results.items()
                if value["status"] == "failed"
            },
        }
        if force or time.monotonic() >= next_snapshot:
            sample["gpu"] = gpu()
            with snapshots.open("a", encoding="utf-8") as log:
                log.write(json.dumps(sample, ensure_ascii=False) + "\n")
            next_snapshot = time.monotonic() + 1200
        write_json(root / "status.json", sample)

    idle_since = time.monotonic()
    terminal = False
    while not terminal:
        jobs = plan(queue)
        remaining = [job for job in jobs if job["id"] not in results]
        if priority:
            rank = {
                job["id"]: next(
                    (
                        i
                        for i, pattern in enumerate(priority)
                        if fnmatchcase(job["id"], pattern)
                    ),
                    len(priority),
                )
                for job in remaining
            }
            remaining.sort(key=lambda job: rank[job["id"]])
        if not remaining:
            terminal = time.monotonic() - idle_since >= idle_seconds
            status(jobs, phase="complete" if terminal else "idle", force=terminal)
            if not terminal:
                time.sleep(5)
            continue
        case = remaining[0]
        directory = Path(case["output"])
        directory.mkdir(parents=True, exist_ok=True)
        marker = directory / case.get("marker", "result.json")
        log_path = directory / "worker.log"
        started = now()
        attempt, code = 0, 0
        if not marker.is_file():
            attempts = case.get("attempts", 1)
            while attempt < attempts:
                attempt += 1
                command = [sys.executable, "-s", "-u", *case["arguments"]]
                write_json(directory / "command.json", command)
                env = os.environ.copy()
                env.update(
                    {
                        name: "1"
                        for name in (
                            "OMP_NUM_THREADS",
                            "OPENBLAS_NUM_THREADS",
                            "MKL_NUM_THREADS",
                            "NUMEXPR_NUM_THREADS",
                        )
                    }
                )
                with log_path.open("a", encoding="utf-8") as log:
                    log.write(f"{now()} Attempt {attempt}\n")
                    log.flush()
                    process = subprocess.Popen(
                        command, stdout=log, stderr=subprocess.STDOUT, env=env
                    )
                    from psutil import NoSuchProcess, Process

                    sampler = Process(process.pid)
                    peak_rss = 0
                    while process.poll() is None:
                        try:
                            peak_rss = max(peak_rss, sampler.memory_info().rss)
                        except NoSuchProcess:
                            pass
                        status(jobs, case, process, peak_rss)
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            pass
                    code = process.returncode
                if code == 0 and marker.is_file():
                    break
        results[case["id"]] = dict(
            status="complete" if code == 0 and marker.is_file() else "failed",
            group=case["group"],
            dataset=case.get("dataset"),
            method=case.get("method"),
            result=str(marker),
            log=str(log_path),
            started=started,
            ended=now(),
            exit_code=code,
            attempts=attempt,
        )
        write_json(root / "outcomes.json", results)
        idle_since = time.monotonic()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=lambda value: Path(value).resolve(), required=True
    )
    parser.add_argument("--idle-seconds", type=int, default=14400)
    parser.add_argument(
        "--priority",
        action="append",
        default=[],
        metavar="CASE_GLOB",
        help="Run matching pending case IDs first, in pattern order.",
    )
    args = parser.parse_args()
    supervise(args.root, args.idle_seconds, args.priority)


if __name__ == "__main__":
    main()
