"""Queue measured replacements for the five Wiki-Talk sensitivity panels."""

import argparse
from pathlib import Path

from experiments.db_a_execution_suite import write_json
from experiments.db_a_matrix import (
    cutoff,
    first_occurrences,
    load,
    mutation_batches,
    prepare_control,
    replay_jobs,
    source,
    topology_novelty,
)
from experiments.db_a_queue import plan


DATASET = "wiki-talk"
METHODS = ("ours_rebuild_moments", "neutronrt_adapt", "head_fused", "ours")


def prepare(root):
    queue = root / "queue"
    target = queue / "060_wiki-talk_sensitivity.json"
    if target.is_file():
        print(f"Already queued: {target}")
        return
    existing = {job["id"] for job in plan(queue)}
    manifest, events_path, events, nodes = source(DATASET, root)
    first = first_occurrences(events, nodes)
    novelty = topology_novelty(events, nodes)
    train, test = cutoff(events, 0.7), cutoff(events, 0.8)
    trace_root = root / "traces" / DATASET
    results = root / DATASET
    batches, initial_start = mutation_batches(
        events, nodes, test, 256, 10, False, novelty
    )
    jobs = []

    def add(tag, trace, methods=METHODS, group="10 workload sensitivity", budget=64):
        for job in replay_jobs(
            trace, results / tag, methods, group, repeats=1, budget=budget
        ):
            if job["id"] not in existing:
                jobs.append(job)
                existing.add(job["id"])

    def controlled(tag, versions, **settings):
        trace = trace_root / tag / "trace.json"
        if trace.is_file():
            return trace
        return prepare_control(
            DATASET,
            root,
            test,
            train,
            versions,
            initial_start,
            tag,
            trace.parent,
            first,
            novelty,
            events_path,
            manifest,
            nodes,
            events,
            **settings,
        )

    for size in (1, 16, 256, 4096):
        tag = f"e6_insert_{size}"
        trace = trace_root / tag / "trace.json"
        add(tag, trace, group="6 effective update batch sizes")

    for count in (1, 16, 256, 4096):
        tag = f"f7_queries_{count}"
        add(tag, controlled(tag, batches[:1], request_count=count))

    for ratio, schedule in (
        (0.01, [100] * 10),
        (0.1, [10] * 10),
        (1, [1] * 10),
        (10, [0] * 9 + [1]),
    ):
        tag = f"f7_mix_{ratio:g}"
        trace = controlled(tag, batches, requests_by_version=schedule)
        observed = load(trace)
        assert len(observed["batches"]) / len(observed["requests"]) == ratio
        add(tag, trace)

    for count in (16, 64, 256, 1024):
        tag = f"f7_candidates_{count}"
        add(
            tag,
            controlled(
                tag, batches[:4], request_count=16, candidates_per_request=count
            ),
        )

    for budget in (0, 4, 16, 64, 256):
        add(
            f"f7_cache_{budget}", trace_root / "e8_budget" / "trace.json", budget=budget
        )

    write_json(target, jobs)
    print(f"Queued {len(jobs)} matched sensitivity jobs in {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=lambda s: Path(s).resolve(), required=True)
    prepare(parser.parse_args().root)
