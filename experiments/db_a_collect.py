"""Verify completed A runs and maintain a compact index of measured cases."""

import argparse
from collections import defaultdict
import csv
from hashlib import sha256
import json
from pathlib import Path
from statistics import fmean, stdev

from experiments.db_a_execution_suite import write_json
from experiments.db_a_matrix import compare_scores, load
from experiments.db_a_native import WORKSPACE
from experiments.db_a_queue import plan, read
from experiments.db_a_report import write_report


def mean_sd(values):
    return {
        "mean": fmean(values),
        "sample_stddev": stdev(values) if len(values) > 1 else None,
    }


def quality_index(root, output):
    quality = root / "quality"
    preexisting = WORKSPACE / "runs/a_cuda_20261118/observed_negatives"
    cohorts = []
    for dataset in (
        "bitcoin-alpha",
        "bitcoin-otc",
        "uci",
        "wikipedia",
        "reddit-body",
        "reddit-title",
        "wiki-talk",
        "stack-overflow",
    ):
        paths = {
            "ours": quality / dataset / "ours/result.json",
            "undergs": quality / dataset / "undergs/result.json",
            "decoupled": quality / "decoupled" / dataset / "gpu/result.json",
        }
        anchored = preexisting / f"{dataset}.json"
        methods = {}
        if anchored.exists():
            for row in load(anchored)["results"]:
                name = "ours" if row["method"] == "Ours" else "undergs"
                methods[name] = {
                    "auc": row["auc"],
                    "mrr": row["mrr"],
                    "snapshots": row["snapshots"],
                    "source": str(anchored),
                    "scope": "existing strict past-observed GPU snapshot replay",
                }
        for name, path in paths.items():
            if path.is_file():
                measured = load(path)
                metric = measured.get("result", measured)
                methods[name] = {
                    "auc": metric["auc"],
                    "mrr": metric["mrr"],
                    "snapshots": metric["snapshots"],
                    "source": str(path),
                    "scope": "seed-42 causal adaptation and frozen validation selection",
                    "test_candidate_pairs": metric.get(
                        "pairs",
                        101 * metric["positives"] if "positives" in metric else None,
                    ),
                }
                if name == "decoupled":
                    representation = (
                        path.parent.parent / "representation" / "result.json"
                    )
                    if representation.is_file():
                        methods[name].update(
                            representation=load(representation)["source"],
                            representation_manifest=str(representation),
                        )
        missing = sorted(set(paths) - set(methods))
        cohorts.append(
            {
                "dataset": dataset,
                "methods": methods,
                "missing": missing,
                "currently_blocked": (
                    ["decoupled"] if dataset in ("wiki-talk", "stack-overflow") else []
                ),
            }
        )
    native_gpu = {}
    for dataset in ("bitcoin-alpha", "bitcoin-otc", "uci"):
        path = quality / "native_1000_gpu" / dataset / "result.json"
        if path.is_file():
            measured = load(path)
            native_gpu[dataset] = {
                "ours": measured["result"],
                "decoupled": measured["native_official_result"],
                "source": str(path),
                "readout": measured["readout_id"],
            }
    write_json(
        output / "quality_index.json",
        dict(
            same_100_negative_candidates_within_dataset=True,
            size_note="Prior small datasets use their locked discrete split; expanded cohorts use complete raw-time bins and event-quantile boundaries",
            native_1000_negative_path=str(
                quality / "decoupled/native_1000_negative.json"
            ),
            native_1000_gpu_replays=native_gpu,
            cohorts=cohorts,
        ),
    )
    return {row["dataset"]: sorted(row["methods"]) for row in cohorts}


def collect(root, output):
    jobs = plan(root / "queue")
    recorded = read(root / "outcomes.json", {})
    grouped = defaultdict(lambda: defaultdict(list))
    summary = defaultdict(lambda: dict(queued=0, complete=0, failed=0))
    verification, rows = {}, []
    for job in jobs:
        group = job["group"]
        if job["id"] in recorded:
            summary[group][recorded[job["id"]]["status"]] += 1
        else:
            summary[group]["queued"] += 1
        if (
            "--trace" not in job["arguments"]
            or recorded.get(job["id"], {}).get("status") != "complete"
        ):
            continue
        out = Path(job["output"])
        result = load(out / "result.json")
        trace_path = Path(job["arguments"][job["arguments"].index("--trace") + 1])
        trace = load(trace_path)
        reference = (
            Path(job["reference"])
            if "reference" in job
            else out.parent / "full_refresh"
        )
        if not (reference / "result.json").exists():
            reference = out.parent.parent / "r00" / "full_refresh"
        if not (reference / "result.json").exists() and not trace.get(
            "groups_per_update"
        ):
            reference = out.parent / "head_fused"
            if not (reference / "result.json").exists() and group.startswith("6 "):
                reference = out.parent / (
                    "ours_rebuild_moments" if out.name == "ours" else "ours"
                )
            if not (
                reference / "result.json"
            ).exists() and out.parent.parent.name.startswith("e8_budget_"):
                reference = (
                    out.parent.parent.parent
                    / "e8_budget_head_fused"
                    / "r00"
                    / "head_fused"
                )
        fidelity = None
        if (reference / "result.json").exists():
            check = out / "verification.json"
            fidelity = read(check, None)
            if fidelity is None or fidelity["reference"] != str(reference):
                fidelity = compare_scores(trace_path, out, reference)
            verification[job["id"]] = dict(
                valid=fidelity["valid"],
                path=str(check),
                method=result["method"],
                reference=str(reference),
                reference_row=out == reference,
                probability_violation_count=fidelity["probability_violation_count"],
                max_abs=fidelity["maximum_probability_abs_error"],
                disagreement=fidelity.get(
                    "rank_position_disagreement",
                    fidelity["request_rank_position_disagreement"],
                ),
            )
            if "actual_quality" in fidelity:
                verification[job["id"]].update(
                    actual_quality=fidelity["actual_quality"],
                    reference_quality=fidelity["reference_quality"],
                )
        history_backend = result.get("temporal_history_backend", "TemporalHistory")
        projection = result.get("unified_device_projection", False)
        deterministic = result.get("deterministic_cuda_algorithms", False)
        sources = result.get("source_record", [])
        revision = (
            sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()[:10]
            if sources
            else ""
        )
        row = dict(
            case=job["id"],
            group=group,
            dataset=result["dataset"],
            method=result["method"],
            condition=out.parent.parent.name,
            path=str(out / "result.json"),
            total_s=result["total_s"],
            requests=result["requests"],
            candidates=result["scored_pairs"],
            query_p50_ms=result["query_p50_ms"],
            query_p95_ms=result["query_p95_ms"],
            query_p99_ms=result["query_p99_ms"],
            initial_edges=trace["initial_edges"],
            final_edges=result["final_edges"],
            topology_versions=result["topology_versions"],
            mean_request_endpoints=fmean(
                req["distinct_endpoints"] for req in trace["requests"]
            ),
            mean_active_node_coverage=fmean(
                req["distinct_endpoints"] / max(1, req["visible_nodes"])
                for req in trace["requests"]
            ),
            processed_updates=sum(batch["raw_events"] for batch in trace["batches"]),
            effective_updates=sum(
                batch.get("effective_updates", batch["new_simple_edges"])
                for batch in trace["batches"]
            ),
            candidate_throughput=result["scored_pairs"] / result["total_s"],
            backbone_edges=result["counters"]["backbone_edges"],
            base_dot_products=result["counters"]["shared_score_evaluations"],
            peak_rss_bytes=result["peak_working_set_bytes"],
            peak_gpu_allocated_bytes=result["peak_gpu_allocated_bytes"],
            peak_gpu_reserved_bytes=result["peak_gpu_reserved_bytes"],
            final_gpu_allocated_bytes=result.get("final_gpu_allocated_bytes"),
            peak_shared_cache_bytes=result["shared_cache"]["peak_bytes"],
            temporal_history_bytes=result.get("temporal_history_bytes"),
            readout_numeric_bytes=result.get("readout_numeric_bytes"),
            resident_resources=result.get("resident_payload"),
            gpu_scratch=result.get("gpu_temporary_above_stage_boundaries"),
            request_reuse=result.get("request_reuse"),
            stages=result["stages"],
            scoped_stages=result.get("scoped_stage_s"),
            work_profile=result.get("work_diagnostics"),
            arrival=result.get("arrival"),
            instrumented_work=result.get("work_diagnostics") is not None,
            detailed_stage_profiling=result.get("detailed_stage_profiling", False),
            temporal_history_backend=history_backend,
            unified_device_projection=projection,
            implementation_revision=revision,
            deterministic_cuda_algorithms=deterministic,
            valid=None if fidelity is None else fidelity["valid"],
        )
        row["update_throughput"] = row["processed_updates"] / result["total_s"]
        if group.startswith("6 "):
            import numpy as np

            update_rows = load(out / "updates.json")
            costs = [
                item["topology_ingest_s"] + item["history_ingest_s"] + item["update_s"]
                for item in update_rows
            ]
            row.update(
                edit_commits=len(costs),
                effective_updates_per_commit=row["effective_updates"] // len(costs),
                commit_p50_ms=float(np.quantile(costs, 0.50) * 1e3),
                commit_p99_ms=float(np.quantile(costs, 0.99) * 1e3),
                effective_update_throughput=row["effective_updates"] / sum(costs),
            )
        rows.append(row)
        key = (
            group,
            result["dataset"],
            row["condition"],
            history_backend,
            projection,
            deterministic,
            revision,
        )
        grouped[key][result["method"]].append(row)

    timing = []
    for (
        group,
        dataset,
        condition,
        history_backend,
        projection,
        deterministic,
        revision,
    ), methods in grouped.items():
        anchors = methods.get("neutronrt_adapt")
        for method, samples in methods.items():
            row = dict(
                group=group,
                dataset=dataset,
                condition=condition,
                method=method,
                repeats=len(samples),
                temporal_history_backend=history_backend,
                unified_device_projection=projection,
                implementation_revision=revision,
                deterministic_cuda_algorithms=deterministic,
                valid=sum(s["valid"] is True for s in samples),
            )
            for key in ("total_s", "candidate_throughput", "update_throughput"):
                row[key] = mean_sd([sample[key] for sample in samples])
            if anchors:
                row["speedup_vs_neutronrt_adapt"] = (
                    fmean(a["total_s"] for a in anchors) / row["total_s"]["mean"]
                )
            timing.append(row)
    write_json(output / "case_index.json", rows)
    write_json(output / "timing_summary.json", timing)
    write_json(output / "fidelity_index.json", verification)
    finished_artifacts = {
        value["result"] for value in recorded.values() if value["status"] == "complete"
    }
    attempts = {
        case: value for case, value in recorded.items() if value["status"] == "failed"
    }
    recovered = {
        case: value["log"]
        for case, value in attempts.items()
        if value["result"] in finished_artifacts
    }
    unresolved = {
        case: value["log"] for case, value in attempts.items() if case not in recovered
    }
    write_json(
        output / "status.json",
        dict(
            live=read(root / "status.json", {}),
            cohort_counts=summary,
            finished_quality_methods=quality_index(root, output),
            recovered_worker_failures=recovered,
            unresolved_worker_failures=unresolved,
            probability_checks={
                "pass": sum(
                    row["valid"] and not row["reference_row"]
                    for row in verification.values()
                ),
                "fail": sum(not row["valid"] for row in verification.values()),
            },
            artifact_root=str(root),
        ),
    )
    table = [
        dict(
            {key: value for key, value in row.items() if not isinstance(value, dict)},
            **{f"stage_{key}": value for key, value in row["stages"].items()},
            **{
                f"operator_{key}": value
                for key, value in (row["scoped_stages"] or {}).items()
            },
            **{
                f"work_{key}": value
                for key, value in (row["work_profile"] or {}).items()
            },
            **{
                f"resident_{key}": value
                for key, value in (row["resident_resources"] or {}).items()
            },
            **{
                f"gpu_scratch_{key}": value
                for key, value in (row["gpu_scratch"] or {}).items()
            },
            **{
                f"request_reuse_{key}": value
                for key, value in (row["request_reuse"] or {}).items()
            },
            **{
                f"arrival_{key}": value for key, value in (row["arrival"] or {}).items()
            },
        )
        for row in rows
    ]
    columns = list(dict.fromkeys(key for row in table for key in row))
    temporary = output / "case_index.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, columns)
        writer.writeheader()
        writer.writerows(table)
    temporary.replace(output / "case_index.csv")
    report = WORKSPACE / "reports/a_experiment_plan_20261118/实验方案.md"
    if output == WORKSPACE / "runs/a_full_gpu_20261118" and report.is_file():
        write_report(root, output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    collect(args.root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
