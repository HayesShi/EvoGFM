"""Single-seed, same-predictor CUDA replay with offline fidelity checks."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from experiments import db_a_shared_fast as replay
from experiments.db_a_execution_suite import fidelity, write_json
from experiments.db_a_native_trace import identity


DEFAULT_METHODS = (
    "full_refresh",
    "head_fused",
    "neutronrt_adapt",
    "ours",
    "demand_only",
    "shared_lru",
)
METHODS = DEFAULT_METHODS + (
    "ours_no_scores",
    "ours_no_projections",
    "ours_rebuild_moments",
    "ours_feature_layer_one",
)


def preload(method):
    from evogfm.shared_attention_cuda import (
        CudaAuthorLinkScorer,
        CudaDemandOnlyLinkScorer,
        CudaFullRefreshLinkScorer,
        CudaFeatureLayerOneLinkScorer,
        CudaLinkScorer,
        CudaSharedLRULinkScorer,
        CudaNoScoresLinkScorer,
        CudaNoProjectionsLinkScorer,
        CudaRebuiltMomentsLinkScorer,
    )

    if method == "neutronrt_adapt":
        from evogfm.strict_dependency import load_author_recompute

        load_author_recompute()
    return {
        "full_refresh": CudaFullRefreshLinkScorer,
        "head_fused": CudaLinkScorer,
        "neutronrt_adapt": CudaAuthorLinkScorer,
        "ours": CudaLinkScorer,
        "demand_only": CudaDemandOnlyLinkScorer,
        "shared_lru": CudaSharedLRULinkScorer,
        "ours_no_scores": CudaNoScoresLinkScorer,
        "ours_no_projections": CudaNoProjectionsLinkScorer,
        "ours_rebuild_moments": CudaRebuiltMomentsLinkScorer,
        "ours_feature_layer_one": CudaFeatureLayerOneLinkScorer,
    }[method]


def worker(args):
    if args.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if args.cpu_affinity:
        import psutil

        psutil.Process().cpu_affinity(args.cpu_affinity)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("The benchmark requires a CUDA-enabled Torch environment")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats()
    path_counts = None
    if args.direct_hops:
        from evogfm.three_hop_csr import three_hop_walk_counts

        path_counts = three_hop_walk_counts
        path_counts(
            np.zeros(2, np.int32),
            np.empty(0, np.int32),
            np.zeros(1, np.int64),
            np.zeros(1, np.int64),
        )
    from evogfm.incremental import FullEdgeMoments
    from experiments.db_a_profile import (
        AllocatorProfile,
        StageProfile,
        WorkLedger,
        resident_payload,
    )

    implementation = preload(args.method)
    profile = StageProfile(torch)
    profile.install(implementation, FullEdgeMoments, args.stage_profile)
    ledger = WorkLedger() if args.work_profile else None
    replay.preload_method = lambda _: implementation
    scorer = None

    def scorer_for(
        method, implementation, model, head, raw, edges, encoded, budget, tile
    ):
        nonlocal scorer
        budget_bytes = (
            int(budget * 2**20)
            if method.startswith("ours") or method == "shared_lru"
            else 0
        )
        scorer = implementation(
            model,
            head,
            raw,
            edges,
            encoded=encoded,
            shared_budget_bytes=budget_bytes,
            tile_edges=tile,
            resident_scores=args.resident_scores,
            path_counts=path_counts,
            async_indices=args.async_indices,
            gpu_propagation=args.gpu_propagation,
            work_ledger=ledger,
            host_buffers=args.host_buffers,
            unified_device_projection=args.unified_device_projection,
        )
        return scorer

    replay.scorer_for = scorer_for
    original_sources = replay.source_record
    replay.source_record = lambda: original_sources() + [
        identity(Path(__file__)),
        identity(replay.WORKSPACE / "evogfm/shared_attention_cuda.py"),
        identity(replay.WORKSPACE / "evogfm/execution_baselines.py"),
        identity(replay.WORKSPACE / "evogfm/three_hop_csr.py"),
        identity(replay.WORKSPACE / "experiments/db_a_profile.py"),
    ]
    replay.VERSION = (
        "cuda-resident-attention-v2"
        if args.resident_scores
        else "cuda-common-attention-v1"
    )
    if args.direct_hops:
        replay.VERSION += "-direct-hop-count"
    if args.async_indices:
        replay.VERSION += "-async-indices"
    if args.gpu_propagation:
        replay.VERSION += "-gpu-propagation"
    replay.write_json = write_json
    args.role, args.defer_result = "timed", True
    args.allocator_profile = (
        AllocatorProfile(torch.cuda) if args.resource_profile else None
    )
    result = replay.worker(args)
    torch.cuda.synchronize()
    backend = (
        "CUDA FP32 attention with CPU-resident hidden buffers and full-neighbor per-row tiles"
        if args.host_buffers
        else (
            "CUDA FP32 attention and hidden states, hybrid sparse propagation; CPU topology and readout"
            if args.gpu_propagation
            else "CUDA FP32 attention and hidden states; CPU graph state, propagation and readout"
        )
    )
    result.update(
        backend=backend,
        sparse_propagation_on_cuda=args.gpu_propagation,
        cpu_affinity=args.cpu_affinity,
        gpu=torch.cuda.get_device_name(),
        cuda_version=torch.version.cuda,
        peak_gpu_allocated_bytes=(
            args.allocator_profile.peak
            if args.allocator_profile
            else torch.cuda.max_memory_allocated()
        ),
        peak_gpu_reserved_bytes=(
            args.allocator_profile.reserved_peak
            if args.allocator_profile
            else torch.cuda.max_memory_reserved()
        ),
        final_gpu_allocated_bytes=torch.cuda.memory_allocated(),
        gpu_temporary_above_stage_boundaries=(
            dict(args.allocator_profile.temporary) if args.allocator_profile else None
        ),
        scoped_stage_s=dict(profile.times),
        resident_payload=resident_payload(scorer),
        detailed_stage_profiling=args.stage_profile,
        work_diagnostics=ledger.summary() if ledger else None,
        deterministic_cuda_algorithms=args.deterministic,
        unified_device_projection=args.unified_device_projection,
        adapter=(
            "NeutronRT author planner on common backend"
            if args.method == "neutronrt_adapt"
            else None
        ),
    )
    write_json(args.output / "result.json", result)


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for method in args.methods:
        output = args.output / method
        output.mkdir(exist_ok=True)
        write_json(
            args.output / "status.json",
            {
                "phase": "running",
                "method": method,
                "completed": len(results),
                "seed": 42,
            },
        )
        with (output / "worker.log").open("w", encoding="utf-8") as log:
            command = [
                sys.executable,
                "-s",
                "-u",
                "-m",
                "experiments.db_a_cuda",
                "worker",
                "--trace",
                str(args.trace),
                "--output",
                str(output),
                "--method",
                method,
                "--budget-mib",
                str(args.budget_mib),
                "--tile-edges",
                str(args.tile_edges),
                "--threads",
                str(args.threads),
                "--repeat",
                str(args.repeat),
            ]
            if args.resident_scores:
                command.append("--resident-scores")
            if args.direct_hops:
                command.append("--direct-hops")
            if args.async_indices:
                command.append("--async-indices")
            if args.gpu_propagation:
                command.append("--gpu-propagation")
            if args.compact:
                command.append("--compact")
            if args.host_buffers:
                command.append("--host-buffers")
            if args.sorted_ingest:
                command.append("--sorted-ingest")
            if args.deterministic:
                command.append("--deterministic")
            if args.unified_device_projection:
                command.append("--unified-device-projection")
            if args.request_reuse_counters:
                command.append("--request-reuse-counters")
            if args.work_profile:
                command.append("--work-profile")
            if args.stage_profile:
                command.append("--stage-profile")
            if args.resource_profile:
                command.append("--resource-profile")
            for request_id in args.capture_request:
                command.extend(("--capture-request", str(request_id)))
            if args.arrival_reference is not None:
                command.extend(
                    (
                        "--arrival-reference",
                        str(args.arrival_reference.resolve()),
                        "--arrival-factor",
                        str(args.arrival_factor),
                    )
                )
            if args.cpu_affinity:
                command += ["--cpu-affinity", *map(str, args.cpu_affinity)]
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            write_json(
                args.output / "status.json",
                {
                    "phase": "failed",
                    "method": method,
                    "log": str(output / "worker.log"),
                },
            )
            raise RuntimeError(f"{method} failed. See {output / 'worker.log'}")
        result = json.loads((output / "result.json").read_text(encoding="utf-8"))
        row = {
            name: result[name]
            for name in (
                "dataset",
                "method",
                "seed",
                "gpu",
                "total_s",
                "scored_pairs",
                "query_p99_ms",
                "peak_gpu_allocated_bytes",
                "stages",
                "counters",
            )
        }
        reference = args.reference or args.output / "full_refresh"
        if (
            args.compact
            and reference != output
            and (reference / "probabilities.npy").is_file()
        ):
            from experiments.db_a_matrix import compare_scores

            row.update(compare_scores(args.trace, output, reference))
        elif reference != output and (reference / "queries.json").is_file():
            try:
                match = fidelity(output, reference)
                row.update(
                    {
                        key: match[key]
                        for key in (
                            "valid",
                            "max_abs",
                            "rank_position_disagreement",
                            "probability_violation_count",
                        )
                    }
                )
            except AssertionError as exc:
                match = json.loads(
                    (output / "verification.json").read_text(encoding="utf-8")
                )
                row.update(
                    valid=False, max_abs=match["max_abs"], validation_error=str(exc)
                )
        trace = json.loads(args.trace.read_text(encoding="utf-8"))
        if trace.get("groups_per_update") and not args.compact:
            row["sampled_query_quality"] = causal_metrics(trace, output)
        results.append(row)
        write_json(args.output / "summary.json", results)
        print(
            json.dumps(
                {
                    key: row[key]
                    for key in (
                        "dataset",
                        "method",
                        "total_s",
                        "peak_gpu_allocated_bytes",
                    )
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    write_json(
        args.output / "status.json",
        {"phase": "complete", "completed": len(results), "seed": 42},
    )


def causal_metrics(trace, output):
    from sklearn.metrics import roc_auc_score

    scores, destinations = [], []
    with np.load(trace["queries_path"]) as pairs:
        for request in trace["requests"]:
            key = request["array_key"]
            with np.load(output / f"{key}.npz") as prediction:
                scores.append(prediction["probability"])
            destinations.append(pairs[key][1])
    predictions = np.concatenate(scores).reshape(-1, 101)
    destinations = np.concatenate(destinations).reshape(-1, 101)
    pos, neg = predictions[:, :1], predictions[:, 1:]
    tie_first = destinations[:, 1:] < destinations[:, :1]
    rank = 1 + ((neg > pos) | ((neg == pos) & tie_first)).sum(axis=1)
    labels = np.zeros_like(predictions, dtype=np.bool_)
    labels[:, 0] = True
    return {
        "positives": len(predictions),
        "auc": float(roc_auc_score(labels.ravel(), predictions.ravel())),
        "mrr": float(np.mean(1 / rank)),
        "contiguous_candidate_order": "positive then 100 negatives",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    handlers = parser.add_subparsers(dest="action", required=True)
    for name, handler in (("worker", worker), ("run", run)):
        child = handlers.add_parser(name)
        child.set_defaults(handler=handler)
        child.add_argument("--trace", type=lambda x: Path(x).resolve(), required=True)
        child.add_argument("--output", type=lambda x: Path(x).resolve(), required=True)
        child.add_argument("--threads", type=int, default=1)
        child.add_argument("--budget-mib", type=float, default=64.0)
        child.add_argument("--tile-edges", type=int, default=32768)
        child.add_argument("--resident-scores", action="store_true")
        child.add_argument("--direct-hops", action="store_true")
        child.add_argument("--async-indices", action="store_true")
        child.add_argument("--gpu-propagation", action="store_true")
        child.add_argument("--compact", action="store_true")
        child.add_argument("--host-buffers", action="store_true")
        child.add_argument("--sorted-ingest", action="store_true")
        child.add_argument("--deterministic", action="store_true")
        child.add_argument("--unified-device-projection", action="store_true")
        child.add_argument("--request-reuse-counters", action="store_true")
        child.add_argument("--capture-request", type=int, action="append", default=[])
        child.add_argument("--repeat", type=int, default=0)
        child.add_argument("--work-profile", action="store_true")
        child.add_argument("--stage-profile", action="store_true")
        child.add_argument("--resource-profile", action="store_true")
        child.add_argument("--arrival-reference", type=Path)
        child.add_argument("--arrival-factor", type=float, default=1.0)
        child.add_argument("--cpu-affinity", nargs="+", type=int)
        if name == "worker":
            child.add_argument("--method", choices=METHODS, required=True)
        else:
            child.add_argument(
                "--methods", nargs="+", choices=METHODS, default=DEFAULT_METHODS
            )
            child.add_argument("--reference", type=lambda x: Path(x).resolve())
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
