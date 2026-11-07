"""Measure complete frozen-model adaptation in one independent process per invocation."""

import argparse
import importlib.util
import json
import platform
import sys
import threading
import time
from pathlib import Path

import psutil
import torch
from threadpoolctl import threadpool_limits, threadpool_info


def snapshot_modules(directory):
    """Use an explicitly saved source revision for the before/after comparison."""
    import evogfm

    for name in ("feature_views", "model", "features", "readout"):
        spec = importlib.util.spec_from_file_location(
            f"evogfm.{name}", Path(directory) / f"{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        setattr(evogfm, name, module)


def main(args):
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    cuda_enabled = "cuda" in (args.device, args.inference_device, args.kernel_device)
    if cuda_enabled:
        torch.backends.cuda.matmul.allow_tf32 = False
    if args.reference:
        snapshot_modules(args.reference)
    from evogfm.data import load_graph, load_protocol, save_json
    from evogfm.features import preprocess, node_representations
    from evogfm.model import InverseLawGFM
    from evogfm.readout import enhanced_ridge

    import evogfm.readout as readout

    fit_name = "ridge" if args.reference else "fit_ridge"
    original_fit = getattr(readout, fit_name)
    fitted_candidates = []

    def timed_fit(features, *fit_args, **fit_kwargs):
        before = time.perf_counter()
        fitted = original_fit(features, *fit_args, **fit_kwargs)
        fitted_candidates.append(
            dict(
                dimensions=features.shape[1],
                alpha_candidates=len(
                    fit_kwargs.get("alphas", (0.01, 0.1, 1.0, 10.0, 100.0))
                ),
                seconds=time.perf_counter() - before,
            )
        )
        print(
            f"candidate {features.shape[1]}D: {time.perf_counter() - before:.3f}s",
            flush=True,
        )
        return fitted

    setattr(readout, fit_name, timed_fit)
    propagation_times = []
    for function_name in ("diffuse_labels", "correct_labels"):
        original_propagation = getattr(readout, function_name)

        def timed_propagation(
            *call_args,
            _function=original_propagation,
            _name=function_name,
            **call_kwargs,
        ):
            before = time.perf_counter()
            value = _function(*call_args, **call_kwargs)
            elapsed = time.perf_counter() - before
            propagation_times.append(dict(operation=_name, seconds=elapsed))
            print(f"{_name}: {elapsed:.3f}s", flush=True)
            return value

        setattr(readout, function_name, timed_propagation)

    stages = {}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    def stage(name, function):
        before = time.perf_counter()
        value = function()
        if args.device == "cuda":
            torch.cuda.synchronize()
        stages[name] = time.perf_counter() - before
        print(f"{name}: {stages[name]:.3f}s", flush=True)
        return value

    graph = stage("data_load", lambda: load_graph(args.dataset, args.data_root))
    protocol = load_protocol(args.dataset, "nc", args.data_root, graph)
    train, val, test = (protocol[k] for k in ("train", "val", "test"))
    if args.shots:
        generator = torch.Generator().manual_seed(42)
        selected = []
        for label in graph["y"][train].unique(sorted=True):
            ids = train[graph["y"][train] == label]
            selected.append(
                ids[torch.randperm(len(ids), generator=generator)[: args.shots]]
            )
        train = torch.cat(selected)
    available_labels = len(train)
    if args.holdout_shots and args.method == "evogfm":
        fitting, heldout = [], []
        for label in graph["y"][train].unique(sorted=True):
            ids = train[graph["y"][train] == label]
            if len(ids) <= args.holdout_shots:
                raise ValueError("Holdout must leave training nodes in every class")
            fitting.append(ids[: -args.holdout_shots])
            heldout.append(ids[-args.holdout_shots :])
        train, val = torch.cat(fitting), torch.cat(heldout)
    if args.method == "evogfm":
        checkpoint = stage(
            "checkpoint_load",
            lambda: torch.load(args.checkpoint, map_location="cpu", weights_only=False),
        )
        model = InverseLawGFM(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["model"])
        model.eval().requires_grad_(False).to(args.device)
    else:
        from evogfm.graphany_benchmark import load_graphany, graphany_predict

        model = stage(
            "checkpoint_load", lambda: load_graphany(args.graphany_root, args.device)
        )

    process = psutil.Process()
    peak = [process.memory_info().rss]
    stop = threading.Event()

    def sample_memory():
        while not stop.wait(0.02):
            peak[0] = max(peak[0], process.memory_info().rss)

    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    adaptation = time.perf_counter()
    if cuda_enabled:
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        if args.method == "evogfm":
            raw, x = stage("feature_preprocessing", lambda: preprocess(graph))
            state = stage(
                "relation_statistics",
                lambda: model.state(
                    x.to(args.device), protocol["edges"].to(args.device)
                ),
            )
            parts, values, transition = stage(
                "backbone_inference",
                lambda: node_representations(
                    model,
                    state,
                    raw,
                    protocol["edges"],
                    hops=4,
                    return_transition=True,
                    **(
                        {"inference_device": args.inference_device}
                        if not args.reference
                        else {}
                    ),
                ),
            )
            scores, info = stage(
                "readout_fit_and_selection",
                lambda: enhanced_ridge(
                    parts,
                    values,
                    graph["y"],
                    train,
                    val,
                    transition,
                    advanced=True,
                    **(
                        {"kernel_device": args.kernel_device}
                        if not args.reference
                        else {}
                    ),
                ),
            )
            classes = graph["y"][train].unique(sorted=True)
        else:
            scores, info = stage(
                "graphany_pipeline",
                lambda: graphany_predict(
                    model,
                    graph,
                    train,
                    args.graphany_root,
                    args.device,
                    stage,
                    propagation_device=args.inference_device,
                ),
            )
            classes = torch.arange(scores.shape[1])
        predictions = stage("prediction", lambda: classes[scores.argmax(1)])
        accuracy = 100 * float((predictions[test] == graph["y"][test]).double().mean())
    seconds = time.perf_counter() - adaptation
    stop.set()
    sampler.join()
    peak[0] = max(peak[0], process.memory_info().rss)
    result = dict(
        dataset=args.dataset,
        method=args.method,
        reference=args.reference,
        device=args.device,
        inference_device=args.inference_device,
        kernel_device=args.kernel_device,
        threads=args.threads,
        seed=42,
        shots_per_class=args.shots,
        holdout_shots_per_class=args.holdout_shots,
        total_available_training_labels=available_labels,
        accuracy_percent=accuracy,
        adaptation_seconds=seconds,
        process_pipeline_seconds=time.perf_counter() - started,
        peak_rss_gib=peak[0] / 2**30,
        stages=stages,
        readout=info,
        support_labels=len(train),
        validation_pool=len(val),
        validation_labels=len(val) if args.method == "evogfm" else 0,
        nodes=len(graph["x"]),
        undirected_edges=protocol["edges"].shape[1],
        checkpoint=args.checkpoint if args.method == "evogfm" else "GraphAny Wisconsin",
        torch_version=torch.__version__,
        numpy_version=__import__("numpy").__version__,
        scipy_version=__import__("scipy").__version__,
        python=platform.python_version(),
        threadpools=threadpool_info(),
        candidate_fits=(
            sum(row["alpha_candidates"] for row in fitted_candidates)
            if args.method == "evogfm"
            else 5
        ),
        fitted_candidates=fitted_candidates,
        label_propagation=propagation_times,
        inverse_updated=0,
        peak_cuda_allocated_gib=(
            torch.cuda.max_memory_allocated() / 2**30 if cuda_enabled else 0
        ),
    )
    torch.save(
        dict(
            predictions=predictions.to(torch.int16),
            validation_scores=scores[val],
            support_nodes=train,
            selection_nodes=val if args.method == "evogfm" else train[:0],
        ),
        output / "predictions.pt",
    )
    save_json(output / "result.json", result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "dataset",
                    "method",
                    "accuracy_percent",
                    "adaptation_seconds",
                    "peak_rss_gib",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", choices=("evogfm", "graphany"), default="evogfm")
    parser.add_argument("--checkpoint", default="runs/reference/best.pt")
    parser.add_argument(
        "--reference", help="Directory containing saved pre-optimization modules"
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--graphany-root", default="../work/GraphAny")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--inference-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--kernel-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--shots",
        type=int,
        default=0,
        help="Training labels per class; 0 uses all official training nodes",
    )
    parser.add_argument(
        "--holdout-shots",
        type=int,
        default=0,
        help="Withhold this many of the sampled labels per class for EvoGFM selection; GraphAny fits all sampled labels",
    )
    args = parser.parse_args()
    with threadpool_limits(limits=args.threads):
        main(args)
