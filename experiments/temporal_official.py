"""Official temporal protocols for incremental predictive analytics over evolving graphs."""

from __future__ import annotations

import argparse
import ast
import csv
from copy import deepcopy
import gc
import importlib
import json
import os
from pathlib import Path
import pdb
import pickle
import runpy
import shutil
import sys
import time
import types

import numpy as np
import torch

from evogfm.data import canonical_edges
from evogfm.incremental import MaintainedLinkScorer, ingest_edges
from evogfm.temporal import TemporalHistory, export_protocol, prepare_dataset
from experiments.full_graph import load_checkpoint
from experiments.idea_a import frozen_structural_features


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


@torch.no_grad()
def build_feature_cache(
    events,
    index,
    protocol_directory,
    directory,
    checkpoint,
    stage="validation",
    maintenance="full",
    threads=2,
):
    """Model sees t'<t; exported t positives enter graph/history after scoring."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(threads)
    model, head, _ = load_checkpoint(checkpoint)
    train = events.phase == "train"
    train_graph = canonical_edges(
        torch.from_numpy(np.stack((events.source[train], events.destination[train]))),
        events.n_nodes,
    )
    # Feature fitting is an offline training operation; no val/test topology enters it.
    raw = frozen_structural_features(
        train_graph, events.n_nodes, model.config["input_dim"]
    )
    encoded = model.encode_features(raw)
    first = index[0]
    warm = np.load(Path(protocol_directory) / first["file"])["positive"]
    visible = canonical_edges(torch.from_numpy(warm), events.n_nodes)
    known = (visible[0] * events.n_nodes + visible[1]).numpy()
    scorer = MaintainedLinkScorer(
        model, head, raw, raw, visible, method=maintenance, encoded=encoded
    )
    history = TemporalHistory(events.n_nodes)
    history.ingest(warm, first["time"])
    selected, maintenance_rows = [], []
    for entry in index[1:]:
        if stage == "validation" and entry["phase"] == "test":
            break
        cached = directory / entry["file"]
        query = np.load(Path(protocol_directory) / entry["file"])
        positive, auc_negative, ranking_negative = (
            query[name] for name in ("positive", "auc_negative", "ranking_negative")
        )
        if not cached.exists():
            pairs = np.concatenate(
                (
                    positive,
                    auc_negative,
                    ranking_negative if entry["phase"] != "train" else positive[:, :0],
                ),
                axis=1,
            )
            start = time.perf_counter()
            tensor = torch.from_numpy(pairs)
            base = torch.cat(
                [scorer.features(block) for block in tensor.split(1024, dim=1)]
            )
            statistics = history.statistics(pairs, entry["time"])
            np.savez_compressed(
                cached,
                base=base.numpy(),
                statistics=statistics.numpy(),
                positive_count=positive.shape[1],
                auc_negative_count=auc_negative.shape[1],
                positive_sources=positive[0],
                rank_sources=query["rank_sources"],
                generation_s=time.perf_counter() - start,
            )
            print(
                json.dumps(
                    dict(
                        action="feature_cache",
                        time=entry["time"],
                        phase=entry["phase"],
                        pairs=pairs.shape[1],
                        seconds=time.perf_counter() - start,
                    )
                ),
                flush=True,
            )
        selected.append(entry)
        history.ingest(positive, entry["time"])
        visible, added, known = ingest_edges(
            visible, torch.from_numpy(positive), events.n_nodes, known
        )
        maintenance_rows.append(
            dict(
                time=entry["time"],
                phase=entry["phase"],
                **scorer.update(visible, added),
            )
        )
    manifest = dict(
        dataset=events.metadata,
        model_config=model.config,
        checkpoint=str(Path(checkpoint).resolve()),
        stage=stage,
        maintenance=maintenance,
        observation_policy="predict-before-ingest; history contains t'<t only",
        feature_fit="frozen structure features fitted on training graph only; offline train fitting",
        warmup_snapshot=first["time"],
        feature_dimensions=int(head[0].in_features),
        backbone_parameters=sum(parameter.numel() for parameter in model.parameters()),
        head_parameters=sum(parameter.numel() for parameter in head.parameters()),
        model_weights_bytes=sum(
            parameter.numel() * parameter.element_size()
            for parameter in model.parameters()
        ),
        head_weights_bytes=sum(
            parameter.numel() * parameter.element_size()
            for parameter in head.parameters()
        ),
        scorer_cache_bytes_final=scorer.cache_bytes(),
        scorer_cache_memory_metric="unique numeric payload storage, excluding model/head weights; includes graph, sparse and nested numeric state",
        maintenance_updates=maintenance_rows,
        temporal_history_state_bytes=history.state_bytes(),
        temporal_history_memory_metric="recursive sys.getsizeof including Python dict keys/values and NumPy payload; excludes allocator fragmentation",
        index=selected,
    )
    write_json(directory / "manifest.json", manifest)
    return manifest


def original_undergs_modules(baselines):
    root = Path(baselines).resolve() / "UnderGS/code"
    name = "_official_undergs"
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(root)]
        sys.modules[name] = package
    return [
        importlib.import_module(f"{name}.{part}")
        for part in ("undergs_model", "util", "evaluation")
    ]


def run_undergs(
    events,
    output,
    baselines,
    seed=20261115,
    epochs=50,
    threads=2,
    learning_rate=0.0001,
    filter_ratio=0.3,
    snapshot_decay=0.05,
):
    """Execute original model/training/validation/evaluator without source edits."""
    module, util, evaluation = original_undergs_modules(baselines)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    np.random.seed(seed)
    args = argparse.Namespace(
        n_nodes=events.n_nodes + 1,
        n_edges=len(events.source) + 1,
        topk=16,
        lambda_1=0.1,
        lambda_2=snapshot_decay,
        filter_ratio=filter_ratio,
        device=torch.device("cpu"),
    )
    model = module.UnderGS(
        node_features=None,
        edge_features=events.edge_features,
        device=args.device,
        dropout=0.1,
        node_dimension=128,
        time_dimension=8,
        neural_network="gcn",
        args=args,
    )
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    stopper = util.EarlyStopMonitor(max_round=5)

    def data(mask):
        return types.SimpleNamespace(
            sources=events.source[mask] + 1,
            destinations=events.destination[mask] + 1,
            timestamps=events.time[mask],
            edge_idxs=events.event_index[mask],
        )

    train, validation, test = (
        data(events.phase == phase) for phase in ("train", "validation", "test")
    )
    train_groups = [
        np.flatnonzero(train.timestamps == time) for time in np.unique(train.timestamps)
    ]
    val_groups = [
        np.flatnonzero(validation.timestamps == time)
        for time in np.unique(validation.timestamps)
    ]
    test_groups = [
        np.flatnonzero(test.timestamps == time) for time in np.unique(test.timestamps)
    ]
    train_sampler = util.RandEdgeSampler(train.sources, train.destinations)
    val_sampler = util.RandEdgeSampler(train.sources, train.destinations, seed=0)
    test_sampler = util.RandEdgeSampler(
        events.source + 1, events.destination + 1, seed=2
    )
    best, rows = None, []
    started = time.perf_counter()
    for epoch in range(epochs):
        model.embedding_module.tcns.reset_tcns()
        losses = []
        for snapshot, selected in enumerate(train_groups):
            model.embedding_module.snapshot = snapshot
            model.train()
            _, negatives = train_sampler.sample(len(selected))
            positive, negative = model.compute_edge_probabilities(
                train.sources[selected],
                train.destinations[selected],
                negatives,
                train.timestamps[selected],
                train.edge_idxs[selected],
                train=True,
            )
            loss = criterion(positive.flatten(), torch.ones(len(selected))) + criterion(
                negative.flatten(), torch.zeros(len(selected))
            )
            model.embedding_module.tcns.snapshot_decay(args.lambda_2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss))
        scheduler.step()
        val_scores = []
        for selected in val_groups:
            backup = model.embedding_module.tcns.backup()
            _, auc = evaluation.val(
                model, val_sampler, validation, selected, args.device, criterion, args
            )
            model.embedding_module.tcns.restore(backup)
            model.embedding_module.tcns.snapshot_decay(args.lambda_2)
            val_scores.append(float(auc))
        metric = float(np.mean(val_scores))
        row = dict(
            epoch=epoch,
            train_loss=float(np.mean(losses)),
            validation_auc=metric,
            elapsed_s=time.perf_counter() - started,
        )
        rows.append(row)
        print(json.dumps(dict(action="official_undergs", **row)), flush=True)
        stop = stopper.early_stop_check(metric)
        if epoch == stopper.best_epoch:
            best = deepcopy(model.state_dict())
        if stop:
            break
    model.load_state_dict(best)
    model.eval()
    training_seconds = time.perf_counter() - started
    # Preserve the public evaluator's two state-mutating calls per snapshot.
    original_compute = model.compute_edge_probabilities
    capture_count = 0

    def capture(*arguments, **kwargs):
        nonlocal capture_count
        result = original_compute(*arguments, **kwargs)
        np.savez_compressed(
            output / f"test_call_{capture_count}.npz",
            positive=result[0].detach().numpy(),
            negative=result[1].detach().numpy(),
            sources=arguments[0],
            destinations=arguments[1],
            negative_nodes=arguments[2],
            time=arguments[3][0],
        )
        capture_count += 1
        return result

    model.compute_edge_probabilities = capture
    with torch.no_grad():
        auc, mrr = evaluation.test(
            model, test_sampler, test, test_groups, criterion, args
        )
    torch.save(best, output / "best_parameters.pt")
    with (output / "training.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = dict(
        auc=float(auc),
        mrr=float(mrr),
        epochs=len(rows),
        best_epoch=stopper.best_epoch,
        training_s=training_seconds,
        timing_scope="development CPU run; concurrent local work possible; not an isolated system-performance measurement",
        seed=seed,
        implementation="original UnderGS model, val and test functions",
        protocol=events.metadata,
        lr=learning_rate,
        filter_ratio=filter_ratio,
        snapshot_decay=snapshot_decay,
        temporal_visibility="official code incorporates current positive edges before scoring; AUC and MRR passes mutate node states separately",
        parameters="paper hyperparameters, original full-snapshot train.py optimizer path; not paper batch-200 pipeline",
        device="cpu",
        pytorch=torch.__version__,
    )
    write_json(output / "summary.json", result)
    print(json.dumps(result), flush=True)
    return result


def run_decoupled(
    events,
    output,
    baselines,
    propagation_build,
    seed=20261115,
    epochs=30,
    threads=2,
    dual_filter=False,
    restore_raw_order=False,
    capture_queries=False,
):
    """Run the complete released low-pass LSTM configuration with API repairs."""
    if events.metadata["dataset"] == "uci" and not events.metadata.get(
        "singleton_csv_map_repair"
    ):
        raise ValueError(
            "Resolve the public UCI singleton-snapshot mismatch before running this baseline"
        )
    root = Path(baselines).resolve() / "DecoupledDGNN"
    propagation_build = Path(propagation_build).resolve()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset = {
        "bitcoin-alpha": "bitcoinalpha",
        "bitcoin-otc": "bitcoinotc",
        "uci": "CollegeMsg",
    }[events.metadata["dataset"]]
    directory = output / "data" / dataset
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"ml_{dataset}_disperse.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("u", "i", "ts", "ts_str", "idx", "label"))
        order = (
            np.argsort(events.event_index, kind="stable")
            if restore_raw_order
            else np.arange(len(events.source))
        )
        writer.writerows(
            zip(
                events.source[order],
                events.destination[order],
                events.time[order],
                events.time[order],
                events.event_index[order],
                np.ones(len(events.source)),
            )
        )
    time_edges = {}
    for timestamp in events.times:
        selected = np.flatnonzero(events.time == timestamp)
        if restore_raw_order:
            selected = selected[np.argsort(events.event_index[selected], kind="stable")]
        time_edges[int(timestamp)] = np.stack(
            (events.source[selected], events.destination[selected]), axis=1
        )
    with (directory / f"{dataset}_time_edge_map_disperse.pkl").open("wb") as handle:
        pickle.dump(time_edges, handle, pickle.HIGHEST_PROTOCOL)
    np.savetxt(
        directory / f"{dataset}_init.txt",
        np.column_stack((np.arange(events.n_nodes), np.arange(events.n_nodes))),
        fmt="%d",
    )
    torch.set_num_threads(threads)
    sys.path.insert(0, str(Path(propagation_build).resolve()))
    sys.path.insert(1, str(root))
    previous_directory, previous_argv, previous_debugger = (
        Path.cwd(),
        sys.argv,
        pdb.set_trace,
    )
    os.chdir(output)
    propagation_started = time.perf_counter()
    try:
        representation = (
            directory / f"{dataset}_nodes_seq_lst_randomize_mul_disperse.pkl"
        )
        low_cached = representation.exists()
        if not low_cached:
            pdb.set_trace = lambda: None
            sys.argv = [
                str(root / "gen_emb.py"),
                "--data",
                dataset,
                "--path",
                str(directory) + "/",
                "--randomize_features",
                "--disperse",
            ]
            generated = runpy.run_path(str(root / "gen_emb.py"), run_name="__main__")
            del generated
            gc.collect()
        high_representation = (
            directory / f"{dataset}_nodes_seq_lst_randomize_mul_disperse_alpha-1.pkl"
        )
        high_cached = high_representation.exists() if dual_filter else None
        if dual_filter and not high_cached:
            high_directory = directory / "highpass_generation"
            high_directory.mkdir(exist_ok=True)
            for name in (
                f"{dataset}_init.txt",
                f"{dataset}_time_edge_map_disperse.pkl",
            ):
                shutil.copyfile(directory / name, high_directory / name)
            pdb.set_trace = lambda: None
            sys.argv = [
                str(root / "gen_emb.py"),
                "--data",
                dataset,
                "--path",
                str(high_directory) + "/",
                "--randomize_features",
                "--disperse",
                "--alpha",
                "1.8",
            ]
            generated = runpy.run_path(str(root / "gen_emb.py"), run_name="__main__")
            del generated
            gc.collect()
            with (high_directory / representation.name).open("rb") as handle:
                high = pickle.load(handle)
            for sequence in high:
                sequence.data *= 0.2 / 1.8
            with high_representation.open("wb") as handle:
                pickle.dump(high, handle, pickle.HIGHEST_PROTOCOL)
            del high
            gc.collect()
        propagation_s = time.perf_counter() - propagation_started
        spec = importlib.util.spec_from_file_location(
            "_official_decoupled_model", root / "src/model.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        helper_source = ast.parse((root / "data_loader.py").read_text())
        helper_definitions = ast.Module(
            body=[
                node
                for node in helper_source.body
                if isinstance(node, ast.ClassDef) and node.name == "EdgeHelper"
            ],
            type_ignores=[],
        )
        import scipy.sparse

        helper_scope = dict(np=np, torch=torch, pickle=pickle, scipy=scipy)
        exec(
            compile(helper_definitions, str(root / "data_loader.py"), "exec"),
            helper_scope,
        )

        query_capture = dict(active=False, current=None, calls=[])

        class EdgeHelper(helper_scope["EdgeHelper"]):
            def get_edges_feats(self, *arguments, **keywords):
                # Source tensor aliasing is preserved; timestamps are unused by LSTM.
                if query_capture["active"]:
                    query_capture["current"] = {
                        "pairs": np.stack((arguments[0], arguments[1])).copy(),
                        "times": np.asarray(arguments[2]).copy(),
                    }
                return super().get_edges_feats(*arguments, **keywords)[:2]

        def data_splits(*arguments, **keywords):
            splits = []
            for phase in (None, "train", "validation", "test"):
                mask = (
                    np.ones(len(events.source), dtype=bool)
                    if phase is None
                    else events.phase == phase
                )
                splits.append(
                    types.SimpleNamespace(
                        sources=events.source[mask],
                        destinations=events.destination[mask],
                        timestamps=events.time[mask],
                        edge_idxs=events.event_index[mask],
                        labels=np.ones(mask.sum()),
                        unique_times=np.unique(events.time[mask]),
                    )
                )
            return (*splits, None, None)

        scatter = types.ModuleType("torch_scatter")

        def scatter_max(src, index, dim_size):
            result = src.new_zeros(dim_size)
            result.scatter_reduce_(
                0, index.long(), src, reduce="amax", include_self=True
            )
            return result, None

        scatter.scatter_max = scatter_max
        sys.modules["torch_scatter"] = scatter
        import math
        import random

        args = argparse.Namespace(
            seed=seed,
            gpu=0,
            batch_size=1024,
            data=dataset,
            use_neg=dual_filter,
            seq_model="lstm",
            emb_size=128,
            hidden_dim=128 if dataset == "bitcoinalpha" else 64,
            lstm_layers=2,
            dropout=0.1,
            window_size=20,
            inductive=False,
            shuffle=False,
            checkpt_path=str(output / "check_point"),
            patience=30,
            learning_rate=0.001,
            epochs=epochs,
        )
        Path(args.checkpt_path).mkdir(parents=True, exist_ok=True)
        source = ast.parse((root / "link_prediction_alongtime.py").read_text())
        definitions = ast.Module(
            body=[
                node
                for node in source.body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef))
            ],
            type_ignores=[],
        )
        scope = dict(
            np=np,
            torch=torch,
            nn=torch.nn,
            optim=torch.optim,
            math=math,
            random=random,
            args=args,
            MyLinkPrediction=module.MyLinkPrediction,
            EdgeHelper=EdgeHelper,
            get_data_transductive=data_splits,
        )
        exec(
            compile(definitions, str(root / "link_prediction_alongtime.py"), "exec"),
            scope,
        )
        controller = scope["Execute"](args)
        original_embedding = controller.model.get_edges_embedding

        def embedding(*arguments, **keywords):
            predictions = original_embedding(*arguments, **keywords)
            if query_capture["active"]:
                query_capture["calls"].append(
                    dict(
                        query_capture["current"],
                        scores=predictions.detach().cpu().numpy().reshape(-1),
                    )
                )
            return predictions

        if capture_queries:
            controller.model.get_edges_embedding = embedding
        # Evaluation graphs are unnecessary and exceed memory on the target runtime.
        controller.valid = torch.no_grad()(controller.valid)
        controller.eval_mrr_and_recall = torch.no_grad()(controller.eval_mrr_and_recall)
        original_test, original_rank = controller.test, controller.eval_mrr_and_recall
        ranking_rows, captured = [], {}

        def ranking(*arguments, **keywords):
            started = time.perf_counter()
            call_start = len(query_capture["calls"])
            metric, recall = original_rank(*arguments, **keywords)
            if query_capture["active"]:
                calls = query_capture["calls"][call_start:]
                if len(calls) != 2:
                    raise RuntimeError(
                        "Released ranking must have positive and negative forward calls"
                    )
                timestamp = int(arguments[1][0])
                path = output / "shared_queries" / f"snapshot_{timestamp}.npz"
                path.parent.mkdir(exist_ok=True)
                auc_calls = query_capture["calls"][:call_start]
                if len(auc_calls) % 2:
                    raise RuntimeError("Unpaired released AUC forward calls")
                np.savez_compressed(
                    path,
                    positive=calls[0]["pairs"],
                    positive_scores=calls[0]["scores"],
                    ranking_negative_all=calls[1]["pairs"],
                    ranking_negative_scores_all=calls[1]["scores"],
                    auc_positive=np.concatenate(
                        [item["pairs"] for item in auc_calls[::2]], axis=1
                    ),
                    auc_positive_scores=np.concatenate(
                        [item["scores"] for item in auc_calls[::2]]
                    ),
                    auc_negative=np.concatenate(
                        [item["pairs"] for item in auc_calls[1::2]], axis=1
                    ),
                    auc_negative_scores=np.concatenate(
                        [item["scores"] for item in auc_calls[1::2]]
                    ),
                    native_mrr=metric,
                )
                query_capture["calls"].clear()
            ranking_rows.append(
                dict(
                    time=int(arguments[1][0]),
                    mrr=float(metric),
                    seconds=time.perf_counter() - started,
                    recall={str(key): value for key, value in recall.items()},
                )
            )
            return metric, recall

        controller.eval_mrr_and_recall = ranking

        @torch.no_grad()
        def test(*arguments, **keywords):
            started = time.perf_counter()
            query_capture["active"] = capture_queries
            if capture_queries:
                torch.save(
                    {
                        "numpy": np.random.get_state(),
                        "torch": torch.get_rng_state(),
                        "cuda": (
                            torch.cuda.get_rng_state_all()
                            if torch.cuda.is_available()
                            else None
                        ),
                        "python": random.getstate(),
                    },
                    output / "pretest_rng.pt",
                )
            result = original_test(*arguments, **keywords)
            query_capture["active"] = False
            captured.update(
                test_s=time.perf_counter() - started,
                loss=float(result[0]),
                accuracy=float(result[1]),
                auc=float(result[2]),
                ap=float(result[3]),
                mrr=float(result[4]),
            )
            return result

        controller.test = test
        started = time.perf_counter()
        controller.train()
        try:
            import resource

            peak_rss_bytes = int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            )
        except ImportError:
            peak_rss_bytes = None

        result = dict(
            **captured,
            train_and_test_s=time.perf_counter() - started,
            propagation_s=propagation_s,
            cached_representations_reused=bool(
                low_cached and (not dual_filter or high_cached)
            ),
            cold_generation_timing_not_measured=bool(
                low_cached or (dual_filter and high_cached)
            ),
            representation_cache_status=dict(
                lowpass_reused=low_cached, highpass_reused=high_cached
            ),
            seed=seed,
            exact_test_queries_captured=capture_queries,
            arguments=vars(args),
            model_parameters=sum(
                parameter.numel() for parameter in controller.model.parameters()
            ),
            device=str(controller.device),
            peak_rss_bytes=peak_rss_bytes,
            peak_cuda_allocated_bytes=(
                int(torch.cuda.max_memory_allocated(controller.device))
                if controller.device.type == "cuda"
                else None
            ),
            gpu_memory_metric="PyTorch max_memory_allocated counter, not measured physical VRAM residency",
            representation_bytes=representation.stat().st_size
            + (high_representation.stat().st_size if dual_filter else 0),
            official_revision=events.metadata["official_revision"],
            protocol=dict(
                events.metadata,
                decoupled_sampler="original training-dependent numpy RNG; exported fixed candidate files are not used by this original-baseline evaluator",
                visibility="original past-window rows only: EdgeHelper uses sequence row index < current snapshot index",
            ),
            auc_aggregation="global pooled test-event AUC, as in released Execute.test; new-method development tables use snapshot-mean AUC",
            mrr_aggregation="unweighted snapshot mean; best positive per unique source and pessimistic ties with 1000 negatives",
            directly_comparable_to_exported_query_results=False,
            comparison_limit="same event split and candidate distribution; candidate realizations depend on training RNG and AUC aggregation differs",
            configuration=(
                "paper Table4 low+high-pass LSTM configuration; missing high-pass generation script reconstructed by original-C++ alpha'=1.8 and output scaling .2/1.8, checked against exact linear operator"
                if dual_filter
                else "complete README low-pass LSTM command; not paper Table4 low+high-pass setting"
            ),
            event_order=(
                "original raw CSV order within each snapshot, restored using raw event_index"
                if restore_raw_order
                else "protocol-export original-timestamp sorted order; differs from public stable snapshot/raw-row ordering on Bitcoin-Alpha"
            ),
            highpass_operator_validation=(
                str(
                    Path(propagation_build).resolve()
                    / "highpass_operator_probe/summary.json"
                )
                if dual_filter
                else None
            ),
            implementation="original C++ propagation, gen_emb.py, MyLinkPrediction, Execute train/valid/test/ranking definitions",
            compatibility_repairs=[
                "build with current compiler/Eigen include paths; unchanged C++/Cython algorithm sources",
                "disable interactive debugger breakpoints in gen_emb.py",
                "supply missing preprocessing idx column and explicit same transductive time splits; bypass unused inductive sampler incompatible with Python3.11 set inputs",
                "drop unused time-channel return values from EdgeHelper for released two-result LSTM caller; preserve original source/destination tensor aliasing",
                "native torch scatter_reduce amax equivalent to torch_scatter scatter_max on sigmoid probabilities",
                "no_grad evaluation preserves model forward calls and original full inference batch sizes",
            ],
            reproducibility="original training-dependent negative RNG, random Xavier LSTM state also during evaluation; no batch-size or sampler redesign",
            timing_scope="development WSL run; no isolated cross-system timing claim",
            ranking=ranking_rows,
        )
        write_json(output / "summary.json", result)
        print(json.dumps(result), flush=True)
        return result
    finally:
        os.chdir(previous_directory)
        sys.argv, pdb.set_trace = previous_argv, previous_debugger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=["undergs", "decoupled"], required=True)
    parser.add_argument(
        "--datasets", nargs="+", default=["uci", "bitcoin-alpha", "bitcoin-otc"]
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[20261115])
    parser.add_argument(
        "--action",
        choices=["prepare", "features", "official_undergs", "official_decoupled"],
        default="prepare",
    )
    parser.add_argument("--output", default="runs/official_a")
    parser.add_argument("--baselines", default="baselines")
    parser.add_argument("--data", default="data/temporal")
    parser.add_argument("--checkpoint", default="checkpoints/reference.pt")
    parser.add_argument("--stage", choices=["validation", "test"], default="validation")
    parser.add_argument(
        "--maintenance", choices=MaintainedLinkScorer.METHODS, default="full"
    )
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--propagation-build", default="runs/official_a/decoupled_build"
    )
    parser.add_argument(
        "--dual-filter",
        action="store_true",
        help="Reconstruct the paper's low+high-pass LSTM baseline",
    )
    args = parser.parse_args()
    for dataset in args.datasets:
        events = prepare_dataset(dataset, args.protocol, args.data, args.baselines)
        for seed in args.seeds:
            directory = Path(args.output) / args.protocol / dataset / f"s{seed}"
            protocol_directory = directory / "protocol"
            if (protocol_directory / "index.json").exists():
                index = json.loads((protocol_directory / "index.json").read_text())
            else:
                index = export_protocol(events, args.protocol, seed, protocol_directory)
            print(json.dumps(dict(action="prepared", **events.metadata)), flush=True)
            if args.action == "features":
                build_feature_cache(
                    events,
                    index,
                    protocol_directory,
                    directory / f"features_{args.maintenance}",
                    args.checkpoint,
                    args.stage,
                    args.maintenance,
                    args.threads,
                )
            elif args.action == "official_undergs":
                if args.protocol != "undergs":
                    parser.error("official_undergs requires --protocol undergs")
                run_undergs(
                    events,
                    directory / "official_undergs",
                    args.baselines,
                    seed,
                    args.epochs,
                    args.threads,
                )
            elif args.action == "official_decoupled":
                if args.protocol != "decoupled":
                    parser.error("official_decoupled requires --protocol decoupled")
                run_decoupled(
                    events,
                    directory
                    / (
                        "official_decoupled_dual_raw_order"
                        if args.dual_filter
                        else "official_decoupled_lowpass"
                    ),
                    args.baselines,
                    args.propagation_build,
                    seed,
                    args.epochs,
                    args.threads,
                    dual_filter=args.dual_filter,
                    restore_raw_order=args.dual_filter,
                )


if __name__ == "__main__":
    main()
