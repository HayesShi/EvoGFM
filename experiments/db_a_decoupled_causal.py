"""Train the author DecoupledDGNN decoder on the shared UnderGS chronology."""

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import time
from zipfile import ZipFile

import numpy as np


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def prepare(dataset, output):
    from evogfm.temporal import prepare_dataset
    from experiments.db_a_cuda_quality import observed_candidates
    from experiments.db_a_native import WORKSPACE

    candidates = (
        WORKSPACE / "runs/protocol_alignment/undergs" / dataset / "s42/candidates"
    )
    index = json.loads((candidates / "index.json").read_text(encoding="utf-8"))
    events = prepare_dataset(
        dataset, "undergs", WORKSPACE / "data/temporal", WORKSPACE / "baselines"
    )
    test = observed_candidates(events, index, candidates)
    offsets, snapshots = [0], []
    variants, rng = {}, np.random.default_rng(42)
    first = np.zeros(events.n_nodes, dtype=bool)
    for row, entry in enumerate(index):
        with np.load(candidates / entry["file"]) as source:
            positive = source["positive"]
            fallback = source["auc_negative"]
        snapshots.append(positive.T)
        offsets.append(offsets[-1] + positive.shape[1])
        if entry["phase"] == "test":
            positive, _negative, ranking = test[entry["time"]]
            variants[f"{row}_positive"] = positive
            variants[f"{row}_negative"] = ranking
        else:
            negative = np.empty_like(positive)
            for u in np.unique(positive[0]):
                slots = np.flatnonzero(positive[0] == u)
                eligible = np.setdiff1d(
                    np.flatnonzero(first), np.r_[u, positive[1, slots]]
                )
                negative[0, slots] = u
                if len(eligible):
                    negative[1, slots] = rng.choice(
                        eligible, size=len(slots), replace=True
                    )
                else:
                    negative[1, slots] = fallback[1, slots]
            variants[f"{row}_positive"] = positive
            variants[f"{row}_negative"] = negative
        first[positive] = True
    np.savez(
        output / "chronology.npz",
        edges=np.concatenate(snapshots),
        offsets=np.asarray(offsets),
        phases=np.asarray([e["phase"] for e in index]),
        n_nodes=np.asarray(events.n_nodes),
    )
    np.savez(output / "pairs.npz", **variants)
    save(
        output / "result.json",
        {
            "dataset": dataset,
            "snapshots": len(index),
            "nodes": events.n_nodes,
            "strict_test_queries": int(sum(q[0].shape[1] for q in test.values())),
            "protocol": "Previously frozen UnderGS causal discrete snapshot split with fixed 100 observed negatives per test event",
            "source_candidate_index": str(candidates / "index.json"),
            "seed": 42,
        },
    )


def prepare_extended(directory, output):
    metadata = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    events = np.load(metadata["event_path"], mmap_mode="r")
    entries = metadata["snapshots"]
    offsets = [row["start"] for row in entries] + [entries[-1]["stop"]]
    variants = {}
    for row in entries:
        snap = row["id"]
        with np.load(directory / row["file"]) as pack:
            selected, negative = pack["indices"], pack["negative"]
        positive = events[row["start"] + selected, :2].T
        variants[f"{snap}_positive"] = positive
        variants[f"{snap}_negative"] = np.stack(
            (np.tile(positive[0], negative.shape[1]), negative.T.ravel())
        )
    np.savez(
        output / "chronology.npz",
        edges=events[:, :2],
        offsets=np.asarray(offsets),
        phases=np.asarray([e["phase"] for e in entries]),
        n_nodes=np.asarray(metadata["n"]),
        features=np.array(172),
        dual_filter=np.array(False),
        undirected=np.array(True),
        dataset=np.array(metadata["dataset"]),
    )
    np.savez(output / "pairs.npz", **variants)
    save(
        output / "result.json",
        {
            "dataset": metadata["dataset"],
            "snapshots": len(entries),
            "nodes": metadata["n"],
            "source_candidate_index": str(directory / "result.json"),
            "seed": 42,
        },
    )


def propagation(chronology, output, compressed):
    from propagation import InstantGNN

    with np.load(chronology) as values:
        edges, offsets, n = values["edges"], values["offsets"], int(values["n_nodes"])
        width = int(values["features"]) if "features" in values else 128
        dual = bool(values["dual_filter"]) if "dual_filter" in values else True
        undirected = bool(values["undirected"]) if "undirected" in values else False
        dataset = str(values["dataset"]) if "dataset" in values else None
    np.savetxt(
        output / "graph_init.txt",
        np.column_stack((np.arange(n), np.arange(n))),
        fmt="%d",
    )
    src = np.random.RandomState(42).rand(n, width)
    alphas = (0.2, 1.8) if dual else (0.2,)
    states, methods = [src.copy() for _ in alphas], [InstantGNN() for _ in alphas]
    for state, operation, alpha in zip(states, methods, alphas):
        operation.initial_operation(
            str(output) + "/", "graph_init", n, n, 1e-7, alpha, state
        )
    steps = len(offsets) - 1
    shape = (steps + 1, n, width * len(alphas))
    channels = None
    if compressed:
        (output / "deltas").mkdir(exist_ok=True)
    else:
        channels = np.lib.format.open_memmap(
            output / "deltas.npy", mode="w+", dtype=np.float32, shape=shape
        )

    def persist(index, values):
        if channels is None:
            np.savez_compressed(
                output / "deltas" / f"{index:04d}.npz", values=values.astype(np.float32)
            )
        else:
            channels[index] = values

    windows = np.lib.format.open_memmap(
        output / "window_indices.npy", mode="w+", dtype=np.int32, shape=(steps, n, 20)
    )
    history = np.full((n, 20), -1, dtype=np.int32)
    history[:, -1] = 0
    persist(
        0,
        (
            np.concatenate((states[0], states[1] * (0.2 / 1.8)), axis=1)
            if dual
            else states[0]
        ),
    )
    batch = output / "active_edges.txt"
    for snap, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
        windows[snap] = history
        incoming = edges[start:stop]
        if undirected:
            incoming = np.concatenate((incoming, incoming[:, ::-1]))
        np.savetxt(batch, incoming, fmt="%d")
        previous = [state.copy() for state in states]
        for state, operation, alpha in zip(states, methods, alphas):
            operation.snapshot_operation(str(batch), 1e-7, alpha, state)
        delta = states[0] - previous[0]
        if dual:
            delta = np.concatenate(
                (delta, (states[1] - previous[1]) * (0.2 / 1.8)), axis=1
            )
        persist(snap + 1, delta)
        affected = np.flatnonzero(delta.sum(axis=1) != 0.0)
        history[affected, :-1] = history[affected, 1:]
        history[affected, -1] = snap + 1
        if (snap + 1) % 10 == 0 or snap + 1 == steps:
            save(
                output / "status.json",
                {
                    "phase": "propagating",
                    "completed_snapshots": snap + 1,
                    "total_snapshots": steps,
                },
            )
    if channels is not None:
        channels.flush()
    windows.flush()
    save(
        output / "result.json",
        {
            "status": "complete",
            "snapshots": steps,
            "n_nodes": n,
            "delta_shape": shape,
            "losslessly_compressed": compressed,
            "seed": 42,
            "source": (
                "Released C++ InstantGNN, low pass and reconstructed high pass"
                if dual
                else f"Released C++ InstantGNN, {dataset} 172-channel low-pass adaptation on undirected edges"
            ),
        },
    )


def train(dataset, prepared, representation, output, resume_test=False):
    import torch
    from sklearn.metrics import roc_auc_score
    from experiments.db_a_cuda_quality import observed_metrics
    from experiments.db_a_native import WORKSPACE

    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(42)
    model_file = WORKSPACE / "baselines/DecoupledDGNN/src/model.py"
    spec = importlib.util.spec_from_file_location("_author_decoupled", model_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    representation_meta = json.loads(
        (representation / "result.json").read_text(encoding="utf-8")
    )
    current_archive = representation_meta.get("working_arrays_archive")
    if current_archive:
        with ZipFile(representation / current_archive) as arrays:
            with arrays.open("deltas.npy") as data:
                deltas = np.load(data, allow_pickle=False)
            with arrays.open("window_indices.npy") as data:
                windows = np.load(data, allow_pickle=False)
    elif representation_meta.get("losslessly_compressed"):
        deltas = np.empty(representation_meta["delta_shape"], dtype=np.float32)
        for i in range(len(deltas)):
            with np.load(representation / "deltas" / f"{i:04d}.npz") as part:
                deltas[i] = part["values"]
            if i % 20 == 0:
                save(
                    output / "status.json",
                    {
                        "phase": "loading exact causal representation",
                        "completed_snapshots": i + 1,
                        "total_snapshots": len(deltas),
                    },
                )
    else:
        deltas = np.load(representation / "deltas.npy", mmap_mode="r")
    if not current_archive:
        windows = np.load(representation / "window_indices.npy", mmap_mode="r")
    config = SimpleNamespace(
        batch_size=1024,
        hidden_dim=64 if dataset in ("bitcoin-otc", "uci") else 128,
        lstm_layers=2,
        emb_size=deltas.shape[2],
        dropout=0.1,
        seq_model="lstm",
    )
    device = torch.device("cuda")
    original = module.MyLinkPrediction(config, device).to(device)
    optimizer = torch.optim.Adam(original.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss()
    with np.load(prepared / "chronology.npz") as metadata:
        phases = metadata["phases"]
    pairs = np.load(prepared / "pairs.npz")

    def examples(snap, on_device=True):
        pos, neg = pairs[f"{snap}_positive"], pairs[f"{snap}_negative"]
        both = np.concatenate((pos, neg), axis=1)
        node, index = np.unique(both, return_inverse=True)
        window = windows[snap, node]
        lookup = deltas[np.maximum(0, window), node[:, None]]
        lookup[window < 0] = 0.0
        lookup = torch.from_numpy(lookup)
        return (
            lookup.cuda() if on_device else lookup,
            index.reshape(both.shape),
            pos,
            neg,
        )

    def forward(lookup, block):
        source, destination = lookup[block[0]], lookup[block[1]]
        return original.get_edges_embedding(
            source.to(device), destination.to(device)
        ).flatten()

    @torch.no_grad()
    def measure(phase):
        original.eval()
        metrics, totals, scores_all = [], [], []
        rng = torch.cuda.get_rng_state()
        torch.cuda.manual_seed(42)
        selected = np.flatnonzero(phases == phase)
        for ordinal, snap in enumerate(selected, 1):
            on_device = not (resume_test and phase == "test")
            lookup, table, pos, neg = examples(snap, on_device=on_device)
            table = torch.from_numpy(table).to(lookup.device)
            scores = np.concatenate(
                [
                    forward(lookup, block).cpu().numpy()
                    for block in table.split(1024, dim=1)
                ]
            )
            count = pos.shape[1]
            labels = np.arange(len(scores)) < count
            totals.append(labels)
            scores_all.append(scores)
            if phase == "test":
                metrics.append(
                    observed_metrics(scores[:count], scores[count:], pos, neg)
                )
                del lookup, table
                # Different snapshot widths leave large cached blocks on WDDM.
                # Released logits and the 1,024-pair decoder batches stay fixed.
                if deltas.shape[1] >= 40_000:
                    torch.cuda.empty_cache()
                save(
                    output / "status.json",
                    {
                        "phase": "test",
                        "completed_snapshots": ordinal,
                        "total_snapshots": len(selected),
                        "scored_candidate_pairs": sum(len(s) for s in scores_all),
                        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
                    },
                )
            else:
                metrics.append({"auc": float(roc_auc_score(labels, scores))})
        torch.cuda.set_rng_state(rng)
        return {
            "auc": float(np.mean([value["auc"] for value in metrics])),
            "pooled_auc": float(
                roc_auc_score(np.concatenate(totals), np.concatenate(scores_all))
            ),
            "mrr": (
                float(np.mean([value["mrr"] for value in metrics]))
                if phase == "test"
                else None
            ),
            "snapshots": len(metrics),
            "pairs": sum(len(row) for row in scores_all),
        }

    best, curve = -np.inf, []
    started = time.perf_counter()
    checkpoint = output / "best_parameters.pt"
    training_record = output / "training_record.json"
    if resume_test:
        record = (
            json.loads(training_record.read_text(encoding="utf-8"))
            if training_record.is_file()
            else {}
        )
        curve, train_s = record.get("validation_curve"), record.get("training_s")
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    else:
        for epoch in range(30):
            original.train()
            for snap in np.flatnonzero(phases == "train"):
                lookup, table, pos, neg = examples(snap)
                if not table.shape[1]:
                    continue
                labels = (
                    torch.arange(table.shape[1], device=device) < pos.shape[1]
                ).float()
                items = torch.from_numpy(table).cuda()
                for chunk in torch.randperm(len(labels), device=device).split(1024):
                    loss = criterion(forward(lookup, items[:, chunk]), labels[chunk])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            validation = measure("validation")
            curve.append(dict(epoch=epoch + 1, **validation))
            if validation["auc"] > best:
                best = validation["auc"]
                torch.save(original.state_dict(), checkpoint)
            save(
                output / "status.json",
                dict(
                    phase="training",
                    completed_epochs=epoch + 1,
                    epochs=30,
                    elapsed_s=time.perf_counter() - started,
                    validation=validation,
                    peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                ),
            )
        train_s = time.perf_counter() - started
        save(training_record, dict(validation_curve=curve, training_s=train_s))
    original.load_state_dict(torch.load(checkpoint, weights_only=True))
    test = measure("test")
    save(
        output / "result.json",
        dict(
            dataset=dataset,
            method="DecoupledDGNN causal adaptation",
            seed=42,
            result=test,
            validation_curve=curve,
            training_s=train_s,
            checkpoint=str(checkpoint),
            gpu=torch.cuda.get_device_name(),
            same_candidates_as=str(prepared / "pairs.npz"),
            sequence_setup=representation_meta["source"]
            + ". Released LSTM with independent source and destination input tensors.",
            representation_manifest=str(representation / "result.json"),
            parameters=sum(p.numel() for p in original.parameters()),
            resumed_test=resume_test,
            batch_host_staging=resume_test,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("prepare", "prepare-extended", "propagate", "train"):
        part = sub.add_parser(action)
        part.add_argument("--output", type=lambda x: Path(x).resolve(), required=True)
        if action == "propagate":
            part.add_argument("--chronology", type=Path, required=True)
            part.add_argument("--compressed", action="store_true")
        elif action == "prepare-extended":
            part.add_argument("--candidates", type=Path, required=True)
        else:
            part.add_argument(
                "--dataset",
                choices=(
                    "bitcoin-alpha",
                    "bitcoin-otc",
                    "uci",
                    "wikipedia",
                    "reddit-body",
                    "reddit-title",
                ),
                required=True,
            )
        if action == "train":
            part.add_argument("--prepared", type=Path, required=True)
            part.add_argument("--representation", type=Path, required=True)
            part.add_argument("--resume-test", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.action == "prepare":
        prepare(args.dataset, args.output)
    elif args.action == "prepare-extended":
        prepare_extended(args.candidates, args.output)
    elif args.action == "propagate":
        propagation(args.chronology, args.output, args.compressed)
    else:
        train(
            args.dataset,
            args.prepared,
            args.representation,
            args.output,
            args.resume_test,
        )


if __name__ == "__main__":
    main()
