"""Official SCGFM training and matched-support node classification.

Supply an unmodified checkout of https://github.com/Xd-He/SCGFM. Source training
uses the release's Photo+Computers configuration, including all 60 epochs.
"""

import argparse
import csv
import gc
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from evogfm.data import load_graph, load_protocol
from evogfm.readout import episode
from . import settings


def encode_nodes(raw, ids, converter, encoder, device, batch_size=64):
    """Encode required nodes using the complete graph, retaining official PPR."""
    with torch.no_grad():
        for nodes in ids.split(batch_size):
            scores = converter.ppr_batch(
                raw.edge_index, raw.num_nodes, nodes.to(device), 0.15, 100, 1e-6
            )
            vectors = []
            for i, node in enumerate(nodes.tolist()):
                ego = converter.extract_subgraph(raw, node, scores[i], 100)
                vectors.append(encoder.encode_single(ego).cpu())
            yield nodes, torch.stack(vectors)


def streamed_predictions(raw, labels, protocol, converter, encoder, device, name):
    """Fit official support readouts, then score every test node in blocks.

    Unused train/validation nodes remain in the full graph for PPR. Their ego
    embeddings need not be materialized: encoding a node is independent of the
    other nodes being evaluated. No feature compression or query subsampling.
    """
    from scgfm.fewshot import ProtoClassifier

    supports = {
        (seed, shot): episode(
            labels,
            protocol["train"],
            torch.Generator().manual_seed(seed),
            shots=shot,
            queries=0,
        )[0]
        for seed in [42, 43, 44]
        for shot in [1, 5, 20]
    }
    ids = torch.cat(list(supports.values())).unique(sorted=True)
    values = torch.cat(
        [vectors for _, vectors in encode_nodes(raw, ids, converter, encoder, device)]
    )
    readouts = {}
    for key, support in supports.items():
        scaler = StandardScaler()
        support_x = torch.from_numpy(
            scaler.fit_transform(values[torch.searchsorted(ids, support)].numpy())
        ).float()
        classifier = ProtoClassifier(device=device)
        classifier.fit(support_x, labels[support])
        readouts[key] = (scaler, classifier)
    del values
    predictions = {key: [] for key in supports}
    completed = 0
    for nodes, vectors in encode_nodes(
        raw, protocol["test"], converter, encoder, device
    ):
        for key, (scaler, classifier) in readouts.items():
            test_x = torch.from_numpy(scaler.transform(vectors.numpy())).float()
            predictions[key].append(classifier.predict(test_x).cpu())
        completed += len(nodes)
        print(
            json.dumps(
                dict(
                    stage="stream_test",
                    graph=name,
                    done=completed,
                    total=len(protocol["test"]),
                )
            ),
            flush=True,
        )
    return {
        key: (support, torch.cat(predictions[key])) for key, support in supports.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["train", "evaluate"])
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=settings.DATA)
    parser.add_argument("--output", type=Path, default=settings.ROOT / "scgfm")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Existing SCGFM checkpoint for evaluation; defaults to OUTPUT/checkpoint/model.pt",
    )
    parser.add_argument(
        "--input-directory",
        type=Path,
        help="Optional shared diagnostic input NPZ directory",
    )
    parser.add_argument(
        "--input-policy", default="degree64", choices=["degree64", "degree_quantile64"]
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Encode supports and all test nodes in blocks, without caching full embeddings",
    )
    parser.add_argument(
        "--graphs", nargs="+", default=["air-brazil", "air-eu", "air-us"]
    )
    args = parser.parse_args()
    sys.path.insert(0, str(args.official.resolve()))
    from torch_geometric.data import Data
    from scgfm.utils import load_config, set_seed
    from scgfm.data.node_to_graph import NodeToGraphConverter
    from scgfm.training import pretrain_bases
    from scgfm.checkpoint import load_model_from_checkpoint
    from scgfm.encoders import SCGFMEncoder
    from scgfm.fewshot import ProtoClassifier

    torch.set_num_threads(2)
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    set_seed(42)

    def load_local(name):
        graph = load_graph(name.lower(), args.data_root)
        if args.phase == "evaluate" and args.input_directory is not None:
            with np.load(
                args.input_directory / f"{name}_{args.input_policy}.npz"
            ) as supplied:
                if not np.array_equal(
                    supplied["y"], graph["y"].numpy()
                ) or not np.array_equal(supplied["edges"], graph["edges"].numpy()):
                    raise ValueError("Diagnostic graph IDs or edges disagree")
                graph["x"] = torch.from_numpy(supplied["x"].copy()).float()
        edges = torch.cat([graph["edges"], graph["edges"].flip(0)], dim=1)
        return Data(
            x=graph["x"], y=graph["y"], edge_index=edges, num_nodes=len(graph["y"])
        )

    converter = NodeToGraphConverter(
        str(args.data_root), str(args.output / "ego"), device, 42
    )
    if args.phase == "train":
        config = load_config(
            args.official / "configs/node/pretrain_photo_computers.yaml"
        )
        build = load_config(args.official / "configs/node/build_ppr_ego_graphs.yaml")[
            "build"
        ]
        (args.output / "configuration.json").write_text(
            json.dumps(
                dict(
                    train=config,
                    build=build,
                    source_graphs=["photo", "computers"],
                    source_seed=42,
                    graph_input="Original full graphs; canonical edges made bidirectional",
                    training_inputs="Official drop_node_features=true; source x discarded after ego extraction",
                ),
                indent=2,
            )
        )
        all_graphs = []
        for name in config["data"]["datasets"]:
            path = args.output / f"{name}_source_ego.pt"
            raw = load_local(name)
            # Advance the sampling RNG even when a source cache is reused, so
            # an interrupted build reproduces the same later source samples.
            nodes = converter.sample_nodes_per_class(
                raw, int(build["samples_per_class"])
            )
            if path.exists():
                graphs = torch.load(path, weights_only=False)
            else:
                raw = raw.to(device)
                graphs = []
                for start in range(0, len(nodes), int(build["batch_size"])):
                    ids = nodes[start : start + int(build["batch_size"])]
                    ppr = converter.ppr_batch(
                        raw.edge_index,
                        raw.num_nodes,
                        torch.tensor(ids, device=device),
                        float(build["alpha"]),
                        int(build["max_iter"]),
                        float(build["tol"]),
                    )
                    for i, node in enumerate(ids):
                        graph = converter.extract_subgraph(
                            raw, node, ppr[i], int(build["size"])
                        )
                        graph.x = None
                        graphs.append(graph)
                    print(
                        json.dumps(
                            dict(
                                stage="source_ego",
                                graph=name,
                                done=len(graphs),
                                total=len(nodes),
                            )
                        ),
                        flush=True,
                    )
                torch.save(graphs, path)
                del ppr
            del raw
            all_graphs.extend(graphs)
        print(
            json.dumps(
                dict(
                    stage="pretrain",
                    graphs=len(all_graphs),
                    epochs=config["train"]["epochs"],
                )
            ),
            flush=True,
        )
        pretrain_bases(all_graphs, config, device, args.output / "checkpoint")
        return

    checkpoint_path = args.checkpoint or args.output / "checkpoint/model.pt"
    model, _ = load_model_from_checkpoint(checkpoint_path, device)
    config = load_config(args.official / "configs/node/fewshot_cross_domain.yaml")
    encoder = SCGFMEncoder(model, device=device, **config["encoder"])
    manifest = dict(
        model="SCGFM",
        checkpoint=str(checkpoint_path),
        encoder=config["encoder"],
        input=(
            args.input_policy
            if args.input_directory is not None
            else "Original full graph features"
        ),
        targets=args.graphs,
        support_seeds=[42, 43, 44],
        shots=[1, 5, 20],
        readout="Official support-only StandardScaler and ProtoClassifier",
        target_split="Standard EvoGFM NC train/val/test; full test nodes",
        streaming=args.stream,
    )
    (args.output / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    rows = []
    if (args.output / "scores.csv").exists():
        with (args.output / "scores.csv").open() as handle:
            rows = list(csv.DictReader(handle))
    done = {(row["graph"], int(row["seed"]), int(row["shots"])) for row in rows}
    for name in args.graphs:
        if all(
            (name, seed, shot) in done for seed in [42, 43, 44] for shot in [1, 5, 20]
        ):
            continue
        path = args.output / f"{name}_embeddings.pt"
        original = load_local(name)
        labels = original.y.cpu()
        protocol = load_protocol(name, "nc", args.data_root)
        streamed = None
        if args.stream:
            streamed = streamed_predictions(
                original.to(device), labels, protocol, converter, encoder, device, name
            )
        elif path.exists():
            h = torch.load(path, weights_only=False)
        else:
            raw = original.to(device)
            vectors = []
            with torch.no_grad():
                for start in range(0, raw.num_nodes, 64):
                    ids = torch.arange(
                        start, min(start + 64, raw.num_nodes), device=device
                    )
                    scores = converter.ppr_batch(
                        raw.edge_index, raw.num_nodes, ids, 0.15, 100, 1e-6
                    )
                    for i, node in enumerate(ids.tolist()):
                        ego = converter.extract_subgraph(raw, node, scores[i], 100)
                        vectors.append(encoder.encode_single(ego).cpu())
                    print(
                        json.dumps(
                            dict(
                                stage="target_encode",
                                graph=name,
                                done=len(vectors),
                                total=raw.num_nodes,
                            )
                        ),
                        flush=True,
                    )
            h = torch.stack(vectors)
            torch.save(h, path)
            original = original.cpu()
        for seed in [42, 43, 44]:
            for shot in [1, 5, 20]:
                if (name, seed, shot) in done:
                    continue
                support = episode(
                    labels,
                    protocol["train"],
                    torch.Generator().manual_seed(seed),
                    shots=shot,
                    queries=0,
                )[0]
                if streamed is not None:
                    support, prediction = streamed[(seed, shot)]
                else:
                    scaler = StandardScaler()
                    support_x = torch.from_numpy(
                        scaler.fit_transform(h[support].numpy())
                    ).float()
                    test_x = torch.from_numpy(
                        scaler.transform(h[protocol["test"]].numpy())
                    ).float()
                    clf = ProtoClassifier(device=device)
                    clf.fit(support_x, labels[support])
                    prediction = clf.predict(test_x).cpu()
                row = dict(
                    graph=name,
                    seed=seed,
                    shots=shot,
                    accuracy_percent=100
                    * float(prediction.eq(labels[protocol["test"]]).double().mean()),
                    support_count=len(support),
                    test_count=len(protocol["test"]),
                )
                rows.append(row)
                np.savez_compressed(
                    args.output / f"{name}_s{seed}_k{shot}.npz",
                    prediction=prediction.numpy(),
                    support=support.numpy(),
                    test=protocol["test"].numpy(),
                )
                print(json.dumps(row), flush=True)
        with (args.output / "scores.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        del original, streamed
        if not args.stream:
            del h
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
