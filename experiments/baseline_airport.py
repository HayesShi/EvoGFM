"""Official GraphAny/GraphPFN inference on exported, matched NC inputs.

Run each model in its official environment. Input and episode NPZ files are
exported by ``experiments.airport_inputs``; no EvoGFM imports are required.
GraphPFN accepts one episode per process to release its ensemble memory.
"""

import argparse
import csv
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch


def load_episode(root, graph, policy, seed, shots):
    with np.load(root / "inputs" / f"{graph}_{policy}.npz") as source:
        data = {key: source[key].copy() for key in source.files}
    episode_path = root / "episodes" / f"{graph}_s{seed}_k{shots}.npz"
    if not episode_path.exists():
        episode_path = root / f"{graph}_{policy}_EvoGFM_s{seed}_k{shots}.npz"
    with np.load(episode_path) as source:
        support, test = source["support"].copy(), source["test"].copy()
    if not np.array_equal(test, data["test"]):
        raise ValueError("Test IDs disagree with the exported input")
    if not np.isin(support, data["train"]).all() or np.intersect1d(support, test).size:
        raise ValueError("Invalid support IDs")
    return data, support, test


def graphany_predict(data, support, model, config, cache):
    import dgl
    from omegaconf import OmegaConf
    from graphany.data import GraphDataset

    dataset = GraphDataset.__new__(GraphDataset)
    dataset.cfg = OmegaConf.create(dict(config, n_per_label_examples=5))
    dataset.name = "matched-airport"
    dataset.label = torch.from_numpy(data["y"]).long()
    dataset.num_class = int(dataset.label.max()) + 1
    dataset.preprocess_device = torch.device("cpu")
    dataset.train_indices = torch.from_numpy(support).long()
    dataset.cache_f_name = str(cache / "features.pt")
    dataset.dist_f_name = str(cache / "distances.pt")
    edges = data["edges"]
    graph = dgl.graph((edges[0], edges[1]), num_nodes=len(data["y"]))
    graph = dgl.to_simple(dgl.to_bidirected(dgl.remove_self_loop(graph)))
    features, _, _ = dataset.prepare_prop_features_logits_and_dist_features(
        graph, torch.from_numpy(data["x"]).float(), 2
    )
    logits = dataset.compute_channel_logits(
        features, dataset.train_indices, sample=False, device="cpu"
    )
    with torch.no_grad():
        prediction, _ = model(
            {key: value[data["test"]] for key, value in logits.items()}
        )
    return prediction.numpy()


def graphpfn_predict(data, support, args):
    import dgl
    from graphpfn import GraphDataset, predict_icl
    import graphpfn.model.limix as limix

    # Resolve locally downloaded official weights without changing inference.
    limix._download_limix_checkpoint = lambda: str(args.limix.resolve())
    n = len(data["y"])
    masks = {key: np.zeros(n, dtype=bool) for key in ["train", "val", "test"]}
    masks["train"][support] = True
    for key in ["val", "test"]:
        masks[key][data[key]] = True
    edges = data["edges"]
    dataset = GraphDataset(
        name=args.graph,
        graph=dgl.graph((edges[0], edges[1]), num_nodes=n),
        features={"other": data["x"].astype(np.float32)},
        targets=data["y"],
        masks=masks,
        task_type="multiclass",
    )
    prediction = predict_icl(
        dataset=dataset,
        model_kwargs={"checkpoint": args.checkpoint},
        device="cpu",
        amp=False,
    )
    return prediction[data["test"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=["graphany", "graphpfn"])
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--limix", type=Path)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", default="degree_quantile64")
    parser.add_argument("--graph")
    parser.add_argument("--seed", type=int, choices=[42, 43, 44])
    parser.add_argument("--shots", type=int, choices=[1, 5, 20])
    args = parser.parse_args()
    if args.model == "graphpfn" and (
        args.limix is None
        or any(value is None for value in [args.graph, args.seed, args.shots])
    ):
        parser.error("GraphPFN requires --limix, --graph, --seed and --shots")
    sys.path.insert(0, str(args.official.resolve()))
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    model, config = None, None
    if args.model == "graphany":
        from graphany.model import GraphAny
        from omegaconf import OmegaConf

        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        config = OmegaConf.to_container(checkpoint["graph_any_config"], resolve=True)
        model = GraphAny(**config)
        model.load_state_dict(
            {
                key.removeprefix("gnn_model."): value
                for key, value in checkpoint["state_dict"].items()
                if key.startswith("gnn_model.")
            }
        )
        model.eval()
    for graph in [args.graph] if args.graph else ["air-brazil", "air-eu", "air-us"]:
        for seed in [args.seed] if args.seed else [42, 43, 44]:
            for shots in [args.shots] if args.shots else [1, 5, 20]:
                key = f"{graph}_s{seed}_k{shots}"
                score_path = args.output / f"{key}.json"
                if score_path.exists():
                    continue
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                data, support, test = load_episode(
                    args.inputs, graph, args.policy, seed, shots
                )
                cache = args.output / "cache" / graph
                cache.mkdir(parents=True, exist_ok=True)
                start = time.perf_counter()
                prediction = (
                    graphany_predict(data, support, model, config, cache)
                    if args.model == "graphany"
                    else graphpfn_predict(data, support, args)
                )
                if (
                    prediction.shape != (len(test), len(np.unique(data["y"])))
                    or not np.isfinite(prediction).all()
                ):
                    raise ValueError("Unexpected official model predictions")
                row = dict(
                    model=args.model,
                    graph=graph,
                    policy=args.policy,
                    seed=seed,
                    shots=shots,
                    accuracy_percent=100
                    * float((prediction.argmax(1) == data["y"][test]).mean()),
                    support_count=len(support),
                    test_count=len(test),
                    seconds=time.perf_counter() - start,
                )
                np.savez_compressed(
                    args.output / f"{key}.npz",
                    scores=prediction,
                    support=support,
                    test=test,
                )
                score_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
                print(json.dumps(row), flush=True)
    rows = [
        json.loads(path.read_text())
        for path in sorted(args.output.glob("*_s*_k*.json"))
    ]
    if rows:
        with (args.output / "scores.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
