"""Train a fixed-capacity readout on harder, past-only negative candidates."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from threadpoolctl import threadpool_limits
import torch

from evogfm.data import canonical_edges
from evogfm.incremental import ingest_edges
from evogfm.shared_attention_cuda import CudaLinkScorer
from evogfm.temporal import TemporalHistory, prepare_dataset
from evogfm.three_hop_csr import three_hop_walk_counts
from experiments.db_a_cuda_quality import arrays
from experiments.db_a_native import WORKSPACE, DEFAULT_CHECKPOINT
from experiments.db_a_quality import baseline, cases, features, write_json
from experiments.db_a_rank_readout import inputs, measure, tree_nodes
from experiments.full_graph import load_checkpoint
from experiments.idea_a import frozen_structural_features


def sample_candidates(
    positive, original_negative, adjacency, history, rng, mode_counts
):
    active = np.flatnonzero(history.in_count + history.out_count)
    ordering = np.argsort(-history.in_count[active], kind="stable")
    popular = active[ordering[: max(16, len(active) // 8)]]
    extra = np.full((4, positive.shape[1]), -1, dtype=np.int64)
    for source in np.unique(positive[0]):
        selected = np.flatnonzero(positive[0] == source)
        one_hop = adjacency.indices[
            adjacency.indptr[source] : adjacency.indptr[source + 1]
        ]
        two_hop = np.unique(adjacency[one_hop].indices)
        blocked = np.r_[source, positive[1, selected]]
        pools = [
            np.setdiff1d(candidates, blocked)
            for candidates in (two_hop, two_hop, popular, one_hop)
        ]
        fallback = np.setdiff1d(active, blocked)
        for row in selected:
            used = [int(original_negative[1, row])]
            for kind, pool in enumerate(pools):
                eligible = np.setdiff1d(pool, used)
                if len(eligible):
                    mode_counts[kind] += 1
                else:
                    eligible = np.setdiff1d(fallback, used)
                    if not len(eligible):
                        continue
                result = int(rng.choice(eligible))
                used.append(result)
                extra[kind, row] = result
    return np.stack((np.broadcast_to(positive[0], extra.shape), extra), axis=0)


@torch.no_grad()
def hard_training_features(case, half_life):
    events = prepare_dataset(
        case["dataset"], "undergs", WORKSPACE / "data/temporal", WORKSPACE / "baselines"
    )
    candidates = Path(case["candidates"])
    index = json.loads((candidates / "index.json").read_text(encoding="utf-8"))
    model, head, _ = load_checkpoint(DEFAULT_CHECKPOINT)
    train = events.phase == "train"
    edges = torch.from_numpy(
        np.stack((events.source[train], events.destination[train]))
    )
    raw = frozen_structural_features(
        canonical_edges(edges, events.n_nodes), events.n_nodes, 100
    )
    first, *_ = arrays(candidates, index[0])
    visible = canonical_edges(torch.from_numpy(first), events.n_nodes)
    known = (visible[0] * events.n_nodes + visible[1]).numpy()
    history = TemporalHistory(events.n_nodes)
    history.ingest(first, index[0]["time"])
    scorer = CudaLinkScorer(
        model,
        head,
        raw,
        visible,
        resident_scores=True,
        path_counts=three_hop_walk_counts,
        async_indices=True,
        tile_edges=32768,
    )
    rng = np.random.default_rng(42)
    selected_counts = np.zeros(4, dtype=np.int64)
    by_time = {}
    for entry in index[1:]:
        if entry["phase"] != "train":
            break
        positive, original_negative, _ = arrays(candidates, entry)
        extra = sample_candidates(
            positive, original_negative, scorer.a, history, rng, selected_counts
        )
        valid = extra[1] >= 0
        pairs = extra.reshape(2, -1)[:, valid.ravel()].copy()
        matrix = []
        result = np.zeros((valid.size, 24), dtype=np.float32)
        if valid.any():
            for request in torch.from_numpy(pairs).split(2048, dim=1):
                graph = scorer.features(request).numpy()
                temporal = history.statistics(request.numpy(), entry["time"]).numpy()
                matrix.append(features(graph, temporal, "union24", half_life))
            result[valid.ravel()] = np.concatenate(matrix)
        by_time[entry["time"]] = (result.reshape(4, positive.shape[1], 24), valid)
        history.ingest(positive, entry["time"])
        visible, added, known = ingest_edges(
            visible, torch.from_numpy(positive), events.n_nodes, known
        )
        scorer.update(visible, added)
    return by_time, selected_counts.tolist()


def fit_with_negatives(model, positives, random_negatives, hard, weights):
    columns = [positives, random_negatives] + [
        hard[i] for i, weight in enumerate(weights) if weight
    ]
    sizes = [len(x) for x in columns]
    sample_weight = np.concatenate(
        [
            np.full(size, weight)
            for size, weight in zip(sizes, [1.0, 1.0, *[w for w in weights if w]])
        ]
    )
    sample_weight[sizes[0] :] *= sizes[1] / sample_weight[sizes[0] :].sum()
    y = np.r_[np.ones(sizes[0]), np.zeros(sum(sizes[1:]))]
    return clone(model).fit(np.concatenate(columns), y, sample_weight=sample_weight)


def select(case, previous, output):
    dataset = case["dataset"]
    previous_path = previous / dataset / "selected_readout.joblib"
    state = joblib.load(previous_path)
    case["half_life"] = state["half_life"]
    reference = state["model"]["fitted"]
    sampled, counts = hard_training_features(case, state["half_life"])
    train = list(inputs(case, "train"))
    positives = np.concatenate([x[: row["npos"]] for row, x in train])
    random_negatives = np.concatenate([x[row["npos"] :] for row, x in train])
    hard = np.concatenate([sampled[row["time"]][0] for row, _ in train], axis=1)
    valid = np.concatenate([sampled[row["time"]][1] for row, _ in train], axis=1)
    hard = [pool[mask] for pool, mask in zip(hard, valid)]
    validation = list(inputs(case, "validation"))
    grid = {
        "two_hop_light": (0.125, 0.125, 0.0, 0.0),
        "two_hop": (0.25, 0.25, 0.0, 0.0),
        "topology_activity": (0.25, 0.25, 0.25, 0.0),
        "topology_activity_repeat": (0.25, 0.25, 0.25, 0.25),
        "activity": (0.0, 0.0, 0.5, 0.0),
    }
    models = {
        name: fit_with_negatives(reference, positives, random_negatives, hard, weights)
        for name, weights in grid.items()
    }
    models["previous"] = reference
    val = {name: measure(model, validation) for name, model in models.items()}
    eligible = [
        name
        for name, row in val.items()
        if row["auc"] >= val["previous"]["auc"] - 0.0025
        and row["pessimistic_mrr"] >= val["previous"]["pessimistic_mrr"]
        and tree_nodes(models[name]) <= tree_nodes(reference)
    ]
    selected = max(
        eligible,
        key=lambda name: (val[name]["mrr"], val[name]["auc"], name == "previous"),
    )
    folder = output / dataset
    folder.mkdir(parents=True, exist_ok=True)
    state["model"]["fitted"] = models[selected]
    if selected != "previous":
        state["ranking_variant"] = selected
    joblib.dump(state, folder / "selected_readout.joblib", compress=3)
    lock = dict(
        dataset=dataset,
        selected=selected,
        val=val,
        hard_pool_matches=counts,
        seed=42,
        training_positives=len(positives),
        tree_nodes=tree_nodes(models[selected]),
        prior_readout=str(previous_path.resolve()),
    )
    write_json(folder / "validation_lock.json", lock)
    print(
        json.dumps({"dataset": dataset, "selected": selected, "val": val[selected]}),
        flush=True,
    )
    return lock


def run(previous, output):
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(42)
    work = [case for case in cases() if case["protocol"] == "undergs"]
    locks = [select(case, previous, output) for case in work]
    write_json(output / "all_validation_locks.json", locks)
    results = []
    for case, lock in zip(work, locks):
        folder = output / case["dataset"]
        state = joblib.load(folder / "selected_readout.joblib")
        reference = joblib.load(lock["prior_readout"])
        held_out = list(inputs(case, "test"))
        row = dict(
            **lock,
            base=baseline(case),
            test=measure(state["model"]["fitted"], held_out),
            previous_test=measure(reference["model"]["fitted"], held_out),
            scope="Existing development split, train-only extra negatives, CUDA feature extraction",
        )
        results.append(row)
        write_json(folder / "result.json", row)
        print(
            json.dumps(
                {"dataset": case["dataset"], "test": row["test"], "base": row["base"]}
            ),
            flush=True,
        )
    write_json(output / "summary.json", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.previous.resolve(), args.output.resolve())
