"""Three full-model continual streams with isolated supervision objectives.

NC / LP / joint update the same 32 coefficients. All arms see the same 60%
structural graph, source checkpoint, arrival order and four updates per graph.
"""

from . import settings

import argparse
from argparse import Namespace
import gc
import json
import math
import time
import types
from . import study as rs
from evogfm.data import make_pack
from evogfm.features import (
    task_features,
    contexts,
    pair_channels,
    pair_context,
    hard_negative_pool,
)
from .io import atomic_json

torch, np = rs.torch, rs.np
HERE = settings.REPO
ORIGINAL = settings.NC_RUN
ROOT = settings.STREAM_RUN
ARMS = ["nc", "lp", "joint"]
SEEDS = [42, 43, 44]
TARGETS = ["twitch-en", "twitch-de", "twitch-fr", "github-developer"]


def event(stage, **kw):
    item = dict(time=rs.now(), stage=stage, **kw)
    print(json.dumps(item), flush=True)
    atomic_json(ROOT / (stage.split("_")[0] + "_progress.json"), item)


def setup():
    for folder in [
        "cache",
        "states",
        "tables_v2",
        "measurements",
        "history",
        "heads",
        "figures",
        "raw",
    ]:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)


def study_instance():
    return rs.Study(
        Namespace(
            output=str(ROOT),
            threads=4,
            eval_device=settings.DEVICE,
            data=str(settings.DATA),
        )
    )


@torch.no_grad()
def remask_state(model, old, edges):
    # Whitening and the SVD feature frame depend only on fixed node features.
    # Reuse that frame while invoking the original full-edge state estimator.
    model.encode_features = types.MethodType(lambda self, x: x, model)
    try:
        return model.state(old.z, edges, 0.6)
    finally:
        del model.encode_features


def safe_hard_pool(edges, full, n):
    pool = hard_negative_pool(edges, n, per_node=32)
    keys = torch.sort(full[0] * n + full[1]).values
    candidates = pool[0] * n + pool[1]
    ix = torch.searchsorted(keys, candidates)
    valid = (ix == len(keys)) | (keys[ix.clamp_max(len(keys) - 1)] != candidates)
    return pool[:, valid]


def prepare_donors(study):
    for name in rs.DONORS:
        path = ROOT / "cache" / f"{name}.pt"
        if path.exists():
            continue
        event("prepare_donor", graph=name)
        old = torch.load(
            ORIGINAL / "cache" / f"{name}.pt", weights_only=False, mmap=True
        )
        native = make_pack(name, old["raw"], old["edges"], seed=42, query_cap=10**9)
        edges = native["state_edges"]
        state = remask_state(study.model, old["state"], edges)
        hard = safe_hard_pool(edges, old["edges"], len(old["y"]))
        perm = torch.randperm(
            hard.shape[1], generator=torch.Generator().manual_seed(31415)
        )
        split = max(1, len(perm) // 2)
        # Negative pools are disjoint between training and held-out selection.
        p = dict(
            name=name,
            raw=old["raw"],
            y=old["y"],
            protocol=old["protocol"],
            edges=edges,
            state=state,
            native=native,
            hard_train=hard[:, perm[:split]],
            hard_val=hard[:, perm[split:]],
        )
        rs.save_pt(path, p)
        del old, p, native
        gc.collect()


def sample_links(pack, split, seed):
    part = pack["native"][split]
    pos = part["pairs"][:, part["y"] == 1]
    uniform = part["pairs"][:, part["y"] == 0]
    hard = pack["hard_train" if split == "calibration" else "hard_val"]
    # Hard candidates must not overlap any uniform-negative split either.
    nnode = len(pack["y"])
    uniform_all = torch.cat(
        [
            pack["native"][k]["pairs"][:, pack["native"][k]["y"] == 0]
            for k in ["calibration", "lp_val", "lp_test"]
        ],
        dim=1,
    )
    uniform_keys = torch.sort(uniform_all[0] * nnode + uniform_all[1]).values
    hard_keys = hard[0] * nnode + hard[1]
    loc = torch.searchsorted(uniform_keys, hard_keys)
    allowed = (loc == len(uniform_keys)) | (
        uniform_keys[loc.clamp_max(len(uniform_keys) - 1)] != hard_keys
    )
    hard = hard[:, allowed]
    # Balanced positives vs a fixed 50:50 uniform/hard negative mixture.
    n = min(256, pos.shape[1], 2 * uniform.shape[1], 2 * hard.shape[1])
    n = (n // 2) * 2
    if n < 2:
        raise RuntimeError(f'Insufficient pair pools: {pack["name"]} {split}')
    g = torch.Generator().manual_seed(seed)
    p = pos[:, torch.randperm(pos.shape[1], generator=g)[:n]]
    u = uniform[:, torch.randperm(uniform.shape[1], generator=g)[: n // 2]]
    h = hard[:, torch.randperm(hard.shape[1], generator=g)[: n // 2]]
    return dict(
        pairs=torch.cat([p, u, h], dim=1), y=torch.cat([torch.ones(n), torch.zeros(n)])
    )


def table(pack, split, seed, prepared):
    path = ROOT / "tables_v2" / f'{pack["name"]}_{split}_{seed}.pt'
    if path.exists():
        return torch.load(path, weights_only=False)
    t = sample_links(pack, split, seed)
    t["ctx"] = contexts(pack["edges"], len(pack["y"]), t["pairs"])
    t["base"] = pair_channels(pack["raw"], pack["edges"], t["pairs"], prepared=prepared)
    rs.save_pt(path, t)
    return t


def node_partition(pack, seed):
    gseed = seed * 1000 + rs.DONORS.index(pack["name"]) * 10
    g = torch.Generator().manual_seed(gseed)
    adapt = []
    held = []
    for label in pack["y"][pack["protocol"]["train"]].unique():
        ids = pack["protocol"]["train"]
        ids = ids[pack["y"][ids] == label]
        ids = ids[torch.randperm(len(ids), generator=g)]
        count = min(20, max(1, len(ids) // 4))
        held.append(ids[:count])
        adapt.append(ids[count:])
    adapt, held = torch.cat(adapt), torch.cat(held)
    support = rs.episode(
        pack["y"], adapt, torch.Generator().manual_seed(gseed), shots=5, queries=0
    )[0]
    return gseed, adapt, held, support


def logits_both(study, pack, t, support, query):
    features, parts = task_features(
        study.model,
        pack["state"],
        pack["raw"],
        pack["edges"],
        t["pairs"],
        t["ctx"],
        t["base"],
    )
    link = study.head(features).squeeze(1)
    node, truth = rs.nc_logits(parts, pack["y"], support, query)
    return node, truth, link


def binary_ggn(study, logits, probes=4):
    prob = logits.detach().sigmoid()
    scale = (prob * (1 - prob) / len(prob)).sqrt()
    metric = torch.zeros(40, 40, dtype=torch.float64)
    for _ in range(probes):
        signs = (torch.randint(2, logits.shape) * 2 - 1).to(logits)
        gradient = study.gradient(logits, scale * signs).detach().double()
        metric += torch.outer(gradient, gradient) / probes
    return metric


def adapt(study, pack, ons, seed, arm, stateid):
    gseed, train, held, held_support = node_partition(pack, seed)
    prepared = pair_context(pack["raw"], pack["edges"])
    selected = table(pack, "lp_val", gseed + 999, prepared)
    steps = [table(pack, "calibration", gseed + i, prepared) for i in range(4)]
    selected_keys = set(
        (selected["pairs"][0] * len(pack["y"]) + selected["pairs"][1]).tolist()
    )
    for t in steps:
        assert not selected_keys.intersection(
            (t["pairs"][0] * len(pack["y"]) + t["pairs"][1]).tolist()
        )
    del prepared
    classes = len(pack["y"][train].unique())
    nc_scale = 1 / math.log(classes)
    lp_scale = 1 / math.log(2)
    weights = {"nc": (1.0, 0.0), "lp": (0.0, 1.0), "joint": (0.5, 0.5)}[arm]
    begin = time.perf_counter()
    records = []
    for step, t in enumerate(steps):
        rs.seed_all(gseed + step)
        event("train_step", state_id=stateid, graph=pack["name"], step=step + 1)
        support, query = rs.episode(
            pack["y"],
            train,
            torch.Generator().manual_seed(gseed + step),
            shots=5,
            queries=20,
        )
        node, truth, link = logits_both(study, pack, t, support, query)
        nl = rs.F.cross_entropy(node, truth) * nc_scale
        ll = rs.F.binary_cross_entropy_with_logits(link, t["y"]) * lp_scale
        ng = study.gradient(nl).detach()
        lg = study.gradient(ll).detach()
        gradient = weights[0] * ng + weights[1] * lg
        curvature = torch.zeros(40, 40, dtype=torch.float64)
        if weights[0]:
            curvature += weights[0] * nc_scale * study.ggn(node, 4)
        if weights[1]:
            curvature += weights[1] * lp_scale * binary_ggn(study, link, 4)
        before = study.model.inverse.theta().detach().clone()
        proposal = ons.step_gradient(gradient, curvature=curvature)
        rec = dict(
            step=step + 1,
            node_loss=float(nl.detach()),
            link_loss=float(ll.detach()),
            gradient_cosine=float(
                torch.dot(ng[:32], lg[:32])
                / (ng[:32].norm() * lg[:32].norm()).clamp_min(1e-20)
            ),
            node_gradient_norm=float(ng[:32].norm()),
            link_gradient_norm=float(lg[:32].norm()),
            nc_labels=len(torch.unique(torch.cat([support, query]))),
            lp_pairs=len(t["y"]),
        )
        del node, truth, link, nl, ll, ng, lg, gradient, curvature
        best = float("inf")
        chosen = None
        held_values = []
        for rate in [0.0, 0.25, 1.0, 4.0]:
            candidate = (before + rate * (proposal - before)).clamp(
                min=before.new_tensor(study.lower), max=before.new_tensor(study.upper)
            )
            candidate[32:] = study.initial[32:]
            study.model.inverse.set_theta(candidate)
            with torch.no_grad():
                node, truth, link = logits_both(
                    study, pack, selected, held_support, held
                )
                nv = float(rs.F.cross_entropy(node, truth)) * nc_scale
                lv = (
                    float(rs.F.binary_cross_entropy_with_logits(link, selected["y"]))
                    * lp_scale
                )
                value = weights[0] * nv + weights[1] * lv
                del node, truth, link
            held_values.append(
                dict(rate=rate, node_loss=nv, link_loss=lv, objective=value)
            )
            if value < best:
                best = value
                chosen = candidate.clone()
                selected_rate = rate
        study.model.inverse.set_theta(chosen)
        ons.theta = chosen.double().numpy().copy()
        rec.update(
            rate=selected_rate,
            held_candidates=held_values,
            change=float((chosen - before).norm()),
        )
        records.append(rec)
    for name, param in study.model.named_parameters():
        if name in study.frozen:
            assert torch.equal(param.detach(), study.frozen[name]), name
    assert torch.equal(study.model.inverse.beta.detach(), study.initial[32:])
    return dict(update_seconds=time.perf_counter() - begin, curve=records)


def train(seeds, arms):
    study = study_instance()
    prepare_donors(study)
    for seed in seeds:
        order = json.loads((ORIGINAL / f"orders_s{seed}.json").read_text())["random"]
        prior = torch.load(
            ORIGINAL / "states" / f"s{seed}_initial.pt", weights_only=False
        )
        for arm in arms:
            ons = study.new_online(prior)
            total = 0.0
            initial = dict(
                ons.state_dict(),
                seed=seed,
                kind=arm,
                prefix=0,
                order=order,
                cumulative_seconds=0.0,
            )
            rs.save_pt(ROOT / "states" / f"s{seed}_{arm}_00.pt", initial)
            for prefix, name in enumerate(order, 1):
                sid = f"s{seed}_{arm}_{prefix:02d}"
                path = ROOT / "states" / f"{sid}.pt"
                if path.exists():
                    state = torch.load(path, weights_only=False)
                    ons = study.load_online(state)
                    total = state["cumulative_seconds"]
                    continue
                pack = torch.load(
                    ROOT / "cache" / f"{name}.pt", weights_only=False, mmap=True
                )
                info = adapt(study, pack, ons, seed, arm, sid)
                total += info["update_seconds"]
                state = dict(
                    ons.state_dict(),
                    seed=seed,
                    kind=arm,
                    prefix=prefix,
                    order=order,
                    graph=name,
                    cumulative_seconds=total,
                    **info,
                )
                rs.save_pt(path, state)
                event("train_saved", state_id=sid)
                del pack
                gc.collect()
    event("train_complete", seeds=seeds, arms=arms)


def manifest():
    atomic_json(
        ROOT / "protocol.json",
        dict(
            source_checkpoint=str(settings.CHECKPOINT),
            source_graphs=["pubmed", "photo", "texas"],
            source_pretraining_replicates=1,
            update_seeds=SEEDS,
            arms=ARMS,
            donors=rs.DONORS,
            recipients=TARGETS,
            order="Original random order for each seed, identical across three arms",
            model="Full checkpoint: 32 shared coefficients; backbone, inverse feature network, 8 offsets and source LP head frozen",
            structure="All arms use the identical native60% structural edges. Hidden donor edges supply LP supervision only.",
            updates="Four GGN steps per graph, 15 graphs, identical eta1/ridge0.1 and source-curvature initialization; rate0/.25/1/4 selected on donor-held-out objective",
            objectives="NC CE/log(C); LP BCE/log(2); joint arithmetic mean. Loss scaling fixed before target evaluation.",
            link_supervision="Up to256 positives and256 negatives per step; negatives half uniform and half RA-hard; hard train/validation pools disjoint; all actual edges excluded.",
            budget="Equal optimizer steps and shared parameter count, not equal annotations or FLOPs; joint consumes both types of supervision.",
            nc_readout="Full original two-hop plus Transformer features, ridge0.1, up to 1/5/20 labels per class",
            lp_readout="All17 full-model channels; frozen source head and target residual head, ridge chosen on target validation only",
            future_evaluation="Recipients never update shared coefficients; NC and LP evaluated after every arrival0..15; every target test positive edge retained",
            history="First4 arrival graphs; per-arm classifier fitted at prefix4, frozen thereafter; report accuracy and change from own prefix4",
            data="New SNAP Twitch EN/DE/FR and GitHub developer graphs, excluded from prior28-graph exploration. Facebook replaced before any new-target scoring because SNAP marks it unavailable.",
            interpretation="Predeclared objective intervention on new recipients; no outcome-based changes to streams or losses.",
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["prepare", "train"])
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=ARMS)
    args = parser.parse_args()
    setup()
    manifest()
    if args.phase == "prepare":
        prepare_donors(study_instance())
    else:
        train(args.seeds, args.arms)
