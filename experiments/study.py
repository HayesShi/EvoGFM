"""Full-checkpoint NC assimilation and independent recipient evaluation.

Use the snapshot's original two-hop + Transformer readout in BOTH adaptation
and evaluation. Full-edge moments are accumulated in float64 blocks. Tests are
read only after calibration; they never select step sizes or stream ordering.
"""

import argparse
import copy
import csv
import gc
import json
import os
from pathlib import Path
import time
from datetime import datetime, timezone
from .io import atomic_json as write_json

from . import settings

HERE = settings.REPO
import numpy as np
import torch
import torch.nn.functional as F
from evogfm.data import load_graph, load_protocol, seed_all
from evogfm.features import preprocess, node_representations
from evogfm.readout import episode, nc_logits, fit_ridge
from .full_graph import load_checkpoint
from evogfm.online import OnlineNewton

DONORS = [
    "actor",
    "amazon-ratings",
    "chameleon",
    "citeseer",
    "coauthor-cs",
    "computers",
    "cora",
    "cornell",
    "dblp",
    "minesweeper",
    "questions",
    "roman-empire",
    "squirrel",
    "tolokers",
    "wisconsin",
]
RECIPIENTS = [
    "air-brazil",
    "air-eu",
    "air-us",
    "coauthor-physics",
    "blogcatalog",
    "deezer",
    "lastfm-asia",
    "ogbn-arxiv",
    "wiki",
    "wikics",
]
DOMAINS = {
    "actor": "film",
    "amazon-ratings": "commerce",
    "chameleon": "web",
    "citeseer": "citation",
    "coauthor-cs": "coauthorship",
    "computers": "commerce",
    "cora": "citation",
    "cornell": "web",
    "dblp": "citation",
    "minesweeper": "synthetic",
    "questions": "social",
    "roman-empire": "language",
    "squirrel": "web",
    "tolokers": "social",
    "wisconsin": "web",
}


def now():
    return datetime.now(timezone.utc).isoformat()


def save_pt(path, value):
    temp = path.with_suffix(".tmp")
    torch.save(value, temp)
    os.replace(temp, path)


def event(out, stage, **info):
    value = dict(time=now(), stage=stage, **info)
    print(json.dumps(value, ensure_ascii=False), flush=True)
    write_json(out / "progress.json", value)


def append_csv(path, row):
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


class Study:
    def __init__(self, args):
        self.args, self.out = args, Path(args.output).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        for folder in ["states", "cache", "measurements", "heads"]:
            (self.out / folder).mkdir(exist_ok=True)
        torch.set_num_threads(args.threads)
        self.model, self.head, self.ckpt = load_checkpoint(
            getattr(args, "checkpoint", settings.CHECKPOINT)
        )
        for p in self.model.parameters():
            p.requires_grad_(False)
        for p in self.head.parameters():
            p.requires_grad_(False)
        self.model.inverse.c.requires_grad_(True)
        self.model.inverse.beta.requires_grad_(True)
        self.parameters = [self.model.inverse.c, self.model.inverse.beta]
        self.initial = self.model.inverse.theta().detach().clone()
        self.k = self.model.inverse.k
        assert self.k == 32 and len(self.initial) == 40
        assert self.model.config.get("joint_task_backbone")
        self.lower, self.upper = self.model.inverse.bounds()
        self.upper[: self.k] = [max(4.0, 4 * float(v)) for v in self.initial[: self.k]]
        self.frozen = {
            n: p.detach().clone()
            for n, p in self.model.named_parameters()
            if n not in {"inverse.c", "inverse.beta"}
        }

    def pack(self, name):
        start = time.perf_counter()
        path = self.out / "cache" / f"{name}.pt"
        if path.exists():
            p = torch.load(path, weights_only=False)
            if name in self.ckpt["source_graphs"]:
                p["protocol"] = self.source_protocol(name, p["y"])
            p["cache_read_seconds"] = time.perf_counter() - start
            return p
        event(self.out, "prepare_graph", graph=name)
        data = load_graph(name, self.args.data)
        raw, x = preprocess(data)
        protocol = (
            self.source_protocol(name, data["y"])
            if name in self.ckpt["source_graphs"]
            else load_protocol(name, "nc", self.args.data)
        )
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
            assert not np.intersect1d(protocol[a].numpy(), protocol[b].numpy()).size
        with torch.no_grad():
            state = self.model.state(x, data["edges"])
        p = dict(
            name=name,
            raw=raw,
            edges=data["edges"],
            y=data["y"],
            protocol=protocol,
            state=state,
            preprocessing_seconds=time.perf_counter() - start,
        )
        save_pt(path, p)
        return p

    def source_protocol(self, name, y):
        # Reuse the actual source-training label partition, including Texas,
        # whose source graph has no public NC masks in the original loader.
        splits = json.loads(settings.SOURCE_SPLITS.read_text())
        train = torch.tensor(splits[name]["train"], dtype=torch.long)
        val = torch.tensor(splits[name]["val"], dtype=torch.long)
        mask = torch.ones(len(y), dtype=torch.bool)
        mask[train] = False
        mask[val] = False
        return dict(train=train, val=val, test=torch.where(mask)[0])

    def parts(self, p, inference=False):
        parts, h = node_representations(
            self.model,
            p["state"],
            p["raw"],
            p["edges"],
            hops=2,
            inference_device=self.args.eval_device if inference else "cpu",
        )
        return tuple(parts) + (h,)

    def gradient(self, output, grad_outputs=None):
        grads = torch.autograd.grad(
            output,
            self.parameters,
            grad_outputs=grad_outputs,
            retain_graph=True,
            allow_unused=True,
        )
        g = torch.cat(
            [
                torch.zeros_like(p) if v is None else v
                for p, v in zip(self.parameters, grads)
            ]
        )
        g[self.k :] = 0
        return g

    def ggn(self, logits, probes):
        p = logits.detach().softmax(1)
        scale = p.sqrt()
        # A Gram matrix is PSD analytically. Accumulating rank-one terms in
        # float32 can nevertheless produce a small negative eigenvalue after
        # the online optimizer casts it to float64 for its PSD check.
        metric = torch.zeros(
            len(self.initial),
            len(self.initial),
            dtype=torch.float64,
            device=logits.device,
        )
        legacy = logits.new_zeros(len(self.initial), len(self.initial))
        for _ in range(probes):
            signs = (torch.randint(2, logits.shape) * 2 - 1).to(logits)
            output_grad = (
                scale
                * (signs - scale * (scale * signs).sum(1, keepdim=True))
                / len(p) ** 0.5
            )
            g = self.gradient(logits, output_grad).detach()
            legacy += torch.outer(g, g) / probes
            gd = g.double()
            metric += torch.outer(gd, gd) / probes
        old_eig = torch.linalg.eigvalsh(legacy.double())
        new_eig = torch.linalg.eigvalsh(metric)
        if float(old_eig.min()) < -1e-8:
            diagnostic = dict(
                time=now(),
                old_float32_min_eigenvalue=float(old_eig.min()),
                old_float32_max_eigenvalue=float(old_eig.max()),
                float64_min_eigenvalue=float(new_eig.min()),
                float64_max_eigenvalue=float(new_eig.max()),
                relative_roundoff=float(
                    (legacy.double() - metric).norm() / metric.norm().clamp_min(1e-30)
                ),
                probes=probes,
            )
            with (self.out / "curvature_numerics.jsonl").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write(json.dumps(diagnostic) + "\n")
            print(
                json.dumps(dict(stage="curvature_precision_diagnostic", **diagnostic)),
                flush=True,
            )
        return metric

    def new_online(self, prior):
        ons = OnlineNewton(self.initial, self.lower, self.upper, eta=1.0, ridge=0.1)
        ons.metric = np.asarray(prior["metric"]).copy()
        self.model.inverse.set_theta(self.initial)
        return ons

    def load_online(self, state):
        ons = self.new_online(state)
        ons.theta = np.asarray(state["theta"]).copy()
        ons.updates = state.get("updates", 0)
        self.model.inverse.set_theta(torch.as_tensor(ons.theta).float())
        return ons

    def prior(self, seed):
        path = self.out / "states" / f"s{seed}_initial.pt"
        if path.exists():
            return torch.load(path, weights_only=False)
        seed_all(seed)
        self.model.inverse.set_theta(self.initial)
        metric = torch.eye(len(self.initial), dtype=torch.float64) * 0.1
        begin = time.perf_counter()
        for name in self.ckpt["source_graphs"]:
            event(self.out, "source_curvature", seed=seed, graph=name)
            p = self.pack(name)
            parts = self.parts(p)
            support, query = episode(
                p["y"], p["protocol"]["train"], torch.Generator().manual_seed(seed)
            )
            if not len(query):
                raise ValueError(f"No source queries: {name}")
            logits, _ = nc_logits(parts, p["y"], support, query)
            metric += self.ggn(logits, 8)
            del parts, logits, p
            gc.collect()
        state = self.new_online(dict(metric=metric.double().numpy())).state_dict()
        state.update(
            seed=seed,
            kind="initial",
            prefix=0,
            graph="",
            cumulative_seconds=0.0,
            prior_seconds=time.perf_counter() - begin,
            order=[],
        )
        save_pt(path, state)
        event(self.out, "source_curvature_saved", seed=seed, checkpoint=str(path))
        return state

    def adapt(self, p, ons, seed, state_id):
        name = p["name"]
        gseed = seed * 1000 + DONORS.index(name) * 10
        generator = torch.Generator().manual_seed(gseed)
        adapt, held = [], []
        for label in p["y"][p["protocol"]["train"]].unique():
            ids = p["protocol"]["train"]
            ids = ids[p["y"][ids] == label]
            ids = ids[torch.randperm(len(ids), generator=generator)]
            count = min(20, max(1, len(ids) // 4))
            held.append(ids[:count])
            adapt.append(ids[count:])
        adapt, held = torch.cat(adapt), torch.cat(held)
        support_held = episode(
            p["y"], adapt, torch.Generator().manual_seed(gseed), shots=5, queries=0
        )[0]
        curve, seen = [], set(held.tolist() + support_held.tolist())
        begin = time.perf_counter()
        for step in range(4):
            seed_all(gseed + step)
            event(
                self.out,
                "calibration_step",
                state_id=state_id,
                graph=name,
                step=step + 1,
            )
            parts = self.parts(p)
            support, query = episode(
                p["y"],
                adapt,
                torch.Generator().manual_seed(gseed + step),
                shots=5,
                queries=20,
            )
            if not len(query):
                raise ValueError(f"No disjoint calibration queries: {name}")
            seen.update(support.tolist())
            seen.update(query.tolist())
            logits, truth = nc_logits(parts, p["y"], support, query)
            loss = F.cross_entropy(logits, truth)
            curvature = self.ggn(logits, 4)
            gradient = self.gradient(loss).detach()
            before = self.model.inverse.theta().detach().clone()
            proposed = ons.step_gradient(gradient, curvature=curvature)
            loss_value = float(loss.detach())
            del parts, logits, loss, curvature
            best, chosen, rate_chosen = float("inf"), None, None
            for rate in [0.0, 0.25, 1.0, 4.0]:
                candidate = (before + rate * (proposed - before)).clamp(
                    min=before.new_tensor(self.lower), max=before.new_tensor(self.upper)
                )
                candidate[self.k :] = self.initial[self.k :]
                self.model.inverse.set_theta(candidate)
                with torch.no_grad():
                    hp = self.parts(p)
                    nc, truth = nc_logits(hp, p["y"], support_held, held)
                    value = float(F.cross_entropy(nc, truth))
                    del hp, nc
                if value < best:
                    best, chosen, rate_chosen = value, candidate.clone(), rate
            self.model.inverse.set_theta(chosen)
            ons.theta = chosen.double().numpy().copy()
            curve.append(
                dict(
                    step=step + 1,
                    training_loss=loss_value,
                    selection_loss=best,
                    rate=rate_chosen,
                    coefficient_change=float((chosen - self.initial).norm()),
                )
            )
        assert torch.equal(self.model.inverse.beta.detach(), self.initial[self.k :])
        for n, p0 in self.model.named_parameters():
            if n in self.frozen:
                assert torch.equal(p0.detach(), self.frozen[n]), n
        return dict(
            update_seconds=time.perf_counter() - begin,
            curve=curve,
            unique_training_labels=len(seen),
            training_nodes=sorted(seen),
        )

    def orders(self, seed):
        rng = np.random.default_rng(seed)
        shuffled = list(rng.permutation(DONORS))
        prefix, remaining = shuffled[:4], shuffled[4:]
        grouped = sorted(remaining, key=lambda n: DOMAINS[n])
        buckets = {}
        for n in grouped:
            buckets.setdefault(DOMAINS[n], []).append(n)
        interleaved = []
        while any(buckets.values()):
            for group in buckets.values():
                if group:
                    interleaved.append(group.pop(0))
        # One reproducible, label-free statistic; not selected by recipient scores.
        descriptor = {}
        for n in remaining:
            p = self.pack(n)
            v = p["state"].spectrum.double().square()
            v = v / v.sum().clamp_min(1e-12)
            descriptor[n] = float((-(v * v.clamp_min(1e-12).log()).sum()).exp())
            del p
        spectral = sorted(remaining, key=lambda n: descriptor[n])
        return {
            "random": shuffled,
            "domain_grouped": prefix + grouped,
            "spectral_grouped": prefix + spectral,
            "interleaved": prefix + interleaved,
        }

    def calibrate(self):
        for seed in self.args.seeds:
            prior = self.prior(seed)
            orders = self.orders(seed)
            write_json(self.out / f"orders_s{seed}.json", orders)
            schedules = [("stream_" + name, order) for name, order in orders.items()]
            schedules += [("donor_" + name, [name]) for name in DONORS]
            for kind, order in schedules:
                ons = self.new_online(prior)
                cumulative = 0.0
                for prefix, name in enumerate(order, 1):
                    state_id = f"s{seed}_{kind}_{prefix:02d}"
                    path = self.out / "states" / f"{state_id}.pt"
                    if path.exists():
                        state = torch.load(path, weights_only=False)
                        ons = self.load_online(state)
                        cumulative = state["cumulative_seconds"]
                        continue
                    p = self.pack(name)
                    info = self.adapt(p, ons, seed, state_id)
                    # Charge uncached preprocessing consistently to each method.
                    assimilation = info["update_seconds"] + p["preprocessing_seconds"]
                    cumulative += assimilation
                    state = copy.deepcopy(ons.state_dict())
                    state.update(
                        seed=seed,
                        kind=kind,
                        graph=name,
                        prefix=prefix,
                        cumulative_seconds=cumulative,
                        order=order,
                        preprocessing_seconds=p["preprocessing_seconds"],
                        **info,
                    )
                    save_pt(path, state)
                    append_csv(
                        self.out / "resources.csv",
                        dict(
                            state_id=state_id,
                            seed=seed,
                            kind=kind,
                            prefix=prefix,
                            graph=name,
                            nodes=len(p["y"]),
                            edges=p["edges"].shape[1],
                            unique_training_labels=info["unique_training_labels"],
                            preprocessing_seconds=p["preprocessing_seconds"],
                            update_seconds=info["update_seconds"],
                            cumulative_seconds=cumulative,
                            persistent_array_bytes=sum(
                                np.asarray(state[k]).nbytes
                                for k in ["theta", "metric", "lower", "upper"]
                            ),
                        ),
                    )
                    event(
                        self.out,
                        "checkpoint_saved",
                        state_id=state_id,
                        checkpoint=str(path),
                        coefficient_change=info["curve"][-1]["coefficient_change"],
                    )
                    del p
                    gc.collect()

    def evaluate(self):
        paths = sorted((self.out / "states").glob("s*.pt"))
        states = [(p.stem, torch.load(p, weights_only=False)) for p in paths]
        for recipient in RECIPIENTS:
            p = self.pack(recipient)
            for state_id, state in states:
                target = self.out / "measurements" / f"{state_id}__{recipient}.json"
                if target.exists():
                    continue
                self.model.inverse.set_theta(torch.as_tensor(state["theta"]).float())
                event(
                    self.out, "evaluate_recipient", state_id=state_id, graph=recipient
                )
                begin = time.perf_counter()
                rows = []
                with torch.no_grad():
                    parts = self.parts(p, inference=True)
                    h = torch.cat(parts, dim=1)
                    del parts
                    for shots in [1, 5, 20]:
                        support, _ = episode(
                            p["y"],
                            p["protocol"]["train"],
                            torch.Generator().manual_seed(state["seed"]),
                            shots=shots,
                            queries=0,
                        )
                        classes = p["y"][support].unique(sorted=True)
                        # Fixed alpha: test labels cannot affect fitting or selection.
                        fitted = fit_ridge(
                            h, p["y"], support, p["protocol"]["val"], alphas=(0.1,)
                        )
                        correct, total = 0, 0
                        for ids in p["protocol"]["test"].split(1024):
                            logits = (
                                h[ids].double() - fitted.mean
                            ) @ fitted.weight + fitted.bias
                            correct += int(
                                (classes[logits.argmax(1)] == p["y"][ids]).sum()
                            )
                            total += len(ids)
                        rows.append(
                            dict(
                                state_id=state_id,
                                graph=recipient,
                                seed=state["seed"],
                                kind=state["kind"],
                                prefix=state["prefix"],
                                shots=shots,
                                support_count=len(support),
                                min_class_support=min(
                                    int((p["y"][support] == c).sum()) for c in classes
                                ),
                                test_count=total,
                                accuracy_percent=100 * correct / total,
                                cumulative_assimilation_seconds=state[
                                    "cumulative_seconds"
                                ],
                            )
                        )
                        del fitted
                    del h
                write_json(
                    target,
                    dict(rows=rows, evaluation_seconds=time.perf_counter() - begin),
                )
                event(self.out, "recipient_saved", state_id=state_id, graph=recipient)
                gc.collect()
            del p
            gc.collect()
        self.collect()

    def collect(self):
        rows = []
        for p in sorted((self.out / "measurements").glob("*.json")):
            rows.extend(json.loads(p.read_text(encoding="utf-8"))["rows"])
        if rows:
            path = self.out / "probe_scores.csv"
            with path.open("w", newline="", encoding="utf-8") as h:
                w = csv.DictWriter(h, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)

    def verify(self):
        """One real-graph critical-path check; never a reported measurement."""
        from evogfm.features import task_features, contexts, pair_channels

        p = self.pack("cornell")
        edges, raw = p["edges"], p["raw"]
        pair = edges[:, :1]
        with torch.no_grad():
            reference = task_features(
                self.model,
                p["state"],
                raw,
                edges,
                pair,
                contexts(edges, len(raw), pair),
                pair_channels(raw, edges, pair),
            )[1]
            actual = self.parts(p)
            errors = [float((a - b).abs().max()) for a, b in zip(reference, actual)]
            assert max(errors) < 1e-5, errors
            z = p["state"].z
            c = (
                p["state"].density
                * (z[edges[0]].double().T @ z[edges[1]].double())
                / edges.shape[1]
            )
            c = (c + c.T) / 2
            reconstructed = (
                p["state"].left.double() * p["state"].spectrum.double()
            ) @ p["state"].right.double().T
            moment_error = float((c - reconstructed).norm() / c.norm())
            assert moment_error < 1e-4, moment_error
            if self.args.eval_device == "cuda":
                gpu = self.parts(p, inference=True)
                gpu_errors = [float((a - b).abs().max()) for a, b in zip(actual, gpu)]
                assert max(gpu_errors) < 2e-4, gpu_errors
            else:
                gpu_errors = None
        ons = self.new_online(dict(metric=np.eye(40) * 0.1))
        info = self.adapt(p, ons, 991, "engineering_check")
        write_json(
            self.out / "verification.json",
            dict(
                readout_max_errors=errors,
                moment_relative_error=moment_error,
                cuda_errors=gpu_errors,
                frozen_parameter_check=True,
                update_info=info,
                paper_result=False,
            ),
        )
        self.model.inverse.set_theta(self.initial)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(settings.DATA))
    parser.add_argument("--output", default=str(settings.NC_RUN))
    parser.add_argument("--checkpoint", default=str(settings.CHECKPOINT))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument(
        "--eval-device", choices=["cpu", "cuda"], default=settings.DEVICE
    )
    parser.add_argument(
        "--phase",
        choices=["verify", "calibrate", "evaluate", "history", "all", "collect"],
        default="all",
    )
    args = parser.parse_args()
    study = Study(args)
    config = dict(
        model_config=study.ckpt["model_config"],
        sources=study.ckpt["source_graphs"],
        source_checkpoint=str(args.checkpoint),
        donor_graphs=DONORS,
        recipients=RECIPIENTS,
        seeds=args.seeds,
        source_seed_replicates=1,
        moment="all visible positive edges, float64 block sum",
        updated_coefficients=32,
        frozen_offsets=8,
        curvature_shape=[40, 40],
        update_steps_per_arrival=4,
        source_ggn_probes=8,
        update_ggn_probes=4,
        readout_hops=2,
        transformer_readout=True,
        readout_ridge=0.1,
        shots=[1, 5, 20],
        eval_split="test",
        eval_device=args.eval_device,
        training_device="cpu",
        source_curvature_view="full graph NC",
        selection="within public training labels; 5-shot support; held out at most 20 per class",
        original_adaptation_changes=[
            "full-edge moment",
            "seed parameterization",
            "full-edge source NC curvature",
            "fixed 5-shot training-only step selection",
            "removed unused extra GGN draw",
        ],
        recipients_previously_inspected_in_project=True,
        formal_scope="Existing public benchmark extension; not independent source-pretraining replications",
        historical_readout="5-shot fixed ridge fitted at common prefix 4; first 4 graphs evaluated at prefixes 4 through 15",
        deferred=[
            "external GFM baselines",
            "conditional Q/K generator",
            "LP/GC probes",
            "independent-domain rule validation",
            "synthetic recovery evolution",
            "offline joint reference",
        ],
    )
    config["curvature_accumulation"] = (
        "float64 rank-one sum; float32 coefficient gradients"
    )
    if args.phase != "collect":
        write_json(study.out / "configuration.json", config)
        event(study.out, "started", phase=args.phase, pid=os.getpid())
    if args.phase == "verify":
        study.verify()
    if args.phase in ["calibrate", "all"]:
        study.calibrate()
    if args.phase in ["evaluate", "all"]:
        study.evaluate()
    if args.phase in ["history", "all"]:
        from .history import evaluate_history

        evaluate_history(study)
    if args.phase == "collect":
        study.collect()
    if args.phase != "collect":
        event(study.out, "completed", phase=args.phase)


if __name__ == "__main__":
    main()
