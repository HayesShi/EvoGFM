"""Task-conditioned continual calibration on a frozen shared backbone.

NC and LP retain two small coefficient/metric states, irrespective of graph count.
The nonlinear final-logit GGN is an optimizer; convex ONS regret bounds do not
apply to this path. Reports use validation splits.
"""

import copy
import json
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
from .data import load_graph, load_protocol, make_pack, seed_all, save_json
from .features import preprocess, contexts, task_features, pair_channels
from .readout import episode, nc_logits, ridge
from .training import load_checkpoint, source_packs, score_source
from .online import OnlineNewton, TaskCoefficientBank, continual_metrics


def calibrate(
    task,
    checkpoint,
    data_root,
    output,
    stream,
    future,
    threads,
    nc_metric="accuracy",
    source_anchor=0.0,
    controls=("static", "cumulative", "reset"),
    source_states=None,
):
    seed_all(42)
    torch.set_num_threads(threads)
    model, head, ckpt = load_checkpoint(checkpoint)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in head.parameters():
        p.requires_grad_(False)
    model.inverse.c.requires_grad_(True)
    model.inverse.beta.requires_grad_(True)
    parameters = [model.inverse.c, model.inverse.beta]
    k = model.inverse.k
    nc_only = task == "nc"
    packs = {}
    nc_packs = {}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name in list(stream) + list(future):
        data = load_graph(name, data_root)
        raw, x = preprocess(data)
        p = make_pack(name, x, data["edges"], seed=42, query_cap=2048)
        p["node_y"] = data["y"]
        if name in stream:
            p["nc_pool"] = load_protocol(name, "nc", data_root)["train"]
            nc_train = []
            nc_held = []
            generator = torch.Generator().manual_seed(42)
            for label in p["node_y"][p["nc_pool"]].unique():
                ids = p["nc_pool"][p["node_y"][p["nc_pool"]] == label]
                ids = ids[torch.randperm(len(ids), generator=generator)]
                held_count = max(1, len(ids) // 4)
                nc_train.append(ids[held_count:])
                nc_held.append(ids[:held_count])
            p["nc_adapt"] = torch.cat(nc_train)
            p["nc_held"] = torch.cat(nc_held)
        p["raw"] = raw
        p["state"] = model.state(x, p["state_edges"], 0.6)
        p["tables"] = {}
        for split in ["calibration", "lp_val"]:
            part = p[split]
            p["tables"][split] = dict(
                ctx=contexts(p["state_edges"], len(x), part["pairs"]),
                base=pair_channels(raw, p["state_edges"], part["pairs"]),
            )
        packs[name] = p
        edges = data["edges"]
        dummy = edges[:, :1]
        nc_packs[name] = dict(
            state=model.state(x, edges),
            raw=raw,
            edges=edges,
            pairs=dummy,
            ctx=contexts(edges, len(raw), dummy),
            base=pair_channels(raw, edges, dummy),
            y=data["y"],
            protocol=load_protocol(name, "nc", data_root),
        )
        print("continual prepared", name, flush=True)
    initial = model.inverse.theta().detach().clone()
    lower, upper = model.inverse.bounds()
    upper[:k] = [max(4.0, 4 * float(v)) for v in initial[:k]]

    def bundle(p, split):
        part = p[split]
        table = p["tables"][split]
        f, parts = task_features(
            model,
            p["state"],
            p["raw"],
            p["state_edges"],
            part["pairs"],
            table["ctx"],
            table["base"],
        )
        return (head(f).squeeze(1), parts)

    def score(p, split):
        return bundle(p, split)[0]

    def flat_gradient(output, grad_outputs=None, retain_graph=False):
        grads = torch.autograd.grad(
            output,
            parameters,
            grad_outputs=grad_outputs,
            retain_graph=retain_graph,
            allow_unused=True,
        )
        g = torch.cat(
            [
                torch.zeros_like(parameter) if grad is None else grad
                for parameter, grad in zip(parameters, grads)
            ]
        )
        g[k:] = 0.0
        return g

    def ggn(logits, probes=4):
        """Unbiased Rademacher estimate of the FINAL-LOGIT GGN, not relation Fisher."""
        multiclass = logits.ndim == 2
        p = logits.detach().softmax(1) if multiclass else logits.detach().sigmoid()
        scale = p.sqrt() if multiclass else (p * (1 - p) / len(p)).sqrt()
        curvature = logits.new_zeros(len(initial), len(initial))
        for probe in range(probes):
            signs = torch.randint(2, logits.shape, dtype=torch.int64).to(logits) * 2 - 1
            if multiclass:
                output_grad = (
                    scale
                    * (signs - scale * (scale * signs).sum(1, keepdim=True))
                    / len(p) ** 0.5
                )
            else:
                output_grad = scale * signs
            g = flat_gradient(logits, output_grad, retain_graph=True).detach()
            curvature += torch.outer(g, g) / probes
        return curvature

    def measure():
        result = {}
        with torch.no_grad():
            for name, p in packs.items():
                if nc_only:
                    n = nc_packs[name]
                    _, parts = task_features(
                        model,
                        n["state"],
                        n["raw"],
                        n["edges"],
                        n["pairs"],
                        n["ctx"],
                        n["base"],
                    )
                    metric = (
                        "auc"
                        if nc_metric == "auto" and len(n["y"].unique()) == 2
                        else "accuracy"
                    )
                    _, info = ridge(
                        torch.cat(parts, dim=1),
                        n["y"],
                        n["protocol"]["train"],
                        n["protocol"]["val"],
                        alphas=(0.1,),
                        metric=metric,
                    )
                    result[name] = dict(
                        score=info[
                            (
                                "validation_auc_percent"
                                if metric == "auc"
                                else "validation_percent"
                            )
                        ]
                        / 100,
                        metric=metric,
                    )
                else:
                    prediction = score(p, "lp_val").numpy()
                    y = p["lp_val"]["y"].numpy()
                    result[name] = dict(
                        score=float(roc_auc_score(y, prediction)),
                        ap=float(average_precision_score(y, prediction)),
                    )
        return result

    source_metric = torch.eye(len(initial)) * 0.1
    source_curvature_samples = []
    if source_states:
        saved = torch.load(
            Path(source_states) / f"{task}_states.pt", weights_only=False
        )
        prior = saved["cumulative"][0]
        if not torch.equal(torch.as_tensor(prior["theta"]).float(), initial):
            raise ValueError("Source state does not match checkpoint coefficients")
        source_metric = torch.as_tensor(prior["metric"]).float()
    for p in [] if source_states else source_packs(ckpt["source_graphs"], data_root):
        logits, parts = score_source(model, head, p, "calibration")
        if not nc_only:
            selected = torch.randperm(
                len(logits), generator=torch.Generator().manual_seed(42)
            )[:512]
            source_metric += ggn(logits[selected], probes=8)
            source_curvature_samples.append(
                dict(
                    dataset=p["name"],
                    count=len(selected),
                    positive=int(p["calibration"]["y"][selected].sum()),
                )
            )
        if nc_only:
            support, query = episode(
                p["node_y"], p["train_nodes"], torch.Generator().manual_seed(42)
            )
            nc, _ = nc_logits(parts, p["node_y"], support, query)
            source_metric += ggn(nc, probes=8)
    settings = dict(
        seed=42,
        task=task,
        sources=ckpt["source_graphs"],
        stream=list(stream),
        future=list(future),
        metric=(
            ("NC accuracy / binary AUC" if nc_metric == "auto" else "NC accuracy")
            if nc_only
            else "LP AUC"
        ),
        split="validation",
        updated_parameters=k,
        frozen_biases=len(initial) - k,
        metric_shape=[len(initial), len(initial)],
        source_ggn_probes=8,
        batch_ggn_probes=4,
        source_curvature_samples=source_curvature_samples,
        steps_per_domain=4,
        line_search=[0, 0.25, 1, 4],
        nc_ridge=0.1,
        source_anchor=source_anchor,
        controls=list(controls),
        source_states=source_states,
    )
    save_json(output / f"{task}_configuration.json", settings)
    trajectories = {}
    summary = {}
    online_states = {}
    for kind in controls:
        seed_all(42)
        model.inverse.set_theta(initial)
        ons = OnlineNewton(initial, lower, upper, eta=1.0, ridge=0.1)
        ons.metric = source_metric.double().numpy().copy()
        records = [dict(prefix=0, scores=measure(), update_seconds=0.0, curve=[])]
        online_states[kind] = [copy.deepcopy(ons.state_dict())]
        for prefix, name in enumerate(tqdm(stream, desc=f"{task} {kind}"), 1):
            begin = time.perf_counter()
            curve = []
            if kind != "static":
                if kind == "reset":
                    model.inverse.set_theta(initial)
                    ons = OnlineNewton(initial, lower, upper, eta=1.0, ridge=0.1)
                    ons.metric = source_metric.double().numpy().copy()
                elif kind == "reset-curvature":
                    ons.metric = source_metric.double().numpy().copy()
                p = packs[name]
                part = p["calibration"]
                order = torch.randperm(
                    len(part["y"]), generator=torch.Generator().manual_seed(42)
                )
                adapt = order[: min(1024, len(order) * 2 // 3)]
                held = order[len(adapt) : len(adapt) + 512]
                batches = adapt.split(256)
                for batch in range(4):
                    ix = batches[batch % len(batches)]
                    logits, parts = bundle(p, "calibration")
                    prediction = logits[ix]
                    lp_loss = F.binary_cross_entropy_with_logits(
                        prediction, part["y"][ix]
                    )
                    loss = lp_loss
                    curvature = ggn(prediction)
                    if nc_only:
                        support, query = episode(
                            p["node_y"],
                            p["nc_adapt"],
                            torch.Generator().manual_seed(42 + batch),
                        )
                        nc, truth = nc_logits(parts, p["node_y"], support, query)
                        nc_loss = F.cross_entropy(nc, truth)
                        # Preserve the random-probe sequence of the recorded
                        # development run, which drew an extra NC estimate.
                        ggn(nc)
                        loss = nc_loss
                        curvature = ggn(nc)
                    if source_anchor:
                        delta = model.inverse.theta() - initial
                        loss = (
                            loss + 0.5 * source_anchor * delta @ source_metric @ delta
                        )
                        curvature = curvature + source_anchor * source_metric
                    gradient = flat_gradient(loss).detach()
                    before = model.inverse.theta().detach().clone()
                    proposed = ons.step_gradient(gradient, curvature=curvature)
                    best = float("inf")
                    chosen = None
                    rate_chosen = None
                    for rate in [0.0, 0.25, 1.0, 4.0]:
                        candidate = before + rate * (proposed - before)
                        candidate = candidate.clamp(
                            min=candidate.new_tensor(lower),
                            max=candidate.new_tensor(upper),
                        )
                        candidate[k:] = initial[k:]
                        model.inverse.set_theta(candidate)
                        with torch.no_grad():
                            held_logits, held_parts = bundle(p, "calibration")
                            held_lp = float(
                                F.binary_cross_entropy_with_logits(
                                    held_logits[held], part["y"][held]
                                )
                            )
                            value = held_lp
                            if nc_only:
                                nc, truth = nc_logits(
                                    held_parts, p["node_y"], p["nc_adapt"], p["nc_held"]
                                )
                                held_nc = float(F.cross_entropy(nc, truth))
                                value = held_nc
                            if source_anchor:
                                delta = candidate - initial
                                value += float(
                                    0.5 * source_anchor * delta @ source_metric @ delta
                                )
                        if value < best:
                            best = value
                            chosen = candidate
                            rate_chosen = rate
                    model.inverse.set_theta(chosen)
                    ons.theta = chosen.double().numpy().copy()
                    curve.append(
                        dict(
                            step=batch + 1,
                            holdout_loss=best,
                            rate=rate_chosen,
                            coefficient_change=float((chosen - initial).norm()),
                        )
                    )
            records.append(
                dict(
                    prefix=prefix,
                    scores=measure(),
                    update_seconds=time.perf_counter() - begin,
                    curve=curve,
                )
            )
            online_states[kind].append(copy.deepcopy(ons.state_dict()))
            print(kind, prefix, json.dumps(records[-1]), flush=True)
        trajectories[kind] = records
        matrix = [
            [record["scores"][name]["score"] for name in stream] for record in records
        ]
        future_curves = {
            name: [record["scores"][name]["score"] for record in records]
            for name in future
        }
        summary[kind] = dict(
            **continual_metrics(matrix),
            matrix=matrix,
            future_curves=future_curves,
            future_gain={
                name: values[-1] - values[0] for name, values in future_curves.items()
            },
        )
        save_json(
            output / f"{task}_validation.json",
            dict(settings=settings, trajectories=trajectories, summary=summary),
        )
        torch.save(online_states, output / f"{task}_states.pt")
    return online_states


def run(
    checkpoint="runs/source/best.pt",
    data_root="data",
    output="runs/continual",
    stream=("cora", "citeseer"),
    future=("roman-empire",),
    task="both",
    threads=4,
    nc_metric="accuracy",
    source_anchor=0.0,
    controls=("static", "cumulative", "reset"),
    source_states=None,
):
    """Run static/cumulative/reset controls and save the final two-task bank."""
    if set(stream) & set(future):
        raise ValueError("Future domains must be excluded from the adaptation stream")
    if source_anchor < 0:
        raise ValueError("source_anchor must be nonnegative")
    tasks = ("nc", "lp") if task == "both" else (task,)
    states = {
        name: calibrate(
            name,
            checkpoint,
            data_root,
            output,
            stream,
            future,
            threads,
            nc_metric,
            source_anchor,
            controls,
            source_states,
        )
        for name in tasks
    }
    model, _, _ = load_checkpoint(checkpoint)
    initial = model.inverse.theta().detach().clone()
    lower, upper = model.inverse.bounds()
    bank = TaskCoefficientBank(initial, lower, upper, tasks=tasks)
    if "cumulative" in controls:
        for name in tasks:
            bank.load_task(name, states[name]["cumulative"][-1])
        torch.save(bank.state_dict(), Path(output) / "task_bank.pt")
