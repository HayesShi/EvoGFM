"""Full-model post hoc mechanism interventions, isolated from benchmark runs.

No recipient result selects a coefficient, power, step size or checkpoint.
All planned sweep points are retained. Import order follows run_study.
"""

from . import settings

import argparse
import gc
import json
import time
import types

from . import study as rs
import pandas as pd
from evogfm.features import node_representations
from evogfm.readout import episode, fit_ridge
from .io import atomic_json

np, torch = rs.np, rs.torch
HERE = settings.REPO
RUN = settings.NC_RUN
OUT = RUN / "internal_exploration"


def intervened_program(self, state):
    left, s, right, b = type(self).attention_program(self, state)
    mode = getattr(self, "diagnostic_mode", "identity")
    if mode == "power":
        exponent = self.diagnostic_power
        if exponent != 1.0:
            s = s.sign() * s.abs().pow(exponent)
            s = s / s.norm().clamp_min(1e-12)
    elif mode == "permute":
        # Same energy spectrum as the matching power control; reassign magnitudes.
        if self.diagnostic_power != 1.0:
            s = s.sign() * s.abs().pow(self.diagnostic_power)
            s = s / s.norm().clamp_min(1e-12)
        permutation = self.diagnostic_permutation
        s = s.sign() * s.abs()[permutation]
    elif mode == "rotation":
        rotation = self.diagnostic_rotation.to(left)
        left = left @ rotation
        right = right @ rotation
    return left, s, right, b


def event(stage, **values):
    data = dict(time=rs.now(), stage=stage, **values)
    atomic_json(OUT / "progress.json", data)
    print(json.dumps(data), flush=True)


def accuracy(h, fit, classes, y, ids):
    correct = 0
    class_correct = torch.zeros(len(classes), dtype=torch.long)
    class_total = torch.zeros_like(class_correct)
    for block in ids.split(1024):
        scores = (h[block].double() - fit.mean) @ fit.weight + fit.bias
        prediction = classes[scores.argmax(1)]
        truth = y[block]
        hit = prediction.eq(truth)
        correct += int(hit.sum())
        for j, c in enumerate(classes):
            mask = truth.eq(c)
            class_correct[j] += hit[mask].sum()
            class_total[j] += mask.sum()
    return dict(
        accuracy_percent=100 * correct / len(ids),
        macro_accuracy_percent=float(
            (class_correct[class_total > 0] / class_total[class_total > 0]).mean()
        )
        * 100,
        per_class_correct=class_correct.tolist(),
        per_class_total=class_total.tolist(),
    )


def internal_stats(model, pack, transition, transformer):
    left, s, right, _ = model.attention_program(pack["state"])
    amplitudes = s.double().numpy()
    energy = amplitudes**2
    energy /= energy.sum()
    w = transition.data.astype(np.float64)
    degree = np.diff(transition.indptr)
    targets = np.repeat(np.arange(len(degree)), degree)
    entropy = np.bincount(
        targets, weights=-w * np.log(np.maximum(w, 1e-30)), minlength=len(degree)
    )
    mask = degree > 1
    # These label summaries are post hoc measurements, never inputs to interventions.
    y = pack["y"].numpy()
    source = transition.indices
    label_match = (y[targets] == y[source]) & (y[targets] >= 0) & (y[source] >= 0)
    class_mass = np.bincount(targets, weights=w * label_match, minlength=len(degree))
    test = pack["protocol"]["test"].numpy()
    test = test[degree[test] > 0]
    z = transformer[pack["protocol"]["test"]].double()
    labels = pack["y"][pack["protocol"]["test"]]
    mean = z.mean(0)
    total = float(((z - mean) ** 2).sum())
    between = 0.0
    for c in labels.unique():
        part = z[labels == c]
        between += len(part) * float(((part.mean(0) - mean) ** 2).sum())
    return dict(
        spectrum=amplitudes.tolist(),
        energy=energy.tolist(),
        spectral_effective_rank=float(
            np.exp(-(energy * np.log(np.maximum(energy, 1e-30))).sum())
        ),
        attention_normalized_entropy=float(
            (entropy[mask] / np.log(degree[mask])).mean()
        ),
        attention_effective_neighbor_fraction=float(
            (np.exp(entropy[mask]) / degree[mask]).mean()
        ),
        test_same_label_attention_mass=float(class_mass[test].mean()),
        transformer_between_class_fraction=between / max(total, 1e-30),
        transformer_mean_norm=float(mean.norm()),
        transformer_total_variance=total / len(z),
    )


def make_views(model, pack):
    with torch.no_grad():
        parts, h, transition = node_representations(
            model,
            pack["state"],
            pack["raw"],
            pack["edges"],
            hops=2,
            inference_device=settings.DEVICE,
            return_transition=True,
        )
        stats = internal_stats(model, pack, transition, h)
        joined = torch.cat(tuple(parts) + (h,), dim=1)
        del parts, h, transition
    return joined, stats


def baseline_heads(h, pack, seeds, shots):
    heads = {}
    for seed in seeds:
        for shot in shots:
            support, _ = episode(
                pack["y"],
                pack["protocol"]["train"],
                torch.Generator().manual_seed(seed),
                shots=shot,
                queries=0,
            )
            fit = fit_ridge(
                h, pack["y"], support, pack["protocol"]["val"], alphas=(0.1,)
            )
            # Avoid retaining the large feature matrix in nine heads.
            fit.features = None
            classes = pack["y"][support].unique(sorted=True)
            heads[(seed, shot)] = (support, classes, fit)
    return heads


def evaluate(model, pack, heads, spec):
    begin = time.perf_counter()
    with torch.no_grad():
        h, stats = make_views(model, pack)
        rows = []
        for (seed, shot), (support, classes, initial_fit) in heads.items():
            # Path checkpoints depend on update seed; other interventions do not.
            if spec.get("update_seed") is not None and seed != spec["update_seed"]:
                continue
            fit = fit_ridge(
                h, pack["y"], support, pack["protocol"]["val"], alphas=(0.1,)
            )
            refit = accuracy(h, fit, classes, pack["y"], pack["protocol"]["test"])
            fixed = accuracy(
                h, initial_fit, classes, pack["y"], pack["protocol"]["test"]
            )
            rows.append(
                dict(
                    **spec,
                    graph=pack["name"],
                    seed=seed,
                    shots=shot,
                    support_count=len(support),
                    min_class_support=min(
                        int((pack["y"][support] == c).sum()) for c in classes
                    ),
                    test_count=len(pack["protocol"]["test"]),
                    **refit,
                    fixed_head_accuracy_percent=fixed["accuracy_percent"],
                    fixed_head_macro_accuracy_percent=fixed["macro_accuracy_percent"],
                )
            )
            del fit
        del h
    return dict(
        spec=spec,
        graph=pack["name"],
        internal=stats,
        rows=rows,
        elapsed_seconds=time.perf_counter() - begin,
    )


def collect():
    rows = []
    internals = []
    for path in sorted((OUT / "measurements").glob("*.json")):
        value = json.loads(path.read_text())
        rows.extend(value["rows"])
        internals.append(
            dict(graph=value["graph"], **value["spec"], **value["internal"])
        )
    pd.DataFrame(rows).to_csv(OUT / "intervention_scores.csv", index=False)
    pd.DataFrame(internals).to_json(
        OUT / "intervention_internals.json", orient="records", indent=2
    )
    return len(rows), len(internals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=["power", "path", "permutation", "channel", "collect"],
        default="power",
    )
    parser.add_argument("--graphs", nargs="+", default=rs.RECIPIENTS)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    (OUT / "measurements").mkdir(exist_ok=True)
    if args.phase == "collect":
        print(collect())
        return
    torch.set_num_threads(4)
    model, _, _ = rs.load_checkpoint(settings.CHECKPOINT)
    model.eval()
    model.attention_program = types.MethodType(intervened_program, model)
    initial = model.inverse.theta().detach().clone()
    seeds = [42, 43, 44]
    shots = [1, 5, 20]
    powers = np.linspace(0.5, 1.5, 17).tolist()
    # Definitions are saved before evaluating any intervention point.
    manifest = dict(
        phase=args.phase,
        graphs=args.graphs,
        seeds=seeds,
        shots=shots,
        powers=powers,
        path_fractions=np.linspace(0, 1, 11).tolist(),
        path_donors=["dblp", "wisconsin"],
        permutation_seeds=list(range(6)),
        model="Full EvoGFM checkpoint, unchanged feature and readout paths",
        channel_sources=["stream_random", "stream_spectral_grouped"],
        channel_components=["full", "max_only", "rest_only"],
        channel_definition="Replay saved endpoint. For split components, first align coefficient norm to the initial norm; retain either coefficient index 31 change or all remaining coefficient changes. Index 31 is the maximum-magnitude inverse basis.",
        source_pretraining_replicates=1,
        selection="All fixed grid points retained; no test-based selection",
        interpretation="Post hoc mechanism controls, not benchmark performance of an optimized variant",
        relation_control="Fixed observed directions and Frobenius norm for power intervention",
        recipient_labels="Ridge fitting and post hoc evaluation only; no shared-coefficient optimization",
        readout="Original fixed alpha 0.1; both refitted and initial frozen heads",
        log_labels="Same-label attention mass and class separation use evaluation labels for post hoc analysis only",
    )
    atomic_json(OUT / f"{args.phase}_manifest.json", manifest)
    for name in args.graphs:
        event("prepare", phase=args.phase, graph=name)
        pack = torch.load(RUN / "cache" / f"{name}.pt", weights_only=False, mmap=True)
        model.inverse.set_theta(initial)
        model.diagnostic_mode = "identity"
        h, stats = make_views(model, pack)
        heads = baseline_heads(h, pack, seeds, shots)
        # Record/reconcile baseline once, using the identical prediction protocol.
        baseline_path = OUT / "measurements" / f"{name}__baseline.json"
        if not baseline_path.exists():
            result = evaluate(model, pack, heads, dict(kind="baseline", value=1.0))
            expected = pd.read_csv(RUN / "probe_scores.csv")
            expected = expected[expected.kind.eq("initial") & expected.graph.eq(name)]
            for row in result["rows"]:
                old = float(
                    expected[
                        expected.seed.eq(row["seed"]) & expected.shots.eq(row["shots"])
                    ].accuracy_percent.iloc[0]
                )
                if abs(row["accuracy_percent"] - old) > 1e-7:
                    raise RuntimeError(
                        f'Baseline reproduction failed: {name}, {row["seed"]}, {row["shots"]}, {old}, {row["accuracy_percent"]}'
                    )
            atomic_json(baseline_path, result)
        del h
        specs = []
        if args.phase == "power":
            specs = [dict(kind="power", value=float(x)) for x in powers]
        elif args.phase == "path":
            specs = [
                dict(kind="path", donor=dn, update_seed=seed, value=float(x))
                for dn in ["dblp", "wisconsin"]
                for seed in seeds
                for x in np.linspace(0, 1, 11)
            ]
        elif args.phase == "permutation":
            specs = [
                dict(kind="permutation", permutation_seed=seed, value=float(power))
                for seed in range(6)
                for power in [0.75, 1.0, 1.25]
            ]
        elif args.phase == "channel":
            specs = [
                dict(
                    kind="channel",
                    source=source,
                    component=component,
                    update_seed=seed,
                    value=1.0,
                )
                for source in ["stream_random", "stream_spectral_grouped"]
                for seed in seeds
                for component in ["full", "max_only", "rest_only"]
            ]
        for index, spec in enumerate(specs):
            key = "_".join(f"{k}-{v}" for k, v in spec.items())
            target = OUT / "measurements" / f"{name}__{key}.json"
            if target.exists():
                continue
            model.inverse.set_theta(initial)
            if args.phase == "power":
                model.diagnostic_mode = "power"
                model.diagnostic_power = spec["value"]
            elif args.phase == "path":
                state = torch.load(
                    RUN
                    / "states"
                    / f's{spec["update_seed"]}_donor_{spec["donor"]}_01.pt',
                    weights_only=False,
                )
                model.inverse.set_theta(
                    initial
                    + spec["value"]
                    * (torch.as_tensor(state["theta"]).to(initial) - initial)
                )
                model.diagnostic_mode = "identity"
            elif args.phase == "channel":
                saved = torch.load(
                    RUN / "states" / f's{spec["update_seed"]}_{spec["source"]}_15.pt',
                    weights_only=False,
                )
                theta = torch.as_tensor(saved["theta"]).to(initial)
                if spec["component"] != "full":
                    aligned = theta[:32] * (initial[:32].norm() / theta[:32].norm())
                    delta = aligned - initial[:32]
                    value = initial.clone()
                    if spec["component"] == "max_only":
                        value[31] += delta[31]
                    else:
                        value[:31] += delta[:31]
                    theta = value
                model.inverse.set_theta(theta)
                model.diagnostic_mode = "identity"
            else:
                # Reassign recovered amplitude magnitudes across fixed directions.
                # This preserves the singular-energy spectrum of the matching power.
                model.diagnostic_mode = "permute"
                model.diagnostic_power = spec["value"]
                model.diagnostic_permutation = torch.randperm(
                    model.rank,
                    generator=torch.Generator().manual_seed(spec["permutation_seed"]),
                )
            event(
                "evaluate",
                phase=args.phase,
                graph=name,
                index=index + 1,
                total=len(specs),
                spec=spec,
            )
            result = evaluate(model, pack, heads, spec)
            atomic_json(target, result)
        collect()
        del pack, heads
        gc.collect()
        torch.cuda.empty_cache()
        event("graph_complete", phase=args.phase, graph=name)
    scores, states = collect()
    event("complete", phase=args.phase, scores=scores, states=states)


if __name__ == "__main__":
    main()
