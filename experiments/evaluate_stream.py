"""Full recipient NC/LP and fixed-head historical evaluation of three streams."""

from . import settings

import argparse
import gc
import json
import time
from . import cross_task as cs
from . import interventions as ex
from evogfm.features import node_representations, task_features
from evogfm.readout import fit_residual, predict_residual
from evogfm.link_inference import LinkScorer
from sklearn.metrics import roc_auc_score, average_precision_score
from .io import atomic_json
import pandas as pd

torch, np = cs.torch, cs.np
LP_PREFIXES = list(range(16))


def state_files():
    return sorted((cs.ROOT / "states").glob("s*_*.pt"))


def nc_scores(joined, p, seed, arm, prefix):
    heads = ex.baseline_heads(joined, p, [seed], [1, 5, 20])
    rows = []
    for (_, shot), (support, classes, fit) in heads.items():
        score = ex.accuracy(joined, fit, classes, p["y"], p["protocol"]["test"])
        rows.append(
            dict(
                graph=p["name"],
                seed=seed,
                arm=arm,
                prefix=prefix,
                shots=shot,
                support_count=len(support),
                test_count=len(p["protocol"]["test"]),
                **score,
            )
        )
    del heads
    return rows


def lp_scores(features, head, p, tables, arm, seed, prefix):
    nf, nv, npos = tables["nfit"], tables["nval"], tables["npos"]
    with torch.no_grad():
        offset = head(features).squeeze(1)
    tf = features[:nf]
    to = offset[:nf]
    vf = features[nf : nf + nv]
    vo = offset[nf : nf + nv]
    ef = features[nf + nv :]
    eo = offset[nf + nv :]
    fits = []
    for ridge in [0.1, 1.0, 10.0, 100.0]:
        fit = fit_residual(tf, to, p["native"]["calibration"]["y"], ridge)
        val = predict_residual(vf, vo, fit)
        loss = float(
            cs.rs.F.binary_cross_entropy_with_logits(
                val, p["native"]["lp_val"]["y"].double()
            )
        )
        fits.append((loss, ridge, fit))
    loss, ridge, fit = min(fits, key=lambda f: f[0])
    initial_path = cs.ROOT / "heads" / f'lp_initial_{p["name"]}.pt'
    if prefix == 0:
        cs.rs.save_pt(initial_path, fit)
    initial_fit = torch.load(initial_path, weights_only=False)
    scores = {
        "source_frozen": eo,
        "target_initial_frozen": predict_residual(ef, eo, initial_fit),
        "target_refitted": predict_residual(ef, eo, fit),
    }
    rows = []
    for readout, score in scores.items():
        for i, difficulty in enumerate(["uniform", "hard"]):
            pos = score[:npos].numpy()
            neg = score[(i + 1) * npos : (i + 2) * npos].numpy()
            y = np.r_[np.ones(npos), np.zeros(npos)]
            s = np.r_[pos, neg]
            rows.append(
                dict(
                    graph=p["name"],
                    seed=seed,
                    arm=arm,
                    prefix=prefix,
                    readout=readout,
                    difficulty=difficulty,
                    selected_ridge=ridge,
                    validation_bce=loss,
                    positive_test_edges=npos,
                    auc_percent=100 * roc_auc_score(y, s),
                    ap_percent=100 * average_precision_score(y, s),
                )
            )
    return rows


def future_once(model, head):
    available = state_files()
    progress = 0
    for name in cs.TARGETS:
        pack_path = cs.ROOT / "cache" / f"{name}.pt"
        table_path = cs.ROOT / "tables_v2" / f"eval_{name}.pt"
        if not pack_path.exists() or not table_path.exists():
            continue
        candidates = [p for p in available if int(p.stem.rsplit("_", 1)[1]) > 0]
        baseline_path = cs.ROOT / "measurements" / f"{name}__initial.json"
        if not baseline_path.exists():
            candidates = [None] + candidates

        def pending(file):
            if file is None:
                return True
            target = cs.ROOT / "measurements" / f"{name}__{file.stem}.json"
            if not target.exists():
                return True
            return not json.loads(target.read_text())["lp"]

        candidates = [p for p in candidates if pending(p)]
        if not candidates:
            continue
        p = torch.load(pack_path, weights_only=False, mmap=True)
        tables = None
        for file in candidates:
            if file is None:
                arm, seed, prefix = "initial", 42, 0
                initial = torch.load(cs.settings.CHECKPOINT, weights_only=False)[
                    "model"
                ]
                theta = torch.cat([initial["inverse.c"], initial["inverse.beta"]])
                target = baseline_path
            else:
                state = torch.load(file, weights_only=False)
                arm, seed, prefix = state["kind"], state["seed"], state["prefix"]
                theta = torch.as_tensor(state["theta"]).float()
                target = cs.ROOT / "measurements" / f"{name}__{file.stem}.json"
            model.inverse.set_theta(theta)
            cs.event("eval_future", graph=name, arm=arm, seed=seed, prefix=prefix)
            begin = time.perf_counter()
            with torch.no_grad():
                parts, h, transition = node_representations(
                    model,
                    p["state"],
                    p["raw"],
                    p["edges"],
                    hops=2,
                    inference_device=settings.DEVICE,
                    return_transition=True,
                )
                joined = torch.cat(tuple(parts) + (h,), dim=1)
                nc = []
                for support_seed in cs.SEEDS if prefix == 0 else [seed]:
                    nc.extend(nc_scores(joined, p, support_seed, arm, prefix))
                del joined
                lp = []
                if prefix in LP_PREFIXES:
                    if tables is None:
                        tables = torch.load(table_path, weights_only=False, mmap=True)
                    weight = torch.from_numpy(transition.data)
                    feats = []
                    for batch in tables["batches"]:
                        f, _ = task_features(
                            model,
                            p["state"],
                            p["raw"],
                            p["edges"],
                            batch["pairs"],
                            batch["ctx"],
                            batch["base"],
                            prepared=(weight, tuple(parts) + (h,)),
                        )
                        feats.append(f)
                    features = torch.cat(feats)
                    if not (cs.ROOT / "inference_check.json").exists():
                        scorer = LinkScorer(
                            model,
                            head,
                            p["raw"],
                            p["x"],
                            p["edges"],
                            observation_rate=0.6,
                        )
                        expected = scorer.features(tables["batches"][0]["pairs"])
                        error = float(
                            (expected - features[: len(expected)]).abs().max()
                        )
                        assert error < 2e-4, error
                        atomic_json(
                            cs.ROOT / "inference_check.json",
                            dict(
                                graph=name,
                                maximum_feature_error=error,
                                channels=features.shape[1],
                                full_reference="LinkScorer.features",
                            ),
                        )
                        del scorer
                    lp = lp_scores(features, head, p, tables, arm, seed, prefix)
                    del features, feats
                del parts, h, transition
            atomic_json(
                target, dict(nc=nc, lp=lp, elapsed_seconds=time.perf_counter() - begin)
            )
            progress += 1
            gc.collect()
            torch.cuda.empty_cache()
        del p, tables
        gc.collect()
    return progress


def history_once(model):
    available = state_files()
    progress = 0
    for seed in cs.SEEDS:
        order = json.loads((cs.ORIGINAL / f"orders_s{seed}.json").read_text())["random"]
        for arm in cs.ARMS:
            anchor = cs.ROOT / "states" / f"s{seed}_{arm}_04.pt"
            if not anchor.exists():
                continue
            candidates = [
                f
                for f in available
                if f.stem.startswith(f"s{seed}_{arm}_")
                and int(f.stem.rsplit("_", 1)[1]) >= 4
            ]
            for graph in order[:4]:
                todo = [
                    f
                    for f in candidates
                    if not (cs.ROOT / "history" / f"{graph}__{f.stem}.json").exists()
                ]
                if not todo:
                    continue
                p = torch.load(
                    cs.ROOT / "cache" / f"{graph}.pt", weights_only=False, mmap=True
                )
                hp = cs.ROOT / "heads" / f"history_{graph}_s{seed}_{arm}.pt"
                if hp.exists():
                    fixed = torch.load(hp, weights_only=False)
                else:
                    state = torch.load(anchor, weights_only=False)
                    model.inverse.set_theta(torch.as_tensor(state["theta"]).float())
                    with torch.no_grad():
                        parts, h = node_representations(
                            model,
                            p["state"],
                            p["raw"],
                            p["edges"],
                            hops=2,
                            inference_device=settings.DEVICE,
                        )
                        joined = torch.cat(tuple(parts) + (h,), dim=1)
                        support, classes, fit = ex.baseline_heads(
                            joined, p, [seed], [5]
                        )[seed, 5]
                        fixed = dict(support=support, classes=classes, fit=fit)
                        cs.rs.save_pt(hp, fixed)
                        del joined, parts, h
                for file in todo:
                    state = torch.load(file, weights_only=False)
                    model.inverse.set_theta(torch.as_tensor(state["theta"]).float())
                    cs.event(
                        "eval_history",
                        graph=graph,
                        arm=arm,
                        seed=seed,
                        prefix=state["prefix"],
                    )
                    with torch.no_grad():
                        parts, h = node_representations(
                            model,
                            p["state"],
                            p["raw"],
                            p["edges"],
                            hops=2,
                            inference_device=settings.DEVICE,
                        )
                        joined = torch.cat(tuple(parts) + (h,), dim=1)
                        value = ex.accuracy(
                            joined,
                            fixed["fit"],
                            fixed["classes"],
                            p["y"],
                            p["protocol"]["test"],
                        )
                        del joined, parts, h
                    atomic_json(
                        cs.ROOT / "history" / f"{graph}__{file.stem}.json",
                        dict(
                            graph=graph,
                            seed=seed,
                            arm=arm,
                            prefix=state["prefix"],
                            shots=5,
                            **value,
                        ),
                    )
                    progress += 1
                    gc.collect()
                    torch.cuda.empty_cache()
                del p, fixed
                gc.collect()
    return progress


def collect():
    nc = []
    lp = []
    for file in (cs.ROOT / "measurements").glob("*.json"):
        obj = json.loads(file.read_text())
        nc.extend(obj["nc"])
        lp.extend(obj["lp"])
    history = [json.loads(f.read_text()) for f in (cs.ROOT / "history").glob("*.json")]
    for name, rows in [("nc", nc), ("lp", lp), ("history", history)]:
        if rows:
            pd.DataFrame(rows).to_csv(cs.ROOT / f"{name}_scores.csv", index=False)
    return dict(nc_rows=len(nc), lp_rows=len(lp), history_rows=len(history))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    cs.setup()
    model, head, _ = cs.rs.load_checkpoint(cs.settings.CHECKPOINT)
    while True:
        progress = future_once(model, head) + history_once(model)
        counts = collect()
        cs.event("eval_snapshot", **counts)
        expected_nc = 4 * (9 + 9 * 15 * 3)
        expected_lp = 4 * (6 + 9 * 15 * 6)
        if counts == dict(
            nc_rows=expected_nc, lp_rows=expected_lp, history_rows=9 * 4 * 12
        ):
            cs.event("eval_complete", **counts)
            break
        if not args.watch:
            break
        if not progress:
            time.sleep(30)


if __name__ == "__main__":
    main()
