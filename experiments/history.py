"""Fixed common-prefix heads: later graphs cannot refit historical readouts."""

import gc
import json
import time
import torch
from evogfm.readout import episode, fit_ridge


def evaluate_history(study):
    from .study import event, save_pt, write_json

    folder = study.out / "history_measurements"
    folder.mkdir(exist_ok=True)
    states = [
        (p.stem, torch.load(p, weights_only=False))
        for p in sorted((study.out / "states").glob("s*_stream_*.pt"))
    ]
    anchors = [(sid, s) for sid, s in states if s["prefix"] == 4]
    for anchor_id, anchor in anchors:
        followups = [
            (sid, s)
            for sid, s in states
            if s["seed"] == anchor["seed"]
            and s["kind"] == anchor["kind"]
            and s["prefix"] >= 4
        ]
        for name in anchor["order"][:4]:
            p = study.pack(name)
            head_path = study.out / "heads" / f"{anchor_id}__{name}.pt"
            if head_path.exists():
                head = torch.load(head_path, weights_only=False)
            else:
                study.model.inverse.set_theta(torch.as_tensor(anchor["theta"]).float())
                event(
                    study.out, "fit_fixed_history_head", state_id=anchor_id, graph=name
                )
                with torch.no_grad():
                    parts = study.parts(p, inference=True)
                    h = torch.cat(parts, dim=1)
                    del parts
                    support, _ = episode(
                        p["y"],
                        p["protocol"]["train"],
                        torch.Generator().manual_seed(anchor["seed"]),
                        shots=5,
                        queries=0,
                    )
                    fit = fit_ridge(
                        h, p["y"], support, p["protocol"]["val"], alphas=(0.1,)
                    )
                    head = dict(
                        mean=fit.mean,
                        weight=fit.weight,
                        bias=fit.bias,
                        classes=p["y"][support].unique(sorted=True),
                        support=support,
                        anchor_state=anchor_id,
                    )
                    save_pt(head_path, head)
                    del fit, h
            for state_id, state in followups:
                path = folder / f"{state_id}__{name}.json"
                if path.exists():
                    continue
                event(study.out, "fixed_head_history", state_id=state_id, graph=name)
                begin = time.perf_counter()
                study.model.inverse.set_theta(torch.as_tensor(state["theta"]).float())
                with torch.no_grad():
                    parts = study.parts(p, inference=True)
                    h = torch.cat(parts, dim=1)
                    del parts
                    correct, total = 0, 0
                    for ids in p["protocol"]["test"].split(1024):
                        logits = (h[ids].double() - head["mean"]) @ head[
                            "weight"
                        ] + head["bias"]
                        correct += int(
                            (head["classes"][logits.argmax(1)] == p["y"][ids]).sum()
                        )
                        total += len(ids)
                    del h
                write_json(
                    path,
                    dict(
                        state_id=state_id,
                        seed=state["seed"],
                        kind=state["kind"],
                        prefix=state["prefix"],
                        graphs_after_common_prefix=state["prefix"] - 4,
                        graph=name,
                        anchor_state=anchor_id,
                        head_policy="fixed_at_common_prefix",
                        shots=5,
                        support_count=len(head["support"]),
                        test_count=total,
                        accuracy_percent=100 * correct / total,
                        evaluation_seconds=time.perf_counter() - begin,
                    ),
                )
                gc.collect()
            del p, head
            gc.collect()
    import csv

    rows = [
        json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("*.json"))
    ]
    if rows:
        with (study.out / "history_scores.csv").open(
            "w", newline="", encoding="utf-8"
        ) as h:
            writer = csv.DictWriter(h, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
