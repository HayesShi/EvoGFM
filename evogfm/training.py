"""Source-only joint training; checkpoints are selected on source validation."""

import time
from pathlib import Path
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from .data import load_graph, make_pack, seed_all, save_json
from .features import preprocess, contexts, task_features, pair_channels
from .model import InverseLawGFM
from .readout import new_head, node_masks, episode, episode_loss


def source_packs(names, data_root="data"):
    result = []
    directory = Path(data_root) / "prepared"
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        cache = directory / (name + "_joint_source.pt")
        if cache.exists():
            p = torch.load(cache, weights_only=False)
        else:
            data = load_graph(name, data_root)
            raw, x = preprocess(data)
            p = make_pack(name, x, data["edges"], data["y"], seed=42, query_cap=4096)
            p["raw"] = raw
            p["train_nodes"], p["validation_nodes"] = node_masks(p["node_y"])
            torch.save(p, cache)
        p["tables"] = {}
        for split in ["calibration", "lp_val"]:
            part = p[split]
            p["tables"][split] = dict(
                ctx=contexts(p["state_edges"], len(p["x"]), part["pairs"]),
                base=pair_channels(p["raw"], p["state_edges"], part["pairs"]),
            )
        generator = torch.Generator().manual_seed(42)
        p["support"], _ = episode(p["node_y"], p["train_nodes"], generator)
        p["query"] = p["validation_nodes"][:1024]
        result.append(p)
        print("source ready", name, len(p["x"]), flush=True)
    return result


def score_source(model, head, p, split):
    state = model.state(p["x"], p["state_edges"], 0.6)
    part = p[split]
    table = p["tables"][split]
    f, rep = task_features(
        model,
        state,
        p["raw"],
        p["state_edges"],
        part["pairs"],
        table["ctx"],
        table["base"],
    )
    return head(f).squeeze(1), rep


def objective(score, y):
    bce = F.binary_cross_entropy_with_logits(score, y)
    pos = score[y > 0]
    neg = score[y == 0]
    # Hard-tail pressure is source-only, using a predeclared top-10% cutoff.
    hard = neg.topk(max(1, len(neg) // 10)).values
    ranking = F.softplus(hard[None, :] - pos[:, None]).mean()
    return bce + 0.2 * ranking


def train(
    epochs=60,
    data_root="data",
    output="runs/source",
    sources=("pubmed", "photo", "texas"),
    threads=4,
):
    seed_all(42)
    torch.set_num_threads(threads)
    packs = source_packs(sources, data_root)
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    config = dict(
        input_dim=100,
        d=100,
        hidden=128,
        layers=2,
        heads=4,
        rank=16,
        k=32,
        dropout=0.1,
        inverse_basis="conditional_hat",
        frame_aligned=True,
        input_encoder="identity",
        whiten_floor=0.05,
        attention_normalization="frobenius",
        relation_gate_init=0.0,
    )
    model = InverseLawGFM(**config)
    model.config["joint_task_backbone"] = True
    head = new_head(True)
    active = list(model.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(active, lr=0.003, weight_decay=0.0001)
    generator = torch.Generator().manual_seed(42)
    settings = dict(
        seed=42,
        sources=list(sources),
        model=config,
        epochs=epochs,
        total_updates=epochs * 4,
        lr=0.003,
        objectives="source BCE + .2 hard-tail ranking + .2 NC episode",
        values="full original features; inverse affinities in SVD100 graph frame",
        task_path="two inverse-programmed local propagations; separate zero-hop; inverse-weighted common neighbors",
        backbone_v_ffn_optimized=True,
        checkpoint_selection="source validation objective only",
        targets_excluded=["cora", "citeseer", "roman-empire"],
        public_test_evaluated=False,
    )
    save_json(directory / "configuration.json", settings)
    best = float("inf")
    history = []
    begin = time.perf_counter()
    progress = tqdm(range(epochs), desc="source training", mininterval=3)
    for epoch in progress:
        model.train()
        head.train()
        losses = []
        for step in range(4):
            p = packs[(epoch * 4 + step) % len(packs)]
            score, rep = score_source(model, head, p, "calibration")
            support, query = episode(p["node_y"], p["train_nodes"], generator)
            loss = objective(score, p["calibration"]["y"]) + 0.2 * episode_loss(
                torch.cat(rep, dim=1), p["node_y"], support, query
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(active, 2.0)
            opt.step()
            model.inverse.constrain()
            losses.append(float(loss.detach()))
        model.eval()
        head.eval()
        values = []
        details = []
        with torch.no_grad():
            for p in packs:
                score, rep = score_source(model, head, p, "lp_val")
                value = objective(score, p["lp_val"]["y"]) + 0.2 * episode_loss(
                    torch.cat(rep, dim=1), p["node_y"], p["support"], p["query"]
                )
                values.append(float(value))
                y = p["lp_val"]["y"]
                details.append(
                    dict(
                        dataset=p["name"],
                        loss=float(value),
                        auc=float(roc_auc_score(y, score)),
                    )
                )
        value = sum(values) / len(values)
        history.append(
            dict(
                epoch=epoch + 1,
                train_loss=sum(losses) / 4,
                validation_loss=value,
                details=details,
            )
        )
        if value < best:
            best = value
            torch.save(
                dict(
                    model=model.state_dict(),
                    head=head.state_dict(),
                    model_config=model.config,
                    epoch=epoch + 1,
                    source_graphs=list(sources),
                    validation_loss=best,
                    settings=settings,
                ),
                directory / "best.pt",
            )
        save_json(directory / "history.json", history)
        save_json(
            directory / "progress.json",
            dict(
                status="running",
                epoch=epoch + 1,
                total_epochs=epochs,
                best_validation_loss=best,
                seconds=time.perf_counter() - begin,
            ),
        )
        progress.set_postfix(val=f"{value:.3f}", best=f"{best:.3f}", refresh=False)
    save_json(
        directory / "progress.json",
        dict(
            status="complete",
            epoch=epochs,
            total_epochs=epochs,
            best_validation_loss=best,
            seconds=time.perf_counter() - begin,
        ),
    )
    print("Saved", directory / "best.pt", flush=True)


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = dict(checkpoint["model_config"])
    backbone = config.pop("joint_task_backbone", False)
    model = InverseLawGFM(**config)
    model.config["joint_task_backbone"] = backbone
    model.load_state_dict(checkpoint["model"])
    model.eval()
    head = new_head(backbone)
    head.load_state_dict(checkpoint["head"])
    head.eval()
    return model, head, checkpoint
