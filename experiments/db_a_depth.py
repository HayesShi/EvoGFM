"""Train and freeze the full two- and three-layer predictors on source graphs."""

import argparse
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from evogfm.data import seed_all
from evogfm.readout import new_head, episode
from evogfm.training import objective, score_source, source_packs
from experiments.db_a_execution_suite import write_json
from experiments.db_a_native import DEFAULT_CHECKPOINT
from experiments.full_graph import FullGraphGFM


def to_cuda(value):
    if torch.is_tensor(value):
        return value.cuda()
    if isinstance(value, dict):
        return {name: to_cuda(child) for name, child in value.items()}
    return value


def episodic_loss(features, y, support, query):
    """Original source ridge episode, preserving the device of its operands."""
    classes = y[support].unique(sorted=True)
    a, b = features[support], features[query]
    t = (y[support, None] == classes[None, :]).float()
    mean, bias = a.mean(0), t.mean(0)
    a, b = a - mean, b - mean
    regularization = torch.eye(len(a), device=a.device) * 0.1
    weights = torch.linalg.solve(a @ a.T + regularization, t - bias)
    truth = (y[query, None] == classes[None, :]).long().argmax(1)
    return F.cross_entropy(8 * (b @ a.T @ weights + bias), truth)


def train(depth, output):

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    seed_all(42)
    released = torch.load(DEFAULT_CHECKPOINT, map_location="cpu", weights_only=False)
    settings = released["settings"]
    epochs, sources = settings["epochs"], settings["sources"]
    config = dict(released["model_config"], layers=depth)
    packs = source_packs(sources)
    device_packs = [to_cuda(pack) for pack in packs]
    model = FullGraphGFM(**config).cuda()
    head = new_head(True).cuda()
    trainable = list(model.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=settings["lr"], weight_decay=0.0001)
    rng = torch.Generator().manual_seed(42)
    curve = []
    best, checkpoint = float("inf"), output / "best.pt"
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        head.train()
        losses = []
        for step in range(4):
            index = (epoch * 4 + step) % len(packs)
            pack, dpack = packs[index], device_packs[index]
            support, query = episode(pack["node_y"], pack["train_nodes"], rng)
            scores, parts = score_source(model, head, dpack, "calibration")
            loss = objective(scores, dpack["calibration"]["y"]) + 0.2 * episodic_loss(
                torch.cat(parts, dim=1), dpack["node_y"], support.cuda(), query.cuda()
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 2.0)
            optimizer.step()
            model.inverse.constrain()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        head.eval()
        values = []
        with torch.no_grad():
            for pack in device_packs:
                scores, parts = score_source(model, head, pack, "lp_val")
                loss = objective(scores, pack["lp_val"]["y"]) + 0.2 * episodic_loss(
                    torch.cat(parts, dim=1),
                    pack["node_y"],
                    pack["support"],
                    pack["query"],
                )
                values.append(float(loss.detach().cpu()))
        value = sum(values) / len(values)
        if value < best:
            best = value
            temporary = output / "best.tmp"
            torch.save(
                dict(
                    model=model.state_dict(),
                    head=head.state_dict(),
                    model_config=model.config,
                    epoch=epoch + 1,
                    source_graphs=sources,
                    validation_loss=best,
                    settings=dict(
                        settings,
                        model=config,
                        estimator="FullGraphGFM full-edge moments",
                        device="CUDA",
                    ),
                ),
                temporary,
            )
            temporary.replace(checkpoint)
        curve.append(
            dict(
                epoch=epoch + 1,
                train_loss=sum(losses) / len(losses),
                validation_loss=value,
            )
        )
        write_json(
            output / "status.json",
            dict(
                phase="training",
                depth=depth,
                completed_epochs=epoch + 1,
                total_epochs=epochs,
                best_validation_loss=best,
                elapsed_s=time.perf_counter() - started,
                peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            ),
        )
    write_json(
        output / "result.json",
        dict(
            depth=depth,
            checkpoint=str(checkpoint),
            seed=42,
            epochs=epochs,
            source_graphs=sources,
            history=curve,
            validation_loss=best,
            training_s=time.perf_counter() - started,
            device=torch.cuda.get_device_name(),
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, choices=(2, 3), required=True)
    parser.add_argument(
        "--output", type=lambda value: Path(value).resolve(), required=True
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    train(args.depth, args.output)


if __name__ == "__main__":
    main()
