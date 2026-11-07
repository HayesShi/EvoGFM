"""Export every V10 accumulation measurement and the paired final-state tables."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import torch

from evogfm.data import save_json


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def final_rows(result):
    groups = defaultdict(list)
    for row in result["rows"]:
        groups[row["dataset"], row["experiment"], row["shots"]].append(row)
    summaries = []
    for (dataset, experiment, shots), rows in sorted(groups.items()):
        prefix = max(row["prefix"] for row in rows)
        final = {row["control"]: row for row in rows if row["prefix"] == prefix}
        initial = [row["score_percent"] for row in rows if row["prefix"] == 0]
        if not initial or max(initial) != min(initial):
            raise ValueError("Controls must start from identical scores")
        if not {"static", "reset", "cumulative"} <= final.keys():
            raise ValueError("Incomplete controls")
        if final["static"]["score_percent"] != initial[0]:
            raise ValueError("Static state changed")
        cumulative = final["cumulative"]
        summaries.append(
            dict(
                dataset=dataset,
                experiment=experiment,
                shots=shots,
                train_labels=cumulative["train_labels"],
                metric=cumulative["metric"],
                final_prefix=prefix,
                initial=initial[0],
                reset=final["reset"]["score_percent"],
                cumulative=cumulative["score_percent"],
                gain=cumulative["score_percent"] - initial[0],
                gain_over_reset=cumulative["score_percent"]
                - final["reset"]["score_percent"],
            )
        )
    return summaries


def collect(args):
    root = Path(args.runs)
    selection = read(root / "selection.json")
    pools = {}
    for name in (
        "original_pool",
        "nc1_pool",
        "revised_pool",
        "assortative_pool",
        "test",
    ):
        directory = root / name
        configuration = read(directory / "configuration.json")
        result = read(directory / "results.json")
        experiments = {}
        coefficients = {}
        for state in configuration["states"]:
            stream = read(Path(state) / "nc_configuration.json")
            experiments[Path(state).name] = stream
            snapshots = torch.load(Path(state) / "nc_states.pt", weights_only=False)
            coefficients[Path(state).name] = {
                control: [torch.as_tensor(s["theta"]).tolist() for s in states]
                for control, states in snapshots.items()
            }
            expected = (
                set(configuration["datasets"])
                - set(stream["sources"])
                - set(stream["stream"])
            )
            if set(stream["sources"] + stream["stream"]) & {"tolokers", "tolokers-2"}:
                expected -= {"tolokers", "tolokers-2"}
            actual = {
                row["dataset"]
                for row in result["rows"]
                if row["experiment"] == Path(state).name
            }
            if actual != expected:
                raise ValueError(f"Incomplete future pool: {name}/{state}")
            prefixes = (
                (0, len(stream["stream"]))
                if configuration.get("final_only", False)
                else range(len(stream["stream"]) + 1)
            )
            cells = {
                (dataset, control, prefix, shots)
                for dataset in expected
                for control in stream["controls"]
                for prefix in prefixes
                for shots in configuration["budgets"]
            }
            actual_cells = {
                (row["dataset"], row["control"], row["prefix"], row["shots"])
                for row in result["rows"]
                if row["experiment"] == Path(state).name
            }
            if cells != actual_cells:
                raise ValueError(f"Incomplete trajectory: {name}/{state}")
        pools[name] = dict(
            configuration=configuration,
            experiments=experiments,
            coefficient_snapshots=coefficients,
            **result,
            final=final_rows(result),
        )
    retention = {
        order: read(Path("runs/round345") / order / "retention/retention.json")
        for order in ("forward", "reverse4")
    }
    checkpoints = {}
    for pool in pools.values():
        path = pool["configuration"]["checkpoint"]
        if path not in checkpoints:
            saved = torch.load(path, map_location="cpu", weights_only=False)
            checkpoints[path] = {
                key: saved[key]
                for key in ("epoch", "source_graphs", "model_config", "settings")
            }
    save_json(
        Path(args.output),
        dict(
            seed=42,
            selection=selection,
            checkpoints=checkpoints,
            pools=pools,
            retention=retention,
        ),
    )
    lines = [
        "# 知识积累：全部最终前缀结果",
        "",
        "seed 42；数值为百分数，差值为百分点。标签预算为每类标签数，all 使用全部官方训练标签。",
        "初始、reset、累积使用同一目标支持集和 alpha=0.1。reset 每次从源状态出发，只保留当前一张学习图的适应。",
        "未来图不更新共享参数。测试配置在测试前固定，见 results_accumulation.json 的 selection；其余候选仅列验证结果。",
        "20-shot 与 all 在部分图上使用完全相同的训练节点，不能算作两份独立证据。Tolokers 两版本来自同一物理图。",
        "",
    ]
    for name, data in pools.items():
        lines.extend([f"## {name}（{data['split']}）", ""])
        for experiment, config in data["experiments"].items():
            lines.extend(
                [
                    f"### {experiment}",
                    "",
                    "学习流：" + " → ".join(config["stream"]) + "。",
                    "",
                    "| 数据集 | 标签/类 | 标签总数 | 指标 | 初始 | reset | 累积 | Δ初始 | Δreset |",
                    "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
                ]
            )
            for row in data["final"]:
                if row["experiment"] != experiment:
                    continue
                values = [
                    row["dataset"],
                    row["shots"] or "all",
                    row["train_labels"],
                    row["metric"],
                    f"{row['initial']:.3f}",
                    f"{row['reset']:.3f}",
                    f"{row['cumulative']:.3f}",
                    f"{row['gain']:+.3f}",
                    f"{row['gain_over_reset']:+.3f}",
                ]
                lines.append("| " + " | ".join(map(str, values)) + " |")
            lines.append("")
    Path(args.tables).write_text("\n".join(lines).rstrip() + "\n", encoding="utf8")
    print("Saved", args.output, "and", args.tables)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="runs/accumulation_v10")
    parser.add_argument("--output", default="results_accumulation.json")
    parser.add_argument("--tables", default="ACCUMULATION_TABLES.md")
    collect(parser.parse_args())
