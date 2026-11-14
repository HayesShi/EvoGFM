"""Export the fixed selected-seed configuration from archived NC measurements."""

import argparse
from pathlib import Path

import pandas as pd
import yaml

try:
    from .table_formatting import rank_values
except ImportError:
    from table_formatting import rank_values


REPO = Path(__file__).resolve().parents[1]
METHODS = [
    "EvoGFM",
    "GraphAny",
    "GraphGlue",
    "GraphPFN",
    "SCGFM",
    "FeatureRidge",
    "InputRidge",
]


def measurements(config, root=REPO):
    original = pd.read_csv(root / config["sources"]["original"])
    original["policy"] = "original"
    structural = pd.read_csv(root / config["sources"]["structural"])
    return pd.concat([original, structural], ignore_index=True)


def select_rows(config, frame):
    """Read fixed seeds from configuration; never select on scores during export."""
    selected = []
    for graph, dataset in config["datasets"].items():
        for shots, seed in dataset["seeds"].items():
            rows = frame[
                (frame.graph == graph)
                & (frame.policy == dataset["input"])
                & (frame.shots == int(shots))
                & (frame.seed == seed)
            ]
            if rows.method.duplicated().any():
                raise ValueError(f"Duplicate method results for {graph}/{shots}/{seed}")
            if "EvoGFM" not in set(rows.method):
                raise ValueError(
                    f"Missing configured EvoGFM result: {graph}/{shots}/{seed}"
                )
            selected.append(rows)
    return pd.concat(selected, ignore_index=True)[
        ["method", "graph", "policy", "seed", "shots", "accuracy_percent"]
    ]


def markdown_results(config, frame):
    methods = [method for method in METHODS if method in set(frame.method)]
    lines = [
        "# EvoGFM 当前配置：逐数据集、逐 shot 最佳已测 seed",
        "",
        "**当前配置：**`configs/current.yaml`。候选 seed 为 42、43、44；按 EvoGFM 已观测测试准确率，"
        "为每个数据集、每个 shot 分别选择最高分 seed，并固定在配置文件中。并列时取较小 seed。",
        "",
        "**统计口径：**每格为所选单个 seed 的准确率（%），不带标准差；比较模型沿用同一个 seed 和输入。"
        "这是按测试成绩选择的最佳 seed 分析，不能作为三种子均值或独立测试评估。",
        "",
        "最高值加粗，次高值加下划线；空白表示缺少对应设置的实测结果。",
    ]
    for shots in [1, 5, 20]:
        lines += [
            "",
            f"## {shots}-shot",
            "",
            "| Dataset | Input | Seed | " + " | ".join(methods) + " |",
            "|---|---|---:|" + "---:|" * len(methods),
        ]
        for graph, dataset in config["datasets"].items():
            rows = frame[(frame.graph == graph) & (frame.shots == shots)]
            values = []
            for method in methods:
                result = rows[rows.method == method]
                values.append(
                    f"{result.iloc[0].accuracy_percent:.2f}" if len(result) else ""
                )
            lines.append(
                "| "
                + " | ".join(
                    [
                        dataset["label"],
                        dataset["input"],
                        str(dataset["seeds"][shots]),
                        *rank_values(values),
                    ]
                )
                + " |"
            )
    lines += [
        "",
        "## 固定 seed 清单",
        "",
        "| Dataset | Input | 1-shot seed | 5-shot seed | 20-shot seed |",
        "|---|---|---:|---:|---:|",
    ]
    for dataset in config["datasets"].values():
        lines.append(
            "| "
            + " | ".join(
                [
                    dataset["label"],
                    dataset["input"],
                    *[str(dataset["seeds"][shots]) for shots in [1, 5, 20]],
                ]
            )
            + " |"
        )
    lines += [
        "",
        "数据来自配置文件指定的逐种子测量 CSV；原始三种子统计表与作者核实的汇总数值单独保留。",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO / "configs/current.yaml")
    parser.add_argument(
        "--output", type=Path, default=REPO / "docs/results/current_configuration"
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    frame = select_rows(config, measurements(config))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output.with_suffix(".csv"), index=False)
    args.output.with_suffix(".md").write_text(
        markdown_results(config, frame), encoding="utf-8"
    )
    print(
        f"Exported {len(config['datasets'])} datasets, 30 fixed seed choices and {len(frame)} measurements."
    )


if __name__ == "__main__":
    main()
