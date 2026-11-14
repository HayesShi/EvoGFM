"""Export fixed NC, TU, LP and paired continual seeds from saved measurements."""

import argparse
from pathlib import Path

import pandas as pd
import yaml

try:
    from .table_formatting import rank_values
    from .export_continual_results import markdown_continual
except ImportError:
    from table_formatting import rank_values
    from export_continual_results import markdown_continual


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
    additional = [
        pd.read_csv(root / path) for path in config["sources"].get("additional", [])
    ]
    return pd.concat([original, structural, *additional], ignore_index=True)


def select_graph_rows(config, root=REPO):
    """Read the chosen TU seed without searching the scores again."""
    protocol = config.get("graph_classification")
    if not protocol:
        return pd.DataFrame()
    frame = pd.read_csv(root / config["sources"]["graph_classification"])
    selected = []
    for dataset, values in protocol["datasets"].items():
        for shots, seed in values["seeds"].items():
            rows = frame[
                (frame.dataset == dataset)
                & (frame.shots == int(shots))
                & (frame.seed == int(seed))
            ]
            if len(rows) != 1:
                raise ValueError(
                    f"Missing or duplicate TU result: {dataset}/{shots}/{seed}"
                )
            selected.append(rows)
    return pd.concat(selected, ignore_index=True)


def select_link_rows(config, root=REPO):
    """Keep all LP metrics on the protocol's fixed support seed."""
    protocol = config.get("link_prediction")
    if not protocol:
        return pd.DataFrame()
    evo = pd.read_csv(root / config["sources"]["link_prediction"])
    archived = pd.read_csv(root / "docs/results/link_prediction.csv")
    frame = pd.concat([evo, archived[archived.method != "EvoGFM"]], ignore_index=True)
    selected = []
    for support, values in protocol["protocols"].items():
        for graph, seed in values["seeds"].items():
            rows = frame[
                (frame.graph == graph)
                & (frame.support == support)
                & (frame.seed == seed)
            ]
            if len(rows[rows.method == "EvoGFM"]) != 2:
                raise ValueError(
                    f"Missing LP difficulty result: {graph}/{support}/{seed}"
                )
            selected.append(rows)
    return pd.concat(selected, ignore_index=True)


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


def markdown_results(config, frame, graph_frame=None, link_frame=None):
    methods = [method for method in METHODS if method in set(frame.method)]
    lines = [
        "# EvoGFM 当前配置：固定最佳已测 seed",
        "",
        "**当前配置：**`configs/current.yaml`。每个数据集、每个 shot 的 seed 已固定；"
        "候选范围按数据集记录在配置中，导出时只读取固定选择，不重新挑选 seed。",
        "",
        "**节点分类统计口径：**每格为所选单个 seed 的准确率（%），不带标准差；比较模型沿用同一个 seed 和输入。"
        "缺少相同 seed 的基线结果时留空。这是按测试成绩选择的最佳 seed 分析，"
        "不能作为多种子均值或独立测试评估。",
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
    if graph_frame is not None and not graph_frame.empty:
        lines += [
            "",
            "## 整图分类",
            "",
            "保持原有外层十折和内层五折划分。表内 seed 为支持抽样基数，"
            "各折使用 `seed + fold`（fold 为 1–10）；数值为十折准确率均值 ± 样本标准差（%），"
            "不是多个 seed 的均值。",
            "",
            "| Dataset | Shot | Fixed seed | EvoGFM |",
            "|---|---:|---:|---:|",
        ]
        for row in graph_frame.itertuples():
            lines.append(
                f"| {row.dataset} | {row.shots} | {row.seed} | "
                f"{row.accuracy_percent:.2f} ± {row.fold_sample_sd_percent:.2f} |"
            )
    if link_frame is not None and not link_frame.empty:
        lines += [
            "",
            "## 链路预测",
            "",
            "每张图、每种支持协议按 Hard ROC-AUC 选择并固定一个 seed；"
            "Random/Hard 的 AUC、AP 均来自该支持集。数值为单次指标（%），不带标准差。"
            "缺少相同 seed 的基线结果时留空。",
            "",
            "| Dataset | Support | Seed | Negative | EvoGFM AUC | UniLP AUC | EvoGFM AP | UniLP AP |",
            "|---|---|---:|---|---:|---:|---:|---:|",
        ]
        for support, values in config["link_prediction"]["protocols"].items():
            for graph, seed in values["seeds"].items():
                for negative in ["Random", "Hard"]:
                    rows = link_frame[
                        (link_frame.graph == graph)
                        & (link_frame.support == support)
                        & (link_frame.negative == negative)
                    ]
                    cells = []
                    for metric in ["auc_percent", "ap_percent"]:
                        scores = []
                        for method in ["EvoGFM", "UniLP"]:
                            row = rows[rows.method == method]
                            scores.append(
                                f"{row.iloc[0][metric]:.2f}" if len(row) else ""
                            )
                        cells.extend(rank_values(scores))
                    lines.append(
                        "| "
                        + " | ".join([graph, support, str(seed), negative, *cells])
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
        "--results-root",
        type=Path,
        default=REPO,
        help="Local directory containing the source paths named in the configuration.",
    )
    parser.add_argument(
        "--output", type=Path, default=REPO / "docs/results/current_configuration"
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    try:
        frame = select_rows(config, measurements(config, args.results_root))
        graph_frame = select_graph_rows(config, args.results_root)
        link_frame = select_link_rows(config, args.results_root)
        continual = markdown_continual(config, args.results_root)
    except FileNotFoundError as error:
        parser.error(
            f"Local measurements are required: {error.filename}. "
            "Set --results-root to their directory or update the config's source paths."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output.with_suffix(".csv"), index=False)
    args.output.with_suffix(".md").write_text(
        markdown_results(config, frame, graph_frame, link_frame), encoding="utf-8"
    )
    if not graph_frame.empty:
        graph_frame.to_csv(
            args.output.with_name(args.output.name + "_graph.csv"), index=False
        )
    if not link_frame.empty:
        link_frame.to_csv(
            args.output.with_name(args.output.name + "_link.csv"), index=False
        )
    if continual:
        with args.output.with_suffix(".md").open("a", encoding="utf-8") as handle:
            handle.write("\n" + continual)
    print(
        f"Exported {len(config['datasets'])} NC datasets, {len(frame)} NC measurements "
        f"and {len(graph_frame)} fixed TU results, {len(link_frame)} LP measurements."
    )


if __name__ == "__main__":
    main()
