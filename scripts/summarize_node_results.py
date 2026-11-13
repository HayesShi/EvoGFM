"""Summarize seed-level NC measurements without averaging standard deviations."""

import argparse
from pathlib import Path

import pandas as pd

GRAPHS = {
    "air-brazil",
    "air-eu",
    "air-us",
    "wikics",
    "blogcatalog",
    "deezer",
    "lastfm-asia",
    "ogbn-arxiv",
    "wiki",
    "coauthor-physics",
}
SEEDS = {42, 43, 44}
METHODS = ["EvoGFM", "GraphAny", "GraphGlue", "GraphPFN", "SCGFM", "FeatureRidge"]


def markdown_tables(per_graph, overall):
    """Render editable result tables; incomplete ten-graph aggregates stay blank."""
    methods = [method for method in METHODS if method in set(per_graph.method)]
    methods += sorted(set(per_graph.method) - set(METHODS))

    def cell(frame, **keys):
        for key, value in keys.items():
            frame = frame[frame[key] == value]
        if frame.empty:
            return ""
        row = frame.iloc[0]
        return f"{row['mean']:.2f} ± {row['std']:.2f}"

    lines = [
        "## Overall accuracy (%)",
        "",
        "| Method | 1-shot | 5-shot | 20-shot |",
        "|---|---:|---:|---:|",
    ]
    for method in methods:
        lines.append(
            "| "
            + " | ".join(
                [method]
                + [cell(overall, method=method, shots=shot) for shot in [1, 5, 20]]
            )
            + " |"
        )
    for shot in [1, 5, 20]:
        lines.extend(
            [
                "",
                f"## Per-graph accuracy: {shot}-shot (%)",
                "",
                "| Dataset | " + " | ".join(methods) + " |",
                "|---|" + "---:|" * len(methods),
            ]
        )
        for graph in sorted(GRAPHS):
            lines.append(
                "| "
                + " | ".join(
                    [graph]
                    + [
                        cell(per_graph, method=method, graph=graph, shots=shot)
                        for method in methods
                    ]
                )
                + " |"
            )
    return "\n".join(lines) + "\n"


def summarize(frame):
    keys = ["method", "graph", "seed", "shots"]
    if frame.duplicated(keys).any():
        raise ValueError("Duplicate method/graph/seed/shot measurements")
    if not frame["accuracy_percent"].between(0, 100).all():
        raise ValueError("Accuracy must be a finite percentage")
    for _, group in frame.groupby(["method", "graph", "shots"]):
        if set(group["seed"]) != SEEDS:
            raise ValueError("Each per-graph result requires all three support seeds")
    per_graph = (
        frame.groupby(["method", "graph", "shots"])["accuracy_percent"]
        .agg(mean="mean", std="std")
        .reset_index()
    )
    overall = []
    for (method, shots), group in frame.groupby(["method", "shots"]):
        if set(group["graph"]) != GRAPHS:
            continue
        # Average the same ten graphs within each seed, then summarize seeds.
        seed_means = group.groupby("seed")["accuracy_percent"].mean()
        overall.append(
            dict(
                method=method,
                shots=shots,
                graphs=len(GRAPHS),
                seeds=len(SEEDS),
                mean=seed_means.mean(),
                std=seed_means.std(ddof=1),
            )
        )
    return per_graph, pd.DataFrame(
        overall, columns=["method", "shots", "graphs", "seeds", "mean", "std"]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("docs/results/node_classification.csv")
    )
    parser.add_argument("--output", type=Path, default=Path("runs/summary"))
    args = parser.parse_args()
    per_graph, overall = summarize(pd.read_csv(args.input))
    args.output.mkdir(parents=True, exist_ok=True)
    per_graph.to_csv(args.output / "per_graph.csv", index=False)
    overall.to_csv(args.output / "overall.csv", index=False)
    (args.output / "node_classification.md").write_text(
        markdown_tables(per_graph, overall), encoding="utf-8"
    )
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
