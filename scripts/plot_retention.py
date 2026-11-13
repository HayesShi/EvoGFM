"""Plot seed-specific historical retention from measured per-graph trajectories."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scores", type=Path, default=Path("figures/data/retention.csv")
    )
    parser.add_argument(
        "--orders", type=Path, default=Path("figures/data/retention_orders.json")
    )
    parser.add_argument("--output", type=Path, default=Path("runs/figures"))
    args = parser.parse_args()
    frame = pd.read_csv(args.scores)
    orders = json.loads(args.orders.read_text())
    seeds = sorted(frame.seed.unique())
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(
        len(seeds),
        4,
        figsize=(10.2, 2.4 * len(seeds)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    colors = {"EvoGFM": "#0F4D92", "GraphKeeper": "#D55E00"}
    for row, seed in enumerate(seeds):
        for col, graph in enumerate(orders[str(seed)]):
            ax = axes[row, col]
            selected = frame[(frame.seed == seed) & (frame.graph == graph)]
            for method, group in selected.groupby("method", sort=False):
                group = group.sort_values("prefix")
                if group.prefix.tolist() != list(range(4, 16)):
                    raise ValueError(
                        f"Incomplete trajectory: seed {seed}, {graph}, {method}"
                    )
                ax.plot(
                    group.new_graphs_after_anchor,
                    group.change_from_prefix4_pp,
                    color=colors[method],
                    linestyle="-" if method == "EvoGFM" else "--",
                    marker="o" if method == "EvoGFM" else "s",
                    markersize=3,
                    linewidth=1.5,
                    label=method,
                )
                if method == "GraphKeeper":
                    misses = group[
                        group.correct_expert_selected.astype(str).str.lower() == "false"
                    ]
                    ax.scatter(
                        misses.new_graphs_after_anchor,
                        misses.change_from_prefix4_pp,
                        marker="x",
                        s=25,
                        color="#333333",
                        zorder=4,
                    )
            ax.axhline(0, color="0.6", linewidth=0.6)
            ax.grid(axis="y", alpha=0.2)
            ax.set_title(f"{graph}  |  Seed {seed}", loc="left", fontsize=9)
            ax.set_xticks([0, 3, 6, 9, 11])
    handles = [
        Line2D(
            [],
            [],
            color=color,
            linestyle="-" if method == "EvoGFM" else "--",
            marker="o" if method == "EvoGFM" else "s",
            label=method,
        )
        for method, color in colors.items()
    ]
    handles.append(
        Line2D(
            [],
            [],
            color="#333333",
            marker="x",
            linestyle="none",
            label="GraphKeeper selects another graph's expert",
        )
    )
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False)
    fig.supxlabel("New graphs learned after the first four")
    fig.supylabel("Accuracy change from each method's own starting point (pp)")
    fig.tight_layout(rect=(0.02, 0.03, 1, 0.94))
    args.output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            args.output / f"historical_retention.{suffix}", dpi=300, bbox_inches="tight"
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
