"""Recreate the paper ROC and PR panels from measured seed-level curves."""

import argparse
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BLUE, ORANGE = "#0F4D92", "#D55E00"


def export(fig, stem):
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(
            OUTPUT / f"{stem}.{suffix}",
            dpi=300,
            facecolor="white",
            bbox_inches="tight",
            pad_inches=0.09,
        )
    plt.close(fig)


def plot_lp(archive):
    data = np.load(archive, allow_pickle=False)
    grid = data["grid"]
    graphs = data["graphs"].tolist()
    methods = data["methods"].tolist()
    curves = {}
    for ki, kind in enumerate(["ROC", "PR"]):
        for gi, graph in enumerate(graphs):
            for di, difficulty in enumerate(["Random", "Hard"]):
                for mi, method in enumerate(methods):
                    curves[(kind, graph, difficulty, method)] = data["curves"][
                        ki, gi, di, mi
                    ]
    styles = [
        (
            m,
            BLUE if m.startswith("EvoGFM") else ORANGE,
            "--" if "random support" in m else "-",
        )
        for m in methods
    ]
    for kind in ["ROC", "PR"]:
        fig, axes = plt.subplots(2, 4, figsize=(13.1, 7.0), sharex=True, sharey=True)
        for row, difficulty in enumerate(["Random", "Hard"]):
            for col, graph in enumerate(graphs):
                ax = axes[row, col]
                for method, color, linestyle in styles:
                    values = np.stack(curves[(kind, graph, difficulty, method)])
                    assert values.shape == (3, len(grid))
                    mean, sd = values.mean(0), values.std(0, ddof=1)
                    ax.fill_between(
                        grid,
                        np.maximum(0, mean - sd),
                        np.minimum(1, mean + sd),
                        color=color,
                        alpha=0.075,
                        linewidth=0,
                    )
                    ax.plot(grid, mean, color=color, ls=linestyle, lw=1.6)
                if kind == "ROC":
                    ax.plot([0, 1], [0, 1], color="#B8B8B8", lw=0.65, ls=":", zorder=0)
                else:
                    ax.axhline(0.5, color="#B8B8B8", lw=0.65, ls=":", zorder=0)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1.02)
                ax.set_xticks([0, 0.5, 1])
                ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
                ax.grid(color="#E9E9E9", linewidth=0.5)
                ax.tick_params(labelsize=9, length=3)
                if row == 0:
                    ax.set_title(graph.capitalize(), loc="left", fontsize=11)
                if col == 0:
                    ax.set_ylabel(
                        f"{difficulty} test negatives\n"
                        + ("True positive rate" if kind == "ROC" else "Precision"),
                        fontsize=10,
                    )
                if row == 1:
                    ax.set_xlabel(
                        "False positive rate" if kind == "ROC" else "Recall",
                        fontsize=10,
                    )
        handles = [
            Line2D([0], [0], color=c, ls=ls, lw=1.8, label=m) for m, c, ls in styles
        ]
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.52, 0.985),
            ncol=2,
            fontsize=10,
            frameon=False,
            columnspacing=3,
            handlelength=3,
        )
        fig.subplots_adjust(
            left=0.078, right=0.99, top=0.82, bottom=0.14, wspace=0.18, hspace=0.30
        )
        fig.text(
            0.53,
            0.035,
            "All original query edges; mean curves and pointwise ±1 SD over three support seeds.\n"
            "Random support: 200 positive + 200 random negative; mixed support: 200 positive + 100 random + 100 hard negative.",
            fontsize=9,
            ha="center",
            va="center",
            color="#444444",
        )
        export(fig, f"link_prediction_{kind.lower()}")


def main():
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--curves", type=Path, default=Path("figures/data/link_curves.npz")
    )
    parser.add_argument("--output", type=Path, default=Path("runs/figures"))
    args = parser.parse_args()
    OUTPUT = args.output
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": ["Arial", "DejaVu Sans"],
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    plot_lp(args.curves)


if __name__ == "__main__":
    main()
