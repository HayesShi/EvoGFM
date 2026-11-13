"""Plot measured per-dataset few-shot differences; tables remain editable text."""

import argparse, csv, math
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BLUE, ORANGE = "#0F4D92", "#D55E00"
LABELS = {
    "air-brazil": "Air-Brazil",
    "air-eu": "Air-EU",
    "air-us": "Air-US",
    "wikics": "WikiCS",
    "blogcatalog": "BlogCatalog",
    "deezer": "Deezer",
    "lastfm-asia": "LastFM-Asia",
    "ogbn-arxiv": "OGBN-Arxiv",
    "wiki": "Wiki",
    "coauthor-physics": "Coauthor-Physics",
}


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def export(fig, stem):
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(OUTPUT / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_nc():
    graphs = list(LABELS)
    shots = [1, 5, 20]
    any_rows = read_csv(DATA / "nc_graphany.csv")
    glue_rows = read_csv(DATA / "nc_graphglue.csv")
    matrices = []
    for method, rows in [("GraphAny", any_rows), ("GraphGlue", glue_rows)]:
        mapping = {(r["graph"], int(r["shots"])): r for r in rows}
        matrix = np.zeros((10, 3))
        for i, graph in enumerate(graphs):
            for j, shot in enumerate(shots):
                r = mapping[(graph, shot)]
                assert int(r["seeds"]) == 3
                matrix[i, j] = float(r["evogfm_initial_accuracy_percent_mean"]) - float(
                    r[f"{method.lower()}_accuracy_percent_mean"]
                )
        matrices.append((method, matrix))
    limit = math.ceil(max(np.abs(m).max() for _, m in matrices) / 5) * 5
    cmap = LinearSegmentedColormap.from_list("paired", [ORANGE, "#FCFCFC", BLUE])
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 5.7), sharey=True)
    for ax, (method, matrix) in zip(axes, matrices):
        image = ax.imshow(matrix, vmin=-limit, vmax=limit, cmap=cmap, aspect="auto")
        for i in range(10):
            for j in range(3):
                value = matrix[i, j]
                label = "0.0" if abs(value) < 0.05 else f"{value:+.1f}"
                ax.text(
                    j,
                    i,
                    label,
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white" if abs(value) > 0.6 * limit else "#222222",
                )
        ax.set_title(f"EvoGFM vs. {method}", loc="left", fontsize=12, pad=12)
        ax.set_xticks(range(3), ["1-shot", "5-shot", "20-shot"])
        ax.set_yticks(range(10), [LABELS[g] for g in graphs])
        ax.tick_params(length=0, pad=8)
        ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 10, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5)
        ax.tick_params(which="minor", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.subplots_adjust(left=0.22, right=0.98, top=0.88, bottom=0.22, wspace=0.17)
    cax = fig.add_axes([0.36, 0.105, 0.46, 0.022])
    colorbar = fig.colorbar(
        image, cax=cax, orientation="horizontal", ticks=[-limit, 0, limit]
    )
    colorbar.outline.set_visible(False)
    colorbar.set_label("EvoGFM accuracy minus baseline (pp)", fontsize=10)
    fig.text(
        0.60,
        0.96,
        "Identical support and test nodes; mean over three support seeds",
        ha="center",
        fontsize=10,
        color="#444444",
    )
    export(fig, "node_classification")


def main():
    global DATA, OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("figures/data"))
    parser.add_argument("--output", type=Path, default=Path("runs/figures"))
    args = parser.parse_args()
    DATA = args.data
    OUTPUT = args.output
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": ["Arial", "DejaVu Sans"],
            "font.size": 10,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    plot_nc()


if __name__ == "__main__":
    main()
