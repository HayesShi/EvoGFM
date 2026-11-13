# Reproducing the experiments

Run all commands from the repository root after installing the package. Each
command prints its execution plan; add `--dry-run` to inspect it without running.
The complete protocols retain all graphs, test queries and original label budgets.

## Reference model

`checkpoints/reference.pt` contains the original epoch-21 checkpoint, selected on
source validation after joint training on Pubmed, Photo and Texas. It contains one
source-training seed (42). The three experiment seeds, 42–44, vary support samples
and stream orders; they are not three independently pretrained models.

`checkpoints/source_splits.json` records its source training/validation node IDs.
For a different source checkpoint, provide its matching `--source-splits` file.
To train from scratch, use `python train.py --epochs 60`; numerical results may
vary across hardware and library versions.

## Standard experiment entry points

| Experiment | Command | Saved under `runs/paper/` |
|---|---|---|
| Node classification, arrival orders, historical retention, single-donor transfer | `python -m experiments.run nc` | `node_classification/` |
| Random- and mixed-support link prediction | `python -m experiments.run lp` | `link_prediction/` |
| TU graph classification, five datasets × 5/20-shot | `python -m experiments.run gc` | `graph_classification/` |
| Node-only, link-only and joint continual supervision | `python -m experiments.run continual` | `cross_task/` |
| Spectral powers, amplitude permutations, shuffled supervision | `python -m experiments.run mechanisms` | `node_classification/internal_exploration/` |

Run `nc` before `continual` and `mechanisms`: these use its source curvature,
arrival orders, graph caches and stream states. LP and GC can run independently.
Existing completed states are reused; use a new `--output` directory when changing
the checkpoint or protocol.

```bash
python -m experiments.run nc --device cuda --data-root data --output runs/paper
python -m experiments.run continual --device cuda
```

`--device` selects the NC evaluation device. Continual updates and the matched LP
solver use CPU, as in the original runs. CUDA requires a compatible PyTorch build.

## Protocols

**Node classification.** Fifteen learning graphs, ten recipient graphs, three
support/order seeds, and 1/5/20 labels per class. The four arrival policies share
the same initial four graphs. GGN calibration updates 32 inverse coefficients;
the eight offsets, backbone and source LP head remain frozen. All-edge moments
are accumulated in float64 blocks. This estimator is defined in
`experiments/full_graph.py`; the library's sampled default remains available to
the original entry points. Target ridge is fixed to 0.1. Historical readouts are
fitted at prefix 4 and held fixed through prefix 15.

**Link prediction.** Chameleon, Cora, Computers and Squirrel use identical test
queries in the two support settings. Both use 200 positive support edges. Random
support adds 200 random nonedges; mixed support adds 100 random and 100 hard
nonedges. Random/Hard in the result rows describes the *test negatives*, separately
from the support setting. The 17-channel source scorer is frozen; the offset
logistic readout uses ridge 0.1 and a converged float64 Armijo Newton solve.
Support, visible structure and query exclusions are checked in
`experiments/protocols.py`. Predictions and sampled pairs are saved for each seed.
The appendix Table A1 uses the first three datasets; the curve collection also
retains Squirrel.

**Graph classification.** MUTAG, PROTEINS, NCI1, DD and IMDB-BINARY use outer
10-fold stratified splits, 5/20 support graphs per class and the native balanced
readout. Support-only inner CV selects SVM parameters and block weights. The
existing graph-classification estimator is retained. Fold variation and support
seed variation are different statistics; compute the paper's sample SD with
`ddof=1` from saved fold scores rather than reusing a population SD field.

**Continual supervision.** Three streams share the same visible 60% structure,
arrival order and four updates per graph. Future targets are Twitch EN/DE/FR and
GitHub. The target preparation script downloads the public SNAP archives. Joint
supervision uses both node and link annotations; equal update counts do not imply
equal annotation or compute budgets.

**Efficiency.** `benchmark_efficiency.py` retains the end-to-end timing and memory
entry point (`pip install -e ".[benchmarks]"`). The planned GGN/Adam/full-tuning
Figure 9 comparison needs matched measurements; it is not generated from the
old development timing snapshots.

## Baselines

The repository contains EvoGFM and its GraphAny preprocessing adapter. External
models use their own official implementations and checkpoints; their source code,
weights and datasets are not bundled here. Export identical support/test IDs
before comparing methods. The archived plotting measurements cover GraphAny,
GraphGlue, UniLP and GraphKeeper; see [figure inputs](../figures/README.md).

The UniLP checkpoint's development data includes Cora. Its Cora comparison is
retained with that overlap. GraphKeeper historical plots retain native expert
routing and restrict predictions to known graph classes. Correct-expert scores
are diagnostic results and are separate from these trajectories.

## Plotting

```bash
python -m pip install -e ".[plots]"
python scripts/plot_node_classification.py
python scripts/plot_link_prediction.py
python scripts/plot_retention.py
```

These read the small measured inputs in `figures/data/` and write editable SVG,
PDF and 300-DPI PNG to `runs/figures/`. Tables remain Markdown/CSV.

To plot a new complete LP run, export its prediction files first:

```bash
python scripts/export_link_curves.py --predictions runs/paper/link_prediction --output runs/new_curves.npz
python scripts/plot_link_prediction.py --curves runs/new_curves.npz
```

An optional `--unilp DIRECTORY` accepts matching baseline files with `scores`,
`query_pairs` and `positive_count` fields. All query IDs must agree. The exporter
also accepts `--source DIRECTORY` for the original baseline-artifact layout.
