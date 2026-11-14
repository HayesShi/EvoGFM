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
before comparing methods. The retained [figure previews](../figures/README.md)
include GraphAny, GraphGlue, UniLP and GraphKeeper comparisons; their measured
inputs are kept locally.

The UniLP checkpoint's development data includes Cora. Its Cora comparison is
retained with that overlap. GraphKeeper historical plots retain native expert
routing and restrict predictions to known graph classes. Correct-expert scores
are diagnostic results and are separate from these trajectories.

## Additional baseline controls

The new controls preserve the original source checkpoint, graph splits and
support sampler. Airport input changes are separate diagnostics. They do not
change `evogfm/` preprocessing or overwrite the main NC run.

```bash
python -m experiments.baseline_feature_ridge --data-root data --output runs/feature_ridge
python -m experiments.airport_inputs --data-root data --output runs/airport_inputs
```

The second command exports shared input NPZ files and the exact support/test
IDs. In each baseline's official environment, run its unchanged model through
the adapter (replace the uppercase paths below with local paths):

```bash
python experiments/baseline_airport.py graphany --official GRAPHANY_SOURCE --checkpoint GRAPHANY_CHECKPOINT --inputs runs/airport_inputs --output runs/graphany_quantile
python experiments/baseline_airport.py graphpfn --official GRAPHPFN_SOURCE/src --checkpoint GRAPHPFN_ADAPTER --limix LIMIX_CHECKPOINT --inputs runs/airport_inputs --output runs/graphpfn_quantile --graph air-brazil --seed 42 --shots 5
```

GraphAny runs all 27 airport episodes by default. Run the GraphPFN command once
per graph, seed and budget to release ensemble memory between processes. Both
default to `degree_quantile64`; pass `--policy degree64` and a new output path
for the earlier encoding. GraphAny uses official least-squares channel readouts;
GraphPFN uses the full ten-member ensemble. DGL belongs to these separate
official environments. The measured GraphAny runtime used PyTorch 2.1.2 and
DGL 2.2.1; GraphPFN used PyTorch 2.4.0 and DGL 2.4.0 on CPU.

For original inputs on any of the ten NC graphs, export the shared data first:

```bash
python -m experiments.export_nc_inputs --data-root data --output runs/nc_inputs --graphs lastfm-asia
```

Then supply `--inputs runs/nc_inputs --policy identity --graph lastfm-asia` to
the same adapter. Here `identity` labels the original-input policy; it does not
replace a graph's native features with node IDs. Larger GraphPFN runs need more
than the current 2 GB WSL budget. The LastFM attempt ended in a kernel-confirmed
out-of-memory kill before its first ensemble member completed and is not scored.

SCGFM uses an unmodified checkout of
[Xd-He/SCGFM](https://github.com/Xd-He/SCGFM) and the base EvoGFM environment
with the optional POT dependency:

```bash
python -m pip install -e ".[scgfm]"
python -m experiments.baseline_scgfm train --official SCGFM_SOURCE --data-root data --output runs/scgfm --device cuda
python -m experiments.baseline_scgfm evaluate --official SCGFM_SOURCE --data-root data --output runs/scgfm --device cuda --stream --graphs air-brazil air-eu air-us wikics blogcatalog lastfm-asia wiki coauthor-physics deezer ogbn-arxiv
```

Training retains the complete official Photo+Computers configuration, all 5391
source ego-graphs and 60 epochs. The command above evaluates all ten graphs;
without `--graphs`, evaluation defaults to the three airport graphs.
`--stream` materializes the support embeddings and predicts every original test
node in blocks. The full graph, original feature width, official PPR and encoder
remain unchanged. Without streaming, full embeddings retain 32 times the original
feature width and can require substantial memory.
A ten-graph mean is reported only after all ten graphs have all three seeds.

For the matched structural-input diagnostic, reuse that source checkpoint in
a separate result directory:

```bash
python -m experiments.baseline_scgfm evaluate --official SCGFM_SOURCE --data-root data --checkpoint runs/scgfm/checkpoint/model.pt --output runs/scgfm_degree64 --stream --input-directory runs/airport_inputs/inputs --input-policy degree64
```

Repeat with `--input-policy degree_quantile64` and another output directory.

## Plotting

```bash
python -m pip install -e ".[plots]"
python scripts/plot_node_classification.py --data runs/plot_inputs
python scripts/plot_link_prediction.py --curves runs/plot_inputs/link_curves.npz
python scripts/plot_retention.py --scores runs/plot_inputs/retention.csv --orders runs/plot_inputs/retention_orders.json
```

Provide measured inputs from your local experiments. CSV/JSON/NPZ inputs and
generated tables are excluded from Git; the image previews are retained.
The renderers write editable SVG, PDF and 300-DPI PNG to `runs/figures/`.

The NC renderer expects `nc_graphany.csv` and `nc_graphglue.csv` in its `--data`
directory, with per-graph, per-shot three-seed comparison summaries. The retention
renderer reads trajectory rows from `--scores` and the four historical graph
names per seed from `--orders`. The LP renderer reads the curve archive generated
by the exporter below.

To plot a new complete LP run, export its prediction files first:

```bash
python scripts/export_link_curves.py --predictions runs/paper/link_prediction --output runs/new_curves.npz
python scripts/plot_link_prediction.py --curves runs/new_curves.npz
```

An optional `--unilp DIRECTORY` accepts matching baseline files with `scores`,
`query_pairs` and `positive_count` fields. All query IDs must agree. The exporter
also accepts `--source DIRECTORY` for the original baseline-artifact layout.

## Local result tables

Summarize your own seed-level NC CSV, retaining editable output:

```bash
python scripts/summarize_node_results.py --input runs/node_scores.csv --output runs/summary
```

The input columns are `method`, `graph`, `seed`, `shots` and `accuracy_percent`.
An optional `--reported-values FILE` applies explicit table-value overrides from
a local CSV. Result tables and override files are not bundled in the repository.

The selected-seed exporter reads the local files listed under `sources` in
`configs/current.yaml`. To use an external local result directory with the same
relative layout:

```bash
python scripts/export_current_results.py --results-root LOCAL_RESULT_ROOT --output runs/current_configuration
```
