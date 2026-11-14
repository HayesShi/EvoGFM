# EvoGFM

Graph-conditioned inverse attention for node classification, link prediction and
graph classification, with continual calibration across graph domains.

EvoGFM estimates relation directions from each graph and shares an inverse map
across graphs. Target evaluation freezes the source model and fits a task readout.
Continual adaptation updates 32 shared inverse coefficients while keeping the
backbone frozen.

## Install

Use Python 3.10+ and PyTorch 2.5–2.6. For CUDA, install a compatible PyTorch build
first. The commands below run from the repository root.

```bash
git clone https://github.com/HayesShi/EvoGFM.git
cd EvoGFM
python -m pip install -r requirements.txt
```

## Quick start

```bash
# Download the default graphs and prepare public evaluation splits.
python prepare.py

# Train on Pubmed, Photo and Texas; select on source validation.
python train.py --epochs 60

# Evaluate the frozen source checkpoint.
python evaluate.py
python evaluate.py --split test

# Adapt over a graph stream.
python continual.py
```

The default checkpoint for these commands is `runs/source/best.pt`. Downloads
are cached in `data/`; generated checkpoints and measurements go under `runs/`.
Use `--help` for individual options.

## Paper experiments

The original epoch-21 reference checkpoint and its source label partitions are
included under `checkpoints/`. Standard experiment entry points use this checkpoint.

```bash
python -m experiments.run nc          # Few-shot NC, arrival orders and retention
python -m experiments.run lp          # Random/mixed-support link prediction
python -m experiments.run gc          # TU graph classification: 5/20-shot
python -m experiments.run continual   # Node-only, link-only and joint streams
python -m experiments.run mechanisms  # Spectral and supervision controls
```

Run `nc` before `continual` and `mechanisms`. Outputs default to `runs/paper/`.
Add `--dry-run` to inspect the plan or `--device cuda` for NC evaluation on GPU.
See the [reproduction guide](docs/reproduction.md) for datasets, seeds, protocols,
checkpoint configuration, external baselines and timing rules.

The [fixed seed configuration](configs/current.yaml) records the selected NC,
graph-classification, LP and continual-study seeds. These are exploratory
selections on observed test results. The standard experiment commands above
retain their multi-seed protocols.

Measurements, result tables and plotting inputs are generated locally and are
excluded from Git. To export fixed-seed tables, provide the local CSV files named
in the configuration and run `python scripts/export_current_results.py`.

## Figures

```bash
python -m pip install -e ".[plots]"
python scripts/plot_node_classification.py --data runs/plot_inputs
python scripts/plot_link_prediction.py --curves runs/plot_inputs/link_curves.npz
python scripts/plot_retention.py --scores runs/plot_inputs/retention.csv --orders runs/plot_inputs/retention_orders.json
```

Supply plotting inputs from local experiment outputs. The scripts export PNG,
PDF and editable SVG to `runs/figures/`. View the [three retained images](figures/README.md).

## Other entry points

| Task | Script |
|---|---|
| Frozen node classification and link prediction | `evaluate.py` |
| Continual adaptation | `continual.py` |
| Fixed-head historical retention | `evaluate_retention.py` |
| Future-domain evaluation | `evaluate_future.py` |
| Few-shot accumulation | `explore_accumulation.py` |
| Nested-CV / episode graph classification | `classify_graphs.py`, `classify_fewshot.py` |
| Official HeaRT / OGB link protocols | `heart.py`, `link_benchmark.py` |
| GraphLand classification and regression | `evaluate_graphland.py` |
| PPA representation ablation | `ablate_ppa.py` |
| End-to-end runtime and memory | `benchmark_efficiency.py` |

`evaluate.py --shots K` limits training labels to K per class; its readout
selection still uses validation labels. The paper runner uses fixed ridge 0.1
for its target NC protocol. HeaRT needs the official files in
`data/heart/<dataset>/`: `train_pos.txt`, `valid_pos.txt`, `test_pos.txt`,
`heart_valid_samples.npy`, `heart_test_samples.npy` and `gnn_feature`.
Install `.[benchmarks]` for the efficiency benchmark's optional dependencies.

## Layout

```text
evogfm/       Model, graph features, data loaders, readouts and continual updates
experiments/  Standard paper protocols and shared experiment settings
scripts/      Figure rendering and prediction-to-curve export
checkpoints/  Reference model and its source label partitions
figures/      PNG figure previews
tests/        Numerical and protocol regression tests
docs/         Reproduction details
```

Load a checkpoint in Python:

```python
from evogfm.training import load_checkpoint

model, link_head, metadata = load_checkpoint("checkpoints/reference.pt")
```

## Development

```bash
python -m pip install -e ".[dev,plots]"
python -m pytest -q
python -m ruff check .
python -m black --check .
```

## References and license

Data and evaluation protocols build on [GraphAny](https://github.com/DeepGraphLearning/GraphAny),
[NCN](https://github.com/GraphPKU/NeuralCommonNeighbor), [HeaRT](https://github.com/Juanhui28/HeaRT),
[OGB](https://ogb.stanford.edu/), [TU datasets](https://chrsmrrs.github.io/datasets/)
and [SNAP](https://snap.stanford.edu/data/).
External models and datasets retain their own licenses. EvoGFM code is released
under the [MIT License](LICENSE).
