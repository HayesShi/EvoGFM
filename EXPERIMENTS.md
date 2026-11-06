# Expanded experiments (V4)

This report preserves the V4 runs. See [V5 experiments](EXPERIMENTS_V5.md) for the later larger-graph, sparse-edge, five-shot, hard-negative and graph-classification results.

The multiscale readout improves all seven node-classification tests in this development run. The source checkpoint and inverse-attention construction stay fixed. The next unresolved issue is whether the inverse map itself provides a reliable advantage in node classification and future-domain transfer.

All runs use seed 42, CPU and four threads. The original source model was trained on Pubmed, Photo and Texas for 60 epochs (240 updates); source validation selected epoch 21. Target readouts use training labels/edges, with validation used for readout selection. The old public tests had already been observed during development. Published baseline means come from their papers; source graphs, training budgets and averaging differ from this one-seed run.

## Readout change

The original NC readout concatenates separately normalized zero-, one- and two-hop features and the Transformer output. The expanded readout tests four fixed choices, each with ridge penalties `{0.01, 0.1, 1, 10, 100}`:

| Choice | Propagation orders | Transformer weight | Extra features |
| :--- | :--- | ---: | :--- |
| current | 0–2 | 1.0 | — |
| light_values | 0–2 | 0.1 | — |
| multiscale | 0–4 | 0.5 | — |
| quadratic | 0–4 | 0.5 | 512 fixed quadratic random features |

Each propagation order remains a separate channel, so ridge can use longer neighborhoods without discarding local features. The quadratic sketch uses two independent fixed Gaussian projections: `phi(x) = (xW) * (xV) / sqrt(512)` for normalized `x`. Its expected inner product is `(xᵀy)²`; it approximates a positive semidefinite degree-two polynomial kernel. Only the final ridge coefficients are fit. This is a readout improvement, with the original graph-to-Q/K method retained.

## Node classification

Both columns use the same frozen source checkpoint and target training masks. Planetoid masks are used for Cora/Citeseer; the heterophily graphs use official mask column 0. Multiclass selection uses accuracy; binary selection uses ROC-AUC. Changes are percentage points.

| Test graph | Metric (%) | Original readout | Expanded readout | Change |
| :--- | :--- | ---: | ---: | ---: |
| Cora | Accuracy | 75.70 | 79.10 | +3.40 |
| Citeseer | Accuracy | 67.00 | 69.80 | +2.80 |
| Roman-empire | Accuracy | 72.20 | 73.42 | +1.22 |
| Amazon-ratings | Accuracy | 42.41 | 44.06 | +1.65 |
| Minesweeper | ROC-AUC | 88.31 | 88.82 | +0.51 |
| Tolokers | ROC-AUC | 78.71 | 80.16 | +1.46 |
| Questions | ROC-AUC | 75.18 | 75.95 | +0.77 |

The four new graphs test different conditions: Amazon-ratings has 24,492 nodes and rating labels; Minesweeper tests local grid structure; Tolokers has 519,000 undirected edges; Questions has 48,921 nodes and severe class imbalance. Questions accuracy is about 97% even with weak discrimination, so ROC-AUC is the main metric.

[GraphAny-Wisconsin, ICLR 2025, Table 5](https://arxiv.org/html/2405.20445#A4.T5) reports Cora 77.82, Citeseer 67.50, Roman-empire 64.06 and Amazon-ratings 42.57 accuracy. The current values exceed those particular published references numerically. Fully supervised models from the [ICLR 2023 heterophily benchmark](https://arxiv.org/html/2302.11640) remain stronger on the new graphs: for example, SAGE reports Amazon-ratings 53.63 accuracy and Minesweeper 93.51 ROC-AUC. The binary GraphAny accuracy numbers cannot be compared to our ROC-AUC column.

## Hard link prediction

The Citeseer HeaRT experiment uses the official 85/5/10 positives and 500 official hard negatives per positive, preserving their ordering. Only training edges enter graph statistics and readout calibration. The original 18-parameter residual head is fitted on four masked training views; its regularizer is chosen on a disjoint calibration holdout. Query batches cache graph-wide representations and compute the requested structural features without constructing full `A²` or `A³` matrices.

| Citeseer HeaRT test | MRR (%) | Hits@10 (%) |
| :--- | ---: | ---: |
| EvoGFM, seed 42 | 27.28 | 54.73 |
| NCN, published | 28.65 ± 1.21 | 53.41 ± 1.46 |
| NCNC, published | 24.10 ± 0.65 | 53.72 ± 0.97 |
| BUDDY, published | 22.84 ± 0.36 | 48.35 ± 1.18 |
| GAE, published | 25.25 ± 0.82 | 49.65 ± 1.48 |

The references are from [HeaRT, NeurIPS 2023 Datasets and Benchmarks, Table 4](https://arxiv.org/html/2306.10453). These are published comparisons with different training budgets. The new result is close to NCN on MRR and numerically above NCNC on both metrics. The previous random-negative public LP results remain Cora 86.35 and Citeseer 92.64 Hits@100; this round found no consistent improvement to that readout.

## What the mechanism ablations show

Setting inverse amplitudes and biases to zero removes the learned affinity and gives uniform neighbor propagation. The shared value path remains. Each ablation receives the same NC readout search or LP calibration budget.

| NC validation graph | Full inverse (%) | Zero inverse (%) |
| :--- | ---: | ---: |
| Cora, accuracy | 79.40 | 80.20 |
| Citeseer, accuracy | 69.00 | 70.20 |
| Roman-empire, accuracy | 74.09 | 74.49 |
| Amazon-ratings, accuracy | 44.31 | 44.23 |
| Minesweeper, ROC-AUC | 88.12 | 89.02 |
| Tolokers, ROC-AUC | 81.41 | 81.76 |

The NC gains currently come mainly from the readout; these ablations do not establish a benefit from the inverse map. HeaRT is mixed: full versus zero inverse gives test MRR 27.28 versus 27.00, but Hits@10 54.73 versus 56.04. Validation MRR is 24.12 versus 24.87. Both outcomes are retained in `results.json`.

## Longer domain stream

The stream is Cora → Citeseer → Amazon-ratings, with Minesweeper and Tolokers reserved as future graphs. The experiment retains the original NC readout with fixed ridge 0.1 and the native masked-edge LP protocol. Its validation scores are separate from the public-test tables above. Future-domain labels provide measurements only.

| Task / control | Final stream average (%) | Forgetting (pp) | Future Minesweeper gain (pp) | Future Tolokers gain (pp) |
| :--- | ---: | ---: | ---: | ---: |
| NC static | 61.565 | 0.000 | 0.000 | 0.000 |
| NC cumulative | 61.320 | 0.200 | +0.195 | −0.256 |
| NC reset | 61.934 | 0.000 | +0.320 | −0.249 |
| LP static | 91.182 | 0.000 | 0.000 | 0.000 |
| LP cumulative | 91.509 | 0.002 | −0.100 | +0.067 |
| LP reset | 91.328 | 0.301 | −0.020 | −0.065 |

Cumulative LP retains earlier graphs better than reset and improves the stream average. Future gains change sign, so stable knowledge accumulation has not been demonstrated. A single additional candidate adds `0.5 λ (θ−θ₀)ᵀ A_source (θ−θ₀)` to calibration with fixed `λ=0.1`. NC final average improves to 61.599 and forgetting falls to 0.100 pp, but future gains are −0.048 and −0.042 pp. The optional `--source-anchor 0.1` reproduces this trial; its default remains zero.

## Other trials and next decision

Two complete 60-epoch source-training trials increased the NC loss weight from 0.2 to 1.0, with attention ranks 16 and 32. Neither improved overall target validation, so the original checkpoint was retained. Adding LP ranking loss, expanding the readout from 18 to 26 parameters, or changing density calibration also failed to give consistent validation gains. Their local development logs are retained.

The implemented progress is stronger frozen-backbone adaptation across seven NC graphs and a working official hard-negative LP evaluator. The next method change should improve the inverse-versus-uniform gap and future-domain transfer under a fixed development protocol. A larger test sweep alone would not resolve those two findings.

Full measured values and local log paths are in [results.json](results.json). Raw data, checkpoints and run directories remain local. The supplied manuscript is unchanged.
