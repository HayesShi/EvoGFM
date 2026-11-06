# V6: broader graph transfer and joint low-resource adaptation

The source encoder and inverse-attention construction stay fixed. This round adds Computers and WikiCS node classification, NCI1 whole-graph classification, a joint five-shot/20%-edge condition, and a four-domain continual stream. All experiments use seed 42. The source checkpoint remains epoch 21 from Pubmed, Photo and Texas. No target graph is added to source training.

## Readout changes

`--node-readout advanced` extends V5's `enhanced` family with two small additions. The existing ridge penalty grid and >0.1-percentage-point validation improvement rule stay fixed.

* **RBF features:** concatenate 256-dimensional Nyström features to the 0–4-hop representations and half-weight Transformer features. Landmarks are sampled without labels using seed 42. The kernel is `exp(-gamma * ||x-y||²)` on row-normalized representations, with `gamma` in `{1,4}`. If `K_LL = U diag(lambda) Uᵀ`, the features are `K_XL U diag(max(lambda,1e-6)^(-1/2))`. Their inner product is a positive semidefinite approximation to the Gaussian kernel. Only the target ridge coefficients are fitted.
* **Training-error correction:** form `R_train = one_hot(y_train) - softmax(8*scores_train)` and zero residuals elsewhere. Iterate `E = 0.2*R + 0.8*P@E` for 20 steps using the inverse-programmed transition. Rescale using the mean absolute residual at training nodes, add the correction to the prediction probabilities, clip and renormalize. Validation selects correction alone or followed by the existing diffusion. Nontraining labels never enter correction. This borrows the residual-propagation principle of [Correct & Smooth, ICLR 2021](https://arxiv.org/html/2010.13993); it is not its exact implementation.

For graph classification, `--balanced-readout` standardizes within each training fold, then divides each feature block by the square root of its dimension. The pooled encoder block receives weight in `{0.1,1,10}` relative to the simple statistics block. Inner five-fold CV selects this weight and SVM C; the ten outer test folds only measure performance. This prevents a large pooled block from dominating RBF distances just because it has more channels. Reusing inner-fold kernels across C values changes computation, not the candidate models or CV criterion.

## Node classification on 11 graphs

Each row uses the same checkpoint, data split and metric on both sides. The first nine compare against V5's recorded test; Computers and WikiCS compare V5 and V6 run in this round. Minesweeper, Tolokers and Questions use ROC-AUC; the other graphs use accuracy. Units are percentages and percentage points.

| Graph | V5 enhanced | V6 advanced | Change |
| :--- | ---: | ---: | ---: |
| cora | 79.60 | 79.60 | +0.00 |
| citeseer | 69.70 | 69.70 | +0.00 |
| roman-empire | 74.37 | 74.37 | +0.00 |
| amazon-ratings | 47.20 | 47.20 | +0.00 |
| minesweeper | 89.16 | 89.16 | +0.00 |
| tolokers | 80.16 | 80.72 | +0.56 |
| questions | 75.95 | 76.24 | +0.28 |
| actor | 32.96 | 32.96 | +0.00 |
| ogbn-arxiv | 67.67 | 67.88 | +0.21 |
| computers | 83.59 | 84.73 | +1.14 |
| wikics | 79.75 | 79.75 | +0.00 |

Six unchanged configurations reuse their V5 test values; the full configuration tuple is checked before reuse. Tolokers, Questions, ogbn-arxiv, Computers and WikiCS receive new test evaluations. Selection records are saved before test execution in `runs/advanced_v6/FROZEN_V6.json`.

Computers reuses locally cached official raw data: 13,752 nodes, 245,861 undirected edges, 767 features and 10 classes. Its masks follow the [GraphAny loader](https://github.com/DeepGraphLearning/GraphAny/blob/main/graphany/data.py): seed-42 stratification with 200 total training nodes, then 6,776 validation and 6,776 test nodes. These are not 20 nodes from every class; GraphAny allocates the total proportionally by stratification.

WikiCS has 11,701 nodes, 215,603 undirected edges, 300 features and 10 classes. Official mask column 0 gives 580 training, 1,769 validation and 5,847 test nodes. Stopping-mask nodes remain unlabeled; the graph is used transductively. This follows GraphAny's saved seed-42 mask convention, not an average over all 20 WikiCS splits.

| New graph | GraphAny-Wisconsin, published accuracy | EvoGFM V6, seed 42 |
| :--- | ---: | ---: |
| Computers | 82.00 ± 1.14 | 84.73 |
| WikiCS | 73.77 ± 0.83 | 79.75 |

Published references come from [GraphAny, ICLR 2025, Appendix D Table 5](https://arxiv.org/html/2405.20445#A4.T5). Its source training, target adaptation and averaging differ from ours. External models were not retrained. These are numerical references from the papers, not matched-budget significance claims. Old public tests had already been observed during development.

## Joint label and edge scarcity

Both constraints now apply simultaneously: five labeled training nodes per class and only 20% of graph edges. Cora/Citeseer/Roman/Amazon retain 35/30/90/25 training labels. The same official validation pool is additionally used to select each readout. Deleted edges are unavailable to the encoder and propagation. Each model uses the same seed-42 label and edge subsets.

| Graph | Original 0–2-hop readout | V5 enhanced | V6 advanced |
| :--- | ---: | ---: | ---: |
| cora | 38.20 | 44.50 | 44.50 |
| citeseer | 42.60 | 43.90 | 43.90 |
| roman-empire | 38.40 | 39.66 | 39.66 |
| amazon-ratings | 23.34 | 23.94 | 23.94 |

This custom condition measures robustness to combined scarcity; it has no directly quoted external baseline. V6 selects the exact V5 configuration on all four validation sets, so those two columns share predictions. V5/V6 are compared to the original readout as a controlled readout ablation. Absolute performance falls substantially under both constraints; this is harder than either separate V5 stress test.

## Whole-graph classification

PROTEINS and NCI1 use official TU data with one-hot node labels and no continuous node attributes. The frozen source encoder processes every graph; graph labels fit only the target SVM. Ten outer stratified folds and five inner folds use seed 42. All scaling is learned inside the corresponding training fold. The C grid is `{0.001,0.01,0.1,1,10,100,1000}`. The statistics-only control searches seven C values; the balanced full readout searches three block weights times seven C values. The ± values below are standard deviations across the ten outer folds, not across seeds.

| Dataset | Graphs | Statistics-only control | EvoGFM balanced pooling |
| :--- | ---: | ---: | ---: |
| PROTEINS | 1,113 | 77.26 ± 3.09 | 77.18 ± 2.61 |
| NCI1 | 4,110 | 72.00 ± 1.98 | 78.39 ± 2.03 |

PROTEINS's original pooled readout scored 75.65%; balancing raises it to 77.18%. The statistics-only control is retained. [GraphCL, NeurIPS 2020, Table 4](https://arxiv.org/html/2010.13902) reports PROTEINS 74.39 ± 0.45 and NCI1 77.87 ± 0.41. Its source training, outer splits and reported variation differ. These are published references; our per-fold values and selected parameters are preserved in `results_v6.json`.

## Retention and future transfer over four domains

The stream is Cora → Citeseer → Actor → Amazon-ratings; Minesweeper and Tolokers remain future domains. The inverse-calibration algorithm and original fixed NC readout are unchanged, keeping this experiment separate from the advanced public NC readout. `static` never updates; `cumulative` carries coefficients and accumulated curvature forward; `reset` restarts from the source state at every domain. All use the same current-domain calibration budget. Only stream training labels/edges update the model; future labels supply measurements.

NC stream scores use official validation masks and accuracy. Future binary NC scores use ROC-AUC. LP uses the native masked-edge validation protocol and ROC-AUC, not NCN Hits@100 or HeaRT MRR. Forgetting is the average post-learning peak minus final score over the first three stream domains. Future gains compare the final prefix to prefix zero.

| Task / control | Final stream average (%) | Forgetting (pp) | Future Minesweeper gain (pp) | Future Tolokers gain (pp) |
| :--- | ---: | ---: | ---: | ---: |
| NC static | 54.130 | 0.000 | +0.000 | +0.000 |
| NC cumulative | 53.907 | 0.708 | +0.005 | -0.258 |
| NC reset | 54.314 | 0.600 | +0.196 | -0.196 |
| LP static | 84.349 | 0.000 | +0.000 | +0.000 |
| LP cumulative | 84.585 | 0.000 | -0.100 | +0.066 |
| LP reset | 84.464 | 0.205 | -0.021 | -0.066 |

The full 5-by-4 score matrices and future-domain trajectories are in `results_v6.json`. This is an additional domain-order observation. Retention is measured separately from future transfer; small or mixed future gains do not establish stable knowledge accumulation.

## Rejected link trial

A content-plus-topology hard-negative pool gave Citeseer HeaRT validation MRR 26.64%, below the retained mixed topology/uniform pool's 28.65%. It was rejected without running the new candidate on test; its exploratory implementation was removed. The V5 HeaRT test remains 31.17 MRR and 51.87 Hits@10. Public random-negative LP remains Cora 86.35 and Citeseer 92.64 Hits@100.

## Commands

The measured checkpoint path is `runs/reference/best.pt`. Use `runs/source/best.pt` when reproducing from a fresh `train.py` run; source-training variation can change numerical results.

```bash
python prepare.py --datasets computers wikics
python evaluate.py --checkpoint runs/reference/best.pt --nc cora citeseer roman-empire amazon-ratings minesweeper tolokers questions actor ogbn-arxiv computers wikics --lp --node-readout advanced --nc-metric auto --output runs/v6

# Inspect validation before adding --split test.
python evaluate.py --checkpoint runs/reference/best.pt --nc cora citeseer roman-empire amazon-ratings --lp --node-readout advanced --shots 5 --edge-rate 0.2 --output runs/v6_joint

python classify_graphs.py --checkpoint runs/reference/best.pt --balanced-readout --output runs/v6_proteins
python classify_graphs.py --checkpoint runs/reference/best.pt --dataset NCI1 --balanced-readout --output runs/v6_nci1

python continual.py --checkpoint runs/reference/best.pt --stream cora citeseer actor amazon-ratings --future minesweeper tolokers --task both --nc-metric auto --output runs/v6_continual
```

V4 and V5 records remain in their original reports/JSON files. The supplied manuscript is unchanged.
