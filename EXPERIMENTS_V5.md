# V5: larger graphs and harder target adaptation

The same frozen source checkpoint improves on the larger ogbn-arxiv graph and on official HeaRT link ranking. This round covers nine node-classification graphs, the existing two public link-prediction graphs, official Citeseer HeaRT, and a new whole-graph classification task on PROTEINS.

All runs use seed 42 and four CPU threads. Source training is unchanged: Pubmed, Photo and Texas, 60 epochs, with epoch 21 selected by source validation. Actor, ogbn-arxiv and PROTEINS are evaluation targets, not new source-training graphs. Existing official raw caches were reused for ogbn-arxiv and PROTEINS.

These are development comparisons: earlier public tests were already observed. V5 configurations use validation selection before V5 test reporting. Published references were read from the original papers; external models were not retrained. Sources, training budgets and averaging differ.

## What changed

**Node readout.** `--node-readout enhanced` starts from V4's four fixed readouts, then considers a 2048-dimensional quadratic sketch and training-label diffusion. The quadratic sketch uses two independent fixed Gaussian projections, `phi(x) = (xW) * (xV) / sqrt(2048)`, after row normalization. Its expected inner product equals the degree-two kernel `(xᵀy)²`. A larger sketch can reduce approximation noise but adds memory and readout computation.

Diffusion starts from `softmax(8 * ridge_scores)` and iterates `p = (1-a)*prior + a*P@p` for 20 steps, clamping only training nodes to their known labels. `P` uses the same inverse-programmed neighbor attention as the encoder. Candidates use `a` in `{0.5, 0.9}`; isolated nodes keep their prior. An added component must improve validation by more than 0.1 percentage point. Otherwise the simpler selected readout stays. Multiclass selection uses accuracy; binary selection uses ROC-AUC. This is a restart label-propagation readout, related to the line of work represented by [Correct & Smooth, ICLR 2021](https://arxiv.org/html/2010.13993); it is not the exact Correct & Smooth algorithm.

**Hard link calibration.** `heart.py --negative-mode mixed` replaces up to half of the calibration negative pool with training-graph nonedges that have high resource-allocation scores. Up to 32 candidates per node are collected; both the hard and uniform portions are sampled using fixed independent generators, deduplicated and split into disjoint calibration train/holdout pair sets across all views. It still fits the original 18 residual coefficients. Public validation/test candidate pairs are never used for fitting. Uniform versus mixed mode is selected on official validation MRR; the default CLI remains uniform for backward compatibility.

The inverse map, Q/K construction and source model weights are unchanged.

## Node classification

Cora/Citeseer use Planetoid masks; heterophily graphs use official mask column 0. Actor uses the official `film_split_0.6_0.2_0.npz` masks (3,648 training nodes out of 7,600). ogbn-arxiv uses the official time split: 90,941 train, 29,799 validation and 48,603 test nodes. Edges are made undirected. Minesweeper, Tolokers and Questions report ROC-AUC; the remaining rows report accuracy. All values are percentages; changes are percentage points.

| Graph | V4 adaptive | V5 enhanced | Change |
| :--- | ---: | ---: | ---: |
| cora | 79.10 | 79.60 | +0.50 |
| citeseer | 69.80 | 69.70 | -0.10 |
| roman-empire | 73.42 | 74.37 | +0.95 |
| amazon-ratings | 44.06 | 47.20 | +3.14 |
| minesweeper | 88.82 | 89.16 | +0.34 |
| tolokers | 80.16 | 80.16 | +0.00 |
| questions | 75.95 | 75.95 | +0.00 |
| actor | 33.62 | 32.96 | -0.66 |
| ogbn-arxiv | 64.25 | 67.67 | +3.42 |

ogbn-arxiv has 169,343 nodes, 128 features, 40 classes and 1,157,799 undirected edges. Its validation accuracy improves from 65.79 to 69.35; test improves by 3.42 points. The selected readout uses quadratic2048 plus diffusion 0.5. Actor and Citeseer decline; the validation-selected configurations are retained rather than changed after seeing those tests.

For the GFM reference, [GraphAny, ICLR 2025, Appendix D Table 5](https://arxiv.org/html/2405.20445#A4.T5) reports the following accuracy values. We keep its Wisconsin-trained variant consistently across graphs, rather than selecting a variant per dataset.

| Graph | GraphAny-Wisconsin, published | EvoGFM V5, seed 42 |
| :--- | ---: | ---: |
| Cora | 77.82 ± 1.15 | 79.60 |
| Citeseer | 67.50 ± 0.44 | 69.70 |
| Roman-empire | 64.06 ± 0.78 | 74.37 |
| Amazon-ratings | 42.57 ± 0.34 | 47.20 |
| Actor | 29.51 ± 0.55 | 32.96 |
| ogbn-arxiv | 57.79 ± 0.56 | 67.67 |

These values exceed that published GFM reference numerically. Strong supervised models can still be better: the [original OGB paper, NeurIPS 2020](https://arxiv.org/html/2005.00687) reports GCN at 71.74 ± 0.29 on ogbn-arxiv. GraphAny's binary-graph accuracy values are not compared to our ROC-AUC values.

## Harder adaptation conditions

The following are our controlled stress tests, not published-baseline rows. Each V4/V5 pair uses exactly the same support/edge budget.

**Five training labels per class.** Samples come from official training pools: 35 Cora, 30 Citeseer, 90 Roman and 25 Amazon labels. Official validation labels are additionally used to select the readout. This is five-shot support with official validation selection, not five labels in total and not the GFMate one-shot protocol.

| Graph | V4 adaptive | V5 enhanced | Change |
| :--- | ---: | ---: | ---: |
| cora | 71.40 | 72.10 | +0.70 |
| citeseer | 64.50 | 65.10 | +0.60 |
| roman-empire | 42.18 | 42.18 | +0.00 |
| amazon-ratings | 21.00 | 21.00 | +0.00 |

**Only 20% of edges observed.** Training/validation/test nodes stay fixed; a seed-42 subset of undirected edges is visible. The encoder receives the known observation rate. The deleted edges are unavailable to propagation and diffusion.

| Graph | V4 adaptive | V5 enhanced | Change |
| :--- | ---: | ---: | ---: |
| cora | 59.30 | 61.50 | +2.20 |
| citeseer | 61.10 | 61.50 | +0.40 |
| roman-empire | 65.20 | 66.25 | +1.06 |
| amazon-ratings | 40.18 | 40.85 | +0.67 |

The sparse-edge condition improves on all four graphs. Five-shot improvements occur on Cora and Citeseer; the other two retain V4's selected readout and scores.

## Official HeaRT link prediction

The dataset, official positive splits and 500 fixed hard negatives per positive are identical to V4. There are 227 validation positives and 455 test positives. The final mixed sampler is selected by validation MRR, which improves from 24.12 to 28.65. Its validation Hits@10 declines from 48.46 to 46.70. Configuration selection is recorded in `runs/heart_mixed_v5_final/FROZEN_V5.json` before test execution.

| Citeseer HeaRT test | MRR (%) | Hits@10 (%) |
| :--- | ---: | ---: |
| EvoGFM V4, uniform calibration | 27.28 | 54.73 |
| EvoGFM V5, mixed calibration | 31.17 | 51.87 |
| NCN, published | 28.65 ± 1.21 | 53.41 ± 1.46 |
| NCNC, published | 24.10 ± 0.65 | 53.72 ± 0.97 |
| BUDDY, published | 22.84 ± 0.36 | 48.35 ± 1.18 |

References are from [HeaRT, NeurIPS 2023 Datasets and Benchmarks, Table 4](https://arxiv.org/html/2306.10453). V5 improves MRR by 3.89 points and exceeds the published NCN MRR mean; Hits@10 falls by 2.86 points and is below NCN. The random-negative public LP results remain Cora 86.35 and Citeseer 92.64 Hits@100. This round does not claim an improvement on those two rows.

## Whole-graph classification

`classify_graphs.py` evaluates 1,113 PROTEINS graphs. Each graph uses one-hot node labels, excluding continuous node attributes. The encoder is frozen. Zero-to-four-hop representations and Transformer outputs are pooled by mean, standard deviation and maximum, then concatenated with feature proportions, graph size and degree statistics. An RBF SVM uses nested stratified 10-fold evaluation and inner 5-fold selection over `C = {0.001, 0.01, 0.1, 1, 10, 100, 1000}`. Standardization is fitted inside each fold. The control uses only the simple graph statistics, with identical folds and tuning budget.

| PROTEINS | Accuracy (%) | Standard deviation across 10 outer folds |
| :--- | ---: | ---: |
| Simple graph statistics, our control | 77.26 | 3.09 |
| EvoGFM pooled representations + statistics | 75.65 | 3.15 |
| GraphCL, published reference | 74.39 | 0.45 (paper's reported variation) |

The GraphCL reference is from [Graph Contrastive Learning with Augmentations, NeurIPS 2020, Table 4](https://arxiv.org/html/2010.13902). Its [official evaluation](https://github.com/Shen-Lab/GraphCL/blob/master/unsupervised_TU/evaluate_embedding.py) uses outer 10-fold SVM evaluation with inner 5-fold C selection, but outer splits, source training, scaling and the meaning of reported variation differ. Treat it as a published reference, not an exact matched-split win. Our full representation is weaker than the statistics-only control; this first graph-classification experiment does not show an encoder advantage.

## Reproduce

Use the same source checkpoint for both columns. The local measured checkpoint is `runs/reference/best.pt`; if training a new checkpoint, replace that path with `runs/source/best.pt`. A fresh source-training run can differ numerically. Downloading is skipped where official raw files or prepared caches already exist.

```bash
python prepare.py --datasets actor ogbn-arxiv

# Validate first, then use --split test after choosing the configuration.
python evaluate.py --checkpoint runs/reference/best.pt --nc cora citeseer roman-empire amazon-ratings minesweeper tolokers questions actor ogbn-arxiv --lp --node-readout enhanced --nc-metric auto --output runs/v5

# Controlled harder settings; run each with --split test after validation.
python evaluate.py --checkpoint runs/reference/best.pt --nc cora citeseer roman-empire amazon-ratings --lp --node-readout enhanced --shots 5 --output runs/v5_five_shot
python evaluate.py --checkpoint runs/reference/best.pt --nc cora citeseer roman-empire amazon-ratings --lp --node-readout enhanced --edge-rate 0.2 --output runs/v5_sparse

# Official HeaRT files are installed as described in README.
python heart.py --checkpoint runs/reference/best.pt --negative-mode mixed --output runs/v5_heart
python heart.py --checkpoint runs/reference/best.pt --negative-mode mixed --split test --output runs/v5_heart

python classify_graphs.py --checkpoint runs/reference/best.pt --output runs/v5_proteins
```

Replace `enhanced` with `adaptive` and use a separate output folder to reproduce each V4 control. `results_v5.json` preserves all measured values, selected variants, PROTEINS folds and local evidence paths. `results.json` and `EXPERIMENTS.md` preserve V4, including inverse ablations and continual trajectories. Retention and accumulation have not been rerun with V5; earlier future-domain gains still change sign. The supplied manuscript remains unchanged.
