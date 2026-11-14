# Baseline results and airport diagnosis

The [editable result tables](results/node_classification.md) combine
[seed-level measurements](results/node_classification.csv) with the author's
confirmed [reported values](results/value_audit.csv). The reported values take
precedence when rendering tables; archived measurements remain unchanged.
For archived measurements, per-graph sample SD is computed across three support
seeds; overall SD is computed after averaging ten graphs within each seed.
The source checkpoint is fixed across these seeds.

## Airport input diagnosis

The airport datasets have no native attributes. Their original identity inputs
are reduced to 100 dimensions in EvoGFM's relation branch. All identity singular
values are tied, so this reduction selects an arbitrary subspace. In the saved
CPU run, 31/131 Brazil nodes, 299/399 EU nodes and 1090/1190 US nodes have zero
projected rows. The full identity input is retained in the propagation branch.

The shared-input control narrows the gap. With the same degree64 encoding
supplied to all methods, Brazil EvoGFM reaches 65.38% at 5-shot and 69.23% at
20-shot. GraphPFN reaches 75.64% and 79.49%, respectively: a 10.26 pp gap remains
at both budgets. The original identity results remain in the main comparison.

Two fixed, label-free controls are included in the
[airport input measurements](results/airport_inputs.csv):

- `degree64`: one-hot encoding of `min(degree, 63)`, previously measured and
  reproduced in this extension.
- `degree_quantile64`: map each degree to its empirical mid-rank, then to one
  of 64 bins. Equal degrees share a bin. This is a new diagnostic encoding.

EvoGFM, GraphAny, GraphPFN and SCGFM use both encodings on all three airport
graphs, three support seeds and three label budgets. The quantile encoding does not improve EvoGFM consistently across
graphs. These controls identify sensitivity to featureless-graph inputs; they
do not replace the main protocol or establish an overall superiority claim.

`FeatureRidge` uses only original row-normalized features and fixed ridge 0.1.
Its complete ten-graph results are included as a simple representation control.
It uses no source training and no graph propagation.

## Baseline coverage

| Method | Matched NC coverage | Implementation and remaining work |
|---|---|---|
| GraphAny | Ten graphs × three budgets × three seeds | Official Wisconsin checkpoint; 90 episodes. |
| GraphGlue | Ten graphs × three budgets × three seeds | Archived official-adapter measurements; 90 episodes. |
| GraphPFN | Three airport graphs × three budgets × three seeds | Official GraphPFN1.3/LimiX-16M, full ten-member ensemble. Non-airport evaluation is incomplete: full runs on Physics, Deezer and LastFM hit the 2 GB WSL limit; no ten-graph aggregate is reported. |
| SCGFM | Ten graphs × three budgets × three seeds | New official-model reproduction: 5391 Photo+Computers source ego-graphs, all 60 epochs, and 90 full-test episodes. |
| FeatureRidge | Ten graphs × three budgets × three seeds | New feature-only control; 90 episodes. |
| R-GFM | Pending | Public code requires a separate CUDA/custom-extension setup and source training; no pretrained checkpoint is provided. The released target-training protocol still needs a matched cross-graph adaptation. |
| PALP | Pending | Official checkpoints use SBERT features and precomputed BUDDY edge features. The current support/query split needs matching structural preprocessing before evaluation. |
| GRAVER / SCR / GAUGE | Pending identification | A verified paper-to-official-code match is still needed before these names can be treated as runnable baselines. |

The SCGFM source seed is 42; target seeds vary only support samples. Its official
encoder and support-fitted StandardScaler/ProtoClassifier are retained. Target
ego-graphs use PPR with alpha 0.15 and size 100; the full test split is scored.

SCGFM's complete ten-graph means are 40.36 / 50.42 / 56.85% at 1/5/20-shot.
It leads the currently completed methods at 1-shot. EvoGFM's corresponding
reported means are 37.09 / 54.79 / 62.24%, with higher 5-shot and 20-shot means.

## Reported values

The nine author-confirmed cells are retained in
[value_audit.csv](results/value_audit.csv), alongside the archived-run values.
EvoGFM's reported 5-shot overall mean is **54.79**; its corresponding SD remains
pending recalculation. The OGBN-Arxiv 5-shot entry reports **29.64** without an
SD, and the 20-shot entry reports **42.91 ± 0.46**.

The LP table entries likewise retain the author's confirmed version. Curve
plots continue to use archived per-seed predictions; those records are separate
from the reported table values. The archived LP measurements are in
[link_prediction.csv](results/link_prediction.csv).

## Reproduce

See [baseline commands and environments](reproduction.md#additional-baseline-controls).
To rebuild the NC tables while retaining the author-confirmed values:

```bash
python scripts/summarize_node_results.py --output runs/summary
```

The generated Markdown applies the reported values and marks the best means in
bold and the second-best means with underlining. Generated statistical CSVs
summarize the archived measurements without changing them.

The source paths for imported measurements and the official baseline revisions
are recorded in [sources.json](results/sources.json). Large datasets, external
weights, per-node predictions and generated caches stay outside the release tree.
