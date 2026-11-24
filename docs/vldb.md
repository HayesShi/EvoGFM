# Incremental Predictive Analytics over Evolving Graphs

## Method

The executor serves a frozen graph-conditioned link predictor against a specified graph version. It maintains the edge moment by adding and subtracting the outer products of edited edges. The updated moment, edge count and graph density define the small symmetric spectral input. The executor recomputes the spectral factors on that input and invalidates versioned projections and scores on graph changes.

At a fixed version, the structural logit factors into a shared scalar (b_g(i,j)=z_i^	op P_gz_j) and a head-specific temperature. The executor can reuse this scalar across consumers of the same directed edge. Each head still normalizes over its own full current neighborhood, and each layer aggregates its own values. Cache admission accounts for the reuse opportunity and cost under a byte budget. The budget applies to retained shared intermediates. GPU peak allocation and process memory are measured independently.

The algebraic comparison assumes the same selected spectral program, version, frozen inputs and query neighborhood. Truncation through a repeated spectral magnitude can make the selected subspace depend on the eigensolver. Floating-point reorderings can also change probabilities and ranking. The verification command reports tolerance violations and rank changes separately.

| Component | Entry point |
| --- | --- |
| Exact graph moments and versioned demand | `evogfm/incremental.py`, `evogfm/demand_maintenance.py` |
| Shared scores, projections, cache and CUDA execution | `evogfm/shared_attention*.py` |
| Same-predictor execution baselines | `evogfm/execution_baselines.py`, `evogfm/strict_dependency.py` |
| Trace and prediction protocols | `experiments/db_a_causal_trace.py`, `experiments/db_a_quality_extended.py` |
| Budget, overlap, layer-depth and sensitivity sweeps | `experiments/db_a_matrix.py`, `experiments/db_a_sensitivity.py` |

## Setup

Run from the repository root. Install the appropriate CUDA build of PyTorch for GPU replay and install the Python package.

```sh
python -m pip install -e ".[benchmarks,dev]"
curl -fL https://raw.githubusercontent.com/HayesShi/EvoGFM/9ced8199f221521d20ceab168974803542dc4574/checkpoints/reference.pt -o checkpoints/reference.pt
```

The public source checkpoint is an input. Use `VLDB_CHECKPOINT` to point at a different compatible source checkpoint. Run `python -m pytest -q` for the CPU regression suite.

The optional author baselines are installed separately under `baselines/`. The NeutronRT adapter uses the released dependency planner on the common executor and requires a compatible DGL installation. DecoupledDGNN needs its released propagation extension for the native protocol. Follow the upstream build instructions.

| Folder | Upstream repository | Revision |
| --- | --- | --- |
| `NeutronRT` | https://github.com/iDC-NEU/NeutronRT | `52f7b6271c66aa17906f7e9e6559c25e22e575a4` |
| `UnderGS` | https://github.com/dani0108/UnderGS | `b87c16f5a5364ec2cec48bb18eca3228edc311c3` |
| `DecoupledDGNN` | https://github.com/zheng-yp/DecoupledDGNN | `3b96b9c509169fc1a7a258a4a52c33d9fdefdba2` |

## Causal replay

For example, download the [Wiki-Talk release](https://snap.stanford.edu/data/wiki-talk-temporal.html) to `data/wiki-talk-temporal.txt.gz`, then use the following commands. Freeze the validation-selected readout for the two executors being compared. Readouts saved with scikit-learn 1.4.2 should be loaded with that version.

```sh
python -m experiments.db_a_data --dataset wiki-talk --source data/wiki-talk-temporal.txt.gz --output runs/vldb/data/wiki-talk
python -m experiments.db_a_quality_extended prepare --dataset wiki-talk --root runs/vldb --output runs/vldb/candidates
python -m experiments.db_a_quality_extended fit --method ours --candidates runs/vldb/candidates --output runs/vldb/readout
python -m experiments.db_a_causal_trace --dataset wiki-talk --source runs/vldb/data/wiki-talk/manifest.json --readout runs/vldb/readout/selected_readout.joblib --output runs/vldb/traces/wiki-talk/causal_test
python -m experiments.db_a_cuda run --trace runs/vldb/traces/wiki-talk/causal_test/trace.json --output runs/vldb/replay --methods head_fused ours --host-buffers --compact --sorted-ingest --resident-scores --direct-hops --async-indices --gpu-propagation
```

Compare matching scores with:

```sh
python -m experiments.db_a_matrix verify --trace runs/vldb/traces/wiki-talk/causal_test/trace.json --actual runs/vldb/replay/ours --reference runs/vldb/replay/head_fused
```

The controlled workload entry points take `--root runs/vldb` and use the frozen readout at `runs/vldb/readout/selected_readout.joblib` by default. Override it with `VLDB_READOUT` when reusing a different compatible readout. Run `python -m experiments.db_a_queue --root runs/vldb` to execute prepared jobs. The queue, trace files, model weights, plots and result logs stay in local ignored directories.
