#!/usr/bin/env bash
set -eu

project=$(cd "$1" && pwd)
results=$(cd "$2" && pwd)
runtime="${VLDB_PYTHON:-python3}"
shift 2
if [[ $# -eq 0 ]]; then
    set -- bitcoin-alpha bitcoin-otc uci
fi
export PYTHONPATH="$project:$project/runs/official_a/decoupled_build${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
for graph; do
    base="$results/quality/decoupled/$graph"
    if [[ -f "$base/representation/result.json" ]]; then
        continue
    fi
    storage=()
    if [[ "$graph" == reddit-title ]]; then
        storage+=(--compressed)
    fi
    if nice -n 8 "$runtime" -m experiments.db_a_decoupled_causal propagate \
        --chronology "$base/candidates/chronology.npz" --output "$base/representation" "${storage[@]}" \
        > "$base/representation.log" 2>&1; then
        echo "$graph complete"
    else
        echo "$graph failed, see $base/representation.log" >&2
    fi
done
