"""Fixed-width, history-only temporal calibration channel replacements."""

from __future__ import annotations

import torch
import numpy as np
import sys

from .temporal import replace_temporal


TRANSFORMS = ("source_conditional_frequency", "target_corrected_recency")


class PastLocalGraph:
    """Incremental simple undirected adjacency, excluding self loops.

    A snapshot is ingested only after all its queries have been scored. Query
    roles and labels are not inputs; duplicate candidate rows are preserved.
    """

    def __init__(self, n_nodes):
        self.neighbors = [set() for _ in range(n_nodes)]

    def ingest(self, pairs):
        for source, target in np.asarray(pairs).T.tolist():
            if source != target:
                self.neighbors[source].add(target)
                self.neighbors[target].add(source)

    def statistics(self, pairs):
        """log1p(CN), log1p(AA), log1p(RA), Jaccard, log1p(deg product).

        AA uses 1/log(max(2,degree)), the existing frozen pair-channel rule;
        this also keeps native self-candidates finite at degree-one neighbors.
        All intersections and weighted sums are exact, without sketching.
        """
        pairs = np.asarray(pairs)
        n = len(self.neighbors)
        keys = np.minimum(pairs[0], pairs[1]) * n + np.maximum(pairs[0], pairs[1])
        unique, inverse = np.unique(keys, return_inverse=True)
        degree = np.asarray([len(row) for row in self.neighbors], dtype=np.float64)
        inv_log = 1 / np.log(np.maximum(2, degree))
        inv_degree = 1 / np.maximum(1, degree)
        values = np.empty((len(unique), 5), dtype=np.float32)
        for index, key in enumerate(unique.tolist()):
            source, target = divmod(key, n)
            common = sorted(self.neighbors[source].intersection(self.neighbors[target]))
            cn = len(common)
            union = degree[source] + degree[target] - cn
            aa = sum(inv_log[node] for node in common)
            ra = sum(inv_degree[node] for node in common)
            values[index] = (
                np.log1p(cn),
                np.log1p(aa),
                np.log1p(ra),
                cn / union if union else 0,
                np.log1p(degree[source] * degree[target]),
            )
        return torch.from_numpy(values[inverse])

    def state_bytes(self):
        """Recursive Python storage estimate including actual set/int objects."""
        seen = set()

        def size(value):
            if id(value) in seen:
                return 0
            seen.add(id(value))
            total = sys.getsizeof(value)
            if isinstance(value, (list, tuple, set)):
                total += sum(size(item) for item in value)
            return total

        return sys.getsizeof(self) + sys.getsizeof(vars(self)) + size(self.neighbors)


def local_calibration_channels(base, temporal_statistics, local_statistics, half_life):
    """Preserve original first ten channels and replace exactly the final seven."""
    if base.shape != (len(temporal_statistics), 17) or local_statistics.shape != (
        len(base),
        5,
    ):
        raise ValueError("Expected 17 base channels and five exact local statistics")
    output = base.clone()
    output[:, 10] = temporal_statistics[:, 1]
    output[:, 11] = torch.exp2(-temporal_statistics[:, 0] / half_life)
    output[:, 12:] = local_statistics
    return output


def calibration_channels(base, statistics, transform, half_life):
    """Return 17 channels without changing the frozen source-head inputs.

    statistics is the existing past-only seven-column TemporalHistory output:
    age, log1p(pair count), reverse age, log1p(reverse count), log1p(source
    out-count), log1p(target in-count), and degree-normalized pair count.
    Neither labels, candidate roles nor candidate-pool statistics are inputs.
    """
    if base.ndim != 2 or base.shape[1] != 17:
        raise ValueError("Calibration requires the existing 17-channel frame")
    if statistics.shape != (len(base), 7):
        raise ValueError("Expected seven history statistics per candidate")
    if half_life <= 0:
        raise ValueError("half_life must be positive")
    output = replace_temporal(base, statistics, "recency_frequency", half_life)
    if transform == "source_conditional_frequency":
        frequency = statistics[:, 1].expm1()
        source_count = statistics[:, 4].expm1().clamp_min(1)
        output[:, 1] = torch.log1p(frequency / source_count)
    elif transform == "target_corrected_recency":
        # sqrt(1 + in_count) = exp(log1p(in_count) / 2), avoiding a
        # count reconstruction and retaining exact zero for unseen pairs.
        output[:, 0] *= torch.exp(-0.5 * statistics[:, 5])
    else:
        raise ValueError(f"Unknown temporal calibration transform: {transform}")
    return output


def selection_grid(reference_half_life):
    """Eight new configurations per dataset, fixed before validation scoring."""
    half_lives = (1.0, 4.0) if reference_half_life <= 4 else (4.0, 16.0)
    return [
        {"transform": transform, "half_life": half_life, "ridge": ridge}
        for transform in TRANSFORMS
        for half_life in half_lives
        for ridge in (1.0, 10.0)
    ]


def select_operating_point(trials, reference):
    """Maximize validation AUC without reducing the reference validation MRR.

    Include the reference itself so a failed hypothesis leaves the locked
    operating point unchanged. All trial metrics remain available for audit.
    """
    candidates = [reference, *trials]
    eligible = [
        row
        for row in candidates
        if row["validation"]["mrr"] >= reference["validation"]["mrr"]
    ]
    selected = max(
        eligible, key=lambda row: (row["validation"]["auc"], row["validation"]["mrr"])
    )
    best_auc = max(
        candidates, key=lambda row: (row["validation"]["auc"], row["validation"]["mrr"])
    )
    frontier = [
        row
        for row in candidates
        if not any(
            other["validation"]["auc"] >= row["validation"]["auc"]
            and other["validation"]["mrr"] >= row["validation"]["mrr"]
            and (
                other["validation"]["auc"] > row["validation"]["auc"]
                or other["validation"]["mrr"] > row["validation"]["mrr"]
            )
            for other in candidates
        )
    ]
    return {"selected": selected, "best_auc": best_auc, "pareto_frontier": frontier}
