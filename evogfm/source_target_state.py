"""Exact compact directed event statistics for the direction-A stream.

Sorted numeric arrays replace per-edge Python dictionaries. This changes the
storage layout, not the recurrence, activity, or recency features.
"""

from __future__ import annotations

import numpy as np
import torch


class CompactTemporalHistory:
    def __init__(self, n_nodes):
        self.n = n_nodes
        self.keys = np.empty(0, dtype=np.int64)
        self.counts = np.empty(0, dtype=np.uint32)
        self.last = np.empty(0, dtype=np.int32)
        self.out_count = np.zeros(n_nodes, dtype=np.float32)
        self.in_count = np.zeros(n_nodes, dtype=np.float32)

    def ingest(self, pairs, time):
        source, destination = np.asarray(pairs)
        keys, counts = np.unique(source * self.n + destination, return_counts=True)
        positions = np.searchsorted(self.keys, keys)
        present = positions < len(self.keys)
        if len(self.keys):
            present &= self.keys[np.minimum(positions, len(self.keys) - 1)] == keys
        if present.any():
            self.counts[positions[present]] += counts[present].astype(np.uint32)
            self.last[positions[present]] = time
        if (~present).any():
            new_keys = keys[~present]
            # Merge unique sorted arrays in numeric storage. Ingest time is
            # included in every method's end-to-end stream transition cost.
            all_keys = np.union1d(self.keys, new_keys)
            all_counts = np.empty(len(all_keys), dtype=np.uint32)
            all_last = np.empty(len(all_keys), dtype=np.int32)
            old_positions = np.searchsorted(all_keys, self.keys)
            new_positions = np.searchsorted(all_keys, new_keys)
            all_counts[old_positions], all_last[old_positions] = self.counts, self.last
            all_counts[new_positions], all_last[new_positions] = counts[~present], time
            self.keys, self.counts, self.last = all_keys, all_counts, all_last
        # Bincount includes every repeated event; graph simple-edge ingestion is
        # a separate operation and must not remove these event counts.
        self.out_count += np.bincount(source, minlength=self.n).astype(np.float32)
        self.in_count += np.bincount(destination, minlength=self.n).astype(np.float32)

    def _lookup(self, keys, time):
        positions = np.searchsorted(self.keys, keys)
        found = positions < len(self.keys)
        if len(self.keys):
            found &= self.keys[np.minimum(positions, len(self.keys) - 1)] == keys
        frequency = np.zeros(len(keys), dtype=np.float32)
        age = np.full(len(keys), np.inf, dtype=np.float32)
        frequency[found] = self.counts[positions[found]]
        age[found] = time - self.last[positions[found]]
        return age, frequency

    def statistics(self, pairs, time):
        source, destination = np.asarray(pairs)
        age, frequency = self._lookup(source * self.n + destination, time)
        reverse_age, reverse = self._lookup(destination * self.n + source, time)
        normalized = frequency / np.sqrt(
            np.maximum(1, self.out_count[source] * self.in_count[destination])
        )
        return torch.from_numpy(
            np.stack(
                (
                    age,
                    np.log1p(frequency),
                    reverse_age,
                    np.log1p(reverse),
                    np.log1p(self.out_count[source]),
                    np.log1p(self.in_count[destination]),
                    normalized,
                ),
                axis=1,
            )
        )

    def state_bytes(self):
        return sum(
            value.nbytes
            for value in vars(self).values()
            if isinstance(value, np.ndarray)
        )
