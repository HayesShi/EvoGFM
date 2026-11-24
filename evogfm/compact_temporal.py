"""Exact temporal history with bounded bootstrap workspace for large replays."""

import numpy as np
from numba import njit

from evogfm.temporal import TemporalHistory


@njit(cache=True)
def merge(
    older_keys,
    older_counts,
    older_last,
    newer_keys,
    newer_counts,
    newer_last,
    max_timestamp,
):
    old = new = overlap = 0
    while old < len(older_keys) and new < len(newer_keys):
        if older_keys[old] < newer_keys[new]:
            old += 1
        elif older_keys[old] > newer_keys[new]:
            new += 1
        else:
            overlap += 1
            old += 1
            new += 1

    total = len(older_keys) + len(newer_keys) - overlap
    keys = np.empty(total, dtype=np.int64)
    counts = np.empty(total, dtype=np.uint32)
    last = np.empty(total, dtype=np.float32)
    old = new = 0
    for offset in range(total):
        if new == len(newer_keys) or (
            old < len(older_keys) and older_keys[old] < newer_keys[new]
        ):
            keys[offset] = older_keys[old]
            counts[offset] = older_counts[old]
            last[offset] = older_last[old]
            old += 1
        else:
            key = newer_keys[new]
            count = newer_counts[new]
            timestamp = newer_last[new]
            if old < len(older_keys) and older_keys[old] == key:
                count += older_counts[old]
                if max_timestamp:
                    timestamp = max(timestamp, older_last[old])
                old += 1
            keys[offset] = key
            counts[offset] = count
            last[offset] = timestamp
            new += 1
    return keys, counts, last


class CompactTemporalHistory(TemporalHistory):
    """Keep sorted directed states in size tiers, with a small mutable buffer."""

    bootstrap_chunk_size = 4_000_000
    buffer_size = 65_536

    def __init__(self, n_nodes):
        super().__init__(n_nodes)
        self._levels = []
        self._buffer_events = 0
        self._event_count = 0

    @staticmethod
    def _add_level(levels, events, arrays, max_timestamp=False):
        while levels and levels[-1][0] <= events:
            previous_events, previous_arrays = levels.pop()
            arrays = merge(*previous_arrays, *arrays, max_timestamp)
            events += previous_events
        levels.append((events, arrays))

    def _flush(self):
        if self._buffer_events:
            keys = sorted(self.counts)
            counts = np.fromiter(
                map(self.counts.__getitem__, keys), np.uint32, len(keys)
            )
            last = np.fromiter(map(self.last.__getitem__, keys), np.float32, len(keys))
            arrays = np.array(keys, dtype=np.int64), counts, last
            self._add_level(self._levels, self._buffer_events, arrays)
            self.counts.clear()
            self.last.clear()
            self._buffer_events = 0

    def _add_events(self, count):
        if self._event_count + count > np.iinfo(np.uint32).max:
            raise OverflowError("Compact history supports up to 2**32 - 1 events")
        self._event_count += count

    def ingest(self, pairs, time):
        pairs = np.asarray(pairs, dtype=np.int64)
        count = pairs.shape[1]
        self._add_events(count)
        super().ingest(pairs, time)
        self._buffer_events += count
        if self._buffer_events >= self.buffer_size:
            self._flush()

    def ingest_native(self, events, origin):
        if not len(events):
            return
        self._add_events(len(events))
        use_buffer = len(events) < self.buffer_size
        batch_levels = []
        if not use_buffer:
            self._flush()
        chunk_size = self.buffer_size if use_buffer else self.bootstrap_chunk_size
        for beginning in range(0, len(events), chunk_size):
            batch = events[beginning : beginning + chunk_size]
            source, destination = batch[:, :2].T
            times = ((batch[:, 2] - origin) / 604800.0).astype(np.float32)
            directed = np.multiply(source, self.n, dtype=np.int64) + destination
            keys, inverse, counts = np.unique(
                directed, return_inverse=True, return_counts=True
            )
            last = np.full(len(keys), -np.inf, dtype=np.float32)
            np.maximum.at(last, inverse, times)
            if use_buffer:
                for key, count, timestamp in zip(
                    keys.tolist(), counts.tolist(), last.tolist()
                ):
                    self.counts[key] = self.counts.get(key, 0) + count
                    self.last[key] = timestamp
                self._buffer_events += len(batch)
            else:
                arrays = keys, counts.astype(np.uint32), last
                self._add_level(batch_levels, len(batch), arrays, max_timestamp=True)
            np.add.at(self.out_count, source, 1)
            np.add.at(self.in_count, destination, 1)
            np.maximum.at(self.last_node, source, times)
            np.maximum.at(self.last_node, destination, times)
        if batch_levels:
            # The native call takes a pair's maximum timestamp within this
            # batch, then replaces its previous last-occurrence timestamp.
            _, arrays = batch_levels.pop()
            while batch_levels:
                _, previous_arrays = batch_levels.pop()
                arrays = merge(*previous_arrays, *arrays, True)
            self._add_level(self._levels, len(events), arrays)
        if self._buffer_events >= self.buffer_size:
            self._flush()

    def lookup(self, keys):
        last = np.full(len(keys), -np.inf, dtype=np.float32)
        counts = np.zeros(len(keys), dtype=np.uint32)
        for _, (stored_keys, stored_counts, stored_last) in self._levels:
            positions = np.searchsorted(stored_keys, keys)
            valid = np.flatnonzero(positions < len(stored_keys))
            matches = valid[stored_keys[positions[valid]] == keys[valid]]
            found = positions[matches]
            last[matches] = stored_last[found]
            counts[matches] += stored_counts[found]
        if self.counts:
            for offset, key in enumerate(keys.tolist()):
                if key in self.counts:
                    last[offset] = self.last[key]
                    counts[offset] += self.counts[key]
        return last, counts.astype(np.float32)

    def state_bytes(self):
        external_payload = sum(
            array.nbytes
            for _, arrays in self._levels
            for array in arrays
            if not array.flags.owndata
        )
        return super().state_bytes() + external_payload
