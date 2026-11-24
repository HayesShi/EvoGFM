"""Exact execution controls for the frozen, frame-aligned attention predictor.

All controls use the same scalar softmax, CSR aggregation and edge tile as the
shared executor. Demand-Only recomputes the structural dot product per head;
Head-Fused computes it once per layer. FullGraph-Refresh eagerly evaluates all
rows and rebuilds edge moments at every changed graph version. Shared-LRU uses
the shared operators with standard object recency instead of protected demand
sets. A full-universe cache has no eviction decisions, so its LRU path is the
same dense layout as the proposed executor.
"""

from __future__ import annotations

import time
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from .incremental import FullEdgeMoments
from .link_inference import LinkScorer
from .shared_attention import SharedAttentionLinkScorer, _Table
from .shared_attention_fast import SharedAttentionFastLinkScorer, _DenseScores


class DemandOnlyLinkScorer(SharedAttentionLinkScorer):
    """Same demand plan and kernels, without structural sharing among heads."""

    def __init__(self, *args, **kwargs):
        kwargs["shared_budget_bytes"] = 0
        super().__init__(*args, **kwargs)

    def _ensure_hidden(self, level, nodes):
        if level == 0:
            return
        missing = nodes[self.hidden_stamps[level][nodes] != self.version]
        if not len(missing):
            return
        csr = self.a[missing]
        support = self._union(missing, csr.indices)
        self._ensure_hidden(level - 1, support)
        layer = self.model.layers[level - 1]
        values = self._values(layer, level, support)
        temperatures = (F.softplus(layer.log_temperature) + 0.05).numpy()
        if len(csr.indices) >= max(1024, len(self.raw) // 32):
            slots = np.empty(len(self.raw), dtype=np.int32)
            slots[support] = np.arange(len(support), dtype=np.int32)
            columns = slots[csr.indices]
        else:
            columns = np.searchsorted(support, csr.indices)
        out = self.raw.new_empty((len(missing), layer.heads, layer.rank))
        for head, temperature in enumerate(temperatures):
            # The only difference from Head-Fused is this call inside the loop.
            base, targets = self._base_scores(missing, csr)
            weight = self._softmax(base, temperature, csr, targets)
            matrix = sp.csr_matrix(
                (weight, columns, csr.indptr), shape=(len(missing), len(support))
            )
            out[:, head] = torch.from_numpy(
                matrix @ values[:, head].contiguous().numpy()
            )
        result = self.hidden[level - 1][torch.from_numpy(missing)] + layer.output(
            out.reshape(len(missing), -1)
        )
        for start in range(0, len(missing), self.tile_edges):
            block = result[start : start + self.tile_edges]
            block.add_(layer.ff(layer.norm2(block)))
        self.hidden[level][torch.from_numpy(missing)] = result
        self.hidden_stamps[level][missing] = self.version
        self.counters["backbone_rows"] += len(missing)
        self.counters["backbone_support_rows"] += len(support)
        self.counters["backbone_edges"] += csr.nnz
        self.counters["backbone_edge_score_evaluations"] += layer.heads * csr.nnz


class FullGraphRefreshLinkScorer(DemandOnlyLinkScorer):
    """One eager full materialization per graph version, including version zero."""

    @torch.no_grad()
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counters["moment_rebuilds"] = 1
        self._refresh_all()

    def _refresh_all(self):
        self._ensure(np.arange(len(self.raw), dtype=np.int64))
        self.materialized_version = self.version
        self.counters["materializations"] += 1

    @torch.no_grad()
    def update(self, edges, added, removed=None):
        started = time.perf_counter()
        changed = added.shape[1] or (removed is not None and removed.shape[1])
        if not changed:
            self.counters["topology_noops"] += 1
            return dict(
                update_s=time.perf_counter() - started,
                noop=1,
                version=self.version,
                refreshed=0,
                affected_fraction=0.0,
            )
        self.moments = FullEdgeMoments.build(self.z, edges)
        self.state = self.moments.state()
        self.version += 1
        self.valid_w2_rows = 0
        self._index(edges)
        self._program()
        self._refresh_all()
        self.counters["moment_rebuilds"] += 1
        self.counters["topology_updates"] += 1
        return dict(
            update_s=time.perf_counter() - started,
            noop=0,
            version=self.version,
            refreshed=1,
            affected_fraction=1.0,
        )


class _LRUTable:
    """Vectorized batch get/put with exact LRU object order.

    Every get touches resident objects in supplied order. A following put admits
    computed objects in supplied order. Keys, validity bits, recency timestamps,
    payload and the sorted lookup permutation are all charged. Roles share one
    node object, as in the protected projection table. Sorting scratch is transient
    operator scratch, included in process RSS rather than retained-cache bytes.
    """

    overhead = 4096

    def __init__(self, budget, width, roles=1, universe=None):
        per_entry = 8 + roles * (4 * width + 1) + 8 + 4
        count = max(0, (int(budget) - self.overhead) // per_entry)
        if universe is not None:
            count = min(count, universe)
        self.keys = np.full(count, -1, dtype=np.int64)
        self.data = np.empty((count, roles, width), dtype=np.float32)
        self.valid = np.zeros((count, roles), dtype=np.bool_)
        self.recency = np.zeros(count, dtype=np.uint64)
        self.order = np.arange(count, dtype=np.int32)
        self.clock = 0
        self.slots = None
        self.evictions = 0

    @property
    def bytes(self):
        return self.overhead + sum(
            a.nbytes
            for a in (self.keys, self.data, self.valid, self.recency, self.order)
        )

    @property
    def payload(self):
        return self.data.nbytes

    def _locate(self, keys):
        if not len(self.keys):
            return np.zeros(len(keys), dtype=np.int32), np.zeros(len(keys), dtype=bool)
        rank = np.searchsorted(self.keys, keys, sorter=self.order)
        pos = self.order[np.minimum(rank, len(self.keys) - 1)]
        present = (rank < len(self.keys)) & (self.keys[pos] == keys)
        return pos, present

    def _touch(self, pos):
        self.recency[pos] = np.arange(
            self.clock + 1, self.clock + len(pos) + 1, dtype=np.uint64
        )
        self.clock += len(pos)

    def get(self, keys, role=0):
        pos, present = self._locate(keys)
        hit = present.copy()
        hit[present] &= self.valid[pos[present], role]
        values = self._read(pos[hit], role)
        self._touch(pos[present])
        return hit, values

    def _read(self, slots, role):
        return self.data[slots, role].copy()

    def _write(self, slots, role, values):
        self.data[slots, role] = values

    def put(self, keys, values, role=0):
        if not len(self.keys) or not len(keys):
            return
        # A batch is an ordered sequence of object writes. Only its last K
        # distinct objects can survive when it is larger than the capacity.
        keys, values = keys[-len(self.keys) :], values[-len(self.keys) :]
        pos, present = self._locate(keys)
        count = int((~present).sum())
        if count:
            priority = self.recency.copy()
            priority[pos[present]] = np.iinfo(np.uint64).max
            victims = (
                np.arange(len(self.keys))
                if count == len(self.keys)
                else np.argpartition(priority, count - 1)[:count]
            )
            self.evictions += int((self.keys[victims] >= 0).sum())
            self.keys[victims] = keys[~present]
            self.valid[victims] = False
            pos[~present] = victims
            self.order[:] = np.argsort(self.keys, kind="stable")
        self._write(pos, role, values)
        self.valid[pos, role] = True
        self._touch(pos)


class SharedLRULinkScorer(SharedAttentionFastLinkScorer):
    """Identical shared kernels and budgets; replace retention by object LRU."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # No demand-wide overlap enumeration is performed by the online LRU.
        self.counters["shared_planned_unique_edges"] = -1
        self.counters["shared_score_recomputations"] = -1
        for name in ("score", "projection", "value"):
            self.counters[f"lru_{name}_evictions"] = 0

    def _put(self, cache, keys, values, role=0, name="score"):
        before = cache.evictions
        cache.put(keys, values, role)
        self.counters[f"lru_{name}_evictions"] += cache.evictions - before

    def _select_mode(self):
        started = time.perf_counter()
        fit = bool(
            self.shared_budget_bytes
            and self.model.layers
            and self._dense_required_bytes() <= self.shared_budget_bytes
        )
        changed = fit != self._dense_mode
        self._edge_cache = self._projection_cache = None
        width = self.model.layers[0].heads * self.model.layers[0].rank
        if fit:
            if not isinstance(self._value_cache, _Table) or len(
                self._value_cache.keys
            ) != len(self.raw):
                self._value_cache = None
                self._value_cache = _Table(np.arange(len(self.raw)), width, prefix=True)
            self._edge_cache = _DenseScores(self.a.nnz)
            self.counters["shared_dense_versions"] += 1
        elif self.shared_budget_bytes:
            if not isinstance(self._value_cache, _LRUTable):
                self._value_cache = None
                self._value_cache = self._make_lru(
                    self._value_budget, width, universe=len(self.raw)
                )
            self._edge_cache = self._make_lru(self._edge_budget, 1, universe=self.a.nnz)
            self._projection_cache = self._make_lru(
                self._projection_budget,
                len(self.program[1]),
                roles=2,
                universe=len(self.raw),
            )
        self._dense_mode = fit
        self.counters["shared_mode_changes"] += int(changed)
        self.counters["shared_mode_selection_s"] += time.perf_counter() - started
        self._record_peak()

    @staticmethod
    def _make_lru(budget, width, roles=1, universe=None):
        if budget < _LRUTable.overhead + 8 + roles * (4 * width + 1) + 12:
            return None
        return _LRUTable(budget, width, roles, universe)

    def cache_stats(self):
        result = super().cache_stats()
        result["retention_policy"] = "lru"
        result["lru_resident_evictions"] = sum(
            c.evictions
            for c in (self._edge_cache, self._projection_cache, self._value_cache)
            if isinstance(c, _LRUTable)
        )
        return result

    def _project(self, nodes, role):
        if self._dense_mode:
            return super()._project(nodes, role)
        cache = self._projection_cache
        output = np.empty((len(nodes), len(self.program[1])), dtype=np.float32)
        hit = np.zeros(len(nodes), dtype=bool)
        if cache is not None:
            hit, values = cache.get(nodes, role)
            output[hit] = values
            self.counters["shared_projection_hits"] += int(hit.sum())
        missing = np.flatnonzero(~hit)
        matrix = self.program[0 if role == 0 else 2]
        for start in range(0, len(missing), self.tile_edges):
            selected = missing[start : start + self.tile_edges]
            output[selected] = (
                self.z[torch.from_numpy(nodes[selected])] @ matrix
            ).numpy()
        if cache is not None:
            self._put(cache, nodes[missing], output[missing], role, name="projection")
        self.counters["shared_projection_rows"] += len(missing)
        self.counters[
            (
                "backbone_query_projection_rows"
                if role == 0
                else "backbone_key_projection_rows"
            )
        ] += len(missing)
        return output

    def _base_scores(self, nodes, csr):
        if self._dense_mode:
            result = super()._base_scores(nodes, csr)
            self.counters["shared_planned_unique_edges"] = -1
            return result
        degrees = np.diff(csr.indptr)
        targets = np.repeat(np.arange(len(nodes)), degrees)
        keys = np.repeat(self.a.indptr[nodes] - csr.indptr[:-1], degrees) + np.arange(
            csr.nnz
        )
        result = np.empty(csr.nnz, dtype=np.float32)
        cache = self._edge_cache
        hit = np.zeros(csr.nnz, dtype=bool)
        if cache is not None:
            hit, values = cache.get(keys)
            result[hit] = values[:, 0]
            self.counters["shared_score_hits"] += int(hit.sum())
        missing = np.flatnonzero(~hit)
        if len(missing):
            requested_rows = targets[missing]
            starts = np.r_[True, requested_rows[1:] != requested_rows[:-1]]
            qnodes = nodes[requested_rows[starts]]
            qi = np.cumsum(starts) - 1
            knodes, ki = self._node_unique(csr.indices[missing], inverse=True)
            q, k = self._project(qnodes, 0), self._project(knodes, 1)
            amplitude = self.program[1].numpy()
            effective = np.sign(amplitude) * np.maximum(np.abs(amplitude), 1e-20)
            result[missing] = self._base_dot_products(q, k, qi, ki, effective)
            if cache is not None:
                self._put(cache, keys[missing], result[missing, None])
        self.counters["shared_score_evaluations"] += len(missing)
        self.counters["shared_planned_score_consumptions"] += csr.nnz
        return result, targets

    def _values(self, layer, level, support):
        if self._dense_mode:
            return super()._values(layer, level, support)
        width = layer.heads * layer.rank
        result = self.raw.new_empty((len(support), width))
        cache = self._value_cache if level == 1 else None
        hit = np.zeros(len(support), dtype=bool)
        if cache is not None:
            hit, values = cache.get(support)
            result[torch.from_numpy(np.flatnonzero(hit))] = torch.from_numpy(values)
            self.counters["shared_value_hits"] += int(hit.sum())
        missing = np.flatnonzero(~hit)
        for start in range(0, len(missing), self.tile_edges):
            selected = missing[start : start + self.tile_edges]
            result[torch.from_numpy(selected)] = layer.value(
                layer.norm1(self.hidden[level - 1][torch.from_numpy(support[selected])])
            )
        if cache is not None:
            self._put(
                cache,
                support[missing],
                result[torch.from_numpy(missing)].numpy(),
                name="value",
            )
        self.counters["backbone_value_projection_rows"] += len(missing)
        return result.reshape(len(support), layer.heads, layer.rank)

    @torch.no_grad()
    def features(self, pairs):
        if self._dense_mode:
            return super().features(pairs)
        self.counters["shared_requests"] += 1
        self.counters["shared_skipped_plans"] += 1
        self._ensure(np.unique(pairs.numpy()))
        return LinkScorer.features(self, pairs)
