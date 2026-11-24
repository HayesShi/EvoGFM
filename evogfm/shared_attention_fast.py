"""Whole-universe shared-cache fast path with the frozen v3 fallback.

Dense allocation is selected only when both full caches fit the added-cache
budget: 4096 + 5*D bytes for lazy scalar scores and 4096 + N*(9+4*H*r)
bytes for fixed first-layer values. D is the directed CSR adjacency nnz.
Scores use the CSR edge position as an implicit key; values retain v3's
explicit int64 node keys, validity byte, and conservative header allowance.
No frame projection is retained. Allocating these buffers does not evaluate
any score or value: only current demand fills entries.

Scores survive requests within a graph version. Frozen first-layer values
survive graph versions. A topology update releases all old program-dependent
caches before selecting the next mode. If the graph outgrows the budget, full
values are released before allocating the unchanged v3 protected partitions.
All allocation/index initialization belongs to constructor/update service work.
The zero-budget baseline is exactly the inherited v3 execution path.

The dense-mode score work identity is F_E = |union of demanded directed edges|
per graph version. This is an exact-work statement, not a latency guarantee.
Total RSS still includes the inherited model/graph/hidden buffers and operator
scratch outside the added shared-cache budget.
"""

from __future__ import annotations

import time
import numpy as np
import torch

from .shared_attention import SharedAttentionLinkScorer, _Table
from .incremental import numeric_payload_bytes
from .link_inference import LinkScorer


class _DenseScores:
    overhead = 4096

    def __init__(self, count):
        self.data = np.empty((count, 1, 1), dtype=np.float32)
        self.valid = np.zeros((count, 1), dtype=np.bool_)
        self.slots = None

    @property
    def bytes(self):
        return self.overhead + self.data.nbytes + self.valid.nbytes

    @property
    def payload(self):
        return self.data.nbytes

    def locate(self, keys, role=0):
        # Every graph edge has exactly one slot; the position is its identity.
        return keys, np.ones(len(keys), dtype=np.bool_), self.valid[keys, role]


class SharedAttentionFastLinkScorer(SharedAttentionLinkScorer):
    """Same constructor/API as v3, with an automatically selected dense mode."""

    @torch.no_grad()
    def __init__(
        self,
        model,
        head,
        raw,
        edges,
        *,
        encoded=None,
        shared_budget_bytes=64 * 1024**2,
        tile_edges=1024,
        policy="lazy",
        materialize_fraction=0.35,
    ):
        super().__init__(
            model,
            head,
            raw,
            edges,
            encoded=encoded,
            shared_budget_bytes=shared_budget_bytes,
            tile_edges=tile_edges,
            policy=policy,
            materialize_fraction=materialize_fraction,
        )
        self._dense_mode = False
        self.counters["shared_dense_requests"] = 0
        self.counters["shared_dense_versions"] = 0
        self.counters["shared_mode_changes"] = 0
        self.counters["shared_mode_selection_s"] = 0.0
        self._select_mode()

    def _dense_required_bytes(self):
        width = self.model.layers[0].heads * self.model.layers[0].rank
        return (
            _DenseScores.overhead
            + 5 * self.a.nnz
            + _Table.overhead
            + len(self.raw) * (9 + 4 * width)
        )

    def _select_mode(self):
        started = time.perf_counter()
        fit = bool(
            self.shared_budget_bytes
            and self.model.layers
            and self._dense_required_bytes() <= self.shared_budget_bytes
        )
        changed = fit != self._dense_mode
        # Old graph-score and projection buffers must be retired first, even
        # when switching from a request-protected plan to dense allocation.
        self._edge_cache = self._projection_cache = None
        width = self.model.layers[0].heads * self.model.layers[0].rank
        if fit:
            if self._value_cache is None or len(self._value_cache.keys) != len(
                self.raw
            ):
                self._value_cache = None
                self._value_cache = _Table(np.arange(len(self.raw)), width, prefix=True)
            self._edge_cache = _DenseScores(self.a.nnz)
            self.counters["shared_dense_versions"] += 1
        elif self.shared_budget_bytes:
            capacity = min(len(self.raw), _Table.capacity(self._value_budget, width))
            if (
                self._value_cache is not None
                and len(self._value_cache.keys) != capacity
            ):
                self._value_cache = None
            if capacity and self._value_cache is None:
                self._value_cache = _Table(np.arange(capacity), width, prefix=True)
        self._dense_mode = fit
        self.counters["shared_mode_changes"] += int(changed)
        self.counters["shared_mode_selection_s"] += time.perf_counter() - started
        self._record_peak()

    def cache_stats(self):
        result = super().cache_stats()
        result.update(
            dense_mode=self._dense_mode,
            dense_required_bytes=self._dense_required_bytes(),
            directed_csr_edges=self.a.nnz,
        )
        return result

    def cache_bytes(self):
        # numeric_payload_bytes does not introspect arbitrary cache classes;
        # supply their array dictionaries explicitly and deduplicate storage.
        state = {
            key: value
            for key, value in vars(self).items()
            if key not in ("model", "head")
        }
        state["shared_cache_numeric"] = [
            vars(c)
            for c in (self._edge_cache, self._projection_cache, self._value_cache)
            if c is not None
        ]
        return numeric_payload_bytes(state)

    def _base_scores(self, nodes, csr):
        if not self._dense_mode:
            return super()._base_scores(nodes, csr)
        before = self.counters["shared_score_evaluations"]
        result = super()._base_scores(nodes, csr)
        self.counters["shared_planned_score_consumptions"] += csr.nnz
        self.counters["shared_planned_unique_edges"] += (
            self.counters["shared_score_evaluations"] - before
        )
        return result

    @torch.no_grad()
    def features(self, pairs):
        if not self._dense_mode:
            return super().features(pairs)
        nodes = np.unique(pairs.numpy())
        self.counters["shared_requests"] += 1
        self.counters["shared_dense_requests"] += 1
        self.counters["shared_skipped_plans"] += 1
        self._ensure(nodes)
        return LinkScorer.features(self, pairs)

    @torch.no_grad()
    def update(self, edges, added, removed=None):
        started = time.perf_counter()
        changed = added.shape[1] or (removed is not None and removed.shape[1])
        # v3.update retires old program caches before changing the graph. It
        # preserves fixed V; _select_mode then keeps or releases that allocation.
        result = super().update(edges, added, removed)
        if changed:
            self._select_mode()
        result["update_s"] = time.perf_counter() - started
        return result
