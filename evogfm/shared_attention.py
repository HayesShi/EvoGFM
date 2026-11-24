"""Exact CPU demand serving with bounded, cross-layer shared intermediates.

The added cache has three protected partitions: 50% scalar edge scores, 25%
left/right frame projections, and 25% frozen first-layer values (including LN).
The first two select a fixed retention set from the *current* request. Equal
size score entries are ranked by known consumer count, with edge ID ties. No
future request is inspected. Projection and score admission use k(c-r)-w,
in scalar-operation units, separately by object type/size; this is a cost
proxy, not a calibrated latency estimate or a global optimality guarantee.

For the chosen score set S, cold score work is sum_e d_e minus
sum_{e in S}(d_e-1). It stays pinned through all current consumers. Planning,
CSR indices, scalar softmax scratch, and the inherited dense hidden/feature
backing are outside this *added shared-intermediate cache* budget and belong
in total RSS. There are no per-edge Python objects. Array payload and metadata
are charged; a conservative 4096-byte control/header allowance per partition
also covers its Python/NumPy wrappers. Invalid program caches are freed eagerly.
An optional int32 dense slot index is charged to its partition, including its
initialization. It is used only when 4*universe <= partition_budget/4. Edge
entries remain 13 bytes; K=floor((B_E-4096-index_bytes)/13). Their universe is
the current directed CSR adjacency nnz (the underlying undirected edges are
expanded in both directions; this backend introduces no backbone self-loops).
"""

from __future__ import annotations

import time
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from .demand_maintenance import DemandLinkScorer
from .link_inference import LinkScorer


class _Table:
    """Sorted fixed-size entries; no resizing or eviction while a request runs."""

    overhead = 4096

    def __init__(self, keys, width, roles=1, universe=0, prefix=False):
        self.keys = np.asarray(keys, dtype=np.int64).copy()
        self.data = np.empty((len(keys), roles, width), dtype=np.float32)
        self.valid = np.zeros((len(keys), roles), dtype=np.bool_)
        self.prefix = prefix
        self.slots = None
        if universe:
            self.slots = np.full(universe, -1, dtype=np.int32)
            self.slots[self.keys] = np.arange(len(keys), dtype=np.int32)

    @classmethod
    def capacity(cls, budget, width, roles=1, universe=0):
        return max(
            0, (budget - cls.overhead - 4 * universe) // (8 + roles * (4 * width + 1))
        )

    @staticmethod
    def mapped_universe(budget, universe):
        return universe if 4 * universe <= budget // 4 else 0

    @property
    def bytes(self):
        return (
            self.overhead
            + self.keys.nbytes
            + self.data.nbytes
            + self.valid.nbytes
            + (0 if self.slots is None else self.slots.nbytes)
        )

    @property
    def payload(self):
        return self.data.nbytes

    def locate(self, keys, role=0):
        if self.prefix:
            pos = keys
            present = keys < len(self.keys)
        elif self.slots is not None:
            pos = self.slots[keys]
            present = pos >= 0
        else:
            pos = np.searchsorted(self.keys, keys)
            present = pos < len(self.keys)
            if len(self.keys):
                present &= self.keys[np.minimum(pos, len(self.keys) - 1)] == keys
        hit = np.zeros(len(keys), dtype=bool)
        hit[present] = self.valid[pos[present], role]
        return pos, present, hit


class SharedAttentionLinkScorer(DemandLinkScorer):
    """Full frozen coupled/frame-aligned model; same public API as DemandLinkScorer."""

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
        if not model.frame_aligned or any(l.mode != "coupled" for l in model.layers):
            raise ValueError("shared attention requires coupled + frame_aligned")
        if model.training or any(l.training for l in model.layers):
            raise ValueError("shared serving requires frozen eval-mode weights")
        if raw.device.type != "cpu" or raw.dtype != torch.float32:
            raise ValueError("this exact serving backend requires CPU float32 inputs")
        if policy != "lazy":
            raise ValueError("shared serving currently uses the lazy demand policy")
        if tile_edges <= 0 or shared_budget_bytes < 0:
            raise ValueError("invalid tile size or budget")
        super().__init__(
            model,
            head,
            raw,
            edges,
            encoded=encoded,
            policy=policy,
            materialize_fraction=materialize_fraction,
        )
        self.shared_budget_bytes = int(shared_budget_bytes)
        self.tile_edges = int(tile_edges)
        self._edge_budget = self.shared_budget_bytes // 2
        self._projection_budget = self.shared_budget_bytes // 4
        self._value_budget = (
            self.shared_budget_bytes - self._edge_budget - self._projection_budget
        )
        self._edge_cache = self._projection_cache = self._value_cache = None
        self._shared_peak = 0
        self._request_unique = 0
        self._request_score_start = 0
        for name in (
            "shared_score_evaluations",
            "shared_score_hits",
            "shared_score_recomputations",
            "shared_projection_rows",
            "shared_projection_hits",
            "shared_value_hits",
            "shared_planned_score_consumptions",
            "shared_planned_unique_edges",
            "shared_protected_edges",
            "shared_admission_benefit_units",
            "shared_invalidated_bytes",
            "shared_requests",
            "shared_score_correction_edges",
        ):
            self.counters[name] = 0
        self.counters["shared_planning_s"] = 0.0
        self.counters["shared_skipped_plans"] = 0
        if self.shared_budget_bytes == 0:
            # The no-cache baseline does not construct an admission plan merely
            # to collect an oracle overlap count. -1 marks unavailable metrics.
            self.counters["shared_planned_unique_edges"] = -1
            self.counters["shared_score_recomputations"] = -1
        if model.layers:
            width = model.layers[0].heads * model.layers[0].rank
            capacity = min(len(raw), _Table.capacity(self._value_budget, width))
            if capacity:
                # Deterministic node-prefix retention is deliberately modest: it
                # needs no forecast, and survives graph changes with frozen z/W.
                self._value_cache = _Table(np.arange(capacity), width, prefix=True)
        self._record_peak()

    def _record_peak(self):
        current = sum(
            c.bytes
            for c in (self._edge_cache, self._projection_cache, self._value_cache)
            if c is not None
        )
        if current > self.shared_budget_bytes:
            raise RuntimeError("shared intermediate cache budget exceeded")
        self._shared_peak = max(self._shared_peak, current)

    def cache_stats(self):
        caches = [
            c
            for c in (self._edge_cache, self._projection_cache, self._value_cache)
            if c is not None
        ]
        resident = sum(c.bytes for c in caches)
        payload = sum(c.payload for c in caches)
        live = sum(int(c.valid.sum()) * c.data.shape[2] * 4 for c in caches)

        def index_bytes(cache):
            return 0 if cache is None or cache.slots is None else cache.slots.nbytes

        return dict(
            budget_bytes=self.shared_budget_bytes,
            resident_bytes=resident,
            peak_bytes=self._shared_peak,
            payload_bytes=payload,
            live_payload_bytes=live,
            unfilled_payload_bytes=payload - live,
            metadata_bytes=resident - payload,
            invalid_bytes=0,
            index_bytes=sum(index_bytes(c) for c in caches),
            edge_index_bytes=index_bytes(self._edge_cache),
            projection_index_bytes=index_bytes(self._projection_cache),
            fixed_value_index_bytes=index_bytes(self._value_cache),
            edge_bytes=0 if self._edge_cache is None else self._edge_cache.bytes,
            projection_bytes=(
                0 if self._projection_cache is None else self._projection_cache.bytes
            ),
            fixed_value_bytes=(
                0 if self._value_cache is None else self._value_cache.bytes
            ),
        )

    def _node_unique(self, values, inverse=False):
        """Bounded-universe indexing scratch, shared by both execution modes."""
        if len(values) < max(1024, len(self.raw) // 32):
            return (
                np.unique(values, return_inverse=True) if inverse else np.unique(values)
            )
        present = np.zeros(len(self.raw), dtype=np.bool_)
        present[values] = True
        nodes = np.flatnonzero(present)
        if not inverse:
            return nodes
        slots = np.empty(len(self.raw), dtype=np.int32)
        slots[nodes] = np.arange(len(nodes), dtype=np.int32)
        return nodes, slots[values]

    def _neighbors(self, nodes):
        return self._node_unique(self.a[nodes].indices)

    def _union(self, left, right):
        return self._node_unique(np.concatenate((left, right)))

    def _plan(self, nodes):
        started = time.perf_counter()
        # Retiring the preceding fixed retention set before allocating the next
        # prevents a transient double-budget allocation.
        self._edge_cache = self._projection_cache = None
        consumers, left_consumers, right_consumers = [], [], []
        w2 = self._missing("w2", nodes)
        w1 = self._union(nodes, self._neighbors(w2)) if len(w2) else nodes
        weighted = self._missing("weight", self._union(self._missing("w1", w1), w2))
        if len(weighted):
            consumers.append(weighted)
            left_consumers.append(weighted)
            right_consumers.append(self._neighbors(weighted))
        current = self._missing("h", nodes)
        if self.model.config.get("joint_task_backbone", False):
            for level in range(len(self.model.layers), 0, -1):
                targets = current[self.hidden_stamps[level][current] != self.version]
                if not len(targets):
                    break
                consumers.append(targets)
                left_consumers.append(targets)
                sources = self._neighbors(targets)
                right_consumers.append(sources)
                current = self._union(targets, sources)
        counts = np.zeros(len(self.raw), dtype=np.int32)
        for rows in consumers:
            counts[rows] += 1
        active = np.flatnonzero(counts)
        consumption = counts[active]
        degree = self.a.indptr[active + 1].astype(np.int64) - self.a.indptr[active]
        self._request_unique = int(degree.sum())
        total = int(degree @ consumption)
        self._request_score_start = self.counters["shared_score_evaluations"]
        self.counters["shared_planned_score_consumptions"] += total
        self.counters["shared_planned_unique_edges"] += self._request_unique
        rank = len(self.program[1])
        mapped = _Table.mapped_universe(self._edge_budget, self.a.nnz)
        capacity = _Table.capacity(self._edge_budget, 1, universe=mapped)
        # c=2r, r_read=1, w_write=1: costs exclude projection work,
        # preventing double-credit with the projection partition.
        benefit = (consumption.astype(np.int64) - 1) * (2 * rank - 1) - 1
        rows = np.flatnonzero((consumption > 1) & (benefit > 0) & (degree > 0))
        if capacity and len(rows):
            order = np.lexsort((active[rows], -consumption[rows]))
            rows = rows[order]
            ends = np.cumsum(degree[rows])
            count = min(capacity, int(ends[-1]))
            take = int(np.searchsorted(ends, count, side="left")) + 1
            rows = rows[:take]
            lengths = degree[rows].copy()
            lengths[-1] -= int(ends[take - 1]) - count
            offsets = np.r_[0, np.cumsum(lengths[:-1])]
            keys = np.repeat(
                self.a.indptr[active[rows]] - offsets, lengths
            ) + np.arange(count)
            keys.sort()
            self._edge_cache = _Table(keys, 1, universe=mapped)
            self.counters["shared_protected_edges"] += count
            self.counters["shared_admission_benefit_units"] += int(
                lengths @ benefit[rows]
            )
        mapped = _Table.mapped_universe(self._projection_budget, len(self.raw))
        capacity = _Table.capacity(
            self._projection_budget, rank, roles=2, universe=mapped
        )
        # Each role has c=2*d*r, read=r, write=r scalar cost units. These
        # are nominal role-consumption counts; score hits can remove a later
        # projection consumer. Thus projection benefit remains a cost proxy,
        # while the protected scalar-score work identity is exact.
        pbenefit = np.zeros(len(self.raw), dtype=np.int64)
        for groups in (left_consumers, right_consumers):
            if groups:
                counts.fill(0)
                for keys in groups:
                    counts[keys] += 1
                keys = np.flatnonzero(counts)
                pbenefit[keys] += (counts[keys] - 1) * (
                    2 * self.z.shape[1] * rank - rank
                ) - rank
        eligible = np.flatnonzero(pbenefit > 0)
        if capacity and len(eligible):
            order = eligible[np.lexsort((eligible, -pbenefit[eligible]))[:capacity]]
            keys = np.sort(order)
            self._projection_cache = _Table(keys, rank, roles=2, universe=mapped)
            self.counters["shared_admission_benefit_units"] += int(
                pbenefit[order].sum()
            )
        self.counters["shared_requests"] += 1
        self.counters["shared_planning_s"] += time.perf_counter() - started
        self._record_peak()

    def _project(self, nodes, role):
        cache = self._projection_cache
        output = np.empty((len(nodes), len(self.program[1])), dtype=np.float32)
        if cache is None:
            missing = np.arange(len(nodes))
        else:
            pos, present, hit = cache.locate(nodes, role)
            output[hit] = cache.data[pos[hit], role]
            self.counters["shared_projection_hits"] += int(hit.sum())
            missing = np.flatnonzero(~hit)
        matrix = self.program[0 if role == 0 else 2]
        # Node projection gathers are bounded, even when a hub exposes the graph.
        for start in range(0, len(missing), self.tile_edges):
            selected = missing[start : start + self.tile_edges]
            values = (self.z[torch.from_numpy(nodes[selected])] @ matrix).numpy()
            output[selected] = values
            if cache is not None:
                keep = present[selected]
                cache.data[pos[selected[keep]], role] = values[keep]
                cache.valid[pos[selected[keep]], role] = True
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
        degrees = np.diff(csr.indptr)
        targets = np.repeat(np.arange(len(nodes)), degrees)
        result = np.empty(csr.nnz, dtype=np.float32)
        cache = self._edge_cache
        if cache is None:
            missing = np.arange(csr.nnz)
        else:
            keys = np.repeat(
                self.a.indptr[nodes] - csr.indptr[:-1], degrees
            ) + np.arange(csr.nnz)
            pos, present, hit = cache.locate(keys)
            result[hit] = cache.data[pos[hit], 0, 0]
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
            values = self._base_dot_products(q, k, qi, ki, effective)
            result[missing] = values
            if cache is not None:
                keep = present[missing]
                cache.data[pos[missing[keep]], 0, 0] = values[keep]
                cache.valid[pos[missing[keep]], 0] = True
        self.counters["shared_score_evaluations"] += len(missing)
        if self.shared_budget_bytes == 0:
            self.counters["shared_planned_score_consumptions"] += csr.nnz
        return result, targets

    def _base_dot_products(self, q, k, qi, ki, effective):
        values = np.empty(len(qi), dtype=np.float32)
        for start in range(0, len(qi), self.tile_edges):
            stop = start + self.tile_edges
            values[start:stop] = (
                (q[qi[start:stop]] * k[ki[start:stop]]) * effective
            ).sum(1)
        return values

    @staticmethod
    def _softmax(base, temperature, csr, targets):
        # Match the released FP32 edge-order reduction. NumPy reduceat uses a
        # different summation tree; on high-degree real graph rows that shifts
        # normalization enough to breach the frozen predictor's tolerance.
        score = torch.from_numpy(base) * float(temperature)
        target = torch.from_numpy(targets.astype(np.int64, copy=False))
        maximum = score.new_full((csr.shape[0],), -torch.inf)
        maximum.scatter_reduce_(0, target, score, reduce="amax", include_self=True)
        weight = (score - maximum[target]).exp()
        denominator = score.new_zeros(csr.shape[0])
        denominator.index_add_(0, target, weight)
        weight /= denominator[target].clamp_min(1e-12)
        return weight.numpy()

    def _ensure_weight(self, nodes):
        missing = self._missing("weight", nodes)
        if not len(missing):
            return
        csr = self.a[missing]
        base, target = self._base_scores(missing, csr)
        amplitude = self.program[1].numpy()
        delta = amplitude - np.sign(amplitude) * np.maximum(np.abs(amplitude), 1e-20)
        if np.any(delta):
            # Weighted propagation uses a, whereas backbone uses a_eff.
            q = self._project(missing, 0)
            sources, reverse = self._node_unique(csr.indices, inverse=True)
            k = self._project(sources, 1)
            for start in range(0, len(base), self.tile_edges):
                stop = start + self.tile_edges
                base[start:stop] += (
                    (q[target[start:stop]] * k[reverse[start:stop]]) * delta
                ).sum(1)
            self.counters["shared_score_correction_edges"] += len(base)
        temperature = float(
            F.softplus(self.model.layers[0].log_temperature).mean() + 0.05
        )
        values = self._softmax(base, temperature, csr, target)
        norm = np.zeros(len(missing), dtype=np.float32)
        nonempty = np.flatnonzero(np.diff(csr.indptr))
        norm[nonempty] = np.sqrt(np.add.reduceat(values * values, csr.indptr[nonempty]))
        self.norm[missing] = norm
        self.norm_tensor[torch.from_numpy(missing)] = torch.from_numpy(norm)
        ids = np.repeat(
            self.a.indptr[missing] - csr.indptr[:-1], np.diff(csr.indptr)
        ) + np.arange(csr.nnz)
        self.weighted.data[ids] = values
        self.stamps["weight"][missing] = self.version
        self.counters["weight_rows"] += len(missing)
        self.counters["weight_edges"] += csr.nnz
        self.counters["weight_support_rows"] += len(self._node_unique(csr.indices))

    def _values(self, layer, level, support):
        width = layer.heads * layer.rank
        result = self.raw.new_empty((len(support), width))
        cache = self._value_cache if level == 1 else None
        if cache is None:
            missing = np.arange(len(support))
        else:
            pos, present, hit = cache.locate(support)
            result[torch.from_numpy(np.flatnonzero(hit))] = torch.from_numpy(
                cache.data[pos[hit], 0]
            )
            self.counters["shared_value_hits"] += int(hit.sum())
            missing = np.flatnonzero(~hit)
        for start in range(0, len(missing), self.tile_edges):
            selected = missing[start : start + self.tile_edges]
            values = layer.value(
                layer.norm1(self.hidden[level - 1][torch.from_numpy(support[selected])])
            )
            result[torch.from_numpy(selected)] = values
            if cache is not None:
                keep = present[selected]
                cache.data[pos[selected[keep]], 0] = values.numpy()[keep]
                cache.valid[pos[selected[keep]], 0] = True
        self.counters["backbone_value_projection_rows"] += len(missing)
        return result.reshape(len(support), layer.heads, layer.rank)

    def _ensure_hidden(self, level, nodes):
        if level == 0:
            return
        missing = nodes[self.hidden_stamps[level][nodes] != self.version]
        if not len(missing):
            return
        csr = self.a[missing]
        support = self._union(missing, csr.indices)
        self._ensure_hidden(level - 1, support)
        self._compute_hidden_rows(level, missing, support, csr)

    def _compute_hidden_rows(self, level, missing, support, csr):
        layer = self.model.layers[level - 1]
        values = self._values(layer, level, support)
        base, targets = self._base_scores(missing, csr)
        temperatures = (F.softplus(layer.log_temperature) + 0.05).numpy()
        if len(csr.indices) >= max(1024, len(self.raw) // 32):
            slots = np.empty(len(self.raw), dtype=np.int32)
            slots[support] = np.arange(len(support), dtype=np.int32)
            columns = slots[csr.indices]
        else:
            columns = np.searchsorted(support, csr.indices)
        out = self.raw.new_empty((len(missing), layer.heads, layer.rank))
        for head, temperature in enumerate(temperatures):
            weight = self._softmax(base, temperature, csr, targets)
            matrix = sp.csr_matrix(
                (weight, columns, csr.indptr), shape=(len(missing), len(support))
            )
            # CSR spMM never creates an E-by-value-dimension neighbor gather.
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
        # One logical base-score consumption, with heads fused; actual dot
        # computations (including weighted consumers) are in shared_score_*.
        self.counters["backbone_edge_score_evaluations"] += csr.nnz

    @torch.no_grad()
    def features(self, pairs):
        nodes = np.unique(pairs.numpy())
        names = ("w1", "w2", "u1", "u2")
        if self.model.config.get("joint_task_backbone", False):
            names += ("h",)
        warm = all(not len(self._missing(name, nodes)) for name in names)
        if warm:
            self.counters["shared_skipped_plans"] += 1
            self.counters["shared_requests"] += 1
            return LinkScorer.features(self, pairs)
        planned = self.shared_budget_bytes != 0
        if planned:
            self._plan(nodes)
        else:
            self.counters["shared_skipped_plans"] += 1
            self.counters["shared_requests"] += 1
        self._ensure(nodes)
        if planned:
            evaluations = (
                self.counters["shared_score_evaluations"] - self._request_score_start
            )
            self.counters["shared_score_recomputations"] += max(
                0, evaluations - self._request_unique
            )
        return LinkScorer.features(self, pairs)

    @torch.no_grad()
    def update(self, edges, added, removed=None):
        changed = added.shape[1] or (removed is not None and removed.shape[1])
        if changed:
            self.counters["shared_invalidated_bytes"] += sum(
                c.bytes
                for c in (self._edge_cache, self._projection_cache)
                if c is not None
            )
            self._edge_cache = self._projection_cache = None
        return super().update(edges, added, removed)
