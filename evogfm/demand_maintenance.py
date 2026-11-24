"""Versioned, exact demand evaluation of the full released link feature path.

Fixed node inputs and fixed weights; all graph attention neighbors are retained.
The cache uses dense backing arrays in this first prototype. It reduces evaluated
rows, not the asymptotic allocated node-state memory.
"""

from __future__ import annotations

import time
import math

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from .incremental import FullEdgeMoments, MaintainedLinkScorer, numeric_payload_bytes
from .link_inference import LinkScorer
from .features import attention


class DemandLinkScorer(LinkScorer):
    """Lazy exact evaluation, with optional coverage-based full materialization."""

    @torch.no_grad()
    def __init__(
        self,
        model,
        head,
        raw,
        edges,
        *,
        encoded=None,
        policy="lazy",
        materialize_fraction=0.35,
    ):
        if policy not in ("lazy", "hybrid", "closure"):
            raise ValueError(policy)
        self.model, self.head, self.raw = model, head, raw
        self.ablation, self.dense = "full", None
        self.policy = policy
        self.materialize_fraction = materialize_fraction
        self.version = 0
        self.valid_w2_rows = 0
        self.materialized_version = -1
        self.z = model.encode_features(raw) if encoded is None else encoded
        self.moments = FullEdgeMoments.build(self.z, edges)
        self.state = self.moments.state()
        n, d = raw.shape
        self.parts = [
            F.normalize(raw, dim=1),
            torch.empty_like(raw),
            torch.empty_like(raw),
        ]
        self.content0 = self.parts[0]
        self.content1, self.content2 = torch.empty_like(raw), torch.empty_like(raw)
        self.weighted1 = torch.empty_like(raw)
        hidden = model.config["hidden"]
        self.hidden = [model.up(self.z)] + [
            raw.new_empty(n, hidden) for _ in model.layers
        ]
        self.h = raw.new_empty(n, hidden)
        self.norm_tensor = raw.new_empty(n)
        self.stamps = {
            name: np.full(n, -1, dtype=np.int64)
            for name in ("weight", "w1", "w2", "u1", "u2", "h")
        }
        self.hidden_stamps = [np.zeros(n, dtype=np.int64)] + [
            np.full(n, -1, dtype=np.int64) for _ in model.layers
        ]
        self._index(edges)
        self._program()
        self.counters = dict(
            weight_rows=0,
            weighted1_rows=0,
            weighted2_rows=0,
            uniform1_rows=0,
            uniform2_rows=0,
            backbone_rows=0,
            backbone_support_rows=0,
            backbone_query_projection_rows=0,
            backbone_key_projection_rows=0,
            backbone_value_projection_rows=0,
            backbone_edges=0,
            backbone_edge_score_evaluations=0,
            weight_edges=0,
            weight_support_rows=0,
            topology_updates=0,
            topology_noops=0,
            materializations=0,
            scheduling_s=0.0,
        )

    def _index(self, edges):
        # Exactly the same graph indexing and uniform normalization as eager.
        MaintainedLinkScorer._index(self, edges)
        self.edges = edges.clone()
        self.weighted = self.a.copy()
        self.weighted.data[:] = 0
        self.norm = np.empty(len(self.raw), dtype=np.float32)

    def _program(self):
        self.program = self.model.attention_program(self.state)
        _, amplitude, _, _ = self.model.programmed(self.state)
        self.amplitude_norm = amplitude.square().sum().sqrt().clamp_min(1e-6)

    def _missing(self, name, nodes):
        return nodes[self.stamps[name][nodes] != self.version]

    def _neighbors(self, nodes):
        return np.unique(self.a[nodes].indices)

    def _ensure_weight(self, nodes):
        missing = self._missing("weight", nodes)
        if not len(missing):
            return
        rows = self.a[missing].tocoo()
        source = torch.from_numpy(rows.col.astype(np.int64, copy=False))
        left, amplitude, right, _ = self.program
        q = self.z[torch.from_numpy(missing)] @ left
        unique, reverse = torch.unique(source, return_inverse=True)
        k = self.z[unique] @ right
        # Match the released full-attention chunk size. Only scalar edge
        # weights persist; the edge-by-program-rank gathers stay bounded.
        batch_size = 32768
        temperature = F.softplus(self.model.layers[0].log_temperature).mean() + 0.05
        group = torch.from_numpy(rows.row).long()
        maximum = q.new_full((len(missing),), -torch.inf)

        def scores(start, stop):
            return ((q[group[start:stop]] * k[reverse[start:stop]]) * amplitude).sum(
                1
            ) * temperature

        for start in range(0, len(group), batch_size):
            stop = min(start + batch_size, len(group))
            maximum.scatter_reduce_(
                0,
                group[start:stop],
                scores(start, stop),
                reduce="amax",
                include_self=True,
            )
        value = q.new_empty(len(group))
        denominator = q.new_zeros(len(missing))
        for start in range(0, len(group), batch_size):
            stop = min(start + batch_size, len(group))
            value[start:stop] = (scores(start, stop) - maximum[group[start:stop]]).exp()
            denominator.index_add_(0, group[start:stop], value[start:stop])
        norm = q.new_zeros(len(missing))
        for start in range(0, len(group), batch_size):
            stop = min(start + batch_size, len(group))
            value[start:stop] /= denominator[group[start:stop]].clamp_min(1e-12)
            norm.index_add_(0, group[start:stop], value[start:stop].square())
        norm.sqrt_()
        self.norm[missing] = norm.numpy()
        self.norm_tensor[torch.from_numpy(missing)] = norm
        value = value.numpy()
        cursor = 0
        for row in missing:
            lo, hi = self.weighted.indptr[row : row + 2]
            size = hi - lo
            self.weighted.data[lo:hi] = value[cursor : cursor + size]
            cursor += size
        self.stamps["weight"][missing] = self.version
        self.counters["weight_rows"] += len(missing)
        self.counters["weight_support_rows"] += len(unique)
        self.counters["weight_edges"] += len(group)

    def _ensure_w1(self, nodes):
        missing = self._missing("w1", nodes)
        if not len(missing):
            return
        self._ensure_weight(missing)
        value = self._propagate_rows(
            self.weighted, missing, self.raw, keep=self.weighted1
        )
        index = torch.from_numpy(missing)
        self.weighted1[index] = value
        self.parts[1][index] = F.normalize(value, dim=1)
        self.stamps["w1"][missing] = self.version
        self.counters["weighted1_rows"] += len(missing)

    def _ensure_w2(self, nodes):
        missing = self._missing("w2", nodes)
        if not len(missing):
            return
        self._ensure_weight(missing)
        neighbors = self._neighbors(missing)
        self._ensure_w1(neighbors)
        value = self._propagate_rows(
            self.weighted, missing, self.weighted1, normalize=True
        )
        self.parts[2][torch.from_numpy(missing)] = value
        self.stamps["w2"][missing] = self.version
        self.valid_w2_rows += len(missing)
        self.counters["weighted2_rows"] += len(missing)

    def _ensure_u1(self, nodes):
        missing = self._missing("u1", nodes)
        if not len(missing):
            return
        value = self._propagate_rows(
            self.uniform, missing, self.raw, normalize=True, keep=self.content1
        )
        self.content1[torch.from_numpy(missing)] = value
        self.stamps["u1"][missing] = self.version
        self.counters["uniform1_rows"] += len(missing)

    def _ensure_u2(self, nodes):
        missing = self._missing("u2", nodes)
        if not len(missing):
            return
        self._ensure_u1(np.union1d(missing, self._neighbors(missing)))
        value = self._propagate_rows(
            self.uniform, missing, self.content1, normalize=True
        )
        self.content2[torch.from_numpy(missing)] = value
        self.stamps["u2"][missing] = self.version
        self.counters["uniform2_rows"] += len(missing)

    def _propagate_rows(self, matrix, rows, source, *, normalize=False, keep=None):
        value = torch.from_numpy(matrix[rows] @ source.numpy())
        return F.normalize(value, dim=1) if normalize else value

    def _ensure_hidden(self, level, nodes):
        if level == 0:
            return
        missing = nodes[self.hidden_stamps[level][nodes] != self.version]
        if not len(missing):
            return
        row = self.a[missing].tocoo()
        support = np.union1d(missing, row.col)
        self._ensure_hidden(level - 1, support)
        index = torch.from_numpy(support)
        visible = torch.from_numpy(
            np.stack(
                (
                    np.searchsorted(support, missing[row.row]),
                    np.searchsorted(support, row.col),
                )
            )
        ).long()
        selected = torch.from_numpy(np.searchsorted(support, missing))
        output = self._forward_rows(
            self.model.layers[level - 1],
            self.hidden[level - 1][index],
            self.z[index],
            selected,
            torch.from_numpy(row.row).long(),
            visible[1],
        )
        self.hidden[level][torch.from_numpy(missing)] = output
        self.hidden_stamps[level][missing] = self.version
        self.counters["backbone_rows"] += len(missing)
        self.counters["backbone_support_rows"] += len(support)

    def _forward_rows(
        self, layer, h, frame, selected, target, source, batch_size=32768
    ):
        """Evaluate full-neighbor attention, output and FF only for target rows.

        Support rows contribute keys/values; their outputs are not dependencies
        of this layer's requested rows. All weights are the original layer's.
        """
        if layer.training:
            raise ValueError("Demand serving requires eval mode")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        x = layer.norm1(h)
        n, m = len(h), len(selected)
        if layer.mode == "vanilla":
            q = layer.q(x[selected]).reshape(m, layer.heads, layer.rank)
            k = layer.k(x).reshape(n, layer.heads, layer.rank)
        else:
            left, amplitude, right, _ = self.program
            rank = len(amplitude)
            scale = rank**0.25
            root = amplitude.abs().clamp_min(1e-20).sqrt()
            wq, wk = (
                scale * left * root[None],
                scale * right * (amplitude.sign() * root)[None],
            )
            if self.model.frame_aligned:
                temperature = (F.softplus(layer.log_temperature) + 0.05).sqrt()[
                    None, :, None
                ]
                q = (frame[selected] @ wq)[:, None] * temperature
                k = (frame @ wk)[:, None] * temperature
            else:
                q = torch.stack([p(x[selected]) @ wq for p in layer.inputs], dim=1)
                k = torch.stack([p(x) @ wk for p in layer.inputs], dim=1)
        v = layer.value(x).reshape(n, layer.heads, layer.rank)
        if layer.mode == "hybrid":
            base_q = layer.q(x[selected]).reshape(m, layer.heads, layer.rank)
            base_k = layer.k(x).reshape(n, layer.heads, layer.rank)

        def scores(a, b):
            score = (q[a] * k[b]).sum(-1) / math.sqrt(q.shape[-1])
            if layer.mode == "hybrid":
                score = (base_q[a] * base_k[b]).sum(-1) / math.sqrt(
                    layer.rank
                ) + layer.program_gate.tanh() * score
            return score

        # Three full-neighbor passes preserve the one global softmax for each
        # target/head without retaining E-by-head scores or E-by-head-by-rank
        # gathers. Chunk boundaries never trim the dependency neighborhood.
        maximum = h.new_full((m, layer.heads), -torch.inf)
        for start in range(0, len(target), batch_size):
            a, b = (
                target[start : start + batch_size],
                source[start : start + batch_size],
            )
            group = a[:, None].expand(-1, layer.heads)
            maximum.scatter_reduce_(
                0, group, scores(a, b), reduce="amax", include_self=True
            )
        denominator = h.new_zeros(m, layer.heads)
        for start in range(0, len(target), batch_size):
            a, b = (
                target[start : start + batch_size],
                source[start : start + batch_size],
            )
            group = a[:, None].expand(-1, layer.heads)
            denominator.scatter_add_(0, group, (scores(a, b) - maximum[a]).exp())
        out = h.new_zeros(m, layer.heads, layer.rank)
        for start in range(0, len(target), batch_size):
            a, b = (
                target[start : start + batch_size],
                source[start : start + batch_size],
            )
            weight = (scores(a, b) - maximum[a]).exp()
            weight /= denominator[a].clamp_min(1e-12)
            out.index_add_(0, a, weight[:, :, None] * v[b])
        self.counters["backbone_query_projection_rows"] += m
        self.counters["backbone_key_projection_rows"] += n
        self.counters["backbone_value_projection_rows"] += n
        self.counters["backbone_edges"] += len(target)
        self.counters["backbone_edge_score_evaluations"] += 3 * len(target)
        result = h[selected] + layer.output(out.reshape(m, -1))
        for start in range(0, m, batch_size):
            block = result[start : start + batch_size]
            block.add_(layer.ff(layer.norm2(block)))
        return result

    def _closure_cost_fraction(self, nodes):
        # Relative dominant FF/projection work. Traverse only demanded adjacency
        # rows and inspect their version stamps, never scan all cache stamps.
        current = nodes
        demanded, complete = 0.0, 0.0
        n = len(self.raw)
        for level in range(len(self.model.layers), 0, -1):
            layer = self.model.layers[level - 1]
            hidden = layer.output.out_features
            rank = layer.rank * layer.heads
            target = current[self.hidden_stamps[level][current] != self.version]
            support = (
                np.union1d(target, self._neighbors(target)) if len(target) else target
            )
            # Dense FF dominates; projections still run for support keys/values.
            demanded += len(target) * (8 * hidden * hidden + rank * hidden)
            demanded += len(support) * rank * hidden
            complete += n * (8 * hidden * hidden + 2 * rank * hidden)
            current = support
        return demanded / max(1.0, complete)

    def _ensure(self, nodes):
        self._ensure_w1(nodes)
        self._ensure_w2(nodes)
        self._ensure_u1(nodes)
        self._ensure_u2(nodes)
        missing = self._missing("h", nodes)
        if self.model.config.get("joint_task_backbone", False) and len(missing):
            self._ensure_hidden(len(self.model.layers), missing)
            index = torch.from_numpy(missing)
            self.h[index] = self._normalized_hidden(index)
            self.stamps["h"][missing] = self.version

    def _normalized_hidden(self, index):
        return F.normalize(self.model.norm(self.hidden[-1][index]), dim=1)

    def _materialize(self):
        # The high-coverage alternative uses vectorized full operators, rather
        # than paying repeated sparse-closure construction for every row.
        row, col = self.a.nonzero()
        visible = torch.from_numpy(np.stack((row, col))).long()
        values = attention(self.model, self.state, visible)
        self.weighted = sp.csr_matrix((values.numpy(), (row, col)), shape=self.a.shape)
        self.weighted1 = torch.from_numpy(self.weighted @ self.raw.numpy())
        self.parts[1] = F.normalize(self.weighted1, dim=1)
        self.parts[2] = F.normalize(
            torch.from_numpy(self.weighted @ self.weighted1.numpy()), dim=1
        )
        self.content1 = F.normalize(
            torch.from_numpy(self.uniform @ self.raw.numpy()), dim=1
        )
        self.content2 = F.normalize(
            torch.from_numpy(self.uniform @ self.content1.numpy()), dim=1
        )
        self.norm = np.sqrt(
            np.asarray(self.weighted.multiply(self.weighted).sum(1)).ravel()
        )
        self.norm_tensor = torch.from_numpy(self.norm)
        left, amplitude, right, _ = self.program
        if self.model.config.get("joint_task_backbone", False):
            for level, layer in enumerate(self.model.layers):
                self.hidden[level + 1] = layer.forward_graph(
                    self.hidden[level],
                    left,
                    amplitude,
                    right,
                    visible,
                    self.z if self.model.frame_aligned else None,
                )
                self.hidden_stamps[level + 1].fill(self.version)
            self.h = F.normalize(self.model.norm(self.hidden[-1]), dim=1)
        n = len(self.raw)
        for name, counter in (
            ("weight", "weight_rows"),
            ("w1", "weighted1_rows"),
            ("w2", "weighted2_rows"),
            ("u1", "uniform1_rows"),
            ("u2", "uniform2_rows"),
        ):
            self.counters[counter] += n
        self.counters["weight_support_rows"] += n
        self.counters["weight_edges"] += len(row)
        if self.model.config.get("joint_task_backbone", False):
            self.counters["backbone_rows"] += n * len(self.model.layers)
            self.counters["backbone_support_rows"] += n * len(self.model.layers)
            for name in (
                "backbone_query_projection_rows",
                "backbone_key_projection_rows",
                "backbone_value_projection_rows",
            ):
                self.counters[name] += n * len(self.model.layers)
            self.counters["backbone_edges"] += len(row) * len(self.model.layers)
            self.counters["backbone_edge_score_evaluations"] += len(row) * len(
                self.model.layers
            )
        for stamp in self.stamps.values():
            stamp.fill(self.version)
        self.counters["materializations"] += 1
        self.valid_w2_rows = n
        self.materialized_version = self.version

    @torch.no_grad()
    def features(self, pairs):
        nodes = np.unique(pairs.numpy())
        started = time.perf_counter()
        materialize = False
        if self.policy == "hybrid" and self.materialized_version != self.version:
            coverage = (self.valid_w2_rows + len(self._missing("w2", nodes))) / len(
                self.raw
            )
            materialize = coverage >= self.materialize_fraction
        elif self.policy == "closure" and self.materialized_version != self.version:
            materialize = (
                self._closure_cost_fraction(nodes) >= self.materialize_fraction
            )
        self.counters["scheduling_s"] += time.perf_counter() - started
        if materialize:
            self._materialize()
        self._ensure(nodes)
        return super().features(pairs)

    @torch.no_grad()
    def update(self, edges, added, removed=None):
        started = time.perf_counter()
        if removed is None:
            removed = edges[:, :0]
        if added.shape[1] == 0 and removed.shape[1] == 0:
            self.counters["topology_noops"] += 1
            return {
                "update_s": time.perf_counter() - started,
                "noop": 1,
                "version": self.version,
                "refreshed": 0,
                "affected_fraction": 0.0,
            }
        self._maintain_moments(edges, added, removed)
        self.state = self.moments.state()
        self.version += 1
        self.valid_w2_rows = 0
        self._index(edges)
        self._program()
        self.counters["topology_updates"] += 1
        return {
            "update_s": time.perf_counter() - started,
            "noop": 0,
            "version": self.version,
            "refreshed": 1,
            "affected_fraction": 1.0,
        }

    def _maintain_moments(self, edges, added, removed):
        self.moments.update(added, removed)

    def cache_bytes(self):
        return numeric_payload_bytes(
            {
                key: value
                for key, value in vars(self).items()
                if key not in ("model", "head")
            }
        )
