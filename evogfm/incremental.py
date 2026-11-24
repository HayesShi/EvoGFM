"""Incremental Predictive Analytics over Evolving Graphs: maintained link caches."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, fields, is_dataclass

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from .features import adjacency, attention, normalized_propagation
from .data import canonical_edges
from .link_inference import LinkScorer
from .model import GraphState, symmetric_eigh


@dataclass
class FullEdgeMoments:
    """Exact sufficient statistic for a fixed, encoded node feature frame."""

    z: torch.Tensor
    total: torch.Tensor
    count: int
    radial: torch.Tensor

    @classmethod
    @torch.no_grad()
    def build(cls, z, edges):
        total = torch.zeros(z.shape[1], z.shape[1], dtype=torch.float64)
        for block in edges.split(32768, dim=1):
            total += z[block[0]].double().T @ z[block[1]].double()
        radius = z.norm(dim=-1) / math.sqrt(z.shape[1])
        radial = torch.stack(
            [
                (radius <= value).float().mean()
                for value in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
            ]
        )
        return cls(z, total, edges.shape[1], radial)

    @torch.no_grad()
    def update(self, added, removed):
        for sign, edges in ((1, added), (-1, removed)):
            for block in edges.split(32768, dim=1):
                self.total += sign * (
                    self.z[block[0]].double().T @ self.z[block[1]].double()
                )
            self.count += sign * edges.shape[1]
        if self.count <= 0:
            raise ValueError("the visible graph must remain nonempty")

    @torch.no_grad()
    def state(self):
        n = len(self.z)
        rho = self.z.new_tensor(self.count / max(1.0, n * (n - 1) / 2)).clamp(
            1e-8, 1 - torch.finfo(self.z.dtype).eps
        )
        matrix = (rho.double() * self.total / self.count).to(self.z)
        matrix = (matrix + matrix.T) / 2
        spectrum, vectors = symmetric_eigh(matrix)
        return GraphState(
            self.z,
            vectors.to(self.z),
            spectrum.to(self.z),
            vectors.to(self.z),
            rho,
            self.radial,
            False,
        )


def program_matrix(model, state):
    left, amplitude, right, _ = model.attention_program(state)
    return (left * amplitude[None, :]) @ right.T


def changed_hop_rows(matrix, rows):
    """Rows whose sparse multiplication uses any changed input row."""
    if len(rows) == 0:
        return rows
    return np.unique(matrix[:, rows].nonzero()[0])


def local_propagation(matrix, source, previous, changed_rows, source_changed=None):
    """Update exactly the affected rows of one sparse multiplication."""
    affected = changed_rows
    if source_changed is not None:
        affected = np.union1d(affected, changed_hop_rows(matrix, source_changed))
    result = previous.copy()
    if len(affected):
        result[affected] = matrix[affected] @ source
    return result, affected


def ingest_edges(previous, incoming, n, known_keys):
    """Consume one event batch, retaining a sorted simple undirected graph."""
    candidates = canonical_edges(incoming, n)
    keys = (candidates[0] * n + candidates[1]).numpy()
    added = candidates[
        :, torch.from_numpy(~np.isin(keys, known_keys, assume_unique=True))
    ]
    visible = canonical_edges(torch.cat((previous, added), dim=1), n)
    return visible, added, (visible[0] * n + visible[1]).numpy()


def numeric_payload_bytes(value):
    """Count distinct numeric storage, including aliases across NumPy/PyTorch.

    Storage pointers and byte lengths define allocation intervals. Merging
    overlapping intervals also handles a ``torch.from_numpy`` tensor made from
    a slice whose NumPy owner is retained elsewhere in the cache.
    """
    allocations = {}
    seen_objects = {}

    def allocation(device, pointer, length):
        if length:
            allocations.setdefault(device, set()).add((pointer, pointer + length))

    def tensor_storage(tensor):
        if tensor.layout == torch.strided:
            storage = tensor.untyped_storage()
            allocation(str(tensor.device), storage.data_ptr(), storage.nbytes())
        elif tensor.layout == torch.sparse_coo:
            visit(tensor._indices())
            visit(tensor._values())
        elif tensor.layout in (torch.sparse_csr, torch.sparse_bsr):
            visit(tensor.crow_indices())
            visit(tensor.col_indices())
            visit(tensor.values())
        elif tensor.layout in (torch.sparse_csc, torch.sparse_bsc):
            visit(tensor.ccol_indices())
            visit(tensor.row_indices())
            visit(tensor.values())

    def array_storage(array):
        if array.dtype.hasobject:
            for item in array.flat:
                visit(item)
            return
        owner = array
        while isinstance(owner, np.ndarray) and owner.base is not None:
            owner = owner.base
        while isinstance(owner, memoryview):
            owner = owner.obj
        if isinstance(owner, torch.Tensor):
            tensor_storage(owner)
        elif isinstance(owner, np.ndarray):
            allocation("cpu", owner.__array_interface__["data"][0], owner.nbytes)
        else:
            # Non-NumPy buffer owners have no portable allocation identity;
            # the accessible strided byte interval is still measurable.
            pointer = array.__array_interface__["data"][0]
            low = sum(
                min(0, (size - 1) * stride)
                for size, stride in zip(array.shape, array.strides)
            )
            high = sum(
                max(0, (size - 1) * stride)
                for size, stride in zip(array.shape, array.strides)
            )
            allocation(
                "cpu", pointer + low, high - low + array.itemsize if array.size else 0
            )

    def visit(item):
        if id(item) in seen_objects:
            return
        # Keep component wrappers alive so temporary sparse Tensor wrappers
        # cannot reuse a previously visited Python identity during this walk.
        seen_objects[id(item)] = item
        if isinstance(item, torch.Tensor):
            tensor_storage(item)
        elif isinstance(item, np.ndarray):
            array_storage(item)
        elif sp.issparse(item):
            visit(vars(item))
        elif isinstance(item, dict):
            for member in item.values():
                visit(member)
        elif isinstance(item, (tuple, list, set)):
            for member in item:
                visit(member)
        elif is_dataclass(item) and not isinstance(item, type):
            for field in fields(item):
                visit(getattr(item, field.name))

    visit(value)
    total = 0
    for intervals in allocations.values():
        high = None
        for start, stop in sorted(intervals):
            if high is None or start >= high:
                total += stop - start
            elif stop > high:
                total += stop - high
            high = max(stop, high) if high is not None else stop
    return total


class MaintainedLinkScorer(LinkScorer):
    """The original 16/17 LP channels with separately maintained preparation.

    Node attributes and encoded/whitened features are fixed during the stream.
    ``exact_incremental`` updates edge moments and refreshes all affinities.
    ``error_aware`` refreshes when relative attention-program drift crosses a
    threshold; between refreshes it updates changed rows and propagation only.
    The drift is a diagnostic trigger, not a bound on prediction loss.
    """

    METHODS = ("full", "periodic", "exact_incremental", "error_aware")

    @torch.no_grad()
    def __init__(
        self,
        model,
        head,
        raw,
        x,
        edges,
        method="full",
        period=5,
        threshold=0.02,
        encoded=None,
    ):
        if method not in self.METHODS:
            raise ValueError(method)
        self.model, self.head, self.raw = model, head, raw
        self.x, self.method, self.period, self.threshold = x, method, period, threshold
        self.ablation, self.dense, self.step = "full", None, 0
        self.version = 0
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
        )
        self.z = model.encode_features(x) if encoded is None else encoded
        self.moments = FullEdgeMoments.build(self.z, edges)
        self.state = self.moments.state()
        self.edges = edges.clone()
        self.timings = {}
        self._full_prepare(edges)
        self.reference_program = program_matrix(model, self.state)
        self.last_refresh = 0

    def _index(self, edges):
        self.a = adjacency(edges, len(self.raw)).astype(np.float32).tocsr()
        self.a.sort_indices()
        self.degree = np.asarray(self.a.sum(1)).ravel()
        self.inv_degree = 1 / np.maximum(1, self.degree)
        self.inv_log_degree = 1 / np.log(np.maximum(2, self.degree))
        self.degree_tensor = torch.from_numpy(self.degree).float()
        self.deg = self.degree_tensor.log1p()
        self.uniform = normalized_propagation(self.a).astype(np.float32).tocsr()

    def _visible(self):
        row, col = self.a.nonzero()
        return torch.from_numpy(np.stack((row, col))).long()

    def _norms(self):
        self.norm = np.sqrt(
            np.asarray(self.weighted.multiply(self.weighted).sum(1)).ravel()
        )
        self.norm_tensor = torch.from_numpy(self.norm).float()
        _, amplitude, _, _ = self.model.programmed(self.state)
        self.amplitude_norm = amplitude.square().sum().sqrt().clamp_min(1e-6)

    def _full_backbone(self, visible):
        if not self.model.config.get("joint_task_backbone", False):
            self.h, self.hidden = None, []
            return
        left, amplitude, right, _ = self.model.attention_program(self.state)
        self.hidden = [self.model.up(self.state.z)]
        for layer in self.model.layers:
            self.hidden.append(
                layer.forward_graph(
                    self.hidden[-1],
                    left,
                    amplitude,
                    right,
                    visible,
                    self.state.z if self.model.frame_aligned else None,
                )
            )
            self._count_backbone(len(self.raw), visible.shape[1])
        self.h = F.normalize(self.model.norm(self.hidden[-1]), dim=1)

    def _count_backbone(self, rows, edges):
        # Eager local forward_graph computes every support output as well as
        # its keys/values, even when only changed outputs are copied back.
        for name in (
            "backbone_rows",
            "backbone_support_rows",
            "backbone_query_projection_rows",
            "backbone_key_projection_rows",
            "backbone_value_projection_rows",
        ):
            self.counters[name] += rows
        self.counters["backbone_edges"] += edges
        self.counters["backbone_edge_score_evaluations"] += edges

    def _local_backbone(self, endpoints):
        if not self.model.config.get("joint_task_backbone", False):
            return np.empty(0, dtype=np.int64)
        left, amplitude, right, _ = self.model.attention_program(self.state)
        changed = np.empty(0, dtype=np.int64)
        for index, layer in enumerate(self.model.layers):
            # Residuals depend on the row itself; messages depend on its neighbors.
            changed = np.union1d(
                endpoints, np.union1d(changed, changed_hop_rows(self.a, changed))
            )
            local = self.a[changed].tocoo()
            target, source = changed[local.row], local.col
            nodes = np.union1d(changed, source)
            if not len(nodes):
                continue
            remapped = torch.from_numpy(
                np.stack(
                    (np.searchsorted(nodes, target), np.searchsorted(nodes, source))
                )
            ).long()
            node_tensor = torch.from_numpy(nodes).long()
            selected = torch.from_numpy(np.searchsorted(nodes, changed)).long()
            result = layer.forward_graph(
                self.hidden[index][node_tensor],
                left,
                amplitude,
                right,
                remapped,
                self.state.z[node_tensor] if self.model.frame_aligned else None,
            )
            self._count_backbone(len(nodes), remapped.shape[1])
            self.hidden[index + 1][torch.from_numpy(changed)] = result[selected]
        if len(changed):
            ids = torch.from_numpy(changed)
            self.h[ids] = F.normalize(self.model.norm(self.hidden[-1][ids]), dim=1)
        return changed

    def _full_prepare(self, edges):
        start = time.perf_counter()
        self._index(edges)
        self.timings["graph_index_s"] = time.perf_counter() - start
        start = time.perf_counter()
        visible = self._visible()
        weight = attention(self.model, self.state, visible)
        self.counters["weight_rows"] += len(self.raw)
        self.counters["weight_support_rows"] += len(self.raw)
        self.counters["weight_edges"] += visible.shape[1]
        self.weighted = sp.csr_matrix(
            (weight.numpy(), tuple(visible.numpy())), shape=self.a.shape
        )
        self.timings["affinity_s"] = time.perf_counter() - start
        start = time.perf_counter()
        raw = self.raw.numpy()
        self.weighted1 = self.weighted @ raw
        self.weighted2 = self.weighted @ self.weighted1
        self.parts = [
            F.normalize(torch.from_numpy(value), dim=1)
            for value in (raw, self.weighted1, self.weighted2)
        ]
        self.content0 = F.normalize(self.raw, dim=1)
        self.uniform1 = self.uniform @ raw
        self.content1 = F.normalize(torch.from_numpy(self.uniform1), dim=1)
        self.uniform2 = self.uniform @ self.content1.numpy()
        self.content2 = F.normalize(torch.from_numpy(self.uniform2), dim=1)
        for name in (
            "weighted1_rows",
            "weighted2_rows",
            "uniform1_rows",
            "uniform2_rows",
        ):
            self.counters[name] += len(self.raw)
        self._norms()
        self.timings["propagation_s"] = time.perf_counter() - start
        start = time.perf_counter()
        self._full_backbone(visible)
        self.timings["backbone_s"] = time.perf_counter() - start

    def _local_prepare(self, edges, endpoints):
        start = time.perf_counter()
        old_a = self.a
        self._index(edges)
        # Symmetric degree normalization changes incident rows of neighbors too.
        uniform_rows = np.union1d(
            endpoints,
            np.union1d(
                changed_hop_rows(old_a, endpoints), changed_hop_rows(self.a, endpoints)
            ),
        )
        self.timings["graph_index_s"] = time.perf_counter() - start
        start = time.perf_counter()
        sub = self.a[endpoints].tocoo()
        visible = torch.from_numpy(np.stack((endpoints[sub.row], sub.col))).long()
        weights = attention(self.model, self.state, visible)
        self.counters["weight_rows"] += len(endpoints)
        # The shared attention helper projects all encoded nodes.
        self.counters["weight_support_rows"] += len(self.raw)
        self.counters["weight_edges"] += visible.shape[1]
        replacement = sp.csr_matrix(
            (weights.numpy(), (sub.row, sub.col)), shape=(len(endpoints), len(self.raw))
        )
        maintained = self.weighted.tolil()
        for index, row in enumerate(endpoints):
            low, high = replacement.indptr[index : index + 2]
            maintained.rows[row] = replacement.indices[low:high].tolist()
            maintained.data[row] = replacement.data[low:high].tolist()
        self.weighted = maintained.tocsr()
        self.timings["affinity_s"] = time.perf_counter() - start
        start = time.perf_counter()
        self.weighted1, first = local_propagation(
            self.weighted, self.raw.numpy(), self.weighted1, endpoints
        )
        self.weighted2, second = local_propagation(
            self.weighted, self.weighted1, self.weighted2, endpoints, first
        )
        self.parts = [self.content0] + [
            F.normalize(torch.from_numpy(value), dim=1)
            for value in (self.weighted1, self.weighted2)
        ]
        self.uniform1, first_uniform = local_propagation(
            self.uniform, self.raw.numpy(), self.uniform1, uniform_rows
        )
        self.content1 = F.normalize(torch.from_numpy(self.uniform1), dim=1)
        self.uniform2, second_uniform = local_propagation(
            self.uniform,
            self.content1.numpy(),
            self.uniform2,
            uniform_rows,
            first_uniform,
        )
        self.content2 = F.normalize(torch.from_numpy(self.uniform2), dim=1)
        for name, rows in (
            ("weighted1_rows", first),
            ("weighted2_rows", second),
            ("uniform1_rows", first_uniform),
            ("uniform2_rows", second_uniform),
        ):
            self.counters[name] += len(rows)
        self._norms()
        self.timings["propagation_s"] = time.perf_counter() - start
        start = time.perf_counter()
        backbone_rows = self._local_backbone(endpoints)
        self.timings["backbone_s"] = time.perf_counter() - start
        return len(np.union1d(np.union1d(second, second_uniform), backbone_rows)) / len(
            self.raw
        )

    @torch.no_grad()
    def update(self, edges, added, removed=None):
        removed = edges[:, :0] if removed is None else removed
        start = time.perf_counter()
        self.timings = {
            key: 0.0
            for key in (
                "statistics_s",
                "spectral_s",
                "graph_index_s",
                "affinity_s",
                "propagation_s",
                "backbone_s",
            )
        }
        if added.shape[1] == 0 and removed.shape[1] == 0:
            self.counters["topology_noops"] += 1
            self.timings.update(
                update_s=time.perf_counter() - start,
                noop=1,
                version=self.version,
                refreshed=0,
                drift=0.0,
                affected_fraction=0.0,
            )
            return self.timings.copy()
        self.step += 1
        self.version += 1
        self.counters["topology_updates"] += 1
        moment_start = time.perf_counter()
        if self.method == "full" or (
            self.method == "periodic" and self.step % self.period == 0
        ):
            self.moments = FullEdgeMoments.build(self.z, edges)
        elif self.method != "periodic":
            self.moments.update(added, removed)
        self.timings["statistics_s"] = time.perf_counter() - moment_start
        if self.method == "periodic" and self.step % self.period != 0:
            self.timings.update(
                update_s=time.perf_counter() - start,
                refreshed=0,
                drift=float("nan"),
                affected_fraction=0.0,
                noop=0,
                version=self.version,
            )
            return self.timings.copy()
        spectral_start = time.perf_counter()
        exact_state = self.moments.state()
        exact_program = program_matrix(self.model, exact_state)
        drift = float(
            (exact_program - self.reference_program).norm()
            / self.reference_program.norm().clamp_min(1e-8)
        )
        self.timings["spectral_s"] = time.perf_counter() - spectral_start
        refresh = (
            self.method != "error_aware"
            or drift > self.threshold
            or self.step - self.last_refresh >= self.period
        )
        if refresh:
            self.state = exact_state
            self._full_prepare(edges)
            self.reference_program = exact_program
            self.last_refresh = self.step
            affected = 1.0
        else:
            endpoints = np.unique(torch.cat((added, removed), dim=1).numpy())
            affected = self._local_prepare(edges, endpoints)
        self.edges = edges.clone()
        self.timings.update(
            update_s=time.perf_counter() - start,
            refreshed=int(refresh),
            drift=drift,
            affected_fraction=affected,
            noop=0,
            version=self.version,
        )
        return self.timings.copy()

    def cache_bytes(self):
        return numeric_payload_bytes(
            {
                key: value
                for key, value in vars(self).items()
                if key not in ("model", "head")
            }
        )
