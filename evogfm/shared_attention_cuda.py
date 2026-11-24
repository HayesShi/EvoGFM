"""CUDA attention for the same full-neighbor predictor and demand protocol."""

from copy import deepcopy
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from .incremental import FullEdgeMoments
from .execution_baselines import SharedLRULinkScorer, _LRUTable
from .shared_attention_fast import SharedAttentionFastLinkScorer
from .strict_dependency import _CSRStorage, load_author_recompute


class CudaLinkScorer(SharedAttentionFastLinkScorer):
    fuse_heads = True
    keep_scores = True
    keep_projections = True
    score_retention_depth = None

    @torch.no_grad()
    def __init__(
        self,
        *args,
        resident_scores=False,
        path_counts=None,
        async_indices=False,
        gpu_propagation=False,
        work_ledger=None,
        host_buffers=False,
        unified_device_projection=False,
        **kwargs,
    ):
        self.resident_scores = resident_scores
        self.path_counts = path_counts
        self.async_indices = async_indices
        self.work_ledger = work_ledger
        self.host_buffers = host_buffers
        self.unified_device_projection = unified_device_projection
        self.score_layer = 0
        super().__init__(*args, **kwargs)
        if not self.keep_scores:
            self._edge_budget = 0
        if not self.keep_projections:
            self._value_budget = self._projection_budget = 0
        self.gpu_model = deepcopy(self.model).cuda().eval()
        self.gpu_encoded = (
            self.z.cuda() if resident_scores and not host_buffers else None
        )
        self.gpu_program_version = -1
        if not host_buffers:
            self.hidden = [
                self.hidden[0].cuda(),
                *[torch.empty_like(rows, device="cuda") for rows in self.hidden[1:]],
            ]
        self.temperatures = [
            F.softplus(layer.log_temperature) + 0.05 for layer in self.gpu_model.layers
        ]
        self.propagation_inputs = None
        self.staged_propagation = gpu_propagation and host_buffers
        if gpu_propagation and not host_buffers:
            self.propagation_inputs = {
                id(self.raw): self.raw.cuda(),
                id(self.weighted1): torch.empty_like(self.weighted1, device="cuda"),
                id(self.content1): torch.empty_like(self.content1, device="cuda"),
            }
        torch.cuda.synchronize()

    def _select_mode(self):
        super()._select_mode()
        if not self.keep_scores:
            self._edge_cache = None
        if not self.keep_projections:
            self._value_cache = self._projection_cache = None
        cache = self._value_cache
        if cache is not None and not torch.is_tensor(cache.data):
            cache.data = torch.empty(
                cache.data.shape, dtype=torch.float32, device="cuda"
            )
        if self.resident_scores and self._dense_mode and self._edge_cache is not None:
            data = self._edge_cache.data
            self._edge_cache.data = torch.empty(
                data.shape, dtype=torch.float32, device="cuda"
            )

    def _index_on_device(self, array):
        return torch.as_tensor(array, dtype=torch.int64).to(
            "cuda", non_blocking=self.async_indices
        )

    def _base_dot_products(self, q, k, qi, ki, effective):
        # Tiny edge sets cost less to evaluate directly than to transfer.
        if len(qi) < 4096 and not self.unified_device_projection:
            return super()._base_dot_products(q, k, qi, ki, effective)
        q = torch.from_numpy(q).cuda()
        k = torch.from_numpy(k).cuda()
        qi, ki = self._index_on_device(qi), self._index_on_device(ki)
        amplitude = torch.from_numpy(effective).cuda()
        values = q.new_empty(len(qi))
        for start in range(0, len(qi), self.tile_edges):
            sl = slice(start, start + self.tile_edges)
            values[sl] = ((q[qi[sl]] * k[ki[sl]]) * amplitude).sum(1)
        return values.cpu().numpy()

    def _program_on_device(self):
        if self.gpu_program_version != self.version:
            left, amp, right, _ = self.program
            effective = amp.sign() * amp.abs().clamp_min(1e-20)
            self.gpu_program = left.cuda(), effective.cuda(), right.cuda()
            self.gpu_program_version = self.version
        return self.gpu_program

    def _project_on_device(self, nodes, matrix):
        def input_rows(selected):
            if self.gpu_encoded is None:
                return self.z[torch.from_numpy(selected)].cuda()
            return self.gpu_encoded[self._index_on_device(selected)]

        if not self.unified_device_projection:
            return input_rows(nodes) @ matrix
        # The padded rows stay in operator scratch. Each requested real row
        # uses the same CUDA GEMM shape regardless of the retention policy.
        width = 1024
        result = matrix.new_empty((len(nodes), matrix.shape[1]))
        for start in range(0, len(nodes), width):
            block = input_rows(nodes[start : start + width])
            count = len(block)
            block = F.pad(block, (0, 0, 0, width - count))
            result[start : start + count] = (block @ matrix)[:count]
        return result

    def _project(self, nodes, role):
        if not self.unified_device_projection:
            return super()._project(nodes, role)
        matrix = self._program_on_device()[2 * role]
        result = np.empty((len(nodes), matrix.shape[1]), dtype=np.float32)
        cache = self._projection_cache
        if isinstance(cache, _LRUTable):
            hit, saved = cache.get(nodes, role)
            result[hit] = saved
        elif cache is not None:
            positions, present, hit = cache.locate(nodes, role)
            result[hit] = cache.data[positions[hit], role]
        else:
            hit = np.zeros(len(nodes), dtype=bool)
        missing = np.flatnonzero(~hit)
        projected = self._project_on_device(nodes[missing], matrix).cpu().numpy()
        result[missing] = projected
        if isinstance(cache, _LRUTable):
            self._put(cache, nodes[missing], projected, role, name="projection")
        elif cache is not None:
            keep = present[missing]
            slots = positions[missing[keep]]
            cache.data[slots, role] = projected[keep]
            cache.valid[slots, role] = True
        self.counters["shared_projection_hits"] += int(hit.sum())
        self.counters["shared_projection_rows"] += len(missing)
        name = (
            "backbone_query_projection_rows"
            if role == 0
            else "backbone_key_projection_rows"
        )
        self.counters[name] += len(missing)
        return result

    def _scores_on_gpu(self, nodes, csr):
        degrees = np.diff(csr.indptr)
        targets = np.repeat(np.arange(len(nodes)), degrees)
        cache = self._active_edge_cache()
        if cache is not None:
            keys = np.repeat(
                self.a.indptr[nodes] - csr.indptr[:-1], degrees
            ) + np.arange(csr.nnz)
            hit = cache.valid[keys, 0]
            slots = self._index_on_device(keys)
            output = cache.data[slots, 0, 0]
            missing = np.flatnonzero(~hit)
            self.counters["shared_score_hits"] += csr.nnz - len(missing)
            self.counters["shared_planned_unique_edges"] += len(missing)
        else:
            output = torch.empty(csr.nnz, dtype=self.z.dtype, device="cuda")
            missing = np.arange(csr.nnz)
        if len(missing):
            left, effective, right = self._program_on_device()
            selected_rows = targets[missing]
            starts = np.r_[True, selected_rows[1:] != selected_rows[:-1]]
            qnodes = nodes[selected_rows[starts]]
            knodes, ki = self._node_unique(csr.indices[missing], inverse=True)
            q = self._project_on_device(qnodes, left)
            k = self._project_on_device(knodes, right)
            qi = self._index_on_device(np.cumsum(starts) - 1)
            ki = self._index_on_device(ki)
            if self.async_indices:
                full_miss = len(missing) == csr.nnz
                dots = output if full_miss else q.new_empty(len(missing))
                for start in range(0, len(missing), self.tile_edges):
                    sl = slice(start, start + self.tile_edges)
                    dots[sl] = ((q[qi[sl]] * k[ki[sl]]) * effective).sum(1)
                if not full_miss:
                    output[self._index_on_device(missing)] = dots
                if cache is not None:
                    destination = (
                        slots if full_miss else self._index_on_device(keys[missing])
                    )
                    cache.data[destination, 0, 0] = dots
            else:
                indices = self._index_on_device(missing)
                for start in range(0, len(missing), self.tile_edges):
                    sl = slice(start, start + self.tile_edges)
                    values = ((q[qi[sl]] * k[ki[sl]]) * effective).sum(1)
                    output[indices[sl]] = values
                    if cache is not None:
                        cache.data[slots[indices[sl]], 0, 0] = values
            if cache is not None:
                cache.valid[keys[missing], 0] = True
            self.counters["shared_projection_rows"] += len(qnodes) + len(knodes)
            self.counters["backbone_query_projection_rows"] += len(qnodes)
            self.counters["backbone_key_projection_rows"] += len(knodes)
        if self.work_ledger is not None:
            self.work_ledger.record(self, nodes, csr, missing)
        self.counters["shared_score_evaluations"] += len(missing)
        self.counters["shared_planned_score_consumptions"] += csr.nnz
        return output, targets

    def _uses_device_scores(self):
        return self.resident_scores and (
            self._dense_mode or self.shared_budget_bytes == 0
        )

    def _active_edge_cache(self):
        if (
            self.score_retention_depth is not None
            and self.score_layer > self.score_retention_depth
        ):
            return None
        return self._edge_cache

    def _base_scores(self, nodes, csr):
        if self._uses_device_scores():
            base, targets = self._scores_on_gpu(nodes, csr)
            return base.cpu().numpy(), targets
        if self.work_ledger is not None:
            degrees = np.diff(csr.indptr)
            keys = np.repeat(
                self.a.indptr[nodes] - csr.indptr[:-1], degrees
            ) + np.arange(csr.nnz)
            cache = self._active_edge_cache()
            if cache is None:
                missing = np.arange(csr.nnz)
            elif isinstance(cache, _LRUTable):
                positions, present = cache._locate(keys)
                missing = np.flatnonzero(~present | ~cache.valid[positions, 0])
            else:
                missing = np.flatnonzero(~cache.locate(keys)[2])
            self.work_ledger.record(self, nodes, csr, missing, keys)
        edge_cache = self._edge_cache
        active_cache = self._active_edge_cache()
        if active_cache is edge_cache:
            return super()._base_scores(nodes, csr)
        self._edge_cache = active_cache
        try:
            return super()._base_scores(nodes, csr)
        finally:
            self._edge_cache = edge_cache

    def _ensure_weight(self, nodes):
        self.score_layer = 0
        return super()._ensure_weight(nodes)

    def features(self, pairs):
        if self.work_ledger is not None:
            self.work_ledger.request += 1
            endpoints = np.unique(pairs.numpy())
            self.work_ledger.counts["hidden_requested_endpoints"] += len(endpoints)
            self.work_ledger.counts["hidden_hit_endpoints"] += int(
                (self.stamps["h"][endpoints] == self.version).sum()
            )
        return super().features(pairs)

    def _cuda_values(self, layer, level, support, inputs):
        cache = self._value_cache if level == 1 else None
        if cache is None:
            self.counters["backbone_value_projection_rows"] += len(support)
            return layer.value(layer.norm1(inputs))
        width = layer.heads * layer.rank
        result = inputs.new_empty((len(support), width))
        positions, present, hit = cache.locate(support)
        if self.resident_scores and hit.all():
            self.counters["shared_value_hits"] += len(support)
            return cache.data[self._index_on_device(positions), 0]
        hit_rows = self._index_on_device(np.flatnonzero(hit))
        result[hit_rows] = cache.data[self._index_on_device(positions[hit]), 0]
        missing = np.flatnonzero(~hit)
        missing_rows = self._index_on_device(missing)
        if len(missing):
            values = layer.value(layer.norm1(inputs[missing_rows]))
            result[missing_rows] = values
            keep = present[missing]
            slots = positions[missing[keep]]
            cache.data[self._index_on_device(slots), 0] = values[
                self._index_on_device(np.flatnonzero(keep))
            ]
            cache.valid[slots, 0] = True
        self.counters["backbone_value_projection_rows"] += len(missing)
        self.counters["shared_value_hits"] += len(support) - len(missing)
        return result

    def _device_base(self, missing, csr):
        if self._uses_device_scores():
            return self._scores_on_gpu(missing, csr)
        host_base, targets = self._base_scores(missing, csr)
        return torch.from_numpy(host_base).cuda(), targets

    def _cuda_rows(self, level, missing, support, csr, inputs):
        self.score_layer = level
        layer = self.gpu_model.layers[level - 1]
        values = self._cuda_values(layer, level, support, inputs).reshape(
            len(support), layer.heads, layer.rank
        )
        out = values.new_zeros((len(missing), layer.heads, layer.rank))
        if csr.nnz:
            base, targets = self._device_base(missing, csr)
            target = self._index_on_device(targets)
            offsets = self._index_on_device(csr.indptr)
            columns = torch.as_tensor(
                np.searchsorted(support, csr.indices), dtype=torch.int32, device="cuda"
            )

            def softmax(scores):
                maxima = torch.segment_reduce(scores, "max", offsets=offsets)
                weights = (scores - maxima[target]).exp()
                denominator = torch.segment_reduce(weights, "sum", offsets=offsets)
                return (weights / denominator[target].clamp_min(1e-12)).T.contiguous()

            temperatures = self.temperatures[level - 1]
            if self.fuse_heads:
                scores = base[:, None] * temperatures
            else:
                # Demand-Only recomputes the scalar for each head but keeps
                # the same FP32 softmax layout and reduction as Head-Fused.
                per_head = [base] + [
                    self._device_base(missing, csr)[0] for _ in range(layer.heads - 1)
                ]
                scores = torch.stack(per_head, dim=1) * temperatures
            weights = softmax(scores)
            csr_offsets = offsets.int()
            for head in range(layer.heads):
                matrix = torch.sparse_csr_tensor(
                    csr_offsets,
                    columns,
                    weights[head],
                    size=(len(missing), len(support)),
                )
                out[:, head] = torch.sparse.mm(matrix, values[:, head].contiguous())
        idx = self._index_on_device(np.searchsorted(support, missing))
        result = inputs[idx] + layer.output(out.reshape(len(missing), -1))
        for block in result.split(self.tile_edges):
            block.add_(layer.ff(layer.norm2(block)))
        self.counters["backbone_rows"] += len(missing)
        self.counters["backbone_support_rows"] += len(support)
        self.counters["backbone_edges"] += csr.nnz
        self.counters["backbone_edge_score_evaluations"] += csr.nnz * (
            1 if self.fuse_heads else layer.heads
        )
        return result

    def _compute_hidden_rows(self, level, missing, support, csr):
        if self.host_buffers:
            for start, stop in self._row_tiles(csr.indptr):
                rows, block = missing[start:stop], csr[start:stop]
                required = self._union(rows, block.indices)
                inputs = self.hidden[level - 1][torch.from_numpy(required)].cuda()
                result = self._cuda_rows(level, rows, required, block, inputs)
                self.hidden[level][torch.from_numpy(rows)] = result.cpu()
        else:
            inputs = self.hidden[level - 1][self._index_on_device(support)]
            result = self._cuda_rows(level, missing, support, csr, inputs)
            self.hidden[level][self._index_on_device(missing)] = result
        self.hidden_stamps[level][missing] = self.version

    @staticmethod
    def _row_tiles(offsets):
        start = 0
        while start < len(offsets) - 1:
            stop = (
                np.searchsorted(offsets, int(offsets[start]) + 262144, side="right") - 1
            )
            stop = min(start + 16384, max(start + 1, stop))
            yield start, stop
            start = stop

    def _normalized_hidden(self, index):
        if self.host_buffers:
            return torch.cat(
                [
                    F.normalize(
                        self.gpu_model.norm(self.hidden[-1][block].cuda()), dim=1
                    ).cpu()
                    for block in index.split(16384)
                ]
            )
        embeddings = self.gpu_model.norm(self.hidden[-1][index.cuda()])
        return F.normalize(embeddings, dim=1).cpu()

    def _three_hop_entries(self, left_ids, right_ids, left, right):
        if self.path_counts is not None:
            return self.path_counts(self.a.indptr, self.a.indices, left_ids, right_ids)
        return super()._three_hop_entries(left_ids, right_ids, left, right)

    def _propagate_rows(self, matrix, rows, source, *, normalize=False, keep=None):
        if self.propagation_inputs is None and not self.staged_propagation:
            return super()._propagate_rows(
                matrix, rows, source, normalize=normalize, keep=keep
            )
        selected = matrix[rows]
        if self.staged_propagation:
            output = source.new_empty(len(rows), source.shape[1])
            for start, stop in self._row_tiles(selected.indptr):
                block = selected[start:stop]
                if not block.nnz:
                    output[start:stop].zero_()
                    continue
                support, inverse = np.unique(block.indices, return_inverse=True)
                inputs = source[torch.from_numpy(support)].cuda()
                output[start:stop] = self._csr_product(
                    block,
                    block,
                    matrix,
                    source,
                    inputs,
                    inverse.astype(np.int32, copy=False),
                    normalize,
                )
            return output
        result = self._csr_product(
            selected,
            selected,
            matrix,
            source,
            self.propagation_inputs[id(source)],
            selected.indices,
            normalize,
            to_host=False,
        )
        if keep is not None:
            self.propagation_inputs[id(keep)][self._index_on_device(rows)] = result
        return result.cpu()

    def _csr_product(
        self,
        selected,
        reference_rows,
        matrix,
        source,
        inputs,
        column_indices,
        normalize,
        to_host=True,
    ):
        pointers = torch.from_numpy(selected.indptr).cuda(non_blocking=True)
        columns = torch.from_numpy(column_indices).cuda(non_blocking=True)
        data = torch.from_numpy(selected.data).cuda(non_blocking=True)
        sparse = torch.sparse_csr_tensor(
            pointers, columns, data, size=(selected.shape[0], len(inputs))
        )
        result = torch.sparse.mm(sparse, inputs)
        if matrix is self.weighted:
            # Very long rows can exceed the FP32 reference's tolerance when
            # summed in cuSPARSE's parallel order. Preserve their CPU order.
            long_rows = np.flatnonzero(np.diff(selected.indptr) >= 1024)
            if len(long_rows):
                reference = torch.from_numpy(reference_rows[long_rows] @ source.numpy())
                result[self._index_on_device(long_rows)] = reference.cuda(
                    non_blocking=True
                )
        if normalize:
            result = F.normalize(result, dim=1)
        return result.cpu() if to_host else result


class CudaDemandOnlyLinkScorer(CudaLinkScorer):
    """Same CUDA operators with one structural dot product for each head."""

    fuse_heads = False


class CudaNoScoresLinkScorer(CudaLinkScorer):
    keep_scores = False


class CudaFeatureLayerOneLinkScorer(CudaLinkScorer):
    score_retention_depth = 1


class CudaNoProjectionsLinkScorer(CudaLinkScorer):
    keep_projections = False


class CudaRebuiltMomentsLinkScorer(CudaLinkScorer):
    def _maintain_moments(self, edges, added, removed):
        self.moments = FullEdgeMoments.build(self.z, edges)


class _CudaLRUValues(_LRUTable):
    def _read(self, slots, role):
        return self.data[torch.as_tensor(slots, dtype=torch.long, device="cuda"), role]

    def _write(self, slots, role, values):
        self.data[torch.as_tensor(slots, dtype=torch.long, device="cuda"), role] = (
            values
        )


class CudaSharedLRULinkScorer(CudaLinkScorer, SharedLRULinkScorer):
    """Object LRU on the same score and value kernels and retention budget."""

    def _make_lru(self, budget, width, roles=1, universe=None):
        if (
            roles == 1
            and width == self.model.layers[0].heads * self.model.layers[0].rank
        ):
            if budget < _LRUTable.overhead + 8 + 4 * width + 1 + 12:
                return None
            return _CudaLRUValues(budget, width, roles, universe)
        return super()._make_lru(budget, width, roles, universe)

    def _cuda_values(self, layer, level, support, inputs):
        cache = self._value_cache if level == 1 else None
        if not isinstance(cache, _CudaLRUValues):
            return super()._cuda_values(layer, level, support, inputs)
        hit, values = cache.get(support)
        result = inputs.new_empty((len(support), layer.heads * layer.rank))
        missing = np.flatnonzero(~hit)
        miss = self._index_on_device(missing)
        result[self._index_on_device(np.flatnonzero(hit))] = values
        fresh = layer.value(layer.norm1(inputs[miss]))
        result[miss] = fresh
        self._put(cache, support[missing], fresh, name="value")
        self.counters["shared_value_hits"] += int(hit.sum())
        self.counters["backbone_value_projection_rows"] += len(missing)
        return result


class _AuthorStorage(_CSRStorage):
    def compile_block(self, targets):
        import dgl

        rows = self.kernel.a[targets.numpy()]
        sources = self.kernel._union(targets.numpy(), rows.indices)
        return SimpleNamespace(
            srcdata={dgl.NID: torch.from_numpy(sources)},
            dstdata={dgl.NID: targets},
            rows=rows,
        )


class CudaAuthorLinkScorer(CudaLinkScorer):
    """Unmodified NeutronRT demand planner with the common CUDA operators."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.author, _ = load_author_recompute()
        self.planner = self.author.RecomputePlanner(
            _AuthorStorage(self),
            len(self.model.layers),
            batch_size=max(1, len(self.raw)),
        )
        self.counters["author_planning_s"] = 0.0

    def _ensure_hidden(self, level, nodes):
        def available(depth, required, version):
            if depth == 0:
                return torch.ones_like(required, dtype=torch.bool)
            return torch.from_numpy(
                self.hidden_stamps[depth][required.numpy()] == version
            )

        started = time.perf_counter()
        plan = self.planner.plan(
            torch.from_numpy(nodes), level, self.version, available=available
        )
        self.counters["author_planning_s"] += time.perf_counter() - started

        def load(depth, sources, _version):
            if self.host_buffers:
                return self.hidden[depth][sources]
            return self.hidden[depth][sources.cuda()]

        def apply(task, inputs):
            if self.host_buffers:
                nodes, support, csr = (
                    task.targets.numpy(),
                    task.sources.numpy(),
                    task.block.rows,
                )
                result = inputs.new_empty(len(nodes), inputs.shape[1])
                for start, stop in self._row_tiles(csr.indptr):
                    rows, block = nodes[start:stop], csr[start:stop]
                    required = self._union(rows, block.indices)
                    subset = inputs[
                        torch.from_numpy(np.searchsorted(support, required))
                    ].cuda()
                    result[start:stop] = self._cuda_rows(
                        task.layer, rows, required, block, subset
                    ).cpu()
                return result
            return self._cuda_rows(
                task.layer,
                task.targets.numpy(),
                task.sources.numpy(),
                task.block.rows,
                inputs,
            )

        def publish(depth, targets, value, version):
            self.hidden[depth][targets if self.host_buffers else targets.cuda()] = value
            self.hidden_stamps[depth][targets.numpy()] = version

        self.author.execute_recomputation(plan, load=load, apply=apply, publish=publish)


class CudaFullRefreshLinkScorer(CudaLinkScorer):
    """Full re-materialization on the common CUDA operators."""

    @torch.no_grad()
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.all_nodes = np.arange(len(self.raw))
        self._refresh()

    def _refresh(self):
        width = 16384 if self.host_buffers else len(self.all_nodes)
        for start in range(0, len(self.all_nodes), width):
            self._ensure(self.all_nodes[start : start + width])

    @torch.no_grad()
    def update(self, edges, added, removed=None):
        started = time.perf_counter()
        if not added.numel() and (removed is None or not removed.numel()):
            return super().update(edges, added, removed)
        self.moments = FullEdgeMoments.build(self.z, edges)
        self.state = self.moments.state()
        self.version += 1
        self.valid_w2_rows = 0
        self._index(edges)
        self._program()
        self._refresh()
        self.counters["topology_updates"] += 1
        self.counters["materializations"] += 1
        return {"update_s": time.perf_counter() - started, "topology_changed": 1}
