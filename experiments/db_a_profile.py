"""Optional scoped diagnostics for CUDA work, stage time, and memory payloads."""

from collections import defaultdict
from functools import wraps
import time

import numpy as np


class WorkLedger:
    def __init__(self):
        self.version = -1
        self.request = -1
        self.counts = defaultdict(int)

    def _finish_version(self):
        if self.version < 0:
            return
        active = self.flags != 0
        first_layer, deeper = (self.flags & 2) != 0, (self.flags & 252) != 0
        self.counts["unique_version_edges"] += int(active.sum())
        self.counts["layer1_and_deeper_edges"] += int((first_layer & deeper).sum())
        self.counts["weighted_branch_and_backbone_edges"] += int(
            ((self.flags & 1 != 0) & (self.flags & 254 != 0)).sum()
        )

    def record(self, scorer, nodes, csr, missing, keys=None):
        started = time.perf_counter()
        if self.version != scorer.version:
            self._finish_version()
            self.version = scorer.version
            self.flags = np.zeros(scorer.a.nnz, dtype=np.uint8)
            self.seen_request = np.full(scorer.a.nnz, -2, dtype=np.int32)
            self.evaluated = np.zeros(scorer.a.nnz, dtype=bool)
        if keys is None:
            degrees = np.diff(csr.indptr)
            keys = np.repeat(
                scorer.a.indptr[nodes] - csr.indptr[:-1], degrees
            ) + np.arange(csr.nnz)
        cross_request = (self.seen_request[keys] != self.request) & (
            self.flags[keys] != 0
        )
        self.counts["cross_request_unique_edge_overlap"] += int(cross_request.sum())
        self.flags[keys] |= 1 << scorer.score_layer
        self.seen_request[keys] = self.request
        evaluated_keys = keys[missing]
        self.counts["recomputed_base_scores"] += int(
            self.evaluated[evaluated_keys].sum()
        )
        self.evaluated[evaluated_keys] = True
        self.counts["base_score_evaluations"] += len(missing)
        self.counts["dependency_edges_consumed"] += len(keys)
        self.counts[f"layer_{scorer.score_layer}_consumers"] += len(keys)
        self.counts["inspection_ns"] += round((time.perf_counter() - started) * 1e9)

    def summary(self):
        self._finish_version()
        return dict(self.counts)


class StageProfile:
    def __init__(self, torch):
        self.torch = torch
        self.stack = []
        self.times = defaultdict(float)

    def wrap(self, original, label, synchronize=False):
        @wraps(original)
        def measure(*args, **kwargs):
            if synchronize:
                self.torch.cuda.synchronize()
            self.stack.append(0.0)
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                if synchronize:
                    self.torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                self.times[label] += max(0.0, elapsed - self.stack.pop())
                if self.stack:
                    self.stack[-1] += elapsed

        return measure

    def install(self, scorer, moments, detailed):
        for name, label in (
            ("build", "moment_rescan"),
            ("state", "spectral_state"),
            ("update", "moment_delta"),
        ):
            original = getattr(moments, name)
            if name == "build":
                setattr(moments, name, staticmethod(self.wrap(original, label)))
            else:
                setattr(moments, name, self.wrap(original, label))
        stages = {
            "_index": ("csr_index", False),
            "_program": ("attention_program", False),
        }
        if detailed:
            stages.update(
                {
                    "_plan": ("protected_cache_plan", False),
                    "_select_mode": ("cache_allocation", False),
                    "_base_scores": ("base_scores", True),
                    "_scores_on_gpu": ("base_scores", True),
                    "_cuda_rows": ("attention_and_value_aggregation", True),
                    "_cuda_values": ("value_projection", True),
                    "_propagate_rows": ("weighted_and_uniform_propagation", True),
                    "_three_hop_entries": ("path_counts", False),
                }
            )
        for name, (label, synchronize) in stages.items():
            setattr(scorer, name, self.wrap(getattr(scorer, name), label, synchronize))


class AllocatorProfile:
    """Measure allocated GPU scratch above both stage-boundary allocations."""

    def __init__(self, cuda):
        self.cuda = cuda
        self.peak = 0
        self.reserved_peak = 0
        self.temporary = defaultdict(int)

    def begin(self):
        self._update_peaks()
        self.cuda.reset_peak_memory_stats()
        return self.cuda.memory_allocated()

    def finish(self, stage, before):
        self._update_peaks()
        after = self.cuda.memory_allocated()
        transient = self.cuda.max_memory_allocated() - max(before, after)
        self.temporary[stage] = max(self.temporary[stage], transient)

    def _update_peaks(self):
        self.peak = max(self.peak, self.cuda.max_memory_allocated())
        self.reserved_peak = max(self.reserved_peak, self.cuda.max_memory_reserved())


def resident_payload(scorer):
    """Final resident tensor/array payload, in addition to isolated allocator peaks."""
    from evogfm.incremental import numeric_payload_bytes

    graph = [
        getattr(scorer, key, None) for key in ("edges", "a", "uniform", "weighted")
    ]
    representation = [
        getattr(scorer, key, None)
        for key in (
            "raw",
            "z",
            "parts",
            "weighted1",
            "content0",
            "content1",
            "content2",
            "h",
            "hidden",
            "hidden_stamps",
            "stamps",
            "norm",
            "norm_tensor",
            "gpu_encoded",
            "propagation_inputs",
            "program",
            "gpu_program",
        )
    ]
    model = [
        list(layer.parameters())
        for layer in (scorer.model, scorer.gpu_model, scorer.head)
    ]
    caches = [
        cache
        for cache in (scorer._edge_cache, scorer._value_cache, scorer._projection_cache)
        if cache is not None
    ]
    return {
        "graph_numeric_bytes": numeric_payload_bytes(graph),
        "model_and_representation_numeric_bytes": numeric_payload_bytes(
            model + representation
        ),
        "graph_retained_gpu_numeric_bytes": 0,
        "model_and_representation_gpu_numeric_bytes": numeric_payload_bytes(
            [
                list(scorer.gpu_model.parameters()),
                scorer.gpu_encoded,
                [value for value in scorer.hidden if value.is_cuda],
                scorer.propagation_inputs,
                getattr(scorer, "gpu_program", None),
                scorer.temperatures,
            ]
        ),
        "retained_shared_cache_bytes": sum(cache.bytes for cache in caches),
        "shared_cache_gpu_numeric_bytes": sum(
            cache.data.nbytes
            for cache in caches
            if getattr(cache.data, "is_cuda", False)
        ),
    }
