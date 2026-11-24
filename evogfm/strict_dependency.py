"""NeutronRT author dependency planner/executor on the common numeric backend.

This is NOT the official end-to-end NeutronRT runtime. Unmodified author
RecomputePlanner/execute_recomputation (revision 52f7b627) drive the backbone.
The complete EvoGFM operator DAG and global epoch are compatibility extensions.
Common storage initialization and numerical kernels are composed from Demand;
its recursive dependency planner, features(), and update() are never called.
See docs/vldb.md for author source setup.
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import sys
import time
import types

import numpy as np
import torch
import torch.nn.functional as F

from .demand_maintenance import DemandLinkScorer
from .link_inference import LinkScorer


def load_author_recompute():
    """Load unmodified author modules; disable only unavailable GraphBolt DLL.

    DGL 2.2.1's regular native CPU graph/SAGE/GAT operators work in the local
    Torch 2.6 environment. Its unrelated GraphBolt DLL does not exist. The
    process-local placeholder prevents that optional import; GraphBolt and
    distributed execution are NOT supported or exercised by this adapter.
    """
    root = Path(__file__).resolve().parents[1] / "baselines" / "NeutronRT"
    if not (root / "lib/graph/recompute.py").is_file():
        raise FileNotFoundError(f"Pinned NeutronRT source required at {root}")
    spec = importlib.util.find_spec("dgl")
    if spec is None:
        raise ImportError("The author planner requires the DGL import dependency")
    disabled = False
    if sys.platform == "win32" and "dgl" not in sys.modules:
        directory = Path(spec.origin).parent
        dll = (
            directory
            / "graphbolt"
            / f"graphbolt_pytorch_{torch.__version__.split('+')[0]}.dll"
        )
        if not dll.exists():
            sys.modules["dgl.graphbolt"] = types.ModuleType("dgl.graphbolt")
            sys.modules["dgl.graphbolt"].__adapter_disabled__ = True
            disabled = True
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    module = importlib.import_module("lib.graph.recompute")
    if Path(module.__file__).resolve() != (root / "lib/graph/recompute.py").resolve():
        raise ImportError("Conflicting lib.graph.recompute module already loaded")
    disabled |= getattr(sys.modules.get("dgl.graphbolt"), "__adapter_disabled__", False)
    return module, disabled


class _CSRStorage:
    """Author-supported compile_block storage hook, retaining every neighbor."""

    device = torch.device("cpu")

    def __init__(self, kernel):
        self.kernel = kernel

    def num_nodes(self):
        return len(self.kernel.raw)

    def in_degrees(self, nodes):
        indptr = self.kernel.a.indptr
        ids = nodes.numpy()
        return torch.from_numpy((indptr[ids + 1] - indptr[ids]).astype(np.int64))

    def compile_block(self, targets):
        import dgl

        ids = targets.numpy()
        rows = self.kernel.a[ids].tocoo()
        sources = np.union1d(ids, rows.col)
        return types.SimpleNamespace(
            srcdata={dgl.NID: torch.from_numpy(sources)},
            dstdata={dgl.NID: targets},
            target_edge=torch.from_numpy(rows.row).long(),
            source_edge=torch.from_numpy(np.searchsorted(sources, rows.col)).long(),
            selected=torch.from_numpy(np.searchsorted(sources, ids)).long(),
        )


class StrictDependencyScorer:
    """Exact same-backend adaptation; no lazy/bulk selection or sampling.

    All graph-derived rows use the current graph epoch, including the global
    spectral-program dependency. This permits O(1) logical invalidation; it
    does not enumerate every node as a synthetic feature-update event.
    """

    label = "NeutronRT author strict planner adaptation (common backend)"
    source_revision = "52f7b6271c66aa17906f7e9e6559c25e22e575a4"

    @torch.no_grad()
    def __init__(self, model, head, raw, edges, encoded=None):
        self._author, self.graphbolt_import_disabled = load_author_recompute()
        self._kernel = DemandLinkScorer(model, head, raw, edges, encoded=encoded)
        # One common request block per layer matches the local Demand backend.
        # The author default 8192 chunk is a storage choice, not a dependency
        # rule; numerical edge/FF chunking remains in the shared kernels.
        self._planner = self._author.RecomputePlanner(
            _CSRStorage(self._kernel), len(model.layers), batch_size=max(1, len(raw))
        )
        self.counters = self._kernel.counters
        self.counters.update(
            planning_s=0.0, plan_requests=0, topology_updates=0, topology_noops=0
        )
        for level in range(1, len(model.layers) + 1):
            for metric in ("target_rows", "support_rows", "edges", "cache_hits"):
                self.counters[f"layer{level}_{metric}"] = 0

    def __getattr__(self, name):
        # Expose the common numeric state for parity diagnostics and readers.
        return getattr(self._kernel, name)

    def _missing(self, name, nodes):
        k = self._kernel
        return nodes[k.stamps[name][nodes] != k.version]

    def _plan(self, nodes):
        """Adapt extra feature branches; author planner compiles the backbone."""
        k = self._kernel
        w2 = self._missing("w2", nodes)
        w1 = self._missing("w1", np.union1d(nodes, k.a[w2].indices))
        weight = self._missing("weight", np.union1d(w1, w2))
        u2 = self._missing("u2", nodes)
        # The symmetric normalized uniform operator includes self loops.
        u1 = self._missing("u1", np.union1d(nodes, k.uniform[u2].indices))
        backbone = None
        h = self._missing("h", nodes)
        if k.model.config.get("joint_task_backbone", False):

            def available(level, required, version):
                if level == 0:
                    return torch.ones_like(required, dtype=torch.bool)
                valid = k.hidden_stamps[level][required.numpy()] == version
                self.counters[f"layer{level}_cache_hits"] += int(valid.sum())
                return torch.from_numpy(valid)

            backbone = self._planner.plan(
                torch.from_numpy(h), len(k.model.layers), k.version, available=available
            )
        return weight, w1, w2, u1, u2, h, backbone

    @torch.no_grad()
    def features(self, pairs):
        k = self._kernel
        started = time.perf_counter()
        plan = self._plan(np.unique(pairs.numpy()))
        elapsed = time.perf_counter() - started
        self.counters["planning_s"] += elapsed
        self.counters["scheduling_s"] += elapsed
        self.counters["plan_requests"] += 1
        weight, w1, w2, u1, u2, h, backbone = plan
        # Only a leaf numerical kernel is shared; dependency control is above.
        k._ensure_weight(weight)
        if len(w1):
            index = torch.from_numpy(w1)
            value = torch.from_numpy(k.weighted[w1] @ k.raw.numpy())
            k.weighted1[index] = value
            k.parts[1][index] = F.normalize(value, dim=1)
        if len(w2):
            value = torch.from_numpy(k.weighted[w2] @ k.weighted1.numpy())
            k.parts[2][torch.from_numpy(w2)] = F.normalize(value, dim=1)
        if len(u1):
            value = torch.from_numpy(k.uniform[u1] @ k.raw.numpy())
            k.content1[torch.from_numpy(u1)] = F.normalize(value, dim=1)
        if len(u2):
            value = torch.from_numpy(k.uniform[u2] @ k.content1.numpy())
            k.content2[torch.from_numpy(u2)] = F.normalize(value, dim=1)
        for name, rows, counter in (
            ("w1", w1, "weighted1_rows"),
            ("w2", w2, "weighted2_rows"),
            ("u1", u1, "uniform1_rows"),
            ("u2", u2, "uniform2_rows"),
        ):
            k.stamps[name][rows] = k.version
            self.counters[counter] += len(rows)
        k.valid_w2_rows += len(w2)

        def load(level, sources, version):
            if version != k.version:
                raise ValueError("Graph epoch changed during strict execution")
            return k.hidden[level][sources]

        def apply(task, inputs):
            level, targets, sources = task.layer, task.targets, task.sources
            output = k._forward_rows(
                k.model.layers[level - 1],
                inputs,
                k.z[sources],
                task.block.selected,
                task.block.target_edge,
                task.block.source_edge,
            )
            self.counters["backbone_rows"] += len(targets)
            self.counters["backbone_support_rows"] += len(sources)
            self.counters[f"layer{level}_target_rows"] += len(targets)
            self.counters[f"layer{level}_support_rows"] += len(sources)
            self.counters[f"layer{level}_edges"] += len(task.block.target_edge)
            return output

        def publish(level, targets, output, version):
            if version != k.version:
                raise ValueError("Graph epoch changed before publication")
            k.hidden[level][targets] = output
            k.hidden_stamps[level][targets.numpy()] = version

        if backbone is not None:
            self._author.execute_recomputation(
                backbone, load=load, apply=apply, publish=publish
            )
        if k.model.config.get("joint_task_backbone", False) and len(h):
            index = torch.from_numpy(h)
            k.h[index] = F.normalize(k.model.norm(k.hidden[-1][index]), dim=1)
            k.stamps["h"][h] = k.version
        return LinkScorer.features(k, pairs)

    @torch.no_grad()
    def update(self, edges, added, removed=None):
        k = self._kernel
        start = time.perf_counter()
        removed = edges[:, :0] if removed is None else removed
        if added.numel() == 0 and removed.numel() == 0:
            self.counters["topology_noops"] += 1
            return {"update_s": time.perf_counter() - start, "topology_changed": 0}
        k.moments.update(added, removed)
        k.state = k.moments.state()
        k.version += 1
        k.valid_w2_rows = 0
        k._index(edges)
        k._program()
        self.counters["topology_updates"] += 1
        return {"update_s": time.perf_counter() - start, "topology_changed": 1}

    def cache_bytes(self):
        return self._kernel.cache_bytes()

    @property
    def provenance(self):
        import dgl

        return dict(
            label=self.label,
            author_revision=self.source_revision,
            author_planner_path=str(Path(self._author.__file__).resolve()),
            author_planner_modified=False,
            official_end_to_end_system=False,
            backend="Common SciPy CSR and released predictor kernels",
            dgl_version=dgl.__version__,
            unused_graphbolt_import_disabled=self.graphbolt_import_disabled,
            target_block_size=self._planner.batch_size,
        )

    def score(self, pairs, fitted, batch_size=512, group_queries=True):
        # Bind the released readout to this adapter's features(), not to the
        # composed Demand object, so public scoring also uses the author plan.
        return LinkScorer.score(self, pairs, fitted, batch_size, group_queries)
