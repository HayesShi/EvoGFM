"""Full EvoGFM link features with graph preparation shared across query batches."""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .features import node_representations, attention, pair_context
from .readout import fit_residual, predict_residual


def three_hop_entries(adjacency, left_ids, right_ids, left, right):
    """Exact A³ entries, sharing a two-hop row across repeated query endpoints."""
    nodes, inverse = np.unique(left_ids, return_inverse=True)
    other_nodes, other_inverse = np.unique(right_ids, return_inverse=True)
    opposite = right
    if len(other_nodes) < len(nodes):
        nodes, inverse, opposite = other_nodes, other_inverse, left
    if len(nodes) * 2 >= len(left_ids):
        return np.asarray((left @ adjacency).multiply(right).sum(1)).ravel()
    result = np.empty(len(left_ids), dtype=adjacency.dtype)
    two_hop_rows = adjacency[nodes] @ adjacency
    two_hop_rows.sort_indices()
    for index in range(len(nodes)):
        selected = np.flatnonzero(inverse == index)
        two_hop = two_hop_rows.getrow(index)
        neighbors = opposite[selected]
        if not two_hop.nnz:
            result[selected] = 0
            continue
        positions = np.searchsorted(two_hop.indices, neighbors.indices)
        matched = (positions < two_hop.nnz) & (
            two_hop.indices[np.minimum(positions, two_hop.nnz - 1)] == neighbors.indices
        )
        values = np.zeros(neighbors.nnz, dtype=adjacency.dtype)
        values[matched] = two_hop.data[positions[matched]]
        rows = np.repeat(np.arange(len(selected)), np.diff(neighbors.indptr))
        result[selected] = np.bincount(
            rows, weights=neighbors.data * values, minlength=len(selected)
        )
    return result


class LinkScorer:
    """The released 17 channels: ten content/topology, six inverse, one encoder."""

    @torch.no_grad()
    def __init__(self, model, head, raw, x, edges, observation_rate=1.0):
        self.model, self.head, self.raw = model, head, raw
        self.state = model.state(x, edges, observation_rate)
        self.parts, self.h = node_representations(model, self.state, raw, edges, hops=2)
        # CSV features can be column-major; queries gather complete node rows.
        self.parts = [part.contiguous() for part in self.parts]
        self.a, self.degree, self.content0, self.content1, self.content2 = pair_context(
            raw, edges, normalized=self.parts[0]
        )
        self.a.sort_indices()
        row, col = self.a.nonzero()
        visible = torch.from_numpy(np.stack((row, col))).long()
        w = attention(model, self.state, visible)
        self.weighted = sp.csr_matrix((w.numpy(), (row, col)), shape=self.a.shape)
        self.norm = np.sqrt(
            np.asarray(self.weighted.multiply(self.weighted).sum(1)).ravel()
        )
        self.inv_degree = 1 / np.maximum(1, self.degree)
        self.inv_log_degree = 1 / np.log(np.maximum(2, self.degree))
        self.degree_tensor = torch.from_numpy(self.degree).float()
        self.norm_tensor = torch.from_numpy(self.norm).float()
        self.deg = self.degree_tensor.log1p()
        _, amplitude, _, _ = model.programmed(self.state)
        self.amplitude_norm = amplitude.square().sum().sqrt().clamp_min(1e-6)
        self.dense = None
        if len(raw) <= 6000 and self.a.nnz / len(raw) ** 2 > 0.02:
            # DDI is small and dense. Compute all exact path entries once.
            a = self.a.toarray()
            cn = a @ a.T
            ra = (a / np.maximum(1, self.degree)[None, :]) @ a.T
            aa = (a / np.log(np.maximum(2, self.degree))[None, :]) @ a.T
            paths = cn @ a
            weighted = self.weighted.toarray()
            shared = weighted @ weighted.T
            self.dense = (cn, ra, aa, paths, shared)

    @torch.no_grad()
    def features(self, pairs):
        i, j = pairs
        ip, jp = i.numpy(), j.numpy()
        if self.dense is None:
            left, right = self.a[ip], self.a[jp]
            common = left.multiply(right)
            structural = [
                np.asarray(common.sum(1)).ravel(),
                np.asarray(common @ self.inv_degree).ravel(),
                np.asarray(common @ self.inv_log_degree).ravel(),
                three_hop_entries(self.a, ip, jp, left, right),
            ]
            shared = np.asarray(
                self.weighted[ip].multiply(self.weighted[jp]).sum(1)
            ).ravel()
        else:
            structural = [matrix[ip, jp] for matrix in self.dense[:4]]
            shared = self.dense[4][ip, jp]
        structure = torch.from_numpy(np.stack(structural, axis=1)).float().log1p()
        raw, p1, p2 = self.content0, self.content1, self.content2
        base = torch.stack(
            (
                (raw[i] * raw[j]).sum(1),
                (p1[i] * p1[j]).sum(1),
                (p2[i] * p2[j]).sum(1),
                (raw[i] * p1[j] + p1[i] * raw[j]).sum(1) / 2,
                torch.minimum(self.deg[i], self.deg[j]),
                torch.maximum(self.deg[i], self.deg[j]),
                *structure.T,
            ),
            dim=1,
        )
        z0, z1, z2 = self.parts
        relation = self.model.relation_logits(self.state, pairs) / self.amplitude_norm
        shared = torch.from_numpy(shared).float()
        degree, norm = self.degree_tensor, self.norm_tensor
        extra = torch.stack(
            (
                (z1[i] * z1[j]).sum(1),
                (z2[i] * z2[j]).sum(1),
                (z0[i] * z1[j] + z1[i] * z0[j]).sum(1) / 2,
                relation.tanh(),
                (shared * degree[i] * degree[j]).clamp_min(0).log1p(),
                shared / (norm[i] * norm[j]).clamp_min(1e-8),
            ),
            dim=1,
        )
        if self.model.config.get("joint_task_backbone", False):
            extra = torch.cat(
                (extra, (self.h[i] * self.h[j]).sum(1, keepdim=True)), dim=1
            )
        return torch.cat((base, extra), dim=1)

    @torch.no_grad()
    def score(self, pairs, fitted, batch_size=512, group_queries=True):
        # Official order is restored before ranking; every query is retained.
        order = None
        if (
            group_queries
            and self.dense is None
            and pairs.shape[1] > min(batch_size, 512)
        ):
            # All channels are symmetric on this undirected inference graph.
            # Put the repeated endpoint first for one-sided negative candidates.
            _, inverse, counts = torch.unique(
                pairs, return_inverse=True, return_counts=True
            )
            frequency = counts[inverse]
            swap = frequency[1] > frequency[0]
            pairs = torch.where(swap[None], pairs.flip(0), pairs)
            order = pairs[0].argsort(stable=True)
            pairs = pairs[:, order]
        output = []
        for batch in tqdm(
            pairs.split(batch_size, dim=1),
            desc="Scoring link queries",
            disable=pairs.shape[1] < 100_000,
        ):
            features = self.features(batch)
            output.append(
                predict_residual(features, self.head(features).squeeze(1), fitted)
            )
        scores = torch.cat(output)
        if order is not None:
            restored = torch.empty_like(scores)
            restored[order] = scores
            return restored
        return scores


def uniform_nonedges(n, edges, count, seed=42):
    """Training-only rejection sampling with a sorted edge index, not a Python set."""
    keys = (edges[0] * n + edges[1]).numpy()
    rng = np.random.default_rng(seed)
    selected = set()
    while len(selected) < count:
        a, b = np.sort(
            rng.integers(n, size=(2, max(1024, 2 * (count - len(selected))))), axis=0
        )
        proposed = a * n + b
        index = np.searchsorted(keys, proposed)
        valid = (a != b) & (
            (index == len(keys)) | (keys[np.minimum(index, len(keys) - 1)] != proposed)
        )
        selected.update(proposed[valid].tolist())
    result = np.array(sorted(selected))
    result = result[rng.permutation(len(result))[:count]]
    return torch.from_numpy(np.stack((result // n, result % n)))


def fit_large_links(model, head, raw, x, edges):
    """Four full masked views; same 4096 fit and 2048 calibration examples."""
    generator = torch.Generator().manual_seed(42)
    count = edges.shape[1]
    global_order = torch.randperm(count, generator=generator)
    nheld = max(1, int(0.05 * count))
    fixed_held, eligible = global_order[:nheld], global_order[nheld:]
    neg = uniform_nonedges(len(raw), edges, 12288)
    neg_held, neg_train = neg[:, :2048], neg[:, 2048:]
    train, held = [], []
    train_ids, held_ids = set(), set()
    for view in range(4):
        print("Calibration full masked view", view + 1, "/4", flush=True)
        order = eligible[torch.randperm(len(eligible), generator=generator)]
        cut = int(0.2 * count) - nheld
        support = edges[:, order[cut:]]
        # Select labeled examples before feature calculation, retaining full support edges.
        pos_ids = order[:cut][torch.randperm(cut, generator=generator)[:512]]
        val_ids = fixed_held[torch.randperm(nheld, generator=generator)[:256]]
        train_pairs = torch.cat(
            (
                edges[:, pos_ids],
                neg_train[
                    :, torch.randperm(neg_train.shape[1], generator=generator)[:512]
                ],
            ),
            dim=1,
        )
        held_pairs = torch.cat(
            (
                edges[:, val_ids],
                neg_held[
                    :, torch.randperm(neg_held.shape[1], generator=generator)[:256]
                ],
            ),
            dim=1,
        )
        train_y, held_y = torch.cat(
            (torch.ones(len(pos_ids)), torch.zeros(512))
        ), torch.cat((torch.ones(len(val_ids)), torch.zeros(256)))
        scorer = LinkScorer(model, head, raw, x, support, 0.8)
        with torch.no_grad():
            tf, hf = [
                torch.cat([scorer.features(batch) for batch in pairs.split(256, dim=1)])
                for pairs in (train_pairs, held_pairs)
            ]
            train.append((tf, head(tf).squeeze(1), train_y))
            held.append((hf, head(hf).squeeze(1), held_y))
        train_ids.update((train_pairs[0] * len(raw) + train_pairs[1]).tolist())
        held_ids.update((held_pairs[0] * len(raw) + held_pairs[1]).tolist())
        del scorer
    if train_ids & held_ids:
        raise RuntimeError("Calibration training/holdout overlap")
    tf, to, ty = [torch.cat([row[i] for row in train]) for i in range(3)]
    hf, ho, hy = [torch.cat([row[i] for row in held]) for i in range(3)]
    choices = []
    for strength in (0.1, 1.0, 10.0, 100.0):
        fitted = fit_residual(tf, to, ty, strength)
        loss = float(
            F.binary_cross_entropy_with_logits(
                predict_residual(hf, ho, fitted), hy.double()
            )
        )
        choices.append((loss, strength, fitted))
    loss, strength, fitted = min(choices, key=lambda row: row[0])
    return fitted, dict(
        selected_ridge=strength,
        calibration_holdout_bce=loss,
        train_pairs=len(ty),
        holdout_pairs=len(hy),
        unique_train_pairs=len(train_ids),
        unique_holdout_pairs=len(held_ids),
    )
