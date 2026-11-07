"""One graph-wide feature frame and inverse-programmed task representations."""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from .data import canonical_edges


def global_svd_features(x, dim=100):
    """Exact graph-wide SVD; never fit independent bases on node chunks.

    Sign canonicalization removes the arbitrary sign of each singular vector.
    It does not align the semantics of different graphs' feature spaces.
    """
    x = torch.as_tensor(x).float()
    if x.shape[1] <= dim:
        return F.pad(x, (0, dim - x.shape[1])).contiguous()
    if not x.is_cuda and x.numel() > 50_000_000 and x.shape[0] >= x.shape[1]:
        basis = covariance_svd_basis(x, dim)
    else:
        options = {"driver": "gesvd"} if x.is_cuda else {}
        _, _, vh = torch.linalg.svd(x, full_matrices=False, **options)
        basis = vh[: min(dim, len(vh))].T
    pivot = basis.abs().argmax(0)
    sign = basis[pivot, torch.arange(basis.shape[1], device=x.device)].sign()
    basis = basis * sign.unsqueeze(0)
    result = x @ basis
    return F.pad(result, (0, dim - result.shape[1])).contiguous()


def covariance_svd_basis(x, dim):
    """Exact right singular frame from the complete float64 Gram matrix."""
    gram = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float64)
    for block in x.split(1024):
        value = block.double()
        gram += value.T @ value
    _, vectors = torch.linalg.eigh(gram)
    return vectors[:, -min(dim, len(vectors)) :].flip(1).float()


def adjacency(edges, n):
    a, b = edges.numpy()
    matrix = sp.csr_matrix(
        (np.ones(2 * len(a)), (np.r_[a, b], np.r_[b, a])), shape=(n, n)
    )
    return matrix


def propagate(x, edges):
    matrix = adjacency(edges, len(x)) + sp.eye(len(x))
    inv = np.asarray(matrix.sum(1)).ravel() ** -0.5
    matrix = sp.diags(inv) @ matrix @ sp.diags(inv)
    return torch.from_numpy(matrix @ x.numpy()).float()


def pair_context(x, edges):
    """Cache query-independent content for large candidate pools."""
    a = adjacency(edges, len(x)).tocsr()
    degree = np.asarray(a.sum(1)).ravel()
    raw = F.normalize(x, dim=1)
    p1 = F.normalize(propagate(x, edges), dim=1)
    p2 = F.normalize(propagate(p1, edges), dim=1)
    return a, degree, raw, p1, p2


def hard_negative_pool(edges, n, per_node=32):
    """Training-graph nonedges with large resource-allocation scores."""
    a = adjacency(edges, n).tocsr()
    degree = np.asarray(a.sum(1)).ravel()
    weighted = a.multiply(1 / np.maximum(1, degree)[None, :]).tocsr()
    pairs = []
    for node in range(n):
        row = (weighted[node] @ a).tocsr()
        valid = (row.indices > node) & ~np.isin(row.indices, a[node].indices)
        candidates, scores = row.indices[valid], row.data[valid]
        order = np.lexsort((candidates, -scores))[:per_node]
        pairs.extend((node, int(j)) for j in candidates[order])
    return (
        torch.tensor(pairs, dtype=torch.long).T.contiguous() if pairs else edges[:, :0]
    )


def pair_channels(x, edges, pairs, batch_size=256, prepared=None):
    """Train-visible topology and label-free content, never query labels."""
    a, degree, raw, p1, p2 = pair_context(x, edges) if prepared is None else prepared
    i, j = pairs
    ip, jp = i.numpy().copy(), j.numpy().copy()
    # Compute only requested entries. Full A² and A³ can become dense even
    # when the original graph is sparse.
    inv_degree = 1 / np.maximum(1, degree)
    inv_log_degree = 1 / np.log(np.maximum(2, degree))
    structural = []
    for start in range(0, len(ip), batch_size):
        left = a[ip[start : start + batch_size]]
        right = a[jp[start : start + batch_size]]
        common = left.multiply(right)
        columns = [
            np.asarray(common.sum(1)).ravel(),
            np.asarray(common @ inv_degree).ravel(),
            np.asarray(common @ inv_log_degree).ravel(),
            np.asarray((left @ a).multiply(right).sum(1)).ravel(),
        ]
        structural.append(torch.from_numpy(np.stack(columns, axis=1)).float())
    structure = torch.cat(structural) if structural else x.new_empty((0, 4))

    deg = torch.from_numpy(degree).float().log1p()
    return torch.stack(
        (
            (raw[i] * raw[j]).sum(1),
            (p1[i] * p1[j]).sum(1),
            (p2[i] * p2[j]).sum(1),
            (raw[i] * p1[j] + p1[i] * raw[j]).sum(1) / 2,
            torch.minimum(deg[i], deg[j]),
            torch.maximum(deg[i], deg[j]),
            structure[:, 0].log1p(),
            structure[:, 1].log1p(),
            structure[:, 2].log1p(),
            structure[:, 3].log1p(),
        ),
        dim=1,
    )


def node_representations(
    model, state, raw, edges, hops=4, uniform=False, return_transition=False
):
    """Keep propagation orders separate so the readout can retain local signal."""
    n = len(raw)
    visible = torch.cat((edges, edges.flip(0)), dim=1)
    visible = visible[:, (visible[0] * n + visible[1]).argsort()]
    weight = attention(model, state, visible, uniform)
    h = F.normalize(model.embed_graph(state, visible), dim=1) if n > 100_000 else None
    value = raw
    parts = [F.normalize(value, dim=1)]
    for _ in range(hops):
        value = weighted_propagate(value, visible, weight)
        parts.append(F.normalize(value, dim=1))
    if h is None:
        h = F.normalize(model.embed_graph(state, visible), dim=1)
    if return_transition:
        degree = torch.bincount(visible[0], minlength=n)
        offsets = torch.cat((degree.new_zeros(1), degree.cumsum(0)))
        transition = sp.csr_matrix(
            (weight.detach().numpy(), visible[1].numpy(), offsets.numpy()), shape=(n, n)
        )
        return parts, h, transition
    return parts, h


def weighted_propagate(value, visible, weight):
    """Identical full-edge aggregation without an E-by-feature temporary at inference."""
    if not torch.is_grad_enabled() and visible.shape[1] * value.shape[1] > 8_000_000:
        matrix = sp.csr_matrix(
            (weight.numpy(), tuple(visible.numpy())), shape=(len(value), len(value))
        )
        return torch.from_numpy(matrix @ value.numpy())
    return value.new_zeros(value.shape).index_add_(
        0, visible[0], weight[:, None] * value[visible[1]]
    )


def label_transition(model, state, edges):
    """The same inverse-programmed row-stochastic affinity used by propagation."""
    visible = contexts(edges, len(state.z), edges[:, :0])["visible"]
    weight = attention(model, state, visible)
    return sp.csr_matrix(
        (weight.detach().numpy(), tuple(visible.numpy())),
        shape=(len(state.z), len(state.z)),
    )


def preprocess(data):
    raw = data["x"]
    raw = F.normalize(raw, p=1, dim=1) if float(raw.min()) >= 0 else raw
    return raw, global_svd_features(raw)


def contexts(edges, n, pairs):
    """Only visible train edges; store common-neighbor edge indices per query."""
    edges = canonical_edges(edges, n)
    visible = torch.cat((edges, edges.flip(0)), dim=1)
    order = (visible[0] * n + visible[1]).argsort()
    visible = visible[:, order]
    degree = torch.bincount(visible[0], minlength=n)
    offsets = torch.cat((degree.new_zeros(1), degree.cumsum(0))).numpy()
    neighbors = visible[1].numpy()
    query = []
    left = []
    right = []
    for index, (i, j) in enumerate(pairs.T.tolist()):
        _, li, ri = np.intersect1d(
            neighbors[offsets[i] : offsets[i + 1]],
            neighbors[offsets[j] : offsets[j + 1]],
            return_indices=True,
        )
        query.extend([index] * len(li))
        left.extend(li + offsets[i])
        right.extend(ri + offsets[j])
    return dict(
        visible=visible,
        degree=degree.float(),
        cn_query=torch.tensor(query, dtype=torch.long),
        cn_left=torch.tensor(left, dtype=torch.long),
        cn_right=torch.tensor(right, dtype=torch.long),
    )


def attention(model, state, visible, uniform=False):
    target, source = visible
    n = len(state.z)
    if uniform:
        return state.z.new_ones(len(target)) / torch.bincount(target, minlength=n)[
            target
        ].clamp_min(1)
    left, s, right, _ = model.attention_program(state)
    q = state.z @ left
    k = state.z @ right
    if not torch.is_grad_enabled() and len(target) > 100_000:
        score = torch.cat(
            [
                ((q[a] * k[b]) * s).sum(1)
                for a, b in zip(target.split(32768), source.split(32768))
            ]
        )
    else:
        score = ((q[target] * k[source]) * s).sum(1)
    temperature = F.softplus(model.layers[0].log_temperature).mean() + 0.05
    score = score * temperature
    maximum = score.new_full((n,), -torch.inf)
    maximum.scatter_reduce_(0, target, score.detach(), reduce="amax", include_self=True)
    numerator = (score - maximum[target]).exp()
    denominator = score.new_zeros(n).index_add_(0, target, numerator)
    return numerator / denominator[target].clamp_min(1e-12)


def task_features(
    model, state, raw, edges, pairs, ctx, base, uniform=False, prepared=None
):
    """All learned affinity comes from the shared inverse-generated Q/K."""
    v = ctx["visible"]
    if prepared is None:
        w = attention(model, state, v, uniform)
        p1 = raw.new_zeros(raw.shape).index_add_(0, v[0], w[:, None] * raw[v[1]])
        p2 = raw.new_zeros(raw.shape).index_add_(0, v[0], w[:, None] * p1[v[1]])
        z0, z1, z2 = [F.normalize(x, dim=1) for x in [raw, p1, p2]]
    else:
        w, parts = prepared
        z0, z1, z2 = parts[:3]
    i, j = pairs
    shared = w.new_zeros(pairs.shape[1]).index_add_(
        0, ctx["cn_query"], w[ctx["cn_left"]] * w[ctx["cn_right"]]
    )
    # Convert row probabilities back to a degree-comparable weighted CN count.
    weighted_count = shared * ctx["degree"][i] * ctx["degree"][j]
    norm = w.new_zeros(len(raw)).index_add_(0, v[0], w.square()).sqrt()
    normalized_shared = shared / (norm[i] * norm[j]).clamp_min(1e-8)
    _, amplitude, _, _ = model.programmed(state)
    relation = model.relation_logits(
        state, pairs
    ) / amplitude.square().sum().sqrt().clamp_min(1e-6)
    extra = torch.stack(
        (
            (z1[i] * z1[j]).sum(1),
            (z2[i] * z2[j]).sum(1),
            (z0[i] * z1[j] + z1[i] * z0[j]).sum(1) / 2,
            relation.tanh(),
            weighted_count.clamp_min(0).log1p(),
            normalized_shared,
        ),
        dim=1,
    )
    if uniform:
        extra = torch.cat(
            (extra[:, :3], torch.zeros_like(extra[:, 3:4]), extra[:, 4:]), dim=1
        )
    parts = (z0, z1, z2)
    if model.config.get("joint_task_backbone", False):
        h = (
            F.normalize(model.embed_graph(state, v), dim=1)
            if prepared is None
            else prepared[1][-1]
        )
        extra = torch.cat((extra, (h[i] * h[j]).sum(1, keepdim=True)), dim=1)
        parts = parts + (h,)
    return torch.cat((base, extra), dim=1), parts
