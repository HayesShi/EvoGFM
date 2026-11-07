"""Full graph inference on an 8 GB GPU, retaining graph/frame storage on CPU."""

import copy
import math

import torch
import torch.nn.functional as F


def cuda_csr(matrix, device="cuda"):
    return torch.sparse_csr_tensor(
        torch.from_numpy(matrix.indptr).to(device),
        torch.from_numpy(matrix.indices).to(device),
        torch.from_numpy(matrix.data).to(device),
        size=matrix.shape,
    )


@torch.no_grad()
def embed_graph_offloaded(model, state, visible, node_batch=32768, edge_batch=1048576):
    """All nodes, all edges and all programmed heads; stream edge indices to CUDA.

    Attention is evaluated per head. CSR row reductions and value aggregation
    use the complete graph, with no node/edge sampling or context truncation.
    """
    if not model.frame_aligned or model.mode != "coupled":
        raise ValueError("CUDA offloading requires the frame-aligned coupled model")
    gpu = copy.deepcopy(model).cuda().eval()
    n = len(state.z)
    target, source = visible
    degree = torch.bincount(target, minlength=n)
    offsets = torch.cat((degree.new_zeros(1), degree.cumsum(0))).cuda()
    columns = source.to(torch.int32).cuda()
    csr_offsets = offsets.to(torch.int32)
    left, s, right, _ = model.attention_program(state)
    scale = len(s) ** 0.25
    wq = (scale * left * s.abs().clamp_min(1e-20).sqrt()[None, :]).cuda()
    wk = (scale * right * (s.sign() * s.abs().clamp_min(1e-20).sqrt())[None, :]).cuda()
    h = torch.empty(n, model.config["hidden"], device="cuda")
    for start in range(0, n, node_batch):
        h[start : start + node_batch] = gpu.up(
            state.z[start : start + node_batch].cuda()
        )

    for cpu_layer, layer in zip(model.layers, gpu.layers):
        q = torch.empty(n, len(s), device="cuda")
        k = torch.empty_like(q)
        v = torch.empty(n, layer.heads, layer.rank, device="cuda")
        for start in range(0, n, node_batch):
            stop = min(n, start + node_batch)
            frame = state.z[start:stop].cuda()
            q[start:stop], k[start:stop] = frame @ wq, frame @ wk
            v[start:stop] = layer.value(layer.norm1(h[start:stop])).reshape(
                -1, layer.heads, layer.rank
            )
        temperature = (F.softplus(cpu_layer.log_temperature) + 0.05).sqrt().cuda()
        out = torch.empty_like(v)
        for head in range(layer.heads):
            weights = torch.empty(len(target), device="cuda")
            for start in range(0, len(target), edge_batch):
                stop = min(len(target), start + edge_batch)
                a, b = target[start:stop].cuda(), source[start:stop].cuda()
                weights[start:stop] = (
                    (q[a] * temperature[head]) * (k[b] * temperature[head])
                ).sum(-1) / math.sqrt(len(s))
            maximum = torch.segment_reduce(weights, "max", offsets=offsets)
            for start in range(0, len(target), edge_batch):
                stop = min(len(target), start + edge_batch)
                a = target[start:stop].cuda()
                weights[start:stop].sub_(maximum[a]).exp_()
            denominator = torch.segment_reduce(weights, "sum", offsets=offsets)
            for start in range(0, len(target), edge_batch):
                stop = min(len(target), start + edge_batch)
                a = target[start:stop].cuda()
                weights[start:stop].div_(denominator[a].clamp_min(1e-12))
            matrix = torch.sparse_csr_tensor(csr_offsets, columns, weights, size=(n, n))
            out[:, head] = torch.sparse.mm(matrix, v[:, head].contiguous())
        del q, k, v, matrix, weights
        for start in range(0, n, node_batch):
            stop = min(n, start + node_batch)
            block = h[start:stop]
            block.add_(layer.output(out[start:stop].reshape(stop - start, -1)))
            block.add_(layer.ff(layer.norm2(block)))
        del out
    return torch.cat([gpu.norm(block).cpu() for block in h.split(node_batch)])
