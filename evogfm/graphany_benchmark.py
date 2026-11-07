"""GraphAny's official Wisconsin weights and CPU gelss, with SciPy mean aggregation.

This is a port of the graph preprocessing for Windows without DGL. Model and
entropy normalization execute the original sources from the supplied repository.
"""

import ast
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold._utils import _binary_search_perplexity


def load_graphany(root, device="cpu"):
    root = Path(root)
    namespace = dict(
        np=np,
        torch=torch,
        th=torch,
        nn=nn,
        F=F,
        rearrange=rearrange,
        pdist=pdist,
        squareform=squareform,
        sklearn_binary_search_perplexity=_binary_search_perplexity,
    )
    for filename, names in (
        ("data.py", {"get_entropy_normed_cond_gaussian_prob"}),
        ("model.py", {"GraphAny", "MLP"}),
    ):
        path = root / "graphany" / filename
        tree = ast.parse(path.read_text(encoding="utf-8"))
        tree.body = [node for node in tree.body if getattr(node, "name", None) in names]
        exec(compile(tree, str(path), "exec"), namespace)
    model = namespace["GraphAny"](
        n_hidden=32,
        feat_channels=["X", "L1", "L2", "H1", "H2"],
        pred_channels=["X", "L1", "L2"],
        att_temperature=5,
        entropy=1,
        n_mlp_layer=2,
    )
    checkpoint = torch.load(
        root / "checkpoints/graph_any_wisconsin.pt",
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(
        {
            key.removeprefix("gnn_model."): value
            for key, value in checkpoint["state_dict"].items()
        }
    )
    return model.eval().requires_grad_(False).to(device)


def graphany_predict(
    model, graph, train, root, device, stage, propagation_device="cpu"
):
    x, n = graph["x"], len(graph["x"])

    def propagation():
        edges = graph["edges"].numpy()
        row, col = np.r_[edges[0], edges[1]], np.r_[edges[1], edges[0]]
        matrix = sp.csr_matrix(
            (np.ones(len(row), dtype=np.float32), (row, col)), shape=(n, n)
        )
        degree = np.asarray(matrix.sum(1)).ravel().clip(1)
        matrix.data /= np.repeat(degree, np.diff(matrix.indptr))
        features = {"X": x}
        if propagation_device == "cuda":
            from .cuda_inference import cuda_csr

            gpu_matrix = cuda_csr(matrix)
            low, high = x.cuda(), x.cuda()
            for hop in (1, 2):
                low = torch.sparse.mm(gpu_matrix, low)
                high = high - torch.sparse.mm(gpu_matrix, high)
                features[f"L{hop}"] = low.cpu()
                features[f"H{hop}"] = high.cpu()
            return features
        low, high = x.numpy(), x.numpy()
        for hop in (1, 2):
            low = matrix @ low
            high = high - matrix @ high
            features[f"L{hop}"] = torch.from_numpy(low)
            features[f"H{hop}"] = torch.from_numpy(high)
        return features

    features = stage("feature_propagation", propagation)

    def fit(features):
        target = F.one_hot(graph["y"][train], int(graph["y"].max()) + 1).float()
        predictions = {}
        for channel in model.feat_channels:
            value = features[channel]
            weight = torch.linalg.lstsq(value[train], target, driver="gelss")[0]
            predictions[channel] = value @ weight
        return predictions

    channels = stage("linear_readout", lambda features=features: fit(features))
    del features

    def predict():
        blocks = []
        for start in range(0, n, 100000):
            cpu_inputs = {
                key: value[start : start + 100000] for key, value in channels.items()
            }
            distance = model.compute_dist(
                torch.stack([cpu_inputs[key] for key in model.feat_channels], dim=1)
            )
            inputs = {key: value.to(device) for key, value in cpu_inputs.items()}
            blocks.append(model(inputs, dist=distance.to(device))[0].cpu())
        return torch.cat(blocks)

    scores = stage("entropy_and_model", predict)
    return scores, dict(
        candidate_fits=5,
        backbone_parameters=sum(p.numel() for p in model.parameters()),
        implementation=(
            "official model/entropy/gelss; SciPy graph construction + CUDA CSR mean propagation"
            if propagation_device == "cuda"
            else "official model/entropy/gelss; SciPy mean propagation port"
        ),
    )
