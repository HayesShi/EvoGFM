"""GraphLand RL archives, following the author's transductive preprocessing."""

from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
from .data import canonical_edges


def load_graphland(name, data_root="data"):
    archive = Path(data_root) / "raw" / "graphland" / f"{name}.zip"
    with zipfile.ZipFile(archive) as source:
        info = yaml.safe_load(source.read(f"{name}/info.yaml"))

        def table(filename, index=True):
            with source.open(f"{name}/{filename}.csv") as stream:
                return pd.read_csv(stream, index_col=0 if index else None)

        frame, masks = table("features"), table("split_masks_RL")
        targets = table("targets").values[:, 0].astype(np.float32)
        edges = table("edgelist", False).values[:, :2].T
    fractions = info["fraction_features_names"]
    numerical = [
        key for key in info["numerical_features_names"] if key not in fractions
    ]
    blocks = []
    for keys, quantile in [(numerical, True), (fractions, False)]:
        if not keys:
            continue
        values = frame[keys].values.astype(np.float32)
        if quantile:
            values = QuantileTransformer(
                n_quantiles=min(1000, len(values)),
                output_distribution="normal",
                subsample=1_000_000_000,
                random_state=0,
            ).fit_transform(values)
        if np.isnan(values).any():
            values = SimpleImputer(strategy="most_frequent").fit_transform(values)
        blocks.append(values)
    keys = info["categorical_features_names"]
    if keys:
        values = frame[keys].values.astype(np.float32)
        blocks.append(
            OneHotEncoder(
                drop="if_binary", sparse_output=False, dtype=np.float32
            ).fit_transform(values)
        )
    x = torch.from_numpy(np.concatenate(blocks, axis=1)).float()
    if not torch.isfinite(x).all():
        raise ValueError("Nonfinite GraphLand features")
    labeled = np.isfinite(targets)
    return dict(
        x=x,
        y=torch.from_numpy(targets),
        edges=canonical_edges(edges, len(x)),
        **{
            key: torch.from_numpy(np.flatnonzero(masks[key].values & labeled))
            for key in ("train", "val", "test")
        },
        task=info["task"],
        preprocessing="author RL: all-visible feature quantiles/one-hot; most-frequent imputation; undirected; no row-L1 normalization",
    )
