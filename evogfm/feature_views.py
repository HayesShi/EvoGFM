"""Lazy blocks of the same readout features, for graphs with wide attributes."""

import torch
import torch.nn.functional as F


class FeatureView:
    def __init__(self, rows, columns, getter):
        self.shape = (rows, columns)
        self.getter = getter

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, ids):
        return self.getter(ids)

    def split(self, size):
        for start in range(0, len(self), size):
            yield self[start : min(len(self), start + size)]


def join_features(parts, values, weight):
    columns = sum(part.shape[1] for part in parts) + values.shape[1]
    if len(values) * columns <= 8_000_000:
        return torch.cat(parts + [weight * values], dim=1)
    return FeatureView(
        len(values),
        columns,
        lambda ids: torch.cat(
            [part[ids] for part in parts] + [weight * values[ids]], dim=1
        ),
    )


def quadratic_features(features, dimension):
    generator = torch.Generator().manual_seed(42)
    w = torch.randn(features.shape[1], dimension, generator=generator)
    v = torch.randn(features.shape[1], dimension, generator=generator)

    def transform(ids):
        base = features[ids]
        x = F.normalize(base, dim=1)
        return torch.cat((base, (x @ w) * (x @ v) / dimension**0.5), dim=1)

    if isinstance(features, FeatureView):
        return FeatureView(len(features), features.shape[1] + dimension, transform)
    return transform(slice(None))


def rbf_features(features, gamma, landmarks=256):
    ids = torch.randperm(len(features), generator=torch.Generator().manual_seed(42))[
        :landmarks
    ]
    centers = F.normalize(features[ids], dim=1)
    gram = (-gamma * (2 - 2 * centers @ centers.T).clamp_min(0)).exp().double()
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    basis = (eigenvectors / eigenvalues.clamp_min(1e-6).sqrt()[None, :]).float()

    def transform(rows):
        base = features[rows]
        x = F.normalize(base, dim=1)
        cross = (-gamma * (2 - 2 * x @ centers.T).clamp_min(0)).exp()
        return torch.cat((base, cross @ basis), dim=1)

    if isinstance(features, FeatureView):
        return FeatureView(len(features), features.shape[1] + len(ids), transform)
    return transform(slice(None))
