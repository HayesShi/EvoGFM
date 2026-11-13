"""Full-edge float64 moments used by the paper's continual experiments.

The library's default sampled estimator remains unchanged. This subclass uses
the exact estimator from the experimental snapshot, in blocks of 32,768 edges.
"""

import math

import torch

from evogfm.model import GraphState, InverseLawGFM, symmetric_eigh
from evogfm.readout import new_head


class FullGraphGFM(InverseLawGFM):
    def state(self, x, edges, observation_rate=1.0, directed=False, budget=None):
        z = self.encode_features(x)
        with torch.no_grad():
            n = len(z)
            e = edges
            pop = n * (n - 1) if directed else n * (n - 1) / 2
            rho = z.new_tensor(len(e[0]) / max(1.0, observation_rate * pop)).clamp(
                1e-8, 1 - torch.finfo(z.dtype).eps
            )
            if budget is not None and e.shape[1] > budget:
                dtype = torch.float64 if e.shape[1] > 2**24 else torch.float32
                ix = torch.linspace(
                    0, e.shape[1] - 1, budget, device=e.device, dtype=dtype
                ).long()
                e = e[:, ix]
            if e.shape[1] == 0:
                raise ValueError("empty visible graph")
            c = torch.zeros(self.d, self.d, dtype=torch.float64, device=z.device)
            for block in e.split(32768, dim=1):
                c += z[block[0]].double().T @ z[block[1]].double()
            c = (rho.double() * c / e.shape[1]).to(z)
            if directed:
                u, g, vh = torch.linalg.svd(c.float(), full_matrices=False)
                v = vh.T
            else:
                c = (c + c.T) / 2
                g, u = symmetric_eigh(c)
                v = u
            radius = z.norm(dim=-1) / math.sqrt(self.d)
            thresholds = radius.new_tensor([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
            radial = torch.stack([(radius <= t).float().mean() for t in thresholds])
        return GraphState(z, u.to(z), g.to(z), v.to(z), rho, radial.to(z), directed)


def load_checkpoint(path):
    """Load the original weights with the full-edge experimental estimator."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = dict(checkpoint["model_config"])
    backbone = config.pop("joint_task_backbone", False)
    model = FullGraphGFM(**config)
    model.config["joint_task_backbone"] = backbone
    model.load_state_dict(checkpoint["model"])
    model.eval()
    head = new_head(backbone)
    head.load_state_dict(checkpoint["head"])
    head.eval()
    return model, head, checkpoint
