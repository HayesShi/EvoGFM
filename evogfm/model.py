from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class GraphState:
    z: torch.Tensor
    left: torch.Tensor
    spectrum: torch.Tensor
    right: torch.Tensor
    density: torch.Tensor
    radial: torch.Tensor
    directed: bool = False


class InverseLaw(nn.Module):
    """Coupled, permutation-equivariant spectral features, linear online coefficients."""

    def __init__(self, k=32, context=32, mode="coupled"):
        super().__init__()
        self.k, self.mode = k, mode
        self.phi = nn.Sequential(
            nn.Linear(1, context), nn.SiLU(), nn.Linear(context, context)
        )
        self.net = nn.Sequential(
            nn.Linear(context + 10, k * 2), nn.SiLU(), nn.Linear(k * 2, k)
        )
        self.local = nn.Sequential(nn.Linear(3, k * 2), nn.SiLU(), nn.Linear(k * 2, k))
        self.bias_features = nn.Sequential(
            nn.Linear(context + 9, 32), nn.SiLU(), nn.Linear(32, 6), nn.Tanh()
        )
        self.register_buffer("knots", torch.linspace(0, math.log(1e4), k))
        self.c = nn.Parameter(self.knots.exp().clone())
        self.beta = nn.Parameter(torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        if mode == "raw":
            self.log_raw_scale = nn.Parameter(torch.tensor(math.log(100.0)))

    def features(self, state):
        gamma = state.spectrum
        # 1-1e-8 rounds to 1 in float32 and makes logit(rho) infinite.
        rho = state.density.clamp(1e-8, 1 - torch.finfo(state.density.dtype).eps)
        lg = torch.log1p(gamma.square() * 1e6).unsqueeze(-1)
        context = self.phi(lg).mean(0, keepdim=True).expand(len(gamma), -1)
        radial = state.radial.reshape(1, -1).expand(len(gamma), -1)
        density = (
            torch.stack((rho, torch.logit(rho))).reshape(1, 2).expand(len(gamma), -1)
        )
        if self.mode == "coordinate":
            logits = self.local(torch.cat((lg, density), dim=1))
        else:
            logits = self.net(torch.cat((lg, context, density, radial), dim=1))
        # Fixed initialization prior; knots do not change when c changes online.
        prior = -0.5 * (self.knots + rho.log()).square()
        q = torch.softmax(logits + prior, dim=-1)
        bias_input = torch.cat((context[0], density[0], state.radial))
        chi = torch.cat(
            (rho.new_ones(1), torch.logit(rho).view(1), self.bias_features(bias_input))
        )
        return q, chi

    def amplitude(self, state):
        q, chi = self.features(state)
        if self.mode == "raw":
            s = state.spectrum * self.log_raw_scale.clamp(-10, 12).exp()
        else:
            s = state.spectrum * (q @ self.c)
        return s, self.beta @ chi

    def moment_spectrum(self, state):
        return state.spectrum

    def theta(self):
        return torch.cat((self.c, self.beta))

    @torch.no_grad()
    def set_theta(self, theta):
        self.c.copy_(theta[: self.k])
        self.beta.copy_(theta[self.k :])

    @torch.no_grad()
    def constrain(self):
        self.c.clamp_(0, 1e5)
        self.beta.clamp_(-5, 5)

    def bounds(self):
        return ([0.0] * self.k + [-5.0] * 8, [1e5] * self.k + [5.0] * 8)


class ConditionalHatInverse(InverseLaw):
    """Same shared coupled inverse, with density-normalized, resolved bases.

    Hat functions separate spectral coordinates before a learned coupled tilt.
    Online coefficients remain nonnegative, and pair logits remain affine.
    """

    def __init__(self, k=32, context=32, mode="coupled"):
        super().__init__(k, context, mode)
        self.knots.copy_(torch.linspace(0.0, 1.0, k))
        with torch.no_grad():
            self.c.fill_(1.0)
            self.beta.zero_()

    def moment_spectrum(self, state):
        return state.spectrum / state.density.clamp_min(1e-8)

    def features(self, state):
        gamma = self.moment_spectrum(state)
        rho = state.density.clamp(1e-8, 1 - torch.finfo(state.density.dtype).eps)
        magnitude = gamma.abs()
        relative = magnitude / magnitude.square().mean().sqrt().clamp_min(1e-6)
        lg = torch.log1p(relative.square()).unsqueeze(-1)
        context = self.phi(lg).mean(0, keepdim=True).expand(len(gamma), -1)
        radial = state.radial.reshape(1, -1).expand(len(gamma), -1)
        density = (
            torch.stack((rho, torch.logit(rho))).reshape(1, 2).expand(len(gamma), -1)
        )
        logits = self.net(torch.cat((lg, context, density, radial), dim=1))
        # The max and RMS depend on the whole spectrum, preserving coupling.
        location = lg[:, 0] / lg[:, 0].max().clamp_min(1e-6)
        hats = (1 - (location[:, None] - self.knots).abs() * (self.k - 1)).clamp_min(0)
        q = torch.softmax(hats.clamp_min(1e-5).log() + 0.1 * logits, dim=-1)
        bias_input = torch.cat((context[0], density[0], state.radial))
        chi = torch.cat(
            (rho.new_ones(1), torch.logit(rho).view(1), self.bias_features(bias_input))
        )
        return q, chi

    def amplitude(self, state):
        q, chi = self.features(state)
        return self.moment_spectrum(state) * (q @ self.c), self.beta @ chi


class ProgrammedLayer(nn.Module):
    def __init__(
        self, hidden, d, heads, rank, dropout=0.1, mode="coupled", frame_aligned=False
    ):
        super().__init__()
        self.heads, self.rank, self.mode = heads, rank, mode
        self.norm1, self.norm2 = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.inputs = nn.ModuleList(
            [nn.Linear(hidden, d, bias=False) for _ in range(heads)]
        )
        self.q = nn.Linear(hidden, heads * rank, bias=False)
        self.k = nn.Linear(hidden, heads * rank, bias=False)
        self.value = nn.Linear(hidden, heads * rank, bias=False)
        self.output = nn.Linear(heads * rank, hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 4, hidden),
        )
        self.dropout = dropout
        if mode == "hybrid":
            self.program_gate = nn.Parameter(torch.tensor(0.0))
        if frame_aligned:
            self.log_temperature = nn.Parameter(torch.linspace(-1.0, 1.0, heads))

    def forward(self, h, left, s, right, frame_z=None):
        single = h.ndim == 2
        if single:
            h = h.unsqueeze(0)
        if frame_z is not None and frame_z.ndim == 2:
            frame_z = frame_z.unsqueeze(0)
        x = self.norm1(h)
        batch, n = x.shape[:2]
        if self.mode == "vanilla":
            q = self.q(x).reshape(batch, n, self.heads, self.rank).transpose(1, 2)
            k = self.k(x).reshape(batch, n, self.heads, self.rank).transpose(1, 2)
        else:
            rank = len(s)
            scale = rank**0.25
            wq = scale * left * s.abs().clamp_min(1e-20).sqrt()[None, :]
            wk = scale * right * (s.sign() * s.abs().clamp_min(1e-20).sqrt())[None, :]
            if frame_z is None:
                q = torch.stack([p(x) @ wq for p in self.inputs], dim=1)
                k = torch.stack([p(x) @ wk for p in self.inputs], dim=1)
            else:
                temperature = (F.softplus(self.log_temperature) + 0.05).sqrt()[
                    None, :, None, None
                ]
                q = (frame_z @ wq)[:, None] * temperature
                k = (frame_z @ wk)[:, None] * temperature
            if self.mode == "hybrid":
                base_q = (
                    self.q(x).reshape(batch, n, self.heads, self.rank).transpose(1, 2)
                )
                base_k = (
                    self.k(x).reshape(batch, n, self.heads, self.rank).transpose(1, 2)
                )
                total = self.rank + rank
                a = (total / self.rank) ** 0.25
                b = (total / rank) ** 0.25
                q = torch.cat((base_q * a, q * b * self.program_gate.tanh()), dim=-1)
                k = torch.cat((base_k * a, k * b), dim=-1)
        v = self.value(x).reshape(batch, n, self.heads, self.rank).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        h = h + F.dropout(
            self.output(out.transpose(1, 2).reshape(batch, n, -1)),
            self.dropout,
            self.training,
        )
        h = h + F.dropout(self.ff(self.norm2(h)), self.dropout, self.training)
        return h[0] if single else h

    def forward_graph(self, h, left, s, right, edge_index, frame_z=None):
        """Local graph attention with programmed Q/K and all visible edges.

        A development layout ablation. The historical BFS-block path is intact.
        edge_index includes both directions and self loops, exactly once each.
        """
        x = self.norm1(h)
        n = len(x)
        if self.mode == "vanilla":
            q = self.q(x).reshape(n, self.heads, self.rank)
            k = self.k(x).reshape(n, self.heads, self.rank)
        else:
            rank = len(s)
            scale = rank**0.25
            wq = scale * left * s.abs().clamp_min(1e-20).sqrt()[None, :]
            wk = scale * right * (s.sign() * s.abs().clamp_min(1e-20).sqrt())[None, :]
            if frame_z is None:
                q = torch.stack([p(x) @ wq for p in self.inputs], dim=1)
                k = torch.stack([p(x) @ wk for p in self.inputs], dim=1)
            else:
                temperature = (F.softplus(self.log_temperature) + 0.05).sqrt()[
                    None, :, None
                ]
                q = (frame_z @ wq)[:, None] * temperature
                k = (frame_z @ wk)[:, None] * temperature
        v = self.value(x).reshape(n, self.heads, self.rank)
        target, source = edge_index
        scores = (q[target] * k[source]).sum(-1) / math.sqrt(q.shape[-1])
        if self.mode == "hybrid":
            base_q = self.q(x).reshape(n, self.heads, self.rank)
            base_k = self.k(x).reshape(n, self.heads, self.rank)
            scores = (base_q[target] * base_k[source]).sum(-1) / math.sqrt(
                self.rank
            ) + self.program_gate.tanh() * scores
        index = target[:, None].expand(-1, self.heads)
        maximum = scores.new_full((n, self.heads), -torch.inf)
        maximum.scatter_reduce_(
            0, index, scores.detach(), reduce="amax", include_self=True
        )
        weight = (scores - maximum[target]).exp()
        denominator = weight.new_zeros(n, self.heads).scatter_add_(0, index, weight)
        weight = weight / denominator[target].clamp_min(1e-12)
        weight = F.dropout(weight, self.dropout, self.training)
        out = h.new_zeros(n, self.heads, self.rank)
        out.index_add_(0, target, weight[:, :, None] * v[source])
        h = h + F.dropout(self.output(out.reshape(n, -1)), self.dropout, self.training)
        h = h + F.dropout(self.ff(self.norm2(h)), self.dropout, self.training)
        return h


class InverseLawGFM(nn.Module):
    def __init__(
        self,
        input_dim=100,
        d=64,
        hidden=128,
        layers=3,
        heads=4,
        rank=16,
        k=32,
        dropout=0.1,
        mode="coupled",
        whiten=True,
        relation_gate_init=None,
        inverse_basis="legacy",
        frame_aligned=False,
        input_encoder="mlp",
        whiten_floor=1e-3,
        attention_normalization="none",
        joint_task_backbone=False,
    ):
        super().__init__()
        self.d, self.rank, self.whiten, self.mode = d, rank, whiten, mode
        self.frame_aligned = frame_aligned
        if attention_normalization not in ["none", "frobenius"]:
            raise ValueError("unknown attention normalization")
        self.attention_normalization = attention_normalization
        if frame_aligned and mode in ["vanilla", "hybrid"]:
            raise ValueError("aligned mainline uses programmed Q/K only")
        if input_encoder not in ["mlp", "identity"]:
            raise ValueError("unknown input encoder")
        if input_encoder == "identity" and input_dim != d:
            raise ValueError("identity encoder requires input_dim == d")
        self.encoder = (
            nn.Identity()
            if input_encoder == "identity"
            else nn.Sequential(nn.Linear(input_dim, d), nn.GELU(), nn.Linear(d, d))
        )
        self.whiten_floor = float(whiten_floor)
        if not 0 < self.whiten_floor <= 1:
            raise ValueError("whiten_floor must be in (0,1]")
        self.up = nn.Linear(d, hidden)
        self.layers = nn.ModuleList(
            [
                ProgrammedLayer(hidden, d, heads, rank, dropout, mode, frame_aligned)
                for _ in range(layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden)
        if inverse_basis not in ["legacy", "conditional_hat"]:
            raise ValueError("unknown inverse basis")
        self.inverse = (
            ConditionalHatInverse(k=k, mode=mode)
            if inverse_basis == "conditional_hat"
            else InverseLaw(k=k, mode=mode)
        )
        if mode == "fixed":
            self.fixed_left = nn.Parameter(torch.eye(d)[:, :rank].clone())
            self.fixed_right = nn.Parameter(torch.eye(d)[:, :rank].clone())
        self.link_decoder = nn.Sequential(
            nn.Linear(3 * hidden + 1, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        self.relation_gate = (
            None
            if relation_gate_init is None
            else nn.Parameter(torch.tensor(float(relation_gate_init)))
        )
        self.config = dict(
            input_dim=input_dim,
            d=d,
            hidden=hidden,
            layers=layers,
            heads=heads,
            rank=rank,
            k=k,
            dropout=dropout,
            mode=mode,
            whiten=whiten,
        )
        if relation_gate_init is not None:
            self.config["relation_gate_init"] = relation_gate_init
        if inverse_basis != "legacy":
            self.config["inverse_basis"] = inverse_basis
        if frame_aligned:
            self.config["frame_aligned"] = True
        if input_encoder != "mlp":
            self.config["input_encoder"] = input_encoder
        if whiten_floor != 1e-3:
            self.config["whiten_floor"] = whiten_floor
        if attention_normalization != "none":
            self.config["attention_normalization"] = attention_normalization
        if joint_task_backbone:
            self.config["joint_task_backbone"] = True

    def encode_features(self, x):
        z = self.encoder(x)
        # Statistics are a graph-conditioned analytical input; gradients through
        # the node encoder remain, gradients through sample whitening do not.
        mean = z.detach().mean(0, keepdim=True)
        z = z - mean
        if self.whiten:
            with torch.no_grad():
                cov = z.detach().T @ z.detach() / max(1, len(z))
                eig, u = torch.linalg.eigh(cov.float())
                floor = eig.max().clamp_min(1e-6) * self.whiten_floor
                w = (u * eig.clamp_min(floor).rsqrt()[None, :]) @ u.T
            z = z @ w.to(z)
        return z

    def state(self, x, edges, observation_rate=1.0, directed=False, budget=8192):
        z = self.encode_features(x)
        # Fixed state estimator is outside the source-optimizer differentiation.
        with torch.no_grad():
            n = len(z)
            e = edges
            pop = n * (n - 1) if directed else n * (n - 1) / 2
            rho = z.new_tensor(len(e[0]) / max(1.0, observation_rate * pop)).clamp(
                1e-8, 1 - torch.finfo(z.dtype).eps
            )
            if e.shape[1] > budget:
                # Deterministic quadrature over the fixed edge order, common to controls.
                ix = torch.linspace(0, e.shape[1] - 1, budget, device=e.device).long()
                e = e[:, ix]
            if e.shape[1] == 0:
                raise ValueError("empty visible graph")
            c = rho * (z[e[0]].T @ z[e[1]]) / e.shape[1]
            if directed:
                u, g, vh = torch.linalg.svd(c.float(), full_matrices=False)
                v = vh.T
            else:
                c = (c + c.T) / 2
                g, u = torch.linalg.eigh(c.float())
                v = u
            # Complete spectrum remains available to the coupled inverse.
            radius = z.norm(dim=-1) / math.sqrt(self.d)
            thresholds = radius.new_tensor([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
            radial = torch.stack([(radius <= t).float().mean() for t in thresholds])
        return GraphState(z, u.to(z), g.to(z), v.to(z), rho, radial.to(z), directed)

    def fixed_directions(self, state, count):
        left = torch.linalg.qr(self.fixed_left, mode="reduced").Q[:, :count]
        right = (
            torch.linalg.qr(self.fixed_right, mode="reduced").Q[:, :count]
            if state.directed
            else left
        )
        return left, right

    def programmed(self, state):
        s, b = self.inverse.amplitude(state)
        # Rank selection depends on observed gamma only, never online theta.
        ix = state.spectrum.abs().argsort(descending=True)[: self.rank]
        left, right = state.left[:, ix], state.right[:, ix]
        if self.mode == "fixed":
            left, right = self.fixed_directions(state, len(ix))
        return left, s[ix], right, b

    def attention_program(self, state):
        """Program Q/K at a graph-normalized scale; relation logits stay affine.

        For isotropic independent z_i,z_j, Var(z_i^T B z_j)=||B||_F^2.
        This controls the analytic attention scale without changing its signs.
        """
        left, s, right, b = self.programmed(state)
        if self.attention_normalization == "frobenius":
            s = s / s.square().sum().sqrt().clamp_min(1e-6)
        return left, s, right, b

    def relation_design(self, state, pairs):
        q, chi = self.inverse.features(state)
        ix = state.spectrum.abs().argsort(descending=True)[: self.rank]
        left, right = state.left[:, ix], state.right[:, ix]
        if self.mode == "fixed":
            left, right = self.fixed_directions(state, len(ix))
        a = state.z[pairs[0]] @ left
        b = state.z[pairs[1]] @ right
        a = (a * b) * self.inverse.moment_spectrum(state)[ix][None, :]
        return torch.cat((a @ q[ix], chi.expand(len(a), -1)), dim=1)

    def relation_logits(self, state, pairs):
        if self.mode == "raw":
            left, s, right, b = self.programmed(state)
            return ((state.z[pairs[0]] @ left) * (state.z[pairs[1]] @ right) * s).sum(
                -1
            ) + b
        return self.relation_design(state, pairs) @ self.inverse.theta()

    def embed_block(self, state, ids):
        left, s, right, _ = self.attention_program(state)
        h = self.up(state.z[ids])
        for layer in self.layers:
            h = layer(h, left, s, right, state.z[ids] if self.frame_aligned else None)
        return self.norm(h)

    def embed_all(self, state, blocks, *, batch_training=False):
        # Equal-length blocks are independent batch elements. This keeps exactly
        # the original attention contexts, including singleton/isolated nodes.
        out = state.z.new_empty((len(state.z), self.config["hidden"]))
        if self.training and not batch_training:
            # Preserve the training-time dropout draw order for existing callers.
            for block in blocks:
                ids = block.to(state.z.device)
                out[ids] = self.embed_block(state, ids)
            return out
        groups = {}
        for block in blocks:
            groups.setdefault(len(block), []).append(block)
        left, s, right, _ = self.attention_program(state)
        for length, group in groups.items():
            batch_size = max(1, min(256, 8192 // length))
            for start in range(0, len(group), batch_size):
                ids = torch.stack(group[start : start + batch_size]).to(state.z.device)
                h = self.up(state.z[ids])
                for layer in self.layers:
                    h = layer(
                        h, left, s, right, state.z[ids] if self.frame_aligned else None
                    )
                out[ids] = self.norm(h)
        return out

    def edge_logits(self, state, h, pairs):
        a, b = h[pairs[0]], h[pairs[1]]
        return self.decode_endpoints(state, a, b, pairs)

    def embed_graph(self, state, edge_index):
        """Graph-local layout; callers supply visible directed edges + self loops."""
        left, s, right, _ = self.attention_program(state)
        h = self.up(state.z)
        for layer in self.layers:
            h = layer.forward_graph(
                h, left, s, right, edge_index, state.z if self.frame_aligned else None
            )
        return self.norm(h)

    def decode_endpoints(self, state, a, b, pairs):
        # A vanilla GT has learned Q/K and no analytical relation shortcut.
        r = (
            a.new_zeros(len(a))
            if self.mode == "vanilla"
            else self.relation_logits(state, pairs)
        )
        residual = r if self.relation_gate is None else self.relation_gate * r
        return (
            self.link_decoder(
                torch.cat((a + b, a * b, (a - b).abs(), r[:, None]), dim=-1)
            ).squeeze(-1)
            + residual
        )

    @staticmethod
    def pair_embeddings(h, pairs, directed=False):
        a, b = h[pairs[0]], h[pairs[1]]
        return (
            torch.cat((a, b, a * b, a - b), dim=-1)
            if directed
            else torch.cat((a + b, a * b, (a - b).abs()), dim=-1)
        )


def prototypes(embeddings, labels, support, query, temperature=0.1):
    classes = labels[support].unique(sorted=True)
    if len(classes) < 2:
        raise ValueError("at least two support classes required")
    proto = torch.stack(
        [embeddings[support[labels[support] == c]].mean(0) for c in classes]
    )
    logits = (
        F.normalize(embeddings[query], dim=-1)
        @ F.normalize(proto, dim=-1).T
        / temperature
    )
    matches = labels[query, None] == classes[None, :]
    valid = matches.any(-1)
    target = matches.long().argmax(-1)
    return logits, target, valid, classes
