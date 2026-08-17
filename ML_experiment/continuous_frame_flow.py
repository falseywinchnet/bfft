#!/usr/bin/env python3
"""Standalone continuous-frame-flow network and CPU Muon helper.

Requires only PyTorch.  The model receives one sample at a time: every chart
probe is generated internally from that sample and from the layer's learned
metric.  No labels, neighbor samples, future coordinates, or task geometry are
used by the mechanism.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LELU(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = math.pi / math.sqrt(3.0)

    def forward(self, x):
        return x * torch.sigmoid(self.scale * x)


class ContinuousFrameFlowLinear(nn.Module):
    """Affine path plus a self-conditioned continuous Stiefel-frame flow."""

    def __init__(self, n_in: int, n_out: int, *, directions: int = 12,
                 rank: int = 4, strength: float = .25,
                 shell_samples: int | None = None,
                 frozen_shell_metric: bool = False):
        super().__init__()
        if n_in % 2:
            raise ValueError("continuous frame flow requires an even hidden width")
        self.n_in = n_in
        self.directions = directions
        self.rank = rank
        self.strength = float(strength)
        self.shell_samples = rank if shell_samples is None else int(shell_samples)
        if not 1 <= self.shell_samples <= rank:
            raise ValueError("shell_samples must be between 1 and rank")
        self.frozen_shell_metric = bool(frozen_shell_metric)

        self.base = nn.Linear(n_in, n_out)
        self.metric = nn.Linear(n_in, rank * rank)
        self.response = nn.Sequential(nn.Linear(4, 12), LELU(), nn.Linear(12, 1))
        self.shared = nn.Parameter(torch.randn(rank, n_out) / math.sqrt(rank))
        self.scale = nn.Parameter(torch.tensor(-1.5))

        generator = torch.Generator().manual_seed(9157 + n_in + n_out)
        seed_frame, _ = torch.linalg.qr(torch.randn(n_in, rank, generator=generator))
        mixing, _ = torch.linalg.qr(torch.randn(n_in, n_in, generator=generator))
        canonical = torch.zeros(n_in, n_in)
        maximum_frequency = max(1, min(5, (directions - 1) // 2))
        for plane in range(n_in // 2):
            frequency = 1 + plane % maximum_frequency
            canonical[2 * plane, 2 * plane + 1] = -frequency
            canonical[2 * plane + 1, 2 * plane] = frequency
        generator_matrix = mixing @ canonical @ mixing.T
        frames = []
        for index in range(directions):
            angle = 2 * math.pi * index / directions
            frames.append((torch.matrix_exp(angle * generator_matrix) @ seed_frame).T)
        self.register_buffer("frame_atlas", torch.stack(frames))

        if self.shell_samples == rank:
            shell_mixer = torch.eye(rank)
        else:
            source = torch.randn(rank, rank, generator=generator)
            orthogonal, _ = torch.linalg.qr(source)
            shell_mixer = orthogonal[:, :self.shell_samples].T
        self.register_buffer("shell_mixer", shell_mixer)

    def _weights(self, metric, projected):
        cost = torch.einsum("bdr,brs,bds->bd", projected, metric, projected)
        norm = projected.square().mean(-1)
        stats = torch.stack((
            torch.log1p(cost), torch.log1p(norm), projected.mean(-1),
            torch.log1p(projected.abs().mean(-1)),
        ), -1)
        logits = self.response(stats).squeeze(-1)
        logits = logits - cost / (cost.mean(1, keepdim=True) + 1e-5)
        return torch.softmax(logits, dim=1)

    def _allocate(self, x):
        factor = self.metric(x).view(len(x), self.rank, self.rank)
        metric = factor @ factor.transpose(1, 2) / self.rank
        projected = torch.einsum("dri,bi->bdr", self.frame_atlas, x)
        return metric, projected, self._weights(metric, projected)

    def _lift(self, projected, weight):
        return torch.einsum(
            "bd,bdr,dri->bi", weight, projected, self.frame_atlas
        ) / self.rank

    @staticmethod
    def _normalize_like(state, reference):
        state_rms = state.square().mean(1, keepdim=True).sqrt().clamp_min(1e-6)
        reference_rms = reference.square().mean(1, keepdim=True).sqrt().detach().clamp_min(1e-6)
        return state * (reference_rms / state_rms)

    def _curvature_context(self, x, metric, projected, weight):
        """Mean even difference over the locally selected tangent shell."""
        center = self._lift(projected, weight)
        frame = torch.einsum("bd,dri->bri", weight, self.frame_atlas)
        frame = F.normalize(frame, dim=-1)

        # Whiten the selected rays: the result depends on their tangent span,
        # not on accidental skew in the ray coordinates.
        gram = frame @ frame.transpose(1, 2)
        identity = torch.eye(self.rank, device=x.device, dtype=x.dtype)[None]
        cholesky = torch.linalg.cholesky(gram + 1e-3 * identity)
        frame = torch.linalg.solve_triangular(cholesky, frame, upper=False)
        if self.shell_samples != self.rank:
            frame = torch.einsum("sr,bri->bsi", self.shell_mixer, frame)

        radius = self.strength * x.norm(dim=1, keepdim=True).detach().clamp_min(1e-3)
        displacement = radius[:, None, :] * frame
        if self.frozen_shell_metric:
            delta = torch.einsum("dki,bsi->bsdk", self.frame_atlas, displacement)
            center_projected = projected[:, None]
            probe_projected = torch.cat(
                (center_projected + delta, center_projected - delta), dim=1
            ).flatten(0, 1)
            probe_metric = metric[:, None].expand(
                -1, 2 * self.shell_samples, -1, -1
            ).flatten(0, 1)
            probe_weight = self._weights(probe_metric, probe_projected)
        else:
            probes = torch.cat((
                x[:, None, :] + displacement,
                x[:, None, :] - displacement,
            ), dim=1)
            _, probe_projected, probe_weight = self._allocate(probes.flatten(0, 1))

        context = self._lift(probe_projected, probe_weight).view(
            len(x), 2 * self.shell_samples, self.n_in
        )
        plus = context[:, :self.shell_samples]
        minus = context[:, self.shell_samples:]
        return (plus + minus - 2 * center[:, None, :]).mean(1)

    def forward(self, x):
        # First interpretation: the chart sees the authentic activation.
        metric, projected, weight = self._allocate(x)
        context = self._normalize_like(self._lift(projected, weight), x)

        # Second interpretation: move along the chart's own first estimate.
        chart = x + self.strength * context
        metric, projected, weight = self._allocate(chart)

        # The change of that interpretation becomes state, then changes the
        # final chart selection.  This is an internal query, not extra data.
        curvature = self._curvature_context(chart, metric, projected, weight)
        curvature = self._normalize_like(curvature, chart)
        chart = chart + self.strength * curvature
        _, projected, weight = self._allocate(chart)

        pooled = torch.einsum("bd,bdr->br", weight, projected)
        correction = F.softplus(self.scale) * (pooled @ self.shared)
        return self.base(x) + correction


class ContinuousFrameFlow(nn.Module):
    """encode -> frame flow -> LELU -> frame flow -> decode."""

    def __init__(self, input_dim: int, output_dim: int, width: int = 24,
                 *, directions: int = 12, rank: int = 4,
                 strength: float = .25, fast: bool = False,
                 frozen_shell_metric: bool = False):
        super().__init__()
        shell_samples = 2 if fast else rank
        options = dict(
            directions=directions, rank=rank, strength=strength,
            shell_samples=shell_samples,
            frozen_shell_metric=frozen_shell_metric,
        )
        self.embed = nn.Linear(input_dim, width)
        self.up = ContinuousFrameFlowLinear(width, 2 * width, **options)
        self.activation = LELU()
        self.down = ContinuousFrameFlowLinear(2 * width, width, **options)
        self.output = nn.Linear(width, output_dim)

    def forward(self, x):
        return self.output(self.down(self.activation(self.up(self.embed(x)))))


def zeropower_newton_schulz5(matrix, steps=5, eps=1e-7):
    """Approximate the semi-orthogonal polar factor used by Muon."""
    a, b, c = 3.4445, -4.7750, 2.0315
    transposed = matrix.shape[0] > matrix.shape[1]
    x = matrix.float().T if transposed else matrix.float()
    x = x / x.norm().clamp_min(eps)
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    return (x.T if transposed else x).to(matrix.dtype)


class Muon(torch.optim.Optimizer):
    """Minimal single-process Muon for hidden matrix parameters."""

    def __init__(self, params, lr=3e-3, momentum=.95, weight_decay=1e-4):
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                     weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.ndim != 2:
                    raise ValueError("Muon parameters must be matrices")
                state = self.state[parameter]
                momentum = state.setdefault("momentum", torch.zeros_like(parameter))
                momentum.lerp_(parameter.grad, 1 - group["momentum"])
                update = parameter.grad.lerp(momentum, group["momentum"])
                update = zeropower_newton_schulz5(update)
                parameter.mul_(1 - group["lr"] * group["weight_decay"])
                adjusted_lr = group["lr"] * .2 * math.sqrt(max(parameter.shape))
                parameter.add_(update, alpha=-adjusted_lr)


class MuonWithAuxAdamW:
    """Muon on hidden matrices; AdamW on edges, biases, and scales."""

    def __init__(self, model, lr=3e-3, weight_decay=1e-4):
        matrices, auxiliary = [], []
        for name, parameter in model.named_parameters():
            edge = name.startswith(("embed.", "output."))
            (matrices if parameter.ndim == 2 and not edge else auxiliary).append(parameter)
        self.muon = Muon(matrices, lr=lr, weight_decay=weight_decay)
        self.adamw = torch.optim.AdamW(
            auxiliary, lr=lr, weight_decay=weight_decay, betas=(.9, .95)
        )

    def zero_grad(self, set_to_none=True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.muon.step()
        self.adamw.step()


if __name__ == "__main__":
    torch.manual_seed(0)
    model = ContinuousFrameFlow(2, 1, width=24)
    x = torch.randn(32, 2)
    loss = model(x).square().mean()
    loss.backward()
    optimizer = MuonWithAuxAdamW(model)
    optimizer.step()
    count = sum(parameter.numel() for parameter in model.parameters())
    print(f"output={tuple(model(x).shape)} parameters={count} loss={loss.item():.6f}")
