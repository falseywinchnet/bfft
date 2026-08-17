"""Continuous Frame Flow (CFF), as a standalone PyTorch module.

Only PyTorch is required. The layer creates every chart and shell query from
the current activation; it consumes no labels, neighbors, task coordinates, or
future samples.

Example:
    model = ContinuousFrameFlow(input_dim=2, output_dim=2, width=38)
    y = model(x)  # x: [batch, 2]
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LELU", "ContinuousFrameFlowLinear", "ContinuousFrameFlow"]


class LELU(nn.Module):
    """Logistic-CDF-matched smooth linear unit."""

    def __init__(self):
        super().__init__()
        self.scale = math.pi / math.sqrt(3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.scale * x)


class ContinuousFrameFlowLinear(nn.Module):
    """Affine map plus self-context and curvature over a continuous frame atlas."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        *,
        directions: int = 12,
        rank: int = 4,
        strength: float = 0.25,
    ):
        super().__init__()
        if n_in % 2:
            raise ValueError("CFF requires an even hidden width")
        if rank > n_in:
            raise ValueError("rank cannot exceed the input width")

        self.n_in = n_in
        self.directions = directions
        self.rank = rank
        self.strength = float(strength)

        # The affine path preserves unbounded range. The frame path supplies
        # a boundedly initialized, input-conditioned correction.
        self.base = nn.Linear(n_in, n_out)
        self.metric = nn.Linear(n_in, rank * rank)
        self.response = nn.Sequential(
            nn.Linear(4, 12),
            LELU(),
            nn.Linear(12, 1),
        )
        self.shared = nn.Parameter(torch.randn(rank, n_out) / math.sqrt(rank))
        self.correction_scale = nn.Parameter(torch.tensor(-1.5))

        # Build discrete samples from one continuous orthogonal frame flow.
        # The atlas is fixed; its input-dependent allocation is learned.
        generator = torch.Generator().manual_seed(9157 + n_in + n_out)
        seed_frame, _ = torch.linalg.qr(
            torch.randn(n_in, rank, generator=generator)
        )
        mixing, _ = torch.linalg.qr(
            torch.randn(n_in, n_in, generator=generator)
        )
        canonical = torch.zeros(n_in, n_in)
        maximum_frequency = max(1, min(5, (directions - 1) // 2))
        for plane in range(n_in // 2):
            frequency = 1 + plane % maximum_frequency
            canonical[2 * plane, 2 * plane + 1] = -frequency
            canonical[2 * plane + 1, 2 * plane] = frequency
        generator_matrix = mixing @ canonical @ mixing.T
        frames = []
        for index in range(directions):
            angle = 2.0 * math.pi * index / directions
            frames.append(
                (torch.matrix_exp(angle * generator_matrix) @ seed_frame).T
            )
        self.register_buffer("frame_atlas", torch.stack(frames))

    def _allocation_weights(
        self,
        metric: torch.Tensor,
        projected: torch.Tensor,
    ) -> torch.Tensor:
        cost = torch.einsum("bdr,brs,bds->bd", projected, metric, projected)
        norm = projected.square().mean(-1)
        statistics = torch.stack(
            (
                torch.log1p(cost),
                torch.log1p(norm),
                projected.mean(-1),
                torch.log1p(projected.abs().mean(-1)),
            ),
            dim=-1,
        )
        logits = self.response(statistics).squeeze(-1)
        logits = logits - cost / (cost.mean(1, keepdim=True) + 1e-5)
        return torch.softmax(logits, dim=1)

    def _allocate(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        factor = self.metric(x).view(len(x), self.rank, self.rank)
        metric = factor @ factor.transpose(1, 2) / self.rank
        projected = torch.einsum("dri,bi->bdr", self.frame_atlas, x)
        weight = self._allocation_weights(metric, projected)
        return metric, projected, weight

    def _lift(
        self,
        projected: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        return torch.einsum(
            "bd,bdr,dri->bi", weight, projected, self.frame_atlas
        ) / self.rank

    @staticmethod
    def _normalize_like(
        state: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        state_rms = state.square().mean(1, keepdim=True).sqrt().clamp_min(1e-6)
        reference_rms = (
            reference.square().mean(1, keepdim=True).sqrt().detach().clamp_min(1e-6)
        )
        return state * (reference_rms / state_rms)

    def _curvature_context(
        self,
        x: torch.Tensor,
        projected: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        """Lift the even second difference over the selected tangent shell."""
        center = self._lift(projected, weight)
        frame = torch.einsum("bd,dri->bri", weight, self.frame_atlas)
        frame = F.normalize(frame, dim=-1)

        # Whiten the rays so the shell depends on their tangent span rather
        # than accidental skew in the selected coordinates.
        gram = frame @ frame.transpose(1, 2)
        identity = torch.eye(self.rank, device=x.device, dtype=x.dtype)[None]
        cholesky = torch.linalg.cholesky(gram + 1e-3 * identity)
        frame = torch.linalg.solve_triangular(cholesky, frame, upper=False)

        radius = (
            self.strength
            * x.norm(dim=1, keepdim=True).detach().clamp_min(1e-3)
        )
        displacement = radius[:, None, :] * frame
        probes = torch.cat(
            (x[:, None, :] + displacement, x[:, None, :] - displacement),
            dim=1,
        )
        _, probe_projected, probe_weight = self._allocate(probes.flatten(0, 1))
        probe_context = self._lift(probe_projected, probe_weight).view(
            len(x), 2 * self.rank, self.n_in
        )
        plus = probe_context[:, : self.rank]
        minus = probe_context[:, self.rank :]
        return (plus + minus - 2.0 * center[:, None, :]).mean(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First chart: interpret the authentic activation.
        _, projected, weight = self._allocate(x)
        context = self._normalize_like(self._lift(projected, weight), x)

        # Second chart: reinterpret x after moving along its own first view.
        chart = x + self.strength * context
        _, projected, weight = self._allocate(chart)

        # Promote the even shell response to state and let it alter selection.
        curvature = self._curvature_context(chart, projected, weight)
        curvature = self._normalize_like(curvature, chart)
        chart = chart + self.strength * curvature
        _, projected, weight = self._allocate(chart)

        pooled = torch.einsum("bd,bdr->br", weight, projected)
        correction = F.softplus(self.correction_scale) * (pooled @ self.shared)
        return self.base(x) + correction


class ContinuousFrameFlow(nn.Module):
    """encode -> CFF expansion -> LELU -> CFF contraction -> decode."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        width: int = 38,
        *,
        directions: int = 12,
        rank: int = 4,
        strength: float = 0.25,
    ):
        super().__init__()
        options = {
            "directions": directions,
            "rank": rank,
            "strength": strength,
        }
        self.embed = nn.Linear(input_dim, width)
        self.up = ContinuousFrameFlowLinear(width, 2 * width, **options)
        self.activation = LELU()
        self.down = ContinuousFrameFlowLinear(2 * width, width, **options)
        self.output = nn.Linear(width, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.embed(x)
        hidden = self.up(hidden)
        hidden = self.activation(hidden)
        hidden = self.down(hidden)
        return self.output(hidden)
