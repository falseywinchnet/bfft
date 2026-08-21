"""Relational response heads for self-context and continuous frame flow.

The enhanced allocator does not receive labels, neighbors, task coordinates,
or future points.  It sees the same sample before and after an internally
generated chart displacement, expressed in every candidate frame.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ML_experiment.models import LELU
from ML_experiment.spectral_response import SpectralResponse
from ML_experiment.variants import make_variant


ORIGINAL_SELF_CONTEXT = "self_context"
RELATIONAL_SCL = "relational_scl"
BASELINE_CFF = "cff"
RELATIONAL_CFF_DEEP = "relational_cff_deep"
SPECTRAL_SCL_MIDDLE = "spectral_scl_middle"
SPECTRAL_SCL_MAX = "spectral_scl_max"
SPECTRAL_CFF_MIDDLE = "spectral_cff_middle"
RESPONSE_VARIANTS = (
    ORIGINAL_SELF_CONTEXT,
    RELATIONAL_SCL,
    BASELINE_CFF,
    RELATIONAL_CFF_DEEP,
)
SPECTRAL_SCREEN_VARIANTS = (
    RELATIONAL_SCL,
    RELATIONAL_CFF_DEEP,
    SPECTRAL_SCL_MIDDLE,
    SPECTRAL_SCL_MAX,
    SPECTRAL_CFF_MIDDLE,
)


def _frame_atlas(n_in: int, n_out: int, directions: int, rank: int, mode: str):
    generator = torch.Generator().manual_seed(9157 + n_in + n_out)
    if mode == "random":
        return F.normalize(
            torch.randn(directions, rank, n_in, generator=generator), dim=-1
        )
    if mode != "stiefel_flow":
        raise ValueError(mode)
    if n_in % 2:
        raise ValueError("continuous frame flow requires an even input width")

    seed_frame, _ = torch.linalg.qr(torch.randn(n_in, rank, generator=generator))
    mixing, _ = torch.linalg.qr(torch.randn(n_in, n_in, generator=generator))
    canonical = torch.zeros(n_in, n_in)
    maximum_frequency = max(1, min(5, (directions - 1) // 2))
    for plane in range(n_in // 2):
        frequency = 1 + plane % maximum_frequency
        canonical[2 * plane, 2 * plane + 1] = -frequency
        canonical[2 * plane + 1, 2 * plane] = frequency
    generator_matrix = mixing @ canonical @ mixing.T
    return torch.stack([
        (
            torch.matrix_exp(
                2.0 * math.pi * index / directions * generator_matrix
            )
            @ seed_frame
        ).T
        for index in range(directions)
    ])


class RelationalSelfContextLinear(nn.Module):
    """Self-context whose allocator sees authentic/current chart relations."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        *,
        directions: int = 12,
        rank: int = 4,
        strength: float = 0.25,
        response_width: int = 12,
        response_depth: int = 1,
        primitive_mode: str = "random",
    ):
        super().__init__()
        if rank > n_in:
            raise ValueError("rank cannot exceed the input width")
        if response_depth < 1:
            raise ValueError("response_depth must be positive")
        self.n_in = n_in
        self.directions = directions
        self.rank = rank
        self.strength = float(strength)

        self.base = nn.Linear(n_in, n_out)
        self.metric = nn.Linear(n_in, rank * rank)
        self.shared = nn.Parameter(torch.randn(rank, n_out) / math.sqrt(rank))
        self.correction_scale = nn.Parameter(torch.tensor(-1.5))
        self.register_buffer(
            "frame_atlas",
            _frame_atlas(n_in, n_out, directions, rank, primitive_mode),
        )

        feature_count = 7 + 5 * rank
        response: list[nn.Module] = [nn.Linear(feature_count, response_width), LELU()]
        for _ in range(response_depth - 1):
            response.extend((nn.Linear(response_width, response_width), LELU()))
        response.append(nn.Linear(response_width, 1))
        self.response = nn.Sequential(*response)
        self.last_weight: torch.Tensor | None = None

    @staticmethod
    def _rms_last(value: torch.Tensor) -> torch.Tensor:
        return value.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)

    @staticmethod
    def _normalize_like(state: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        state_rms = state.square().mean(1, keepdim=True).sqrt().clamp_min(1e-6)
        reference_rms = (
            reference.square().mean(1, keepdim=True).sqrt().detach().clamp_min(1e-6)
        )
        return state * (reference_rms / state_rms)

    def _features(
        self,
        metric: torch.Tensor,
        projected: torch.Tensor,
        anchor_projected: torch.Tensor,
        cost: torch.Tensor,
    ) -> torch.Tensor:
        current = projected / self._rms_last(projected)
        anchor = anchor_projected / self._rms_last(anchor_projected)
        displacement = (projected - anchor_projected) / self._rms_last(anchor_projected)

        metric_view = torch.einsum("brs,bds->bdr", metric, current)
        metric_view = metric_view / self._rms_last(metric_view)
        coordinate_alignment = current * anchor

        energy = projected.square().mean(-1)
        anchor_energy = anchor_projected.square().mean(-1)
        scalar = torch.stack(
            (
                torch.log1p(cost.clamp_min(0)),
                torch.log1p(energy),
                current.mean(-1),
                current.abs().mean(-1),
                coordinate_alignment.mean(-1),
                torch.log1p(displacement.square().mean(-1)),
                torch.log((energy + 1e-6) / (anchor_energy + 1e-6)),
            ),
            dim=-1,
        )
        return torch.cat(
            (
                scalar,
                current,
                anchor,
                displacement,
                metric_view,
                coordinate_alignment,
            ),
            dim=-1,
        )

    def _allocation_weights(
        self,
        metric: torch.Tensor,
        projected: torch.Tensor,
        anchor_projected: torch.Tensor,
    ) -> torch.Tensor:
        cost = torch.einsum("bdr,brs,bds->bd", projected, metric, projected)
        response = self.response(
            self._features(metric, projected, anchor_projected, cost)
        ).squeeze(-1)
        relative_cost = cost / (cost.mean(1, keepdim=True) + 1e-5)
        return torch.softmax(response - relative_cost, dim=1)

    def _allocate(
        self,
        state: torch.Tensor,
        anchor_projected: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        factor = self.metric(state).view(len(state), self.rank, self.rank)
        metric = factor @ factor.transpose(1, 2) / self.rank
        projected = torch.einsum("dri,bi->bdr", self.frame_atlas, state)
        if anchor_projected is None:
            anchor_projected = projected
        weight = self._allocation_weights(metric, projected, anchor_projected)
        return metric, projected, weight

    def _lift(self, projected: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.einsum(
            "bd,bdr,dri->bi", weight, projected, self.frame_atlas
        ) / self.rank

    def _correct(
        self,
        x: torch.Tensor,
        projected: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        self.last_weight = weight
        pooled = torch.einsum("bd,bdr->br", weight, projected)
        correction = F.softplus(self.correction_scale) * (pooled @ self.shared)
        return self.base(x) + correction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, authentic_projected, weight = self._allocate(x)
        context = self._normalize_like(self._lift(authentic_projected, weight), x)
        chart = x + self.strength * context
        _, projected, weight = self._allocate(chart, authentic_projected)
        return self._correct(x, projected, weight)


class RelationalCFFLinear(RelationalSelfContextLinear):
    """Continuous frame flow with relational, two-LELU response heads."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        *,
        directions: int = 12,
        rank: int = 4,
        strength: float = 0.25,
        response_width: int = 12,
        response_depth: int = 2,
    ):
        super().__init__(
            n_in,
            n_out,
            directions=directions,
            rank=rank,
            strength=strength,
            response_width=response_width,
            response_depth=response_depth,
            primitive_mode="stiefel_flow",
        )

    def _curvature_context(
        self,
        x: torch.Tensor,
        projected: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        center = self._lift(projected, weight)
        frame = torch.einsum("bd,dri->bri", weight, self.frame_atlas)
        frame = F.normalize(frame, dim=-1)
        gram = frame @ frame.transpose(1, 2)
        identity = torch.eye(self.rank, device=x.device, dtype=x.dtype)[None]
        cholesky = torch.linalg.cholesky(gram + 1e-3 * identity)
        frame = torch.linalg.solve_triangular(cholesky, frame, upper=False)

        radius = self.strength * x.norm(dim=1, keepdim=True).detach().clamp_min(1e-3)
        displacement = radius[:, None, :] * frame
        probes = torch.cat(
            (x[:, None, :] + displacement, x[:, None, :] - displacement), dim=1
        )
        probe_anchor = projected[:, None].expand(
            -1, 2 * self.rank, -1, -1
        ).flatten(0, 1)
        _, probe_projected, probe_weight = self._allocate(
            probes.flatten(0, 1), probe_anchor
        )
        probe_context = self._lift(probe_projected, probe_weight).view(
            len(x), 2 * self.rank, self.n_in
        )
        plus = probe_context[:, : self.rank]
        minus = probe_context[:, self.rank :]
        return (plus + minus - 2.0 * center[:, None, :]).mean(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, authentic_projected, weight = self._allocate(x)
        context = self._normalize_like(self._lift(authentic_projected, weight), x)

        chart = x + self.strength * context
        _, chart_projected, weight = self._allocate(chart, authentic_projected)

        curvature = self._normalize_like(
            self._curvature_context(chart, chart_projected, weight), chart
        )
        curved_chart = chart + self.strength * curvature
        _, projected, weight = self._allocate(curved_chart, chart_projected)
        return self._correct(x, projected, weight)


class SpectralSelfContextLinear(RelationalSelfContextLinear):
    """Relational self-context whose chart response is an ordered eigenvalue."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        *,
        matrix_dim: int = 5,
        eigen_index: int | None = None,
        **options,
    ):
        super().__init__(n_in, n_out, **options)
        self.response = SpectralResponse(
            7 + 5 * self.rank,
            matrix_dim=matrix_dim,
            eigen_index=eigen_index,
        )


class SpectralCFFLinear(RelationalCFFLinear):
    """CFF geometry with a spectral chart-response mechanism."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        *,
        matrix_dim: int = 5,
        eigen_index: int | None = None,
        **options,
    ):
        super().__init__(n_in, n_out, **options)
        self.response = SpectralResponse(
            7 + 5 * self.rank,
            matrix_dim=matrix_dim,
            eigen_index=eigen_index,
        )


class RelationalLayerNet(nn.Module):
    """Matched encode -> expansion -> LELU -> contraction -> decode network."""

    def __init__(self, input_dim: int, output_dim: int, width: int, layer):
        super().__init__()
        self.embed = nn.Linear(input_dim, width)
        self.up = layer(width, 2 * width)
        self.activation = LELU()
        self.down = layer(2 * width, width)
        self.output = nn.Linear(width, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.down(self.activation(self.up(self.embed(x)))))

    def allocation_weights(self):
        return self.up.last_weight, self.down.last_weight


def make_response_variant(
    name: str,
    input_dim: int,
    output_dim: int,
    width: int,
) -> nn.Module:
    if name == ORIGINAL_SELF_CONTEXT:
        return make_variant("self_context", input_dim, output_dim, width)
    if name == BASELINE_CFF:
        return make_variant(
            "self_context_stiefel_flow_curvature", input_dim, output_dim, width
        )
    if name == RELATIONAL_SCL:
        return RelationalLayerNet(
            input_dim, output_dim, width, RelationalSelfContextLinear
        )
    if name == RELATIONAL_CFF_DEEP:
        return RelationalLayerNet(input_dim, output_dim, width, RelationalCFFLinear)
    if name == SPECTRAL_SCL_MIDDLE:
        return RelationalLayerNet(
            input_dim, output_dim, width, SpectralSelfContextLinear
        )
    if name == SPECTRAL_SCL_MAX:
        return RelationalLayerNet(
            input_dim,
            output_dim,
            width,
            lambda n_in, n_out: SpectralSelfContextLinear(
                n_in, n_out, matrix_dim=5, eigen_index=4
            ),
        )
    if name == SPECTRAL_CFF_MIDDLE:
        return RelationalLayerNet(input_dim, output_dim, width, SpectralCFFLinear)
    raise KeyError(name)


def allocation_summary(model: nn.Module) -> dict[str, float]:
    """Return normalized entropy and maximum chart ownership after a forward."""
    weights = model.allocation_weights()
    entropy = []
    maximum = []
    for weight in weights:
        if weight is None:
            continue
        entropy.append(
            float(
                (
                    -(weight * torch.log(weight + 1e-9)).sum(1)
                    / math.log(weight.shape[1])
                ).mean()
            )
        )
        maximum.append(float(weight.max(1).values.mean()))
    return {
        "allocation_entropy": sum(entropy) / len(entropy),
        "allocation_max_weight": sum(maximum) / len(maximum),
    }
