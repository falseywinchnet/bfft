"""Odd third-order degrees of freedom grafted onto self-context and CFF.

The goal is not to replace the parent geometry with a cubic model.  Each
variant asks where a small odd relational channel can enter while leaving the
parent's ordinary function class intact.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ML_experiment.models import LELU, parameter_count
from ML_experiment.nd_spiral_wall import ShallowOddCubicNet
from ML_experiment.response_enhanced import (
    RELATIONAL_CFF_DEEP,
    RELATIONAL_SCL,
    make_response_variant,
)


SELF = RELATIONAL_SCL
FLOW = RELATIONAL_CFF_DEEP


class CubicResidual(nn.Module):
    """A pure odd triple-product residual with selectable radial behavior."""

    def __init__(self, n_in: int, n_out: int, channels: int, mode: str):
        super().__init__()
        if mode not in {"degree2", "angular"}:
            raise ValueError(mode)
        self.mode = mode
        self.a = nn.Linear(n_in, channels, bias=False)
        self.b = nn.Linear(n_in, channels, bias=False)
        self.c = nn.Linear(n_in, channels, bias=False)
        self.output = nn.Linear(channels, n_out, bias=False)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        a, b, c = self.a(x), self.b(x), self.c(x)
        product = a * b * c
        if self.mode == "degree2":
            # Cubic numerator / degree-one radius: positive radial degree two,
            # while retaining odd parity under x -> -x.
            radius = (a.square() + b.square() + c.square()).clamp_min(1e-6).sqrt()
            return math.sqrt(3.0) * product / radius
        # Separate direction from magnitude.  The normalized cubic supplies an
        # angular signature; authentic input RMS restores only degree-one range.
        angular = F.rms_norm(product, (product.shape[-1],))
        magnitude = x.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
        return angular * magnitude

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.features(x))


class AncientPolyNormResidual(nn.Module):
    """The old square/cubic mixture followed by norm coupling and LELU."""

    def __init__(self, n_in: int, n_out: int, channels: int):
        super().__init__()
        self.project = nn.Linear(n_in, channels)
        self.norm = nn.LayerNorm(channels)
        self.activation = LELU()
        self.output = nn.Linear(channels, n_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = self.project(x)
        state = state.square() + 0.5 * state.pow(3)
        return self.output(self.activation(self.norm(state)))


class ParallelGraft(nn.Module):
    """Parent logits plus a separately learnable structural residual."""

    def __init__(self, parent: nn.Module, branch: nn.Module):
        super().__init__()
        self.parent = parent
        self.branch = branch
        self.branch_scale = nn.Parameter(torch.tensor(-2.0))

    def components(self, x: torch.Tensor):
        parent = self.parent(x)
        branch = F.softplus(self.branch_scale) * self.branch(x)
        return parent, branch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parent, branch = self.components(x)
        return parent + branch


class ContextualCubicBridge(nn.Module):
    """A cubic relation between authentic hidden state and chart response."""

    def __init__(self, source_dim: int, context_dim: int, output_dim: int,
                 channels: int, *, tied_source: bool = False,
                 conditioning: str = "none",
                 homogeneity: str = "fixed"):
        super().__init__()
        if conditioning not in {
            "none", "factor_rms", "row_unit", "tight", "tight_rms",
            "learned_cone", "weight_norm",
        }:
            raise ValueError(conditioning)
        if homogeneity not in {"fixed", "global", "ray"}:
            raise ValueError(homogeneity)
        self.conditioning = conditioning
        self.homogeneity = homogeneity
        self.a = nn.Linear(source_dim, channels, bias=False)
        self.b = (self.a if tied_source
                  else nn.Linear(source_dim, channels, bias=False))
        self.c = nn.Linear(context_dim, channels, bias=False)
        self.output = nn.Linear(channels, output_dim, bias=False)
        self.cone_logits = (nn.Parameter(torch.zeros(3))
                            if conditioning == "learned_cone" else None)
        if conditioning == "weight_norm":
            layers = (self.a, self.b, self.c)
            self.log_row_gains = nn.ParameterList([
                nn.Parameter(layer.weight.detach().norm(dim=1).clamp_min(1e-6).log())
                for layer in layers
            ])
        else:
            self.log_row_gains = None
        # degree = 3 sigmoid(logit), initialized at degree one. This leaves the
        # original bridge unchanged at initialization while permitting either
        # the atlas or each ray to discover a continuous radial scaling law.
        degree_count = channels if homogeneity == "ray" else 1
        self.degree_logits = (
            nn.Parameter(torch.full((degree_count,), -math.log(2.0)))
            if homogeneity != "fixed" else None
        )

    @staticmethod
    def _semi_orthogonal(weight: torch.Tensor) -> torch.Tensor:
        rows, columns = weight.shape
        if rows <= columns:
            q, r = torch.linalg.qr(weight.T, mode="reduced")
            sign = torch.diagonal(r).sign().detach().clamp_min(0).mul(2).sub(1)
            return (q * sign).T
        q, r = torch.linalg.qr(weight, mode="reduced")
        sign = torch.diagonal(r).sign().detach().clamp_min(0).mul(2).sub(1)
        return q * sign

    def _project(self, layer: nn.Linear, x: torch.Tensor,
                 factor_index: int) -> torch.Tensor:
        weight = layer.weight
        if self.conditioning == "row_unit":
            weight = F.normalize(weight, dim=1)
        elif self.conditioning == "learned_cone":
            exponent = torch.sigmoid(self.cone_logits[factor_index])
            row_norm = weight.norm(dim=1, keepdim=True).clamp_min(1e-6)
            weight = weight / row_norm.pow(exponent)
        elif self.conditioning == "weight_norm":
            direction = F.normalize(weight, dim=1)
            gain = self.log_row_gains[factor_index].exp()[:, None]
            weight = direction * gain
        elif self.conditioning in {"tight", "tight_rms"}:
            weight = self._semi_orthogonal(weight)
        return F.linear(x, weight)

    def forward(self, source: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        a = self._project(self.a, source, 0)
        b = self._project(self.b, source, 1)
        c = self._project(self.c, context, 2)
        if self.conditioning in {"factor_rms", "tight_rms"}:
            shape = (a.shape[-1],)
            a, b, c = F.rms_norm(a, shape), F.rms_norm(b, shape), F.rms_norm(c, shape)
        product = a * b * c
        angular = F.rms_norm(product, (product.shape[-1],))
        magnitude = source.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
        if self.degree_logits is None:
            radial = magnitude
        else:
            degree = 3.0 * torch.sigmoid(self.degree_logits)
            radial = torch.exp(torch.log(magnitude) * degree)
        return self.output(angular * radial)


class ContextualHiddenGraft(nn.Module):
    """Let the first chart response participate in an odd hidden relation.

    The bridge returns to the parent's normal LELU/down/chart path.  Its
    backward signal also reaches the first allocator through the context
    factor, which an output-only parallel branch cannot do.
    """

    def __init__(self, parent, width: int, *, channels: int | None = None,
                 tied_source: bool = False, antithetic: bool = False,
                 modulated: bool = False, conditioning: str = "none",
                 homogeneity: str = "fixed"):
        super().__init__()
        self.parent = parent
        channels = channels or 2 * width
        self.bridge = ContextualCubicBridge(width, 2 * width, 2 * width,
                                             channels=channels,
                                             tied_source=tied_source,
                                             conditioning=conditioning,
                                             homogeneity=homogeneity)
        self.branch_scale = nn.Parameter(torch.tensor(-2.0))
        self.antithetic = bool(antithetic)
        self.modulated = bool(modulated)
        self.last_parent_state = None
        self.last_branch_state = None

    def hidden_components(self, x: torch.Tensor):
        authentic = self.parent.embed(x)
        chart = self.parent.up(authentic)
        relation = self.bridge(authentic, chart)
        if self.antithetic:
            negative = -authentic
            negative_chart = self.parent.up(negative)
            relation = 0.5 * (
                relation - self.bridge(negative, negative_chart)
            )
        branch = F.softplus(self.branch_scale) * relation
        return authentic, chart, branch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, chart, branch = self.hidden_components(x)
        self.last_parent_state, self.last_branch_state = chart, branch
        if self.modulated:
            state = self.parent.activation(chart * (1.0 + torch.tanh(branch)))
        else:
            state = self.parent.activation(chart + branch)
        return self.parent.output(self.parent.down(state))


VARIANTS = (
    "self_context",
    "cff",
    "shallow_odd_cubic",
    "self_parallel_degree2",
    "self_parallel_angular",
    "self_contextual_angular",
    "cff_contextual_angular",
    "self_ancient_polynorm",
)


CONTROLLED_VARIANTS = (
    "self_context",
    "self_capacity_match_rank8",
    "self_capacity_match_full",
    "self_contextual_angular_rank4",
    "self_contextual_angular_rank8",
    "self_contextual_tied_rank8",
    "self_contextual_modulated_rank8",
    "self_contextual_antithetic_rank8",
    "self_contextual_angular",
)


FACTOR_VARIANTS = (
    "self_contextual_angular",
    "self_contextual_full_factor_rms",
    "self_contextual_full_row_unit",
    "self_contextual_full_tight",
    "self_contextual_full_tight_rms",
    "self_contextual_rank8_tight",
    "self_contextual_full_learned_cone",
    "self_contextual_full_weight_norm",
    "self_contextual_full_learned_cone_global_degree",
    "self_contextual_full_learned_cone_ray_degrees",
    "self_contextual_full_row_unit_ray_degrees",
)


def _matched_self_context(n_in: int, n_out: int, width: int,
                          target_parameters: int) -> nn.Module:
    """Return the closest wider plain SCL under the requested parameter count."""
    candidates = [
        make_response_variant(SELF, n_in, n_out, candidate_width)
        for candidate_width in range(width, 3 * width + 1)
    ]
    return min(
        candidates,
        key=lambda model: abs(parameter_count(model) - target_parameters),
    )


def make_hybrid(name: str, n_in: int, n_out: int, width: int) -> nn.Module:
    channels = max(48, 2 * width)
    if name == "self_context":
        return make_response_variant(SELF, n_in, n_out, width)
    if name == "cff":
        return make_response_variant(FLOW, n_in, n_out, width)
    if name == "shallow_odd_cubic":
        return ShallowOddCubicNet(n_in, n_out, width)
    if name == "self_parallel_degree2":
        return ParallelGraft(make_response_variant(SELF, n_in, n_out, width),
                             CubicResidual(n_in, n_out, channels, "degree2"))
    if name == "self_parallel_angular":
        return ParallelGraft(make_response_variant(SELF, n_in, n_out, width),
                             CubicResidual(n_in, n_out, channels, "angular"))
    if name == "self_contextual_angular":
        return ContextualHiddenGraft(
            make_response_variant(SELF, n_in, n_out, width), width
        )
    geometry_variants = {
        "self_contextual_full_learned_cone_global_degree": (
            "learned_cone", "global"
        ),
        "self_contextual_full_learned_cone_ray_degrees": (
            "learned_cone", "ray"
        ),
        "self_contextual_full_row_unit_ray_degrees": ("row_unit", "ray"),
    }
    if name in geometry_variants:
        conditioning, homogeneity = geometry_variants[name]
        return ContextualHiddenGraft(
            make_response_variant(SELF, n_in, n_out, width),
            width,
            channels=2 * width,
            conditioning=conditioning,
            homogeneity=homogeneity,
        )
    if name.startswith("self_contextual_full_") or name == "self_contextual_rank8_tight":
        conditioning = name.removeprefix("self_contextual_full_")
        channels = 2 * width
        if name == "self_contextual_rank8_tight":
            conditioning, channels = "tight", 8
        return ContextualHiddenGraft(
            make_response_variant(SELF, n_in, n_out, width),
            width,
            channels=channels,
            conditioning=conditioning,
        )
    if name in {
        "self_contextual_angular_rank4",
        "self_contextual_angular_rank8",
        "self_contextual_modulated_rank8",
        "self_contextual_antithetic_rank8",
    } or name.startswith("self_contextual_tied_rank"):
        channels = 4 if name.endswith("rank4") else 8
        if name.startswith("self_contextual_tied_rank"):
            channels = int(name.removeprefix("self_contextual_tied_rank"))
        return ContextualHiddenGraft(
            make_response_variant(SELF, n_in, n_out, width),
            width,
            channels=channels,
            tied_source=name.startswith("self_contextual_tied_rank"),
            modulated=name == "self_contextual_modulated_rank8",
            antithetic=name == "self_contextual_antithetic_rank8",
        )
    if name in {"self_capacity_match_rank8", "self_capacity_match_full"}:
        channels = 8 if name.endswith("rank8") else 2 * width
        target = ContextualHiddenGraft(
            make_response_variant(SELF, n_in, n_out, width),
            width,
            channels=channels,
        )
        return _matched_self_context(
            n_in, n_out, width, parameter_count(target)
        )
    if name == "cff_contextual_angular":
        return ContextualHiddenGraft(
            make_response_variant(FLOW, n_in, n_out, width), width
        )
    if name == "self_ancient_polynorm":
        return ParallelGraft(make_response_variant(SELF, n_in, n_out, width),
                             AncientPolyNormResidual(n_in, n_out, channels))
    raise KeyError(name)
