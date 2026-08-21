"""A deliberately broad, leakage-safe model screen for the 16-D spiral.

Every model receives only the observed 16-vector.  None receives radius, phase,
turn index, the task rotation, a neighbor, or a future point.  The unusual
layers are generic hypotheses about how a hidden representation might expose
structure; they do not encode a spiral frequency.
"""
from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ML_experiment.continuous_frame_flow import ContinuousFrameFlow
from ML_experiment.models import LELU
from ML_experiment.response_enhanced import RELATIONAL_SCL, make_response_variant


class OrdinaryMLP(nn.Module):
    """The agreed encode -> up -> LELU -> down -> decode reference."""

    def __init__(self, n_in: int, n_out: int, width: int):
        super().__init__()
        self.embed = nn.Linear(n_in, width)
        self.up = nn.Linear(width, 2 * width)
        self.activation = LELU()
        self.down = nn.Linear(2 * width, width)
        self.output = nn.Linear(width, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.down(self.activation(self.up(self.embed(x)))))


class LayerNet(nn.Module):
    def __init__(self, n_in: int, n_out: int, width: int,
                 layer: Callable[[int, int], nn.Module]):
        super().__init__()
        self.embed = nn.Linear(n_in, width)
        self.up = layer(width, 2 * width)
        self.activation = LELU()
        self.down = layer(2 * width, width)
        self.output = nn.Linear(width, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.down(self.activation(self.up(self.embed(x)))))


class DynamicSubspaceLinear(nn.Module):
    """Context-dependent rank-one conduits with a residual complement.

    Each sample chooses several directions.  Their union can span much more
    than the local rank, while no fixed orthogonal partition is imposed.
    """

    def __init__(self, n_in: int, n_out: int, slices: int = 6):
        super().__init__()
        self.direction = nn.ModuleList(
            nn.Linear(n_in, n_in, bias=False) for _ in range(slices)
        )
        self.slice_out = nn.ModuleList(nn.Linear(n_in, n_out) for _ in range(slices))
        self.residual = nn.Linear(n_in, n_out)
        self.mix = nn.Parameter(torch.full((slices,), -0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        components = []
        for direction in self.direction:
            ray = F.normalize(direction(x), dim=-1, eps=1e-6)
            components.append((x * ray).sum(-1, keepdim=True) * ray)
        stacked = torch.stack(components, 1)
        used = stacked.sum(1) / math.sqrt(len(components))
        output = self.residual(x - used)
        for index, head in enumerate(self.slice_out):
            output = output + torch.sigmoid(self.mix[index]) * head(stacked[:, index])
        return output


class OddCubicLinear(nn.Module):
    """Learned low-rank odd third-order tensor sketch.

    The denominator changes cubic growth back toward linear growth at large
    norm.  Unlike a square/phase-retrieval lift, f(-x) = -f(x) before bias.
    """

    def __init__(self, n_in: int, n_out: int, channels: int | None = None):
        super().__init__()
        channels = channels or max(16, min(64, 2 * n_out))
        self.a = nn.Linear(n_in, channels, bias=False)
        self.b = nn.Linear(n_in, channels, bias=False)
        self.c = nn.Linear(n_in, channels, bias=False)
        self.out = nn.Linear(channels, n_out)
        self.base = nn.Linear(n_in, n_out)
        self.scale = nn.Parameter(torch.tensor(-0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b, c = self.a(x), self.b(x), self.c(x)
        normalizer = (a.square() + b.square() + c.square()).clamp_min(1e-6)
        feature = math.sqrt(3.0) * a * b * c / normalizer.sqrt()
        return self.base(x) + F.softplus(self.scale) * self.out(feature)


class BispectralLinear(nn.Module):
    """Learned complex triple products, with no prescribed frequencies."""

    def __init__(self, n_in: int, n_out: int, channels: int | None = None):
        super().__init__()
        channels = channels or max(12, min(48, n_out))
        self.project = nn.Linear(n_in, 6 * channels, bias=False)
        self.out = nn.Linear(2 * channels, n_out)
        self.base = nn.Linear(n_in, n_out)
        self.scale = nn.Parameter(torch.tensor(-0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values = self.project(x).view(len(x), 3, 2, -1)
        z0 = torch.complex(values[:, 0, 0], values[:, 0, 1])
        z1 = torch.complex(values[:, 1, 0], values[:, 1, 1])
        z2 = torch.complex(values[:, 2, 0], values[:, 2, 1])
        triple = z0 * z1 * z2.conj()
        energy = (z0.abs().square() + z1.abs().square() + z2.abs().square()).clamp_min(1e-6)
        triple = math.sqrt(3.0) * triple / energy.sqrt()
        feature = torch.cat((triple.real, triple.imag), -1)
        return self.base(x) + F.softplus(self.scale) * self.out(feature)


class MidpointHessianLinear(nn.Module):
    """A bank of learned cosets exposing gradient and Hessian responses.

    This is a low-rank analogue of reading a hidden direction from the local
    Hessian of a smoothed periodic/coset potential.  It uses neither an
    eigensolver nor a task coordinate.
    """

    def __init__(self, n_in: int, n_out: int, rank: int = 8,
                 anchors: int = 10, probes: int = 4):
        super().__init__()
        rank = min(rank, n_in)
        self.project = nn.Linear(n_in, rank, bias=False)
        self.anchors = nn.Parameter(torch.randn(anchors, rank) / math.sqrt(rank))
        self.probes = nn.Parameter(torch.randn(probes, rank) / math.sqrt(rank))
        self.log_width = nn.Parameter(torch.tensor(0.0))
        self.response = nn.Sequential(
            nn.Linear(anchors * probes * 2, max(16, n_out)), LELU(),
            nn.Linear(max(16, n_out), n_out),
        )
        self.base = nn.Linear(n_in, n_out)
        self.scale = nn.Parameter(torch.tensor(-0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = self.project(x)
        delta = state[:, None, :] - self.anchors[None]
        width = F.softplus(self.log_width) + 0.25
        weight = torch.exp(-0.5 * delta.square().mean(-1) / width.square())
        directional = torch.einsum("bar,pr->bap", delta, F.normalize(self.probes, dim=-1))
        gradient = weight[:, :, None] * directional / width
        hessian = weight[:, :, None] * (directional.square() / width.square() - 1.0)
        feature = torch.cat((gradient, hessian), -1).flatten(1)
        return self.base(x) + F.softplus(self.scale) * self.response(feature)


class SoftHypothesisLinear(nn.Module):
    """Keep several affine support hypotheses alive and mix them softly."""

    def __init__(self, n_in: int, n_out: int, hypotheses: int = 6):
        super().__init__()
        self.experts = nn.ModuleList(nn.Linear(n_in, n_out) for _ in range(hypotheses))
        self.gate = nn.Sequential(
            nn.Linear(n_in + hypotheses, max(12, hypotheses * 2)), LELU(),
            nn.Linear(max(12, hypotheses * 2), hypotheses),
        )
        self.temperature = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        candidates = torch.stack([expert(x) for expert in self.experts], 1)
        disagreement = (candidates - candidates.mean(1, keepdim=True)).square().mean(-1)
        logits = self.gate(torch.cat((x, disagreement), -1))
        temperature = F.softplus(self.temperature) + 0.25
        weight = torch.softmax(logits / temperature, 1)
        return (weight[:, :, None] * candidates).sum(1)


class GaussianDerivativeLinear(nn.Module):
    """Learned variation-diminishing kernels plus their signed derivatives."""

    def __init__(self, n_in: int, n_out: int, channels: int | None = None):
        super().__init__()
        channels = channels or max(16, 2 * n_out)
        self.project = nn.Linear(n_in, channels, bias=False)
        self.center = nn.Parameter(torch.zeros(channels))
        self.log_width = nn.Parameter(torch.zeros(channels))
        self.out = nn.Linear(2 * channels, n_out)
        self.base = nn.Linear(n_in, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        width = F.softplus(self.log_width) + 0.15
        coordinate = (self.project(x) - self.center) / width
        kernel = torch.exp(-0.5 * coordinate.square())
        feature = torch.cat((kernel, coordinate * kernel), -1)
        return self.base(x) + self.out(feature)


class MovingFrameNet(nn.Module):
    """A tied short recurrence: update state, then recompute its geometry."""

    def __init__(self, n_in: int, n_out: int, width: int, steps: int = 4):
        super().__init__()
        self.embed = nn.Linear(n_in, width)
        self.drive = nn.Linear(n_in, width, bias=False)
        self.state = nn.Linear(width, width)
        self.context = OddCubicLinear(width, width, channels=width)
        self.gate = nn.Linear(2 * width, width)
        self.norm = nn.RMSNorm(width)
        self.output = nn.Linear(width, n_out)
        self.steps = steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        source = self.drive(x)
        state = self.embed(x)
        for _ in range(self.steps):
            proposal = self.context(self.norm(state)) + source
            gate = torch.sigmoid(self.gate(torch.cat((state, proposal - state), -1)))
            state = state + gate * LELU()(self.state(proposal) - state) / self.steps
        return self.output(state)


class CayleyFlowNet(nn.Module):
    """Input-conditioned, norm-preserving rank-two rotations of hidden state."""

    def __init__(self, n_in: int, n_out: int, width: int, steps: int = 3):
        super().__init__()
        self.embed = nn.Linear(n_in, width)
        self.u = nn.Linear(width, width, bias=False)
        self.v = nn.Linear(width, width, bias=False)
        self.response = nn.Linear(width, width)
        self.output = nn.Linear(width, n_out)
        self.steps = steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = self.embed(x)
        eye = torch.eye(state.shape[-1], device=x.device, dtype=x.dtype)[None]
        for _ in range(self.steps):
            u, v = F.normalize(self.u(state), dim=-1), F.normalize(self.v(state), dim=-1)
            skew = 0.35 * (u[:, :, None] * v[:, None, :] - v[:, :, None] * u[:, None, :])
            rotated = torch.linalg.solve(eye - 0.5 * skew, (eye + 0.5 * skew) @ state[:, :, None]).squeeze(-1)
            state = state + LELU()(self.response(rotated)) / self.steps
        return self.output(state)


class LivingGraphNet(nn.Module):
    """The hidden coordinates form a sample-conditioned diffusion graph."""

    def __init__(self, n_in: int, n_out: int, width: int, depth: int = 3):
        super().__init__()
        self.embed = nn.Linear(n_in, width)
        self.location = nn.Parameter(torch.linspace(-1, 1, width))
        self.key_scale = nn.Parameter(torch.randn(width) / math.sqrt(width))
        self.mix = nn.ModuleList(nn.Linear(3 * width, width) for _ in range(depth))
        self.output = nn.Linear(width, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = self.embed(x)
        for layer in self.mix:
            key = self.location[None] + self.key_scale[None] * torch.tanh(state)
            distance = (key[:, :, None] - key[:, None, :]).square()
            adjacency = torch.softmax(-4.0 * distance, -1)
            first = (adjacency @ state[:, :, None]).squeeze(-1)
            second = (adjacency @ first[:, :, None]).squeeze(-1)
            state = state + LELU()(layer(torch.cat((state, first, second), -1))) / len(self.mix)
        return self.output(state)


class ShallowOddCubicNet(nn.Module):
    """One odd tensor sketch directly from observation to logits."""

    def __init__(self, n_in: int, n_out: int, width: int):
        super().__init__()
        self.layer = OddCubicLinear(n_in, n_out, channels=max(48, 2 * width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class ShallowBispectrumNet(nn.Module):
    """One learned complex triple product directly to logits."""

    def __init__(self, n_in: int, n_out: int, width: int):
        super().__init__()
        self.layer = BispectralLinear(n_in, n_out, channels=max(48, 2 * width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class FixedRandomBispectrumNet(nn.Module):
    """A non-learned random third-order feature atlas with a linear readout."""

    def __init__(self, n_in: int, n_out: int, width: int, channels: int = 192):
        super().__init__()
        generator = torch.Generator().manual_seed(260808003)
        self.register_buffer(
            "projection", torch.randn(6 * channels, n_in, generator=generator) / math.sqrt(n_in)
        )
        self.output = nn.Linear(2 * channels, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values = F.linear(x, self.projection).view(len(x), 3, 2, -1)
        z0 = torch.complex(values[:, 0, 0], values[:, 0, 1])
        z1 = torch.complex(values[:, 1, 0], values[:, 1, 1])
        z2 = torch.complex(values[:, 2, 0], values[:, 2, 1])
        triple = z0 * z1 * z2.conj()
        energy = (z0.abs().square() + z1.abs().square() + z2.abs().square()).clamp_min(1e-6)
        triple = math.sqrt(3.0) * triple / energy.sqrt()
        return self.output(torch.cat((triple.real, triple.imag), -1))


class EvenQuadraticNet(nn.Module):
    """Deliberate parity control: identical features for x and -x."""

    def __init__(self, n_in: int, n_out: int, width: int, channels: int = 192):
        super().__init__()
        self.project = nn.Linear(n_in, 2 * channels, bias=False)
        self.output = nn.Linear(channels, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.project(x).chunk(2, -1)
        feature = a * b / (a.square() + b.square()).clamp_min(1e-6).sqrt()
        return self.output(feature)


def make_wall_model(name: str, n_in: int, n_out: int, width: int) -> nn.Module:
    if name == "ordinary_mlp":
        return OrdinaryMLP(n_in, n_out, width)
    if name == "self_context":
        return make_response_variant(RELATIONAL_SCL, n_in, n_out, width)
    if name == "cff_fast":
        return ContinuousFrameFlow(n_in, n_out, width=width, fast=True)
    if name == "dynamic_subspace":
        return LayerNet(n_in, n_out, width, DynamicSubspaceLinear)
    if name == "odd_cubic":
        return LayerNet(n_in, n_out, width, OddCubicLinear)
    if name == "learned_bispectrum":
        return LayerNet(n_in, n_out, width, BispectralLinear)
    if name == "midpoint_hessian":
        return LayerNet(n_in, n_out, width, MidpointHessianLinear)
    if name == "soft_hypotheses":
        return LayerNet(n_in, n_out, width, SoftHypothesisLinear)
    if name == "gaussian_derivatives":
        return LayerNet(n_in, n_out, width, GaussianDerivativeLinear)
    if name == "moving_frame":
        return MovingFrameNet(n_in, n_out, width)
    if name == "cayley_flow":
        return CayleyFlowNet(n_in, n_out, width)
    if name == "living_graph":
        return LivingGraphNet(n_in, n_out, width)
    if name == "shallow_odd_cubic":
        return ShallowOddCubicNet(n_in, n_out, width)
    if name == "shallow_bispectrum":
        return ShallowBispectrumNet(n_in, n_out, width)
    if name == "fixed_random_bispectrum":
        return FixedRandomBispectrumNet(n_in, n_out, width)
    if name == "even_quadratic_control":
        return EvenQuadraticNet(n_in, n_out, width)
    raise KeyError(name)


WALL_MODELS = (
    "ordinary_mlp",
    "self_context",
    "cff_fast",
    "dynamic_subspace",
    "odd_cubic",
    "learned_bispectrum",
    "midpoint_hessian",
    "soft_hypotheses",
    "gaussian_derivatives",
    "moving_frame",
    "cayley_flow",
    "living_graph",
    "shallow_odd_cubic",
    "shallow_bispectrum",
    "fixed_random_bispectrum",
    "even_quadratic_control",
)
