#!/usr/bin/env python3
"""Focused diagnosis of the periodic N-D regression failure.

The target is a sum of independent one-dimensional functions.  This study
separates three questions which a scalar test score conflates:

1. can the optimizer acquire the individual coordinate harmonics;
2. can the model discover the frame in which the function is separable; and
3. does its learned curvature remain in one commuting frame, as the truth does.

The project models are compared with generic ridge-chart banks.  The latter do
not contain sine or Fourier features: each chart is an ordinary two-stage LELU
network living on a scalar learned projection.  Explicit sinusoidal models are
included only as diagnostic ceilings and are named accordingly.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from ML_experiment.metrics import evaluate
from ML_experiment.models import LELU, parameter_count
from ML_experiment.odd_context_hybrids import make_hybrid
from ML_experiment.tasks import periodic_nd


PROJECT_VARIANTS = {
    "mlp_26k": "ordinary_mlp_cone_budget",
    "self_context": "self_context",
    "cff": "cff",
    "learned_cone": "self_contextual_full_learned_cone",
    "operator_sphere": "self_contextual_operator_sphere_global_r2",
}


class DenseLELUNet(nn.Module):
    """Ordinary dense LELU control with no geometric structure."""

    def __init__(self, n_in: int, n_out: int, width: int = 96, depth: int = 3):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(n_in, width), LELU()]
        for _ in range(depth - 1):
            layers.extend((nn.Linear(width, width), LELU()))
        layers.append(nn.Linear(width, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ScalarChartBank(nn.Module):
    """A sum of generic one-dimensional LELU charts on projection rays.

    No periodic activation is present.  The restriction is instead geometric:
    each nonlinear response belongs to one scalar chart, so the model does not
    have to undo arbitrary cross-dimensional mixing to represent an additive
    function.  ``frame_mode`` controls whether the rays are the observed axes,
    a learned orthogonal frame, or an overcomplete learned atlas.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        *,
        rays: int,
        units: int,
        frame_mode: str,
        seed: int = 7813,
    ):
        super().__init__()
        if n_out != 1:
            raise ValueError("the focused scalar chart bank expects one output")
        if frame_mode not in {"axes", "orthogonal", "orthogonal_identity", "qr", "atlas"}:
            raise ValueError(frame_mode)
        if frame_mode in {"axes", "orthogonal", "orthogonal_identity", "qr"} and rays != n_in:
            raise ValueError("axis and orthogonal frames require rays == n_in")
        self.n_in = n_in
        self.rays = rays
        self.units = units
        self.frame_mode = frame_mode

        generator = torch.Generator().manual_seed(seed + n_in + rays + units)
        if frame_mode == "axes":
            self.register_buffer("fixed_frame", torch.eye(n_in))
            self.frame_delta = None
            self.frame = None
        elif frame_mode in {"orthogonal", "orthogonal_identity"}:
            if frame_mode == "orthogonal_identity":
                initial = torch.eye(n_in)
            else:
                initial, _ = torch.linalg.qr(
                    torch.randn(n_in, n_in, generator=generator)
                )
            self.register_buffer("initial_frame", initial.T)
            self.frame_delta = nn.Parameter(torch.zeros(n_in, n_in))
            self.frame = None
        elif frame_mode == "qr":
            initial, _ = torch.linalg.qr(
                torch.randn(n_in, n_in, generator=generator)
            )
            self.frame_raw = nn.Parameter(initial.T)
            self.frame_delta = None
            self.frame = None
        else:
            self.frame = nn.Parameter(
                F.normalize(torch.randn(rays, n_in, generator=generator), dim=1)
            )
            self.frame_delta = None

        # A broad, nonperiodic bank of slopes ensures that acquisition does not
        # begin with only the lowest spatial scale.  Biases tile the observed
        # scalar range; both remain fully trainable.
        base_slopes = torch.logspace(math.log10(0.35), math.log10(6.0), units)
        permutation = torch.randperm(units, generator=generator)
        slopes = base_slopes[permutation][None].repeat(rays, 1)
        slopes *= torch.where(
            torch.rand((rays, units), generator=generator) > 0.5, 1.0, -1.0
        )
        self.input_scale = nn.Parameter(slopes)
        self.input_bias = nn.Parameter(
            torch.linspace(-math.pi, math.pi, units)[None].repeat(rays, 1)
        )
        self.mix = nn.Parameter(
            torch.randn(rays, units, units, generator=generator) / math.sqrt(units)
        )
        self.mix_bias = nn.Parameter(torch.zeros(rays, units))
        self.readout = nn.Parameter(
            torch.randn(rays, units, generator=generator)
            / math.sqrt(rays * units)
        )
        self.output_bias = nn.Parameter(torch.zeros(1))

    def rays_matrix(self) -> torch.Tensor:
        if self.frame_mode == "axes":
            return self.fixed_frame
        if self.frame_mode in {"orthogonal", "orthogonal_identity"}:
            skew = self.frame_delta - self.frame_delta.T
            return torch.matrix_exp(skew) @ self.initial_frame
        if self.frame_mode == "qr":
            frame, triangular = torch.linalg.qr(self.frame_raw.T)
            sign = torch.diagonal(triangular).sign().detach()
            sign = torch.where(sign == 0, torch.ones_like(sign), sign)
            return (frame * sign).T
        return F.normalize(self.frame, dim=1)

    def chart_values(self, x: torch.Tensor) -> torch.Tensor:
        projected = x @ self.rays_matrix().T
        first = LELU()(projected[:, :, None] * self.input_scale + self.input_bias)
        second = torch.einsum("bru,ruv->brv", first, self.mix)
        second = LELU()(second / math.sqrt(self.units) + self.mix_bias)
        return torch.einsum("bru,ru->br", second, self.readout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.chart_values(x).sum(1, keepdim=True) + self.output_bias


class SineDiagnosticCeiling(nn.Module):
    """A task-aligned periodic ceiling, not a candidate general layer."""

    def __init__(self, n_in: int, n_out: int, width: int = 96):
        super().__init__()
        if n_out != 1:
            raise ValueError
        self.project = nn.Linear(n_in, width)
        self.output = nn.Linear(width, 1)
        with torch.no_grad():
            self.project.weight.uniform_(-2.0, 2.0)
            self.project.bias.uniform_(-math.pi, math.pi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(torch.sin(self.project(x)))


class FourierOracleCeiling(nn.Module):
    """Linear regression on explicit per-axis harmonics, for diagnosis only."""

    def __init__(self, n_in: int, n_out: int, maximum_frequency: int = 6):
        super().__init__()
        if n_out != 1:
            raise ValueError
        self.maximum_frequency = maximum_frequency
        self.output = nn.Linear(2 * n_in * maximum_frequency, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frequency = torch.arange(
            1, self.maximum_frequency + 1, device=x.device, dtype=x.dtype
        )
        phase = x[:, :, None] * frequency
        features = torch.cat((torch.sin(phase), torch.cos(phase)), dim=2)
        return self.output(features.flatten(1))


class ParallelChartGraft(nn.Module):
    """Keep a project model intact while adding a commuting chart residual."""

    def __init__(self, parent: nn.Module, chart: ScalarChartBank):
        super().__init__()
        self.parent = parent
        self.chart = chart
        self.chart_scale = nn.Parameter(torch.tensor(-1.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.parent(x) + F.softplus(self.chart_scale) * self.chart(x)


def make_model(name: str, n_in: int, n_out: int, width: int) -> nn.Module:
    if name in PROJECT_VARIANTS:
        return make_hybrid(PROJECT_VARIANTS[name], n_in, n_out, width)
    if name == "dense_lelu_96":
        return DenseLELUNet(n_in, n_out, width=96, depth=3)
    if name == "axis_chart_16":
        return ScalarChartBank(n_in, n_out, rays=n_in, units=16, frame_mode="axes")
    if name == "axis_chart_32":
        return ScalarChartBank(n_in, n_out, rays=n_in, units=32, frame_mode="axes")
    if name == "orthogonal_chart_24":
        return ScalarChartBank(
            n_in, n_out, rays=n_in, units=24, frame_mode="orthogonal"
        )
    if name == "orthogonal_chart_24_fast":
        model = ScalarChartBank(
            n_in, n_out, rays=n_in, units=24, frame_mode="orthogonal"
        )
        model.frame_lr_multiplier = 5.0
        return model
    if name == "orthogonal_identity_24":
        return ScalarChartBank(
            n_in, n_out, rays=n_in, units=24, frame_mode="orthogonal_identity"
        )
    if name == "qr_chart_24":
        return ScalarChartBank(n_in, n_out, rays=n_in, units=24, frame_mode="qr")
    if name == "free_chart_8x24":
        return ScalarChartBank(n_in, n_out, rays=8, units=24, frame_mode="atlas")
    if name == "self_commuting_chart":
        return ParallelChartGraft(
            make_hybrid("self_context", n_in, n_out, width),
            ScalarChartBank(
                n_in, n_out, rays=n_in, units=24,
                frame_mode="orthogonal_identity",
            ),
        )
    if name == "cone_commuting_chart":
        return ParallelChartGraft(
            make_hybrid(
                "self_contextual_full_learned_cone", n_in, n_out, width
            ),
            ScalarChartBank(
                n_in, n_out, rays=n_in, units=24,
                frame_mode="orthogonal_identity",
            ),
        )
    if name == "atlas_chart_16x16":
        return ScalarChartBank(n_in, n_out, rays=16, units=16, frame_mode="atlas")
    if name == "atlas_chart_32x12":
        return ScalarChartBank(n_in, n_out, rays=32, units=12, frame_mode="atlas")
    if name == "sine_ceiling":
        return SineDiagnosticCeiling(n_in, n_out)
    if name == "fourier_oracle":
        return FourierOracleCeiling(n_in, n_out)
    raise KeyError(name)


def train_model(
    model: nn.Module,
    task,
    *,
    seed: int,
    steps: int,
    batch: int,
    lr: float,
    evaluate_every: int,
):
    frame_names = {"frame", "frame_delta", "frame_raw"}
    frame_parameters = [
        parameter for name, parameter in model.named_parameters()
        if name.rsplit(".", 1)[-1] in frame_names
    ]
    frame_ids = {id(parameter) for parameter in frame_parameters}
    other_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in frame_ids
    ]
    multiplier = float(getattr(model, "frame_lr_multiplier", 1.0))
    groups = [{"params": other_parameters, "lr": lr}]
    if frame_parameters:
        groups.append({"params": frame_parameters, "lr": lr * multiplier})
    optimizer = torch.optim.AdamW(groups, lr=lr, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(91000 + seed)
    history = []
    best = None
    started = time.perf_counter()
    for step in range(1, steps + 1):
        indices = torch.randint(len(task.x_train), (batch,), generator=generator)
        x, y = task.x_train[indices], task.y_train[indices]
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        if step == 1 or step % evaluate_every == 0 or step == steps:
            metrics = evaluate(model, task, task.x_val, task.y_val)
            point = {"step": step, "loss": float(loss.detach()), **metrics}
            history.append(point)
            if best is None or metrics["r2"] > best[0]:
                best = (metrics["r2"], copy.deepcopy(model.state_dict()), step)
    seconds = time.perf_counter() - started
    model.load_state_dict(best[1])
    return history, seconds, best[2]


def _correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten() - a.mean()
    b = b.flatten() - b.mean()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-8))


@torch.no_grad()
def partial_dependence(model: nn.Module, task, *, points: int = 129, contexts: int = 128):
    model.eval()
    dim = task.input_dim
    grid = torch.linspace(-math.pi, math.pi, points)
    anchors = task.x_val[:contexts].clone()
    target_std = float(task.target_std.squeeze())
    frequencies = torch.arange(dim).remainder(5) + 1
    curves = []
    for axis in range(dim):
        batches = anchors[None].repeat(points, 1, 1)
        batches[:, :, axis] = grid[:, None]
        prediction = model(batches.flatten(0, 1)).view(points, contexts).mean(1)
        prediction = prediction - prediction.mean()
        truth = (
            torch.cos(grid * frequencies[axis]) + 0.3 * torch.sin(2 * grid)
        ) / (dim * target_std)
        truth = truth - truth.mean()
        residual = (prediction - truth).square().mean()
        variance = truth.square().mean().clamp_min(1e-9)
        curves.append({
            "axis": axis,
            "frequency": int(frequencies[axis]),
            "correlation": _correlation(prediction, truth),
            "r2": float(1.0 - residual / variance),
            "amplitude_ratio": float(prediction.std() / truth.std().clamp_min(1e-8)),
            "prediction": [round(float(v), 6) for v in prediction],
            "truth": [round(float(v), 6) for v in truth],
        })
    return {
        "grid": [round(float(v), 6) for v in grid],
        "curves": curves,
        "mean_correlation": sum(row["correlation"] for row in curves) / dim,
        "mean_r2": sum(row["r2"] for row in curves) / dim,
        "mean_amplitude_ratio": sum(row["amplitude_ratio"] for row in curves) / dim,
    }


def differential_diagnostics(model: nn.Module, task, maximum: int = 8):
    """Measure gradient recovery and whether learned Hessians commute."""
    model.eval()
    points = task.x_val[:maximum].detach()
    frequencies = torch.arange(task.input_dim, dtype=points.dtype).remainder(5) + 1
    target_std = float(task.target_std.squeeze())
    predicted_gradients = []
    true_gradients = []
    hessians = []
    for point in points:
        local = point.clone().requires_grad_(True)
        value = model(local[None]).sum()
        gradient = torch.autograd.grad(value, local, create_graph=True)[0]
        rows = []
        for index in range(task.input_dim):
            rows.append(torch.autograd.grad(
                gradient[index], local, retain_graph=True
            )[0])
        hessians.append(torch.stack(rows).detach())
        predicted_gradients.append(gradient.detach())
        true_gradients.append((
            -frequencies * torch.sin(frequencies * point)
            + 0.6 * torch.cos(2 * point)
        ) / (task.input_dim * target_std))
    predicted_gradient = torch.stack(predicted_gradients)
    true_gradient = torch.stack(true_gradients)
    hessian = torch.stack(hessians)
    diagonal = torch.diagonal(hessian, dim1=1, dim2=2)
    off_diagonal = hessian - torch.diag_embed(diagonal)
    off_ratio = float(
        off_diagonal.square().sum().sqrt()
        / hessian.square().sum().sqrt().clamp_min(1e-8)
    )
    commutators = []
    for index in range(1, len(hessian)):
        a, b = hessian[0], hessian[index]
        commutator = a @ b - b @ a
        scale = (a @ b).norm() + (b @ a).norm()
        commutators.append(float(commutator.norm() / scale.clamp_min(1e-8)))
    return {
        "gradient_correlation": _correlation(predicted_gradient, true_gradient),
        "gradient_relative_error": float(
            (predicted_gradient - true_gradient).norm()
            / true_gradient.norm().clamp_min(1e-8)
        ),
        "hessian_off_diagonal_ratio": off_ratio,
        "hessian_commutator_ratio": sum(commutators) / max(1, len(commutators)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/tmp/periodic_nd_study"))
    parser.add_argument(
        "--variants",
        default=(
            "mlp_26k,self_context,cff,learned_cone,operator_sphere,"
            "dense_lelu_96,axis_chart_16,axis_chart_32,orthogonal_chart_24,"
            "atlas_chart_16x16,atlas_chart_32x12,sine_ceiling,fourier_oracle"
        ),
    )
    parser.add_argument("--width", type=int, default=38)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    partial = args.out / "results.partial.json"
    payload = json.loads(partial.read_text()) if args.resume and partial.exists() else {"runs": []}
    done = {(row["variant"], row["seed"]) for row in payload["runs"]}
    for seed in range(args.seeds):
        task = periodic_nd(seed)
        for name in args.variants.split(","):
            if (name, seed) in done:
                continue
            torch.manual_seed(90000 + seed)
            model = make_model(name, task.input_dim, task.output_dim, args.width)
            print(f"START variant={name} seed={seed} params={parameter_count(model)}", flush=True)
            history, seconds, best_step = train_model(
                model, task, seed=seed, steps=args.steps, batch=args.batch,
                lr=args.lr, evaluate_every=args.eval_every,
            )
            metrics = evaluate(model, task)
            partials = partial_dependence(model, task)
            differentials = differential_diagnostics(model, task)
            row = {
                "variant": name,
                "seed": seed,
                "parameters": parameter_count(model),
                "seconds": seconds,
                "best_step": best_step,
                **metrics,
                **differentials,
                "partial_dependence": partials,
                "history": history,
            }
            payload["runs"].append(row)
            partial.write_text(json.dumps(payload, indent=2))
            print(json.dumps({
                "variant": name,
                "r2": round(row["r2"], 5),
                "score": round(row["score"], 5),
                "partial_r2": round(partials["mean_r2"], 5),
                "gradient_correlation": round(row["gradient_correlation"], 5),
                "hessian_offdiag": round(row["hessian_off_diagonal_ratio"], 5),
                "seconds": round(seconds, 3),
            }), flush=True)
    configuration = {**vars(args), "out": str(args.out)}
    final = {"configuration": configuration, "runs": payload["runs"]}
    (args.out / "results.json").write_text(json.dumps(final, indent=2))
    print(json.dumps({"complete": True, "runs": len(payload["runs"])}))


if __name__ == "__main__":
    main()
