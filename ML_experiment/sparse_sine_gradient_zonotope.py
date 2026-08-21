#!/usr/bin/env python3
"""Continuous low-rank candidate-gradient selection on the sparse sine.

Several globally support-balanced gradients define a tiny empirical update
cloud.  Its mean is the common descent signal; an SVD of the centered cloud
supplies the anisotropic directions.  A structured witness fold selects a
continuous point in that update zonotope, and exactly one AdamW update is
committed to the model.
"""
from __future__ import annotations

import argparse
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
from torch.func import functional_call

from ML_experiment.models import parameter_count
from ML_experiment.sparse_sine_study import (
    AcquisitionData,
    SupportWhitened,
    curve_metrics,
    curve_probe,
    make_model,
    train,
)
from ML_experiment.sparse_sine_witness_descent import InterleavedWitnessAtlas
from ML_experiment.tasks import sparse_sine_1d


torch.set_num_threads(8)


def gradient_covariance_frame(
    gradients: torch.Tensor, rank: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return mean gradient, covariance-scaled axes, and retained variance."""
    if gradients.ndim != 2 or len(gradients) < 2:
        raise ValueError("gradients must contain at least two flattened samples")
    common = gradients.mean(0)
    residual = gradients - common
    _, singular, vh = torch.linalg.svd(residual, full_matrices=False)
    usable = min(int(rank), len(gradients) - 1, len(singular))
    scale = singular[:usable] / math.sqrt(max(1, len(gradients) - 1))
    directions = scale[:, None] * vh[:usable]
    variance = singular.square()
    retained = variance[:usable].sum() / variance.sum().clamp_min(1e-12)
    return common, directions, retained


def _named_parameters(model: nn.Module):
    return [(name, parameter) for name, parameter in model.named_parameters()
            if parameter.requires_grad]


def _flatten_gradients(named_parameters) -> torch.Tensor:
    chunks = []
    for _, parameter in named_parameters:
        if parameter.grad is None:
            chunks.append(torch.zeros_like(parameter).flatten())
        else:
            chunks.append(parameter.grad.detach().flatten())
    return torch.cat(chunks)


def _split_vector(vector: torch.Tensor, named_parameters):
    result = []
    offset = 0
    for name, parameter in named_parameters:
        count = parameter.numel()
        result.append((name, parameter, vector[offset:offset + count].view_as(parameter)))
        offset += count
    if offset != vector.numel():
        raise ValueError("flat update does not match the model parameters")
    return result


def preview_adamw(
    model: nn.Module,
    optimizer: torch.optim.AdamW,
    gradient: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Differentiably preview the next AdamW parameters for a flat gradient."""
    return preview_adamw_transition(model, optimizer, gradient)[0]


def preview_adamw_transition(
    model: nn.Module,
    optimizer: torch.optim.AdamW,
    gradient: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, torch.Tensor | int]]]:
    """Preview both parameters and moment state for one AdamW transition."""
    named = _named_parameters(model)
    group_by_parameter = {
        parameter: group
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    candidate = {}
    transition = {}
    for name, parameter, grad in _split_vector(gradient, named):
        group = group_by_parameter[parameter]
        beta1, beta2 = group["betas"]
        state = optimizer.state.get(parameter, {})
        exp_avg = state.get("exp_avg", torch.zeros_like(parameter))
        exp_avg_sq = state.get("exp_avg_sq", torch.zeros_like(parameter))
        old_step = state.get("step", 0)
        old_step = int(old_step.item()) if torch.is_tensor(old_step) else int(old_step)
        step = old_step + 1
        next_avg = beta1 * exp_avg + (1.0 - beta1) * grad
        next_sq = beta2 * exp_avg_sq + (1.0 - beta2) * grad.square()
        bias1 = 1.0 - beta1 ** step
        bias2 = 1.0 - beta2 ** step
        denominator = next_sq.sqrt() / math.sqrt(bias2) + group["eps"]
        decayed = parameter * (1.0 - group["lr"] * group["weight_decay"])
        candidate[name] = decayed - (group["lr"] / bias1) * next_avg / denominator
        transition[name] = {
            "exp_avg": next_avg,
            "exp_avg_sq": next_sq,
            "step": step,
        }
    return candidate, transition


def commit_adamw(
    model: nn.Module,
    optimizer: torch.optim.AdamW,
    gradient: torch.Tensor,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    for _, parameter, grad in _split_vector(gradient, _named_parameters(model)):
        parameter.grad = grad.detach().clone()
    optimizer.step()


@torch.no_grad()
def commit_transition_mixture(
    model: nn.Module,
    optimizer: torch.optim.AdamW,
    candidate_parameters: list[dict[str, torch.Tensor]],
    candidate_states: list[dict[str, dict[str, torch.Tensor | int]]],
    weight: torch.Tensor,
) -> None:
    """Commit a barycenter of parameter and Adam moment histories."""
    for name, parameter in _named_parameters(model):
        value = sum(
            weight[index] * candidate_parameters[index][name]
            for index in range(len(candidate_parameters))
        )
        parameter.copy_(value)
        state = optimizer.state[parameter]
        state["exp_avg"] = sum(
            weight[index] * candidate_states[index][name]["exp_avg"]
            for index in range(len(candidate_states))
        ).detach().clone()
        state["exp_avg_sq"] = sum(
            weight[index] * candidate_states[index][name]["exp_avg_sq"]
            for index in range(len(candidate_states))
        ).detach().clone()
        step = int(candidate_states[0][name]["step"])
        old_step = state.get("step")
        if torch.is_tensor(old_step):
            state["step"] = old_step.new_tensor(float(step))
        else:
            state["step"] = torch.tensor(float(step))
    optimizer.zero_grad(set_to_none=True)


def _functional_prediction(
    model: nn.Module, parameters: dict[str, torch.Tensor], x: torch.Tensor
) -> torch.Tensor:
    state = {**dict(model.named_buffers()), **parameters}
    return functional_call(model, state, (x,))


def smooth_witness_objective(
    model: nn.Module,
    candidate_parameters: dict[str, torch.Tensor],
    atlas: InterleavedWitnessAtlas,
    held_out: int,
    *,
    worst_weight: float,
    temperature: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    index = atlas.witness(held_out)
    prediction = _functional_prediction(model, candidate_parameters, atlas.data.x[index])
    error = (prediction - atlas.data.y[index]).square().mean(-1)
    weight = atlas.data.voronoi[index, 0]
    mean = (weight * error).sum() / weight.sum().clamp_min(1e-12)
    cell_losses = []
    for cell_index in range(atlas.cells):
        selected = atlas.cell[index] == cell_index
        if selected.any():
            local_weight = weight[selected]
            cell_losses.append(
                (local_weight * error[selected]).sum()
                / local_weight.sum().clamp_min(1e-12)
            )
    cells = torch.stack(cell_losses)
    top_count = max(1, math.ceil(0.25 * len(cells)))
    smooth_worst = cells.topk(top_count).values.mean()
    return mean + worst_weight * smooth_worst, mean, smooth_worst


@torch.no_grad()
def quadratic_zonotope_selection(
    model: nn.Module,
    optimizer: torch.optim.AdamW,
    common: torch.Tensor,
    directions: torch.Tensor,
    atlas: InterleavedWitnessAtlas,
    held_out: int,
    *,
    radius: float,
    probe: float,
    coefficient_penalty: float,
    worst_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Fit separable local witness quadratics and select an interior point.

    Only forward evaluations are required.  The final tanh map keeps every
    coefficient continuous and strictly inside the zonotope rather than
    snapping it to a face.
    """
    rank = len(directions)

    def evaluate(coefficient: torch.Tensor) -> float:
        gradient = common + torch.einsum("r,rp->p", coefficient, directions)
        parameters = preview_adamw(model, optimizer, gradient)
        objective, _, _ = smooth_witness_objective(
            model, parameters, atlas, held_out, worst_weight=worst_weight
        )
        objective = objective + coefficient_penalty * coefficient.square().mean()
        return float(objective)

    zero = torch.zeros(rank)
    center = evaluate(zero)
    probe = min(float(probe), float(radius))
    raw = torch.zeros(rank)
    curvatures = []
    for axis in range(rank):
        offset = torch.zeros(rank)
        offset[axis] = probe
        plus = evaluate(offset)
        minus = evaluate(-offset)
        slope = (plus - minus) / (2.0 * probe)
        curvature = (plus + minus - 2.0 * center) / (probe * probe)
        curvatures.append(curvature)
        if curvature > 1e-8:
            raw[axis] = -slope / curvature
        else:
            # Locally monotone directions still remain inside the smooth shell.
            raw[axis] = -radius * math.copysign(1.0, slope) if slope else 0.0
    coefficient = radius * torch.tanh(raw / max(radius, 1e-8))

    # Cross-axis curvature was not fitted. A tiny radial trust search retains
    # the continuous direction while preventing an adverse combined move.
    scales = (1.0, 0.5, 0.25, 0.0)
    candidates = [(scale, evaluate(scale * coefficient)) for scale in scales]
    scale, selected_score = min(candidates, key=lambda row: row[1])
    coefficient = scale * coefficient
    return coefficient, {
        "center_score": center,
        "selected_score": selected_score,
        "selected_scale": scale,
        "positive_curvature_fraction": sum(value > 0 for value in curvatures)
        / max(1, len(curvatures)),
    }


@torch.no_grad()
def covariance_ray_selection(
    model: nn.Module,
    optimizer: torch.optim.AdamW,
    gradients: torch.Tensor,
    common: torch.Tensor,
    directions: torch.Tensor,
    atlas: InterleavedWitnessAtlas,
    held_out: int,
    *,
    radius: float,
    coefficient_penalty: float,
    worst_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Select and continuously refine a candidate ray in covariance space."""
    rank = len(directions)

    def evaluate(coefficient: torch.Tensor) -> float:
        gradient = common + torch.einsum("r,rp->p", coefficient, directions)
        parameters = preview_adamw(model, optimizer, gradient)
        objective, _, _ = smooth_witness_objective(
            model, parameters, atlas, held_out, worst_weight=worst_weight
        )
        objective = objective + coefficient_penalty * coefficient.square().mean()
        return float(objective)

    denominator = directions.square().sum(-1).clamp_min(1e-12)
    # Orthogonal covariance axes make this the least-squares coordinate of
    # each observed gradient residual in the retained update frame.
    coordinates = torch.einsum(
        "bp,rp->br", gradients - common, directions
    ) / denominator
    maximum = coordinates.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    coordinates = coordinates * torch.clamp(radius / maximum, max=1.0)
    zero = torch.zeros(rank)
    center_score = evaluate(zero)
    endpoint_scores = [evaluate(value) for value in coordinates]
    best_index = min(range(len(endpoint_scores)), key=endpoint_scores.__getitem__)
    ray = coordinates[best_index]
    endpoint_score = endpoint_scores[best_index]
    midpoint_score = evaluate(0.5 * ray)

    # f(t)=a*t^2+b*t+c through t={0,.5,1}; the stationary point is a
    # continuous selection along the retained ray rather than a branch ID.
    d_half = midpoint_score - center_score
    d_end = endpoint_score - center_score
    curvature = 2.0 * d_end - 4.0 * d_half
    slope = 4.0 * d_half - d_end
    if curvature > 1e-12:
        raw_t = -slope / (2.0 * curvature)
    else:
        raw_t = 1.0 if endpoint_score < center_score else 0.0
    fitted_t = min(1.0, max(0.0, raw_t))
    fitted = fitted_t * ray
    fitted_score = evaluate(fitted)

    options = [
        (0.0, zero, center_score),
        (0.5, 0.5 * ray, midpoint_score),
        (1.0, ray, endpoint_score),
        (fitted_t, fitted, fitted_score),
    ]
    selected_t, coefficient, selected_score = min(options, key=lambda row: row[2])
    return coefficient, {
        "center_score": center_score,
        "selected_score": selected_score,
        "selected_scale": selected_t,
        "positive_curvature_fraction": float(curvature > 0.0),
        "selected_ray": float(best_index),
    }


@torch.no_grad()
def soft_barycentric_selection(
    model: nn.Module,
    optimizer: torch.optim.AdamW,
    gradients: torch.Tensor,
    common: torch.Tensor,
    directions: torch.Tensor,
    atlas: InterleavedWitnessAtlas,
    held_out: int,
    *,
    radius: float,
    coefficient_penalty: float,
    worst_weight: float,
    temperature: float,
    score_ema: torch.Tensor | None,
    decay: float,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    """Continuously retain witness-compatible candidate coordinates.

    The endpoint scores are converted to barycentric weights.  Low temperature
    approaches discrete branch retention, while near-tied candidates combine
    inside the covariance zonotope instead of forcing an arbitrary winner.
    """
    denominator = directions.square().sum(-1).clamp_min(1e-12)
    coordinates = torch.einsum(
        "bp,rp->br", gradients - common, directions
    ) / denominator
    maximum = coordinates.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    coordinates = coordinates * torch.clamp(radius / maximum, max=1.0)

    scores = []
    for coefficient in coordinates:
        gradient = common + torch.einsum("r,rp->p", coefficient, directions)
        parameters = preview_adamw(model, optimizer, gradient)
        objective, _, _ = smooth_witness_objective(
            model, parameters, atlas, held_out, worst_weight=worst_weight
        )
        objective = objective + coefficient_penalty * coefficient.square().mean()
        scores.append(objective)
    scores = torch.stack(scores)
    relative = (scores - scores.mean()) / scores.abs().mean().clamp_min(1e-8)
    if score_ema is None:
        score_ema = relative
    else:
        score_ema = decay * score_ema + (1.0 - decay) * relative
    weight = torch.softmax(-score_ema / max(temperature, 1e-8), dim=0)
    coefficient = torch.einsum("b,br->r", weight, coordinates)
    entropy = -(weight * weight.clamp_min(1e-12).log()).sum()
    return coefficient, {
        "center_score": math.nan,
        "selected_score": float((weight * scores).sum()),
        "selected_scale": float(weight.max()),
        "positive_curvature_fraction": math.nan,
        "selected_ray": float(weight.argmax()),
        "selection_entropy": float(entropy),
    }, score_ema


@torch.no_grad()
def transported_soft_selection(
    model: nn.Module,
    optimizer: torch.optim.AdamW,
    gradients: torch.Tensor,
    atlas: InterleavedWitnessAtlas,
    held_out: int,
    *,
    rank: int,
    temperature: float,
    score_ema: torch.Tensor | None,
    decay: float,
    worst_weight: float,
) -> tuple[
    torch.Tensor, dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor,
]:
    """Reduce and mix candidate histories in Adam-transported update space."""
    named = _named_parameters(model)
    candidate_parameters = []
    candidate_states = []
    displacements = []
    for gradient in gradients:
        parameters, states = preview_adamw_transition(model, optimizer, gradient)
        candidate_parameters.append(parameters)
        candidate_states.append(states)
        displacements.append(torch.cat([
            (parameters[name] - parameter).flatten()
            for name, parameter in named
        ]))
    displacements = torch.stack(displacements)
    common, directions, retained = gradient_covariance_frame(displacements, rank)
    denominator = directions.square().sum(-1).clamp_min(1e-12)
    coordinates = torch.einsum(
        "bp,rp->br", displacements - common, directions
    ) / denominator
    projected = common + torch.einsum("br,rp->bp", coordinates, directions)

    scores = []
    projected_parameters = []
    for displacement in projected:
        values = {}
        offset = 0
        for name, parameter in named:
            count = parameter.numel()
            values[name] = parameter + displacement[offset:offset + count].view_as(parameter)
            offset += count
        projected_parameters.append(values)
        objective, _, _ = smooth_witness_objective(
            model, values, atlas, held_out, worst_weight=worst_weight
        )
        scores.append(objective)
    scores = torch.stack(scores)
    relative = (scores - scores.mean()) / scores.abs().mean().clamp_min(1e-8)
    if score_ema is None:
        score_ema = relative
    else:
        score_ema = decay * score_ema + (1.0 - decay) * relative
    weight = torch.softmax(-score_ema / max(temperature, 1e-8), dim=0)

    # Parameter values use the reduced displacement geometry. Optimizer moments
    # use the same barycentric weights over the authentic candidate histories.
    commit_transition_mixture(
        model, optimizer, projected_parameters, candidate_states, weight
    )
    coefficient = torch.einsum("b,br->r", weight, coordinates)
    entropy = -(weight * weight.clamp_min(1e-12).log()).sum()
    selection = {
        "center_score": math.nan,
        "selected_score": float((weight * scores).sum()),
        "selected_scale": float(weight.max()),
        "positive_curvature_fraction": math.nan,
        "selected_ray": float(weight.argmax()),
        "selection_entropy": float(entropy),
    }
    selected_displacement = torch.einsum("b,bp->p", weight, projected)
    return coefficient, selection, score_ema, common, selected_displacement, retained


def train_gradient_zonotope(
    task,
    *,
    model_name: str,
    width: int,
    seed: int,
    steps: int,
    gradient_samples: int,
    covariance_rank: int,
    radius: float,
    selector: str,
    quadratic_probe: float,
    selection_temperature: float,
    selection_decay: float,
    outer_steps: int,
    outer_lr: float,
    outer_optimizer_name: str,
    coefficient_penalty: float,
    batch: int,
    lr: float,
    cells: int,
    folds: int,
    witness_period: int,
    worst_weight: float,
    evaluate_every: int,
):
    torch.manual_seed(61000 + seed)
    data = AcquisitionData(task)
    model = SupportWhitened(make_model(model_name, width), data)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    atlas = InterleavedWitnessAtlas(data, cells=cells, folds=folds)
    named = _named_parameters(model)
    history = []
    coefficient_norms = []
    retained_variances = []
    selection_score_ema = None
    selection_fold = None
    started = time.perf_counter()

    for step in range(steps):
        held_out = (step // witness_period) % folds
        if held_out != selection_fold:
            selection_score_ema = None
            selection_fold = held_out
        gradients = []
        for sample in range(gradient_samples):
            generator = torch.Generator().manual_seed(
                71000 + 1000003 * seed + 1009 * step + 37 * sample
            )
            phase = (sample * 0.6180339887498949 + step * 0.4142135623730950) % 1.0
            index = atlas.stratified_train(
                batch, held_out, generator, phase=phase
            )
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(data.x[index]), data.y[index])
            loss.backward()
            gradient = _flatten_gradients(named)
            norm = gradient.norm().clamp_min(1e-12)
            # Match the experiment's existing clipping before constructing the
            # covariance geometry, without erasing relative gradient scale.
            gradient = gradient * torch.clamp(torch.tensor(10.0) / norm, max=1.0)
            gradients.append(gradient)

        gradient_matrix = torch.stack(gradients)
        transported = selector == "transport"
        if transported:
            (
                coefficient, selection, selection_score_ema, common,
                selected_update, retained,
            ) = transported_soft_selection(
                model, optimizer, gradient_matrix, atlas, held_out,
                rank=covariance_rank, temperature=selection_temperature,
                score_ema=selection_score_ema, decay=selection_decay,
                worst_weight=worst_weight,
            )
            directions = None
            selected_gradient = None
        else:
            common, directions, retained = gradient_covariance_frame(
                gradient_matrix, covariance_rank
            )
        if selector == "softmax":
            coefficient, selection, selection_score_ema = soft_barycentric_selection(
                model, optimizer, gradient_matrix, common, directions,
                atlas, held_out, radius=radius,
                coefficient_penalty=coefficient_penalty,
                worst_weight=worst_weight,
                temperature=selection_temperature,
                score_ema=selection_score_ema,
                decay=selection_decay,
            )
        elif selector == "ray":
            coefficient, selection = covariance_ray_selection(
                model, optimizer, gradient_matrix, common, directions,
                atlas, held_out, radius=radius,
                coefficient_penalty=coefficient_penalty,
                worst_weight=worst_weight,
            )
        elif selector == "quadratic":
            coefficient, selection = quadratic_zonotope_selection(
                model, optimizer, common, directions, atlas, held_out,
                radius=radius, probe=quadratic_probe,
                coefficient_penalty=coefficient_penalty,
                worst_weight=worst_weight,
            )
        elif selector == "gradient":
            beta = torch.zeros(len(directions), requires_grad=True)
            if outer_optimizer_name == "adam":
                outer_optimizer = torch.optim.Adam([beta], lr=outer_lr)
            elif outer_optimizer_name == "sgd":
                outer_optimizer = torch.optim.SGD([beta], lr=outer_lr)
            else:
                raise ValueError(outer_optimizer_name)
            initial_sensitivity = math.nan
            for outer_index in range(outer_steps):
                outer_optimizer.zero_grad(set_to_none=True)
                coefficient = radius * torch.tanh(beta)
                candidate_gradient = common + torch.einsum(
                    "r,rp->p", coefficient, directions
                )
                candidate_parameters = preview_adamw(model, optimizer, candidate_gradient)
                objective, _, _ = smooth_witness_objective(
                    model, candidate_parameters, atlas, held_out,
                    worst_weight=worst_weight,
                )
                objective = objective + coefficient_penalty * coefficient.square().mean()
                objective.backward()
                if outer_index == 0:
                    initial_sensitivity = float(beta.grad.norm())
                outer_optimizer.step()
            with torch.no_grad():
                coefficient = radius * torch.tanh(beta)
            selection = {
                "center_score": math.nan,
                "selected_score": math.nan,
                "selected_scale": 1.0,
                "positive_curvature_fraction": math.nan,
                "initial_sensitivity": initial_sensitivity,
            }
        elif selector != "transport":
            raise ValueError(selector)
        if not transported:
            with torch.no_grad():
                selected_gradient = common + torch.einsum(
                    "r,rp->p", coefficient, directions
                )
            commit_adamw(model, optimizer, selected_gradient)
            selected_update = selected_gradient
        coefficient_norms.append(float(coefficient.norm()))
        retained_variances.append(float(retained))

        accepted = step + 1
        if step == 0 or accepted % evaluate_every == 0 or accepted == steps:
            metrics = curve_metrics(model, task)
            common_norm = float(common.norm())
            shell_norm = float((selected_update - common).norm())
            history.append({
                "step": accepted,
                "gradient_evaluations": accepted * gradient_samples,
                "held_out_fold": held_out,
                "r2": metrics["r2"],
                "sparse_tail_r2": metrics["sparse_tail_r2"],
                "minimum_segment_r2": metrics["minimum_segment_r2"],
                "extrapolation_r2": metrics["extrapolation_r2"],
                "retained_covariance": float(retained),
                "coefficient_norm": float(coefficient.norm()),
                "shell_to_common": shell_norm / max(common_norm, 1e-12),
                **selection,
            })

    return {
        "model": model,
        "history": history,
        "seconds": time.perf_counter() - started,
        "mean_coefficient_norm": sum(coefficient_norms) / len(coefficient_norms),
        "mean_retained_covariance": sum(retained_variances) / len(retained_variances),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/tmp/sparse_sine_gradient_zonotope"))
    parser.add_argument("--model", default="self_context")
    parser.add_argument("--width", type=int, default=38)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--gradient-samples", type=int, default=3)
    parser.add_argument("--covariance-rank", type=int, default=2)
    parser.add_argument("--radius", type=float, default=4.0)
    parser.add_argument(
        "--selector",
        choices=("transport", "softmax", "ray", "quadratic", "gradient"),
        default="transport",
    )
    parser.add_argument("--quadratic-probe", type=float, default=0.75)
    parser.add_argument("--selection-temperature", type=float, default=1e-6)
    parser.add_argument("--selection-decay", type=float, default=0.0)
    parser.add_argument("--outer-steps", type=int, default=4)
    parser.add_argument("--outer-lr", type=float, default=0.35)
    parser.add_argument("--outer-optimizer", choices=("sgd", "adam"), default="sgd")
    parser.add_argument("--coefficient-penalty", type=float, default=0.0)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--cells", type=int, default=16)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--witness-period", type=int, default=50)
    parser.add_argument("--worst-weight", type=float, default=0.10)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--skip-baselines", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    runs = []
    gradient_budget = args.steps * args.gradient_samples
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        task = sparse_sine_1d(seed)
        if not args.skip_baselines:
            for label, baseline_steps in (
                ("baseline_path_matched", args.steps),
                ("baseline_compute_matched", gradient_budget),
            ):
                model, history, seconds = train(
                    args.model, "stratified_whiten", task,
                    width=args.width, seed=seed, steps=baseline_steps,
                    batch=args.batch, lr=args.lr,
                    evaluate_every=max(1, baseline_steps // 50),
                )
                runs.append({
                    "method": label,
                    "seed": seed,
                    "parameters": parameter_count(model),
                    "accepted_steps": baseline_steps,
                    "gradient_evaluations": baseline_steps,
                    "seconds": seconds,
                    **curve_metrics(model, task),
                    "history": history,
                    "probe": curve_probe(model, task),
                })

        print(f"START gradient_zonotope seed={seed}", flush=True)
        result = train_gradient_zonotope(
            task, model_name=args.model, width=args.width, seed=seed,
            steps=args.steps, gradient_samples=args.gradient_samples,
            covariance_rank=args.covariance_rank, radius=args.radius,
            selector=args.selector, quadratic_probe=args.quadratic_probe,
            selection_temperature=args.selection_temperature,
            selection_decay=args.selection_decay,
            outer_steps=args.outer_steps, outer_lr=args.outer_lr,
            outer_optimizer_name=args.outer_optimizer,
            coefficient_penalty=args.coefficient_penalty,
            batch=args.batch, lr=args.lr, cells=args.cells, folds=args.folds,
            witness_period=args.witness_period, worst_weight=args.worst_weight,
            evaluate_every=args.eval_every,
        )
        model = result.pop("model")
        row = {
            "method": "gradient_zonotope",
            "seed": seed,
            "parameters": parameter_count(model),
            "accepted_steps": args.steps,
            "gradient_evaluations": gradient_budget,
            **result,
            **curve_metrics(model, task),
            "probe": curve_probe(model, task),
        }
        runs.append(row)
        print(json.dumps({
            "seed": seed,
            "r2": round(row["r2"], 6),
            "sparse_tail_r2": round(row["sparse_tail_r2"], 6),
            "minimum_segment_r2": round(row["minimum_segment_r2"], 6),
            "extrapolation_r2": round(row["extrapolation_r2"], 6),
            "seconds": round(row["seconds"], 3),
            "retained_covariance": round(row["mean_retained_covariance"], 5),
            "coefficient_norm": round(row["mean_coefficient_norm"], 5),
        }), flush=True)

    payload = {
        "configuration": {**vars(args), "out": str(args.out)},
        "interpretation": {
            "candidate_gradient_uses_witness_labels": False,
            "continuous_coefficient_selection_uses_witness_labels": True,
            "inference_is_ensemble": False,
            "gradient_budget": gradient_budget,
        },
        "runs": runs,
    }
    (args.out / "results.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({"complete": True, "runs": len(runs)}))


if __name__ == "__main__":
    main()
