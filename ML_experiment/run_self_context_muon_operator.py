#!/usr/bin/env python3
"""Embed the Muon operator witness inside the complete self-context network."""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from ML_experiment.muon_primacy import make_operator_problem
from ML_experiment.optimizers import make_optimizer, zeropower_newton_schulz5
from ML_experiment.variants import make_variant


OPTIMIZERS = (
    "adamw",
    "muon",
    "matrix_transport_nesterov",
    "matrix_transport_longitudinal_fusion",
    "matrix_transport_residual_polar",
)


def make_samples(problem, count: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    normal = torch.randn(count, problem.target.shape[0], generator=generator)
    x = normal @ problem.covariance_root.float()
    y = x @ problem.target.float().T
    return x, y


@torch.no_grad()
def relative_mse(model, x, y):
    model.eval()
    prediction = torch.cat([model(part) for part in x.split(512)])
    return float(F.mse_loss(prediction, y) / y.square().mean().clamp_min(1e-12))


def zero_jacobian_error(model, target):
    model.eval()
    origin = torch.zeros(target.shape[1], requires_grad=True)
    jacobian = torch.autograd.functional.jacobian(
        lambda value: model(value[None])[0], origin,
    ).detach()
    return float(
        torch.linalg.vector_norm(jacobian - target)
        / torch.linalg.vector_norm(target)
    )


def first_below(history, threshold):
    return next((row["step"] for row in history
                 if row["relative_mse"] <= threshold), None)


def _weighted_mean(values):
    total = sum(weight for _, weight in values)
    return (sum(value * weight for value, weight in values) / total
            if total else 0.0)


def _spectral_rank_fractions(matrix):
    """Return three scale-free ranks normalized by the smaller dimension.

    ``stable`` is the conventional Frobenius/spectral stable rank,
    ``participation`` is the inverse participation ratio, and ``entropy`` is
    the exponential spectral-entropy rank.  All are one for equal singular
    values and approach ``1 / min(matrix.shape)`` for a rank-one matrix.
    """
    singular = torch.linalg.svdvals(matrix.float())
    energy = singular.square()
    total = energy.sum().clamp_min(1e-20)
    probability = energy / total
    dimension = min(matrix.shape)
    stable = total / energy.max().clamp_min(1e-20)
    participation = total.square() / energy.square().sum().clamp_min(1e-20)
    entropy = torch.exp(-torch.sum(
        probability * torch.log(probability.clamp_min(1e-20))
    ))
    return tuple(float(value / dimension)
                 for value in (stable, participation, entropy))


@torch.no_grad()
def output_residual_diagnostics(model, x, y):
    """Measure unresolved output and operator geometry on training examples."""
    model.eval()
    prediction = torch.cat([model(part) for part in x.split(512)])
    residual = prediction - y
    stable, participation, entropy = _spectral_rank_fractions(residual)
    # Regress the residual back onto the input.  This removes sample covariance
    # and estimates the unresolved linear operator (the average residual
    # Jacobian for this task) using training examples only.
    gram = x.float().T @ x.float()
    ridge = 1e-7 * torch.trace(gram) / gram.shape[0]
    identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    residual_operator = torch.linalg.solve(
        gram + ridge * identity,
        x.float().T @ residual.float(),
    )
    op_stable, op_participation, op_entropy = _spectral_rank_fractions(
        residual_operator
    )
    return {
        "train_residual_stable_rank_fraction": stable,
        "train_residual_participation_rank_fraction": participation,
        "train_residual_entropy_rank_fraction": entropy,
        "train_operator_residual_stable_rank_fraction": op_stable,
        "train_operator_residual_participation_rank_fraction": op_participation,
        "train_operator_residual_entropy_rank_fraction": op_entropy,
    }


@torch.no_grad()
def operator_residual_entropy_rank(prediction, x, y):
    """Training-batch residual-operator rank used by the polar governor."""
    residual = prediction.detach().float() - y.float()
    x = x.float()
    gram = x.T @ x
    ridge = 1e-7 * torch.trace(gram) / gram.shape[0]
    identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    residual_operator = torch.linalg.solve(
        gram + ridge * identity, x.T @ residual
    )
    return _spectral_rank_fractions(residual_operator)[2]


@torch.no_grad()
def advance_shadow_muon(optimizer, probe_state, momentum=.95):
    """Advance a raw-gradient Muon buffer along MatrixTransport's trajectory."""
    if hasattr(optimizer, "muon"):
        return
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            gradient = parameter.grad
            if gradient is None or parameter.ndim < 2:
                continue
            key = ("shadow_muon_buffer", id(parameter))
            buffer = probe_state.get(key)
            if buffer is None:
                buffer = torch.zeros_like(gradient)
                probe_state[key] = buffer
            buffer.lerp_(gradient, 1.0 - momentum)


@torch.no_grad()
def optimizer_state_diagnostics(optimizer, probe_state):
    """Read intrinsic optimizer state without inspecting loss or targets."""
    if hasattr(optimizer, "muon"):
        alignments, request_alignments = [], []
        stable_ranks, conditions, polar_defects, update_turns = [], [], [], []
        for group in optimizer.muon.param_groups:
            momentum = group["momentum"]
            for parameter in group["params"]:
                gradient = parameter.grad
                state = optimizer.muon.state[parameter]
                if gradient is None or "momentum_buffer" not in state:
                    continue
                weight = parameter.numel()
                buffer = state["momentum_buffer"]
                gnorm = torch.linalg.vector_norm(gradient).clamp_min(1e-12)
                bnorm = torch.linalg.vector_norm(buffer).clamp_min(1e-12)
                alignments.append((float(torch.sum(gradient * buffer) / (gnorm * bnorm)), weight))
                request = gradient.lerp(buffer, momentum) if group["nesterov"] else buffer
                rnorm = torch.linalg.vector_norm(request).clamp_min(1e-12)
                request_alignments.append((float(torch.sum(gradient * request) / (gnorm * rnorm)), weight))
                singular = torch.linalg.svdvals(request.float())
                energy = singular.square()
                stable_rank = energy.sum().square() / energy.square().sum().clamp_min(1e-20)
                stable_ranks.append((float(stable_rank / min(parameter.shape)), weight))
                floor = singular.max().clamp_min(1e-12) * 1e-7
                conditions.append((float(torch.log10(
                    singular.max().clamp_min(floor) / singular.min().clamp_min(floor)
                )), weight))
                polar = zeropower_newton_schulz5(request, group["ns_steps"]).float()
                size = min(parameter.shape)
                identity = torch.eye(size, device=polar.device, dtype=polar.dtype)
                gram = polar @ polar.T if polar.shape[0] <= polar.shape[1] else polar.T @ polar
                polar_defects.append((float(
                    torch.linalg.vector_norm(gram - identity) / math.sqrt(size)
                ), weight))
                key = ("muon_polar", id(parameter))
                previous = probe_state.get(key)
                if previous is not None:
                    cosine = torch.sum(previous * polar) / (
                        torch.linalg.vector_norm(previous).clamp_min(1e-12)
                        * torch.linalg.vector_norm(polar).clamp_min(1e-12)
                    )
                    update_turns.append((float(.5 * (1.0 - cosine.clamp(-1, 1))), weight))
                probe_state[key] = polar.clone()
        return {
            "state_buffer_gradient_cosine": _weighted_mean(alignments),
            "state_request_gradient_cosine": _weighted_mean(request_alignments),
            "state_request_stable_rank_fraction": _weighted_mean(stable_ranks),
            "state_request_log10_condition": _weighted_mean(conditions),
            "state_polar_defect": _weighted_mean(polar_defects),
            "state_polar_interval_turn": _weighted_mean(update_turns),
        }

    frame_confidences, momentum_cosines, axial_fractions = [], [], []
    metric_conditions, root_drifts, root_ages = [], [], []
    request_stable_ranks, request_conditions = [], []
    request_polar_defects, request_polar_cosines = [], []
    shadow_ranks, shadow_alignments, shadow_conditions = [], [], []
    gradient_stable_ranks, gradient_participation_ranks = [], []
    gradient_entropy_ranks = []
    residual_stable_ranks, residual_participation_ranks = [], []
    residual_entropy_ranks, residual_energy_fractions = [], []
    for group in optimizer.param_groups:
        beta1 = group["betas"][0]
        beta2 = group["betas"][1]
        for parameter in group["params"]:
            state = optimizer.state[parameter]
            if not state or "momentum" not in state:
                continue
            weight = parameter.numel()
            frame_confidences.append((float(state.get("frame_confidence", 1.0)), weight))
            signature = state.get("signature")
            momentum = state["momentum"].reshape(1, -1)
            if signature is not None and float(torch.linalg.vector_norm(momentum)) > 1e-12:
                cosine = torch.sum(momentum * signature) / torch.linalg.vector_norm(momentum)
                momentum_cosines.append((float(cosine.clamp(-1, 1)), weight))
                axial_fractions.append((float(cosine.square().clamp(0, 1)), weight))
            if parameter.ndim >= 2 and "root_condition" in state:
                metric_conditions.append((math.log10(max(state["root_condition"], 1.0)), weight))
                bias2 = 1.0 - beta2 ** state["step"]
                left = state["left"] / bias2
                right = state["right"] / bias2
                left_ref = state["root_reference_left"]
                right_ref = state["root_reference_right"]
                drift = max(
                    float(torch.linalg.vector_norm(left - left_ref)
                          / torch.linalg.vector_norm(left_ref).clamp_min(1e-12)),
                    float(torch.linalg.vector_norm(right - right_ref)
                          / torch.linalg.vector_norm(right_ref).clamp_min(1e-12)),
                )
                root_drifts.append((drift, weight))
                root_ages.append((float(state["step"] - state["last_root_step"]), weight))
                gradient = parameter.grad
                if gradient is not None:
                    rows = parameter.shape[0]
                    live = (
                        state["left_root"] @ gradient.reshape(rows, -1)
                        @ state["right_root"]
                    ).reshape_as(parameter)
                    averaged = state["momentum"] / (1.0 - beta1 ** state["step"])
                    flat_live = live.reshape(1, -1)
                    signature = state["signature"]
                    averaged_flat = averaged.reshape(1, -1)
                    nesterov = beta1 * averaged_flat + (1.0 - beta1) * flat_live
                    axial = torch.sum(averaged_flat * signature, dim=1)
                    live_norm = torch.linalg.vector_norm(flat_live, dim=1)
                    correction = (
                        state["frame_confidence"] * beta1
                        * (live_norm - axial)[:, None] * signature
                    )
                    request = (nesterov + correction).reshape(rows, -1)
                    inner = torch.sum(request.double() * gradient.reshape(rows, -1).double())
                    gradient_norm2 = gradient.double().square().sum()
                    if float(inner) < 0 and float(gradient_norm2) > group["eps"]:
                        request = request - float(inner / gradient_norm2) * gradient.reshape(rows, -1)
                    gradient_matrix = gradient.reshape(rows, -1)
                    gradient_ranks = _spectral_rank_fractions(gradient_matrix)
                    gradient_stable_ranks.append((gradient_ranks[0], weight))
                    gradient_participation_ranks.append((gradient_ranks[1], weight))
                    gradient_entropy_ranks.append((gradient_ranks[2], weight))
                    request_norm2 = request.double().square().sum().clamp_min(1e-20)
                    projection_scale = float(
                        torch.sum(gradient_matrix.double() * request.double())
                        / request_norm2
                    )
                    unexplained = gradient_matrix - projection_scale * request
                    residual_ranks = _spectral_rank_fractions(unexplained)
                    residual_stable_ranks.append((residual_ranks[0], weight))
                    residual_participation_ranks.append((residual_ranks[1], weight))
                    residual_entropy_ranks.append((residual_ranks[2], weight))
                    residual_energy_fractions.append((float(
                        unexplained.double().square().sum()
                        / gradient_norm2.clamp_min(1e-20)
                    ), weight))
                    singular = torch.linalg.svdvals(request.float())
                    energy = singular.square()
                    stable_rank = energy.sum().square() / energy.square().sum().clamp_min(1e-20)
                    request_stable_ranks.append((float(
                        stable_rank / min(request.shape)
                    ), weight))
                    floor = singular.max().clamp_min(1e-12) * 1e-7
                    request_conditions.append((float(torch.log10(
                        singular.max().clamp_min(floor)
                        / singular.min().clamp_min(floor)
                    )), weight))
                    polar = zeropower_newton_schulz5(request, 5).float()
                    size = min(request.shape)
                    identity = torch.eye(size, device=polar.device, dtype=polar.dtype)
                    gram = polar @ polar.T if polar.shape[0] <= polar.shape[1] else polar.T @ polar
                    request_polar_defects.append((float(
                        torch.linalg.vector_norm(gram - identity) / math.sqrt(size)
                    ), weight))
                    request_polar_cosines.append((float(
                        torch.sum(request.float() * polar)
                        / (torch.linalg.vector_norm(request.float()).clamp_min(1e-12)
                           * torch.linalg.vector_norm(polar).clamp_min(1e-12))
                    ), weight))
                    shadow_buffer = probe_state.get(("shadow_muon_buffer", id(parameter)))
                    if shadow_buffer is not None:
                        shadow_request = gradient.lerp(shadow_buffer, .95).reshape(rows, -1)
                        shadow_singular = torch.linalg.svdvals(shadow_request.float())
                        shadow_energy = shadow_singular.square()
                        shadow_rank = (
                            shadow_energy.sum().square()
                            / shadow_energy.square().sum().clamp_min(1e-20)
                        )
                        shadow_ranks.append((float(
                            shadow_rank / min(shadow_request.shape)
                        ), weight))
                        shadow_alignments.append((float(
                            torch.sum(shadow_request.double() * gradient.reshape(rows, -1).double())
                            / (torch.linalg.vector_norm(shadow_request.double()).clamp_min(1e-12)
                               * torch.linalg.vector_norm(gradient.double()).clamp_min(1e-12))
                        ), weight))
                        shadow_floor = shadow_singular.max().clamp_min(1e-12) * 1e-7
                        shadow_conditions.append((float(torch.log10(
                            shadow_singular.max().clamp_min(shadow_floor)
                            / shadow_singular.min().clamp_min(shadow_floor)
                        )), weight))
    return {
        "state_frame_confidence": _weighted_mean(frame_confidences),
        "state_momentum_live_cosine": _weighted_mean(momentum_cosines),
        "state_momentum_axial_fraction": _weighted_mean(axial_fractions),
        "state_metric_log10_condition": _weighted_mean(metric_conditions),
        "state_root_drift": _weighted_mean(root_drifts),
        "state_root_age": _weighted_mean(root_ages),
        "state_fusion_request_stable_rank_fraction": _weighted_mean(request_stable_ranks),
        "state_fusion_request_log10_condition": _weighted_mean(request_conditions),
        "state_fusion_request_polar_defect": _weighted_mean(request_polar_defects),
        "state_fusion_request_polar_cosine": _weighted_mean(request_polar_cosines),
        "state_live_gradient_stable_rank_fraction": _weighted_mean(gradient_stable_ranks),
        "state_live_gradient_participation_rank_fraction": _weighted_mean(gradient_participation_ranks),
        "state_live_gradient_entropy_rank_fraction": _weighted_mean(gradient_entropy_ranks),
        "state_unexplained_gradient_stable_rank_fraction": _weighted_mean(residual_stable_ranks),
        "state_unexplained_gradient_participation_rank_fraction": _weighted_mean(residual_participation_ranks),
        "state_unexplained_gradient_entropy_rank_fraction": _weighted_mean(residual_entropy_ranks),
        "state_unexplained_gradient_energy_fraction": _weighted_mean(residual_energy_fractions),
        "state_shadow_muon_stable_rank_fraction": _weighted_mean(shadow_ranks),
        "state_shadow_muon_request_gradient_cosine": _weighted_mean(shadow_alignments),
        "state_shadow_muon_log10_condition": _weighted_mean(shadow_conditions),
    }


def train_one(
    problem, train_x, train_y, test_x, test_y, optimizer_name, scenario,
    seed, width, steps, batch_size, lr, evaluate_every,
):
    torch.manual_seed(10000 + seed)
    model = make_variant(
        "self_context", problem.target.shape[0], problem.target.shape[0], width
    )
    initial_state = copy.deepcopy(model.state_dict())
    optimizer = make_optimizer(model, optimizer_name, lr, weight_decay=1e-4)
    sample_generator = torch.Generator().manual_seed(190000 + seed)
    initial_relative_mse = relative_mse(model, test_x, test_y)
    history = []
    probe_state = {}
    failure = None
    started = time.perf_counter()

    for step in range(1, steps + 1):
        model.train()
        if scenario == "population":
            x, y = train_x, train_y
        else:
            index = torch.randint(
                len(train_x), (batch_size,), generator=sample_generator
            )
            x, y = train_x[index], train_y[index]
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x)
        loss = F.mse_loss(prediction, y)
        if not torch.isfinite(loss):
            failure = {"step": step, "stage": "loss"}
            break
        if hasattr(optimizer, "set_residual_entropy_rank"):
            optimizer.set_residual_entropy_rank(
                operator_residual_entropy_rank(prediction, x, y)
            )
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if not torch.isfinite(gradient_norm):
            failure = {"step": step, "stage": "gradient"}
            break
        advance_shadow_muon(optimizer, probe_state)
        optimizer.step()
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            failure = {"step": step, "stage": "parameters"}
            break
        if step == 1 or step % evaluate_every == 0 or step == steps:
            value = relative_mse(model, test_x, test_y)
            row = {
                "step": step,
                "relative_mse": value,
                "train_relative_mse": float(
                    loss.detach() / y.square().mean().clamp_min(1e-12)
                ),
            }
            for key, attribute in (
                ("optimizer_lr_scale", "last_lr_scale"),
                ("optimizer_turn", "last_turn"),
                ("optimizer_longitudinal_closure", "last_longitudinal_closure"),
                ("optimizer_longitudinal_certificate", "last_longitudinal_certificate"),
                ("optimizer_residual_entropy_rank", "last_residual_entropy_rank"),
                ("optimizer_polar_weight", "last_polar_weight"),
            ):
                if hasattr(optimizer, attribute):
                    row[key] = float(getattr(optimizer, attribute))
            row.update(optimizer_state_diagnostics(optimizer, probe_state))
            row.update(output_residual_diagnostics(model, train_x, train_y))
            history.append(row)
            if not math.isfinite(value):
                failure = {"step": step, "stage": "evaluation"}
                break

    if failure and not history:
        model.load_state_dict(initial_state)
    final_relative_mse = relative_mse(model, test_x, test_y)
    jacobian_error = zero_jacobian_error(model, problem.target.float())
    best = min(history, key=lambda row: row["relative_mse"], default=None)
    return {
        "optimizer": optimizer_name,
        "scenario": scenario,
        "seed": seed,
        "lr": lr,
        "status": "failed" if failure else "complete",
        "failure": failure,
        "seconds": time.perf_counter() - started,
        "initial_relative_mse": initial_relative_mse,
        "final_relative_mse": final_relative_mse,
        "best_relative_mse": best["relative_mse"] if best else float("nan"),
        "best_step": best["step"] if best else None,
        "zero_jacobian_error": jacobian_error,
        "steps_to_1e-1": first_below(history, 1e-1),
        "steps_to_3e-2": first_below(history, 3e-2),
        "steps_to_1e-2": first_below(history, 1e-2),
        "steps_to_3e-3": first_below(history, 3e-3),
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/tmp/self_context_muon.json"))
    parser.add_argument("--dimension", type=int, default=16)
    parser.add_argument("--condition", type=float, default=1e4)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--train-count", type=int, default=1024)
    parser.add_argument("--test-count", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed-base", type=int, default=731)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--optimizers", default=",".join(OPTIMIZERS))
    parser.add_argument("--scenarios", default="population,minibatch")
    args = parser.parse_args()

    optimizer_names = tuple(x.strip() for x in args.optimizers.split(",") if x.strip())
    unknown = set(optimizer_names) - set(OPTIMIZERS)
    if unknown:
        raise ValueError(f"unknown optimizers: {sorted(unknown)}")
    scenarios = tuple(x.strip() for x in args.scenarios.split(",") if x.strip())
    if set(scenarios) - {"population", "minibatch"}:
        raise ValueError("scenarios must be population and/or minibatch")

    runs = []
    for seed_index in range(args.seeds):
        problem_seed = args.seed_base + seed_index
        problem = make_operator_problem(
            args.dimension, args.condition, problem_seed
        )
        train_x, train_y = make_samples(
            problem, args.train_count, 30000 + problem_seed
        )
        test_x, test_y = make_samples(
            problem, args.test_count, 40000 + problem_seed
        )
        for scenario in scenarios:
            for optimizer_name in optimizer_names:
                row = train_one(
                    problem, train_x, train_y, test_x, test_y,
                    optimizer_name, scenario, seed_index, args.width,
                    args.steps, args.batch_size, args.lr, args.eval_every,
                )
                row["problem_seed"] = problem_seed
                runs.append(row)
                print(json.dumps({k: v for k, v in row.items() if k != "history"}), flush=True)

    payload = {
        "configuration": {**vars(args), "out": str(args.out)},
        "optimizers": list(optimizer_names),
        "scenarios": list(scenarios),
        "runs": runs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"complete": True, "runs": len(runs)}, indent=2))


if __name__ == "__main__":
    main()
