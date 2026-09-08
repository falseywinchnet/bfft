"""A matrix-regression problem on which Muon's primacy is algebraic.

The population objective is

    F(W) = 1/2 tr((W-Q) Sigma (W-Q)^T),

where Q is orthogonal and Sigma is positive definite and anisotropic.  At
W=0, grad F=-Q Sigma, whose polar factor is -Q.  Exact spectral descent with
unit operator step therefore reaches the optimum in one update independently
of the condition number of Sigma.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class OperatorProblem:
    target: torch.Tensor
    covariance: torch.Tensor
    covariance_root: torch.Tensor

    def loss(self, weight: torch.Tensor) -> torch.Tensor:
        error = weight - self.target
        return 0.5 * torch.sum((error @ self.covariance) * error)

    def gradient(self, weight: torch.Tensor) -> torch.Tensor:
        return (weight - self.target) @ self.covariance

    def relative_operator_error(self, weight: torch.Tensor) -> float:
        return float(
            torch.linalg.matrix_norm(weight - self.target, ord=2)
            / torch.linalg.matrix_norm(self.target, ord=2)
        )


def _orthogonal(dimension: int, generator: torch.Generator) -> torch.Tensor:
    source = torch.randn(
        dimension, dimension, generator=generator, dtype=torch.float64
    )
    result, factor = torch.linalg.qr(source)
    signs = torch.sign(torch.diagonal(factor)).clamp_min(0.0) * 2.0 - 1.0
    return result * signs.unsqueeze(0)


def make_operator_problem(
    dimension: int = 32,
    condition: float = 1e4,
    seed: int = 731,
) -> OperatorProblem:
    if dimension < 2:
        raise ValueError("dimension must be at least two")
    if condition < 1.0:
        raise ValueError("condition must be at least one")
    generator = torch.Generator().manual_seed(seed)
    target = _orthogonal(dimension, generator)
    eigenvectors = _orthogonal(dimension, generator)
    eigenvalues = torch.logspace(
        0.0, -math.log10(condition), dimension, dtype=torch.float64
    )
    covariance = (
        eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T
    )
    covariance_root = (
        eigenvectors @ torch.diag(torch.sqrt(eigenvalues)) @ eigenvectors.T
    )
    return OperatorProblem(target, covariance, covariance_root)


def support_polar(matrix: torch.Tensor, tolerance: float = 1e-12) -> torch.Tensor:
    """Return the polar factor on the numerically supported singular space."""
    # A normalized matrix step is discontinuous at zero.  Treat an absolutely
    # negligible residual as zero so floating-point roundoff cannot launch a
    # fresh unit step after the analytic one-step solution has been reached.
    if float(torch.linalg.vector_norm(matrix)) <= tolerance:
        return torch.zeros_like(matrix)
    left, singular, right = torch.linalg.svd(matrix, full_matrices=False)
    if singular.numel() == 0 or float(singular[0]) == 0.0:
        return torch.zeros_like(matrix)
    keep = singular > tolerance * singular[0]
    if not bool(torch.any(keep)):
        return torch.zeros_like(matrix)
    return left[:, keep] @ right[keep]


def weighted_inner(
    left: torch.Tensor, right: torch.Tensor, covariance: torch.Tensor
) -> torch.Tensor:
    return torch.sum((left @ covariance) * right)


def best_scalar_step(
    direction: torch.Tensor,
    target: torch.Tensor,
    covariance: torch.Tensor,
) -> tuple[float, float]:
    """Best one-step scalar and residual population-loss ratio from W=0."""
    numerator = weighted_inner(direction, target, covariance)
    denominator = weighted_inner(direction, direction, covariance)
    step = numerator / denominator.clamp_min(1e-300)
    initial = 0.5 * weighted_inner(target, target, covariance)
    error = step * direction - target
    residual = 0.5 * weighted_inner(error, error, covariance)
    return float(step), float(residual / initial)


def one_step_certificate(problem: OperatorProblem) -> dict:
    target, covariance = problem.target, problem.covariance
    gradient = problem.gradient(torch.zeros_like(target))
    directions = {
        "exact_muon": -support_polar(gradient),
        "sgd": -gradient,
        "adam_sign_limit": -torch.sign(gradient),
    }
    result = {}
    for name, direction in directions.items():
        step, ratio = best_scalar_step(direction, target, covariance)
        result[name] = {
            "best_scalar_step": step,
            "relative_loss_after_one_step": ratio,
            "direction_operator_norm": float(
                torch.linalg.matrix_norm(direction, ord=2)
            ),
        }
    muon_weight = directions["exact_muon"]
    result["exact_muon"]["frobenius_target_error"] = float(
        torch.linalg.vector_norm(muon_weight - target)
    )
    return result


def _first_below(history: list[dict], threshold: float) -> int | None:
    return next(
        (point["step"] for point in history if point["relative_loss"] <= threshold),
        None,
    )


def _first_operator_below(history: list[dict], threshold: float) -> int | None:
    return next(
        (point["step"] for point in history if point["operator_error"] <= threshold),
        None,
    )


def run_population(
    problem: OperatorProblem,
    method: str,
    lr: float,
    steps: int,
) -> dict:
    weight = torch.zeros_like(problem.target)
    first = torch.zeros_like(weight)
    second = torch.zeros_like(weight)
    initial = float(problem.loss(weight))
    history = []
    for step in range(1, steps + 1):
        gradient = problem.gradient(weight)
        if method == "exact_muon":
            update = support_polar(gradient)
        elif method == "sgd":
            update = gradient
        elif method == "adam":
            beta1, beta2 = 0.9, 0.999
            first.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
            second.mul_(beta2).addcmul_(
                gradient, gradient, value=1.0 - beta2
            )
            corrected_first = first / (1.0 - beta1 ** step)
            corrected_second = second / (1.0 - beta2 ** step)
            update = corrected_first / (torch.sqrt(corrected_second) + 1e-8)
        else:
            raise KeyError(method)
        weight.add_(update, alpha=-lr)
        value = float(problem.loss(weight))
        history.append({
            "step": step,
            "relative_loss": value / initial,
            "operator_error": problem.relative_operator_error(weight),
        })
    return {
        "method": method,
        "lr": lr,
        "steps": steps,
        "final_relative_loss": history[-1]["relative_loss"],
        "final_operator_error": history[-1]["operator_error"],
        "steps_to_1e-2": _first_below(history, 1e-2),
        "steps_to_1e-4": _first_below(history, 1e-4),
        "steps_to_1e-8": _first_below(history, 1e-8),
        "steps_to_operator_1e-2": _first_operator_below(history, 1e-2),
        "history": history,
    }


def make_minibatch_covariances(
    problem: OperatorProblem,
    batch_size: int,
    steps: int,
    seed: int,
) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    dimension = problem.target.shape[0]
    result = []
    for _ in range(steps):
        normal = torch.randn(
            dimension, batch_size, generator=generator, dtype=torch.float64
        )
        values = problem.covariance_root @ normal
        result.append(values @ values.T / batch_size)
    return result


def make_block_sweep_covariances(
    problem: OperatorProblem,
    batch_size: int,
    cycles: int = 1,
) -> list[torch.Tensor]:
    """Partition the covariance into orthogonal, anisotropic batch blocks.

    The eigenvectors are interleaved so every block spans both loud and quiet
    directions.  Consequently each batch is rank deficient but internally
    ill-conditioned.  The blocks sum to the population covariance and their
    support projectors sum to the identity.
    """
    dimension = problem.target.shape[0]
    if batch_size <= 0 or dimension % batch_size:
        raise ValueError("batch_size must be a positive divisor of dimension")
    if cycles <= 0:
        raise ValueError("cycles must be positive")
    eigenvalues, eigenvectors = torch.linalg.eigh(problem.covariance)
    block_count = dimension // batch_size
    blocks = []
    for block in range(block_count):
        indices = torch.arange(block, dimension, block_count)
        basis = eigenvectors[:, indices]
        values = eigenvalues[indices]
        blocks.append(basis @ torch.diag(values) @ basis.T)
    return blocks * cycles


def run_streaming(
    problem: OperatorProblem,
    covariances: list[torch.Tensor],
    method: str,
    lr: float,
) -> dict:
    weight = torch.zeros_like(problem.target)
    first = torch.zeros_like(weight)
    second = torch.zeros_like(weight)
    initial = float(problem.loss(weight))
    history = []
    for step, covariance in enumerate(covariances, 1):
        gradient = (weight - problem.target) @ covariance
        if method == "exact_muon":
            update = support_polar(gradient)
        elif method == "sgd":
            update = gradient
        elif method == "adam":
            beta1, beta2 = 0.9, 0.999
            first.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
            second.mul_(beta2).addcmul_(
                gradient, gradient, value=1.0 - beta2
            )
            update = (
                first / (1.0 - beta1 ** step)
                / (torch.sqrt(second / (1.0 - beta2 ** step)) + 1e-8)
            )
        else:
            raise KeyError(method)
        weight.add_(update, alpha=-lr)
        value = float(problem.loss(weight))
        history.append({
            "step": step,
            "relative_loss": value / initial,
            "operator_error": problem.relative_operator_error(weight),
        })
    return {
        "method": method,
        "lr": lr,
        "steps": len(covariances),
        "final_relative_loss": history[-1]["relative_loss"],
        "final_operator_error": history[-1]["operator_error"],
        "steps_to_1e-2": _first_below(history, 1e-2),
        "steps_to_1e-4": _first_below(history, 1e-4),
        "steps_to_1e-8": _first_below(history, 1e-8),
        "steps_to_operator_1e-2": _first_operator_below(history, 1e-2),
        "history": history,
    }


def _select_best(runs: list[dict], threshold: str) -> dict:
    def key(row):
        reached = row[threshold]
        return (
            reached is None,
            reached if reached is not None else math.inf,
            row["final_relative_loss"],
        )
    return min(runs, key=key)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/tmp/muon_primacy.json"))
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--condition", type=float, default=1e4)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--stream-steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--block-cycles", type=int, default=100)
    parser.add_argument("--seed", type=int, default=731)
    args = parser.parse_args()
    problem = make_operator_problem(args.dimension, args.condition, args.seed)

    population_grids = {
        "exact_muon": (0.3, 0.5, 1.0),
        "sgd": (0.1, 0.3, 0.7, 1.0, 1.5, 1.9),
        "adam": (0.003, 0.01, 0.03, 0.1, 0.3),
    }
    population_runs = [
        run_population(problem, method, lr, args.steps)
        for method, rates in population_grids.items()
        for lr in rates
    ]
    population_best = {
        method: _select_best(
            [row for row in population_runs if row["method"] == method],
            "steps_to_1e-8",
        )
        for method in population_grids
    }

    covariances = make_minibatch_covariances(
        problem, args.batch_size, args.stream_steps, args.seed + 1
    )
    streaming_grids = {
        "exact_muon": (0.01, 0.03, 0.1, 0.3),
        "sgd": (0.03, 0.1, 0.3, 1.0),
        "adam": (0.001, 0.003, 0.01, 0.03),
    }
    streaming_runs = [
        run_streaming(problem, covariances, method, lr)
        for method, rates in streaming_grids.items()
        for lr in rates
    ]
    streaming_best = {
        method: _select_best(
            [row for row in streaming_runs if row["method"] == method],
            "steps_to_1e-4",
        )
        for method in streaming_grids
    }

    block_covariances = make_block_sweep_covariances(
        problem, args.batch_size, args.block_cycles
    )
    block_grids = {
        "exact_muon": (0.3, 0.5, 1.0),
        "sgd": (0.1, 0.3, 0.7, 1.0, 1.5, 1.9),
        "adam": (0.001, 0.003, 0.01, 0.03, 0.1),
    }
    block_runs = [
        run_streaming(problem, block_covariances, method, lr)
        for method, rates in block_grids.items()
        for lr in rates
    ]
    block_best = {
        method: _select_best(
            [row for row in block_runs if row["method"] == method],
            "steps_to_1e-8",
        )
        for method in block_grids
    }

    payload = {
        "configuration": {**vars(args), "out": str(args.out)},
        "one_step_certificate": one_step_certificate(problem),
        "population_best": population_best,
        "streaming_best": streaming_best,
        "block_sweep_best": block_best,
        "population_runs": population_runs,
        "streaming_runs": streaming_runs,
        "block_sweep_runs": block_runs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    compact = {
        "one_step_certificate": payload["one_step_certificate"],
        "population_best": {
            name: {key: value for key, value in row.items() if key != "history"}
            for name, row in population_best.items()
        },
        "streaming_best": {
            name: {key: value for key, value in row.items() if key != "history"}
            for name, row in streaming_best.items()
        },
        "block_sweep_best": {
            name: {key: value for key, value in row.items() if key != "history"}
            for name, row in block_best.items()
        },
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
