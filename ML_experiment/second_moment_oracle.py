"""Exact second-moment controls for the Standing Population witness.

This module is deliberately an assay, not a production optimizer.  It asks
which information Adam loses when it replaces the covariance operator

    E[g \\otimes g]

by its coordinate diagonal.  The exact full operator is practical only for
small witnesses, but it gives us an oracle against which cheaper structured
approximations can be judged.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from ML_experiment.muon_primacy import (
    OperatorProblem,
    best_scalar_step,
    make_operator_problem,
    support_polar,
)


MODES = ("diagonal", "scalar", "full", "right_matrix", "exact_muon")


def _inverse_square_root_action(
    matrix: torch.Tensor,
    vector: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Apply ``(matrix.sqrt() + epsilon I)^-1`` to ``vector``.

    Eigh is intentionally used here: this is the exact small-problem oracle.
    Production candidates must approximate the measured useful structure
    without storing or diagonalizing the full covariance operator.
    """
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    eigenvalues = eigenvalues.clamp_min(0.0)
    coordinates = eigenvectors.T @ vector
    largest = eigenvalues.max()
    numerical_null = torch.finfo(matrix.dtype).eps * matrix.shape[0] * largest
    supported = eigenvalues > torch.maximum(
        numerical_null, eigenvalues.new_tensor(epsilon ** 2)
    )
    inverse = torch.where(
        supported,
        1.0 / (eigenvalues.sqrt() + epsilon),
        torch.zeros_like(eigenvalues),
    )
    if coordinates.ndim > 1:
        inverse = inverse.unsqueeze(-1)
    coordinates = coordinates * inverse
    return eigenvectors @ coordinates


@dataclass
class MomentState:
    """Adam first moment plus one explicitly chosen second-moment geometry."""

    shape: torch.Size
    mode: str
    beta1: float = 0.9
    beta2: float = 0.999
    # Double-precision oracle regularization.  A production epsilon is a
    # separate stability choice; using 1e-8 here would materially damp the
    # witness's quietest singular directions and obscure the algebraic assay.
    epsilon: float = 1e-12
    dtype: torch.dtype = torch.float64

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"unknown second-moment mode: {self.mode}")
        self.step = 0
        self.first = torch.zeros(self.shape, dtype=self.dtype)
        parameter_count = self.first.numel()
        if self.mode == "diagonal":
            self.second = torch.zeros(self.shape, dtype=self.dtype)
        elif self.mode == "scalar":
            self.second = torch.zeros((), dtype=self.dtype)
        elif self.mode == "full":
            self.second = torch.zeros(
                parameter_count, parameter_count, dtype=self.dtype
            )
        elif self.mode == "right_matrix":
            if len(self.shape) != 2:
                raise ValueError("right-matrix moment requires a matrix parameter")
            self.second = torch.zeros(
                self.shape[1], self.shape[1], dtype=self.dtype
            )
        else:
            self.second = None

    def update(self, gradient: torch.Tensor) -> torch.Tensor:
        if gradient.shape != self.shape:
            raise ValueError("gradient shape does not match moment state")
        self.step += 1
        self.first.mul_(self.beta1).add_(gradient, alpha=1.0 - self.beta1)
        corrected_first = self.first / (1.0 - self.beta1 ** self.step)

        if self.mode == "exact_muon":
            return support_polar(corrected_first)

        if self.mode == "diagonal":
            self.second.mul_(self.beta2).addcmul_(
                gradient, gradient, value=1.0 - self.beta2
            )
            corrected = self.second / (1.0 - self.beta2 ** self.step)
            return corrected_first / (corrected.sqrt() + self.epsilon)

        if self.mode == "scalar":
            mean_square = gradient.square().mean()
            self.second.mul_(self.beta2).add_(
                mean_square, alpha=1.0 - self.beta2
            )
            corrected = self.second / (1.0 - self.beta2 ** self.step)
            return corrected_first / (corrected.sqrt() + self.epsilon)

        if self.mode == "full":
            flat_gradient = gradient.reshape(-1)
            self.second.mul_(self.beta2).add_(
                torch.outer(flat_gradient, flat_gradient),
                alpha=1.0 - self.beta2,
            )
            corrected = self.second / (1.0 - self.beta2 ** self.step)
            result = _inverse_square_root_action(
                corrected, corrected_first.reshape(-1), self.epsilon
            )
            return result.reshape(self.shape)

        # This is the literal matrix analogue of Adam's second moment on the
        # input side.  Its first deterministic step is the polar factor, so it
        # is included to mark the exact boundary at which "matrix Adam"
        # becomes Muon rather than as a proposed candidate.
        self.second.mul_(self.beta2).addmm_(
            gradient.T, gradient, beta=1.0, alpha=1.0 - self.beta2
        )
        corrected = self.second / (1.0 - self.beta2 ** self.step)
        return _inverse_square_root_action(
            corrected, corrected_first.T, self.epsilon
        ).T


def first_step_certificate(problem: OperatorProblem) -> dict:
    """Compare first-step directions after optimizing away scalar step size."""
    gradient = problem.gradient(torch.zeros_like(problem.target))
    result = {}
    for mode in MODES:
        state = MomentState(gradient.shape, mode, dtype=gradient.dtype)
        direction = -state.update(gradient)
        step, residual = best_scalar_step(
            direction, problem.target, problem.covariance
        )
        result[mode] = {
            "best_scalar_step": step,
            "relative_loss_after_one_step": residual,
            "direction_frobenius_norm": float(
                torch.linalg.vector_norm(direction)
            ),
            "direction_operator_norm": float(
                torch.linalg.matrix_norm(direction, ord=2)
            ),
        }
    result["right_matrix"]["distance_to_muon_direction"] = float(
        torch.linalg.vector_norm(
            -MomentState(
                gradient.shape, "right_matrix", dtype=gradient.dtype
            ).update(gradient)
            + support_polar(gradient)
        )
    )
    return result


def _first(history: list[dict], key: str, threshold: float) -> int | None:
    return next(
        (point["step"] for point in history if point[key] <= threshold), None
    )


def run_population(
    problem: OperatorProblem,
    mode: str,
    learning_rate: float,
    steps: int,
) -> dict:
    weight = torch.zeros_like(problem.target)
    state = MomentState(weight.shape, mode, dtype=weight.dtype)
    initial_loss = float(problem.loss(weight))
    history = []
    for step in range(1, steps + 1):
        gradient = problem.gradient(weight)
        weight.add_(state.update(gradient), alpha=-learning_rate)
        history.append({
            "step": step,
            "relative_loss": float(problem.loss(weight)) / initial_loss,
            "operator_error": problem.relative_operator_error(weight),
        })
    return {
        "mode": mode,
        "learning_rate": learning_rate,
        "final_relative_loss": history[-1]["relative_loss"],
        "final_operator_error": history[-1]["operator_error"],
        "steps_to_loss_1e-2": _first(history, "relative_loss", 1e-2),
        "steps_to_loss_1e-4": _first(history, "relative_loss", 1e-4),
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/tmp/second_moment_oracle.json"))
    parser.add_argument("--dimension", type=int, default=8)
    parser.add_argument("--condition", type=float, default=1e4)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed-base", type=int, default=731)
    parser.add_argument(
        "--learning-rates", type=str,
        default="diagonal=.003,scalar=.003,full=.003,right_matrix=.003,exact_muon=.003",
    )
    args = parser.parse_args()
    rates = {}
    for assignment in args.learning_rates.split(","):
        name, value = assignment.split("=", 1)
        rates[name] = float(value)
    if set(rates) != set(MODES):
        raise ValueError("learning rates must cover every declared mode exactly")

    certificates, runs = [], []
    for seed_index in range(args.seeds):
        problem = make_operator_problem(
            args.dimension, args.condition, args.seed_base + seed_index
        )
        certificates.append({
            "seed": seed_index,
            "problem_seed": args.seed_base + seed_index,
            "directions": first_step_certificate(problem),
        })
        for mode in MODES:
            row = run_population(problem, mode, rates[mode], args.steps)
            row.update({
                "seed": seed_index,
                "problem_seed": args.seed_base + seed_index,
            })
            runs.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "history"}))

    payload = {
        "configuration": {**vars(args), "out": str(args.out)},
        "modes": list(MODES),
        "learning_rates": rates,
        "first_step_certificates": certificates,
        "runs": runs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"complete": True, "runs": len(runs)}, indent=2))


if __name__ == "__main__":
    main()
