#!/usr/bin/env python3
"""Deterministic curved-valley oracle for response transport restraint."""

from __future__ import annotations

import argparse
import copy

import torch

from experiments.sgd_transport_restraint.optimizer import ResponseTransportSGD


def rosenbrock(x: torch.Tensor) -> torch.Tensor:
    return torch.sum(100.0 * (x[1:] - x[:-1].square()).square() + (1.0 - x[:-1]).square())


def run(method: str, initial: torch.Tensor, lr: float, alpha: float, steps: int):
    x = torch.nn.Parameter(initial.clone())
    mode = {"SGD": "none", "TR-scalar": "scalar", "TR-shape": "shape"}[method]
    optimizer = ResponseTransportSGD([x], lr=lr, alpha=alpha, mode=mode)
    trace = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = rosenbrock(x)
        loss.backward()
        optimizer.step(observe=(step > 0))
        if step % 100 == 0 or step == steps - 1:
            trace.append((step, float(loss.item()), optimizer.last_ratio))
        if not torch.isfinite(loss):
            break
    return trace, x.detach()


def first_below(trace, threshold: float):
    return next((step for step, loss, _ in trace if loss <= threshold), None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimensions", type=int, default=16)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()

    initial = torch.ones(args.dimensions, dtype=torch.float64)
    initial[::2] = -1.2
    results = {}
    for method in ("SGD", "TR-scalar", "TR-shape"):
        trace, final = run(method, initial, args.lr, args.alpha, args.steps)
        results[method] = trace
        print(
            f"{method:10s} final={trace[-1][1]:.6g} "
            f"step<1e-4={first_below(trace, 1e-4)} "
            f"step<1e-8={first_below(trace, 1e-8)} "
            f"distance={torch.linalg.vector_norm(final - 1.0).item():.6g} "
            f"mean_ratio={sum(point[2] for point in trace) / len(trace):.4f}"
        )


if __name__ == "__main__":
    main()
