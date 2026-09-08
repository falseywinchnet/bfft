#!/usr/bin/env python3
"""Caveman oracle for the projected-signature SGD transplant."""

from __future__ import annotations

import argparse

import torch

from experiments.sgd_transport_restraint.optimizer import SignatureTransportSGD


def rosenbrock(value: torch.Tensor) -> torch.Tensor:
    return torch.sum(
        100.0 * (value[1:] - value[:-1].square()).square()
        + (1.0 - value[:-1]).square()
    )


def run(
    method: str,
    mode: str,
    initial: torch.Tensor,
    lr: float,
    alpha: float,
    steps: int,
    gate: str,
    fusion: int,
    observer: str,
    cell: str,
    normal_rank: int,
    momentum: float,
    transport_momentum: bool,
    project_momentum: bool,
):
    value = torch.nn.Parameter(initial.clone())
    optimizer = SignatureTransportSGD(
        [value], lr=lr, alpha=alpha, gate=gate, fusion=fusion,
        observer=observer, cell=cell, normal_rank=normal_rank, mode=mode,
        momentum=momentum, transport_momentum=transport_momentum,
        normalized_momentum=momentum > 0.0,
        project_momentum=project_momentum,
    )
    trace = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = rosenbrock(value)
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == steps - 1:
            trace.append((step, float(loss.item()), optimizer.last_ratio))
        if not torch.isfinite(loss):
            break
    return trace, value.detach()


def first_below(trace, threshold: float):
    return next((step for step, loss, _ in trace if loss <= threshold), None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimensions", type=int, default=16)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--fusion", type=int, default=1)
    parser.add_argument(
        "--observer",
        choices=("single", "capacity", "cycle2"),
        default="single",
    )
    parser.add_argument("--cell", choices=("row", "tensor"), default="row")
    parser.add_argument("--normal-rank", type=int, default=1)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument(
        "--gate",
        choices=("soft", "quartic", "amplitude", "reversal", "hard"),
        default="soft",
    )
    args = parser.parse_args()

    initial = torch.ones(args.dimensions, dtype=torch.float64)
    initial[::2] = -1.2
    if args.momentum > 0.0:
        methods = (
            ("SGD", "none", 0.0, False, False),
            ("Momentum", "none", args.momentum, False, False),
            ("SIG-shape", "shape", 0.0, False, False),
            ("SIG+M", "shape", args.momentum, False, False),
            ("T-M", "none", args.momentum, True, False),
            ("P-M", "shape", args.momentum, False, True),
            ("TP-M", "shape", args.momentum, True, True),
        )
    else:
        methods = (
            ("SGD", "none", 0.0, False, False),
            ("SIG-scalar", "scalar", 0.0, False, False),
            ("SIG-shape", "shape", 0.0, False, False),
            ("SIG-rotate", "rotate", 0.0, False, False),
        )
    for method, mode, momentum, transport_momentum, project_momentum in methods:
        trace, final = run(
            method, mode, initial, args.lr, args.alpha, args.steps,
            args.gate, args.fusion, args.observer, args.cell,
            args.normal_rank, momentum, transport_momentum,
            project_momentum,
        )
        print(
            f"{method:10s} final={trace[-1][1]:.6g} "
            f"step<1e-4={first_below(trace, 1e-4)} "
            f"step<1e-8={first_below(trace, 1e-8)} "
            f"distance={torch.linalg.vector_norm(final - 1.0).item():.6g} "
            f"mean_ratio={sum(point[2] for point in trace) / len(trace):.4f}"
        )


if __name__ == "__main__":
    main()
