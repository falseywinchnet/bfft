#!/usr/bin/env python3
"""Wolf plus transport restraint on the fixed curved-valley oracle."""

from __future__ import annotations

import argparse

import torch

from experiments.sgd_transport_restraint.optimizer import SignatureTransportSGD
from experiments.sgd_transport_restraint.run_signature_oracle import (
    first_below,
    rosenbrock,
)
from experiments.sgd_transport_restraint.wolf_optimizer import TransportWolf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimensions", type=int, default=16)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    initial = torch.ones(args.dimensions, dtype=torch.float64)
    initial[::2] = -1.2
    methods = (
        ("SGD", "sgd", "request"),
        ("SIG-shape", "signature", "request"),
        ("Wolf", "none", "request"),
        ("Wolf-shape", "shape", "request"),
        ("G-Wolf-scalar", "scalar", "gradient"),
        ("G-Wolf-shape", "shape", "gradient"),
        ("G-Wolf-rotate", "rotate", "gradient"),
    )
    for name, kind, observer_source in methods:
        torch.manual_seed(args.seed)
        value = torch.nn.Parameter(initial.clone())
        if kind == "sgd":
            optimizer = SignatureTransportSGD(
                [value], lr=args.lr, mode="none"
            )
        elif kind == "signature":
            optimizer = SignatureTransportSGD(
                [value], lr=args.lr, mode="shape"
            )
        else:
            optimizer = TransportWolf(
                [value], lr=args.lr, mode=kind,
                observer_source=observer_source,
            )
        trace = []
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            loss = rosenbrock(value)
            loss.backward()
            optimizer.step()
            if step % 100 == 0 or step + 1 == args.steps:
                trace.append((step, float(loss.item()), optimizer.last_ratio))
            if not torch.isfinite(loss):
                break
        fallback = getattr(optimizer, "last_fallback_fraction", 0.0)
        print(
            f"{name:12s} final={trace[-1][1]:.6g} "
            f"step<1e-4={first_below(trace, 1e-4)} "
            f"step<1e-8={first_below(trace, 1e-8)} "
            f"ratio={sum(x[2] for x in trace) / len(trace):.4f} "
            f"fallback={fallback:.3f}"
        )


if __name__ == "__main__":
    main()
