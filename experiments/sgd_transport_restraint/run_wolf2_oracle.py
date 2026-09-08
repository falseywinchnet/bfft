#!/usr/bin/env python3
"""Rewritten Wolf lead-lag momentum plus transport on Rosenbrock."""

from __future__ import annotations

import argparse

import torch

from experiments.sgd_transport_restraint.optimizer import SignatureTransportSGD
from experiments.sgd_transport_restraint.run_signature_oracle import (
    first_below,
    rosenbrock,
)
from experiments.sgd_transport_restraint.wolf_optimizer import WolfTransportV2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimensions", type=int, default=16)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--lr", type=float, default=0.002)
    args = parser.parse_args()
    initial = torch.ones(args.dimensions, dtype=torch.float64)
    initial[::2] = -1.2
    methods = (
        ("SGD", None),
        ("SIG-shape", "signature"),
        ("Wolf2", (False, False, False)),
        ("Wolf2-T", (False, False, True)),
        ("Wolf2-R", (True, False, False)),
        ("Wolf2-RS", (True, True, False)),
        ("Wolf2-TR", (True, False, True)),
        ("Wolf2-TRS", (True, True, True)),
    )
    for name, config in methods:
        value = torch.nn.Parameter(initial.clone())
        if config is None:
            optimizer = SignatureTransportSGD(
                [value], lr=args.lr, mode="none"
            )
        elif config == "signature":
            optimizer = SignatureTransportSGD(
                [value], lr=args.lr, mode="shape"
            )
        else:
            restrain_output, restrain_state, transport_state = config
            optimizer = WolfTransportV2(
                [value], lr=args.lr,
                restrain_output=restrain_output,
                restrain_state=restrain_state,
                transport_state=transport_state,
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
        print(
            f"{name:10s} final={trace[-1][1]:.6g} "
            f"step<1e-4={first_below(trace, 1e-4)} "
            f"step<1e-8={first_below(trace, 1e-8)} "
            f"ratio={sum(x[2] for x in trace) / len(trace):.4f}"
        )


if __name__ == "__main__":
    main()
