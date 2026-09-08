#!/usr/bin/env python3
"""Diagnostics for Anchor on the fixed ill-conditioned MLP assay."""

from __future__ import annotations

import argparse
import copy
import math

import torch
from torch import nn

from ML_experiment.anchor import Anchor
from experiments.sgd_transport_restraint.run_signature_mlp import MLP, evaluate, target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--condition", type=float, default=1e4)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--target", type=float, default=0.310903)
    parser.add_argument("--trust-radius", type=float)
    parser.add_argument("--isotropy-cap", type=float)
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--n-train", type=int, default=16384)
    parser.add_argument("--n-test", type=int, default=4096)
    parser.add_argument("--eval-every", type=int, default=25)
    args = parser.parse_args()

    generator = torch.Generator().manual_seed(args.seed)
    rotation, _ = torch.linalg.qr(torch.randn(
        args.dimension, args.dimension, generator=generator
    ))
    scales = torch.logspace(0.0, -math.log10(args.condition), args.dimension)
    transform = rotation @ torch.diag(scales) @ rotation.T
    train_latent = torch.randn(args.n_train, args.dimension, generator=generator)
    test_latent = torch.randn(args.n_test, args.dimension, generator=generator)
    train_x = (train_latent @ transform.T).float()
    test_x = (test_latent @ transform.T).float()
    train_y = target(train_latent).float()
    test_y = target(test_latent).float()

    torch.manual_seed(args.seed + 1000)
    initial_model = MLP(args.dimension, args.width, args.depth)
    initial = copy.deepcopy(initial_model.state_dict())
    model = MLP(args.dimension, args.width, args.depth)
    model.load_state_dict(initial)
    optimizer = Anchor(
        model.parameters(), lr=args.lr, trust_radius=args.trust_radius,
        isotropy_cap=args.isotropy_cap,
    )
    schedule = torch.randint(
        0, args.n_train, (args.steps, args.batch_size),
        generator=torch.Generator().manual_seed(918273),
    )

    first = None
    relative_steps = []
    for step, indices in enumerate(schedule):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(
            model(train_x[indices]), train_y[indices]
        )
        loss.backward()
        optimizer.step()
        relative_steps.append(optimizer.last_relative_step)
        if step % args.eval_every == 0 or step + 1 == args.steps:
            test_loss = evaluate(model, test_x, test_y)
            if first is None and test_loss <= args.target:
                first = step
            print(
                f"step={step:4d} test={test_loss:.6g} "
                f"turn={optimizer.last_turn:.4f} "
                f"align={optimizer.last_alignment:.4f} "
                f"relative={optimizer.last_relative_step:.6f} "
                f"lr-scale={optimizer.last_lr_scale:.4f} "
                f"concentration={optimizer.last_update_concentration:.3f}"
            )
    ordered = sorted(relative_steps)
    print(
        f"first={first} max_relative={max(ordered):.6f} "
        f"p50_relative={ordered[len(ordered)//2]:.6f} "
        f"p95_relative={ordered[int(.95*(len(ordered)-1))]:.6f}"
    )


if __name__ == "__main__":
    main()
