#!/usr/bin/env python3
"""Wolf plus projected-signature restraint on the fixed MLP assay."""

from __future__ import annotations

import argparse
import copy
import math

import torch
from torch import nn

from experiments.sgd_transport_restraint.optimizer import SignatureTransportSGD
from experiments.sgd_transport_restraint.run_signature_mlp import MLP, evaluate, target
from experiments.sgd_transport_restraint.wolf_optimizer import TransportWolf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--condition", type=float, default=1e4)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--wolf-lr", type=float)
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--n-train", type=int, default=16384)
    parser.add_argument("--n-test", type=int, default=4096)
    parser.add_argument("--eval-every", type=int, default=25)
    args = parser.parse_args()
    wolf_lr = args.lr if args.wolf_lr is None else args.wolf_lr

    generator = torch.Generator().manual_seed(args.seed)
    rotation, _ = torch.linalg.qr(torch.randn(
        args.dimension, args.dimension, generator=generator
    ))
    scales = torch.logspace(0.0, -math.log10(args.condition), args.dimension)
    transform = rotation @ torch.diag(scales) @ rotation.T
    train_latent = torch.randn(
        args.n_train, args.dimension, generator=generator
    )
    test_latent = torch.randn(
        args.n_test, args.dimension, generator=generator
    )
    train_x = (train_latent @ transform.T).float()
    test_x = (test_latent @ transform.T).float()
    train_y = target(train_latent).float()
    test_y = target(test_latent).float()

    torch.manual_seed(args.seed + 1000)
    base = MLP(args.dimension, args.width, args.depth)
    initial = copy.deepcopy(base.state_dict())
    schedule = torch.randint(
        0, args.n_train, (args.steps, args.batch_size),
        generator=torch.Generator().manual_seed(918273),
    )
    methods = (
        ("SGD", "sgd", "request"),
        ("SIG-shape", "signature", "request"),
        ("Wolf", "none", "request"),
        ("Wolf-shape", "shape", "request"),
        ("G-Wolf-scalar", "scalar", "gradient"),
        ("G-Wolf-shape", "shape", "gradient"),
        ("G-Wolf-rotate", "rotate", "gradient"),
    )
    results = []
    for name, kind, observer_source in methods:
        model = MLP(args.dimension, args.width, args.depth)
        model.load_state_dict(initial)
        if kind == "sgd":
            optimizer = SignatureTransportSGD(
                model.parameters(), lr=args.lr, mode="none"
            )
        elif kind == "signature":
            optimizer = SignatureTransportSGD(
                model.parameters(), lr=args.lr, mode="shape"
            )
        else:
            optimizer = TransportWolf(
                model.parameters(), lr=wolf_lr, mode=kind,
                observer_source=observer_source,
            )
        torch.manual_seed(args.seed + 3000)
        trace = []
        fallback_sum = 0.0
        for step, indices in enumerate(schedule):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(
                model(train_x[indices]), train_y[indices]
            )
            loss.backward()
            optimizer.step()
            fallback_sum += getattr(
                optimizer, "last_fallback_fraction", 0.0
            )
            if step % args.eval_every == 0 or step + 1 == args.steps:
                trace.append({
                    "step": step,
                    "train": evaluate(model, train_x, train_y),
                    "test": evaluate(model, test_x, test_y),
                    "ratio": optimizer.last_ratio,
                })
        results.append({"method": name, "trace": trace})
        print(
            f"{name:12s} test={trace[-1]['test']:.6g} "
            f"ratio={sum(x['ratio'] for x in trace) / len(trace):.4f} "
            f"fallback={fallback_sum / args.steps:.3f}"
        )

    sgd_floor = results[0]["trace"][-1]["test"]
    print(f"\ntransition target: SGD final test floor {sgd_floor:.6g}")
    for record in results:
        first = next((
            point["step"] for point in record["trace"]
            if point["test"] <= sgd_floor
        ), None)
        print(f"{record['method']:12s} step-to-SGD-floor={first}")


if __name__ == "__main__":
    main()
