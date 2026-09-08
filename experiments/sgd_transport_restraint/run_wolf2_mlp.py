#!/usr/bin/env python3
"""High-LR rewritten Wolf transport on the fixed MLP assay."""

from __future__ import annotations

import argparse
import copy
import math

import torch
from torch import nn

from experiments.sgd_transport_restraint.optimizer import SignatureTransportSGD
from experiments.sgd_transport_restraint.run_signature_mlp import MLP, evaluate, target
from experiments.sgd_transport_restraint.wolf_optimizer import WolfTransportV2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--condition", type=float, default=1e4)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--baseline-lr", type=float, default=0.03)
    parser.add_argument("--wolf-lr", type=float, default=0.3)
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
        ("SGD", None),
        ("SIG-shape", "signature"),
        ("Wolf2", (False, False, False)),
        ("Wolf2-T", (False, False, True)),
        ("Wolf2-TR", (True, False, True)),
        ("Wolf2-TRS", (True, True, True)),
    )
    results = []
    for name, config in methods:
        model = MLP(args.dimension, args.width, args.depth)
        model.load_state_dict(initial)
        if config is None:
            optimizer = SignatureTransportSGD(
                model.parameters(), lr=args.baseline_lr, mode="none"
            )
        elif config == "signature":
            optimizer = SignatureTransportSGD(
                model.parameters(), lr=args.baseline_lr, mode="shape"
            )
        else:
            restrain_output, restrain_state, transport_state = config
            optimizer = WolfTransportV2(
                model.parameters(), lr=args.wolf_lr,
                restrain_output=restrain_output,
                restrain_state=restrain_state,
                transport_state=transport_state,
            )
        trace = []
        for step, indices in enumerate(schedule):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(
                model(train_x[indices]), train_y[indices]
            )
            loss.backward()
            optimizer.step()
            if step % args.eval_every == 0 or step + 1 == args.steps:
                trace.append({
                    "step": step,
                    "test": evaluate(model, test_x, test_y),
                    "ratio": optimizer.last_ratio,
                })
        results.append({"method": name, "trace": trace})
        print(
            f"{name:10s} test={trace[-1]['test']:.6g} "
            f"ratio={sum(x['ratio'] for x in trace) / len(trace):.4f}"
        )

    sgd_floor = results[0]["trace"][-1]["test"]
    print(f"\ntransition target: SGD final test floor {sgd_floor:.6g}")
    for record in results:
        first = next((
            point["step"] for point in record["trace"]
            if point["test"] <= sgd_floor
        ), None)
        sustained = next((
            point["step"] for index, point in enumerate(record["trace"])
            if point["test"] <= sgd_floor
            and all(
                later["test"] <= sgd_floor
                for later in record["trace"][index:]
            )
        ), None)
        best = min(record["trace"], key=lambda point: point["test"])
        print(
            f"{record['method']:10s} first={first} sustained={sustained} "
            f"best={best['test']:.6g}@{best['step']}"
        )


if __name__ == "__main__":
    main()
