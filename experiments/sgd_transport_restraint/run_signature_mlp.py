#!/usr/bin/env python3
"""Projected-signature restraint on an ordinary ill-conditioned MLP."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import torch
from torch import nn

from experiments.sgd_transport_restraint.optimizer import SignatureTransportSGD


class MLP(nn.Module):
    def __init__(self, dimension: int, width: int, depth: int) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(dimension, width), nn.GELU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width), nn.GELU()])
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


def make_data(count: int, dimension: int, condition: float, generator):
    latent = torch.randn(count, dimension, generator=generator)
    return latent


def target(latent: torch.Tensor) -> torch.Tensor:
    value = (
        0.65 * torch.sin(1.7 * latent[:, 0] + 0.8 * latent[:, 7])
        + 0.35 * torch.sin(3.1 * latent[:, 14] - 0.4 * latent[:, 3])
        + 0.25 * torch.tanh(latent[:, 20] * latent[:, 27])
        + 0.15 * latent[:, -1]
    )
    return value[:, None]


@torch.no_grad()
def evaluate(model, values, labels):
    return float(nn.functional.mse_loss(model(values), labels).item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--condition", type=float, default=1e4)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=0.03)
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
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--n-train", type=int, default=16384)
    parser.add_argument("--n-test", type=int, default=4096)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    generator = torch.Generator().manual_seed(args.seed)
    rotation, _ = torch.linalg.qr(torch.randn(
        args.dimension, args.dimension, generator=generator
    ))
    scales = torch.logspace(0.0, -math.log10(args.condition), args.dimension)
    transform = rotation @ torch.diag(scales) @ rotation.T
    train_latent = make_data(args.n_train, args.dimension, args.condition, generator)
    test_latent = make_data(args.n_test, args.dimension, args.condition, generator)
    train_x, train_y = (train_latent @ transform.T).float(), target(train_latent).float()
    test_x, test_y = (test_latent @ transform.T).float(), target(test_latent).float()

    torch.manual_seed(args.seed + 1000)
    base = MLP(args.dimension, args.width, args.depth)
    initial = copy.deepcopy(base.state_dict())
    schedule = torch.randint(
        0, args.n_train, (args.steps, args.batch_size),
        generator=torch.Generator().manual_seed(918273),
    )

    results = []
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
        model = MLP(args.dimension, args.width, args.depth)
        model.load_state_dict(initial)
        optimizer = SignatureTransportSGD(
            model.parameters(), lr=args.lr, alpha=args.alpha,
            gate=args.gate, fusion=args.fusion,
            observer=args.observer, cell=args.cell,
            normal_rank=args.normal_rank, mode=mode,
            momentum=momentum, transport_momentum=transport_momentum,
            normalized_momentum=momentum > 0.0,
            project_momentum=project_momentum,
        )
        trace = []
        for step, indices in enumerate(schedule):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(train_x[indices]), train_y[indices])
            loss.backward()
            optimizer.step()
            if step % args.eval_every == 0 or step + 1 == args.steps:
                trace.append({
                    "step": step,
                    "train": evaluate(model, train_x, train_y),
                    "test": evaluate(model, test_x, test_y),
                    "ratio": optimizer.last_ratio,
                    "turn": optimizer.last_turn,
                })
        record = {"method": method, "trace": trace}
        results.append(record)
        print(
            f"{method:10s} train={trace[-1]['train']:.6g} "
            f"test={trace[-1]['test']:.6g} best={min(x['test'] for x in trace):.6g} "
            f"mean_ratio={sum(x['ratio'] for x in trace) / len(trace):.4f} "
            f"mean_turn={sum(x['turn'] for x in trace) / len(trace):.4f}"
        )

    sgd_floor = results[0]["trace"][-1]["test"]
    print(f"\ntransition target: SGD final test floor {sgd_floor:.6g}")
    for record in results:
        first = next(
            (
                point["step"]
                for point in record["trace"]
                if point["test"] <= sgd_floor
            ),
            None,
        )
        print(f"{record['method']:10s} step-to-SGD-floor={first}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "config": vars(args) | {"output": str(args.output)},
            "results": results,
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
