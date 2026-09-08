#!/usr/bin/env python3
"""Caveman-grade same-batch response-restraint experiment.

Every minibatch is used for exactly two consecutive state transitions. This
gives the transport observer an uncontaminated finite response

    s = theta_1 - theta_0,
    y = grad(theta_1; B) - grad(theta_0; B).

All methods see the same initial model and paired minibatch schedule. There is
one learning rate and no momentum, scheduler, clipping, or parameter sweep.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.sgd_transport_restraint.optimizer import ResponseTransportSGD


@dataclass
class Trace:
    method: str
    condition: float
    steps: list[int]
    train_loss: list[float]
    test_loss: list[float]
    coherence: list[float]
    gradient_ratio: list[float]
    active_fraction: list[float]


class MLP(nn.Module):
    def __init__(self, dimension: int, width: int, depth: int) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(dimension, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width), nn.Tanh()])
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_problem(
    *, seed: int, condition: float, n_train: int, n_test: int, dimension: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(dimension, dimension, generator=generator))
    scales = torch.logspace(0.0, -math.log10(condition), dimension)
    transform = q @ torch.diag(scales) @ q.T

    def sample(count: int) -> tuple[torch.Tensor, torch.Tensor]:
        latent = torch.randn(count, dimension, generator=generator)
        observed = latent @ transform.T
        target = (
            0.65 * torch.sin(1.7 * latent[:, 0] + 0.8 * latent[:, 7])
            + 0.35 * torch.sin(3.1 * latent[:, 14] - 0.4 * latent[:, 3])
            + 0.25 * torch.tanh(latent[:, 20] * latent[:, 27])
            + 0.15 * latent[:, -1]
        )
        return observed.float(), target[:, None].float()

    train_x, train_y = sample(n_train)
    test_x, test_y = sample(n_test)
    return train_x, train_y, test_x, test_y


def make_optimizer(method: str, model: nn.Module, lr: float, alpha: float):
    mode = {"SGD": "none", "TR-scalar": "scalar", "TR-shape": "shape"}[method]
    return ResponseTransportSGD(model.parameters(), lr=lr, alpha=alpha, mode=mode)


@torch.no_grad()
def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    return float(nn.functional.mse_loss(model(x), y).item())


def run_method(
    method: str,
    *,
    initial_state: dict,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    schedule: torch.Tensor,
    lr: float,
    alpha: float,
    dimension: int,
    width: int,
    depth: int,
    eval_every: int,
) -> Trace:
    model = MLP(dimension, width, depth)
    model.load_state_dict(initial_state)
    optimizer = make_optimizer(method, model, lr, alpha)

    steps: list[int] = []
    train_losses: list[float] = []
    test_losses: list[float] = []
    coherences: list[float] = []
    ratios: list[float] = []
    active_fractions: list[float] = []
    step = 0

    for indices in schedule:
        xb = train_x[indices]
        yb = train_y[indices]
        for phase in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(xb), yb)
            loss.backward()
            optimizer.step(observe=(phase == 1))

            # Checkpoint after an observed (second-in-pair) transition so the
            # transport diagnostics describe an active observer step.
            if (step + 1) % eval_every == 0 or step == 2 * len(schedule) - 1:
                steps.append(step)
                train_losses.append(evaluate(model, train_x, train_y))
                test_losses.append(evaluate(model, test_x, test_y))
                coherences.append(optimizer.last_coherence)
                ratios.append(optimizer.last_ratio)
                active_fractions.append(optimizer.last_active_fraction)
            step += 1

    return Trace(
        method=method,
        condition=0.0,
        steps=steps,
        train_loss=train_losses,
        test_loss=test_losses,
        coherence=coherences,
        gradient_ratio=ratios,
        active_fraction=active_fractions,
    )


def first_at_or_below(trace: Trace, threshold: float) -> int | None:
    for step, value in zip(trace.steps, trace.train_loss):
        if value <= threshold:
            return step
    return None


def run_condition(args: argparse.Namespace, condition: float) -> list[Trace]:
    train_x, train_y, test_x, test_y = make_problem(
        seed=args.seed,
        condition=condition,
        n_train=args.n_train,
        n_test=args.n_test,
        dimension=args.dimension,
    )
    torch.manual_seed(args.seed + 1000)
    base = MLP(args.dimension, args.width, args.depth)
    initial_state = copy.deepcopy(base.state_dict())
    generator = torch.Generator().manual_seed(args.seed + 2000)
    schedule = torch.randint(
        0,
        args.n_train,
        (args.pairs, args.batch_size),
        generator=generator,
    )

    traces = []
    for method in ("SGD", "TR-scalar", "TR-shape"):
        trace = run_method(
            method,
            initial_state=initial_state,
            train_x=train_x,
            train_y=train_y,
            test_x=test_x,
            test_y=test_y,
            schedule=schedule,
            lr=args.lr,
            alpha=args.alpha,
            dimension=args.dimension,
            width=args.width,
            depth=args.depth,
            eval_every=args.eval_every,
        )
        trace.condition = condition
        traces.append(trace)

    floor = min(min(trace.train_loss) for trace in traces)
    threshold = floor * 1.05
    print(f"\ncondition={condition:g}, empirical floor={floor:.6g}, 1.05x={threshold:.6g}")
    print("method       final-train  best-train   final-test   step-to-1.05x   mean-ratio  coherence  active")
    for trace in traces:
        print(
            f"{trace.method:10s}  {trace.train_loss[-1]:10.6g}  "
            f"{min(trace.train_loss):10.6g}  "
            f"{trace.test_loss[-1]:10.6g}  "
            f"{str(first_at_or_below(trace, threshold)):>13s}  "
            f"{np.mean(trace.gradient_ratio):10.4f}  "
            f"{np.mean(trace.coherence):14.4f}  "
            f"{np.mean(trace.active_fraction):6.3f}"
        )
    return traces


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--conditions", default="1,10000")
    parser.add_argument("--pairs", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-train", type=int, default=8192)
    parser.add_argument("--n-test", type=int, default=2048)
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    all_traces: list[Trace] = []
    for condition in (float(value) for value in args.conditions.split(",")):
        all_traces.extend(run_condition(args, condition))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps([asdict(trace) for trace in all_traces], indent=2) + "\n"
        )
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
