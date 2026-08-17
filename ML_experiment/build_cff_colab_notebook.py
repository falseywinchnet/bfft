#!/usr/bin/env python3
"""Generate a standalone Colab notebook for CFF on the 8-turn dual spiral."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def markdown(source: str):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=HERE / "CFF_8_turn_spiral_colab.ipynb",
    )
    args = parser.parse_args()
    cff_source = (HERE / "cff.py").read_text()

    cells = [
        markdown(
            """# Continuous Frame Flow on the fixed-pitch 8-turn dual spiral

This notebook is standalone and Colab-ready. It contains:

- the complete `ContinuousFrameFlowLinear(nn.Module)` layer;
- a two-layer CFF network;
- exactly parameter-matched Vanilla LELU MLP and Self-context controls;
- the fixed-pitch **8 visible / 8 withheld turn** dual-spiral problem;
- AdamW training, observed/extrapolation metrics, learning curves, and geometric decision plots.

No repository imports or task-derived coordinates are used. The default is the strongest tested CFF setting: width 38, 2,000 steps. Runtime → Change runtime type → GPU is optional; CPU also works.
"""
        ),
        code(
            """import copy
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WIDTH = 38
STEPS = 2_000
BATCH = 256
LR = 3e-3
SEED = 0
EVAL_EVERY = 50
GRID = 181

if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")
print("device:", DEVICE)
"""
        ),
        markdown("## Standalone CFF layer and network"),
        code(cff_source),
        markdown("## Exactly parameter-matched controls"),
        code(
            """class SelfContextLinear(nn.Module):
    \"\"\"The first-order self-context control: no frame flow or curvature shell.\"\"\"

    def __init__(self, n_in, n_out, *, directions=12, rank=4, strength=.25):
        super().__init__()
        self.rank = rank
        self.strength = float(strength)
        self.base = nn.Linear(n_in, n_out)
        self.metric = nn.Linear(n_in, rank * rank)
        self.response = nn.Sequential(nn.Linear(4, 12), LELU(), nn.Linear(12, 1))
        self.shared = nn.Parameter(torch.randn(rank, n_out) / math.sqrt(rank))
        self.correction_scale = nn.Parameter(torch.tensor(-1.5))
        generator = torch.Generator().manual_seed(9157 + n_in + n_out)
        atlas = F.normalize(torch.randn(directions, rank, n_in, generator=generator), dim=-1)
        self.register_buffer("frame_atlas", atlas)

    def _allocate(self, x):
        factor = self.metric(x).view(len(x), self.rank, self.rank)
        metric = factor @ factor.transpose(1, 2) / self.rank
        projected = torch.einsum("dri,bi->bdr", self.frame_atlas, x)
        cost = torch.einsum("bdr,brs,bds->bd", projected, metric, projected)
        norm = projected.square().mean(-1)
        stats = torch.stack((torch.log1p(cost), torch.log1p(norm), projected.mean(-1),
                             torch.log1p(projected.abs().mean(-1))), -1)
        logits = self.response(stats).squeeze(-1)
        logits = logits - cost / (cost.mean(1, keepdim=True) + 1e-5)
        return projected, torch.softmax(logits, 1)

    def _lift(self, projected, weight):
        return torch.einsum("bd,bdr,dri->bi", weight, projected, self.frame_atlas) / self.rank

    def forward(self, x):
        projected, weight = self._allocate(x)
        context = self._lift(projected, weight)
        context_rms = context.square().mean(1, keepdim=True).sqrt().clamp_min(1e-6)
        input_rms = x.square().mean(1, keepdim=True).sqrt().detach().clamp_min(1e-6)
        chart = x + self.strength * context * (input_rms / context_rms)
        projected, weight = self._allocate(chart)
        pooled = torch.einsum("bd,bdr->br", weight, projected)
        correction = F.softplus(self.correction_scale) * (pooled @ self.shared)
        return self.base(x) + correction


class SelfContext(nn.Module):
    def __init__(self, input_dim, output_dim, width=38):
        super().__init__()
        self.embed = nn.Linear(input_dim, width)
        self.up = SelfContextLinear(width, 2 * width)
        self.activation = LELU()
        self.down = SelfContextLinear(2 * width, width)
        self.output = nn.Linear(width, output_dim)

    def forward(self, x):
        return self.output(self.down(self.activation(self.up(self.embed(x)))))


class BudgetMatchedMLP(nn.Module):
    \"\"\"Ordinary encode-expand-LELU-contract-decode MLP at an exact budget.\"\"\"

    def __init__(self, input_dim, output_dim, width, parameter_budget):
        super().__init__()
        fixed = (input_dim + 1) * width + width + (width + 1) * output_dim
        per_hidden = 2 * width + 1
        expansion = (parameter_budget - fixed) // per_hidden
        self.encode = nn.Linear(input_dim, width)
        self.up = nn.Linear(width, expansion)
        self.activation = LELU()
        self.down = nn.Linear(expansion, width)
        self.decode = nn.Linear(width, output_dim)
        remainder = parameter_budget - (fixed + per_hidden * expansion)
        self.extra = nn.Parameter(torch.zeros(remainder))
        generator = torch.Generator().manual_seed(17159 + input_dim + output_dim + parameter_budget)
        if remainder:
            bw = torch.randn(remainder, output_dim, input_dim, generator=generator)
            bb = torch.randn(remainder, output_dim, generator=generator)
            self.register_buffer("basis_weight", bw / math.sqrt(input_dim * output_dim))
            self.register_buffer("basis_bias", bb / math.sqrt(output_dim))
        else:
            self.register_buffer("basis_weight", torch.empty(0, output_dim, input_dim))
            self.register_buffer("basis_bias", torch.empty(0, output_dim))

    def forward(self, x):
        result = self.decode(self.down(self.activation(self.up(self.encode(x)))))
        if self.extra.numel():
            weight = torch.einsum("r,roi->oi", self.extra, self.basis_weight)
            bias = torch.einsum("r,ro->o", self.extra, self.basis_bias)
            result = result + x @ weight.T + bias
        return result


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_models(seed=0):
    torch.manual_seed(10_000 + seed)
    cff = ContinuousFrameFlow(2, 2, width=WIDTH)
    budget = parameter_count(cff)
    torch.manual_seed(10_000 + seed)
    self_context = SelfContext(2, 2, width=WIDTH)
    torch.manual_seed(10_000 + seed)
    vanilla = BudgetMatchedMLP(2, 2, WIDTH, budget)
    models = {"Vanilla MLP": vanilla, "Self-context": self_context, "CFF": cff}
    assert {parameter_count(model) for model in models.values()} == {budget}
    return models


models = make_models(SEED)
print({name: parameter_count(model) for name, model in models.items()})
"""
        ),
        markdown("## Fixed-pitch 8-visible / 8-withheld dual spiral"),
        code(
            """RADIUS_START = .10
RADIUS_PITCH = .45
PHASE_START = .55
VISIBLE_TURNS = 8


def dual_spiral_points(count_per_branch, lo_turn, hi_turn, seed):
    generator = torch.Generator().manual_seed(seed)
    t = torch.rand(count_per_branch, generator=generator) * (hi_turn - lo_turn) + lo_turn
    theta = PHASE_START + 2 * math.pi * t
    radius = RADIUS_START + RADIUS_PITCH * t
    arm = torch.stack((radius * torch.cos(theta), radius * torch.sin(theta)), 1)
    x = torch.cat((arm, -arm))
    x += .04 * RADIUS_PITCH * torch.randn(x.shape, generator=generator)
    y = torch.cat((torch.zeros(count_per_branch, dtype=torch.long),
                   torch.ones(count_per_branch, dtype=torch.long)))
    return x, y


def spiral_truth(x):
    angle = torch.atan2(x[:, 1], x[:, 0])
    radius = torch.linalg.vector_norm(x, dim=1)
    turn = (radius - RADIUS_START) / RADIUS_PITCH
    expected = PHASE_START + 2 * math.pi * turn
    return (torch.cos(angle - expected) < 0).long()


def make_problem(seed=0, points_per_turn=900):
    x, y = dual_spiral_points(points_per_turn * VISIBLE_TURNS, 0, VISIBLE_TURNS, 4100 + seed)
    generator = torch.Generator().manual_seed(4200 + seed)
    order = torch.randperm(len(x), generator=generator)
    cut = int(.75 * len(x))
    x_train, y_train = x[order[:cut]], y[order[:cut]]
    x_val, y_val = x[order[cut:]], y[order[cut:]]
    tails = [dual_spiral_points(450, VISIBLE_TURNS + turn, VISIBLE_TURNS + turn + 1,
                                5000 + seed * 97 + turn)
             for turn in range(VISIBLE_TURNS)]
    x_test = torch.cat([item[0] for item in tails])
    y_test = torch.cat([item[1] for item in tails])
    return {"train": (x_train, y_train), "val": (x_val, y_val),
            "test": (x_test, y_test), "tails": tails}


problem = make_problem(SEED)
print({split: tuple(value.shape for value in problem[split])
       for split in ("train", "val", "test")})
"""
        ),
        markdown("## Training"),
        code(
            """@torch.no_grad()
def accuracy(model, pair):
    x, y = pair
    return float((model(x).argmax(1) == y).float().mean())


def train(model, problem, seed=0):
    model = model.to(DEVICE)
    train_pair = tuple(value.to(DEVICE) for value in problem["train"])
    val_pair = tuple(value.to(DEVICE) for value in problem["val"])
    test_pair = tuple(value.to(DEVICE) for value in problem["test"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(190_000 + seed)
    best = (-1., None, 0)
    history = []
    started = time.perf_counter()
    for step in range(1, STEPS + 1):
        index = torch.randint(len(train_pair[0]), (BATCH,), generator=generator).to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(train_pair[0][index]), train_pair[1][index])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        if step == 1 or step % EVAL_EVERY == 0 or step == STEPS:
            observed = accuracy(model, val_pair)
            withheld = accuracy(model, test_pair)
            history.append((step, observed, withheld, float(loss.detach())))
            if observed > best[0]:
                best = (observed, copy.deepcopy(model.state_dict()), step)
    model.load_state_dict(best[1])
    elapsed = time.perf_counter() - started
    return model, history, {"observed": accuracy(model, val_pair),
                            "withheld": accuracy(model, test_pair),
                            "best_step": best[2], "seconds": elapsed}


trained, histories, results = {}, {}, {}
for index, (name, model) in enumerate(models.items()):
    trained[name], histories[name], results[name] = train(model, problem, SEED)
    print(name, results[name])
"""
        ),
        markdown("## Learning curves"),
        code(
            """fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for name, history in histories.items():
    steps = [row[0] for row in history]
    axes[0].plot(steps, [row[1] for row in history], label=name)
    axes[1].plot(steps, [row[2] for row in history], label=name)
axes[0].set_title("Observed-region accuracy")
axes[1].set_title("Eight-turn extrapolation")
for axis in axes:
    axis.set_xlabel("training step")
    axis.set_ylabel("accuracy")
    axis.set_ylim(.45, 1.01)
    axis.grid(alpha=.2)
axes[0].legend()
plt.tight_layout()
plt.show()
"""
        ),
        markdown("## Decision geometry (solid = visible, dashed = withheld)"),
        code(
            """@torch.no_grad()
def probability_field(model, points, chunk=8192):
    values = []
    for start in range(0, len(points), chunk):
        logits = model(points[start:start + chunk].to(DEVICE))
        values.append(torch.softmax(logits, 1)[:, 1].cpu())
    return torch.cat(values)


limit = RADIUS_START + 2 * RADIUS_PITCH * VISIBLE_TURNS + .15
axis = torch.linspace(-limit, limit, GRID)
yy, xx = torch.meshgrid(axis, axis, indexing="ij")
grid_points = torch.stack((xx.flatten(), yy.flatten()), 1)
truth = spiral_truth(grid_points).reshape(GRID, GRID)


def arm(turn_lo, turn_hi, branch, count=1600):
    t = torch.linspace(turn_lo, turn_hi, count)
    radius = RADIUS_START + RADIUS_PITCH * t
    theta = PHASE_START + 2 * math.pi * t + branch * math.pi
    return radius * torch.cos(theta), radius * torch.sin(theta)


fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
fields = [("Truth", truth.float().numpy())]
for name, model in trained.items():
    fields.append((f"{name}\\nobs {results[name]['observed']:.3f} · out {results[name]['withheld']:.3f}",
                   probability_field(model, grid_points).reshape(GRID, GRID).numpy()))

for plot_axis, (title, field) in zip(axes, fields):
    plot_axis.imshow(field, origin="lower", extent=(-limit, limit, -limit, limit),
                     cmap="coolwarm", vmin=0, vmax=1, interpolation="nearest")
    for branch, color in ((0, "#2468d8"), (1, "#e17628")):
        x_in, y_in = arm(0, VISIBLE_TURNS, branch)
        x_out, y_out = arm(VISIBLE_TURNS, 2 * VISIBLE_TURNS, branch)
        plot_axis.plot(x_in, y_in, color=color, linewidth=1.2)
        plot_axis.plot(x_out, y_out, color=color, linewidth=1.0, linestyle="--")
    plot_axis.add_patch(plt.Circle((0, 0), RADIUS_START + RADIUS_PITCH * VISIBLE_TURNS,
                                   fill=False, color="black", linestyle=":", alpha=.6))
    plot_axis.set_title(title)
    plot_axis.set_xlabel("x₁")
    plot_axis.set_ylabel("x₂")
    plot_axis.set_aspect("equal")
    plot_axis.set_xlim(-limit, limit)
    plot_axis.set_ylim(-limit, limit)
plt.show()
"""
        ),
        markdown(
            """## Notes

- Checkpoints are selected only by observed-region validation accuracy.
- The test set is the next eight complete turns, not random held-out points from the observed disk.
- The CFF atlas is fixed and task-agnostic. Input-dependent metric allocation, self-context, and symmetric curvature probes are learned end-to-end by ordinary backpropagation.
- To reproduce the four-seed study, set `SEED` to 0–3 and rerun the model/problem/training cells for each seed.
"""
        ),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": args.out.name, "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    args.out.write_text(json.dumps(notebook, indent=1))
    print(args.out)


if __name__ == "__main__":
    main()
