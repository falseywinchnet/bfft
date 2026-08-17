#!/usr/bin/env python3
"""Test whether more observed dual-spiral turns improve equal-turn extrapolation."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from ML_experiment.models import parameter_count
from ML_experiment.run_benchmark import auc, train_variant
from ML_experiment.tasks import Task


VARIANTS = (
    "ordinary_mlp",
    "self_context",
    "self_context_stiefel_flow_curvature",
)
VISIBLE_TURNS = (2, 4, 8)
RADIUS_START = 0.10
RADIUS_PITCH = 0.45
PHASE_START = 0.55


def _split(x: torch.Tensor, y: torch.Tensor, seed: int):
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(x), generator=generator)
    cut = int(0.75 * len(x))
    return x[order[:cut]], y[order[:cut]], x[order[cut:]], y[order[cut:]]


def _dual_spiral_points(
    count_per_branch: int,
    lo: float,
    hi: float,
    seed: int,
):
    """Sample one fixed-pitch infinite dual spiral in turn coordinates."""
    generator = torch.Generator().manual_seed(seed)
    turn_coordinate = torch.rand(count_per_branch, generator=generator) * (hi - lo) + lo
    theta = PHASE_START + 2.0 * math.pi * turn_coordinate
    radius = RADIUS_START + RADIUS_PITCH * turn_coordinate
    base = torch.stack((radius * torch.cos(theta), radius * torch.sin(theta)), dim=1)
    points = torch.cat((base, -base), dim=0)
    noise = 0.04 * RADIUS_PITCH
    points += noise * torch.randn(points.shape, generator=generator)
    labels = torch.cat((
        torch.zeros(count_per_branch, dtype=torch.long),
        torch.ones(count_per_branch, dtype=torch.long),
    ))
    return points, labels


def spiral_truth(points: torch.Tensor):
    angle = torch.atan2(points[:, 1], points[:, 0])
    radius = torch.linalg.vector_norm(points, dim=1)
    turn_coordinate = (radius - RADIUS_START) / RADIUS_PITCH
    expected = PHASE_START + 2.0 * math.pi * turn_coordinate
    return (torch.cos(angle - expected) < 0).long()


def make_task(visible_turns: int, seed: int, points_per_turn: int = 900):
    count = points_per_turn * visible_turns
    x, y = _dual_spiral_points(count, 0.0, visible_turns, 4100 + seed)
    x_train, y_train, x_val, y_val = _split(x, y, 4200 + seed)
    tail_x, tail_y = [], []
    for turn in range(visible_turns):
        lo = visible_turns + turn
        hi = lo + 1.0
        tx, ty = _dual_spiral_points(450, lo, hi, 5000 + seed * 97 + turn)
        tail_x.append(tx)
        tail_y.append(ty)
    x_test = torch.cat(tail_x)
    y_test = torch.cat(tail_y)
    limit = RADIUS_START + 2 * RADIUS_PITCH * visible_turns + 0.15
    return Task(
        f"dual_spiral_{visible_turns}_turns",
        "classification",
        2,
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        tail_x,
        tail_y,
        (-limit, limit, -limit, limit),
        spiral_truth,
    )


@torch.no_grad()
def accuracy(model, x, y):
    return float((model(x).argmax(1) == y).float().mean())


@torch.no_grad()
def field_probe(model, visible_turns: int, size: int):
    limit = RADIUS_START + 2 * RADIUS_PITCH * visible_turns + 0.15
    axis = torch.linspace(-limit, limit, size)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    points = torch.stack((xx.flatten(), yy.flatten()), dim=1)
    probability = torch.softmax(model(points), dim=1)[:, 1]
    return {
        "size": size,
        "limits": [-limit, limit, -limit, limit],
        "truth": spiral_truth(points).tolist(),
        "probability": [round(float(value), 5) for value in probability],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/tmp/spiral_evidence_horizon"))
    parser.add_argument("--turns", default="2,4,8")
    parser.add_argument("--width", type=int, default=38)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--grid", type=int, default=101)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    run_partial = args.out / "runs.partial.json"
    probe_partial = args.out / "probes.partial.json"
    runs = (json.loads(run_partial.read_text())["runs"]
            if args.resume and run_partial.exists() else [])
    probes = (json.loads(probe_partial.read_text())["probes"]
              if args.resume and probe_partial.exists() else [])
    done = {(row["visible_turns"], row["seed"], row["variant"]) for row in runs}

    for visible_turns in map(int, args.turns.split(",")):
        for seed in range(args.seeds):
            task = make_task(visible_turns, seed)
            for variant in VARIANTS:
                key = (visible_turns, seed, variant)
                if key in done:
                    continue
                model, history, seconds, best_step = train_variant(
                    variant,
                    task,
                    args.width,
                    seed,
                    args.steps,
                    args.batch,
                    args.lr,
                    args.eval_every,
                    optimizer_name="adamw",
                )
                tail_scores = [
                    accuracy(model, x_tail, y_tail)
                    for x_tail, y_tail in zip(task.tail_x, task.tail_y)
                ]
                row = {
                    "visible_turns": visible_turns,
                    "withheld_turns": visible_turns,
                    "seed": seed,
                    "variant": variant,
                    "width": args.width,
                    "parameters": parameter_count(model),
                    "seconds": seconds,
                    "best_step": best_step,
                    "validation_score": accuracy(model, task.x_val, task.y_val),
                    "test_score": accuracy(model, task.x_test, task.y_test),
                    "final_turn_score": tail_scores[-1],
                    "tail_scores": tail_scores,
                    "learning_auc": auc(history, args.steps),
                    "history": history,
                }
                runs.append(row)
                run_partial.write_text(json.dumps({"runs": runs}, indent=2))
                probes.append({
                    "visible_turns": visible_turns,
                    "seed": seed,
                    "variant": variant,
                    **field_probe(model, visible_turns, args.grid),
                })
                probe_partial.write_text(json.dumps({"probes": probes}))
                print(json.dumps({
                    "turns": visible_turns,
                    "seed": seed,
                    "variant": variant,
                    "parameters": row["parameters"],
                    "validation": row["validation_score"],
                    "test": row["test_score"],
                    "final_turn": row["final_turn_score"],
                    "auc": row["learning_auc"],
                    "seconds": seconds,
                }), flush=True)

    configuration = {**vars(args), "out": str(args.out), "variants": VARIANTS}
    (args.out / "results.json").write_text(json.dumps({
        "configuration": configuration,
        "runs": runs,
    }, indent=2))
    (args.out / "probes.json").write_text(json.dumps({
        "configuration": configuration,
        "probes": probes,
    }))
    print(json.dumps({"complete": True, "runs": len(runs), "probes": len(probes)}))


if __name__ == "__main__":
    main()
