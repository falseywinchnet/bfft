#!/usr/bin/env python3
"""Paired Ripple experiment: AdamW versus source-aware self-context AdamW."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from ML_experiment.metrics import evaluate
from ML_experiment.run_benchmark import auc, fit_visualization, threshold, train_variant
from ML_experiment.tasks import ripple


def summarize(rows):
    result = {}
    for key in ("auc", "best_score", "best_step", "final_score", "final_loss"):
        values = np.asarray([row[key] for row in rows], dtype=float)
        result[key] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "values": values.tolist(),
        }
    for score in (.8, .9, .95):
        values = [row["thresholds"][str(score)] for row in rows]
        reached = [value for value in values if value is not None]
        result[f"steps_to_{score}"] = {
            "reached": len(reached),
            "of": len(values),
            "median": float(np.median(reached)) if reached else None,
            "values": values,
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=750)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--eval-every", type=int, default=5)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    arms = {
        "adamw": dict(
            optimizer_name="adamw",
            context_backward_mode="exact",
            context_gradient_mode="blended",
        ),
        "adamw_source_aware": dict(
            optimizer_name="adamw_source_aware",
            context_backward_mode="nonexpansive",
            context_gradient_mode="split_trust",
        ),
    }
    runs = []
    for seed in range(args.seeds):
        task = ripple(seed)
        for arm, configuration in arms.items():
            model, history, seconds, best_step, failure = train_variant(
                "self_context", task, args.width, seed, args.steps,
                args.batch, args.lr, args.eval_every, **configuration,
            )
            heldout = evaluate(model, task)
            row = {
                "arm": arm,
                "seed": seed,
                "history": history,
                "auc": auc(history, args.steps),
                "best_score": float(max(item["score"] for item in history)),
                "best_step": int(best_step),
                "final_score": float(history[-1]["score"]),
                "final_loss": float(history[-1]["loss"]),
                "heldout": heldout,
                "thresholds": {
                    str(value): threshold(history, value)
                    for value in (.8, .9, .95)
                },
                "seconds": seconds,
                "failure": failure,
                "fit_visualization": (
                    fit_visualization(model, task) if seed == 0 else None
                ),
            }
            runs.append(row)
            print(json.dumps({
                "arm": arm,
                "seed": seed,
                "auc": row["auc"],
                "best_score": row["best_score"],
                "best_step": best_step,
                "heldout": heldout["score"],
                "failure": failure,
            }))

    payload = {
        "experiment": "adamw_vs_source_aware_adamw_on_ripple",
        "protocol": {
            "variant": "self_context",
            "task": "ripple",
            "paired_initialization_and_batches": True,
            "seeds": args.seeds,
            "steps": args.steps,
            "width": args.width,
            "batch": args.batch,
            "lr": args.lr,
            "weight_decay": 1e-4,
            "eval_every": args.eval_every,
            "gradient_clip": 5.0,
        },
        "summaries": {
            arm: summarize([row for row in runs if row["arm"] == arm])
            for arm in arms
        },
        "runs": runs,
    }
    destination = args.out / "results.json"
    destination.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"complete": True, "output": str(destination)}, indent=2))


if __name__ == "__main__":
    torch.set_num_threads(8)
    main()
