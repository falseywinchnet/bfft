"""Paired six-optimizer analysis for the neural and operator batteries."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


OPTIMIZERS = (
    "adamw",
    "sgd",
    "anchor",
    "anchor_restrained_momentum",
    "muon",
    "lepton_transport",
)
METRICS = ("learning_auc", "validation_score", "score", "tail_score", "seconds")


def _finite(values):
    return [float(value) for value in values if value is not None and math.isfinite(float(value))]


def _mean(values):
    values = _finite(values)
    return statistics.fmean(values) if values else float("nan")


def _median(values):
    values = _finite(values)
    return statistics.median(values) if values else None


def _first_score(row: dict, target: float) -> int | None:
    return next(
        (
            int(point["step"])
            for point in row["history"]
            if math.isfinite(float(point["score"]))
            and float(point["score"]) >= target - 1e-12
        ),
        None,
    )


def _pairwise(runs: list[dict], metric: str) -> list[dict]:
    keyed = {
        (row["task"], row["variant"], row["seed"], row["optimizer"]): row
        for row in runs
    }
    contexts = sorted({
        (row["task"], row["variant"], row["seed"]) for row in runs
    })
    result = []
    for left in OPTIMIZERS:
        for right in OPTIMIZERS:
            if left == right:
                continue
            deltas = []
            for task, variant, seed in contexts:
                a = keyed.get((task, variant, seed, left))
                b = keyed.get((task, variant, seed, right))
                if not a or not b:
                    continue
                av, bv = float(a[metric]), float(b[metric])
                if math.isfinite(av) and math.isfinite(bv):
                    deltas.append(av - bv)
            result.append({
                "left": left,
                "right": right,
                "metric": metric,
                "pairs": len(deltas),
                "wins": sum(delta > 1e-12 for delta in deltas),
                "ties": sum(abs(delta) <= 1e-12 for delta in deltas),
                "mean_delta": _mean(deltas),
                "median_delta": _median(deltas),
            })
    return result


def analyze_neural(payload: dict) -> dict:
    runs = payload["runs"]
    missing = set(OPTIMIZERS) - {row["optimizer"] for row in runs}
    if missing:
        raise ValueError(f"missing optimizer arms: {sorted(missing)}")
    overall = []
    for optimizer in OPTIMIZERS:
        rows = [row for row in runs if row["optimizer"] == optimizer]
        overall.append({
            "optimizer": optimizer,
            "runs": len(rows),
            "failures": sum(row["status"] != "complete" for row in rows),
            **{metric: _mean(row.get(metric) for row in rows) for metric in METRICS},
        })

    grouped = defaultdict(list)
    for row in runs:
        grouped[row["task"], row["variant"]].append(row)
    pairs = []
    robust_counts = {name: 0 for name in OPTIMIZERS}
    for (task, variant), rows in sorted(grouped.items()):
        by_optimizer = {
            name: sorted(
                (row for row in rows if row["optimizer"] == name),
                key=lambda row: row["seed"],
            )
            for name in OPTIMIZERS
        }
        if not all(by_optimizer.values()):
            continue
        means = {
            name: {metric: _mean(row.get(metric) for row in values) for metric in METRICS}
            for name, values in by_optimizer.items()
        }
        ordered = sorted(
            OPTIMIZERS, key=lambda name: means[name]["learning_auc"], reverse=True
        )
        winner, runner_up = ordered[:2]
        per_seed_wins = 0
        reached = 0
        speedups = []
        for seed in sorted({row["seed"] for row in rows}):
            seed_rows = {
                name: next(row for row in by_optimizer[name] if row["seed"] == seed)
                for name in OPTIMIZERS
            }
            winner_auc = seed_rows[winner]["learning_auc"]
            per_seed_wins += winner_auc > max(
                seed_rows[name]["learning_auc"] for name in OPTIMIZERS if name != winner
            )
            target = max(seed_rows[name]["validation_score"] for name in OPTIMIZERS)
            winner_step = _first_score(seed_rows[winner], target)
            comparison_steps = [
                step for name in OPTIMIZERS if name != winner
                if (step := _first_score(seed_rows[name], target)) is not None
            ]
            if winner_step is not None:
                reached += 1
                if comparison_steps:
                    speedups.append(min(comparison_steps) / max(winner_step, 1))
        robust = per_seed_wins >= math.ceil(len(by_optimizer[winner]) / 2)
        if robust:
            robust_counts[winner] += 1
        pairs.append({
            "task": task,
            "variant": variant,
            "kind": rows[0]["kind"],
            "winner": winner,
            "runner_up": runner_up,
            "auc_margin": means[winner]["learning_auc"] - means[runner_up]["learning_auc"],
            "paired_seed_wins": per_seed_wins,
            "robust": robust,
            "winner_target_reached": reached,
            "median_speedup_to_best_shared_target": _median(speedups),
            "means": means,
        })

    architecture = []
    for variant in sorted({row["variant"] for row in runs}):
        for optimizer in OPTIMIZERS:
            rows = [
                row for row in runs
                if row["variant"] == variant and row["optimizer"] == optimizer
            ]
            architecture.append({
                "variant": variant,
                "optimizer": optimizer,
                "learning_auc": _mean(row["learning_auc"] for row in rows),
                "validation_score": _mean(row["validation_score"] for row in rows),
                "score": _mean(row["score"] for row in rows),
            })

    return {
        "configuration": payload["configuration"],
        "overall": overall,
        "architecture": architecture,
        "pairs": pairs,
        "robust_winner_counts": robust_counts,
        "pairwise": {
            metric: _pairwise(runs, metric)
            for metric in ("learning_auc", "validation_score", "score")
        },
    }


def analyze_operator(payload: dict) -> dict:
    grouped = defaultdict(list)
    for row in payload["runs"]:
        grouped[row["protocol"], row["scenario"], row["optimizer"]].append(row)
    summaries = []
    for (protocol, scenario, optimizer), rows in sorted(grouped.items()):
        summaries.append({
            "protocol": protocol,
            "scenario": scenario,
            "optimizer": optimizer,
            "lr": rows[0]["lr"],
            "runs": len(rows),
            "failures": sum(row["status"] != "complete" for row in rows),
            "final_relative_loss": _mean(row["final_relative_loss"] for row in rows),
            "final_operator_error": _mean(row["final_operator_error"] for row in rows),
            "steps_to_loss_1e-2": _median(row["steps_to_loss_1e-2"] for row in rows),
            "steps_to_loss_1e-4": _median(row["steps_to_loss_1e-4"] for row in rows),
            "steps_to_loss_1e-8": _median(row["steps_to_loss_1e-8"] for row in rows),
            "steps_to_operator_1e-2": _median(row["steps_to_operator_1e-2"] for row in rows),
        })
    return {
        "configuration": payload["configuration"],
        "protocols": payload["protocols"],
        "summaries": summaries,
        "exact_one_step_certificates": payload["exact_one_step_certificates"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("neural", type=Path)
    parser.add_argument("operator", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    neural_payload = json.loads(args.neural.read_text())
    operator_payload = json.loads(args.operator.read_text())
    result = {
        "neural_source": str(args.neural),
        "operator_source": str(args.operator),
        "neural": analyze_neural(neural_payload),
        "operator": analyze_operator(operator_payload),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({
        "overall": result["neural"]["overall"],
        "robust_winner_counts": result["neural"]["robust_winner_counts"],
        "operator": result["operator"]["summaries"],
    }, indent=2))


if __name__ == "__main__":
    main()
