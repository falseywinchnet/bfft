#!/usr/bin/env python3
"""Structured held-out branch selection for the progressively sparse sine.

This is deliberately one model, not an inference ensemble.  At each macro-step
the current optimizer state is copied into a few short candidate trajectories.
Every candidate trains on a different jittered, support-stratified view of the
same non-witness observations.  A common interleaved witness fold selects the
trajectory to retain; the other candidates are discarded.

Witness targets never participate in a candidate gradient.  They are an outer
supervised signal, so this is honestly a form of repeated cross-validation (or
"in-context cheating"), not unsupervised learning.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from ML_experiment.models import parameter_count
from ML_experiment.sparse_sine_study import (
    AcquisitionData,
    SupportWhitened,
    curve_metrics,
    curve_probe,
    make_model,
    train,
)
from ML_experiment.tasks import sparse_sine_1d


torch.set_num_threads(8)


class InterleavedWitnessAtlas:
    """Continuous-support cells with locally interleaved witness folds.

    Cells are equal in coordinate-support mass, not equal in observation count.
    Within each cell, consecutive observations go to different folds.  Thus a
    fold spans the whole interval while never becoming a single contiguous
    validation region.
    """

    def __init__(self, data: AcquisitionData, *, cells: int = 16, folds: int = 4):
        if cells < 2 or folds < 2:
            raise ValueError("cells and folds must both be at least two")
        self.data = data
        self.cells = int(cells)
        self.folds = int(folds)
        count = len(data.x)
        cell = torch.empty(count, dtype=torch.long)
        fold = torch.empty(count, dtype=torch.long)

        # The midpoint support CDF is a coordinate-like variable even under a
        # highly nonuniform empirical measure.
        ordered_probability = data.support_probability[data.support_order]
        support_midpoint = data.support_cdf - 0.5 * ordered_probability
        ordered_cell = torch.clamp(
            (support_midpoint * self.cells).long(), 0, self.cells - 1
        )
        cell[data.support_order] = ordered_cell
        for cell_index in range(self.cells):
            position = torch.where(ordered_cell == cell_index)[0]
            original = data.support_order[position]
            # A cell-dependent rotation prevents the same geometric phase from
            # always receiving the same fold while retaining local interleaving.
            local_fold = (
                torch.arange(len(original)) + 3 * cell_index
            ) % self.folds
            fold[original] = local_fold

        self.cell = cell
        self.fold = fold
        self._train_order: list[torch.Tensor] = []
        self._train_cdf: list[torch.Tensor] = []
        self._witness: list[torch.Tensor] = []
        for held_out in range(self.folds):
            train_order = data.support_order[fold[data.support_order] != held_out]
            probability = data.support_probability[train_order]
            probability = probability / probability.sum().clamp_min(1e-12)
            self._train_order.append(train_order)
            self._train_cdf.append(probability.cumsum(0))
            self._witness.append(torch.where(fold == held_out)[0])

    def witness(self, held_out: int) -> torch.Tensor:
        return self._witness[held_out]

    def stratified_train(
        self,
        batch: int,
        held_out: int,
        generator: torch.Generator,
        *,
        phase: float = 0.0,
    ) -> torch.Tensor:
        """A jittered lattice over support, excluding one witness fold.

        ``phase`` rotates the lattice. Candidate pools therefore have the same
        support measure but see meaningfully different local observations.
        """
        jitter = torch.rand(batch, generator=generator)
        coordinate = (
            torch.arange(batch, dtype=self.data.x.dtype) + jitter + phase
        ) / batch
        coordinate = coordinate.remainder(1.0)
        coordinate, _ = coordinate.sort()
        position = torch.searchsorted(
            self._train_cdf[held_out], coordinate
        ).clamp_max(len(self._train_order[held_out]) - 1)
        return self._train_order[held_out][position]

    @torch.no_grad()
    def score(self, model: nn.Module, held_out: int, worst_weight: float = 0.25):
        """Support-weighted witness error plus a worst-cell anti-collapse term."""
        index = self.witness(held_out)
        error = (model(self.data.x[index]) - self.data.y[index]).square().mean(-1)
        weight = self.data.voronoi[index, 0]
        mean = (weight * error).sum() / weight.sum().clamp_min(1e-12)
        cell_losses = []
        for cell_index in range(self.cells):
            selected = self.cell[index] == cell_index
            if selected.any():
                local_weight = weight[selected]
                cell_losses.append(
                    (local_weight * error[selected]).sum()
                    / local_weight.sum().clamp_min(1e-12)
                )
        cells = torch.stack(cell_losses)
        # The smooth top quartile is less brittle than a single maximum point.
        top_count = max(1, math.ceil(0.25 * len(cells)))
        worst = cells.topk(top_count).values.mean()
        return {
            "score": float(mean + worst_weight * worst),
            "mean": float(mean),
            "worst_quartile": float(worst),
        }


def _optimizer(model: nn.Module, lr: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)


def train_witness_descent(
    task,
    *,
    model_name: str,
    width: int,
    seed: int,
    macro_steps: int,
    branches: int,
    micro_steps: int,
    batch: int,
    lr: float,
    cells: int,
    folds: int,
    worst_weight: float,
    evaluate_every: int,
    witness_period: int = 50,
):
    torch.manual_seed(61000 + seed)
    data = AcquisitionData(task)
    model = SupportWhitened(make_model(model_name, width), data)
    optimizer = _optimizer(model, lr)
    atlas = InterleavedWitnessAtlas(data, cells=cells, folds=folds)
    history = []
    branch_wins = [0 for _ in range(branches)]
    started = time.perf_counter()

    for macro in range(macro_steps):
        if witness_period <= 0:
            held_out = seed % folds
        else:
            held_out = (macro // witness_period) % folds
        candidate_models = []
        candidate_optimizers = []
        candidate_scores = []
        optimizer_state = copy.deepcopy(optimizer.state_dict())

        for branch in range(branches):
            candidate = copy.deepcopy(model)
            candidate_optimizer = _optimizer(candidate, lr)
            candidate_optimizer.load_state_dict(copy.deepcopy(optimizer_state))
            generator = torch.Generator().manual_seed(
                71000 + 1000003 * seed + 1009 * macro + 37 * branch
            )
            # Irrationally spaced rotations prevent candidate views from
            # repeatedly sharing the same local sample phase.
            phase = (branch * 0.6180339887498949 + macro * 0.4142135623730950) % 1.0
            candidate.train()
            for _ in range(micro_steps):
                index = atlas.stratified_train(
                    batch, held_out, generator, phase=phase
                )
                candidate_optimizer.zero_grad(set_to_none=True)
                loss = F.mse_loss(candidate(data.x[index]), data.y[index])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(candidate.parameters(), 10.0)
                candidate_optimizer.step()
                phase = (phase + 0.3819660112501051) % 1.0
            candidate.eval()
            candidate_models.append(candidate)
            candidate_optimizers.append(candidate_optimizer)
            candidate_scores.append(
                atlas.score(candidate, held_out, worst_weight=worst_weight)
            )

        winner = min(
            range(branches), key=lambda index: candidate_scores[index]["score"]
        )
        branch_wins[winner] += 1
        model = candidate_models[winner]
        optimizer = candidate_optimizers[winner]

        step = (macro + 1) * micro_steps
        if macro == 0 or (macro + 1) % evaluate_every == 0 or macro + 1 == macro_steps:
            metrics = curve_metrics(model, task)
            scores = [row["score"] for row in candidate_scores]
            history.append({
                "macro_step": macro + 1,
                "accepted_steps": step,
                "gradient_evaluations": (macro + 1) * branches * micro_steps,
                "held_out_fold": held_out,
                "winner": winner,
                "candidate_score_min": min(scores),
                "candidate_score_max": max(scores),
                "candidate_score_spread": max(scores) - min(scores),
                "r2": metrics["r2"],
                "sparse_tail_r2": metrics["sparse_tail_r2"],
                "minimum_segment_r2": metrics["minimum_segment_r2"],
                "extrapolation_r2": metrics["extrapolation_r2"],
            })

    return model, history, branch_wins, time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/tmp/sparse_sine_witness"))
    parser.add_argument("--model", default="self_context")
    parser.add_argument("--width", type=int, default=38)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--macro-steps", type=int, default=1000)
    parser.add_argument("--branches", type=int, default=3)
    parser.add_argument("--micro-steps", type=int, default=1)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--cells", type=int, default=16)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--worst-weight", type=float, default=0.10)
    parser.add_argument(
        "--witness-period", type=int, default=50,
        help="Macro-steps per witness fold; zero keeps one fold fixed.",
    )
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--skip-baselines", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    runs = []
    accepted_steps = args.macro_steps * args.micro_steps
    gradient_budget = accepted_steps * args.branches
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        task = sparse_sine_1d(seed)
        if not args.skip_baselines:
            for label, steps in (
                ("baseline_path_matched", accepted_steps),
                ("baseline_compute_matched", gradient_budget),
            ):
                print(f"START {label} seed={seed} steps={steps}", flush=True)
                model, history, seconds = train(
                    args.model, "stratified_whiten", task,
                    width=args.width, seed=seed, steps=steps,
                    batch=args.batch, lr=args.lr,
                    evaluate_every=max(1, steps // 50),
                )
                runs.append({
                    "method": label,
                    "seed": seed,
                    "parameters": parameter_count(model),
                    "accepted_steps": steps,
                    "gradient_evaluations": steps,
                    "seconds": seconds,
                    **curve_metrics(model, task),
                    "history": history,
                    "probe": curve_probe(model, task),
                })

        print(
            f"START zonotopic_witness seed={seed} "
            f"accepted={accepted_steps} gradients={gradient_budget}",
            flush=True,
        )
        model, history, wins, seconds = train_witness_descent(
            task, model_name=args.model, width=args.width, seed=seed,
            macro_steps=args.macro_steps, branches=args.branches,
            micro_steps=args.micro_steps, batch=args.batch, lr=args.lr,
            cells=args.cells, folds=args.folds,
            worst_weight=args.worst_weight,
            evaluate_every=args.eval_every,
            witness_period=args.witness_period,
        )
        metrics = curve_metrics(model, task)
        row = {
            "method": "zonotopic_witness",
            "seed": seed,
            "parameters": parameter_count(model),
            "accepted_steps": accepted_steps,
            "gradient_evaluations": gradient_budget,
            "seconds": seconds,
            "branch_wins": wins,
            **metrics,
            "history": history,
            "probe": curve_probe(model, task),
        }
        runs.append(row)
        print(json.dumps({
            "seed": seed,
            "r2": round(row["r2"], 6),
            "sparse_tail_r2": round(row["sparse_tail_r2"], 6),
            "minimum_segment_r2": round(row["minimum_segment_r2"], 6),
            "extrapolation_r2": round(row["extrapolation_r2"], 6),
            "seconds": round(seconds, 3),
            "branch_wins": wins,
        }), flush=True)

    payload = {
        "configuration": {**vars(args), "out": str(args.out)},
        "interpretation": {
            "candidate_gradient_uses_witness_labels": False,
            "candidate_selection_uses_witness_labels": True,
            "inference_is_ensemble": False,
            "accepted_steps": accepted_steps,
            "gradient_budget": gradient_budget,
        },
        "runs": runs,
    }
    (args.out / "results.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({"complete": True, "runs": len(runs)}))


if __name__ == "__main__":
    main()
