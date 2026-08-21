#!/usr/bin/env python3
"""Problem battery for optimizer-transported gradient zonotope descent."""
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
import torch.nn.functional as F
from torch.func import functional_call

from ML_experiment.metrics import evaluate, jacobian_variability, tail_metrics
from ML_experiment.models import parameter_count
from ML_experiment.odd_context_hybrids import make_hybrid
from ML_experiment.run_benchmark import auc, threshold
from ML_experiment.run_frame_refinement import make_probe
from ML_experiment.run_odd_context_battery import train as train_baseline
from ML_experiment.sparse_sine_gradient_zonotope import (
    _flatten_gradients,
    _named_parameters,
    commit_transition_mixture,
    gradient_covariance_frame,
    preview_adamw_transition,
)
from ML_experiment.tasks import TASK_BUILDERS


torch.set_num_threads(8)


class ProjectiveWitnessAtlas:
    """Interleaved witness folds along a data-derived principal projection."""

    def __init__(self, task, folds: int = 4, cells: int = 16):
        self.task = task
        self.x = task.x_train
        self.y = task.y_train
        self.folds = int(folds)
        self.cells = int(cells)
        centered = self.x - self.x.mean(0, keepdim=True)
        covariance = centered.T @ centered / max(1, len(centered))
        direction = torch.linalg.eigh(covariance).eigenvectors[:, -1]
        score = centered @ direction
        self.score = score
        self.fold = torch.empty(len(self.x), dtype=torch.long)
        self.cell = torch.empty(len(self.x), dtype=torch.long)

        groups = (
            [torch.where(self.y == label)[0] for label in torch.unique(self.y)]
            if task.kind == "classification"
            else [torch.arange(len(self.x))]
        )
        for group_index, group in enumerate(groups):
            order = group[torch.argsort(score[group])]
            rank = torch.arange(len(order))
            self.fold[order] = (rank + 3 * group_index) % self.folds
            self.cell[order] = torch.clamp(
                rank * self.cells // max(1, len(order)), 0, self.cells - 1
            )
        self.train_pool = [
            torch.where(self.fold != held_out)[0]
            for held_out in range(self.folds)
        ]
        self.witness_pool = [
            torch.where(self.fold == held_out)[0]
            for held_out in range(self.folds)
        ]

    def sample(self, batch: int, held_out: int, generator: torch.Generator):
        pool = self.train_pool[held_out]
        return pool[torch.randint(len(pool), (batch,), generator=generator)]

    def witness(self, held_out: int):
        return self.witness_pool[held_out]


def _functional_prediction(model, parameters, x):
    return functional_call(
        model, {**dict(model.named_buffers()), **parameters}, (x,)
    )


def witness_objective(model, parameters, atlas, held_out, worst_weight):
    index = atlas.witness(held_out)
    output = _functional_prediction(model, parameters, atlas.x[index])
    target = atlas.y[index]
    if atlas.task.kind == "classification":
        point_loss = F.cross_entropy(output, target, reduction="none")
        local = []
        for label in torch.unique(target):
            selected = target == label
            local.append(point_loss[selected].mean())
    else:
        point_loss = (output - target).square().mean(-1)
        local = []
        for cell in range(atlas.cells):
            selected = atlas.cell[index] == cell
            if selected.any():
                local.append(point_loss[selected].mean())
    local = torch.stack(local)
    top_count = max(1, math.ceil(0.25 * len(local)))
    return point_loss.mean() + worst_weight * local.topk(top_count).values.mean()


@torch.no_grad()
def transported_step(
    model,
    optimizer,
    gradients,
    atlas,
    held_out,
    *,
    rank,
    temperature,
    worst_weight,
):
    named = _named_parameters(model)
    candidate_parameters, candidate_states, displacements = [], [], []
    for gradient in gradients:
        parameters, states = preview_adamw_transition(model, optimizer, gradient)
        candidate_parameters.append(parameters)
        candidate_states.append(states)
        displacements.append(torch.cat([
            (parameters[name] - parameter).flatten()
            for name, parameter in named
        ]))
    displacements = torch.stack(displacements)
    common, directions, retained = gradient_covariance_frame(displacements, rank)
    denominator = directions.square().sum(-1).clamp_min(1e-12)
    coordinates = torch.einsum(
        "bp,rp->br", displacements - common, directions
    ) / denominator
    projected = common + torch.einsum("br,rp->bp", coordinates, directions)

    projected_parameters, scores = [], []
    for displacement in projected:
        values, offset = {}, 0
        for name, parameter in named:
            count = parameter.numel()
            values[name] = parameter + displacement[offset:offset + count].view_as(parameter)
            offset += count
        projected_parameters.append(values)
        scores.append(witness_objective(
            model, values, atlas, held_out, worst_weight
        ))
    scores = torch.stack(scores)
    relative = (scores - scores.mean()) / scores.abs().mean().clamp_min(1e-8)
    weight = torch.softmax(-relative / max(temperature, 1e-8), dim=0)
    commit_transition_mixture(
        model, optimizer, projected_parameters, candidate_states, weight
    )
    coefficient = torch.einsum("b,br->r", weight, coordinates)
    entropy = -(weight * weight.clamp_min(1e-12).log()).sum()
    selected = torch.einsum("b,bp->p", weight, projected)
    return {
        "retained_covariance": float(retained),
        "coefficient_norm": float(coefficient.norm()),
        "selection_entropy": float(entropy),
        "shell_to_common": float((selected - common).norm() / common.norm().clamp_min(1e-12)),
    }


def train_transport(
    task,
    width,
    seed,
    steps,
    batch,
    lr,
    evaluate_every,
    *,
    gradient_samples,
    covariance_rank,
    temperature,
    witness_period,
    worst_weight,
):
    torch.manual_seed(41000 + seed)
    model = make_hybrid("self_context", task.input_dim, task.output_dim, width)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    atlas = ProjectiveWitnessAtlas(task)
    named = _named_parameters(model)
    history, best = [], None
    diagnostics = []
    started = time.perf_counter()
    for step in range(steps):
        held_out = (step // witness_period) % atlas.folds
        gradients = []
        model.train()
        for sample in range(gradient_samples):
            generator = torch.Generator().manual_seed(
                52000 + 1000003 * seed + 1009 * step + 37 * sample
            )
            index = atlas.sample(batch, held_out, generator)
            optimizer.zero_grad(set_to_none=True)
            output = model(task.x_train[index])
            loss = (
                F.cross_entropy(output, task.y_train[index])
                if task.kind == "classification"
                else F.mse_loss(output, task.y_train[index])
            )
            loss.backward()
            gradient = _flatten_gradients(named)
            norm = gradient.norm().clamp_min(1e-12)
            gradient = gradient * torch.clamp(torch.tensor(10.0) / norm, max=1.0)
            gradients.append(gradient)
        diagnostic = transported_step(
            model, optimizer, torch.stack(gradients), atlas, held_out,
            rank=covariance_rank, temperature=temperature,
            worst_weight=worst_weight,
        )
        diagnostics.append(diagnostic)
        accepted = step + 1
        if step == 0 or accepted % evaluate_every == 0 or accepted == steps:
            metrics = evaluate(model, task, task.x_val, task.y_val)
            history.append({"step": accepted, **metrics, **diagnostic})
            if best is None or metrics["score"] > best[0]:
                best = (metrics["score"], copy.deepcopy(model.state_dict()), accepted)
    seconds = time.perf_counter() - started
    model.load_state_dict(best[1])
    summary = {
        key: sum(row[key] for row in diagnostics) / len(diagnostics)
        for key in diagnostics[0]
    }
    return model, history, seconds, best[2], summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/tmp/gradient_zonotope_battery"))
    parser.add_argument(
        "--tasks", default=",".join(
            name for name in TASK_BUILDERS if name != "sparse_sine_1d"
        )
    )
    parser.add_argument("--variants", default="self_context,transport_cold,transport_warm")
    parser.add_argument("--width", type=int, default=38)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--gradient-samples", type=int, default=3)
    parser.add_argument("--covariance-rank", type=int, default=2)
    parser.add_argument("--cold-temperature", type=float, default=1e-6)
    parser.add_argument("--warm-temperature", type=float, default=1e-4)
    parser.add_argument("--witness-period", type=int, default=50)
    parser.add_argument("--worst-weight", type=float, default=0.10)
    parser.add_argument("--grid", type=int, default=71)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    partial = args.out / "results.partial.json"
    payload = json.loads(partial.read_text()) if args.resume and partial.exists() else {"runs": [], "probes": []}
    done = {(row["task"], row["variant"], row["seed"]) for row in payload["runs"]}

    for task_name in args.tasks.split(","):
        for seed in range(args.seeds):
            task = TASK_BUILDERS[task_name](seed)
            for variant in args.variants.split(","):
                if (task_name, variant, seed) in done:
                    continue
                print(f"START task={task_name} variant={variant} seed={seed}", flush=True)
                if variant == "self_context":
                    model, history, seconds, best_step = train_baseline(
                        "self_context", task, args.width, seed, args.steps,
                        args.batch, args.lr, args.eval_every,
                    )
                    summary = {}
                    gradient_evaluations = args.steps
                else:
                    temperature = (
                        args.cold_temperature if variant == "transport_cold"
                        else args.warm_temperature
                    )
                    model, history, seconds, best_step, summary = train_transport(
                        task, args.width, seed, args.steps, args.batch, args.lr,
                        args.eval_every, gradient_samples=args.gradient_samples,
                        covariance_rank=args.covariance_rank,
                        temperature=temperature,
                        witness_period=args.witness_period,
                        worst_weight=args.worst_weight,
                    )
                    gradient_evaluations = args.steps * args.gradient_samples
                test = evaluate(model, task)
                tails = tail_metrics(model, task)
                variability, jacobian_rank = jacobian_variability(model, task.x_val)
                row = {
                    "task": task_name,
                    "kind": task.kind,
                    "input_dim": task.input_dim,
                    "output_dim": task.output_dim,
                    "variant": variant,
                    "seed": seed,
                    "parameters": parameter_count(model),
                    "steps": args.steps,
                    "gradient_evaluations": gradient_evaluations,
                    "seconds": seconds,
                    "best_step": best_step,
                    "learning_auc": auc(history, args.steps),
                    "steps_to_80": threshold(history, 0.8),
                    "steps_to_90": threshold(history, 0.9),
                    **test,
                    **tails,
                    "jacobian_variability": variability,
                    "jacobian_change_rank": jacobian_rank,
                    **summary,
                    "history": history,
                }
                payload["runs"].append(row)
                if seed == 0:
                    payload["probes"].append({
                        "task": task_name,
                        "variant": variant,
                        **make_probe(task, model, args.grid),
                    })
                partial.write_text(json.dumps(payload))
                print(json.dumps({
                    "task": task_name,
                    "variant": variant,
                    "score": round(row["score"], 6),
                    "tail_score": None if row.get("tail_score") is None else round(row["tail_score"], 6),
                    "seconds": round(seconds, 3),
                }), flush=True)
    result = {"configuration": {**vars(args), "out": str(args.out)}, **payload}
    (args.out / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"complete": True, "runs": len(payload["runs"])}))


if __name__ == "__main__":
    main()
