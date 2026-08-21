#!/usr/bin/env python3
"""Screen odd third-order grafts on the 16-D eight-plane spiral.

The benchmark is deliberately narrow.  It asks whether the degree of freedom
that solves the antipodal N-D spiral can be acquired inside self-context/CFF,
or only as an independent shortcut.  No phase, radius, neighbor, rotation, or
tail coordinate is supplied to a model.
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
import torch.nn.functional as F

from ML_experiment.models import parameter_count
from ML_experiment.odd_context_hybrids import (
    CONTROLLED_VARIANTS,
    FACTOR_VARIANTS,
    ContextualHiddenGraft,
    ParallelGraft,
    VARIANTS,
    make_hybrid,
)
from ML_experiment.run_nd_spiral_wall import visual_probe
from ML_experiment.tasks import TASK_BUILDERS


LABELS = {
    "self_context": "Relational self-context",
    "cff": "Relational continuous frame flow",
    "shallow_odd_cubic": "Shallow odd-cubic control",
    "self_parallel_degree2": "Self-context + parallel degree-2 odd channel",
    "self_parallel_angular": "Self-context + parallel angular odd channel",
    "self_contextual_angular": "Self-context with chart-coupled odd bridge",
    "cff_contextual_angular": "CFF with chart-coupled odd bridge",
    "self_ancient_polynorm": "Self-context + old square/cubic norm branch",
    "self_capacity_match_rank8": "Plain self-context, rank-8 capacity matched",
    "self_capacity_match_full": "Plain self-context, full-bridge capacity matched",
    "self_contextual_angular_rank4": "Self-context + rank-4 chart odd bridge",
    "self_contextual_angular_rank8": "Self-context + rank-8 chart odd bridge",
    "self_contextual_tied_rank8": "Self-context + tied rank-8 chart odd bridge",
    "self_contextual_modulated_rank8": "Self-context + modulated rank-8 odd bridge",
    "self_contextual_antithetic_rank8": "Self-context + parity-projected rank-8 bridge",
    "self_contextual_full_factor_rms": "Full chart bridge + factor RMS",
    "self_contextual_full_row_unit": "Full chart bridge + unit rays",
    "self_contextual_full_tight": "Full chart bridge + tight frames",
    "self_contextual_full_tight_rms": "Full chart bridge + tight frames + factor RMS",
    "self_contextual_rank8_tight": "Rank-8 chart bridge + tight frames",
    "self_contextual_full_learned_cone": "Full chart bridge + learned cone",
    "self_contextual_full_weight_norm": "Full chart bridge + direction/gain coordinates",
}

torch.set_num_threads(8)


@torch.no_grad()
def accuracy(model, x, y):
    return float((model(x).argmax(1) == y).float().mean())


def _gradient_groups(model):
    totals: dict[str, float] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        group = name.split(".", 1)[0]
        totals[group] = totals.get(group, 0.0) + float(parameter.grad.square().sum())
    return {name: math.sqrt(value) for name, value in totals.items()}


def _margin_gradient(model, x):
    points = x.detach().clone().requires_grad_(True)
    logits = model(points)
    margin = logits[:, 1] - logits[:, 0]
    return torch.autograd.grad(margin.sum(), points)[0].detach()


def gradient_geometry(model, x):
    """Describe the learned derivative field without using it as an objective."""
    x = x[:128]
    positive = _margin_gradient(model, x)
    negative = _margin_gradient(model, -x)
    centered = positive - positive.mean(0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    energy = singular.square()
    distribution = energy / energy.sum().clamp_min(1e-12)
    effective_rank = torch.exp(
        -(distribution * torch.log(distribution.clamp_min(1e-12))).sum()
    )

    generator = torch.Generator().manual_seed(8675309)
    direction = F.normalize(torch.randn(x.shape, generator=generator), dim=-1)
    epsilon = 0.03 * x.square().mean().sqrt().clamp_min(1e-3)
    plus = _margin_gradient(model, x + epsilon * direction)
    minus = _margin_gradient(model, x - epsilon * direction)
    curvature = (plus - minus) / (2 * epsilon)
    denominator = positive.norm(dim=1).mean().clamp_min(1e-8)

    radial_alignment = F.cosine_similarity(positive, x, dim=1).abs().mean()
    return {
        "gradient_effective_rank": float(effective_rank),
        "gradient_top_energy": float(distribution[0]),
        # The derivative of an odd logit margin is even.  Lower is therefore
        # the task-aligned parity signature, not a generic quality measure.
        "gradient_even_error": float(
            (positive - negative).norm(dim=1).mean()
            / (0.5 * (positive.norm(dim=1) + negative.norm(dim=1))).mean().clamp_min(1e-8)
        ),
        "gradient_radial_alignment": float(radial_alignment),
        "directional_curvature_ratio": float(curvature.norm(dim=1).mean() / denominator),
    }


@torch.no_grad()
def component_diagnostics(model, task):
    x, y = task.x_val[:512], task.y_val[:512]
    result = {}
    if hasattr(model, "branch_scale"):
        result["branch_scale"] = float(F.softplus(model.branch_scale))
    if (hasattr(model, "bridge") and
            getattr(model.bridge, "cone_logits", None) is not None):
        cone = torch.sigmoid(model.bridge.cone_logits)
        result.update({
            "cone_source_a": float(cone[0]),
            "cone_source_b": float(cone[1]),
            "cone_chart": float(cone[2]),
        })
    if isinstance(model, ParallelGraft):
        parent, branch = model.components(x)
        result.update({
            "parent_only_accuracy": float((parent.argmax(1) == y).float().mean()),
            "branch_only_accuracy": float((branch.argmax(1) == y).float().mean()),
            "branch_to_parent_rms": float(
                branch.square().mean().sqrt() / parent.square().mean().sqrt().clamp_min(1e-8)
            ),
        })
    elif isinstance(model, ContextualHiddenGraft):
        _, chart, branch = model.hidden_components(x)
        parent = model.parent(x)
        result.update({
            "parent_only_accuracy": float((parent.argmax(1) == y).float().mean()),
            "branch_to_parent_rms": float(
                branch.square().mean().sqrt() / chart.square().mean().sqrt().clamp_min(1e-8)
            ),
        })
    return result


@torch.no_grad()
def endpoint_diagnostics(model, task):
    prediction = model(task.x_test).argmax(1)
    probabilities = model(task.x_test).softmax(1)[:, 1]
    flipped = model(-task.x_test).softmax(1)[:, 1]
    return {
        "validation_accuracy": accuracy(model, task.x_val, task.y_val),
        "tail_accuracy": float((prediction == task.y_test).float().mean()),
        "tail_class_0": float((prediction[task.y_test == 0] == 0).float().mean()),
        "tail_class_1": float((prediction[task.y_test == 1] == 1).float().mean()),
        "antipodal_error": float((probabilities + flipped - 1).abs().mean()),
        "tail_bins": [accuracy(model, x, y) for x, y in zip(task.tail_x, task.tail_y)],
    }


def train(name, task, width, seed, steps, batch, lr, evaluate_every):
    torch.manual_seed(31000 + seed)
    model = make_hybrid(name, task.input_dim, task.output_dim, width)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(32000 + seed)
    history = []
    best_state, best_score, best_step = None, -1.0, 0
    started = time.perf_counter()
    for step in range(1, steps + 1):
        index = torch.randint(len(task.x_train), (batch,), generator=generator)
        x, y = task.x_train[index], task.y_train[index]
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        gradient_groups = _gradient_groups(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        if step == 1 or step % evaluate_every == 0 or step == steps:
            score = accuracy(model, task.x_val, task.y_val)
            history.append({
                "step": step,
                "accuracy": score,
                "loss": float(loss.detach()),
                "gradient_groups": gradient_groups,
            })
            if score > best_score:
                best_state = copy.deepcopy(model.state_dict())
                best_score, best_step = score, step
    seconds = time.perf_counter() - started
    model.load_state_dict(best_state)
    row = {
        "configuration": name,
        "label": LABELS.get(
            name,
            name.replace("self_contextual_tied_rank", "Tied chart bridge, rank "),
        ),
        "seed": seed,
        "parameters": parameter_count(model),
        "seconds": seconds,
        "best_step": best_step,
        "history": history,
        **endpoint_diagnostics(model, task),
        **component_diagnostics(model, task),
        **gradient_geometry(model, task.x_val),
    }
    return model, row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/tmp/odd_context_hybrids"))
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--models", default=",".join(VARIANTS))
    parser.add_argument("--controlled", action="store_true",
                        help="run the parameter-controlled chart-bridge screen")
    parser.add_argument("--factors", action="store_true",
                        help="run the factor-atlas conditioning screen")
    args = parser.parse_args()
    names = (list(CONTROLLED_VARIANTS) if args.controlled else
             list(FACTOR_VARIANTS) if args.factors else args.models.split(","))
    args.out.mkdir(parents=True, exist_ok=True)
    rows, probes = [], []
    for seed in range(args.seeds):
        task = TASK_BUILDERS["nd_spiral_high_rank"](seed)
        for name in names:
            print(f"START seed={seed} model={name}", flush=True)
            model, row = train(
                name, task, args.width, seed, args.steps, args.batch, args.lr,
                args.eval_every,
            )
            rows.append(row)
            if seed == 0:
                probes.append({
                    "configuration": name,
                    **visual_probe(model, task, seed),
                })
            (args.out / "results.partial.json").write_text(
                json.dumps({"runs": rows}, indent=2)
            )
            (args.out / "probes.partial.json").write_text(json.dumps({"probes": probes}))
            print(json.dumps({
                key: row[key] for key in (
                    "configuration", "seed", "validation_accuracy", "tail_accuracy",
                    "parameters", "seconds", "gradient_effective_rank",
                    "gradient_even_error",
                )
            }), flush=True)
    configuration = {**vars(args), "out": str(args.out), "models": names}
    (args.out / "results.json").write_text(json.dumps({
        "configuration": configuration,
        "runs": rows,
    }, indent=2))
    (args.out / "probes.json").write_text(json.dumps({
        "configuration": configuration,
        "probes": probes,
    }))
    print(json.dumps({"complete": True, "runs": len(rows), "out": str(args.out)}))


if __name__ == "__main__":
    main()
