#!/usr/bin/env python3
"""Observed-graph Hermite acquisition for the pivotal geometry tasks.

This study deliberately does not expose polar coordinates, curve parameters,
analytic targets, or test points to the model.  It constructs three views of
the observed training set:

* empirical points;
* points reweighted by local target-boundary/jet curvature;
* cubic Hermite points transported between compatible observations.

The views can either be averaged in one ordinary gradient or offered as the
three candidate updates of optimizer-transported zonotope descent.  A held-out
witness fold is used only by the latter to choose a continuous mixture.
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

from ML_experiment.metrics import evaluate, jacobian_variability, tail_metrics
from ML_experiment.models import parameter_count
from ML_experiment.odd_context_hybrids import make_hybrid
from ML_experiment.run_benchmark import auc, threshold
from ML_experiment.run_frame_refinement import make_probe
from ML_experiment.run_gradient_zonotope_battery import (
    ProjectiveWitnessAtlas,
    transported_step,
)
from ML_experiment.run_odd_context_battery import mechanism_diagnostics
from ML_experiment.sparse_sine_gradient_zonotope import (
    _flatten_gradients,
    _named_parameters,
)
from ML_experiment.tasks import TASK_BUILDERS


torch.set_num_threads(8)

PIVOTAL_TASKS = ("radial_stripes", "spiral", "complex_spiral_3d")
BASE_SELF = "self_context"
OPERATOR = "self_contextual_operator_sphere_global_r2"
CURVATURE = "self_context_curvature_response"
CURVATURE2 = "self_context_curvature_response2"
CURVATURE_BOUNDED = "self_context_curvature_response_bounded"
CURVATURE_GEOMETRIC = "self_context_curvature_response_geometric"
CURVATURE_SIGNED = "self_context_curvature_response_signed"
CURVATURE_SIGNED_GEOMETRIC = "self_context_curvature_response_signed_geometric"
SELECTION_CURVATURE = "self_context_selection_curvature"
LEARNED_CONE = "self_contextual_full_learned_cone"
NESTED_OPERATOR = "self_contextual_nested_operator_r2"


def _loss(task, output, target):
    return (F.cross_entropy(output, target) if task.kind == "classification"
            else F.mse_loss(output, target))


def _nonuniform_jets(x: torch.Tensor, y: torch.Tensor):
    """First and second output derivatives on sorted scalar observations."""
    order = torch.argsort(x[:, 0])
    xs, ys = x[order, 0], y[order]
    first, second = torch.empty_like(ys), torch.empty_like(ys)
    h0 = (xs[1:-1] - xs[:-2]).clamp_min(1e-8)
    h1 = (xs[2:] - xs[1:-1]).clamp_min(1e-8)
    first[1:-1] = (
        -h1[:, None] / (h0 * (h0 + h1))[:, None] * ys[:-2]
        + (h1 - h0)[:, None] / (h0 * h1)[:, None] * ys[1:-1]
        + h0[:, None] / (h1 * (h0 + h1))[:, None] * ys[2:]
    )
    second[1:-1] = 2.0 * (
        ys[:-2] / (h0 * (h0 + h1))[:, None]
        - ys[1:-1] / (h0 * h1)[:, None]
        + ys[2:] / (h1 * (h0 + h1))[:, None]
    )
    first[0] = (ys[1] - ys[0]) / (xs[1] - xs[0]).clamp_min(1e-8)
    first[-1] = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2]).clamp_min(1e-8)
    second[0], second[-1] = second[1], second[-2]
    return xs[:, None], ys, first, second


def _robust_unit_scale(value: torch.Tensor) -> torch.Tensor:
    positive = value[value.isfinite() & (value > 0)]
    scale = positive.median() if len(positive) else value.new_tensor(1.0)
    return (value / scale.clamp_min(1e-8)).clamp(0, 8)


class ObservedHermiteAtlas:
    """A task-agnostic local chart made exclusively from observed pairs."""

    def __init__(self, task, indices: torch.Tensor | None = None,
                 *, neighbours: int = 12, pool_size: int = 16384,
                 seed: int = 0):
        self.task = task
        self.x = task.x_train if indices is None else task.x_train[indices]
        self.y = task.y_train if indices is None else task.y_train[indices]
        self.neighbours = int(neighbours)
        self.generator = torch.Generator().manual_seed(seed)
        if task.kind == "regression" and task.input_dim == 1:
            self._build_scalar_regression(pool_size)
        elif task.kind == "classification":
            self._build_classification(pool_size)
        else:
            raise ValueError(
                "Hermite acquisition currently supports scalar-input regression "
                "or classification"
            )

    def _build_scalar_regression(self, pool_size: int):
        xs, ys, first, second = _nonuniform_jets(self.x, self.y)
        interval = (xs[1:, 0] - xs[:-1, 0]).clamp_min(1e-8)
        curvature = 0.5 * (second[:-1].norm(dim=1) + second[1:].norm(dim=1))
        structural = interval * (1.0 + _robust_unit_scale(curvature))
        structural = structural / structural.sum()
        interval_probability = interval / interval.sum()
        self.real_probability = torch.ones(len(self.x)) / len(self.x)
        self.geometry_diagnostics = {
            "mean_jet_curvature": float(curvature.mean()),
            "maximum_jet_curvature": float(curvature.max()),
        }

        def generate(probability, hermite: bool):
            edge = torch.multinomial(
                probability, pool_size, replacement=True,
                generator=self.generator,
            )
            alpha = torch.rand((pool_size, 1), generator=self.generator)
            x0, x1 = xs[edge], xs[edge + 1]
            y0, y1 = ys[edge], ys[edge + 1]
            query_x = x0 + alpha * (x1 - x0)
            if not hermite:
                return query_x, y0 + alpha * (y1 - y0)
            a2, a3 = alpha.square(), alpha.pow(3)
            h = x1 - x0
            query_y = (
                (2 * a3 - 3 * a2 + 1) * y0
                + (a3 - 2 * a2 + alpha) * h * first[edge]
                + (-2 * a3 + 3 * a2) * y1
                + (a3 - a2) * h * first[edge + 1]
            )
            return query_x, query_y

        self.linear_x, self.linear_y = generate(interval_probability, False)
        self.hermite_x, self.hermite_y = generate(structural, True)

    @staticmethod
    def _chunked_neighbours(z: torch.Tensor, y: torch.Tensor, count: int):
        same_rows, opposite_rows = [], []
        for start in range(0, len(z), 384):
            stop = min(len(z), start + 384)
            distance = torch.cdist(z[start:stop], z)
            row = torch.arange(stop - start)
            own = torch.arange(start, stop)
            distance[row, own] = torch.inf
            same = y[start:stop, None] == y[None]
            same_distance = distance.masked_fill(~same, torch.inf)
            opposite_distance = distance.masked_fill(same, torch.inf)
            same_rows.append(same_distance.topk(count, largest=False).indices)
            opposite_rows.append(opposite_distance.argmin(1))
        return torch.cat(same_rows), torch.cat(opposite_rows)

    def _build_classification(self, pool_size: int):
        mean = self.x.mean(0, keepdim=True)
        centered = self.x - mean
        covariance = centered.T @ centered / max(1, len(centered) - 1)
        eigenvalue, basis = torch.linalg.eigh(covariance)
        eigenvalue, basis = eigenvalue.flip(0), basis.flip(1)
        gaps = eigenvalue[:-1] / eigenvalue[1:].clamp_min(1e-12)
        largest_gap, gap_index = gaps.max(0)
        # Only collapse to an empirical support subspace when the observations
        # themselves contain decisive evidence for it. Otherwise retain the
        # full ambient geometry. This is rotation invariant; unlike per-axis
        # standardization it cannot promote tiny off-manifold noise merely
        # because the signal plane is rotated across every coordinate.
        intrinsic_rank = (
            int(gap_index) + 1 if float(largest_gap) >= 8.0 else self.x.shape[1]
        )
        basis = basis[:, :intrinsic_rank]
        scale = eigenvalue[:intrinsic_rank].sqrt().clamp_min(1e-5)
        z = (centered @ basis) / scale
        count = min(self.neighbours, max(1, int((self.y == self.y[0]).sum()) - 1))
        same, opposite = self._chunked_neighbours(z, self.y, count)
        normal = F.normalize(z[opposite] - z, dim=1)
        if z.shape[1] == 2:
            tangent = torch.stack((-normal[:, 1], normal[:, 0]), dim=1)
            neighbour_delta = z[same] - z[:, None]
            alignment = torch.einsum(
                "bkd,bd->bk", F.normalize(neighbour_delta, dim=2), tangent
            ).abs()
            partner = same[torch.arange(len(z)), alignment.argmax(1)]
        else:
            local = z[same] - z[:, None]
            _, _, vectors = torch.linalg.svd(local, full_matrices=False)
            tangent = vectors[:, 0]
            alignment = torch.einsum(
                "bkd,bd->bk", F.normalize(local, dim=2), tangent
            ).abs()
            partner = same[torch.arange(len(z)), alignment.argmax(1)]

        local_normal = normal[same]
        angular_change = 1.0 - torch.einsum(
            "bkd,bd->bk", local_normal, normal
        ).abs().mean(1)
        local_scale = (z[same[:, 0]] - z).norm(dim=1).clamp_min(1e-5)
        curvature = angular_change / local_scale
        boundary_distance = (z[opposite] - z).norm(dim=1).clamp_min(1e-5)
        complexity = (
            1.0 / _robust_unit_scale(boundary_distance).clamp_min(0.15)
            * (1.0 + 2.0 * _robust_unit_scale(curvature))
        ).clamp_max(32)
        # Equalize classes before applying local geometric complexity.
        for label in torch.unique(self.y):
            selected = self.y == label
            complexity[selected] /= complexity[selected].sum().clamp_min(1e-8)
        self.real_probability = complexity / complexity.sum()
        self.geometry_diagnostics = {
            "mean_boundary_distance": float(boundary_distance.mean()),
            "mean_boundary_curvature": float(curvature.mean()),
            "maximum_boundary_curvature": float(curvature.max()),
            "observed_intrinsic_rank": intrinsic_rank,
            "observed_eigengap": float(largest_gap),
        }

        anchor = torch.multinomial(
            self.real_probability, pool_size, replacement=True,
            generator=self.generator,
        )
        endpoint = partner[anchor]
        alpha = torch.rand((pool_size, 1), generator=self.generator)
        z0, z1 = z[anchor], z[endpoint]
        chord = z1 - z0
        length = chord.norm(dim=1, keepdim=True).clamp_min(1e-6)
        t0, t1 = tangent[anchor], tangent[endpoint]
        t0 = t0 * torch.sign((t0 * chord).sum(1, keepdim=True)).clamp_min(0).mul(2).sub(1)
        t1 = t1 * torch.sign((t1 * chord).sum(1, keepdim=True)).clamp_min(0).mul(2).sub(1)
        a2, a3 = alpha.square(), alpha.pow(3)
        curve = (
            (2 * a3 - 3 * a2 + 1) * z0
            + (a3 - 2 * a2 + alpha) * length * t0
            + (-2 * a3 + 3 * a2) * z1
            + (a3 - a2) * length * t1
        )
        line = z0 + alpha * chord
        curve_x = (curve * scale) @ basis.T + mean
        line_x = (line * scale) @ basis.T + mean
        # The observed 1-NN label is a conservative support test, not an
        # analytic oracle.  Failed curves fall back to their safe chord.
        accepted = []
        for start in range(0, pool_size, 512):
            nearest = torch.cdist(curve[start:start + 512], z).argmin(1)
            accepted.append(self.y[nearest] == self.y[anchor[start:start + 512]])
        accepted = torch.cat(accepted)
        curve_x[~accepted] = line_x[~accepted]
        self.hermite_x, self.hermite_y = curve_x, self.y[anchor]
        self.linear_x, self.linear_y = line_x, self.y[anchor]
        self.geometry_diagnostics["hermite_acceptance"] = float(accepted.float().mean())

    def sample_real(self, batch: int, generator: torch.Generator, *, structural=False):
        if structural:
            index = torch.multinomial(
                self.real_probability, batch, replacement=True,
                generator=generator,
            )
        else:
            index = torch.randint(len(self.x), (batch,), generator=generator)
        return self.x[index], self.y[index]

    def sample_pool(self, mode: str, batch: int, generator: torch.Generator):
        x = self.hermite_x if mode == "hermite" else self.linear_x
        y = self.hermite_y if mode == "hermite" else self.linear_y
        index = torch.randint(len(x), (batch,), generator=generator)
        return x[index], y[index]


def _record_best(model, task, history, step, loss, best):
    metrics = evaluate(model, task, task.x_val, task.y_val)
    history.append({"step": step, "loss": float(loss), **metrics})
    if best is None or metrics["score"] > best[0]:
        best = (metrics["score"], copy.deepcopy(model.state_dict()), step)
    return best


def train_ordinary(variant, task, *, acquisition, width, seed, steps, batch,
                   lr, evaluate_every):
    torch.manual_seed(71000 + seed)
    model = make_hybrid(variant, task.input_dim, task.output_dim, width)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    atlas = ObservedHermiteAtlas(task, seed=72000 + seed)
    generator = torch.Generator().manual_seed(73000 + seed)
    history, best = [], None
    started = time.perf_counter()
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        if acquisition in {"hermite_mix", "linear_mix"}:
            empirical_count = batch // 2
            x0, y0 = atlas.sample_real(empirical_count, generator)
            x1, y1 = atlas.sample_pool(
                acquisition.removesuffix("_mix"),
                batch - empirical_count,
                generator,
            )
            loss = _loss(task, model(torch.cat((x0, x1))), torch.cat((y0, y1)))
        elif acquisition in {"hermite_only", "linear_only"}:
            x0, y0 = atlas.sample_pool(
                acquisition.removesuffix("_only"), batch, generator
            )
            loss = _loss(task, model(x0), y0)
        else:
            x0, y0 = atlas.sample_real(batch, generator)
            loss = _loss(task, model(x0), y0)
        if acquisition not in {
            "empirical", "hermite_mix", "linear_mix",
            "hermite_only", "linear_only",
        }:
            if acquisition == "structural":
                x1, y1 = atlas.sample_real(batch, generator, structural=True)
            else:
                x1, y1 = atlas.sample_pool(acquisition, batch, generator)
            loss = loss + 0.75 * _loss(task, model(x1), y1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        if step == 1 or step % evaluate_every == 0 or step == steps:
            best = _record_best(model, task, history, step, loss.detach(), best)
    seconds = time.perf_counter() - started
    model.load_state_dict(best[1])
    return model, history, seconds, best[2], atlas.geometry_diagnostics


def train_zonotope(variant, task, *, width, seed, steps, batch, lr,
                   evaluate_every, temperature, witness_period=50,
                   worst_weight=0.1, radial_bands=False):
    torch.manual_seed(71000 + seed)
    model = make_hybrid(variant, task.input_dim, task.output_dim, width)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    witness = ProjectiveWitnessAtlas(task)
    atlases = ([] if radial_bands else [
        ObservedHermiteAtlas(
            task, witness.train_pool[fold], pool_size=8192,
            seed=74000 + 101 * seed + fold,
        )
        for fold in range(witness.folds)
    ])
    named = _named_parameters(model)
    history, best, diagnostics = [], None, []
    started = time.perf_counter()
    loss_value = 0.0
    for zero_step in range(steps):
        held_out = (zero_step // witness_period) % witness.folds
        atlas = None if radial_bands else atlases[held_out]
        gradients = []
        for candidate, mode in enumerate(("empirical", "structural", "hermite")):
            generator = torch.Generator().manual_seed(
                75000 + 1000003 * seed + 1009 * zero_step + 37 * candidate
            )
            if radial_bands:
                band = (held_out + candidate + 1) % witness.folds
                index = witness.sample_band(batch, band, generator)
                x, y = task.x_train[index], task.y_train[index]
            elif mode == "empirical":
                index = witness.sample(batch, held_out, generator)
                x, y = task.x_train[index], task.y_train[index]
            elif mode == "structural":
                x, y = atlas.sample_real(batch, generator, structural=True)
            else:
                x, y = atlas.sample_pool("hermite", batch, generator)
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(task, model(x), y)
            loss.backward()
            gradient = _flatten_gradients(named)
            norm = gradient.norm().clamp_min(1e-12)
            gradients.append(gradient * torch.clamp(gradient.new_tensor(10.0) / norm, max=1.0))
            loss_value = float(loss.detach())
        diagnostic = transported_step(
            model, optimizer, torch.stack(gradients), witness, held_out,
            rank=2, temperature=temperature, worst_weight=worst_weight,
            band_witness=radial_bands,
        )
        diagnostics.append(diagnostic)
        step = zero_step + 1
        if zero_step == 0 or step % evaluate_every == 0 or step == steps:
            best = _record_best(model, task, history, step, loss_value, best)
            history[-1].update(diagnostic)
    seconds = time.perf_counter() - started
    model.load_state_dict(best[1])
    summary = {
        key: sum(row[key] for row in diagnostics) / len(diagnostics)
        for key in diagnostics[0]
    }
    if radial_bands:
        summary.update({
            "witness_intrinsic_rank": witness.intrinsic_rank,
            "witness_intrinsic_radius": float(
                witness.geometry_mode == "intrinsic_radius"
            ),
        })
    else:
        summary.update({
            f"atlas_{key}": sum(a.geometry_diagnostics[key] for a in atlases) / len(atlases)
            for key in atlases[0].geometry_diagnostics
        })
    return model, history, seconds, best[2], summary


def run_one(task_name, method, seed, args):
    task = TASK_BUILDERS[task_name](seed)
    zonotope_variants = {
        "self_zonotope": BASE_SELF,
        "operator_zonotope": OPERATOR,
        "curvature_zonotope": CURVATURE,
        "curvature2_zonotope": CURVATURE2,
        "self_band_zonotope": BASE_SELF,
        "curvature_band_zonotope": CURVATURE,
    }
    zonotope_prefix = next(
        (prefix for prefix in zonotope_variants if method.startswith(prefix)),
        None,
    )
    if zonotope_prefix is not None:
        variant = zonotope_variants[zonotope_prefix]
        temperature = args.cold_temperature if method.endswith("cold") else args.warm_temperature
        model, history, seconds, best_step, diagnostics = train_zonotope(
            variant, task, width=args.width, seed=seed, steps=args.steps,
            batch=args.batch, lr=args.lr, evaluate_every=args.eval_every,
            temperature=temperature,
            radial_bands="band_zonotope" in zonotope_prefix,
        )
        gradient_evaluations = 3 * args.steps
    else:
        if method.startswith("operator_"):
            variant, acquisition = OPERATOR, method.removeprefix("operator_")
        elif method.startswith("cone_"):
            variant, acquisition = LEARNED_CONE, method.removeprefix("cone_")
        elif method.startswith("nested_"):
            variant, acquisition = NESTED_OPERATOR, method.removeprefix("nested_")
        elif method.startswith("selection_"):
            variant = SELECTION_CURVATURE
            acquisition = method.removeprefix("selection_")
        elif method.startswith("curvature_"):
            variant, acquisition = CURVATURE, method.removeprefix("curvature_")
        elif method.startswith("curvature2_"):
            variant, acquisition = CURVATURE2, method.removeprefix("curvature2_")
        elif method.startswith("bounded_"):
            variant, acquisition = CURVATURE_BOUNDED, method.removeprefix("bounded_")
        elif method.startswith("geometric_"):
            variant, acquisition = CURVATURE_GEOMETRIC, method.removeprefix("geometric_")
        elif method.startswith("signed_geometric_"):
            variant = CURVATURE_SIGNED_GEOMETRIC
            acquisition = method.removeprefix("signed_geometric_")
        elif method.startswith("signed_"):
            variant, acquisition = CURVATURE_SIGNED, method.removeprefix("signed_")
        else:
            variant, acquisition = BASE_SELF, method.removeprefix("self_")
        model, history, seconds, best_step, diagnostics = train_ordinary(
            variant, task, acquisition=acquisition, width=args.width, seed=seed,
            steps=args.steps, batch=args.batch, lr=args.lr,
            evaluate_every=args.eval_every,
        )
        gradient_evaluations = args.steps * (
            2 if acquisition in {"structural", "hermite", "linear"} else 1
        )
    variability, rank = jacobian_variability(model, task.x_val)
    row = {
        "task": task_name,
        "method": method,
        "seed": seed,
        "parameters": parameter_count(model),
        "steps": args.steps,
        "gradient_evaluations": gradient_evaluations,
        "seconds": seconds,
        "best_step": best_step,
        "learning_auc": auc(history, args.steps),
        "steps_to_80": threshold(history, .8),
        "steps_to_90": threshold(history, .9),
        **evaluate(model, task),
        **tail_metrics(model, task),
        "jacobian_variability": variability,
        "jacobian_change_rank": rank,
        **mechanism_diagnostics(model),
        **diagnostics,
        "history": history,
    }
    probe = {"task": task_name, "method": method, "seed": seed,
             **make_probe(task, model, args.grid)}
    return row, probe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/tmp/hermite_zonotope_pivotal"))
    parser.add_argument("--tasks", default=",".join(PIVOTAL_TASKS))
    parser.add_argument(
        "--methods",
        default=("self_empirical,self_structural,self_hermite,"
                 "operator_empirical,operator_hermite,"
                 "self_zonotope_cold,self_zonotope_warm,"
                 "operator_zonotope_warm"),
    )
    parser.add_argument("--width", type=int, default=38)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--cold-temperature", type=float, default=1e-6)
    parser.add_argument("--warm-temperature", type=float, default=1e-4)
    parser.add_argument("--grid", type=int, default=81)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    partial = args.out / "results.partial.json"
    payload = (json.loads(partial.read_text()) if args.resume and partial.exists()
               else {"runs": [], "probes": []})
    done = {(row["task"], row["method"], row["seed"]) for row in payload["runs"]}
    for task_name in args.tasks.split(","):
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            for method in args.methods.split(","):
                if (task_name, method, seed) in done:
                    continue
                print(f"START task={task_name} method={method} seed={seed}", flush=True)
                row, probe = run_one(task_name, method, seed, args)
                payload["runs"].append(row)
                if seed == 0:
                    payload["probes"].append(probe)
                partial.write_text(json.dumps(payload))
                print(json.dumps({
                    "task": task_name, "method": method,
                    "score": round(row["score"], 6),
                    "tail": None if row.get("tail_score") is None else round(row["tail_score"], 6),
                    "auc": round(row["learning_auc"], 6),
                    "seconds": round(row["seconds"], 3),
                }), flush=True)
    result = {"configuration": {**vars(args), "out": str(args.out)}, **payload}
    (args.out / "results.json").write_text(json.dumps(result, indent=2))
    (args.out / "probes.json").write_text(json.dumps({
        "configuration": result["configuration"], "probes": payload["probes"]
    }))
    print(json.dumps({"complete": True, "runs": len(payload["runs"])}))


if __name__ == "__main__":
    main()
