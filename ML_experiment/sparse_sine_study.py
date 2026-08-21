#!/usr/bin/env python3
"""Progressively thinned sine acquisition and transport study.

The experiment separates empirical-risk weighting from geometric coverage.
Every candidate receives exactly the same observed pairs.  Acquisition modes
change only how those pairs contribute to training and whether the empty
interval between adjacent observations is represented by a label-derived local
Hermite transport.  No sine, frequency, or future point enters a candidate
model.
"""
from __future__ import annotations

import argparse
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

from ML_experiment.metrics import regression_metrics
from ML_experiment.models import parameter_count
from ML_experiment.odd_context_hybrids import make_hybrid
from ML_experiment.periodic_nd_study import ParallelChartGraft, ScalarChartBank
from ML_experiment.tasks import sparse_sine_1d


torch.set_num_threads(8)

PROJECT_MODELS = {
    "mlp_26k": "ordinary_mlp_cone_budget",
    "self_context": "self_context",
    "cff": "cff",
    "learned_cone": "self_contextual_full_learned_cone",
    "operator_sphere": "self_contextual_operator_sphere_global_r2",
    "nested_operator": "self_contextual_nested_operator_r2",
}


def make_model(name: str, width: int) -> nn.Module:
    if name in PROJECT_MODELS:
        return make_hybrid(PROJECT_MODELS[name], 1, 1, width)
    if name == "scalar_chart_32":
        return ScalarChartBank(1, 1, rays=1, units=32, frame_mode="axes")
    if name == "self_commuting_chart":
        return ParallelChartGraft(
            make_hybrid("self_context", 1, 1, width),
            ScalarChartBank(1, 1, rays=1, units=32, frame_mode="axes"),
        )
    raise KeyError(name)


def nonuniform_jets(x: torch.Tensor, y: torch.Tensor):
    """Three-point first/second derivatives on sorted nonuniform samples."""
    order = torch.argsort(x[:, 0])
    xs = x[order, 0]
    ys = y[order, 0]
    count = len(xs)
    first = torch.empty_like(ys)
    second = torch.empty_like(ys)
    h0 = (xs[1:-1] - xs[:-2]).clamp_min(1e-8)
    h1 = (xs[2:] - xs[1:-1]).clamp_min(1e-8)
    first[1:-1] = (
        -h1 / (h0 * (h0 + h1)) * ys[:-2]
        + (h1 - h0) / (h0 * h1) * ys[1:-1]
        + h0 / (h1 * (h0 + h1)) * ys[2:]
    )
    second[1:-1] = 2.0 * (
        ys[:-2] / (h0 * (h0 + h1))
        - ys[1:-1] / (h0 * h1)
        + ys[2:] / (h1 * (h0 + h1))
    )
    first[0] = (ys[1] - ys[0]) / (xs[1] - xs[0]).clamp_min(1e-8)
    first[-1] = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2]).clamp_min(1e-8)
    second[0], second[-1] = second[1], second[-2]
    inverse = torch.empty(count, dtype=torch.long)
    inverse[order] = torch.arange(count)
    return first[inverse, None], second[inverse, None], order


def voronoi_weights(x: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(x[:, 0])
    xs = x[order, 0]
    cell = torch.empty_like(xs)
    cell[1:-1] = 0.5 * (xs[2:] - xs[:-2])
    cell[0] = 0.5 * (xs[1] - xs[0])
    cell[-1] = 0.5 * (xs[-1] - xs[-2])
    weight = torch.empty_like(cell)
    weight[order] = cell
    return (weight / weight.mean().clamp_min(1e-8))[:, None]


class AcquisitionData:
    def __init__(self, task):
        self.x = task.x_train
        self.y = task.y_train
        self.segment = torch.clamp(
            (self.x[:, 0] * task.observed_periods).long(),
            0, task.observed_periods - 1,
        )
        self.by_segment = [
            torch.where(self.segment == index)[0]
            for index in range(task.observed_periods)
        ]
        self.first, self.second, order = nonuniform_jets(self.x, self.y)
        self.voronoi = voronoi_weights(self.x)
        self.support_probability = (
            self.voronoi[:, 0] / self.voronoi[:, 0].sum().clamp_min(1e-8)
        )
        self.support_order = order
        self.support_cdf = self.support_probability[order].cumsum(0)
        self.derivative_scale = self.first.square().mean().sqrt().clamp_min(1e-6)
        self.curvature_scale = self.second.square().mean().sqrt().clamp_min(1e-6)
        self.sorted_x = self.x[order]
        self.sorted_y = self.y[order]
        self.sorted_first = self.first[order]
        self.interval = (self.sorted_x[1:, 0] - self.sorted_x[:-1, 0]).clamp_min(1e-8)

    def empirical(self, batch: int, generator: torch.Generator):
        return torch.randint(len(self.x), (batch,), generator=generator)

    def support(self, batch: int, generator: torch.Generator):
        """Draw points in proportion to the coordinate support they own.

        This is importance-sampled Voronoi quadrature. It places sparse cells
        in every batch without the rare, very large gradients produced by
        empirical sampling followed by inverse-density weights.
        """
        return torch.multinomial(
            self.support_probability, batch, replacement=True,
            generator=generator,
        )

    def stratified_support(self, batch: int, generator: torch.Generator):
        """One jittered draw from every equal-mass coordinate stratum."""
        coordinate = (
            torch.arange(batch, dtype=self.x.dtype)
            + torch.rand(batch, generator=generator)
        ) / batch
        position = torch.searchsorted(self.support_cdf, coordinate).clamp_max(
            len(self.x) - 1
        )
        return self.support_order[position]

    def balanced(self, batch: int, generator: torch.Generator,
                 segment_count: int | None = None):
        segment_count = segment_count or len(self.by_segment)
        segment = torch.randint(segment_count, (batch,), generator=generator)
        index = torch.empty(batch, dtype=torch.long)
        for value in range(len(self.by_segment)):
            selected = torch.where(segment == value)[0]
            if len(selected):
                pool = self.by_segment[value]
                draw = torch.randint(len(pool), (len(selected),), generator=generator)
                index[selected] = pool[draw]
        return index

    def hermite(self, batch: int, generator: torch.Generator,
                maximum_x: float | None = None):
        # Sampling intervals in proportion to their length makes the synthetic
        # queries uniform in coordinate measure, not uniform over observed gaps.
        interval_weight = self.interval
        if maximum_x is not None:
            interval_weight = interval_weight * (
                self.sorted_x[1:, 0] <= maximum_x + 1e-8
            )
        interval_index = torch.multinomial(
            interval_weight, batch, replacement=True, generator=generator
        )
        alpha = torch.rand((batch, 1), generator=generator)
        x0 = self.sorted_x[interval_index]
        x1 = self.sorted_x[interval_index + 1]
        y0 = self.sorted_y[interval_index]
        y1 = self.sorted_y[interval_index + 1]
        m0 = self.sorted_first[interval_index]
        m1 = self.sorted_first[interval_index + 1]
        h = x1 - x0
        a2, a3 = alpha.square(), alpha.pow(3)
        h00 = 2 * a3 - 3 * a2 + 1
        h10 = a3 - 2 * a2 + alpha
        h01 = -2 * a3 + 3 * a2
        h11 = a3 - a2
        target = h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1
        return x0 + alpha * h, target

    def linear(self, batch: int, generator: torch.Generator):
        interval_index = torch.multinomial(
            self.interval, batch, replacement=True, generator=generator
        )
        alpha = torch.rand((batch, 1), generator=generator)
        x0 = self.sorted_x[interval_index]
        x1 = self.sorted_x[interval_index + 1]
        y0 = self.sorted_y[interval_index]
        y1 = self.sorted_y[interval_index + 1]
        return x0 + alpha * (x1 - x0), y0 + alpha * (y1 - y0)


class SupportWhitened(nn.Module):
    """Condition coordinates using the geometric rather than empirical measure."""

    def __init__(self, model: nn.Module, data: AcquisitionData, *, gain: float = 1.0,
                 learnable_gain: bool = False):
        super().__init__()
        self.model = model
        probability = data.support_probability[:, None]
        mean = (probability * data.x).sum(0, keepdim=True)
        variance = (probability * (data.x - mean).square()).sum(0, keepdim=True)
        self.register_buffer("input_mean", mean)
        self.register_buffer("input_scale", variance.sqrt().clamp_min(1e-5))
        if learnable_gain:
            self.log_gain = nn.Parameter(torch.tensor(math.log(gain)))
        else:
            self.register_buffer("log_gain", torch.tensor(math.log(gain)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(
            (x - self.input_mean) / self.input_scale * self.log_gain.exp()
        )


@torch.no_grad()
def curve_metrics(model: nn.Module, task):
    model.eval()
    observed_prediction = model(task.x_test)
    observed = regression_metrics(observed_prediction, task.y_test)
    segments = []
    for index in range(task.observed_periods):
        lo, hi = index / task.observed_periods, (index + 1) / task.observed_periods
        selected = (task.x_test[:, 0] >= lo) & (
            task.x_test[:, 0] <= hi if index + 1 == task.observed_periods
            else task.x_test[:, 0] < hi
        )
        metrics = regression_metrics(
            observed_prediction[selected], task.y_test[selected]
        )
        segments.append({"segment": index, "count": task.segment_counts[index], **metrics})
    extrapolation = []
    for index, (x, y) in enumerate(zip(task.tail_x, task.tail_y)):
        extrapolation.append({"segment": index, **regression_metrics(model(x), y)})
    sparse = segments[-3:]
    return {
        **observed,
        "segment_metrics": segments,
        "sparse_tail_r2": sum(row["r2"] for row in sparse) / len(sparse),
        "minimum_segment_r2": min(row["r2"] for row in segments),
        "extrapolation": extrapolation,
        "extrapolation_r2": sum(row["r2"] for row in extrapolation) / len(extrapolation),
    }


@torch.no_grad()
def curve_probe(
    model: nn.Module,
    task,
    points: int = 7201,
    diagnostic_limit: float = 3.0,
):
    """Render far beyond the scored tail without changing evaluation.

    The benchmark still scores only the five explicit ``tail_x`` periods.
    The four-times-longer diagnostic tail is intentionally observational: it
    exposes phase drift, frequency renormalization, or slower oscillatory
    structure without selecting a model for those post-hoc behaviors.
    """
    model.eval()
    x = torch.linspace(0.0, diagnostic_limit, points)[:, None]
    physical_truth = torch.sin(
        2 * math.pi * task.observed_periods * x + task.phase_offset
    )
    normalized_truth = (physical_truth - task.target_mean) / task.target_std
    prediction = model(x) * task.target_std + task.target_mean
    return {
        "x": [round(float(v), 6) for v in x[:, 0]],
        "truth": [round(float(v), 6) for v in physical_truth[:, 0]],
        "prediction": [round(float(v), 6) for v in prediction[:, 0]],
        "observed_limit": 1.0,
        "scored_extrapolation_limit": 1.5,
        "diagnostic_limit": diagnostic_limit,
        "normalized_truth_mse": float((model(x) - normalized_truth).square().mean()),
    }


def train(name: str, mode: str, task, *, width: int, seed: int, steps: int,
          batch: int, lr: float, evaluate_every: int):
    torch.manual_seed(61000 + seed)
    model = make_model(name, width)
    generator = torch.Generator().manual_seed(62000 + seed)
    data = AcquisitionData(task)
    if mode in {
        "empirical_whiten", "support_whiten", "support_whiten2",
        "support_learned_scale", "stratified_whiten", "stratified_whiten2",
    }:
        model = SupportWhitened(
            model, data,
            gain=2.0 if mode in {
                "support_whiten2", "support_learned_scale", "stratified_whiten2",
            } else 1.0,
            learnable_gain=mode == "support_learned_scale",
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    history = []
    started = time.perf_counter()
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        if mode in {"empirical", "empirical_whiten"}:
            index = data.empirical(batch, generator)
            loss = F.mse_loss(model(data.x[index]), data.y[index])
        elif mode in {
            "support", "support_whiten", "support_whiten2",
            "support_learned_scale",
        }:
            index = data.support(batch, generator)
            loss = F.mse_loss(model(data.x[index]), data.y[index])
        elif mode in {"stratified_whiten", "stratified_whiten2"}:
            index = data.stratified_support(batch, generator)
            loss = F.mse_loss(model(data.x[index]), data.y[index])
        elif mode == "support_full":
            error = (model(data.x) - data.y).square()
            loss = (error * data.voronoi).mean()
        elif mode == "voronoi":
            index = data.empirical(batch, generator)
            error = (model(data.x[index]) - data.y[index]).square()
            loss = (error * data.voronoi[index]).mean()
        else:
            frontier = None
            if mode in {"frontier_hermite", "frontier_tangent"}:
                warm_steps = max(1, int(0.10 * steps))
                expansion_steps = max(1, int(0.60 * steps))
                progress = max(0.0, min(1.0, (step - warm_steps) / expansion_steps))
                frontier = 0.2 + 0.8 * progress
                visible_segments = max(
                    2, min(task.observed_periods, math.ceil(frontier * task.observed_periods))
                )
                index = data.balanced(
                    batch, generator, segment_count=visible_segments
                )
            else:
                index = data.balanced(batch, generator)
            x = data.x[index]
            y = data.y[index]
            if mode == "balanced":
                loss = F.mse_loss(model(x), y)
            elif mode == "tangent":
                x = x.detach().requires_grad_(True)
                prediction = model(x)
                derivative = torch.autograd.grad(
                    prediction.sum(), x, create_graph=True
                )[0]
                value_loss = F.mse_loss(prediction, y)
                tangent_loss = F.mse_loss(
                    derivative / data.derivative_scale,
                    data.first[index] / data.derivative_scale,
                )
                loss = value_loss + 0.35 * tangent_loss
            elif mode == "linear":
                query_x, query_y = data.linear(batch, generator)
                loss = (
                    F.mse_loss(model(x), y)
                    + 0.75 * F.mse_loss(model(query_x), query_y)
                )
            elif mode in {
                "hermite", "hermite_tangent",
                "frontier_hermite", "frontier_tangent",
            }:
                query_x, query_y = data.hermite(
                    batch, generator, maximum_x=frontier
                )
                value_loss = F.mse_loss(model(x), y)
                transport_loss = F.mse_loss(model(query_x), query_y)
                loss = value_loss + 0.75 * transport_loss
                if mode in {"hermite_tangent", "frontier_tangent"}:
                    x = x.detach().requires_grad_(True)
                    prediction = model(x)
                    derivative = torch.autograd.grad(
                        prediction.sum(), x, create_graph=True
                    )[0]
                    tangent_loss = F.mse_loss(
                        derivative / data.derivative_scale,
                        data.first[index] / data.derivative_scale,
                    )
                    loss = loss + 0.2 * tangent_loss
            else:
                raise ValueError(mode)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        if step == 1 or step % evaluate_every == 0 or step == steps:
            metrics = curve_metrics(model, task)
            history.append({
                "step": step,
                "loss": float(loss.detach()),
                "r2": metrics["r2"],
                "sparse_tail_r2": metrics["sparse_tail_r2"],
                "minimum_segment_r2": metrics["minimum_segment_r2"],
                "extrapolation_r2": metrics["extrapolation_r2"],
            })
    return model, history, time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/tmp/sparse_sine_study"))
    parser.add_argument("--models", default="self_context")
    parser.add_argument(
        "--modes",
        default=(
            "empirical,empirical_whiten,voronoi,support,support_whiten,"
            "support_whiten2,support_learned_scale,support_full,balanced,tangent,"
            "stratified_whiten,stratified_whiten2,"
            "hermite,hermite_tangent"
        ),
    )
    parser.add_argument("--width", type=int, default=38)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    partial = args.out / "results.partial.json"
    payload = json.loads(partial.read_text()) if args.resume and partial.exists() else {"runs": []}
    done = {(row["model"], row["mode"], row["seed"]) for row in payload["runs"]}
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        task = sparse_sine_1d(seed)
        for name in args.models.split(","):
            for mode in args.modes.split(","):
                if (name, mode, seed) in done:
                    continue
                print(f"START model={name} mode={mode} seed={seed}", flush=True)
                model, history, seconds = train(
                    name, mode, task, width=args.width, seed=seed,
                    steps=args.steps, batch=args.batch, lr=args.lr,
                    evaluate_every=args.eval_every,
                )
                metrics = curve_metrics(model, task)
                row = {
                    "model": name,
                    "mode": mode,
                    "seed": seed,
                    "parameters": parameter_count(model),
                    "seconds": seconds,
                    **metrics,
                    "history": history,
                    "probe": curve_probe(model, task),
                }
                payload["runs"].append(row)
                partial.write_text(json.dumps(payload))
                print(json.dumps({
                    "model": name,
                    "mode": mode,
                    "r2": round(row["r2"], 5),
                    "sparse_tail_r2": round(row["sparse_tail_r2"], 5),
                    "minimum_segment_r2": round(row["minimum_segment_r2"], 5),
                    "extrapolation_r2": round(row["extrapolation_r2"], 5),
                    "seconds": round(seconds, 3),
                }), flush=True)
    final = {
        "configuration": {**vars(args), "out": str(args.out)},
        "sampling": {
            "segment_counts": task.segment_counts,
            "segment_edges": task.segment_edges,
            "observed_limit": 1.0,
            "extrapolation_limit": 1.5,
        },
        "runs": payload["runs"],
    }
    (args.out / "results.json").write_text(json.dumps(final, indent=2))
    print(json.dumps({"complete": True, "runs": len(payload["runs"])}))


if __name__ == "__main__":
    main()
