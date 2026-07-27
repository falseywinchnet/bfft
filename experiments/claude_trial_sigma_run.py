#!/usr/bin/env python3
"""Sigma round harness.  Prints a JSON record only after a trial has run.

The control is the validated workflow: grow to the cell ceiling under the
expected-affine-gain currency, then couple all cells with a broad cartoon
overlap and a sharp texture overlap.

    PYTHONPATH=.:viewer .venv/bin/python \
        experiments/claude_trial_sigma_run.py --images pikachu \
        --max-side 256 --cells 2400
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "viewer"))

import bfft  # noqa: E402
import gallery  # noqa: E402
from transport_voronoi import Config  # noqa: E402
from transport_voronoi import _normalize  # noqa: E402
from claude_trial_sigma import (  # noqa: E402
    LAB_WEIGHTS, SigmaVoronoi, residual_structure)

from bfft.effects import lab_to_srgb  # noqa: E402


TRIALS = {
    "coupled_control": (
        "Validated workflow: LSMR coupled multiscale solve at overlap "
        "4 / 16."),
    "direct_normal": (
        "Same system, factored instead of iterated.  Direct sparse "
        "factorization of the renderer's normal matrix."),
    "ridge_enriched": (
        "Direct solve with one bounded tanh ridge column per cell, whose "
        "axis and offset come from the residual's own cumulative sign "
        "statistics."),
    "graph_descent": (
        "Learn a scalar barrier field on the pixel graph from the exact "
        "envelope-theorem gradient carried back over both shortest-path "
        "forests."),
    "leverage_exchange": (
        "Retire cells by exact constrained-deletion cost and re-seed at "
        "exact Schur-complement gain."),
    "matched_budget": (
        "Control for enrichment: plain affine cells, cell count raised so "
        "the two carry the same number of transmitted numbers "
        "(14 per enriched cell against 9 per affine cell)."),
    "ridge_geometric": (
        "Control for enrichment: the same bounded ridge, but its axis is "
        "the cartoon edge normal at the site and its offset is zero.  This "
        "is the rejected geometry-only dipole wearing the new basis."),
    "ridge_random": (
        "Control for enrichment: the same bounded ridge on a random axis "
        "through the site.  Isolates the value of the measurement from "
        "the value of owning a discontinuous basis at all."),
    "graph_random": (
        "Control for graph descent: a random smooth scale field with the "
        "same span, no gradient information."),
    "weight_descent": (
        "The same exact gradient restricted to one additive weight per "
        "site.  Asks whether the free barrier field is only relocating "
        "cell boundaries, which 2,400 numbers can already do."),
    "weight_ridge": (
        "Weight descent, then the measured bounded ridge on the resulting "
        "geometry.  Tests whether geometry and local function space are "
        "independent faults."),
    "weight_matched": (
        "Control for weight_ridge: weight descent alone at the raised cell "
        "count that carries the same number of transmitted numbers."),
    "graph_features": (
        "Constrained graph descent: the scale field is the exponential of "
        "a seven-term combination of the BFFT fields, so the same exact "
        "gradient learns how much of each ingredient the metric should "
        "contain instead of learning a free field."),
}


def feature_basis(model):
    """The BFFT-derived ingredients the metric is allowed to be made of."""
    flow = _normalize(np.abs(model.flow), 99.0)
    names = ("constant", "cartoon edge", "texture activity",
             "texture entropy", "gradient consistency", "texture demand",
             "flow magnitude")
    stack = np.stack([
        np.ones((model.h, model.w)),
        model.edge_strength,
        model.texture_activity,
        model.texture_entropy,
        model.gradient_consistency,
        _normalize(model.texture_demand, 99.0),
        flow,
    ]).reshape(len(names), -1)
    return names, stack

RIDGE_BUDGET_RATIO = 14.0 / 9.0


def native_components(image, cfg):
    split = bfft.meyer_channels(
        image, space="oklab_lc", lam=cfg.lam, mu=cfg.mu,
        passes=cfg.passes, threads=4)
    scale = np.maximum(split.scale[None, None, :], 1e-12)
    return split.cartoon / scale, split.texture / scale


def score(model, cfg, extra=None):
    reconstruction = np.clip(
        lab_to_srgb(model.reconstruction), 0.0, 1.0)
    target_cartoon, target_texture = native_components(model.rgb, cfg)
    recon_cartoon, recon_texture = native_components(reconstruction, cfg)
    rgb_mse = float(np.mean((model.rgb - reconstruction) ** 2))
    record = {
        "cells": int(len(model.seeds)),
        "rgb_mse": rgb_mse,
        "psnr": float(-10.0 * math.log10(max(rgb_mse, 1e-12))),
        "cartoon_mse": float(np.mean(
            (target_cartoon - recon_cartoon) ** 2)),
        "texture_mse": float(np.mean(
            (target_texture - recon_texture) ** 2)),
        "residual_structure": float(residual_structure(model)),
    }
    record["objective"] = (
        record["rgb_mse"] + record["cartoon_mse"] + record["texture_mse"])
    if extra:
        record.update(extra)
    return record


def _apply_fields(model, base, detail):
    precision = float(np.clip(model.cfg.detail_precision, 0.0, 1.5))
    model.cartoon_reconstruction = base
    model.texture_reconstruction = detail
    model.reconstruction = base + precision * detail
    base_delta = model.base_lab - base
    detail_delta = model.detail_lab - detail
    delta = model.lab - model.reconstruction
    model.cartoon_error = np.sqrt(
        base_delta[..., 0] ** 2 + 1.5 * np.sum(base_delta[..., 1:] ** 2, 2))
    model.texture_error = np.sqrt(
        detail_delta[..., 0] ** 2 +
        1.5 * np.sum(detail_delta[..., 1:] ** 2, 2))
    model.error = np.sqrt(
        delta[..., 0] ** 2 + 1.5 * np.sum(delta[..., 1:] ** 2, 2))
    rgb = np.clip(lab_to_srgb(model.reconstruction), 0.0, 1.0)
    model.rgb_mse = float(np.mean((model.rgb - rgb) ** 2))
    model.psnr = -10.0 * math.log10(max(model.rgb_mse, 1e-12))


def set_ridge(model, source, residual, context, rng):
    """Choose the enrichment axis and offset by one of three rules."""
    n = context["n"]
    if source == "measured":
        model.measure_ridge(residual, context)
        return
    if source == "geometric":
        from scipy import ndimage as ndi
        gx = ndi.sobel(model.cartoon, axis=1, mode="reflect")
        gy = ndi.sobel(model.cartoon, axis=0, mode="reflect")
        xi = np.clip(np.rint(model.seeds[:, 0]).astype(int), 0, model.w - 1)
        yi = np.clip(np.rint(model.seeds[:, 1]).astype(int), 0, model.h - 1)
        model.ridge_axis = np.arctan2(gy[yi, xi], gx[yi, xi])
    else:
        model.ridge_axis = rng.uniform(0.0, np.pi, size=n)
    model.ridge_offset = np.zeros(n, dtype=np.float64)


def solve_pair(model, enriched, cartoon_softness, texture_softness,
               residuals=None, ridge_source="measured", rng=None):
    """Direct coupled solve of the cartoon and texture fields."""
    out = {}
    for name, target, softness in (
            ("base", model.base_lab, cartoon_softness),
            ("detail", model.detail_lab, texture_softness)):
        context = model.assemble(softness, enriched=False)
        if enriched:
            set_ridge(model, ridge_source, residuals[name], context, rng)
            context = model.assemble(softness, enriched=True)
        solved = model.solve_exact(target, context)
        out[name] = (context, solved)
    return out


_GROWN = {}


def grown_model(image, cfg, key):
    """Grow once per (image, budget); every trial restarts from that state.

    Growth is the expensive part and is identical across trials, so sharing
    it keeps the comparison exactly paired rather than merely similar.
    """
    cache_key = (key, cfg.max_side, cfg.max_cells, cfg.initial_cells)
    if cache_key not in _GROWN:
        started = time.perf_counter()
        model = SigmaVoronoi(image, cfg)
        while len(model.seeds) < cfg.max_cells:
            model.step()
        for _ in range(3):
            model.step()
        _GROWN[cache_key] = (
            model, model.seeds.copy(), time.perf_counter() - started)
    model, seeds, grow_s = _GROWN[cache_key]
    model.seeds = seeds.copy()
    model.scale_field = np.ones(model.npix, dtype=np.float64)
    model.ridge_axis = None
    model.ridge_offset = None
    model.cfg.detail_precision = 1.0
    model._assign()
    model._fit_models()
    model._render()
    return model, grow_s


def run_trial(image, cfg, trial, steps=6, learning_rate=0.35, key="image"):
    if trial in ("matched_budget", "weight_matched"):
        cfg.max_cells = int(round(cfg.max_cells * RIDGE_BUDGET_RATIO))
    rng = np.random.default_rng(20260726)
    model, grow_s = grown_model(image, cfg, key)

    cartoon_softness, texture_softness = 4.0, 16.0
    extra = {"grow_s": grow_s}

    if trial == "coupled_control":
        t0 = time.perf_counter()
        model.solve_coupled(
            multiscale=True, cartoon_softness=cartoon_softness,
            texture_softness=texture_softness)
        extra["solve_s"] = time.perf_counter() - t0
        return score(model, cfg, extra)

    if trial in ("direct_normal", "matched_budget"):
        t0 = time.perf_counter()
        fields = solve_pair(
            model, False, cartoon_softness, texture_softness)
        _apply_fields(
            model, fields["base"][1]["field"], fields["detail"][1]["field"])
        extra["solve_s"] = time.perf_counter() - t0
        extra["unknowns"] = model.solve_stats["unknowns"]
        extra["gram_nnz"] = model.solve_stats["gram_nnz"]
        return score(model, cfg, extra)

    if trial.startswith("ridge_"):
        source = {
            "ridge_enriched": "measured",
            "ridge_geometric": "geometric",
            "ridge_random": "random",
        }[trial]
        t0 = time.perf_counter()
        plain = solve_pair(model, False, cartoon_softness, texture_softness)
        residuals = {
            "base": model.base_lab - plain["base"][1]["field"],
            "detail": model.detail_lab - plain["detail"][1]["field"],
        }
        fields = solve_pair(
            model, True, cartoon_softness, texture_softness,
            residuals=residuals, ridge_source=source, rng=rng)
        _apply_fields(
            model, fields["base"][1]["field"], fields["detail"][1]["field"])
        extra["solve_s"] = time.perf_counter() - t0
        extra["numbers_per_cell"] = 14
        extra["ridge_source"] = source
        extra.update(ridge_alignment(model))
        return score(model, cfg, extra)

    if trial in ("graph_descent", "graph_random", "graph_features"):
        t0 = time.perf_counter()
        history = []
        scale = np.ones(model.npix, dtype=np.float64)
        best_scale, best_mse = scale.copy(), np.inf
        names, basis = (
            feature_basis(model) if trial == "graph_features"
            else (None, None))
        theta = (np.zeros(len(names)) if names else None)
        best_theta = None if theta is None else theta.copy()
        theta_history = []
        rate = learning_rate
        previous_mse = np.inf
        previous_state = (scale.copy(), None if theta is None else theta.copy())
        for sweep in range(steps + 1):
            if trial == "graph_random" and sweep > 0:
                # Same number of evaluations, same span, no gradient: keep
                # the best draw, which is the honest control for a search.
                draw = rng.standard_normal(model.npix)
                draw = ndi_smooth(draw, model.h, model.w)
                draw /= max(float(np.std(draw)), 1e-12)
                scale = np.clip(np.exp(0.35 * draw), 0.35, 3.0)
                scale /= math.exp(float(np.mean(np.log(scale))))
            model.scale_field = scale
            forest = model.walk_with_predecessors(scale)
            fields = solve_pair(
                model, False, cartoon_softness, texture_softness)
            _apply_fields(
                model, fields["base"][1]["field"],
                fields["detail"][1]["field"])
            history.append(float(model.psnr))
            if model.rgb_mse < best_mse:
                best_mse = model.rgb_mse
                best_scale = scale.copy()
                if theta is not None:
                    best_theta = theta.copy()
            if trial != "graph_random":
                # Backtracking.  The gradient is exact but the step size is
                # not, and without this the trace wanders and stops being
                # evidence about the parameterization.
                if model.rgb_mse > previous_mse:
                    scale, reverted = previous_state[0].copy(), previous_state[1]
                    if reverted is not None:
                        theta = reverted.copy()
                    rate *= 0.5
                    if rate < 0.02:
                        break
                    model.scale_field = scale
                    forest = model.walk_with_predecessors(scale)
                    fields = solve_pair(
                        model, False, cartoon_softness, texture_softness)
                    _apply_fields(
                        model, fields["base"][1]["field"],
                        fields["detail"][1]["field"])
                else:
                    previous_mse = model.rgb_mse
                    previous_state = (
                        scale.copy(),
                        None if theta is None else theta.copy())
            if sweep == steps or trial == "graph_random":
                if sweep == steps:
                    break
                continue
            precision = float(np.clip(model.cfg.detail_precision, 0.0, 1.5))
            packed = []
            for name, weight, softness in (
                    ("base", 1.0, cartoon_softness),
                    ("detail", precision, texture_softness)):
                context, solved = fields[name]
                packed.append({
                    "w1": context["w1"], "w2": context["w2"],
                    "softness": softness, "scale": weight,
                    "pred_first": solved["pred_first"],
                    "pred_second": solved["pred_second"],
                })
            grad, _ = model.metric_gradient(packed, forest)
            if theta is not None:
                # Same gradient, projected onto the ingredients the metric
                # is allowed to be made of.  Seven numbers per image cannot
                # memorize an image; they can only reweight the recipe.
                projected = basis @ (grad * scale)
                magnitude = float(np.max(np.abs(projected)))
                if magnitude <= 0.0:
                    break
                theta = theta - rate * projected / magnitude
                theta_history.append([float(v) for v in theta])
                scale = np.exp(basis.T @ theta)
                scale = np.clip(scale, 0.35, 3.0)
                scale /= math.exp(float(np.mean(np.log(scale))))
                continue
            magnitude = float(np.percentile(np.abs(grad), 99.0))
            if magnitude <= 0.0:
                break
            update = np.clip(grad / magnitude, -4.0, 4.0)
            scale = scale * np.exp(-rate * update)
            scale = ndi_smooth(scale, model.h, model.w)
            scale = np.clip(scale, 0.35, 3.0)
            scale /= math.exp(float(np.mean(np.log(scale))))
        if theta is not None:
            extra["feature_names"] = list(names)
            extra["theta"] = [float(v) for v in best_theta]
            extra["theta_history"] = theta_history
        model.scale_field = best_scale
        model.walk_with_predecessors(best_scale)
        fields = solve_pair(model, False, cartoon_softness, texture_softness)
        _apply_fields(
            model, fields["base"][1]["field"], fields["detail"][1]["field"])
        extra["solve_s"] = time.perf_counter() - t0
        extra["psnr_history"] = history
        extra["scale_span"] = [float(best_scale.min()),
                               float(best_scale.max())]
        return score(model, cfg, extra)

    if trial in ("weight_descent", "weight_ridge", "weight_matched"):
        t0 = time.perf_counter()
        n = len(model.seeds)
        offset = np.zeros(n, dtype=np.float64)
        best_offset, best_mse = offset.copy(), np.inf
        history = []
        rate, previous_mse, previous = learning_rate, np.inf, offset.copy()
        precision = float(np.clip(model.cfg.detail_precision, 0.0, 1.5))
        for sweep in range(steps + 1):
            forest = model.walk_with_predecessors(
                model.scale_field, reach_offset=offset)
            fields = solve_pair(
                model, False, cartoon_softness, texture_softness)
            _apply_fields(
                model, fields["base"][1]["field"],
                fields["detail"][1]["field"])
            history.append(float(model.psnr))
            if model.rgb_mse < best_mse:
                best_mse, best_offset = model.rgb_mse, offset.copy()
            if model.rgb_mse > previous_mse:
                offset = previous.copy()
                rate *= 0.5
                if rate < 0.02:
                    break
            else:
                previous_mse, previous = model.rgb_mse, offset.copy()
            if sweep == steps:
                break
            packed = []
            for name, weight, softness in (
                    ("base", 1.0, cartoon_softness),
                    ("detail", precision, texture_softness)):
                context, solved = fields[name]
                packed.append({
                    "w1": context["w1"], "w2": context["w2"],
                    "softness": softness, "scale": weight,
                    "pred_first": solved["pred_first"],
                    "pred_second": solved["pred_second"]})
            _, sensitivity = model.metric_gradient(packed, forest)
            gradient = model.weight_gradient(sensitivity)
            magnitude = float(np.percentile(np.abs(gradient), 95.0))
            if magnitude <= 0.0:
                break
            spacing = math.sqrt(model.npix / max(n, 1))
            offset = offset - rate * spacing * np.clip(
                gradient / magnitude, -4.0, 4.0)
            offset -= float(np.mean(offset))
        model.walk_with_predecessors(
            model.scale_field, reach_offset=best_offset)
        fields = solve_pair(model, False, cartoon_softness, texture_softness)
        _apply_fields(
            model, fields["base"][1]["field"], fields["detail"][1]["field"])
        if trial == "weight_ridge":
            # Geometry and local function space are separate faults.  Fixing
            # the reach does not give a cell a discontinuity it lacks, and a
            # ridge does not move a boundary.  Composed, not alternated.
            residuals = {
                "base": model.base_lab - fields["base"][1]["field"],
                "detail": model.detail_lab - fields["detail"][1]["field"]}
            fields = solve_pair(
                model, True, cartoon_softness, texture_softness,
                residuals=residuals, ridge_source="measured", rng=rng)
            _apply_fields(
                model, fields["base"][1]["field"],
                fields["detail"][1]["field"])
            extra["numbers_per_cell"] = 14
        extra["solve_s"] = time.perf_counter() - t0
        extra["psnr_history"] = history
        extra["parameters"] = int(n)
        extra["offset_span"] = [float(best_offset.min()),
                                float(best_offset.max())]
        return score(model, cfg, extra)

    if trial == "leverage_exchange":
        t0 = time.perf_counter()
        fields = solve_pair(model, False, cartoon_softness, texture_softness)
        _apply_fields(
            model, fields["base"][1]["field"], fields["detail"][1]["field"])
        before = float(model.psnr)
        context, solved = fields["base"]
        exact, rough = model.deletion_costs(solved, context)
        residual = model.lab - model.reconstruction
        pressure = model.allocation_pressure.ravel().copy()
        candidates = choose_candidates(model, pressure, count=96)
        gains = model.addition_gains(solved, context, candidates, residual)
        budget = min(96, max(1, len(model.seeds) // 24))
        dying = np.argsort(exact)[:budget]
        order = np.argsort(gains)[::-1][:budget]
        moved = 0
        for slot, cell in zip(order, dying):
            if not np.isfinite(exact[cell]):
                continue
            pixel = int(candidates[slot])
            y, x = divmod(pixel, model.w)
            model.seeds[cell] = (float(x), float(y))
            moved += 1
        model.walk_with_predecessors(model.scale_field)
        fields = solve_pair(model, False, cartoon_softness, texture_softness)
        _apply_fields(
            model, fields["base"][1]["field"], fields["detail"][1]["field"])
        extra["solve_s"] = time.perf_counter() - t0
        extra["exchanged"] = moved
        extra["psnr_before_exchange"] = before
        return score(model, cfg, extra)

    raise ValueError(trial)


def ridge_alignment(model):
    """Does the measured discontinuity agree with the cartoon contour?

    If it does, the contour normal would have been a usable proxy and the
    measurement is a convenience.  If it does not, the measurement is
    carrying information no geometric field contains.
    """
    from scipy import ndimage as ndi
    gx = ndi.sobel(model.cartoon, axis=1, mode="reflect")
    gy = ndi.sobel(model.cartoon, axis=0, mode="reflect")
    xi = np.clip(np.rint(model.seeds[:, 0]).astype(int), 0, model.w - 1)
    yi = np.clip(np.rint(model.seeds[:, 1]).astype(int), 0, model.h - 1)
    normal = np.arctan2(gy[yi, xi], gx[yi, xi])
    strength = np.hypot(gx[yi, xi], gy[yi, xi])
    gap = np.abs(np.angle(np.exp(2j * (model.ridge_axis - normal)))) / 2.0
    strong = strength >= np.percentile(strength, 70.0)
    return {
        "ridge_axis_gap_deg": float(np.degrees(np.mean(gap))),
        "ridge_axis_gap_deg_strong_edges": float(
            np.degrees(np.mean(gap[strong]))),
        "ridge_axis_within_30deg": float(np.mean(gap < np.radians(30.0))),
    }


def ndi_smooth(scale, h, w):
    from scipy import ndimage as ndi
    return ndi.gaussian_filter(
        scale.reshape(h, w), 0.9, mode="reflect").ravel()


def choose_candidates(model, pressure, count):
    """Blue-noise thinned maxima of the current allocation pressure."""
    order = np.argsort(pressure)[::-1]
    blocked = np.zeros(model.npix, dtype=bool)
    chosen = []
    radius = max(2, int(0.6 * math.sqrt(
        model.npix / max(len(model.seeds), 1))))
    for pixel in order:
        if blocked[pixel]:
            continue
        chosen.append(int(pixel))
        y, x = divmod(int(pixel), model.w)
        y0, y1 = max(0, y - radius), min(model.h, y + radius + 1)
        x0, x1 = max(0, x - radius), min(model.w, x + radius + 1)
        blocked.reshape(model.h, model.w)[y0:y1, x0:x1] = True
        if len(chosen) >= count:
            break
    return np.asarray(chosen, dtype=np.int64)


def check(image, cfg):
    """Falsify the two claims the round rests on before trusting a score."""
    model = SigmaVoronoi(image, cfg)
    for _ in range(4):
        model.step()
    reference_owner = model.owner.copy()
    reference_d1 = model.d1.copy()
    model.walk_with_predecessors(np.ones(model.npix))
    same_owner = float(np.mean(reference_owner == model.owner))
    distance_gap = float(np.max(np.abs(
        np.where(np.isfinite(reference_d1), reference_d1, 0.0) -
        np.where(np.isfinite(model.d1), model.d1, 0.0))))

    context = model.assemble(4.0, enriched=False)
    solved = model.solve_exact(model.base_lab, context)
    from scipy.sparse.linalg import lsmr
    from scipy import sparse
    padded_design = sparse.vstack([
        context["design"],
        sparse.diags(np.sqrt(context["reg"]))]).tocsr()
    rhs = np.zeros(padded_design.shape[0])
    rhs[:model.npix] = model.base_lab[..., 0].ravel()
    iterative = lsmr(padded_design, rhs, atol=2e-6, btol=2e-6,
                     maxiter=160)[0]
    direct = solved["coeff"][:, :, 0].ravel()
    relative = float(np.linalg.norm(iterative - direct) /
                     max(np.linalg.norm(direct), 1e-12))

    # Envelope-theorem gradient against a finite difference on the scale
    # field, which is the only claim in the round that a bug could fake.
    forest = model.walk_with_predecessors(np.ones(model.npix))
    fields = solve_pair(model, False, 4.0, 16.0)
    packed = []
    for name, weight, softness in (
            ("base", 1.0, 4.0), ("detail", 1.0, 16.0)):
        ctx, sol = fields[name]
        packed.append({
            "w1": ctx["w1"], "w2": ctx["w2"], "softness": softness,
            "scale": weight, "pred_first": sol["pred_first"],
            "pred_second": sol["pred_second"]})
    grad, sensitivity = model.metric_gradient(packed, forest)

    rng = np.random.default_rng(11)
    probes = rng.choice(
        np.flatnonzero(np.abs(grad) > np.percentile(np.abs(grad), 99.5)),
        size=6, replace=False)
    epsilon = 1e-4
    agreements = []
    for pixel in probes:
        analytic = grad[pixel]
        numeric = 0.0
        for sign in (1.0, -1.0):
            perturbed = np.ones(model.npix)
            perturbed[pixel] += sign * epsilon
            model.walk_with_predecessors(perturbed)
            local = solve_pair(model, False, 4.0, 16.0)
            recon = (local["base"][1]["field"] +
                     local["detail"][1]["field"])
            delta = recon - model.lab
            loss = float(np.sum(
                LAB_WEIGHTS[None, None, :] * delta * delta))
            numeric += sign * loss
        numeric /= 2.0 * epsilon
        agreements.append((float(analytic), float(numeric)))
    model.walk_with_predecessors(np.ones(model.npix))
    return {
        "owner_match_fraction": same_owner,
        "max_distance_gap": distance_gap,
        "lsmr_vs_direct_relative": relative,
        "sensitivity_support": float(np.mean(sensitivity != 0.0)),
        "gradient_probe_analytic_numeric": agreements,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-side", type=int, default=256)
    parser.add_argument("--cells", type=int, default=2400)
    parser.add_argument("--initial-cells", type=int, default=180)
    parser.add_argument("--images", nargs="+", default=["pikachu"])
    parser.add_argument("--trials", nargs="+", choices=list(TRIALS),
                        default=list(TRIALS))
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    def make_cfg():
        return Config(
            max_side=args.max_side, initial_cells=args.initial_cells,
            max_cells=args.cells, split_batch=36,
            allocation_mode="Expected affine gain",
            recursive_memory_stages=1, residual_memory_weight=0.0,
            composition_discrepancy_weight=0.0)

    record = {
        "protocol": {
            "images": args.images, "max_side": args.max_side,
            "maximum_cells": args.cells, "initial_cells": args.initial_cells,
            "allocation_mode": "Expected affine gain",
            "coupled_overlap": [4.0, 16.0],
            "objective": "rgb_mse + cartoon_mse + texture_mse, lower better",
        },
        "results": [],
    }

    if args.check:
        cfg = make_cfg()
        cfg.max_side = min(args.max_side, 96)
        cfg.initial_cells = 64
        cfg.max_cells = 120
        record["check"] = check(gallery.load(args.images[0]), cfg)
        print(json.dumps(record["check"], indent=2), file=sys.stderr)

    for key in args.images:
        image = gallery.load(key)
        for trial in args.trials:
            cfg = make_cfg()
            result = run_trial(
                image, cfg, trial, steps=args.steps, key=key)
            result["trial"] = trial
            result["description"] = TRIALS[trial]
            result["image"] = key
            record["results"].append(result)
            print(
                f"{key:9s} {trial:18s} {result['psnr']:6.2f} dB "
                f"obj {result['objective']:.3e} "
                f"struct {result['residual_structure']:.4f} "
                f"solve {result.get('solve_s', 0.0):.2f}s",
                file=sys.stderr, flush=True)
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
