#!/usr/bin/env python3
"""Marching fusion on the renderer's measured co-ownership graph.

This is deliberately a *soft* fusion experiment.  Pixels, sites, and current
owner/runner geometry are left unchanged.  Compatible neighboring cell jets
are transported into one common affine frame and tied by a prolongation
``x = P z``.  The contracted normal equations are therefore exact:

    G_group = P.T @ G @ P
    h_group = P.T @ h

Only after such a tied boundary is shown to be harmless should a later hard
fusion delete its redundant geometric support.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "viewer"))

import gallery  # noqa: E402
from bfft.vision import SingleStageDecompositionObjective, render_partition
from claude_trial_sigma import LAB_WEIGHTS, SigmaVoronoi  # noqa: E402
from transport_voronoi import Config  # noqa: E402


def normalize_labels(labels):
    _, normalized = np.unique(np.asarray(labels, dtype=np.int64),
                              return_inverse=True)
    return normalized.astype(np.int32)


def group_geometry(model, labels, spacing):
    """Mass-weighted anchor and axial frame for every current soft group."""
    labels = normalize_labels(labels)
    groups = int(labels.max()) + 1
    cells = len(model.seeds)
    pixel_mass = np.bincount(model.owner, minlength=cells).astype(np.float64)
    pixel_mass = np.maximum(pixel_mass, 1.0)
    mass = np.bincount(labels, weights=pixel_mass, minlength=groups)
    anchor_x = np.bincount(
        labels, weights=pixel_mass * model.seeds[:, 0],
        minlength=groups) / mass
    anchor_y = np.bincount(
        labels, weights=pixel_mass * model.seeds[:, 1],
        minlength=groups) / mass
    angles, _ = model._site_frames()
    # Cell axes are unoriented.  Average on the doubled-angle circle.
    cosine = np.bincount(
        labels, weights=pixel_mass * np.cos(2.0 * angles),
        minlength=groups)
    sine = np.bincount(
        labels, weights=pixel_mass * np.sin(2.0 * angles),
        minlength=groups)
    group_angle = 0.5 * np.arctan2(sine, cosine)
    return labels, np.column_stack((anchor_x, anchor_y)), group_angle, angles


def affine_prolongation(model, labels, spacing):
    """Map one group jet to every member cell's local affine coordinates."""
    labels, anchors, group_angle, cell_angle = group_geometry(
        model, labels, spacing)
    cells = len(labels)
    groups = int(labels.max()) + 1
    rows = []
    columns = []
    values = []
    for cell in range(cells):
        group = int(labels[cell])
        ai = float(cell_angle[cell])
        ag = float(group_angle[group])
        ti = np.array([math.cos(ai), math.sin(ai)])
        ni = np.array([-math.sin(ai), math.cos(ai)])
        tg = np.array([math.cos(ag), math.sin(ag)])
        ng = np.array([-math.sin(ag), math.cos(ag)])
        delta = model.seeds[cell] - anchors[group]
        transform = np.array([
            [1.0, float(delta @ tg) / spacing,
             float(delta @ ng) / spacing],
            [0.0, float(ti @ tg), float(ti @ ng)],
            [0.0, float(ni @ tg), float(ni @ ng)],
        ])
        for a in range(3):
            for b in range(3):
                value = float(transform[a, b])
                if value != 0.0:
                    rows.append(3 * cell + a)
                    columns.append(3 * group + b)
                    values.append(value)
    prolongation = sparse.coo_matrix(
        (values, (rows, columns)),
        shape=(3 * cells, 3 * groups)).tocsc()
    return labels, prolongation, anchors, group_angle


def solve_contracted(model, context, solved, labels):
    """Solve one field exactly in the affine space selected by ``labels``."""
    labels, prolongation, anchors, angles = affine_prolongation(
        model, labels, context["spacing"])
    gram = (prolongation.T @ solved["gram"] @ prolongation).tocsc()
    rhs = np.asarray(prolongation.T @ solved["rhs"])
    factor = splu(
        gram, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
        options={"SymmetricMode": True})
    group_coeff = factor.solve(rhs)
    cell_coeff = np.asarray(prolongation @ group_coeff).reshape(
        len(model.seeds), 3, 3)
    field, first, second = render_partition(
        cell_coeff, model.owner, context["other"], context["valid"],
        context["w1"], context["w2"], context["first"], context["second"])
    return {
        "field": field.reshape(model.h, model.w, 3),
        "coeff": cell_coeff,
        "pred_first": first,
        "pred_second": second,
        "group_coeff": group_coeff.reshape(-1, 3, 3),
        "labels": labels,
        "anchors": anchors,
        "angles": angles,
        "gram": gram,
        "rhs": rhs,
        "lu": factor,
    }


def measured_group_edges(model, labels, fields):
    """Compatibility of actual adjacent groups under their current fits."""
    labels = normalize_labels(labels)
    valid = (
        (model.second >= 0) &
        (model.owner >= 0) &
        (model.owner != model.second))
    pixel = np.flatnonzero(valid)
    owner_group = labels[model.owner[pixel]]
    runner_group = labels[model.second[pixel]]
    crossing = owner_group != runner_group
    pixel = pixel[crossing]
    owner_group = owner_group[crossing]
    runner_group = runner_group[crossing]
    if pixel.size == 0:
        return []
    groups = int(labels.max()) + 1
    low = np.minimum(owner_group, runner_group)
    high = np.maximum(owner_group, runner_group)
    key = low.astype(np.int64) * groups + high
    unique, inverse = np.unique(key, return_inverse=True)

    joint = np.zeros(pixel.size, dtype=np.float64)
    gap_energy = np.zeros(pixel.size, dtype=np.float64)
    precision = float(np.clip(model.cfg.detail_precision, 0.0, 1.5))
    for name, scale in (("base", 1.0), ("detail", precision)):
        context, solved = fields[name]
        delta = (
            solved["pred_first"][pixel] -
            solved["pred_second"][pixel])
        gap_energy += scale * scale * np.sum(
            LAB_WEIGHTS[None, :] * delta * delta, axis=1)
        joint += context["w1"][pixel] * context["w2"][pixel]
    joint = np.maximum(joint * 0.5, 1e-6)
    shared = np.bincount(
        inverse, weights=joint, minlength=unique.size)
    mismatch = np.bincount(
        inverse, weights=joint * gap_energy,
        minlength=unique.size) / np.maximum(shared, 1e-12)
    barrier = np.bincount(
        inverse, weights=joint * model.edge_strength.ravel()[pixel],
        minlength=unique.size) / np.maximum(shared, 1e-12)
    texture = np.bincount(
        inverse, weights=joint * model.texture_activity.ravel()[pixel],
        minlength=unique.size) / np.maximum(shared, 1e-12)

    _, _, group_angle, _ = group_geometry(
        model, labels, math.sqrt(model.npix / max(len(model.seeds), 1)))
    records = []
    for slot, encoded in enumerate(unique):
        a = int(encoded // groups)
        b = int(encoded % groups)
        alignment = abs(math.cos(float(group_angle[a] - group_angle[b])))
        orientation_penalty = texture[slot] * (1.0 - alignment) ** 2
        # Mismatch is primary.  A real cartoon/texture boundary or an
        # incoherent texture direction makes an otherwise similar pair wait.
        score = (
            mismatch[slot] *
            (1.0 + 2.0 * barrier[slot] + 1.5 * orientation_penalty))
        records.append({
            "a": a, "b": b, "score": float(score),
            "mismatch": float(mismatch[slot]),
            "shared": float(shared[slot]),
            "barrier": float(barrier[slot]),
            "texture": float(texture[slot]),
            "alignment": float(alignment),
        })
    return records


def greedy_front(edges, max_merges, quantile=0.35):
    """One non-overlapping low-energy front on the current measured graph."""
    if not edges or max_merges <= 0:
        return []
    scores = np.array([edge["score"] for edge in edges])
    threshold = float(np.quantile(scores, np.clip(quantile, 0.0, 1.0)))
    used = set()
    selected = []
    for edge in sorted(edges, key=lambda record: record["score"]):
        if edge["score"] > threshold:
            break
        if edge["a"] in used or edge["b"] in used:
            continue
        used.add(edge["a"])
        used.add(edge["b"])
        selected.append(edge)
        if len(selected) >= max_merges:
            break
    return selected


def merge_labels(labels, edges):
    labels = normalize_labels(labels)
    groups = int(labels.max()) + 1
    parent = np.arange(groups, dtype=np.int32)

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    for edge in edges:
        a, b = root(edge["a"]), root(edge["b"])
        if a != b:
            parent[b] = a
    group_map = np.array([root(i) for i in range(groups)], dtype=np.int32)
    return normalize_labels(group_map[labels])


def evaluate_grouping(model, base_fields, labels, objective):
    output = {}
    for name in ("base", "detail"):
        context, solved = base_fields[name]
        output[name] = (
            context, solve_contracted(model, context, solved, labels))
    precision = float(np.clip(model.cfg.detail_precision, 0.0, 1.5))
    reconstruction = (
        output["base"][1]["field"] +
        precision * output["detail"][1]["field"])
    from bfft.effects import lab_to_srgb
    rgb = np.clip(lab_to_srgb(reconstruction), 0.0, 1.0)
    return output, reconstruction, objective.evaluate(rgb)


def marching_fusion(model, rounds=6, fraction=0.08, quantile=0.35,
                    objective_tolerance=0.0025, psnr_tolerance=0.03,
                    cartoon_softness=4.0, texture_softness=16.0):
    """March compatible support contractions with one half-front retry."""
    base_fields = model._solve_direct_pair(
        cartoon_softness, texture_softness, enriched=False)
    model._apply_direct_fields(
        base_fields["base"][1]["field"],
        base_fields["detail"][1]["field"], "fusion baseline")
    objective = SingleStageDecompositionObjective(
        model.rgb, lam=model.cfg.lam, mu=model.cfg.mu,
        passes=model.cfg.passes, threads=4)
    from bfft.effects import lab_to_srgb
    baseline = objective.evaluate(np.clip(
        lab_to_srgb(model.reconstruction), 0.0, 1.0))
    current = baseline
    labels = np.arange(len(model.seeds), dtype=np.int32)
    current_fields = base_fields
    trace = []

    for round_index in range(max(0, int(rounds))):
        edges = measured_group_edges(model, labels, current_fields)
        front = greedy_front(
            edges,
            max(1, int(math.ceil((labels.max() + 1) * fraction))),
            quantile=quantile)
        if not front:
            break
        accepted = None
        attempts = (front, front[:max(1, len(front) // 2)])
        seen_sizes = set()
        for attempt, selected in enumerate(attempts):
            if len(selected) in seen_sizes:
                continue
            seen_sizes.add(len(selected))
            proposal = merge_labels(labels, selected)
            fields, reconstruction, measured = evaluate_grouping(
                model, base_fields, proposal, objective)
            relative = (
                measured["objective"] / max(baseline["objective"], 1e-12) -
                1.0)
            psnr_loss = baseline["psnr"] - measured["psnr"]
            record = {
                "round": round_index,
                "attempt": attempt,
                "merges": len(selected),
                "groups": int(proposal.max()) + 1,
                "relative_objective": float(relative),
                "psnr": float(measured["psnr"]),
                "psnr_loss": float(psnr_loss),
            }
            trace.append(record)
            if (relative <= objective_tolerance and
                    psnr_loss <= psnr_tolerance):
                accepted = proposal, fields, reconstruction, measured
                break
        if accepted is None:
            break
        labels, current_fields, model.reconstruction, current = accepted
        model.cartoon_reconstruction = current_fields["base"][1]["field"]
        model.texture_reconstruction = current_fields["detail"][1]["field"]

    model.fusion_labels = labels
    model.fusion_groups = int(labels.max()) + 1
    model.fusion_trace = trace
    model._apply_direct_fields(
        current_fields["base"][1]["field"],
        current_fields["detail"][1]["field"], "marching soft fusion")
    return {
        "cells": len(model.seeds),
        "groups": model.fusion_groups,
        "fused": len(model.seeds) - model.fusion_groups,
        "baseline": baseline,
        "final": current,
        "trace": trace,
    }


def grow(image, cfg):
    model = SigmaVoronoi(image, cfg)
    while len(model.seeds) < cfg.max_cells:
        model.step_direct()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="+", default=["pikachu", "camera"])
    parser.add_argument("--max-side", type=int, default=128)
    parser.add_argument("--cells", type=int, default=700)
    parser.add_argument("--passes", type=int, default=6)
    parser.add_argument("--flow-sweeps", type=int, default=24)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--fraction", type=float, default=0.08)
    parser.add_argument("--quantile", type=float, default=0.35)
    parser.add_argument("--objective-tolerance", type=float, default=0.0025)
    parser.add_argument("--psnr-tolerance", type=float, default=0.03)
    args = parser.parse_args()
    results = []
    for image in args.images:
        cfg = Config(
            max_side=args.max_side, passes=args.passes,
            flow_sweeps=args.flow_sweeps, initial_cells=min(96, args.cells),
            max_cells=args.cells, split_batch=36,
            allocation_mode="Expected affine gain")
        model = grow(gallery.load(image), cfg)
        report = marching_fusion(
            model, rounds=args.rounds, fraction=args.fraction,
            quantile=args.quantile,
            objective_tolerance=args.objective_tolerance,
            psnr_tolerance=args.psnr_tolerance)
        results.append({"image": image, **report})
        print(
            f"{image:10s} {report['cells']} cells -> {report['groups']} "
            f"soft groups; fused {report['fused']}; "
            f"PSNR {report['baseline']['psnr']:.3f} -> "
            f"{report['final']['psnr']:.3f}",
            file=sys.stderr)
    print(json.dumps({"protocol": vars(args), "results": results},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
