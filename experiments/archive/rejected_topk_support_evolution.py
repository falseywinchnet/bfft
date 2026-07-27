#!/usr/bin/env python3
"""ARCHIVED: rejected top-K probe for BFFT-guided cell supports.

Top-K candidate expansion was explicitly belayed on 2026-07-26.  This file is
retained only because its controls established that residual anisotropy and
receiver-driven support bias were useful signals, while adaptive temperature,
coefficient-only fusion, and forced deletion were negative.  Do not backport
this candidate representation into the live model.

The active direction remains the exact owner/runner co-ownership graph.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy import sparse
from scipy.sparse.linalg import splu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "viewer"))
sys.path.insert(0, str(ROOT / "experiments"))

import gallery  # noqa: E402
from bfft.effects import lab_to_srgb  # noqa: E402
from bfft.vision import SingleStageDecompositionObjective  # noqa: E402
from receiver_guided_graph import ReceiverGuidedVoronoi  # noqa: E402
from transport_voronoi import Config  # noqa: E402


@dataclass
class SupportState:
    seeds: np.ndarray
    angles: np.ndarray
    ratios: np.ndarray
    bias: np.ndarray
    temperature: np.ndarray
    capacity: np.ndarray

    def clone(self):
        return copy.deepcopy(self)


@dataclass
class SupportFit:
    logits: np.ndarray
    phi: np.ndarray
    top_index: np.ndarray
    top_weight: np.ndarray
    design: sparse.csr_matrix
    gram: sparse.csc_matrix
    rhs: np.ndarray
    coeff: np.ndarray
    reconstruction: np.ndarray
    rgb: np.ndarray
    psnr: float
    rgb_mse: float
    objective: dict | None
    metrics: dict


def initial_state(model, temperature=4.0):
    seed_x = np.clip(
        np.rint(model.seeds[:, 0]).astype(int), 0, model.w - 1)
    seed_y = np.clip(
        np.rint(model.seeds[:, 1]).astype(int), 0, model.h - 1)
    angle = model.angle[seed_y, seed_x].astype(np.float64)
    coherence = np.clip(
        model.coherence[seed_y, seed_x].astype(np.float64), 0.0, 1.0)
    ratio = np.clip(1.0 + 5.0 * coherence, 1.0, 10.0)
    capacity = np.bincount(
        model.owner, minlength=len(model.seeds)).astype(np.float64)
    capacity = np.maximum(capacity, 1.0)
    capacity *= model.npix / float(np.sum(capacity))
    return SupportState(
        seeds=model.seeds.copy(),
        angles=angle,
        ratios=ratio,
        bias=np.zeros(len(model.seeds), dtype=np.float64),
        temperature=np.full(
            len(model.seeds), float(temperature), dtype=np.float64),
        capacity=capacity)


def raw_logits(model, state):
    """Continuous determinant-one anisotropic site scores."""
    spacing = math.sqrt(model.npix / max(len(state.seeds), 1))
    x = model.xf[:, None]
    y = model.yf[:, None]
    dx = (x - state.seeds[None, :, 0]) / spacing
    dy = (y - state.seeds[None, :, 1]) / spacing
    cosine = np.cos(state.angles)[None, :]
    sine = np.sin(state.angles)[None, :]
    tangent = dx * cosine + dy * sine
    normal = -dx * sine + dy * cosine
    ratio = np.maximum(state.ratios[None, :], 1.0)
    distance = np.sqrt(
        tangent * tangent / ratio + normal * normal * ratio + 1e-12)
    return (
        state.bias[None, :] -
        state.temperature[None, :] * distance)


def diffuse_scores(model, logits, steps=0, strength=0.55):
    """Diffuse candidate explanations through the BFFT support graph."""
    if steps <= 0 or strength <= 0.0:
        return logits
    z = logits.reshape(model.h, model.w, -1).copy()
    conductance = np.exp(
        -1.5 * np.maximum(model._edge_cost_volume.astype(np.float64) - 1.0,
                          0.0))
    directions = (
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1))
    alpha = float(np.clip(strength, 0.0, 0.95))
    for _ in range(int(steps)):
        accumulated = np.zeros_like(z)
        mass = np.zeros((model.h, model.w), dtype=np.float64)
        for direction, (dy, dx) in enumerate(directions):
            ys = slice(max(0, -dy), min(model.h, model.h - dy))
            xs = slice(max(0, -dx), min(model.w, model.w - dx))
            yd = slice(max(0, dy), min(model.h, model.h + dy))
            xd = slice(max(0, dx), min(model.w, model.w + dx))
            weight = conductance[direction, ys, xs]
            accumulated[ys, xs] += weight[..., None] * z[yd, xd]
            mass[ys, xs] += weight
        average = accumulated / np.maximum(mass[..., None], 1e-12)
        z = (1.0 - alpha) * z + alpha * average
    return z.reshape(model.npix, -1)


def topk_partition(logits, k):
    """Fixed-budget soft ownership, returned dense for analysis."""
    k = max(2, min(int(k), logits.shape[1]))
    index = np.argpartition(logits, -k, axis=1)[:, -k:]
    value = np.take_along_axis(logits, index, axis=1)
    value -= np.max(value, axis=1, keepdims=True)
    weight = np.exp(np.clip(value, -60.0, 0.0))
    weight /= np.sum(weight, axis=1, keepdims=True)
    phi = np.zeros_like(logits)
    row = np.arange(logits.shape[0])[:, None]
    phi[row, index] = weight
    return phi, index.astype(np.int32), weight


def support_field(model, state, k=8, diffusion_steps=0,
                  diffusion_strength=0.55):
    logits = diffuse_scores(
        model, raw_logits(model, state),
        diffusion_steps, diffusion_strength)
    phi, index, weight = topk_partition(logits, k)
    return logits, phi, index, weight


def balance_capacity(model, state, k=8, diffusion_steps=0,
                     diffusion_strength=0.55, iterations=4, trust=2.0):
    """Semi-discrete-OT-style Newton balance on the soft ownership graph."""
    report = []
    for iteration in range(max(0, int(iterations))):
        logits, phi, _, _ = support_field(
            model, state, k, diffusion_steps, diffusion_strength)
        mass = np.sum(phi, axis=0)
        defect = state.capacity - mass
        jacobian = np.diag(mass) - phi.T @ phi
        positive = np.diag(jacobian)
        scale = max(float(np.median(positive[positive > 1e-10]))
                    if np.any(positive > 1e-10) else 1.0, 1e-8)
        system = jacobian + np.eye(len(mass)) * (1e-3 * scale)
        step = np.linalg.solve(system, defect)
        step -= float(np.mean(step))
        p95 = max(float(np.percentile(np.abs(step), 95.0)), 1e-12)
        if p95 > trust:
            step *= trust / p95
        state.bias += step
        state.bias -= float(np.mean(state.bias))
        report.append({
            "iteration": iteration,
            "relative_mass_error": float(
                np.linalg.norm(defect) /
                max(np.linalg.norm(state.capacity), 1e-12)),
            "bias_p95": float(np.percentile(np.abs(state.bias), 95.0)),
        })
    return report


def affine_design(model, top_index, top_weight, cells):
    """Top-K partition-of-unity design with a shared global affine frame."""
    pixels, k = top_index.shape
    x = (model.xf - 0.5 * (model.w - 1)) / max(model.w, model.h)
    y = (model.yf - 0.5 * (model.h - 1)) / max(model.w, model.h)
    basis = np.column_stack((np.ones(model.npix), x, y))
    rows = np.repeat(np.arange(pixels, dtype=np.int64), k * 3)
    columns = (
        3 * top_index[:, :, None] +
        np.arange(3, dtype=np.int32)[None, None, :]).reshape(-1)
    data = (top_weight[:, :, None] * basis[:, None, :]).reshape(-1)
    return sparse.coo_matrix(
        (data, (rows, columns)),
        shape=(pixels, 3 * int(cells))).tocsr()


def compatible_laplacian(phi, coeff, strength):
    """Continuous graph fusion: compatible jets attract but are never tied."""
    cells = phi.shape[1]
    if strength <= 0.0:
        return sparse.csc_matrix((3 * cells, 3 * cells))
    overlap = phi.T @ phi
    np.fill_diagonal(overlap, 0.0)
    active = overlap > max(float(np.max(overlap)) * 1e-4, 1e-10)
    if not np.any(active):
        return sparse.csc_matrix((3 * cells, 3 * cells))
    delta = coeff[:, None, :] - coeff[None, :, :]
    coordinate_weight = np.repeat([1.0, 0.25, 0.25], 3)
    mismatch = np.sum(
        delta * delta * coordinate_weight[None, None, :], axis=2)
    scale = max(float(np.median(mismatch[active])), 1e-10)
    affinity = overlap * np.exp(-mismatch / (2.0 * scale))
    affinity[~active] = 0.0
    affinity = 0.5 * (affinity + affinity.T)
    laplacian = sparse.diags(np.sum(affinity, axis=1)) - sparse.csr_matrix(
        affinity)
    channel_metric = sparse.diags([1.0, 0.25, 0.25])
    return float(strength) * sparse.kron(
        laplacian, channel_metric, format="csc")


def solve_support(model, state, objective=None, k=8, diffusion_steps=0,
                  diffusion_strength=0.55, fusion_strength=0.0):
    logits, phi, top_index, top_weight = support_field(
        model, state, k, diffusion_steps, diffusion_strength)
    cells = len(state.seeds)
    design = affine_design(model, top_index, top_weight, cells)
    regularization = np.tile([1e-6, 2e-4, 2e-4], cells)
    gram = (design.T @ design).tocsc() + sparse.diags(regularization)
    target = model.lab.reshape(-1, 3)
    rhs = np.asarray(design.T @ target)
    factor = splu(
        gram, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
        options={"SymmetricMode": True})
    coeff = factor.solve(rhs).reshape(cells, 3, 3)

    if fusion_strength > 0.0:
        # IRLS-like single reweight: compatibility is measured from the
        # unconstrained optimum, then enters the same normal equations.
        graph_penalty = compatible_laplacian(
            phi, coeff.reshape(cells, 9), fusion_strength)
        # Apply the same site graph separately to each output channel.  The
        # affine-coordinate penalty is already encoded in graph_penalty.
        gram = gram + graph_penalty
        factor = splu(
            gram, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
            options={"SymmetricMode": True})
        coeff = factor.solve(rhs).reshape(cells, 3, 3)

    reconstruction = np.asarray(
        design @ coeff.reshape(3 * cells, 3)).reshape(model.h, model.w, 3)
    rgb = np.clip(lab_to_srgb(reconstruction), 0.0, 1.0)
    rgb_mse = float(np.mean((model.rgb - rgb) ** 2))
    psnr = -10.0 * math.log10(max(rgb_mse, 1e-12))
    measured = objective.evaluate(rgb) if objective is not None else None
    metrics = geometry_metrics(model, phi, top_index, top_weight)
    return SupportFit(
        logits, phi, top_index, top_weight, design, gram, rhs, coeff,
        reconstruction, rgb, psnr, rgb_mse, measured, metrics)


def evolve_support_competition(
        model, state, k=8, diffusion_steps=0, diffusion_strength=0.55,
        iterations=3, capacity_weight=0.08, survival_weight=0.0,
        trust=1.0):
    """Gauss-Newton bias evolution with capacity and survival pressure.

    The data Jacobian is exact for the current fixed candidate set. Capacity
    is the boundary budget: it prevents one explanation from purchasing the
    whole image. The concave square-root mass cost is an IRL1 survival term;
    it makes already-small redundant supports more expensive and can drive
    them continuously toward zero.
    """
    trace = []
    lab_weight = np.sqrt(np.array([1.0, 1.5, 1.5]))
    x = (model.xf - 0.5 * (model.w - 1)) / max(model.w, model.h)
    y = (model.yf - 0.5 * (model.h - 1)) / max(model.w, model.h)
    basis = np.column_stack((np.ones(model.npix), x, y))
    target = model.lab.reshape(-1, 3)
    cells = len(state.seeds)

    for iteration in range(max(0, int(iterations))):
        fit = solve_support(
            model, state, None, k, diffusion_steps,
            diffusion_strength, 0.0)
        site_coeff = fit.coeff[fit.top_index]
        prediction = np.einsum(
            "pa,pkac->pkc", basis, site_coeff, optimize=False)
        rendered = fit.reconstruction.reshape(-1, 3)
        derivative = fit.top_weight[..., None] * (
            prediction - rendered[:, None, :])
        derivative *= lab_weight[None, None, :]
        rows = np.broadcast_to(
            3 * np.arange(model.npix, dtype=np.int64)[:, None, None] +
            np.arange(3, dtype=np.int64)[None, None, :],
            derivative.shape).reshape(-1)
        columns = np.repeat(fit.top_index[..., None], 3, axis=2).reshape(-1)
        jacobian = sparse.coo_matrix(
            (derivative.reshape(-1), (rows, columns)),
            shape=(3 * model.npix, cells)).tocsr()
        residual = ((rendered - target) *
                    lab_weight[None, :]).reshape(-1)
        gradient = 2.0 * np.asarray(jacobian.T @ residual).ravel()
        hessian = 2.0 * np.asarray(
            (jacobian.T @ jacobian).toarray())

        mass = np.sum(fit.phi, axis=0)
        mass_jacobian = np.diag(mass) - fit.phi.T @ fit.phi
        if capacity_weight > 0.0:
            inverse_capacity = 1.0 / np.maximum(state.capacity, 1.0)
            defect = (mass - state.capacity) * inverse_capacity
            gradient += (
                float(capacity_weight) * mass_jacobian @ defect)
            hessian += (
                float(capacity_weight) *
                (mass_jacobian * inverse_capacity[None, :]) @
                mass_jacobian)
        if survival_weight > 0.0:
            marginal = (
                0.5 * float(survival_weight) /
                np.sqrt(np.maximum(mass, 1e-3)))
            gradient += mass_jacobian @ marginal

        diagonal = np.diag(hessian)
        scale = max(float(np.median(diagonal[diagonal > 1e-12]))
                    if np.any(diagonal > 1e-12) else 1.0, 1e-9)
        hessian += np.eye(cells) * (1e-3 * scale)
        step = np.linalg.solve(hessian, -gradient)
        step -= float(np.mean(step))
        p95 = max(float(np.percentile(np.abs(step), 95.0)), 1e-12)
        if p95 > trust:
            step *= float(trust) / p95
        state.bias += step
        state.bias -= float(np.mean(state.bias))
        trace.append({
            "iteration": iteration,
            "psnr": fit.psnr,
            "mass_cv": float(
                np.std(mass) / max(float(np.mean(mass)), 1e-12)),
            "minimum_capacity_ratio": float(np.min(
                mass / np.maximum(state.capacity, 1.0))),
            "step_p95": float(np.percentile(np.abs(step), 95.0)),
            "gradient_norm": float(np.linalg.norm(gradient)),
        })
    return trace


def delete_extinguished_supports(state, phi, minimum_capacity_ratio=0.15):
    """Hard deletion only after soft competition has removed the support."""
    mass = np.sum(phi, axis=0)
    ratio = mass / np.maximum(state.capacity, 1.0)
    keep = ratio >= float(minimum_capacity_ratio)
    if np.sum(keep) < 3:
        keep[np.argsort(mass)[-3:]] = True
    removed = np.flatnonzero(~keep)
    if removed.size == 0:
        return state.clone(), removed
    reduced = SupportState(
        seeds=state.seeds[keep].copy(),
        angles=state.angles[keep].copy(),
        ratios=state.ratios[keep].copy(),
        bias=state.bias[keep].copy(),
        temperature=state.temperature[keep].copy(),
        capacity=state.capacity[keep].copy())
    reduced.bias -= float(np.mean(reduced.bias))
    reduced.capacity *= np.sum(state.capacity) / np.sum(reduced.capacity)
    return reduced, removed


def evolve_site_positions(
        model, state, k=8, diffusion_steps=0, diffusion_strength=0.55,
        iterations=2, trust=0.3, rebalance_steps=2):
    """Move sites by the receiver Jacobian, accepting only measured descent."""
    trace = []
    lab_weight = np.array([1.0, 1.5, 1.5])
    x_global = (
        model.xf - 0.5 * (model.w - 1)) / max(model.w, model.h)
    y_global = (
        model.yf - 0.5 * (model.h - 1)) / max(model.w, model.h)
    basis = np.column_stack((np.ones(model.npix), x_global, y_global))
    target = model.lab.reshape(-1, 3)
    spacing = math.sqrt(model.npix / max(len(state.seeds), 1))

    for iteration in range(max(0, int(iterations))):
        baseline = solve_support(
            model, state, None, k, diffusion_steps,
            diffusion_strength, 0.0)
        index = baseline.top_index
        coeff = baseline.coeff[index]
        prediction = np.einsum(
            "pa,pkac->pkc", basis, coeff, optimize=False)
        rendered = baseline.reconstruction.reshape(-1, 3)
        common = baseline.top_weight[..., None] * (
            prediction - rendered[:, None, :])

        sx = state.seeds[index, 0]
        sy = state.seeds[index, 1]
        dx = (model.xf[:, None] - sx) / spacing
        dy = (model.yf[:, None] - sy) / spacing
        cosine = np.cos(state.angles[index])
        sine = np.sin(state.angles[index])
        ratio = np.maximum(state.ratios[index], 1.0)
        tangent = dx * cosine + dy * sine
        normal = -dx * sine + dy * cosine
        distance = np.sqrt(
            tangent * tangent / ratio +
            normal * normal * ratio + 1e-12)
        temperature = state.temperature[index]
        score_x = (
            temperature / spacing *
            (tangent * cosine / ratio - normal * ratio * sine) /
            distance)
        score_y = (
            temperature / spacing *
            (tangent * sine / ratio + normal * ratio * cosine) /
            distance)
        derivative_x = common * score_x[..., None]
        derivative_y = common * score_y[..., None]
        residual = rendered - target
        grad_x = 2.0 * np.sum(
            residual[:, None, :] * derivative_x *
            lab_weight[None, None, :], axis=2)
        grad_y = 2.0 * np.sum(
            residual[:, None, :] * derivative_y *
            lab_weight[None, None, :], axis=2)
        h_xx = 2.0 * np.sum(
            derivative_x * derivative_x *
            lab_weight[None, None, :], axis=2)
        h_xy = 2.0 * np.sum(
            derivative_x * derivative_y *
            lab_weight[None, None, :], axis=2)
        h_yy = 2.0 * np.sum(
            derivative_y * derivative_y *
            lab_weight[None, None, :], axis=2)

        cells = len(state.seeds)
        gradient = np.zeros((cells, 2), dtype=np.float64)
        hessian = np.zeros((cells, 2, 2), dtype=np.float64)
        flat = index.ravel()
        np.add.at(gradient[:, 0], flat, grad_x.ravel())
        np.add.at(gradient[:, 1], flat, grad_y.ravel())
        np.add.at(hessian[:, 0, 0], flat, h_xx.ravel())
        np.add.at(hessian[:, 0, 1], flat, h_xy.ravel())
        np.add.at(hessian[:, 1, 0], flat, h_xy.ravel())
        np.add.at(hessian[:, 1, 1], flat, h_yy.ravel())
        diagonal = np.concatenate(
            (hessian[:, 0, 0], hessian[:, 1, 1]))
        damping = 1e-3 * max(
            float(np.median(diagonal[diagonal > 1e-12]))
            if np.any(diagonal > 1e-12) else 1.0, 1e-8)
        hessian[:, 0, 0] += damping
        hessian[:, 1, 1] += damping
        step = np.zeros_like(gradient)
        for site in range(cells):
            try:
                step[site] = np.linalg.solve(
                    hessian[site], -gradient[site])
            except np.linalg.LinAlgError:
                pass
        length = np.sqrt(np.sum(step * step, axis=1))
        limit = max(float(trust) * spacing, 1e-8)
        scale = np.minimum(1.0, limit / np.maximum(length, 1e-12))
        step *= scale[:, None]

        accepted = None
        for alpha in (1.0, 0.5):
            proposal = state.clone()
            proposal.seeds += alpha * step
            proposal.seeds[:, 0] = np.clip(
                proposal.seeds[:, 0], 0.0, model.w - 1.0)
            proposal.seeds[:, 1] = np.clip(
                proposal.seeds[:, 1], 0.0, model.h - 1.0)
            balance_capacity(
                model, proposal, k, diffusion_steps,
                diffusion_strength, rebalance_steps, 1.0)
            fitted = solve_support(
                model, proposal, None, k, diffusion_steps,
                diffusion_strength, 0.0)
            if fitted.rgb_mse < baseline.rgb_mse:
                accepted = proposal, fitted, alpha
                break
        if accepted is None:
            trace.append({
                "iteration": iteration, "accepted_alpha": 0.0,
                "psnr": baseline.psnr,
                "step_p95": float(np.percentile(length, 95.0)),
            })
            break
        proposal, fitted, alpha = accepted
        state.seeds = proposal.seeds
        state.bias = proposal.bias
        trace.append({
            "iteration": iteration, "accepted_alpha": alpha,
            "psnr": fitted.psnr,
            "gain_db": fitted.psnr - baseline.psnr,
            "step_p95": float(np.percentile(
                alpha * np.sqrt(np.sum(step * step, axis=1)), 95.0)),
        })
    return trace


def adapt_temperature(model, state, phi, base=4.0):
    """Hard at measured barriers, soft in uncertain smooth interiors."""
    uncertainty = phi * (1.0 - phi)
    denominator = np.sum(uncertainty, axis=0)
    edge = (
        uncertainty.T @ model.edge_strength.ravel() /
        np.maximum(denominator, 1e-12))
    lo, hi = np.percentile(edge, (15.0, 85.0))
    normalized = np.clip((edge - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    state.temperature = float(base) * (0.55 + 1.45 * normalized)
    return edge


def adapt_anisotropy(model, state, phi, reconstruction, amount=0.55):
    """Use residual demand covariance, with BFFT orientation as the prior."""
    residual = model.lab - reconstruction
    demand = np.sum(
        residual * residual *
        np.array([1.0, 1.5, 1.5])[None, None, :], axis=2).ravel()
    scale = max(float(np.percentile(demand, 90.0)), 1e-12)
    demand = 0.08 + np.clip(demand / scale, 0.0, 3.0)
    x = model.xf
    y = model.yf
    for site in range(len(state.seeds)):
        weight = phi[:, site] * demand
        mass = float(np.sum(weight))
        if mass < 1e-8:
            continue
        cx = float(weight @ x / mass)
        cy = float(weight @ y / mass)
        dx, dy = x - cx, y - cy
        covariance = np.array([
            [float(weight @ (dx * dx)), float(weight @ (dx * dy))],
            [float(weight @ (dx * dy)), float(weight @ (dy * dy))],
        ]) / mass
        value, vector = np.linalg.eigh(covariance + np.eye(2) * 1e-6)
        direction = vector[:, int(np.argmax(value))]
        learned_angle = math.atan2(float(direction[1]), float(direction[0]))
        learned_ratio = math.sqrt(
            max(float(value.max()), 1e-8) /
            max(float(value.min()), 1e-8))
        learned_ratio = float(np.clip(learned_ratio, 1.0, 12.0))
        old = state.angles[site]
        c = ((1.0 - amount) * math.cos(2.0 * old) +
             amount * math.cos(2.0 * learned_angle))
        s = ((1.0 - amount) * math.sin(2.0 * old) +
             amount * math.sin(2.0 * learned_angle))
        state.angles[site] = 0.5 * math.atan2(s, c)
        state.ratios[site] = math.exp(
            (1.0 - amount) * math.log(max(state.ratios[site], 1.0)) +
            amount * math.log(learned_ratio))


def geometry_metrics(model, phi, top_index, top_weight):
    hard = np.take_along_axis(
        top_index, np.argmax(top_weight, axis=1)[:, None], axis=1
    ).reshape(model.h, model.w)
    horizontal = hard[:, 1:] != hard[:, :-1]
    vertical = hard[1:, :] != hard[:-1, :]
    boundary_count = int(np.sum(horizontal) + np.sum(vertical))
    edge_sum = (
        float(np.sum(model.edge_strength[:, 1:][horizontal])) +
        float(np.sum(model.edge_strength[1:, :][vertical])))
    components = []
    for site in np.unique(hard):
        _, count = ndi.label(hard == site)
        components.append(count)
    entropy = -np.sum(
        top_weight * np.log(np.maximum(top_weight, 1e-12)), axis=1)
    mass = np.sum(phi, axis=0)
    overlap = phi.T @ phi
    np.fill_diagonal(overlap, 0.0)
    return {
        "boundary_edges": boundary_count,
        "boundary_edge_alignment": edge_sum / max(boundary_count, 1),
        "mean_entropy": float(np.mean(entropy)),
        "mass_cv": float(np.std(mass) / max(np.mean(mass), 1e-12)),
        "fragmented_cells": int(np.sum(np.asarray(components) > 1)),
        "extra_components": int(np.sum(np.asarray(components) - 1)),
        "mean_max_overlap": float(np.mean(np.max(overlap, axis=1))),
    }


def grow_model(image, side, cells, passes, flow_sweeps):
    cfg = Config(
        max_side=side, passes=passes, flow_sweeps=flow_sweeps,
        initial_cells=min(96, cells), max_cells=cells,
        split_batch=min(36, max(1, cells - min(96, cells))),
        territory_count=1, marked_cells=False,
        allocation_mode="Expected affine gain")
    model = ReceiverGuidedVoronoi(image, cfg)
    while len(model.seeds) < cells:
        model.step_direct(
            split=True, cartoon_softness=4.0, texture_softness=16.0)
    return model


def run_variants(model, args):
    objective = SingleStageDecompositionObjective(
        model.rgb, lam=model.cfg.lam, mu=model.cfg.mu,
        passes=model.cfg.passes, threads=4)
    state0 = initial_state(model, args.temperature)
    variants = []

    def measure(name, state, diffusion=0, fusion=0.0, notes=None):
        started = time.perf_counter()
        fit = solve_support(
            model, state, objective, args.top_k, diffusion,
            args.diffusion_strength, fusion)
        record = {
            "name": name,
            "elapsed_s": time.perf_counter() - started,
            "psnr": fit.psnr,
            "rgb_mse": fit.rgb_mse,
            "objective": fit.objective["objective"],
            "cartoon_mse": fit.objective["cartoon_mse"],
            "texture_mse": fit.objective["texture_mse"],
            **fit.metrics,
        }
        if notes:
            record["notes"] = notes
        variants.append(record)
        return fit

    fixed = state0.clone()
    measure("continuous anisotropic top-k", fixed)

    diffused = state0.clone()
    diffuse_fit = measure(
        "transport-diffused support", diffused, args.diffusion_steps)

    capacity = state0.clone()
    capacity_trace = balance_capacity(
        model, capacity, args.top_k, args.diffusion_steps,
        args.diffusion_strength, args.capacity_steps, args.capacity_trust)
    capacity_fit = measure(
        "diffusion + capacity", capacity, args.diffusion_steps,
        notes={"capacity_trace": capacity_trace})

    temperature_only = capacity.clone()
    adapt_temperature(
        model, temperature_only, capacity_fit.phi, args.temperature)
    temperature_trace = balance_capacity(
        model, temperature_only, args.top_k, args.diffusion_steps,
        args.diffusion_strength, args.capacity_steps,
        args.capacity_trust)
    temperature_fit = measure(
        "adaptive temperature only", temperature_only,
        args.diffusion_steps,
        notes={"capacity_trace": temperature_trace})

    anisotropy_only = capacity.clone()
    adapt_anisotropy(
        model, anisotropy_only, capacity_fit.phi,
        capacity_fit.reconstruction, args.anisotropy_amount)
    anisotropy_trace = balance_capacity(
        model, anisotropy_only, args.top_k, args.diffusion_steps,
        args.diffusion_strength, args.capacity_steps,
        args.capacity_trust)
    anisotropy_fit = measure(
        "residual anisotropy only", anisotropy_only,
        args.diffusion_steps,
        notes={"capacity_trace": anisotropy_trace})

    adaptive = capacity.clone()
    adapt_temperature(
        model, adaptive, capacity_fit.phi, args.temperature)
    adapt_anisotropy(
        model, adaptive, capacity_fit.phi,
        capacity_fit.reconstruction, args.anisotropy_amount)
    adaptive_trace = balance_capacity(
        model, adaptive, args.top_k, args.diffusion_steps,
        args.diffusion_strength, args.capacity_steps,
        args.capacity_trust)
    adaptive_fit = measure(
        "adaptive temperature + residual anisotropy",
        adaptive, args.diffusion_steps,
        notes={"capacity_trace": adaptive_trace})

    competitive = adaptive.clone()
    competition_trace = evolve_support_competition(
        model, competitive, args.top_k, args.diffusion_steps,
        args.diffusion_strength, args.competition_steps,
        args.capacity_weight, args.survival_weight,
        args.competition_trust)
    competitive_fit = measure(
        "receiver competition + survival", competitive,
        args.diffusion_steps,
        notes={"competition_trace": competition_trace})

    transported = competitive.clone()
    position_trace = evolve_site_positions(
        model, transported, args.top_k, args.diffusion_steps,
        args.diffusion_strength, args.position_steps,
        args.position_trust, max(1, args.capacity_steps // 2))
    transported_fit = measure(
        "receiver-transported site positions", transported,
        args.diffusion_steps,
        notes={"position_trace": position_trace})

    reduced, removed = delete_extinguished_supports(
        transported, transported_fit.phi, args.deletion_ratio)
    if removed.size:
        balance_capacity(
            model, reduced, args.top_k, args.diffusion_steps,
            args.diffusion_strength, args.capacity_steps,
            args.capacity_trust)
        measure(
            "hard delete extinguished supports", reduced,
            args.diffusion_steps,
            notes={"removed": removed.tolist()})
    else:
        variants.append({
            "name": "hard delete extinguished supports",
            "skipped": True,
            "reason": "no support fell below the measured mass threshold",
            "removed": 0,
        })

    measure(
        "joint support + soft graph fusion", adaptive,
        args.diffusion_steps, args.fusion_strength)
    return variants, {
        "fixed": fixed, "diffused": diffused,
        "capacity": capacity, "temperature_only": temperature_only,
        "anisotropy_only": anisotropy_only, "adaptive": adaptive,
        "competitive": competitive, "transported": transported,
    }, {
        "diffuse": diffuse_fit, "capacity": capacity_fit,
        "adaptive": adaptive_fit, "temperature_only": temperature_fit,
        "anisotropy_only": anisotropy_fit,
        "competitive": competitive_fit, "transported": transported_fit,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="+", default=["pikachu", "camera"])
    parser.add_argument("--max-side", type=int, default=96)
    parser.add_argument("--cells", type=int, default=180)
    parser.add_argument("--passes", type=int, default=4)
    parser.add_argument("--flow-sweeps", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--diffusion-steps", type=int, default=3)
    parser.add_argument("--diffusion-strength", type=float, default=0.55)
    parser.add_argument("--capacity-steps", type=int, default=4)
    parser.add_argument("--capacity-trust", type=float, default=2.0)
    parser.add_argument("--anisotropy-amount", type=float, default=0.55)
    parser.add_argument("--fusion-strength", type=float, default=0.0001)
    parser.add_argument("--competition-steps", type=int, default=3)
    parser.add_argument("--capacity-weight", type=float, default=0.08)
    parser.add_argument("--survival-weight", type=float, default=0.002)
    parser.add_argument("--competition-trust", type=float, default=1.0)
    parser.add_argument("--deletion-ratio", type=float, default=0.15)
    parser.add_argument("--position-steps", type=int, default=2)
    parser.add_argument("--position-trust", type=float, default=0.3)
    args = parser.parse_args()

    results = []
    for image_name in args.images:
        started = time.perf_counter()
        model = grow_model(
            gallery.load(image_name), args.max_side, args.cells,
            args.passes, args.flow_sweeps)
        variants, _, _ = run_variants(model, args)
        result = {
            "image": image_name,
            "working_shape": [model.h, model.w],
            "cells": len(model.seeds),
            "baseline_psnr": float(model.psnr),
            "variants": variants,
            "elapsed_s": time.perf_counter() - started,
        }
        results.append(result)
        print(
            f"{image_name}: baseline {model.psnr:.3f} dB; " +
            "; ".join(
                (f"{item['name']} {item['psnr']:.3f}"
                 if "psnr" in item else
                 f"{item['name']} skipped")
                for item in variants),
            file=sys.stderr)
    print(json.dumps({
        "protocol": vars(args),
        "results": results,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
