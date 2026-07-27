#!/usr/bin/env python3
"""Soft-to-hard Wasserstein bifurcation from one fixed BFFT decomposition.

This experiment deliberately separates target representation from allocation.
Exactly one finished Meyer/TGFD split of the target supplies a support measure
and a physical metric.  Intermediate solver iterates are *not* treated as
scale-time and the reconstruction is never recursively decomposed.

One allocation atom begins at the global barycenter.  A leaf bifurcates when
its conditional transport covariance cannot be covered by the fixed metric:

    lambda_max(C_i Q_i) > kappa.

The two branches are obtained by capacity-balanced soft transport along the
unstable generalized eigenvector.  Their soft barycentres define two power
sites, whose additive bias is then sharpened to one hard partition.  Population
is therefore a property of instability in the allocation plan, not pixel
count, residual rank, candidate selection, or a requested cell budget.

Every pixel has one hard cell ID.  The final RGB/Lab readout is an independent
affine jet per region, assembled with linear bincount reductions.  No sparse
global factorization and no overlapping support raster are used.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from scipy.special import expit

try:
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "viewer"))
sys.path.insert(0, str(ROOT / "experiments"))

import gallery  # noqa: E402
import bfft  # noqa: E402
from bfft.vision import (  # noqa: E402
    SingleStageDecompositionObjective,
    measure_residual_ridges,
)
from dual_aperture_support import score  # noqa: E402
from transport_measure_cells import site_id_colours  # noqa: E402
from transport_voronoi import (  # noqa: E402
    _dijkstra_two_best_packed,
    _fit_rgb,
    srgb_to_lab,
)


def _dijkstra_two_best_bucket_adapter(
    seed_x: np.ndarray,
    seed_y: np.ndarray,
    reach: np.ndarray,
    costs: np.ndarray,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exact Dial queue from the sigma optimization study.

    The study kernel also returns predecessor metadata used by receiver
    gradients.  Allocation needs only the first two labels and distances.
    """
    from experiments.sigma_opt.opt_dijkstra_bucket import (
        _dijkstra_bucket,
        queue_geometry,
    )

    seed_pixel = (
        seed_y.astype(np.int64) * width + seed_x.astype(np.int64))
    scale = np.ones(height * width, dtype=np.float64)
    delta, span, shift = queue_geometry(costs, scale, reach)
    result = _dijkstra_bucket(
        seed_pixel,
        reach,
        costs,
        scale,
        height,
        width,
        delta,
        span,
        shift,
    )
    return result[0], result[1], result[2], result[3]


def _identity(function):  # pragma: no cover
    return function


_compile = njit(cache=True) if njit is not None else _identity


def _bincount(
    labels: np.ndarray,
    values: np.ndarray,
    cells: int,
) -> np.ndarray:
    return np.bincount(
        labels, weights=values, minlength=cells).astype(np.float64)


def _moments(
    labels: np.ndarray,
    measure: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    cells: int,
) -> dict[str, np.ndarray]:
    mass = _bincount(labels, measure, cells)
    safe = np.maximum(mass, 1e-30)
    cx = _bincount(labels, measure * x, cells) / safe
    cy = _bincount(labels, measure * y, cells) / safe
    dx = x - cx[labels]
    dy = y - cy[labels]
    return {
        "mass": mass,
        "cx": cx,
        "cy": cy,
        "cxx": _bincount(labels, measure * dx * dx, cells) / safe,
        "cxy": _bincount(labels, measure * dx * dy, cells) / safe,
        "cyy": _bincount(labels, measure * dy * dy, cells) / safe,
    }


def _physical_precision(
    qxx: np.ndarray,
    qxy: np.ndarray,
    qyy: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray(qxx, dtype=np.float64) * width * width,
        np.asarray(qxy, dtype=np.float64) * width * height,
        np.asarray(qyy, dtype=np.float64) * height * height,
    )


def single_decomposition_geometry(
    rgb: np.ndarray,
    *,
    lam: float = 0.05,
    mu: float = 40.0,
    tgfd_sweeps: int = 24,
    flow_sweeps: int = 24,
    max_support_fraction: float = 0.18,
    coherent_tangent_fraction: float = 0.02,
    threads: int = 4,
    meyer_solver: int = 1,
) -> dict[str, np.ndarray | float]:
    """One frozen BFFT support measure and metric.

    ``tgfd_sweeps`` is the internal convergence work of the one split, not a
    segmentation stage axis.  The additional ROF map evaluates the fixed
    cartoon-side outer defect (the glass state) without decomposing a changed
    target.
    """
    lab = srgb_to_lab(rgb)
    light = lab[..., 0] * 255.0
    cartoon, texture = bfft.meyer_split(
        light,
        lam=lam,
        mu=mu,
        passes=tgfd_sweeps,
        threads=threads,
        solver=meyer_solver,
    )
    projected = bfft.rof(
        light - texture,
        c=lam,
        eta=2.0 * lam,
        sweeps=flow_sweeps,
        tol=0.0,
        threads=threads,
        solver=meyer_solver,
    )
    cartoon = cartoon / 255.0
    texture = texture / 255.0
    glass = (cartoon * 255.0 - projected) / 255.0

    # The tensor is amplitude-normalized: it measures local inverse support
    # scale rather than merely preferring high-contrast edges.  Cartoon and
    # glass carry region topology; texture supplies fine directional demand.
    smooth_cartoon = ndi.gaussian_filter(
        cartoon, 3.0, mode="reflect")
    channels = (
        cartoon - smooth_cartoon,
        math.sqrt(0.65) * texture,
        math.sqrt(0.70) * glass,
    )
    energy = np.zeros_like(cartoon, dtype=np.float64)
    jxx = np.zeros_like(cartoon, dtype=np.float64)
    jxy = np.zeros_like(cartoon, dtype=np.float64)
    jyy = np.zeros_like(cartoon, dtype=np.float64)
    for channel in channels:
        energy += channel * channel
        gx = ndi.sobel(channel, axis=1, mode="reflect") / 8.0
        gy = ndi.sobel(channel, axis=0, mode="reflect") / 8.0
        jxx += gx * gx
        jxy += gx * gy
        jyy += gy * gy
    energy = ndi.gaussian_filter(energy, 1.5, mode="reflect")
    jxx = ndi.gaussian_filter(jxx, 1.5, mode="reflect")
    jxy = ndi.gaussian_filter(jxy, 1.5, mode="reflect")
    jyy = ndi.gaussian_filter(jyy, 1.5, mode="reflect")

    scale = max(float(np.percentile(energy, 99.5)), 1e-20)
    transport_reliability = energy / (energy + 1e-5 * scale)

    # A Meyer/G-norm response is nonlocal: a genuine contour can carry a
    # diminishing transport field well into an otherwise constant panel.
    # That field is useful orientation, but it is not local evidence for new
    # allocation mass.  Gate its magnitude by variation in the unchanged
    # source.  This makes a flat white/black region broad again while keeping
    # the BFFT direction at its actual boundary.
    source_activity = np.zeros_like(cartoon, dtype=np.float64)
    for channel in np.moveaxis(lab, -1, 0):
        local = channel - ndi.gaussian_filter(
            channel, 1.5, mode="reflect")
        gx = ndi.sobel(channel, axis=1, mode="reflect") / 8.0
        gy = ndi.sobel(channel, axis=0, mode="reflect") / 8.0
        source_activity += local * local + 0.35 * (gx * gx + gy * gy)
    source_activity = ndi.gaussian_filter(
        source_activity, 1.0, mode="reflect")
    source_scale = max(
        float(np.percentile(source_activity, 99.5)), 1e-20)
    source_reliability = source_activity / (
        source_activity + 1e-5 * source_scale)
    reliability = transport_reliability * source_reliability
    denominator = energy + 1e-5 * scale
    qxx = reliability * jxx / denominator
    qxy = reliability * jxy / denominator
    qyy = reliability * jyy / denominator

    height, width = cartoon.shape
    max_length = max(
        float(max_support_fraction) * max(height, width), 1.0)
    frequency_floor = 1.0 / (max_length * max_length)
    qxx += frequency_floor
    qyy += frequency_floor

    # A coherent contour should be expensive across its normal but cheap
    # along its tangent.  Numerical curvature and channel disagreement give
    # the raw tensor a spurious second eigenvalue, which otherwise turns one
    # long edge into a necklace of round cells.  Isotropic texture retains
    # both eigenvalues because its coherence is low.
    trace = qxx + qyy
    disc = np.hypot(qxx - qyy, 2.0 * qxy)
    coherence = disc / np.maximum(trace, 1e-30)
    high = np.maximum(0.5 * (trace + disc), frequency_floor)
    low = np.maximum(0.5 * (trace - disc), frequency_floor)
    tangent_fraction = float(np.clip(
        coherent_tangent_fraction, 0.0, 1.0))
    low_factor = 1.0 - (1.0 - tangent_fraction) * coherence
    low = frequency_floor + low_factor * (low - frequency_floor)
    normal_angle = 0.5 * np.arctan2(2.0 * qxy, qxx - qyy)
    nx = np.cos(normal_angle)
    ny = np.sin(normal_angle)
    tx = -ny
    ty = nx
    qxx = high * nx * nx + low * tx * tx
    qxy = high * nx * ny + low * tx * ty
    qyy = high * ny * ny + low * ty * ty

    determinant = np.maximum(qxx * qyy - qxy * qxy, 0.0)
    measure = np.sqrt(determinant) / math.pi
    implied_cells = float(np.sum(measure))
    measure /= max(implied_cells, 1e-30)
    return {
        "measure": measure,
        "precision_xx": qxx,
        "precision_xy": qxy,
        "precision_yy": qyy,
        "cartoon": cartoon,
        "texture": texture,
        "glass": glass,
        "energy": energy,
        "source_reliability": source_reliability,
        "implied_cells": implied_cells,
        "max_support_px": max_length,
        "coherent_tangent_fraction": tangent_fraction,
        "target_decompositions": 1.0,
    }


def pyramid_geometry(geometry: dict, maximum_side: int) -> dict:
    """Restrict one frozen support geometry to a coarser transport grid.

    This is not another decomposition.  The measure and metric are sampled
    from the already-finished target geometry, and final sites are prolonged
    in normalized coordinates to the full-resolution solve.
    """
    source = np.asarray(geometry["measure"])
    height, width = source.shape
    scale = min(float(maximum_side) / max(height, width), 1.0)
    if scale >= 1.0:
        return {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in geometry.items()
        }
    target_height = max(int(round(height * scale)), 2)
    target_width = max(int(round(width * scale)), 2)
    zoom = (target_height / height, target_width / width)
    coarse: dict = {}
    for key, value in geometry.items():
        if isinstance(value, np.ndarray) and value.ndim == 2:
            coarse[key] = ndi.zoom(
                value,
                zoom,
                order=1,
                mode="reflect",
                prefilter=False,
            )
        else:
            coarse[key] = value
    coarse["measure"] = np.maximum(coarse["measure"], 0.0)
    coarse["measure"] /= max(float(np.sum(coarse["measure"])), 1e-30)
    coarse["max_support_px"] = (
        float(geometry["max_support_px"]) * max(zoom))
    coarse["pyramid_source_shape"] = (height, width)
    return coarse


def _unstable_direction(
    cxx: float,
    cxy: float,
    cyy: float,
    qxx: float,
    qxy: float,
    qyy: float,
) -> tuple[float, float, float, float]:
    """Largest eigenpair of C Q; eigenvalues match Q^(1/2) C Q^(1/2)."""
    a = cxx * qxx + cxy * qxy
    b = cxx * qxy + cxy * qyy
    c = cxy * qxx + cyy * qxy
    d = cxy * qxy + cyy * qyy
    trace = a + d
    determinant = max(a * d - b * c, 0.0)
    disc = math.sqrt(max(trace * trace - 4.0 * determinant, 0.0))
    eigenvalue = max(0.5 * (trace + disc), 0.0)
    vx, vy = b, eigenvalue - a
    if abs(vx) + abs(vy) < 1e-15:
        vx, vy = eigenvalue - d, c
    norm = math.hypot(vx, vy)
    if norm < 1e-15:
        if cxx >= cyy:
            return eigenvalue, 1.0, 0.0, max(trace - eigenvalue, 0.0)
        return eigenvalue, 0.0, 1.0, max(trace - eigenvalue, 0.0)
    return (
        eigenvalue,
        vx / norm,
        vy / norm,
        max(trace - eigenvalue, 0.0),
    )


def _edge_cost_stack(
    geometry: dict,
    metric_strength: float,
) -> np.ndarray:
    """Eight-neighbour costs of the fixed BFFT support metric."""
    qxx = np.asarray(geometry["precision_xx"], dtype=np.float64)
    qxy = np.asarray(geometry["precision_xy"], dtype=np.float64)
    qyy = np.asarray(geometry["precision_yy"], dtype=np.float64)
    height, width = qxx.shape
    scale = max(float(np.percentile(qxx + qyy, 90.0)), 1e-12)
    # Q is measured per pixel while the stability radius is expressed as a
    # fraction of the support horizon.  A one-pixel discontinuity must retain
    # the same cost relative to that horizon at every resolution, hence the
    # squared-horizon conversion.
    strength = (
        max(float(metric_strength), 0.0)
        * float(geometry["max_support_px"]) ** 2
    )
    mxx = 1.0 + strength * qxx / scale
    mxy = strength * qxy / scale
    myy = 1.0 + strength * qyy / scale
    costs = np.full((8, height, width), np.inf, dtype=np.float32)
    directions = (
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    )
    for index, (dy, dx) in enumerate(directions):
        ys = slice(max(0, -dy), min(height, height - dy))
        xs = slice(max(0, -dx), min(width, width - dx))
        yd = slice(max(0, dy), min(height, height + dy))
        xd = slice(max(0, dx), min(width, width + dx))
        ax = 0.5 * (mxx[ys, xs] + mxx[yd, xd])
        axy = 0.5 * (mxy[ys, xs] + mxy[yd, xd])
        ay = 0.5 * (myy[ys, xs] + myy[yd, xd])
        costs[index, ys, xs] = np.sqrt(np.maximum(
            dx * dx * ax + 2.0 * dx * dy * axy + dy * dy * ay,
            1e-8,
        ))
    return costs


def _soft_transport_moments(
    owner: np.ndarray,
    runner: np.ndarray,
    first_distance: np.ndarray,
    second_distance: np.ndarray,
    temperature: float,
    measure: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    qxx_pixel: np.ndarray,
    qxy_pixel: np.ndarray,
    qyy_pixel: np.ndarray,
    cells: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Two-nearest Gibbs transport moments without a dense pixel×site plan."""
    valid = runner >= 0
    gap = np.zeros_like(first_distance)
    gap[valid] = second_distance[valid] - first_distance[valid]
    safe_second_distance = np.where(valid, second_distance, 0.0)
    owner_weight = np.ones_like(first_distance)
    owner_weight[valid] = expit(np.clip(
        gap[valid] / max(float(temperature), 1e-6), 0.0, 40.0))
    runner_weight = np.where(valid, 1.0 - owner_weight, 0.0)
    runner_safe = np.where(valid, runner, owner)

    def both(values: np.ndarray) -> np.ndarray:
        return (
            _bincount(owner, measure * owner_weight * values, cells)
            + _bincount(
                runner_safe,
                measure * runner_weight * values,
                cells,
            )
        )

    mass = both(np.ones_like(measure))
    safe = np.maximum(mass, 1e-30)
    cx = both(x) / safe
    cy = both(y) / safe
    dx_owner = x - cx[owner]
    dy_owner = y - cy[owner]
    dx_runner = x - cx[runner_safe]
    dy_runner = y - cy[runner_safe]

    def centered(
        owner_values: np.ndarray,
        runner_values: np.ndarray,
    ) -> np.ndarray:
        return (
            _bincount(
                owner,
                measure * owner_weight * owner_values,
                cells,
            )
            + _bincount(
                runner_safe,
                measure * runner_weight * runner_values,
                cells,
            )
        ) / safe

    moments = {
        "mass": mass,
        "cx": cx,
        "cy": cy,
        "cxx": centered(dx_owner * dx_owner, dx_runner * dx_runner),
        "cxy": centered(dx_owner * dy_owner, dx_runner * dy_runner),
        "cyy": centered(dy_owner * dy_owner, dy_runner * dy_runner),
        "transport_rms": np.sqrt(np.maximum(centered(
            first_distance * first_distance,
            safe_second_distance * safe_second_distance,
        ), 0.0)),
    }
    return (
        moments,
        both(qxx_pixel) / safe,
        both(qxy_pixel) / safe,
        both(qyy_pixel) / safe,
    )


def _balanced_branch_barycentres(
    owner: np.ndarray,
    runner: np.ndarray,
    first_distance: np.ndarray,
    second_distance: np.ndarray,
    temperature: float,
    measure: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    moments: dict[str, np.ndarray],
    direction: np.ndarray,
    split: np.ndarray,
    balance_steps: int = 14,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split each soft allocation into two conserved half-mass branches."""
    cells = len(split)
    valid_runner = runner >= 0
    gap = np.zeros_like(first_distance)
    gap[valid_runner] = (
        second_distance[valid_runner] - first_distance[valid_runner])
    owner_fraction = np.ones_like(first_distance)
    owner_fraction[valid_runner] = expit(np.clip(
        gap[valid_runner] / max(float(temperature), 1e-6),
        0.0,
        40.0,
    ))
    runner_fraction = np.where(
        valid_runner, 1.0 - owner_fraction, 0.0)
    runner_safe = np.where(valid_runner, runner, owner)
    owner_mass = measure * owner_fraction
    runner_mass = measure * runner_fraction
    owner_projection = (
        (x - moments["cx"][owner]) * direction[owner, 0]
        + (y - moments["cy"][owner]) * direction[owner, 1]
    )
    runner_projection = (
        (x - moments["cx"][runner_safe]) * direction[runner_safe, 0]
        + (y - moments["cy"][runner_safe]) * direction[runner_safe, 1]
    )

    low = np.full(cells, np.inf, dtype=np.float64)
    high = np.full(cells, -np.inf, dtype=np.float64)
    np.minimum.at(low, owner, owner_projection)
    np.maximum.at(high, owner, owner_projection)
    np.minimum.at(
        low, runner_safe[valid_runner], runner_projection[valid_runner])
    np.maximum.at(
        high, runner_safe[valid_runner], runner_projection[valid_runner])
    low[~split] = 0.0
    high[~split] = 0.0
    for _ in range(max(int(balance_steps), 1)):
        boundary = 0.5 * (low + high)
        owner_plus = owner_projection > boundary[owner]
        runner_plus = runner_projection > boundary[runner_safe]
        plus_mass = (
            _bincount(
                owner, owner_mass * owner_plus, cells)
            + _bincount(
                runner_safe, runner_mass * runner_plus, cells)
        )
        too_much = split & (
            plus_mass > 0.5 * moments["mass"])
        low[too_much] = boundary[too_much]
        high[split & ~too_much] = boundary[split & ~too_much]

    boundary = 0.5 * (low + high)
    owner_plus = owner_projection > boundary[owner]
    runner_plus = runner_projection > boundary[runner_safe]
    plus_owner_mass = owner_mass * owner_plus
    plus_runner_mass = runner_mass * runner_plus
    plus_mass = (
        _bincount(owner, plus_owner_mass, cells)
        + _bincount(runner_safe, plus_runner_mass, cells)
    )
    total_x = moments["mass"] * moments["cx"]
    total_y = moments["mass"] * moments["cy"]
    plus_x_sum = (
        _bincount(owner, plus_owner_mass * x, cells)
        + _bincount(runner_safe, plus_runner_mass * x, cells)
    )
    plus_y_sum = (
        _bincount(owner, plus_owner_mass * y, cells)
        + _bincount(runner_safe, plus_runner_mass * y, cells)
    )
    minus_mass = moments["mass"] - plus_mass
    plus = np.column_stack([
        plus_x_sum / np.maximum(plus_mass, 1e-30),
        plus_y_sum / np.maximum(plus_mass, 1e-30),
    ])
    minus = np.column_stack([
        (total_x - plus_x_sum) / np.maximum(minus_mass, 1e-30),
        (total_y - plus_y_sum) / np.maximum(minus_mass, 1e-30),
    ])
    valid = (
        split
        & (plus_mass > 1e-12)
        & (minus_mass > 1e-12)
    )
    return minus, plus, valid


@_compile
def _balanced_branch_histogram(
    owner: np.ndarray,
    runner: np.ndarray,
    first_distance: np.ndarray,
    second_distance: np.ndarray,
    temperature: float,
    measure: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    mass: np.ndarray,
    center_x: np.ndarray,
    center_y: np.ndarray,
    direction: np.ndarray,
    split: np.ndarray,
    bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-shot weighted branch medians and barycentres.

    The old implementation found each median with fourteen full-image
    bisection reductions.  Here each pixel visits its owner and runner once
    per finite assembly pass; the only scan is over ``bins`` local slots.
    """
    cells = len(split)
    pixels = len(owner)
    low = np.full(cells, np.inf)
    high = np.full(cells, -np.inf)

    for p in range(pixels):
        first = owner[p]
        projection = (
            (x[p] - center_x[first]) * direction[first, 0]
            + (y[p] - center_y[first]) * direction[first, 1])
        if projection < low[first]:
            low[first] = projection
        if projection > high[first]:
            high[first] = projection
        second = runner[p]
        if second >= 0:
            projection = (
                (x[p] - center_x[second]) * direction[second, 0]
                + (y[p] - center_y[second]) * direction[second, 1])
            if projection < low[second]:
                low[second] = projection
            if projection > high[second]:
                high[second] = projection

    histogram = np.zeros((cells, bins))
    safe_temperature = max(temperature, 1e-6)
    for p in range(pixels):
        first = owner[p]
        second = runner[p]
        first_fraction = 1.0
        second_fraction = 0.0
        if second >= 0:
            z = (second_distance[p] - first_distance[p]) / safe_temperature
            z = min(max(z, 0.0), 40.0)
            first_fraction = 1.0 / (1.0 + math.exp(-z))
            second_fraction = 1.0 - first_fraction
        projection = (
            (x[p] - center_x[first]) * direction[first, 0]
            + (y[p] - center_y[first]) * direction[first, 1])
        span = max(high[first] - low[first], 1e-30)
        slot = int((projection - low[first]) / span * bins)
        slot = min(max(slot, 0), bins - 1)
        histogram[first, slot] += measure[p] * first_fraction
        if second >= 0:
            projection = (
                (x[p] - center_x[second]) * direction[second, 0]
                + (y[p] - center_y[second]) * direction[second, 1])
            span = max(high[second] - low[second], 1e-30)
            slot = int((projection - low[second]) / span * bins)
            slot = min(max(slot, 0), bins - 1)
            histogram[second, slot] += measure[p] * second_fraction

    boundary = np.zeros(cells)
    for cell in range(cells):
        if not split[cell]:
            continue
        target = 0.5 * mass[cell]
        cumulative = 0.0
        selected = bins - 1
        for slot in range(bins):
            cumulative += histogram[cell, slot]
            if cumulative >= target:
                selected = slot
                break
        boundary[cell] = (
            low[cell]
            + (selected + 0.5) / bins * (high[cell] - low[cell]))

    plus_mass = np.zeros(cells)
    plus_x = np.zeros(cells)
    plus_y = np.zeros(cells)
    for p in range(pixels):
        first = owner[p]
        second = runner[p]
        first_fraction = 1.0
        second_fraction = 0.0
        if second >= 0:
            z = (second_distance[p] - first_distance[p]) / safe_temperature
            z = min(max(z, 0.0), 40.0)
            first_fraction = 1.0 / (1.0 + math.exp(-z))
            second_fraction = 1.0 - first_fraction
        projection = (
            (x[p] - center_x[first]) * direction[first, 0]
            + (y[p] - center_y[first]) * direction[first, 1])
        if split[first] and projection > boundary[first]:
            value = measure[p] * first_fraction
            plus_mass[first] += value
            plus_x[first] += value * x[p]
            plus_y[first] += value * y[p]
        if second >= 0:
            projection = (
                (x[p] - center_x[second]) * direction[second, 0]
                + (y[p] - center_y[second]) * direction[second, 1])
            if split[second] and projection > boundary[second]:
                value = measure[p] * second_fraction
                plus_mass[second] += value
                plus_x[second] += value * x[p]
                plus_y[second] += value * y[p]

    minus_mass = mass - plus_mass
    minus = np.zeros((cells, 2))
    plus = np.zeros((cells, 2))
    valid = np.zeros(cells, dtype=np.bool_)
    for cell in range(cells):
        if (not split[cell] or plus_mass[cell] <= 1e-12
                or minus_mass[cell] <= 1e-12):
            continue
        plus[cell, 0] = plus_x[cell] / plus_mass[cell]
        plus[cell, 1] = plus_y[cell] / plus_mass[cell]
        minus[cell, 0] = (
            mass[cell] * center_x[cell] - plus_x[cell]
        ) / minus_mass[cell]
        minus[cell, 1] = (
            mass[cell] * center_y[cell] - plus_y[cell]
        ) / minus_mass[cell]
        valid[cell] = True
    return minus, plus, valid


@_compile
def _balanced_metric_multibranch(
    owner: np.ndarray,
    runner: np.ndarray,
    first_distance: np.ndarray,
    second_distance: np.ndarray,
    temperature: float,
    measure: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    mass: np.ndarray,
    center_x: np.ndarray,
    center_y: np.ndarray,
    direction: np.ndarray,
    branch_count: np.ndarray,
    bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Emit all equal-mass branches required along each worst metric axis.

    If a cell spans ``rho`` in its largest eigen-direction and the admissible
    extent is ``kappa``, ``ceil(sqrt(rho / kappa))`` equal-width pieces are
    the first-order population required to bring that variance under kappa.
    This kernel realizes those pieces as transported *mass quantiles*, not
    uniformly spaced guesses.  Every final site is subsequently released
    into the shared global domain.
    """
    cells = len(branch_count)
    pixels = len(owner)
    low = np.full(cells, np.inf)
    high = np.full(cells, -np.inf)

    for p in range(pixels):
        first = owner[p]
        projection = (
            (x[p] - center_x[first]) * direction[first, 0]
            + (y[p] - center_y[first]) * direction[first, 1])
        if projection < low[first]:
            low[first] = projection
        if projection > high[first]:
            high[first] = projection
        second = runner[p]
        if second >= 0:
            projection = (
                (x[p] - center_x[second]) * direction[second, 0]
                + (y[p] - center_y[second]) * direction[second, 1])
            if projection < low[second]:
                low[second] = projection
            if projection > high[second]:
                high[second] = projection

    histogram = np.zeros((cells, bins))
    safe_temperature = max(temperature, 1e-6)
    for p in range(pixels):
        first = owner[p]
        second = runner[p]
        first_fraction = 1.0
        second_fraction = 0.0
        if second >= 0:
            z = (second_distance[p] - first_distance[p]) / safe_temperature
            z = min(max(z, 0.0), 40.0)
            first_fraction = 1.0 / (1.0 + math.exp(-z))
            second_fraction = 1.0 - first_fraction
        if branch_count[first] > 1:
            projection = (
                (x[p] - center_x[first]) * direction[first, 0]
                + (y[p] - center_y[first]) * direction[first, 1])
            span = max(high[first] - low[first], 1e-30)
            slot = int((projection - low[first]) / span * bins)
            slot = min(max(slot, 0), bins - 1)
            histogram[first, slot] += measure[p] * first_fraction
        if second >= 0 and branch_count[second] > 1:
            projection = (
                (x[p] - center_x[second]) * direction[second, 0]
                + (y[p] - center_y[second]) * direction[second, 1])
            span = max(high[second] - low[second], 1e-30)
            slot = int((projection - low[second]) / span * bins)
            slot = min(max(slot, 0), bins - 1)
            histogram[second, slot] += measure[p] * second_fraction

    branch_for_bin = np.zeros((cells, bins), dtype=np.int32)
    for cell in range(cells):
        branches = branch_count[cell]
        if branches <= 1:
            continue
        target = mass[cell] / branches
        cumulative = 0.0
        for slot in range(bins):
            midpoint = cumulative + 0.5 * histogram[cell, slot]
            branch = int(midpoint / max(target, 1e-30))
            branch_for_bin[cell, slot] = min(
                max(branch, 0), branches - 1)
            cumulative += histogram[cell, slot]

    offsets = np.zeros(cells + 1, dtype=np.int32)
    for cell in range(cells):
        offsets[cell + 1] = offsets[cell] + branch_count[cell]
    total = offsets[cells]
    child_mass = np.zeros(total)
    child_x = np.zeros(total)
    child_y = np.zeros(total)

    for p in range(pixels):
        first = owner[p]
        second = runner[p]
        first_fraction = 1.0
        second_fraction = 0.0
        if second >= 0:
            z = (second_distance[p] - first_distance[p]) / safe_temperature
            z = min(max(z, 0.0), 40.0)
            first_fraction = 1.0 / (1.0 + math.exp(-z))
            second_fraction = 1.0 - first_fraction

        branch = 0
        if branch_count[first] > 1:
            projection = (
                (x[p] - center_x[first]) * direction[first, 0]
                + (y[p] - center_y[first]) * direction[first, 1])
            span = max(high[first] - low[first], 1e-30)
            slot = int((projection - low[first]) / span * bins)
            slot = min(max(slot, 0), bins - 1)
            branch = branch_for_bin[first, slot]
        child = offsets[first] + branch
        value = measure[p] * first_fraction
        child_mass[child] += value
        child_x[child] += value * x[p]
        child_y[child] += value * y[p]

        if second >= 0:
            branch = 0
            if branch_count[second] > 1:
                projection = (
                    (x[p] - center_x[second]) * direction[second, 0]
                    + (y[p] - center_y[second]) * direction[second, 1])
                span = max(high[second] - low[second], 1e-30)
                slot = int((projection - low[second]) / span * bins)
                slot = min(max(slot, 0), bins - 1)
                branch = branch_for_bin[second, slot]
            child = offsets[second] + branch
            value = measure[p] * second_fraction
            child_mass[child] += value
            child_x[child] += value * x[p]
            child_y[child] += value * y[p]

    child_centers = np.zeros((total, 2))
    valid = np.zeros(total, dtype=np.bool_)
    for cell in range(cells):
        for branch in range(branch_count[cell]):
            child = offsets[cell] + branch
            if child_mass[child] <= 1e-12:
                continue
            child_centers[child, 0] = child_x[child] / child_mass[child]
            child_centers[child, 1] = child_y[child] / child_mass[child]
            valid[child] = True
    return child_centers, valid


@_compile
def _recursive_metric_bifurcation(
    owner: np.ndarray,
    runner: np.ndarray,
    first_distance: np.ndarray,
    second_distance: np.ndarray,
    temperature: float,
    measure: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    qxx_pixel: np.ndarray,
    qxy_pixel: np.ndarray,
    qyy_pixel: np.ndarray,
    force_split: np.ndarray,
    metric_limit: float,
    minimum_samples: int,
    maximum_levels: int,
    bins: int,
    safety_cells: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Resolve several metric generations inside one frozen transport plan.

    Pixel-to-site soft mass is assembled once.  Each local child then measures
    its own covariance and mean BFFT precision before deciding and orienting
    the next split.  No child inherits final ownership: only its barycentre
    survives, and all centres re-enter the next shared-domain walk.
    """
    pixels = len(owner)
    valid_runner_count = 0
    for p in range(pixels):
        if runner[p] >= 0:
            valid_runner_count += 1
    contributions = pixels + valid_runner_count
    group = np.empty(contributions, dtype=np.int32)
    pixel = np.empty(contributions, dtype=np.int32)
    weight = np.empty(contributions)
    safe_temperature = max(temperature, 1e-6)
    cursor = 0
    for p in range(pixels):
        second = runner[p]
        first_fraction = 1.0
        second_fraction = 0.0
        if second >= 0:
            z = (second_distance[p] - first_distance[p]) / safe_temperature
            z = min(max(z, 0.0), 40.0)
            first_fraction = 1.0 / (1.0 + math.exp(-z))
            second_fraction = 1.0 - first_fraction
        group[cursor] = owner[p]
        pixel[cursor] = p
        weight[cursor] = measure[p] * first_fraction
        cursor += 1
        if second >= 0:
            group[cursor] = second
            pixel[cursor] = p
            weight[cursor] = measure[p] * second_fraction
            cursor += 1

    groups = len(force_split)
    levels_used = 0
    for level in range(max(maximum_levels, 1)):
        mass = np.zeros(groups)
        sx = np.zeros(groups)
        sy = np.zeros(groups)
        sxx = np.zeros(groups)
        sxy = np.zeros(groups)
        syy = np.zeros(groups)
        sqxx = np.zeros(groups)
        sqxy = np.zeros(groups)
        sqyy = np.zeros(groups)
        samples = np.zeros(groups, dtype=np.int32)
        for item in range(contributions):
            label = group[item]
            p = pixel[item]
            value = weight[item]
            mass[label] += value
            sx[label] += value * x[p]
            sy[label] += value * y[p]
            sqxx[label] += value * qxx_pixel[p]
            sqxy[label] += value * qxy_pixel[p]
            sqyy[label] += value * qyy_pixel[p]
            samples[label] += 1
        center_x = sx / np.maximum(mass, 1e-30)
        center_y = sy / np.maximum(mass, 1e-30)
        for item in range(contributions):
            label = group[item]
            p = pixel[item]
            value = weight[item]
            dx = x[p] - center_x[label]
            dy = y[p] - center_y[label]
            sxx[label] += value * dx * dx
            sxy[label] += value * dx * dy
            syy[label] += value * dy * dy

        direction = np.zeros((groups, 2))
        split = np.zeros(groups, dtype=np.bool_)
        for label in range(groups):
            safe_mass = max(mass[label], 1e-30)
            cxx = sxx[label] / safe_mass
            cxy = sxy[label] / safe_mass
            cyy = syy[label] / safe_mass
            qxx = sqxx[label] / safe_mass
            qxy = sqxy[label] / safe_mass
            qyy = sqyy[label] / safe_mass
            a = cxx * qxx + cxy * qxy
            b = cxx * qxy + cxy * qyy
            c = cxy * qxx + cyy * qxy
            d = cxy * qxy + cyy * qyy
            trace = a + d
            determinant = max(a * d - b * c, 0.0)
            disc = math.sqrt(max(
                trace * trace - 4.0 * determinant, 0.0))
            largest = max(0.5 * (trace + disc), 0.0)
            vx = b
            vy = largest - a
            if abs(vx) + abs(vy) < 1e-15:
                vx = largest - d
                vy = c
            norm = math.hypot(vx, vy)
            if norm < 1e-15:
                if cxx >= cyy:
                    vx, vy = 1.0, 0.0
                else:
                    vx, vy = 0.0, 1.0
            else:
                vx /= norm
                vy /= norm
            direction[label, 0] = vx
            direction[label, 1] = vy
            requested = largest > metric_limit
            if level == 0 and force_split[label]:
                requested = True
            split[label] = (
                requested
                and samples[label] >= 2 * max(minimum_samples, 1)
                and mass[label] > 1e-12
            )

        split_count = 0
        for label in range(groups):
            if split[label]:
                split_count += 1
        if split_count == 0 or groups + split_count > safety_cells:
            break

        low = np.full(groups, np.inf)
        high = np.full(groups, -np.inf)
        for item in range(contributions):
            label = group[item]
            if not split[label]:
                continue
            p = pixel[item]
            projection = (
                (x[p] - center_x[label]) * direction[label, 0]
                + (y[p] - center_y[label]) * direction[label, 1])
            if projection < low[label]:
                low[label] = projection
            if projection > high[label]:
                high[label] = projection
        histogram = np.zeros((groups, bins))
        for item in range(contributions):
            label = group[item]
            if not split[label]:
                continue
            p = pixel[item]
            projection = (
                (x[p] - center_x[label]) * direction[label, 0]
                + (y[p] - center_y[label]) * direction[label, 1])
            span = max(high[label] - low[label], 1e-30)
            slot = int((projection - low[label]) / span * bins)
            slot = min(max(slot, 0), bins - 1)
            histogram[label, slot] += weight[item]
        boundary = np.zeros(groups)
        for label in range(groups):
            if not split[label]:
                continue
            half = 0.5 * mass[label]
            cumulative = 0.0
            selected = bins - 1
            for slot in range(bins):
                cumulative += histogram[label, slot]
                if cumulative >= half:
                    selected = slot
                    break
            boundary[label] = (
                low[label]
                + (selected + 0.5) / bins * (
                    high[label] - low[label]))

        negative = np.zeros(groups, dtype=np.int32)
        positive = np.zeros(groups, dtype=np.int32)
        next_group = 0
        for label in range(groups):
            negative[label] = next_group
            next_group += 1
            if split[label]:
                positive[label] = next_group
                next_group += 1
            else:
                positive[label] = negative[label]
        for item in range(contributions):
            label = group[item]
            if not split[label]:
                group[item] = negative[label]
                continue
            p = pixel[item]
            projection = (
                (x[p] - center_x[label]) * direction[label, 0]
                + (y[p] - center_y[label]) * direction[label, 1])
            if projection > boundary[label]:
                group[item] = positive[label]
            else:
                group[item] = negative[label]
        groups = next_group
        levels_used += 1

    final_mass = np.zeros(groups)
    final_x = np.zeros(groups)
    final_y = np.zeros(groups)
    for item in range(contributions):
        label = group[item]
        p = pixel[item]
        value = weight[item]
        final_mass[label] += value
        final_x[label] += value * x[p]
        final_y[label] += value * y[p]
    centers = np.zeros((groups, 2))
    valid = np.zeros(groups, dtype=np.bool_)
    for label in range(groups):
        if final_mass[label] <= 1e-12:
            continue
        centers[label, 0] = final_x[label] / final_mass[label]
        centers[label, 1] = final_y[label] / final_mass[label]
        valid[label] = True
    return centers, valid, levels_used


def bifurcate_allocation(
    geometry: dict,
    threshold: float = 4.0,
    softness_start: float = 0.20,
    softness_end: float = 0.0025,
    minimum_region_pixels: int = 12,
    safety_cells: int = 4096,
    maximum_rounds: int = 24,
    metric_strength: float = 1.5,
    centroid_relaxation: float = 0.85,
    balance_steps: int = 14,
    branch_bins: int = 64,
    exact_branch_balance: bool = True,
    initial_centers: np.ndarray | None = None,
    metric_extent_threshold: float = math.inf,
    direct_metric_branches: bool = False,
    maximum_direct_branches: int = 32,
    direct_metric_start_round: int = 0,
    direct_metric_rounds: int = 0,
    direct_metric_coherence: float = 0.0,
    direct_metric_local_levels: int = 0,
    transport_queue: str = "heap",
    transport_stencil_radius: int = 1,
    transport_model: str = "edge",
    trace_topology: bool = False,
    capture_center_history: bool = False,
) -> tuple[np.ndarray, dict, list[dict]]:
    """Globally transport, migrate, and bifurcate fixed-target allocations."""
    measure_2d = np.asarray(geometry["measure"], dtype=np.float64)
    qxx_2d = np.asarray(geometry["precision_xx"], dtype=np.float64)
    qxy_2d = np.asarray(geometry["precision_xy"], dtype=np.float64)
    qyy_2d = np.asarray(geometry["precision_yy"], dtype=np.float64)
    height, width = measure_2d.shape
    pixels = height * width
    yy, xx = np.mgrid[:height, :width]
    x = (xx.ravel().astype(np.float64) + 0.5) / width
    y = (yy.ravel().astype(np.float64) + 0.5) / height
    measure = measure_2d.ravel()
    qxx_p, qxy_p, qyy_p = _physical_precision(
        qxx_2d, qxy_2d, qyy_2d, width, height)
    qxx_p = qxx_p.ravel()
    qxy_p = qxy_p.ravel()
    qyy_p = qyy_p.ravel()
    stencil_radius = max(int(transport_stencil_radius), 1)
    if transport_model == "continuous":
        from port_needed.continuous_eikonal_transport import (
            continuous_first_partition_prepared,
            prepare_continuous_metric,
        )
        from port_needed.wide_stencil_transport import _metric_fields

        prepared_continuous = prepare_continuous_metric(
            *_metric_fields(geometry, metric_strength))
        costs = np.empty((0, height, width), dtype=np.float32)

        def walk(seed_x, seed_y, reach, edge_costs, h, w):
            normalized_centers = np.column_stack((
                (seed_x.astype(np.float64) + 0.5) / w,
                (seed_y.astype(np.float64) + 0.5) / h,
            ))
            result = continuous_first_partition_prepared(
                normalized_centers,
                prepared_continuous,
                reach=np.asarray(reach, dtype=np.float64),
            )
            first = result["distance"].ravel()
            owner_result = result["labels"].ravel()
            return (
                owner_result,
                np.full(owner_result.shape, -1, dtype=np.int32),
                first,
                np.full(first.shape, np.inf, dtype=np.float64),
            )
    elif transport_model == "edge" and stencil_radius > 1:
        from port_needed.wide_stencil_transport import (
            build_wide_edge_costs,
            _dijkstra_two_best_wide,
        )

        costs, wide_directions = build_wide_edge_costs(
            geometry, metric_strength, stencil_radius)

        def walk(seed_x, seed_y, reach, edge_costs, h, w):
            result = _dijkstra_two_best_wide(
                seed_x,
                seed_y,
                reach,
                edge_costs,
                wide_directions,
                h,
                w,
            )
            return result[0], result[1], result[2], result[3]
    elif transport_model == "edge":
        costs = _edge_cost_stack(geometry, metric_strength)
    else:
        raise ValueError(f"unknown transport model: {transport_model}")
    if (
        transport_model == "edge"
        and stencil_radius == 1
        and transport_queue == "bucket"
    ):
        walk = _dijkstra_two_best_bucket_adapter
    elif (
        transport_model == "edge"
        and stencil_radius == 1
        and transport_queue == "heap"
    ):
        walk = _dijkstra_two_best_packed
    elif transport_model == "edge" and stencil_radius == 1:
        raise ValueError(f"unknown transport queue: {transport_queue}")
    trace: list[dict] = []
    if initial_centers is None:
        centers = np.array([[
            float(np.sum(measure * x)),
            float(np.sum(measure * y)),
        ]], dtype=np.float64)
    else:
        centers = np.asarray(initial_centers, dtype=np.float64).copy()
        if centers.ndim != 2 or centers.shape[1] != 2 or len(centers) == 0:
            raise ValueError("initial_centers must have shape (cells, 2)")
        centers[:, 0] = np.clip(
            centers[:, 0], 0.5 / width, 1.0 - 0.5 / width)
        centers[:, 1] = np.clip(
            centers[:, 1], 0.5 / height, 1.0 - 0.5 / height)
    cells = len(centers)
    safety_limit_hit = False
    owner = np.zeros(pixels, dtype=np.int32)
    previous_owner: np.ndarray | None = None
    previous_adjacency: np.ndarray | None = None
    # Maps the sites used by the next walk back to the sites of this walk.
    # It exists only for diagnostics: ancestry never restricts ownership.
    parent_for_next_walk: np.ndarray | None = None
    center_history: list[np.ndarray] | None = (
        [] if capture_center_history else None)

    for round_index in range(max(int(maximum_rounds), 1)):
        use_direct_branches = (
            direct_metric_branches
            and round_index >= max(int(direct_metric_start_round), 0)
            and (
                int(direct_metric_rounds) <= 0
                or round_index < (
                    max(int(direct_metric_start_round), 0)
                    + int(direct_metric_rounds)
                )
            )
        )
        progress = round_index / max(int(maximum_rounds) - 1, 1)
        softness = (
            float(softness_start)
            * (float(softness_end) / float(softness_start)) ** progress)
        temperature = (
            softness * max(float(geometry["max_support_px"]), 1.0))
        seed_x = np.clip(
            np.rint(centers[:, 0] * width - 0.5).astype(np.int32),
            0,
            width - 1,
        )
        seed_y = np.clip(
            np.rint(centers[:, 1] * height - 0.5).astype(np.int32),
            0,
            height - 1,
        )
        owner, runner, first_distance, second_distance = (
            walk(
                seed_x,
                seed_y,
                np.zeros(cells, dtype=np.float64),
                costs,
                height,
                width,
            )
        )
        topology_pixel_change = float("nan")
        topology_edge_jaccard = float("nan")
        topology_new_edge_fraction = float("nan")
        if (
            trace_topology
            and previous_owner is not None
            and previous_adjacency is not None
            and parent_for_next_walk is not None
            and len(parent_for_next_walk) == cells
        ):
            mapped_owner = parent_for_next_walk[owner]
            topology_pixel_change = float(np.mean(
                mapped_owner != previous_owner))
            valid_pair = runner >= 0
            mapped_runner = np.full_like(runner, -1)
            mapped_runner[valid_pair] = parent_for_next_walk[
                runner[valid_pair]]
            distinct = valid_pair & (mapped_owner != mapped_runner)
            edge_lo = np.minimum(
                mapped_owner[distinct], mapped_runner[distinct])
            edge_hi = np.maximum(
                mapped_owner[distinct], mapped_runner[distinct])
            prior_cells = int(np.max(parent_for_next_walk)) + 1
            mapped_adjacency = np.unique(
                edge_lo.astype(np.int64) * prior_cells
                + edge_hi.astype(np.int64)
            )
            union = np.union1d(
                previous_adjacency, mapped_adjacency)
            intersection = np.intersect1d(
                previous_adjacency, mapped_adjacency,
                assume_unique=True,
            )
            topology_edge_jaccard = (
                float(len(intersection) / len(union))
                if len(union) else 1.0
            )
            topology_new_edge_fraction = (
                float(
                    np.count_nonzero(
                        ~np.isin(
                            mapped_adjacency,
                            previous_adjacency,
                            assume_unique=True,
                        )
                    ) / len(mapped_adjacency)
                )
                if len(mapped_adjacency) else 0.0
            )
        if trace_topology:
            valid_pair = runner >= 0
            distinct = valid_pair & (owner != runner)
            edge_lo = np.minimum(owner[distinct], runner[distinct])
            edge_hi = np.maximum(owner[distinct], runner[distinct])
            current_adjacency = np.unique(
                edge_lo.astype(np.int64) * cells
                + edge_hi.astype(np.int64)
            )
        else:
            current_adjacency = np.empty(0, dtype=np.int64)
        pixel_count = np.bincount(owner, minlength=cells)
        moments, qxx, qxy, qyy = _soft_transport_moments(
            owner,
            runner,
            first_distance,
            second_distance,
            temperature,
            measure,
            x,
            y,
            qxx_p,
            qxy_p,
            qyy_p,
            cells,
        )
        relaxation = float(np.clip(centroid_relaxation, 0.0, 1.0))
        centers[:, 0] += relaxation * (
            moments["cx"] - centers[:, 0])
        centers[:, 1] += relaxation * (
            moments["cy"] - centers[:, 1])

        instability = np.zeros(cells, dtype=np.float64)
        minor_instability = np.zeros(cells, dtype=np.float64)
        direction = np.zeros((cells, 2), dtype=np.float64)
        for cell in range(cells):
            value, vx, vy, minor = _unstable_direction(
                moments["cxx"][cell],
                moments["cxy"][cell],
                moments["cyy"][cell],
                qxx[cell],
                qxy[cell],
                qyy[cell],
            )
            instability[cell] = value
            minor_instability[cell] = minor
            direction[cell] = (vx, vy)

        transport_extent = (
            moments["transport_rms"]
            / max(float(geometry["max_support_px"]), 1e-12)
        )
        transport_split = transport_extent > float(threshold)
        metric_split = instability > float(metric_extent_threshold)
        split = (
            (transport_split | metric_split)
            & (pixel_count >= 2 * max(int(minimum_region_pixels), 1))
            & (moments["mass"] > 1e-8)
        )
        branch_count = np.ones(cells, dtype=np.int32)
        branch_count[split] = 2
        metric_coherence = np.hypot(
            qxx - qyy, 2.0 * qxy
        ) / np.maximum(qxx + qyy, 1e-30)
        if (
            use_direct_branches
            and int(direct_metric_local_levels) <= 0
            and math.isfinite(float(metric_extent_threshold))
        ):
            requested = np.ceil(np.sqrt(
                instability / max(float(metric_extent_threshold), 1e-30)
            )).astype(np.int32)
            requested = np.clip(
                requested,
                1,
                max(int(maximum_direct_branches), 2),
            )
            available = pixel_count // max(int(minimum_region_pixels), 1)
            requested = np.minimum(requested, np.maximum(available, 1))
            direct_cells = (
                metric_split
                & split
                & (
                    metric_coherence
                    >= float(np.clip(direct_metric_coherence, 0.0, 1.0))
                )
            )
            branch_count[direct_cells] = np.maximum(
                branch_count[direct_cells],
                requested[direct_cells],
            )
        split_ids = np.flatnonzero(split)
        requested_cells = int(np.sum(branch_count))
        if requested_cells > int(safety_cells):
            # This is only a guard against a misspecified stability threshold.
            # Do not select a privileged subset of leaves to satisfy it.
            safety_limit_hit = True
            break
        old_cells = cells
        local_levels_used = 0
        parent_for_next_walk = np.arange(
            old_cells, dtype=np.int32)

        if len(split_ids):
            if use_direct_branches and int(direct_metric_local_levels) > 0:
                child_centers, valid_children, local_levels_used = (
                    _recursive_metric_bifurcation(
                        owner,
                        runner,
                        first_distance,
                        second_distance,
                        temperature,
                        measure,
                        x,
                        y,
                        qxx_p,
                        qxy_p,
                        qyy_p,
                        split,
                        float(metric_extent_threshold),
                        max(int(minimum_region_pixels), 1),
                        max(int(direct_metric_local_levels), 1),
                        max(int(branch_bins), 256),
                        int(safety_cells),
                    )
                )
                centers = child_centers[valid_children]
                # The local-recursion experiment does not preserve a simple
                # one-level parent map.  Mark its topology comparison as
                # unavailable rather than inventing ancestry.
                parent_for_next_walk = None
                centers[:, 0] = np.clip(
                    centers[:, 0], 0.5 / width, 1.0 - 0.5 / width)
                centers[:, 1] = np.clip(
                    centers[:, 1], 0.5 / height, 1.0 - 0.5 / height)
                cells = len(centers)
            elif use_direct_branches:
                child_centers, valid_children = (
                    _balanced_metric_multibranch(
                        owner,
                        runner,
                        first_distance,
                        second_distance,
                        temperature,
                        measure,
                        x,
                        y,
                        moments["mass"],
                        moments["cx"],
                        moments["cy"],
                        direction,
                        branch_count,
                        max(int(branch_bins), 256),
                    )
                )
                centers = child_centers[valid_children]
                direct_parent = np.repeat(
                    np.arange(old_cells, dtype=np.int32),
                    branch_count,
                )
                parent_for_next_walk = direct_parent[valid_children]
                centers[:, 0] = np.clip(
                    centers[:, 0], 0.5 / width, 1.0 - 0.5 / width)
                centers[:, 1] = np.clip(
                    centers[:, 1], 0.5 / height, 1.0 - 0.5 / height)
                cells = len(centers)
            elif exact_branch_balance:
                negative, positive, valid_split = (
                    _balanced_branch_barycentres(
                        owner,
                        runner,
                        first_distance,
                        second_distance,
                        temperature,
                        measure,
                        x,
                        y,
                        moments,
                        direction,
                        split,
                        balance_steps=balance_steps,
                    )
                )
            else:
                negative, positive, valid_split = (
                    _balanced_branch_histogram(
                    owner,
                    runner,
                    first_distance,
                    second_distance,
                    temperature,
                    measure,
                    x,
                    y,
                    moments["mass"],
                    moments["cx"],
                    moments["cy"],
                    direction,
                    split,
                    max(int(branch_bins), 8),
                )
                )
            if not use_direct_branches:
                separation = np.hypot(
                    (positive[:, 0] - negative[:, 0]) * width,
                    (positive[:, 1] - negative[:, 1]) * height,
                )
                valid_split &= separation >= 0.75
                split_ids = np.flatnonzero(valid_split)
                centers[split_ids] = negative[split_ids]
                centers = np.vstack([centers, positive[split_ids]])
                parent_for_next_walk = np.concatenate([
                    np.arange(old_cells, dtype=np.int32),
                    split_ids.astype(np.int32),
                ])
                centers[:, 0] = np.clip(
                    centers[:, 0], 0.5 / width, 1.0 - 0.5 / width)
                centers[:, 1] = np.clip(
                    centers[:, 1], 0.5 / height, 1.0 - 0.5 / height)
                cells = len(centers)

        trace.append({
            "round": round_index + 1,
            "cells_before": old_cells,
            "splits": int(cells - old_cells),
            "split_parents": int(len(split_ids)),
            "maximum_requested_branches": int(np.max(branch_count)),
            "direct_metric_branches": bool(use_direct_branches),
            "direct_metric_local_levels": int(local_levels_used),
            "cells_after": cells,
            "softness": softness,
            "temperature_px": temperature,
            "transport_extent_p50": float(np.median(transport_extent)),
            "transport_extent_p90": float(np.percentile(
                transport_extent, 90.0)),
            "transport_extent_max": float(np.max(transport_extent)),
            "instability_p50": float(np.median(instability)),
            "instability_p90": float(np.percentile(instability, 90.0)),
            "instability_max": float(np.max(instability)),
            "minor_instability_p90": float(np.percentile(
                minor_instability, 90.0)),
            "metric_aspect_p90": float(np.percentile(
                instability / np.maximum(minor_instability, 1e-12),
                90.0,
            )),
            "metric_aspect_max": float(np.max(
                instability / np.maximum(minor_instability, 1e-12))),
            "metric_coherence_p50": float(np.median(metric_coherence)),
            "metric_coherence_p90": float(np.percentile(
                metric_coherence, 90.0)),
            "topology_pixel_change": topology_pixel_change,
            "topology_edge_jaccard": topology_edge_jaccard,
            "topology_new_edge_fraction": topology_new_edge_fraction,
            "coownership_edges": int(len(current_adjacency)),
            "minimum_region_pixels": int(np.min(
                np.bincount(owner, minlength=old_cells))),
        })
        if center_history is not None:
            center_history.append(centers.copy())
        if trace_topology:
            previous_owner = owner.copy()
            previous_adjacency = current_adjacency
        if len(split_ids) == 0:
            break

    # Zero-temperature readout: ancestry has no ownership.  Every final site
    # re-enters the same hard transport domain.
    seed_x = np.clip(
        np.rint(centers[:, 0] * width - 0.5).astype(np.int32),
        0,
        width - 1,
    )
    seed_y = np.clip(
        np.rint(centers[:, 1] * height - 0.5).astype(np.int32),
        0,
        height - 1,
    )
    owner, _, _, _ = walk(
        seed_x,
        seed_y,
        np.zeros(cells, dtype=np.float64),
        costs,
        height,
        width,
    )
    return owner.reshape(height, width), {
        "centers": centers,
        "cells": cells,
        "safety_limit_hit": safety_limit_hit,
        "center_history": center_history,
        "transport_stencil_radius": stencil_radius,
        "transport_model": transport_model,
    }, trace


def fit_hard_regions(
    labels_2d: np.ndarray,
    target_lab: np.ndarray,
    objective: SingleStageDecompositionObjective,
) -> tuple[dict, np.ndarray]:
    """Independent full affine Lab jet per hard allocation region."""
    labels = np.asarray(labels_2d, dtype=np.intp).ravel()
    height, width = labels_2d.shape
    cells = int(np.max(labels)) + 1
    yy, xx = np.mgrid[:height, :width]
    x = (xx.ravel() + 0.5) / width - 0.5
    y = (yy.ravel() + 0.5) / height - 0.5
    basis = np.column_stack([np.ones_like(x), x, y])
    target = np.asarray(target_lab, dtype=np.float64).reshape(-1, 3)
    normal = np.empty((cells, 3, 3), dtype=np.float64)
    rhs = np.empty((cells, 3, 3), dtype=np.float64)
    for a in range(3):
        for b in range(3):
            normal[:, a, b] = _bincount(
                labels, basis[:, a] * basis[:, b], cells)
        for channel in range(3):
            rhs[:, a, channel] = _bincount(
                labels, basis[:, a] * target[:, channel], cells)
    count = np.maximum(np.bincount(labels, minlength=cells), 1)
    scale = count.astype(np.float64)
    normal[:, 0, 0] += 1e-7 * scale
    normal[:, 1, 1] += 1e-5 * scale
    normal[:, 2, 2] += 1e-5 * scale
    coefficient = np.linalg.solve(normal, rhs)
    reconstruction = np.einsum(
        "ni,nic->nc", basis, coefficient[labels], optimize=False
    ).reshape(target_lab.shape)
    return score(
        objective, objective.target_rgb, reconstruction), reconstruction


def fit_hard_regions_with_ridge(
    labels_2d: np.ndarray,
    centers: np.ndarray,
    target_lab: np.ndarray,
    objective: SingleStageDecompositionObjective,
    *,
    ridge_kappa: float = 4.0,
    ridge_angles: int = 16,
    ridge_bins: int = 41,
    ridge_count: int = 1,
    initial_affine: np.ndarray | None = None,
) -> tuple[dict, np.ndarray, dict]:
    """Refit hard regions with a small measured bounded ridge ladder."""
    if initial_affine is None:
        _, affine = fit_hard_regions(labels_2d, target_lab, objective)
    else:
        affine = np.asarray(initial_affine, dtype=np.float64)
        if affine.shape != target_lab.shape:
            raise ValueError("initial affine reconstruction shape mismatch")
    labels = np.asarray(labels_2d, dtype=np.int32).ravel()
    height, width = labels_2d.shape
    cells = int(np.max(labels)) + 1
    yy, xx = np.mgrid[:height, :width]
    xf = xx.ravel().astype(np.float64)
    yf = yy.ravel().astype(np.float64)
    center_x = np.asarray(centers[:cells, 0]) * width - 0.5
    center_y = np.asarray(centers[:cells, 1]) * height - 0.5
    dx = xf - center_x[labels]
    dy = yf - center_y[labels]
    spacing = max(math.sqrt(height * width / max(cells, 1)), 1e-9)
    x = (xf + 0.5) / width - 0.5
    y = (yf + 0.5) / height - 0.5
    basis = np.column_stack([np.ones_like(x), x, y])
    target = np.asarray(target_lab, dtype=np.float64).reshape(-1, 3)
    count = np.maximum(np.bincount(labels, minlength=cells), 1)
    reconstruction = affine
    ridge_means = []
    ridge_nonzero = []

    def refit(active_basis: np.ndarray) -> np.ndarray:
        width_basis = active_basis.shape[1]
        normal = np.empty(
            (cells, width_basis, width_basis), dtype=np.float64)
        rhs = np.empty((cells, width_basis, 3), dtype=np.float64)
        for a in range(width_basis):
            for b in range(width_basis):
                normal[:, a, b] = _bincount(
                    labels,
                    active_basis[:, a] * active_basis[:, b],
                    cells,
                )
            for channel in range(3):
                rhs[:, a, channel] = _bincount(
                    labels,
                    active_basis[:, a] * target[:, channel],
                    cells,
                )
        regularization = np.array(
            [1e-7, 1e-5, 1e-5]
            + [2e-5] * max(width_basis - 3, 0),
            dtype=np.float64,
        )
        for component in range(width_basis):
            normal[:, component, component] += (
                regularization[component] * count)
        coefficient = np.linalg.solve(normal, rhs)
        return np.einsum(
            "ni,nic->nc",
            active_basis,
            coefficient[labels],
            optimize=False,
        ).reshape(target_lab.shape)

    for _ in range(max(int(ridge_count), 0)):
        residual = target - reconstruction.reshape(-1, 3)
        ridge_score, ridge_axis, ridge_offset = measure_residual_ridges(
            labels,
            np.ones(labels.size, dtype=np.float64),
            residual,
            dx,
            dy,
            spacing,
            cells,
            angles=ridge_angles,
            bins=ridge_bins,
            channel_weights=(1.0, 1.5, 1.5),
        )
        projection = (
            dx * np.cos(ridge_axis[labels])
            + dy * np.sin(ridge_axis[labels])
        ) / spacing
        ridge = np.tanh(
            float(ridge_kappa) * (
                projection - ridge_offset[labels]))
        basis = np.column_stack([basis, ridge])
        reconstruction = refit(basis)
        ridge_means.append(float(np.mean(ridge_score)))
        ridge_nonzero.append(int(np.count_nonzero(
            ridge_score > 1e-12)))

    return (
        score(objective, objective.target_rgb, reconstruction),
        reconstruction,
        {
            "ridge_count": max(int(ridge_count), 0),
            "ridge_score_mean": ridge_means,
            "ridge_score_nonzero": ridge_nonzero,
            "ridge_angles": int(ridge_angles),
            "ridge_bins": int(ridge_bins),
        },
    )


def site_id_image(labels: np.ndarray) -> np.ndarray:
    colours = site_id_colours(int(np.max(labels)) + 1)
    return colours[np.asarray(labels, dtype=np.intp)]


def save_panel(
    rgb: np.ndarray,
    labels: np.ndarray,
    record: dict,
    geometry: dict,
    trace: list[dict],
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("target")
    axes[0, 1].imshow(record["rgb"])
    axes[0, 1].set_title(
        f"hard allocation readout\n{record['psnr']:.2f} dB")
    axes[0, 2].imshow(site_id_image(labels))
    axes[0, 2].set_title(f"site IDs — {int(np.max(labels))+1} cells")
    axes[1, 0].imshow(np.asarray(geometry["measure"]), cmap="viridis")
    axes[1, 0].set_title(
        "one-decomposition support measure\n"
        f"{geometry['implied_cells']:.0f} continuous cells")
    axes[1, 1].step(
        [item["round"] for item in trace],
        [item["cells_after"] for item in trace],
        where="post")
    axes[1, 1].set_title("allocation bifurcation")
    axes[1, 1].set_xlabel("transport-temperature round")
    axes[1, 1].set_ylabel("hard cells")
    axes[1, 2].semilogy(
        [item["round"] for item in trace],
        [max(item["instability_max"], 1e-12) for item in trace],
        label="maximum")
    axes[1, 2].semilogy(
        [item["round"] for item in trace],
        [max(item["instability_p90"], 1e-12) for item in trace],
        label="p90")
    axes[1, 2].set_title("metric instability")
    axes[1, 2].legend()
    for axis in axes.ravel():
        if axis not in (axes[1, 1], axes[1, 2]):
            axis.set_xticks([])
            axis.set_yticks([])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?")
    parser.add_argument("--gallery", default="pikachu")
    parser.add_argument("--side", type=int, default=256)
    parser.add_argument(
        "--tgfd-sweeps", type=int, default=24,
        help="internal convergence work of the one target decomposition")
    parser.add_argument("--flow-sweeps", type=int, default=24)
    parser.add_argument("--max-support-fraction", type=float, default=0.18)
    parser.add_argument(
        "--coherent-tangent-fraction", type=float, default=0.02)
    parser.add_argument("--threshold", type=float, default=2.5)
    parser.add_argument(
        "--metric-strength", type=float, default=1.5,
        help="dimensionless BFFT barrier, scaled by support horizon squared")
    parser.add_argument(
        "--metric-extent-threshold", type=float, default=math.inf,
        help=(
            "split when the largest eigenvalue of the transported "
            "cell/support product exceeds this value"))
    parser.add_argument(
        "--direct-metric-branches", action="store_true",
        help=(
            "emit ceil(sqrt(metric extent / limit)) mass-balanced branches "
            "along the worst supported axis in one allocation action"))
    parser.add_argument(
        "--maximum-direct-branches", type=int, default=32,
        help="per-cell guard for direct metric multibranching")
    parser.add_argument(
        "--direct-metric-start-round", type=int, default=0,
        help="zero-based transport round at which direct branching begins")
    parser.add_argument(
        "--direct-metric-rounds", type=int, default=0,
        help="number of direct rounds; zero keeps it enabled")
    parser.add_argument(
        "--direct-metric-coherence", type=float, default=0.0,
        help=(
            "minimum coherence of the transported BFFT precision required "
            "for more than two branches"))
    parser.add_argument(
        "--direct-metric-local-levels", type=int, default=0,
        help=(
            "recompute child support tensors this many times inside each "
            "frozen transport action; zero uses the direct count formula"))
    parser.add_argument(
        "--softness-start", type=float, default=0.20,
        help="initial temperature as a fraction of support horizon")
    parser.add_argument(
        "--softness-end", type=float, default=0.0025,
        help="final temperature as a fraction of support horizon")
    parser.add_argument("--minimum-region-pixels", type=int, default=12)
    parser.add_argument("--safety-cells", type=int, default=4096)
    parser.add_argument("--maximum-rounds", type=int, default=24)
    parser.add_argument(
        "--pyramid-side", type=int, default=0,
        help=(
            "preallocate on this maximum side using the same frozen support; "
            "zero disables the transport pyramid"))
    parser.add_argument(
        "--pyramid-rounds", type=int, default=5,
        help="full-resolution correction rounds after coarse prolongation")
    parser.add_argument(
        "--branch-bins", type=int, default=64,
        help="bins in the fused one-pass branch mass quantile")
    parser.add_argument(
        "--exact-branch-balance", action="store_true",
        help="use the slower repeated exact branch bisection reference")
    parser.add_argument(
        "--approximate-branch-balance", action="store_true",
        help="use the faster histogram branch quantile (experimental)")
    parser.add_argument(
        "--transport-queue", choices=("heap", "bucket"), default="heap",
        help="exact priority queue used by the two-label transport walk")
    parser.add_argument(
        "--trace-topology", action="store_true",
        help=(
            "measure parent-territory and co-ownership graph changes "
            "between allocation events (diagnostic overhead)"))
    parser.add_argument(
        "--ridges", type=int, default=1,
        help="bounded residual ridge columns per hard cell")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "experiments/out/wasserstein_allocation_tree.png")
    parser.add_argument(
        "--json", type=Path,
        default=ROOT / "experiments/out/wasserstein_allocation_tree.json")
    args = parser.parse_args()
    if args.image:
        from skimage.io import imread

        source = imread(Path(args.image).expanduser())
        source_name = str(Path(args.image).expanduser())
    else:
        source = gallery.load(args.gallery)
        source_name = f"gallery:{args.gallery}"
    rgb = _fit_rgb(source, args.side)
    started = time.perf_counter()
    geometry = single_decomposition_geometry(
        rgb,
        tgfd_sweeps=args.tgfd_sweeps,
        flow_sweeps=args.flow_sweeps,
        max_support_fraction=args.max_support_fraction,
        coherent_tangent_fraction=args.coherent_tangent_fraction,
    )
    decomposition_seconds = time.perf_counter() - started
    exact_branch_balance = not args.approximate_branch_balance
    coarse_seconds = 0.0
    initial_centers = None
    if 0 < args.pyramid_side < max(rgb.shape[:2]):
        coarse_geometry = pyramid_geometry(geometry, args.pyramid_side)
        coarse_scale = (
            max(np.asarray(coarse_geometry["measure"]).shape)
            / max(np.asarray(geometry["measure"]).shape)
        )
        coarse_started = time.perf_counter()
        _, coarse_allocation, _ = bifurcate_allocation(
            coarse_geometry,
            threshold=args.threshold,
            softness_start=args.softness_start,
            softness_end=args.softness_end,
            minimum_region_pixels=max(
                1,
                int(round(args.minimum_region_pixels * coarse_scale ** 2)),
            ),
            safety_cells=args.safety_cells,
            maximum_rounds=args.maximum_rounds,
            metric_strength=args.metric_strength,
            branch_bins=args.branch_bins,
            exact_branch_balance=exact_branch_balance,
            metric_extent_threshold=args.metric_extent_threshold,
            direct_metric_branches=args.direct_metric_branches,
            maximum_direct_branches=args.maximum_direct_branches,
            direct_metric_start_round=args.direct_metric_start_round,
            direct_metric_rounds=args.direct_metric_rounds,
            direct_metric_coherence=args.direct_metric_coherence,
            direct_metric_local_levels=args.direct_metric_local_levels,
            transport_queue=args.transport_queue,
            trace_topology=args.trace_topology,
        )
        coarse_seconds = time.perf_counter() - coarse_started
        initial_centers = coarse_allocation["centers"]
    allocation_started = time.perf_counter()
    labels, allocation, trace = bifurcate_allocation(
        geometry,
        threshold=args.threshold,
        minimum_region_pixels=args.minimum_region_pixels,
        safety_cells=args.safety_cells,
        maximum_rounds=(
            args.pyramid_rounds
            if initial_centers is not None else args.maximum_rounds
        ),
        metric_strength=args.metric_strength,
        branch_bins=args.branch_bins,
        exact_branch_balance=exact_branch_balance,
        initial_centers=initial_centers,
        metric_extent_threshold=args.metric_extent_threshold,
        direct_metric_branches=args.direct_metric_branches,
        maximum_direct_branches=args.maximum_direct_branches,
        direct_metric_start_round=args.direct_metric_start_round,
        direct_metric_rounds=args.direct_metric_rounds,
        direct_metric_coherence=args.direct_metric_coherence,
        direct_metric_local_levels=args.direct_metric_local_levels,
        transport_queue=args.transport_queue,
        trace_topology=args.trace_topology,
        softness_start=(
            0.05 if initial_centers is not None else args.softness_start),
        softness_end=(
            0.006 if initial_centers is not None else args.softness_end),
    )
    allocation_seconds = time.perf_counter() - allocation_started
    objective = SingleStageDecompositionObjective(
        rgb, passes=args.tgfd_sweeps)
    fit_started = time.perf_counter()
    affine_record, _ = fit_hard_regions(
        labels, srgb_to_lab(rgb), objective)
    if args.ridges > 0:
        record, _, ridge_info = fit_hard_regions_with_ridge(
            labels,
            allocation["centers"],
            srgb_to_lab(rgb),
            objective,
            ridge_count=args.ridges,
        )
    else:
        record = affine_record
        ridge_info = {"ridge_count": 0}
    fit_seconds = time.perf_counter() - fit_started
    save_panel(rgb, labels, record, geometry, trace, args.output)
    report = {
        "source": source_name,
        "shape": list(rgb.shape),
        "cells": allocation["cells"],
        "threshold": args.threshold,
        "metric_strength": args.metric_strength,
        "metric_extent_threshold": args.metric_extent_threshold,
        "direct_metric_branches": args.direct_metric_branches,
        "maximum_direct_branches": args.maximum_direct_branches,
        "direct_metric_start_round": args.direct_metric_start_round,
        "direct_metric_rounds": args.direct_metric_rounds,
        "direct_metric_coherence": args.direct_metric_coherence,
        "direct_metric_local_levels": args.direct_metric_local_levels,
        "transport_queue": args.transport_queue,
        "trace_topology": args.trace_topology,
        "branch_balance": (
            "exact_bisection" if exact_branch_balance
            else f"fused_histogram_{max(args.branch_bins, 8)}"
        ),
        "pyramid_side": args.pyramid_side,
        "pyramid_initial_cells": (
            0 if initial_centers is None else len(initial_centers)),
        "coherent_tangent_fraction": args.coherent_tangent_fraction,
        "target_decompositions_for_allocation": 1,
        "tgfd_internal_sweeps": args.tgfd_sweeps,
        "continuous_implied_cells": geometry["implied_cells"],
        "decomposition_seconds": decomposition_seconds,
        "coarse_allocation_seconds": coarse_seconds,
        "allocation_seconds": allocation_seconds,
        "fit_seconds": fit_seconds,
        "ridge_info": ridge_info,
        "safety_limit_hit": allocation["safety_limit_hit"],
        "affine_record": {
            key: float(value)
            for key, value in affine_record.items() if key != "rgb"
        },
        "record": {
            key: float(value)
            for key, value in record.items() if key != "rgb"
        },
        "trace": trace,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2))
    print(json.dumps({
        "cells": allocation["cells"],
        "psnr": record["psnr"],
        "objective": record["objective"],
        "affine_psnr": affine_record["psnr"],
        "ridges": args.ridges,
        "continuous_implied_cells": geometry["implied_cells"],
        "decomposition_seconds": decomposition_seconds,
        "coarse_allocation_seconds": coarse_seconds,
        "allocation_seconds": allocation_seconds,
        "fit_seconds": fit_seconds,
        "safety_limit_hit": allocation["safety_limit_hit"],
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
