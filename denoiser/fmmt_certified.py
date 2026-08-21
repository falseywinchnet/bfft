#!/usr/bin/env python3
"""FMMT-2D with cross-predictive certified support birth.

A blind grayscale denoiser designed around one transported probabilistic state,
not a bank of named noise models.

Core idea
---------
1. Build a robust transport-derived provisional chart x0 from the full local
   intensity distribution.
2. Before x0 becomes geometry, independently held-out observation lattices
   certify coarse support and fine ancestry. Unsupported coarse support is
   allowed one bounded conservative evolution; hard-boundary censoring reduces
   the authority of this witness rather than forcing a noise label.
3. Estimate a robust local residual scale from y-x0.
4. Define an 8-neighbour discrete eikonal metric. Supported contrast retains
   the ordinary FMMT crossing cost; unsupported contrast is cheaper to cross,
   so an unverified bootstrap pit cannot become a hereditary barrier.
5. Every anchor carries TWO full empirical measures on the SAME ordered fronts:
      pi_a(x): latent-signal proposal measure
      nu_a(n): residual/noise measure
   They are transported with exp(-geodesic_distance/tau) attenuation.
6. Couple the transported measures through the additive observation equation:
      p_i(x | y_i) proportional to Pi_i(x) * Nu_i(y_i - x)
   No Gaussian/Poisson/impulse/speckle branch exists.
7. The Bayes action is the posterior mean. Posterior entropy controls inertia:
   uncertain transported state cannot overwrite the bootstrap aggressively.

The fast-marching backend is exact shortest-path transport on an 8-neighbour
weighted graph (Dijkstra's ordered method), i.e. a discrete eikonal model. It is
not a subpixel continuous-grid Sethian FMM solver; that distinction is explicit.

Dependencies: numpy, scipy, Pillow, numba
Benchmark additionally requires scikit-image.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from functools import lru_cache
from pathlib import Path
import time

try:
    import numba
except ImportError:  # The GUI can still run the reference path without JIT.
    class _NumbaFallback:
        @staticmethod
        def njit(*args, **kwargs):
            if args and callable(args[0]) and len(args) == 1 and not kwargs:
                return args[0]

            def decorate(function):
                return function

            return decorate

    numba = _NumbaFallback()
import numpy as np
from PIL import Image
from scipy import ndimage, sparse
from scipy.signal import fftconvolve
from scipy.sparse.csgraph import dijkstra
from scipy.stats import norm


# -----------------------------------------------------------------------------
# Measure bootstrap: fast separable path-sum approximation used ONLY to obtain
# a robust scalar chart for the later eikonal metric. The final estimator is the
# FMM posterior, not this bootstrap.

@numba.njit(cache=True)
def _geo_axis(fields, guide, scale, tau, alpha, axis):
    h, w, channels = fields.shape
    out = np.empty_like(fields)
    if axis == 1:
        for y in range(h):
            for c in range(channels):
                out[y, 0, c] = fields[y, 0, c]
            for x in range(1, w):
                local_scale = 0.5 * (scale[y, x] + scale[y, x - 1]) + 1e-6
                ds = 1.0 + alpha * abs(guide[y, x] - guide[y, x - 1]) / local_scale
                transmission = math.exp(-ds / tau)
                for c in range(channels):
                    out[y, x, c] = fields[y, x, c] + transmission * out[y, x - 1, c]
        for y in range(h):
            acc = np.empty(channels, np.float64)
            for c in range(channels):
                acc[c] = fields[y, w - 1, c]
            for x in range(w - 2, -1, -1):
                local_scale = 0.5 * (scale[y, x] + scale[y, x + 1]) + 1e-6
                ds = 1.0 + alpha * abs(guide[y, x] - guide[y, x + 1]) / local_scale
                transmission = math.exp(-ds / tau)
                for c in range(channels):
                    acc[c] = fields[y, x, c] + transmission * acc[c]
                    out[y, x, c] += acc[c] - fields[y, x, c]
    else:
        for x in range(w):
            for c in range(channels):
                out[0, x, c] = fields[0, x, c]
            for y in range(1, h):
                local_scale = 0.5 * (scale[y, x] + scale[y - 1, x]) + 1e-6
                ds = 1.0 + alpha * abs(guide[y, x] - guide[y - 1, x]) / local_scale
                transmission = math.exp(-ds / tau)
                for c in range(channels):
                    out[y, x, c] = fields[y, x, c] + transmission * out[y - 1, x, c]
        for x in range(w):
            acc = np.empty(channels, np.float64)
            for c in range(channels):
                acc[c] = fields[h - 1, x, c]
            for y in range(h - 2, -1, -1):
                local_scale = 0.5 * (scale[y, x] + scale[y + 1, x]) + 1e-6
                ds = 1.0 + alpha * abs(guide[y, x] - guide[y + 1, x]) / local_scale
                transmission = math.exp(-ds / tau)
                for c in range(channels):
                    acc[c] = fields[y, x, c] + transmission * acc[c]
                    out[y, x, c] += acc[c] - fields[y, x, c]
    return out


@numba.njit(cache=True)
def _geo_transport(fields, guide, scale, tau, alpha):
    hv = _geo_axis(_geo_axis(fields, guide, scale, tau, alpha, 1), guide, scale, tau, alpha, 0)
    vh = _geo_axis(_geo_axis(fields, guide, scale, tau, alpha, 0), guide, scale, tau, alpha, 1)
    return 0.5 * (hv + vh)


def _geo_axis_vectorized_transmission(fields, transmission, axis):
    """Advance one recurrence axis using a precomputed edge transmission."""
    fields = np.asarray(fields, np.float64)
    h, w, _channels = fields.shape
    out = np.empty_like(fields)
    if axis == 1:
        if transmission.shape != (h, w - 1):
            raise ValueError("horizontal transmission does not align")
        out[:, 0, :] = fields[:, 0, :]
        for x in range(1, w):
            out[:, x, :] = (
                fields[:, x, :]
                + transmission[:, x - 1, None] * out[:, x - 1, :]
            )
        accumulator = fields[:, -1, :].copy()
        for x in range(w - 2, -1, -1):
            accumulator = (
                fields[:, x, :]
                + transmission[:, x, None] * accumulator
            )
            out[:, x, :] += accumulator - fields[:, x, :]
    else:
        if transmission.shape != (h - 1, w):
            raise ValueError("vertical transmission does not align")
        out[0, :, :] = fields[0, :, :]
        for y in range(1, h):
            out[y, :, :] = (
                fields[y, :, :]
                + transmission[y - 1, :, None] * out[y - 1, :, :]
            )
        accumulator = fields[-1, :, :].copy()
        for y in range(h - 2, -1, -1):
            accumulator = (
                fields[y, :, :]
                + transmission[y, :, None] * accumulator
            )
            out[y, :, :] += accumulator - fields[y, :, :]
    return out


def _geo_edge_transmissions(guide, scale, tau, alpha):
    """Resolve the two physical edge laws once for both sweep orderings."""
    epsilon = 1e-6
    horizontal_scale = 0.5 * (scale[:, 1:] + scale[:, :-1]) + epsilon
    horizontal_distance = 1.0 + alpha * np.abs(
        guide[:, 1:] - guide[:, :-1]) / horizontal_scale
    vertical_scale = 0.5 * (scale[1:, :] + scale[:-1, :]) + epsilon
    vertical_distance = 1.0 + alpha * np.abs(
        guide[1:, :] - guide[:-1, :]) / vertical_scale
    return (
        np.exp(-horizontal_distance / tau),
        np.exp(-vertical_distance / tau),
    )


def _geo_axis_vectorized(fields, guide, scale, tau, alpha, axis):
    """The same two-sided recurrence, vectorized across lines and packets.

    ``_geo_axis`` predates the present profiling harness and advances every
    histogram channel in a scalar compiled loop.  On the M4 CPU that loop owns
    almost the entire FMMT runtime.  This form changes only the representation:
    one Python iteration advances all independent lines and packet channels at
    once.  Boundary initialization and recurrence order are deliberately
    identical to the reference implementation.
    """
    horizontal, vertical = _geo_edge_transmissions(
        guide, scale, tau, alpha)
    return _geo_axis_vectorized_transmission(
        fields, horizontal if axis == 1 else vertical, axis)


def _geo_transport_vectorized(fields, guide, scale, tau, alpha):
    horizontal, vertical = _geo_edge_transmissions(
        guide, scale, tau, alpha)
    hv = _geo_axis_vectorized_transmission(
        _geo_axis_vectorized_transmission(fields, horizontal, 1),
        vertical,
        0,
    )
    vh = _geo_axis_vectorized_transmission(
        _geo_axis_vectorized_transmission(fields, vertical, 0),
        horizontal,
        1,
    )
    return 0.5 * (hv + vh)


def _soft_histogram(image, bins, lo=0.0, hi=1.0, window=1):
    image = np.asarray(image, np.float64)
    p = np.clip((image - lo) / (hi - lo), 0.0, 1.0) * (bins - 1)
    low = np.floor(p).astype(np.int32)
    high = np.minimum(low + 1, bins - 1)
    frac = p - low
    h, w = image.shape
    hist = np.zeros((h, w, bins), np.float32)
    rr, cc = np.indices((h, w))
    hist[rr, cc, low] = 1.0 - frac
    hist[rr, cc, high] += frac
    if window > 1:
        # All bins share the same separable spatial box operator.  A unit
        # channel extent applies it in one compiled call without mixing bins.
        ndimage.uniform_filter(
            hist,
            size=(window, window, 1),
            mode="reflect",
            output=hist,
        )
    hist /= np.maximum(hist.sum(axis=-1, keepdims=True), 1e-12)
    return hist


def _hist_quantile(hist, q, lo=0.0, hi=1.0):
    cdf = np.cumsum(hist, axis=-1)
    idx = np.argmax(cdf >= q * cdf[..., -1, None], axis=-1)
    return lo + (hi - lo) * idx / (hist.shape[-1] - 1)


def _bootstrap_chart(y, bins=48):
    raw = _soft_histogram(y, bins, 0.0, 1.0, 1).astype(np.float64)
    z = np.zeros_like(y)
    one = np.ones_like(y)
    probe = _geo_transport_vectorized(raw, z, one, 1.2, 0.0)
    q25 = _hist_quantile(probe, 0.25)
    q75 = _hist_quantile(probe, 0.75)
    broadness = float(np.median(q75 - q25))
    tau = float(np.clip(1.1 + 2.5 * max(broadness - 0.05, 0.0), 1.1, 3.2))
    spatial = _geo_transport_vectorized(raw, z, one, tau, 0.0)
    guide = _hist_quantile(spatial, 0.50)
    residual = y - guide
    abs_r = np.abs(residual)
    q = ndimage.percentile_filter(abs_r, percentile=25, size=5, mode="reflect")
    scale = q / norm.ppf(0.625) + 1e-4
    scale = np.clip(scale, 0.012, 0.15)
    transported = _geo_transport_vectorized(
        raw, guide, scale, max(1.1, 0.9 * tau), 0.2)
    # For the bootstrap only, the median is intentionally conservative.
    return _hist_quantile(transported, 0.50)




# -----------------------------------------------------------------------------
# Cross-predictive support certification.
#
# These witnesses are computed only from the unchanged observation.  They do
# not decide a noise family and they never edit the final reconstruction.
# Their sole job is to decide whether coarse structure in the provisional
# bootstrap is allowed to become hereditary geometry for the eikonal stage.

def _smoothstep(x, a, b):
    # Deliberately linear ramp: evidence earns support continuously rather
    # than through a sharpened classification boundary.
    return np.clip((np.asarray(x, np.float64) - a) / max(b - a, 1e-12), 0.0, 1.0)


def _residue_masks(shape, period):
    yy, xx = np.indices(shape)
    return np.stack([((yy % period) == a) & ((xx % period) == b)
                     for a in range(period) for b in range(period)]).astype(np.float64)


def _lane_lowpass(y, masks, sigma):
    values = []
    for mask in masks:
        den = ndimage.gaussian_filter(mask, sigma, mode="reflect")
        num = ndimage.gaussian_filter(y * mask, sigma, mode="reflect")
        values.append(num / np.maximum(den, 1e-12))
    return np.stack(values)


def _fine_support_backbone(y, open_level=0.06, full_level=0.13):
    """Independent fine-scale ancestry from disjoint residue lattices."""
    masks = _residue_masks(y.shape, 2)
    bands = _lane_lowpass(y, masks, 1.5) - _lane_lowpass(y, masks, 3.0)
    center = np.median(bands, axis=0)
    spread = 1.4826 * np.median(np.abs(bands - center), axis=0) + 1e-6
    signed = bands / np.maximum(np.abs(bands), 0.15 * spread[None])
    phase = np.abs(np.mean(np.clip(signed, -1.0, 1.0), axis=0))
    se = spread / np.sqrt(bands.shape[0]) + 1e-5
    z = np.abs(center) / se
    agreement = _smoothstep(z, 2.0, 4.0) * _smoothstep(phase, 0.45, 0.80)
    density = ndimage.gaussian_filter(agreement, 2.0, mode="reflect")
    pooled = ndimage.gaussian_filter(density, 4.0, mode="reflect")
    return ndimage.gaussian_filter(
        _smoothstep(pooled, open_level, full_level), 1.0, mode="reflect")


def _conv_reflect_fft(y, kernel):
    ry, rx = kernel.shape[0] // 2, kernel.shape[1] // 2
    padded = np.pad(y, ((ry, ry), (rx, rx)), mode="reflect")
    return fftconvolve(padded, kernel, mode="valid")


@lru_cache(None)
def _poly_intercept_kernel(sigma: float, order: int,
                           train_parity: int, center_parity: int):
    radius = int(math.ceil(3.0 * sigma))
    ys, xs = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    weight = np.exp(-(xs * xs + ys * ys) / (2.0 * sigma * sigma))
    weight *= (((center_parity + xs + ys) & 1) == train_parity)
    if order == 1:
        design = np.stack((np.ones_like(xs), xs, ys), axis=-1).reshape(-1, 3)
    elif order == 2:
        design = np.stack((np.ones_like(xs), xs, ys, xs * xs, xs * ys, ys * ys),
                          axis=-1).reshape(-1, 6)
    else:
        raise ValueError("order must be 1 or 2")
    wf = weight.ravel()
    gram = design.T @ (wf[:, None] * design)
    inverse = np.linalg.pinv(gram, rcond=1e-10)
    coeff = (np.eye(gram.shape[0])[0] @ inverse @ design.T) * wf
    return coeff.reshape(weight.shape)


def _crossfit_poly(y, sigma):
    yy, xx = np.indices(y.shape)
    parity = (yy + xx) & 1
    affine = np.empty_like(y)
    quad = np.empty_like(y)
    for held in (0, 1):
        train = 1 - held
        pa = _conv_reflect_fft(y, _poly_intercept_kernel(float(sigma), 1, train, held))
        pq = _conv_reflect_fft(y, _poly_intercept_kernel(float(sigma), 2, train, held))
        use = parity == held
        affine[use] = pa[use]
        quad[use] = pq[use]
    return affine, quad


def _masked_gaussian(values, mask, sigma):
    den = ndimage.gaussian_filter(mask.astype(np.float64), sigma, mode="reflect")
    num = ndimage.gaussian_filter(values * mask, sigma, mode="reflect")
    return num / np.maximum(den, 1e-8)


def _cross_predictive_curvature(y, sigma):
    """Evidence that curvature predicts samples not used to fit it."""
    affine, quad = _crossfit_poly(y, sigma)
    affine_error = np.abs(y - affine)
    quad_error = np.abs(y - quad)
    size = max(5, int(2 * sigma + 1) // 2 * 2 + 1)
    scale = ndimage.percentile_filter(affine_error, 50, size=size, mode="reflect") + 1e-4
    advantage = scale * np.tanh((affine_error - quad_error) / scale)
    yy, xx = np.indices(y.shape)
    parity = (yy + xx) & 1
    means, variances = [], []
    for held in (0, 1):
        mask = parity == held
        mean = _masked_gaussian(advantage, mask, sigma)
        second = _masked_gaussian(advantage * advantage, mask, sigma)
        means.append(mean)
        variances.append(np.maximum(second - mean * mean, 1e-10))
    neff = max(math.pi * sigma * sigma, 2.0)
    z0 = means[0] / np.sqrt(variances[0] / neff + 1e-10)
    z1 = means[1] / np.sqrt(variances[1] / neff + 1e-10)
    return np.minimum(z0, z1)


def _coarse_support_witness(y, provisional, scales=(8.0, 12.0)):
    residual = np.abs(y - provisional)
    tail_ratio = float(np.quantile(residual, 0.90) /
                       (np.quantile(residual, 0.50) + 1e-6))
    tail = np.clip(np.log2(max(tail_ratio / 3.0, 1.0)), 0.0, 4.0)
    z_open = 0.8 + 0.5 * tail
    z_full = 2.5 + 1.4 * tail
    supports = []
    for sigma in scales:
        z = _cross_predictive_curvature(y, float(sigma))
        support = _smoothstep(z, z_open, z_full)
        supports.append(ndimage.gaussian_filter(support, 0.12 * sigma, mode="reflect"))
    return np.maximum.reduce(supports), tail_ratio


def _conservative_support_step(state, conductance, dt=0.18):
    """Mass-conserving 4-neighbour flux; dt<=.25 cannot create a new extremum."""
    out = state.copy()
    edge = 0.5 * (conductance[:, :-1] + conductance[:, 1:])
    flux = dt * edge * (state[:, 1:] - state[:, :-1])
    out[:, :-1] += flux
    out[:, 1:] -= flux
    edge = 0.5 * (conductance[:-1, :] + conductance[1:, :])
    flux = dt * edge * (state[1:, :] - state[:-1, :])
    out[:-1, :] += flux
    out[1:, :] -= flux
    return out


def _certify_bootstrap_support(y, provisional, sweeps=64, dt=0.18,
                               split_sigma=1.5):
    """Evolve provisional coarse support before it becomes FMMT geometry.

    A coarse event is protected only when it predicts held-out observations or
    descends from independently reproducible fine support.  Large hard-boundary
    atom mass reduces the authority of this witness because censored/degenerate
    observations are poor evidence for support evolution; in that regime FMMT's
    original full empirical state is inherited almost unchanged.
    """
    coarse_support, tail_ratio = _coarse_support_witness(y, provisional)
    fine_support = _fine_support_backbone(y, 0.06, 0.13)

    # Reliability of the observation as a support witness.  This is not a
    # salt/pepper classifier: it is simply the mass collapsed onto the two hard
    # measurement bounds, where interior relation evidence has been censored.
    censored_mass = float(np.mean((y <= 1.0 / 255.0) | (y >= 254.0 / 255.0)))
    t = np.clip((censored_mass - 0.14) / (0.22 - 0.14), 0.0, 1.0)
    censor_authority = 1.0 - t * t * (3.0 - 2.0 * t)
    # A huge remote residual support means the observation is fragmented: a
    # coarse witness fitted on it is not trustworthy enough to rewrite FMMT's
    # robust empirical bootstrap.  Website-like broad corruption (~12x in the
    # regression) remains admitted; sparse replacement (~40x) is inherited.
    u = np.clip((tail_ratio - 16.0) / (32.0 - 16.0), 0.0, 1.0)
    tail_authority = 1.0 - u * u * (3.0 - 2.0 * u)
    witness_authority = censor_authority * tail_authority

    low = ndimage.gaussian_filter(provisional, split_sigma, mode="reflect")
    fine = provisional - low
    conductance = ndimage.gaussian_filter(
        (1.0 - np.maximum(coarse_support, fine_support)) * witness_authority,
        1.2, mode="reflect")
    state = low.copy()
    for _ in range(int(sweeps)):
        state = _conservative_support_step(state, conductance, dt=dt)
    certified = np.clip(fine + state, 0.0, 1.0)
    admitted_support = np.maximum(coarse_support, fine_support)
    # Geometry uses the same support statement but not the same operation.
    # A reliable unsupported contrast is made cheaper to cross; supported or
    # censored contrast retains the original FMMT resistance.
    barrier_gate = ndimage.gaussian_filter(
        1.0 - witness_authority * (1.0 - admitted_support),
        1.0, mode="reflect")
    return certified, barrier_gate, {
        "cross_predictive_tail_ratio": tail_ratio,
        "censored_observation_mass": censored_mass,
        "support_witness_authority": float(witness_authority),
        "censoring_authority": float(censor_authority),
        "remote_support_authority": float(tail_authority),
        "mean_coarse_support": float(np.mean(coarse_support)),
        "mean_fine_ancestry": float(np.mean(fine_support)),
        "mean_bootstrap_conductance": float(np.mean(conductance)),
        "mean_eikonal_barrier_gate": float(np.mean(barrier_gate)),
        "bootstrap_support_sweeps": int(sweeps),
    }


# -----------------------------------------------------------------------------
# Discrete eikonal geometry and joint measure transport.

def _build_graph(pilot, scale, alpha=3.0, barrier_gate=None):
    h, w = pilot.shape
    n = h * w
    idx = np.arange(n).reshape(h, w)
    rows, cols, vals = [], [], []
    for dy, dx, length in ((0, 1, 1.0), (1, 0, 1.0),
                           (1, 1, math.sqrt(2.0)), (1, -1, math.sqrt(2.0))):
        y0, y1 = max(0, -dy), min(h, h - dy)
        x0, x1 = max(0, -dx), min(w, w - dx)
        a = idx[y0:y1, x0:x1].ravel()
        b = idx[y0 + dy:y1 + dy, x0 + dx:x1 + dx].ravel()
        pa = pilot[y0:y1, x0:x1].ravel()
        pb = pilot[y0 + dy:y1 + dy, x0 + dx:x1 + dx].ravel()
        sa = scale[y0:y1, x0:x1].ravel()
        sb = scale[y0 + dy:y1 + dy, x0 + dx:x1 + dx].ravel()
        standardized = np.abs(pa - pb) / (0.5 * (sa + sb) + 0.01)
        if barrier_gate is None:
            edge_gate = 1.0
        else:
            ga = barrier_gate[y0:y1, x0:x1].ravel()
            gb = barrier_gate[y0 + dy:y1 + dy, x0 + dx:x1 + dx].ravel()
            edge_gate = 0.5 * (ga + gb)
        cost = length * (1.0 + alpha * edge_gate
                         * np.minimum(standardized, 4.0) ** 1.5)
        rows.extend((a, b)); cols.extend((b, a)); vals.extend((cost, cost))
    return sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    )


def _anchors(shape, stride):
    h, w = shape
    ys = np.arange(stride // 2, h, stride)
    xs = np.arange(stride // 2, w, stride)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    return (yy * w + xx).ravel().astype(np.int64)


def _joint_transport(graph, anchors, packet_fields, tau, limit_factor=3.5, batch=16):
    """Transport several measures on exactly the same ordered fronts.

    packet_fields is a list of (N, K_j) arrays. Each anchor contributes a full
    vector-valued packet. Dijkstra distances are computed once per batch and
    used for every measure, so signal prior and residual law share identical
    path geometry and attenuation.
    """
    n = graph.shape[0]
    widths = [field.shape[1] for field in packet_fields]
    packet = np.concatenate(packet_fields, axis=1)
    accumulated_packet = np.zeros((n, packet.shape[1]), np.float64)
    mass = np.zeros(n, np.float64)
    limit = tau * limit_factor
    for start in range(0, len(anchors), batch):
        ids = anchors[start:start + batch]
        dist = dijkstra(graph, directed=False, indices=ids, limit=limit,
                        return_predecessors=False)
        if dist.ndim == 1:
            dist = dist[None, :]
        finite = np.isfinite(dist)
        # The distance array is dead after attenuation. Reuse it in place so
        # front batching pays for one BxN float64 field rather than both
        # distance and weight fields.
        np.minimum(dist, 700.0, out=dist)
        np.negative(dist, out=dist)
        dist /= tau
        np.exp(dist, out=dist)
        weight = dist
        weight[~finite] = 0.0
        mass += weight.sum(axis=0)
        accumulated_packet += weight.T @ packet[ids]
    split = np.cumsum(widths[:-1])
    return list(np.split(accumulated_packet, split, axis=1)), mass


def _interp_residual_law(hist, residual, lo=-1.0, hi=1.0):
    bins = hist.shape[1]
    p = np.clip((residual - lo) / (hi - lo), 0.0, 1.0) * (bins - 1)
    low = np.floor(p).astype(np.int32)
    high = np.minimum(low + 1, bins - 1)
    frac = p - low
    rr = np.arange(hist.shape[0])[:, None]
    return hist[rr, low] * (1.0 - frac) + hist[rr, high] * frac


def denoise_fmmt(image, *, stride=None, bins=24, residual_bins=31,
                  alpha=3.0, packet_ratio=0.4, local_mass=4.0,
                  certify_support=True, support_sweeps=128,
                  support_operator=None, front_batch=None,
                  front_workspace_mib=256.0):
    started = time.perf_counter()
    stage_started = started
    stage_seconds = {}
    y = np.clip(np.asarray(image, np.float64), 0.0, 1.0)
    if y.ndim != 2:
        raise ValueError("FMMT-2D currently expects one grayscale 2-D image")
    h, w = y.shape
    n = h * w
    if front_batch is not None and int(front_batch) < 1:
        raise ValueError("front_batch must be positive")
    if not np.isfinite(front_workspace_mib) or front_workspace_mib <= 0.0:
        raise ValueError("front_workspace_mib must be finite and positive")
    if stride is None:
        stride = max(6, int(round(min(h, w) / 32.0)))

    # 1. Transport-derived bootstrap chart.
    provisional_x0 = _bootstrap_chart(y, bins=48)
    stage_seconds["bootstrap_chart"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    if support_operator is not None:
        x0, barrier_gate, support_diagnostics = support_operator(
            y, provisional_x0)
    elif certify_support:
        x0, barrier_gate, support_diagnostics = _certify_bootstrap_support(
            y, provisional_x0, sweeps=support_sweeps)
    else:
        x0 = provisional_x0
        barrier_gate = None
        support_diagnostics = {
            "support_witness_authority": 0.0,
            "bootstrap_support_sweeps": 0,
        }
    stage_seconds["support_birth"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()

    # 2. Robust residual scale. The 25th |residual| quantile has a 75% gross
    # contamination breakdown point, useful without naming any noise family.
    residual = y - x0
    scale = ndimage.percentile_filter(np.abs(residual), 25, size=7, mode="reflect")
    scale = scale / norm.ppf(0.625) + 0.004
    scale = np.clip(scale, 0.008, 0.20)
    stage_seconds["residual_scale"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()

    # 3. Eikonal graph.
    graph = _build_graph(x0, scale, alpha=alpha, barrier_gate=barrier_gate)
    anchors = _anchors(y.shape, stride)
    tau = 0.75 * stride
    if front_batch is None:
        # Dijkstra distance is transformed into attenuation in place, so one
        # float64 per anchor/pixel owns the explicit front workspace. The cap
        # is a representation policy only; changing it cannot alter the
        # estimator.
        workspace_bytes = float(front_workspace_mib) * 1024.0 * 1024.0
        # SciPy's sparse Dijkstra has a shallow optimum: tiny batches repeat
        # setup, while very wide batches lose cache locality on small images.
        # This cap grows only with representation size and remains independent
        # of image content or reconstruction quality.
        cache_batch = max(256, min(1024, n // 128))
        front_batch = max(1, min(
            len(anchors), cache_batch,
            int(workspace_bytes // max(8 * n, 1))))
    front_batch = int(front_batch)
    stage_seconds["eikonal_graph"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()

    # 4. Packet state. The signal packet's spatial support is not fixed. Its
    # width is the ratio of local latent-signal variation to residual scale.
    # Legitimate high-frequency structure therefore remains point-like; smooth
    # exchangeable regions use a wider empirical signal packet.
    signal_atom = _soft_histogram(x0, bins, 0.0, 1.0, 1)
    signal_local = _soft_histogram(x0, bins, 0.0, 1.0, 5)
    q25 = ndimage.percentile_filter(x0, 25, size=5, mode="reflect")
    q75 = ndimage.percentile_filter(x0, 75, size=5, mode="reflect")
    signal_variation = (q75 - q25) / 1.349
    packet_mix = 1.0 / (1.0 + (signal_variation / (packet_ratio * scale + 0.01)) ** 2)
    signal_packet = ((1.0 - packet_mix[..., None]) * signal_atom
                     + packet_mix[..., None] * signal_local).reshape(n, bins)
    noise_packet = _soft_histogram(residual, residual_bins, -1.0, 1.0, 7).reshape(n, residual_bins)
    stage_seconds["packet_measures"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()

    # Same fronts, same attenuation, both full measures.
    (transported, transported_noise), front_mass = _joint_transport(
        graph, anchors, [signal_packet, noise_packet], tau=tau,
        batch=int(front_batch),
    )
    stage_seconds["ordered_front_transport"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    transported += local_mass * signal_packet
    transported_noise += local_mass * noise_packet
    front_mass = front_mass + local_mass
    signal_prior = transported / np.maximum(front_mass[:, None], 1e-12)
    noise_law = transported_noise / np.maximum(front_mass[:, None], 1e-12)

    # 5. Nonparametric observation update.
    values = np.linspace(0.0, 1.0, bins)
    candidate_residual = y.ravel()[:, None] - values[None, :]
    likelihood = _interp_residual_law(noise_law, candidate_residual, -1.0, 1.0)
    # A tiny uniform contamination floor prevents zero histogram bins from
    # creating impossible observations; it is not a named noise component.
    likelihood += 0.01 * np.mean(likelihood, axis=1, keepdims=True) + 1e-12
    posterior = signal_prior * likelihood
    posterior /= np.maximum(posterior.sum(axis=1, keepdims=True), 1e-12)
    posterior_mean = (posterior @ values).reshape(h, w)

    # 6. Entropy-controlled state update.
    entropy_norm = (-np.sum(posterior * np.log(posterior + 1e-12), axis=1)
                    / np.log(bins)).reshape(h, w)
    inertia = np.clip(1.5 * (entropy_norm - 0.20), 0.08, 0.60)
    out = inertia * x0 + (1.0 - inertia) * posterior_mean
    stage_seconds["posterior_update"] = time.perf_counter() - stage_started
    stage_seconds["total"] = time.perf_counter() - started

    diagnostics = {
        "backend": "8-neighbor ordered graph fast marching (Dijkstra), discrete eikonal",
        "shape": [int(h), int(w)],
        "stride": int(stride),
        "anchors": int(len(anchors)),
        "front_batch": int(front_batch),
        "front_workspace_mib": float(front_workspace_mib),
        "alpha": float(alpha),
        "travel_tau": float(tau),
        "bins": int(bins),
        "residual_bins": int(residual_bins),
        "residual_scale_median": float(np.median(scale)),
        "signal_packet_local_mix_mean": float(np.mean(packet_mix)),
        "signal_packet_local_mix_median": float(np.median(packet_mix)),
        "posterior_entropy_normalized_mean": float(np.mean(entropy_norm)),
        "inertia_mean": float(np.mean(inertia)),
        "certified_support_birth": bool(certify_support),
        "stage_seconds": stage_seconds,
        **support_diagnostics,
    }
    return np.clip(out, 0.0, 1.0), diagnostics


# -----------------------------------------------------------------------------
# IO and reproducible benchmark.

def _load_gray(path):
    return np.asarray(Image.open(path).convert("L"), np.float64) / 255.0


def _save_gray(path, image):
    Image.fromarray(np.uint8(np.round(np.clip(image, 0, 1) * 255.0))).save(path)


def _benchmark(csv_path):
    from scipy.signal import wiener
    from skimage import color, data, img_as_float
    from skimage.metrics import structural_similarity
    from skimage.restoration import denoise_tv_chambolle
    from skimage.transform import resize

    def gray(a):
        a = img_as_float(a)
        if a.ndim == 3:
            a = color.rgb2gray(a)
        return resize(a, (96, 96), anti_aliasing=True)

    sources = {name: gray(getattr(data, name)())
               for name in ("astronaut", "coffee", "chelsea", "rocket")}

    def corrupt(clean, kind, rng):
        if kind == "gaussian":
            return np.clip(clean + rng.normal(0, 0.08, clean.shape), 0, 1)
        if kind == "poisson":
            return np.clip(rng.poisson(clean * 18) / 18, 0, 1)
        if kind == "speckle":
            return np.clip(clean * (1 + rng.normal(0, 0.25, clean.shape)), 0, 1)
        if kind == "heteroscedastic":
            return np.clip(clean + rng.normal(0, 0.025 + 0.10 * np.sqrt(clean), clean.shape), 0, 1)
        if kind in ("sp20", "sp61"):
            density = 0.20 if kind == "sp20" else 0.61
            out = clean.copy(); m = rng.random(clean.shape)
            out[m < density / 2] = 0.0
            out[(m >= density / 2) & (m < density)] = 1.0
            return out
        if kind == "mixed":
            out = rng.poisson(clean * 25) / 25 + rng.normal(0, 0.025, clean.shape)
            m = rng.random(clean.shape)
            out[m < 0.025] = 0.0; out[(m >= 0.025) & (m < 0.05)] = 1.0
            return np.clip(out, 0, 1)
        raise ValueError(kind)

    def metric(clean, estimate):
        return (float(np.mean((estimate - clean) ** 2)),
                float(structural_similarity(clean, estimate, data_range=1.0)))

    rows = []
    noises = ("gaussian", "poisson", "speckle", "heteroscedastic", "sp20", "sp61", "mixed")
    for ni, noise in enumerate(noises):
        for ii, (name, clean) in enumerate(sources.items()):
            rng = np.random.default_rng(41000 + 100 * ni + ii)
            noisy = corrupt(clean, noise, rng)
            start = time.perf_counter(); ours, diag = denoise_fmmt(noisy); elapsed = time.perf_counter() - start

            medians = {s: ndimage.median_filter(noisy, size=s, mode="reflect") for s in (3, 5, 7, 9, 11)}
            tvs = {w: denoise_tv_chambolle(noisy, weight=w, channel_axis=None)
                   for w in (0.03, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5)}
            wieners = {s: np.clip(wiener(noisy, (s, s)), 0, 1) for s in (3, 5, 7, 9)}
            median_param, median = min(medians.items(), key=lambda kv: metric(clean, kv[1])[0])
            tv_param, tv = min(tvs.items(), key=lambda kv: metric(clean, kv[1])[0])
            wiener_param, wi = min(wieners.items(), key=lambda kv: metric(clean, kv[1])[0])
            for method, estimate, param in (
                ("FMMT_certified", ours, "blind"),
                ("median_oracle", median, median_param),
                ("tv_oracle", tv, tv_param),
                ("wiener_oracle", wi, wiener_param),
            ):
                mse, ssim = metric(clean, estimate)
                rows.append({"noise": noise, "image": name, "method": method,
                             "mse": mse, "ssim": ssim, "parameter": param,
                             "seconds": elapsed if method == "FMMT" else 0.0})

    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    for method in sorted({row["method"] for row in rows}):
        rr = [row for row in rows if row["method"] == method]
        print(f"{method:14s} MSE={np.mean([x['mse'] for x in rr]):.6f} "
              f"SSIM={np.mean([x['ssim'] for x in rr]):.6f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    den = sub.add_parser("denoise")
    den.add_argument("input")
    den.add_argument("output")
    den.add_argument("--stride", type=int)
    den.add_argument("--alpha", type=float, default=3.0)
    den.add_argument("--packet-ratio", type=float, default=0.4)
    den.add_argument("--diagnostics")
    den.add_argument("--plain-fmmt", action="store_true", help="disable certified support birth")
    bench = sub.add_parser("benchmark")
    bench.add_argument("--csv", default="fmmt_benchmark.csv")
    args = parser.parse_args()

    if args.command == "denoise":
        image = _load_gray(args.input)
        start = time.perf_counter()
        out, diag = denoise_fmmt(image, stride=args.stride, alpha=args.alpha,
                                 packet_ratio=args.packet_ratio,
                                 certify_support=not args.plain_fmmt)
        diag["seconds"] = time.perf_counter() - start
        _save_gray(args.output, out)
        if args.diagnostics:
            Path(args.diagnostics).write_text(json.dumps(diag, indent=2))
        print(json.dumps(diag, indent=2))
    else:
        _benchmark(args.csv)


if __name__ == "__main__":
    main()
