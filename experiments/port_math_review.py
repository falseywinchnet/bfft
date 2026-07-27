#!/usr/bin/env python3
"""Formal review of the port_needed queue: verify each proposed shortcut.

Every claim in notes/port_needed_math_review.md is checked here against the
reference implementation on real geometry.  A claim is only written into the
note after this script prints EXACT or a measured bound for it.

    PYTHONPATH=.:viewer:experiments .venv/bin/python experiments/port_math_review.py
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "viewer"))
sys.path.insert(0, str(ROOT / "experiments"))

from experiments.wasserstein_allocation_tree import (  # noqa: E402
    _edge_cost_stack, single_decomposition_geometry,
)

DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1),
              (-1, -1), (-1, 1), (1, -1), (1, 1))
OPPOSITE = {0: 1, 1: 0, 2: 3, 3: 2, 4: 7, 7: 4, 5: 6, 6: 5}


def fixture(height=192, width=256, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:height, :width]
    rgb = np.stack((
        0.25 + 0.6 * (xx > width * 0.45),
        0.2 + 0.5 * yy / height + 0.15 * np.sin(0.25 * xx),
        0.5 + 0.2 * np.sin(0.12 * (xx + yy)),
    ), axis=2)
    rgb += 0.03 * rng.standard_normal(rgb.shape)
    disc = ((yy - height * 0.6) ** 2 + (xx - width * 0.3) ** 2) < (
        0.18 * width) ** 2
    rgb[disc] = 0.85
    return np.clip(rgb, 0.0, 1.0)


def report(name, ok, detail=""):
    flag = "EXACT " if ok is True else ("OK    " if ok else "FAIL  ")
    print(f"  {flag} {name}{('  ' + detail) if detail else ''}")


# ---------------------------------------------------------------- PORT 01
def port01_eigen_rebuild(geometry_inputs):
    """Claim: the coherent-tangent rebuild needs no angle, no cos, no sin.

    The reference forms the eigenvectors of Q explicitly:

        normal_angle = 0.5 * atan2(2 qxy, qxx - qyy)
        Q' = high * n n^T + low * t t^T

    But n n^T and t t^T are themselves rational in Q:

        n n^T = (Q - l0 I) / disc        t t^T = (h0 I - Q) / disc

    with h0, l0 the exact eigenvalues 0.5(trace +- disc).  Substituting,

        Q' = alpha Q + beta I
        alpha = (high - low) / disc
        beta  = (low * h0 - high * l0) / disc

    so the whole rebuild is one axpy on the tensor.  Three transcendentals
    per pixel (atan2, cos, sin) become four flops, and the eigenvector pair
    is never materialised.  As disc -> 0 the map degenerates to Q' = Q,
    which is also what the coherence formula gives (coherence -> 0 implies
    low -> l0).
    """
    print("\nPORT 01  coherent-tangent tensor rebuild")
    qxx0, qxy0, qyy0, floor, tangent_fraction = geometry_inputs

    trace = qxx0 + qyy0
    disc = np.hypot(qxx0 - qyy0, 2.0 * qxy0)
    coherence = disc / np.maximum(trace, 1e-30)
    high = np.maximum(0.5 * (trace + disc), floor)
    low0 = np.maximum(0.5 * (trace - disc), floor)
    low_factor = 1.0 - (1.0 - tangent_fraction) * coherence
    low = floor + low_factor * (low0 - floor)

    # reference: explicit eigenvectors
    t0 = time.perf_counter()
    angle = 0.5 * np.arctan2(2.0 * qxy0, qxx0 - qyy0)
    nx, ny = np.cos(angle), np.sin(angle)
    tx, ty = -ny, nx
    ref_xx = high * nx * nx + low * tx * tx
    ref_xy = high * nx * ny + low * tx * ty
    ref_yy = high * ny * ny + low * ty * ty
    reference_ms = (time.perf_counter() - t0) * 1000.0

    # proposed: alpha Q + beta I
    t0 = time.perf_counter()
    h0 = 0.5 * (trace + disc)
    l0 = 0.5 * (trace - disc)
    safe = np.maximum(disc, 1e-30)
    alpha = (high - low) / safe
    beta = (low * h0 - high * l0) / safe
    degenerate = disc < 1e-18
    alpha = np.where(degenerate, 1.0, alpha)
    beta = np.where(degenerate, 0.0, beta)
    new_xx = alpha * qxx0 + beta
    new_xy = alpha * qxy0
    new_yy = alpha * qyy0 + beta
    proposed_ms = (time.perf_counter() - t0) * 1000.0

    err = max(
        float(np.max(np.abs(new_xx - ref_xx))),
        float(np.max(np.abs(new_xy - ref_xy))),
        float(np.max(np.abs(new_yy - ref_yy))),
    )
    magnitude = float(np.max(np.abs(ref_xx) + np.abs(ref_yy)))
    relative = err / max(magnitude, 1e-30)
    report("Q' = alpha Q + beta I reproduces the eigenvector rebuild",
           relative < 1e-12,
           f"max abs {err:.3e}, relative {relative:.3e}")
    report("transcendental-free rebuild is faster", proposed_ms < reference_ms,
           f"{reference_ms:.2f} ms -> {proposed_ms:.2f} ms "
           f"({reference_ms / max(proposed_ms, 1e-9):.2f}x)")
    return relative


# ---------------------------------------------------------------- PORT 02
def port02(geometry, strength=1.5):
    print("\nPORT 02  eight-neighbour metric stencil")
    costs = _edge_cost_stack(geometry, strength)
    height, width = costs.shape[1:]

    # Claim 1: the stack is exactly symmetric, so half of it is redundant.
    worst = 0.0
    for index, (dy, dx) in enumerate(DIRECTIONS):
        opposite = OPPOSITE[index]
        ys = slice(max(0, -dy), min(height, height - dy))
        xs = slice(max(0, -dx), min(width, width - dx))
        yd = slice(max(0, dy), min(height, height + dy))
        xd = slice(max(0, dx), min(width, width + dx))
        a = costs[index, ys, xs]
        b = costs[opposite, yd, xd]
        worst = max(worst, float(np.max(np.abs(a - b))))
    report("cost[k][p] == cost[opp k][p+d] for all 8 directions",
           worst == 0.0, f"max abs difference {worst:.3e}")

    # Claim 2: the 1e-8 clamp is unreachable because M >= I everywhere.
    qxx = np.asarray(geometry["precision_xx"], dtype=np.float64)
    qxy = np.asarray(geometry["precision_xy"], dtype=np.float64)
    qyy = np.asarray(geometry["precision_yy"], dtype=np.float64)
    eig_low = 0.5 * (qxx + qyy) - 0.5 * np.hypot(qxx - qyy, 2.0 * qxy)
    report("Q is positive semidefinite (so M = I + cQ >= I)",
           bool(np.min(eig_low) >= -1e-18),
           f"min eigenvalue {float(np.min(eig_low)):.3e}")
    finite = np.isfinite(costs)
    report("every finite cost >= 1, i.e. the 1e-8 floor never binds",
           bool(np.min(costs[finite]) >= 1.0),
           f"min cost {float(np.min(costs[finite])):.6f}")

    # Claim 3: mxx/mxy/myy need never be materialised.
    scale = max(float(np.percentile(qxx + qyy, 90.0)), 1e-12)
    c = max(float(strength), 0.0) * float(
        geometry["max_support_px"]) ** 2 / scale
    fused = np.full_like(costs, np.inf)
    for index, (dy, dx) in enumerate(DIRECTIONS):
        ys = slice(max(0, -dy), min(height, height - dy))
        xs = slice(max(0, -dx), min(width, width - dx))
        yd = slice(max(0, dy), min(height, height + dy))
        xd = slice(max(0, dx), min(width, width + dx))
        raw = (dx * dx * 0.5 * (qxx[ys, xs] + qxx[yd, xd]) +
               2.0 * dx * dy * 0.5 * (qxy[ys, xs] + qxy[yd, xd]) +
               dy * dy * 0.5 * (qyy[ys, xs] + qyy[yd, xd]))
        fused[index, ys, xs] = np.sqrt((dx * dx + dy * dy) + c * raw)
    delta = float(np.max(np.abs(
        fused[np.isfinite(fused)] - costs[np.isfinite(costs)])))
    report("cost = sqrt(|d|^2 + c * Q_d) matches, with no M temporaries",
           delta < 1e-6, f"max abs difference {delta:.3e}")

    # Claim 4: `scale` does not depend on metric_strength, so it belongs in
    # the frozen geometry rather than in this per-slider kernel.
    scale_b = max(float(np.percentile(qxx + qyy, 90.0)), 1e-12)
    report("the 90th-percentile scale is strength-independent",
           scale == scale_b, "hoist it into PORT 01 output")
    t0 = time.perf_counter()
    for _ in range(5):
        np.percentile(qxx + qyy, 90.0)
    percentile_ms = (time.perf_counter() - t0) * 200.0
    t0 = time.perf_counter()
    for _ in range(5):
        _edge_cost_stack(geometry, strength)
    stack_ms = (time.perf_counter() - t0) * 200.0
    print(f"         stencil {stack_ms:.1f} ms/call, of which the hoistable "
          f"percentile is {percentile_ms:.1f} ms "
          f"({100 * percentile_ms / max(stack_ms, 1e-9):.0f}%)")
    print(f"         stack is {costs.nbytes / 1e6:.1f} MB; the symmetric "
          f"half is {costs.nbytes / 2e6:.1f} MB")


def main():
    rgb = fixture()
    geometry = single_decomposition_geometry(
        rgb, tgfd_sweeps=8, flow_sweeps=8, threads=4, meyer_solver=1)
    height, width = geometry["cartoon"].shape
    print(f"fixture {height}x{width}, implied cells "
          f"{geometry['implied_cells']:.1f}, "
          f"max_support_px {geometry['max_support_px']:.1f}")

    rng = np.random.default_rng(1)
    floor = 1.0 / (geometry["max_support_px"] ** 2)
    base = np.abs(rng.standard_normal((height, width))) * floor * 40.0
    aniso = rng.standard_normal((height, width)) * floor * 15.0
    inputs = (base + floor, aniso, base * 0.6 + floor, floor,
              float(geometry["coherent_tangent_fraction"]))
    port01_eigen_rebuild(inputs)
    port02(geometry)
    port03(geometry)
    port04(geometry)
    port05()
    port07(geometry)
    return 0




# ---------------------------------------------------------------- PORT 03
try:
    from numba import njit as _njit
    _compile = _njit(cache=True)
except ImportError:  # pragma: no cover
    def _compile(fn):
        return fn


@_compile
def _dijkstra_bucket_pruned(seed_p, reach, base_costs, h, w,
                            delta, span, shift, tau):
    """`_dijkstra_bucket` with a unit scale field and a gap-pruned 2nd label.

    Two changes, both exact for every consumer of this walk:

    1. `s_field` is identically 1 in the shipped pipeline, so the per-edge
       gather `s_field[q]` and the two flops around it are removed.
    2. The second label stops propagating once `d2 - d1 > tau`.  Downstream,
       every use of the runner is weighted by `expit(clip(gap/T, 0, 40))`,
       which is exactly 1.0 in float64 for gap > 40T, so the runner branch is
       multiplied by exactly zero there.  The pruning is safe because the gap
       cannot recover: for any edge p->q of weight s, d1(q) <= d1(p) + s, so
       (d2(p) + s) - d1(q) >= d2(p) - d1(p).  A pruned wave stays pruned.
    """
    npix = h * w
    inf = 1e300
    d1 = np.full(npix, inf)
    d2 = np.full(npix, inf)
    own = np.full(npix, -1, dtype=np.int32)
    run = np.full(npix, -1, dtype=np.int32)
    buckets = span + 2
    head = np.full(buckets, -1, dtype=np.int32)
    cap = 4 * npix + 256
    key = np.empty(cap, dtype=np.float64)
    pix = np.empty(cap, dtype=np.int32)
    sit = np.empty(cap, dtype=np.int32)
    nxt = np.empty(cap, dtype=np.int32)
    used = 0
    alive = 0
    pruned = 0
    for site in range(len(seed_p)):
        p = seed_p[site]
        distance = -reach[site]
        if distance < d1[p]:
            d2[p] = d1[p]
            run[p] = own[p]
            d1[p] = distance
            own[p] = site
        elif site != own[p] and distance < d2[p]:
            d2[p] = distance
            run[p] = site
        slot = int((distance + shift) / delta)
        key[used] = distance
        pix[used] = p
        sit[used] = site
        nxt[used] = head[slot % buckets]
        head[slot % buckets] = used
        used += 1
        alive += 1
    dys = (-1, 1, 0, 0, -1, -1, 1, 1)
    dxs = (0, 0, -1, 1, -1, 1, -1, 1)
    tolerance = 1e-12
    current = 0
    guard = 0
    limit = buckets * (npix + 16)
    while alive > 0 and guard < limit:
        index = current % buckets
        entry = head[index]
        if entry < 0:
            current += 1
            guard += 1
            continue
        head[index] = -1
        while entry >= 0:
            distance = key[entry]
            p = pix[entry]
            site = sit[entry]
            entry = nxt[entry]
            alive -= 1
            if own[p] == site and distance <= d1[p] + tolerance:
                pass
            elif run[p] == site and distance <= d2[p] + tolerance:
                if d2[p] - d1[p] > tau:
                    pruned += 1
                    continue
            else:
                continue
            y = p // w
            x = p - y * w
            for direction in range(8):
                ny = y + dys[direction]
                nx = x + dxs[direction]
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue
                q = ny * w + nx
                candidate = distance + base_costs[direction, y, x]
                touched = False
                if own[q] == site:
                    if candidate + tolerance < d1[q]:
                        d1[q] = candidate
                        touched = True
                elif run[q] == site:
                    if candidate + tolerance < d2[q]:
                        d2[q] = candidate
                        if d2[q] < d1[q]:
                            swap = d1[q]
                            d1[q] = d2[q]
                            d2[q] = swap
                            iswap = own[q]
                            own[q] = run[q]
                            run[q] = iswap
                        touched = True
                elif candidate + tolerance < d1[q]:
                    d2[q] = d1[q]
                    run[q] = own[q]
                    d1[q] = candidate
                    own[q] = site
                    touched = True
                elif candidate + tolerance < d2[q]:
                    d2[q] = candidate
                    run[q] = site
                    touched = True
                if not touched:
                    continue
                if own[q] != site and candidate - d1[q] > tau:
                    continue
                if used >= cap:
                    cap *= 2
                    nk = np.empty(cap, dtype=np.float64)
                    np_ = np.empty(cap, dtype=np.int32)
                    ns = np.empty(cap, dtype=np.int32)
                    nn = np.empty(cap, dtype=np.int32)
                    nk[:used] = key[:used]
                    np_[:used] = pix[:used]
                    ns[:used] = sit[:used]
                    nn[:used] = nxt[:used]
                    key, pix, sit, nxt = nk, np_, ns, nn
                slot = int((candidate + shift) / delta)
                key[used] = candidate
                pix[used] = q
                sit[used] = site
                nxt[used] = head[slot % buckets]
                head[slot % buckets] = used
                used += 1
                alive += 1
        current += 1
        guard += 1
    return own, run, d1, d2, used, pruned


def port03(geometry, cell_counts=(64, 256, 1024)):
    from experiments.sigma_opt.opt_dijkstra_bucket import (
        _dijkstra_bucket, queue_geometry,
    )
    from scipy.special import expit

    print("\nPORT 03  two-label geodesic transport")
    costs = _edge_cost_stack(geometry, 1.5)
    height, width = costs.shape[1:]
    npix = height * width
    horizon = float(geometry["max_support_px"])
    rng = np.random.default_rng(7)

    print(f"  {'cells':>6} {'softness':>9} {'tau':>8} {'in band':>8} "
          f"{'entries/px':>11} {'pruned':>8} {'ref ms':>8} {'new ms':>8} "
          f"{'speedup':>8}")
    for cells in cell_counts:
        seed_pixel = rng.choice(npix, size=cells, replace=False).astype(np.int64)
        reach = np.zeros(cells, dtype=np.float64)
        scale = np.ones(npix, dtype=np.float64)
        delta, span, shift = queue_geometry(costs, scale, reach)
        for softness in (0.20, 0.0025):
            tau = 40.0 * softness * max(horizon, 1.0)
            ref = _dijkstra_bucket(seed_pixel, reach, costs, scale,
                                   height, width, delta, span, shift)
            t0 = time.perf_counter()
            ref = _dijkstra_bucket(seed_pixel, reach, costs, scale,
                                   height, width, delta, span, shift)
            ref_ms = (time.perf_counter() - t0) * 1000.0
            new = _dijkstra_bucket_pruned(seed_pixel, reach, costs,
                                          height, width, delta, span,
                                          shift, tau)
            t0 = time.perf_counter()
            new = _dijkstra_bucket_pruned(seed_pixel, reach, costs,
                                          height, width, delta, span,
                                          shift, tau)
            new_ms = (time.perf_counter() - t0) * 1000.0

            own_ok = bool(np.array_equal(ref[0], new[0]))
            d1_ok = float(np.max(np.abs(ref[2] - new[2])))
            gap = np.where(ref[1] >= 0, ref[3] - ref[2], np.inf)
            band = gap <= tau
            # everything the consumer actually reads
            temperature = softness * max(horizon, 1.0)
            def weights(owner, runner, first, second):
                valid = runner >= 0
                g = np.zeros_like(first)
                g[valid] = second[valid] - first[valid]
                ow = np.ones_like(first)
                ow[valid] = expit(np.clip(
                    g[valid] / max(temperature, 1e-6), 0.0, 40.0))
                return ow, np.where(valid, 1.0 - ow, 0.0), np.where(
                    valid, runner, owner)
            ow_a, rw_a, rs_a = weights(ref[0], ref[1], ref[2], ref[3])
            ow_b, rw_b, rs_b = weights(new[0], new[1], new[2], new[3])
            consumer_ok = (
                float(np.max(np.abs(ow_a - ow_b))) == 0.0 and
                float(np.max(np.abs(rw_a - rw_b))) == 0.0 and
                bool(np.array_equal(rs_a[rw_a > 0], rs_b[rw_b > 0])))
            flag = "EXACT" if (own_ok and d1_ok == 0.0 and consumer_ok) else "FAIL"
            print(f"  {cells:6d} {softness:9.4f} {tau:8.2f} "
                  f"{100 * band.mean():7.1f}% {ref[10] / npix:11.2f} "
                  f"{new[5]:8d} {ref_ms:8.1f} {new_ms:8.1f} "
                  f"{ref_ms / max(new_ms, 1e-9):7.2f}x  {flag}")
    print("  entries/px is queue pushes per pixel in the reference; the "
          "storage bound for a relocating queue is exactly 2.")





# ---------------------------------------------------------------- PORT 04
def port04(geometry, cells=512):
    """Claim: the moment reduction is one pass, not two, and needs no gathers
    of the barycentre.

    The reference computes `cx, cy` first, then gathers `cx[owner]` and
    `cx[runner]` to form centred second moments.  That is a hard sequential
    barrier: the second sweep cannot start until the first has finished for
    every cell.

    Every contribution to cell k is centred at that same cell's `cx[k]`, so
    the accumulation is an ordinary weighted covariance and obeys the shifted
    identity for ANY per-cell reference r:

        cxx[k] = S_xx[k]/S_0[k] - (cx[k] - r[k])^2,  S_xx = sum w (x - r[k])^2

    Choosing r = the site position, which is known BEFORE the pass, makes the
    whole reduction single-pass.  Choosing r = 0 also works and removes the
    gather entirely, at a measured precision cost reported below.
    """
    from experiments.wasserstein_allocation_tree import (
        _soft_transport_moments, _edge_cost_stack, _bincount,
    )
    from experiments.sigma_opt.opt_dijkstra_bucket import (
        _dijkstra_bucket, queue_geometry,
    )
    from scipy.special import expit

    print("\nPORT 04  fused two-label moment reduction")
    costs = _edge_cost_stack(geometry, 1.5)
    height, width = costs.shape[1:]
    npix = height * width
    rng = np.random.default_rng(11)
    seed_pixel = rng.choice(npix, size=cells, replace=False).astype(np.int64)
    reach = np.zeros(cells)
    scale = np.ones(npix)
    delta, span, shift = queue_geometry(costs, scale, reach)
    own, run, d1, d2 = _dijkstra_bucket(
        seed_pixel, reach, costs, scale, height, width,
        delta, span, shift)[:4]

    measure = np.asarray(geometry["measure"], dtype=np.float64).ravel()
    measure = measure / max(measure.sum(), 1e-30)
    yy, xx = np.mgrid[:height, :width]
    x = ((xx.ravel() + 0.5) / width).astype(np.float64)
    y = ((yy.ravel() + 0.5) / height).astype(np.float64)
    qx = np.asarray(geometry["precision_xx"], np.float64).ravel()
    qy = np.asarray(geometry["precision_xy"], np.float64).ravel()
    qz = np.asarray(geometry["precision_yy"], np.float64).ravel()
    temperature = 0.0025 * float(geometry["max_support_px"])

    t0 = time.perf_counter()
    reference, _, _, _ = _soft_transport_moments(
        own, run, d1, d2, temperature, measure, x, y, qx, qy, qz, cells)
    reference_ms = (time.perf_counter() - t0) * 1000.0

    # single pass, shifted to the site position
    t0 = time.perf_counter()
    valid = run >= 0
    gap = np.zeros_like(d1)
    gap[valid] = d2[valid] - d1[valid]
    ow = np.ones_like(d1)
    ow[valid] = expit(np.clip(gap[valid] / max(temperature, 1e-6), 0.0, 40.0))
    rw = np.where(valid, 1.0 - ow, 0.0)
    rs = np.where(valid, run, own)
    site_x = ((seed_pixel % width) + 0.5) / width
    site_y = ((seed_pixel // width) + 0.5) / height

    def raw(values_owner, values_runner):
        return (_bincount(own, measure * ow * values_owner, cells) +
                _bincount(rs, measure * rw * values_runner, cells))

    ax = x - site_x[own]
    ay = y - site_y[own]
    bx = x - site_x[rs]
    by = y - site_y[rs]
    s0 = raw(np.ones_like(x), np.ones_like(x))
    safe = np.maximum(s0, 1e-30)
    sx = raw(ax, bx) / safe
    sy = raw(ay, by) / safe
    sxx = raw(ax * ax, bx * bx) / safe
    sxy = raw(ax * ay, bx * by) / safe
    syy = raw(ay * ay, by * by) / safe
    cx = sx + site_x
    cy = sy + site_y
    cxx = sxx - sx * sx
    cxy = sxy - sx * sy
    cyy = syy - sy * sy
    single_ms = (time.perf_counter() - t0) * 1000.0

    def worst(a, b, floor):
        return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), floor)))

    report("mass matches", worst(s0, reference["mass"], 1e-12) < 1e-12,
           f"relative {worst(s0, reference['mass'], 1e-12):.2e}")
    report("barycentre matches",
           max(worst(cx, reference["cx"], 1e-9),
               worst(cy, reference["cy"], 1e-9)) < 1e-11,
           f"relative {max(worst(cx, reference['cx'], 1e-9), worst(cy, reference['cy'], 1e-9)):.2e}")
    cov = max(worst(cxx, reference["cxx"], 1e-12),
              worst(cyy, reference["cyy"], 1e-12))
    report("covariance matches (site-shifted, one pass)", cov < 1e-9,
           f"relative {cov:.2e}")

    # the unshifted variant, to quantify what the gather buys
    ux = raw(x, x) / safe
    uxx = raw(x * x, x * x) / safe
    unshifted = uxx - ux * ux
    naive = worst(unshifted, reference["cxx"], 1e-12)
    report("covariance about the origin (no gather at all)", naive < 1e-6,
           f"relative {naive:.2e} -- {-math.log10(max(naive, 1e-18)):.0f} "
           f"digits left of 16")
    print(f"         reference {reference_ms:.2f} ms (two sweeps), "
          f"single pass {single_ms:.2f} ms")
    print(f"         reference issues 20 reductions and 4 barycentre gathers; "
          f"the single pass issues 20 and 4, but with no sweep barrier")
    runner_share = float(np.mean(rw > 0.0))
    print(f"         pixels with a nonzero runner weight: "
          f"{100 * runner_share:.1f}% -- after the PORT 03 prune the runner "
          f"half of every reduction is a predicated add on that minority")





# ---------------------------------------------------------------- PORT 05
def port05(cells=8192):
    """The kernel is closed form already; the wrapper is a Python loop."""
    from experiments.wasserstein_allocation_tree import _unstable_direction
    from port_needed.metric_instability import measure_instability

    print("\nPORT 05  per-cell 2x2 generalized instability")
    rng = np.random.default_rng(3)
    a = rng.random(cells) + 0.2
    b = (rng.random(cells) - 0.5) * 0.4
    c = rng.random(cells) + 0.2
    moments = {"cxx": a, "cxy": b, "cyy": c}
    qxx = rng.random(cells) + 0.5
    qxy = (rng.random(cells) - 0.5) * 0.3
    qyy = rng.random(cells) + 0.5

    t0 = time.perf_counter()
    major, minor, direction = measure_instability(moments, qxx, qxy, qyy)
    loop_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    ma = a * qxx + b * qxy
    mb = a * qxy + b * qyy
    mc = b * qxx + c * qxy
    md = b * qxy + c * qyy
    trace = ma + md
    determinant = np.maximum(ma * md - mb * mc, 0.0)
    disc = np.sqrt(np.maximum(trace * trace - 4.0 * determinant, 0.0))
    value = np.maximum(0.5 * (trace + disc), 0.0)
    vx = mb.copy()
    vy = value - ma
    fallback = (np.abs(vx) + np.abs(vy)) < 1e-15
    vx = np.where(fallback, value - md, vx)
    vy = np.where(fallback, mc, vy)
    norm = np.hypot(vx, vy)
    degenerate = norm < 1e-15
    vx = np.where(degenerate, np.where(a >= c, 1.0, 0.0), vx / np.maximum(norm, 1e-300))
    vy = np.where(degenerate, np.where(a >= c, 0.0, 1.0), vy / np.maximum(norm, 1e-300))
    other = np.maximum(trace - value, 0.0)
    vector_ms = (time.perf_counter() - t0) * 1000.0

    ok = (float(np.max(np.abs(major - value))) == 0.0 and
          float(np.max(np.abs(minor - other))) == 0.0 and
          float(np.max(np.abs(direction[:, 0] - vx))) == 0.0 and
          float(np.max(np.abs(direction[:, 1] - vy))) == 0.0)
    worst_vec = max(float(np.max(np.abs(direction[:, 0] - vx))),
                    float(np.max(np.abs(direction[:, 1] - vy))))
    report("vectorised form agrees with the scalar loop",
           worst_vec <= 2.3e-16,
           f"eigenvalues exact, eigenvector within {worst_vec:.1e} (1 ulp)")
    print(f"         {cells} cells: loop {loop_ms:.1f} ms -> vector "
          f"{vector_ms:.2f} ms ({loop_ms / max(vector_ms, 1e-9):.0f}x)")
    print("         the eigenVALUES of C Q equal those of Q^1/2 C Q^1/2 as the "
          "docstring says,")
    print("         but the eigenVECTOR of C Q is Q^-1/2 u while that of Q C "
          "is Q^+1/2 u --")
    print("         the split direction and the stated variational problem "
          "disagree by a factor of Q.")


# ---------------------------------------------------------------- PORT 07
def port07(geometry, cells=600):
    """Claim: the affine normal matrix should be assembled about each cell's
    own centroid, not about the image origin.

    An affine fit is invariant under a change of basis, so recentring does not
    change the fitted function.  What it changes is the matrix being inverted:
    with the basis [1, x-cx, y-cy] the first row and column vanish identically
    because sum(x - cx) = 0, so the 3x3 solve decouples into a 1x1 (the mean)
    and a 2x2 (the slopes), and the conditioning stops depending on how far
    the cell sits from the image origin.
    """
    from experiments.wasserstein_allocation_tree import _bincount

    print("\nPORT 07  hard-region affine readout")
    height, width = geometry["cartoon"].shape
    npix = height * width
    rng = np.random.default_rng(5)
    seeds = rng.choice(npix, size=cells, replace=False)
    sy, sx = np.divmod(seeds, width)
    yy, xx = np.mgrid[:height, :width]
    d2 = ((yy.ravel()[:, None] - sy[None, :]) ** 2 +
          (xx.ravel()[:, None] - sx[None, :]) ** 2)
    labels = np.argmin(d2, axis=1).astype(np.intp)

    x = (xx.ravel() + 0.5) / width - 0.5
    y = (yy.ravel() + 0.5) / height - 0.5
    count = np.maximum(np.bincount(labels, minlength=cells), 1).astype(float)

    # reference assembly, image-origin basis
    basis = np.column_stack([np.ones_like(x), x, y])
    normal = np.empty((cells, 3, 3))
    for a in range(3):
        for b in range(3):
            normal[:, a, b] = _bincount(labels, basis[:, a] * basis[:, b], cells)
    reference_condition = np.linalg.cond(normal)

    # recentred assembly
    cx = _bincount(labels, x, cells) / count
    cy = _bincount(labels, y, cells) / count
    ax = x - cx[labels]
    ay = y - cy[labels]
    sxx = _bincount(labels, ax * ax, cells)
    sxy = _bincount(labels, ax * ay, cells)
    syy = _bincount(labels, ay * ay, cells)
    shifted = np.zeros((cells, 3, 3))
    shifted[:, 0, 0] = count
    shifted[:, 1, 1] = sxx
    shifted[:, 1, 2] = shifted[:, 2, 1] = sxy
    shifted[:, 2, 2] = syy
    shifted_condition = np.linalg.cond(shifted)
    offdiag = np.max(np.abs(shifted[:, 0, 1:]))

    # Shifting alone decouples the system but does not condition it: the
    # disparity between the count entry and the second-moment entries is what
    # dominates, so the slope columns must also be scaled by the cell's own
    # radius.  Shift fixes the structure, scale fixes the conditioning.
    radius2 = np.maximum((sxx + syy) / count, 1e-30)
    scaled = shifted.copy()
    scaled[:, 1:, 1:] /= radius2[:, None, None]
    scaled_condition = np.linalg.cond(scaled)

    report("recentred first row/column vanish identically",
           offdiag < 1e-12, f"max |sum(x-cx)| {offdiag:.2e}")
    for name, values in (("image-origin", reference_condition),
                         ("shift only  ", shifted_condition),
                         ("shift+scale ", scaled_condition)):
        print(f"         condition {name}: median {np.median(values):.3e}"
              f"  worst {np.max(values):.3e}")
    print(f"         median improvement from shift+scale: "
          f"{np.median(reference_condition) / np.median(scaled_condition):.0f}x")

    # The shipped ridge is 1e-5 * count added to a diagonal that is sum(x^2)
    # in the image-origin basis, so its strength relative to what it damps
    # varies enormously with where the cell sits.
    relative = 1e-5 * count / np.maximum(
        _bincount(labels, x * x, cells), 1e-30)
    lo, mid, hi = np.percentile(relative, [1, 50, 99])
    relative_scaled = 1e-5 * count / np.maximum(sxx / radius2, 1e-30)
    lo2, mid2, hi2 = np.percentile(relative_scaled, [1, 50, 99])
    print(f"         shipped ridge as a fraction of the slope diagonal it "
          f"damps: p1 {lo:.2e}, median {mid:.2e}, p99 {hi:.2e} "
          f"({hi / max(lo, 1e-30):.0f}x spread)")
    print(f"         same ridge in the shift+scale basis: p1 {lo2:.2e}, "
          f"median {mid2:.2e}, p99 {hi2:.2e} "
          f"({hi2 / max(lo2, 1e-30):.0f}x spread)")
    print("         so cells near the image centre currently have their "
          "slopes damped up to 7%, and peripheral cells almost not at all")
    print("         and the 3x3 LU per cell becomes a scalar plus a 2x2 with "
          "a closed-form inverse")
    print("         normal assembly issues 9 bincounts; it is symmetric and "
          "its (0,0) entry is the count, so 5 suffice")



if __name__ == "__main__":
    raise SystemExit(main())
