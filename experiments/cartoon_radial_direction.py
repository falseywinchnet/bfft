#!/usr/bin/env python3
"""Can an early dual direction field make the isotropic support decision?

The anisotropic taut-string experiment found jump sets quickly by changing
the functional.  This experiment keeps the original isotropic Euclidean TV
and tests a different observation: Split Bregman's dual *directions* appear
to settle much earlier than the set of saturated dual vectors.

Write a feasible isotropic ROF dual field as

    p_i = rho_i n_i,       0 <= rho_i <= 1,       |n_i| = 1.

For directions ``n`` captured after an early Split Bregman pass, the best
radial field is the convex box-constrained quadratic

    min_rho  1/2 ||g + div(rho n)/c||^2,    0 <= rho <= 1.

L-BFGS-B solves that problem only as an oracle control.  The actual candidate
copies the causal segmentation machinery: freeze the characteristic order,
take the exact clipped scalar quadratic minimizer for each radial coordinate,
update the local divergence residual immediately, and accept that coordinate
once.  It is one finite low-to-high pass, not convergence by iteration.

This is a falsification experiment, not a proposed production solver: if the
causal radial drop does not jump toward the converged isotropic objective,
local analytical minimizers are insufficient and the next construction must
reverse-accumulate whole characteristic trees before applying capacity
events.

Nothing here changes the library or approximates the Euclidean unit disk.

    PYTHONPATH=.:viewer .venv/bin/python \
        experiments/cartoon_radial_direction.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

try:
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "viewer"))

import gallery  # noqa: E402

from cartoon_stage_tautstring import _div, _grad, _solve_neumann  # noqa: E402


def _identity(function):  # pragma: no cover
    return function


_compile = njit(cache=True) if njit is not None else _identity


def isotropic_objective(u, g, c):
    gx, gy = _grad(u)
    return float(0.5 * c * np.sum((u - g) ** 2)
                 + np.sum(np.hypot(gx, gy)))


def dual_bound(px, py, g, c):
    """Feasible ROF dual value; never exceeds the primal optimum."""
    q = _div(px, py)
    return float(-np.vdot(g, q).real - 0.5 * np.vdot(q, q).real / c)


def relative_gap(u, px, py, g, c):
    primal = isotropic_objective(u, g, c)
    dual = dual_bound(px, py, g, c)
    return (primal - dual) / max(abs(primal), 1.0)


def split_bregman_states(g, c, eta, iterations, checkpoints):
    """Neumann isotropic Split Bregman, retaining its dual projection."""
    bx = np.zeros_like(g)
    by = np.zeros_like(g)
    reflected_x = np.zeros_like(g)
    reflected_y = np.zeros_like(g)
    theta = 1.0 / eta
    wanted = set(checkpoints)
    states = {}
    objectives = []

    for iteration in range(1, iterations + 1):
        u = _solve_neumann(
            c * g - eta * _div(reflected_x, reflected_y), c, eta)
        gx, gy = _grad(u)
        tx, ty = gx + bx, gy + by
        magnitude = np.hypot(tx, ty)
        shrink = np.maximum(magnitude - theta, 0.0) / np.maximum(
            magnitude, 1e-300)
        dx, dy = tx * shrink, ty * shrink
        bx, by = tx - dx, ty - dy
        reflected_x, reflected_y = dx - bx, dy - by
        objectives.append(isotropic_objective(u, g, c))
        if iteration in wanted:
            # eta*b is feasible for the unit-disk ROF dual.
            states[iteration] = (
                u.copy(), (eta * bx).copy(), (eta * by).copy())
    return states, np.asarray(objectives)


def fgp_reference(g, c, max_iterations=12000, tolerance=2e-8):
    """High-accuracy isotropic dual reference with matching Neumann ends."""
    px = np.zeros_like(g)
    py = np.zeros_like(g)
    previous_x = px.copy()
    previous_y = py.copy()
    momentum = 1.0
    step = c / 8.0
    last_check = np.inf

    for iteration in range(1, max_iterations + 1):
        next_momentum = 0.5 * (
            1.0 + np.sqrt(1.0 + 4.0 * momentum * momentum))
        weight = (momentum - 1.0) / next_momentum
        yx = px + weight * (px - previous_x)
        yy = py + weight * (py - previous_y)
        u_y = g + _div(yx, yy) / c
        gx, gy = _grad(u_y)
        qx, qy = yx + step * gx, yy + step * gy
        magnitude = np.maximum(1.0, np.hypot(qx, qy))
        next_x, next_y = qx / magnitude, qy / magnitude
        previous_x, previous_y = px, py
        px, py = next_x, next_y
        momentum = next_momentum

        if iteration % 100 == 0:
            u = g + _div(px, py) / c
            gap = relative_gap(u, px, py, g, c)
            if gap <= tolerance:
                return u, px, py, iteration, gap
            # A large gap increase means FISTA crossed an active-set event.
            # Restart momentum without discarding the feasible dual field.
            if gap > last_check * 1.05:
                previous_x = px.copy()
                previous_y = py.copy()
                momentum = 1.0
            last_check = gap

    u = g + _div(px, py) / c
    return (u, px, py, max_iterations,
            relative_gap(u, px, py, g, c))


def directions_from_state(u, px, py):
    """Continuous unit directions, with a primal fallback near zero flux."""
    magnitude = np.hypot(px, py)
    gx, gy = _grad(u)
    gradient_magnitude = np.hypot(gx, gy)
    use_flux = magnitude > 1e-10
    nx = np.where(
        use_flux, px / np.maximum(magnitude, 1e-300),
        gx / np.maximum(gradient_magnitude, 1e-300))
    ny = np.where(
        use_flux, py / np.maximum(magnitude, 1e-300),
        gy / np.maximum(gradient_magnitude, 1e-300))
    undefined = np.hypot(nx, ny) < 0.5
    nx[undefined] = 1.0
    ny[undefined] = 0.0
    # Free-end divergence assumes the unused outward edge fluxes are zero.
    nx[:, -1] = 0.0
    ny[-1, :] = 0.0
    return nx, ny, np.clip(magnitude, 0.0, 1.0)


@_compile
def _causal_radial_kernel(g, c, nx, ny, rho, order):
    """One exact coordinate pass in frozen characteristic order.

    The column of ``div(rho*n)/c`` belonging to one interior rho has three
    nonzeros: ``(nx+ny)/c`` at the pixel, ``-nx/c`` to its right, and
    ``-ny/c`` below it.  Holding all other coordinates fixed therefore leaves
    a scalar bounded quadratic.  Its clipped stationary point is exact.
    Updating the residual immediately makes the pass Gauss--Seidel causal:
    every coordinate is accepted once and never revisited.
    """
    height, width = g.shape
    px = rho * nx
    py = rho * ny
    residual = g.copy()
    for y in range(height):
        for x in range(width):
            divergence = px[y, x] + py[y, x]
            if x > 0:
                divergence -= px[y, x - 1]
            if y > 0:
                divergence -= py[y - 1, x]
            residual[y, x] += divergence / c
    for index in range(order.size):
        pixel = order[index]
        y = pixel // width
        x = pixel - y * width
        ax = nx[y, x] / c
        ay = ny[y, x] / c
        center = ax + ay
        hessian = center * center
        derivative = center * residual[y, x]
        if x + 1 < width:
            hessian += ax * ax
            derivative -= ax * residual[y, x + 1]
        if y + 1 < height:
            hessian += ay * ay
            derivative -= ay * residual[y + 1, x]
        if hessian <= 1e-30:
            rho[y, x] = 0.0
            continue
        previous = rho[y, x]
        proposed = min(max(previous - derivative / hessian, 0.0), 1.0)
        delta = proposed - previous
        rho[y, x] = proposed
        residual[y, x] += center * delta
        if x + 1 < width:
            residual[y, x + 1] -= ax * delta
        if y + 1 < height:
            residual[y + 1, x] -= ay * delta
    return residual, rho


def causal_radial_drop(g, c, nx, ny, initial, order_field):
    """One noniterative radial pass, ordered like a first-arrival front."""
    # n follows the increasing primal direction, so low-to-high is the
    # corresponding accepted characteristic order. Stable sorting makes ties
    # deterministic. Production can replace this with fixed monotone buckets.
    order = np.argsort(
        np.asarray(order_field, dtype=np.float64).ravel(),
        kind="stable").astype(np.int64)
    started = time.perf_counter()
    u, rho = _causal_radial_kernel(
        np.asarray(g, dtype=np.float64),
        float(c),
        np.asarray(nx, dtype=np.float64),
        np.asarray(ny, dtype=np.float64),
        np.asarray(initial, dtype=np.float64).copy(),
        order,
    )
    elapsed = time.perf_counter() - started
    px, py = rho * nx, rho * ny
    return u, px, py, elapsed


def solve_radial(g, c, nx, ny, initial, max_iterations=1000):
    """Accurately solve the fixed-direction dual box-QP."""
    shape = g.shape
    evaluations = 0

    def value_gradient(flat):
        nonlocal evaluations
        evaluations += 1
        rho = flat.reshape(shape)
        px, py = rho * nx, rho * ny
        u = g + _div(px, py) / c
        gx, gy = _grad(u)
        value = 0.5 * np.vdot(u, u).real
        gradient = -(nx * gx + ny * gy) / c
        return float(value), gradient.ravel()

    started = time.perf_counter()
    result = minimize(
        value_gradient, initial.ravel(), method="L-BFGS-B", jac=True,
        bounds=[(0.0, 1.0)] * initial.size,
        options={"maxiter": max_iterations, "maxls": 50,
                 "ftol": 1e-14, "gtol": 1e-9,
                 "maxcor": 20})
    elapsed = time.perf_counter() - started
    rho = result.x.reshape(shape)
    px, py = rho * nx, rho * ny
    u = g + _div(px, py) / c
    return u, px, py, result, evaluations, elapsed


def correction_steps(g, c, px, py, targets=(1e-2, 1e-3),
                     max_iterations=512):
    """Unrestricted isotropic FGP steps needed after a feasible dual start."""
    previous_x = px.copy()
    previous_y = py.copy()
    momentum = 1.0
    step = c / 8.0
    reached = {target: None for target in targets}
    initial_u = g + _div(px, py) / c
    initial_gap = relative_gap(initial_u, px, py, g, c)

    for iteration in range(1, max_iterations + 1):
        next_momentum = 0.5 * (
            1.0 + np.sqrt(1.0 + 4.0 * momentum * momentum))
        weight = (momentum - 1.0) / next_momentum
        yx = px + weight * (px - previous_x)
        yy = py + weight * (py - previous_y)
        u_y = g + _div(yx, yy) / c
        gx, gy = _grad(u_y)
        qx, qy = yx + step * gx, yy + step * gy
        magnitude = np.maximum(1.0, np.hypot(qx, qy))
        next_x, next_y = qx / magnitude, qy / magnitude
        previous_x, previous_y = px, py
        px, py = next_x, next_y
        momentum = next_momentum

        u = g + _div(px, py) / c
        gap = relative_gap(u, px, py, g, c)
        for target in targets:
            if reached[target] is None and gap <= target:
                reached[target] = iteration
        if all(value is not None for value in reached.values()):
            break
    return initial_gap, reached


def jump_set(u, cut):
    gx, gy = _grad(u)
    return np.hypot(gx, gy) > cut


def jaccard(a, b):
    return (float(np.count_nonzero(a & b))
            / max(np.count_nonzero(a | b), 1))


def synthetic(side=256, seed=4):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:side, 0:side]
    y, x = yy / (side - 1), xx / (side - 1)
    image = np.zeros((side, side))
    for cy, cx, radius, value in (
            (0.30, 0.35, 0.22, 200.0),
            (0.68, 0.62, 0.18, 60.0)):
        image[(y - cy) ** 2 + (x - cx) ** 2 < radius ** 2] = value
    image += 40.0 * x
    image += 18.0 * np.sin(2.0 * np.pi * 26.0 * x) * np.cos(
        2.0 * np.pi * 21.0 * y)
    image += 6.0 * rng.standard_normal(image.shape)
    return np.clip(image, 0.0, 255.0)


def report(name, g, c=0.05, eta=0.10,
           checkpoints=(2, 4, 8, 12, 16, 24, 32)):
    print(f"\n=== {name}: {g.shape[0]}x{g.shape[1]} ===")
    reference, ref_px, ref_py, ref_iterations, ref_gap = fgp_reference(g, c)
    ref_objective = isotropic_objective(reference, g, c)
    ref_dual = dual_bound(ref_px, ref_py, g, c)
    cut = float(np.percentile(np.hypot(*_grad(reference)), 95.0))
    ref_jump = jump_set(reference, cut)
    states, bregman_objectives = split_bregman_states(
        g, c, eta, max(128, max(checkpoints)), checkpoints)
    ref_scale = max(float(np.linalg.norm(reference)), 1e-30)
    print(f"reference: FGP {ref_iterations} iterations, "
          f"relative gap {ref_gap:.2e}, objective {ref_objective:.6e}")
    print(" pass | analytic obj | dual gain | equiv | jump | correct 1% | "
          "time ms || QP gain/iters")

    rows = []
    for checkpoint in checkpoints:
        u, px, py = states[checkpoint]
        nx, ny, initial = directions_from_state(u, px, py)
        analytic, analytic_px, analytic_py, analytic_elapsed = (
            causal_radial_drop(g, c, nx, ny, initial, u))
        radial, radial_px, radial_py, result, evaluations, elapsed = (
            solve_radial(g, c, nx, ny, initial))
        b_objective = isotropic_objective(u, g, c)
        analytic_objective = isotropic_objective(analytic, g, c)
        radial_objective = isotropic_objective(radial, g, c)
        b_excess = (b_objective - ref_objective) / ref_objective
        analytic_excess = (
            analytic_objective - ref_objective) / ref_objective
        radial_excess = (radial_objective - ref_objective) / ref_objective
        b_dual_deficit = (
            ref_dual - dual_bound(px, py, g, c)) / ref_objective
        analytic_dual_deficit = (
            ref_dual - dual_bound(
                analytic_px, analytic_py, g, c)) / ref_objective
        radial_dual_deficit = (
            ref_dual - dual_bound(radial_px, radial_py, g, c)) / ref_objective
        analytic_dual_gain = (
            b_dual_deficit / max(analytic_dual_deficit, 1e-30))
        dual_gain = b_dual_deficit / max(radial_dual_deficit, 1e-30)
        analytic_equivalent = next(
            (index + 1 for index, value in enumerate(bregman_objectives)
             if value <= analytic_objective), None)
        equivalent = next(
            (index + 1 for index, value in enumerate(bregman_objectives)
             if value <= radial_objective), None)
        b_jump = jaccard(jump_set(u, cut), ref_jump)
        analytic_jump = jaccard(jump_set(analytic, cut), ref_jump)
        radial_jump = jaccard(jump_set(radial, cut), ref_jump)
        gap = relative_gap(radial, radial_px, radial_py, g, c)
        b_initial_gap, b_correction = correction_steps(g, c, px, py)
        analytic_initial_gap, analytic_correction = correction_steps(
            g, c, analytic_px, analytic_py)
        radial_initial_gap, radial_correction = correction_steps(
            g, c, radial_px, radial_py)
        relative_error = (
            float(np.linalg.norm(radial - reference)) / ref_scale)
        rows.append({
            "checkpoint": checkpoint,
            "bregman_excess": b_excess,
            "analytic_excess": analytic_excess,
            "radial_excess": radial_excess,
            "analytic_equivalent_pass": analytic_equivalent,
            "equivalent_pass": equivalent,
            "bregman_jump": b_jump,
            "analytic_jump": analytic_jump,
            "radial_jump": radial_jump,
            "radial_gap": gap,
            "bregman_dual_deficit": b_dual_deficit,
            "analytic_dual_deficit": analytic_dual_deficit,
            "radial_dual_deficit": radial_dual_deficit,
            "analytic_dual_gain": analytic_dual_gain,
            "dual_gain": dual_gain,
            "bregman_initial_gap": b_initial_gap,
            "analytic_initial_gap": analytic_initial_gap,
            "radial_initial_gap": radial_initial_gap,
            "bregman_correction": b_correction,
            "analytic_correction": analytic_correction,
            "radial_correction": radial_correction,
            "relative_error": relative_error,
            "qp_iterations": result.nit,
            "qp_evaluations": evaluations,
            "qp_success": bool(result.success),
            "qp_message": str(result.message),
            "analytic_seconds": analytic_elapsed,
            "seconds": elapsed,
        })
        analytic_equivalent_text = (
            str(analytic_equivalent)
            if analytic_equivalent is not None else ">128")
        analytic_one = analytic_correction[1e-2]
        analytic_one_text = (
            str(analytic_one) if analytic_one is not None else ">512")
        print(f"{checkpoint:5d} | {analytic_excess:12.3e} | "
              f"{analytic_dual_gain:9.2f}x | "
              f"{analytic_equivalent_text:>5s} | "
              f"{analytic_jump:4.2f} | {analytic_one_text:>10s} | "
              f"{analytic_elapsed * 1e3:7.2f} || "
              f"{dual_gain:5.2f}x/{result.nit:<4d}")
    return rows


def main():
    camera = np.asarray(gallery.load("camera"), dtype=np.float64)
    if camera.shape != (256, 256):
        camera = camera[::2, ::2]
    report("cameraman", camera)
    report("synthetic", synthetic())


if __name__ == "__main__":
    main()
