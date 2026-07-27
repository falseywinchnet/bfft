"""Intrinsic position force of an irreversible first-arrival cell.

For the semi-discrete path-length objective

    E(s, w) = integral min_i[d_M(s_i, x) - w_i] rho(x) dx
              + sum_i w_i m_i,

the envelope theorem removes interface-motion terms. At fixed optimal
weights, the source-position derivative is

    dE/ds_i = integral_{C_i} d_s d_M(s_i, x) rho(x) dx.

In a constant metric, the terminal arrival covector is

    p(x) = M(x - s_i) / d_M(s_i, x)

and d_s d_M = -p. Therefore each site is in equilibrium precisely when the
accepted transport momentum sums to zero. This is a Riemannian geometric
median condition, not a centroid or PCA condition.

For an image-varying metric, the terminal covector must be parallel
transported back along the achieving characteristic to obtain the exact
source derivative. The reduction below deliberately exposes both the
measured terminal momentum and a local constant-metric Newton surrogate; it
does not pretend the latter is exact across a bent geodesic.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None


def _identity(function):  # pragma: no cover
    return function


_compile = njit(cache=True) if njit is not None else _identity


def terminal_momentum_reduction(
    labels: np.ndarray,
    distance: np.ndarray,
    gradient_x: np.ndarray,
    gradient_y: np.ndarray,
    mxx: np.ndarray,
    mxy: np.ndarray,
    myy: np.ndarray,
    measure: np.ndarray,
    reach: np.ndarray,
) -> dict[str, np.ndarray]:
    """Reduce accepted terminal covectors and local curvature by cell."""
    labels = np.asarray(labels, dtype=np.int32)
    distance = np.asarray(distance, dtype=np.float64)
    gx = np.asarray(gradient_x, dtype=np.float64)
    gy = np.asarray(gradient_y, dtype=np.float64)
    mxx = np.asarray(mxx, dtype=np.float64)
    mxy = np.asarray(mxy, dtype=np.float64)
    myy = np.asarray(myy, dtype=np.float64)
    rho = np.maximum(np.asarray(measure, dtype=np.float64), 0.0)
    reach = np.asarray(reach, dtype=np.float64)
    if not (
        labels.shape
        == distance.shape
        == gx.shape
        == gy.shape
        == mxx.shape
        == mxy.shape
        == myy.shape
        == rho.shape
    ):
        raise ValueError("all first-arrival fields must have identical shape")
    if np.any(labels < 0):
        raise ValueError("first-arrival partition must be complete")
    cells = len(reach)
    flat_label = labels.ravel()

    def reduce(values):
        return np.bincount(
            flat_label,
            weights=np.asarray(values, dtype=np.float64).ravel(),
            minlength=cells,
        ).astype(np.float64)

    mass = reduce(rho)
    momentum_x = reduce(rho * gx)
    momentum_y = reduce(rho * gy)

    # distance = path_length - reach[label].
    travel = distance + reach[labels]
    valid = travel > 1e-6
    inverse_travel = np.zeros_like(travel)
    inverse_travel[valid] = 1.0 / travel[valid]
    curvature_xx = reduce(
        rho * (mxx - gx * gx) * inverse_travel)
    curvature_xy = reduce(
        rho * (mxy - gx * gy) * inverse_travel)
    curvature_yy = reduce(
        rho * (myy - gy * gy) * inverse_travel)

    determinant = (
        curvature_xx * curvature_yy - curvature_xy * curvature_xy)
    trace = curvature_xx + curvature_yy
    ridge = np.maximum(trace, 1e-12) * 1e-8
    regular_xx = curvature_xx + ridge
    regular_yy = curvature_yy + ridge
    regular_determinant = np.maximum(
        regular_xx * regular_yy - curvature_xy * curvature_xy,
        1e-30,
    )
    # The objective gradient is -momentum. Thus H^-1 momentum is the local
    # constant-metric Newton displacement in pixel coordinates.
    step_x = (
        regular_yy * momentum_x - curvature_xy * momentum_y
    ) / regular_determinant
    step_y = (
        regular_xx * momentum_y - curvature_xy * momentum_x
    ) / regular_determinant
    momentum_norm = np.hypot(momentum_x, momentum_y)
    return {
        "mass": mass,
        "momentum_x": momentum_x,
        "momentum_y": momentum_y,
        "momentum_norm": momentum_norm,
        "momentum_per_mass": momentum_norm / np.maximum(mass, 1e-30),
        "curvature_xx": curvature_xx,
        "curvature_xy": curvature_xy,
        "curvature_yy": curvature_yy,
        "curvature_determinant": determinant,
        "newton_step_x_px": step_x,
        "newton_step_y_px": step_y,
        "newton_step_px": np.hypot(step_x, step_y),
    }


def source_force_reduction(
    labels: np.ndarray,
    source_gradient_x: np.ndarray,
    source_gradient_y: np.ndarray,
    measure: np.ndarray,
    cells: int,
) -> dict[str, np.ndarray]:
    """Sum the exact discrete source derivative along achieving fronts.

    ``source_gradient_*`` is propagated through same-label Hopf--Lax
    simplices by the front solver.  It is therefore the derivative of each
    accepted numerical arrival action with respect to its own continuous
    source position while the causal topology is fixed.
    """
    labels = np.asarray(labels, dtype=np.int32)
    sgx = np.asarray(source_gradient_x, dtype=np.float64)
    sgy = np.asarray(source_gradient_y, dtype=np.float64)
    rho = np.maximum(np.asarray(measure, dtype=np.float64), 0.0)
    if not (labels.shape == sgx.shape == sgy.shape == rho.shape):
        raise ValueError("all first-arrival fields must have identical shape")
    flat_label = labels.ravel()
    mass = np.bincount(
        flat_label, weights=rho.ravel(), minlength=cells).astype(np.float64)
    gradient_x = np.bincount(
        flat_label,
        weights=(rho * sgx).ravel(),
        minlength=cells,
    ).astype(np.float64)
    gradient_y = np.bincount(
        flat_label,
        weights=(rho * sgy).ravel(),
        minlength=cells,
    ).astype(np.float64)
    # Negative objective gradient is the intrinsic descent force.
    force_x = -gradient_x
    force_y = -gradient_y
    force_norm = np.hypot(force_x, force_y)
    return {
        "mass": mass,
        "objective_gradient_x": gradient_x,
        "objective_gradient_y": gradient_y,
        "force_x": force_x,
        "force_y": force_y,
        "force_norm": force_norm,
        "force_per_mass": force_norm / np.maximum(mass, 1e-30),
    }


@_compile
def _backtransport_force_kernel(
    labels,
    gradient_x,
    gradient_y,
    parent_first,
    parent_second,
    parent_fraction,
    acceptance_order,
    measure,
    center_x,
    center_y,
    core_radius_px,
    cells,
):
    load = measure.copy()
    force_x = np.zeros(cells)
    force_y = np.zeros(cells)
    captured = np.zeros(cells)
    width = labels.shape[1]
    flat_label = labels.ravel()
    flat_gx = gradient_x.ravel()
    flat_gy = gradient_y.ravel()
    flat_first = parent_first.ravel()
    flat_second = parent_second.ravel()
    flat_fraction = parent_fraction.ravel()
    for order_index in range(len(acceptance_order) - 1, -1, -1):
        pixel = acceptance_order[order_index]
        value = load[pixel]
        if value == 0.0:
            continue
        label = flat_label[pixel]
        py = pixel // width
        px = pixel - py * width
        dx = px - center_x[label]
        dy = py - center_y[label]
        first = flat_first[pixel]
        if (
            first < 0
            or dx * dx + dy * dy <= core_radius_px * core_radius_px
        ):
            force_x[label] += value * flat_gx[pixel]
            force_y[label] += value * flat_gy[pixel]
            captured[label] += value
            continue
        second = flat_second[pixel]
        if second < 0:
            load[first] += value
        else:
            fraction = flat_fraction[pixel]
            load[first] += (1.0 - fraction) * value
            load[second] += fraction * value
    return force_x, force_y, captured


def backtransport_source_force(
    centers: np.ndarray,
    partition: dict[str, np.ndarray],
    measure: np.ndarray,
    *,
    core_radius_px: float = 2.0,
) -> dict[str, np.ndarray]:
    """Reverse accepted resource to a small characteristic shell per site.

    A single reverse pass over the already causal Hopf--Lax DAG transports
    every pixel's resource toward its source. At the first small shell around
    that source, the local arrival covector estimates the initial geodesic
    momentum. As pixel size tends to zero the shell can shrink and this
    converges to the continuum source force. No per-site path tracing,
    directional bins, runner field, search, or sorting is introduced.
    """
    labels = np.asarray(partition["labels"], dtype=np.int32)
    height, width = labels.shape
    centers = np.asarray(centers, dtype=np.float64)
    center_x = centers[:, 0] * width - 0.5
    center_y = centers[:, 1] * height - 0.5
    rho = np.ascontiguousarray(
        np.maximum(np.asarray(measure, dtype=np.float64), 0.0).ravel())
    force_x, force_y, captured = _backtransport_force_kernel(
        labels,
        np.asarray(partition["gradient_x"], dtype=np.float64),
        np.asarray(partition["gradient_y"], dtype=np.float64),
        np.asarray(partition["parent_first"], dtype=np.int32),
        np.asarray(partition["parent_second"], dtype=np.int32),
        np.asarray(partition["parent_fraction"], dtype=np.float64),
        np.asarray(partition["acceptance_order"], dtype=np.int32),
        rho,
        np.asarray(center_x, dtype=np.float64),
        np.asarray(center_y, dtype=np.float64),
        max(float(core_radius_px), 0.5),
        len(centers),
    )
    mass = np.bincount(
        labels.ravel(), weights=rho, minlength=len(centers)).astype(np.float64)
    norm = np.hypot(force_x, force_y)
    return {
        "force_x": force_x,
        "force_y": force_y,
        "force_norm": norm,
        "force_per_mass": norm / np.maximum(mass, 1e-30),
        "mass": mass,
        "captured_mass": captured,
        "captured_fraction": captured / np.maximum(mass, 1e-30),
        "core_radius_px": float(core_radius_px),
    }


def apply_terminal_momentum_step(
    centers: np.ndarray,
    reduction: dict[str, np.ndarray],
    image_shape: tuple[int, int],
    *,
    scale: float = 1.0,
) -> np.ndarray:
    """Apply the exposed constant-metric surrogate without directional bins."""
    height, width = image_shape
    result = np.asarray(centers, dtype=np.float64).copy()
    result[:, 0] += (
        float(scale) * reduction["newton_step_x_px"] / width)
    result[:, 1] += (
        float(scale) * reduction["newton_step_y_px"] / height)
    result[:, 0] = np.clip(
        result[:, 0], 0.5 / width, 1.0 - 0.5 / width)
    result[:, 1] = np.clip(
        result[:, 1], 0.5 / height, 1.0 - 0.5 / height)
    return result


def cell_boundary_inradius(
    labels: np.ndarray,
    centers: np.ndarray,
) -> np.ndarray:
    """Distance from each germ to its nearest literal hard-cell interface."""
    labels = np.asarray(labels, dtype=np.int32)
    centers = np.asarray(centers, dtype=np.float64)
    height, width = labels.shape
    cells = len(centers)
    center_x = centers[:, 0] * width - 0.5
    center_y = centers[:, 1] * height - 0.5
    radius = np.full(cells, np.inf, dtype=np.float64)

    # The interface between cardinal pixel centres lies at their midpoint.
    crossing = labels[:, :-1] != labels[:, 1:]
    if np.any(crossing):
        yy, xx = np.nonzero(crossing)
        bx = xx.astype(np.float64) + 0.5
        by = yy.astype(np.float64)
        first = labels[yy, xx]
        second = labels[yy, xx + 1]
        np.minimum.at(
            radius,
            first,
            np.hypot(bx - center_x[first], by - center_y[first]),
        )
        np.minimum.at(
            radius,
            second,
            np.hypot(bx - center_x[second], by - center_y[second]),
        )
    crossing = labels[:-1, :] != labels[1:, :]
    if np.any(crossing):
        yy, xx = np.nonzero(crossing)
        bx = xx.astype(np.float64)
        by = yy.astype(np.float64) + 0.5
        first = labels[yy, xx]
        second = labels[yy + 1, xx]
        np.minimum.at(
            radius,
            first,
            np.hypot(bx - center_x[first], by - center_y[first]),
        )
        np.minimum.at(
            radius,
            second,
            np.hypot(bx - center_x[second], by - center_y[second]),
        )

    # The image boundary is also a hard topological boundary.
    domain_radius = np.minimum.reduce((
        center_x + 0.5,
        width - 0.5 - center_x,
        center_y + 0.5,
        height - 0.5 - center_y,
    ))
    radius = np.minimum(radius, domain_radius)
    radius[~np.isfinite(radius)] = domain_radius[~np.isfinite(radius)]
    return np.maximum(radius, 0.0)


def safe_characteristic_site_step(
    centers: np.ndarray,
    partition: dict[str, np.ndarray],
    prepared_metric: dict[str, np.ndarray],
    measure: np.ndarray,
    *,
    trust_fraction: float = 0.5,
    core_radius_px: float = 3.0,
    maximum_trials: int = 6,
    armijo_fraction: float = 1e-4,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict]:
    """Take one topology-safe reverse-characteristic position step.

    Safety has two independent parts:

    1. Every raw Newton displacement is projected into half of the largest
       raster ball around its germ contained in its current hard cell.
       Simultaneously moving neighbours therefore cannot carry either germ
       across the old interface.
    2. Exact remarches accept only a complete partition with every germ alive
       and a sufficient decrease of accepted transport action.

    Trial scales are the dyadic trust-region contractions used by globally
    convergent semi-discrete transport Newton methods.  This is not image
    reconstruction search: it tests only the causal transport energy whose
    derivative generated the step.
    """
    from .continuous_eikonal_transport import (
        continuous_first_partition_prepared,
    )

    centers = np.asarray(centers, dtype=np.float64)
    measure = np.asarray(measure, dtype=np.float64)
    labels = np.asarray(partition["labels"], dtype=np.int32)
    height, width = labels.shape
    cells = len(centers)
    reach = np.zeros(cells, dtype=np.float64)

    force = backtransport_source_force(
        centers,
        partition,
        measure,
        core_radius_px=core_radius_px,
    )
    curvature = terminal_momentum_reduction(
        labels,
        partition["distance"],
        partition["gradient_x"],
        partition["gradient_y"],
        prepared_metric["mxx"],
        prepared_metric["mxy"],
        prepared_metric["myy"],
        measure,
        reach,
    )
    hxx = curvature["curvature_xx"]
    hxy = curvature["curvature_xy"]
    hyy = curvature["curvature_yy"]
    trace = hxx + hyy
    disc = np.hypot(hxx - hyy, 2.0 * hxy)
    eigen_low = 0.5 * (trace - disc)
    eigen_high = np.maximum(0.5 * (trace + disc), 0.0)
    eigen_floor = np.maximum(1e-5 * eigen_high, 1e-12)
    diagonal_shift = np.maximum(eigen_floor - eigen_low, 0.0)
    axx = hxx + diagonal_shift
    ayy = hyy + diagonal_shift
    determinant = axx * ayy - hxy * hxy
    raw_x = (
        ayy * force["force_x"] - hxy * force["force_y"]
    ) / determinant
    raw_y = (
        axx * force["force_y"] - hxy * force["force_x"]
    ) / determinant

    inradius = cell_boundary_inradius(labels, centers)
    trust_radius = np.clip(float(trust_fraction), 0.0, 0.5) * inradius
    raw_norm = np.hypot(raw_x, raw_y)
    limiter = np.minimum(
        1.0,
        trust_radius / np.maximum(raw_norm, 1e-30),
    )
    step_x = limiter * raw_x
    step_y = limiter * raw_y
    predicted_linear_decrease = float(np.sum(
        force["force_x"] * step_x + force["force_y"] * step_y))
    before_action = float(np.sum(
        measure * np.asarray(partition["distance"], dtype=np.float64)))

    accepted = False
    accepted_scale = 0.0
    after_action = before_action
    candidate_partition = partition
    candidate_centers = centers.copy()
    trials = 0
    converged = bool(
        np.max(np.hypot(step_x, step_y), initial=0.0) <= 1e-8)
    descent_direction = bool(predicted_linear_decrease > 1e-20)
    trial_count = (
        max(int(maximum_trials), 1)
        if descent_direction and not converged
        else 0
    )
    for trial in range(trial_count):
        trials = trial + 1
        scale = 2.0 ** (-trial)
        proposed = centers.copy()
        proposed[:, 0] += scale * step_x / width
        proposed[:, 1] += scale * step_y / height
        proposed[:, 0] = np.clip(
            proposed[:, 0], 0.5 / width, 1.0 - 0.5 / width)
        proposed[:, 1] = np.clip(
            proposed[:, 1], 0.5 / height, 1.0 - 0.5 / height)
        marched = continuous_first_partition_prepared(
            proposed, prepared_metric)
        alive = np.bincount(
            marched["labels"].ravel(), minlength=cells) > 0
        value = float(np.sum(
            measure
            * np.asarray(marched["distance"], dtype=np.float64)))
        required = (
            before_action
            - float(armijo_fraction)
            * scale
            * max(predicted_linear_decrease, 0.0)
        )
        if np.all(alive) and value <= required + 1e-15:
            accepted = True
            accepted_scale = scale
            after_action = value
            candidate_partition = marched
            candidate_centers = proposed
            break

    diagnostic = {
        "accepted": accepted,
        "converged": converged,
        "descent_direction": descent_direction,
        "accepted_scale": accepted_scale,
        "trials": trials,
        "before_action": before_action,
        "after_action": after_action,
        "relative_action_change": (
            (after_action - before_action) / max(abs(before_action), 1e-30)
        ),
        "predicted_linear_decrease": predicted_linear_decrease,
        "inradius_px": inradius,
        "trust_radius_px": trust_radius,
        "raw_step_px": raw_norm,
        "limited_step_px": np.hypot(step_x, step_y),
        "limited_fraction": limiter,
        "force": force,
        "curvature": curvature,
        "curvature_diagonal_shift": diagonal_shift,
        "regularized_curvature_determinant": determinant,
        "front_updates_before": int(partition.get(
            "front_pushes", 0)),
        "front_updates_after": int(candidate_partition.get(
            "front_pushes", 0)),
        "front_maximum_heap_after": int(candidate_partition.get(
            "front_maximum_heap", 0)),
        "initial_centers": centers.copy(),
        "final_centers": candidate_centers.copy(),
    }
    return candidate_centers, candidate_partition, diagnostic
