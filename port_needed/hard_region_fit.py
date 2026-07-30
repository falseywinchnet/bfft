"""PORT 07: hard-cell affine/ridge readout in each cell's own frame.

An affine field is invariant to translation and scaling of its basis.  Using
``[1, (x-cx)/r, (y-cy)/r]`` makes the intercept exactly orthogonal to both
slopes and keeps the 2x2 slope block at order one.  The affine solve therefore
reduces to a mean and a closed-form 2x2 solve instead of one ill-conditioned
3x3 LU per cell.
"""

from __future__ import annotations

import math
import time

import numpy as np

from bfft.vision import (
    hard_affine_fit_native,
    hard_basis_refit_native,
    measure_paired_offsets,
    measure_residual_ridges,
)
from experiments.dual_aperture_support import score


def _reduce(
    labels: np.ndarray,
    values: np.ndarray,
    cells: int,
) -> np.ndarray:
    return np.bincount(
        labels, weights=values, minlength=cells).astype(np.float64)


def _eliminate_small_systems(
    normal: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """Directly eliminate tiny independent cell systems.

    This is only the portable fallback for an older shared library; the live
    viewer uses the fused C++ reduction and elimination kernel.
    """
    coefficient = np.empty_like(rhs)
    width = normal.shape[1]
    for cell in range(normal.shape[0]):
        matrix = normal[cell].copy()
        values = rhs[cell].copy()
        for column in range(width):
            pivot = column
            pivot_size = abs(matrix[column, column])
            for row in range(column + 1, width):
                candidate = abs(matrix[row, column])
                if candidate > pivot_size:
                    pivot = row
                    pivot_size = candidate
            if pivot != column:
                matrix[[column, pivot]] = matrix[[pivot, column]]
                values[[column, pivot]] = values[[pivot, column]]
            diagonal = matrix[column, column]
            for row in range(column + 1, width):
                multiplier = matrix[row, column] / diagonal
                matrix[row, column + 1:] -= (
                    multiplier * matrix[column, column + 1:])
                values[row] -= multiplier * values[column]
        for row in range(width - 1, -1, -1):
            values[row] = (
                values[row]
                - matrix[row, row + 1:] @ values[row + 1:]
            ) / matrix[row, row]
        coefficient[cell] = values
    return coefficient


def _local_affine_basis(
    labels_2d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(labels_2d, dtype=np.intp).ravel()
    height, width = labels_2d.shape
    cells = int(np.max(labels)) + 1
    yy, xx = np.mgrid[:height, :width]
    x = (xx.ravel() + 0.5) / width - 0.5
    y = (yy.ravel() + 0.5) / height - 0.5
    count = np.maximum(
        np.bincount(labels, minlength=cells), 1).astype(np.float64)
    cx = _reduce(labels, x, cells) / count
    cy = _reduce(labels, y, cells) / count
    dx = x - cx[labels]
    dy = y - cy[labels]
    sxx = _reduce(labels, dx * dx, cells)
    syy = _reduce(labels, dy * dy, cells)
    radius = np.sqrt(np.maximum((sxx + syy) / count, 1e-30))
    ux = dx / radius[labels]
    uy = dy / radius[labels]
    basis = np.column_stack((np.ones_like(ux), ux, uy))
    return labels, basis, count, radius, np.column_stack((cx, cy))


def _fit_local_affine(
    labels_2d: np.ndarray,
    target_lab: np.ndarray,
    objective,
) -> tuple[dict, np.ndarray, dict, tuple]:
    solve_started = time.perf_counter()
    native = hard_affine_fit_native(labels_2d, target_lab)
    if native is not None:
        labels, basis, count, radius, centroid, reconstruction = native
    else:
        labels, basis, count, radius, centroid = _local_affine_basis(
            labels_2d)
        cells = len(count)
        target = np.asarray(target_lab, dtype=np.float64).reshape(-1, 3)
        ux, uy = basis[:, 1], basis[:, 2]

        rhs0 = np.column_stack([
            _reduce(labels, target[:, channel], cells)
            for channel in range(3)
        ])
        rhsx = np.column_stack([
            _reduce(labels, ux * target[:, channel], cells)
            for channel in range(3)
        ])
        rhsy = np.column_stack([
            _reduce(labels, uy * target[:, channel], cells)
            for channel in range(3)
        ])
        # Coefficients of ux=(x-cx)/r represent gradient*r.  A fixed penalty
        # on the physical image-space gradient therefore transforms by 1/r^2.
        gradient_regularization = (
            1e-5 * count / np.maximum(radius * radius, 1e-30))
        a = _reduce(labels, ux * ux, cells) + gradient_regularization
        b = _reduce(labels, ux * uy, cells)
        c = _reduce(labels, uy * uy, cells) + gradient_regularization
        determinant = np.maximum(a * c - b * b, 1e-30)

        coefficient = np.empty((cells, 3, 3), dtype=np.float64)
        coefficient[:, 0, :] = (
            rhs0 / ((1.0 + 1e-7) * count[:, None]))
        coefficient[:, 1, :] = (
            c[:, None] * rhsx - b[:, None] * rhsy
        ) / determinant[:, None]
        coefficient[:, 2, :] = (
            a[:, None] * rhsy - b[:, None] * rhsx
        ) / determinant[:, None]
        reconstruction = np.einsum(
            "ni,nic->nc",
            basis,
            coefficient[labels],
            optimize=False,
        ).reshape(target_lab.shape)
    solve_ms = 1000.0 * (time.perf_counter() - solve_started)
    score_started = time.perf_counter()
    record = score(objective, objective.target_rgb, reconstruction)
    score_ms = 1000.0 * (time.perf_counter() - score_started)
    return (
        record,
        reconstruction,
        {
            "count": count,
            "radius": radius,
            "centroid": centroid,
            "conditioned_frame": "cell_centroid_radius",
            "affine_solve_ms": solve_ms,
            "affine_score_ms": score_ms,
        },
        (labels, basis, count, radius, centroid),
    )


def _fit_local_ridges(
    labels_2d: np.ndarray,
    centers: np.ndarray,
    target_lab: np.ndarray,
    objective,
    *,
    ridge_kappa: float = 16.0,
    ridge_angles: int = 16,
    ridge_bins: int = 161,
    ridge_count: int = 1,
    initial_affine: np.ndarray,
    initial_record: dict,
    basis_data=None,
    geometry: dict | None = None,
) -> tuple[dict, np.ndarray, dict]:
    if basis_data is None:
        basis_data = _local_affine_basis(labels_2d)
    labels, base_basis, count, radius, centroid = basis_data
    height, width = labels_2d.shape
    cells = len(count)
    requested_ridge_bins = max(int(ridge_bins), 3)
    geometry_fixed = geometry is not None and all(
        name in geometry
        for name in ("boundary_xx", "boundary_xy", "boundary_yy")
    )
    if geometry_fixed:
        # The streaming paired-side scan owns only B x RGB accumulation,
        # independent of the number of cells.  It can retain the full finite
        # offset table at populations where C x A x B would be prohibitive.
        ridge_bins = requested_ridge_bins
    else:
        # Compatibility path for callers without BFFT geometry.  Its free
        # angle reference materializes C x A x B x RGB doubles.
        bytes_per_bin = max(cells * max(int(ridge_angles), 1) * 3 * 8, 1)
        maximum_bins = max((256 * 1024 * 1024) // bytes_per_bin, 3)
        ridge_bins = min(requested_ridge_bins, int(maximum_bins))
    if ridge_bins % 2 == 0:
        ridge_bins = max(ridge_bins - 1, 3)
    yy, xx = np.mgrid[:height, :width]
    xf = xx.ravel().astype(np.float64)
    yf = yy.ravel().astype(np.float64)
    center_x = np.asarray(centers[:cells, 0]) * width - 0.5
    center_y = np.asarray(centers[:cells, 1]) * height - 0.5
    dx = xf - center_x[labels]
    dy = yf - center_y[labels]
    spacing = max(math.sqrt(height * width / max(cells, 1)), 1e-9)
    direction = None
    fixed_projection = None
    if geometry_fixed:
        # A germ normally lies in a flat region where the boundary tensor is
        # zero.  Sampling it at the germ would therefore collapse most normals
        # to an arbitrary axis.  Integrating its traceless (doubled-angle)
        # components over the literal cell instead extracts that cell's
        # dominant measured border and remains invariant to normal sign.
        doubled_x = _reduce(
            labels,
            (
                np.asarray(
                    geometry["boundary_xx"], dtype=np.float64)
                - np.asarray(
                    geometry["boundary_yy"], dtype=np.float64)
            ).ravel(),
            cells,
        )
        doubled_y = _reduce(
            labels,
            (
                2.0 * np.asarray(
                    geometry["boundary_xy"], dtype=np.float64)
            ).ravel(),
            cells,
        )
        # BFFT's slope table represents the doubled-angle line field.  The
        # algebraic half-angle below recovers a unit normal without atan2,
        # sin, cos, angle wrapping, or an iterative eigensolve.
        discriminant = np.hypot(doubled_x, doubled_y)
        cosine_double = np.divide(
            doubled_x,
            discriminant,
            out=np.ones_like(discriminant),
            where=discriminant > 1e-30,
        )
        sine_double = np.divide(
            doubled_y,
            discriminant,
            out=np.zeros_like(discriminant),
            where=discriminant > 1e-30,
        )
        normal_x = np.sqrt(np.maximum(
            0.5 * (1.0 + cosine_double), 0.0))
        normal_y = np.copysign(
            np.sqrt(np.maximum(0.5 * (1.0 - cosine_double), 0.0)),
            np.where(np.abs(sine_double) > 1e-30, sine_double, 1.0),
        )
        direction = np.column_stack((normal_x, normal_y))
        fixed_projection = (
            dx * normal_x[labels] + dy * normal_y[labels]
        ) / spacing
    target = np.asarray(target_lab, dtype=np.float64).reshape(-1, 3)
    reconstruction = np.asarray(initial_affine, dtype=np.float64)
    active_basis = base_basis
    ridge_means = []
    ridge_nonzero = []
    ridge_measure_ms = []
    ridge_refit_ms = []
    ridge_score_ms = []
    best_record = initial_record
    best_reconstruction = reconstruction
    best_objective_state = objective.capture_state()
    best_rung = 0
    last_scored_rung = 0

    def refit(design: np.ndarray) -> np.ndarray:
        native = hard_basis_refit_native(
            labels, design, target, count, radius)
        if native is not None:
            return native.reshape(target_lab.shape)
        width_basis = design.shape[1]
        normal = np.empty(
            (cells, width_basis, width_basis), dtype=np.float64)
        rhs = np.empty((cells, width_basis, 3), dtype=np.float64)
        for first in range(width_basis):
            for second in range(first, width_basis):
                value = _reduce(
                    labels,
                    design[:, first] * design[:, second],
                    cells,
                )
                normal[:, first, second] = value
                normal[:, second, first] = value
            for channel in range(3):
                rhs[:, first, channel] = _reduce(
                    labels,
                    design[:, first] * target[:, channel],
                    cells,
                )
        normal[:, 0, 0] += 1e-7 * count
        gradient_regularization = (
            1e-5 * count / np.maximum(radius * radius, 1e-30))
        normal[:, 1, 1] += gradient_regularization
        normal[:, 2, 2] += gradient_regularization
        for component in range(3, width_basis):
            normal[:, component, component] += 2e-5 * count
        coefficient = _eliminate_small_systems(normal, rhs)
        return np.einsum(
            "ni,nic->nc",
            design,
            coefficient[labels],
            optimize=False,
        ).reshape(target_lab.shape)

    for _ in range(max(int(ridge_count), 0)):
        residual = target - reconstruction.reshape(-1, 3)
        measure_started = time.perf_counter()
        if geometry_fixed:
            projection = fixed_projection
            ridge_score, ridge_offset = measure_paired_offsets(
                labels,
                np.ones(labels.size, dtype=np.float64),
                residual,
                projection,
                cells,
                bins=ridge_bins,
                channel_weights=(1.0, 1.5, 1.5),
            )
            # Intercept plus this signed fractional coordinate is exactly a
            # pair of one-sided cell values away from the narrow transition.
            # The linear shoulder retains subpixel coverage; a hard sign loses
            # that information and the former tanh needlessly calls exp.
            ridge = np.clip(
                float(ridge_kappa)
                * (projection - ridge_offset[labels]),
                -1.0,
                1.0,
            )
        else:
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
                float(ridge_kappa)
                * (projection - ridge_offset[labels]))
        ridge_measure_ms.append(
            1000.0 * (time.perf_counter() - measure_started))
        active_basis = np.column_stack((active_basis, ridge))
        refit_started = time.perf_counter()
        reconstruction = refit(active_basis)
        ridge_refit_ms.append(
            1000.0 * (time.perf_counter() - refit_started))
        ridge_means.append(float(np.mean(ridge_score)))
        ridge_nonzero.append(int(np.count_nonzero(ridge_score > 1e-12)))
        score_started = time.perf_counter()
        candidate_record = score(
            objective, objective.target_rgb, reconstruction)
        ridge_score_ms.append(
            1000.0 * (time.perf_counter() - score_started))
        last_scored_rung = len(ridge_means)
        if candidate_record["objective"] < best_record["objective"]:
            best_record = candidate_record
            best_reconstruction = reconstruction
            best_objective_state = objective.capture_state()
            best_rung = len(ridge_means)

    # The objective stores the residual field of its latest evaluation.
    # Restore it to the rung actually returned when a later rung regressed.
    if best_rung != last_scored_rung:
        objective.restore_state(best_objective_state)
    return (
        best_record,
        best_reconstruction,
        {
            "ridge_count": max(int(ridge_count), 0),
            "selected_ridge_count": best_rung,
            "ridge_score_mean": ridge_means,
            "ridge_score_nonzero": ridge_nonzero,
            "ridge_measure_ms": ridge_measure_ms,
            "ridge_refit_ms": ridge_refit_ms,
            "ridge_candidate_score_ms": ridge_score_ms,
            "ridge_angles": int(ridge_angles),
            "ridge_bins": int(ridge_bins),
            "requested_ridge_bins": requested_ridge_bins,
            "ridge_kappa": float(ridge_kappa),
            "ridge_model": (
                "geometry_fixed_paired_sides"
                if geometry_fixed else "free_angle_ridge"),
            "direction_source": (
                "boundary_tensor_half_angle"
                if geometry_fixed else "residual_angle_scan"),
            "cell_direction": direction,
            "conditioned_frame": "cell_centroid_radius",
            "cell_radius_p50": float(np.median(radius)),
            "cell_centroid": centroid,
        },
    )


def fit_regions(
    labels,
    centers,
    target_lab,
    objective,
    *,
    ridge_count=0,
    affine_record=None,
    affine=None,
    geometry=None,
):
    if affine_record is None or affine is None:
        (
            affine_record,
            affine,
            affine_information,
            basis_data,
        ) = _fit_local_affine(labels, target_lab, objective)
    else:
        basis_data = None
        affine_information = {
            "conditioned_frame": "provided",
        }
    if int(ridge_count) <= 0:
        return affine_record, affine, {
            "ridge_count": 0,
            "selected": "affine",
            **affine_information,
        }
    ridge_record, ridge, information = _fit_local_ridges(
        labels,
        centers,
        target_lab,
        objective,
        ridge_count=int(ridge_count),
        initial_affine=affine,
        initial_record=affine_record,
        basis_data=basis_data,
        geometry=geometry,
    )
    information.update(affine_information)
    if affine_record["objective"] <= ridge_record["objective"]:
        information["selected"] = "affine"
        information["rejected_ridge_objective"] = ridge_record["objective"]
        return affine_record, affine, information
    information["selected"] = "ridge"
    information["affine_objective"] = affine_record["objective"]
    return ridge_record, ridge, information
