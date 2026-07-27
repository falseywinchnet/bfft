"""PORT 07: hard-cell affine/ridge readout in each cell's own frame.

An affine field is invariant to translation and scaling of its basis.  Using
``[1, (x-cx)/r, (y-cy)/r]`` makes the intercept exactly orthogonal to both
slopes and keeps the 2x2 slope block at order one.  The affine solve therefore
reduces to a mean and a closed-form 2x2 solve instead of one ill-conditioned
3x3 LU per cell.
"""

from __future__ import annotations

import math

import numpy as np

from bfft.vision import measure_residual_ridges
from experiments.dual_aperture_support import score


def _reduce(
    labels: np.ndarray,
    values: np.ndarray,
    cells: int,
) -> np.ndarray:
    return np.bincount(
        labels, weights=values, minlength=cells).astype(np.float64)


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
) -> tuple[dict, np.ndarray, dict]:
    labels, basis, count, radius, centroid = _local_affine_basis(labels_2d)
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
    # Coefficients of ux=(x-cx)/r represent gradient*r.  A fixed penalty on
    # the physical image-space gradient therefore transforms by 1/r^2 in
    # this conditioned basis.  Keeping the same numeric ridge after scaling
    # would silently change the model to penalize variation per cell.
    gradient_regularization = (
        1e-5 * count / np.maximum(radius * radius, 1e-30))
    a = _reduce(labels, ux * ux, cells) + gradient_regularization
    b = _reduce(labels, ux * uy, cells)
    c = _reduce(labels, uy * uy, cells) + gradient_regularization
    determinant = np.maximum(a * c - b * b, 1e-30)

    coefficient = np.empty((cells, 3, 3), dtype=np.float64)
    coefficient[:, 0, :] = rhs0 / ((1.0 + 1e-7) * count[:, None])
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
    return (
        score(objective, objective.target_rgb, reconstruction),
        reconstruction,
        {
            "count": count,
            "radius": radius,
            "centroid": centroid,
            "conditioned_frame": "cell_centroid_radius",
        },
    )


def _fit_local_ridges(
    labels_2d: np.ndarray,
    centers: np.ndarray,
    target_lab: np.ndarray,
    objective,
    *,
    ridge_kappa: float = 4.0,
    ridge_angles: int = 16,
    ridge_bins: int = 41,
    ridge_count: int = 1,
    initial_affine: np.ndarray,
    initial_record: dict,
) -> tuple[dict, np.ndarray, dict]:
    labels, base_basis, count, radius, centroid = _local_affine_basis(
        labels_2d)
    height, width = labels_2d.shape
    cells = len(count)
    yy, xx = np.mgrid[:height, :width]
    xf = xx.ravel().astype(np.float64)
    yf = yy.ravel().astype(np.float64)
    center_x = np.asarray(centers[:cells, 0]) * width - 0.5
    center_y = np.asarray(centers[:cells, 1]) * height - 0.5
    dx = xf - center_x[labels]
    dy = yf - center_y[labels]
    spacing = max(math.sqrt(height * width / max(cells, 1)), 1e-9)
    target = np.asarray(target_lab, dtype=np.float64).reshape(-1, 3)
    reconstruction = np.asarray(initial_affine, dtype=np.float64)
    active_basis = base_basis
    ridge_means = []
    ridge_nonzero = []
    best_record = initial_record
    best_reconstruction = reconstruction
    best_rung = 0
    last_scored_rung = 0

    def refit(design: np.ndarray) -> np.ndarray:
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
        coefficient = np.linalg.solve(normal, rhs)
        return np.einsum(
            "ni,nic->nc",
            design,
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
            float(ridge_kappa)
            * (projection - ridge_offset[labels]))
        active_basis = np.column_stack((active_basis, ridge))
        reconstruction = refit(active_basis)
        ridge_means.append(float(np.mean(ridge_score)))
        ridge_nonzero.append(int(np.count_nonzero(ridge_score > 1e-12)))
        candidate_record = score(
            objective, objective.target_rgb, reconstruction)
        last_scored_rung = len(ridge_means)
        if candidate_record["objective"] < best_record["objective"]:
            best_record = candidate_record
            best_reconstruction = reconstruction
            best_rung = len(ridge_means)

    # The objective stores the residual field of its latest evaluation.
    # Restore it to the rung actually returned when a later rung regressed.
    if best_rung != last_scored_rung:
        objective.evaluate(best_record["rgb"])
    return (
        best_record,
        best_reconstruction,
        {
            "ridge_count": max(int(ridge_count), 0),
            "selected_ridge_count": best_rung,
            "ridge_score_mean": ridge_means,
            "ridge_score_nonzero": ridge_nonzero,
            "ridge_angles": int(ridge_angles),
            "ridge_bins": int(ridge_bins),
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
):
    if affine_record is None or affine is None:
        affine_record, affine, affine_information = _fit_local_affine(
            labels, target_lab, objective)
    else:
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
    )
    if affine_record["objective"] <= ridge_record["objective"]:
        information["selected"] = "affine"
        information["rejected_ridge_objective"] = ridge_record["objective"]
        return affine_record, affine, information
    information["selected"] = "ridge"
    information["affine_objective"] = affine_record["objective"]
    return ridge_record, ridge, information
