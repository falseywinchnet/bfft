"""Invertible eikonal observer lens with residual-space smoothing.

The accepted Hopf--Lax forest is used as an analysis/synthesis transform, not
as a set of candidate observations.  In reverse causal order every child is
predicted from its transported parent simplex.  The innovation is retained as
an exact detail coordinate, while the Euclidean least-squares projection of
that innovation is absorbed into the parents.  Reversing the lifting steps
reconstructs the observation exactly.

Only after this exact decomposition is the left-behind detail smoothed, on the
causal forest itself.  The observer/root state is never smoothed.  Forward
lifting inversion then recomposes the scene.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from .causal_information_lineage_2d import causal_information_lineage_law_2d
from .witnessed_characteristic_transport_2d import _validate


def _forest_arrays(forest: dict[str, np.ndarray]) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    first = np.asarray(forest["parent_first"], dtype=np.int64).reshape(-1)
    second = np.asarray(forest["parent_second"], dtype=np.int64).reshape(-1)
    fraction = np.asarray(
        forest["parent_fraction"], dtype=np.float64).reshape(-1)
    order = np.asarray(forest["acceptance_order"], dtype=np.int64).reshape(-1)
    pixels = first.size
    if second.shape != first.shape or fraction.shape != first.shape:
        raise ValueError("forest parent arrays must align")
    if order.size != pixels or not np.array_equal(
        np.sort(order), np.arange(pixels)):
        raise ValueError("forest acceptance order must permute every pixel")
    if np.any(first >= pixels) or np.any(second >= pixels):
        raise ValueError("forest parent index is outside the raster")
    return first, second, fraction, order


def _parent_weights(
    child: int,
    first: np.ndarray,
    second: np.ndarray,
    fraction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    parent_a = int(first[child])
    parent_b = int(second[child])
    if parent_a < 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    if parent_b < 0:
        return np.asarray((parent_a,), dtype=np.int64), np.asarray(
            (1.0,), dtype=np.float64)
    t = float(fraction[child])
    return np.asarray((parent_a, parent_b), dtype=np.int64), np.asarray(
        (1.0 - t, t), dtype=np.float64)


def eikonal_lens_analysis_2d(
    observation: np.ndarray,
    forest: dict[str, np.ndarray],
    *,
    prediction_offset: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """Absorb predictable structure into roots and retain exact innovations."""
    image = _validate(observation)
    first, second, fraction, order = _forest_arrays(forest)
    if first.size != image.size:
        raise ValueError("forest must cover the observation raster")
    state = image.reshape(-1).copy()
    offset = (
        np.zeros(image.size, dtype=np.float64)
        if prediction_offset is None
        else np.asarray(prediction_offset, dtype=np.float64).reshape(-1)
    )
    if offset.size != image.size or not np.all(np.isfinite(offset)):
        raise ValueError("lens prediction offset must be finite and raster-sized")
    root = first < 0
    for child in order[::-1]:
        parents, weight = _parent_weights(
            int(child), first, second, fraction)
        if parents.size == 0:
            continue
        prediction = float(np.dot(weight, state[parents])) + offset[child]
        detail = state[child] - prediction
        # Orthogonal projection of (parents, child) onto child=w^T parents,
        # expressed as an in-place lifting update.  It is canonical in the
        # Euclidean observation measure and is exactly invertible.
        update = weight / (1.0 + float(np.dot(weight, weight)))
        state[parents] += update * detail
        state[child] = detail
    detail = state.copy()
    detail[root] = 0.0
    coarse = state.copy()
    coarse[~root] = 0.0
    return coarse.reshape(image.shape), detail.reshape(image.shape), {
        "root_count": int(np.sum(root)),
        "detail_rms": float(np.sqrt(np.mean(detail[~root] ** 2))),
        "root_rms": float(np.sqrt(np.mean(state[root] ** 2))),
        "analysis_energy": float(np.dot(state, state)),
    }


def eikonal_lens_synthesis_2d(
    coarse: np.ndarray,
    detail: np.ndarray,
    forest: dict[str, np.ndarray],
    *,
    prediction_offset: np.ndarray | None = None,
) -> np.ndarray:
    """Invert the causal lifting operations and render the scene."""
    root_value = np.asarray(coarse, dtype=np.float64)
    residual = np.asarray(detail, dtype=np.float64)
    if root_value.shape != residual.shape or root_value.ndim != 2:
        raise ValueError("coarse and detail images must align")
    first, second, fraction, order = _forest_arrays(forest)
    if first.size != root_value.size:
        raise ValueError("forest must cover the lifting state")
    state = (root_value + residual).reshape(-1).copy()
    offset = (
        np.zeros(first.size, dtype=np.float64)
        if prediction_offset is None
        else np.asarray(prediction_offset, dtype=np.float64).reshape(-1)
    )
    if offset.size != first.size or not np.all(np.isfinite(offset)):
        raise ValueError("lens prediction offset must be finite and raster-sized")
    for child in order:
        parents, weight = _parent_weights(
            int(child), first, second, fraction)
        if parents.size == 0:
            continue
        innovation = float(state[child])
        update = weight / (1.0 + float(np.dot(weight, weight)))
        state[parents] -= update * innovation
        state[child] = (
            innovation + float(np.dot(weight, state[parents])) + offset[child])
    return state.reshape(root_value.shape)


def eikonal_jet_prediction_offset_2d(
    law: dict[str, np.ndarray],
    forest: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, float]]:
    """Parallel-transport the terminal jet across each parent simplex."""
    mass = np.asarray(law["hj_simplex_collision_mass"], dtype=np.float64)
    jet_x = np.asarray(law["jet_x"], dtype=np.float64)
    jet_y = np.asarray(law["jet_y"], dtype=np.float64)
    tangent = np.asarray(law["tangent"], dtype=np.float64)
    if mass.shape != jet_x.shape or mass.shape != jet_y.shape:
        raise ValueError("terminal mass and jet particles must align")
    if tangent.shape != (mass.shape[-1], 2):
        raise ValueError("terminal tangent catalogue must index jet particles")
    derivative = (
        jet_y * tangent[None, None, :, 0]
        + jet_x * tangent[None, None, :, 1])
    ty = tangent[:, 0]
    tx = tangent[:, 1]
    qxx = np.sum(mass * tx * tx, axis=-1)
    qxy = np.sum(mass * tx * ty, axis=-1)
    qyy = np.sum(mass * ty * ty, axis=-1)
    bx = np.sum(mass * derivative * tx, axis=-1)
    by = np.sum(mass * derivative * ty, axis=-1)
    determinant = qxx * qyy - qxy * qxy
    scale = np.maximum(qxx + qyy, np.finfo(float).tiny)
    valid = determinant > np.finfo(float).eps * scale * scale
    gradient_x = np.zeros(mass.shape[:2], dtype=np.float64)
    gradient_y = np.zeros_like(gradient_x)
    gradient_x[valid] = (
        qyy[valid] * bx[valid] - qxy[valid] * by[valid]
    ) / determinant[valid]
    gradient_y[valid] = (
        qxx[valid] * by[valid] - qxy[valid] * bx[valid]
    ) / determinant[valid]

    first, second, fraction, order = _forest_arrays(forest)
    height, width = mass.shape[:2]
    yy, xx = np.mgrid[:height, :width]
    coordinate_x = xx.reshape(-1).astype(np.float64)
    coordinate_y = yy.reshape(-1).astype(np.float64)
    flat_gx = gradient_x.reshape(-1)
    flat_gy = gradient_y.reshape(-1)
    offset = np.zeros(first.size, dtype=np.float64)
    for child in order:
        parents, weight = _parent_weights(
            int(child), first, second, fraction)
        if parents.size == 0:
            continue
        base_x = float(np.dot(weight, coordinate_x[parents]))
        base_y = float(np.dot(weight, coordinate_y[parents]))
        transported_gx = float(np.dot(weight, flat_gx[parents]))
        transported_gy = float(np.dot(weight, flat_gy[parents]))
        offset[child] = (
            transported_gx * (coordinate_x[child] - base_x)
            + transported_gy * (coordinate_y[child] - base_y))
    return offset.reshape(mass.shape[:2]), {
        "valid_jet_fraction": float(np.mean(valid)),
        "mean_gradient_magnitude": float(np.mean(np.hypot(
            gradient_x, gradient_y))),
        "prediction_offset_rms": float(np.sqrt(np.mean(offset * offset))),
    }


def smooth_eikonal_lens_detail_2d(
    detail: np.ndarray,
    forest: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Apply one canonical screened Dirichlet resolvent in observer space."""
    value = np.asarray(detail, dtype=np.float64)
    if value.ndim != 2 or not np.all(np.isfinite(value)):
        raise ValueError("lens detail must be a finite HxW field")
    first, second, fraction, order = _forest_arrays(forest)
    if first.size != value.size:
        raise ValueError("forest must cover lens detail")
    root = first < 0
    nonroot = np.flatnonzero(~root)
    detail_index = np.full(first.size, -1, dtype=np.int64)
    detail_index[nonroot] = np.arange(nonroot.size, dtype=np.int64)
    rows = []
    columns = []
    entries = []
    edge = 0
    for child in nonroot:
        parents, weight = _parent_weights(
            int(child), first, second, fraction)
        child_column = int(detail_index[child])
        for parent, parent_weight in zip(parents, weight):
            scale = np.sqrt(max(float(parent_weight), 0.0))
            rows.append(edge)
            columns.append(child_column)
            entries.append(scale)
            if not root[parent]:
                rows.append(edge)
                columns.append(int(detail_index[parent]))
                entries.append(-scale)
            # A root is the zero-detail observer boundary.
            edge += 1
    incidence = sparse.coo_matrix(
        (entries, (rows, columns)), shape=(edge, nonroot.size)).tocsr()
    identity = sparse.identity(nonroot.size, dtype=np.float64, format="csr")
    augmented = sparse.vstack((identity, incidence), format="csr")
    rhs = np.concatenate((
        value.reshape(-1)[nonroot],
        np.zeros(edge, dtype=np.float64),
    ))
    numerical = np.sqrt(np.finfo(float).eps)
    solved = sparse_linalg.lsmr(
        augmented,
        rhs,
        atol=numerical,
        btol=numerical,
        maxiter=max(4 * nonroot.size, 1),
    )
    smoothed = np.zeros(first.size, dtype=np.float64)
    smoothed[nonroot] = solved[0]
    before = incidence @ value.reshape(-1)[nonroot]
    after = incidence @ solved[0]
    return smoothed.reshape(value.shape), {
        "detail_count": int(nonroot.size),
        "forest_incidence_count": int(edge),
        "least_squares_stop_code": int(solved[1]),
        "least_squares_iterations": int(solved[2]),
        "detail_displacement_rms": float(np.sqrt(np.mean(
            (solved[0] - value.reshape(-1)[nonroot]) ** 2))),
        "forest_action_before": float(np.dot(before, before)),
        "forest_action_after": float(np.dot(after, after)),
    }


def eikonal_observer_phase_model_2d(
    observation: np.ndarray,
    law: dict[str, np.ndarray],
    forest: dict[str, np.ndarray],
    *,
    phase_signal: np.ndarray | None = None,
    fit_signal: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """Absorb affine and graph-unrolled phase into the observer model."""
    image = _validate(observation)
    root_identity = np.asarray(forest["root_identity"], dtype=np.int64)
    if root_identity.shape != image.shape:
        raise ValueError("forest root identity must align with observation")
    height, width = image.shape
    first, second, fraction, order = _forest_arrays(forest)
    flat_root_identity = root_identity.reshape(-1)
    root_nodes = order[first[order] < 0]
    root_count = int(root_nodes.size)
    if root_count < 1:
        raise ValueError("observer forest must retain at least one raster root")
    # The lens cells are the spatial Hopf--Lax root germs themselves.  Do not
    # collapse them through the branch posterior before building phase; that
    # would erase the very observer positions meant to absorb structure.
    root_probability = np.zeros((image.size, root_count), dtype=np.float64)
    germ_identity = np.full(image.size, -1, dtype=np.int64)
    germ_identity[root_nodes] = np.arange(root_count, dtype=np.int64)
    for child in order:
        if first[child] < 0:
            identity = int(germ_identity[child])
            if identity < 0:
                raise ValueError("parentless lens point has no root identity")
            root_probability[child, identity] = 1.0
        elif second[child] < 0:
            root_probability[child] = root_probability[first[child]]
        else:
            t = float(fraction[child])
            root_probability[child] = (
                (1.0 - t) * root_probability[first[child]]
                + t * root_probability[second[child]])
    labels = np.argmax(root_probability, axis=-1).reshape(
        image.shape).astype(np.int32)
    centers = np.empty((root_count, 2), dtype=np.float64)
    for root in range(root_count):
        location = np.asarray((root_nodes[root],), dtype=np.int64)
        if location.size:
            y, x = divmod(int(location[0]), width)
        else:
            owned = np.flatnonzero(labels.reshape(-1) == root)
            if not owned.size:
                y, x = 0, 0
            else:
                y, x = divmod(int(owned[0]), width)
        centers[root] = ((x + 0.5) / width, (y + 0.5) / height)

    from experiments.segmenting_v3 import _graph_unrolled_texture_columns

    phase_source = (
        image
        if phase_signal is None
        else _validate(np.asarray(phase_signal, dtype=np.float64))
    )
    if phase_source.shape != image.shape:
        raise ValueError("phase-estimation signal must align with observation")
    phase_columns, phase_diagnostic = _graph_unrolled_texture_columns(
        labels, centers, phase_source)
    yy, xx = np.mgrid[:height, :width]
    center_x = centers[:, 0] * width - 0.5
    center_y = centers[:, 1] * height - 0.5
    scale = np.sqrt(image.size / max(root_count, 1))
    dx = (xx - center_x[labels]) / max(scale, np.finfo(float).tiny)
    dy = (yy - center_y[labels]) / max(scale, np.finfo(float).tiny)
    basis = np.column_stack((
        np.ones(image.size, dtype=np.float64),
        dx.reshape(-1),
        dy.reshape(-1),
        *(np.asarray(column, dtype=np.float64).reshape(-1)
          for column in phase_columns),
    ))
    flat_labels = labels.reshape(-1)
    fit_source = (
        image
        if fit_signal is None
        else _validate(np.asarray(fit_signal, dtype=np.float64))
    )
    if fit_source.shape != image.shape:
        raise ValueError("observer fit signal must align with observation")
    target = fit_source.reshape(-1)
    model = np.empty_like(target)
    maximum_normal_error = 0.0
    active_cells = 0
    for root in range(root_count):
        selected = flat_labels == root
        if not np.any(selected):
            continue
        active_cells += 1
        design = basis[selected]
        coefficient = np.linalg.lstsq(design, target[selected], rcond=None)[0]
        model[selected] = design @ coefficient
        normal_error = design.T @ (target[selected] - model[selected])
        maximum_normal_error = max(
            maximum_normal_error,
            float(np.max(np.abs(normal_error))),
        )
    residual = target - model
    return model.reshape(image.shape), {
        "status": (
            "transport-root affine jet plus paired graph-unrolled phase "
            "observer model"
        ),
        "root_count": int(root_count),
        "active_root_count": int(active_cells),
        "basis_dimension_per_root": int(basis.shape[1]),
        "maximum_projection_normal_error": maximum_normal_error,
        "model_residual_rms": float(np.sqrt(np.mean(residual * residual))),
        "phase_estimation_source": (
            "observation" if phase_signal is None else "transported_structure"),
        "coefficient_fit_source": (
            "observation" if fit_signal is None else "transported_structure"),
        **phase_diagnostic,
    }


def denoise_phase_eikonal_observer_lens_2d(
    observation: np.ndarray,
    *,
    angular_count: int = 4,
    quantile_count: int = 16,
    complete_residual_moment: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Absorb affine/phase structure, smooth residual detail, invert lens."""
    image = _validate(observation)
    law, transport = causal_information_lineage_law_2d(
        image,
        angular_count=angular_count,
        quantile_count=quantile_count,
        complete_residual_moment=complete_residual_moment,
    )
    forest = transport["forest"]
    transported_structure = law["causal_hj_simplex_collision_barycenter"]
    model, phase = eikonal_observer_phase_model_2d(
        image,
        law,
        forest,
        phase_signal=transported_structure,
        fit_signal=transported_structure,
    )
    residual = image - model
    coarse, detail, analysis = eikonal_lens_analysis_2d(residual, forest)
    exact_residual = eikonal_lens_synthesis_2d(coarse, detail, forest)
    smoothed, smoothing = smooth_eikonal_lens_detail_2d(detail, forest)
    residual_estimate = eikonal_lens_synthesis_2d(coarse, smoothed, forest)
    estimate = model + residual_estimate
    return estimate, {
        "status": (
            "graph-unrolled phase/jet absorbed by virtual lens; only exact "
            "residual detail smoothed before inverse rendering"
        ),
        "phase": phase,
        "analysis": analysis,
        "smoothing": smoothing,
        "exact_analysis_synthesis_maximum_error": float(np.max(np.abs(
            model + exact_residual - image))),
        "observation_displacement_rms": float(np.sqrt(np.mean(
            (estimate - image) ** 2))),
        "transport": transport,
        "readouts": {
            "observer_phase_model": model,
            "observer_phase_residual": residual,
            "observer_phase_detail": detail,
            "smoothed_observer_phase_detail": smoothed,
            "phase_eikonal_observer_lens": estimate,
        },
        "theory_status": (
            "first phase-absorbing lens; same-observation phase estimation "
            "must survive low-SNR self-alignment falsification"
        ),
    }


def denoise_eikonal_observer_lens_2d(
    observation: np.ndarray,
    *,
    angular_count: int = 4,
    quantile_count: int = 16,
    complete_residual_moment: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Infer one predictive eikonal lens, smooth only its detail, and invert."""
    image = _validate(observation)
    law, transport = causal_information_lineage_law_2d(
        image,
        angular_count=angular_count,
        quantile_count=quantile_count,
        complete_residual_moment=complete_residual_moment,
    )
    forest = transport["forest"]
    prediction_offset, jet = eikonal_jet_prediction_offset_2d(law, forest)
    coarse, detail, analysis = eikonal_lens_analysis_2d(
        image, forest, prediction_offset=prediction_offset)
    exact = eikonal_lens_synthesis_2d(
        coarse, detail, forest, prediction_offset=prediction_offset)
    smoothed, smoothing = smooth_eikonal_lens_detail_2d(detail, forest)
    estimate = eikonal_lens_synthesis_2d(
        coarse, smoothed, forest, prediction_offset=prediction_offset)
    return estimate, {
        "status": (
            "predictive eikonal virtual lens; structure absorbed backward, "
            "detail smoothed in observer space, scene rendered forward"
        ),
        "analysis": analysis,
        "jet": jet,
        "smoothing": smoothing,
        "exact_analysis_synthesis_maximum_error": float(np.max(np.abs(
            exact - image))),
        "observation_displacement_rms": float(np.sqrt(np.mean(
            (estimate - image) ** 2))),
        "transport": transport,
        "readouts": {
            "observer_coarse": coarse,
            "observer_detail": detail,
            "smoothed_observer_detail": smoothed,
            "jet_prediction_offset": prediction_offset,
            "eikonal_observer_lens": estimate,
        },
        "theory_status": (
            "first invertible lens decomposition; parent-simplex prediction "
            "does not yet carry a transported jet"
        ),
    }
