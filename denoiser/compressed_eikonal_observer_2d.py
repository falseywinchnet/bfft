"""One-shot compressed observation on transport-derived eikonal support.

This module deliberately does not model an optical lens.  The observer is a
finite measurement space emitted directly by the V3 support measure.  A
continuous first-arrival partition transports those support germs through the
measured precision tensor.  On each resulting cell the sensor measures the
three affine moments ``(1, x, y)`` in one direct solve.

If ``B_S`` is that sparse, support-indexed measurement matrix, one observation
is the orthogonal extraction

    e = B_S (B_S^T B_S)^+ B_S^T y,       r = y - e.

Consequently ``B_S^T r = 0`` to numerical precision: the residual is exactly
what the current observer can no longer explain.  Repeating the construction
on ``r`` changes the support and therefore asks a new compressed question; it
is not an inner descent on the preceding coefficients.

The current file is a theorem probe, not a promoted denoiser.  In particular,
support is measured from the same samples whose explanation is scored.  The
reported dimension-corrected yield is therefore a diagnostic, not yet a
calibrated noise hypothesis test.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage as ndi
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from .witnessed_characteristic_transport_2d import _validate


_COMPLETE_PARITY_COVECTORS = ((1, 0), (0, 1), (1, 1))


def _geometry_signal(field: np.ndarray) -> np.ndarray:
    """Map a signed residual to an affine-equivalent unit-range scene."""
    lower = float(np.min(field))
    span = float(np.max(field) - lower)
    if span <= np.finfo(float).eps * max(float(np.max(np.abs(field))), 1.0):
        return np.full(field.shape, 0.5, dtype=np.float64)
    return (field - lower) / span


def _direct_affine_projection(
    field: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Project onto the exact local affine sensor without global iteration."""
    value = np.asarray(field, dtype=np.float64)
    owner = np.asarray(labels, dtype=np.int32)
    if owner.shape != value.shape or np.any(owner < 0):
        raise ValueError("eikonal owners must cover the complete sensor raster")

    height, width = value.shape
    yy, xx = np.mgrid[:height, :width]
    flat_owner = owner.reshape(-1)
    flat_value = value.reshape(-1)
    flat_x = xx.reshape(-1).astype(np.float64)
    flat_y = yy.reshape(-1).astype(np.float64)
    explanation = np.zeros(value.size, dtype=np.float64)
    rank = 0
    maximum_normal_error = 0.0
    active_cells = 0

    for cell in range(int(np.max(flat_owner)) + 1):
        selected = np.flatnonzero(flat_owner == cell)
        if not selected.size:
            continue
        active_cells += 1
        local_x = flat_x[selected]
        local_y = flat_y[selected]
        # Translation and RMS normalization change conditioning but not the
        # affine measurement space.  Machine precision is the only floor.
        local_x = local_x - float(np.mean(local_x))
        local_y = local_y - float(np.mean(local_y))
        scale_x = max(float(np.sqrt(np.mean(local_x * local_x))), 1.0)
        scale_y = max(float(np.sqrt(np.mean(local_y * local_y))), 1.0)
        design = np.column_stack((
            np.ones(selected.size, dtype=np.float64),
            local_x / scale_x,
            local_y / scale_y,
        ))
        coefficient, _residual, local_rank, _singular = np.linalg.lstsq(
            design, flat_value[selected], rcond=None)
        fitted = design @ coefficient
        explanation[selected] = fitted
        rank += int(local_rank)
        normal_error = design.T @ (flat_value[selected] - fitted)
        maximum_normal_error = max(
            maximum_normal_error,
            float(np.max(np.abs(normal_error))),
        )

    residual = flat_value - explanation
    return explanation.reshape(value.shape), {
        "active_supports": int(active_cells),
        "sensor_rank": int(rank),
        "maximum_normal_error": maximum_normal_error,
        "residual_rms": float(np.sqrt(np.mean(residual * residual))),
    }


def _eikonal_support_from_scene(
    support_scene: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Emit and march the V3 structural support of one sensing scene."""
    from bfft.effects import srgb_to_lab
    from port_needed.continuous_eikonal_transport import (
        continuous_first_partition_prepared,
        prepare_continuous_metric,
    )
    from port_needed.density_population import emit_density_population
    from port_needed.frozen_meyer_geometry import build_frozen_geometry

    field = _validate(support_scene)
    geometry_field = _geometry_signal(field)
    rgb = np.repeat(geometry_field[..., None], 3, axis=2)
    target_lab = srgb_to_lab(rgb)
    geometry = build_frozen_geometry(
        rgb,
        target_lab=target_lab,
        # One is the V3 one-stage support measurement, not a denoising time.
        tgfd_sweeps=1,
        flow_sweeps=1,
        texture_support_weight=0.0,
        glass_support_weight=0.0,
        null_evidence_strength=1.0,
        threads=1,
    )
    centers, population = emit_density_population(
        geometry,
        safety_cells=field.size,
    )
    # Use the measured precision tensor itself.  No extra metric-strength or
    # boundary-jump control is introduced by the observer experiment.
    metric = prepare_continuous_metric(
        geometry["precision_xx"],
        geometry["precision_xy"],
        geometry["precision_yy"],
        consistency_limit=np.finfo(float).max,
    )
    forest = continuous_first_partition_prepared(centers, metric)
    return np.asarray(forest["labels"], dtype=np.int32), {
        "support_count": int(len(centers)),
        "implied_support_count": float(geometry["implied_cells"]),
        "realized_support_count": int(population["realized_cells"]),
        "support_quantization_error": float(population["quantization_error"]),
        "forest": forest,
        "geometry": geometry,
        "centers": centers,
    }


def interlaced_scene_views_2d(
    scene: np.ndarray,
    *,
    parity_covector: tuple[int, int] = (1, 1),
) -> tuple[np.ndarray, np.ndarray]:
    """Return two full fields generated from one disjoint parity covector.

    Each chart retains one lattice parity exactly. Missing points use the
    barycenter of the nearest lattice points in the opposite parity. Thus no
    observed sample enters both charts. The fill is only a
    support/measurement chart; it is never returned as a denoised image.
    """
    field = _validate(scene)
    covector = tuple(int(value) & 1 for value in parity_covector)
    if len(covector) != 2 or covector == (0, 0):
        raise ValueError("parity covector must be nonzero modulo two")
    covector_y, covector_x = covector
    yy, xx = np.mgrid[:field.shape[0], :field.shape[1]]
    candidates = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if (dy or dx) and ((covector_y * dy + covector_x * dx) & 1):
                candidates.append((dy * dy + dx * dx, dy, dx))
    minimum_length = min(row[0] for row in candidates)
    kernel = np.zeros((3, 3), dtype=np.float64)
    for length, dy, dx in candidates:
        if length == minimum_length:
            kernel[dy + 1, dx + 1] = 1.0
    views = []
    for parity in (0, 1):
        retained = ((covector_y * yy + covector_x * xx) & 1) == parity
        numerator = ndi.convolve(
            np.where(retained, field, 0.0),
            kernel,
            mode="constant",
            cval=0.0,
        )
        denominator = ndi.convolve(
            retained.astype(np.float64),
            kernel,
            mode="constant",
            cval=0.0,
        )
        filled = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(field),
            where=denominator > 0.0,
        )
        views.append(np.where(retained, field, filled))
    return views[0], views[1]


def _cross_covariance_gain(
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[float, float, float]:
    """Return the nonnegative shared-energy contraction of two measures."""
    a = np.asarray(first, dtype=np.float64).reshape(-1)
    b = np.asarray(second, dtype=np.float64).reshape(-1)
    cross_action = float(np.dot(a, b))
    mean_measure = 0.5 * (a + b)
    measured_action = float(np.dot(mean_measure, mean_measure))
    gain = (
        max(cross_action, 0.0) / measured_action
        if measured_action > np.finfo(float).tiny
        else 0.0
    )
    # 4<a,b> <= ||a+b||^2 for positive cross action, so this is at most one
    # up to roundoff. The clip enforces the physical shared-variance cone.
    return float(np.clip(gain, 0.0, 1.0)), cross_action, measured_action


def _transport_cross_gain(
    labels: np.ndarray,
    cross_action: np.ndarray,
    measured_action: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Transport local shared-energy statistics on the eikonal cell graph.

    Each cell supplies the screened data term ``M_i g_i = [C_i]_+``. An
    interface transports gain with its measured action density integrated over
    interface length. The resulting matrix ``diag(M)+L`` is a positive
    screened graph Laplacian. Its maximum principle keeps the solution in
    ``[0,1]`` without a gain threshold or smoothing strength.
    """
    owner = np.asarray(labels, dtype=np.int32)
    cross = np.asarray(cross_action, dtype=np.float64)
    measure = np.asarray(measured_action, dtype=np.float64)
    cells = int(np.max(owner)) + 1
    if cross.shape != (cells,) or measure.shape != (cells,):
        raise ValueError("cross statistics must index every eikonal cell")
    count = np.bincount(owner.reshape(-1), minlength=cells).astype(np.float64)
    density = np.divide(
        measure,
        count,
        out=np.zeros_like(measure),
        where=count > 0.0,
    )
    horizontal = np.column_stack((
        owner[:, :-1].reshape(-1), owner[:, 1:].reshape(-1)))
    vertical = np.column_stack((
        owner[:-1, :].reshape(-1), owner[1:, :].reshape(-1)))
    pair = np.vstack((horizontal, vertical))
    pair = pair[pair[:, 0] != pair[:, 1]]
    if pair.size:
        pair.sort(axis=1)
        unique, interface_count = np.unique(
            pair, axis=0, return_counts=True)
        first = unique[:, 0].astype(np.int64)
        second = unique[:, 1].astype(np.int64)
        denominator = density[first] + density[second]
        harmonic_density = np.divide(
            2.0 * density[first] * density[second],
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 0.0,
        )
        weight = interface_count.astype(np.float64) * harmonic_density
        retained = weight > 0.0
        first = first[retained]
        second = second[retained]
        weight = weight[retained]
    else:
        first = second = np.empty(0, dtype=np.int64)
        weight = np.empty(0, dtype=np.float64)

    diagonal = measure.copy()
    if weight.size:
        diagonal += np.bincount(first, weights=weight, minlength=cells)
        diagonal += np.bincount(second, weights=weight, minlength=cells)
        row = np.concatenate((first, second))
        column = np.concatenate((second, first))
        entry = np.concatenate((-weight, -weight))
    else:
        row = column = np.empty(0, dtype=np.int64)
        entry = np.empty(0, dtype=np.float64)
    positive = measure > np.finfo(float).tiny
    # Empty-action cells have neither a measurement nor a meaningful gain.
    # A unit numerical screen makes those isolated rows nonsingular without
    # changing any physical row carrying measured action.
    diagonal[~positive] = 1.0
    matrix = sparse.coo_matrix((
        np.concatenate((entry, diagonal)),
        (
            np.concatenate((row, np.arange(cells))),
            np.concatenate((column, np.arange(cells))),
        ),
    ), shape=(cells, cells)).tocsr()
    rhs = np.maximum(cross, 0.0)
    rhs[~positive] = 0.0
    gain = np.asarray(sparse_linalg.spsolve(matrix, rhs), dtype=np.float64)
    numerical = 32.0 * np.finfo(float).eps
    if np.min(gain) < -numerical or np.max(gain) > 1.0 + numerical:
        raise RuntimeError("positive screened gain violated its maximum principle")
    gain = np.clip(gain, 0.0, 1.0)
    defect = matrix @ gain - rhs
    return gain, {
        "transport_edges": int(weight.size),
        "transported_gain_mean": float(np.mean(gain)),
        "transported_gain_median": float(np.median(gain)),
        "transported_gain_minimum": float(np.min(gain)),
        "transported_gain_maximum": float(np.max(gain)),
        "screened_transport_residual_maximum": float(np.max(np.abs(defect))),
    }


def _transport_signed_phase(
    labels: np.ndarray,
    cross_action: np.ndarray,
    total_action: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Transport symmetric/antisymmetric phase on the eikonal cell graph.

    For reciprocal chart projections ``a`` and ``b``, put

    ``m=(a+b)/2``, ``d=(a-b)/2``, ``C=<a,b>`` and
    ``T=||m||^2+||d||^2``.  Cauchy--Schwarz gives ``|C| <= T``.  The screened
    equation

    ``(diag(T) + L_T) h = C``

    therefore has ``h`` in ``[-1,1]`` by the same M-matrix maximum principle
    as the positive gain solve.  Unlike clipping ``C`` to its positive part,
    this coordinate retains an antisymmetric structural phase.
    """
    owner = np.asarray(labels, dtype=np.int32)
    cross = np.asarray(cross_action, dtype=np.float64)
    action = np.asarray(total_action, dtype=np.float64)
    cells = int(np.max(owner)) + 1
    if cross.shape != (cells,) or action.shape != (cells,):
        raise ValueError("phase statistics must index every eikonal cell")
    if np.any(np.abs(cross) > action + 128.0 * np.finfo(float).eps * np.maximum(
        action, 1.0
    )):
        raise ValueError("signed cross action lies outside its total-action cone")

    count = np.bincount(owner.reshape(-1), minlength=cells).astype(np.float64)
    density = np.divide(
        action,
        count,
        out=np.zeros_like(action),
        where=count > 0.0,
    )
    horizontal = np.column_stack((
        owner[:, :-1].reshape(-1), owner[:, 1:].reshape(-1)))
    vertical = np.column_stack((
        owner[:-1, :].reshape(-1), owner[1:, :].reshape(-1)))
    pair = np.vstack((horizontal, vertical))
    pair = pair[pair[:, 0] != pair[:, 1]]
    if pair.size:
        pair.sort(axis=1)
        unique, interface_count = np.unique(pair, axis=0, return_counts=True)
        first = unique[:, 0].astype(np.int64)
        second = unique[:, 1].astype(np.int64)
        denominator = density[first] + density[second]
        harmonic_density = np.divide(
            2.0 * density[first] * density[second],
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 0.0,
        )
        weight = interface_count.astype(np.float64) * harmonic_density
        retained = weight > 0.0
        first = first[retained]
        second = second[retained]
        weight = weight[retained]
    else:
        first = second = np.empty(0, dtype=np.int64)
        weight = np.empty(0, dtype=np.float64)

    diagonal = action.copy()
    if weight.size:
        diagonal += np.bincount(first, weights=weight, minlength=cells)
        diagonal += np.bincount(second, weights=weight, minlength=cells)
        row = np.concatenate((first, second))
        column = np.concatenate((second, first))
        entry = np.concatenate((-weight, -weight))
    else:
        row = column = np.empty(0, dtype=np.int64)
        entry = np.empty(0, dtype=np.float64)
    positive = action > np.finfo(float).tiny
    diagonal[~positive] = 1.0
    matrix = sparse.coo_matrix((
        np.concatenate((entry, diagonal)),
        (
            np.concatenate((row, np.arange(cells))),
            np.concatenate((column, np.arange(cells))),
        ),
    ), shape=(cells, cells)).tocsr()
    rhs = cross.copy()
    rhs[~positive] = 0.0
    phase = np.asarray(sparse_linalg.spsolve(matrix, rhs), dtype=np.float64)
    numerical = 64.0 * np.finfo(float).eps
    if np.min(phase) < -1.0 - numerical or np.max(phase) > 1.0 + numerical:
        raise RuntimeError("signed screened phase violated its maximum principle")
    phase = np.clip(phase, -1.0, 1.0)
    defect = matrix @ phase - rhs
    return phase, {
        "phase_transport_edges": int(weight.size),
        "transported_phase_mean": float(np.mean(phase)),
        "transported_phase_absolute_mean": float(np.mean(np.abs(phase))),
        "transported_phase_minimum": float(np.min(phase)),
        "transported_phase_maximum": float(np.max(phase)),
        "signed_screened_transport_residual_maximum": float(
            np.max(np.abs(defect))),
    }


def _phase_measured_chart(
    support_source: np.ndarray,
    coefficient_source: np.ndarray,
    common_mean: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return smooth phase-resolved estimates in both chart coordinates."""
    labels, support = _eikonal_support_from_scene(support_source)
    first, first_projection = _direct_affine_projection(
        support_source - common_mean, labels)
    second, second_projection = _direct_affine_projection(
        coefficient_source - common_mean, labels)
    symmetric = 0.5 * (first + second)
    antisymmetric = 0.5 * (first - second)

    cross_action = []
    total_action = []
    for cell in range(int(np.max(labels)) + 1):
        selected = labels == cell
        cross_action.append(float(np.vdot(
            first[selected], second[selected]).real))
        total_action.append(float(
            np.vdot(symmetric[selected], symmetric[selected]).real
            + np.vdot(antisymmetric[selected], antisymmetric[selected]).real
        ))
    cross_array = np.asarray(cross_action, dtype=np.float64)
    total_array = np.asarray(total_action, dtype=np.float64)
    phase, phase_diagnostic = _transport_signed_phase(
        labels, cross_array, total_array)
    pixel_phase = phase[labels]

    # h^2 is the smooth phase-certainty coordinate: it is zero where the
    # transported statistic has no preferred phase and one at either pure
    # phase.  Multiplying by the Bernoulli phase barycentric coordinates gives
    # nonnegative cubic weights with no positive-part kink.
    certainty = pixel_phase * pixel_phase
    symmetric_weight = 0.5 * certainty * (1.0 + pixel_phase)
    antisymmetric_weight = 0.5 * certainty * (1.0 - pixel_phase)
    first_estimate = (
        symmetric_weight * symmetric
        + antisymmetric_weight * antisymmetric
    )
    second_estimate = (
        symmetric_weight * symmetric
        - antisymmetric_weight * antisymmetric
    )
    return first_estimate, second_estimate, {
        **support,
        **phase_diagnostic,
        "mean_phase_certainty": float(np.mean(certainty)),
        "mean_pixel_phase": float(np.mean(pixel_phase)),
        "mean_pixel_phase_square": float(np.mean(pixel_phase * pixel_phase)),
        "mean_symmetric_weight": float(np.mean(symmetric_weight)),
        "mean_antisymmetric_weight": float(np.mean(antisymmetric_weight)),
        "negative_phase_cell_fraction": float(np.mean(phase < 0.0)),
        "pixel_phase": pixel_phase,
        "support_projection_normal_error": float(
            first_projection["maximum_normal_error"]),
        "coefficient_projection_normal_error": float(
            second_projection["maximum_normal_error"]),
    }


def _cross_measured_chart(
    support_source: np.ndarray,
    coefficient_source: np.ndarray,
    common_mean: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Measure coefficients on a chart disjoint from its support source."""
    labels, support = _eikonal_support_from_scene(support_source)
    first, first_projection = _direct_affine_projection(
        support_source - common_mean, labels)
    second, second_projection = _direct_affine_projection(
        coefficient_source - common_mean, labels)
    mean_measure = 0.5 * (first + second)
    (
        global_gain,
        global_cross_action,
        global_measured_action,
    ) = _cross_covariance_gain(first, second)
    global_prior = common_mean + global_gain * mean_measure

    local_prior = np.full(first.shape, common_mean, dtype=np.float64)
    local_gain = []
    local_cross_action = []
    local_measured_action = []
    for cell in range(int(np.max(labels)) + 1):
        selected = labels == cell
        gain, action, measured = _cross_covariance_gain(
            first[selected], second[selected])
        local_prior[selected] += gain * mean_measure[selected]
        local_gain.append(gain)
        local_cross_action.append(action)
        local_measured_action.append(measured)
    gain_array = np.asarray(local_gain, dtype=np.float64)
    action_array = np.asarray(local_cross_action, dtype=np.float64)
    measured_array = np.asarray(local_measured_action, dtype=np.float64)
    transported_gain, gain_transport = _transport_cross_gain(
        labels, action_array, measured_array)
    transported_prior = (
        common_mean + transported_gain[labels] * mean_measure)
    return transported_prior, local_prior, global_prior, {
        **support,
        "global_cross_gain": global_gain,
        "global_cross_action": global_cross_action,
        "global_measured_action": global_measured_action,
        "mean_local_cross_gain": float(np.mean(gain_array)),
        "median_local_cross_gain": float(np.median(gain_array)),
        "positive_local_cross_fraction": float(np.mean(action_array > 0.0)),
        "total_positive_local_cross_action": float(np.sum(np.maximum(
            action_array, 0.0))),
        "total_negative_local_cross_action": float(np.sum(np.minimum(
            action_array, 0.0))),
        "support_projection_normal_error": float(
            first_projection["maximum_normal_error"]),
        "coefficient_projection_normal_error": float(
            second_projection["maximum_normal_error"]),
        **gain_transport,
    }


def cross_measured_eikonal_observation_2d(
    scene: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Extract a prior whose support and coefficient evidence are disjoint.

    Chart A emits support while chart B supplies its independent coefficient
    measure; the roles are then reversed. Cross covariance estimates shared
    structural energy. After that authority has been measured, both
    coefficients are averaged to lower variance. This is a direct cross-fit,
    not a bank of competing reconstructed observations.
    """
    field = _validate(scene)
    if float(np.ptp(field)) <= np.finfo(float).eps:
        prior = field.copy()
        residual = np.zeros_like(field)
        return prior, residual, {
            "status": "constant scene is shared by both interlaced measures",
            "common_mean": float(field.flat[0]),
            "exact_bookkeeping_maximum_error": 0.0,
            "charts": [],
            "readouts": {"global_cross_prior": prior.copy()},
        }

    first_view, second_view = interlaced_scene_views_2d(field)
    common_mean = 0.5 * (
        float(np.mean(first_view)) + float(np.mean(second_view)))
    (
        first_transported,
        first_local,
        first_global,
        first_diagnostic,
    ) = _cross_measured_chart(first_view, second_view, common_mean)
    (
        second_transported,
        second_local,
        second_global,
        second_diagnostic,
    ) = _cross_measured_chart(second_view, first_view, common_mean)
    transported_prior = 0.5 * (first_transported + second_transported)
    local_prior = 0.5 * (first_local + second_local)
    global_prior = 0.5 * (first_global + second_global)
    residual = field - transported_prior
    return transported_prior, residual, {
        "status": (
            "reciprocal interlaced support/coefficient cross-measure with "
            "support-local shared-energy contraction"
        ),
        "common_mean": common_mean,
        "interlaced_view_disagreement_rms": float(np.sqrt(np.mean(
            (first_view - second_view) ** 2))),
        "charts": (first_diagnostic, second_diagnostic),
        "mean_local_cross_gain": float(np.mean([
            first_diagnostic["mean_local_cross_gain"],
            second_diagnostic["mean_local_cross_gain"],
        ])),
        "positive_local_cross_fraction": float(np.mean([
            first_diagnostic["positive_local_cross_fraction"],
            second_diagnostic["positive_local_cross_fraction"],
        ])),
        "exact_bookkeeping_maximum_error": float(np.max(np.abs(
            transported_prior + residual - field))),
        "readouts": {
            "global_cross_prior": global_prior,
            "local_cross_prior": local_prior,
            "transported_cross_prior": transported_prior,
        },
        "theory_status": (
            "first disjoint cross-measure with screened graph transport of "
            "shared variance; checkerboard chart convergence and the "
            "nonsmooth positive shared-variance cone remain open"
        ),
    }


def phase_resolved_eikonal_observation_2d(
    scene: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Extract a conservative prior without identifying phase with sameness.

    Reciprocal interlaced charts are still causally disjoint.  Their projected
    relation is now carried by a signed screened coordinate: positive phase is
    symmetric structure and negative phase is antisymmetric structure.  The
    two modes are recomposed on their original lattice owners, so an
    alternating supported signal is not averaged away merely because its two
    chart values have opposite sign.

    This is intentionally a conservative prior, not a finished denoiser.  The
    exact observation residual is returned for the next posterior experiment.
    """
    field = _validate(scene)
    if float(np.ptp(field)) <= np.finfo(float).eps:
        prior = field.copy()
        residual = np.zeros_like(field)
        return prior, residual, {
            "status": "constant scene is exactly shared in symmetric phase",
            "common_mean": float(field.flat[0]),
            "exact_bookkeeping_maximum_error": 0.0,
            "charts": [],
            "theory_status": "exact constant fixed point",
        }

    first_view, second_view = interlaced_scene_views_2d(field)
    common_mean = 0.5 * (
        float(np.mean(first_view)) + float(np.mean(second_view)))
    first_in_first, second_in_first, first_diagnostic = (
        _phase_measured_chart(first_view, second_view, common_mean))
    second_in_second, first_in_second, second_diagnostic = (
        _phase_measured_chart(second_view, first_view, common_mean))

    first_coordinate = 0.5 * (first_in_first + first_in_second)
    second_coordinate = 0.5 * (second_in_first + second_in_second)
    yy, xx = np.mgrid[:field.shape[0], :field.shape[1]]
    first_owner = ((xx + yy) & 1) == 0
    prior = common_mean + np.where(
        first_owner, first_coordinate, second_coordinate)
    residual = field - prior
    mean_phase = float(np.mean([
        first_diagnostic["mean_pixel_phase"],
        second_diagnostic["mean_pixel_phase"],
    ]))
    mean_phase_square = float(np.mean([
        first_diagnostic["mean_pixel_phase_square"],
        second_diagnostic["mean_pixel_phase_square"],
    ]))
    phase_order = (
        mean_phase * mean_phase / mean_phase_square
        if mean_phase_square > np.finfo(float).tiny
        else 0.0
    )
    first_phase = np.asarray(first_diagnostic["pixel_phase"])
    second_phase = np.asarray(second_diagnostic["pixel_phase"])
    local_phase_mean = 0.5 * (first_phase + second_phase)
    local_phase_square = 0.5 * (
        first_phase * first_phase + second_phase * second_phase)
    local_phase_order = np.divide(
        local_phase_mean * local_phase_mean,
        local_phase_square,
        out=np.zeros_like(local_phase_mean),
        where=local_phase_square > np.finfo(float).tiny,
    )
    local_phase_authority = local_phase_mean * local_phase_mean
    return prior, residual, {
        "status": (
            "reciprocal disjoint support observation with signed screened "
            "phase transport and owner-coordinate recomposition"
        ),
        "common_mean": common_mean,
        "charts": (first_diagnostic, second_diagnostic),
        "mean_absolute_phase": float(np.mean([
            first_diagnostic["transported_phase_absolute_mean"],
            second_diagnostic["transported_phase_absolute_mean"],
        ])),
        "mean_phase_certainty": float(np.mean([
            first_diagnostic["mean_phase_certainty"],
            second_diagnostic["mean_phase_certainty"],
        ])),
        "mean_phase": mean_phase,
        "mean_phase_square": mean_phase_square,
        "phase_order": float(np.clip(phase_order, 0.0, 1.0)),
        "local_phase_order": np.clip(local_phase_order, 0.0, 1.0),
        "local_phase_authority": np.clip(
            local_phase_authority, 0.0, 1.0),
        "negative_phase_cell_fraction": float(np.mean([
            first_diagnostic["negative_phase_cell_fraction"],
            second_diagnostic["negative_phase_cell_fraction"],
        ])),
        "exact_bookkeeping_maximum_error": float(np.max(np.abs(
            prior + residual - field))),
        "theory_status": (
            "smooth phase-certainty prior; outer posterior transport and "
            "continuous chart-phase refinement remain open"
        ),
    }


def phase_ordered_cross_observation_2d(
    scene: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Contract the high-yield cross prior by its transported phase order.

    The positive cross observer is a useful engineering estimate but can still
    aggregate chance-positive cells in a null field.  The signed observer
    supplies an order parameter

    ``kappa = E[h]^2 / E[h^2]``.

    It is one only when transported phase is globally aligned and tends to
    zero when local phases cancel.  This is a Rayleigh quotient, not a fitted
    acceptance threshold.  Applying it to the aggressive cross prior produces
    a conservative, continuously contracted seed for the later residual
    posterior.
    """
    field = _validate(scene)
    if float(np.ptp(field)) <= np.finfo(float).eps:
        prior = field.copy()
        residual = np.zeros_like(field)
        return prior, residual, {
            "status": "constant scene has unit phase order",
            "common_mean": float(field.flat[0]),
            "phase_order": 1.0,
            "exact_bookkeeping_maximum_error": 0.0,
        }

    cross_prior, _cross_residual, cross_diagnostic = (
        cross_measured_eikonal_observation_2d(field))
    _phase_prior, _phase_residual, phase_diagnostic = (
        phase_resolved_eikonal_observation_2d(field))
    common_mean = float(phase_diagnostic["common_mean"])
    phase_order = float(phase_diagnostic["phase_order"])
    prior = common_mean + phase_order * (cross_prior - common_mean)
    residual = field - prior
    return prior, residual, {
        "status": (
            "transported positive cross prior contracted by the signed "
            "phase-order Rayleigh quotient"
        ),
        "common_mean": common_mean,
        "phase_order": phase_order,
        "mean_phase": float(phase_diagnostic["mean_phase"]),
        "mean_phase_square": float(phase_diagnostic["mean_phase_square"]),
        "local_phase_order": np.asarray(
            phase_diagnostic["local_phase_order"], dtype=np.float64),
        "local_phase_authority": np.asarray(
            phase_diagnostic["local_phase_authority"], dtype=np.float64),
        "cross_mean_local_gain": float(
            cross_diagnostic["mean_local_cross_gain"]),
        "exact_bookkeeping_maximum_error": float(np.max(np.abs(
            prior + residual - field))),
        "readouts": {
            "transported_cross_prior": cross_prior,
            "phase_resolved_prior": _phase_prior,
            "phase_ordered_cross_prior": prior,
        },
        "theory_status": (
            "validated conservative scene seed; unsupported residual still "
            "requires an action-contracting posterior transport"
        ),
    }


def phase_union_eikonal_observation_2d(
    scene: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Integrate the complete mod-two covector fibre without phase aliasing.

    Horizontal, vertical, and diagonal parity covectors are the complete
    nonzero covectors of the two-dimensional lattice modulo two.  Each emits a
    reciprocal disjoint cross estimate and its own signed phase law.  Phase
    signs from different covectors are gauge coordinates and are therefore
    never averaged.  Their order parameters are joined by the smooth union

    ``kappa_union = 1 - product_q (1 - kappa_q)``.

    Cross estimates are barycentered by their order mass before the union
    contracts the result.  A coherent section in any covector can survive,
    while no hard maximum, selected direction, or fitted acceptance threshold
    is introduced.
    """
    field = _validate(scene)
    if float(np.ptp(field)) <= np.finfo(float).eps:
        prior = field.copy()
        residual = np.zeros_like(field)
        return prior, residual, {
            "status": "constant scene is exact in every parity covector",
            "common_mean": float(field.flat[0]),
            "covectors": _COMPLETE_PARITY_COVECTORS,
            "covector_phase_order": (1.0, 1.0, 1.0),
            "phase_union_order": 1.0,
            "local_phase_authority": np.ones_like(field),
            "exact_bookkeeping_maximum_error": 0.0,
        }

    view_pairs = [
        interlaced_scene_views_2d(field, parity_covector=covector)
        for covector in _COMPLETE_PARITY_COVECTORS
    ]
    common_mean = float(np.mean([
        float(np.mean(view)) for pair in view_pairs for view in pair
    ]))
    cross_priors = []
    covector_orders = []
    covector_authorities = []
    covector_diagnostics = []
    for covector, (first_view, second_view) in zip(
        _COMPLETE_PARITY_COVECTORS, view_pairs
    ):
        first_cross, _first_local, _first_global, first_cross_diagnostic = (
            _cross_measured_chart(first_view, second_view, common_mean))
        second_cross, _second_local, _second_global, second_cross_diagnostic = (
            _cross_measured_chart(second_view, first_view, common_mean))
        cross_priors.append(0.5 * (first_cross + second_cross))

        _first_in_first, _second_in_first, first_phase_diagnostic = (
            _phase_measured_chart(first_view, second_view, common_mean))
        _second_in_second, _first_in_second, second_phase_diagnostic = (
            _phase_measured_chart(second_view, first_view, common_mean))
        first_phase = np.asarray(first_phase_diagnostic["pixel_phase"])
        second_phase = np.asarray(second_phase_diagnostic["pixel_phase"])
        phase_mean = 0.5 * (
            float(np.mean(first_phase)) + float(np.mean(second_phase)))
        phase_square = 0.5 * (
            float(np.mean(first_phase * first_phase))
            + float(np.mean(second_phase * second_phase)))
        order = (
            phase_mean * phase_mean / phase_square
            if phase_square > np.finfo(float).tiny
            else 0.0
        )
        order = float(np.clip(order, 0.0, 1.0))
        covector_orders.append(order)
        reciprocal_phase = 0.5 * (first_phase + second_phase)
        covector_authorities.append(
            order * reciprocal_phase * reciprocal_phase)
        covector_diagnostics.append({
            "covector": covector,
            "phase_order": order,
            "phase_mean": phase_mean,
            "phase_square": phase_square,
            "first_cross": {
                key: value for key, value in first_cross_diagnostic.items()
                if key not in {"forest", "geometry", "centers"}
            },
            "second_cross": {
                key: value for key, value in second_cross_diagnostic.items()
                if key not in {"forest", "geometry", "centers"}
            },
            "first_phase": {
                key: value for key, value in first_phase_diagnostic.items()
                if key not in {"forest", "geometry", "centers", "pixel_phase"}
            },
            "second_phase": {
                key: value for key, value in second_phase_diagnostic.items()
                if key not in {"forest", "geometry", "centers", "pixel_phase"}
            },
        })

    order_array = np.asarray(covector_orders, dtype=np.float64)
    order_mass = float(np.sum(order_array))
    if order_mass > np.finfo(float).tiny:
        barycentric_mass = order_array / order_mass
    else:
        barycentric_mass = np.full(
            len(_COMPLETE_PARITY_COVECTORS),
            1.0 / len(_COMPLETE_PARITY_COVECTORS),
        )
    cross_barycenter = np.tensordot(
        barycentric_mass, np.stack(cross_priors), axes=1)
    phase_union_order = float(1.0 - np.prod(1.0 - order_array))
    authority_stack = np.stack(covector_authorities)
    local_phase_authority = 1.0 - np.prod(1.0 - authority_stack, axis=0)
    prior = common_mean + phase_union_order * (
        cross_barycenter - common_mean)
    residual = field - prior
    return prior, residual, {
        "status": (
            "complete parity-covector union of reciprocal disjoint phase "
            "laws and their cross-estimate barycenter"
        ),
        "common_mean": common_mean,
        "covectors": _COMPLETE_PARITY_COVECTORS,
        "covector_phase_order": tuple(float(value) for value in order_array),
        "phase_union_order": phase_union_order,
        "covector_barycentric_mass": tuple(
            float(value) for value in barycentric_mass),
        "mean_local_phase_authority": float(np.mean(local_phase_authority)),
        "local_phase_authority": np.clip(
            local_phase_authority, 0.0, 1.0),
        "cross_barycenter": cross_barycenter,
        "covector_diagnostics": tuple(covector_diagnostics),
        "exact_bookkeeping_maximum_error": float(np.max(np.abs(
            prior + residual - field))),
        "theory_status": (
            "finite complete mod-two covector fibre; refinement beyond the "
            "lattice parity quotient remains a convergence obligation"
        ),
    }


def screened_selling_posterior_observation_2d(
    scene: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Remeasure support on the residual, then transport that component.

    Let ``s_0=O(y)`` be the complete-covector phase-union observation and
    ``r_0=y-s_0``.  A second direct observation ``s_1=O(r_0)`` asks what the
    sensor can explain *after* the first scene component has been removed.
    Only this newly supported component enters the screened posterior:

    ``(I + L(s_0,r_0^2)) z = s_1``,       ``x = s_0 + z``.

    Raw residual samples never receive authority from structure elsewhere in
    the image.  The Selling resolvent regularizes a component that has already
    survived a new compressed observation; it does not smooth the entire
    residual and hope that noise disappears.
    """
    from .continual_eikonal_noise_transport_2d import (
        _continual_flux_laplacian,
        continual_transport_metric,
    )

    field = _validate(scene)
    prior, unresolved, prior_diagnostic = phase_union_eikonal_observation_2d(
        field)
    if not np.any(unresolved):
        return prior, unresolved, {
            "status": "exact support observation leaves no posterior action",
            "phase_union_order": float(
                prior_diagnostic["phase_union_order"]),
            "exact_bookkeeping_maximum_error": 0.0,
            "screened_system_residual_maximum": 0.0,
        }

    residual_component, _unobserved_residual, residual_diagnostic = (
        phase_union_eikonal_observation_2d(unresolved))

    # Squared unexplained action is a local second moment, not a named-noise
    # variance fit.  If it contains unresolved structure, the resulting metric
    # becomes more isotropic and therefore errs toward conservative transport.
    metric = continual_transport_metric(prior, unresolved * unresolved)
    laplacian, _markov, stencil_diagnostic = _continual_flux_laplacian(
        metric, np.ones_like(field))
    screen = sparse.eye(field.size, format="csr") + laplacian
    rhs = residual_component.reshape(-1)
    correction = np.asarray(
        sparse_linalg.spsolve(screen, rhs), dtype=np.float64).reshape(field.shape)
    posterior = prior + correction
    posterior_residual = field - posterior
    defect = screen @ correction.reshape(-1) - rhs
    correction_norm = float(np.linalg.norm(correction))
    contractive_bound = float(np.linalg.norm(residual_component))
    numerical = 128.0 * np.finfo(float).eps * max(contractive_bound, 1.0)
    if correction_norm > contractive_bound + numerical:
        raise RuntimeError("screened posterior violated resolvent contraction")
    return posterior, posterior_residual, {
        "status": (
            "complete-covector scene observation, residual support "
            "remeasurement, and screened Selling transport of only the newly "
            "observed component"
        ),
        "phase_union_order": float(prior_diagnostic["phase_union_order"]),
        "residual_phase_union_order": float(
            residual_diagnostic["phase_union_order"]),
        "support_prior": prior,
        "support_prior_diagnostic": prior_diagnostic,
        "residual_component": residual_component,
        "residual_component_diagnostic": residual_diagnostic,
        "posterior_correction": correction,
        "unresolved_before_posterior": unresolved,
        "correction_rms": float(np.sqrt(np.mean(correction * correction))),
        "unresolved_rms": float(np.sqrt(np.mean(unresolved * unresolved))),
        "resolvent_contraction_ratio": (
            correction_norm / contractive_bound
            if contractive_bound > np.finfo(float).tiny
            else 0.0
        ),
        "screened_system_residual_maximum": float(np.max(np.abs(defect))),
        "dirichlet_action": float(
            correction.reshape(-1) @ (laplacian @ correction.reshape(-1))),
        "exact_bookkeeping_maximum_error": float(np.max(np.abs(
            posterior + posterior_residual - field))),
        "metric": {
            key: value for key, value in metric.items()
            if not isinstance(value, np.ndarray)
        },
        "stencil": stencil_diagnostic,
        "theory_status": (
            "first support-remeasured residual posterior; terminal action law "
            "for further outer observations remains open"
        ),
    }


def compressed_eikonal_observation_2d(
    scene: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Extract one support-justified scene component in a direct pass.

    The V3 support machinery is intentionally used only as a sensor emitter:
    no residual split/refill, candidate scoring, characteristic relaxation,
    ridge selection, or coefficient descent is enabled here.
    """
    field = _validate(scene)
    if float(np.ptp(field)) <= np.finfo(float).eps:
        explanation = field.copy()
        residual = np.zeros_like(field)
        return explanation, residual, {
            "status": "constant scene is exactly observable",
            "support_count": 1,
            "sensor_rank": 1,
            "active_supports": 1,
            "maximum_normal_error": 0.0,
            "exact_bookkeeping_maximum_error": 0.0,
            "dimension_corrected_explanation_yield": float("inf"),
            "support_self_measured": True,
        }

    labels, support = _eikonal_support_from_scene(field)
    explanation, projection = _direct_affine_projection(
        field, labels)
    residual = field - explanation

    centered_scene = field - float(np.mean(field))
    centered_explanation = explanation - float(np.mean(field))
    ambient_dimension = max(field.size - 1, 1)
    effective_rank = max(int(projection["sensor_rank"]) - 1, 1)
    ambient_energy = float(np.dot(
        centered_scene.reshape(-1), centered_scene.reshape(-1)))
    explained_energy = float(np.dot(
        centered_explanation.reshape(-1),
        centered_explanation.reshape(-1),
    ))
    if ambient_energy <= np.finfo(float).tiny:
        yield_ratio = float("inf")
    else:
        yield_ratio = (
            explained_energy / effective_rank
        ) / (ambient_energy / ambient_dimension)

    return explanation, residual, {
        "status": (
            "one-shot V3 support emission, continuous eikonal allocation, "
            "and direct affine compressed observation"
        ),
        **support,
        "sensor_rank": int(projection["sensor_rank"]),
        "active_supports": int(projection["active_supports"]),
        "compression_ratio": float(
            projection["sensor_rank"] / field.size),
        "maximum_normal_error": float(projection["maximum_normal_error"]),
        "exact_bookkeeping_maximum_error": float(np.max(np.abs(
            explanation + residual - field))),
        "residual_rms": float(np.sqrt(np.mean(residual * residual))),
        "dimension_corrected_explanation_yield": float(yield_ratio),
        "support_self_measured": True,
        "support_selection_caveat": (
            "yield is descriptive until support discovery and explanation "
            "are separated by a transport-valid cross-measure"
        ),
    }


def pursue_compressed_eikonal_scene_2d(
    observation: np.ndarray,
    *,
    maximum_observations: int = 8,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Repeat direct support observations until a new sensor has no yield.

    ``maximum_observations`` is a numerical safety ceiling for this probe, not
    a denoising strength.  The physical stop is a per-degree explanation yield
    no greater than that of the unresolved ambient field.
    """
    field = _validate(observation)
    explained_scene = np.zeros_like(field)
    unexplained = field.copy()
    trace = []
    for index in range(max(int(maximum_observations), 1)):
        component, candidate_residual, diagnostic = (
            compressed_eikonal_observation_2d(unexplained))
        yield_ratio = float(
            diagnostic["dimension_corrected_explanation_yield"])
        accepted = bool(np.isinf(yield_ratio) or yield_ratio > 1.0)
        trace.append({
            "observation": index + 1,
            "accepted": accepted,
            **{
                key: value
                for key, value in diagnostic.items()
                if key not in {"forest", "geometry", "centers"}
            },
        })
        if not accepted:
            break
        explained_scene += component
        unexplained = candidate_residual
        if not np.any(unexplained):
            break

    return explained_scene, {
        "status": (
            "repeated one-shot compressed observations; no coefficient or "
            "geometry descent inside an observation"
        ),
        "accepted_observations": int(sum(row["accepted"] for row in trace)),
        "attempted_observations": int(len(trace)),
        "trace": trace,
        "explained_scene": explained_scene,
        "unexplained_scene": unexplained,
        "exact_bookkeeping_maximum_error": float(np.max(np.abs(
            explained_scene + unexplained - field))),
        "theory_status": (
            "first direct support sensor; self-measured support bias must be "
            "removed before the yield stop can be considered a posterior"
        ),
    }
