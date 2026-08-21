"""Curvilinear exposure charts with an exact reflected-boundary adjoint.

A curved translation path is not a one-parameter translation group, so no
global coordinate warp can turn its exposure integral into the exact box
recurrence used for a line.  The honest object is a lifted exposure tube:
every positive PSF atom has an ordered Eikonal coordinate along a fitted path,
a transverse residual, and a path Jacobian.  The image operator gathers along
those transported states; its adjoint scatters through the identical discrete
reflection map.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .kernels import TransportKernel


@dataclass(frozen=True)
class CurvilinearExposureChart:
    """Ordered positive exposure atoms in fitted path coordinates."""

    displacements_xy: np.ndarray
    weights: np.ndarray
    eikonal_coordinate: np.ndarray
    transverse_coordinate: np.ndarray
    path_jacobian: np.ndarray
    tangents_xy: np.ndarray
    endpoint_displacements_xy: np.ndarray
    quadratic_coefficients: np.ndarray
    path_length: float
    tangent_turn_degrees: float
    curvature_rms: float
    tube_rms: float


@dataclass(frozen=True)
class CurvilinearInverseResult:
    image: np.ndarray
    uncertainty: np.ndarray
    diagnostics: dict[str, object]


def residual_discrepancy(
    observation: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    """Measure structured residual and robust fine-scale noise continuously."""
    from scipy.ndimage import uniform_filter

    first = np.asarray(observation, dtype=np.float64)
    second = np.asarray(prediction, dtype=np.float64)
    if first.shape != second.shape:
        raise ValueError("discrepancy fields must share one shape")
    if first.ndim == 3:
        first = (
            0.2126 * first[..., 0]
            + 0.7152 * first[..., 1]
            + 0.0722 * first[..., 2]
        )
        second = (
            0.2126 * second[..., 0]
            + 0.7152 * second[..., 1]
            + 0.0722 * second[..., 2]
        )
    residual = first - second
    local = uniform_filter(residual, size=3, mode="reflect")
    high = residual - local
    median = float(np.median(high))
    mad = float(np.median(np.abs(high - median)))
    read_sigma = mad / (
        0.6744897501960817 * math.sqrt(8.0 / 9.0) + 1e-15)
    total_rms = float(np.sqrt(np.mean(residual * residual)))
    structured_rms = float(np.sqrt(np.mean(local * local)))
    signal_scale = max(
        1.0,
        float(np.sqrt(np.mean(first * first))),
        float(np.sqrt(np.mean(second * second))),
    )
    numerical_floor = 64.0 * np.finfo(float).eps * signal_scale
    ratio = (
        0.0
        if total_rms <= numerical_floor
        else total_rms / max(read_sigma, np.finfo(float).tiny)
    )
    return {
        "read_sigma": float(read_sigma),
        "total_rms": total_rms,
        "structured_rms": structured_rms,
        "numerical_floor": float(numerical_floor),
        "total_to_read_ratio": float(ratio),
    }


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    order = np.argsort(values)
    sorted_values = np.asarray(values, dtype=np.float64)[order]
    sorted_weights = np.asarray(weights, dtype=np.float64)[order]
    cumulative = np.cumsum(sorted_weights)
    target = float(np.clip(quantile, 0.0, 1.0)) * cumulative[-1]
    return float(sorted_values[min(
        int(np.searchsorted(cumulative, target, side="left")),
        len(sorted_values) - 1,
    )])


def fit_curvilinear_exposure_chart(
    kernel: TransportKernel,
) -> CurvilinearExposureChart:
    """Fit one quadratic Eikonal coordinate to a positive displacement law.

    The PSF atoms themselves remain the forward operator, so fitting never
    changes image formation.  The curve supplies only their ordering,
    Jacobian, endpoints, and transverse model-discrepancy coordinate.
    """
    psf = np.asarray(kernel.psf, dtype=np.float64)
    yy, xx = np.mgrid[: psf.shape[0], : psf.shape[1]]
    center = 0.5 * (np.asarray(psf.shape, dtype=np.float64) - 1.0)
    mask = psf > 0.0
    points = np.column_stack((
        (xx[mask] - center[1]),
        (yy[mask] - center[0]),
    ))
    weights = psf[mask]
    weights = weights / np.sum(weights)
    centroid = np.sum(points * weights[:, None], axis=0)
    centered = points - centroid[None, :]
    covariance = centered.T @ (weights[:, None] * centered)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    major = eigenvectors[:, 1]
    # Fix the sign so endpoint labels and diagnostics are reproducible.
    dominant = int(np.argmax(np.abs(major)))
    if major[dominant] < 0.0:
        major = -major
    normal = np.asarray((-major[1], major[0]), dtype=np.float64)
    s = centered @ major
    n = centered @ normal
    s2_mean = float(np.sum(weights * s * s))
    design = np.column_stack((s * s - s2_mean, s, np.ones_like(s)))
    normal_matrix = design.T @ (weights[:, None] * design)
    rhs = design.T @ (weights * n)
    coefficient = np.linalg.solve(
        normal_matrix + 1e-10 * np.eye(3), rhs)
    fitted_n = design @ coefficient
    transverse = n - fitted_n
    slope = 2.0 * coefficient[0] * s + coefficient[1]
    jacobian = np.sqrt(1.0 + slope * slope)
    tangents = (
        major[None, :] + slope[:, None] * normal[None, :]
    ) / jacobian[:, None]

    supported = weights > max(float(np.max(weights)) * 0.02, 1e-12)
    support_s = s[supported]
    support_w = weights[supported]
    s_start = _weighted_quantile(support_s, support_w, 0.01)
    s_stop = _weighted_quantile(support_s, support_w, 0.99)
    if s_stop <= s_start:
        s_start, s_stop = float(np.min(s)), float(np.max(s))
    dense_s = np.linspace(s_start, s_stop, 2049, dtype=np.float64)
    dense_slope = 2.0 * coefficient[0] * dense_s + coefficient[1]
    dense_jacobian = np.sqrt(1.0 + dense_slope * dense_slope)
    increments = 0.5 * (
        dense_jacobian[:-1] + dense_jacobian[1:]
    ) * np.diff(dense_s)
    dense_arc = np.concatenate(([0.0], np.cumsum(increments)))
    atom_arc = np.interp(s, dense_s, dense_arc)
    arc_middle = np.interp(0.0, dense_s, dense_arc)
    eikonal = atom_arc - arc_middle
    curvature = (
        2.0 * coefficient[0]
        / np.maximum(1.0 + slope * slope, 1e-12) ** 1.5
    )
    endpoint_s = np.asarray((s_start, s_stop), dtype=np.float64)
    endpoint_n = (
        coefficient[0] * (endpoint_s * endpoint_s - s2_mean)
        + coefficient[1] * endpoint_s
        + coefficient[2]
    )
    endpoints = (
        centroid[None, :]
        + endpoint_s[:, None] * major[None, :]
        + endpoint_n[:, None] * normal[None, :]
    )
    endpoint_slopes = 2.0 * coefficient[0] * endpoint_s + coefficient[1]
    endpoint_tangents = (
        major[None, :] + endpoint_slopes[:, None] * normal[None, :]
    )
    endpoint_angles = np.degrees(np.arctan2(
        endpoint_tangents[:, 1], endpoint_tangents[:, 0]))
    tangent_turn = abs(
        ((endpoint_angles[1] - endpoint_angles[0] + 90.0) % 180.0)
        - 90.0
    )
    order = np.lexsort((transverse, eikonal))
    return CurvilinearExposureChart(
        displacements_xy=np.ascontiguousarray(points[order]),
        weights=np.ascontiguousarray(weights[order]),
        eikonal_coordinate=np.ascontiguousarray(eikonal[order]),
        transverse_coordinate=np.ascontiguousarray(transverse[order]),
        path_jacobian=np.ascontiguousarray(jacobian[order]),
        tangents_xy=np.ascontiguousarray(tangents[order]),
        endpoint_displacements_xy=np.ascontiguousarray(endpoints),
        quadratic_coefficients=np.ascontiguousarray(coefficient),
        path_length=float(dense_arc[-1]),
        tangent_turn_degrees=float(tangent_turn),
        curvature_rms=float(np.sqrt(np.sum(weights * curvature * curvature))),
        tube_rms=float(np.sqrt(np.sum(weights * transverse * transverse))),
    )


def _reflect_indices(indices: np.ndarray, size: int) -> np.ndarray:
    """Half-sample symmetric reflection used by scipy ``mode='reflect'``."""
    count = int(size)
    if count <= 1:
        return np.zeros_like(indices, dtype=np.int64)
    period = 2 * count
    folded = np.mod(np.asarray(indices, dtype=np.int64), period)
    return np.where(folded < count, folded, period - 1 - folded)


class ReflectedPathOperator:
    """Exact gather/scatter pair for the chart's discrete positive atoms."""

    backend = "numpy_flat_gather_bincount_scatter"

    def __init__(
        self,
        chart: CurvilinearExposureChart,
        shape: tuple[int, int],
    ) -> None:
        self.chart = chart
        self.shape = (int(shape[0]), int(shape[1]))
        height, width = self.shape
        base_y = np.arange(height, dtype=np.int64)
        base_x = np.arange(width, dtype=np.int64)
        self._source_flat: list[np.ndarray] = []
        for displacement in chart.displacements_xy:
            dx = int(round(float(displacement[0])))
            dy = int(round(float(displacement[1])))
            source_y = _reflect_indices(base_y - dy, height)
            source_x = _reflect_indices(base_x - dx, width)
            self._source_flat.append(np.ascontiguousarray(
                (source_y[:, None] * width + source_x[None, :]).ravel()
            ))
        self._native_plan = None
        try:
            from .native_backend import (
                NativeReflectedPathPlan,
                native_available,
            )
            if native_available():
                self._native_plan = NativeReflectedPathPlan(
                    np.stack(self._source_flat, axis=0),
                    chart.weights,
                    self.shape,
                )
                self.backend = self._native_plan.backend
        except (OSError, RuntimeError, ValueError):
            self._native_plan = None

    def _validate(self, image: np.ndarray) -> np.ndarray:
        value = np.asarray(image, dtype=np.float64)
        if value.shape[:2] != self.shape or value.ndim not in (2, 3):
            raise ValueError("path operator image shape does not match its chart")
        return value

    def _forward_numpy(self, image: np.ndarray) -> np.ndarray:
        value = self._validate(image)
        channels = 1 if value.ndim == 2 else value.shape[2]
        flat = value.reshape((-1, channels))
        output = np.zeros_like(flat)
        for weight, source_flat in zip(
            self.chart.weights, self._source_flat
        ):
            output += weight * flat[source_flat]
        return output.reshape(value.shape)

    def _adjoint_numpy(self, image: np.ndarray) -> np.ndarray:
        value = self._validate(image)
        channels = 1 if value.ndim == 2 else value.shape[2]
        flat = value.reshape((-1, channels))
        output = np.zeros_like(flat)
        for weight, source_flat in zip(
            self.chart.weights, self._source_flat
        ):
            for channel in range(channels):
                output[:, channel] += np.bincount(
                    source_flat,
                    weights=weight * flat[:, channel],
                    minlength=flat.shape[0],
                )
        return output.reshape(value.shape)

    def forward(self, image: np.ndarray) -> np.ndarray:
        value = self._validate(image)
        if self._native_plan is not None:
            return self._native_plan.forward(value)
        return self._forward_numpy(value)

    def adjoint(self, image: np.ndarray) -> np.ndarray:
        value = self._validate(image)
        if self._native_plan is not None:
            return self._native_plan.adjoint(value)
        return self._adjoint_numpy(value)

    def adjoint_normalization(self, channels: int | None = None) -> np.ndarray:
        shape = self.shape if channels is None else (*self.shape, int(channels))
        return self.adjoint(np.ones(shape, dtype=np.float64))


def _endpoint_seed(
    observation: np.ndarray,
    displacement_xy: np.ndarray,
) -> np.ndarray:
    from scipy.ndimage import shift

    value = np.asarray(observation, dtype=np.float64)
    shift_vector: tuple[float, ...] = (
        -float(displacement_xy[1]),
        -float(displacement_xy[0]),
    )
    if value.ndim == 3:
        shift_vector = (*shift_vector, 0.0)
    return shift(
        value,
        shift=shift_vector,
        order=1,
        mode="reflect",
        prefilter=False,
    )


def _positive_branch(
    observation: np.ndarray,
    operator: ReflectedPathOperator,
    initial: np.ndarray,
    *,
    passes: int,
    ratio_limit: float,
    discrepancy_ratio: float = 1.1,
    normalization: np.ndarray | None = None,
    descent_method: str = "optimal_positive_line",
) -> tuple[np.ndarray, dict[str, object]]:
    measured = np.asarray(observation, dtype=np.float64)
    latent = np.clip(np.asarray(initial, dtype=np.float64), 1e-8, 1.0)
    channels = None if measured.ndim == 2 else measured.shape[2]
    if normalization is None:
        normalization = operator.adjoint_normalization(channels)
    normalization = np.maximum(
        np.asarray(normalization, dtype=np.float64), 1e-8)
    if normalization.shape != measured.shape:
        raise ValueError("path normalization must match the observation")
    if descent_method not in ("multiplicative", "optimal_positive_line"):
        raise ValueError("unknown curvilinear descent method")
    residual_trace: list[float] = []
    step_trace: list[float] = []
    discrepancy_trace: list[dict[str, float]] = []
    action = 0.0
    limit = max(float(ratio_limit), 1.0)
    target = max(float(discrepancy_ratio), 1.0)
    prediction = operator.forward(latent)
    initial_discrepancy = residual_discrepancy(measured, prediction)
    stopped_by = "maximum_passes"
    if initial_discrepancy["total_to_read_ratio"] <= target:
        stopped_by = "noise_discrepancy"
    for _ in range(
        0 if stopped_by == "noise_discrepancy" else max(int(passes), 0)
    ):
        ratio = np.clip(
            measured / np.maximum(prediction, 1e-8),
            1.0 / limit,
            limit,
        )
        correction = np.maximum(operator.adjoint(ratio) / normalization, 1e-8)
        proposed = np.clip(latent * correction, 0.0, 1.0)
        direction = proposed - latent
        if descent_method == "optimal_positive_line":
            direction_prediction = operator.forward(direction)
            numerator = float(np.sum(
                (measured - prediction) * direction_prediction))
            denominator = float(np.sum(
                direction_prediction * direction_prediction))
            step = max(numerator / max(denominator, 1e-20), 0.0)
            negative = direction < 0.0
            if np.any(negative):
                step = min(step, 0.999 * float(np.min(
                    -latent[negative] / direction[negative])))
            positive = direction > 0.0
            if np.any(positive):
                step = min(step, max(0.999 * float(np.min(
                    (1.0 - latent[positive]) / direction[positive])), 0.0))
            updated = latent + step * direction
            updated_prediction = prediction + step * direction_prediction
        else:
            step = 1.0
            updated = proposed
            updated_prediction = operator.forward(updated)
        action += float(np.mean((updated - latent) ** 2))
        latent = updated
        prediction = updated_prediction
        step_trace.append(float(step))
        residual = prediction - measured
        residual_trace.append(float(np.sqrt(np.mean(residual * residual))))
        discrepancy = residual_discrepancy(measured, prediction)
        discrepancy_trace.append(discrepancy)
        if discrepancy["total_to_read_ratio"] <= target:
            stopped_by = "noise_discrepancy"
            break
        if descent_method == "optimal_positive_line" and step <= 1e-5:
            stopped_by = "optimal_positive_line_stationarity"
            break
    terminal_discrepancy = (
        discrepancy_trace[-1] if discrepancy_trace else initial_discrepancy)
    return latent, {
        "residual_trace": residual_trace,
        "terminal_forward_rms": (
            residual_trace[-1] if residual_trace else float(np.sqrt(np.mean(
                (prediction - measured) ** 2)))
        ),
        "descent_action": action,
        "descent_method": descent_method,
        "step_trace": step_trace,
        "passes_used": len(residual_trace),
        "stopped_by": stopped_by,
        "discrepancy_ratio_target": target,
        "initial_discrepancy": initial_discrepancy,
        "terminal_discrepancy": terminal_discrepancy,
        "discrepancy_trace": discrepancy_trace,
    }


def _endpoint_inverse_with_plan(
    measured: np.ndarray,
    chart: CurvilinearExposureChart,
    operator: ReflectedPathOperator,
    normalization: np.ndarray,
    *,
    passes: int,
    ratio_limit: float,
) -> CurvilinearInverseResult:
    """Transport endpoint gauges through an already verified operator plan."""
    branches: list[np.ndarray] = []
    records: list[dict[str, object]] = []
    for label, endpoint in zip(
        ("negative_endpoint", "positive_endpoint"),
        chart.endpoint_displacements_xy,
    ):
        initial = _endpoint_seed(measured, endpoint)
        branch, record = _positive_branch(
            measured,
            operator,
            initial,
            passes=passes,
            ratio_limit=ratio_limit,
            normalization=normalization,
        )
        branches.append(branch)
        records.append({
            **record,
            "seed": label,
            "endpoint_displacement_xy": endpoint.tolist(),
        })
    stack = np.stack(branches, axis=0)
    residuals = np.asarray([
        float(record["terminal_forward_rms"]) for record in records
    ])
    actions = np.asarray([
        float(record["descent_action"]) for record in records
    ])
    residual_scale = max(float(np.median(residuals)), 1e-8)
    action_scale = max(float(np.median(actions)), 1e-8)
    log_weight = -residuals / residual_scale - 0.25 * actions / action_scale
    log_weight -= np.max(log_weight)
    weights = np.exp(log_weight)
    weights /= np.sum(weights)
    image = np.sum(stack * weights.reshape((-1,) + (1,) * measured.ndim), axis=0)
    uncertainty = np.sqrt(np.sum(
        weights.reshape((-1,) + (1,) * measured.ndim)
        * (stack - image[None, ...]) ** 2,
        axis=0,
    ))
    prediction = operator.forward(image)
    return CurvilinearInverseResult(
        image=np.clip(image, 0.0, 1.0),
        uncertainty=uncertainty,
        diagnostics={
            "method": "curvilinear_eikonal_endpoint_transport",
            "path_atom_count": int(len(chart.weights)),
            "path_length": chart.path_length,
            "tangent_turn_degrees": chart.tangent_turn_degrees,
            "curvature_rms": chart.curvature_rms,
            "tube_rms": chart.tube_rms,
            "jacobian_min": float(np.min(chart.path_jacobian)),
            "jacobian_max": float(np.max(chart.path_jacobian)),
            "endpoint_branches": records,
            "endpoint_weights": weights.tolist(),
            "endpoint_disagreement_rms": float(np.sqrt(np.mean(
                uncertainty * uncertainty))),
            "forward_rms": float(np.sqrt(np.mean(
                (prediction - measured) ** 2))),
            "adjoint_normalization_min": float(np.min(normalization)),
            "adjoint_normalization_max": float(np.max(normalization)),
            "boundary": "exact_discrete_half_sample_reflection",
        },
    )


def curvilinear_endpoint_inverse(
    observation: np.ndarray,
    kernel: TransportKernel,
    *,
    passes: int = 8,
    ratio_limit: float = 4.0,
) -> CurvilinearInverseResult:
    """Transport the two endpoint seed gauges through one exact path operator."""
    measured = np.asarray(observation, dtype=np.float64)
    chart = fit_curvilinear_exposure_chart(kernel)
    operator = ReflectedPathOperator(chart, measured.shape[:2])
    normalization = operator.adjoint_normalization(
        None if measured.ndim == 2 else measured.shape[2])
    return _endpoint_inverse_with_plan(
        measured,
        chart,
        operator,
        normalization,
        passes=passes,
        ratio_limit=ratio_limit,
    )


def _coverage_gate_correction(
    candidate: np.ndarray,
    initial: np.ndarray,
    kernel: TransportKernel,
    coverage_floor: float,
) -> tuple[np.ndarray, float]:
    """Retain only the refinement correction supported by the path OTF."""
    first = np.asarray(initial, dtype=np.float64)
    second = np.asarray(candidate, dtype=np.float64)
    coverage = np.abs(kernel.otf(first.shape[:2])) ** 2
    floor = max(float(coverage_floor), 0.0)
    authority = (
        np.ones_like(coverage)
        if floor == 0.0 else coverage / (coverage + floor)
    )
    channels = 1 if first.ndim == 2 else first.shape[2]
    planes = []
    energy_before = 0.0
    energy_after = 0.0
    for channel in range(channels):
        before_plane = first if channels == 1 else first[..., channel]
        after_plane = second if channels == 1 else second[..., channel]
        correction_fft = np.fft.fft2(after_plane - before_plane)
        gated_fft = authority * correction_fft
        energy_before += float(np.sum(np.abs(correction_fft) ** 2))
        energy_after += float(np.sum(np.abs(gated_fft) ** 2))
        planes.append(before_plane + np.fft.ifft2(gated_fft).real)
    image = planes[0] if channels == 1 else np.stack(planes, axis=2)
    retained = energy_after / max(energy_before, np.finfo(float).tiny)
    return np.clip(image, 0.0, 1.0), float(retained)


def refine_curvilinear_exposure(
    observation: np.ndarray,
    kernel: TransportKernel,
    initial: np.ndarray,
    *,
    passes: int = 32,
    endpoint_passes: int = 4,
    ratio_limit: float = 2.0,
    coverage_floor: float = 5e-4,
    endpoint_gauge_floor: float = 0.01,
    endpoint_basin_uncertainty_weight: float = 0.0,
    local_constancy_floor: float = 0.004,
) -> CurvilinearInverseResult:
    """Refine a positive basin through the exact lifted path operator.

    Endpoint-conditioned branches are transported only to measure the seed
    gauge.  Reconstruction starts from the already stable positive basin and
    spends a small additional action through the exact reflect gather/scatter
    pair.  Its correction is projected by the original Fourier support law.
    """
    measured = np.asarray(observation, dtype=np.float64)
    starting = np.asarray(initial, dtype=np.float64)
    if starting.shape != measured.shape:
        raise ValueError("curvilinear initial state must match the observation")
    chart = fit_curvilinear_exposure_chart(kernel)
    operator = ReflectedPathOperator(chart, measured.shape[:2])
    normalization = operator.adjoint_normalization(
        None if measured.ndim == 2 else measured.shape[2])
    raw, refinement_record = _positive_branch(
        measured,
        operator,
        starting,
        passes=passes,
        ratio_limit=ratio_limit,
        normalization=normalization,
    )
    supported_image, retained = _coverage_gate_correction(
        raw, starting, kernel, coverage_floor)
    endpoint = _endpoint_inverse_with_plan(
        measured,
        chart,
        operator,
        normalization,
        passes=endpoint_passes,
        ratio_limit=max(ratio_limit, 2.0),
    )
    supported_correction = supported_image - starting
    gauge_floor = max(float(endpoint_gauge_floor), 1e-8)
    endpoint_seed_basin_before_gating = np.abs(
        endpoint.image - supported_image)
    basin_weight = max(float(endpoint_basin_uncertainty_weight), 0.0)
    gauge_uncertainty_squared = (
        endpoint.uncertainty * endpoint.uncertainty
        + basin_weight
        * endpoint_seed_basin_before_gating
        * endpoint_seed_basin_before_gating
    )
    gauge_authority = (
        supported_correction * supported_correction + gauge_floor * gauge_floor
    ) / (
        supported_correction * supported_correction
        + gauge_uncertainty_squared
        + gauge_floor * gauge_floor
    )
    gauge_correction = gauge_authority * supported_correction
    constancy_floor = max(float(local_constancy_floor), 0.0)
    if constancy_floor > 0.0:
        moment_input = np.concatenate(
            (measured[..., None], (measured * measured)[..., None]), axis=2
        ) if measured.ndim == 2 else np.concatenate(
            (measured, measured * measured), axis=2)
        transported_moments = operator.forward(moment_input)
        if measured.ndim == 2:
            local_mean = transported_moments[..., 0]
            local_second = transported_moments[..., 1]
        else:
            channels = measured.shape[2]
            local_mean = transported_moments[..., :channels]
            local_second = transported_moments[..., channels:]
        local_variance = np.maximum(
            local_second - local_mean * local_mean, 0.0)
        variance_floor = constancy_floor * constancy_floor
        constancy_authority = local_variance / (
            local_variance + variance_floor)
    else:
        constancy_authority = np.ones_like(gauge_correction)
    image = np.clip(
        starting + constancy_authority * gauge_correction, 0.0, 1.0)
    prediction = operator.forward(image)
    correction = image - starting
    endpoint_seed_basin = np.abs(endpoint.image - image)
    uncertainty = np.sqrt(
        endpoint.uncertainty * endpoint.uncertainty
        + endpoint_seed_basin * endpoint_seed_basin
    )
    return CurvilinearInverseResult(
        image=image,
        uncertainty=uncertainty,
        diagnostics={
            **endpoint.diagnostics,
            "method": "exact_curvilinear_eikonal_refinement",
            "refinement_passes": max(int(passes), 0),
            "refinement_passes_used": refinement_record["passes_used"],
            "refinement_stopped_by": refinement_record["stopped_by"],
            "refinement_discrepancy_ratio_target": refinement_record[
                "discrepancy_ratio_target"],
            "refinement_initial_discrepancy": refinement_record[
                "initial_discrepancy"],
            "refinement_terminal_discrepancy": refinement_record[
                "terminal_discrepancy"],
            "endpoint_uncertainty_passes": max(int(endpoint_passes), 0),
            "refinement_residual_trace": refinement_record["residual_trace"],
            "refinement_action": refinement_record["descent_action"],
            "correction_rms": float(np.sqrt(np.mean(correction * correction))),
            "endpoint_seed_basin_rms": float(np.sqrt(np.mean(
                endpoint_seed_basin * endpoint_seed_basin))),
            "coverage_floor": max(float(coverage_floor), 0.0),
            "correction_spectral_energy_retained": retained,
            "endpoint_gauge_floor": gauge_floor,
            "endpoint_basin_uncertainty_weight": basin_weight,
            "endpoint_gauge_authority_mean": float(np.mean(gauge_authority)),
            "endpoint_gauge_authority_range": [
                float(np.min(gauge_authority)),
                float(np.max(gauge_authority)),
            ],
            "local_constancy_floor": constancy_floor,
            "local_constancy_authority_mean": float(np.mean(
                constancy_authority)),
            "local_constancy_authority_range": [
                float(np.min(constancy_authority)),
                float(np.max(constancy_authority)),
            ],
            "moment_transport_evaluations": int(constancy_floor > 0.0),
            "separate_moment_transports_avoided": int(
                constancy_floor > 0.0),
            "forward_rms": float(np.sqrt(np.mean(
                (prediction - measured) ** 2))),
            "operator_role": (
                "exact_ordered_path_gather_with_matched_reflect_scatter"
            ),
            "operator_backend": operator.backend,
            "operator_plan_reused_across_branches": True,
            "uncertainty_role": (
                "transported_endpoint_seed_gauge_not_calibrated_interval"
            ),
        },
    )
