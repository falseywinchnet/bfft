"""Shift-first factorization of deterministic transport and centered mixing.

The image-formation object is not a blur-family label.  A positive exposure
measure ``mu`` has a centroid ``m`` and a centered residual ``nu``:

    mu = translate(m) # nu
    H_mu(f) = exp(-2 pi i f dot m) H_nu(f).

The phase ramp is deterministic transport.  The centered characteristic
function is mixing around transported centers.  This module removes the first
before applying a positive multiplicative inverse to the second.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np

from .curvilinear import (
    fit_curvilinear_exposure_chart,
    refine_curvilinear_exposure,
    residual_discrepancy,
)
from .kernels import (
    CircularTransportPlan,
    TransportKernel,
    gaussian_kernel,
    identity_kernel,
    line_kernel,
    path_kernel,
)


@dataclass(frozen=True)
class RelativeShiftEstimate:
    """Relative shift needed to register ``moving`` onto ``reference``."""

    shift_xy: tuple[float, float]
    peak_ratio: float
    observable: bool
    reason: str


@dataclass(frozen=True)
class TransportMixFactorization:
    deterministic_shift_xy: tuple[float, float]
    centered_mixing: TransportKernel
    mixing_covariance: np.ndarray
    shift_detected: bool
    shift_norm: float
    phase_factorization_error: float


@dataclass(frozen=True)
class TwoStageDeblurResult:
    image: np.ndarray
    aligned_observation: np.ndarray
    factorization: TransportMixFactorization
    diagnostics: dict[str, object]
    uncertainty: np.ndarray | None = None


@dataclass(frozen=True)
class CenteredMixEstimate:
    kernel: TransportKernel
    covariance: np.ndarray
    anisotropy_ratio: float
    estimator_branch: str
    confidence: float


def image_fingerprint(image: np.ndarray) -> str:
    """Stable evidence fingerprint used to prove an input was not overwritten."""
    value = np.ascontiguousarray(np.asarray(image, dtype=np.float64))
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def _luminance(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image, dtype=np.float64)
    if value.ndim == 2:
        return value
    if value.ndim == 3 and value.shape[2] >= 3:
        return (
            0.2126 * value[..., 0]
            + 0.7152 * value[..., 1]
            + 0.0722 * value[..., 2]
        )
    raise ValueError("an image must be HxW or RGB")


def estimate_relative_shift(
    reference: np.ndarray,
    moving: np.ndarray,
    *,
    minimum_peak_ratio: float = 1.2,
) -> RelativeShiftEstimate:
    """Estimate observable relative translation by phase correlation.

    Absolute translation of one unknown latent image is a gauge and is not
    estimated here.  This function requires a second raster carrying a shared
    scene coordinate system.
    """
    first = _luminance(reference)
    second = _luminance(moving)
    if first.shape != second.shape:
        raise ValueError("relative-shift observations must share one shape")
    window = np.hanning(first.shape[0])[:, None] * np.hanning(first.shape[1])[None, :]
    first_fft = np.fft.fft2((first - np.mean(first)) * window)
    second_fft = np.fft.fft2((second - np.mean(second)) * window)
    cross = first_fft * np.conj(second_fft)
    cross /= np.maximum(np.abs(cross), np.finfo(float).tiny)
    correlation = np.abs(np.fft.ifft2(cross))
    peak = np.unravel_index(int(np.argmax(correlation)), correlation.shape)
    suppressed = correlation.copy()
    py, px = peak
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            suppressed[(py + dy) % first.shape[0], (px + dx) % first.shape[1]] = 0.0
    runner = float(np.max(suppressed))
    ratio = float(correlation[peak] / max(runner, np.finfo(float).tiny))
    shift_y = float(py if py <= first.shape[0] // 2 else py - first.shape[0])
    shift_x = float(px if px <= first.shape[1] // 2 else px - first.shape[1])
    observable = bool(ratio >= float(minimum_peak_ratio))
    return RelativeShiftEstimate(
        shift_xy=(shift_x, shift_y),
        peak_ratio=ratio,
        observable=observable,
        reason=(
            "relative_phase_ramp_supported"
            if observable else "relative_phase_peak_ambiguous"
        ),
    )


def _centered_kernel(kernel: TransportKernel) -> TransportKernel:
    yy, xx = np.mgrid[: kernel.psf.shape[0], : kernel.psf.shape[1]]
    center = 0.5 * (np.asarray(kernel.psf.shape, dtype=np.float64) - 1.0)
    points = np.column_stack((
        (xx - center[1]).ravel(),
        (yy - center[0]).ravel(),
    ))
    points -= kernel.centroid[None, :]
    return path_kernel(
        points,
        weights=kernel.psf.ravel(),
        name=f"centered_{kernel.name}",
        recenter=False,
    )


def factor_transport_mix(
    kernel: TransportKernel,
    *,
    shift_threshold: float = 0.25,
    audit_shape: tuple[int, int] = (128, 128),
) -> TransportMixFactorization:
    """Factor a known positive displacement law into shift then mixing."""
    shift = np.asarray(kernel.centroid, dtype=np.float64)
    centered = _centered_kernel(kernel)
    height = max(int(audit_shape[0]), kernel.psf.shape[0], centered.psf.shape[0])
    width = max(int(audit_shape[1]), kernel.psf.shape[1], centered.psf.shape[1])
    shape = (height, width)
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.fftfreq(width)[None, :]
    phase = np.exp(-2j * np.pi * (fx * shift[0] + fy * shift[1]))
    expected = phase * centered.otf(shape)
    actual = kernel.otf(shape)
    error = float(np.sqrt(np.mean(np.abs(expected - actual) ** 2)))
    norm = float(np.linalg.norm(shift))
    return TransportMixFactorization(
        deterministic_shift_xy=(float(shift[0]), float(shift[1])),
        centered_mixing=centered,
        mixing_covariance=np.asarray(centered.covariance, dtype=np.float64),
        shift_detected=bool(norm >= float(shift_threshold)),
        shift_norm=norm,
        phase_factorization_error=error,
    )


def shift_image_reflect(image: np.ndarray, shift_xy: tuple[float, float]) -> np.ndarray:
    """Translate without exposing the reconstruction to periodic wraparound."""
    from scipy.ndimage import shift as nd_shift

    value = np.asarray(image, dtype=np.float64)
    shift = (float(shift_xy[1]), float(shift_xy[0]))
    if value.ndim == 3:
        shift = (*shift, 0.0)
    return nd_shift(value, shift=shift, order=1, mode="reflect", prefilter=False)


def apply_reflect(image: np.ndarray, kernel: TransportKernel) -> np.ndarray:
    """Apply a positive PSF with camera-like reflected boundary extension."""
    from scipy.ndimage import convolve

    value = np.asarray(image, dtype=np.float64)
    if value.ndim == 2:
        return convolve(value, kernel.psf, mode="reflect")
    return np.stack([
        convolve(value[..., channel], kernel.psf, mode="reflect")
        for channel in range(value.shape[2])
    ], axis=2)


def _positive_center_inverse(
    observation: np.ndarray,
    kernel: TransportKernel,
    *,
    passes: int,
    ratio_limit: float,
    coverage_floor: float,
    local_constancy_floor: float = 0.004,
    discrepancy_ratio: float = 1.1,
    descent_method: str = "optimal_positive_line",
) -> tuple[np.ndarray, list[float], dict[str, object]]:
    """Invert centered mixing by positive adjoint mass transport.

    Reflection padding isolates the image from the circular FFT boundary.  A
    multiplicative forward/adjoint correction preserves positivity and avoids
    the signed inverse-filter oscillations responsible for severe ringing.
    """
    value = np.asarray(observation, dtype=np.float64)
    if kernel.psf.size == 1 or max(int(passes), 0) == 0:
        return value.copy(), [0.0], {
            "coverage_floor": 0.0,
            "dead_fraction": 0.0,
            "unsupported_energy_removed_fraction": 0.0,
        }
    padding = max(max(kernel.psf.shape) * 2, 8)
    pad_width = ((padding, padding), (padding, padding))
    if value.ndim == 3:
        pad_width += ((0, 0),)
    measured = np.pad(value, pad_width, mode="reflect")
    plan = CircularTransportPlan(kernel, measured.shape[:2])
    latent = np.maximum(measured, 1e-8)
    crop = (slice(padding, -padding), slice(padding, -padding))
    if value.ndim == 3:
        crop += (slice(None),)
    if descent_method not in ("multiplicative", "optimal_positive_line"):
        raise ValueError("unknown positive inverse descent method")
    trace: list[float] = []
    step_trace: list[float] = []
    discrepancy_trace: list[dict[str, float]] = []
    limit = max(float(ratio_limit), 1.0)
    target = max(float(discrepancy_ratio), 1.0)
    prediction = plan.forward(latent)
    initial_discrepancy = residual_discrepancy(
        measured[crop], prediction[crop])
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
        correction = np.maximum(plan.adjoint(ratio), 1e-8)
        if descent_method == "optimal_positive_line":
            proposed = np.clip(latent * correction, 0.0, 1.0)
            direction = proposed - latent
            direction_prediction = plan.forward(direction)
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
            latent = latent + step * direction
            prediction = prediction + step * direction_prediction
        else:
            step = 1.0
            latent = np.clip(latent * correction, 0.0, 1.0)
            prediction = plan.forward(latent)
        step_trace.append(float(step))
        residual = prediction - measured
        trace.append(float(np.sqrt(np.mean(residual * residual))))
        discrepancy = residual_discrepancy(
            measured[crop], prediction[crop])
        discrepancy_trace.append(discrepancy)
        if discrepancy["total_to_read_ratio"] <= target:
            stopped_by = "noise_discrepancy"
            break
        if descent_method == "optimal_positive_line" and step <= 1e-5:
            stopped_by = "optimal_positive_line_stationarity"
            break
    terminal_discrepancy = (
        discrepancy_trace[-1] if discrepancy_trace else initial_discrepancy)

    # The positive update can still create oscillatory content in exact or
    # near nulls of a line/path OTF.  Those coefficients were not measured.
    # Project them back toward the observation instead of allowing positivity
    # to masquerade as evidence.  This is an absolute support gate, distinct
    # from the relative correction used above.
    floor = max(float(coverage_floor), 0.0)
    coverage = np.abs(plan.transfer) ** 2
    if floor > 0.0:
        authority = coverage / (coverage + floor)
        authority_field = (
            authority if measured.ndim == 2 else authority[..., None])
        unsupported_weight = 1.0 - authority
        unsupported_field = (
            unsupported_weight
            if measured.ndim == 2 else unsupported_weight[..., None])
        candidate_fft = np.fft.fft2(latent, axes=(0, 1))
        measured_fft = np.fft.fft2(measured, axes=(0, 1))
        output_fft = (
            authority_field * candidate_fft
            + unsupported_field * measured_fft
        )
        delta_fft = candidate_fft - measured_fft
        output_delta_fft = output_fft - measured_fft
        energy_before = float(np.sum(
            unsupported_field * np.abs(delta_fft) ** 2))
        energy_after = float(np.sum(
            unsupported_field * np.abs(output_delta_fft) ** 2))
        latent = np.fft.ifft2(output_fft, axes=(0, 1)).real
        latent = np.clip(latent, 0.0, 1.0)
        removed = float(1.0 - energy_after / max(
            energy_before, np.finfo(float).tiny))
    else:
        removed = 0.0
    # Every locally constant field is an exact fixed point of a positive
    # exposure operator.  A correction in such a region therefore has no
    # observational support: it is null-space ringing, even when positivity
    # keeps it inside the display range.  Measure variation over the actual
    # transport reach and continuously withhold corrections that exceed it.
    constancy_floor = max(float(local_constancy_floor), 0.0)
    if constancy_floor > 0.0:
        moment_input = np.concatenate(
            (measured[..., None], (measured * measured)[..., None]), axis=2
        ) if measured.ndim == 2 else np.concatenate(
            (measured, measured * measured), axis=2)
        transported_moments = plan.forward(moment_input)
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
        correction = latent - measured
        latent = np.clip(
            measured + constancy_authority * correction, 0.0, 1.0)
    else:
        constancy_authority = np.ones_like(latent)
    support_record = {
        "coverage_floor": floor,
        "dead_fraction": float(np.mean(coverage <= floor)),
        "minimum_coverage": float(np.min(coverage)),
        "median_coverage": float(np.median(coverage)),
        "unsupported_energy_removed_fraction": removed,
        "local_constancy_floor": constancy_floor,
        "local_constancy_authority_mean": float(np.mean(
            constancy_authority[crop])),
        "local_constancy_authority_range": [
            float(np.min(constancy_authority[crop])),
            float(np.max(constancy_authority[crop])),
        ],
        "moment_transport_evaluations": int(constancy_floor > 0.0),
        "separate_moment_transports_avoided": int(constancy_floor > 0.0),
        "passes_used": len(trace),
        "descent_method": descent_method,
        "step_trace": step_trace,
        "stopped_by": stopped_by,
        "discrepancy_ratio_target": target,
        "initial_discrepancy": initial_discrepancy,
        "terminal_discrepancy": terminal_discrepancy,
        "transport_backend": plan.backend,
        "otf_evaluations": 1,
        "forward_evaluations": 1 + len(trace),
        "adjoint_evaluations": len(trace),
        "redundant_forward_evaluations_removed": len(trace),
    }
    return latent[crop], trace, support_record


def _line_characteristic_inverse(
    observation: np.ndarray,
    *,
    length: int,
    vertical: bool,
    seed_penalty: float = 0.01,
    flux_penalty: float = 0.01,
) -> tuple[np.ndarray, dict[str, object]]:
    """Invert an axis-aligned uniform exposure by its exact path recurrence.

    For an odd box length ``L``, differencing adjacent exposure integrals gives

        L (y_i - y_(i-1)) = x_(i+r) - x_(i-r-1).

    Each residue class modulo ``L`` is therefore one transported
    characteristic.  Its seed is the honest null-space state.  We select the
    finite seed vector by minimum correction and longitudinal flux action,
    rather than dividing by the zero-bearing Fourier transfer function.
    """
    from scipy.ndimage import convolve1d

    value = np.asarray(observation, dtype=np.float64)
    working = np.swapaxes(value, 0, 1) if vertical else value
    if working.ndim == 2:
        working = working[..., None]
    height, width, channels = working.shape
    count = max(int(length), 3)
    if count % 2 == 0:
        count += 1
    count = min(count, width - (1 - width % 2))
    if count < 3:
        return value.copy(), {
            "selected": False, "reason": "line_shorter_than_raster"}
    radius = count // 2
    basis = np.zeros((width, count), dtype=np.float64)
    basis[np.arange(width), np.arange(width) % count] = 1.0
    box = np.ones(count, dtype=np.float64) / count
    blurred_basis = np.stack([
        convolve1d(basis[:, column], box, mode="reflect")
        for column in range(count)
    ], axis=1)
    difference_basis = np.diff(basis, axis=0)
    normal = (
        blurred_basis.T @ blurred_basis
        + max(float(seed_penalty), 0.0) * (basis.T @ basis)
        + max(float(flux_penalty), 0.0)
        * (difference_basis.T @ difference_basis)
        + 1e-8 * np.eye(count)
    )
    output = np.empty_like(working)
    residuals: list[float] = []
    for channel in range(channels):
        for row in range(height):
            measured = working[row, :, channel]
            transported = np.zeros(width, dtype=np.float64)
            for index in range(count, width):
                transported[index] = (
                    transported[index - count]
                    + count * (
                        measured[index - radius]
                        - measured[index - radius - 1]
                    )
                )
            blurred_transport = convolve1d(
                transported, box, mode="reflect")
            difference_transport = np.diff(transported)
            rhs = (
                blurred_basis.T @ (measured - blurred_transport)
                + max(float(seed_penalty), 0.0)
                * basis.T @ (measured - transported)
                - max(float(flux_penalty), 0.0)
                * difference_basis.T @ difference_transport
            )
            seed = np.linalg.solve(normal, rhs)
            recovered = np.clip(transported + basis @ seed, 0.0, 1.0)
            output[row, :, channel] = recovered
            residuals.append(float(np.sqrt(np.mean((
                convolve1d(recovered, box, mode="reflect") - measured
            ) ** 2))))
    if value.ndim == 2:
        output = output[..., 0]
    if vertical:
        output = np.swapaxes(output, 0, 1)
    return output, {
        "selected": True,
        "method": "line_characteristic_seed_transport",
        "axis": "vertical" if vertical else "horizontal",
        "line_length": count,
        "seed_gauge_dimension_per_ray": count - 1,
        "seed_policy": "minimum_correction_plus_longitudinal_flux_action",
        "seed_penalty": float(seed_penalty),
        "flux_penalty": float(flux_penalty),
        "mean_uniform_forward_rms": float(np.mean(residuals)),
    }


def _principal_path_geometry(
    factor: TransportMixFactorization,
) -> tuple[int, float, float]:
    covariance = np.asarray(factor.mixing_covariance, dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if eigenvalues[1] <= 0.5:
        return 1, 0.0, 1.0
    straightness = float(eigenvalues[0] / max(eigenvalues[1], 1e-12))
    direction = eigenvectors[:, 1]
    angle = math.degrees(math.atan2(direction[1], direction[0])) % 180.0
    length = max(int(round(math.sqrt(12.0 * eigenvalues[1]))), 3)
    if length % 2 == 0:
        length += 1
    return length, angle, straightness


def _center_crop_shape(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = map(int, shape)
    y0 = max((image.shape[0] - height) // 2, 0)
    x0 = max((image.shape[1] - width) // 2, 0)
    return image[y0 : y0 + height, x0 : x0 + width]


def _transported_line_candidate(
    observation: np.ndarray,
    *,
    length: int,
    angle_degrees: float,
) -> tuple[np.ndarray, dict[str, object], np.ndarray]:
    """Pull an oblique line into an expanded chart, recur, and push it back."""
    from scipy.ndimage import rotate

    value = np.asarray(observation, dtype=np.float64)
    angle = float(angle_degrees) % 180.0
    horizontal_distance = min(angle, 180.0 - angle)
    vertical_distance = abs(angle - 90.0)
    if min(horizontal_distance, vertical_distance) <= 0.75:
        candidate, record = _line_characteristic_inverse(
            value,
            length=length,
            vertical=bool(vertical_distance < horizontal_distance),
        )
        return candidate, {
            **record,
            "chart": "native_raster_axis",
            "chart_angle_degrees": angle,
            "chart_roundtrip_rms": 0.0,
        }, np.zeros_like(value)
    chart = rotate(
        value,
        angle,
        reshape=True,
        order=1,
        mode="reflect",
        prefilter=False,
    )
    recovered_chart, record = _line_characteristic_inverse(
        chart, length=length, vertical=False)
    pushed = rotate(
        recovered_chart,
        -angle,
        reshape=True,
        order=1,
        mode="reflect",
        prefilter=False,
    )
    candidate = _center_crop_shape(pushed, value.shape[:2])
    roundtrip = rotate(
        chart,
        -angle,
        reshape=True,
        order=1,
        mode="reflect",
        prefilter=False,
    )
    roundtrip = _center_crop_shape(roundtrip, value.shape[:2])
    chart_error = np.abs(roundtrip - value)
    return candidate, {
        **record,
        "chart": "expanded_rotated_transport_coordinates",
        "chart_angle_degrees": angle,
        "chart_shape": list(chart.shape[:2]),
        "chart_roundtrip_rms": float(np.sqrt(np.mean(chart_error ** 2))),
    }, chart_error


def two_stage_deblur_known(
    observation: np.ndarray,
    kernel: TransportKernel,
    *,
    passes: int = 64,
    ratio_limit: float = 6.0,
    coverage_floor: float = 5e-4,
    local_constancy_floor: float = 0.004,
    reference: np.ndarray | None = None,
    path_authority_scale: float = 1.0,
    line_constraint_scale: float = 1.0,
    path_basin_uncertainty_weight: float = 0.0,
) -> TwoStageDeblurResult:
    """Undo deterministic shift, then invert the centered positive mixing law."""
    measured = np.asarray(observation, dtype=np.float64)
    before = image_fingerprint(measured)
    factor = factor_transport_mix(kernel, audit_shape=measured.shape[:2])
    inverse_shift = (
        -factor.deterministic_shift_xy[0],
        -factor.deterministic_shift_xy[1],
    )
    aligned = (
        shift_image_reflect(measured, inverse_shift)
        if factor.shift_detected else measured.copy()
    )
    image, residual_trace, support_record = _positive_center_inverse(
        aligned,
        factor.centered_mixing,
        passes=passes,
        ratio_limit=ratio_limit,
        coverage_floor=coverage_floor,
        local_constancy_floor=local_constancy_floor,
    )
    path_length, principal_angle, straightness = _principal_path_geometry(factor)
    curvature_ratio = straightness
    curvilinear_chart = fit_curvilinear_exposure_chart(
        factor.centered_mixing)
    tangent_turn = curvilinear_chart.tangent_turn_degrees
    relative_tube_width = (
        curvilinear_chart.tube_rms
        / max(curvilinear_chart.path_length, 1e-12)
    )
    requested_authority_scale = float(np.clip(path_authority_scale, 0.0, 1.0))
    # The recurrence contributes missing path geometry early.  Once the
    # positive basin has already spent many descent passes, retaining its full
    # authority duplicates sharpening and can reintroduce faint halos.  The
    # square-root law is a finite action dilution, with the original 12-pass
    # checkpoint as unit action.
    descent_authority = min(
        1.0, math.sqrt(12.0 / max(float(passes), 1.0)))
    authority_scale = requested_authority_scale * descent_authority
    path_uncertainty: np.ndarray | None = None
    characteristic_record: dict[str, object]
    mixing_active = bool(
        factor.centered_mixing.psf.size > 1
        and max(int(passes), 0) > 0
        and requested_authority_scale > 0.0
    )
    if mixing_active:
        pre_constraint_discrepancy = residual_discrepancy(
            aligned,
            apply_reflect(image, factor.centered_mixing),
        )
        discrepancy_target = float(
            support_record.get("discrepancy_ratio_target", 1.1))
        discrepancy_excess = max(
            pre_constraint_discrepancy["total_to_read_ratio"]
            - discrepancy_target,
            0.0,
        )
        residual_demand = 1.0 - math.exp(-discrepancy_excess)
        anisotropy_coherence = math.exp(
            -0.5 * (straightness / 0.08) ** 2)
        tangent_coherence = math.exp(
            -0.5 * (tangent_turn / 5.0) ** 2)
        extent_coherence = 1.0 - math.exp(
            -max(float(path_length - 1), 0.0))
        line_coherence = (
            anisotropy_coherence
            * tangent_coherence
            * extent_coherence
        )
        lattice_alignment = (
            math.cos(math.radians(2.0 * principal_angle)) ** 2)
        line_authority = (
            authority_scale
            * float(np.clip(line_constraint_scale, 0.0, 1.0))
            * residual_demand
            * line_coherence
            * (0.025 + 0.225 * lattice_alignment)
        )
        line_record: dict[str, object] = {
            "method": "continuous_line_characteristic_constraint",
            "principal_angle_degrees": principal_angle,
            "minor_major_covariance_ratio": straightness,
            "fitted_tangent_turn_degrees": tangent_turn,
            "coherence": line_coherence,
            "anisotropy_coherence": anisotropy_coherence,
            "tangent_coherence": tangent_coherence,
            "extent_coherence": extent_coherence,
            "lattice_alignment": lattice_alignment,
            "residual_demand": residual_demand,
            "pre_constraint_discrepancy": pre_constraint_discrepancy,
            "authority": line_authority,
            "evaluated": False,
        }
        line_uncertainty = np.zeros_like(image)
        # Omitting a recurrence whose maximum possible pixel contribution is
        # below this scale is numerical dead-code elimination, not a
        # blur-family choice.  The continuous coherence law remains recorded.
        line_evaluation_floor = 1e-6
        line_record["evaluation_floor"] = line_evaluation_floor
        if path_length >= 3 and line_authority > line_evaluation_floor:
            pre_line = image
            characteristic_image, recurrence_record, chart_uncertainty = (
                _transported_line_candidate(
                    aligned,
                    length=path_length,
                    angle_degrees=principal_angle,
                )
            )
            rejected = (
                (1.0 - line_authority)
                * np.abs(characteristic_image - pre_line)
            )
            line_uncertainty = line_coherence * np.sqrt(
                chart_uncertainty * chart_uncertainty
                + rejected * rejected
            )
            image = np.clip(
                pre_line + line_authority * (characteristic_image - pre_line),
                0.0,
                1.0,
            )
            line_record = {
                **line_record,
                **recurrence_record,
                "evaluated": True,
            }

        refinement_passes = max(
            1, min(32, int(math.ceil(max(float(passes), 1.0) / 2.0))))
        endpoint_passes = max(
            1, min(4, int(round(math.sqrt(refinement_passes)))))
        pre_refinement = image
        continuous = refine_curvilinear_exposure(
            aligned,
            factor.centered_mixing,
            pre_refinement,
            passes=refinement_passes,
            endpoint_passes=endpoint_passes,
            ratio_limit=min(max(float(ratio_limit), 1.0), 2.0),
            coverage_floor=coverage_floor,
            endpoint_basin_uncertainty_weight=(
                path_basin_uncertainty_weight),
            local_constancy_floor=local_constancy_floor,
        )
        continuous_correction = continuous.image - pre_refinement
        image = np.clip(
            pre_refinement
            + requested_authority_scale * continuous_correction,
            0.0,
            1.0,
        )
        withheld = (
            (1.0 - requested_authority_scale)
            * np.abs(continuous_correction)
        )
        path_uncertainty = np.sqrt(
            continuous.uncertainty * continuous.uncertainty
            + line_uncertainty * line_uncertainty
            + withheld * withheld
        )
        characteristic_record = {
            **continuous.diagnostics,
            "selected": True,
            "method": "continuous_positive_exposure_transport",
            "role": "one_exact_operator_with_continuous_constraint_weights",
            "authority": requested_authority_scale,
            "principal_angle_degrees": principal_angle,
            "minor_major_covariance_ratio": curvature_ratio,
            "fitted_tangent_turn_degrees": tangent_turn,
            "relative_path_tube_width": relative_tube_width,
            "line_constraint": line_record,
            "line_constraint_authority": line_authority,
            "uncertainty_rms": float(np.sqrt(np.mean(
                path_uncertainty * path_uncertainty))),
            "uncertainty_q95": float(np.quantile(path_uncertainty, 0.95)),
            "seed_policy": "principal_endpoint_gauges_transported_by_exact_adjoint",
        }
        method = "shift_first_continuous_positive_exposure_transport"
    else:
        characteristic_record = {
            "selected": False,
            "reason": (
                "path_authority_withheld_by_operator_trust_policy"
                if requested_authority_scale <= 0.0
                else "identity_or_zero_transport_action"
            ),
            "principal_angle_degrees": principal_angle,
            "minor_major_covariance_ratio": straightness,
            "fitted_tangent_turn_degrees": tangent_turn,
            "relative_path_tube_width": relative_tube_width,
        }
        method = "shift_first_positive_center_transport"
    # Audit against the original, unfactored forward operator.
    transported = (
        shift_image_reflect(image, factor.deterministic_shift_xy)
        if factor.shift_detected else image
    )
    prediction = apply_reflect(transported, factor.centered_mixing)
    forward_rms = float(np.sqrt(np.mean((prediction - measured) ** 2)))
    after = image_fingerprint(measured)
    diagnostics: dict[str, object] = {
        "method": method,
        "decision_order": ["deterministic_transport", "centered_mixing"],
        "deterministic_shift_xy": list(factor.deterministic_shift_xy),
        "shift_detected": factor.shift_detected,
        "shift_observability": "known_forward_operator",
        "mixing_covariance": factor.mixing_covariance.tolist(),
        "mixing_passes": max(int(passes), 0),
        "mixing_residual_trace": residual_trace,
        "support_gate": support_record,
        "characteristic_transport": characteristic_record,
        "path_authority_scale": authority_scale,
        "requested_path_authority_scale": requested_authority_scale,
        "line_constraint_scale": float(np.clip(
            line_constraint_scale, 0.0, 1.0)),
        "path_basin_uncertainty_weight": max(
            float(path_basin_uncertainty_weight), 0.0),
        "local_constancy_floor": max(float(local_constancy_floor), 0.0),
        "descent_authority_dilution": descent_authority,
        "forward_rms": forward_rms,
        "observation_fingerprint_before": before,
        "observation_fingerprint_after": after,
        "observation_unchanged": bool(before == after),
        "boundary": "reflect_padded_no_periodic_wrap",
        "phase_factorization_error": factor.phase_factorization_error,
    }
    if reference is not None:
        truth = np.asarray(reference, dtype=np.float64)
        if truth.shape != measured.shape:
            raise ValueError("synthetic reference and observation must share one shape")
        mse_before = max(float(np.mean((measured - truth) ** 2)), np.finfo(float).tiny)
        mse_after = max(float(np.mean((image - truth) ** 2)), np.finfo(float).tiny)
        diagnostics.update({
            "truth_role": "evaluation_only",
            "observation_psnr": float(-10.0 * math.log10(mse_before)),
            "result_psnr": float(-10.0 * math.log10(mse_after)),
            "psnr_gain": float(10.0 * math.log10(mse_before / mse_after)),
        })
    return TwoStageDeblurResult(
        image=image,
        aligned_observation=aligned,
        factorization=factor,
        diagnostics=diagnostics,
        uncertainty=path_uncertainty,
    )


def single_observation_shift_policy() -> RelativeShiftEstimate:
    """Return the honest absolute-shift verdict for one unknown observation."""
    return RelativeShiftEstimate(
        shift_xy=(0.0, 0.0),
        peak_ratio=0.0,
        observable=False,
        reason="absolute_translation_is_single_image_gauge",
    )


def estimate_centered_mixing_phase(
    observation: np.ndarray,
    *,
    radius: int = 18,
    directional_threshold: float = 2.7,
) -> CenteredMixEstimate:
    """Estimate centered mixing moments from one phase-only image.

    This estimation-time branch distinguishes a strongly directional path from
    an isotropic center cloud.  It does not select the deblurring algorithm:
    both estimates are passed to the same positive center-transport inverse.
    The estimate is deliberately centered because absolute translation is not
    identifiable from a single unknown image.
    """
    image = _luminance(observation)
    if min(image.shape) < 32:
        raise ValueError("blind mixing estimation needs at least 32x32 pixels")
    window = np.hanning(image.shape[0])[:, None] * np.hanning(image.shape[1])[None, :]
    spectrum = np.fft.fft2((image - np.mean(image)) * window)
    phase_only = np.abs(np.fft.ifft2(
        spectrum / np.maximum(np.abs(spectrum), 1e-10)))
    autocorrelation = np.fft.fftshift(np.fft.ifft2(
        np.abs(np.fft.fft2(phase_only)) ** 2).real)
    maximum_radius = max(3, (min(image.shape) - 1) // 2)
    support = min(max(int(radius), 3), maximum_radius)
    cy, cx = np.asarray(image.shape) // 2
    patch = autocorrelation[
        cy - support : cy + support + 1,
        cx - support : cx + support + 1,
    ].copy()
    yy, xx = np.mgrid[-support : support + 1, -support : support + 1]
    off_center = (xx != 0) | (yy != 0)
    baseline = float(np.quantile(patch[off_center], 0.60))
    evidence = np.maximum(patch - baseline, 0.0)
    off_cap = float(np.quantile(evidence[off_center], 0.95))
    evidence[support, support] = min(evidence[support, support], off_cap)
    positive = evidence[evidence > 0.0]
    if not len(positive):
        return CenteredMixEstimate(
            kernel=identity_kernel(), covariance=np.zeros((2, 2)),
            anisotropy_ratio=1.0, estimator_branch="insufficient_phase_evidence",
            confidence=0.0)
    evidence[evidence < np.quantile(positive, 0.78)] = 0.0
    mass = evidence / max(float(np.sum(evidence)), np.finfo(float).tiny)
    # Autocorrelation doubles the covariance of a displacement law.
    covariance = 0.5 * np.asarray((
        (np.sum(mass * xx * xx), np.sum(mass * xx * yy)),
        (np.sum(mass * xx * yy), np.sum(mass * yy * yy)),
    ))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    anisotropy = float(eigenvalues[1] / max(eigenvalues[0], 1e-8))
    if anisotropy >= float(directional_threshold):
        direction = eigenvectors[:, 1]
        angle = math.degrees(math.atan2(direction[1], direction[0]))
        # Empirical finite-window correction removes residual scene spread.
        length = max(float(np.sqrt(12.0 * 0.70 * eigenvalues[1])), 1.0)
        kernel = line_kernel(length, angle)
        branch = "directional_path_moment"
        confidence = float(np.clip(
            (anisotropy - directional_threshold) / directional_threshold,
            0.0, 1.0))
    else:
        sigma = max(float(np.sqrt(0.70 * np.mean(eigenvalues))), 0.0)
        kernel = gaussian_kernel(sigma) if sigma >= 0.45 else identity_kernel()
        branch = "isotropic_center_moment"
        confidence = float(np.clip(
            1.0 - abs(anisotropy - 1.0) / directional_threshold,
            0.0, 1.0))
    return CenteredMixEstimate(
        kernel=kernel,
        covariance=covariance,
        anisotropy_ratio=anisotropy,
        estimator_branch=branch,
        confidence=confidence,
    )


def two_stage_deblur_blind(
    observation: np.ndarray,
    *,
    passes: int = 8,
    ratio_limit: float = 4.0,
    coverage_floor: float = 5e-4,
) -> TwoStageDeblurResult:
    """Single-observation provisional: abstain on shift, estimate center mix."""
    shift = single_observation_shift_policy()
    estimate = estimate_centered_mixing_phase(observation)
    requested_passes = max(int(passes), 0)
    effective_passes = min(requested_passes, 8)
    result = two_stage_deblur_known(
        observation,
        estimate.kernel,
        passes=effective_passes,
        ratio_limit=ratio_limit,
        coverage_floor=coverage_floor,
        path_authority_scale=0.0,
    )
    # A single-image kernel estimate is evidence, not an oracle.  Transport
    # only the supported fraction of its proposed correction.  Directional
    # phase consensus can earn more authority; an isotropic moment remains a
    # deliberately conservative nudge because scene texture and blur are
    # confounded in one observation.
    if estimate.estimator_branch == "directional_path_moment":
        authority = float(np.clip(
            0.10 + 0.35 * estimate.confidence, 0.10, 0.45))
    elif estimate.estimator_branch == "isotropic_center_moment":
        authority = 0.18
    else:
        authority = 0.0
    measured = np.asarray(observation, dtype=np.float64)
    image = np.clip(
        measured + authority * (result.image - measured), 0.0, 1.0)
    prediction = apply_reflect(image, estimate.kernel)
    forward_rms = float(np.sqrt(np.mean((prediction - measured) ** 2)))
    diagnostics = {
        **result.diagnostics,
        "forward_rms": forward_rms,
        "shift_detected": False,
        "shift_observability": shift.reason,
        "kernel_origin": "single_image_phase_only_center_moments",
        "estimation_branch": estimate.estimator_branch,
        "estimation_confidence": estimate.confidence,
        "estimated_anisotropy_ratio": estimate.anisotropy_ratio,
        "estimated_raw_covariance": estimate.covariance.tolist(),
        "estimated_centered_kernel": estimate.kernel.name,
        "blind_inverse_authority": authority,
        "path_recurrence_policy": (
            "disabled_until_known_operator_or_multi_observation_consensus"
        ),
        "requested_mixing_passes": requested_passes,
        "effective_mixing_passes": effective_passes,
        "decision": (
            "evidence_weighted_center_mix"
            if authority > 0.0 else "abstain_insufficient_mix_evidence"
        ),
        "claim_boundary": (
            "provisional single-image centered-mix estimate; absolute shift "
            "and curved-path handedness are not identified"
        ),
    }
    return TwoStageDeblurResult(
        image=image,
        aligned_observation=result.aligned_observation,
        factorization=result.factorization,
        diagnostics=diagnostics,
        uncertainty=(
            None
            if result.uncertainty is None
            else authority * result.uncertainty
        ),
    )


def identity_two_stage(image: np.ndarray) -> TwoStageDeblurResult:
    """Convenience identity result for explicit as-is abstention paths."""
    return two_stage_deblur_known(image, identity_kernel(), passes=0)
