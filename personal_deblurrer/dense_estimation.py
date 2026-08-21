"""Continuous dense barycentric-flow estimation and exposure consensus.

The estimator carries one 2-D displacement field. Translation, affine motion,
rotation, shear, and local deformation are limiting shapes of that field, not
classes selected by a classifier. Forward/reverse registration, a robust
photometric law, and an image-induced Eikonal conductance all contribute
continuously. Confidence shortens unsupported transport before the field is
converted into a positive exposure measure.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .spatial_consensus import solve_spatial_field_consensus
from .spatial_transport import SpatialExposureField


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
    raise ValueError("dense flow images must be HxW or RGB")


def _sample(value: np.ndarray, flow_xy: np.ndarray) -> np.ndarray:
    from scipy.ndimage import map_coordinates

    array = np.asarray(value, dtype=np.float64)
    height, width = flow_xy.shape[:2]
    yy, xx = np.mgrid[:height, :width]
    coordinates = (yy + flow_xy[..., 1], xx + flow_xy[..., 0])
    if array.ndim == 2:
        return map_coordinates(
            array, coordinates, order=1, mode="reflect", prefilter=False)
    return np.stack([
        map_coordinates(
            array[..., channel], coordinates,
            order=1, mode="reflect", prefilter=False)
        for channel in range(array.shape[2])
    ], axis=2)


def _resize_image(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    from scipy.ndimage import gaussian_filter, zoom

    value = np.asarray(image, dtype=np.float64)
    if value.shape == shape:
        return value.copy()
    ratio_y = shape[0] / value.shape[0]
    ratio_x = shape[1] / value.shape[1]
    sigma = max(0.0, 0.5 * (1.0 / min(ratio_x, ratio_y) - 1.0))
    filtered = gaussian_filter(value, sigma=sigma, mode="reflect")
    return zoom(
        filtered,
        (ratio_y, ratio_x),
        order=1,
        mode="reflect",
        prefilter=False,
    )


def _resize_flow(flow_xy: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    from scipy.ndimage import zoom

    value = np.asarray(flow_xy, dtype=np.float64)
    if value.shape[:2] == shape:
        return value.copy()
    ratio_y = shape[0] / value.shape[0]
    ratio_x = shape[1] / value.shape[1]
    result = zoom(
        value,
        (ratio_y, ratio_x, 1.0),
        order=1,
        mode="nearest",
        prefilter=False,
    )
    result[..., 0] *= ratio_x
    result[..., 1] *= ratio_y
    return result


def _phase_translation(reference: np.ndarray, moving: np.ndarray) -> np.ndarray:
    """Return the continuous sampling offset that aligns moving to reference."""
    from scipy.ndimage import gaussian_filter
    from scipy.optimize import minimize

    first = gaussian_filter(_luminance(reference), 1.0, mode="reflect")
    second = gaussian_filter(_luminance(moving), 1.0, mode="reflect")
    height, width = first.shape
    window = np.outer(np.hanning(height), np.hanning(width))
    a = np.fft.fft2((first - np.mean(first)) * window)
    b = np.fft.fft2((second - np.mean(second)) * window)
    cross = b * np.conjugate(a)
    cross /= np.maximum(np.abs(cross), 1e-12)
    correlation = np.fft.ifft2(cross).real
    peak_y, peak_x = np.unravel_index(np.argmax(correlation), correlation.shape)
    if peak_x > width // 2:
        peak_x -= width
    if peak_y > height // 2:
        peak_y -= height
    seed = np.asarray((float(peak_x), float(peak_y)), dtype=np.float64)
    yy, xx = np.mgrid[:height, :width]
    margin = max(3, int(round(0.08 * min(height, width))))
    crop = (slice(margin, -margin), slice(margin, -margin))
    target = first[crop]

    def objective(offset: np.ndarray) -> float:
        flow = np.empty((height, width, 2), dtype=np.float64)
        flow[..., 0] = float(offset[0])
        flow[..., 1] = float(offset[1])
        residual = _sample(second, flow)[crop] - target
        scale = max(1.4826 * float(np.median(np.abs(residual))), 0.01)
        return float(np.mean(
            np.sqrt(residual * residual + scale * scale) - scale))

    bound_x = min(0.25 * width, 16.0)
    bound_y = min(0.25 * height, 16.0)
    seed[0] = np.clip(seed[0], -bound_x, bound_x)
    seed[1] = np.clip(seed[1], -bound_y, bound_y)
    result = minimize(
        objective,
        seed,
        method="Powell",
        bounds=((-bound_x, bound_x), (-bound_y, bound_y)),
        options={"xtol": 1e-3, "ftol": 1e-5, "maxiter": 48},
    )
    return np.asarray(result.x, dtype=np.float64)


def _conductance_laplacian_weights(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    horizontal_difference = image[:, 1:] - image[:, :-1]
    vertical_difference = image[1:, :] - image[:-1, :]
    sample = np.concatenate((
        np.abs(horizontal_difference).ravel(),
        np.abs(vertical_difference).ravel(),
    ))
    scale = max(float(np.quantile(sample, 0.65)), 0.01)
    horizontal = 0.03 + 0.97 / np.sqrt(
        1.0 + (horizontal_difference / scale) ** 2)
    vertical = 0.03 + 0.97 / np.sqrt(
        1.0 + (vertical_difference / scale) ** 2)
    return horizontal, vertical


def _laplacian(
    value: np.ndarray,
    horizontal: np.ndarray,
    vertical: np.ndarray,
) -> np.ndarray:
    result = np.zeros_like(value)
    difference = value[:, :-1] - value[:, 1:]
    result[:, :-1] += horizontal * difference
    result[:, 1:] -= horizontal * difference
    difference = value[:-1, :] - value[1:, :]
    result[:-1, :] += vertical * difference
    result[1:, :] -= vertical * difference
    return result


def _laplacian_degree(
    horizontal: np.ndarray,
    vertical: np.ndarray,
) -> np.ndarray:
    height = horizontal.shape[0]
    width = vertical.shape[1]
    degree = np.zeros((height, width), dtype=np.float64)
    degree[:, :-1] += horizontal
    degree[:, 1:] += horizontal
    degree[:-1, :] += vertical
    degree[1:, :] += vertical
    return degree


def _flow_energy(
    reference: np.ndarray,
    moving: np.ndarray,
    reference_gradient_x: np.ndarray,
    reference_gradient_y: np.ndarray,
    moving_gradient_x: np.ndarray,
    moving_gradient_y: np.ndarray,
    flow: np.ndarray,
    horizontal: np.ndarray,
    vertical: np.ndarray,
    smoothness: float,
    gradient_weight: float,
    robust_flow: bool,
) -> float:
    residual = _sample(moving, flow) - reference
    scale = max(1.4826 * float(np.median(np.abs(residual))), 0.008)
    data = np.mean(scale * scale * (
        np.sqrt(1.0 + (residual / scale) ** 2) - 1.0))
    gradient_data = 0.0
    if float(gradient_weight) > 0.0:
        residual_x = _sample(moving_gradient_x, flow) - reference_gradient_x
        residual_y = _sample(moving_gradient_y, flow) - reference_gradient_y
        for gradient_residual in (residual_x, residual_y):
            gradient_scale = max(
                1.4826 * float(np.median(np.abs(gradient_residual))), 0.004)
            gradient_data += float(np.mean(
                gradient_scale * gradient_scale * (
                    np.sqrt(1.0 + (gradient_residual / gradient_scale) ** 2)
                    - 1.0)))
    dx = flow[:, 1:] - flow[:, :-1]
    dy = flow[1:, :] - flow[:-1, :]
    if robust_flow:
        flow_scale = 0.5
        regularity = (
            np.mean(horizontal * flow_scale * flow_scale * (
                np.sqrt(
                    1.0 + np.sum(dx * dx, axis=2) / (flow_scale * flow_scale))
                - 1.0))
            + np.mean(vertical * flow_scale * flow_scale * (
                np.sqrt(
                    1.0 + np.sum(dy * dy, axis=2) / (flow_scale * flow_scale))
                - 1.0))
        )
    else:
        regularity = (
            np.mean(horizontal[..., None] * dx * dx)
            + np.mean(vertical[..., None] * dy * dy))
    return float(
        data
        + float(gradient_weight) * gradient_data
        + 0.5 * smoothness * smoothness * regularity)


def _transport_connection_confidence(
    evidence: np.ndarray,
    support: np.ndarray,
    image: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Harmonically extend measured connection through the image metric."""
    from scipy.sparse.linalg import LinearOperator, cg

    measured = np.clip(np.asarray(evidence, dtype=np.float64), 0.0, 1.0)
    anchor = np.clip(np.asarray(support, dtype=np.float64), 0.0, 1.0)
    horizontal, vertical = _conductance_laplacian_weights(image)
    height, width = measured.shape
    count = measured.size
    transport_scale = 0.35
    numerical_anchor = 1e-5

    def product(flat: np.ndarray) -> np.ndarray:
        value = flat.reshape((height, width))
        return (
            anchor * value
            + transport_scale * _laplacian(value, horizontal, vertical)
            + numerical_anchor * value
        ).ravel()

    operator = LinearOperator(
        (count, count), matvec=product, dtype=np.float64)
    inverse_diagonal = 1.0 / np.maximum(
        anchor
        + transport_scale * _laplacian_degree(horizontal, vertical)
        + numerical_anchor,
        numerical_anchor,
    )
    preconditioner = LinearOperator(
        (count, count),
        matvec=lambda flat: inverse_diagonal.ravel() * flat,
        dtype=np.float64,
    )
    iterations = 0

    def count_iteration(_value: np.ndarray) -> None:
        nonlocal iterations
        iterations += 1

    transported, status = cg(
        operator,
        (anchor * measured).ravel(),
        rtol=1e-5,
        atol=0.0,
        maxiter=160,
        M=preconditioner,
        callback=count_iteration,
    )
    authority = np.clip(transported.reshape((height, width)), 0.0, 1.0)
    if not np.any(anchor > 1e-6):
        authority.fill(0.0)
    return authority, {
        "iterations": iterations,
        "status": int(status),
        "measured_mean": float(np.mean(anchor * measured)),
        "transported_mean": float(np.mean(authority)),
        "transported_min": float(np.min(authority)),
        "transported_max": float(np.max(authority)),
    }


def _refine_level(
    reference: np.ndarray,
    moving: np.ndarray,
    flow: np.ndarray,
    *,
    smoothness: float,
    warp_iterations: int,
    cg_iterations: int,
    gradient_weight: float,
    robust_flow: bool,
) -> tuple[np.ndarray, list[float], list[int], list[int]]:
    from scipy.ndimage import gaussian_filter
    from scipy.sparse.linalg import LinearOperator, cg

    first = gaussian_filter(reference, 0.7, mode="reflect")
    second = gaussian_filter(moving, 0.7, mode="reflect")
    first_gradient_y, first_gradient_x = np.gradient(first)
    gradient_y, gradient_x = np.gradient(second)
    if float(gradient_weight) > 0.0:
        gradient_x_y, gradient_x_x = np.gradient(gradient_x)
        gradient_y_y, gradient_y_x = np.gradient(gradient_y)
    metric_horizontal, metric_vertical = _conductance_laplacian_weights(first)
    height, width = first.shape
    count = height * width
    energy_trace = [_flow_energy(
        first,
        second,
        first_gradient_x,
        first_gradient_y,
        gradient_x,
        gradient_y,
        flow,
        metric_horizontal,
        metric_vertical,
        smoothness,
        gradient_weight,
        robust_flow,
    )]
    cg_trace: list[int] = []
    cg_status_trace: list[int] = []
    damping = 1e-5
    alpha2 = float(smoothness) ** 2
    for _ in range(max(int(warp_iterations), 1)):
        if robust_flow:
            flow_difference_x = flow[:, 1:] - flow[:, :-1]
            flow_difference_y = flow[1:, :] - flow[:-1, :]
            flow_scale = 0.5
            horizontal = metric_horizontal / np.sqrt(
                1.0 + np.sum(flow_difference_x * flow_difference_x, axis=2)
                / (flow_scale * flow_scale))
            vertical = metric_vertical / np.sqrt(
                1.0 + np.sum(flow_difference_y * flow_difference_y, axis=2)
                / (flow_scale * flow_scale))
        else:
            horizontal = metric_horizontal
            vertical = metric_vertical
        warped = _sample(second, flow)
        gx = _sample(gradient_x, flow)
        gy = _sample(gradient_y, flow)
        residual = warped - first
        scale = max(1.4826 * float(np.median(np.abs(residual))), 0.008)
        data_weight = 1.0 / np.sqrt(1.0 + (residual / scale) ** 2)
        data_terms = [(gx, gy, residual, data_weight)]
        if float(gradient_weight) > 0.0:
            residual_gradient_x = gx - first_gradient_x
            residual_gradient_y = gy - first_gradient_y
            gxx = _sample(gradient_x_x, flow)
            gxy = 0.5 * (
                _sample(gradient_x_y, flow) + _sample(gradient_y_x, flow))
            gyy = _sample(gradient_y_y, flow)
            gradient_scale_x = max(
                1.4826 * float(np.median(np.abs(residual_gradient_x))), 0.004)
            gradient_scale_y = max(
                1.4826 * float(np.median(np.abs(residual_gradient_y))), 0.004)
            gradient_data_weight_x = (
                gradient_weight
                / np.sqrt(1.0 + (residual_gradient_x / gradient_scale_x) ** 2)
            )
            gradient_data_weight_y = (
                gradient_weight
                / np.sqrt(1.0 + (residual_gradient_y / gradient_scale_y) ** 2)
            )
            data_terms.extend((
                (gxx, gxy, residual_gradient_x, gradient_data_weight_x),
                (gxy, gyy, residual_gradient_y, gradient_data_weight_y),
            ))
        lx = _laplacian(flow[..., 0], horizontal, vertical)
        ly = _laplacian(flow[..., 1], horizontal, vertical)
        right_x = -alpha2 * lx
        right_y = -alpha2 * ly
        for jacobian_x, jacobian_y, term_residual, term_weight in data_terms:
            right_x -= term_weight * jacobian_x * term_residual
            right_y -= term_weight * jacobian_y * term_residual
        right = np.concatenate((right_x.ravel(), right_y.ravel()))

        def product(flat: np.ndarray) -> np.ndarray:
            dx = flat[:count].reshape((height, width))
            dy = flat[count:].reshape((height, width))
            out_x = alpha2 * _laplacian(dx, horizontal, vertical) + damping * dx
            out_y = alpha2 * _laplacian(dy, horizontal, vertical) + damping * dy
            for jacobian_x, jacobian_y, _term_residual, term_weight in data_terms:
                coupled = term_weight * (
                    jacobian_x * dx + jacobian_y * dy)
                out_x += jacobian_x * coupled
                out_y += jacobian_y * coupled
            return np.concatenate((out_x.ravel(), out_y.ravel()))

        operator = LinearOperator(
            (2 * count, 2 * count), matvec=product, dtype=np.float64)
        regularity_diagonal = (
            alpha2 * _laplacian_degree(horizontal, vertical) + damping)
        block_a = regularity_diagonal.copy()
        block_b = np.zeros_like(block_a)
        block_c = regularity_diagonal.copy()
        for jacobian_x, jacobian_y, _term_residual, term_weight in data_terms:
            block_a += term_weight * jacobian_x * jacobian_x
            block_b += term_weight * jacobian_x * jacobian_y
            block_c += term_weight * jacobian_y * jacobian_y
        block_determinant = np.maximum(
            block_a * block_c - block_b * block_b,
            damping * damping,
        )

        def precondition(flat: np.ndarray) -> np.ndarray:
            value_x = flat[:count].reshape((height, width))
            value_y = flat[count:].reshape((height, width))
            output_x = (
                block_c * value_x - block_b * value_y) / block_determinant
            output_y = (
                block_a * value_y - block_b * value_x) / block_determinant
            return np.concatenate((output_x.ravel(), output_y.ravel()))

        preconditioner = LinearOperator(
            (2 * count, 2 * count), matvec=precondition, dtype=np.float64)
        iteration_count = 0

        def count_iteration(_value: np.ndarray) -> None:
            nonlocal iteration_count
            iteration_count += 1

        increment, _status = cg(
            operator,
            right,
            rtol=2e-3,
            atol=0.0,
            maxiter=max(int(cg_iterations), 1),
            M=preconditioner,
            callback=count_iteration,
        )
        increment = increment.reshape((2, height, width)).transpose(1, 2, 0)
        magnitude = np.sqrt(np.sum(increment * increment, axis=2))
        maximum_increment = max(float(np.quantile(magnitude, 0.98)), 1e-12)
        increment *= min(1.0, 2.0 / maximum_increment)
        best = flow
        best_energy = energy_trace[-1]
        for step in (1.0,):
            candidate = flow + step * increment
            energy = _flow_energy(
                first,
                second,
                first_gradient_x,
                first_gradient_y,
                gradient_x,
                gradient_y,
                candidate,
                metric_horizontal,
                metric_vertical,
                smoothness,
                gradient_weight,
                robust_flow,
            )
            if energy < best_energy:
                best = candidate
                best_energy = energy
        cg_trace.append(iteration_count)
        cg_status_trace.append(int(_status))
        energy_trace.append(best_energy)
        update = float(np.sqrt(np.mean((best - flow) ** 2)))
        flow = best
        if update <= 2e-4:
            break
    return flow, energy_trace, cg_trace, cg_status_trace


def _one_way_dense_flow(
    reference: np.ndarray,
    moving: np.ndarray,
    *,
    pyramid_levels: int,
    warp_iterations: int,
    cg_iterations: int,
    smoothness: float,
    finest_gradient_weight: float = 0.2,
) -> tuple[np.ndarray, dict[str, object]]:
    first = _luminance(reference)
    second = _luminance(moving)
    if first.shape != second.shape:
        raise ValueError("dense flow observations must share one raster")
    level_count = max(int(pyramid_levels), 1)
    factors = [1.0 / (2 ** level) for level in reversed(range(level_count))]
    factors = [factor for factor in factors if min(first.shape) * factor >= 16]
    if not factors or factors[-1] != 1.0:
        factors.append(1.0)
    translation = _phase_translation(first, second)
    flow = None
    records = []
    for factor in factors:
        shape = (
            max(8, int(round(first.shape[0] * factor))),
            max(8, int(round(first.shape[1] * factor))),
        )
        level_first = _resize_image(first, shape)
        level_second = _resize_image(second, shape)
        if flow is None:
            flow = np.empty((*shape, 2), dtype=np.float64)
            flow[..., 0] = translation[0] * shape[1] / first.shape[1]
            flow[..., 1] = translation[1] * shape[0] / first.shape[0]
        else:
            flow = _resize_flow(flow, shape)
        flow, energy, cg_steps, cg_status = _refine_level(
            level_first,
            level_second,
            flow,
            smoothness=smoothness,
            warp_iterations=warp_iterations,
            cg_iterations=cg_iterations,
            gradient_weight=(
                float(finest_gradient_weight) if factor == 1.0 else 0.0),
            robust_flow=(factor == 1.0),
        )
        records.append({
            "shape": list(shape),
            "energy_trace": energy,
            "cg_iterations": cg_steps,
            "cg_status": cg_status,
            "gradient_constancy_weight": (
                float(finest_gradient_weight) if factor == 1.0 else 0.0),
            "robust_flow_action": bool(factor == 1.0),
        })
    assert flow is not None
    return flow, {
        "initial_translation_xy": translation.tolist(),
        "pyramid_factors": factors,
        "levels": records,
        "smoothness": float(smoothness),
        "flow_rms": float(np.sqrt(np.mean(np.sum(flow * flow, axis=2)))),
    }


@dataclass(frozen=True)
class DensePairEstimate:
    forward_sampling_flow_xy: np.ndarray
    reverse_sampling_flow_xy: np.ndarray
    confidence: np.ndarray
    cycle_error_pixels: np.ndarray
    photometric_error: np.ndarray
    flow_standard_deviation_pixels: np.ndarray
    visibility_confidence: tuple[np.ndarray, np.ndarray]
    fields: tuple[SpatialExposureField, SpatialExposureField]
    relative_motion_observable: bool
    common_warp_gauge_unidentifiable: bool
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class DenseConsensusResult:
    image: np.ndarray
    uncertainty: np.ndarray
    estimate: DensePairEstimate
    diagnostics: dict[str, object]


def _pair_fields(
    relative_flow_xy: np.ndarray,
    authority: np.ndarray,
    *,
    duty_cycle: float,
    atoms: int,
) -> tuple[SpatialExposureField, SpatialExposureField]:
    count = max(int(atoms), 1)
    trusted_flow = authority[..., None] * relative_flow_xy
    exposure_velocity = max(float(duty_cycle), 0.0) * trusted_flow
    times = np.linspace(-0.5, 0.5, count, dtype=np.float64)
    residual = times[:, None, None, None] * exposure_velocity[None, ...]
    weights = np.ones((count, *relative_flow_xy.shape[:2]), dtype=np.float64)
    return tuple(
        SpatialExposureField.from_barycentric_paths(
            name=f"dense_pair_{index}_continuous_exposure",
            barycentric_flow_xy=sign * 0.5 * trusted_flow,
            residual_displacements_xy=residual,
            weights=weights,
        )
        for index, sign in enumerate((-1.0, 1.0))
    )  # type: ignore[return-value]


def _connection_visibility(
    reference: np.ndarray,
    moving: np.ndarray,
    sampling_flow_xy: np.ndarray,
    reverse_flow_xy: np.ndarray,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    dict[str, object],
]:
    """Measure and transport one direction's visibility/connection evidence."""
    reverse_on_reference = _sample(reverse_flow_xy, sampling_flow_xy)
    cycle_vector = sampling_flow_xy + reverse_on_reference
    cycle_error = np.sqrt(np.sum(cycle_vector * cycle_vector, axis=2))
    reference_luminance = _luminance(reference)
    moving_luminance = _luminance(moving)
    warped_moving = _sample(moving_luminance, sampling_flow_xy)
    photometric = np.abs(warped_moving - reference_luminance)
    metric_image = _resize_image(reference_luminance, reference_luminance.shape)
    gradient_y, gradient_x = np.gradient(metric_image)
    texture = gradient_x * gradient_x + gradient_y * gradient_y
    positive_texture = texture[texture > np.finfo(float).eps]
    texture_scale = (
        float(np.quantile(positive_texture, 0.35))
        if positive_texture.size else 1.0)
    texture_authority = texture / (texture + max(texture_scale, 1e-8))
    cycle_scale = max(float(np.median(cycle_error)), 0.15)
    photo_scale = max(float(np.median(photometric)), 0.01)
    connection_evidence = (
        1.0 / np.sqrt(1.0 + (cycle_error / (2.5 * cycle_scale)) ** 2)
        / np.sqrt(1.0 + (photometric / (3.0 * photo_scale)) ** 2)
    )
    confidence, confidence_transport = _transport_connection_confidence(
        connection_evidence,
        texture_authority,
        metric_image,
    )
    flow_magnitude = np.sqrt(np.sum(
        sampling_flow_xy * sampling_flow_xy, axis=2))
    visibility = (
        1.0 / (1.0 + (
            cycle_error / (0.25 + 0.05 * flow_magnitude)) ** 4)
        / (1.0 + (photometric / max(3.0 * photo_scale, 0.03)) ** 4)
    )
    from scipy.ndimage import gaussian_filter
    visibility = np.clip(
        gaussian_filter(visibility, 0.6, mode="reflect"), 0.0, 1.0)
    flow_sigma = 0.5 * cycle_error + (1.0 - confidence) * flow_magnitude
    return confidence, visibility, cycle_error, photometric, flow_sigma, {
        "cycle_rms": float(np.sqrt(np.mean(cycle_error * cycle_error))),
        "photometric_rms": float(np.sqrt(np.mean(photometric * photometric))),
        "texture_support": float(np.mean(texture_authority)),
        "consensus_confidence": float(np.mean(confidence)),
        "visibility_mean": float(np.mean(visibility)),
        "visibility_min": float(np.min(visibility)),
        "visibility_max": float(np.max(visibility)),
        "connection_confidence_transport": confidence_transport,
    }


def estimate_dense_pair_exposure(
    first: np.ndarray,
    second: np.ndarray,
    *,
    duty_cycle: float = 0.5,
    atoms: int = 7,
    pyramid_levels: int = 3,
    warp_iterations: int = 5,
    cg_iterations: int = 60,
    smoothness: float = 0.12,
) -> DensePairEstimate:
    """Estimate one continuous relative field with reverse-cycle evidence."""
    first_value = np.asarray(first, dtype=np.float64)
    second_value = np.asarray(second, dtype=np.float64)
    if first_value.shape != second_value.shape:
        raise ValueError("dense pair observations must share one raster")
    signal_scale = max(
        float(np.max(np.abs(first_value))),
        float(np.max(np.abs(second_value))),
        1.0,
    )
    numerical_floor = 64.0 * np.finfo(np.float64).eps * signal_scale
    photometric_identity_error = np.abs(
        _luminance(first_value) - _luminance(second_value))
    if float(np.max(photometric_identity_error)) <= numerical_floor:
        height, width = first_value.shape[:2]
        zero_flow = np.zeros((height, width, 2), dtype=np.float64)
        zero_scalar = np.zeros((height, width), dtype=np.float64)
        fields = _pair_fields(
            zero_flow, zero_scalar, duty_cycle=duty_cycle, atoms=atoms)
        one_way_record = {
            "initial_translation_xy": [0.0, 0.0],
            "pyramid_factors": [],
            "levels": [],
            "smoothness": float(smoothness),
            "flow_rms": 0.0,
            "fast_path": "machine_precision_identical_observations",
        }
        diagnostics = {
            "method": "continuous_dense_barycentric_flow_cycle_consensus",
            "forward": one_way_record,
            "reverse": one_way_record,
            "flow_rms": 0.0,
            "cycle_rms": 0.0,
            "photometric_rms": float(np.sqrt(np.mean(
                photometric_identity_error * photometric_identity_error))),
            "texture_support": 0.0,
            "consensus_confidence": 0.0,
            "relative_motion_observable": False,
            "common_warp_gauge_unidentifiable": True,
            "transport_authority_mean": 0.0,
            "transport_authority_min": 0.0,
            "transport_authority_max": 0.0,
            "connection_confidence_transport": {
                "iterations": 0,
                "status": 0,
                "measured_mean": 0.0,
                "transported_mean": 0.0,
                "transported_min": 0.0,
                "transported_max": 0.0,
            },
            "estimation_decision": "abstain_common_warp_and_exposure_gauge",
            "fast_path": "machine_precision_identical_observations",
        }
        return DensePairEstimate(
            forward_sampling_flow_xy=zero_flow,
            reverse_sampling_flow_xy=zero_flow.copy(),
            confidence=zero_scalar,
            cycle_error_pixels=zero_scalar.copy(),
            photometric_error=photometric_identity_error,
            flow_standard_deviation_pixels=zero_scalar.copy(),
            visibility_confidence=(zero_scalar.copy(), zero_scalar.copy()),
            fields=fields,
            relative_motion_observable=False,
            common_warp_gauge_unidentifiable=True,
            diagnostics=diagnostics,
        )
    forward, forward_record = _one_way_dense_flow(
        first_value,
        second_value,
        pyramid_levels=pyramid_levels,
        warp_iterations=warp_iterations,
        cg_iterations=cg_iterations,
        smoothness=smoothness,
    )
    reverse, reverse_record = _one_way_dense_flow(
        second_value,
        first_value,
        pyramid_levels=pyramid_levels,
        warp_iterations=warp_iterations,
        cg_iterations=cg_iterations,
        smoothness=smoothness,
    )
    (
        confidence,
        forward_visibility_weight,
        cycle_error,
        photometric,
        flow_sigma,
        visibility_forward,
    ) = _connection_visibility(first_value, second_value, forward, reverse)
    (
        reverse_confidence,
        reverse_visibility_weight,
        _,
        _,
        _,
        visibility_reverse,
    ) = _connection_visibility(second_value, first_value, reverse, forward)
    flow_rms = float(np.sqrt(np.mean(np.sum(forward * forward, axis=2))))
    texture_support = float(visibility_forward["texture_support"])
    consensus_confidence = float(np.mean(confidence))
    observable = bool(
        flow_rms >= 0.1
        and texture_support >= 0.02
        and consensus_confidence >= 0.03
    )
    authority = confidence if observable else np.zeros_like(confidence)
    fields = _pair_fields(
        forward, authority, duty_cycle=duty_cycle, atoms=atoms)
    return DensePairEstimate(
        forward_sampling_flow_xy=forward,
        reverse_sampling_flow_xy=reverse,
        confidence=confidence,
        cycle_error_pixels=cycle_error,
        photometric_error=photometric,
        flow_standard_deviation_pixels=flow_sigma,
        visibility_confidence=(
            forward_visibility_weight, reverse_visibility_weight),
        fields=fields,
        relative_motion_observable=observable,
        common_warp_gauge_unidentifiable=True,
        diagnostics={
            "method": "continuous_dense_barycentric_flow_cycle_consensus",
            "forward": forward_record,
            "reverse": reverse_record,
            "flow_rms": flow_rms,
            "cycle_rms": visibility_forward["cycle_rms"],
            "photometric_rms": visibility_forward["photometric_rms"],
            "texture_support": texture_support,
            "consensus_confidence": consensus_confidence,
            "relative_motion_observable": observable,
            "common_warp_gauge_unidentifiable": True,
            "transport_authority_mean": float(np.mean(authority)),
            "transport_authority_min": float(np.min(authority)),
            "transport_authority_max": float(np.max(authority)),
            "connection_confidence_transport": visibility_forward[
                "connection_confidence_transport"],
            "forward_visibility": visibility_forward,
            "reverse_visibility": visibility_reverse,
            "estimation_decision": (
                "relative_dense_flow_supported"
                if observable else "abstain_common_warp_and_exposure_gauge"
            ),
        },
    )


def deblur_dense_pair_consensus(
    first: np.ndarray,
    second: np.ndarray,
    *,
    duty_cycle: float = 0.5,
    atoms: int = 7,
    passes: int = 64,
    pyramid_levels: int = 3,
    warp_iterations: int = 5,
    cg_iterations: int = 60,
    smoothness: float = 0.12,
) -> DenseConsensusResult:
    """Estimate a dense field and use the common spatial consensus inverse."""
    images = (np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64))
    estimate = estimate_dense_pair_exposure(
        images[0],
        images[1],
        duty_cycle=duty_cycle,
        atoms=atoms,
        pyramid_levels=pyramid_levels,
        warp_iterations=warp_iterations,
        cg_iterations=cg_iterations,
        smoothness=smoothness,
    )
    if not estimate.relative_motion_observable:
        mean = np.mean(np.stack(images, axis=0), axis=0)
        uncertainty = np.std(np.stack(images, axis=0), axis=0)
        return DenseConsensusResult(
            image=mean,
            uncertainty=uncertainty,
            estimate=estimate,
            diagnostics={
                **estimate.diagnostics,
                "reconstruction_method": (
                    "shared_latent_spatial_positive_exposure_transport"),
                "passes_used": 0,
                "stopped_by": "common_warp_gauge_abstention",
                "uncertainty_rms": float(np.sqrt(np.mean(
                    uncertainty * uncertainty))),
            },
        )
    mean_image = 0.5 * (_luminance(images[0]) + _luminance(images[1]))
    gy, gx = np.gradient(mean_image)
    gradient = np.sqrt(gx * gx + gy * gy)
    geometric = estimate.flow_standard_deviation_pixels * gradient
    solution = solve_spatial_field_consensus(
        images,
        estimate.fields,
        frame_weights=np.asarray((0.5, 0.5), dtype=np.float64),
        geometric_uncertainty=geometric,
        passes=passes,
        ratio_limit=4.0,
    )
    return DenseConsensusResult(
        image=solution.image,
        uncertainty=solution.uncertainty,
        estimate=estimate,
        diagnostics={
            **estimate.diagnostics,
            **solution.diagnostics,
            "estimation_method": estimate.diagnostics["method"],
            "reconstruction_method": solution.diagnostics["method"],
            "estimation_decision": estimate.diagnostics["estimation_decision"],
            "coverage_decision": solution.diagnostics["estimation_decision"],
        },
    )
