"""Exchange-symmetric multi-capture center and mixing transport.

Every observation pair contributes a relative exposure gain, Fourier-circle
center displacement, and low-frequency centered-mixing covariance difference.
Graph least squares transports those pair relations into one zero-mean gauge.
A directional positive-cone program then finds the minimum-trace set of
per-capture covariances compatible with the graph.  All captures enter one
shared-latent positive forward/adjoint solve; no capture or blur family wins a
selection branch.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np
from scipy.optimize import linprog

from .circles import (
    phase_circle_translation_from_spectra,
    prepare_phase_circle_spectrum,
)
from .dense_estimation import _luminance
from .full_quartic_transport import estimate_full_quartic_transport
from .radiometric_transport import _quantile_log_gain, _soft_sensor_precision
from .quartic_shape_transport import estimate_quartic_shape_transport
from .relative_mixing_transport import (
    _positive_sigma_measure,
    estimate_adaptive_relative_mixing_from_spectra,
    estimate_relative_mixing_from_spectra,
    prepare_mixing_magnitude_spectrum,
)
from .spatial_consensus import (
    SpatialFieldConsensusSolution,
    solve_spatial_field_consensus,
)
from .spatial_transport import (
    CompactGlobalExposureField,
    CovarianceExposureField,
    SpatialExposureField,
    pullback_compact_global_values,
)


@dataclass(frozen=True)
class MultiCaptureTransportResult:
    image: np.ndarray
    uncertainty: np.ndarray
    predicted_transport_gauge_observations: np.ndarray
    fields: tuple[
        SpatialExposureField
        | CompactGlobalExposureField
        | CovarianceExposureField, ...]
    radiometric_images: tuple[np.ndarray, ...]
    diagnostics: dict[str, object]


def _graph_coordinates(
    count: int,
    edges: Sequence[tuple[int, int]],
    values: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve x_j-x_i=value_ij with exact zero-mean gauge and robust weights."""
    edge_values = np.asarray(values, dtype=np.float64)
    if edge_values.ndim == 1:
        edge_values = edge_values[:, None]
    incidence = np.zeros((len(edges), count), dtype=np.float64)
    for row, (first, second) in enumerate(edges):
        incidence[row, first] = -1.0
        incidence[row, second] = 1.0
    base_weight = np.maximum(np.asarray(weights, dtype=np.float64), 1e-6)
    robust_weight = np.ones(len(edges), dtype=np.float64)
    coordinates = np.zeros((count, edge_values.shape[1]), dtype=np.float64)
    for _ in range(5):
        weight = base_weight * robust_weight
        laplacian = incidence.T @ (weight[:, None] * incidence)
        right = incidence.T @ (weight[:, None] * edge_values)
        system = np.block([
            [laplacian, np.ones((count, 1), dtype=np.float64)],
            [np.ones((1, count), dtype=np.float64), np.zeros((1, 1))],
        ])
        target = np.vstack((right, np.zeros((1, edge_values.shape[1]))))
        coordinates = np.linalg.solve(system, target)[:count]
        residual = incidence @ coordinates - edge_values
        norm = np.sqrt(np.sum(residual * residual, axis=1))
        scale = max(1.4826 * float(np.median(norm)), 1e-6)
        robust_weight = 1.0 / np.sqrt(1.0 + (norm / scale) ** 2)
    residual = incidence @ coordinates - edge_values
    return coordinates, residual


def _minimum_trace_positive_covariances(
    relative_covariances: np.ndarray,
    *,
    direction_count: int = 256,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Fix the common gauge by the positive realization of least total trace."""
    angles = np.linspace(0.0, np.pi, max(int(direction_count), 32), endpoint=False)
    cosine = np.cos(angles)
    sine = np.sin(angles)
    directions = np.stack((cosine, sine), axis=1)
    design = np.stack((cosine * cosine, 2.0 * cosine * sine,
                       sine * sine), axis=1)
    directional_relative = np.einsum(
        "di,nij,dj->nd", directions, relative_covariances, directions)
    required = -np.min(directional_relative, axis=0)
    program = linprog(
        np.asarray((1.0, 0.0, 1.0)),
        A_ub=-design,
        b_ub=-required,
        bounds=((None, None), (None, None), (None, None)),
        method="highs",
    )
    if not program.success:
        raise RuntimeError(f"positive covariance gauge failed: {program.message}")
    coefficient = program.x
    gauge = np.asarray(((coefficient[0], coefficient[1]),
                        (coefficient[1], coefficient[2])))
    covariances = relative_covariances + gauge[None, ...]
    minimum_eigenvalue = min(
        float(np.min(np.linalg.eigvalsh(item))) for item in covariances)
    isotropic_correction = max(-minimum_eigenvalue, 0.0) + 1e-10
    if isotropic_correction > 1e-9:
        gauge = gauge + isotropic_correction * np.eye(2)
        covariances = relative_covariances + gauge[None, ...]
    return covariances, gauge, {
        "positive_gauge_method": (
            "directional_linear_program_minimum_total_trace"),
        "positive_gauge_direction_count": len(angles),
        "positive_gauge_trace": float(np.trace(gauge)),
        "positive_gauge_isotropic_correction": isotropic_correction,
        "minimum_frame_covariance_eigenvalue": float(min(
            np.min(np.linalg.eigvalsh(item)) for item in covariances)),
        "identifiability_boundary": (
            "common_positive_centered_blur_is_a_gauge_minimum_trace_is_not_"
            "evidence_that_any_capture_is_sharp"),
    }


def _spatial_positive_sigma_measure(
    covariance_field: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Realize every 2-D positive covariance as one nine-atom measure."""
    covariance = np.asarray(covariance_field, dtype=np.float64)
    if covariance.ndim != 4 or covariance.shape[-2:] != (2, 2):
        raise ValueError("spatial covariance field must have shape HxWx2x2")
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    extents = np.sqrt(3.0 * eigenvalues)
    coefficients = (-1.0, 0.0, 1.0)
    axis_weights = (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0)
    points = []
    weights = []
    for first_index, first in enumerate(coefficients):
        for second_index, second in enumerate(coefficients):
            displacement = (
                first * extents[..., 0, None] * eigenvectors[..., :, 0]
                + second * extents[..., 1, None] * eigenvectors[..., :, 1]
            )
            points.append(displacement)
            weights.append(np.full(
                covariance.shape[:2],
                axis_weights[first_index] * axis_weights[second_index],
                dtype=np.float64,
            ))
    return np.stack(points, axis=0), np.stack(weights, axis=0)


def estimate_spatial_mixing_covariance_atlas(
    observations: Sequence[np.ndarray],
    *,
    patch_size: int = 48,
    stride: int | None = None,
    quartic_shape: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, object]]:
    """Close local relative-covariance graphs and blend their positive gauges.

    The returned array has shape ``NxHxWx2x2``. Every chart and capture
    contributes through positive spatial mass; there is no winning chart,
    capture, or blur family.
    """
    images = tuple(np.asarray(item, dtype=np.float64) for item in observations)
    if len(images) < 3:
        raise ValueError("spatial mixing atlas needs at least three observations")
    shape = images[0].shape[:2]
    if any(item.shape[:2] != shape for item in images[1:]):
        raise ValueError("spatial mixing atlas observations must share one raster")
    count = len(images)
    edges = tuple(combinations(range(count), 2))
    height, width = shape
    extent = min(max(int(patch_size), 16), height, width)
    step = int(stride or max(8, round(0.5 * extent)))
    half = extent // 2
    pad_after = extent - half
    padded = tuple(np.pad(
        _luminance(item), ((half, pad_after), (half, pad_after)),
        mode="reflect",
    ) for item in images)
    centers_y = list(range(step // 2, height, step))
    centers_x = list(range(step // 2, width, step))
    if not centers_y or centers_y[-1] != height - 1:
        centers_y.append(height - 1)
    if not centers_x or centers_x[-1] != width - 1:
        centers_x.append(width - 1)
    yy, xx = np.mgrid[:height, :width]
    spatial_sigma = max(0.55 * extent, 1.0)
    covariance_numerator = np.zeros((count, height, width, 2, 2))
    side_weight_numerator = (
        np.zeros((count, height, width, 2), dtype=np.float64)
        if quartic_shape else None)
    mass = np.zeros((height, width), dtype=np.float64)
    chart_records = []
    gauge_sum = np.zeros((2, 2), dtype=np.float64)
    gauge_mass = 0.0
    # The full-raster graph is the statistically stronger common anchor. Local
    # charts transport only their continuously trusted deviation from it.
    global_spectra = tuple(
        prepare_mixing_magnitude_spectrum(item) for item in images)
    global_values = []
    global_weights = []
    for first, second in edges:
        estimate = estimate_relative_mixing_from_spectra(
            global_spectra[first], global_spectra[second])
        covariance = estimate.covariance_difference_second_minus_first
        global_values.append((covariance[0, 0], covariance[0, 1],
                              covariance[1, 1]))
        global_weights.append(max(estimate.authority ** 2, 1e-4))
    global_components, global_residual = _graph_coordinates(
        count, edges, np.asarray(global_values), np.asarray(global_weights))
    global_relative = np.empty((count, 2, 2), dtype=np.float64)
    global_relative[:, 0, 0] = global_components[:, 0]
    global_relative[:, 0, 1] = global_components[:, 1]
    global_relative[:, 1, 0] = global_components[:, 1]
    global_relative[:, 1, 1] = global_components[:, 2]
    global_positive, global_gauge, _ = _minimum_trace_positive_covariances(
        global_relative, direction_count=256)
    for center_y in centers_y:
        for center_x in centers_x:
            patches = tuple(
                item[center_y:center_y + extent,
                     center_x:center_x + extent]
                for item in padded)
            spectra = tuple(
                prepare_mixing_magnitude_spectrum(item) for item in patches)
            values = []
            weights = []
            authorities = []
            adaptive_maximum_frequencies = []
            for first, second in edges:
                estimate = estimate_adaptive_relative_mixing_from_spectra(
                    spectra[first], spectra[second])
                covariance = estimate.covariance_difference_second_minus_first
                values.append((covariance[0, 0], covariance[0, 1],
                               covariance[1, 1]))
                weights.append(max(estimate.authority ** 2, 1e-4))
                authorities.append(estimate.authority)
                adaptive_maximum_frequencies.append(float(
                    estimate.diagnostics[
                        "adaptive_maximum_frequency_cycles_per_pixel"]))
            components, graph_residual = _graph_coordinates(
                count, edges, np.asarray(values), np.asarray(weights))
            relative = np.empty((count, 2, 2), dtype=np.float64)
            relative[:, 0, 0] = components[:, 0]
            relative[:, 0, 1] = components[:, 1]
            relative[:, 1, 0] = components[:, 1]
            relative[:, 1, 1] = components[:, 2]
            positive, gauge, gauge_record = (
                _minimum_trace_positive_covariances(
                    relative, direction_count=128))
            authority = float(np.median(authorities))
            graph_residual_rms = float(np.sqrt(np.mean(
                graph_residual * graph_residual)))
            edge_signal_rms = float(np.sqrt(np.mean(
                np.asarray(values, dtype=np.float64) ** 2)))
            graph_closure = float(
                edge_signal_rms ** 2
                / (edge_signal_rms ** 2 + graph_residual_rms ** 2 + 0.03 ** 2))
            local_deviation_authority = float(
                np.clip(authority ** 2 * graph_closure, 0.0, 1.0))
            positive = (
                (1.0 - local_deviation_authority) * global_positive
                + local_deviation_authority * positive)
            gauge = (
                (1.0 - local_deviation_authority) * global_gauge
                + local_deviation_authority * gauge)
            chart_shape_authority = 0.0
            chart_side_weights = np.full((count, 2), 1.0 / 6.0)
            if quartic_shape:
                quartic_maximum_frequency = max(
                    0.12,
                    min(0.30, 2.0 * max(adaptive_maximum_frequencies)),
                )
                shape_estimate = estimate_quartic_shape_transport(
                    patches, positive,
                    maximum_frequency=quartic_maximum_frequency,
                )
                chart_shape_authority = shape_estimate.authority
                chart_side_weights = (
                    1.0 / 6.0
                    + local_deviation_authority
                    * (shape_estimate.side_weights - 1.0 / 6.0))
            chart_mass = 0.05 + 0.95 * authority
            window = chart_mass * np.exp(-0.5 * (
                ((xx - center_x) / spatial_sigma) ** 2
                + ((yy - center_y) / spatial_sigma) ** 2
            ))
            covariance_numerator += (
                positive[:, None, None, :, :] * window[None, ..., None, None])
            if side_weight_numerator is not None:
                side_weight_numerator += (
                    chart_side_weights[:, None, None, :]
                    * window[None, ..., None])
            mass += window
            gauge_sum += chart_mass * gauge
            gauge_mass += chart_mass
            chart_records.append({
                "center_xy": [int(center_x), int(center_y)],
                "median_relative_authority": authority,
                "edge_signal_rms": edge_signal_rms,
                "graph_closure": graph_closure,
                "local_deviation_authority": local_deviation_authority,
                "quartic_shape_authority": chart_shape_authority,
                "axis_side_weight_range": [
                    float(np.min(chart_side_weights)),
                    float(np.max(chart_side_weights)),
                ],
                "adaptive_maximum_frequency_range": [
                    min(adaptive_maximum_frequencies),
                    max(adaptive_maximum_frequencies),
                ],
                "positive_chart_mass": chart_mass,
                "covariance_graph_residual_rms": graph_residual_rms,
                "positive_gauge_trace": gauge_record["positive_gauge_trace"],
            })
    covariance_fields = covariance_numerator / np.maximum(
        mass[None, ..., None, None], np.finfo(float).tiny)
    side_weight_fields = (
        None if side_weight_numerator is None else
        side_weight_numerator / np.maximum(
            mass[None, ..., None], np.finfo(float).tiny))
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(covariance_fields)))
    return covariance_fields, side_weight_fields, {
        "method": "positive_overlapping_local_covariance_graph_atlas",
        "patch_size": extent,
        "stride": step,
        "chart_count": len(chart_records),
        "chart_records": chart_records,
        "minimum_covariance_eigenvalue": minimum_eigenvalue,
        "mean_common_covariance_gauge": (
            gauge_sum / max(gauge_mass, np.finfo(float).tiny)).tolist(),
        "global_anchor_covariances": global_positive.tolist(),
        "global_anchor_graph_residual_rms": float(np.sqrt(np.mean(
            global_residual * global_residual))),
        "local_deviation_authority_range": [
            float(min(item["local_deviation_authority"] for item in chart_records)),
            float(max(item["local_deviation_authority"] for item in chart_records)),
        ],
        "spatial_covariance_trace_range": [
            float(np.min(np.trace(covariance_fields, axis1=-2, axis2=-1))),
            float(np.max(np.trace(covariance_fields, axis1=-2, axis2=-1))),
        ],
        "capture_role": (
            "all_local_charts_and_captures_have_positive_mass_no_selection"),
        "quartic_shape_atlas": bool(quartic_shape),
        "quartic_shape_authority_range": (
            None if not quartic_shape else [
                float(min(item["quartic_shape_authority"]
                          for item in chart_records)),
                float(max(item["quartic_shape_authority"]
                          for item in chart_records)),
            ]),
    }


def deblur_multicapture_consensus(
    observations: Sequence[np.ndarray],
    *,
    passes: int = 64,
    descent_method: str = "optimal_positive_line",
    mixing_patch_size: int | None = None,
    mixing_stride: int | None = None,
    quartic_shape: bool = False,
    full_quartic_shape: bool = False,
) -> MultiCaptureTransportResult:
    """Recover one latent gauge from three or more exchangeable captures."""
    raw = tuple(np.asarray(item, dtype=np.float64) for item in observations)
    if len(raw) < 3:
        raise ValueError("multi-capture consensus needs at least three observations")
    if any(item.shape != raw[0].shape for item in raw[1:]):
        raise ValueError("multi-capture observations must share one raster")
    if any(np.any(~np.isfinite(item)) for item in raw):
        raise ValueError("multi-capture observations must be finite")
    count = len(raw)
    if all(np.array_equal(raw[0], item) for item in raw[1:]):
        zero_flow = np.zeros((*raw[0].shape[:2], 2), dtype=np.float64)
        identity_fields = tuple(SpatialExposureField.from_barycentric_paths(
            name=f"multicapture_{index}_exact_identity",
            barycentric_flow_xy=zero_flow,
            residual_displacements_xy=np.zeros((1, 2), dtype=np.float64),
            weights=np.ones(1, dtype=np.float64),
        ) for index in range(count))
        return MultiCaptureTransportResult(
            image=raw[0].copy(),
            uncertainty=np.zeros_like(raw[0]),
            predicted_transport_gauge_observations=np.stack(raw, axis=0),
            fields=identity_fields,
            radiometric_images=raw,
            diagnostics={
                "method": "exchange_symmetric_multicapture_positive_transport",
                "capture_count": count,
                "pair_count": count * (count - 1) // 2,
                "frame_covariances": np.zeros((count, 2, 2)).tolist(),
                "center_coordinates_xy": np.zeros((count, 2)).tolist(),
                "relative_covariances": np.zeros((count, 2, 2)).tolist(),
                "radiometric_authority": 0.0,
                "passes_used": 0,
                "zero_measure_fast_path": True,
                "minimum_frame_covariance_eigenvalue": 0.0,
                "capture_role": (
                    "all_captures_remain_positive_measures_no_frame_or_family_"
                    "selection"),
            },
        )
    edges = tuple(combinations(range(count), 2))

    gain_values = []
    gain_weights = []
    for first, second in edges:
        log_gain, support = _quantile_log_gain(raw[first], raw[second])
        gain_values.append(float(np.clip(log_gain, -np.log(8.0), np.log(8.0))))
        gain_weights.append(max(float(support) / 38.0, 1e-3))
    log_exposures, gain_residual = _graph_coordinates(
        count, edges, np.asarray(gain_values), np.asarray(gain_weights))
    log_exposures = log_exposures[:, 0]
    radiometric_authority = float(1.0 - np.exp(-(
        np.std(log_exposures) / 0.12) ** 4))
    normalized = tuple(
        (1.0 - radiometric_authority) * image
        + radiometric_authority * image / np.exp(log_exposures[index])
        for index, image in enumerate(raw)
    )
    raw_precision = np.stack([_soft_sensor_precision(item) for item in raw])
    sensor_precision = (
        (1.0 - radiometric_authority) * np.ones_like(raw_precision)
        + radiometric_authority * raw_precision)

    translations = []
    translation_weights = []
    translation_records = []
    phase_spectra = tuple(
        prepare_phase_circle_spectrum(_luminance(item)) for item in normalized)
    for first, second in edges:
        vector, record = phase_circle_translation_from_spectra(
            phase_spectra[first], phase_spectra[second])
        ambiguity = float(record["weighted_peak_ambiguity"])
        dispersion = float(record["translation_dispersion_pixels"])
        scale = 0.5 + float(np.linalg.norm(vector))
        weight = max(
            (1.0 - ambiguity) ** 2 * np.exp(-(dispersion / scale) ** 2),
            1e-4,
        )
        translations.append(vector)
        translation_weights.append(weight)
        translation_records.append({
            "edge": [first, second],
            "translation_xy": vector.tolist(),
            "weight": weight,
            "dispersion_pixels": dispersion,
            "ambiguity": ambiguity,
        })
    center_coordinates, center_residual = _graph_coordinates(
        count, edges, np.stack(translations), np.asarray(translation_weights))
    # The graph is permutation-equivariant analytically. Canonicalize
    # sub-picometer floating summation differences before interpolation so a
    # downstream local spectrum cannot amplify edge-order roundoff.
    center_coordinates = np.round(
        center_coordinates - np.mean(center_coordinates, axis=0, keepdims=True),
        decimals=12,
    )

    # Deterministic transport is analytically removed before any local mixing
    # evidence is measured. Global Fourier magnitudes are translation
    # invariant, but finite local charts are not; this ordering prevents
    # residual edge motion from being misread as a wider positive measure.
    centered_for_mixing = []
    for index, image in enumerate(normalized):
        center_field = CompactGlobalExposureField(
            name=f"multicapture_{index}_estimation_center_chart",
            raster_shape=raw[0].shape[:2],
            residual_displacements_xy=np.zeros((1, 2), dtype=np.float64),
            residual_weights=np.ones(1, dtype=np.float64),
            barycentric_translation_xy=center_coordinates[index],
        )
        centered_image, _ = pullback_compact_global_values(
            image, center_field)
        centered_for_mixing.append(centered_image)
    mixing_images = (
        normalized
        if mixing_patch_size is None
        else tuple(centered_for_mixing))

    covariance_records = []
    covariance_atlas_record: dict[str, object] | None = None
    if mixing_patch_size is None:
        covariance_values = []
        covariance_weights = []
        mixing_spectra = tuple(
            prepare_mixing_magnitude_spectrum(item) for item in mixing_images)
        for first, second in edges:
            estimate = estimate_relative_mixing_from_spectra(
                mixing_spectra[first], mixing_spectra[second])
            covariance = estimate.covariance_difference_second_minus_first
            covariance_values.append((covariance[0, 0], covariance[0, 1],
                                      covariance[1, 1]))
            covariance_weights.append(max(estimate.authority ** 2, 1e-4))
            covariance_records.append({
                "edge": [first, second],
                "covariance_difference": covariance.tolist(),
                "authority": estimate.authority,
                "fit_coherence": estimate.diagnostics["fit_coherence"],
            })
        relative_components, covariance_graph_residual = _graph_coordinates(
            count, edges, np.asarray(covariance_values),
            np.asarray(covariance_weights))
        relative_covariances = np.empty((count, 2, 2), dtype=np.float64)
        relative_covariances[:, 0, 0] = relative_components[:, 0]
        relative_covariances[:, 0, 1] = relative_components[:, 1]
        relative_covariances[:, 1, 0] = relative_components[:, 1]
        relative_covariances[:, 1, 1] = relative_components[:, 2]
        covariances, common_gauge, gauge_record = (
            _minimum_trace_positive_covariances(relative_covariances))
        covariance_fields = None
        side_weight_fields = None
    else:
        covariance_fields, side_weight_fields, covariance_atlas_record = (
            estimate_spatial_mixing_covariance_atlas(
                mixing_images,
                patch_size=mixing_patch_size,
                stride=mixing_stride,
                quartic_shape=quartic_shape,
            ))
        covariances = np.mean(covariance_fields, axis=(1, 2))
        relative_covariances = covariances - np.mean(
            covariances, axis=0, keepdims=True)
        common_gauge = np.asarray(
            covariance_atlas_record["mean_common_covariance_gauge"])
        covariance_graph_residual = np.asarray([
            item["covariance_graph_residual_rms"]
            for item in covariance_atlas_record["chart_records"]
        ])
        gauge_record = {
            "positive_gauge_method": (
                "overlapping_chart_directional_linear_programs"),
            "positive_gauge_direction_count": 128,
            "positive_gauge_trace": float(np.trace(common_gauge)),
            "positive_gauge_isotropic_correction": 0.0,
            "minimum_frame_covariance_eigenvalue": float(np.min(
                np.linalg.eigvalsh(covariance_fields))),
            "identifiability_boundary": (
                "chart_local_common_positive_blur_remains_a_spatial_gauge"),
        }

    shape_estimate = None
    full_shape_estimate = None
    if full_quartic_shape and covariance_fields is None:
        full_shape_estimate = estimate_full_quartic_transport(
            mixing_images, covariances)
    elif quartic_shape and covariance_fields is None:
        shape_estimate = estimate_quartic_shape_transport(
            mixing_images, covariances)
    spatial_covariance_mode = covariance_fields is not None
    full_shape_active = bool(
        full_shape_estimate is not None
        and np.max(np.abs(full_shape_estimate.standardized_cumulants)) > 1e-10
    )
    fields = []
    for index in range(count):
        if covariance_fields is None:
            if full_shape_active:
                points = full_shape_estimate.residual_displacements[index]
                weights = full_shape_estimate.residual_weights[index]
            elif shape_estimate is None:
                points, weights = _positive_sigma_measure(covariances[index])
            else:
                points = shape_estimate.residual_displacements[index]
                weights = shape_estimate.residual_weights[index]
            fields.append(CompactGlobalExposureField(
                name=f"multicapture_{index}_center_plus_positive_mixing",
                raster_shape=raw[0].shape[:2],
                residual_displacements_xy=points,
                residual_weights=weights,
                barycentric_translation_xy=center_coordinates[index],
            ))
        else:
            flow = np.broadcast_to(
                center_coordinates[index][None, None, :],
                (*raw[0].shape[:2], 2),
            )
            fields.append(CovarianceExposureField(
                name=f"multicapture_{index}_center_plus_covariance_atlas",
                barycentric_flow_xy=flow,
                covariance_components=covariance_fields[index],
                axis_side_weights=(
                    side_weight_fields[index]
                    if side_weight_fields is not None else None),
            ))
    if spatial_covariance_mode:
        # CovarianceExposureField has compacted each symmetric HxWx2x2 matrix
        # into HxWx3 storage. Release the temporary atlas tensor before the
        # iterative solver creates its pulled covariance operators.
        covariance_fields = None
    solution: SpatialFieldConsensusSolution = solve_spatial_field_consensus(
        normalized,
        tuple(fields),
        frame_weights=sensor_precision / float(count),
        passes=passes,
        descent_method=descent_method,
    )
    diagnostics = {
        **solution.diagnostics,
        "method": "exchange_symmetric_multicapture_positive_transport",
        "capture_count": count,
        "pair_count": len(edges),
        "log_exposure_coordinates": log_exposures.tolist(),
        "radiometric_authority": radiometric_authority,
        "radiometric_graph_residual_rms": float(np.sqrt(np.mean(
            gain_residual * gain_residual))),
        "center_coordinates_xy": center_coordinates.tolist(),
        "center_graph_residual_rms": float(np.sqrt(np.mean(
            center_residual * center_residual))),
        "mixing_estimation_chart": (
            "radiometric_then_deterministic_center_pullback_before_local_mixing"
            if mixing_patch_size is not None else
            "radiometric_then_global_fourier_translation_quotient_after_center_"
            "estimation"),
        "translation_edges": translation_records,
        "relative_covariances": relative_covariances.tolist(),
        "frame_covariances": covariances.tolist(),
        "common_covariance_gauge": common_gauge.tolist(),
        "covariance_graph_residual_rms": float(np.sqrt(np.mean(
            covariance_graph_residual * covariance_graph_residual))),
        "covariance_edges": covariance_records,
        "spatial_mixing_atlas": covariance_atlas_record,
        "quartic_shape_transport": (
            None if shape_estimate is None else shape_estimate.diagnostics),
        "full_quartic_transport": (
            None if full_shape_estimate is None
            else full_shape_estimate.diagnostics),
        "full_quartic_spatial_status": (
            "global_positive_directional_measure_applied"
            if full_shape_active else
            "exact_zero_tensor_covariance_identity_fast_path"
            if full_shape_estimate is not None else
            "deferred_in_spatial_atlas_mode"
            if full_quartic_shape and covariance_atlas_record is not None else
            "not_requested"),
        "shared_spectral_preparation": {
            "phase_fft_count": count,
            "mixing_fft_count": (
                count if covariance_atlas_record is None
                else count * (1 + int(
                    covariance_atlas_record["chart_count"]))),
            "total_observation_fft_count": (
                2 * count if covariance_atlas_record is None
                else count * (2 + int(
                    covariance_atlas_record["chart_count"]))),
            "former_pair_recomputed_fft_count": (
                4 * len(edges) if covariance_atlas_record is None
                else 2 * len(edges) * (
                    2 + int(covariance_atlas_record["chart_count"]))),
        },
        **gauge_record,
        "capture_role": (
            "all_captures_remain_positive_measures_no_frame_or_family_selection"),
    }
    return MultiCaptureTransportResult(
        image=solution.image,
        uncertainty=solution.uncertainty,
        predicted_transport_gauge_observations=(
            solution.predicted_transport_gauge_observations),
        fields=tuple(fields),
        radiometric_images=normalized,
        diagnostics=diagnostics,
    )
