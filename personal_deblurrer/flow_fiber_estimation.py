"""Blind positive flow-atlas estimation without sheet classification.

Dense local transport and global/local Fourier-circle phase transport supply
complementary connections. A tensor quadrature over displacement scale and
atlas coordinate carries distinct latent appearances and a soft
cross-predictive ownership density. Continuous coherence/disagreement
authority reconciles it with the dense common gauge. Quadrature resolution is
numerical support for one measure, not a proposed number of scene layers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from .circles import phase_circle_flow, phase_circle_translation
from .dense_estimation import (
    DensePairEstimate,
    _luminance,
    _sample,
    _transport_connection_confidence,
    estimate_dense_pair_exposure,
)
from .multisheet_transport import (
    MultiSheetConsensusSolution,
    solve_multisheet_consensus,
)
from .radiometric_transport import transport_radiometric_pair
from .relative_mixing_transport import estimate_relative_mixing_transport
from .spatial_consensus import solve_spatial_field_consensus
from .spatial_transport import SpatialExposureField, SpatialReflectedExposureOperator


@dataclass(frozen=True)
class FlowFiberConsensusResult:
    image: np.ndarray
    common_image: np.ndarray
    uncertainty: np.ndarray
    estimate: DensePairEstimate
    fiber_solution: MultiSheetConsensusSolution | None
    diagnostics: dict[str, object]


def _flow_fiber_fields(
    relative_flow_xy: np.ndarray,
    authority: np.ndarray,
    support: np.ndarray,
    *,
    duty_cycle: float,
    atoms: int,
) -> tuple[tuple[SpatialExposureField, ...], tuple[SpatialExposureField, ...]]:
    trusted = authority[..., None] * np.asarray(relative_flow_xy, dtype=np.float64)
    shape = trusted.shape[:2]
    atom_count = max(int(atoms), 1)
    times = np.linspace(-0.5, 0.5, atom_count, dtype=np.float64)
    mass = np.ones((atom_count, *shape), dtype=np.float64)
    frames = []
    for frame, sign in enumerate((-1.0, 1.0)):
        frames.append(tuple(
            SpatialExposureField.from_barycentric_paths(
                name=f"flow_fiber_frame_{frame}_lambda_{value:.6f}",
                barycentric_flow_xy=sign * 0.5 * float(value) * trusted,
                residual_displacements_xy=(
                    times[:, None, None, None]
                    * max(float(duty_cycle), 0.0)
                    * float(value)
                    * trusted[None, ...]
                ),
                weights=mass,
            )
            for value in support
        ))
    return frames[0], frames[1]


def _flow_atlas_fields(
    support_flows_xy: np.ndarray,
    *,
    duty_cycle: float,
    atoms: int,
    residual_measures: tuple[
        tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]
    ] | None = None,
) -> tuple[tuple[SpatialExposureField, ...], tuple[SpatialExposureField, ...]]:
    """Lift arbitrary flow quadrature points into exchangeable positive fields."""
    flows = np.asarray(support_flows_xy, dtype=np.float64)
    if flows.ndim != 4 or flows.shape[-1] != 2:
        raise ValueError("flow-atlas support must have shape SxHxWx2")
    atom_count = max(int(atoms), 1)
    times = np.linspace(-0.5, 0.5, atom_count, dtype=np.float64)
    frames = []
    for frame, sign in enumerate((-1.0, 1.0)):
        frame_fields = []
        if residual_measures is None:
            mixing_points = np.zeros((1, 2), dtype=np.float64)
            mixing_weights = np.ones(1, dtype=np.float64)
        else:
            mixing_points, mixing_weights = residual_measures[frame]
        for sheet, flow in enumerate(flows):
            velocity_points = (
                times[:, None, None, None]
                * max(float(duty_cycle), 0.0)
                * flow[None, ...]
            )
            residual = (
                velocity_points[:, None, ...]
                + mixing_points[None, :, None, None, :]
            ).reshape(-1, *flows.shape[1:3], 2)
            weights = (
                np.ones(atom_count, dtype=np.float64)[:, None]
                * mixing_weights[None, :]
            ).reshape(-1)
            frame_fields.append(SpatialExposureField.from_barycentric_paths(
                name=f"flow_atlas_frame_{frame}_support_{sheet}",
                barycentric_flow_xy=sign * 0.5 * flow,
                residual_displacements_xy=residual,
                weights=weights,
            ))
        frames.append(tuple(frame_fields))
    return frames[0], frames[1]


def _compose_relative_mixing(
    fields: tuple[SpatialExposureField, SpatialExposureField],
    residual_measures: tuple[
        tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]
    ],
) -> tuple[SpatialExposureField, SpatialExposureField]:
    """Convolve spatial exposure fields with frame-relative positive measures."""
    composed = []
    for frame, field in enumerate(fields):
        points, mass = residual_measures[frame]
        centered = field.centered_displacements_xy
        residual = (
            centered[:, None, ...]
            + points[None, :, None, None, :]
        ).reshape(-1, *field.shape, 2)
        weights = (
            field.weights[:, None, ...] * mass[None, :, None, None]
        ).reshape(-1, *field.shape)
        composed.append(SpatialExposureField.from_barycentric_paths(
            name=f"{field.name}_plus_relative_centered_mixing",
            barycentric_flow_xy=field.barycentric_flow_xy,
            residual_displacements_xy=residual,
            weights=weights,
        ))
    return composed[0], composed[1]


def _backprojected_closure_measure(
    observations: tuple[np.ndarray, np.ndarray],
    fields: tuple[tuple[SpatialExposureField, ...], ...],
    *,
    entropy_floor: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Build a soft latent measure from matched-adjoint pair closure."""
    sheet_count = len(fields[0])
    image_ndim = observations[0].ndim
    closures = []
    backprojections = []
    operators = []
    for sheet in range(sheet_count):
        sheet_operators = tuple(
            SpatialReflectedExposureOperator(fields[frame][sheet])
            for frame in range(2)
        )
        operators.append(sheet_operators)
        pulled = []
        for frame in range(2):
            normalization = sheet_operators[frame].adjoint(
                np.ones_like(observations[frame]))
            pulled.append(
                sheet_operators[frame].adjoint(observations[frame])
                / np.maximum(normalization, 1e-8)
            )
        backprojections.append(pulled)
        difference = pulled[0] - pulled[1]
        if image_ndim == 3:
            difference = np.mean(difference * difference, axis=2)
        else:
            difference = difference * difference
        closures.append(gaussian_filter(difference, 0.8, mode="reflect"))
    closure = np.stack(closures, axis=0)
    best = np.min(closure, axis=0)
    gap = closure - best[None, ...]
    positive_gap = gap[gap > np.finfo(float).eps]
    temperature = max(
        float(np.quantile(positive_gap, 0.35)) if positive_gap.size else 0.0,
        1e-6,
    )
    likelihood = np.exp(-gap / temperature)
    floor = np.clip(float(entropy_floor), 0.0, 0.5)
    latent_measure = likelihood + floor
    latent_measure /= np.sum(latent_measure, axis=0, keepdims=True)

    sensor_measure = np.empty((2, sheet_count, *closure.shape[1:]), dtype=np.float64)
    for frame in range(2):
        for sheet in range(sheet_count):
            sensor_measure[frame, sheet] = np.maximum(
                operators[sheet][frame].forward(latent_measure[sheet]), 0.0)
        sensor_measure[frame] /= np.maximum(
            np.sum(sensor_measure[frame], axis=0, keepdims=True), 1e-8)
    entropy = -np.sum(
        latent_measure * np.log(np.maximum(
            latent_measure, np.finfo(float).tiny)),
        axis=0,
    ) / np.log(float(sheet_count))
    return latent_measure, sensor_measure, {
        "closure_temperature": temperature,
        "closure_error_min_mean": float(np.mean(best)),
        "closure_error_max_mean": float(np.mean(np.max(closure, axis=0))),
        "latent_measure_entropy_mean": float(np.mean(entropy)),
        "latent_measure_entropy_min": float(np.min(entropy)),
        "entropy_floor": floor,
        "measure_construction": (
            "continuous_matched_adjoint_closure_density_on_flow_fiber"),
    }


def _transport_latent_measure_to_sensors(
    latent_measure: np.ndarray,
    fields: tuple[tuple[SpatialExposureField, ...], ...],
) -> np.ndarray:
    sensor_measure = np.empty(
        (len(fields), *latent_measure.shape), dtype=np.float64)
    for frame, frame_fields in enumerate(fields):
        for sheet, field in enumerate(frame_fields):
            sensor_measure[frame, sheet] = np.maximum(
                SpatialReflectedExposureOperator(field).forward(
                    latent_measure[sheet]),
                0.0,
            )
        sensor_measure[frame] /= np.maximum(
            np.sum(sensor_measure[frame], axis=0, keepdims=True), 1e-8)
    return sensor_measure


def _motion_coordinate_measure(
    relative_flow_xy: np.ndarray,
    authority: np.ndarray,
    support: np.ndarray,
    fields: tuple[tuple[SpatialExposureField, ...], ...],
    *,
    entropy_floor: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Lift local flow magnitude into a continuous density on its scale fiber."""
    trusted = authority[..., None] * np.asarray(relative_flow_xy, dtype=np.float64)
    magnitude = np.sqrt(np.sum(trusted * trusted, axis=2))
    positive = magnitude[magnitude > 64.0 * np.finfo(float).eps]
    scale = max(
        float(np.quantile(positive, 0.90)) if positive.size else 0.0,
        1e-6,
    )
    coordinate = np.clip(magnitude / scale, 0.0, 1.0)
    bandwidth = max(0.20 / max(len(support) - 1, 1), 0.03)
    density = np.exp(
        -0.5 * ((support[:, None, None] - coordinate[None, ...])
                / bandwidth) ** 2
    )
    # Lobatto endpoint weights are half cells; they make support refinement a
    # quadrature change rather than a preference for semantic endpoints.
    quadrature = np.ones(len(support), dtype=np.float64)
    quadrature[[0, -1]] = 0.5
    density *= quadrature[:, None, None]
    floor = np.clip(float(entropy_floor), 0.0, 0.5)
    density += floor * quadrature[:, None, None]
    latent_measure = density / np.sum(density, axis=0, keepdims=True)
    sensor_measure = _transport_latent_measure_to_sensors(
        latent_measure, fields)
    entropy = -np.sum(
        latent_measure * np.log(np.maximum(
            latent_measure, np.finfo(float).tiny)),
        axis=0,
    ) / np.log(float(len(support)))
    return latent_measure, sensor_measure, {
        "motion_coordinate_scale_pixels": scale,
        "motion_coordinate_mean": float(np.mean(coordinate)),
        "motion_coordinate_q90": float(np.quantile(coordinate, 0.90)),
        "latent_measure_entropy_mean": float(np.mean(entropy)),
        "latent_measure_entropy_min": float(np.min(entropy)),
        "entropy_floor": floor,
        "fiber_bandwidth": bandwidth,
        "measure_construction": (
            "continuous_motion_coordinate_density_on_flow_fiber"),
    }


def _fourier_circle_cross_predictive_measure(
    observations: tuple[np.ndarray, np.ndarray],
    support_flows_xy: np.ndarray,
    support_prior: np.ndarray,
    fields: tuple[tuple[SpatialExposureField, ...], ...],
    operators: tuple[
        tuple[SpatialReflectedExposureOperator, ...], ...
    ],
    sensor_precision: np.ndarray,
    *,
    entropy_floor: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Cross-predict both frames along one Fourier-circle transport atlas."""
    first = _luminance(observations[0])
    second = _luminance(observations[1])
    sensor_likelihood = []
    residual_records = []
    for frame, (reference, moving, sign, moving_frame) in enumerate((
        (first, second, 1.0, 1),
        (second, first, -1.0, 0),
    )):
        residuals = []
        for flow in support_flows_xy:
            prediction = _sample(
                moving, sign * flow)
            transported_precision = _sample(
                sensor_precision[moving_frame], sign * flow)
            joint_precision = np.sqrt(np.maximum(
                sensor_precision[frame] * transported_precision, 0.0))
            residual = gaussian_filter(
                joint_precision * (prediction - reference) ** 2,
                0.8,
                mode="reflect",
            )
            residuals.append(residual)
        residual_stack = np.stack(residuals, axis=0)
        best = np.min(residual_stack, axis=0)
        gap = residual_stack - best[None, ...]
        positive_gap = gap[gap > np.finfo(float).eps]
        temperature = max(
            float(np.quantile(positive_gap, 0.35))
            if positive_gap.size else 0.0,
            1e-6,
        )
        likelihood = (
            np.exp(-gap / temperature)
            * support_prior[:, None, None]
        )
        likelihood += np.clip(float(entropy_floor), 0.0, 0.5)
        likelihood /= np.sum(likelihood, axis=0, keepdims=True)
        sensor_likelihood.append(likelihood)
        residual_records.append({
            "frame": frame,
            "temperature": temperature,
            "minimum_residual_mean": float(np.mean(best)),
            "maximum_residual_mean": float(np.mean(
                np.max(residual_stack, axis=0))),
        })
    sensor_measure = np.stack(sensor_likelihood, axis=0)

    latent_evidence = np.zeros_like(sensor_measure[0])
    for frame in range(2):
        for sheet, field in enumerate(fields[frame]):
            operator = operators[frame][sheet]
            normalization = operator.adjoint(
                sensor_precision[frame])
            latent_evidence[sheet] += (
                operator.adjoint(
                    sensor_measure[frame, sheet] * sensor_precision[frame])
                / np.maximum(normalization, 1e-8))
    latent_measure = latent_evidence / np.maximum(
        np.sum(latent_evidence, axis=0, keepdims=True), 1e-8)
    entropy = -np.sum(
        latent_measure * np.log(np.maximum(
            latent_measure, np.finfo(float).tiny)),
        axis=0,
    ) / np.log(float(len(support_flows_xy)))
    certainty = 1.0 - entropy
    return latent_measure, sensor_measure, {
        "latent_measure_entropy_mean": float(np.mean(entropy)),
        "latent_measure_entropy_min": float(np.min(entropy)),
        "cross_predictive_certainty_mean": float(np.mean(certainty)),
        "cross_predictive_certainty_max": float(np.max(certainty)),
        "cross_predictive_certainty": certainty,
        "cross_predictive_residual_records": residual_records,
        "entropy_floor": float(entropy_floor),
        "measure_construction": (
            "forward_reverse_cross_prediction_on_fourier_circle_flow_atlas"),
    }


def deblur_flow_fiber_consensus(
    first: np.ndarray,
    second: np.ndarray,
    *,
    support_count: int = 3,
    entropy_floor: float = 0.001,
    measure_method: str = "fourier_circle_cross_predictive",
    duty_cycle: float = 0.0,
    atoms: int = 7,
    fiber_passes: int = 1,
    passes: int = 64,
    pyramid_levels: int = 3,
    warp_iterations: int = 5,
    cg_iterations: int = 60,
    smoothness: float = 0.12,
    automatic_relative_mixing: bool = False,
) -> FlowFiberConsensusResult:
    """Estimate and invert one continuous positive measure over flow support."""
    raw_images = (
        np.asarray(first, dtype=np.float64),
        np.asarray(second, dtype=np.float64),
    )
    if raw_images[0].shape != raw_images[1].shape:
        raise ValueError("flow-atlas observations must share one raster")
    radiometric = transport_radiometric_pair(*raw_images)
    images = radiometric.images
    sensor_precision = radiometric.precision
    relative_mixing = (
        estimate_relative_mixing_transport(*images)
        if automatic_relative_mixing else None
    )
    relative_mixing_diagnostics = (
        relative_mixing.diagnostics
        if relative_mixing is not None else {
            "relative_mixing_method": "disabled_exact_zero_measure",
            "relative_mixing_authority": 0.0,
        }
    )
    estimate = estimate_dense_pair_exposure(
        images[0], images[1],
        duty_cycle=duty_cycle,
        atoms=atoms,
        pyramid_levels=pyramid_levels,
        warp_iterations=warp_iterations,
        cg_iterations=cg_iterations,
        smoothness=smoothness,
    )
    mixing_observable = bool(
        relative_mixing is not None and relative_mixing.authority > 1e-8)
    if not estimate.relative_motion_observable and not mixing_observable:
        precision_for_image = (
            sensor_precision
            if images[0].ndim == 2 else sensor_precision[..., None])
        image_stack = np.stack(images, axis=0)
        mass = np.sum(precision_for_image, axis=0)
        mean = np.sum(
            precision_for_image * image_stack, axis=0) / np.maximum(mass, 1e-8)
        uncertainty = np.sqrt(np.sum(
            precision_for_image * (image_stack - mean) ** 2,
            axis=0,
        ) / np.maximum(mass, 1e-8))
        return FlowFiberConsensusResult(
            image=mean,
            common_image=mean,
            uncertainty=uncertainty,
            estimate=estimate,
            fiber_solution=None,
            diagnostics={
                **estimate.diagnostics,
                **radiometric.diagnostics,
                **relative_mixing_diagnostics,
                "method": "continuous_positive_flow_fiber_consensus",
                "estimation_decision": "common_connection_gauge_abstention",
                "support_count": 0,
                "passes_used": 0,
            },
        )

    count = max(int(support_count), 2)
    # Chebyshev-Lobatto support resolves fiber endpoints without privileging
    # them as semantic layers.
    angle = np.linspace(np.pi, 0.0, count)
    support = 0.5 * (1.0 + np.cos(angle))
    if measure_method not in (
        "fourier_circle_cross_predictive",
        "motion_coordinate",
        "matched_adjoint_closure",
    ):
        raise ValueError(
            "unknown flow-atlas measure method")
    residual_measures = (
        tuple(zip(
            relative_mixing.residual_displacements,
            relative_mixing.residual_weights,
        ))
        if relative_mixing is not None else None
    )
    common_fields = (
        _compose_relative_mixing(estimate.fields, residual_measures)
        if residual_measures is not None else estimate.fields
    )
    common = solve_spatial_field_consensus(
        images,
        common_fields,
        frame_weights=0.5 * sensor_precision,
        passes=passes,
    )
    minimum_jacobian = np.minimum(
        estimate.fields[0].sensor_to_latent_jacobian_determinant,
        estimate.fields[1].sensor_to_latent_jacobian_determinant,
    )
    # The positive part of the fold defect is zero throughout an injective
    # chart and grows continuously once the one-sheet map ceases to exist.
    jacobian_pressure = np.clip(
        np.maximum(-minimum_jacobian, 0.0) / 0.25, 0.0, 1.0)
    fold_fractions = [
        field.diagnostics()["fold_fraction"] for field in estimate.fields]
    if (
        not np.any(jacobian_pressure > 0.0)
        and measure_method != "fourier_circle_cross_predictive"
    ):
        return FlowFiberConsensusResult(
            image=common.image,
            common_image=common.image,
            uncertainty=common.uncertainty,
            estimate=estimate,
            fiber_solution=None,
            diagnostics={
                **estimate.diagnostics,
                **common.diagnostics,
                **radiometric.diagnostics,
                **relative_mixing_diagnostics,
                "method": "continuous_positive_flow_fiber_consensus",
                "flow_support": support.tolist(),
                "support_count": count,
                "fiber_passes": 0,
                "fold_fractions": fold_fractions,
                "execution_chart": "injective_common_gauge_zero_fiber_measure",
                "support_role": (
                    "quadrature_of_one_continuous_measure_not_a_layer_catalog"),
                "correction_authority_mean": 0.0,
                "correction_authority_max": 0.0,
                "jacobian_pressure_mean": 0.0,
                "jacobian_pressure_max": 0.0,
                "jacobian_pressure_transport": {
                    "iterations": 0,
                    "status": 0,
                    "measured_mean": 0.0,
                    "transported_mean": 0.0,
                    "transported_min": 0.0,
                    "transported_max": 0.0,
                },
                "correction_authority_role": (
                    "exact_zero_positive_fold_defect_elides_null_correction"),
                "common_gauge_method": common.diagnostics["method"],
                "latent_measure_entropy_mean": 0.0,
                "estimation_decision": "injective_single_sheet_chart_exact",
                "zero_measure_fast_path": True,
            },
        )
    circle_translation, circle_record = phase_circle_translation(
        _luminance(images[0]), _luminance(images[1]))
    global_circle_flow = np.broadcast_to(
        circle_translation[None, None, :],
        estimate.forward_sampling_flow_xy.shape,
    ).copy()
    use_circle_fiber = measure_method == "fourier_circle_cross_predictive"
    circle_atlas_flow, circle_atlas_record = phase_circle_flow(
        _luminance(images[0]), _luminance(images[1]))
    circle_atlas_diagnostics = {
        key: value for key, value in circle_atlas_record.items()
        if not isinstance(value, np.ndarray)
    }
    circle_flow = (
        circle_atlas_flow if use_circle_fiber else global_circle_flow)
    atlas_center = None
    global_reliability = None
    atlas_reliability = None
    if use_circle_fiber:
        geometry_support = np.asarray((0.0, 1.0), dtype=np.float64)
        geometry_quadrature = np.asarray((0.5, 0.5), dtype=np.float64)
        global_magnitude_for_prior = float(np.linalg.norm(circle_translation))
        atlas_rms_for_prior = float(circle_atlas_record["flow_rms_pixels"])
        magnitude_total = (
            global_magnitude_for_prior ** 8 + atlas_rms_for_prior ** 8)
        global_mass_share = (
            global_magnitude_for_prior ** 8
            / max(magnitude_total, np.finfo(float).tiny)
        )
        global_dispersion_for_prior = (
            float(circle_record["translation_dispersion_pixels"])
            / (0.25 + global_magnitude_for_prior)
        )
        atlas_dispersion_for_prior = float(np.mean(
            circle_atlas_record["spectral_dispersion_field"]
            / (0.25 + np.sqrt(np.sum(
                circle_atlas_flow * circle_atlas_flow, axis=2)))
        ))
        global_reliability = global_mass_share * np.exp(-(
            global_dispersion_for_prior / 0.22) ** 4)
        atlas_reliability = (1.0 - global_mass_share) * np.exp(-(
            atlas_dispersion_for_prior / 0.80) ** 4)
        atlas_center = (
            atlas_reliability
            / max(global_reliability + atlas_reliability, 1e-8)
        )
        geometry_density = (
            np.exp(-0.5 * ((geometry_support - atlas_center) / 0.28) ** 2)
            + 0.002
        ) * geometry_quadrature
        geometry_density /= np.sum(geometry_density)
        scale_quadrature = np.ones(len(support), dtype=np.float64)
        support_flows = [np.zeros_like(circle_flow)]
        support_coordinates = [{"scale": 0.0, "atlas_coordinate": 0.5}]
        support_prior = [1.0]
        for scale, scale_mass in zip(
            support[1:], scale_quadrature[1:],
        ):
            for coordinate, geometry_mass in zip(
                geometry_support, geometry_density,
            ):
                support_flows.append(float(scale) * (
                    (1.0 - float(coordinate)) * global_circle_flow
                    + float(coordinate) * circle_atlas_flow
                ))
                support_coordinates.append({
                    "scale": float(scale),
                    "atlas_coordinate": float(coordinate),
                })
                support_prior.append(float(scale_mass * geometry_mass))
        support_flow_array = np.stack(support_flows, axis=0)
        support_prior_array = np.asarray(support_prior, dtype=np.float64)
        support_prior_array /= np.sum(support_prior_array)
        fields = _flow_atlas_fields(
            support_flow_array,
            duty_cycle=duty_cycle,
            atoms=atoms,
            residual_measures=residual_measures,
        )
    else:
        support_flow_array = (
            support[:, None, None, None]
            * estimate.forward_sampling_flow_xy[None, ...]
            * estimate.confidence[None, ..., None]
        )
        support_coordinates = [
            {"scale": float(value), "atlas_coordinate": 0.0}
            for value in support
        ]
        support_prior_array = np.ones(len(support), dtype=np.float64)
        support_prior_array[[0, -1]] = 0.5
        support_prior_array /= np.sum(support_prior_array)
        fields = _flow_fiber_fields(
            estimate.forward_sampling_flow_xy,
            estimate.confidence,
            support,
            duty_cycle=duty_cycle,
            atoms=atoms,
        )
    if measure_method == "fourier_circle_cross_predictive":
        operator_plans = tuple(tuple(
            SpatialReflectedExposureOperator(field)
            for field in frame_fields
        ) for frame_fields in fields)
        latent_measure, sensor_measure, measure_record = (
            _fourier_circle_cross_predictive_measure(
                images,
                support_flow_array,
                support_prior_array,
                fields,
                operator_plans,
                sensor_precision,
                entropy_floor=entropy_floor,
            )
        )
    elif measure_method == "motion_coordinate":
        operator_plans = None
        latent_measure, sensor_measure, measure_record = (
            _motion_coordinate_measure(
                estimate.forward_sampling_flow_xy,
                estimate.confidence,
                support,
                fields,
                entropy_floor=entropy_floor,
            )
        )
    elif measure_method == "matched_adjoint_closure":
        operator_plans = None
        latent_measure, sensor_measure, measure_record = (
            _backprojected_closure_measure(
                images,
                fields,
                entropy_floor=entropy_floor,
            )
        )
    else:
        raise AssertionError("validated flow-atlas measure method")
    solution = solve_multisheet_consensus(
        images,
        fields,
        operator_plans=operator_plans,
        sensor_ownership=sensor_measure,
        reference_ownership=latent_measure,
        frame_weights=sensor_precision,
        appearance_coupling=1.0,
        passes=min(max(int(fiber_passes), 1), max(int(passes), 1)),
    )
    cycle_scale = max(
        float(np.quantile(estimate.cycle_error_pixels, 0.75)), 0.25)
    photo_scale = max(
        float(np.quantile(estimate.photometric_error, 0.75)), 0.02)
    cycle_ratio = estimate.cycle_error_pixels / cycle_scale
    photo_ratio = estimate.photometric_error / photo_scale
    measured_nonclosure = (
        cycle_ratio * cycle_ratio / (1.0 + cycle_ratio * cycle_ratio)
        * photo_ratio * photo_ratio / (1.0 + photo_ratio * photo_ratio)
        * estimate.confidence
    )
    transported_nonclosure, pressure_transport = (
        _transport_connection_confidence(
            measured_nonclosure,
            jacobian_pressure,
            0.5 * (_luminance(images[0]) + _luminance(images[1])),
        )
    )
    correction_authority = np.sqrt(
        np.maximum(measured_nonclosure, 0.0)
        * np.maximum(transported_nonclosure, 0.0))
    fold_authority = np.clip(
        gaussian_filter(correction_authority, 0.8, mode="reflect"),
        0.0,
        0.5,
    )
    global_magnitude = float(np.linalg.norm(circle_translation))
    atlas_magnitude_field = np.sqrt(np.sum(
        circle_atlas_flow * circle_atlas_flow, axis=2))
    global_disagreement = float(np.mean(
        np.sqrt(np.sum(
            (estimate.forward_sampling_flow_xy - global_circle_flow) ** 2,
            axis=2,
        )) / (0.25 + global_magnitude)
    ))
    atlas_disagreement = float(np.mean(
        np.sqrt(np.sum(
            (estimate.forward_sampling_flow_xy - circle_atlas_flow) ** 2,
            axis=2,
        )) / (0.25 + atlas_magnitude_field)
    ))
    normalized_disagreement = float(np.sqrt(
        global_disagreement * atlas_disagreement))
    if use_circle_fiber:
        atlas_spectral_dispersion = np.asarray(
            circle_atlas_record["spectral_dispersion_field"],
            dtype=np.float64,
        )
        atlas_observability = np.asarray(
            circle_atlas_record["observability_field"],
            dtype=np.float64,
        )
        global_dispersion = float(
            circle_record["translation_dispersion_pixels"])
        spectral_dispersion = atlas_spectral_dispersion
        circle_observability = atlas_observability
    else:
        spectral_dispersion = np.full_like(
            circle_magnitude_field,
            float(circle_record["translation_dispersion_pixels"]),
        )
        circle_observability = np.ones_like(circle_magnitude_field)
    normalized_dispersion_field = (
        spectral_dispersion / (0.25 + atlas_magnitude_field))
    normalized_dispersion = float(np.mean(normalized_dispersion_field))
    global_dispersion_ratio = (
        float(circle_record["translation_dispersion_pixels"])
        / (0.25 + global_magnitude)
    )
    global_disagreement_power = global_disagreement ** 12
    global_authority = (
        global_disagreement_power
        / (global_disagreement_power + 0.12 ** 12)
        * np.exp(-((global_dispersion_ratio / 0.22) ** 4))
    )
    atlas_disagreement_power = normalized_disagreement ** 20
    atlas_disagreement_gate = (
        atlas_disagreement_power
        / (atlas_disagreement_power + 0.42 ** 20)
    )
    atlas_rms = float(circle_atlas_record["flow_rms_pixels"])
    locality_ratio = atlas_rms / (0.25 + global_magnitude)
    locality_power = locality_ratio ** 12
    locality_gate = locality_power / (locality_power + 0.80 ** 12)
    sparse_atlas_support = 1.0 - float(np.mean(circle_observability))
    sparse_support_power = sparse_atlas_support ** 12
    sparse_support_gate = sparse_support_power / (
        sparse_support_power + 0.55 ** 12)
    observability_power = circle_observability ** 4
    observability_gate = observability_power / (
        observability_power + 0.03 ** 4)
    atlas_authority = (
        atlas_disagreement_gate
        * locality_gate
        * sparse_support_gate
        * observability_gate
        * np.exp(-((normalized_dispersion_field / 0.80) ** 4))
    )
    coherent_disagreement_authority = (
        1.0 - (1.0 - global_authority) * (1.0 - atlas_authority))
    correction_authority = np.clip(
        1.0
        - (1.0 - fold_authority)
        * (1.0 - coherent_disagreement_authority),
        0.0,
        1.0,
    )
    authority_for_image = (
        correction_authority
        if images[0].ndim == 2 else correction_authority[..., None])
    reconstructed = (
        (1.0 - authority_for_image) * common.image
        + authority_for_image * solution.image)
    uncertainty = np.sqrt(
        (1.0 - authority_for_image) ** 2 * common.uncertainty ** 2
        + authority_for_image ** 2 * solution.uncertainty ** 2
        + authority_for_image * (1.0 - authority_for_image)
        * (solution.image - common.image) ** 2
    )
    return FlowFiberConsensusResult(
        image=reconstructed,
        common_image=common.image,
        uncertainty=uncertainty,
        estimate=estimate,
        fiber_solution=solution,
        diagnostics={
            **estimate.diagnostics,
            **radiometric.diagnostics,
            **relative_mixing_diagnostics,
            **solution.diagnostics,
            **measure_record,
            "method": "continuous_positive_flow_fiber_consensus",
            "flow_support": support_coordinates,
            "flow_support_prior": support_prior_array.tolist(),
            "flow_atlas_prior_center": atlas_center,
            "global_circle_prior_reliability": global_reliability,
            "local_atlas_prior_reliability": atlas_reliability,
            "fourier_circle_translation_xy": circle_translation.tolist(),
            "fourier_circle_transport": circle_record,
            "fourier_circle_atlas_transport": circle_atlas_diagnostics,
            "support_count": len(support_flow_array),
            "fiber_passes": min(
                max(int(fiber_passes), 1), max(int(passes), 1)),
            "fold_fractions": fold_fractions,
            "execution_chart": "continuous_positive_flow_fiber",
            "support_role": (
                "quadrature_of_one_continuous_measure_not_a_layer_catalog"),
            "correction_authority_mean": float(np.mean(correction_authority)),
            "correction_authority_max": float(np.max(correction_authority)),
            "fold_authority_mean": float(np.mean(fold_authority)),
            "fold_authority_max": float(np.max(fold_authority)),
            "dense_circle_disagreement": normalized_disagreement,
            "dense_global_circle_disagreement": global_disagreement,
            "dense_atlas_circle_disagreement": atlas_disagreement,
            "global_circle_authority": float(global_authority),
            "atlas_locality_ratio": float(locality_ratio),
            "atlas_locality_gate": float(locality_gate),
            "sparse_atlas_support": float(sparse_atlas_support),
            "sparse_atlas_support_gate": float(sparse_support_gate),
            "normalized_circle_dispersion": normalized_dispersion,
            "coherent_disagreement_authority": float(
                np.mean(coherent_disagreement_authority)),
            "jacobian_pressure_mean": float(np.mean(jacobian_pressure)),
            "jacobian_pressure_max": float(np.max(jacobian_pressure)),
            "jacobian_pressure_transport": pressure_transport,
            "correction_authority_role": (
                "continuous_union_of_fold_nonclosure_and_coherent_fourier_"
                "circle_dense_disagreement_not_model_choice"),
            "common_gauge_method": common.diagnostics["method"],
            "uncertainty_rms": float(np.sqrt(np.mean(uncertainty ** 2))),
            "uncertainty_q95": float(np.quantile(uncertainty, 0.95)),
            "uncertainty_role": (
                "continuous_common_gauge_plus_fourier_circle_fiber_disagreement_"
                "not_calibrated_interval"),
            "estimation_decision": "positive_flow_fiber_supported",
            "zero_measure_fast_path": False,
        },
    )
