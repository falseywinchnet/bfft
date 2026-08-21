"""Causal joint-information branch transport on the Hopf--Lax parent DAG.

This is the two-dimensional lift of the positive 1-D lineage experiment.  It
uses the existing continuous tangent law and predictive horizontal geometry;
no spatial smoother or residual continuation is applied.  Population phase,
angular count, and quantile count remain numerical refinement coordinates for
the theorem probe and are not promoted denoising controls.
"""

from __future__ import annotations

from typing import Any
from functools import lru_cache
import math

import numpy as np

from .causal_ancestry import shared_label_continuous_causal_forest
from .continuous_tangent_transport_2d import (
    continuous_tangent_joint_population_2d,
)
from .fused_transport_geometry import (
    predictive_horizontal_wasserstein_geometry,
    weighted_empirical_quantiles,
)
from .witnessed_characteristic_transport_2d import _validate, _weighted_median


@lru_cache(maxsize=None)
def _discrete_gradient_operator(shape: tuple[int, int]):
    """Return the incidence matrix for every unoriented grid edge."""
    from scipy import sparse

    height, width = shape
    horizontal = height * (width - 1)
    vertical = (height - 1) * width
    edges = horizontal + vertical
    row = np.repeat(np.arange(edges, dtype=np.int64), 2)
    column = np.empty(2 * edges, dtype=np.int64)
    value = np.tile(np.array([-1.0, 1.0]), edges)
    cursor = 0
    for y in range(height):
        for x in range(width - 1):
            column[2 * cursor:2 * cursor + 2] = (
                y * width + x, y * width + x + 1)
            cursor += 1
    for y in range(height - 1):
        for x in range(width):
            column[2 * cursor:2 * cursor + 2] = (
                y * width + x, (y + 1) * width + x)
            cursor += 1
    return sparse.csr_matrix((value, (row, column)), shape=(edges, height * width))


def _integrable_jet_potential(
    jet_y: np.ndarray,
    jet_x: np.ndarray,
    mean_value: float,
) -> np.ndarray:
    """Hodge-project a transported jet and fix only its additive gauge."""
    from scipy.sparse.linalg import lsmr

    jy = np.asarray(jet_y, dtype=np.float64)
    jx = np.asarray(jet_x, dtype=np.float64)
    if jy.shape != jx.shape or jy.ndim != 2:
        raise ValueError("jet components must be aligned 2-D fields")
    edge_x = 0.5 * (jx[:, :-1] + jx[:, 1:])
    edge_y = 0.5 * (jy[:-1, :] + jy[1:, :])
    edge_jet = np.concatenate((edge_x.ravel(), edge_y.ravel()))
    operator = _discrete_gradient_operator(jy.shape)
    tolerance = math.sqrt(np.finfo(float).eps)
    solution = lsmr(
        operator,
        edge_jet,
        atol=tolerance,
        btol=tolerance,
        maxiter=4 * jy.size,
    )[0].reshape(jy.shape)
    solution += float(mean_value) - float(np.mean(solution))
    return solution


def _joint_geometric_median(
    signal: np.ndarray,
    jet_y: np.ndarray,
    jet_x: np.ndarray,
    residual: np.ndarray,
    mass: np.ndarray,
    floor: float,
) -> np.ndarray:
    """Return the z-section of the joint determinant-one W1 barycenter."""
    coordinate = np.stack((signal, jet_y, jet_x, residual), axis=-1)
    shape = signal.shape[:2]
    atoms = signal.shape[-1]
    point = coordinate.reshape(-1, atoms, 4)
    probability = mass.reshape(-1, atoms)
    section = np.empty(point.shape[0], dtype=np.float64)
    numerical_tolerance = math.sqrt(np.finfo(float).eps)
    for pixel in range(point.shape[0]):
        branch = point[pixel]
        weight = probability[pixel]
        precision, _anisotropy = _determinant_one_precision(
            branch, weight, floor * floor)
        location = weight @ branch
        scale = max(
            float(np.ptp(branch, axis=0).max()),
            float(np.max(np.abs(location))),
            floor,
        )
        for _iteration in range(4 * atoms):
            defect = location[None, :] - branch
            distance = np.sqrt(np.maximum(np.einsum(
                "...a,ab,...b->...", defect, precision, defect), 0.0))
            authority = weight / np.maximum(distance, floor)
            candidate = authority @ branch / np.sum(authority)
            step = candidate - location
            location = candidate
            if float(np.linalg.norm(step)) <= numerical_tolerance * scale:
                break
        section[pixel] = location[0]
    return section.reshape(shape)


def _determinant_one_precision(
    coordinates: np.ndarray,
    weights: np.ndarray,
    covariance_floor: float,
    complete_residual_moment: bool = False,
) -> tuple[np.ndarray, float]:
    """Return covariance inverse with unit determinant and its anisotropy."""
    point = np.asarray(coordinates, dtype=np.float64)
    mass = np.asarray(weights, dtype=np.float64)
    mass = mass / np.sum(mass)
    center = mass @ point
    centered = point - center
    covariance = (centered * mass[:, None]).T @ centered
    if complete_residual_moment:
        if point.shape[1] != 4:
            raise ValueError(
                "complete residual moment requires (signal,jy,jx,residual)")
        # Cov(r)+E[r]^2 = E[r^2].  Other coordinates remain translation
        # invariant; zero residual is a physical hypothesis, not a gauge.
        covariance[3, 3] += center[3] * center[3]
    eigenvalue, eigenvector = np.linalg.eigh(covariance)
    precision_eigenvalue = 1.0 / np.maximum(eigenvalue, covariance_floor)
    geometric = float(np.exp(np.mean(np.log(precision_eigenvalue))))
    precision_eigenvalue /= geometric
    precision = (
        eigenvector * precision_eigenvalue[None, :]
    ) @ eigenvector.T
    anisotropy = float(
        np.max(precision_eigenvalue) / np.min(precision_eigenvalue))
    return precision, anisotropy


def _parent_child_kernel(
    parent: int,
    child: int,
    shape: tuple[int, int],
    signal: np.ndarray,
    residual: np.ndarray,
    jet_y: np.ndarray,
    jet_x: np.ndarray,
    local_mass: np.ndarray,
    reference_mass: np.ndarray,
    floor: float,
    complete_residual_moment: bool = False,
) -> tuple[np.ndarray, float]:
    """Couple two branch laws after parallel transport to their midpoint."""
    height, width = shape
    py, px = divmod(int(parent), width)
    cy, cx = divmod(int(child), width)
    dy = float(cy - py)
    dx = float(cx - px)
    parent_signal = signal[parent]
    child_signal = signal[child]
    parent_midpoint = parent_signal + 0.5 * (
        jet_y[parent] * dy + jet_x[parent] * dx)
    child_midpoint = child_signal - 0.5 * (
        jet_y[child] * dy + jet_x[child] * dx)
    parent_coordinate = np.column_stack((
        parent_midpoint,
        jet_y[parent],
        jet_x[parent],
        residual[parent],
    ))
    child_coordinate = np.column_stack((
        child_midpoint,
        jet_y[child],
        jet_x[child],
        residual[child],
    ))
    coordinate = np.concatenate((parent_coordinate, child_coordinate), axis=0)
    weight = 0.5 * np.concatenate((local_mass[parent], local_mass[child]))
    precision, anisotropy = _determinant_one_precision(
        coordinate,
        weight,
        floor * floor,
        complete_residual_moment=complete_residual_moment,
    )
    defect = parent_coordinate[:, None, :] - child_coordinate[None, :, :]
    distance_squared = np.einsum(
        "...a,ab,...b->...", defect, precision, defect)
    distance = np.sqrt(np.maximum(distance_squared, 0.0))
    kernel = reference_mass[child][None, :] / np.maximum(distance, floor)
    row_mass = np.sum(kernel, axis=1, keepdims=True)
    if np.any(row_mass <= 0.0):
        raise RuntimeError("causal branch transition lost all child support")
    kernel /= row_mass
    return kernel, anisotropy


def causal_information_lineage_law_2d(
    observation: np.ndarray,
    *,
    angular_count: int = 4,
    quantile_count: int = 16,
    population_phase: float = 0.0,
    memory_ceiling_bytes: int | None = None,
    complete_residual_moment: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Push the joint branch law through exact causal parent fractions."""
    image = _validate(observation)
    population, population_diagnostic = continuous_tangent_joint_population_2d(
        image, angular_count=angular_count)
    quantiles = weighted_empirical_quantiles(
        population["signal"], population["prior_mass"], quantile_count)
    horizontal = predictive_horizontal_wasserstein_geometry(quantiles)

    from port_needed.continuous_eikonal_transport import prepare_continuous_metric
    from port_needed.density_population import emit_density_population

    centers, population_realization = emit_density_population(
        {
            "measure": horizontal["measure"],
            "implied_cells": horizontal["implied_support"],
        },
        safety_cells=image.size,
        phase_shift=float(population_phase),
    )
    if population_realization["safety_limit_hit"]:
        raise RuntimeError("domain-sized numerical population ceiling was hit")
    prepared = prepare_continuous_metric(
        horizontal["metric_xx"],
        horizontal["metric_xy"],
        horizontal["metric_yy"],
        consistency_limit=np.finfo(float).max,
    )
    forest, ancestry = shared_label_continuous_causal_forest(
        centers,
        prepared,
        memory_ceiling_bytes=memory_ceiling_bytes,
    )

    height, width = image.shape
    pixels = image.size
    signal = np.asarray(population["signal"], dtype=np.float64).reshape(
        pixels, -1)
    residual = np.asarray(population["residual"], dtype=np.float64).reshape(
        pixels, -1)
    local_mass = np.asarray(population["mass"], dtype=np.float64).reshape(
        pixels, -1)
    scale = np.asarray(
        population["scale_conductance"], dtype=np.float64).reshape(pixels, -1)
    reference_mass = scale / np.sum(scale, axis=1, keepdims=True)
    derivative = np.asarray(
        population["directional_derivative"], dtype=np.float64).reshape(
            pixels, -1)
    tangent = np.asarray(population["tangent"], dtype=np.float64)
    source_identity = np.asarray(population["source_identity"], dtype=np.int64)
    source_coefficient = np.asarray(
        population["source_coefficient"], dtype=np.float64)
    jet_y = derivative * tangent[None, :, 0]
    jet_x = derivative * tangent[None, :, 1]
    action = np.asarray(population["joint_action"], dtype=np.float64).reshape(
        pixels, -1)
    magnitude = max(float(np.max(np.abs(image))), float(np.ptp(image)))
    floor = (
        np.finfo(float).tiny
        if magnitude == 0.0
        else math.sqrt(np.finfo(float).eps) * magnitude
    )
    likelihood = 1.0 / np.maximum(action, floor)
    likelihood /= np.max(likelihood, axis=1, keepdims=True)

    first = np.asarray(forest["parent_first"], dtype=np.int64).reshape(-1)
    second = np.asarray(forest["parent_second"], dtype=np.int64).reshape(-1)
    fraction = np.asarray(forest["parent_fraction"], dtype=np.float64).reshape(-1)
    order = np.asarray(forest["acceptance_order"], dtype=np.int64).reshape(-1)
    source_count = int(ancestry.source_count)
    root_identity = np.asarray(
        forest["root_identity"], dtype=np.int64).reshape(-1)
    state = np.zeros(
        (pixels, source_count, local_mass.shape[-1]), dtype=np.float64)
    path_score = np.full_like(local_mass, -np.inf)
    anisotropies = []
    root_count = 0
    for child in order:
        parent_a = int(first[child])
        parent_b = int(second[child])
        if parent_a < 0:
            identity = int(root_identity[child])
            if identity < 0:
                raise RuntimeError("parentless point has no continuous lineage")
            state[child, identity] = local_mass[child]
            path_score[child] = np.log(np.maximum(
                local_mass[child] / np.maximum(
                    reference_mass[child], np.finfo(float).tiny),
                np.finfo(float).tiny))
            path_score[child] -= np.max(path_score[child])
            root_count += 1
            continue
        kernel_a, anisotropy_a = _parent_child_kernel(
            parent_a, int(child), image.shape,
            signal, residual, jet_y, jet_x,
            local_mass,
            reference_mass,
            floor,
            complete_residual_moment=complete_residual_moment,
        )
        predicted = state[parent_a] @ kernel_a
        message_a = np.max(
            path_score[parent_a][:, None]
            + np.log(np.maximum(
                kernel_a / np.maximum(
                    reference_mass[child][None, :], np.finfo(float).tiny),
                np.finfo(float).tiny)),
            axis=0,
        )
        predicted_score = message_a
        anisotropies.append(anisotropy_a)
        if parent_b >= 0:
            kernel_b, anisotropy_b = _parent_child_kernel(
                parent_b, int(child), image.shape,
                signal, residual, jet_y, jet_x,
                local_mass,
                reference_mass,
                floor,
                complete_residual_moment=complete_residual_moment,
            )
            t = float(fraction[child])
            predicted = (1.0 - t) * predicted + t * (state[parent_b] @ kernel_b)
            message_b = np.max(
                path_score[parent_b][:, None]
                + np.log(np.maximum(
                    kernel_b / np.maximum(
                        reference_mass[child][None, :], np.finfo(float).tiny),
                    np.finfo(float).tiny)),
                axis=0,
            )
            predicted_score = (1.0 - t) * message_a + t * message_b
            anisotropies.append(anisotropy_b)
        state[child] = predicted * likelihood[child][None, :]
        total = float(np.sum(state[child]))
        if total <= 0.0:
            raise RuntimeError("causal branch likelihood removed all mass")
        state[child] /= total
        path_score[child] = predicted_score + np.log(np.maximum(
            likelihood[child], np.finfo(float).tiny))
        path_score[child] -= np.max(path_score[child])

    marginal_mass = np.sum(state, axis=1)
    collision_path_score = 2.0 * path_score
    collision_path_mass = reference_mass * np.exp(collision_path_score)
    collision_path_mass /= np.sum(
        collision_path_mass, axis=1, keepdims=True)
    path_barycenter = np.sum(signal * collision_path_mass, axis=1)
    path_w1_barycenter = _weighted_median(signal, collision_path_mass)
    root_probability = np.sum(state, axis=2)
    effective_root_count = 1.0 / np.sum(
        root_probability * root_probability, axis=1)
    ancestry_collision_order = 1.0 + effective_root_count
    ancestry_collision_mass = reference_mass * np.exp(
        ancestry_collision_order[:, None] * path_score)
    ancestry_collision_mass /= np.sum(
        ancestry_collision_mass, axis=1, keepdims=True)
    ancestry_path_barycenter = np.sum(
        signal * ancestry_collision_mass, axis=1)
    ancestry_path_w1_barycenter = _weighted_median(
        signal, ancestry_collision_mass)
    simplex_collision_order = np.ones(pixels, dtype=np.float64)
    has_parent = first >= 0
    simplex_collision_order[has_parent] = 2.0
    has_parent_pair = second >= 0
    parent_pair_fraction = fraction[has_parent_pair]
    simplex_collision_order[has_parent_pair] = 1.0 + 1.0 / (
        (1.0 - parent_pair_fraction) ** 2 + parent_pair_fraction ** 2)
    simplex_collision_mass = reference_mass * np.exp(
        simplex_collision_order[:, None] * path_score)
    simplex_collision_mass /= np.sum(
        simplex_collision_mass, axis=1, keepdims=True)
    simplex_path_barycenter = np.sum(
        signal * simplex_collision_mass, axis=1)
    simplex_path_w1_barycenter = _weighted_median(
        signal, simplex_collision_mass)
    path_index = np.argmax(collision_path_score, axis=1)
    path_section = signal[np.arange(pixels), path_index]
    ordered_path_score = np.sort(collision_path_score, axis=1)
    path_gap = ordered_path_score[:, -1] - ordered_path_score[:, -2]
    causal_quantiles = weighted_empirical_quantiles(
        signal.reshape(image.shape + (signal.shape[-1],)),
        marginal_mass.reshape(image.shape + (marginal_mass.shape[-1],)),
        quantile_count,
    )
    causal_horizontal = predictive_horizontal_wasserstein_geometry(
        causal_quantiles)
    initial_measure = np.asarray(horizontal["measure"], dtype=np.float64)
    causal_measure = np.asarray(causal_horizontal["measure"], dtype=np.float64)
    measure_rms = float(np.sqrt(np.mean(
        (causal_measure - initial_measure) ** 2)))
    measure_reference = float(np.mean(initial_measure))
    return {
        "mass": marginal_mass.reshape(
            image.shape + (marginal_mass.shape[-1],)),
        "root_mass": state.reshape(
            image.shape + (source_count, state.shape[-1])),
        "local_mass": local_mass.reshape(
            image.shape + (local_mass.shape[-1],)),
        # The current observation enters the transported branch state exactly
        # once through this likelihood.  Exposing it permits an exact cavity
        # law without rerunning causal transport or inventing a noise model.
        "likelihood": likelihood.reshape(
            image.shape + (likelihood.shape[-1],)),
        "signal": signal.reshape(image.shape + (signal.shape[-1],)),
        "residual": residual.reshape(image.shape + (residual.shape[-1],)),
        "jet_y": jet_y.reshape(image.shape + (jet_y.shape[-1],)),
        "jet_x": jet_x.reshape(image.shape + (jet_x.shape[-1],)),
        # Projective tangent is the operator-chart identity of every branch.
        # Keeping it in the terminal law lets later estimators retain
        # uncertainty between transport maps instead of collapsing all maps
        # into one scalar branch mixture.
        "tangent": tangent.copy(),
        # The exact target-free observation graph is the analogue of a
        # deblurring forward operator.  Later relative-closure estimators need
        # its coefficient Gram matrix to distinguish normalized chart
        # ownership from absolute independent evidence coverage.
        "source_identity": source_identity.copy(),
        "source_coefficient": source_coefficient.copy(),
        "reference_mass": reference_mass.reshape(
            image.shape + (reference_mass.shape[-1],)),
        # Expose the gauge-fixed terminal action and its causal collision
        # dimension for theorem probes.  They are state coordinates, not
        # scalar denoising controls.
        "hj_path_score": path_score.reshape(
            image.shape + (path_score.shape[-1],)),
        "hj_simplex_collision_order": simplex_collision_order.reshape(
            image.shape),
        "hj_collision_mass": collision_path_mass.reshape(
            image.shape + (collision_path_mass.shape[-1],)),
        "hj_ancestry_collision_mass": ancestry_collision_mass.reshape(
            image.shape + (ancestry_collision_mass.shape[-1],)),
        "hj_simplex_collision_mass": simplex_collision_mass.reshape(
            image.shape + (simplex_collision_mass.shape[-1],)),
        "causal_hj_collision_section": path_section.reshape(image.shape),
        "causal_hj_collision_barycenter": path_barycenter.reshape(image.shape),
        "causal_hj_collision_w1_barycenter": path_w1_barycenter.reshape(
            image.shape),
        "causal_hj_ancestry_collision_barycenter": (
            ancestry_path_barycenter.reshape(image.shape)),
        "causal_hj_ancestry_collision_w1_barycenter": (
            ancestry_path_w1_barycenter.reshape(image.shape)),
        "causal_hj_simplex_collision_barycenter": (
            simplex_path_barycenter.reshape(image.shape)),
        "causal_hj_simplex_collision_w1_barycenter": (
            simplex_path_w1_barycenter.reshape(image.shape)),
    }, {
        "status": "forward joint-information branch transport on Hopf--Lax DAG",
        "angular_count": int(angular_count),
        "quantile_count": int(quantile_count),
        "population_phase": float(population_phase),
        "complete_residual_moment": bool(complete_residual_moment),
        "joint_population": population_diagnostic,
        "horizontal_geometry": horizontal,
        "causal_horizontal_geometry": causal_horizontal,
        "population": population_realization,
        "centers": centers,
        "forest": forest,
        "continuous_root_count": int(len(centers)),
        "raster_root_count": int(root_count),
        "mean_source_collision_population": float(np.mean(
            1.0 / np.sum(
                np.sum(state, axis=2) ** 2, axis=1))),
        "mean_geometric_source_collision_population": float(np.mean(
            ancestry.collision_population)),
        "mean_branch_collision_population": float(np.mean(
            1.0 / np.sum(marginal_mass * marginal_mass, axis=1))),
        "mean_bundle_anisotropy": float(np.mean(anisotropies)),
        "mass_maximum_error": float(np.max(np.abs(
            np.sum(state, axis=(1, 2)) - 1.0))),
        "marginal_maximum_error": float(np.max(np.abs(
            marginal_mass - np.sum(state, axis=1)))),
        "root_branch_state_bytes": int(state.nbytes),
        "initial_implied_support": float(horizontal["implied_support"]),
        "causal_implied_support": float(
            causal_horizontal["implied_support"]),
        "causal_measure_relative_rms": float(
            measure_rms / max(measure_reference, np.finfo(float).tiny)),
        "mean_hj_collision_score_gap": float(np.mean(path_gap)),
        "mean_hj_collision_participation": float(np.mean(
            1.0 / np.sum(collision_path_mass * collision_path_mass, axis=1))),
        "mean_hj_ancestry_collision_order": float(np.mean(
            ancestry_collision_order)),
        "minimum_hj_ancestry_collision_order": float(np.min(
            ancestry_collision_order)),
        "maximum_hj_ancestry_collision_order": float(np.max(
            ancestry_collision_order)),
        "mean_hj_simplex_collision_order": float(np.mean(
            simplex_collision_order)),
        "minimum_hj_simplex_collision_order": float(np.min(
            simplex_collision_order)),
        "maximum_hj_simplex_collision_order": float(np.max(
            simplex_collision_order)),
        "theory_status": (
            "root-resolved 2-D causal information-lineage probe; angular and "
            "phase convergence pending"
        ),
    }


def _readouts_from_law(
    image: np.ndarray,
    causal: dict[str, np.ndarray],
    *,
    include_experimental: bool = False,
) -> dict[str, np.ndarray]:
    """Project one causal measure after all representation integration."""
    signal = causal["signal"]
    mass = causal["mass"]
    reference = causal["reference_mass"]
    causal_median = _weighted_median(signal, mass)
    causal_mean = np.sum(mass * signal, axis=-1)
    causal_maximum = np.take_along_axis(
        signal, np.argmax(mass, axis=-1)[..., None], axis=-1)[..., 0]
    collision_mass = mass * mass / np.maximum(reference, np.finfo(float).tiny)
    collision_mass /= np.sum(collision_mass, axis=-1, keepdims=True)
    collision_median = _weighted_median(signal, collision_mass)
    collision_mean = np.sum(collision_mass * signal, axis=-1)
    collision_maximum = np.take_along_axis(
        signal,
        np.argmax(collision_mass, axis=-1)[..., None],
        axis=-1,
    )[..., 0]
    local_mass = causal["local_mass"]
    local_median = _weighted_median(signal, local_mass)
    local_maximum = np.take_along_axis(
        signal, np.argmax(local_mass, axis=-1)[..., None],
        axis=-1)[..., 0]
    lower = float(np.min(image))
    upper = float(np.max(image))
    result = {
        "local_median": np.clip(local_median, lower, upper),
        "local_maximum": np.clip(local_maximum, lower, upper),
        "causal_mean": np.clip(causal_mean, lower, upper),
        "causal_median": np.clip(causal_median, lower, upper),
        "causal_maximum": np.clip(causal_maximum, lower, upper),
        "causal_collision_median": np.clip(
            collision_median, lower, upper),
        "causal_collision_mean": np.clip(
            collision_mean, lower, upper),
        "causal_collision_maximum": np.clip(
            collision_maximum, lower, upper),
    }
    if include_experimental:
        collision_jet_y = np.sum(
            collision_mass * causal["jet_y"], axis=-1)
        collision_jet_x = np.sum(
            collision_mass * causal["jet_x"], axis=-1)
        collision_jet_potential = _integrable_jet_potential(
            collision_jet_y, collision_jet_x, float(np.mean(collision_mean)))
        magnitude = max(float(np.max(np.abs(image))), float(np.ptp(image)))
        coordinate_floor = (
            np.finfo(float).tiny
            if magnitude == 0.0
            else math.sqrt(np.finfo(float).eps) * magnitude
        )
        collision_joint_median = _joint_geometric_median(
            signal,
            causal["jet_y"],
            causal["jet_x"],
            causal["residual"],
            collision_mass,
            coordinate_floor,
        )
        result["causal_collision_jet_potential"] = np.clip(
            collision_jet_potential, lower, upper)
        result["causal_collision_joint_median"] = np.clip(
            collision_joint_median, lower, upper)
    if "root_mass" in causal:
        root_mass = causal["root_mass"]
        root_probability = np.sum(root_mass, axis=-1)
        same_root_probability = np.sum(
            root_probability * root_probability, axis=-1)
        cross_root_density = (
            mass * mass - np.sum(root_mass * root_mass, axis=-2)
        ) / np.maximum(reference, np.finfo(float).tiny)
        cross_total = np.sum(cross_root_density, axis=-1, keepdims=True)
        cross_law = np.divide(
            cross_root_density,
            cross_total,
            out=np.zeros_like(cross_root_density),
            where=cross_total > np.finfo(float).tiny,
        )
        cross_lineage_mass = (
            (1.0 - same_root_probability)[..., None] * cross_law
            + same_root_probability[..., None] * local_mass
        )
        cross_lineage_mass /= np.sum(
            cross_lineage_mass, axis=-1, keepdims=True)
        result["causal_cross_lineage_median"] = np.clip(
            _weighted_median(signal, cross_lineage_mass), lower, upper)
    if "phase_collision_maximum" in causal:
        result["causal_phase_average_collision_maximum"] = np.clip(
            causal["phase_collision_maximum"], lower, upper)
    if "phase_collision_mode" in causal:
        result["causal_phase_average_collision_mode"] = np.clip(
            causal["phase_collision_mode"], lower, upper)
    if "phase_hj_collision_section" in causal:
        result["causal_phase_average_hj_collision_section"] = np.clip(
            causal["phase_hj_collision_section"], lower, upper)
    elif "causal_hj_collision_section" in causal:
        result["causal_hj_collision_section"] = np.clip(
            causal["causal_hj_collision_section"], lower, upper)
    if "phase_hj_collision_barycenter" in causal:
        result["causal_phase_average_hj_collision_barycenter"] = np.clip(
            causal["phase_hj_collision_barycenter"], lower, upper)
    elif "causal_hj_collision_barycenter" in causal:
        result["causal_hj_collision_barycenter"] = np.clip(
            causal["causal_hj_collision_barycenter"], lower, upper)
    if "phase_hj_collision_w1_barycenter" in causal:
        result["causal_phase_average_hj_collision_w1_barycenter"] = np.clip(
            causal["phase_hj_collision_w1_barycenter"], lower, upper)
    elif "causal_hj_collision_w1_barycenter" in causal:
        result["causal_hj_collision_w1_barycenter"] = np.clip(
            causal["causal_hj_collision_w1_barycenter"], lower, upper)
    if "phase_hj_ancestry_collision_barycenter" in causal:
        result["causal_phase_average_hj_ancestry_collision_barycenter"] = (
            np.clip(
                causal["phase_hj_ancestry_collision_barycenter"],
                lower,
                upper,
            )
        )
        result[
            "causal_phase_average_hj_ancestry_collision_w1_barycenter"
        ] = np.clip(
            causal["phase_hj_ancestry_collision_w1_barycenter"],
            lower,
            upper,
        )
    elif "causal_hj_ancestry_collision_barycenter" in causal:
        result["causal_hj_ancestry_collision_barycenter"] = np.clip(
            causal["causal_hj_ancestry_collision_barycenter"], lower, upper)
        result["causal_hj_ancestry_collision_w1_barycenter"] = np.clip(
            causal["causal_hj_ancestry_collision_w1_barycenter"],
            lower,
            upper,
        )
    if "phase_hj_simplex_collision_barycenter" in causal:
        result["causal_phase_average_hj_simplex_collision_barycenter"] = (
            np.clip(
                causal["phase_hj_simplex_collision_barycenter"],
                lower,
                upper,
            )
        )
        result[
            "causal_phase_average_hj_simplex_collision_w1_barycenter"
        ] = np.clip(
            causal["phase_hj_simplex_collision_w1_barycenter"],
            lower,
            upper,
        )
    elif "causal_hj_simplex_collision_barycenter" in causal:
        result["causal_hj_simplex_collision_barycenter"] = np.clip(
            causal["causal_hj_simplex_collision_barycenter"], lower, upper)
        result["causal_hj_simplex_collision_w1_barycenter"] = np.clip(
            causal["causal_hj_simplex_collision_w1_barycenter"],
            lower,
            upper,
        )
    return result


def _collision_maximum_from_law(
    law: dict[str, np.ndarray],
) -> np.ndarray:
    mass = law["mass"]
    reference = law["reference_mass"]
    density = mass * mass / np.maximum(reference, np.finfo(float).tiny)
    return np.take_along_axis(
        law["signal"],
        np.argmax(density, axis=-1)[..., None],
        axis=-1,
    )[..., 0]


def _collision_mode_from_law(
    law: dict[str, np.ndarray],
) -> np.ndarray:
    """Select the collision density mode relative to branch Haar measure."""
    density = law["mass"] / np.maximum(
        law["reference_mass"], np.finfo(float).tiny)
    return np.take_along_axis(
        law["signal"],
        np.argmax(density, axis=-1)[..., None],
        axis=-1,
    )[..., 0]


def causal_information_lineage_readouts_2d(
    observation: np.ndarray,
    **kwargs: Any,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read local and causal sections from one population realization."""
    image = _validate(observation)
    causal, diagnostic = causal_information_lineage_law_2d(image, **kwargs)
    return _readouts_from_law(image, causal), diagnostic


def nested_population_phases(count: int) -> tuple[float, ...]:
    """Return the first ``count`` base-two radical-inverse phases.

    Powers of two form nested equal-mass quadratures of the representation
    fibre.  The count is numerical resolution, never denoising time.
    """
    size = int(count)
    if size < 1 or size & (size - 1):
        raise ValueError("population phase count must be a positive power of two")

    def radical_inverse(index: int) -> float:
        value = 0.0
        place = 0.5
        while index:
            value += place * (index & 1)
            index >>= 1
            place *= 0.5
        return value

    return tuple(radical_inverse(index) for index in range(size))


def causal_information_phase_integrated_law_2d(
    observation: np.ndarray,
    *,
    angular_count: int = 4,
    quantile_count: int = 16,
    phase_count: int = 4,
    memory_ceiling_bytes: int | None = None,
    complete_residual_moment: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Integrate the causal branch law over numerical population phase."""
    image = _validate(observation)
    phases = nested_population_phases(phase_count)
    laws = []
    diagnostics = []
    for phase in phases:
        law, diagnostic = causal_information_lineage_law_2d(
            image,
            angular_count=angular_count,
            quantile_count=quantile_count,
            population_phase=phase,
            memory_ceiling_bytes=memory_ceiling_bytes,
            complete_residual_moment=complete_residual_moment,
        )
        laws.append(law)
        diagnostics.append(diagnostic)
    mass = np.mean(np.stack([law["mass"] for law in laws]), axis=0)
    combined = {
        key: value for key, value in laws[0].items() if key != "root_mass"
    }
    combined["mass"] = mass
    combined["hj_collision_mass"] = np.mean(np.stack([
        law["hj_collision_mass"] for law in laws]), axis=0)
    combined["phase_collision_maximum"] = np.mean(np.stack([
        _collision_maximum_from_law(law) for law in laws]), axis=0)
    combined["phase_collision_mode"] = np.mean(np.stack([
        _collision_mode_from_law(law) for law in laws]), axis=0)
    combined["phase_hj_collision_section"] = np.mean(np.stack([
        law["causal_hj_collision_section"] for law in laws]), axis=0)
    combined["phase_hj_collision_barycenter"] = np.mean(np.stack([
        law["causal_hj_collision_barycenter"] for law in laws]), axis=0)
    combined["phase_hj_collision_w1_barycenter"] = np.mean(np.stack([
        law["causal_hj_collision_w1_barycenter"] for law in laws]), axis=0)
    combined["phase_hj_ancestry_collision_barycenter"] = np.mean(np.stack([
        law["causal_hj_ancestry_collision_barycenter"] for law in laws
    ]), axis=0)
    combined["phase_hj_ancestry_collision_w1_barycenter"] = np.mean(np.stack([
        law["causal_hj_ancestry_collision_w1_barycenter"] for law in laws
    ]), axis=0)
    combined["phase_hj_simplex_collision_barycenter"] = np.mean(np.stack([
        law["causal_hj_simplex_collision_barycenter"] for law in laws
    ]), axis=0)
    combined["phase_hj_simplex_collision_w1_barycenter"] = np.mean(np.stack([
        law["causal_hj_simplex_collision_w1_barycenter"] for law in laws
    ]), axis=0)
    return combined, _phase_integral_diagnostic(
        mass, laws, diagnostics, phases, angular_count, quantile_count)


def _phase_integral_diagnostic(
    mass: np.ndarray,
    laws: list[dict[str, np.ndarray]],
    diagnostics: list[dict[str, Any]],
    phases: tuple[float, ...],
    angular_count: int,
    quantile_count: int,
) -> dict[str, Any]:
    phase_rms = [float(np.sqrt(np.mean((law["mass"] - mass) ** 2)))
                 for law in laws]
    return {
        "status": "causal branch measure integrated over population phase",
        "phase_count": int(len(phases)),
        "phases": phases,
        "angular_count": int(angular_count),
        "quantile_count": int(quantile_count),
        "phase_mass_mean_rms": float(np.mean(phase_rms)),
        "phase_mass_maximum_rms": float(np.max(phase_rms)),
        "mass_maximum_error": float(np.max(np.abs(
            np.sum(mass, axis=-1) - 1.0))),
        "continuous_root_count_minimum": int(min(
            row["continuous_root_count"] for row in diagnostics)),
        "continuous_root_count_maximum": int(max(
            row["continuous_root_count"] for row in diagnostics)),
        "continuous_root_count_mean": float(np.mean([
            row["continuous_root_count"] for row in diagnostics])),
        "initial_implied_support": float(np.mean([
            row["initial_implied_support"] for row in diagnostics])),
        "causal_implied_support": float(np.mean([
            row["causal_implied_support"] for row in diagnostics])),
        "theory_status": (
            "phase is integrated before scalar projection; convergence under "
            "nested phase refinement remains an empirical obligation"
        ),
    }


def causal_information_phase_integrated_readouts_2d(
    observation: np.ndarray,
    **kwargs: Any,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read scalar sections only after population-phase integration."""
    image = _validate(observation)
    causal, diagnostic = causal_information_phase_integrated_law_2d(
        image, **kwargs)
    return _readouts_from_law(image, causal), diagnostic


def causal_information_phase_refinement_readouts_2d(
    observation: np.ndarray,
    *,
    phase_counts: tuple[int, ...] = (1, 2, 4, 8),
    angular_count: int = 4,
    quantile_count: int = 16,
    memory_ceiling_bytes: int | None = None,
) -> dict[int, tuple[dict[str, np.ndarray], dict[str, Any]]]:
    """Resolve nested phase integrals while reusing every finer realization."""
    image = _validate(observation)
    counts = tuple(int(value) for value in phase_counts)
    if not counts or tuple(sorted(set(counts))) != counts:
        raise ValueError("phase counts must be nonempty, unique, and increasing")
    for count in counts:
        nested_population_phases(count)
    phases = nested_population_phases(counts[-1])
    laws = []
    diagnostics = []
    for phase in phases:
        law, diagnostic = causal_information_lineage_law_2d(
            image,
            angular_count=angular_count,
            quantile_count=quantile_count,
            population_phase=phase,
            memory_ceiling_bytes=memory_ceiling_bytes,
        )
        laws.append(law)
        diagnostics.append(diagnostic)
    result = {}
    for count in counts:
        selected_laws = laws[:count]
        mass = np.mean(np.stack([
            law["mass"] for law in selected_laws]), axis=0)
        combined = {
            key: value for key, value in selected_laws[0].items()
            if key != "root_mass"
        }
        combined["mass"] = mass
        combined["hj_collision_mass"] = np.mean(np.stack([
            law["hj_collision_mass"] for law in selected_laws]), axis=0)
        combined["phase_collision_maximum"] = np.mean(np.stack([
            _collision_maximum_from_law(law) for law in selected_laws]), axis=0)
        combined["phase_collision_mode"] = np.mean(np.stack([
            _collision_mode_from_law(law) for law in selected_laws]), axis=0)
        combined["phase_hj_collision_section"] = np.mean(np.stack([
            law["causal_hj_collision_section"] for law in selected_laws]), axis=0)
        combined["phase_hj_collision_barycenter"] = np.mean(np.stack([
            law["causal_hj_collision_barycenter"]
            for law in selected_laws]), axis=0)
        combined["phase_hj_collision_w1_barycenter"] = np.mean(np.stack([
            law["causal_hj_collision_w1_barycenter"]
            for law in selected_laws]), axis=0)
        combined["phase_hj_ancestry_collision_barycenter"] = np.mean(np.stack([
            law["causal_hj_ancestry_collision_barycenter"]
            for law in selected_laws]), axis=0)
        combined["phase_hj_ancestry_collision_w1_barycenter"] = np.mean(
            np.stack([
                law["causal_hj_ancestry_collision_w1_barycenter"]
                for law in selected_laws
            ]), axis=0)
        combined["phase_hj_simplex_collision_barycenter"] = np.mean(np.stack([
            law["causal_hj_simplex_collision_barycenter"]
            for law in selected_laws]), axis=0)
        combined["phase_hj_simplex_collision_w1_barycenter"] = np.mean(
            np.stack([
                law["causal_hj_simplex_collision_w1_barycenter"]
                for law in selected_laws
            ]), axis=0)
        diagnostic = _phase_integral_diagnostic(
            mass,
            selected_laws,
            diagnostics[:count],
            phases[:count],
            angular_count,
            quantile_count,
        )
        result[count] = (
            _readouts_from_law(
                image, combined, include_experimental=True),
            diagnostic,
        )
    return result
