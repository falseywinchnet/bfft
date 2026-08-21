"""Continuous source-measure transport under relation-derived geometry.

This experiment replaces hard first-arrival roots with a probability measure
carried by every observation. A Selling decomposition of the determinant-one
metric gives nonnegative lattice fluxes whose second moment exactly reconstructs
the inverse metric. Their reversible Markov discretization transports complete
source ancestry without a time, radius, bandwidth, or chosen direction list.

The full-scale characteristic measure supplies the residual velocity. Exact
source ancestry supplies only its local authority: every source's held-out
prediction error is transported to the target, and continuation survives only
above that finite-population variance. The dense ancestry matrix is a research
representation, not a production implementation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import sparse

from .cross_predictive_transport_2d import (
    relation_characteristic_measure_2d,
    relation_transport_metric_2d,
)


@dataclass(frozen=True)
class ContinuousSourceResolution:
    """Numerical guards that do not choose the physical smoothing state."""

    maximum_continuations: int = 32
    ancestry_memory_ceiling_bytes: int | None = None


def selling_decomposition(
    metric_xx: np.ndarray,
    metric_xy: np.ndarray,
    metric_yy: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Decompose ``M^-1`` into nonnegative metric-reduced lattice fluxes."""
    from port_needed.metric_reduced_stencil import metric_reduced_superbase

    mxx = np.asarray(metric_xx, dtype=np.float64)
    mxy = np.asarray(metric_xy, dtype=np.float64)
    myy = np.asarray(metric_yy, dtype=np.float64)
    if mxx.ndim != 2 or mxy.shape != mxx.shape or myy.shape != mxx.shape:
        raise ValueError("metric fields must be aligned 2-D arrays")
    determinant = mxx * myy - mxy * mxy
    if np.any(mxx <= 0.0) or np.any(determinant <= 0.0):
        raise ValueError("transport metric must be positive definite")
    vectors = metric_reduced_superbase(mxx, mxy, myy).astype(np.float64)
    basis = np.stack((
        vectors[..., 0] ** 2,
        vectors[..., 0] * vectors[..., 1],
        vectors[..., 1] ** 2,
    ), axis=-2)
    inverse = np.stack((
        myy / determinant,
        -mxy / determinant,
        mxx / determinant,
    ), axis=-1)
    coefficient = np.einsum(
        "...ij,...j->...i", np.linalg.inv(basis), inverse)
    roundoff = 64.0 * np.finfo(float).eps * np.maximum(
        np.max(np.abs(coefficient), axis=-1, keepdims=True), 1.0)
    if np.any(coefficient < -roundoff):
        raise RuntimeError("metric reduction produced a negative Selling flux")
    coefficient = np.maximum(coefficient, 0.0)
    reconstructed_xx = np.sum(
        coefficient * vectors[..., 0] ** 2, axis=-1)
    reconstructed_xy = np.sum(
        coefficient * vectors[..., 0] * vectors[..., 1], axis=-1)
    reconstructed_yy = np.sum(
        coefficient * vectors[..., 1] ** 2, axis=-1)
    error = np.maximum.reduce((
        np.abs(reconstructed_xx - inverse[..., 0]),
        np.abs(reconstructed_xy - inverse[..., 1]),
        np.abs(reconstructed_yy - inverse[..., 2]),
    ))
    return {
        "vectors": vectors.astype(np.int32),
        "coefficient": coefficient,
        "inverse_xx": inverse[..., 0],
        "inverse_xy": inverse[..., 1],
        "inverse_yy": inverse[..., 2],
        "maximum_reconstruction_error": float(np.max(error)),
        "minimum_coefficient": float(np.min(coefficient)),
    }


def source_measure_operator(
    metric_xx: np.ndarray,
    metric_xy: np.ndarray,
    metric_yy: np.ndarray,
) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    """Build a reversible zero-diagonal Markov transport from Selling flux."""
    decomposition = selling_decomposition(metric_xx, metric_xy, metric_yy)
    vectors = np.asarray(decomposition["vectors"], dtype=np.int64)
    coefficient = np.asarray(decomposition["coefficient"], dtype=np.float64)
    height, width = coefficient.shape[:2]
    pixels = height * width
    yy, xx = np.mgrid[:height, :width]
    source = np.arange(pixels).reshape(height, width)
    rows = []
    columns = []
    values = []
    for direction in range(3):
        for sign in (-1, 1):
            nx = xx + sign * vectors[..., direction, 0]
            ny = yy + sign * vectors[..., direction, 1]
            valid = (0 <= nx) & (nx < width) & (0 <= ny) & (ny < height)
            rows.append(source[valid])
            columns.append((ny[valid] * width + nx[valid]).astype(np.int64))
            values.append(coefficient[..., direction][valid])
    directed = sparse.coo_matrix(
        (np.concatenate(values),
         (np.concatenate(rows), np.concatenate(columns))),
        shape=(pixels, pixels),
    ).tocsr()
    directed.sum_duplicates()
    conductance = (0.5 * (directed + directed.T)).tocsr()
    conductance.setdiag(0.0)
    conductance.eliminate_zeros()
    degree = np.asarray(conductance.sum(axis=1)).ravel()
    if np.any(degree <= 0.0):
        raise RuntimeError("source transport graph contains an isolated point")
    operator = (sparse.diags(1.0 / degree) @ conductance).tocsr()
    stationary = degree / np.sum(degree)
    stationary_error = float(np.max(np.abs(
        np.asarray(stationary @ operator).ravel() - stationary)))
    return operator, {
        "selling_maximum_reconstruction_error": decomposition[
            "maximum_reconstruction_error"],
        "selling_minimum_coefficient": decomposition["minimum_coefficient"],
        "row_mass_maximum_error": float(np.max(np.abs(
            np.asarray(operator.sum(axis=1)).ravel() - 1.0))),
        "maximum_diagonal_mass": float(np.max(np.abs(operator.diagonal()))),
        "stationary_mass_maximum_error": stationary_error,
        "stationary_mass": stationary,
        "undirected_edge_count": int(conductance.nnz // 2),
        "source_measure": (
            "reversible Markov measure from exact Selling decomposition"
        ),
    }


def _exclude_target_identity(ancestry: np.ndarray) -> np.ndarray:
    """Condition every transported source law on target exclusion."""
    conditioned = np.asarray(ancestry, dtype=np.float64).copy()
    np.fill_diagonal(conditioned, 0.0)
    mass = np.sum(conditioned, axis=1, keepdims=True)
    if np.any(mass <= 0.0):
        raise RuntimeError("target exclusion removed all predictive ancestry")
    conditioned /= mass
    return conditioned


def _local_cross_predictive_coefficient(
    ancestry: np.ndarray,
    residual: np.ndarray,
    prediction: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Debias target prediction by transported held-out source errors.

    Every source predicts its own residual through the same zero-diagonal law.
    For target prediction ``q_x``, exact ancestry gives the variance of its
    weighted error mean as ``sum_i A_xi^2 (r_i-q_i)^2``. Only prediction energy
    above that finite-population uncertainty is returned.
    """
    square_ancestry = ancestry * ancestry
    held_out_error = residual - prediction
    variance = square_ancestry @ (held_out_error * held_out_error)
    prediction_energy = prediction * prediction
    excess = np.maximum(prediction_energy - variance, 0.0)
    coefficient = np.zeros_like(prediction)
    positive = prediction_energy > 0.0
    coefficient[positive] = excess[positive] / prediction_energy[positive]
    coefficient = np.clip(coefficient, 0.0, 1.0)
    collision_population = 1.0 / np.sum(square_ancestry, axis=1)
    return coefficient, {
        "positive_excess_fraction": float(np.mean(excess > 0.0)),
        "held_out_error_variance_mean": float(np.mean(variance)),
        "mean_coefficient": float(np.mean(coefficient)),
        "maximum_coefficient": float(np.max(coefficient)),
        "collision_population_mean": float(np.mean(collision_population)),
        "collision_population_minimum": float(np.min(collision_population)),
        "collision_population_maximum": float(np.max(collision_population)),
    }


def denoise_continuous_source_transport(
    observation: np.ndarray,
    resolution: ContinuousSourceResolution = ContinuousSourceResolution(),
) -> tuple[np.ndarray, dict[str, Any]]:
    """Advance the continuous source measure until no descent remains."""
    image = np.asarray(observation, dtype=np.float64)
    if image.ndim != 2 or min(image.shape) < 8:
        raise ValueError("continuous source transport expects an HxW field")
    if not np.all(np.isfinite(image)):
        raise ValueError("continuous source transport requires finite samples")
    ceiling = int(resolution.maximum_continuations)
    if ceiling < 1:
        raise ValueError("maximum_continuations must be positive")
    pixels = image.size
    required_bytes = pixels * pixels * np.dtype(np.float64).itemsize
    if (
        resolution.ancestry_memory_ceiling_bytes is not None
        and required_bytes > int(resolution.ancestry_memory_ceiling_bytes)
    ):
        raise MemoryError(
            f"dense source ancestry needs {required_bytes} bytes, above the "
            "numerical memory ceiling"
        )

    geometry = relation_transport_metric_2d(image)
    operator, operator_diagnostic = source_measure_operator(
        geometry["metric_xx"], geometry["metric_xy"], geometry["metric_yy"])
    observation_flat = image.ravel()
    ancestry = _exclude_target_identity(operator.toarray())
    state = relation_characteristic_measure_2d(image)[0].ravel()
    lower = float(np.min(observation_flat))
    upper = float(np.max(observation_flat))
    residual_action = float(np.mean((observation_flat - state) ** 2))
    records = []
    equilibrium = False
    for continuation in range(ceiling):
        residual = observation_flat - state
        prediction = relation_characteristic_measure_2d(
            residual.reshape(image.shape))[0].ravel()
        coefficient, local = _local_cross_predictive_coefficient(
            ancestry, residual, prediction)
        update = coefficient * prediction
        update_energy = float(np.mean(update * update))
        projection = float(np.mean(residual * update))
        if update_energy == 0.0 or projection <= 0.0:
            equilibrium = True
            records.append({
                "continuation": continuation,
                "accepted": False,
                "residual_action_before": residual_action,
                "residual_action_after": residual_action,
                "global_descent_coefficient": 0.0,
                **local,
            })
            break
        descent = min(1.0, projection / update_energy)
        candidate = np.clip(state + descent * update, lower, upper)
        candidate_action = float(np.mean((observation_flat - candidate) ** 2))
        numerical = np.finfo(float).eps * max(residual_action, 1.0)
        if candidate_action >= residual_action - numerical:
            equilibrium = True
            records.append({
                "continuation": continuation,
                "accepted": False,
                "residual_action_before": residual_action,
                "residual_action_after": residual_action,
                "global_descent_coefficient": descent,
                **local,
            })
            break
        records.append({
            "continuation": continuation,
            "accepted": True,
            "residual_action_before": residual_action,
            "residual_action_after": candidate_action,
            "global_descent_coefficient": descent,
            **local,
        })
        state = candidate
        residual_action = candidate_action
        ancestry = _exclude_target_identity(operator @ ancestry)

    ceiling_hit = len(records) == ceiling and not equilibrium
    return np.clip(state.reshape(image.shape), 0.0, 1.0), {
        "status": (
            "continuous source covariance equilibrium"
            if equilibrium
            else "numerical continuation ceiling reached; unresolved"
        ),
        "theory_status": "continuous-measure 2-D candidate; M4 gate pending",
        "residual_velocity": (
            "full-scale characteristic prediction with continuous-source "
            "held-out error authority"
        ),
        "accepted_continuations": int(sum(
            record["accepted"] for record in records)),
        "continuation_ceiling_hit": ceiling_hit,
        "final_residual_action": residual_action,
        "operator": operator_diagnostic,
        "continuations": records,
        "numerical_resolution": asdict(resolution),
        "dense_ancestry_bytes": required_bytes,
        "unresolved": [
            "relation metric still reads the target observation",
            "Selling stencil changes discretely under metric reduction",
            "dense exact ancestry has quadratic representation cost",
            "full tangent-sphere and Sasaki jet transport remain pending",
        ],
    }
