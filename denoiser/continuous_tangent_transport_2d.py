"""Common-physical-scale, target-free continuous tangent proposals in 2-D.

Every angular quadrature uses the same physical radii.  Off-grid ray samples
come from affine-exact cell coordinates.  Whenever ordinary bilinear sampling
would read the target observation, its coefficient is eliminated and the
remaining three cell corners are solved as an affine triangle.  This separates
angular refinement from radial refinement while retaining exact source
identity and the observation graph.
"""

from __future__ import annotations

from typing import Any
import math

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from .crossfit_characteristic_transport_2d import (
    crossfit_characteristic_population_2d,
)
from .witnessed_characteristic_transport_2d import (
    _crps_against_witness,
    _four_colour_leave_one_out_residual_crps,
    _lineage_covariance_authority,
    _source_influence_and_lineage,
    _validate,
    _weighted_median,
)


def uniform_projective_tangents(
    angular_count: int,
) -> tuple[tuple[float, float], ...]:
    """Return nested periodic trapezoidal nodes on the projective circle."""
    count = int(angular_count)
    if count < 4 or count & (count - 1):
        raise ValueError("angular count must be a power of two no smaller than four")
    return tuple(
        (math.sin(math.pi * index / count),
         math.cos(math.pi * index / count))
        for index in range(count)
    )


def _target_free_affine_sample(
    image: np.ndarray,
    offset_y: float,
    offset_x: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample one translated field with affine-exact zero target influence."""
    height, width = image.shape
    yy, xx = np.mgrid[:height, :width]
    py = yy + float(offset_y)
    px = xx + float(offset_x)
    valid = (0.0 <= py) & (py <= height - 1) & (0.0 <= px) & (px <= width - 1)
    y0 = np.clip(np.floor(py).astype(np.int64), 0, height - 2)
    x0 = np.clip(np.floor(px).astype(np.int64), 0, width - 2)
    y1 = y0 + 1
    x1 = x0 + 1
    a = px - x0
    b = py - y0
    source = np.stack((
        y0 * width + x0,
        y0 * width + x1,
        y1 * width + x0,
        y1 * width + x1,
    ), axis=-1)
    coefficient = np.stack((
        (1.0 - a) * (1.0 - b),
        a * (1.0 - b),
        (1.0 - a) * b,
        a * b,
    ), axis=-1)
    target = (yy * width + xx)[..., None]
    active_target = (source == target) & (np.abs(coefficient) > 0.0)
    remove = np.argmax(active_target, axis=-1)
    needs_removal = np.any(active_target, axis=-1)
    triangle = (
        np.stack((np.zeros_like(a), 1.0 - b, 1.0 - a, a + b - 1.0), axis=-1),
        np.stack((1.0 - b, np.zeros_like(a), b - a, a), axis=-1),
        np.stack((1.0 - a, a - b, np.zeros_like(a), b), axis=-1),
        np.stack((1.0 - a - b, a, b, np.zeros_like(a)), axis=-1),
    )
    for removed, replacement in enumerate(triangle):
        mask = needs_removal & (remove == removed)
        coefficient[mask] = replacement[mask]
    flat = image.reshape(-1)
    value = np.sum(coefficient * flat[source], axis=-1)
    return value, valid, source, coefficient


def continuous_tangent_proposals_2d(
    observation: np.ndarray,
    *,
    angular_count: int = 4,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Return common-radius first-jet particles over continuous tangents."""
    image = _validate(observation)
    height, width = image.shape
    tangents = uniform_projective_tangents(angular_count)
    maximum_radius = max(
        int(math.floor(0.5 * math.hypot(height - 1, width - 1))), 1)
    angular_weight = math.pi / len(tangents)
    magnitude = max(float(np.max(np.abs(image))), float(np.ptp(image)))
    floor = (
        np.finfo(float).tiny
        if magnitude == 0.0
        else math.sqrt(np.finfo(float).eps) * magnitude
    )
    predictions = []
    variations = []
    scale_conductances = []
    source_identities = []
    source_coefficients = []
    directional_derivatives = []
    tangent_vectors = []
    for dy, dx in tangents:
        for radius in range(1, maximum_radius + 1):
            minus_one = _target_free_affine_sample(
                image, -dy * radius, -dx * radius)
            plus_one = _target_free_affine_sample(
                image, dy * radius, dx * radius)
            minus_next = _target_free_affine_sample(
                image, -dy * (radius + 1), -dx * (radius + 1))
            plus_next = _target_free_affine_sample(
                image, dy * (radius + 1), dx * (radius + 1))
            minus_far = _target_free_affine_sample(
                image, -dy * (radius + 2), -dx * (radius + 2))
            plus_far = _target_free_affine_sample(
                image, dy * (radius + 2), dx * (radius + 2))
            minus_jet = minus_one[0] - minus_next[0]
            plus_jet = plus_next[0] - plus_one[0]
            minus_next_jet = minus_next[0] - minus_far[0]
            plus_next_jet = plus_far[0] - plus_next[0]
            characteristics = (
                (
                    0.5 * (minus_one[0] + plus_one[0]),
                    0.5 * np.abs(plus_jet - minus_jet),
                    minus_one[1] & plus_one[1]
                    & minus_next[1] & plus_next[1],
                    np.concatenate((minus_one[2], plus_one[2]), axis=-1),
                    np.concatenate((
                        0.5 * minus_one[3], 0.5 * plus_one[3]), axis=-1),
                    0.5 * (minus_jet + plus_jet),
                ),
                (
                    minus_one[0] + radius * minus_jet,
                    np.abs(minus_jet - minus_next_jet),
                    minus_one[1] & minus_next[1] & minus_far[1],
                    np.concatenate((minus_one[2], minus_next[2]), axis=-1),
                    np.concatenate((
                        (1.0 + radius) * minus_one[3],
                        -radius * minus_next[3]), axis=-1),
                    minus_jet,
                ),
                (
                    plus_one[0] - radius * plus_jet,
                    np.abs(plus_jet - plus_next_jet),
                    plus_one[1] & plus_next[1] & plus_far[1],
                    np.concatenate((plus_one[2], plus_next[2]), axis=-1),
                    np.concatenate((
                        (1.0 + radius) * plus_one[3],
                        -radius * plus_next[3]), axis=-1),
                    plus_jet,
                ),
            )
            for (
                prediction, variation, valid, source, coefficient,
                directional_derivative,
            ) in characteristics:
                predictions.append(prediction)
                variations.append(variation)
                scale_conductances.append(np.where(
                    valid, angular_weight / radius, 0.0))
                source_identities.append(source)
                source_coefficients.append(coefficient)
                directional_derivatives.append(directional_derivative)
                tangent_vectors.append((dy, dx))
    prediction = np.stack(predictions, axis=-1)
    variation = np.stack(variations, axis=-1)
    scale_conductance = np.stack(scale_conductances, axis=-1)
    if np.any(np.sum(scale_conductance, axis=-1) <= 0.0):
        raise RuntimeError("continuous tangent proposal left a point unsupported")
    source_identity = np.stack(source_identities, axis=-2)
    source_coefficient = np.stack(source_coefficients, axis=-2)
    target = np.arange(image.size).reshape(image.shape + (1, 1))
    maximum_self = float(np.max(np.abs(np.where(
        source_identity == target, source_coefficient, 0.0))))
    return {
        "prediction": prediction,
        "variation": variation,
        "scale_conductance": scale_conductance,
        "source_identity": source_identity,
        "source_coefficient": source_coefficient,
        "directional_derivative": np.stack(
            directional_derivatives, axis=-1),
        "tangent": np.asarray(tangent_vectors, dtype=np.float64),
    }, {
        "angular_count": len(tangents),
        "angular_weight": angular_weight,
        "angular_mass": angular_weight * len(tangents),
        "maximum_radius": maximum_radius,
        "proposal_count": int(prediction.shape[-1]),
        "maximum_target_self_coefficient": maximum_self,
        "scale_measure": "common integer physical radii under ds/s",
        "boundary_condition": "no reflection; invalid rays carry zero mass",
        "interpolation": "target-free affine cell coordinates",
        "jet_transport": (
            "unit local directional jet parallel-transported across radius"
        ),
        "numerical_floor": floor,
    }


def continuous_tangent_signal_population_2d(
    observation: np.ndarray,
    *,
    angular_count: int = 4,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Return the target-independent signal law on continuous tangents."""
    image = _validate(observation)
    proposal, proposal_diagnostic = continuous_tangent_proposals_2d(
        image, angular_count=angular_count)
    witness, witness_diagnostic = crossfit_characteristic_population_2d(image)
    signal_score, self_action = _crps_against_witness(
        proposal["prediction"], witness["prediction"], witness["mass"])
    magnitude = max(float(np.max(np.abs(image))), float(np.ptp(image)))
    floor = (
        np.finfo(float).tiny
        if magnitude == 0.0
        else math.sqrt(np.finfo(float).eps) * magnitude
    )
    conductance = proposal["scale_conductance"] / np.maximum(
        proposal["variation"] + signal_score, floor)
    mass = conductance / np.sum(conductance, axis=-1, keepdims=True)
    return {
        "prediction": proposal["prediction"],
        "mass": mass,
        "crps": signal_score,
        "variation": proposal["variation"],
        "scale_conductance": proposal["scale_conductance"],
        "source_identity": proposal["source_identity"],
        "source_coefficient": proposal["source_coefficient"],
        "directional_derivative": proposal["directional_derivative"],
        "tangent": proposal["tangent"],
    }, {
        "status": "target-independent common-scale continuous tangent signal law",
        "proposal": proposal_diagnostic,
        "witness": witness_diagnostic,
        "mean_witness_self_action": float(np.mean(self_action)),
        "target_identity_excluded": (
            proposal_diagnostic["maximum_target_self_coefficient"] == 0.0
        ),
    }


def continuous_tangent_joint_population_2d(
    observation: np.ndarray,
    *,
    angular_count: int = 4,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build the joint signal/residual law on common-scale tangent particles."""
    image = _validate(observation)
    signal, signal_diagnostic = continuous_tangent_signal_population_2d(
        image, angular_count=angular_count)
    witness, witness_diagnostic = crossfit_characteristic_population_2d(image)
    signal_score = signal["crps"]
    residual = image[..., None] - signal["prediction"]
    residual_score = _four_colour_leave_one_out_residual_crps(
        image, witness, residual)
    magnitude = max(float(np.max(np.abs(image))), float(np.ptp(image)))
    floor = (
        np.finfo(float).tiny
        if magnitude == 0.0
        else math.sqrt(np.finfo(float).eps) * magnitude
    )
    action = signal["variation"] + signal_score + residual_score
    conductance = signal["scale_conductance"] / np.maximum(action, floor)
    mass = conductance / np.sum(conductance, axis=-1, keepdims=True)
    return {
        "signal": signal["prediction"],
        "residual": residual,
        "mass": mass,
        "prior_mass": signal["mass"],
        "joint_action": action,
        "variation": signal["variation"],
        "scale_conductance": signal["scale_conductance"],
        "signal_crps": signal_score,
        "residual_crps": residual_score,
        "source_identity": signal["source_identity"],
        "source_coefficient": signal["source_coefficient"],
        "directional_derivative": signal["directional_derivative"],
        "tangent": signal["tangent"],
    }, {
        "status": "joint law on common-scale continuous tangent quadrature",
        "proposal": signal_diagnostic["proposal"],
        "witness": witness_diagnostic,
        "mean_witness_self_action": signal_diagnostic[
            "mean_witness_self_action"],
        "observation_graph_maximum_error": float(np.max(np.abs(
            signal["prediction"] + residual - image[..., None]))),
        "target_identity_excluded": (
            signal_diagnostic["target_identity_excluded"]
        ),
        "theory_status": "continuous tangent convergence experiment; not promoted",
    }


def continuous_tangent_joint_measure_2d(
    observation: np.ndarray,
    *,
    barycenter: str = "mean",
    angular_count: int = 4,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read the common-scale joint law by its W2 or W1 signal barycenter."""
    population, diagnostic = continuous_tangent_joint_population_2d(
        observation, angular_count=angular_count)
    if barycenter == "mean":
        readout = np.sum(population["mass"] * population["signal"], axis=-1)
    elif barycenter == "median":
        readout = _weighted_median(population["signal"], population["mass"])
    else:
        raise ValueError("barycenter must be 'mean' or 'median'")
    return readout, {**diagnostic, "barycenter": barycenter}


def continuous_tangent_jet_field_2d(
    population: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Return the posterior least-squares spatial jet at every target."""
    mass = np.asarray(population["mass"], dtype=np.float64)
    derivative = np.asarray(
        population["directional_derivative"], dtype=np.float64)
    tangent = np.asarray(population["tangent"], dtype=np.float64)
    if derivative.shape != mass.shape or tangent.shape != (mass.shape[-1], 2):
        raise ValueError("joint population has inconsistent tangent state")
    dy = tangent[:, 0]
    dx = tangent[:, 1]
    qxx = np.sum(mass * dx * dx, axis=-1)
    qxy = np.sum(mass * dx * dy, axis=-1)
    qyy = np.sum(mass * dy * dy, axis=-1)
    bx = np.sum(mass * derivative * dx, axis=-1)
    by = np.sum(mass * derivative * dy, axis=-1)
    determinant = qxx * qyy - qxy * qxy
    if np.any(determinant <= np.finfo(float).tiny):
        raise RuntimeError("tangent population does not span the spatial plane")
    gradient_x = (qyy * bx - qxy * by) / determinant
    gradient_y = (qxx * by - qxy * bx) / determinant
    residual = (
        derivative
        - gradient_x[..., None] * dx
        - gradient_y[..., None] * dy
    )
    return gradient_x, gradient_y, {
        "mean_directional_jet_residual": float(np.mean(
            mass * residual * residual)),
        "minimum_tangent_tensor_determinant": float(np.min(determinant)),
        "mean_tangent_tensor_determinant": float(np.mean(determinant)),
    }


def continuous_tangent_jet_particles_2d(
    population: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, float | int]]:
    """Integrate angular covectors into full-jet scale/path particles."""
    mass = np.asarray(population["mass"], dtype=np.float64)
    signal_key = "signal" if "signal" in population else "prediction"
    signal = np.asarray(population[signal_key], dtype=np.float64)
    derivative = np.asarray(
        population["directional_derivative"], dtype=np.float64)
    tangent = np.asarray(population["tangent"], dtype=np.float64)
    if signal.shape != mass.shape or derivative.shape != mass.shape:
        raise ValueError("signal, mass, and directional jets must align")
    if tangent.shape != (mass.shape[-1], 2):
        raise ValueError("tangent catalogue must index the particle axis")
    directions, first, inverse = np.unique(
        tangent, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first)
    directions = directions[order]
    direction_indices = [
        np.flatnonzero(inverse == index)
        for index in order
    ]
    per_direction = {indices.size for indices in direction_indices}
    if len(per_direction) != 1:
        raise ValueError("every tangent must carry the same particle catalogue")
    catalogue_size = per_direction.pop()
    index = np.stack(direction_indices, axis=0)
    selected_mass = mass[..., index]
    selected_signal = signal[..., index]
    selected_derivative = derivative[..., index]
    dy = directions[:, 0].reshape((1, 1, -1, 1))
    dx = directions[:, 1].reshape((1, 1, -1, 1))
    group_mass = np.sum(selected_mass, axis=-2)
    qxx = np.sum(selected_mass * dx * dx, axis=-2)
    qxy = np.sum(selected_mass * dx * dy, axis=-2)
    qyy = np.sum(selected_mass * dy * dy, axis=-2)
    bx = np.sum(selected_mass * selected_derivative * dx, axis=-2)
    by = np.sum(selected_mass * selected_derivative * dy, axis=-2)
    determinant = qxx * qyy - qxy * qxy
    scale = np.maximum(qxx + qyy, np.finfo(float).tiny)
    valid = determinant > np.finfo(float).eps * scale * scale
    gradient_x = np.zeros_like(group_mass)
    gradient_y = np.zeros_like(group_mass)
    gradient_x[valid] = (
        qyy[valid] * bx[valid] - qxy[valid] * by[valid]
    ) / determinant[valid]
    gradient_y[valid] = (
        qxx[valid] * by[valid] - qxy[valid] * bx[valid]
    ) / determinant[valid]
    value = np.divide(
        np.sum(selected_mass * selected_signal, axis=-2),
        group_mass,
        out=np.zeros_like(group_mass),
        where=group_mass > 0.0,
    )
    particle_mass = np.where(valid, group_mass, 0.0)
    total = np.sum(particle_mass, axis=-1, keepdims=True)
    coarsened = total[..., 0] <= 0.0
    if np.any(coarsened):
        flat_dx = tangent[:, 1]
        flat_dy = tangent[:, 0]
        global_qxx = np.sum(mass * flat_dx * flat_dx, axis=-1)
        global_qxy = np.sum(mass * flat_dx * flat_dy, axis=-1)
        global_qyy = np.sum(mass * flat_dy * flat_dy, axis=-1)
        global_bx = np.sum(mass * derivative * flat_dx, axis=-1)
        global_by = np.sum(mass * derivative * flat_dy, axis=-1)
        global_determinant = (
            global_qxx * global_qyy - global_qxy * global_qxy)
        if np.any(global_determinant[coarsened] <= np.finfo(float).tiny):
            raise RuntimeError("complete angular jet law does not span the plane")
        global_gradient_x = (
            global_qyy * global_bx - global_qxy * global_by
        ) / global_determinant
        global_gradient_y = (
            global_qxx * global_by - global_qxy * global_bx
        ) / global_determinant
        global_value = np.sum(mass * signal, axis=-1)
        particle_mass[coarsened] = 0.0
        particle_mass[coarsened, 0] = 1.0
        value[coarsened, 0] = global_value[coarsened]
        gradient_x[coarsened, 0] = global_gradient_x[coarsened]
        gradient_y[coarsened, 0] = global_gradient_y[coarsened]
        total = np.sum(particle_mass, axis=-1, keepdims=True)
    particle_mass /= total
    return {
        "signal": value,
        "mass": particle_mass,
        "gradient_x": gradient_x,
        "gradient_y": gradient_y,
    }, {
        "projective_tangent_count": int(len(directions)),
        "scale_characteristic_particle_count": int(catalogue_size),
        "minimum_valid_particle_count": int(np.min(np.sum(valid, axis=-1))),
        "mean_valid_particle_count": float(np.mean(np.sum(valid, axis=-1))),
        "coarsened_boundary_fraction": float(np.mean(coarsened)),
        "maximum_mass_error": float(np.max(np.abs(
            np.sum(particle_mass, axis=-1) - 1.0))),
    }


def _integrable_jet_projection(
    gradient_x: np.ndarray,
    gradient_y: np.ndarray,
    mean_value: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Hodge-project a discrete jet field onto exact scalar gradients."""
    gx = np.asarray(gradient_x, dtype=np.float64)
    gy = np.asarray(gradient_y, dtype=np.float64)
    if gx.ndim != 2 or gy.shape != gx.shape:
        raise ValueError("jet components must be aligned 2-D fields")
    height, width = gx.shape
    pixels = height * width
    rows = []
    columns = []
    values = []
    target = []
    edge = 0
    for y in range(height):
        for x in range(width - 1):
            left = y * width + x
            rows.extend((edge, edge))
            columns.extend((left, left + 1))
            values.extend((-1.0, 1.0))
            target.append(0.5 * (gx[y, x] + gx[y, x + 1]))
            edge += 1
    for y in range(height - 1):
        for x in range(width):
            top = y * width + x
            rows.extend((edge, edge))
            columns.extend((top, top + width))
            values.extend((-1.0, 1.0))
            target.append(0.5 * (gy[y, x] + gy[y + 1, x]))
            edge += 1
    incidence = sparse.coo_matrix(
        (values, (rows, columns)), shape=(edge, pixels)).tocsr()
    numerical = math.sqrt(np.finfo(float).eps)
    solved = sparse_linalg.lsmr(
        incidence,
        np.asarray(target, dtype=np.float64),
        atol=numerical,
        btol=numerical,
        maxiter=4 * pixels,
    )
    field = solved[0].reshape(gx.shape)
    field += float(mean_value) - float(np.mean(field))
    mismatch = incidence @ solved[0] - np.asarray(target)
    return field, {
        "jet_projection_iterations": int(solved[2]),
        "jet_projection_stop_code": int(solved[1]),
        "jet_projection_residual_norm": float(np.linalg.norm(mismatch)),
        "jet_projection_mean_error": float(np.mean(field)) - float(mean_value),
    }


def continuous_tangent_jet_projection_2d(
    observation: np.ndarray,
    *,
    angular_count: int = 16,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read the joint law through its integrable first-jet barycenter."""
    image = _validate(observation)
    population, diagnostic = continuous_tangent_joint_population_2d(
        image, angular_count=angular_count)
    scalar = _weighted_median(population["signal"], population["mass"])
    gradient_x, gradient_y, jet_diagnostic = continuous_tangent_jet_field_2d(
        population)
    field, projection_diagnostic = _integrable_jet_projection(
        gradient_x, gradient_y, float(np.mean(scalar)))
    return np.clip(field, 0.0, 1.0), {
        "status": "integrable continuous-tangent jet projection",
        "angular_count": int(angular_count),
        "scalar_gauge": "mean of joint W1 barycenter",
        "joint_measure": diagnostic,
        **jet_diagnostic,
        **projection_diagnostic,
        "theory_status": "parameter-free jet readout experiment; not promoted",
    }


def denoise_continuous_tangent_lineage_covariance_2d(
    observation: np.ndarray,
    *,
    angular_count: int = 16,
    maximum_continuations: int = 32,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Continue the convergent tangent law under conserved-source covariance."""
    image = _validate(observation)
    ceiling = int(maximum_continuations)
    if ceiling < 1:
        raise ValueError("maximum continuations must be positive")
    state, initial = continuous_tangent_joint_measure_2d(
        image, barycenter="median", angular_count=angular_count)
    lower = float(np.min(image))
    upper = float(np.max(image))
    action = float(np.mean((image - state) ** 2))
    records = []
    equilibrium = False
    for continuation in range(ceiling):
        residual = image - state
        population, measure = continuous_tangent_joint_population_2d(
            residual, angular_count=angular_count)
        prediction = np.sum(
            population["mass"] * population["signal"], axis=-1)
        held_out_prediction = np.sum(
            population["prior_mass"] * population["signal"], axis=-1)
        _influence, lineage = _source_influence_and_lineage(
            population["prior_mass"],
            population["source_identity"],
            population["source_coefficient"],
        )
        authority, covariance = _lineage_covariance_authority(
            lineage, residual, held_out_prediction)
        authority = authority.reshape(image.shape)
        update = authority * prediction
        update_energy = float(np.mean(update * update))
        projection = float(np.mean(residual * update))
        local = {
            **covariance,
            "lineage_row_mass_maximum_error": float(np.max(np.abs(
                np.sum(lineage, axis=1) - 1.0))),
            "maximum_target_self_lineage": float(np.max(np.abs(
                np.diag(lineage)))),
        }
        if update_energy == 0.0 or projection <= 0.0:
            equilibrium = True
            records.append({
                "continuation": continuation,
                "accepted": False,
                "residual_action_before": action,
                "residual_action_after": action,
                "global_descent_coefficient": 0.0,
                **local,
            })
            break
        descent = min(1.0, projection / update_energy)
        candidate = np.clip(state + descent * update, lower, upper)
        candidate_action = float(np.mean((image - candidate) ** 2))
        numerical = np.finfo(float).eps * max(action, 1.0)
        if candidate_action >= action - numerical:
            equilibrium = True
            records.append({
                "continuation": continuation,
                "accepted": False,
                "residual_action_before": action,
                "residual_action_after": action,
                "global_descent_coefficient": descent,
                **local,
            })
            break
        records.append({
            "continuation": continuation,
            "accepted": True,
            "residual_action_before": action,
            "residual_action_after": candidate_action,
            "global_descent_coefficient": descent,
            **local,
            "measure": measure,
        })
        state = candidate
        action = candidate_action
    ceiling_hit = len(records) == ceiling and not equilibrium
    return np.clip(state, 0.0, 1.0), {
        "status": (
            "continuous tangent lineage covariance equilibrium"
            if equilibrium else "continuation ceiling reached; unresolved"
        ),
        "angular_count": int(angular_count),
        "angular_resolution_status": (
            "numerical tangent quadrature; 4/8/16/32 convergence measured"
        ),
        "initial_measure": initial,
        "accepted_continuations": int(sum(row["accepted"] for row in records)),
        "continuation_ceiling_hit": ceiling_hit,
        "final_residual_action": action,
        "continuations": records,
        "authority_law": (
            "positive debiased residual-prediction covariance under prior lineage"
        ),
        "theory_status": "continuous tangent covariance experiment; not promoted",
    }
