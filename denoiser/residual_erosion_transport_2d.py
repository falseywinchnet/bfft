"""Cavity-certified erosion and re-entry of a denoiser residual.

This experiment starts from the continuous-scale phase posterior ``x`` and
its exact residual ``r=y-x``.  It does not call the residual noise.  Instead,
it asks whether a residual correction is approximately explained by the
posterior's normalized Selling curvature.

The regression is evaluated through two target cavities.  First, posterior
curvature is replaced by its off-diagonal Selling-neighbour prediction.
Second, the residual/curvature coefficient and its complete second moment are
estimated only from off-diagonal neighbours.  The Schur complement is the
unexplained bounded innovation; only positive covariance in excess of that
innovation has authority.  A screened Selling step erodes the admitted
correction before it is recomposed.  The exact quadratic descent coefficient
then contracts observation residual action, and continuation stops when the
next relation has nonpositive action projection.

This is a research posterior, not a promoted denoiser.  In particular, the
screened routing of independently admitted neighbouring corrections has not
yet been given a complete source-lineage coverage proof.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse

from .causal_scale_transport_2d import (
    _screened_transport,
    causal_scale_transport_observation_2d,
)
from .continual_eikonal_noise_transport_2d import (
    _continual_flux_laplacian,
    continual_transport_metric,
)
from .witnessed_characteristic_transport_2d import _validate


def _off_diagonal_conductance(
    laplacian: sparse.csr_matrix,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    degree = np.asarray(laplacian.diagonal(), dtype=np.float64)
    conductance = (sparse.diags(degree) - laplacian).tocsr()
    conductance.setdiag(0.0)
    conductance.eliminate_zeros()
    mass = np.asarray(conductance.sum(axis=1)).ravel()
    return conductance, mass


def _cavity_residual_relation(
    state: np.ndarray,
    residual: np.ndarray,
    laplacian: sparse.csr_matrix,
    maximum_degree: float,
) -> tuple[np.ndarray, dict[str, np.ndarray | float]]:
    """Return the target-excluded raw correction and its Z-style moments."""
    x = np.asarray(state, dtype=np.float64)
    r = np.asarray(residual, dtype=np.float64)
    if x.shape != r.shape:
        raise ValueError("state and residual must align")
    conductance, mass = _off_diagonal_conductance(laplacian)
    tiny = np.finfo(float).tiny
    flat_state = x.reshape(-1)
    flat_residual = r.reshape(-1)
    if maximum_degree <= 0.0:
        posterior_curvature = np.zeros_like(flat_state)
    else:
        posterior_curvature = np.asarray(
            laplacian @ flat_state) / float(maximum_degree)
    cavity_curvature = np.divide(
        conductance @ posterior_curvature,
        mass,
        out=np.zeros_like(mass),
        where=mass > tiny,
    )
    cross_moment = np.divide(
        conductance @ (flat_residual * cavity_curvature),
        mass,
        out=np.zeros_like(mass),
        where=mass > tiny,
    )
    curvature_moment = np.divide(
        conductance @ (cavity_curvature * cavity_curvature),
        mass,
        out=np.zeros_like(mass),
        where=mass > tiny,
    )
    residual_moment = np.divide(
        conductance @ (flat_residual * flat_residual),
        mass,
        out=np.zeros_like(mass),
        where=mass > tiny,
    )
    positive_cross = np.maximum(cross_moment, 0.0)
    coefficient = np.divide(
        positive_cross,
        curvature_moment,
        out=np.zeros_like(positive_cross),
        where=curvature_moment > tiny,
    )
    explained_action = np.divide(
        positive_cross * positive_cross,
        curvature_moment * residual_moment,
        out=np.zeros_like(positive_cross),
        where=(curvature_moment * residual_moment) > tiny,
    )
    explained_action = np.clip(explained_action, 0.0, 1.0)
    schur_innovation = np.maximum(
        residual_moment
        - np.divide(
            positive_cross * positive_cross,
            curvature_moment,
            out=np.zeros_like(positive_cross),
            where=curvature_moment > tiny,
        ),
        0.0,
    )
    raw_correction = (
        explained_action * coefficient * cavity_curvature
    ).reshape(x.shape)
    return raw_correction, {
        "cavity_curvature": cavity_curvature.reshape(x.shape),
        "cross_moment": cross_moment.reshape(x.shape),
        "curvature_moment": curvature_moment.reshape(x.shape),
        "residual_moment": residual_moment.reshape(x.shape),
        "explained_action": explained_action.reshape(x.shape),
        "schur_innovation": schur_innovation.reshape(x.shape),
        "mean_positive_relation_authority": float(np.mean(explained_action)),
        "positive_relation_fraction": float(np.mean(positive_cross > 0.0)),
        "mean_schur_innovation": float(np.mean(schur_innovation)),
        "off_diagonal_edge_count": int(conductance.nnz // 2),
    }


def denoise_cavity_residual_erosion_2d(
    observation: np.ndarray,
    *,
    initial_state: np.ndarray | None = None,
    continuation_guard: int = 32,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Erode and re-enter only action-contracting cavity residual relations."""
    image = _validate(observation)
    guard = int(continuation_guard)
    if guard < 1:
        raise ValueError("continuation guard must be positive")
    if initial_state is None:
        _base, _base_residual, scale = (
            causal_scale_transport_observation_2d(image))
        state = np.asarray(
            scale["readouts"]["phase_susceptibility"],
            dtype=np.float64,
        ).copy()
        base_diagnostic: dict[str, Any] = scale
    else:
        state = _validate(initial_state).copy()
        if state.shape != image.shape:
            raise ValueError("initial state must align with observation")
        base_diagnostic = {"status": "caller-supplied initial state"}

    lower = float(np.min(image))
    upper = float(np.max(image))
    residual = image - state
    action = float(np.mean(residual * residual))
    records = []
    equilibrium = False
    for continuation in range(guard):
        metric = continual_transport_metric(state, residual * residual)
        laplacian, _markov, stencil = _continual_flux_laplacian(
            metric, np.ones_like(image))
        maximum_degree = float(stencil["maximum_degree"])
        raw_correction, relation = _cavity_residual_relation(
            state, residual, laplacian, maximum_degree)
        correction = (
            _screened_transport(
                laplacian,
                1.0 / maximum_degree,
                raw_correction[None, ...],
            )[0]
            if maximum_degree > 0.0 else raw_correction
        )
        update_action = float(np.mean(correction * correction))
        projection = float(np.mean(residual * correction))
        local = {
            "continuation": int(continuation),
            "residual_action_before": action,
            "update_action": update_action,
            "residual_update_projection": projection,
            "raw_correction_rms": float(np.sqrt(np.mean(
                raw_correction * raw_correction))),
            "screened_correction_rms": float(np.sqrt(update_action)),
            "stencil_maximum_degree": maximum_degree,
            "mean_positive_relation_authority": relation[
                "mean_positive_relation_authority"],
            "positive_relation_fraction": relation[
                "positive_relation_fraction"],
            "mean_schur_innovation": relation["mean_schur_innovation"],
        }
        if update_action <= np.finfo(float).tiny or projection <= 0.0:
            equilibrium = True
            records.append({
                **local,
                "accepted": False,
                "descent_coefficient": 0.0,
                "residual_action_after": action,
            })
            break
        descent = float(np.clip(projection / update_action, 0.0, 1.0))
        candidate = np.clip(state + descent * correction, lower, upper)
        candidate_residual = image - candidate
        candidate_action = float(np.mean(
            candidate_residual * candidate_residual))
        numerical = np.finfo(float).eps * max(action, 1.0)
        if candidate_action >= action - numerical:
            equilibrium = True
            records.append({
                **local,
                "accepted": False,
                "descent_coefficient": descent,
                "residual_action_after": action,
            })
            break
        records.append({
            **local,
            "accepted": True,
            "descent_coefficient": descent,
            "residual_action_after": candidate_action,
        })
        state = candidate
        residual = candidate_residual
        action = candidate_action

    accepted = sum(bool(record["accepted"]) for record in records)
    ceiling_hit = len(records) == guard and not equilibrium
    exact_residual = image - state
    return state, {
        "status": (
            "cavity residual relation equilibrium"
            if equilibrium
            else "continuation guard reached; unresolved"
        ),
        "theory_status": (
            "target-excluded relation authority and exact action contraction; "
            "screened source-lineage coverage remains unproved"
        ),
        "accepted_continuations": int(accepted),
        "continuation_guard_hit": bool(ceiling_hit),
        "initial_residual_action": float(records[0][
            "residual_action_before"] if records else action),
        "final_residual_action": float(np.mean(
            exact_residual * exact_residual)),
        "observation_graph_maximum_error": float(np.max(np.abs(
            state + exact_residual - image))),
        "continuations": tuple(records),
        "base": base_diagnostic,
        "laws": {
            "relation": (
                "off-diagonal Selling cavity regression of residual on "
                "posterior generator curvature"
            ),
            "uncertainty": (
                "positive transported covariance versus Schur innovation"
            ),
            "erosion": (
                "one operator-normalized screened Selling transport of each "
                "admitted correction"
            ),
            "stop": (
                "nonpositive residual/update projection or failed residual "
                "action contraction"
            ),
        },
    }


__all__ = ["denoise_cavity_residual_erosion_2d"]
