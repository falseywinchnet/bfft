"""Analytical bloom on explicit support--manifold proposal topology.

The support/manifold Gram used proposal features ``sqrt(w_ms)(e_s + p_m)``.
Its support term necessarily contributes a full-rank identity diagonal, which
obscures cross-resolution topology.  Here the same complete proposal catalog
is retained as the measured cross-incidence operator

    C = P.T @ W,

where ``P[m,r]`` is region ``r``'s participation in bounded manifold ``m``
and ``W[m,s]`` is the measured support weight.  ``C + C.T`` transports between
support and manifold members.  Only its diagonal is removed: a self-loop does
not transport state, while every proposal still contributes all off-diagonal
incidences.

Symmetric degree normalization gives ``S``.  The unit combinatorial heat
kernel ``exp(S-I)`` is evaluated spectrally in one closed form.  It contains
all path orders with factorial measure and introduces no merge threshold,
chosen seed, expected object count, or iterative stopping rule.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse


def _normalize_kernel(kernel: np.ndarray) -> np.ndarray:
    value = 0.5 * (
        np.asarray(kernel, dtype=np.float64)
        + np.asarray(kernel, dtype=np.float64).T
    )
    diagonal = np.maximum(np.diag(value), 0.0)
    denominator = np.sqrt(diagonal[:, None] * diagonal[None, :])
    return np.divide(
        value, denominator, out=np.zeros_like(value),
        where=denominator > 1e-30)


def build_proposal_connection(
    participation: sparse.spmatrix,
    support_manifold_weight: np.ndarray,
) -> dict[str, np.ndarray]:
    """Construct the exact normalized off-diagonal proposal connection."""
    member = participation.tocsr().astype(np.float64)
    weight = np.asarray(support_manifold_weight, dtype=np.float64)
    if member.shape != weight.shape:
        raise ValueError("manifold participation and support weights disagree")
    cross = np.asarray(member.T @ weight, dtype=np.float64)
    adjacency = cross + cross.T
    self_measure = np.diag(adjacency).copy()
    np.fill_diagonal(adjacency, 0.0)
    adjacency = np.maximum(0.5 * (adjacency + adjacency.T), 0.0)
    degree = np.sum(adjacency, axis=1)
    inverse = np.divide(
        1.0, np.sqrt(degree), out=np.zeros_like(degree), where=degree > 0.0)
    normalized = inverse[:, None] * adjacency * inverse[None, :]
    return {
        "cross_incidence": cross,
        "adjacency": adjacency,
        "self_measure": self_measure,
        "degree": degree,
        "normalized_connection": 0.5 * (normalized + normalized.T),
    }


def analytical_proposal_bloom(
    normalized_connection: np.ndarray,
    base_kernel: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Evaluate proposal heat and optional transported base kernel exactly."""
    connection = 0.5 * (
        np.asarray(normalized_connection, dtype=np.float64)
        + np.asarray(normalized_connection, dtype=np.float64).T
    )
    if connection.ndim != 2 or connection.shape[0] != connection.shape[1]:
        raise ValueError("normalized connection must be square")
    eigenvalue, eigenvector = np.linalg.eigh(connection)
    # Numerical normalization noise can only move the spectrum infinitesimally
    # outside the normalized-adjacency interval.
    eigenvalue = np.clip(eigenvalue, -1.0, 1.0)
    half_multiplier = np.exp(0.5 * (eigenvalue - 1.0))
    half_heat = (eigenvector * half_multiplier[None, :]) @ eigenvector.T
    heat = half_heat @ half_heat
    result = {
        "connection_eigenvalue": eigenvalue,
        "heat_kernel": _normalize_kernel(heat),
    }
    if base_kernel is not None:
        base = np.asarray(base_kernel, dtype=np.float64)
        if base.shape != connection.shape:
            raise ValueError("base kernel must match proposal connection")
        transported = half_heat @ base @ half_heat
        result["transported_base_kernel"] = _normalize_kernel(transported)
    return result


def summarize_proposal_connection(connection: dict) -> dict:
    adjacency = np.asarray(connection["adjacency"], dtype=np.float64)
    degree = np.asarray(connection["degree"], dtype=np.float64)
    nonzero = adjacency > 0.0
    return {
        "directed_cross_incidences": int(np.count_nonzero(nonzero)),
        "isolated_regions": int(np.count_nonzero(degree == 0.0)),
        "degree_quantiles": [
            float(value) for value in np.quantile(
                degree, (0.0, 0.25, 0.5, 0.75, 1.0))
        ] if len(degree) else [0.0] * 5,
    }
