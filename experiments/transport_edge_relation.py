#!/usr/bin/env python3
"""Centered relational information carried by literal transport edges.

Let phi_i be the phase-preserving scattering coordinate of support fragment i,
and let m_i be its image area.  We form the centered node covariance

    G = E_m[(phi - mean(phi))(phi - mean(phi))^T]

and the symmetric joint moment J of feature pairs joined by literal support
interfaces.  Because the features are centered, J is the empirical edge-joint
measure minus the corresponding independent product contribution.

The whitened operator B = G^(-1/2) J G^(-1/2) therefore asks a precise
question: which patterns co-occur across transport support more (positive) or
less (negative) than their prevalence alone predicts?

All eigenmodes are retained.  Their signed Krein embedding gives a bounded
relation score without selecting candidates, modes, or object rules.
"""

from __future__ import annotations

import numpy as np

from experiments.transport_graph_scattering import (
    transport_graph_scattering,
)


def _robust_columns(value: np.ndarray) -> np.ndarray:
    x = np.asarray(value, dtype=np.float64)
    center = np.median(x, axis=0)
    scale = np.maximum(
        1.4826 * np.median(np.abs(x - center), axis=0),
        1e-8,
    )
    return np.clip((x - center) / scale, -8.0, 8.0)


def transport_edge_relation(
    objects: dict,
    *,
    scattering: dict | None = None,
    scales: int = 3,
    whitening_floor: float = 1e-7,
) -> dict[str, np.ndarray | float | int]:
    """Construct the complete centered edge-relation spectrum."""
    graph = objects["graph"]
    if scattering is None or int(scattering["scales"]) != int(scales):
        scattering = transport_graph_scattering(objects, scales=scales)
    feature = _robust_columns(scattering["descriptor"])
    area = np.asarray(graph["area"], dtype=np.float64)
    mass = area / max(float(np.sum(area)), 1e-30)
    mean = np.sum(mass[:, None] * feature, axis=0)
    centered = feature - mean
    covariance = centered.T @ (mass[:, None] * centered)
    covariance = 0.5 * (covariance + covariance.T)

    edge = graph["edge"]
    first = np.asarray(edge["first"], dtype=np.int32)
    second = np.asarray(edge["second"], dtype=np.int32)
    length = np.asarray(
        edge.get("length", np.ones(len(first))), dtype=np.float64)
    edge_mass = length / max(2.0 * float(np.sum(length)), 1e-30)
    joint = (
        centered[first].T @ (edge_mass[:, None] * centered[second])
        + centered[second].T @ (edge_mass[:, None] * centered[first])
    )
    joint = 0.5 * (joint + joint.T)

    covariance_value, covariance_vector = np.linalg.eigh(covariance)
    spectral_ceiling = max(float(
        covariance_value.max(initial=0.0)), 0.0)
    floor = max(
        spectral_ceiling * max(float(whitening_floor), 0.0),
        1e-10,
    )
    inverse_sqrt = (
        covariance_vector
        * (1.0 / np.sqrt(np.maximum(covariance_value, floor)))
    ) @ covariance_vector.T
    operator = inverse_sqrt @ joint @ inverse_sqrt
    operator = 0.5 * (operator + operator.T)
    relation_value, relation_vector = np.linalg.eigh(operator)

    # |B| supplies the positive norm; sign(B) supplies association polarity.
    whitened_coordinates = centered @ inverse_sqrt @ relation_vector
    coordinates = (
        whitened_coordinates
        * np.sqrt(np.abs(relation_value))[None, :]
    )
    norm = np.linalg.norm(coordinates, axis=1)
    return {
        "scales": int(scales),
        "feature_count": int(feature.shape[1]),
        "whitening_floor": float(floor),
        "covariance_eigenvalue": covariance_value,
        "relation_eigenvalue": relation_value,
        "relation_sign": np.sign(relation_value),
        "whitened_coordinates": np.ascontiguousarray(
            whitened_coordinates),
        "coordinates": np.ascontiguousarray(coordinates),
        "norm": norm,
    }


def signed_relation_field(
    relation: dict,
    anchor_cell: int,
    *,
    order: int = 1,
) -> np.ndarray:
    """Return normalized positive/negative association to one fragment."""
    relation_order = max(int(order), 0)
    base = np.asarray(
        relation["whitened_coordinates"], dtype=np.float64)
    eigenvalue = np.asarray(
        relation["relation_eigenvalue"], dtype=np.float64)
    coordinate = (
        base * np.abs(eigenvalue)[None, :] ** (0.5 * relation_order)
    )
    sign = np.sign(eigenvalue) ** relation_order
    norm = np.linalg.norm(coordinate, axis=1)
    anchor = int(np.clip(anchor_cell, 0, max(len(coordinate) - 1, 0)))
    score = (coordinate * sign[None, :]) @ coordinate[anchor]
    score /= np.maximum(norm * norm[anchor], 1e-30)
    return np.clip(score, -1.0, 1.0)


def aggregate_signed_relations(
    relation: dict,
    owner: np.ndarray,
    weight: np.ndarray,
    *,
    order: int = 1,
) -> np.ndarray:
    """Aggregate the same signed embedding over any evaluation partition."""
    label = np.asarray(owner, dtype=np.int32)
    value = np.asarray(weight, dtype=np.float64)
    relation_order = max(int(order), 0)
    base = np.asarray(
        relation["whitened_coordinates"], dtype=np.float64)
    eigenvalue = np.asarray(
        relation["relation_eigenvalue"], dtype=np.float64)
    coordinate = (
        base * np.abs(eigenvalue)[None, :] ** (0.5 * relation_order)
    )
    sign = np.sign(eigenvalue) ** relation_order
    count = int(label.max(initial=-1)) + 1
    total = np.bincount(label, weights=value, minlength=count)
    mean = np.column_stack([
        np.bincount(
            label,
            weights=value * coordinate[:, column],
            minlength=count,
        )
        for column in range(coordinate.shape[1])
    ])
    mean /= np.maximum(total[:, None], 1e-30)
    score = (mean * sign[None, :]) @ mean.T
    norm = np.linalg.norm(mean, axis=1)
    score /= np.maximum(norm[:, None] * norm[None, :], 1e-30)
    return np.clip(score, -1.0, 1.0)
