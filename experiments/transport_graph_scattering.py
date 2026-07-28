#!/usr/bin/env python3
"""Phase-preserving scattering of the frozen transport graph.

The unordered empirical measures in ``transport_relation_forensics`` discard
where a transport state occurs and how neighbouring metric directors turn.
This experiment retains that order without inventing object decisions.

Let P be the lazy, interface-measure random walk on the literal connected
support-fragment graph.  Its dyadic diffusion bands are

    Psi_j x = P^(2^(j-1)) x - P^(2^j) x,

with the first band interpreted as ``x - P x``.  The transport precision
tensor supplies the complex spin-2 field

    z = ((q_xx - q_yy) + 2 i q_xy) / tr(q).

A rigid image rotation multiplies every z by one constant unit complex phase.
Linear diffusion commutes with that multiplication, so ``abs(Psi_j z)`` is
rotation invariant while still measuring director coherence *before* phase is
discarded.  A second scattering order measures how those first-order
responses are arranged over larger graph scales.

The output remains one descriptor per canonical support fragment.  It neither
forms nor compares objects.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix


def _robust_scalar(value: np.ndarray) -> np.ndarray:
    x = np.asarray(value, dtype=np.float64)
    center = float(np.median(x))
    scale = max(
        1.4826 * float(np.median(np.abs(x - center))),
        1e-8,
    )
    return np.clip((x - center) / scale, -12.0, 12.0)


def _lazy_diffusion(graph: dict):
    count = int(graph["cells"])
    edge = graph["edge"]
    first = np.asarray(edge["first"], dtype=np.int32)
    second = np.asarray(edge["second"], dtype=np.int32)
    weight = np.asarray(
        edge.get("length", np.ones(len(first))), dtype=np.float64)
    adjacency = coo_matrix(
        (
            np.concatenate((weight, weight)),
            (
                np.concatenate((first, second)),
                np.concatenate((second, first)),
            ),
        ),
        shape=(count, count),
    ).tocsr()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    isolated = degree <= 1e-30
    inverse_degree = 1.0 / np.maximum(degree, 1e-30)

    def apply(signal: np.ndarray) -> np.ndarray:
        value = np.asarray(signal)
        neighbour = adjacency @ value
        if value.ndim == 1:
            neighbour = neighbour * inverse_degree
            neighbour[isolated] = value[isolated]
        else:
            neighbour = neighbour * inverse_degree[:, None]
            neighbour[isolated] = value[isolated]
        return 0.5 * (value + neighbour)

    return apply


def _dyadic_bands(
    signal: np.ndarray,
    diffuse,
    scales: int,
) -> list[np.ndarray]:
    """Return x-Px, Px-P^2x, P^2x-P^4x, ... exactly."""
    low = np.asarray(signal)
    exponent = 0
    bands = []
    for scale in range(max(int(scales), 0)):
        target_exponent = 1 << scale
        next_low = low
        for _ in range(target_exponent - exponent):
            next_low = diffuse(next_low)
        bands.append(low - next_low)
        low = next_low
        exponent = target_exponent
    return bands


def transport_graph_scattering(
    objects: dict,
    *,
    scales: int = 5,
    second_order: bool = True,
) -> dict[str, np.ndarray | int]:
    """Return a multiscale, rotation-invariant descriptor per graph node."""
    graph = objects["graph"]
    qxx = np.asarray(graph["node_qxx"], dtype=np.float64)
    qxy = np.asarray(graph["node_qxy"], dtype=np.float64)
    qyy = np.asarray(graph["node_qyy"], dtype=np.float64)
    trace = np.maximum(qxx + qyy, 1e-30)
    director = ((qxx - qyy) + 2.0j * qxy) / trace
    action = _robust_scalar(
        np.log(np.maximum(graph["node_energy"], 1e-30)))
    log_trace = _robust_scalar(np.log(trace))
    diffuse = _lazy_diffusion(graph)

    # The complex product keeps action modulation attached to the same
    # transported line field; it is not a separately weighted decision cue.
    signals = (
        action.astype(np.complex128),
        log_trace.astype(np.complex128),
        director,
        action * director,
    )
    columns = [
        action,
        log_trace,
        np.abs(director),
        np.abs(action * director),
    ]
    column_scale = [0, 0, 0, 0]
    first_order: list[list[np.ndarray]] = []
    for signal in signals:
        bands = _dyadic_bands(signal, diffuse, scales)
        modulus = [np.abs(band) for band in bands]
        first_order.append(modulus)
        columns.extend(modulus)
        column_scale.extend(range(1, len(modulus) + 1))

    if second_order:
        for modulus in first_order:
            for first_scale, first_band in enumerate(modulus):
                second = _dyadic_bands(
                    first_band.astype(np.complex128),
                    diffuse,
                    scales,
                )
                columns.extend(
                    np.abs(second[second_scale])
                    for second_scale in range(first_scale + 1, len(second))
                )
                column_scale.extend(
                    range(first_scale + 2, len(second) + 1))

    descriptor = np.ascontiguousarray(
        np.column_stack(columns), dtype=np.float64)
    return {
        "scales": int(scales),
        "second_order": bool(second_order),
        "descriptor": descriptor,
        "column_scale": np.asarray(column_scale, dtype=np.int32),
        "feature_count": int(descriptor.shape[1]),
    }


def scattering_anchor_field(
    scattering: dict,
    anchor_cell: int,
    *,
    maximum_scale: int | None = None,
) -> np.ndarray:
    """Robust Gaussian likeness in the scattering coordinate system."""
    value = np.asarray(scattering["descriptor"], dtype=np.float64)
    if maximum_scale is not None:
        selected = (
            np.asarray(scattering["column_scale"], dtype=np.int32)
            <= max(int(maximum_scale), 0)
        )
        value = value[:, selected]
    center = np.median(value, axis=0)
    scale = np.maximum(
        1.4826 * np.median(np.abs(value - center), axis=0),
        1e-8,
    )
    standardized = np.clip((value - center) / scale, -12.0, 12.0)
    anchor = int(np.clip(anchor_cell, 0, max(len(value) - 1, 0)))
    difference = standardized - standardized[anchor]
    return np.exp(-0.5 * np.mean(difference * difference, axis=1))
