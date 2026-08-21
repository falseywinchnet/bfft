"""Exact one-sided contour components of the V3 incidence bundle.

A boundary component is not a color family and not an object rule.  It is the
connected component obtained by following one fixed region side through exact
embedded junction continuations.  The opposite regions encountered along that
component are simultaneous participants in one witnessed contour relation.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


def build_contour_transport(complex_: dict, bundle: dict) -> dict:
    """Construct connected one-sided contours and their opposite-region law."""
    incidence = bundle["incidence"]
    continuation = bundle["continuation"]
    count = len(incidence["arc"])
    region_count = int(complex_["region_count"])
    lookup = {
        (int(arc), int(region)): identifier
        for identifier, (arc, region) in enumerate(zip(
            incidence["arc"], incidence["region"]))
    }
    first = []
    second = []
    for first_arc, second_arc, shared in zip(
        continuation["first_arc"],
        continuation["second_arc"],
        continuation["region"],
    ):
        first.append(lookup[(int(first_arc), int(shared))])
        second.append(lookup[(int(second_arc), int(shared))])
    if first:
        row = np.asarray(first + second, dtype=np.int32)
        column = np.asarray(second + first, dtype=np.int32)
        adjacency = sparse.csr_matrix(
            (np.ones(len(row), dtype=np.int8), (row, column)),
            shape=(count, count),
        )
    else:
        adjacency = sparse.csr_matrix((count, count), dtype=np.int8)
    component_count, component = connected_components(
        adjacency, directed=False, return_labels=True)
    component = component.astype(np.int32, copy=False)

    owner = np.full(component_count, -1, dtype=np.int32)
    owner_consistent = np.ones(component_count, dtype=bool)
    incidence_region = np.asarray(incidence["region"], dtype=np.int32)
    for identifier in range(component_count):
        members = np.flatnonzero(component == identifier)
        owners = np.unique(incidence_region[members])
        owner_consistent[identifier] = len(owners) == 1
        if len(owners):
            owner[identifier] = int(owners[0])
    if not np.all(owner_consistent):
        raise RuntimeError("one-sided contour continuation changed owner region")

    length = np.asarray(incidence["length"], dtype=np.float64)
    component_length = np.bincount(
        component, weights=length, minlength=component_count)
    component_arcs = np.bincount(component, minlength=component_count)
    component_closed_arcs = np.bincount(
        component,
        weights=np.asarray(incidence["closed"], dtype=np.int32),
        minlength=component_count,
    ).astype(np.int32)

    outside = np.asarray(incidence["outside"], dtype=np.int32)
    key = component.astype(np.int64) * region_count + outside
    unique, inverse = np.unique(key, return_inverse=True)
    pair_component = (unique // region_count).astype(np.int32)
    pair_region = (unique % region_count).astype(np.int32)
    pair_length = np.bincount(inverse, weights=length).astype(np.float64)
    pair_fraction = pair_length / np.maximum(
        component_length[pair_component], 1e-30)

    opposite = sparse.csr_matrix(
        (pair_fraction, (pair_component, pair_region)),
        shape=(component_count, region_count),
    )
    # Columns are empirical distributions over distinct, localized contour
    # components.  Their normalized Gram is a positive participation kernel.
    column_norm = np.sqrt(np.asarray(opposite.power(2).sum(axis=0)).ravel())
    normalized = opposite @ sparse.diags(np.divide(
        1.0,
        column_norm,
        out=np.zeros_like(column_norm),
        where=column_norm > 0.0,
    ))
    kernel = (normalized.T @ normalized).toarray()
    kernel = np.clip(0.5 * (kernel + kernel.T), -1.0, 1.0)
    return {
        "incidence_component": component,
        "component_count": component_count,
        "component_owner": owner,
        "component_length": component_length,
        "component_arcs": component_arcs.astype(np.int32),
        "component_closed_arcs": component_closed_arcs,
        "pair_component": pair_component,
        "pair_region": pair_region,
        "pair_length": pair_length,
        "pair_fraction": pair_fraction,
        "opposite_participation": opposite,
        "region_kernel": kernel,
    }


def summarize_contour_transport(transport: dict) -> dict:
    component_arcs = transport["component_arcs"]
    pair_component = transport["pair_component"]
    pair_count = np.bincount(
        pair_component, minlength=int(transport["component_count"]))
    return {
        "contour_components": int(transport["component_count"]),
        "single_arc_components": int(np.count_nonzero(component_arcs == 1)),
        "multi_arc_components": int(np.count_nonzero(component_arcs > 1)),
        "components_with_closed_arc": int(np.count_nonzero(
            transport["component_closed_arcs"] > 0)),
        "components_with_multiple_opposite_regions": int(np.count_nonzero(
            pair_count > 1)),
        "maximum_arcs_per_component": int(
            np.max(component_arcs, initial=0)),
        "maximum_opposite_regions_per_component": int(
            np.max(pair_count, initial=0)),
    }
