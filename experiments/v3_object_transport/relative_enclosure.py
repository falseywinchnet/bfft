"""Bounded relative-complement manifolds of the planar V3 region complex.

For every possible owner region simultaneously, remove that vertex from the
literal region adjacency graph.  Each remaining connected component that has
no path to a frame-touching region is a bounded complement relative to that
owner.  This is an exact topological observable, not a foreground rule or an
appearance threshold.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


def build_relative_enclosures(complex_: dict) -> dict:
    region_count = int(complex_["region_count"])
    arc = complex_["arc"]
    first = np.asarray(arc["cell_first"], dtype=np.int32)
    second = np.asarray(arc["cell_second"], dtype=np.int32)
    if len(first):
        pair = np.unique(np.column_stack((first, second)), axis=0)
        row = np.concatenate((pair[:, 0], pair[:, 1]))
        column = np.concatenate((pair[:, 1], pair[:, 0]))
        adjacency = sparse.csr_matrix(
            (np.ones(len(row), dtype=np.int8), (row, column)),
            shape=(region_count, region_count),
        )
    else:
        adjacency = sparse.csr_matrix(
            (region_count, region_count), dtype=np.int8)
    frame = np.asarray(complex_["node"]["touches_frame"], dtype=bool)
    area = np.asarray(complex_["node"]["area"], dtype=np.float64)

    manifold_owner: list[int] = []
    manifold_offset = [0]
    member_record: list[int] = []
    member_area_fraction: list[float] = []
    manifold_area: list[float] = []
    for owner in range(region_count):
        keep = np.ones(region_count, dtype=bool)
        keep[owner] = False
        retained = np.flatnonzero(keep)
        if not len(retained):
            continue
        subgraph = adjacency[keep][:, keep]
        component_count, component = connected_components(
            subgraph, directed=False, return_labels=True)
        for identifier in range(component_count):
            member = retained[component == identifier]
            if np.any(frame[member]):
                continue
            total_area = float(np.sum(area[member]))
            if total_area <= 0.0:
                continue
            manifold_owner.append(owner)
            member_record.extend(member.tolist())
            member_area_fraction.extend((area[member] / total_area).tolist())
            manifold_area.append(total_area)
            manifold_offset.append(len(member_record))

    owner_array = np.asarray(manifold_owner, dtype=np.int32)
    offset_array = np.asarray(manifold_offset, dtype=np.int64)
    member_array = np.asarray(member_record, dtype=np.int32)
    fraction_array = np.asarray(member_area_fraction, dtype=np.float64)
    manifold_count = len(owner_array)
    if manifold_count:
        manifold_index = np.repeat(
            np.arange(manifold_count, dtype=np.int32),
            np.diff(offset_array),
        )
        participation = sparse.csr_matrix(
            (fraction_array, (manifold_index, member_array)),
            shape=(manifold_count, region_count),
        )
    else:
        participation = sparse.csr_matrix(
            (0, region_count), dtype=np.float64)
    column_norm = np.sqrt(np.asarray(
        participation.power(2).sum(axis=0)).ravel())
    normalized = participation @ sparse.diags(np.divide(
        1.0,
        column_norm,
        out=np.zeros_like(column_norm),
        where=column_norm > 0.0,
    ))
    kernel = (normalized.T @ normalized).toarray()
    kernel = np.clip(0.5 * (kernel + kernel.T), 0.0, 1.0)
    return {
        "adjacency": adjacency,
        "manifold_owner": owner_array,
        "manifold_offset": offset_array,
        "manifold_member": member_array,
        "manifold_member_area_fraction": fraction_array,
        "manifold_area": np.asarray(manifold_area, dtype=np.float64),
        "participation": participation,
        "region_kernel": kernel,
    }


def summarize_relative_enclosures(enclosure: dict) -> dict:
    offset = enclosure["manifold_offset"]
    sizes = np.diff(offset)
    owners = enclosure["manifold_owner"]
    return {
        "bounded_manifolds": int(len(owners)),
        "owners_with_bounded_complement": int(len(np.unique(owners))),
        "single_region_manifolds": int(np.count_nonzero(sizes == 1)),
        "multi_region_manifolds": int(np.count_nonzero(sizes > 1)),
        "maximum_regions_in_manifold": int(np.max(sizes, initial=0)),
        "maximum_manifold_area": float(np.max(
            enclosure["manifold_area"], initial=0.0)),
    }
