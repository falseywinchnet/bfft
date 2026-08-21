"""Exact winding supports emitted by closed one-sided contour components.

Every even-degree embedded contour component emits the pixels selected by its
mod-2 winding field.  Region participation is measured both as literal area
overlap and as overlap weighted by the contour interior's own second moment.
No foreground side, semantic seed, radius, or inclusion threshold is chosen.
"""

from __future__ import annotations

import hashlib

import numpy as np
from scipy import sparse


def _normalized_gram(participation: sparse.csr_matrix) -> sparse.csr_matrix:
    norm = np.sqrt(np.asarray(participation.power(2).sum(axis=0)).ravel())
    normalized = participation @ sparse.diags(np.divide(
        1.0, norm, out=np.zeros_like(norm), where=norm > 0.0))
    kernel = (normalized.T @ normalized).tocsr()
    kernel = (0.5 * (kernel + kernel.T)).tocsr()
    kernel.data = np.clip(kernel.data, 0.0, 1.0)
    kernel.eliminate_zeros()
    return kernel


def build_contour_cycle_nesting(
    complex_: dict,
    bundle: dict,
    contour: dict,
) -> dict:
    labels = np.asarray(complex_["labels"], dtype=np.int32)
    height, width = labels.shape
    region_count = int(complex_["region_count"])
    topology = complex_["topology"]
    edgel = topology["edgel"]
    incidence = bundle["incidence"]
    incidence_component = np.asarray(
        contour["incidence_component"], dtype=np.int32)
    component_count = int(contour["component_count"])
    area = np.asarray(complex_["node"]["area"], dtype=np.float64)

    # Arc sets are recovered from the exact incidence components.  Different
    # one-sided owners can induce the same physical cycle, so identical
    # winding fields are retained once.
    component_arcs = [set() for _ in range(component_count)]
    for identifier, arc in zip(incidence_component, incidence["arc"]):
        component_arcs[int(identifier)].add(int(arc))
    edgel_arc = np.asarray(edgel["arc"], dtype=np.int32)
    vertex_first = np.asarray(edgel["vertex_first"], dtype=np.int64)
    vertex_second = np.asarray(edgel["vertex_second"], dtype=np.int64)
    orientation = np.asarray(edgel["orientation"], dtype=np.int8)

    source_component = []
    winding_area = []
    centroid_x = []
    centroid_y = []
    covariance_xx = []
    covariance_xy = []
    covariance_yy = []
    overlap_row = []
    overlap_column = []
    overlap_value = []
    centered_value = []
    seen: set[bytes] = set()
    yy, xx = np.indices(labels.shape, dtype=np.float64)
    for component_identifier, arcs in enumerate(component_arcs):
        if not arcs:
            continue
        selected = np.isin(edgel_arc, np.fromiter(arcs, dtype=np.int32))
        if not np.any(selected):
            continue
        vertices = np.concatenate((
            vertex_first[selected], vertex_second[selected]))
        degree = np.bincount(
            vertices, minlength=(height + 1) * (width + 1))
        if np.any(degree & 1):
            continue

        vertical = selected & (orientation == 1)
        grid_vertex = vertex_first[vertical]
        grid_x = grid_vertex % (width + 1)
        grid_y = grid_vertex // (width + 1)
        valid = (
            (grid_y >= 0) & (grid_y < height)
            & (grid_x >= 0) & (grid_x <= width)
        )
        crossing = np.zeros((height, width + 1), dtype=np.uint8)
        np.bitwise_xor.at(
            crossing, (grid_y[valid], grid_x[valid]), np.uint8(1))
        inside = np.bitwise_and(
            np.cumsum(crossing[:, :width], axis=1), 1).astype(bool)
        count = int(np.count_nonzero(inside))
        if count == 0:
            continue
        signature = hashlib.sha256(np.packbits(inside).tobytes()).digest()
        if signature in seen:
            continue
        seen.add(signature)

        x_center = float(np.mean(xx[inside]))
        y_center = float(np.mean(yy[inside]))
        dx = xx[inside] - x_center
        dy = yy[inside] - y_center
        covariance = np.asarray((
            (float(np.mean(dx * dx)), float(np.mean(dx * dy))),
            (float(np.mean(dx * dy)), float(np.mean(dy * dy))),
        ))
        inverse = np.linalg.pinv(covariance, rcond=1e-12)
        quadratic = (
            inverse[0, 0] * (xx - x_center) ** 2
            + 2.0 * inverse[0, 1] * (xx - x_center) * (yy - y_center)
            + inverse[1, 1] * (yy - y_center) ** 2
        )
        # The Cauchy feature is parameter-free after normalization by the
        # contour's measured covariance; literal overlap remains separate.
        centrality = 1.0 / np.sqrt(1.0 + np.maximum(quadratic, 0.0))
        region_area_inside = np.bincount(
            labels[inside], minlength=region_count).astype(np.float64)
        region_centered_inside = np.bincount(
            labels[inside], weights=centrality[inside],
            minlength=region_count).astype(np.float64)
        occupied = np.flatnonzero(region_area_inside > 0.0)
        row_identifier = len(source_component)
        overlap_row.extend([row_identifier] * len(occupied))
        overlap_column.extend(occupied.tolist())
        overlap_value.extend(
            (region_area_inside[occupied] / area[occupied]).tolist())
        centered_value.extend(
            (region_centered_inside[occupied] / area[occupied]).tolist())
        source_component.append(component_identifier)
        winding_area.append(count)
        centroid_x.append(x_center)
        centroid_y.append(y_center)
        covariance_xx.append(covariance[0, 0])
        covariance_xy.append(covariance[0, 1])
        covariance_yy.append(covariance[1, 1])

    shape = (len(source_component), region_count)
    row = np.asarray(overlap_row, dtype=np.int32)
    column = np.asarray(overlap_column, dtype=np.int32)
    overlap = sparse.csr_matrix(
        (np.asarray(overlap_value), (row, column)), shape=shape)
    centered = sparse.csr_matrix(
        (np.asarray(centered_value), (row, column)), shape=shape)
    return {
        "source_component": np.asarray(source_component, dtype=np.int32),
        "winding_area": np.asarray(winding_area, dtype=np.float64),
        "centroid_x": np.asarray(centroid_x, dtype=np.float64),
        "centroid_y": np.asarray(centroid_y, dtype=np.float64),
        "covariance_xx": np.asarray(covariance_xx, dtype=np.float64),
        "covariance_xy": np.asarray(covariance_xy, dtype=np.float64),
        "covariance_yy": np.asarray(covariance_yy, dtype=np.float64),
        "overlap_participation": overlap,
        "centered_participation": centered,
        "overlap_kernel": _normalized_gram(overlap),
        "centered_kernel": _normalized_gram(centered),
    }


def summarize_contour_cycle_nesting(nesting: dict) -> dict:
    overlap = nesting["overlap_participation"]
    member_count = np.diff(overlap.indptr)
    return {
        "distinct_closed_winding_fields": int(overlap.shape[0]),
        "single_region_fields": int(np.count_nonzero(member_count == 1)),
        "multi_region_fields": int(np.count_nonzero(member_count > 1)),
        "maximum_regions_in_field": int(np.max(member_count, initial=0)),
        "maximum_winding_area": float(np.max(
            nesting["winding_area"], initial=0.0)),
    }
