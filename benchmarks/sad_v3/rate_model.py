"""Transparent bitrate estimates for the not-yet-serialized v3 model.

V3 currently renders and discards its fitted coefficients.  Consequently this
module deliberately reports estimates, not a fictional encoded file size.
The topology estimate is real zlib output over a reversible canonical label
stream.  The parameter estimate declares every assumed fixed-width field.
"""

from __future__ import annotations

import zlib

import numpy as np


SAD_SITE_BYTES = 16
STREAM_HEADER_BYTES = 128
MAP_HEADER_BYTES = 16


def _canonicalize_labels(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Relabel in first-raster-appearance order and return old IDs per new ID."""
    flat = np.ascontiguousarray(labels, dtype=np.int32).ravel()
    unique, first, inverse = np.unique(
        flat,
        return_index=True,
        return_inverse=True,
    )
    order = np.argsort(first, kind="stable")
    old_for_new = unique[order]
    new_for_unique = np.empty(len(unique), dtype=np.int32)
    new_for_unique[order] = np.arange(len(unique), dtype=np.int32)
    canonical = new_for_unique[inverse].reshape(labels.shape)
    return np.ascontiguousarray(canonical), old_for_new


def _compressed_integer_map_bytes(values: np.ndarray) -> tuple[int, str]:
    """Return the best reversible zlib raster predictor and its byte count."""
    field = np.ascontiguousarray(values, dtype=np.int32)
    candidates = {"raw": field}
    if field.ndim == 2:
        horizontal = field.copy()
        horizontal[:, 1:] -= field[:, :-1]
        vertical = field.copy()
        vertical[1:, :] -= field[:-1, :]
        candidates["horizontal_delta"] = horizontal
        candidates["vertical_delta"] = vertical
    compressed = {
        name: len(zlib.compress(candidate.tobytes(), level=9))
        for name, candidate in candidates.items()
    }
    predictor = min(compressed, key=compressed.get)
    return compressed[predictor] + MAP_HEADER_BYTES, predictor


def _smallest_unsigned_bytes(values: np.ndarray) -> bytes:
    maximum = int(np.max(values, initial=0))
    if maximum <= np.iinfo(np.uint8).max:
        dtype = np.uint8
    elif maximum <= np.iinfo(np.uint16).max:
        dtype = np.dtype("<u2")
    else:
        dtype = np.dtype("<u4")
    return np.ascontiguousarray(values, dtype=dtype).tobytes()


def _topology_estimate(
    structural_labels: np.ndarray,
    texture_labels: np.ndarray,
) -> dict:
    structural, structural_old = _canonicalize_labels(structural_labels)
    texture, texture_old = _canonicalize_labels(texture_labels)
    texture_bytes, texture_predictor = _compressed_integer_map_bytes(texture)

    flat_structural = np.asarray(structural_labels, dtype=np.int32).ravel()
    flat_texture = np.asarray(texture_labels, dtype=np.int32).ravel()
    texture_limit = int(np.max(flat_texture, initial=-1)) + 1
    minimum_parent = np.full(
        texture_limit,
        np.iinfo(np.int32).max,
        dtype=np.int32,
    )
    maximum_parent = np.full(texture_limit, -1, dtype=np.int32)
    np.minimum.at(minimum_parent, flat_texture, flat_structural)
    np.maximum.at(maximum_parent, flat_texture, flat_structural)
    nested = bool(np.array_equal(
        minimum_parent[texture_old],
        maximum_parent[texture_old],
    ))

    if nested:
        structural_old_to_new = np.full(
            int(np.max(structural_old, initial=-1)) + 1,
            -1,
            dtype=np.int32,
        )
        structural_old_to_new[structural_old] = np.arange(
            len(structural_old),
            dtype=np.int32,
        )
        parents = structural_old_to_new[minimum_parent[texture_old]]
        parent_bytes = len(zlib.compress(
            _smallest_unsigned_bytes(parents),
            level=9,
        )) + MAP_HEADER_BYTES
        structural_bytes = 0
        structural_predictor = "derived_from_texture_parent"
    else:
        parent_bytes = 0
        structural_bytes, structural_predictor = (
            _compressed_integer_map_bytes(structural))

    total = texture_bytes + parent_bytes + structural_bytes
    return {
        "bytes": int(total),
        "texture_map_bytes": int(texture_bytes),
        "texture_predictor": texture_predictor,
        "parent_map_bytes": int(parent_bytes),
        "structural_map_bytes": int(structural_bytes),
        "structural_predictor": structural_predictor,
        "nested_parent_map": nested,
    }


def estimate_v3_rate(
    result: dict,
    *,
    structural_ridges: int,
    texture_ridges: int,
    graph_phase: bool,
) -> dict:
    """Estimate a decoder-oriented fp16 stream and SAD-compatible proxy."""
    height, width = result["reconstruction_rgb"].shape[:2]
    pixels = height * width
    structural_cells = len(result["centers"])
    texture_cells = len(result["texture_centers"])
    structural_ridges = max(int(structural_ridges), 0)
    texture_ridges = max(int(texture_ridges), 0)

    structural_basis = 3 + structural_ridges
    texture_basis = 3 + texture_ridges + (2 if graph_phase else 0)

    # Both layers store a 16-bit x/y site.  Coefficients are RGB fp16.
    # A ridged layer stores one uint16 frame angle and one fp16 offset/ridge.
    structural_bytes_per_cell = (
        4
        + structural_basis * 3 * 2
        + structural_ridges * 2
        + (2 if structural_ridges else 0)
    )
    texture_bytes_per_cell = (
        4
        + texture_basis * 3 * 2
        + texture_ridges * 2
        + (2 if texture_ridges else 0)
        # Two wave components plus the final phase of both paired normals.
        + (8 if graph_phase else 0)
    )
    parameter_bytes = (
        structural_cells * structural_bytes_per_cell
        + texture_cells * texture_bytes_per_cell
    )
    topology = _topology_estimate(
        result["labels"],
        result["texture_labels"],
    )
    estimated_stream_bytes = (
        STREAM_HEADER_BYTES + parameter_bytes + topology["bytes"]
    )
    layered_site_proxy_bytes = (
        structural_cells + texture_cells
    ) * SAD_SITE_BYTES

    to_bpp = lambda byte_count: 8.0 * byte_count / pixels
    return {
        "status": "estimate_not_bitstream",
        "pixels": pixels,
        "structural_cells": structural_cells,
        "texture_cells": texture_cells,
        "total_layered_cells": structural_cells + texture_cells,
        "structural_basis_width": structural_basis,
        "texture_basis_width": texture_basis,
        "structural_bytes_per_cell": structural_bytes_per_cell,
        "texture_bytes_per_cell": texture_bytes_per_cell,
        "parameter_bytes": int(parameter_bytes),
        "parameter_only_bpp": to_bpp(parameter_bytes),
        "topology": topology,
        "topology_bpp": to_bpp(topology["bytes"]),
        "estimated_stream_bytes": int(estimated_stream_bytes),
        "estimated_stream_bpp": to_bpp(estimated_stream_bytes),
        "sad_proxy_bytes_per_site": SAD_SITE_BYTES,
        "sad_layered_site_proxy_bytes": int(layered_site_proxy_bytes),
        "sad_layered_site_proxy_bpp": to_bpp(layered_site_proxy_bytes),
        "sad_texture_only_site_proxy_bpp": to_bpp(
            texture_cells * SAD_SITE_BYTES),
    }
