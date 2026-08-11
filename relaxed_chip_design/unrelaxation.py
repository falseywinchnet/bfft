"""Dimension-sparse unrelaxation for a transported circuit measure.

The inverse carries a cell's physical anchor phase through two CDFs.  It does
not score alternative locations.  The reference coupling says where the cell
started in its local support; the transported coupling says where that same
mass coordinate ends.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class QuantileUnrelaxation:
    """Hard targets decoded from one identity-preserving transport map."""

    target_slots: np.ndarray
    target_segments: np.ndarray
    quantiles: np.ndarray


def _as_vector(name: str, value: np.ndarray, length: int) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (length,):
        raise ValueError("{} must have shape ({},)".format(name, length))
    return array


def _validate_feature_inputs(
    cell_features: np.ndarray,
    site_features: np.ndarray,
    site_segments: np.ndarray,
    site_x: np.ndarray,
    anchor_segments: np.ndarray,
    anchor_x: np.ndarray,
    segment_count: int,
) -> Tuple[np.ndarray, ...]:
    cells = np.asarray(cell_features, dtype=np.float64)
    sites = np.asarray(site_features, dtype=np.float64)
    if cells.ndim != 2 or sites.ndim != 2:
        raise ValueError("cell_features and site_features must be matrices")
    if cells.shape[1] != sites.shape[1]:
        raise ValueError("cell and site feature ranks must match")
    segments = _as_vector("site_segments", site_segments, len(sites)).astype(
        np.int64, copy=False
    )
    site_positions = _as_vector("site_x", site_x, len(sites)).astype(
        np.float64, copy=False
    )
    anchors = _as_vector(
        "anchor_segments", anchor_segments, len(cells)
    ).astype(np.int64, copy=False)
    anchor_positions = _as_vector("anchor_x", anchor_x, len(cells)).astype(
        np.float64, copy=False
    )
    if segment_count <= 0:
        raise ValueError("segment_count must be positive")
    if np.any(segments < 0) or np.any(segments >= segment_count):
        raise ValueError("site segment is outside segment_count")
    if np.any(anchors < 0) or np.any(anchors >= segment_count):
        raise ValueError("anchor segment is outside segment_count")
    if not np.all(np.isfinite(cells)) or not np.all(np.isfinite(sites)):
        raise ValueError("features must be finite")
    if np.any(cells < 0.0) or np.any(sites < 0.0):
        raise ValueError("rank features must be nonnegative")
    return (
        cells,
        sites,
        segments,
        site_positions,
        anchors,
        anchor_positions,
    )


def reference_anchor_phases(
    cell_features: np.ndarray,
    site_features: np.ndarray,
    site_segments: np.ndarray,
    site_x: np.ndarray,
    anchor_segments: np.ndarray,
    anchor_x: np.ndarray,
    segment_count: int,
) -> np.ndarray:
    """Evaluate each hard anchor in the rank-factor reference CDF.

    Sites are visited one physical segment at a time.  The routine forms one
    ``capacity x rank`` prefix, reads every anchor in that segment, and then
    releases the prefix.  It never materializes a cell-by-site coupling.
    """

    (
        cells,
        sites,
        segments,
        site_positions,
        anchors,
        anchor_positions,
    ) = _validate_feature_inputs(
        cell_features,
        site_features,
        site_segments,
        site_x,
        anchor_segments,
        anchor_x,
        segment_count,
    )

    phase = np.empty(len(cells), dtype=np.float64)
    site_order = np.lexsort((site_positions, segments))
    ordered_segments = segments[site_order]
    offsets = np.searchsorted(
        ordered_segments, np.arange(segment_count + 1)
    )

    for segment in range(segment_count):
        cell_indices = np.flatnonzero(anchors == segment)
        if not len(cell_indices):
            continue
        indices = site_order[offsets[segment] : offsets[segment + 1]]
        if not len(indices):
            raise ValueError("anchor segment has no reference sites")

        x = site_positions[indices]
        prefix = np.vstack(
            (
                np.zeros((1, sites.shape[1]), dtype=np.float64),
                np.cumsum(sites[indices], axis=0),
            )
        )
        left = np.searchsorted(x, anchor_positions[cell_indices], side="left")
        right = np.searchsorted(
            x, anchor_positions[cell_indices], side="right"
        )
        midpoint_summary = prefix[left] + 0.5 * (
            prefix[right] - prefix[left]
        )
        numerator = np.einsum(
            "cr,cr->c",
            cells[cell_indices],
            midpoint_summary,
            optimize=True,
        )
        denominator = cells[cell_indices] @ prefix[-1]
        if np.any(denominator <= 0.0):
            raise ValueError("reference feature mass must be positive")
        phase[cell_indices] = numerator / denominator

    return np.clip(phase, 0.0, 1.0)


def conjugate_support_quantiles(
    active_segments: np.ndarray,
    anchor_segments: np.ndarray,
    transported_support: np.ndarray,
    reference_support: np.ndarray,
    segment_y: np.ndarray,
    segment_x: np.ndarray,
    source_phase: np.ndarray,
) -> QuantileUnrelaxation:
    """Carry each hard anchor phase through reference and transported CDFs.

    Negative entries in ``active_segments`` are padding.  Every row must
    contain its anchor exactly once, and both support measures must have
    positive mass over their valid slots.
    """

    active = np.asarray(active_segments, dtype=np.int64)
    transported = np.asarray(transported_support, dtype=np.float64)
    reference = np.asarray(reference_support, dtype=np.float64)
    if active.ndim != 2:
        raise ValueError("active_segments must be a cell-by-support matrix")
    if transported.shape != active.shape or reference.shape != active.shape:
        raise ValueError("support measures must match active_segments")

    cell_count = len(active)
    anchors = _as_vector(
        "anchor_segments", anchor_segments, cell_count
    ).astype(np.int64, copy=False)
    phases = _as_vector("source_phase", source_phase, cell_count).astype(
        np.float64, copy=False
    )
    y = np.asarray(segment_y, dtype=np.float64)
    x = np.asarray(segment_x, dtype=np.float64)
    if y.ndim != 1 or x.shape != y.shape:
        raise ValueError("segment_x and segment_y must be equal-length vectors")
    if np.any(active >= len(y)):
        raise ValueError("active support contains an unknown segment")
    if not np.all(np.isfinite(phases)) or np.any(phases < 0.0) or np.any(
        phases > 1.0
    ):
        raise ValueError("source phases must be finite values in [0, 1]")
    if not np.all(np.isfinite(transported)) or not np.all(
        np.isfinite(reference)
    ):
        raise ValueError("support measures must be finite")

    target_slots = np.full(cell_count, -1, dtype=np.int32)
    target_segments = np.full(cell_count, -1, dtype=np.int64)
    quantiles = np.empty(cell_count, dtype=np.float64)

    for cell in range(cell_count):
        valid_slots = np.flatnonzero(active[cell] >= 0)
        if not len(valid_slots):
            raise ValueError("every cell needs a nonempty active support")
        physical_order = valid_slots[
            np.lexsort(
                (
                    x[active[cell, valid_slots]],
                    y[active[cell, valid_slots]],
                )
            )
        ]
        ordered_segments = active[cell, physical_order]
        anchor_position = np.flatnonzero(ordered_segments == anchors[cell])
        if len(anchor_position) != 1:
            raise ValueError("anchor must occur once in active support")

        reference_row = np.maximum(reference[cell, physical_order], 0.0)
        transported_row = np.maximum(
            transported[cell, physical_order], 0.0
        )
        reference_total = float(np.sum(reference_row))
        transported_total = float(np.sum(transported_row))
        if reference_total <= 0.0 or transported_total <= 0.0:
            raise ValueError("support measures must have positive row mass")
        reference_row /= reference_total
        transported_row /= transported_total

        position = int(anchor_position[0])
        quantile = float(np.sum(reference_row[:position])) + float(
            phases[cell]
        ) * float(reference_row[position])
        transported_position = min(
            int(
                np.searchsorted(
                    np.cumsum(transported_row), quantile, side="left"
                )
            ),
            len(physical_order) - 1,
        )
        slot = int(physical_order[transported_position])
        target_slots[cell] = slot
        target_segments[cell] = int(active[cell, slot])
        quantiles[cell] = quantile

    return QuantileUnrelaxation(
        target_slots=target_slots,
        target_segments=target_segments,
        quantiles=quantiles,
    )
