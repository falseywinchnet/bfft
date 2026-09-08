"""Conditional-entropy change points for generic-bitmap region proposals."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GridRegion:
    x0: int
    y0: int
    x1: int
    y1: int


@dataclass(frozen=True)
class RegionPartition:
    regions: tuple[GridRegion, ...]
    estimated_bits: float
    last_gain_bits: float | None
    x_edges: tuple[int, ...]
    y_edges: tuple[int, ...]


def _shift(bits, dx: int, dy: int):
    import numpy as np

    height, width = bits.shape
    result = np.zeros_like(bits, dtype=np.uint8)
    tx0, tx1 = max(0, -dx), min(width, width - dx)
    ty0, ty1 = max(0, -dy), min(height, height - dy)
    if tx0 < tx1 and ty0 < ty1:
        result[ty0:ty1, tx0:tx1] = bits[ty0 + dy:ty1 + dy, tx0 + dx:tx1 + dx]
    return result


def propose_entropy_partitions(
    ink,
    *,
    cell_size: int = 128,
    max_regions: int = 8,
    header_bits: float = 31 * 8,
    adaptation_bits_per_context: float = 5.0,
    minimum_gain_bits: float = -256.0,
    first_split_frontier: int = 1,
) -> tuple[RegionPartition, ...]:
    """Greedily split at changes in predictive-context distributions.

    A 4-neighbor causal context is used only as a cheap proposal model. Actual
    JBIG2 byte size remains the terminal acceptance criterion.
    """

    import numpy as np

    bits = np.asarray(ink, dtype=np.uint8)
    if bits.ndim != 2 or not bits.any():
        return ()
    ys, xs = np.nonzero(bits)
    left, up = _shift(bits, -1, 0), _shift(bits, 0, -1)
    up_left, up_right = _shift(bits, -1, -1), _shift(bits, 1, -1)
    context = left | (up << 1) | (up_left << 2) | (up_right << 3)
    symbols = (context << 1) | bits

    px0, py0, px1, py1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    x_edges = list(range(px0, px1, cell_size)) + [px1]
    y_edges = list(range(py0, py1, cell_size)) + [py1]
    nx, ny = len(x_edges) - 1, len(y_edges) - 1
    block_counts = np.zeros((ny, nx, 32), dtype=np.int64)
    for gy in range(ny):
        for gx in range(nx):
            block = symbols[y_edges[gy]:y_edges[gy + 1], x_edges[gx]:x_edges[gx + 1]]
            block_counts[gy, gx] = np.bincount(block.ravel(), minlength=32)
    integral = block_counts.cumsum(axis=0).cumsum(axis=1)

    def histogram(region: GridRegion):
        value = integral[region.y1 - 1, region.x1 - 1].copy()
        if region.x0:
            value -= integral[region.y1 - 1, region.x0 - 1]
        if region.y0:
            value -= integral[region.y0 - 1, region.x1 - 1]
        if region.x0 and region.y0:
            value += integral[region.y0 - 1, region.x0 - 1]
        return value

    def cost(region: GridRegion) -> float:
        counts = histogram(region).reshape(16, 2)
        totals = counts.sum(axis=1)
        nonzero = totals > 0
        probabilities = counts[nonzero] / totals[nonzero, None]
        terms = np.zeros_like(probabilities, dtype=np.float64)
        positive = probabilities > 0
        terms[positive] = -probabilities[positive] * np.log2(probabilities[positive])
        entropy_bits = float((terms.sum(axis=1) * totals[nonzero]).sum())
        active_contexts = int(nonzero.sum())
        return entropy_bits + header_bits + adaptation_bits_per_context * active_contexts

    regions = [GridRegion(0, 0, nx, ny)]
    edges = (tuple(x_edges), tuple(y_edges))
    partitions = [RegionPartition(tuple(regions), cost(regions[0]), None, *edges)]
    while len(regions) < max_regions:
        candidates = []
        for index, region in enumerate(regions):
            parent_cost = cost(region)
            for split in range(region.x0 + 1, region.x1):
                first = GridRegion(region.x0, region.y0, split, region.y1)
                second = GridRegion(split, region.y0, region.x1, region.y1)
                candidates.append((parent_cost - cost(first) - cost(second), index, first, second))
            for split in range(region.y0 + 1, region.y1):
                first = GridRegion(region.x0, region.y0, region.x1, split)
                second = GridRegion(region.x0, split, region.x1, region.y1)
                candidates.append((parent_cost - cost(first) - cost(second), index, first, second))
        if not candidates:
            break
        candidates.sort(key=lambda item: item[0], reverse=True)
        gain, index, first, second = candidates[0]
        if gain < minimum_gain_bits:
            break
        if len(regions) == 1 and first_split_frontier > 1:
            for alternate_gain, alternate_index, alternate_first, alternate_second in candidates[:first_split_frontier]:
                if alternate_gain < minimum_gain_bits:
                    break
                alternate_regions = list(regions)
                alternate_regions[alternate_index:alternate_index + 1] = [alternate_first, alternate_second]
                partitions.append(
                    RegionPartition(
                        tuple(alternate_regions),
                        sum(cost(region) for region in alternate_regions),
                        alternate_gain,
                        *edges,
                    )
                )
        regions[index:index + 1] = [first, second]
        estimated = sum(cost(region) for region in regions)
        if first_split_frontier <= 1 or len(regions) > 2:
            partitions.append(RegionPartition(tuple(regions), estimated, gain, *edges))
    return tuple(partitions)


def partition_pixel_boxes(partition: RegionPartition, ink):
    """Convert the most recently proposed grid into disjoint tight ink boxes."""

    import numpy as np

    x_edges = partition.x_edges
    y_edges = partition.y_edges
    boxes = []
    for region in partition.regions:
        x0, x1 = x_edges[region.x0], x_edges[region.x1]
        y0, y1 = y_edges[region.y0], y_edges[region.y1]
        ys, xs = np.nonzero(ink[y0:y1, x0:x1])
        if len(xs):
            boxes.append((x0 + int(xs.min()), y0 + int(ys.min()),
                          x0 + int(xs.max()) + 1, y0 + int(ys.max()) + 1))
    return boxes
