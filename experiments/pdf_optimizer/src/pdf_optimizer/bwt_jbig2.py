"""BWT-trained adaptive-template experiments for lossless JBIG2 generic regions.

The transform is used only to choose the four GBAT coordinates.  The original
bitmap is always what the standards-compatible encoder receives.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log2
from pathlib import Path
from typing import Iterable


DEFAULT_GBAT = ((3, -1), (-3, -1), (2, -2), (-2, -2))
_SLOTS = (4, 10, 11, 15)
_FIXED = (
    (-1, 0, 0), (-2, 0, 1), (-3, 0, 2), (-4, 0, 3),
    (2, -1, 5), (1, -1, 6), (0, -1, 7), (-1, -1, 8), (-2, -1, 9),
    (1, -2, 12), (0, -2, 13), (-1, -2, 14),
)


@dataclass(frozen=True)
class GbatProposal:
    coordinates: tuple[tuple[int, int], ...]
    conditional_bits_per_pixel: float

    @property
    def cli_value(self) -> str:
        return ";".join(f"{x},{y}" for x, y in self.coordinates)


def cyclic_bwt(block: bytes) -> bytes:
    """Return the cyclic Burrows-Wheeler last column without a sentinel."""

    size = len(block)
    if size < 2:
        return block
    values = list(block)
    order = sorted(range(size), key=values.__getitem__)
    classes = [0] * size
    class_id = 0
    for index in range(1, size):
        if values[order[index]] != values[order[index - 1]]:
            class_id += 1
        classes[order[index]] = class_id
    span = 1
    while span < size and class_id + 1 < size:
        order.sort(key=lambda index: (classes[index], classes[(index + span) % size]))
        new_classes = [0] * size
        class_id = 0
        for index in range(1, size):
            previous = order[index - 1]
            current = order[index]
            if (classes[current], classes[(current + span) % size]) != (
                classes[previous], classes[(previous + span) % size]
            ):
                class_id += 1
            new_classes[current] = class_id
        classes = new_classes
        span <<= 1
    return bytes(values[(start - 1) % size] for start in order)


def rowwise_byte_bwt(bits):
    """Apply byte BWT independently to packed rows and return unpacked bits."""

    import numpy as np

    array = np.asarray(bits, dtype=np.uint8)
    if array.ndim != 2:
        raise ValueError("bits must be a two-dimensional array")
    width = array.shape[1]
    packed = np.packbits(array, axis=1, bitorder="big")
    transformed = np.empty_like(packed)
    for row in range(packed.shape[0]):
        transformed[row] = np.frombuffer(cyclic_bwt(packed[row].tobytes()), dtype=np.uint8)
    return np.unpackbits(transformed, axis=1, bitorder="big")[:, :width]


def _shifted(bits, dx: int, dy: int):
    import numpy as np

    height, width = bits.shape
    shifted = np.zeros((height, width), dtype=np.uint8)
    target_x0 = max(0, -dx)
    target_x1 = min(width, width - dx)
    target_y0 = max(0, -dy)
    target_y1 = min(height, height - dy)
    if target_x0 < target_x1 and target_y0 < target_y1:
        shifted[target_y0:target_y1, target_x0:target_x1] = bits[
            target_y0 + dy:target_y1 + dy,
            target_x0 + dx:target_x1 + dx,
        ]
    return shifted


def _candidate_coordinates() -> tuple[tuple[int, int], ...]:
    fixed = {(dx, dy) for dx, dy, _ in _FIXED}
    candidates: list[tuple[int, int]] = []
    candidates.extend((dx, 0) for dx in range(-5, -17, -1))
    for dy, radius in ((-1, 12), (-2, 12), (-3, 10), (-4, 8), (-6, 6), (-8, 4)):
        candidates.extend((dx, dy) for dx in range(-radius, radius + 1))
    return tuple(dict.fromkeys(coordinate for coordinate in candidates if coordinate not in fixed))


def _conditional_entropy(context, target) -> float:
    import numpy as np

    packed = (context.astype(np.uint32) << 1) | target.astype(np.uint32)
    counts = np.bincount(packed, minlength=1 << 17).reshape(-1, 2)
    totals = counts.sum(axis=1)
    nonzero = totals > 0
    probabilities = counts[nonzero] / totals[nonzero, None]
    terms = np.zeros_like(probabilities, dtype=np.float64)
    positive = probabilities > 0
    terms[positive] = -probabilities[positive] * np.log2(probabilities[positive])
    return float((terms.sum(axis=1) * totals[nonzero]).sum() / target.size)


def train_gbat_tables(
    bits,
    *,
    sample_rows: int = 96,
    sample_columns: int = 120_000,
    beam_width: int = 4,
    candidates: Iterable[tuple[int, int]] | None = None,
) -> list[GbatProposal]:
    """Greedily choose template-0 adaptive pixels by empirical conditional entropy."""

    import numpy as np

    source = np.asarray(bits, dtype=np.uint8)
    if source.ndim != 2 or min(source.shape) < 2:
        raise ValueError("a nontrivial two-dimensional bitmap is required")
    height, width = source.shape
    row_count = min(sample_rows, height)
    rows = np.unique(np.linspace(0, height - 1, row_count, dtype=np.int64))
    stride = max(1, int((len(rows) * width + sample_columns - 1) // sample_columns))
    columns = np.arange(0, width, stride, dtype=np.int64)
    sample = np.ix_(rows, columns)
    target = source[sample].reshape(-1)

    base = np.zeros_like(source, dtype=np.uint16)
    for dx, dy, bit in _FIXED:
        base |= _shifted(source, dx, dy).astype(np.uint16) << bit
    base_sample = base[sample].reshape(-1)

    pool = tuple(candidates) if candidates is not None else _candidate_coordinates()
    features = {
        coordinate: _shifted(source, *coordinate)[sample].reshape(-1).astype(np.uint16)
        for coordinate in pool
    }
    beams: list[tuple[float, tuple[tuple[int, int], ...], object]] = [
        (_conditional_entropy(base_sample, target), (), base_sample)
    ]
    for slot in _SLOTS:
        expanded: list[tuple[float, tuple[tuple[int, int], ...], object]] = []
        for _, selected, context in beams:
            for coordinate in pool:
                if coordinate in selected:
                    continue
                next_context = context | (features[coordinate] << slot)
                score = _conditional_entropy(next_context, target)
                expanded.append((score, selected + (coordinate,), next_context))
        expanded.sort(key=lambda item: (item[0], item[1]))
        beams = expanded[:beam_width]
    return [
        GbatProposal(coordinates=selected, conditional_bits_per_pixel=score)
        for score, selected, _ in beams
    ]


def rank_gbat_coordinates(bits, *, sample_rows: int = 96, sample_columns: int = 120_000):
    """Rank single adaptive pixels against the twelve fixed template pixels."""

    import numpy as np

    source = np.asarray(bits, dtype=np.uint8)
    height, width = source.shape
    rows = np.unique(np.linspace(0, height - 1, min(sample_rows, height), dtype=np.int64))
    stride = max(1, int((len(rows) * width + sample_columns - 1) // sample_columns))
    sample = np.ix_(rows, np.arange(0, width, stride, dtype=np.int64))
    target = source[sample].reshape(-1)
    base = np.zeros_like(source, dtype=np.uint16)
    for dx, dy, bit in _FIXED:
        base |= _shifted(source, dx, dy).astype(np.uint16) << bit
    context = base[sample].reshape(-1)
    ranked = []
    for coordinate in _candidate_coordinates():
        feature = _shifted(source, *coordinate)[sample].reshape(-1).astype(np.uint16)
        ranked.append((coordinate, _conditional_entropy(context | (feature << 4), target)))
    ranked.sort(key=lambda item: (item[1], item[0]))
    return ranked


def zero_order_huffman_cost(data: bytes) -> int:
    """Return ideal integer Huffman payload bits for a byte string."""

    from collections import Counter
    import heapq

    heap = list(Counter(data).values())
    if len(heap) < 2:
        return 0
    heapq.heapify(heap)
    cost = 0
    while len(heap) > 1:
        merged = heapq.heappop(heap) + heapq.heappop(heap)
        cost += merged
        heapq.heappush(heap, merged)
    return cost


def shannon_entropy(data: bytes) -> float:
    """Return zero-order entropy in bits per byte symbol."""

    from collections import Counter

    if not data:
        return 0.0
    size = len(data)
    return -sum((count / size) * log2(count / size) for count in Counter(data).values())


def load_pbm_ink(path: str | Path):
    """Load PBM with one for black ink, matching the JBIG2 coding value."""

    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        return np.logical_not(np.asarray(image.convert("1"), dtype=bool)).astype(np.uint8)
