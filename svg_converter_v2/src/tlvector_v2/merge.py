"""Error-per-byte merging of flat-color connected components."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from tlvector.core import _feature_to_rgba


@dataclass(frozen=True)
class MergeReport:
    rounds: int
    merges: int
    components_before: int
    components_after: int
    estimated_bytes_saved: int
    mse_before: float
    mse_after: float


def _component_map(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    result = np.empty(labels.shape, dtype=np.int32)
    component_labels: list[int] = []
    component_sizes: list[int] = []
    offset = 0
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    for label in np.unique(labels):
        local, count = ndimage.label(labels == label, structure=structure)
        mask = local > 0
        result[mask] = local[mask] + offset - 1
        sizes = np.bincount(local.ravel(), minlength=count + 1)[1:]
        component_labels.extend([int(label)] * count)
        component_sizes.extend(sizes.astype(int).tolist())
        offset += count
    return (
        result,
        np.asarray(component_labels, dtype=np.int32),
        np.asarray(component_sizes, dtype=np.int64),
    )


def _adjacency(component_map: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left = component_map[:, :-1].ravel()
    right = component_map[:, 1:].ravel()
    top = component_map[:-1].ravel()
    bottom = component_map[1:].ravel()
    first = np.concatenate((left, top))
    second = np.concatenate((right, bottom))
    changed = first != second
    low = np.minimum(first[changed], second[changed]).astype(np.int64)
    high = np.maximum(first[changed], second[changed]).astype(np.int64)
    keys = low * np.int64(count) + high
    unique, shared = np.unique(keys, return_counts=True)
    return unique // count, unique % count, shared.astype(np.int64)


def _mse(source: np.ndarray, labels: np.ndarray, palette: np.ndarray) -> float:
    delta = source.astype(np.float64) - palette[labels].astype(np.float64)
    return float(np.mean(delta * delta))


def merge_error_per_byte(
    source: np.ndarray,
    labels: np.ndarray,
    palette: np.ndarray,
    *,
    target_mse: float,
    maximum_area: int = 32,
    rounds: int = 2,
) -> tuple[np.ndarray, np.ndarray, MergeReport]:
    """Greedily absorb cheap small islands while respecting a hard MSE cap."""
    current = labels.copy()
    current_palette = palette.copy()
    before_mse = _mse(source, current, current_palette)
    total_merges = 0
    estimated_saved = 0
    components_before = 0
    completed_rounds = 0
    flat_source = source.reshape(-1, 4).astype(np.float64)
    budget_sse = float(target_mse) * source.size

    for round_index in range(max(0, int(rounds))):
        component_map, component_label, area = _component_map(current)
        component_count = len(component_label)
        if round_index == 0:
            components_before = component_count
        first, second, shared = _adjacency(component_map, component_count)
        if not len(first):
            break

        flat_components = component_map.ravel()
        sums = np.stack([
            np.bincount(
                flat_components, weights=flat_source[:, channel],
                minlength=component_count,
            )
            for channel in range(4)
        ], axis=1)
        sumsq = np.bincount(
            flat_components,
            weights=np.sum(flat_source * flat_source, axis=1),
            minlength=component_count,
        )
        current_colors = current_palette[component_label].astype(np.float64)
        current_sse_by_component = (
            sumsq
            - 2.0 * np.sum(current_colors * sums, axis=1)
            + area * np.sum(current_colors * current_colors, axis=1)
        )

        source_component = np.concatenate((first, second))
        target_component = np.concatenate((second, first))
        shared_edge = np.concatenate((shared, shared))
        eligible = area[source_component] <= max(1, int(maximum_area))
        source_component = source_component[eligible]
        target_component = target_component[eligible]
        shared_edge = shared_edge[eligible]
        if not len(source_component):
            break
        target_colors = current_palette[component_label[target_component]].astype(np.float64)
        moved_sse = (
            sumsq[source_component]
            - 2.0 * np.sum(target_colors * sums[source_component], axis=1)
            + area[source_component] * np.sum(target_colors * target_colors, axis=1)
        )
        delta = moved_sse - current_sse_by_component[source_component]
        byte_saving = 12 + 6 * shared_edge
        score = delta / byte_saving
        ranking = np.lexsort((target_component, source_component, score))

        fixed_sse = float(np.sum(current_sse_by_component))
        new_label = component_label.copy()
        consumed = np.zeros(component_count, dtype=bool)
        protected = np.zeros(component_count, dtype=bool)
        accepted = 0
        saved = 0
        for candidate in ranking:
            source_id = int(source_component[candidate])
            target_id = int(target_component[candidate])
            if consumed[source_id] or protected[source_id] or consumed[target_id]:
                continue
            change = float(delta[candidate])
            if fixed_sse + change > budget_sse:
                continue
            new_label[source_id] = component_label[target_id]
            consumed[source_id] = True
            protected[target_id] = True
            fixed_sse += change
            accepted += 1
            saved += int(byte_saving[candidate])
        if accepted == 0:
            break
        current = new_label[component_map]
        current_palette = _feature_to_rgba(current_palette, source, current)
        total_merges += accepted
        estimated_saved += saved
        completed_rounds = round_index + 1

    final_map, _final_labels, _final_sizes = _component_map(current)
    components_after = int(np.max(final_map)) + 1
    after_mse = _mse(source, current, current_palette)
    if after_mse > float(target_mse) + 1e-9:
        raise RuntimeError(
            f"component merging exceeded MSE target: {after_mse} > {target_mse}"
        )
    return current, current_palette, MergeReport(
        rounds=completed_rounds,
        merges=total_merges,
        components_before=components_before,
        components_after=components_after,
        estimated_bytes_saved=estimated_saved,
        mse_before=before_mse,
        mse_after=after_mse,
    )
