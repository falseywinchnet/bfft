"""Perceptually allocated raster posterization pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from time import perf_counter

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy import sparse
from scipy.sparse import csgraph

from tlvector.core import (
    VectorizerConfig,
    _feature_to_rgba,
    _nearest_assign,
    _normalize_alpha,
    _resize_rgba,
    _rgba_features,
    _seed_palette,
    _srgb_to_oklab,
    _trim_transparent,
)

from .oklch import (
    bifurcate_palette,
    lab_alpha_to_rgba,
    oklch_distance2,
    oklch_pair_distance2,
    separate_nodes,
)


@dataclass(frozen=True)
class PosterizerConfig:
    colors: int = 24
    method: str = "oklch"
    lightness_weight: float = 1.0
    chroma_weight: float = 1.0
    hue_weight: float = 1.0
    alpha_weight: float = 0.7
    node_separation: float = 1.08
    detail_priority: float = 2.0
    population_exponent: float = 0.65
    family_priority: float = 1.0
    structure_radius: int = 2
    structure_threshold: float = 0.065
    texture_priority: float = 0.25
    mixing_strength: float = 0.0
    mixing_neighbors: int = 3
    minimum_leaf: int = 16
    sample_limit: int = 65536
    coarse_side: int = 160
    minimum_island: int = 6
    cleanup_rounds: int = 1
    alpha_mode: str = "auto"
    alpha_cutoff: int = 128
    trim_transparent: bool = True


@dataclass
class PosterizerResult:
    labels: np.ndarray
    palette_rgba: np.ndarray
    posterized_rgba: np.ndarray
    diagnostics: dict[str, float | int | str | list]

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        suffix = destination.suffix.lower()
        if suffix == ".png":
            Image.fromarray(self.posterized_rgba, "RGBA").save(destination)
        elif suffix in {".jpg", ".jpeg"}:
            image = Image.fromarray(self.posterized_rgba, "RGBA")
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            background.alpha_composite(image)
            background.convert("RGB").save(
                destination, quality=95, subsampling=0, optimize=True
            )
        elif suffix == ".json":
            destination.write_text(
                json.dumps(self.diagnostics, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        else:
            raise ValueError("posterizer output must end in .png, .jpg, .jpeg, or .json")


def _rgba_to_lab_alpha(rgba: np.ndarray) -> np.ndarray:
    rgb = rgba[..., :3].astype(np.float64) / 255.0
    alpha = rgba[..., 3:4].astype(np.float64) / 255.0
    return np.concatenate((_srgb_to_oklab(rgb), alpha), axis=-1)


def _deduplicate_palette(
    palette_rgba: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse display-identical nodes while preserving palette order."""
    palette = np.asarray(palette_rgba, dtype=np.uint8).reshape(-1, 4)
    unique: list[np.ndarray] = []
    indices: dict[tuple[int, int, int, int], int] = {}
    remap = np.empty(len(palette), dtype=np.int32)
    for position, color in enumerate(palette):
        key = tuple(int(channel) for channel in color)
        label = indices.get(key)
        if label is None:
            label = len(unique)
            indices[key] = label
            unique.append(color.copy())
        remap[position] = label
    return np.stack(unique), remap


def _component_map(
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = labels.shape
    pixel_count = height * width
    indices = np.arange(pixel_count, dtype=np.int32).reshape(height, width)
    horizontal = labels[:, :-1] == labels[:, 1:]
    vertical = labels[:-1, :] == labels[1:, :]
    first = np.concatenate((
        indices[:, :-1][horizontal], indices[:-1, :][vertical]
    ))
    second = np.concatenate((
        indices[:, 1:][horizontal], indices[1:, :][vertical]
    ))
    graph = sparse.coo_matrix(
        (np.ones(len(first), dtype=np.uint8), (first, second)),
        shape=(pixel_count, pixel_count),
    ).tocsr()
    count, flat_components = csgraph.connected_components(
        graph, directed=False, return_labels=True
    )
    area = np.bincount(flat_components, minlength=count).astype(np.int64)
    flat_labels = labels.ravel()
    component_label = np.rint(
        np.bincount(flat_components, weights=flat_labels, minlength=count) / area
    ).astype(np.int32)
    # Preserve the former deterministic ID order: palette label first, then
    # raster-order discovery within that label. Merge conflict tie-breaking
    # intentionally depends on these IDs.
    first_pixel = np.full(count, pixel_count, dtype=np.int64)
    np.minimum.at(first_pixel, flat_components, np.arange(pixel_count))
    order = np.lexsort((first_pixel, component_label))
    remap = np.empty(count, dtype=np.int32)
    remap[order] = np.arange(count, dtype=np.int32)
    flat_components = remap[flat_components]
    area = area[order]
    component_label = component_label[order]
    return flat_components.reshape(labels.shape), component_label, area


def _adjacency(
    component_map: np.ndarray, count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first = np.concatenate((
        component_map[:, :-1].ravel(), component_map[:-1].ravel()
    ))
    second = np.concatenate((
        component_map[:, 1:].ravel(), component_map[1:].ravel()
    ))
    changed = first != second
    low = np.minimum(first[changed], second[changed]).astype(np.int64)
    high = np.maximum(first[changed], second[changed]).astype(np.int64)
    keys = low * np.int64(count) + high
    unique, shared = np.unique(keys, return_counts=True)
    return unique // count, unique % count, shared.astype(np.int64)


def _perceptual_importance(
    lab_alpha: np.ndarray,
    visible: np.ndarray,
    config: PosterizerConfig,
) -> np.ndarray:
    """Balance spatial detail and color rarity against raw pixel population."""
    lab = np.asarray(lab_alpha[..., :3], dtype=np.float64)
    gradient2 = np.zeros(lab.shape[:2], dtype=np.float64)
    for channel in range(3):
        gx = ndimage.sobel(lab[..., channel], axis=1, mode="reflect") / 8.0
        gy = ndimage.sobel(lab[..., channel], axis=0, mode="reflect") / 8.0
        gradient2 += gx * gx + gy * gy
    gradient = np.sqrt(gradient2)
    blurred = ndimage.gaussian_filter(lab, sigma=(2.0, 2.0, 0.0), mode="reflect")
    local_contrast = np.linalg.norm(lab - blurred, axis=2)
    detail = gradient + 0.75 * local_contrast
    active_detail = detail[visible]
    scale = float(np.quantile(active_detail, 0.9)) if len(active_detail) else 1.0
    detail = np.clip(detail / max(scale, 1e-9), 0.0, 4.0)
    detail_factor = 1.0 + max(0.0, float(config.detail_priority)) * detail

    lightness = np.clip((lab[..., 0] * 15.999).astype(np.int32), 0, 15)
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    chroma_bin = np.clip((chroma / 0.4 * 11.999).astype(np.int32), 0, 11)
    hue = (np.arctan2(lab[..., 2], lab[..., 1]) + np.pi) / (2.0 * np.pi)
    hue_bin = np.clip((hue * 23.999).astype(np.int32), 0, 23)
    hue_bin[chroma < 0.015] = 0
    keys = (lightness * 12 + chroma_bin) * 24 + hue_bin
    active_keys = keys[visible]
    counts = np.bincount(active_keys, minlength=16 * 12 * 24).astype(np.float64)
    exponent = float(np.clip(config.population_exponent, 0.0, 1.0))
    rarity = np.ones(keys.shape, dtype=np.float64)
    if len(active_keys):
        rarity[visible] = np.maximum(counts[active_keys], 1.0) ** (exponent - 1.0)
    importance = detail_factor * rarity
    active = importance[visible]
    if len(active):
        importance /= max(float(np.mean(active)), 1e-12)
    importance = np.clip(importance, 0.03, 30.0)
    importance[~visible] = 0.03
    return importance


def _edge_aware_structure(
    lab_alpha: np.ndarray,
    visible: np.ndarray,
    config: PosterizerConfig,
) -> np.ndarray:
    """Build a locally consensual color field without crossing strong edges."""
    source = np.asarray(lab_alpha, dtype=np.float64)
    radius = min(12, max(0, int(config.structure_radius)))
    threshold = max(0.0, float(config.structure_threshold))
    if radius == 0 or threshold <= 0.0:
        return source.copy()

    height, width = source.shape[:2]
    guide = source[..., :3]
    accumulated = np.zeros_like(guide)
    normalization = np.zeros((height, width), dtype=np.float64)
    spatial_sigma = max(0.5, 0.65 * radius)
    inverse_spatial = 0.5 / (spatial_sigma * spatial_sigma)
    inverse_range = 0.5 / (threshold * threshold)
    support = np.asarray(visible, dtype=bool)

    for dy in range(-radius, radius + 1):
        source_y = slice(max(0, -dy), min(height, height - dy))
        target_y = slice(max(0, dy), min(height, height + dy))
        for dx in range(-radius, radius + 1):
            source_x = slice(max(0, -dx), min(width, width - dx))
            target_x = slice(max(0, dx), min(width, width + dx))
            difference = (
                guide[target_y, target_x] - guide[source_y, source_x]
            )
            weight = np.exp(
                -(dx * dx + dy * dy) * inverse_spatial
                - np.einsum("ijk,ijk->ij", difference, difference)
                * inverse_range
            )
            weight *= (
                support[target_y, target_x] == support[source_y, source_x]
            )
            accumulated[target_y, target_x] += (
                guide[source_y, source_x] * weight[..., None]
            )
            normalization[target_y, target_x] += weight

    result = source.copy()
    result[..., :3] = accumulated / np.maximum(normalization[..., None], 1e-15)
    return result


def _transport_texture(
    lab_alpha: np.ndarray,
    config: PosterizerConfig,
) -> np.ndarray:
    """Push measured local lightness detail across palette boundaries."""
    source = np.asarray(lab_alpha, dtype=np.float64)
    amount = max(0.0, float(config.texture_priority))
    if amount <= 0.0:
        return source.copy()
    lightness = source[..., 0]
    local_field = ndimage.gaussian_filter(lightness, sigma=1.25, mode="reflect")
    result = source.copy()
    result[..., 0] = np.clip(
        lightness + amount * (lightness - local_field), 0.0, 1.0
    )
    return result


def _smoothstep(low: float, high: float, value: np.ndarray) -> np.ndarray:
    scaled = np.clip((value - low) / max(high - low, 1e-12), 0.0, 1.0)
    return scaled * scaled * (3.0 - 2.0 * scaled)


def _mix_plan(
    source_lab_alpha: np.ndarray,
    labels: np.ndarray,
    palette_lab_alpha: np.ndarray,
    neighbors: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project each pixel onto nearby palette segments rooted at its label."""
    height, width = labels.shape
    source = source_lab_alpha[..., :3].reshape(-1, 3)
    flat_labels = labels.ravel()
    partner = flat_labels.copy()
    fraction = np.zeros(len(flat_labels), dtype=np.float64)
    improvement = np.zeros(len(flat_labels), dtype=np.float64)
    candidate_count = max(1, int(neighbors))

    for base in range(len(palette_lab_alpha)):
        active = np.flatnonzero(flat_labels == base)
        if not len(active) or palette_lab_alpha[base, 3] <= 4.0 / 255.0:
            continue
        target = source[active]
        origin = palette_lab_alpha[base, :3]
        base_error = np.sum((target - origin) ** 2, axis=1)
        best_error = base_error.copy()
        best_fraction = np.zeros(len(active), dtype=np.float64)
        best_partner = np.full(len(active), base, dtype=np.int32)
        palette_distance = np.sum(
            (palette_lab_alpha[:, :3] - origin) ** 2, axis=1
        )
        same_support = palette_lab_alpha[:, 3] > 4.0 / 255.0
        palette_distance[~same_support] = np.inf
        candidates = np.argsort(palette_distance, kind="stable")[1:]
        candidates = candidates[np.isfinite(palette_distance[candidates])]
        candidates = candidates[:candidate_count]
        for other in candidates:
            direction = palette_lab_alpha[other, :3] - origin
            denominator = float(np.dot(direction, direction))
            if denominator <= 1e-14:
                continue
            projection = np.clip(
                np.einsum("ij,j->i", target - origin, direction)
                / denominator,
                0.0,
                1.0,
            )
            approximation = origin + projection[:, None] * direction
            error = np.sum((target - approximation) ** 2, axis=1)
            better = error < best_error
            best_error[better] = error[better]
            best_fraction[better] = projection[better]
            best_partner[better] = other
        partner[active] = best_partner
        fraction[active] = best_fraction
        improvement[active] = np.maximum(base_error - best_error, 0.0)
    return (
        partner.reshape(height, width),
        fraction.reshape(height, width),
        improvement.reshape(height, width),
    )


def _segmented_error_diffusion(
    probability: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    """Realize local mixture density without crossing palette-pair regions."""
    height, width = probability.shape
    work = probability.astype(np.float64).copy()
    output = np.zeros((height, width), dtype=bool)
    for y in range(height):
        if y % 2 == 0:
            xs = range(width)
            direction = 1
        else:
            xs = range(width - 1, -1, -1)
            direction = -1
        for x in xs:
            if probability[y, x] <= 0.0:
                work[y, x] = 0.0
                continue
            bit = work[y, x] >= 0.5
            output[y, x] = bit
            error = work[y, x] - float(bit)
            following = x + direction
            previous = x - direction
            group = groups[y, x]
            if 0 <= following < width and groups[y, following] == group:
                work[y, following] += error * 7.0 / 16.0
            if y + 1 >= height:
                continue
            if 0 <= previous < width and groups[y + 1, previous] == group:
                work[y + 1, previous] += error * 3.0 / 16.0
            if groups[y + 1, x] == group:
                work[y + 1, x] += error * 5.0 / 16.0
            if (
                0 <= following < width
                and groups[y + 1, following] == group
            ):
                work[y + 1, following] += error * 1.0 / 16.0
    return output


def _spatial_mix_labels(
    source_lab_alpha: np.ndarray,
    labels: np.ndarray,
    palette_lab_alpha: np.ndarray,
    visible: np.ndarray,
    config: PosterizerConfig,
) -> tuple[np.ndarray, float]:
    """Synthesize missing tones from nearby nodes inside stable regions."""
    strength = max(0.0, float(config.mixing_strength))
    if strength <= 0.0 or len(palette_lab_alpha) < 2:
        return labels.copy(), 0.0
    partner, fraction, improvement = _mix_plan(
        source_lab_alpha,
        labels,
        palette_lab_alpha,
        max(1, int(config.mixing_neighbors)),
    )
    stable = (
        ndimage.minimum_filter(labels, size=3, mode="nearest")
        == ndimage.maximum_filter(labels, size=3, mode="nearest")
    )
    stable &= np.asarray(visible, dtype=bool)

    lab = source_lab_alpha[..., :3]
    gradient2 = np.zeros(labels.shape, dtype=np.float64)
    for channel in range(3):
        gx = ndimage.sobel(lab[..., channel], axis=1, mode="reflect") / 8.0
        gy = ndimage.sobel(lab[..., channel], axis=0, mode="reflect") / 8.0
        gradient2 += gx * gx + gy * gy
    gradient = np.sqrt(gradient2)
    active_gradient = gradient[stable]
    gradient_scale = (
        float(np.quantile(active_gradient, 0.7)) if len(active_gradient) else 0.02
    )
    smoothness = 1.0 - _smoothstep(
        0.7 * gradient_scale, 2.2 * gradient_scale, gradient
    )
    active_improvement = improvement[stable & (improvement > 0.0)]
    improvement_scale = (
        float(np.quantile(active_improvement, 0.55))
        if len(active_improvement)
        else 1e-5
    )
    worthwhile = _smoothstep(
        0.12 * improvement_scale, 1.8 * improvement_scale, improvement
    )
    probability = np.clip(
        strength * fraction * stable * smoothness * worthwhile, 0.0, 1.0
    )
    groups = labels.astype(np.int64) * len(palette_lab_alpha) + partner
    groups[probability <= 0.0] = -1
    choose_partner = _segmented_error_diffusion(probability, groups)
    mixed = labels.copy()
    mixed[choose_partner] = partner[choose_partner]
    return mixed, float(np.mean(choose_partner[visible])) if np.any(visible) else 0.0


def _safe_correlation(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=np.float64).ravel()
    right = np.asarray(second, dtype=np.float64).ravel()
    if len(left) < 2:
        return 1.0
    left = left - np.mean(left)
    right = right - np.mean(right)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-15:
        return 1.0 if np.max(np.abs(left - right)) <= 1e-12 else 0.0
    return float(np.dot(left, right) / denominator)


def _weighted_mean(value: np.ndarray, weight: np.ndarray) -> float:
    return float(np.sum(weight * value) / max(float(np.sum(weight)), 1e-15))


def _chart_relation_metrics(
    source: np.ndarray,
    target: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    importance: np.ndarray,
) -> dict[str, float]:
    source_delta = source[first] - source[second]
    target_delta = target[first] - target[second]
    source_distance = np.linalg.norm(source_delta, axis=1)
    target_distance = np.linalg.norm(target_delta, axis=1)
    weight = np.sqrt(importance[first] * importance[second])
    source_energy = _weighted_mean(source_distance * source_distance, weight)
    error = target_distance - source_distance
    stress = np.sqrt(
        _weighted_mean(error * error, weight) / max(source_energy, 1e-30)
    )
    source_mean = _weighted_mean(source_distance, weight)
    target_mean = _weighted_mean(target_distance, weight)
    source_centered = source_distance - source_mean
    target_centered = target_distance - target_mean
    covariance = _weighted_mean(source_centered * target_centered, weight)
    variance = _weighted_mean(source_centered * source_centered, weight)
    variance *= _weighted_mean(target_centered * target_centered, weight)
    correlation = covariance / np.sqrt(max(variance, 1e-30))
    relation_energy = weight * source_distance * source_distance
    collapsed = float(np.sum(
        relation_energy[target_distance < 0.25 * source_distance]
    ) / max(float(np.sum(relation_energy)), 1e-30))
    return {
        "stress": float(stress),
        "correlation": float(correlation),
        "collapsed": collapsed,
    }


def _chart_diagnostics(
    source_lab: np.ndarray,
    poster_lab: np.ndarray,
    importance: np.ndarray,
    visible: np.ndarray,
) -> dict[str, float]:
    """Measure deformation of source color relations at a viewing scale."""
    active = np.flatnonzero(np.asarray(visible, dtype=bool).ravel())
    if len(active) < 2:
        return {
            "chart_global_stress": 0.0,
            "chart_local_stress": 0.0,
            "chart_distance_correlation": 1.0,
            "chart_collapsed_relation_energy": 0.0,
            "chart_worst_sector_alignment": 1.0,
            "chart_worst_sector_relative_error": 0.0,
            "chart_worst_sector_start_degrees": 0.0,
        }
    source_field = ndimage.gaussian_filter(
        source_lab[..., :3], sigma=(1.5, 1.5, 0.0), mode="reflect"
    )
    poster_field = ndimage.gaussian_filter(
        poster_lab[..., :3], sigma=(1.5, 1.5, 0.0), mode="reflect"
    )
    source_flat = source_field.reshape(-1, 3)
    poster_flat = poster_field.reshape(-1, 3)
    flat_importance = importance.ravel()
    pair_limit = 65536
    random = np.random.default_rng(7351)
    global_first = random.choice(active, size=pair_limit, replace=True)
    global_second = random.choice(active, size=pair_limit, replace=True)
    equal = global_first == global_second
    while np.any(equal):
        global_second[equal] = random.choice(
            active, size=int(np.sum(equal)), replace=True
        )
        equal = global_first == global_second
    global_metrics = _chart_relation_metrics(
        source_flat,
        poster_flat,
        global_first,
        global_second,
        flat_importance,
    )

    height, width = visible.shape
    index = np.arange(height * width, dtype=np.int64).reshape(height, width)
    local_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for radius in (1, 2, 4, 8, 16, 32):
        for dy, dx in (
            (0, radius), (radius, 0), (radius, radius), (radius, -radius)
        ):
            if abs(dy) >= height or abs(dx) >= width:
                continue
            y0 = slice(max(0, -dy), min(height, height - dy))
            y1 = slice(max(0, dy), min(height, height + dy))
            x0 = slice(max(0, -dx), min(width, width - dx))
            x1 = slice(max(0, dx), min(width, width + dx))
            pair_visible = visible[y0, x0] & visible[y1, x1]
            local_pairs.append((
                index[y0, x0][pair_visible], index[y1, x1][pair_visible]
            ))
    local_first = np.concatenate([pair[0] for pair in local_pairs])
    local_second = np.concatenate([pair[1] for pair in local_pairs])
    if len(local_first) > pair_limit:
        chosen = np.random.default_rng(9277).choice(
            len(local_first), size=pair_limit, replace=False
        )
        local_first = local_first[chosen]
        local_second = local_second[chosen]
    local_metrics = _chart_relation_metrics(
        source_flat,
        poster_flat,
        local_first,
        local_second,
        flat_importance,
    )

    source_chroma = np.hypot(source_lab[..., 1], source_lab[..., 2])
    poster_chroma = np.hypot(poster_lab[..., 1], poster_lab[..., 2])
    source_hue = (
        np.degrees(np.arctan2(source_lab[..., 2], source_lab[..., 1])) + 360.0
    ) % 360.0
    chroma_energy = importance * source_chroma * source_chroma
    active_chroma_energy = float(np.sum(chroma_energy[visible]))
    relative_chroma_error = np.hypot(
        poster_lab[..., 1] - source_lab[..., 1],
        poster_lab[..., 2] - source_lab[..., 2],
    ) / np.maximum(source_chroma, 0.02)
    chroma_alignment = (
        source_lab[..., 1] * poster_lab[..., 1]
        + source_lab[..., 2] * poster_lab[..., 2]
    ) / np.maximum(source_chroma * poster_chroma, 1e-12)
    worst_error = -1.0
    worst_alignment = 1.0
    worst_sector = 0.0
    for low in range(0, 360, 30):
        sector = visible & (source_chroma > 0.03)
        sector &= (source_hue >= low) & (source_hue < low + 30)
        sector_energy = float(np.sum(chroma_energy[sector]))
        if sector_energy < 0.01 * active_chroma_energy:
            continue
        sector_error = _weighted_mean(
            relative_chroma_error[sector], chroma_energy[sector]
        )
        if sector_error > worst_error:
            worst_error = sector_error
            worst_alignment = _weighted_mean(
                chroma_alignment[sector], chroma_energy[sector]
            )
            worst_sector = float(low)
    return {
        "chart_global_stress": global_metrics["stress"],
        "chart_local_stress": local_metrics["stress"],
        "chart_distance_correlation": local_metrics["correlation"],
        "chart_collapsed_relation_energy": local_metrics["collapsed"],
        "chart_worst_sector_alignment": worst_alignment,
        "chart_worst_sector_relative_error": max(0.0, worst_error),
        "chart_worst_sector_start_degrees": worst_sector,
    }


def _utility_diagnostics(
    source_lab: np.ndarray,
    poster_lab: np.ndarray,
    visible: np.ndarray,
) -> dict[str, float]:
    """Measure retained texture and chromatic relationships, not only error."""
    active = np.asarray(visible, dtype=bool)
    source_lightness = source_lab[..., 0]
    poster_lightness = poster_lab[..., 0]
    source_texture = source_lightness - ndimage.gaussian_filter(
        source_lightness, sigma=1.25, mode="reflect"
    )
    poster_texture = poster_lightness - ndimage.gaussian_filter(
        poster_lightness, sigma=1.25, mode="reflect"
    )
    texture_mask = active.copy()
    if np.any(active):
        threshold = float(np.quantile(np.abs(source_texture[active]), 0.75))
        texture_mask &= np.abs(source_texture) >= threshold

    source_chroma = np.hypot(source_lab[..., 1], source_lab[..., 2])
    poster_chroma = np.hypot(poster_lab[..., 1], poster_lab[..., 2])
    chroma_mask = active & (source_chroma > 0.02)
    hue_dot = (
        source_lab[..., 1] * poster_lab[..., 1]
        + source_lab[..., 2] * poster_lab[..., 2]
    )
    hue_norm = np.maximum(source_chroma * poster_chroma, 1e-12)
    return {
        "texture_correlation": _safe_correlation(
            source_texture[texture_mask], poster_texture[texture_mask]
        ),
        "chroma_correlation": _safe_correlation(
            source_chroma[chroma_mask], poster_chroma[chroma_mask]
        ),
        "mean_hue_alignment": float(np.mean(
            hue_dot[chroma_mask] / hue_norm[chroma_mask]
        )) if np.any(chroma_mask) else 1.0,
    }


def _assign_oklch(
    values: np.ndarray,
    palette: np.ndarray,
    config: PosterizerConfig,
) -> np.ndarray:
    flat = values.reshape(-1, 4)
    labels = np.empty(len(flat), dtype=np.int32)
    for start in range(0, len(flat), 16384):
        stop = min(start + 16384, len(flat))
        labels[start:stop] = np.argmin(
            oklch_distance2(
                flat[start:stop], palette,
                lightness_weight=config.lightness_weight,
                chroma_weight=config.chroma_weight,
                hue_weight=config.hue_weight,
                alpha_weight=config.alpha_weight,
            ),
            axis=1,
        )
    return labels.reshape(values.shape[:2])


def _reserve_family_anchor(
    samples: np.ndarray,
    sample_weights: np.ndarray,
    primary_palette: np.ndarray,
    family_palette: np.ndarray,
    config: PosterizerConfig,
) -> np.ndarray:
    """Trade the least costly tonal node for one missing color-family anchor."""
    primary = np.asarray(primary_palette, dtype=np.float64)
    family = np.asarray(family_palette, dtype=np.float64)
    if len(primary) < 2 or not len(family):
        return primary.copy()

    family_to_primary = oklch_distance2(
        family,
        primary,
        lightness_weight=config.lightness_weight,
        chroma_weight=config.chroma_weight,
        hue_weight=config.hue_weight,
        alpha_weight=config.alpha_weight,
    )
    family_assignment = np.argmin(
        oklch_distance2(
            samples,
            family,
            lightness_weight=config.lightness_weight,
            chroma_weight=config.chroma_weight,
            hue_weight=config.hue_weight,
            alpha_weight=config.alpha_weight,
        ),
        axis=1,
    )
    family_mass = np.bincount(
        family_assignment,
        weights=sample_weights,
        minlength=len(family),
    )
    representation_gap = np.min(family_to_primary, axis=1)
    anchor_score = representation_gap * np.sqrt(
        family_mass / max(float(np.sum(family_mass)), 1e-15)
    )
    anchor = family[int(np.argmax(anchor_score))]

    primary_distance = oklch_distance2(
        samples,
        primary,
        lightness_weight=config.lightness_weight,
        chroma_weight=config.chroma_weight,
        hue_weight=config.hue_weight,
        alpha_weight=config.alpha_weight,
    )
    rows = np.arange(len(samples))
    best_label = np.argmin(primary_distance, axis=1)
    best_distance = primary_distance[rows, best_label].copy()
    primary_distance[rows, best_label] = np.inf
    second_distance = np.min(primary_distance, axis=1)
    anchor_distance = oklch_distance2(
        samples,
        anchor[None, :],
        lightness_weight=config.lightness_weight,
        chroma_weight=config.chroma_weight,
        hue_weight=config.hue_weight,
        alpha_weight=config.alpha_weight,
    )[:, 0]
    losses = np.empty(len(primary), dtype=np.float64)
    for dropped in range(len(primary)):
        remaining = np.where(
            best_label == dropped, second_distance, best_distance
        )
        losses[dropped] = np.sum(
            sample_weights * np.minimum(remaining, anchor_distance)
        )
    result = primary.copy()
    result[int(np.argmin(losses))] = anchor
    return result


def _cleanup_components(
    labels: np.ndarray,
    palette_lab_alpha: np.ndarray,
    config: PosterizerConfig,
) -> tuple[np.ndarray, int]:
    current = labels.copy()
    merged = 0
    for _ in range(max(0, int(config.cleanup_rounds))):
        component_map, component_label, area = _component_map(current)
        count = len(component_label)
        first, second, shared = _adjacency(component_map, count)
        if not len(first):
            break
        source = np.concatenate((first, second))
        target = np.concatenate((second, first))
        boundaries = np.concatenate((shared, shared))
        eligible = area[source] <= max(0, int(config.minimum_island))
        eligible &= (
            (area[target] > area[source])
            | ((area[target] == area[source]) & (target < source))
        )
        source = source[eligible]
        target = target[eligible]
        boundaries = boundaries[eligible]
        if not len(source):
            break
        source_color = component_label[source]
        target_color = component_label[target]
        source_visible = palette_lab_alpha[source_color, 3] > 4.0 / 255.0
        target_visible = palette_lab_alpha[target_color, 3] > 4.0 / 255.0
        same_support = source_visible == target_visible
        source = source[same_support]
        target = target[same_support]
        boundaries = boundaries[same_support]
        source_color = source_color[same_support]
        target_color = target_color[same_support]
        if not len(source):
            break
        pair_distance = oklch_pair_distance2(
            palette_lab_alpha[source_color],
            palette_lab_alpha[target_color],
            lightness_weight=config.lightness_weight,
            chroma_weight=config.chroma_weight,
            hue_weight=config.hue_weight,
            alpha_weight=config.alpha_weight,
        )
        score = pair_distance / np.maximum(boundaries, 1)
        ranking = np.lexsort((target, score, source))
        mapping = component_label.copy()
        consumed = np.zeros(count, dtype=bool)
        protected = np.zeros(count, dtype=bool)
        accepted = 0
        for candidate in ranking:
            source_id = int(source[candidate])
            target_id = int(target[candidate])
            if consumed[source_id] or protected[source_id] or consumed[target_id]:
                continue
            mapping[source_id] = component_label[target_id]
            consumed[source_id] = True
            protected[target_id] = True
            accepted += 1
        if not accepted:
            break
        current = mapping[component_map]
        merged += accepted
    return current, merged


def posterize_array(
    rgba: np.ndarray,
    config: PosterizerConfig = PosterizerConfig(),
    *,
    title: str = "posterized image",
) -> PosterizerResult:
    started = perf_counter()
    source = np.asarray(rgba, dtype=np.uint8)
    if source.ndim != 3 or source.shape[2] != 4:
        raise ValueError("expected an H x W x 4 uint8 RGBA array")
    base_config = VectorizerConfig(
        colors=max(1, int(config.colors)),
        coarse_side=max(1, int(config.coarse_side)),
        alpha_mode=config.alpha_mode,
        alpha_cutoff=int(config.alpha_cutoff),
        trim_transparent=bool(config.trim_transparent),
    )
    source, alpha_mode = _normalize_alpha(source, base_config)
    source, crop = _trim_transparent(source, base_config)
    lab_alpha = _rgba_to_lab_alpha(source)
    visible = source[..., 3] > 4
    importance_map = _perceptual_importance(lab_alpha, visible, config)
    method = str(config.method).lower()
    structure_lab_alpha = (
        _edge_aware_structure(lab_alpha, visible, config)
        if method == "oklch"
        else lab_alpha.copy()
    )
    assignment_lab_alpha = (
        _transport_texture(lab_alpha, config)
        if method == "oklch"
        else lab_alpha.copy()
    )
    prepared = perf_counter()

    bifurcation_gain = 0.0
    if method == "oklch":
        samples = structure_lab_alpha[visible]
        sample_weights = importance_map[visible]
        if not len(samples):
            samples = lab_alpha.reshape(-1, 4)
            sample_weights = np.ones(len(samples), dtype=np.float64)
        limit = max(1, int(config.sample_limit))
        if len(samples) > limit:
            stride = int(np.ceil(len(samples) / limit))
            samples = samples[::stride]
            sample_weights = sample_weights[::stride]
        tree = bifurcate_palette(
            samples,
            config.colors,
            sample_weights=sample_weights,
            lightness_weight=config.lightness_weight,
            chroma_weight=config.chroma_weight,
            hue_weight=config.hue_weight,
            alpha_weight=config.alpha_weight,
            minimum_leaf=config.minimum_leaf,
            family_priority=0.0,
        )
        intended_palette = separate_nodes(tree, config.node_separation)
        if config.family_priority > 0.0 and int(config.colors) >= 4:
            family_tree = bifurcate_palette(
                samples,
                config.colors,
                sample_weights=sample_weights,
                lightness_weight=config.lightness_weight,
                chroma_weight=config.chroma_weight,
                hue_weight=config.hue_weight,
                alpha_weight=config.alpha_weight,
                minimum_leaf=config.minimum_leaf,
                family_priority=config.family_priority,
            )
            intended_palette = _reserve_family_anchor(
                samples,
                sample_weights,
                intended_palette,
                separate_nodes(family_tree, config.node_separation),
                config,
            )
        palette_rgba, _remap = _deduplicate_palette(
            lab_alpha_to_rgba(intended_palette)
        )
        # Assignment must use the colors that will actually be displayed,
        # after gamut mapping and 8-bit rounding.
        palette_lab_alpha = _rgba_to_lab_alpha(palette_rgba)
        labels = _assign_oklch(assignment_lab_alpha, palette_lab_alpha, config)
        bifurcation_gain = tree.total_gain
    elif method == "inherited":
        coarse = _resize_rgba(source, config.coarse_side)
        feature_palette = _seed_palette(_rgba_features(coarse), config.colors)
        labels = _nearest_assign(_rgba_features(source), feature_palette)
        palette_rgba = _feature_to_rgba(feature_palette, source, labels)
        palette_rgba, remap = _deduplicate_palette(palette_rgba)
        labels = remap[labels]
        palette_lab_alpha = _rgba_to_lab_alpha(palette_rgba)
    else:
        raise ValueError("method must be 'oklch' or 'inherited'")
    if np.any(~visible):
        transparent = np.flatnonzero(np.all(palette_rgba == 0, axis=1))
        if len(transparent):
            transparent_label = int(transparent[0])
        else:
            transparent_label = len(palette_rgba)
            palette_rgba = np.vstack((
                palette_rgba, np.zeros((1, 4), dtype=np.uint8)
            ))
            palette_lab_alpha = np.vstack((
                palette_lab_alpha, np.zeros((1, 4), dtype=np.float64)
            ))
        labels = labels.copy()
        labels[~visible] = transparent_label
    quantized = perf_counter()

    labels, cleaned = _cleanup_components(labels, palette_lab_alpha, config)
    cleaned_at = perf_counter()
    labels, mixed_fraction = _spatial_mix_labels(
        lab_alpha, labels, palette_lab_alpha, visible, config
    )
    mixed_at = perf_counter()
    posterized = palette_rgba[labels]
    poster_lab_alpha = _rgba_to_lab_alpha(posterized)

    rgba_delta = source.astype(np.float64) - posterized.astype(np.float64)
    mse = float(np.mean(rgba_delta * rgba_delta))
    assigned = labels.ravel()
    assigned_error = oklch_pair_distance2(
        lab_alpha.reshape(-1, 4),
        palette_lab_alpha[assigned],
        lightness_weight=config.lightness_weight,
        chroma_weight=config.chroma_weight,
        hue_weight=config.hue_weight,
        alpha_weight=config.alpha_weight,
    )
    perceptual_rmse = float(np.sqrt(np.mean(assigned_error)))
    flat_importance = importance_map.ravel()
    weighted_perceptual_rmse = float(np.sqrt(
        np.sum(flat_importance * assigned_error) / np.sum(flat_importance)
    ))
    source_lowpass = ndimage.gaussian_filter(
        lab_alpha[..., :3], sigma=(1.5, 1.5, 0.0), mode="reflect"
    )
    poster_lowpass = ndimage.gaussian_filter(
        poster_lab_alpha[..., :3], sigma=(1.5, 1.5, 0.0), mode="reflect"
    )
    lowpass_perceptual_rmse = float(np.sqrt(np.mean(
        (source_lowpass[visible] - poster_lowpass[visible]) ** 2
    ))) if np.any(visible) else 0.0
    palette_hex = [
        f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}{color[3]:02x}"
        for color in palette_rgba
    ]
    completed = perf_counter()
    diagnostics: dict[str, float | int | str | list] = {
        "method": f"posterizer_{method}",
        "width": int(source.shape[1]),
        "height": int(source.shape[0]),
        "crop_x": int(crop[0]),
        "crop_y": int(crop[1]),
        "alpha_mode": alpha_mode,
        "requested_colors": int(config.colors),
        "palette_colors": int(len(palette_rgba)),
        "visible_palette_colors": int(np.sum(palette_rgba[:, 3] > 4)),
        "palette": palette_hex,
        "node_separation": float(config.node_separation),
        "detail_priority": float(config.detail_priority),
        "population_exponent": float(config.population_exponent),
        "family_priority": float(config.family_priority),
        "structure_radius": int(config.structure_radius),
        "structure_threshold": float(config.structure_threshold),
        "structure_delta": float(np.mean(np.linalg.norm(
            lab_alpha[..., :3] - structure_lab_alpha[..., :3], axis=2
        ))),
        "texture_priority": float(config.texture_priority),
        "texture_delta": float(np.mean(np.abs(
            lab_alpha[..., 0] - assignment_lab_alpha[..., 0]
        ))),
        "mixing_strength": float(config.mixing_strength),
        "mixing_neighbors": int(config.mixing_neighbors),
        "mixed_fraction": mixed_fraction,
        "bifurcation_gain": float(bifurcation_gain),
        "cleaned_components": int(cleaned),
        "rgba_mse_255": mse,
        "perceptual_rmse": perceptual_rmse,
        "importance_weighted_perceptual_rmse": weighted_perceptual_rmse,
        "lowpass_perceptual_rmse": lowpass_perceptual_rmse,
        **_chart_diagnostics(
            lab_alpha, poster_lab_alpha, importance_map, visible
        ),
        **_utility_diagnostics(lab_alpha, poster_lab_alpha, visible),
        "importance_p90": float(np.quantile(importance_map[visible], 0.9))
        if np.any(visible) else 1.0,
        "preparation_ms": 1000.0 * (prepared - started),
        "quantization_ms": 1000.0 * (quantized - prepared),
        "cleanup_ms": 1000.0 * (cleaned_at - quantized),
        "mixing_ms": 1000.0 * (mixed_at - cleaned_at),
        "total_ms": 1000.0 * (completed - started),
    }
    return PosterizerResult(
        labels, palette_rgba, posterized, diagnostics
    )


def posterize_image(
    source: str | Path,
    destination: str | Path,
    config: PosterizerConfig = PosterizerConfig(),
) -> PosterizerResult:
    source_path = Path(source)
    with Image.open(source_path) as image:
        rgba = np.asarray(image.convert("RGBA"))
    result = posterize_array(rgba, config, title=source_path.name)
    result.save(destination)
    return result
