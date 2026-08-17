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
    prepared = perf_counter()

    method = str(config.method).lower()
    bifurcation_gain = 0.0
    if method == "oklch":
        samples = lab_alpha[visible]
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
        )
        palette_lab_alpha = separate_nodes(tree, config.node_separation)
        labels = _assign_oklch(lab_alpha, palette_lab_alpha, config)
        palette_rgba = lab_alpha_to_rgba(palette_lab_alpha)
        bifurcation_gain = tree.total_gain
    elif method == "inherited":
        coarse = _resize_rgba(source, config.coarse_side)
        feature_palette = _seed_palette(_rgba_features(coarse), config.colors)
        labels = _nearest_assign(_rgba_features(source), feature_palette)
        palette_rgba = _feature_to_rgba(feature_palette, source, labels)
        palette_lab_alpha = _rgba_to_lab_alpha(palette_rgba)
    else:
        raise ValueError("method must be 'oklch' or 'inherited'")
    if np.any(~visible):
        transparent_label = len(palette_rgba)
        palette_rgba = np.vstack((palette_rgba, np.zeros((1, 4), dtype=np.uint8)))
        palette_lab_alpha = np.vstack((
            palette_lab_alpha, np.zeros((1, 4), dtype=np.float64)
        ))
        labels = labels.copy()
        labels[~visible] = transparent_label
    quantized = perf_counter()

    labels, cleaned = _cleanup_components(labels, palette_lab_alpha, config)
    cleaned_at = perf_counter()
    posterized = palette_rgba[labels]

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
        "bifurcation_gain": float(bifurcation_gain),
        "cleaned_components": int(cleaned),
        "rgba_mse_255": mse,
        "perceptual_rmse": perceptual_rmse,
        "importance_weighted_perceptual_rmse": weighted_perceptual_rmse,
        "importance_p90": float(np.quantile(importance_map[visible], 0.9))
        if np.any(visible) else 1.0,
        "preparation_ms": 1000.0 * (prepared - started),
        "quantization_ms": 1000.0 * (quantized - prepared),
        "cleanup_ms": 1000.0 * (cleaned_at - quantized),
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
