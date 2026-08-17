"""Posterization pipeline with inherited compact exact-lattice output."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from time import perf_counter
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

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
from tlvector_v2.lattice import compact_lattice_svg, deterministic_svgz
from tlvector_v2.merge import _adjacency, _component_map

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
    minimum_leaf: int = 16
    sample_limit: int = 65536
    coarse_side: int = 160
    minimum_island: int = 6
    cleanup_rounds: int = 1
    alpha_mode: str = "auto"
    alpha_cutoff: int = 128
    trim_transparent: bool = True
    gzip_level: int = 9


@dataclass
class PosterizerResult:
    labels: np.ndarray
    palette_rgba: np.ndarray
    posterized_rgba: np.ndarray
    svg: str
    svgz: bytes
    diagnostics: dict[str, float | int | str | list]

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        suffix = destination.suffix.lower()
        if suffix == ".svgz":
            destination.write_bytes(self.svgz)
        elif suffix == ".svg":
            destination.write_text(self.svg, encoding="utf-8")
        elif suffix == ".png":
            Image.fromarray(self.posterized_rgba, "RGBA").save(destination)
        elif suffix == ".json":
            destination.write_text(
                json.dumps(self.diagnostics, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        else:
            raise ValueError("posterizer output must end in .png, .svg, .svgz, or .json")


def _rgba_to_lab_alpha(rgba: np.ndarray) -> np.ndarray:
    rgb = rgba[..., :3].astype(np.float64) / 255.0
    alpha = rgba[..., 3:4].astype(np.float64) / 255.0
    return np.concatenate((_srgb_to_oklab(rgb), alpha), axis=-1)


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
    prepared = perf_counter()

    method = str(config.method).lower()
    bifurcation_gain = 0.0
    if method == "oklch":
        visible = source[..., 3] > 4
        samples = lab_alpha[visible]
        if not len(samples):
            samples = lab_alpha.reshape(-1, 4)
        limit = max(1, int(config.sample_limit))
        if len(samples) > limit:
            stride = int(np.ceil(len(samples) / limit))
            samples = samples[::stride]
        tree = bifurcate_palette(
            samples,
            config.colors,
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
    visible = source[..., 3] > 4
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
    svg, svg_stats = compact_lattice_svg(labels, palette_rgba, title=title)
    svg_at = perf_counter()
    svgz = deterministic_svgz(svg, level=config.gzip_level)
    completed = perf_counter()

    rgba_delta = source.astype(np.float64) - posterized.astype(np.float64)
    mse = float(np.mean(rgba_delta * rgba_delta))
    perceptual = oklch_distance2(
        lab_alpha.reshape(-1, 4),
        palette_lab_alpha,
        lightness_weight=config.lightness_weight,
        chroma_weight=config.chroma_weight,
        hue_weight=config.hue_weight,
        alpha_weight=config.alpha_weight,
    )
    assigned = labels.ravel()
    perceptual_rmse = float(np.sqrt(np.mean(perceptual[np.arange(len(assigned)), assigned])))
    palette_hex = [
        f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}{color[3]:02x}"
        for color in palette_rgba
    ]
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
        "bifurcation_gain": float(bifurcation_gain),
        "cleaned_components": int(cleaned),
        "rgba_mse_255": mse,
        "perceptual_rmse": perceptual_rmse,
        **svg_stats,
        "svgz_bytes": len(svgz),
        "svgz_ratio": len(svgz) / max(1, svg_stats["svg_bytes"]),
        "preparation_ms": 1000.0 * (prepared - started),
        "quantization_ms": 1000.0 * (quantized - prepared),
        "cleanup_ms": 1000.0 * (cleaned_at - quantized),
        "svg_ms": 1000.0 * (svg_at - cleaned_at),
        "gzip_ms": 1000.0 * (completed - svg_at),
        "total_ms": 1000.0 * (completed - started),
    }
    return PosterizerResult(
        labels, palette_rgba, posterized, svg, svgz, diagnostics
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
    if Path(destination).suffix.lower() == ".svg":
        ET.parse(destination)
    return result
