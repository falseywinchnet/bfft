"""Converter v2: compact lattice output plus rate-distortion island merging."""

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
    _adaptive_quality_children,
    _feature_to_rgba,
    _nearest_assign,
    _normalize_alpha,
    _resize_rgba,
    _rgba_features,
    _seed_palette,
    _trim_transparent,
)

from .lattice import compact_lattice_svg, deterministic_svgz
from .merge import merge_error_per_byte


@dataclass(frozen=True)
class V2Config:
    colors: int = 128
    split_budget: int = 96
    split_target_mse: float = 20.0
    final_target_mse: float = 30.0
    coarse_side: int = 160
    minimum_region: int = 10
    merge_maximum_area: int = 32
    merge_rounds: int = 2
    alpha_mode: str = "auto"
    alpha_cutoff: int = 128
    trim_transparent: bool = True
    gzip_level: int = 9


@dataclass
class V2Result:
    labels: np.ndarray
    palette_rgba: np.ndarray
    svg: str
    svgz: bytes
    diagnostics: dict[str, float | int | str]

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        if destination.suffix.lower() == ".svgz":
            destination.write_bytes(self.svgz)
        else:
            destination.write_text(self.svg, encoding="utf-8")


def _mse(source: np.ndarray, labels: np.ndarray, palette: np.ndarray) -> float:
    delta = source.astype(np.float64) - palette[labels].astype(np.float64)
    return float(np.mean(delta * delta))


def vectorize_array_v2(
    rgba: np.ndarray,
    config: V2Config = V2Config(),
    *,
    title: str = "converter v2 image",
) -> V2Result:
    started = perf_counter()
    source = np.asarray(rgba, dtype=np.uint8)
    if source.ndim != 3 or source.shape[2] != 4:
        raise ValueError("expected an H x W x 4 uint8 RGBA array")
    base_config = VectorizerConfig(
        colors=int(config.colors),
        detail_colors=int(config.split_budget),
        target_mse=float(config.split_target_mse),
        coarse_side=int(config.coarse_side),
        minimum_region=int(config.minimum_region),
        alpha_mode=config.alpha_mode,
        alpha_cutoff=int(config.alpha_cutoff),
        trim_transparent=bool(config.trim_transparent),
    )
    source, alpha_mode = _normalize_alpha(source, base_config)
    source, crop = _trim_transparent(source, base_config)
    prepared = perf_counter()

    coarse = _resize_rgba(source, config.coarse_side)
    palette = _seed_palette(_rgba_features(coarse), config.colors)
    features = _rgba_features(source)
    structural = _nearest_assign(features, palette)
    labels, feature_palette, parents = _adaptive_quality_children(
        source, structural, palette, base_config
    )
    rgba_palette = _feature_to_rgba(feature_palette, source, labels)
    split_mse = _mse(source, labels, rgba_palette)
    split_done = perf_counter()

    labels, rgba_palette, merge_report = merge_error_per_byte(
        source,
        labels,
        rgba_palette,
        target_mse=float(config.final_target_mse),
        maximum_area=int(config.merge_maximum_area),
        rounds=int(config.merge_rounds),
    )
    merge_done = perf_counter()
    svg, svg_stats = compact_lattice_svg(
        labels, rgba_palette, title=title
    )
    svg_done = perf_counter()
    svgz = deterministic_svgz(svg, level=config.gzip_level)
    completed = perf_counter()
    final_mse = _mse(source, labels, rgba_palette)
    diagnostics: dict[str, float | int | str] = {
        "method": "converter_v2_rate_distortion_lattice",
        "width": int(source.shape[1]),
        "height": int(source.shape[0]),
        "crop_x": int(crop[0]),
        "crop_y": int(crop[1]),
        "alpha_mode": alpha_mode,
        "structural_colors": int(len(palette)),
        "accepted_splits": int(len(parents) - len(palette)),
        "palette_colors": int(len(rgba_palette)),
        "split_mse": split_mse,
        "final_mse": final_mse,
        "final_target_mse": float(config.final_target_mse),
        "target_met": int(final_mse <= config.final_target_mse),
        "merge_rounds": merge_report.rounds,
        "component_merges": merge_report.merges,
        "components_before": merge_report.components_before,
        "components_after": merge_report.components_after,
        "estimated_merge_bytes_saved": merge_report.estimated_bytes_saved,
        **svg_stats,
        "svgz_bytes": len(svgz),
        "svgz_ratio": len(svgz) / max(1, svg_stats["svg_bytes"]),
        "preparation_ms": 1000.0 * (prepared - started),
        "split_ms": 1000.0 * (split_done - prepared),
        "merge_ms": 1000.0 * (merge_done - split_done),
        "svg_ms": 1000.0 * (svg_done - merge_done),
        "gzip_ms": 1000.0 * (completed - svg_done),
        "total_ms": 1000.0 * (completed - started),
    }
    return V2Result(labels, rgba_palette, svg, svgz, diagnostics)


def vectorize_png_v2(
    source: str | Path,
    destination: str | Path,
    config: V2Config = V2Config(),
) -> V2Result:
    source_path = Path(source)
    with Image.open(source_path) as image:
        rgba = np.asarray(image.convert("RGBA"))
    result = vectorize_array_v2(rgba, config, title=source_path.name)
    result.save(destination)
    if Path(destination).suffix.lower() == ".svg":
        ET.parse(destination)
    return result


def diagnostics_json(result: V2Result) -> str:
    return json.dumps(result.diagnostics, indent=2, sort_keys=True) + "\n"
