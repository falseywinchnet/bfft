"""Prototype affine-gradient SVG painting on segmenting-v3 texture cells."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import html
import os
from time import perf_counter

import numpy as np
from PIL import Image
from scipy import ndimage

from tlvector.core import _boundary_loops

from .lattice import _number, compact_lattice_loop, deterministic_svgz


@dataclass
class AffineResult:
    labels: np.ndarray
    reconstruction_rgb: np.ndarray
    svg: str
    svgz: bytes
    diagnostics: dict[str, float | int | str]

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        if destination.suffix.lower() == ".svgz":
            destination.write_bytes(self.svgz)
        else:
            destination.write_text(self.svg, encoding="utf-8")


def _dense_labels(labels: np.ndarray) -> np.ndarray:
    _unique, inverse = np.unique(labels, return_inverse=True)
    return inverse.reshape(labels.shape).astype(np.int32)


def fit_rank1_affine_cells(
    source_rgb: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float | int]]:
    """Fit the best single spatial color-gradient direction in every cell."""
    source = np.asarray(source_rgb, dtype=np.float64)
    dense = _dense_labels(labels)
    height, width = dense.shape
    flat_labels = dense.ravel()
    count = int(flat_labels.max()) + 1
    yy, xx = np.indices((height, width), dtype=np.float64)
    x = xx.ravel()
    y = yy.ravel()
    pixels = source.reshape(-1, 3)
    population = np.bincount(flat_labels, minlength=count).astype(np.float64)
    safe_population = np.maximum(population, 1.0)
    sum_x = np.bincount(flat_labels, weights=x, minlength=count)
    sum_y = np.bincount(flat_labels, weights=y, minlength=count)
    center_x = sum_x / safe_population
    center_y = sum_y / safe_population
    dx = x - center_x[flat_labels]
    dy = y - center_y[flat_labels]
    sxx = np.bincount(flat_labels, weights=dx * dx, minlength=count)
    sxy = np.bincount(flat_labels, weights=dx * dy, minlength=count)
    syy = np.bincount(flat_labels, weights=dy * dy, minlength=count)
    mean = np.stack([
        np.bincount(flat_labels, weights=pixels[:, channel], minlength=count)
        / safe_population
        for channel in range(3)
    ], axis=1)
    sxc = np.stack([
        np.bincount(
            flat_labels, weights=dx * pixels[:, channel], minlength=count
        )
        for channel in range(3)
    ], axis=1)
    syc = np.stack([
        np.bincount(
            flat_labels, weights=dy * pixels[:, channel], minlength=count
        )
        for channel in range(3)
    ], axis=1)
    determinant = sxx * syy - sxy * sxy
    scale = np.maximum(sxx + syy, 1.0)
    valid = determinant > 1e-10 * scale * scale
    gradient = np.zeros((count, 2, 3), dtype=np.float64)
    gradient[valid, 0] = (
        syy[valid, None] * sxc[valid]
        - sxy[valid, None] * syc[valid]
    ) / determinant[valid, None]
    gradient[valid, 1] = (
        sxx[valid, None] * syc[valid]
        - sxy[valid, None] * sxc[valid]
    ) / determinant[valid, None]

    spatial, singular, color_axis = np.linalg.svd(
        gradient, full_matrices=False
    )
    direction = spatial[:, :, 0]
    slope = singular[:, 0, None] * color_axis[:, 0, :]
    projection = (
        direction[flat_labels, 0] * dx
        + direction[flat_labels, 1] * dy
    )
    reconstruction = np.clip(
        mean[flat_labels] + slope[flat_labels] * projection[:, None],
        0.0,
        1.0,
    ).reshape(height, width, 3)
    flat_reconstruction = mean[flat_labels].reshape(height, width, 3)
    affine_mse = float(np.mean((source - reconstruction) ** 2) * 65025.0)
    flat_mse = float(np.mean((source - flat_reconstruction) ** 2) * 65025.0)

    minimum = np.full(count, np.inf, dtype=np.float64)
    maximum = np.full(count, -np.inf, dtype=np.float64)
    np.minimum.at(minimum, flat_labels, projection)
    np.maximum.at(maximum, flat_labels, projection)
    model = {
        "mean": mean,
        "center_x": center_x,
        "center_y": center_y,
        "direction": direction,
        "slope": slope,
        "minimum": minimum,
        "maximum": maximum,
        "population": population,
    }
    return reconstruction, model, {
        "cells": count,
        "flat_rgb_mse_255": flat_mse,
        "affine_rgb_mse_255": affine_mse,
        "flat_rgba_mse_255_equivalent": 0.75 * flat_mse,
        "affine_rgba_mse_255_equivalent": 0.75 * affine_mse,
    }


def affine_gradient_svg(
    source_rgb: np.ndarray,
    labels: np.ndarray,
    *,
    title: str,
) -> tuple[str, np.ndarray, dict[str, float | int]]:
    dense = _dense_labels(labels)
    reconstruction, model, fit = fit_rank1_affine_cells(source_rgb, dense)
    height, width = dense.shape
    count = int(fit["cells"])
    objects = ndimage.find_objects(dense + 1, max_label=count)
    order = np.argsort(-model["population"], kind="stable")
    definitions: list[str] = []
    paths: list[str] = []
    loops = 0
    gradient_count = 0
    for cell in order:
        region = objects[cell] if cell < len(objects) else None
        if region is None:
            continue
        y_slice, x_slice = region
        offset = np.array([x_slice.start, y_slice.start], dtype=np.float64)
        cell_paths = [
            compact_lattice_loop(loop + offset)
            for loop in _boundary_loops(dense[region] == cell)
        ]
        cell_paths = [path for path in cell_paths if path]
        if not cell_paths:
            continue
        loops += len(cell_paths)
        low = float(model["minimum"][cell])
        high = float(model["maximum"][cell])
        mean = model["mean"][cell]
        slope = model["slope"][cell]
        direction = model["direction"][cell]
        first_color = np.clip(mean + slope * low, 0.0, 1.0)
        second_color = np.clip(mean + slope * high, 0.0, 1.0)
        delta = float(np.max(np.abs(first_color - second_color)))
        if high - low > 1e-6 and delta >= 1.0 / 255.0:
            x1 = model["center_x"][cell] + direction[0] * low
            y1 = model["center_y"][cell] + direction[1] * low
            x2 = model["center_x"][cell] + direction[0] * high
            y2 = model["center_y"][cell] + direction[1] * high
            c1 = np.clip(np.round(first_color * 255), 0, 255).astype(np.uint8)
            c2 = np.clip(np.round(second_color * 255), 0, 255).astype(np.uint8)
            definitions.append(
                f'<linearGradient id="g{cell}" gradientUnits="userSpaceOnUse" '
                f'x1="{_number(x1)}" y1="{_number(y1)}" '
                f'x2="{_number(x2)}" y2="{_number(y2)}">'
                f'<stop stop-color="#{c1[0]:02x}{c1[1]:02x}{c1[2]:02x}"/>'
                f'<stop offset="1" stop-color="#{c2[0]:02x}{c2[1]:02x}{c2[2]:02x}"/>'
                "</linearGradient>"
            )
            fill = f"url(#g{cell})"
            gradient_count += 1
        else:
            color = np.clip(np.round(mean * 255), 0, 255).astype(np.uint8)
            fill = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
        paths.append(
            f'<path fill="{fill}" fill-rule="evenodd" '
            f'd="{"".join(cell_paths)}"/>'
        )
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" '
            'shape-rendering="crispEdges">'
        ),
        f"<title>{html.escape(title)}</title>",
        "<desc>Converter v2 rank-one affine gradients on segmenting-v3 cells</desc>",
        "<defs>",
        *definitions,
        "</defs>",
        *paths,
        "</svg>",
    ]
    svg = "\n".join(parts) + "\n"
    return svg, reconstruction, {
        **fit,
        "paths": len(paths),
        "loops": loops,
        "gradients": gradient_count,
        "svg_bytes": len(svg.encode("utf-8")),
    }


def build_v3_affine_svg(
    source: str | Path,
    *,
    bfft_library: str | None = None,
    threads: int = 4,
    gzip_level: int = 9,
) -> AffineResult:
    if bfft_library:
        os.environ["BFFT_LIBRARY"] = bfft_library
    # The optional experiment import stays out of the ordinary v2 path.
    from experiments.segmenting_v3 import SegmentingV3Config, build_segmenting_v3

    source_path = Path(source)
    with Image.open(source_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    started = perf_counter()
    result = build_segmenting_v3(
        rgb,
        SegmentingV3Config(
            structural_topology="canonical_v2",
            meyer_operator="jump_measure",
            threads=int(threads),
        ),
    )
    segmented = perf_counter()
    svg, reconstruction, diagnostics = affine_gradient_svg(
        rgb, result["texture_labels"], title=source_path.name
    )
    compiled = perf_counter()
    svgz = deterministic_svgz(svg, level=gzip_level)
    diagnostics.update({
        "method": "converter_v2_v3_rank1_affine",
        "v3_structural_cells": int(len(result["centers"])),
        "v3_texture_cells": int(len(result["texture_centers"])),
        "v3_modeled_rgb_mse_255": float(
            result["record"]["rgb_mse"] * 65025.0
        ),
        "v3_modeled_psnr_db": float(result["record"]["psnr"]),
        "svgz_bytes": len(svgz),
        "segmentation_ms": 1000.0 * (segmented - started),
        "affine_svg_ms": 1000.0 * (compiled - segmented),
        "total_ms": 1000.0 * (compiled - started),
    })
    return AffineResult(
        _dense_labels(result["texture_labels"]),
        reconstruction,
        svg,
        svgz,
        diagnostics,
    )
