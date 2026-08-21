"""Probe spatial palette mixing while retaining an exact global color budget.

This is deliberately an experiment rather than part of the public engine.  It
compares flat nearest-palette assignment with region-constrained two-color
halftoning using exactly the same display palette.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from posterizer.core import (
    PosterizerConfig,
    _rgba_to_lab_alpha,
    _utility_diagnostics,
    posterize_array,
)


def _rank_field(shape: tuple[int, int], seed: int = 17) -> np.ndarray:
    """Return a deterministic, approximately blue-noise threshold field."""
    rng = np.random.default_rng(seed)
    white = rng.random(shape)
    low = ndimage.gaussian_filter(white, sigma=1.15, mode="wrap")
    high = white - low
    order = np.argsort(high, axis=None, kind="stable")
    rank = np.empty(high.size, dtype=np.float64)
    rank[order] = (np.arange(high.size) + 0.5) / high.size
    return rank.reshape(shape)


def _bayer_field(shape: tuple[int, int], side: int = 8) -> np.ndarray:
    """Return a compact ordered screen with uniform counts in every tile."""
    matrix = np.array([[0]], dtype=np.int32)
    while len(matrix) < side:
        matrix = np.block([
            [4 * matrix, 4 * matrix + 2],
            [4 * matrix + 3, 4 * matrix + 1],
        ])
    matrix = (matrix.astype(np.float64) + 0.5) / matrix.size
    repeats = (
        int(np.ceil(shape[0] / side)), int(np.ceil(shape[1] / side))
    )
    return np.tile(matrix, repeats)[: shape[0], : shape[1]]


def _error_diffuse(probability: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Diffuse density without carrying residuals across palette-pair regions."""
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
            if y + 1 < height:
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


def _smoothstep(low: float, high: float, value: np.ndarray) -> np.ndarray:
    scaled = np.clip((value - low) / max(high - low, 1e-12), 0.0, 1.0)
    return scaled * scaled * (3.0 - 2.0 * scaled)


def _mix_plan(
    source_lab: np.ndarray,
    labels: np.ndarray,
    palette_lab: np.ndarray,
    neighbors: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find the best palette segment starting at each flat-assigned node."""
    height, width = labels.shape
    flat_source = source_lab[..., :3].reshape(-1, 3)
    flat_labels = labels.ravel()
    partner = flat_labels.copy()
    fraction = np.zeros(len(flat_labels), dtype=np.float64)
    improvement = np.zeros(len(flat_labels), dtype=np.float64)

    for base in range(len(palette_lab)):
        active = np.flatnonzero(flat_labels == base)
        if not len(active):
            continue
        target = flat_source[active]
        origin = palette_lab[base, :3]
        base_error = np.sum((target - origin) ** 2, axis=1)
        best_error = base_error.copy()
        best_fraction = np.zeros(len(active), dtype=np.float64)
        best_partner = np.full(len(active), base, dtype=np.int32)
        palette_distance = np.sum(
            (palette_lab[:, :3] - origin) ** 2, axis=1
        )
        candidates = np.argsort(palette_distance, kind="stable")[1:]
        if neighbors is not None:
            candidates = candidates[: max(1, int(neighbors))]
        for other in candidates:
            if other == base:
                continue
            direction = palette_lab[other, :3] - origin
            denominator = float(np.dot(direction, direction))
            if denominator <= 1e-14:
                continue
            projection = np.clip(
                np.einsum("ij,j->i", target - origin, direction) / denominator,
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


def _mix_gate(
    source_lab: np.ndarray,
    labels: np.ndarray,
    improvement: np.ndarray,
    *,
    smooth_only: bool,
) -> np.ndarray:
    """Protect contours and admit mixing only where it buys real tone."""
    stable = (
        ndimage.minimum_filter(labels, size=3, mode="nearest")
        == ndimage.maximum_filter(labels, size=3, mode="nearest")
    )
    lab = source_lab[..., :3]
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
    gate = stable.astype(np.float64) * worthwhile
    if smooth_only:
        gate *= smoothness
    return gate


def spatial_mix(
    source: np.ndarray,
    flat: np.ndarray,
    labels: np.ndarray,
    palette: np.ndarray,
    *,
    strength: float,
    smooth_only: bool,
    carrier: str = "blue",
    neighbors: int | None = 2,
    seed: int = 17,
) -> tuple[np.ndarray, dict[str, float]]:
    source_lab = _rgba_to_lab_alpha(source)
    palette_lab = _rgba_to_lab_alpha(palette)
    partner, fraction, improvement = _mix_plan(
        source_lab, labels, palette_lab, neighbors
    )
    gate = _mix_gate(
        source_lab, labels, improvement, smooth_only=smooth_only
    )
    probability = np.clip(float(strength) * fraction * gate, 0.0, 1.0)
    if carrier == "blue":
        choose_partner = _rank_field(labels.shape, seed=seed) < probability
    elif carrier == "bayer":
        choose_partner = _bayer_field(labels.shape) < probability
    elif carrier == "diffusion":
        groups = labels.astype(np.int64) * len(palette) + partner
        groups[probability <= 0.0] = -1
        choose_partner = _error_diffuse(probability, groups)
    else:
        raise ValueError("carrier must be 'blue', 'bayer', or 'diffusion'")
    mixed_labels = labels.copy()
    mixed_labels[choose_partner] = partner[choose_partner]
    output = palette[mixed_labels]

    output_lab = _rgba_to_lab_alpha(output)
    source_low = ndimage.gaussian_filter(
        source_lab[..., :3], sigma=(1.5, 1.5, 0.0), mode="reflect"
    )
    output_low = ndimage.gaussian_filter(
        output_lab[..., :3], sigma=(1.5, 1.5, 0.0), mode="reflect"
    )
    rgba_delta = source.astype(np.float64) - output.astype(np.float64)
    diagnostics = {
        "mixed_fraction": float(np.mean(choose_partner)),
        "rgba_mse_255": float(np.mean(rgba_delta * rgba_delta)),
        "lowpass_oklab_rmse": float(np.sqrt(np.mean((source_low - output_low) ** 2))),
        **_utility_diagnostics(
            source_lab, output_lab, source[..., 3] > 4
        ),
    }
    return output, diagnostics


def _montage(
    panels: list[tuple[str, np.ndarray, dict[str, float] | None]],
    destination: Path,
) -> None:
    columns = 3
    image_height, image_width = panels[0][1].shape[:2]
    label_height = 42
    rows = (len(panels) + columns - 1) // columns
    canvas = Image.new(
        "RGB", (columns * image_width, rows * (image_height + label_height)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for index, (name, array, diagnostics) in enumerate(panels):
        x = (index % columns) * image_width
        y = (index // columns) * (image_height + label_height)
        caption = name
        if diagnostics is not None:
            caption += (
                f" | mix {100 * diagnostics['mixed_fraction']:.1f}%"
                f" low {diagnostics['lowpass_oklab_rmse']:.4f}"
                f" tex {diagnostics['texture_correlation']:.3f}"
            )
        draw.text((x + 4, y + 3), caption, fill="black")
        canvas.paste(Image.fromarray(array, "RGBA").convert("RGB"), (x, y + label_height))
    canvas.save(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    source = np.asarray(Image.open(args.source).convert("RGBA"))
    config = replace(
        PosterizerConfig(), colors=8, cleanup_rounds=0, minimum_island=0
    )
    result = posterize_array(source, config)
    panels: list[tuple[str, np.ndarray, dict[str, float] | None]] = [
        ("original", source, None),
        ("flat exact-8", result.posterized_rgba, None),
    ]
    for carrier, neighbors, strength in product(
        ("bayer", "diffusion"), (1, 2, 3, None), (0.5, 1.0)
    ):
        output, diagnostics = spatial_mix(
            source,
            result.posterized_rgba,
            result.labels,
            result.palette_rgba,
            strength=strength,
            smooth_only=True,
            carrier=carrier,
            neighbors=neighbors,
        )
        neighbor_name = "all" if neighbors is None else str(neighbors)
        panels.append((
            f"{carrier} smooth n={neighbor_name} x{strength:g}",
            output,
            diagnostics,
        ))
        print(carrier, neighbor_name, strength, diagnostics)
    _montage(panels, args.destination)


if __name__ == "__main__":
    main()
