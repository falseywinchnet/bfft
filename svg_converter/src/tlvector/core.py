"""Topology-first PNG to SVG conversion.

The implementation is deliberately standalone.  Its hierarchy is inspired by
the coarse-owner/full-interface/residual-child discipline of BFFT segmenting
v3, while its contour compiler uses only the general lessons from the
Portsmouth font archive: order the boundary, preserve hard events, and fit a
curve only after measuring its geometric error.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import html
import math
from time import perf_counter
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
from scipy import ndimage


@dataclass(frozen=True)
class VectorizerConfig:
    colors: int = 12
    detail_colors: int = 6
    coarse_side: int = 160
    interface_radius: int = 3
    interface_sweeps: int = 3
    smoothness: float = 0.028
    residual_quantile: float = 0.82
    residual_gain: float = 0.12
    minimum_region: int = 10
    simplify: float = 0.85
    curve_tolerance: float = 0.65
    corner_degrees: float = 52.0
    corner_window: int = 5
    subpixel_smoothing: int = 4
    seam_overlap: float = 0.65
    alpha_mode: str = "auto"
    alpha_cutoff: int = 128
    trim_transparent: bool = True
    trim_padding: int = 0
    alpha_threshold: int = 4


@dataclass
class Vectorization:
    width: int
    height: int
    labels: np.ndarray
    structural_labels: np.ndarray
    palette_rgba: np.ndarray
    parent_of: np.ndarray
    svg: str
    diagnostics: dict[str, float | int | str]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.svg, encoding="utf-8")


def _srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    value = np.asarray(rgb, dtype=np.float64)
    linear = np.where(
        value <= 0.04045,
        value / 12.92,
        ((value + 0.055) / 1.055) ** 2.4,
    )
    lms = linear @ np.array([
        [0.4122214708, 0.2119034982, 0.0883024619],
        [0.5363325363, 0.6806995451, 0.2817188376],
        [0.0514459929, 0.1073969566, 0.6299787005],
    ])
    lms = np.cbrt(np.maximum(lms, 0.0))
    return lms @ np.array([
        [0.2104542553, 1.9779984951, 0.0259040371],
        [0.7936177850, -2.4285922050, 0.7827717662],
        [-0.0040720468, 0.4505937099, -0.8086757660],
    ])


def _oklab_distance2(features: np.ndarray, palette: np.ndarray) -> np.ndarray:
    # Weighted squared Euclidean distance via the norm identity. This avoids
    # the former (..., colors, 4) broadcast, whose temporary was four times
    # larger than the actual color-cost matrix.
    weights = np.array([1.0, 1.15, 1.15, 0.7])
    shape = features.shape[:-1] + (len(palette),)
    sample = np.asarray(features, dtype=np.float64).reshape(-1, 4) * weights
    centers = np.asarray(palette, dtype=np.float64).reshape(-1, 4) * weights
    result = (
        np.sum(sample * sample, axis=1)[:, None]
        + np.sum(centers * centers, axis=1)[None, :]
        - 2.0 * (sample @ centers.T)
    )
    np.maximum(result, 0.0, out=result)
    return result.reshape(shape)


def _rgba_features(rgba: np.ndarray) -> np.ndarray:
    rgb = rgba[..., :3].astype(np.float64) / 255.0
    alpha = rgba[..., 3:4].astype(np.float64) / 255.0
    lab = _srgb_to_oklab(rgb)
    # Transparent color is undefined. Premultiplication prevents hidden RGB
    # values from manufacturing regions.
    return np.concatenate((lab * alpha, alpha), axis=-1)


def _resize_rgba(rgba: np.ndarray, side: int) -> np.ndarray:
    height, width = rgba.shape[:2]
    scale = min(1.0, float(side) / max(height, width))
    shape = (max(1, round(width * scale)), max(1, round(height * scale)))
    if shape == (width, height):
        return rgba.copy()
    return np.asarray(
        Image.fromarray(rgba, mode="RGBA").resize(shape, Image.Resampling.LANCZOS)
    )


def _seed_palette(features: np.ndarray, count: int) -> np.ndarray:
    samples = features.reshape(-1, 4)
    if len(samples) > 32768:
        stride = int(math.ceil(len(samples) / 32768))
        samples = samples[::stride]
    unique = np.unique(np.round(samples, 5), axis=0)
    count = max(1, min(int(count), len(unique)))
    center = np.mean(unique, axis=0)
    first = int(np.argmin(np.sum((unique - center) ** 2, axis=1)))
    chosen = [unique[first]]
    nearest = _oklab_distance2(unique, np.asarray(chosen))[:, 0]
    for _ in range(1, count):
        # Coverage pressure replaces random k-means initialization.
        index = int(np.argmax(nearest * (0.2 + unique[:, 3])))
        chosen.append(unique[index])
        nearest = np.minimum(
            nearest, _oklab_distance2(unique, np.asarray(chosen[-1:]))[:, 0]
        )
    palette = np.asarray(chosen)
    for _ in range(6):
        labels = np.argmin(_oklab_distance2(samples, palette), axis=1)
        updated = palette.copy()
        for index in range(len(palette)):
            members = samples[labels == index]
            if len(members):
                updated[index] = np.mean(members, axis=0)
        if np.max(np.abs(updated - palette)) < 1e-6:
            break
        palette = updated
    return palette


def _neighbor_disagreement(labels: np.ndarray, candidate: int) -> np.ndarray:
    padded = np.pad(labels, 1, mode="edge")
    neighbors = (
        padded[:-2, 1:-1], padded[2:, 1:-1],
        padded[1:-1, :-2], padded[1:-1, 2:],
    )
    return sum((neighbor != candidate).astype(np.float64) for neighbor in neighbors)


def _regularized_assign(
    features: np.ndarray,
    palette: np.ndarray,
    labels: np.ndarray,
    active: np.ndarray,
    smoothness: float,
    sweeps: int,
) -> np.ndarray:
    result = labels.copy()
    yy, xx = np.indices(result.shape)
    active_by_parity = [
        np.nonzero(active & (((yy + xx) & 1) == parity))
        for parity in (0, 1)
    ]
    chunk_size = 16384
    for _ in range(max(0, sweeps)):
        before = result.copy()
        for parity in (0, 1):
            rows, columns = active_by_parity[parity]
            if not len(rows):
                continue
            padded = np.pad(result, 1, mode="edge")
            for start in range(0, len(rows), chunk_size):
                stop = min(start + chunk_size, len(rows))
                cy = rows[start:stop]
                cx = columns[start:stop]
                costs = _oklab_distance2(features[cy, cx], palette)
                if smoothness:
                    neighbors = (
                        padded[cy, cx + 1], padded[cy + 2, cx + 1],
                        padded[cy + 1, cx], padded[cy + 1, cx + 2],
                    )
                    costs += 4.0 * smoothness
                    sample = np.arange(stop - start)
                    for neighbor in neighbors:
                        costs[sample, neighbor] -= smoothness
                result[cy, cx] = np.argmin(costs, axis=1).astype(np.int32)
        if np.array_equal(result, before):
            break
    return result


def _interface_mask(labels: np.ndarray, radius: int) -> np.ndarray:
    edge = np.zeros(labels.shape, dtype=bool)
    edge[1:] |= labels[1:] != labels[:-1]
    edge[:-1] |= labels[:-1] != labels[1:]
    edge[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    edge[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    if radius > 0:
        edge = ndimage.binary_dilation(edge, iterations=radius)
    return edge


def _lift_labels(labels: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    low_h, low_w = labels.shape
    y = np.minimum(((np.arange(height) + 0.5) * low_h / height).astype(int), low_h - 1)
    x = np.minimum(((np.arange(width) + 0.5) * low_w / width).astype(int), low_w - 1)
    return labels[y[:, None], x[None, :]].astype(np.int32)


def _remove_small_components(
    labels: np.ndarray,
    parent_labels: np.ndarray,
    minimum: int,
    *,
    first_label: int = 0,
) -> np.ndarray:
    result = labels.copy()
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    for value in np.unique(result):
        if value < first_label:
            continue
        components, count = ndimage.label(result == value, structure=structure)
        sizes = np.bincount(components.ravel())
        for component in range(1, count + 1):
            if sizes[component] < minimum:
                mask = components == component
                result[mask] = parent_labels[mask]
    return result


def _nested_residual_children(
    features: np.ndarray,
    structural: np.ndarray,
    palette: np.ndarray,
    config: VectorizerConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = structural.copy()
    palettes = [row.copy() for row in palette]
    parents = list(range(len(palette)))
    if config.detail_colors <= 0:
        return labels, np.asarray(palettes), np.asarray(parents, dtype=np.int32)
    delta = (features - palette[structural]) * np.array([1.0, 1.15, 1.15, 0.7])
    base_error = np.sum(delta * delta, axis=-1)
    flat_features = features.reshape(-1, features.shape[-1])
    flat_structural = structural.ravel()
    flat_labels = labels.ravel()
    flat_error = base_error.ravel()
    order = np.argsort(flat_structural, kind="stable")
    population = np.bincount(flat_structural, minlength=len(palette))
    offsets = np.concatenate(([0], np.cumsum(population)))
    for _ in range(max(0, config.detail_colors)):
        best: tuple[float, int, np.ndarray, np.ndarray] | None = None
        for parent in range(len(palette)):
            # Children are siblings, never replacement layers over one another.
            # Limiting proposals to still-unexplained parent pixels keeps every
            # accepted child represented in the final quotient.
            parent_indices = order[offsets[parent]:offsets[parent + 1]]
            available = parent_indices[flat_labels[parent_indices] == parent]
            if len(available) < 2 * config.minimum_region:
                continue
            values = flat_error[available]
            threshold = float(np.quantile(values, config.residual_quantile))
            hot = available[values >= threshold]
            if len(hot) < config.minimum_region:
                continue
            candidate = np.mean(flat_features[hot], axis=0)
            old = _oklab_distance2(
                flat_features[hot], palette[parent:parent + 1]
            )[:, 0]
            new = _oklab_distance2(flat_features[hot], candidate[None, :])[:, 0]
            gain = float(np.sum(old - new))
            if best is None or gain > best[0]:
                best = (gain, parent, candidate, hot)
        if best is None or best[0] <= 0.0:
            break
        _gain, parent, candidate, hot = best
        child = len(palettes)
        old_cost = _oklab_distance2(
            flat_features[hot], np.asarray(palettes[parent:parent + 1])
        )[:, 0]
        new_cost = _oklab_distance2(flat_features[hot], candidate[None, :])[:, 0]
        accepted = hot[new_cost <= old_cost * (1.0 - config.residual_gain)]
        improve = np.zeros(labels.size, dtype=bool)
        improve[accepted] = True
        improve = improve.reshape(labels.shape)
        components, count = ndimage.label(improve)
        keep = np.zeros_like(improve)
        sizes = np.bincount(components.ravel())
        for component in range(1, count + 1):
            if sizes[component] >= config.minimum_region:
                keep |= components == component
        if not np.any(keep):
            flat_error[hot] = 0.0
            continue
        labels[keep] = child
        palettes.append(candidate)
        parents.append(parent)
        base_error[keep] = 0.0
    labels = _remove_small_components(
        labels, structural, config.minimum_region, first_label=len(palette)
    )
    return labels, np.asarray(palettes), np.asarray(parents, dtype=np.int32)


def _feature_to_rgba(feature: np.ndarray, source: np.ndarray, labels: np.ndarray) -> np.ndarray:
    # Means in source sRGB avoid a lossy inverse Oklab conversion.
    count = int(np.max(labels)) + 1
    palette = np.zeros((count, 4), dtype=np.uint8)
    flat_labels = labels.ravel()
    population = np.bincount(flat_labels, minlength=count)
    present = population > 0
    flat_source = source.reshape(-1, 4)
    for channel in range(4):
        total = np.bincount(
            flat_labels, weights=flat_source[:, channel], minlength=count
        )
        palette[present, channel] = np.clip(
            np.round(total[present] / population[present]), 0, 255
        ).astype(np.uint8)
    return palette


def _boundary_loops(mask: np.ndarray) -> list[np.ndarray]:
    """Trace exact pixel-union boundaries on the integer lattice."""
    height, width = mask.shape
    edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    padded = np.pad(mask, 1, mode="constant")
    neighbors = (
        mask & ~padded[:-2, 1:-1],
        mask & ~padded[1:-1, 2:],
        mask & ~padded[2:, 1:-1],
        mask & ~padded[1:-1, :-2],
    )
    for side, boundary in enumerate(neighbors):
        rows, columns = np.nonzero(boundary)
        for y, x in zip(rows, columns):
            x = int(x)
            y = int(y)
            if side == 0:
                edges.add(((x, y), (x + 1, y)))
            elif side == 1:
                edges.add(((x + 1, y), (x + 1, y + 1)))
            elif side == 2:
                edges.add(((x + 1, y + 1), (x, y + 1)))
            else:
                edges.add(((x, y + 1), (x, y)))
    outgoing: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for start, end in edges:
        outgoing.setdefault(start, []).append(end)
    loops: list[np.ndarray] = []
    while edges:
        first = min(edges)
        start, current = first
        previous = start
        points = [start]
        edges.remove(first)
        guard = 0
        while current != start:
            points.append(current)
            choices = [end for end in outgoing.get(current, []) if (current, end) in edges]
            if not choices:
                break
            incoming = (current[0] - previous[0], current[1] - previous[1])
            # Interior lies on the right; prefer right, straight, left, reverse.
            rank = {
                (-incoming[1], incoming[0]): 0,
                incoming: 1,
                (incoming[1], -incoming[0]): 2,
                (-incoming[0], -incoming[1]): 3,
            }
            end = min(choices, key=lambda p: rank.get((p[0] - current[0], p[1] - current[1]), 4))
            edges.remove((current, end))
            previous, current = current, end
            guard += 1
            if guard > 4 * height * width + 4:
                raise RuntimeError("boundary trace failed to close")
        if len(points) >= 4 and current == start:
            loops.append(np.asarray(points, dtype=np.float64))
    return loops


def _point_line_distance(points: np.ndarray, first: np.ndarray, last: np.ndarray) -> np.ndarray:
    delta = last - first
    length2 = float(np.dot(delta, delta))
    if length2 <= 1e-12:
        return np.linalg.norm(points - first, axis=1)
    t = np.clip(((points - first) @ delta) / length2, 0.0, 1.0)
    return np.linalg.norm(points - (first + t[:, None] * delta), axis=1)


def _rdp_open(points: np.ndarray, tolerance: float) -> np.ndarray:
    if len(points) <= 2:
        return points
    distance = _point_line_distance(points[1:-1], points[0], points[-1])
    index = int(np.argmax(distance)) + 1
    if distance[index - 1] <= tolerance:
        return points[[0, -1]]
    return np.vstack((_rdp_open(points[:index + 1], tolerance)[:-1], _rdp_open(points[index:], tolerance)))


def _simplify_closed(points: np.ndarray, tolerance: float) -> np.ndarray:
    # Remove grid-collinear vertices first.
    previous = points - np.roll(points, 1, axis=0)
    following = np.roll(points, -1, axis=0) - points
    cross = previous[:, 0] * following[:, 1] - previous[:, 1] * following[:, 0]
    points = points[np.abs(cross) > 1e-12]
    if len(points) <= 4:
        return points
    pivot = int(np.argmax(np.sum((points - points[0]) ** 2, axis=1)))
    first = _rdp_open(points[:pivot + 1], tolerance)
    second = _rdp_open(np.vstack((points[pivot:], points[:1])), tolerance)
    return np.vstack((first[:-1], second[:-1]))


def _turn_degrees(points: np.ndarray, reach: int = 1) -> np.ndarray:
    reach = max(1, min(int(reach), max(1, (len(points) - 1) // 2)))
    incoming = points - np.roll(points, reach, axis=0)
    outgoing = np.roll(points, -reach, axis=0) - points
    ni = np.maximum(np.linalg.norm(incoming, axis=1), 1e-9)
    no = np.maximum(np.linalg.norm(outgoing, axis=1), 1e-9)
    cosine = np.clip(np.sum(incoming * outgoing, axis=1) / (ni * no), -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def _relax_subpixel(points: np.ndarray, config: VectorizerConfig) -> np.ndarray:
    """Remove pixel-staircase energy without moving persistent corners.

    Corner evidence is measured across several unit boundary steps. A raster
    circle therefore looks locally smooth, while the turn at a rectangle
    remains a hard event. The relaxation is a conservative periodic Laplacian
    step and never changes contour order or topology.
    """
    result = np.asarray(points, dtype=np.float64).copy()
    if len(result) < 8 or config.subpixel_smoothing <= 0:
        return result
    hard = _turn_degrees(result, config.corner_window) >= config.corner_degrees
    # Pin a short neighborhood so smoothing cannot shave a detected corner.
    hard |= np.roll(hard, 1) | np.roll(hard, -1)
    for _ in range(config.subpixel_smoothing):
        midpoint = 0.5 * (np.roll(result, 1, axis=0) + np.roll(result, -1, axis=0))
        updated = result + 0.45 * (midpoint - result)
        result[~hard] = updated[~hard]
    return result


def _format_number(value: float) -> str:
    rounded = round(float(value), 3)
    if abs(rounded - round(rounded)) < 1e-9:
        return str(int(round(rounded)))
    return f"{rounded:.3f}".rstrip("0").rstrip(".")


def _loop_path(points: np.ndarray, config: VectorizerConfig) -> str:
    points = _relax_subpixel(points, config)
    points = _simplify_closed(points, config.simplify)
    if len(points) < 3:
        return ""
    turns = _turn_degrees(points)
    smooth = turns < config.corner_degrees
    previous = np.roll(points, 1, axis=0)
    following = np.roll(points, -1, axis=0)
    bend = np.linalg.norm(previous + following - 2.0 * points, axis=1)
    # Use the largest symmetric trim whose quadratic displacement stays inside
    # the requested error budget. Unlike the old accept/reject test, a smooth
    # event now receives a smaller curve when the local bend is strong.
    trim = np.minimum(0.5, 4.0 * config.curve_tolerance / np.maximum(bend, 1e-9))
    trim = np.where(smooth, trim, 0.0)
    entry = points + trim[:, None] * (previous - points)
    exit_points = points + trim[:, None] * (following - points)
    start = exit_points[0]
    commands = [f"M{_format_number(start[0])} {_format_number(start[1])}"]
    for offset in range(1, len(points) + 1):
        index = offset % len(points)
        vertex = points[index]
        commands.append(
            f"L{_format_number(entry[index, 0])} {_format_number(entry[index, 1])}"
        )
        if smooth[index]:
            commands.append(
                f"Q{_format_number(vertex[0])} {_format_number(vertex[1])} "
                f"{_format_number(exit_points[index, 0])} {_format_number(exit_points[index, 1])}"
            )
        else:
            commands.append(f"L{_format_number(vertex[0])} {_format_number(vertex[1])}")
    commands.append("Z")
    return "".join(commands)


def _trim_transparent(
    source: np.ndarray, config: VectorizerConfig
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    height, width = source.shape[:2]
    if not config.trim_transparent:
        return source, (0, 0, width, height)
    visible = source[..., 3] > int(config.alpha_threshold)
    if not np.any(visible):
        return source, (0, 0, width, height)
    rows, columns = np.nonzero(visible)
    padding = max(0, int(config.trim_padding))
    x0 = max(0, int(columns.min()) - padding)
    y0 = max(0, int(rows.min()) - padding)
    x1 = min(width, int(columns.max()) + 1 + padding)
    y1 = min(height, int(rows.max()) + 1 + padding)
    return np.ascontiguousarray(source[y0:y1, x0:x1]), (x0, y0, x1, y1)


def _normalize_alpha(
    source: np.ndarray, config: VectorizerConfig
) -> tuple[np.ndarray, str]:
    """Separate cutout coverage from genuine region translucency.

    Raster antialiasing is not an image region. In cutout mode its coverage is
    thresholded once and the SVG renderer recreates a smooth edge. Auto mode
    chooses that route when partially transparent pixels hug the support
    boundary; broad translucent interiors remain preserved.
    """
    mode = str(config.alpha_mode).lower()
    if mode not in {"auto", "preserve", "cutout"}:
        raise ValueError("alpha_mode must be 'auto', 'preserve', or 'cutout'")
    alpha = source[..., 3]
    partial = (alpha > 0) & (alpha < 255)
    if mode == "auto":
        if not np.any(partial):
            mode = "cutout"
        else:
            depth = ndimage.distance_transform_edt(alpha > 0)
            deep_fraction = float(np.mean(depth[partial] > 3.0))
            mode = "cutout" if deep_fraction < 0.15 else "preserve"
    if mode == "preserve":
        return source, mode
    result = source.copy()
    result[..., 3] = np.where(alpha >= int(config.alpha_cutoff), 255, 0).astype(np.uint8)
    return result, mode


def _svg_document(
    labels: np.ndarray,
    palette: np.ndarray,
    config: VectorizerConfig,
    title: str,
) -> tuple[str, int, int]:
    height, width = labels.shape
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f"<title>{html.escape(title)}</title>",
        "<desc>Transport-locked residual contour vectorization</desc>",
    ]
    path_count = loop_count = 0
    objects = ndimage.find_objects(labels + 1, max_label=len(palette))
    for index in range(len(palette)):
        region = objects[index] if index < len(objects) else None
        if region is None:
            continue
        color = palette[index]
        if int(color[3]) <= int(config.alpha_threshold):
            continue
        y_slice, x_slice = region
        mask = labels[region] == index
        offset = np.array([x_slice.start, y_slice.start], dtype=np.float64)
        paths = [
            _loop_path(loop + offset, config) for loop in _boundary_loops(mask)
        ]
        paths = [path for path in paths if path]
        if not paths:
            continue
        opacity = color[3] / 255.0
        hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
        attributes = f'fill="{hex_color}" fill-rule="evenodd"'
        if config.seam_overlap > 0.0:
            # Adjacent independently fitted paths can expose subpixel slivers
            # under antialiasing. A small same-color under-stroke makes the
            # cover closed without materially moving the visible interface.
            attributes += (
                f' stroke="{hex_color}" stroke-width="{_format_number(config.seam_overlap)}"'
                ' stroke-linejoin="round" stroke-linecap="round" paint-order="stroke fill"'
            )
        if opacity < 0.999:
            attributes += f' fill-opacity="{_format_number(opacity)}"'
            if config.seam_overlap > 0.0:
                attributes += f' stroke-opacity="{_format_number(opacity)}"'
        parts.append(f'<path {attributes} d="{"".join(paths)}"/>')
        path_count += 1
        loop_count += len(paths)
    parts.append("</svg>")
    return "\n".join(parts) + "\n", path_count, loop_count


def vectorize_array(
    rgba: np.ndarray,
    config: VectorizerConfig = VectorizerConfig(),
    *,
    title: str = "vectorized image",
) -> Vectorization:
    started = perf_counter()
    source = np.asarray(rgba, dtype=np.uint8)
    if source.ndim != 3 or source.shape[2] != 4:
        raise ValueError("expected an H x W x 4 uint8 RGBA array")
    original_height, original_width = source.shape[:2]
    source, alpha_mode = _normalize_alpha(source, config)
    source, crop = _trim_transparent(source, config)
    preprocessing_done = perf_counter()
    coarse_rgba = _resize_rgba(source, config.coarse_side)
    coarse_features = _rgba_features(coarse_rgba)
    palette = _seed_palette(coarse_features, config.colors)
    coarse_labels = np.argmin(_oklab_distance2(coarse_features, palette), axis=-1).astype(np.int32)
    palette_done = perf_counter()
    coarse_labels = _regularized_assign(
        coarse_features, palette, coarse_labels, np.ones(coarse_labels.shape, dtype=bool),
        config.smoothness, 3,
    )
    coarse_done = perf_counter()
    features = _rgba_features(source)
    structural = _lift_labels(coarse_labels, source.shape[:2])
    active = _interface_mask(structural, config.interface_radius)
    structural = _regularized_assign(
        features, palette, structural, active, config.smoothness,
        config.interface_sweeps,
    )
    interface_done = perf_counter()
    labels, feature_palette, parents = _nested_residual_children(
        features, structural, palette, config
    )
    detail_done = perf_counter()
    rgba_palette = _feature_to_rgba(feature_palette, source, labels)
    color_done = perf_counter()
    svg, path_count, loop_count = _svg_document(labels, rgba_palette, config, title)
    svg_done = perf_counter()
    approximation = rgba_palette[labels].astype(np.float64)
    mse = float(np.mean((source.astype(np.float64) - approximation) ** 2))
    diagnostics: dict[str, float | int | str] = {
        "width": int(source.shape[1]),
        "height": int(source.shape[0]),
        "original_width": int(original_width),
        "original_height": int(original_height),
        "crop_x": int(crop[0]),
        "crop_y": int(crop[1]),
        "alpha_mode": alpha_mode,
        "structural_colors": int(len(palette)),
        "detail_colors": int(len(parents) - len(palette)),
        "paths": int(path_count),
        "loops": int(loop_count),
        "interface_pixels": int(np.count_nonzero(active)),
        "rgba_mse": mse,
        "svg_bytes": int(len(svg.encode("utf-8"))),
        "preprocessing_ms": 1000.0 * (preprocessing_done - started),
        "palette_ms": 1000.0 * (palette_done - preprocessing_done),
        "coarse_regularization_ms": 1000.0 * (coarse_done - palette_done),
        "interface_refinement_ms": 1000.0 * (interface_done - coarse_done),
        "detail_refinement_ms": 1000.0 * (detail_done - interface_done),
        "color_reduction_ms": 1000.0 * (color_done - detail_done),
        "svg_compilation_ms": 1000.0 * (svg_done - color_done),
        "total_ms": 1000.0 * (svg_done - started),
    }
    return Vectorization(
        width=source.shape[1], height=source.shape[0], labels=labels,
        structural_labels=structural, palette_rgba=rgba_palette,
        parent_of=parents, svg=svg, diagnostics=diagnostics,
    )


def vectorize_png(
    source: str | Path,
    destination: str | Path,
    config: VectorizerConfig = VectorizerConfig(),
) -> Vectorization:
    path = Path(source)
    with Image.open(path) as image:
        rgba = np.asarray(image.convert("RGBA"))
    result = vectorize_array(rgba, config, title=path.name)
    result.save(destination)
    # Parse what we wrote: malformed SVG must fail at the conversion boundary.
    ET.parse(destination)
    return result
