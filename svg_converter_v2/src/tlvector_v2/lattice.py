"""Compact exact-lattice SVG serialization and deterministic SVGZ output."""

from __future__ import annotations

from io import BytesIO
import gzip
import html

import numpy as np
from scipy import ndimage

from tlvector.core import _boundary_loops


def _number(value: float) -> str:
    rounded = round(float(value), 3)
    if abs(rounded - round(rounded)) < 1e-9:
        return str(int(round(rounded)))
    return f"{rounded:.3f}".rstrip("0").rstrip(".")


def _collinear_vertices(points: np.ndarray) -> np.ndarray:
    value = np.asarray(points, dtype=np.float64)
    if len(value) <= 4:
        return value
    incoming = value - np.roll(value, 1, axis=0)
    outgoing = np.roll(value, -1, axis=0) - value
    cross = incoming[:, 0] * outgoing[:, 1] - incoming[:, 1] * outgoing[:, 0]
    retained = value[np.abs(cross) > 1e-12]
    return retained if len(retained) >= 4 else value


def compact_lattice_loop(points: np.ndarray) -> str:
    """Encode an exact axis-aligned loop with the shortest H/V command form."""
    value = _collinear_vertices(points)
    if len(value) < 4:
        return ""
    x, y = value[0]
    commands = [f"M{_number(x)} {_number(y)}"]
    for next_x, next_y in value[1:]:
        if abs(next_y - y) < 1e-9:
            absolute = f"H{_number(next_x)}"
            relative = f"h{_number(next_x - x)}"
        elif abs(next_x - x) < 1e-9:
            absolute = f"V{_number(next_y)}"
            relative = f"v{_number(next_y - y)}"
        else:
            absolute = f"L{_number(next_x)} {_number(next_y)}"
            relative = f"l{_number(next_x - x)} {_number(next_y - y)}"
        commands.append(relative if len(relative) < len(absolute) else absolute)
        x, y = next_x, next_y
    commands.append("Z")
    return "".join(commands)


def deterministic_svgz(svg: str, *, level: int = 9) -> bytes:
    output = BytesIO()
    with gzip.GzipFile(
        filename="", mode="wb", fileobj=output,
        compresslevel=int(level), mtime=0,
    ) as stream:
        stream.write(svg.encode("utf-8"))
    return output.getvalue()


def compact_lattice_svg(
    labels: np.ndarray,
    palette: np.ndarray,
    *,
    title: str,
) -> tuple[str, dict[str, int]]:
    """Compile one exact even-odd path per present color, largest colors first."""
    height, width = labels.shape
    population = np.bincount(labels.ravel(), minlength=len(palette))
    order = np.argsort(-population, kind="stable")
    objects = ndimage.find_objects(labels + 1, max_label=len(palette))
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" '
            'shape-rendering="crispEdges">'
        ),
        f"<title>{html.escape(title)}</title>",
        "<desc>Converter v2 compact exact-lattice vectorization</desc>",
    ]
    path_count = 0
    loop_count = 0
    command_bytes = 0
    for index in order:
        if population[index] <= 0:
            continue
        color = palette[index]
        if int(color[3]) <= 4:
            continue
        region = objects[index] if index < len(objects) else None
        if region is None:
            continue
        y_slice, x_slice = region
        offset = np.array([x_slice.start, y_slice.start], dtype=np.float64)
        paths = [
            compact_lattice_loop(loop + offset)
            for loop in _boundary_loops(labels[region] == index)
        ]
        paths = [path for path in paths if path]
        if not paths:
            continue
        data = "".join(paths)
        opacity = color[3] / 255.0
        attributes = (
            f'fill="#{color[0]:02x}{color[1]:02x}{color[2]:02x}" '
            'fill-rule="evenodd"'
        )
        if opacity < 0.999:
            attributes += f' fill-opacity="{_number(opacity)}"'
        parts.append(f'<path {attributes} d="{data}"/>')
        path_count += 1
        loop_count += len(paths)
        command_bytes += len(data.encode("utf-8"))
    parts.append("</svg>")
    svg = "\n".join(parts) + "\n"
    return svg, {
        "paths": path_count,
        "loops": loop_count,
        "command_bytes": command_bytes,
        "svg_bytes": len(svg.encode("utf-8")),
    }
