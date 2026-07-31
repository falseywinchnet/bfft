#!/usr/bin/env python3
"""Paired one-sided eikonal-Lanczos jump estimation for finite Meyer.

This reuses the viewer resampler's algebraic tensor chart and Lanczos-2
lookup table as a fixed analysis operator.  It never resizes an image.  Two
identical kernels are centred on opposite sides of the local structural
normal; their difference estimates a plateau jump while their elongated
tangent support cancels a crossing carrier.  All validation sources are
authored analytic truth.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

try:
    from numba import njit, prange
except ImportError:  # pragma: no cover
    njit = None
    prange = range

from port_needed.eikonal_lanczos import LANCZOS2_TABLE
from experiments.meyer_first_pass_conditioning import (
    checker_support_scene,
    gaussian_periodic,
    lap_hat,
)
from experiments.meyer_preconditioning_research import junction_texture_scene
from experiments.meyer_transverse_route_research import (
    divergence,
    jump_cancelled_texture,
    native_structural_gate,
    tangent_reservoir_route,
)
from experiments.meyer_tsv_validation import (
    multiscale_crossing_scene,
    score_split,
    symmetric_support_scene,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "out" / "meyer_lanczos_jump"


def render_audit(scene: dict, columns: tuple, path: Path) -> None:
    """Two-row truth comparison with common cartoon/texture scales."""
    panels = []
    for name, split in columns:
        value = split[0]
        shown = np.clip(value, 0.0, 255.0)
        image = Image.fromarray(shown.astype(np.uint8), mode="L").convert("RGB")
        panel = Image.new("RGB", (image.width, image.height + 22), "white")
        panel.paste(image, (0, 22))
        ImageDraw.Draw(panel).text((4, 4), f"{name} / cartoon", fill="black")
        panels.append(panel)
    for name, split in columns:
        value = split[1]
        error = value - scene["texture"]
        rmse = float(np.sqrt(np.mean(error * error)))
        suffix = "" if name == "TRUTH" else f" / RMSE {rmse:.2f}"
        shown = np.clip(127.5 + 2.0 * value, 0.0, 255.0)
        image = Image.fromarray(shown.astype(np.uint8), mode="L").convert("RGB")
        panel = Image.new("RGB", (image.width, image.height + 22), "white")
        panel.paste(image, (0, 22))
        ImageDraw.Draw(panel).text(
            (4, 4), f"{name} / texture{suffix}", fill="black")
        panels.append(panel)
    count = len(columns)
    output = Image.new(
        "RGB", (count * panels[0].width, 2 * panels[0].height))
    for index, panel in enumerate(panels):
        output.paste(panel, (
            (index % count) * panel.width,
            (index // count) * panel.height,
        ))
    output.save(path)


def _identity(function):  # pragma: no cover
    return function


_compile_parallel = (
    njit(cache=True, parallel=True, fastmath=False)
    if njit is not None else _identity
)


@_compile_parallel
def _paired_lanczos_kernel(
    source,
    normal_x,
    normal_y,
    coherence,
    offset,
    anisotropy,
    table,
):
    """Return plus-minus normalized one-sided Lanczos moments."""
    height, width = source.shape
    output = np.empty_like(source)
    table_scale = (len(table) - 1) / 2.0
    radius = int(np.ceil(offset + 2.0 * (1.0 + max(anisotropy, 0.0))))
    for y in prange(height):
        for x in range(width):
            nx = normal_x[y, x]
            ny = normal_y[y, x]
            tx = -ny
            ty = nx
            stretch = 1.0 + anisotropy * coherence[y, x]
            plus_sum = 0.0
            plus_weight = 0.0
            minus_sum = 0.0
            minus_weight = 0.0
            for side in (-1.0, 1.0):
                center_x = x + side * offset * nx
                center_y = y + side * offset * ny
                accum = 0.0
                total = 0.0
                for sample_y_raw in range(y - radius, y + radius + 1):
                    dy = sample_y_raw - center_y
                    sample_y = sample_y_raw % height
                    for sample_x_raw in range(x - radius, x + radius + 1):
                        dx = sample_x_raw - center_x
                        normal_distance = abs(
                            (dx * nx + dy * ny) * stretch)
                        tangent_distance = abs(
                            (dx * tx + dy * ty) / stretch)
                        if normal_distance >= 2.0 or tangent_distance >= 2.0:
                            continue
                        normal_position = normal_distance * table_scale
                        normal_index = min(
                            int(normal_position), len(table) - 2)
                        normal_fraction = normal_position - normal_index
                        normal_weight = (
                            table[normal_index]
                            + normal_fraction * (
                                table[normal_index + 1]
                                - table[normal_index]
                            )
                        )
                        tangent_position = tangent_distance * table_scale
                        tangent_index = min(
                            int(tangent_position), len(table) - 2)
                        tangent_fraction = tangent_position - tangent_index
                        tangent_weight = (
                            table[tangent_index]
                            + tangent_fraction * (
                                table[tangent_index + 1]
                                - table[tangent_index]
                            )
                        )
                        weight = normal_weight * tangent_weight
                        accum += weight * source[
                            sample_y, sample_x_raw % width]
                        total += weight
                if side > 0.0:
                    plus_sum = accum
                    plus_weight = total
                else:
                    minus_sum = accum
                    minus_weight = total
            plus = plus_sum / plus_weight if abs(plus_weight) > 1e-12 \
                else source[y, x]
            minus = minus_sum / minus_weight if abs(minus_weight) > 1e-12 \
                else source[y, x]
            output[y, x] = plus - minus
    return output


def structural_frame(source: np.ndarray) -> tuple[np.ndarray, ...]:
    """Algebraic dominant tensor eigenvector, coherence, and raw gradient."""
    source = np.asarray(source, dtype=np.float64)
    gx = np.roll(source, -1, axis=1) - source
    gy = np.roll(source, -1, axis=0) - source
    xx = gaussian_periodic(gx * gx, 1.15)
    xy = gaussian_periodic(gx * gy, 1.15)
    yy = gaussian_periodic(gy * gy, 1.15)
    difference = xx - yy
    doubled_xy = 2.0 * xy
    discriminant = np.hypot(difference, doubled_xy)
    cosine_double = np.divide(
        difference,
        discriminant,
        out=np.ones_like(discriminant),
        where=discriminant > 1e-30,
    )
    sine_double = np.divide(
        doubled_xy,
        discriminant,
        out=np.zeros_like(discriminant),
        where=discriminant > 1e-30,
    )
    normal_x = np.sqrt(np.maximum(0.5 * (1.0 + cosine_double), 0.0))
    normal_y = np.copysign(
        np.sqrt(np.maximum(0.5 * (1.0 - cosine_double), 0.0)),
        np.where(np.abs(sine_double) > 1e-30, sine_double, 1.0),
    )
    coherence = np.clip(
        discriminant / np.maximum(xx + yy, 1e-30), 0.0, 1.0)
    return normal_x, normal_y, coherence, gx, gy


def lanczos_jump_cancelled_texture(
    source: np.ndarray,
    gate: np.ndarray,
    *,
    lam: float = 0.05,
    virtual_passes: int = 8,
    offset: float = 1.5,
    anisotropy: float = 0.75,
    localized: bool = True,
) -> tuple[np.ndarray, dict]:
    """Finite texture proposal from paired eikonal-Lanczos jump moments."""
    source = np.asarray(source, dtype=np.float64)
    gate = np.asarray(gate, dtype=np.float64)
    nx, ny, coherence, gx, gy = structural_frame(source)
    jump = _paired_lanczos_kernel(
        source,
        nx,
        ny,
        coherence,
        float(offset),
        float(anisotropy),
        LANCZOS2_TABLE,
    )
    eta = 2.0 * float(lam)
    threshold = 1.0 / (2.0 * eta)
    jump_magnitude = np.abs(jump)
    selected = np.maximum(jump_magnitude - threshold, 0.0)
    signed_selected = np.copysign(selected, jump)
    if localized:
        raw_normal = gx * nx + gy * ny
        localization = np.minimum(
            np.abs(raw_normal) / np.maximum(jump_magnitude, 1e-30), 1.0)
    else:
        localization = np.ones_like(source)
    flux_x = gate * localization * signed_selected * nx
    flux_y = gate * localization * signed_selected * ny

    laplacian = lap_hat(source.shape)
    safe = np.where(np.abs(laplacian) > 1e-15, laplacian, 1.0)
    jump_spectrum = np.fft.fft2(divergence(flux_x, flux_y)) / safe
    jump_spectrum[0, 0] = 0.0
    transfer = lam / (lam - eta * laplacian)
    highpass = 1.0 - transfer ** int(virtual_passes)
    proposed = np.fft.ifft2(
        highpass * (np.fft.fft2(source) - jump_spectrum)
    ).real
    return proposed, {
        "offset": float(offset),
        "anisotropy": float(anisotropy),
        "localized": bool(localized),
        "half_threshold": float(threshold),
        "selected_fraction": float(np.mean(selected > 0.0)),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"quality_sources": "authored analytic truth only", "scenes": {}}
    for scene in (
        symmetric_support_scene(256),
        multiscale_crossing_scene(256),
        checker_support_scene(256),
        junction_texture_scene(256),
    ):
        source = scene["source"]
        gate = native_structural_gate(source)
        raw_proposed, _ = jump_cancelled_texture(
            source, gate, lam=0.05, virtual_passes=8)
        raw_texture, _ = tangent_reservoir_route(
            raw_proposed, gate, radius=40.0)
        raw_split = (source - raw_texture, raw_texture)
        rows = {}
        splits = {"RAW JUMP": raw_split}
        for name, localized in (("direct", False), ("localized", True)):
            proposed, diagnostic = lanczos_jump_cancelled_texture(
                source, gate, localized=localized)
            texture, route = tangent_reservoir_route(
                proposed, gate, radius=40.0)
            split = (source - texture, texture)
            splits[f"LANCZOS {name.upper()}"] = split
            rows[name] = {
                "scores": score_split(*split, scene),
                "texture_relative_l2": float(
                    np.linalg.norm(texture - scene["texture"])
                    / np.linalg.norm(scene["texture"])
                ),
                "diagnostic": diagnostic,
                "route": route,
            }
        rows["raw_gradient"] = {
            "scores": score_split(*raw_split, scene),
            "texture_relative_l2": float(
                np.linalg.norm(raw_texture - scene["texture"])
                / np.linalg.norm(scene["texture"])
            ),
        }
        report["scenes"][scene["name"]] = rows
        if scene["name"] == "multiscale_crossing":
            render_audit(
                scene,
                (
                    ("TRUTH", (scene["cartoon"], scene["texture"])),
                    ("RAW JUMP", splits["RAW JUMP"]),
                    ("LANCZOS DIRECT", splits["LANCZOS DIRECT"]),
                    ("LANCZOS LOCALIZED", splits["LANCZOS LOCALIZED"]),
                ),
                OUT / "multiscale_truth_audit.png",
            )
        print(f"\n{scene['name']}")
        for name in ("raw_gradient", "direct", "localized"):
            row = rows[name]
            score = row["scores"]
            print(
                f"  {name:12s} rel {row['texture_relative_l2']:.3f}  "
                f"gain {score['interior_texture_gain']:.3f}  "
                f"error {score['interior_texture_relative_rms_error']:.3f}  "
                f"contour {score['contour_excess_texture_rms']:.3f}"
            )
    path = OUT / "results.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
