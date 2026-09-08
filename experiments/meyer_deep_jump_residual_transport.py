#!/usr/bin/env python3
"""Two-stage residual-current repair for the fixed-cost Meyer deep jump.

Stage 1 retains the existing feed-forward jump observation and learns a
material seed from the jump-cancelled residual.  It does *not* emit the
scalar jump complement as texture.  Stage 2 treats high-confidence structural
support as missing data in that material seed and continues the already
witnessed material current across it from target-excluded one-dimensional
charts.  Thus a monotone edge cannot seed texture, while a carrier observed on
both sides of the edge can cross it.

This is a falsification experiment, not yet the public implementation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bfft  # noqa: E402
from experiments.meyer_conv_current_pyramid_split import _otsu_confidence  # noqa: E402
from experiments.meyer_deep_jump_causal_audit import (  # noqa: E402
    canonical_scene,
    pure_carrier_scene,
    pure_edge_scene,
)
from experiments.meyer_preconditioning_research import (  # noqa: E402
    junction_texture_scene,
)
from experiments.meyer_transverse_route_research import (  # noqa: E402
    jump_texture_components,
    native_structural_gate,
    tangent_reservoir_route,
)
from experiments.meyer_tsv_validation import (  # noqa: E402
    multiscale_crossing_scene,
    symmetric_support_scene,
)


OUT = ROOT / "experiments" / "out" / "meyer_deep_jump_residual_transport"


def dilate8(mask: np.ndarray, radius: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    for _ in range(int(radius)):
        source = result.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                result |= np.roll(source, (dy, dx), axis=(0, 1))
    return result


def _directional_hermite(
    value: np.ndarray,
    missing: np.ndarray,
    dy: int,
    dx: int,
    maximum_distance: int,
) -> np.ndarray:
    """Continue across a missing run using only samples outside that run."""

    source = np.asarray(value, dtype=np.float64)
    mask = np.asarray(missing, dtype=bool)
    shape = source.shape
    left_value = np.zeros(shape)
    right_value = np.zeros(shape)
    left_slope = np.zeros(shape)
    right_slope = np.zeros(shape)
    left_distance = np.zeros(shape, dtype=np.int16)
    right_distance = np.zeros(shape, dtype=np.int16)

    unresolved_left = mask.copy()
    unresolved_right = mask.copy()
    for distance in range(1, int(maximum_distance) + 1):
        left_mask = np.roll(mask, (dy * distance, dx * distance), axis=(0, 1))
        right_mask = np.roll(mask, (-dy * distance, -dx * distance), axis=(0, 1))
        found_left = unresolved_left & ~left_mask
        found_right = unresolved_right & ~right_mask
        if np.any(found_left):
            left = np.roll(
                source, (dy * distance, dx * distance), axis=(0, 1)
            )
            farther = np.roll(
                source, (dy * (distance + 1), dx * (distance + 1)),
                axis=(0, 1),
            )
            farther_mask = np.roll(
                mask, (dy * (distance + 1), dx * (distance + 1)),
                axis=(0, 1),
            )
            left_value[found_left] = left[found_left]
            secant = np.zeros(shape)
            secant[found_left] = left[found_left] - farther[found_left]
            secant[found_left & farther_mask] = 0.0
            left_slope[found_left] = secant[found_left]
            left_distance[found_left] = distance
            unresolved_left[found_left] = False
        if np.any(found_right):
            right = np.roll(
                source, (-dy * distance, -dx * distance), axis=(0, 1)
            )
            farther = np.roll(
                source, (-dy * (distance + 1), -dx * (distance + 1)),
                axis=(0, 1),
            )
            farther_mask = np.roll(
                mask, (-dy * (distance + 1), -dx * (distance + 1)),
                axis=(0, 1),
            )
            right_value[found_right] = right[found_right]
            secant = np.zeros(shape)
            secant[found_right] = farther[found_right] - right[found_right]
            secant[found_right & farther_mask] = 0.0
            right_slope[found_right] = secant[found_right]
            right_distance[found_right] = distance
            unresolved_right[found_right] = False
        if not np.any(unresolved_left | unresolved_right):
            break

    valid = mask & (left_distance > 0) & (right_distance > 0)
    span = left_distance.astype(np.float64) + right_distance.astype(np.float64)
    t = np.divide(
        left_distance,
        span,
        out=np.zeros_like(span),
        where=span > 0.0,
    )
    t2, t3 = t * t, t * t * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    prediction = (
        h00 * left_value + h10 * span * left_slope
        + h01 * right_value + h11 * span * right_slope
    )
    return np.where(valid, prediction, np.nan)


def transport_material_across_structure(
    material_seed: np.ndarray,
    structural_missing: np.ndarray,
    *,
    maximum_distance: int = 16,
) -> tuple[np.ndarray, dict[str, float]]:
    """Median of four target-excluded current continuations."""

    seed = np.asarray(material_seed, dtype=np.float64)
    missing = np.asarray(structural_missing, dtype=bool)
    family = np.stack([
        _directional_hermite(seed, missing, dy, dx, maximum_distance)
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1))
    ])
    finite = np.isfinite(family)
    count = np.sum(finite, axis=0)
    ordered = np.sort(np.where(finite, family, np.inf), axis=0)
    # Lower median for two charts, ordinary median for three/four.  All
    # candidates are target-excluded; no source value inside the mask enters.
    index = np.maximum((count - 1) // 2, 0)
    prediction = np.take_along_axis(
        ordered, index[None, ...], axis=0
    )[0]
    valid = missing & (count > 0) & np.isfinite(prediction)
    transported = seed.copy()
    transported[valid] = prediction[valid]
    transported[missing & ~valid] = 0.0
    return transported, {
        "structural_missing_fraction": float(np.mean(missing)),
        "transported_missing_fraction": float(np.mean(valid)),
        "chart_count_minimum_on_missing": (
            int(np.min(count[missing])) if np.any(missing) else 0
        ),
        "chart_count_median_on_missing": (
            float(np.median(count[missing])) if np.any(missing) else 0.0
        ),
        "maximum_distance": int(maximum_distance),
    }


def split(
    source: np.ndarray,
    *,
    lam: float = 0.05,
    mu: float = 40.0,
    virtual_passes: int = 8,
    mask_dilation: int = 1,
    maximum_distance: int = 16,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    image = np.asarray(source, dtype=np.float64)
    gate = native_structural_gate(image)
    confidence = _otsu_confidence(gate)
    missing = dilate8(confidence > 0.0, mask_dilation)
    material_seed, jump, jump_diagnostic = jump_texture_components(
        image, gate, lam=lam, virtual_passes=virtual_passes
    )
    transported, transport_diagnostic = transport_material_across_structure(
        material_seed, missing, maximum_distance=maximum_distance
    )
    texture, route_diagnostic = tangent_reservoir_route(
        transported, gate, radius=mu
    )
    cartoon = image - texture
    return cartoon, texture, {
        "jump_potential_rms": float(np.sqrt(np.mean(jump * jump))),
        "material_seed_rms": float(np.sqrt(np.mean(material_seed * material_seed))),
        "transported_material_rms": float(np.sqrt(np.mean(transported * transported))),
        "recomposition_linf": float(np.max(np.abs(image - cartoon - texture))),
        "jump": jump_diagnostic,
        "transport": transport_diagnostic,
        "route": route_diagnostic,
    }


def _rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(value, dtype=np.float64) ** 2)))


def metrics(cartoon: np.ndarray, texture: np.ndarray, scene: dict) -> dict[str, float]:
    truth_u = np.asarray(scene["truth_cartoon"])
    truth_v = np.asarray(scene["truth_texture"])
    contour = np.asarray(scene["contour"], dtype=bool)
    scale = max(_rms(truth_v), 1.0)
    return {
        "cartoon_relative_rms_error": _rms(cartoon - truth_u) / max(_rms(truth_u), 1.0),
        "texture_relative_rms_error": _rms(texture - truth_v) / scale,
        "texture_gain": float(np.sum(texture * truth_v) / max(np.sum(truth_v * truth_v), 1e-30)),
        "contour_texture_rms": _rms(texture[contour]),
        "contour_texture_rms_over_texture_scale": _rms(texture[contour]) / scale,
    }


def _plain(value: np.ndarray) -> Image.Image:
    return Image.fromarray(np.uint8(np.clip(value, 0, 255)), mode="L").convert("RGB")


def _signed(value: np.ndarray, scale: float) -> Image.Image:
    shown = np.uint8(np.clip(127.5 + 127.5 * value / max(scale, 1e-30), 0, 255))
    return Image.fromarray(shown, mode="L").convert("RGB")


def render(rows: list[tuple], path: Path) -> None:
    titles = (
        "source", "truth texture", "old deep texture", "new cartoon", "new texture"
    )
    cell, header, label = 176, 32, 150
    canvas = Image.new("RGB", (label + len(titles) * cell, header + len(rows) * cell), "white")
    draw = ImageDraw.Draw(canvas)
    for j, title in enumerate(titles):
        draw.text((label + j * cell + 3, 8), title, fill="black")
    for i, (name, source, truth_v, old_v, new_u, new_v) in enumerate(rows):
        y0 = header + i * cell
        draw.text((4, y0 + 8), name, fill="black")
        scale = max(float(np.percentile(np.abs(truth_v), 99.5)), 1.0)
        panels = (
            _plain(source), _signed(truth_v, scale), _signed(old_v, scale),
            _plain(new_u), _signed(new_v, scale),
        )
        for j, panel in enumerate(panels):
            canvas.paste(panel.resize((cell, cell)), (label + j * cell, y0))
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--mask-dilation", type=int, default=1)
    parser.add_argument("--maximum-distance", type=int, default=16)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    scenes = (
        pure_edge_scene(args.size),
        pure_carrier_scene(args.size),
        canonical_scene(symmetric_support_scene(args.size)),
        canonical_scene(multiscale_crossing_scene(args.size)),
        canonical_scene(junction_texture_scene(args.size)),
    )
    report = {"scenes": {}, "mask_dilation": args.mask_dilation,
              "maximum_distance": args.maximum_distance}
    rows = []
    arrays: dict[str, np.ndarray] = {}
    for scene in scenes:
        source = np.asarray(scene["source"])
        plan = bfft.MeyerPlan(source.shape, lam=0.05, mu=40.0, passes=1, threads=1)
        old_u, old_v = plan.split_jump_measure(source)
        new_u, new_v, diagnostic = split(
            source, mask_dilation=args.mask_dilation,
            maximum_distance=args.maximum_distance,
        )
        name = str(scene["name"])
        report["scenes"][name] = {
            "old": metrics(old_u, old_v, scene),
            "residual_transport": metrics(new_u, new_v, scene),
            "diagnostic": diagnostic,
        }
        score = report["scenes"][name]["residual_transport"]
        print(
            f"{name:20s} gain={score['texture_gain']:.4f} "
            f"texture={score['texture_relative_rms_error']:.4f} "
            f"contour={score['contour_texture_rms']:.4f}"
        )
        rows.append((name, source, scene["truth_texture"], old_v, new_u, new_v))
        for label, value in (("source", source), ("truth_texture", scene["truth_texture"]),
                             ("old_texture", old_v), ("new_cartoon", new_u),
                             ("new_texture", new_v)):
            arrays[f"{name}_{label}"] = np.asarray(value, dtype=np.float32)
    (args.out / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(args.out / "arrays.npz", **arrays)
    render(rows, args.out / "comparison.png")


if __name__ == "__main__":
    main()
