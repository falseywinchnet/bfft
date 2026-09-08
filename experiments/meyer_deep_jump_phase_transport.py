#!/usr/bin/env python3
"""Contour excision plus witnessed second-current phase transport.

The rejected scalar deep jump makes the same local error in two directions:
its response to a coherent jump is a thin bright/dark contour population, and
blindly suppressing that population also suppresses a real carrier crossing
the contour.  The repair must therefore be ordered:

1. learn and excise the structural contour population;
2. restore a value inside that population only when independent samples on
   both sides witness the same stable second-order recurrence.

For a centred harmonic current ``x`` the recurrence is

    x[n+1] - 2 c x[n] + x[n-1] = 0,       |c| <= 1.

Each one-dimensional chart estimates ``c`` exclusively outside the excised
run.  Both sides must contain a sign reversal, so a single two-lobed edge
response cannot authorize its own return.  Four Cartesian/diagonal chart
families propose continuations; their weighted barycentre is routed through
the unchanged Meyer G-capacity readout.  There is no cubic extrapolation,
source value inside the mask, learned model, or natural-image selection.
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
from experiments.meyer_deep_jump_causal_audit import (  # noqa: E402
    pure_carrier_scene,
    pure_edge_scene,
)
from experiments.meyer_first_pass_conditioning import checker_support_scene  # noqa: E402
from experiments.meyer_preconditioning_research import junction_texture_scene  # noqa: E402
from experiments.meyer_transverse_route_research import (  # noqa: E402
    jump_texture_components,
    native_structural_gate,
    tangent_reservoir_route,
)
from experiments.meyer_tsv_validation import (  # noqa: E402
    multiscale_crossing_scene,
    symmetric_support_scene,
)


OUT = ROOT / "experiments" / "out" / "meyer_deep_jump_phase_transport"


def otsu_high_population(value: np.ndarray) -> tuple[np.ndarray, float]:
    sample = np.asarray(value, dtype=np.float64)
    maximum = float(np.max(sample))
    if not maximum > 0.0:
        return np.zeros_like(sample, dtype=bool), 0.0
    histogram, edges = np.histogram(sample, bins=256, range=(0.0, maximum))
    centres = 0.5 * (edges[:-1] + edges[1:])
    mass = histogram.astype(np.float64)
    cumulative = np.cumsum(mass)
    moment = np.cumsum(mass * centres)
    total = float(cumulative[-1])
    total_moment = float(moment[-1])
    right = total - cumulative
    valid = (cumulative > 0.0) & (right > 0.0)
    between = np.zeros_like(centres)
    between[valid] = (
        (total_moment * cumulative[valid] - moment[valid] * total) ** 2
        / (cumulative[valid] * right[valid])
    )
    split = int(np.argmax(between[:-1]))
    boundary = float(edges[split + 1])
    return sample >= boundary, boundary


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    for _ in range(int(radius)):
        source = result.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                result |= np.roll(source, (dy, dx), axis=(0, 1))
    return result


def sign_changes(value: np.ndarray) -> int:
    sample = np.asarray(value, dtype=np.float64)
    tolerance = 128.0 * np.finfo(float).eps * max(
        float(np.max(np.abs(sample), initial=0.0)), 1.0
    )
    signs = np.sign(sample[np.abs(sample) > tolerance])
    return int(np.sum(signs[1:] != signs[:-1])) if signs.size else 0


def recurrence_coefficient(value: np.ndarray) -> tuple[float, float, float]:
    """Return centred harmonic coefficient, residual confidence, and mean."""

    source = np.asarray(value, dtype=np.float64)
    mean = float(np.mean(source))
    centred = source - mean
    if centred.size < 5 or sign_changes(centred) < 1:
        return 0.0, 0.0, mean
    centre = centred[1:-1]
    neighbour_sum = centred[:-2] + centred[2:]
    energy = float(np.sum(centre * centre))
    if not energy > 1e-30:
        return 0.0, 0.0, mean
    coefficient = float(np.clip(
        np.sum(centre * neighbour_sum) / (2.0 * energy), -0.999, 0.999
    ))
    residual = neighbour_sum - 2.0 * coefficient * centre
    residual_energy = float(np.sum(residual * residual))
    confidence = energy / (energy + residual_energy + 1e-30)
    return coefficient, confidence, mean


def fill_run(
    line: np.ndarray,
    start: int,
    stop: int,
    *,
    witness: int,
) -> tuple[np.ndarray | None, float]:
    left = np.asarray(line[max(0, start - witness):start], dtype=np.float64)
    right = np.asarray(line[stop:min(line.size, stop + witness)], dtype=np.float64)
    if left.size < 5 or right.size < 5:
        return None, 0.0
    left_c, left_q, left_mean = recurrence_coefficient(left)
    right_c, right_q, right_mean = recurrence_coefficient(right)
    # Both independent sides must witness at least one phase reversal.
    if left_q <= 0.0 or right_q <= 0.0:
        return None, 0.0
    coefficient = float(np.clip(
        (left_q * left_c + right_q * right_c) / (left_q + right_q),
        -0.999, 0.999,
    ))
    # Agreement is continuous. Opposed recurrences receive no authority;
    # there is no fitted accept/reject tolerance.
    agreement = max(0.0, 1.0 - abs(left_c - right_c) / 2.0)
    quality = agreement * (2.0 * left_q * right_q / (left_q + right_q))
    if not quality > 0.0:
        return None, 0.0
    length = stop - start
    forward = np.empty(length, dtype=np.float64)
    x0, x1 = left[-2] - left_mean, left[-1] - left_mean
    for index in range(length):
        x2 = 2.0 * coefficient * x1 - x0
        forward[index] = x2 + left_mean
        x0, x1 = x1, x2
    backward = np.empty(length, dtype=np.float64)
    x0, x1 = right[1] - right_mean, right[0] - right_mean
    for reverse_index in range(length - 1, -1, -1):
        x2 = 2.0 * coefficient * x1 - x0
        backward[reverse_index] = x2 + right_mean
        x0, x1 = x1, x2
    coordinate = (np.arange(length, dtype=np.float64) + 1.0) / (length + 1.0)
    prediction = (1.0 - coordinate) * forward + coordinate * backward
    # Bound the bridge by witnessed amplitude. A stable recurrence should
    # already satisfy this; the interval is a conservative numerical guard.
    bound = max(float(np.max(np.abs(left - left_mean))),
                float(np.max(np.abs(right - right_mean))), 1e-30)
    local_mean = (1.0 - coordinate) * left_mean + coordinate * right_mean
    prediction = local_mean + np.clip(prediction - local_mean, -bound, bound)
    return prediction, float(quality)


def fill_lines(
    value: np.ndarray,
    missing: np.ndarray,
    lines: list[tuple[np.ndarray, np.ndarray]],
    *,
    witness: int,
) -> tuple[np.ndarray, np.ndarray]:
    proposal = np.zeros_like(value, dtype=np.float64)
    authority = np.zeros_like(value, dtype=np.float64)
    for yy, xx in lines:
        line = value[yy, xx]
        mask = missing[yy, xx]
        padded = np.concatenate(([False], mask, [False]))
        transitions = np.flatnonzero(padded[1:] != padded[:-1])
        for start, stop in transitions.reshape(-1, 2):
            prediction, quality = fill_run(
                line, int(start), int(stop), witness=witness
            )
            if prediction is None:
                continue
            target_y = yy[start:stop]
            target_x = xx[start:stop]
            proposal[target_y, target_x] += quality * prediction
            authority[target_y, target_x] += quality
    return proposal, authority


def chart_lines(shape: tuple[int, int]) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    height, width = shape
    horizontal = [
        (np.full(width, y, dtype=np.int64), np.arange(width, dtype=np.int64))
        for y in range(height)
    ]
    vertical = [
        (np.arange(height, dtype=np.int64), np.full(height, x, dtype=np.int64))
        for x in range(width)
    ]
    positive = []
    negative = []
    for offset in range(-height + 1, width):
        y = np.arange(max(0, -offset), min(height, width - offset), dtype=np.int64)
        x = y + offset
        if y.size:
            positive.append((y, x))
        x_negative = width - 1 - x
        if y.size:
            negative.append((y, x_negative))
    return [horizontal, vertical, positive, negative]


def phase_transport(
    candidate: np.ndarray,
    missing: np.ndarray,
    *,
    witness: int,
) -> tuple[np.ndarray, dict[str, float]]:
    value = np.asarray(candidate, dtype=np.float64)
    total = np.zeros_like(value)
    authority = np.zeros_like(value)
    for family in chart_lines(value.shape):
        proposal, weight = fill_lines(
            value, missing, family, witness=witness
        )
        total += proposal
        authority += weight
    bridge = np.divide(
        total, authority, out=np.zeros_like(total), where=authority > 0.0
    )
    result = value.copy()
    result[missing] = bridge[missing]
    return result, {
        "missing_fraction": float(np.mean(missing)),
        "transported_missing_fraction": float(np.mean(missing & (authority > 0.0))),
        "mean_chart_authority_on_missing": float(
            np.mean(authority[missing]) if np.any(missing) else 0.0
        ),
        "maximum_chart_authority": float(np.max(authority)),
        "witness": int(witness),
    }


def split(
    source: np.ndarray,
    *,
    virtual_depth: int = 12,
    mask_dilation: int = 3,
    witness: int = 16,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    image = np.asarray(source, dtype=np.float64)
    gate = native_structural_gate(image)
    high, boundary = otsu_high_population(gate)
    missing = dilate(high, mask_dilation)
    candidate, jump, jump_diagnostic = jump_texture_components(
        image, gate, lam=0.05, virtual_passes=virtual_depth
    )
    transported, transport_diagnostic = phase_transport(
        candidate, missing, witness=witness
    )
    texture, route_diagnostic = tangent_reservoir_route(
        transported, gate, radius=40.0
    )
    cartoon = image - texture
    return cartoon, texture, {
        "virtual_depth": int(virtual_depth),
        "mask_dilation": int(mask_dilation),
        "support_boundary": boundary,
        "jump_rms": float(np.sqrt(np.mean(jump * jump))),
        "candidate_rms": float(np.sqrt(np.mean(candidate * candidate))),
        "transport": transport_diagnostic,
        "route": route_diagnostic,
        "recomposition_linf": float(np.max(np.abs(image - cartoon - texture))),
    }


def truth(scene: dict) -> tuple[np.ndarray, np.ndarray]:
    if "truth_cartoon" in scene:
        return np.asarray(scene["truth_cartoon"]), np.asarray(scene["truth_texture"])
    if "hard_composition" in scene:
        return np.asarray(scene["hard_composition"]), np.asarray(scene["material_texture"])
    return np.asarray(scene["cartoon"]), np.asarray(scene["texture"])


def score(cartoon: np.ndarray, texture: np.ndarray, scene: dict) -> dict[str, float]:
    truth_u, truth_v = truth(scene)
    contour = np.asarray(scene["contour"], dtype=bool)
    scale = max(float(np.sqrt(np.mean(truth_v * truth_v))), 1.0)
    return {
        "texture_relative_rms_error": float(np.sqrt(np.mean((texture - truth_v) ** 2)) / scale),
        "texture_gain": float(np.sum(texture * truth_v) / max(np.sum(truth_v * truth_v), 1e-30)),
        "contour_texture_rms": float(np.sqrt(np.mean(texture[contour] ** 2))),
        "cartoon_relative_rms_error": float(
            np.linalg.norm(cartoon - truth_u) / max(np.linalg.norm(truth_u), 1.0)
        ),
    }


def plain(value: np.ndarray) -> Image.Image:
    return Image.fromarray(np.uint8(np.clip(value, 0.0, 255.0)), mode="L").convert("RGB")


def signed(value: np.ndarray, scale: float) -> Image.Image:
    shown = np.clip(127.5 + 127.5 * value / max(scale, 1e-30), 0.0, 255.0)
    return Image.fromarray(np.uint8(shown), mode="L").convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    scenes = (
        pure_edge_scene(args.size), pure_carrier_scene(args.size),
        symmetric_support_scene(args.size), checker_support_scene(args.size),
        multiscale_crossing_scene(args.size), junction_texture_scene(args.size),
    )
    arms = tuple(
        (depth, dilation, witness)
        for depth in (12, 16)
        for dilation in (1, 2, 3, 4)
        for witness in (12, 16, 24)
    )
    report: dict[str, object] = {"arms": {}, "ranking": []}
    outputs = {}
    for depth, dilation, witness in arms:
        key = f"K{depth}_D{dilation}_W{witness}"
        rows = {}
        for scene in scenes:
            result = split(
                scene["source"], virtual_depth=depth,
                mask_dilation=dilation, witness=witness,
            )
            outputs[(scene["name"], key)] = result[:2]
            rows[scene["name"]] = {**score(result[0], result[1], scene), "diagnostic": result[2]}
        edge = rows["pure_edge"]["contour_texture_rms"]
        material = float(np.mean([
            rows[scene["name"]]["texture_relative_rms_error"] for scene in scenes[1:]
        ]))
        crossing = float(np.mean([
            rows[name]["texture_relative_rms_error"]
            for name in ("multiscale_crossing", "junction_texture")
        ]))
        objective = edge + 4.0 * material + 2.0 * crossing
        report["arms"][key] = {
            "virtual_depth": depth, "mask_dilation": dilation, "witness": witness,
            "objective": objective, "scenes": rows,
        }
        report["ranking"].append({"key": key, "objective": objective})
        print(f"{key:14s} edge={edge:.4f} material={material:.4f} cross={crossing:.4f}")
    report["ranking"].sort(key=lambda row: row["objective"])
    selected = [row["key"] for row in report["ranking"][:5]]
    titles = ("source", "truth cartoon", "truth texture", "public texture", *selected)
    cell, header, label = 160, 30, 150
    canvas = Image.new(
        "RGB", (label + len(titles) * cell, header + len(scenes) * cell), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for column, title in enumerate(titles):
        draw.text((label + column * cell + 3, 8), title, fill="black")
    for row, scene in enumerate(scenes):
        truth_u, truth_v = truth(scene)
        public = bfft.MeyerPlan(
            np.asarray(scene["source"]).shape, passes=1, threads=8
        ).split(scene["source"])
        scale = max(float(np.percentile(np.abs(truth_v), 99.5)), 1.0)
        panels = [
            plain(scene["source"]), plain(truth_u), signed(truth_v, scale),
            signed(public[1], scale),
        ]
        panels.extend(signed(outputs[(scene["name"], key)][1], scale) for key in selected)
        y0 = header + row * cell
        draw.text((4, y0 + 8), scene["name"], fill="black")
        for column, panel in enumerate(panels):
            canvas.paste(
                panel.resize((cell, cell), Image.Resampling.NEAREST),
                (label + column * cell, y0),
            )
    canvas.save(args.out / "comparison.png")
    (args.out / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
