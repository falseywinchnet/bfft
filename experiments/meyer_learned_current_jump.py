#!/usr/bin/env python3
"""Learn Meyer ownership first, then jump in conservative current coordinates.

This is the surgical replacement probe for the rejected source-only scalar
deep jump.  A short ordinary Meyer prefix supplies *observed nonlinear state*.
The prefix texture is analyzed in compact, exactly invertible CONV block
coordinates.  A coordinate may enter the deep jump only when

1. its last two nonlinear transfer increments have the same orientation;
2. that recurrence is locally coherent; and
3. the observed current occupies a two-dimensional oscillatory population,
   rather than only a sparse two-lobed contour response.

The jump is consequently a bounded continuation of learned current.  It is
not a source high-pass, an edge mask, or a scalar Poisson potential.  Exact
recomposition is imposed last by ``cartoon = source - texture``.

This file is a falsification experiment.  Nothing here is a public operator
until the pure-edge, pure-carrier, supported-carrier, crossing, and junction
visual gates all pass with one fixed parameter set.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bfft  # noqa: E402
from experiments.conv_conservative_multiresolution import (  # noqa: E402
    ConservativeBlockState,
    conservative_analysis_2d,
    conservative_synthesis_2d,
)
from experiments.meyer_conv_second_transport_split import (  # noqa: E402
    _conv_current_alternation,
)
from experiments.meyer_deep_jump_causal_audit import (  # noqa: E402
    canonical_scene,
    pure_carrier_scene,
    pure_edge_scene,
)
from experiments.meyer_first_pass_conditioning import (  # noqa: E402
    checker_support_scene,
)
from experiments.meyer_preconditioning_research import (  # noqa: E402
    junction_texture_scene,
)
from experiments.meyer_tsv_validation import (  # noqa: E402
    multiscale_crossing_scene,
    symmetric_support_scene,
)


OUT = ROOT / "experiments" / "out" / "meyer_learned_current_jump"


@dataclass(frozen=True)
class CurrentPyramid:
    coarse: np.ndarray
    levels: tuple[ConservativeBlockState, ...]


def analyze_current(value: np.ndarray, levels: int) -> CurrentPyramid:
    current = np.asarray(value, dtype=np.float64)
    states: list[ConservativeBlockState] = []
    for _ in range(int(levels)):
        if min(current.shape) < 10 or current.shape[0] % 2 or current.shape[1] % 2:
            break
        state = conservative_analysis_2d(current)
        states.append(state)
        current = state.coarse
    return CurrentPyramid(current, tuple(states))


def synthesize_current(
    coarse: np.ndarray,
    details: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> np.ndarray:
    current = np.asarray(coarse, dtype=np.float64)
    for horizontal, vertical, mixed in reversed(details):
        current = conservative_synthesis_2d(
            current, horizontal, vertical, mixed
        )
    return current


def periodic_box_mean(value: np.ndarray, radius: int) -> np.ndarray:
    """Small fixed periodic population observer with no fitted threshold."""

    source = np.asarray(value, dtype=np.float64)
    result = np.zeros_like(source)
    count = 0
    for dy in range(-int(radius), int(radius) + 1):
        for dx in range(-int(radius), int(radius) + 1):
            result += np.roll(source, (dy, dx), axis=(0, 1))
            count += 1
    return result / float(count)


def block_mean(value: np.ndarray) -> np.ndarray:
    source = np.asarray(value, dtype=np.float64)
    return 0.25 * (
        source[0::2, 0::2] + source[0::2, 1::2]
        + source[1::2, 0::2] + source[1::2, 1::2]
    )


def learned_population(value: np.ndarray, radius: int) -> np.ndarray:
    """Return broad alternating-current population, not edge magnitude.

    A monotone jump can create one reversal in the response current, but it
    cannot populate the complete six-node alternating chart throughout a
    two-dimensional neighbourhood.  Repeated carrier current can.  Squaring
    the bounded population retains that distinction without an acceptance
    threshold or a named edge/texture class.
    """

    alternation = _conv_current_alternation(value)
    return np.clip(periodic_box_mean(alternation, radius), 0.0, 1.0) ** 2


def local_recurrence_jump(
    before: np.ndarray,
    previous: np.ndarray,
    current: np.ndarray,
    *,
    remaining: int,
    radius: int,
    gain_cap: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Bounded geometric continuation of a witnessed current coordinate."""

    old_increment = previous - before
    new_increment = current - previous
    numerator = periodic_box_mean(old_increment * new_increment, radius)
    old_energy = periodic_box_mean(old_increment * old_increment, radius)
    new_energy = periodic_box_mean(new_increment * new_increment, radius)
    ratio = np.clip(
        np.divide(
            numerator,
            old_energy + 1e-30,
            out=np.zeros_like(numerator),
            where=old_energy > 0.0,
        ),
        0.0,
        0.95,
    )
    coherence = np.clip(
        np.divide(
            np.maximum(numerator, 0.0) ** 2,
            old_energy * new_energy + 1e-30,
            out=np.zeros_like(numerator),
            where=(old_energy > 0.0) & (new_energy > 0.0),
        ),
        0.0,
        1.0,
    )
    geometric = np.divide(
        ratio * (1.0 - ratio ** int(remaining)),
        1.0 - ratio,
        out=np.zeros_like(ratio),
        where=ratio < 1.0,
    )
    gain = coherence * np.minimum(geometric, float(gain_cap))
    proposal = current + gain * new_increment
    return proposal, {
        "mean_ratio": float(np.mean(ratio)),
        "mean_coherence": float(np.mean(coherence)),
        "mean_gain": float(np.mean(gain)),
        "maximum_gain": float(np.max(gain)),
    }


def learned_current_jump(
    source: np.ndarray,
    *,
    learning_passes: int = 4,
    virtual_target: int = 16,
    levels: int = 4,
    population_radius: int = 4,
    recurrence_radius: int = 1,
    gain_cap: float = 6.0,
    threads: int = 8,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Two-stage split: nonlinear learning prefix, then one current jump."""

    image = np.ascontiguousarray(source, dtype=np.float64)
    if learning_passes < 3:
        raise ValueError("at least three learning passes are required")
    if virtual_target <= learning_passes:
        raise ValueError("virtual target must exceed the learning prefix")
    plan = bfft.MeyerPlan(
        image.shape, lam=0.05, mu=40.0,
        passes=int(learning_passes), threads=int(threads),
    )
    started = time.perf_counter()
    _cartoon_trace, texture_trace = plan.trace(image)
    learning_ms = 1000.0 * (time.perf_counter() - started)
    pyramids = [analyze_current(texture, levels) for texture in texture_trace[-3:]]
    actual_levels = len(pyramids[-1].levels)
    if any(len(pyramid.levels) != actual_levels for pyramid in pyramids):
        raise RuntimeError("inconsistent conservative pyramid depth")

    remaining = int(virtual_target - learning_passes)
    before, previous, current = pyramids
    coarse, coarse_diag = local_recurrence_jump(
        before.coarse, previous.coarse, current.coarse,
        remaining=remaining,
        radius=recurrence_radius,
        gain_cap=min(float(gain_cap), 2.0),
    )
    # A genuinely oscillatory coarse remainder must still be witnessed as a
    # current population at that scale.  Otherwise it belongs to cartoon.
    population = learned_population(current.coarse, max(1, population_radius // 2))
    coarse = population * coarse

    details: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    level_diagnostics = []
    current_field = texture_trace[-1]
    for level in range(actual_levels):
        left = before.levels[level]
        middle = previous.levels[level]
        right = current.levels[level]
        population = block_mean(learned_population(
            current_field, population_radius
        ))
        components = []
        component_diagnostics = []
        for name in ("horizontal_detail", "vertical_detail", "mixed_detail"):
            proposal, diagnostic = local_recurrence_jump(
                getattr(left, name), getattr(middle, name), getattr(right, name),
                remaining=remaining,
                radius=recurrence_radius,
                gain_cap=gain_cap,
            )
            components.append(population * proposal)
            component_diagnostics.append({"component": name, **diagnostic})
        details.append(tuple(components))
        level_diagnostics.append({
            "level": level + 1,
            "population_mean": float(np.mean(population)),
            "population_maximum": float(np.max(population)),
            "components": component_diagnostics,
        })
        current_field = right.coarse

    texture = synthesize_current(coarse, details)
    cartoon = image - texture
    return cartoon, texture, {
        "learning_passes": int(learning_passes),
        "virtual_target": int(virtual_target),
        "conservative_levels": int(actual_levels),
        "population_radius": int(population_radius),
        "recurrence_radius": int(recurrence_radius),
        "gain_cap": float(gain_cap),
        "learning_milliseconds": learning_ms,
        "coarse": coarse_diag,
        "levels": level_diagnostics,
        "texture_rms": float(np.sqrt(np.mean(texture * texture))),
        "recomposition_linf": float(np.max(np.abs(image - cartoon - texture))),
    }


def _plain(value: np.ndarray) -> Image.Image:
    return Image.fromarray(np.uint8(np.clip(value, 0.0, 255.0)), mode="L").convert("RGB")


def _signed(value: np.ndarray, scale: float) -> Image.Image:
    shown = np.clip(127.5 + 127.5 * value / max(scale, 1e-30), 0.0, 255.0)
    return Image.fromarray(np.uint8(shown), mode="L").convert("RGB")


def _canonical(scene: dict) -> dict:
    if "truth_cartoon" in scene and "truth_texture" in scene:
        return scene
    return canonical_scene(scene)


def _truth(scene: dict) -> tuple[np.ndarray, np.ndarray]:
    canonical = _canonical(scene)
    return (
        np.asarray(canonical["truth_cartoon"], dtype=np.float64),
        np.asarray(canonical["truth_texture"], dtype=np.float64),
    )


def _score(
    cartoon: np.ndarray, texture: np.ndarray, scene: dict
) -> dict[str, float]:
    truth_u, truth_v = _truth(scene)
    contour = np.asarray(scene["contour"], dtype=bool)
    texture_norm2 = float(np.sum(truth_v * truth_v))
    texture_scale = max(float(np.sqrt(np.mean(truth_v * truth_v))), 1.0)
    return {
        "texture_relative_rms_error": float(
            np.sqrt(np.mean((texture - truth_v) ** 2)) / texture_scale
        ),
        "cartoon_relative_rms_error": float(
            np.linalg.norm(cartoon - truth_u) / max(np.linalg.norm(truth_u), 1.0)
        ),
        "texture_gain": float(
            np.sum(texture * truth_v) / max(texture_norm2, 1e-30)
        ),
        "contour_texture_rms": float(
            np.sqrt(np.mean(np.asarray(texture)[contour] ** 2))
        ),
        "recomposition_linf": float(
            np.max(np.abs(np.asarray(scene["source"]) - cartoon - texture))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--learning-passes", type=int, default=4)
    parser.add_argument("--virtual-target", type=int, default=16)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--population-radius", type=int, default=4)
    parser.add_argument("--recurrence-radius", type=int, default=1)
    parser.add_argument("--gain-cap", type=float, default=6.0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    scenes = (
        pure_edge_scene(args.size),
        pure_carrier_scene(args.size),
        symmetric_support_scene(args.size),
        checker_support_scene(args.size),
        multiscale_crossing_scene(args.size),
        junction_texture_scene(args.size),
    )
    titles = (
        "source", "truth cartoon", "truth texture", "public cartoon",
        "public texture", "learned-current cartoon", "learned-current texture",
    )
    cell, header, label = 176, 30, 150
    canvas = Image.new(
        "RGB", (label + len(titles) * cell, header + len(scenes) * cell), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for column, title in enumerate(titles):
        draw.text((label + column * cell + 3, 8), title, fill="black")

    report: dict[str, object] = {
        "hypothesis": (
            "learn nonlinear ownership first; continue only populated, "
            "sign-consistent conservative current coordinates"
        ),
        "parameters": vars(args) | {"out": str(args.out)},
        "scenes": {},
    }
    arrays: dict[str, np.ndarray] = {}
    for row, scene in enumerate(scenes):
        name = str(scene["name"])
        source = np.asarray(scene["source"], dtype=np.float64)
        truth_u, truth_v = _truth(scene)
        public = bfft.MeyerPlan(
            source.shape, lam=0.05, mu=40.0, passes=1, threads=args.threads
        ).split(source)
        proposed = learned_current_jump(
            source,
            learning_passes=args.learning_passes,
            virtual_target=args.virtual_target,
            levels=args.levels,
            population_radius=args.population_radius,
            recurrence_radius=args.recurrence_radius,
            gain_cap=args.gain_cap,
            threads=args.threads,
        )
        public_score = _score(public[0], public[1], scene)
        proposed_score = _score(proposed[0], proposed[1], scene)
        report["scenes"][name] = {
            "public": public_score,
            "learned_current": proposed_score,
            "diagnostic": proposed[2],
        }
        scale = max(float(np.percentile(np.abs(truth_v), 99.5)), 1.0)
        panels = (
            _plain(source), _plain(truth_u), _signed(truth_v, scale),
            _plain(public[0]), _signed(public[1], scale),
            _plain(proposed[0]), _signed(proposed[1], scale),
        )
        y0 = header + row * cell
        draw.text((4, y0 + 8), name, fill="black")
        for column, panel in enumerate(panels):
            canvas.paste(
                panel.resize((cell, cell), Image.Resampling.NEAREST),
                (label + column * cell, y0),
            )
        for key, value in (
            ("source", source), ("truth_cartoon", truth_u),
            ("truth_texture", truth_v), ("public_cartoon", public[0]),
            ("public_texture", public[1]), ("learned_cartoon", proposed[0]),
            ("learned_texture", proposed[1]),
        ):
            arrays[f"{name}_{key}"] = np.asarray(value, dtype=np.float32)
        print(
            f"{name:22s} public={public_score['texture_relative_rms_error']:.4f} "
            f"learned={proposed_score['texture_relative_rms_error']:.4f} "
            f"edge={proposed_score['contour_texture_rms']:.3f}",
            flush=True,
        )

    canvas.save(args.out / "comparison.png")
    np.savez_compressed(args.out / "arrays.npz", **arrays)
    (args.out / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
