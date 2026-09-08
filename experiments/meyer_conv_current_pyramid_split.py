#!/usr/bin/env python3
"""Surgical replacement study for the scalar deep-jump Meyer path.

The native jump path first integrates an oriented jump observation to a scalar
potential and then applies global scalar spectral complements.  This probe
keeps decomposition in conservative CONV current coordinates instead.  Each
dyadic level has an exact coarse mean plus three compact child moments.  Only
moments owned by oscillatory current evidence are assigned to texture; the
remaining state is reconstructed as cartoon.  No scalar jump high-pass or
post-hoc boundary complement is used.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.conv_conservative_multiresolution import (
    ConservativeBlockState,
    conservative_analysis_2d,
    conservative_synthesis_2d,
)
from experiments.meyer_conv_second_transport_split import (
    _conv_current_alternation,
)
from experiments.meyer_first_pass_conditioning import checker_support_scene
from experiments.meyer_preconditioning_research import junction_texture_scene
from experiments.meyer_transverse_route_research import native_structural_gate
from experiments.meyer_tsv_validation import (
    multiscale_crossing_scene,
    score_split,
    symmetric_support_scene,
)


OUT = ROOT / "experiments" / "out" / "meyer_conv_current_pyramid_split"


def _otsu_confidence(value: np.ndarray, bins: int = 256) -> np.ndarray:
    """Map a source statistic to its data-derived high-population confidence."""

    sample = np.asarray(value, dtype=np.float64)
    maximum = float(np.max(sample))
    if not maximum > 0.0:
        return np.zeros_like(sample)
    histogram, edges = np.histogram(sample, bins=bins, range=(0.0, maximum))
    centres = 0.5 * (edges[:-1] + edges[1:])
    mass = histogram.astype(np.float64)
    total = float(np.sum(mass))
    cumulative = np.cumsum(mass)
    moment = np.cumsum(mass * centres)
    global_moment = float(moment[-1])
    left = cumulative
    right = total - cumulative
    valid = (left > 0.0) & (right > 0.0)
    between = np.zeros_like(centres)
    between[valid] = (
        (global_moment * left[valid] - moment[valid] * total) ** 2
        / (left[valid] * right[valid])
    )
    index = int(np.argmax(between))
    boundary = float(edges[index + 1])
    high = sample[sample >= boundary]
    high_mean = float(np.mean(high)) if high.size else maximum
    span = max(high_mean - boundary, np.finfo(float).eps * maximum)
    return np.clip((sample - boundary) / span, 0.0, 1.0)


def _block_reduce(value: np.ndarray, mode: str) -> np.ndarray:
    blocks = np.stack((
        value[0::2, 0::2], value[0::2, 1::2],
        value[1::2, 0::2], value[1::2, 1::2],
    ))
    if mode == "mean":
        return np.mean(blocks, axis=0)
    if mode == "max":
        return np.max(blocks, axis=0)
    raise ValueError("block reduction must be 'mean' or 'max'")


@dataclass(frozen=True)
class OwnedLevel:
    horizontal: np.ndarray
    vertical: np.ndarray
    mixed: np.ndarray
    owner: np.ndarray


def current_owner(source: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return data-derived structural and alternating-current confidence.

    The structural complement owns ordinary material interiors.  The Otsu
    high class of the six-node current-alternation statistic restores a
    carrier where it crosses a structural front.  A monotone jump has no
    alternating-current class, so it cannot create a texture ring.
    """

    structural = _otsu_confidence(native_structural_gate(source))
    alternation = _otsu_confidence(_conv_current_alternation(source))
    return structural, alternation


def conv_current_pyramid_split(
    image: np.ndarray,
    *,
    levels: int = 4,
    gate_power: int = 8,
    block_reduction: str = "mean",
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Exact two-product split in compact conservative current coordinates."""

    source = np.asarray(image, dtype=np.float64)
    current = source
    owned: list[OwnedLevel] = []
    level_diagnostics = []
    for level in range(int(levels)):
        if min(current.shape) < 10 or current.shape[0] % 2 or current.shape[1] % 2:
            break
        state: ConservativeBlockState = conservative_analysis_2d(current)
        structural, alternation = current_owner(current)
        # One witnessed structural pixel owns the whole compact child block;
        # this removes the support-broadening that created the scalar annulus.
        # Alternation can return that block to texture only as an independent
        # current observation, which is what carries a real front-crossing
        # carrier through the structural exclusion.
        structural_block = _block_reduce(structural, "max")
        alternation_block = _block_reduce(alternation, block_reduction)
        owner = np.maximum(1.0 - structural_block, alternation_block)
        owned.append(OwnedLevel(
            owner * state.horizontal_detail,
            owner * state.vertical_detail,
            owner * state.mixed_detail,
            owner,
        ))
        level_diagnostics.append({
            "level": level + 1,
            "height": int(current.shape[0]),
            "width": int(current.shape[1]),
            "owner_mean": float(np.mean(owner)),
            "owner_p90": float(np.percentile(owner, 90.0)),
            "owner_maximum": float(np.max(owner)),
            "detail_rms": float(np.sqrt(np.mean(
                state.horizontal_detail ** 2
                + state.vertical_detail ** 2
                + state.mixed_detail ** 2
            ))),
        })
        current = state.coarse

    texture = np.zeros_like(current)
    for level in reversed(owned):
        texture = conservative_synthesis_2d(
            texture, level.horizontal, level.vertical, level.mixed
        )
    cartoon = source - texture
    return cartoon, texture, {
        "levels": len(owned),
        "gate_power": int(gate_power),
        "block_reduction": block_reduction,
        "level_diagnostics": level_diagnostics,
        "texture_rms": float(np.sqrt(np.mean(texture * texture))),
        "recomposition_linf": float(np.max(np.abs(source - cartoon - texture))),
    }


def _display_gray(value: np.ndarray) -> Image.Image:
    return Image.fromarray(np.uint8(np.clip(value, 0.0, 255.0)), mode="L").convert("RGB")


def _display_signed(value: np.ndarray, scale: float) -> Image.Image:
    shown = np.clip(127.5 + 127.5 * value / max(scale, 1e-30), 0.0, 255.0)
    return Image.fromarray(np.uint8(shown), mode="L").convert("RGB")


def render(rows: list[tuple], path: Path) -> None:
    titles = ("source", "truth cartoon", "truth texture", "current deep cartoon",
              "current deep texture", "CONV-current cartoon", "CONV-current texture")
    cell, header, label_width = 192, 28, 150
    canvas = Image.new("RGB", (label_width + len(titles) * cell,
                               header + len(rows) * cell), "white")
    draw = ImageDraw.Draw(canvas)
    for index, title in enumerate(titles):
        draw.text((label_width + index * cell + 3, 7), title, fill="black")
    for row_index, row in enumerate(rows):
        name, source, truth_u, truth_v, deep_u, deep_v, conv_u, conv_v = row
        y0 = header + row_index * cell
        draw.text((4, y0 + 8), name, fill="black")
        scale = max(float(np.percentile(np.abs(truth_v), 99.5)), 1.0)
        panels = (
            _display_gray(source), _display_gray(truth_u), _display_signed(truth_v, scale),
            _display_gray(deep_u), _display_signed(deep_v, scale),
            _display_gray(conv_u), _display_signed(conv_v, scale),
        )
        for column, panel in enumerate(panels):
            canvas.paste(panel.resize((cell, cell)), (label_width + column * cell, y0))
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--block-reduction", choices=("mean", "max"), default="mean")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    scenes = (
        symmetric_support_scene(args.size),
        checker_support_scene(args.size),
        multiscale_crossing_scene(args.size),
        junction_texture_scene(args.size),
    )
    native_arrays_path = (
        ROOT / "paper" / "fast_meyer_bregman" / "results" / "intrinsic"
        / "intrinsic_arrays.npz"
    )
    native = np.load(native_arrays_path)
    report = {
        "hypothesis": (
            "scalar jump integration followed by global spectral complements "
            "jointly causes boundary ringing and carrier retention"
        ),
        "replacement": "compact conservative CONV current pyramid",
        "scenes": {},
    }
    arrays: dict[str, np.ndarray] = {}
    rows = []
    for scene in scenes:
        name = scene["name"]
        source = np.asarray(scene["source"], dtype=np.float64)
        # The surgical contract keeps coherent jumps in cartoon.  Earlier
        # deep-jump experiments redefined part of each jump as truth texture;
        # that made the two-lobed boundary response score as correct and is
        # precisely the assumption under examination here.
        truth_scene = dict(scene)
        if "hard_composition" in scene:
            truth_scene["cartoon"] = scene["hard_composition"]
            truth_scene["texture"] = scene["material_texture"]
        deep = (
            np.asarray(native[f"{name}_deep_jump_default_cartoon"], dtype=np.float64),
            np.asarray(native[f"{name}_deep_jump_default_texture"], dtype=np.float64),
        )
        conv_u, conv_v, diagnostic = conv_current_pyramid_split(
            source,
            levels=args.levels,
            block_reduction=args.block_reduction,
        )
        report["scenes"][name] = {
            "truth_contract": (
                "coherent jumps remain wholly in cartoon; texture contains "
                "only authored material oscillation"
            ),
            "current_deep": score_split(*deep, truth_scene),
            "conv_current": score_split(conv_u, conv_v, truth_scene),
            "conv_current_diagnostic": diagnostic,
        }
        rows.append((name, source, truth_scene["cartoon"], truth_scene["texture"],
                     deep[0], deep[1], conv_u, conv_v))
        for label, value in (
            ("source", source), ("truth_cartoon", truth_scene["cartoon"]),
            ("truth_texture", truth_scene["texture"]),
            ("deep_cartoon", deep[0]), ("deep_texture", deep[1]),
            ("conv_cartoon", conv_u), ("conv_texture", conv_v),
        ):
            arrays[f"{name}_{label}"] = np.asarray(value, dtype=np.float32)
        print(name)
        for method in ("current_deep", "conv_current"):
            score = report["scenes"][name][method]
            print(
                f"  {method:14s} texture {score['texture_relative_rms_error']:.4f} "
                f"cartoon {score['cartoon_relative_rms_error']:.4f} "
                f"contour {score['contour_excess_texture_rms']:.4f} "
                f"AUC {score['texture_over_contour_allocation_auc']:.4f}"
            )

    (args.out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(args.out / "arrays.npz", **arrays)
    render(rows, args.out / "comparison.png")


if __name__ == "__main__":
    main()
