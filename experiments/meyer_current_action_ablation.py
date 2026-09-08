#!/usr/bin/env python3
"""Compact current-action admission after the two-observation deep proposal.

The two-observation jump estimator leaves a low-amplitude contour response but
already observes the material carrier at substantially larger sustained
action.  This probe does not classify source gradients.  It measures only the
compact RMS action of the proposed texture current and admits a complete
proposal where that action exceeds one of the two existing Meyer reflection
radii, ``1/(8 lambda)`` or ``1/(4 lambda)``.  Hard admission preserves phase;
garrote arms are retained as falsification controls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import uniform_filter


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bfft  # noqa: E402
from experiments.meyer_capacity_route_ablation import (  # noqa: E402
    plain,
    score,
    signed,
    truth,
)
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


OUT = ROOT / "experiments" / "out" / "meyer_current_action_ablation"


def periodic_box_mean(value: np.ndarray, radius: int) -> np.ndarray:
    source = np.asarray(value, dtype=np.float64)
    width = 2 * int(radius) + 1
    return uniform_filter(source, size=(width, width), mode="wrap")


def admit_current_action(
    proposal: np.ndarray,
    *,
    radius: int,
    threshold: float,
    law: str,
) -> tuple[np.ndarray, dict[str, float]]:
    value = np.asarray(proposal, dtype=np.float64)
    action = np.sqrt(np.maximum(
        periodic_box_mean(value * value, radius), 0.0
    ))
    if law == "hard":
        authority = (action > threshold).astype(np.float64)
    elif law == "garrote":
        authority = np.maximum(
            1.0 - (threshold / np.maximum(action, 1e-30)) ** 2, 0.0
        )
    elif law == "root_garrote":
        authority = np.sqrt(np.maximum(
            1.0 - (threshold / np.maximum(action, 1e-30)) ** 2, 0.0
        ))
    else:
        raise ValueError(f"unknown action law {law!r}")
    return authority * value, {
        "action_mean": float(np.mean(action)),
        "action_p90": float(np.percentile(action, 90.0)),
        "action_maximum": float(np.max(action)),
        "authority_mean": float(np.mean(authority)),
        "authority_active_fraction": float(np.mean(authority > 0.0)),
    }


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
        (depth, radius, threshold, law)
        for depth in (12, 16, 20)
        for radius in (1, 2, 3, 4)
        for threshold in (2.5, 5.0)
        for law in ("hard", "root_garrote", "garrote")
    )
    gates = {scene["name"]: native_structural_gate(scene["source"]) for scene in scenes}
    proposals = {
        (scene["name"], depth): jump_texture_components(
            scene["source"], gates[scene["name"]], lam=0.05,
            virtual_passes=depth,
        )[0]
        for scene in scenes for depth in (12, 16, 20)
    }
    report: dict[str, object] = {"arms": {}, "ranking": []}
    outputs = {}
    for depth, radius, threshold, law in arms:
        key = f"K{depth}_R{radius}_T{threshold:g}_{law}"
        rows = {}
        for scene in scenes:
            admitted, action = admit_current_action(
                proposals[(scene["name"], depth)], radius=radius,
                threshold=threshold, law=law,
            )
            texture, route = tangent_reservoir_route(
                admitted, gates[scene["name"]], radius=40.0
            )
            cartoon = np.asarray(scene["source"]) - texture
            outputs[(scene["name"], key)] = (cartoon, texture)
            rows[scene["name"]] = {
                **score(cartoon, texture, scene),
                "action": action,
                "route_alpha": float(route["alpha"]),
            }
        edge = rows["pure_edge"]["contour_texture_rms"]
        material = float(np.mean([
            rows[scene["name"]]["texture_relative_rms_error"] for scene in scenes[1:]
        ]))
        crossing = float(np.mean([
            rows[name]["texture_relative_rms_error"]
            for name in ("multiscale_crossing", "junction_texture")
        ]))
        objective = 4.0 * edge + 4.0 * material + 2.0 * crossing
        report["arms"][key] = {
            "virtual_depth": depth, "action_radius": radius,
            "action_threshold": threshold, "action_law": law,
            "objective": objective, "scenes": rows,
        }
        report["ranking"].append({"key": key, "objective": objective})
        print(
            f"{key:31s} edge={edge:.4f} material={material:.4f} "
            f"cross={crossing:.4f}"
        )
    report["ranking"].sort(key=lambda row: row["objective"])
    selected = [row["key"] for row in report["ranking"][:6]]
    titles = ("source", "truth cartoon", "truth texture", "public texture", *selected)
    cell, header, label = 144, 30, 150
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
