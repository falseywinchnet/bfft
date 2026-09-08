#!/usr/bin/env python3
"""Surgical ablation of the deep-jump transverse G-capacity route.

The revised two-observation jump estimator produces a substantially complete
carrier before the final G-ball readout.  The current native route multiplies
its divergence-free correction by ``1 - structural_gate``.  That is suspect:
a divergence-free correction cannot change the proposed texture, but refusing
to route current at a structural crossing can make the later pointwise disk
projection reject a real carrier there.  The rejected part necessarily stays
in cartoon.

This probe keeps the scalar proposal fixed and varies only the conservative
capacity route: structural suppression, and a bounded number of transverse
Newton corrections before the single final disk projection.  Pure edge is a
hard veto; carrier, supported carrier, crossing, and junction truth decide the
remaining frontier.
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

from experiments.meyer_deep_jump_causal_audit import (  # noqa: E402
    pure_carrier_scene,
    pure_edge_scene,
)
from experiments.meyer_first_pass_conditioning import checker_support_scene  # noqa: E402
from experiments.meyer_preconditioning_research import junction_texture_scene  # noqa: E402
from experiments.meyer_transverse_route_research import (  # noqa: E402
    disk_readout,
    divergence,
    jump_texture_components,
    longitudinal_flux,
    native_structural_gate,
    transverse_projection,
)
from experiments.meyer_tsv_validation import (  # noqa: E402
    multiscale_crossing_scene,
    symmetric_support_scene,
)


OUT = ROOT / "experiments" / "out" / "meyer_capacity_route_ablation"


def iterative_route(
    proposed: np.ndarray,
    gate: np.ndarray,
    *,
    radius: float,
    cycles: int,
    suppression: str,
    slack_fraction: float = 0.20,
) -> tuple[np.ndarray, dict[str, object]]:
    px, py = longitudinal_flux(proposed)
    initial_x, initial_y = px.copy(), py.copy()
    gate = np.asarray(gate, dtype=np.float64)
    if suppression == "native":
        authority = 1.0 - gate
    elif suppression == "root":
        authority = np.sqrt(1.0 - gate)
    elif suppression == "none":
        authority = np.ones_like(gate)
    else:
        raise ValueError(f"unknown suppression {suppression!r}")

    history = []
    for cycle in range(int(cycles)):
        magnitude = np.hypot(px, py)
        active = magnitude > radius
        before_energy = float(np.sum(np.maximum(magnitude - radius, 0.0) ** 2))
        if not before_energy > 0.0:
            history.append({
                "cycle": cycle + 1,
                "alpha": 0.0,
                "overload_energy_before": 0.0,
                "overload_energy_after": 0.0,
                "accepted": False,
            })
            break
        inverse = 1.0 / np.maximum(magnitude, 1e-30)
        normal_x, normal_y = px * inverse, py * inverse
        demand = np.where(
            active, radius - magnitude,
            slack_fraction * (radius - magnitude),
        )
        source_x = 2.0 * demand * normal_x * authority
        source_y = 2.0 * demand * normal_y * authority
        qx, qy = transverse_projection(source_x, source_y)
        radial = normal_x * qx + normal_y * qy
        q2 = qx * qx + qy * qy
        excess = magnitude - radius
        first = 2.0 * float(np.sum(excess[active] * radial[active]))
        second = 2.0 * float(np.sum(
            radial[active] ** 2
            + excess[active] / magnitude[active]
            * (q2[active] - radial[active] ** 2)
        ))
        alpha = float(np.clip(-first / max(second, 1e-30), 0.0, 2.0))
        candidate_x = px + alpha * qx
        candidate_y = py + alpha * qy
        after_energy = float(np.sum(np.maximum(
            np.hypot(candidate_x, candidate_y) - radius, 0.0
        ) ** 2))
        accepted = bool(after_energy < before_energy)
        if accepted:
            px, py = candidate_x, candidate_y
        else:
            alpha = 0.0
            after_energy = before_energy
        history.append({
            "cycle": cycle + 1,
            "alpha": alpha,
            "overload_energy_before": before_energy,
            "overload_energy_after": after_energy,
            "energy_ratio": after_energy / max(before_energy, 1e-30),
            "active_fraction": float(np.mean(active)),
            "accepted": accepted,
            "divergence_change": float(np.linalg.norm(divergence(qx, qy))),
        })

    baseline, _, _ = disk_readout(initial_x, initial_y, radius)
    texture, feasible_x, feasible_y = disk_readout(px, py, radius)
    return texture, {
        "cycles_requested": int(cycles),
        "suppression": suppression,
        "history": history,
        "baseline_loss": float(np.linalg.norm(baseline - proposed)),
        "routed_loss": float(np.linalg.norm(texture - proposed)),
        "postprojection_maximum": float(np.max(np.hypot(feasible_x, feasible_y))),
        "preprojection_divergence_error": float(np.linalg.norm(
            divergence(px, py) - proposed
        )),
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
    truth_rms = float(np.sqrt(np.mean(truth_v * truth_v)))
    return {
        "texture_relative_rms_error": float(
            np.sqrt(np.mean((texture - truth_v) ** 2)) / max(truth_rms, 1.0)
        ),
        "texture_gain": float(
            np.sum(texture * truth_v) / max(float(np.sum(truth_v * truth_v)), 1e-30)
        ),
        "contour_texture_rms": float(np.sqrt(np.mean(texture[contour] ** 2))),
        "cartoon_relative_rms_error": float(
            np.linalg.norm(cartoon - truth_u) / max(np.linalg.norm(truth_u), 1.0)
        ),
    }


def signed(value: np.ndarray, scale: float) -> Image.Image:
    shown = np.clip(127.5 + 127.5 * value / max(scale, 1e-30), 0.0, 255.0)
    return Image.fromarray(np.uint8(shown), mode="L").convert("RGB")


def plain(value: np.ndarray) -> Image.Image:
    return Image.fromarray(np.uint8(np.clip(value, 0.0, 255.0)), mode="L").convert("RGB")


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
        (depth, suppression, cycles)
        for depth in (12, 16, 20)
        for suppression in ("native", "root", "none")
        for cycles in (1, 2, 3)
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
    for depth, suppression, cycles in arms:
        key = f"K{depth}_{suppression}_C{cycles}"
        rows = {}
        for scene in scenes:
            texture, diagnostic = iterative_route(
                proposals[(scene["name"], depth)], gates[scene["name"]],
                radius=40.0, cycles=cycles, suppression=suppression,
            )
            cartoon = np.asarray(scene["source"]) - texture
            outputs[(scene["name"], key)] = (cartoon, texture)
            rows[scene["name"]] = {
                **score(cartoon, texture, scene), "route": diagnostic,
            }
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
            "virtual_depth": depth, "suppression": suppression,
            "cycles": cycles, "objective": objective, "scenes": rows,
        }
        report["ranking"].append({"key": key, "objective": objective})
        print(
            f"{key:20s} edge={edge:.4f} material={material:.4f} "
            f"cross={crossing:.4f}"
        )
    report["ranking"].sort(key=lambda row: row["objective"])
    selected = [row["key"] for row in report["ranking"][:5]]
    titles = ("source", "truth cartoon", "truth texture", *selected)
    cell, header, label = 160, 30, 150
    canvas = Image.new(
        "RGB", (label + len(titles) * cell, header + len(scenes) * cell), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for column, title in enumerate(titles):
        draw.text((label + column * cell + 3, 8), title, fill="black")
    for row, scene in enumerate(scenes):
        truth_u, truth_v = truth(scene)
        scale = max(float(np.percentile(np.abs(truth_v), 99.5)), 1.0)
        panels = [plain(scene["source"]), plain(truth_u), signed(truth_v, scale)]
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
