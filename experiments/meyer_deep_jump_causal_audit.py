#!/usr/bin/env python3
"""Causal audit and repair certificate for the fixed-cost Meyer deep jump.

The old scalar-complement form failed in two apparently opposite ways: a
perfect step generated a signed texture halo, while a perfect carrier left
its low resolvent tail in cartoon.  The common cause was deciding ownership
from isolated scalar samples after a global high-pass.

The repaired form keeps the two source observations, but treats their final
oscillation only as a proposal.  It admits the complete proposal where a
compact, scale-relative population has RMS current action above the existing
Meyer observation radius.  No accepted sample is shrunk.  This file retains
the two old oracle limits as falsification controls, then verifies that the
action-admitted Python construction agrees with the production C++ path.
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
from experiments.meyer_first_pass_conditioning import (  # noqa: E402
    lap_hat,
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
    _dilate,
    _gradient_magnitude,
    multiscale_crossing_scene,
    symmetric_support_scene,
)


OUT = ROOT / "experiments" / "out" / "meyer_deep_jump_causal_audit"


def _rms(value: np.ndarray, mask: np.ndarray | None = None) -> float:
    sample = np.asarray(value, dtype=np.float64)
    if mask is not None:
        sample = sample[np.asarray(mask, dtype=bool)]
    if not sample.size:
        return 0.0
    return float(np.sqrt(np.mean(sample * sample)))


def _relative_rms(value: np.ndarray, reference: np.ndarray) -> float:
    return _rms(value) / max(_rms(reference), 1e-30)


def _linear_gain(value: np.ndarray, reference: np.ndarray) -> float:
    value = np.asarray(value, dtype=np.float64).ravel()
    reference = np.asarray(reference, dtype=np.float64).ravel()
    return float(value @ reference / max(float(reference @ reference), 1e-30))


def _resolvents(
    shape: tuple[int, int], *, lam: float, virtual_passes: int
) -> tuple[np.ndarray, np.ndarray]:
    symbol = lap_hat(shape)
    hu = lam / (lam - 2.0 * lam * symbol)
    return hu, hu ** int(virtual_passes)


def pure_edge_scene(size: int) -> dict[str, object]:
    parent = symmetric_support_scene(size)
    cartoon = np.asarray(parent["cartoon"], dtype=np.float64)
    y, x = np.mgrid[:size, :size].astype(np.float64)
    smooth = 92.0 + 10.0 * x / size
    jump = cartoon - smooth
    contour = _dilate(_gradient_magnitude(cartoon) > 4.0, 5)
    return {
        "name": "pure_edge",
        "source": cartoon,
        "truth_cartoon": cartoon,
        "truth_texture": np.zeros_like(cartoon),
        "structural_jump": jump,
        "contour": contour,
    }


def pure_carrier_scene(size: int) -> dict[str, object]:
    y, x = np.mgrid[:size, :size].astype(np.float64)
    smooth = 96.0 + 8.0 * x / size - 5.0 * y / size
    x0, x1 = 0.16 * size, 0.87 * size
    y0, y1 = 0.18 * size, 0.84 * size
    distance = np.minimum.reduce((x - x0, x1 - x, y - y0, y1 - y))
    taper = np.clip((distance - 4.0) / 18.0, 0.0, 1.0)
    carrier = 18.0 * taper * (
        np.cos(2.0 * np.pi * (x + 0.35 * y) / 17.0 + 0.21)
        + 0.55 * np.cos(2.0 * np.pi * (x - 1.4 * y) / 7.0 - 0.43)
    )
    support = taper > 0.01
    return {
        "name": "pure_carrier",
        "source": smooth + carrier,
        "truth_cartoon": smooth,
        "truth_texture": carrier,
        "structural_jump": np.zeros_like(smooth),
        "contour": ~support,
    }


def canonical_scene(scene: dict) -> dict[str, object]:
    """Use ordinary cartoon/texture semantics, not the old halo-as-truth one."""
    if "hard_composition" in scene:
        cartoon = np.asarray(scene["hard_composition"], dtype=np.float64)
        texture = np.asarray(scene["material_texture"], dtype=np.float64)
        jump = np.asarray(scene["jump_potential"], dtype=np.float64)
    else:
        cartoon = np.asarray(scene["cartoon"], dtype=np.float64)
        texture = np.asarray(scene["texture"], dtype=np.float64)
        # Only the discontinuous component is needed by the ideal edge limit.
        smooth = np.fft.ifft2(
            np.fft.fft2(cartoon) * _resolvents(
                cartoon.shape, lam=0.05, virtual_passes=8
            )[0]
        ).real
        jump = cartoon - smooth
    return {
        "name": scene["name"],
        "source": cartoon + texture,
        "truth_cartoon": cartoon,
        "truth_texture": texture,
        "structural_jump": jump,
        "contour": np.asarray(scene["contour"], dtype=bool),
    }


def audit_scene(
    scene: dict[str, object], *, lam: float, mu: float, virtual_passes: int
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    source = np.asarray(scene["source"], dtype=np.float64)
    truth_cartoon = np.asarray(scene["truth_cartoon"], dtype=np.float64)
    truth_texture = np.asarray(scene["truth_texture"], dtype=np.float64)
    structural_jump = np.asarray(scene["structural_jump"], dtype=np.float64)
    contour = np.asarray(scene["contour"], dtype=bool)
    hu, huk = _resolvents(
        source.shape, lam=lam, virtual_passes=virtual_passes
    )

    # Oracle limits: neither result depends on a learned/estimated owner.
    ideal_boundary = np.fft.ifft2(
        (1.0 - hu) * np.fft.fft2(structural_jump)
    ).real
    forced_cartoon_carrier = np.fft.ifft2(
        huk * np.fft.fft2(truth_texture)
    ).real
    ideal_captured_carrier = truth_texture - forced_cartoon_carrier

    gate = native_structural_gate(source)
    proposed_material, jump_potential, jump_diagnostic = jump_texture_components(
        source, gate, lam=lam, virtual_passes=virtual_passes
    )
    forbidden_boundary = np.fft.ifft2(
        (1.0 - hu) * np.fft.fft2(jump_potential)
    ).real
    unadmitted_material, unadmitted_route = tangent_reservoir_route(
        proposed_material, gate, radius=mu
    )
    action_radius = max(1, min(source.shape) // 8)
    action_threshold = 1.0 / (4.0 * lam)
    action = np.sqrt(np.maximum(uniform_filter(
        proposed_material * proposed_material,
        size=(2 * action_radius + 1, 2 * action_radius + 1),
        mode="wrap",
    ), 0.0))
    action_authority = action > action_threshold
    admitted_material = np.where(action_authority, proposed_material, 0.0)
    routed_material, route_diagnostic = tangent_reservoir_route(
        admitted_material, gate, radius=mu
    )
    reconstructed_texture = routed_material
    reconstructed_cartoon = source - reconstructed_texture

    plan = bfft.MeyerPlan(
        source.shape, lam=lam, mu=mu, passes=1, threads=1
    )
    native_cartoon, native_texture = plan.split_jump_measure(
        source, virtual_passes=virtual_passes
    )
    component_delta = reconstructed_texture - native_texture
    texture_scale = max(_rms(truth_texture), 1.0)
    report = {
        "oracle_edge_limit": {
            "boundary_halo_rms": _rms(ideal_boundary),
            "boundary_halo_peak": float(np.max(np.abs(ideal_boundary))),
            "boundary_halo_rms_over_jump_rms": _relative_rms(
                ideal_boundary, structural_jump
            ),
            "boundary_halo_contour_rms": _rms(ideal_boundary, contour),
            "statement": (
                "Even a perfect jump estimate produces this nonzero signed "
                "texture term because the implementation applies (I-Hu)."
            ),
        },
        "oracle_carrier_limit": {
            "forced_cartoon_carrier_rms": _rms(forced_cartoon_carrier),
            "forced_cartoon_carrier_relative_rms": _relative_rms(
                forced_cartoon_carrier, truth_texture
            ),
            "forced_cartoon_carrier_gain": _linear_gain(
                forced_cartoon_carrier, truth_texture
            ),
            "captured_carrier_gain_before_capacity": _linear_gain(
                ideal_captured_carrier, truth_texture
            ),
            "statement": (
                "Even with a perfect zero jump estimate, Hu**K of the carrier "
                "is assigned to cartoon before the capacity projection."
            ),
        },
        "production_components": {
            "jump_potential_rms": _rms(jump_potential),
            "forbidden_boundary_component_rms": _rms(forbidden_boundary),
            "routed_material_component_rms": _rms(routed_material),
            "boundary_contour_rms_over_texture_scale": (
                _rms(forbidden_boundary, contour) / texture_scale
            ),
            "native_texture_relative_error": _relative_rms(
                native_texture - truth_texture, truth_texture
            ),
            "native_cartoon_relative_error": _relative_rms(
                native_cartoon - truth_cartoon, truth_cartoon
            ),
            "component_sum_native_rms": _rms(component_delta),
            "component_sum_native_linf": float(np.max(np.abs(component_delta))),
        },
        "current_action_admission": {
            "observer_radius": action_radius,
            "observer_side": 2 * action_radius + 1,
            "threshold": action_threshold,
            "active_fraction": float(np.mean(action_authority)),
            "action_mean": float(np.mean(action)),
            "action_p90": float(np.percentile(action, 90.0)),
            "action_maximum": float(np.max(action)),
            "unadmitted_texture_relative_error": _relative_rms(
                unadmitted_material - truth_texture, truth_texture
            ),
            "admitted_texture_relative_error": _relative_rms(
                routed_material - truth_texture, truth_texture
            ),
        },
        "jump_diagnostic": jump_diagnostic,
        "unadmitted_route_diagnostic": unadmitted_route,
        "route_diagnostic": route_diagnostic,
    }
    arrays = {
        "source": source,
        "truth_cartoon": truth_cartoon,
        "truth_texture": truth_texture,
        "ideal_boundary": ideal_boundary,
        "forced_cartoon_carrier": forced_cartoon_carrier,
        "jump_potential": jump_potential,
        "forbidden_boundary": forbidden_boundary,
        "unadmitted_material": unadmitted_material,
        "action": action,
        "action_authority": action_authority,
        "admitted_material": admitted_material,
        "routed_material": routed_material,
        "native_cartoon": native_cartoon,
        "native_texture": native_texture,
    }
    return report, arrays


def _plain(value: np.ndarray) -> Image.Image:
    shown = np.uint8(np.clip(value, 0.0, 255.0))
    return Image.fromarray(shown, mode="L").convert("RGB")


def _signed(value: np.ndarray, scale: float) -> Image.Image:
    shown = np.uint8(np.clip(127.5 + 127.5 * value / max(scale, 1e-30), 0, 255))
    return Image.fromarray(shown, mode="L").convert("RGB")


def render(rows: list[tuple[str, dict[str, np.ndarray]]], path: Path) -> None:
    titles = (
        "source", "truth texture", "old scalar limit", "raw proposal",
        "current action", "admitted proposal", "routed material",
        "native texture", "native cartoon",
    )
    cell, header, label_width = 176, 34, 156
    canvas = Image.new(
        "RGB", (label_width + len(titles) * cell, header + len(rows) * cell), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for column, title in enumerate(titles):
        draw.text((label_width + column * cell + 3, 8), title, fill="black")
    for row, (name, arrays) in enumerate(rows):
        y0 = header + row * cell
        draw.text((4, y0 + 7), name, fill="black")
        signed_values = (
            arrays["truth_texture"], arrays["ideal_boundary"],
            arrays["unadmitted_material"], arrays["admitted_material"],
            arrays["routed_material"], arrays["native_texture"],
        )
        scale = max(
            float(np.percentile(np.abs(value), 99.5)) for value in signed_values
        )
        panels = (
            _plain(arrays["source"]),
            _signed(arrays["truth_texture"], scale),
            _signed(
                arrays["ideal_boundary"] + arrays["forced_cartoon_carrier"],
                scale,
            ),
            _signed(arrays["unadmitted_material"], scale),
            _signed(
                arrays["action"],
                max(float(np.percentile(arrays["action"], 99.5)), 1.0),
            ),
            _signed(arrays["admitted_material"], scale),
            _signed(arrays["routed_material"], scale),
            _signed(arrays["native_texture"], scale),
            _plain(arrays["native_cartoon"]),
        )
        for column, panel in enumerate(panels):
            canvas.paste(
                panel.resize((cell, cell)), (label_width + column * cell, y0)
            )
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--lam", type=float, default=0.05)
    parser.add_argument("--mu", type=float, default=40.0)
    parser.add_argument("--virtual-passes", type=int, default=12)
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
    report: dict[str, object] = {
        "method": "native fixed-cost Meyer deep-jump causal component audit",
        "parameters": {
            "size": args.size,
            "lambda": args.lam,
            "mu": args.mu,
            "virtual_passes": args.virtual_passes,
        },
        "falsification_contract": {
            "pure_edge": "texture must be identically zero up to numerical error",
            "pure_carrier": "cartoon must contain no component correlated with carrier",
            "mixed": "coherent hard jumps remain wholly cartoon",
        },
        "scenes": {},
    }
    rows = []
    archive: dict[str, np.ndarray] = {}
    for scene in scenes:
        scene_report, arrays = audit_scene(
            scene, lam=args.lam, mu=args.mu, virtual_passes=args.virtual_passes
        )
        name = str(scene["name"])
        report["scenes"][name] = scene_report
        rows.append((name, arrays))
        for label, value in arrays.items():
            archive[f"{name}_{label}"] = np.asarray(value, dtype=np.float32)
        production = scene_report["production_components"]
        print(
            f"{name:20s} forbidden-boundary="
            f"{production['forbidden_boundary_component_rms']:.4f} "
            f"routed={production['routed_material_component_rms']:.4f} "
            f"native-agreement={production['component_sum_native_linf']:.3e}"
        )

    (args.out / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(args.out / "components.npz", **archive)
    render(rows, args.out / "component_atlas.png")


if __name__ == "__main__":
    main()
