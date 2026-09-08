#!/usr/bin/env python3
"""Ablate soft scalar shrinkage inside the Meyer jump-current observation.

The current public deep jump observes structural bonds with two multiplicative
soft factors: a ramp from the Otsu structural population and a nonnegative
garrote in gradient magnitude.  On a true jump both factors underestimate the
observed current, leaving a step residual for the subsequent virtual-depth
high-pass to turn into a ring.  This probe asks the more precise question:
once the first carrier observation has removed oscillatory current, should the
second structural observation retain the *complete witnessed bond current*?

All arms keep the same data-derived population partition, two observations,
Hodge integration, G-capacity route, and exact two-product recomposition.
Only the second-observation amplitude law changes.  The synthetic controls are
the authority; no natural-image score or Gilles similarity selects an arm.
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
from experiments.conv_distilled_core import distilled_conv_resize  # noqa: E402
from experiments.meyer_first_pass_conditioning import (  # noqa: E402
    checker_support_scene,
    lap_hat,
)
from experiments.meyer_preconditioning_research import (  # noqa: E402
    junction_texture_scene,
)
from experiments.meyer_transverse_route_research import (  # noqa: E402
    divergence,
    native_structural_gate,
    tangent_reservoir_route,
)
from experiments.meyer_tsv_validation import (  # noqa: E402
    multiscale_crossing_scene,
    symmetric_support_scene,
)


OUT = ROOT / "experiments" / "out" / "meyer_jump_measure_support_ablation"


def otsu_population(gate: np.ndarray) -> tuple[float, float, np.ndarray]:
    histogram, edges = np.histogram(gate, bins=256, range=(0.0, 1.0))
    centres = 0.5 * (edges[:-1] + edges[1:])
    mass = histogram.astype(np.float64)
    mass /= max(float(np.sum(mass)), 1.0)
    cumulative = np.cumsum(mass)
    moment = np.cumsum(mass * centres)
    total_moment = float(moment[-1])
    between = (
        (total_moment * cumulative - moment) ** 2
        / np.maximum(cumulative * (1.0 - cumulative), 1e-30)
    )
    split = int(np.argmax(between[:-1]))
    boundary = float(edges[split + 1])
    high_mass = max(1.0 - cumulative[split], 1e-30)
    high_mean = float(np.sum(mass[split + 1:] * centres[split + 1:]) / high_mass)
    ramp = np.clip(
        (gate - boundary) / max(high_mean - boundary, 1e-30), 0.0, 1.0
    )
    return boundary, high_mean, ramp


def observe_bonds(
    value: np.ndarray,
    confidence: np.ndarray,
    threshold: float,
    amplitude_law: str,
) -> tuple[np.ndarray, np.ndarray]:
    gx = np.roll(value, -1, axis=1) - value
    gy = np.roll(value, -1, axis=0) - value
    magnitude = np.hypot(gx, gy)
    if amplitude_law == "garrote":
        activation = np.maximum(
            1.0 - (threshold / np.maximum(magnitude, 1e-30)) ** 2, 0.0
        )
    elif amplitude_law == "hard":
        activation = magnitude > threshold
    elif amplitude_law == "root_garrote":
        activation = np.sqrt(np.maximum(
            1.0 - (threshold / np.maximum(magnitude, 1e-30)) ** 2, 0.0
        ))
    elif amplitude_law == "full":
        activation = np.ones_like(magnitude)
    else:
        raise ValueError(f"unknown amplitude law {amplitude_law!r}")
    weight = confidence * activation
    return weight * gx, weight * gy


def conv_persistent_current(
    value: np.ndarray, gradient_scale: float, *, amplitude_weight: bool = True
) -> np.ndarray:
    """Dyadically persistent oriented-current confidence.

    The returned field is not an edge-magnitude detector.  It compares the
    fine bond current with the current of its conservative half-scale
    prediction.  The factor two is the geometric gradient rescaling of a
    dyadic restriction; the ``conv4`` arm is retained only as a falsification
    control for a broader current reservoir.
    """

    source = np.asarray(value, dtype=np.float64)
    coarse = distilled_conv_resize(
        source, (max(5, source.shape[0] // 2), max(5, source.shape[1] // 2))
    )
    predicted = distilled_conv_resize(coarse, source.shape)
    gx = np.roll(source, -1, axis=1) - source
    gy = np.roll(source, -1, axis=0) - source
    px = np.roll(predicted, -1, axis=1) - predicted
    py = np.roll(predicted, -1, axis=0) - predicted
    fine_magnitude = np.hypot(gx, gy)
    predicted_magnitude = float(gradient_scale) * np.hypot(px, py)
    agreement = np.clip(
        np.divide(
            gx * px + gy * py,
            fine_magnitude * np.hypot(px, py) + 1e-30,
            out=np.zeros_like(source),
            where=(fine_magnitude > 0.0) & (np.hypot(px, py) > 0.0),
        ),
        0.0,
        1.0,
    )
    amplitude = np.divide(
        2.0 * np.minimum(fine_magnitude, predicted_magnitude),
        fine_magnitude + predicted_magnitude + 1e-30,
        out=np.zeros_like(source),
        where=(fine_magnitude + predicted_magnitude) > 0.0,
    )
    if not amplitude_weight:
        return agreement
    return np.clip(agreement * amplitude, 0.0, 1.0)


def integrate_bonds(
    flux_x: np.ndarray, flux_y: np.ndarray, safe_laplacian: np.ndarray
) -> np.ndarray:
    spectrum = np.fft.fft2(divergence(flux_x, flux_y)) / safe_laplacian
    spectrum[0, 0] = 0.0
    return spectrum


def split(
    source: np.ndarray,
    *,
    virtual_depth: int,
    confidence_law: str,
    amplitude_law: str,
    mu: float = 40.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    image = np.asarray(source, dtype=np.float64)
    gate = native_structural_gate(image)
    boundary, high_mean, ramp = otsu_population(gate)
    if confidence_law == "ramp":
        confidence = ramp
    elif confidence_law == "population":
        confidence = (gate >= boundary).astype(np.float64)
    elif confidence_law in ("conv2", "conv4", "conv_align"):
        # The current is observed after the first carrier removal below.
        confidence = None
    else:
        raise ValueError(f"unknown confidence law {confidence_law!r}")

    lam = 0.05
    eta = 2.0 * lam
    laplacian = lap_hat(image.shape)
    safe_laplacian = np.where(np.abs(laplacian) > 1e-15, laplacian, 1.0)
    transfer = lam / (lam - eta * laplacian)
    highpass = 1.0 - transfer ** int(virtual_depth)
    spectrum = np.fft.fft2(image)

    # The first observation remains deliberately soft: its job is only to
    # learn a carrier seed without asserting structural ownership.
    first_x, first_y = observe_bonds(image, ramp, 1.0 / (4.0 * lam), "garrote")
    first_jump = integrate_bonds(first_x, first_y, safe_laplacian)
    carrier_seed = np.fft.ifft2(highpass * (spectrum - first_jump)).real

    resident = image - carrier_seed
    if confidence_law == "conv2":
        confidence = conv_persistent_current(resident, 2.0)
    elif confidence_law == "conv4":
        confidence = conv_persistent_current(resident, 4.0)
    elif confidence_law == "conv_align":
        confidence = conv_persistent_current(
            resident, 2.0, amplitude_weight=False
        )
    assert confidence is not None
    second_x, second_y = observe_bonds(
        resident, confidence, 1.0 / (2.0 * lam), amplitude_law
    )
    jump = integrate_bonds(second_x, second_y, safe_laplacian)
    proposal = np.fft.ifft2(highpass * (spectrum - jump)).real
    texture, route = tangent_reservoir_route(proposal, gate, radius=float(mu))
    cartoon = image - texture
    return cartoon, texture, {
        "support_boundary": boundary,
        "support_high_mean": high_mean,
        "carrier_seed_rms": float(np.sqrt(np.mean(carrier_seed * carrier_seed))),
        "proposal_rms": float(np.sqrt(np.mean(proposal * proposal))),
        "texture_rms": float(np.sqrt(np.mean(texture * texture))),
        "route_alpha": float(route["alpha"]),
        "recomposition_linf": float(np.max(np.abs(image - cartoon - texture))),
    }


def canonical_truth(scene: dict) -> tuple[np.ndarray, np.ndarray]:
    if "truth_cartoon" in scene:
        return (
            np.asarray(scene["truth_cartoon"], dtype=np.float64),
            np.asarray(scene["truth_texture"], dtype=np.float64),
        )
    if "hard_composition" in scene:
        return (
            np.asarray(scene["hard_composition"], dtype=np.float64),
            np.asarray(scene["material_texture"], dtype=np.float64),
        )
    return (
        np.asarray(scene["cartoon"], dtype=np.float64),
        np.asarray(scene["texture"], dtype=np.float64),
    )


def score(cartoon: np.ndarray, texture: np.ndarray, scene: dict) -> dict[str, float]:
    truth_u, truth_v = canonical_truth(scene)
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
        (depth, confidence, amplitude)
        for depth in (12, 16, 20)
        for confidence, amplitude in (
            ("ramp", "garrote"),
            ("ramp", "root_garrote"),
            ("ramp", "hard"),
            ("population", "garrote"),
            ("population", "root_garrote"),
            ("population", "hard"),
            ("conv2", "garrote"),
            ("conv2", "hard"),
            ("conv2", "full"),
            ("conv4", "garrote"),
            ("conv4", "hard"),
            ("conv4", "full"),
            ("conv_align", "garrote"),
            ("conv_align", "hard"),
            ("conv_align", "full"),
        )
    )
    report: dict[str, object] = {"arms": {}, "ranking": []}
    outputs: dict[tuple[str, tuple], tuple[np.ndarray, np.ndarray]] = {}
    for arm in arms:
        depth, confidence, amplitude = arm
        key = f"K{depth}_{confidence}_{amplitude}"
        rows = {}
        for scene in scenes:
            result = split(
                scene["source"], virtual_depth=depth,
                confidence_law=confidence, amplitude_law=amplitude,
            )
            outputs[(scene["name"], arm)] = result[:2]
            rows[scene["name"]] = {**score(result[0], result[1], scene), **result[2]}
        # Pure edge is a hard veto, not a small weighted term.  Among arms
        # under that control, rank the five material scenes by mean error.
        edge = rows["pure_edge"]["contour_texture_rms"]
        material_error = float(np.mean([
            rows[scene["name"]]["texture_relative_rms_error"]
            for scene in scenes[1:]
        ]))
        objective = edge + 5.0 * material_error
        report["arms"][key] = {
            "virtual_depth": depth,
            "confidence_law": confidence,
            "amplitude_law": amplitude,
            "objective": objective,
            "scenes": rows,
        }
        report["ranking"].append({"key": key, "objective": objective})
        print(f"{key:34s} edge={edge:.4f} material={material_error:.4f}")
    report["ranking"].sort(key=lambda item: item["objective"])

    selected_keys = [item["key"] for item in report["ranking"][:4]]
    arm_by_key = {
        f"K{depth}_{confidence}_{amplitude}": (depth, confidence, amplitude)
        for depth, confidence, amplitude in arms
    }
    titles = ("source", "truth cartoon", "truth texture", *selected_keys)
    cell, header, label = 160, 30, 150
    canvas = Image.new(
        "RGB", (label + len(titles) * cell, header + len(scenes) * cell), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for column, title in enumerate(titles):
        draw.text((label + column * cell + 3, 8), title, fill="black")
    for row, scene in enumerate(scenes):
        truth_u, truth_v = canonical_truth(scene)
        scale = max(float(np.percentile(np.abs(truth_v), 99.5)), 1.0)
        panels = [plain(scene["source"]), plain(truth_u), signed(truth_v, scale)]
        for key in selected_keys:
            _u, texture = outputs[(scene["name"], arm_by_key[key])]
            panels.append(signed(texture, scale))
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
