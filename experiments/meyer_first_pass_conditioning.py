#!/usr/bin/env python3
"""One-shot structural conditioning for the first Meyer cartoon solve.

The shipped one-pass split emits the result of a *linear* screened-Poisson
solve.  Its nonlinear shrink is only retained as state for pass two.  This
experiment predicts that missing reflected-gradient state directly from the
unchanged source and injects its divergence into the first equation:

    (c I - eta Delta) u = c f - eta div(r_0).

There is no outer iteration, path march, per-cell choice, or candidate scan
in a selected method.  The offline strength grid only answers whether a
single fixed conditioning coefficient exists across two known-truth rigs.

The structural gates are deliberately frozen:

* ``raw`` is the ungated predicted reflected state (a falsification control);
* ``persistent`` is a cheap cross-scale/eikonal-normal confidence;
* ``tsv`` uses the four-direction symmetric cancellation diagnostic;
* ``tsv_tail`` admits only TSV's high-certainty structural tail;
* ``joint`` requires both forms of evidence.

Run:

    PYTHONPATH=.:experiments python experiments/meyer_first_pass_conditioning.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from experiments.meyer_tsv_validation import (
    _periodic_kernel,
    multiscale_crossing_scene,
    score_split,
    symmetric_support_scene,
    tsv_four_direction,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "out" / "meyer_first_pass_conditioning"
_TSV_SYMBOLS: dict[tuple, tuple[np.ndarray, ...]] = {}


def grad(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.roll(value, -1, axis=1) - value,
        np.roll(value, -1, axis=0) - value,
    )


def div(px: np.ndarray, py: np.ndarray) -> np.ndarray:
    return (
        px - np.roll(px, 1, axis=1)
        + py - np.roll(py, 1, axis=0)
    )


def lap_hat(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    wy = 2.0 * np.cos(2.0 * np.pi * np.arange(h) / h) - 2.0
    wx = 2.0 * np.cos(2.0 * np.pi * np.arange(w) / w) - 2.0
    return wy[:, None] + wx[None, :]


def screened(
    source: np.ndarray,
    c: float,
    eta: float,
    reflected: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    rhs = c * np.asarray(source, dtype=np.float64)
    if reflected is not None:
        rhs = rhs - eta * div(*reflected)
    denominator = c - eta * lap_hat(source.shape)
    return np.fft.ifft2(np.fft.fft2(rhs) / denominator).real


def gaussian_periodic(value: np.ndarray, sigma: float) -> np.ndarray:
    h, w = value.shape
    ky = 2.0 * np.pi * np.fft.fftfreq(h)
    kx = 2.0 * np.pi * np.fft.fftfreq(w)
    symbol = np.exp(-0.5 * sigma * sigma * (
        ky[:, None] * ky[:, None] + kx[None, :] * kx[None, :]
    ))
    return np.fft.ifft2(np.fft.fft2(value) * symbol).real


def tsv_one_forward(
    source: np.ndarray,
    *,
    sigma_long: float = 12.0,
    sigma_width: float = 0.75,
    radius: int = 12,
) -> np.ndarray:
    """Four-direction TSV using one shared source FFT.

    A directional difference and its convolution are both multipliers of the
    same source spectrum.  Cache their product, perform one forward transform,
    then only the four unavoidable scalar inverse transforms.
    """
    shape = tuple(source.shape)
    key = (shape, float(sigma_long), float(sigma_width), int(radius))
    symbols = _TSV_SYMBOLS.get(key)
    if symbols is None:
        h, w = shape
        ky = np.arange(h, dtype=np.float64)[:, None] / h
        kx = np.arange(w // 2 + 1, dtype=np.float64)[None, :] / w
        directions = (
            (1, 0, 0.0),
            (0, 1, np.pi / 2.0),
            (1, 1, np.pi / 4.0),
            (1, -1, 3.0 * np.pi / 4.0),
        )
        built = []
        for dy, dx, theta in directions:
            difference = np.exp(2j * np.pi * (dy * ky + dx * kx)) - 1.0
            kernel = _periodic_kernel(
                shape, theta, sigma_long, sigma_width, radius
            )
            built.append(difference * np.fft.rfft2(kernel))
        symbols = tuple(built)
        _TSV_SYMBOLS[key] = symbols
    source_spectrum = np.fft.rfft2(np.asarray(source, dtype=np.float64))
    total = np.zeros(shape, dtype=np.float64)
    for symbol in symbols:
        total += np.abs(np.fft.irfft2(source_spectrum * symbol, s=shape))
    return total


def predicted_reflection(
    source: np.ndarray,
    eta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict the C++ ``db = d-b`` field from the source gradient."""
    gx, gy = grad(source)
    magnitude = np.hypot(gx, gy)
    coefficient = np.maximum(magnitude - 1.0 / eta, 0.0) / np.maximum(
        magnitude, 1e-30
    )
    reflected_scale = 2.0 * coefficient - 1.0
    return reflected_scale * gx, reflected_scale * gy


def persistent_gate(source: np.ndarray) -> np.ndarray:
    """Cheap frozen eikonal-normal confidence from cross-scale agreement."""
    gx, gy = grad(source)
    smooth = gaussian_periodic(source, 1.35)
    sx, sy = grad(smooth)
    magnitude = np.hypot(gx, gy)
    smooth_magnitude = np.hypot(sx, sy)
    agreement = np.maximum(gx * sx + gy * sy, 0.0) / np.maximum(
        magnitude * smooth_magnitude, 1e-30
    )

    # A local structure tensor distinguishes a persistent normal front from
    # an isotropic or crossing high-frequency field.  No angle extraction is
    # needed: the two eigenvalues follow from trace and discriminant.
    jxx = gaussian_periodic(gx * gx, 1.15)
    jxy = gaussian_periodic(gx * gy, 1.15)
    jyy = gaussian_periodic(gy * gy, 1.15)
    trace = jxx + jyy
    coherence = np.hypot(jxx - jyy, 2.0 * jxy) / np.maximum(trace, 1e-30)
    scale = max(float(np.percentile(smooth_magnitude, 82.0)), 1e-12)
    amplitude = smooth_magnitude / (smooth_magnitude + scale)
    return np.clip(agreement * coherence * coherence * amplitude, 0.0, 1.0)


def normalize_gate(value: np.ndarray, percentile: float = 82.0) -> np.ndarray:
    scale = max(float(np.percentile(value, percentile)), 1e-12)
    ratio = np.asarray(value, dtype=np.float64) / scale
    # The square rejects weak texture-interior responses while saturating
    # decisive discontinuities smoothly.
    return np.clip(ratio * ratio / (1.0 + ratio * ratio), 0.0, 1.0)


def symmetric_chord(source: np.ndarray, radii: tuple[int, ...]) -> np.ndarray:
    """Box-support TSV in O(len(radii) * directions * pixels).

    Integrating a signed directional difference over a symmetric box support
    telescopes to its two endpoints.  This is the rectangular-kernel sibling
    of TSV's Gaussian integral, with no convolution or march.
    """
    total = np.zeros_like(source, dtype=np.float64)
    for radius in radii:
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
            positive = np.roll(source, (-radius * dy, -radius * dx), axis=(0, 1))
            negative = np.roll(source, (radius * dy, radius * dx), axis=(0, 1))
            total += np.abs(positive - negative) / max(float(radius), 1.0)
    return total


def gates(source: np.ndarray) -> dict[str, np.ndarray]:
    persistent = persistent_gate(source)
    tsv = normalize_gate(tsv_one_forward(source))
    chord6 = normalize_gate(symmetric_chord(source, (6,)))
    chord12 = normalize_gate(symmetric_chord(source, (12,)))
    chord_multi = normalize_gate(symmetric_chord(source, (4, 8, 12)))
    return {
        "raw": np.ones_like(source),
        "persistent": persistent,
        "tsv": tsv,
        "tsv_tail": tsv ** 6,
        "joint": np.sqrt(persistent * tsv),
        "chord6": chord6,
        "chord12": chord12,
        "chord_multi": chord_multi,
        "chord_tail": chord_multi ** 4,
    }


def checker_support_scene(size: int = 256, period: float = 8.0) -> dict:
    """Hard square-wave carrier control with the same structural contours."""
    scene = symmetric_support_scene(size)
    y, x = np.mgrid[:size, :size].astype(np.float64)
    x0, x1 = 0.43 * size, 0.92 * size
    y0, y1 = 0.16 * size, 0.86 * size
    rectangle = (x > x0) & (x < x1) & (y > y0) & (y < y1)
    distance = np.minimum.reduce((x - x0, x1 - x, y - y0, y1 - y))
    taper = np.clip((distance - 5.0) / 13.0, 0.0, 1.0) * rectangle
    checker = (
        np.sign(np.sin(2.0 * np.pi * x / period))
        * np.sign(np.sin(2.0 * np.pi * y / period))
    )
    scene["name"] = "checker_support"
    scene["texture"] = 22.0 * taper * checker
    scene["source"] = scene["cartoon"] + scene["texture"]
    return scene


def first_split(
    source: np.ndarray,
    reflected: tuple[np.ndarray, np.ndarray] | None,
    *,
    lam: float,
    mu: float,
) -> tuple[np.ndarray, np.ndarray]:
    cartoon = screened(source, lam, 2.0 * lam, reflected)
    remainder = source - cartoon
    survivor = screened(remainder, 1.0 / mu, 10.0 / mu)
    return cartoon, remainder - survivor


def conditioned_first_split_tsv_tail(
    source: np.ndarray,
    *,
    lam: float = 0.05,
    mu: float = 40.0,
    strength: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Selected noniterative candidate: one frozen gate and one Meyer pass."""
    source = np.asarray(source, dtype=np.float64)
    gate = normalize_gate(tsv_one_forward(source)) ** 6
    rx, ry = predicted_reflection(source, 2.0 * lam)
    cartoon, texture = first_split(
        source,
        (strength * gate * rx, strength * gate * ry),
        lam=lam,
        mu=mu,
    )
    return cartoon, texture, gate


def conditioned_first_split_chord_tail(
    source: np.ndarray,
    *,
    lam: float = 0.05,
    mu: float = 40.0,
    strength: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """O(N) rectangular-support alternative to Gaussian TSV conditioning."""
    source = np.asarray(source, dtype=np.float64)
    gate = normalize_gate(symmetric_chord(source, (4, 8, 12))) ** 4
    rx, ry = predicted_reflection(source, 2.0 * lam)
    cartoon, texture = first_split(
        source,
        (strength * gate * rx, strength * gate * ry),
        lam=lam,
        mu=mu,
    )
    return cartoon, texture, gate


def objective(score: dict) -> float:
    # Known-truth decomposition loss.  Contour leakage is explicitly charged
    # because it is the conditioning failure the TSV control localized.
    return (
        score["cartoon_relative_rms_error"] ** 2
        + score["texture_relative_rms_error"] ** 2
        + 0.25 * score["contour_excess_texture_rms"] ** 2
    )


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=np.float64).ravel()
    bv = np.asarray(b, dtype=np.float64).ravel()
    av -= np.mean(av)
    bv -= np.mean(bv)
    return float(av @ bv / max(np.linalg.norm(av) * np.linalg.norm(bv), 1e-30))


def ideal_divergence(scene: dict, c: float, eta: float) -> np.ndarray:
    source = scene["source"]
    truth = scene["cartoon"]
    lap_truth = div(*grad(truth))
    return (c * source - c * truth + eta * lap_truth) / eta


def evaluate_scene(
    scene: dict,
    strengths: tuple[float, ...],
    lam: float,
    mu: float,
) -> tuple[dict, dict[str, tuple[np.ndarray, np.ndarray]]]:
    source = np.asarray(scene["source"], dtype=np.float64)
    reflected = predicted_reflection(source, 2.0 * lam)
    t0 = time.perf_counter()
    field_gates = gates(source)
    gate_ms = 1000.0 * (time.perf_counter() - t0)
    wanted_divergence = ideal_divergence(scene, lam, 2.0 * lam)

    splits: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    records = []
    cartoon, texture = first_split(source, None, lam=lam, mu=mu)
    baseline_score = score_split(cartoon, texture, scene)
    baseline_score["conditioning_objective"] = objective(baseline_score)
    records.append({"method": "baseline", "strength": 0.0, **baseline_score})
    splits["baseline"] = (cartoon, texture)

    for method, gate in field_gates.items():
        candidate_divergence = div(gate * reflected[0], gate * reflected[1])
        correlation = _correlation(candidate_divergence, wanted_divergence)
        for strength in strengths:
            flux = (
                strength * gate * reflected[0],
                strength * gate * reflected[1],
            )
            cartoon, texture = first_split(
                source, flux, lam=lam, mu=mu
            )
            scored = score_split(cartoon, texture, scene)
            scored["conditioning_objective"] = objective(scored)
            label = f"{method}@{strength:g}"
            records.append({
                "method": method,
                "strength": strength,
                "ideal_divergence_correlation": correlation,
                **scored,
            })
            splits[label] = (cartoon, texture)

    best = min(records, key=lambda row: row["conditioning_objective"])
    return {
        "name": scene["name"],
        "gate_milliseconds": gate_ms,
        "baseline": records[0],
        "best": best,
        "records": records,
    }, splits


def common_fixed_choice(results: list[dict]) -> tuple[str, float, list[dict]]:
    grouped: dict[tuple[str, float], list[dict]] = {}
    for result in results:
        baseline = result["baseline"]["conditioning_objective"]
        for row in result["records"]:
            key = (row["method"], row["strength"])
            grouped.setdefault(key, []).append({
                "scene": result["name"],
                "relative_objective": row["conditioning_objective"] / baseline,
                **row,
            })
    eligible = {
        key: rows for key, rows in grouped.items()
        if len(rows) == len(results)
    }
    key, rows = min(
        eligible.items(),
        key=lambda item: max(row["relative_objective"] for row in item[1]),
    )
    return key[0], key[1], rows


def render(scene: dict, splits: dict, labels: list[str], path: Path) -> None:
    panels = []
    source = scene["source"]
    arrays = [("source", source)]
    for label in labels:
        cartoon, texture = splits[label]
        arrays.extend(((f"{label} cartoon", cartoon), (f"{label} texture", texture)))
    for label, value in arrays:
        if "texture" in label:
            shown = np.clip(127.5 + 2.2 * value, 0.0, 255.0)
        else:
            shown = np.clip(value, 0.0, 255.0)
        image = Image.fromarray(shown.astype(np.uint8), mode="L").convert("RGB")
        canvas = Image.new("RGB", (image.width, image.height + 24), "white")
        canvas.paste(image, (0, 24))
        ImageDraw.Draw(canvas).text((5, 5), label, fill="black")
        panels.append(canvas)
    output = Image.new("RGB", (panels[0].width * len(panels), panels[0].height), "white")
    for index, panel in enumerate(panels):
        output.paste(panel, (index * panel.width, 0))
    output.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--lam", type=float, default=0.05)
    parser.add_argument("--mu", type=float, default=40.0)
    args = parser.parse_args()
    strengths = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
    scenes = (
        symmetric_support_scene(args.size),
        multiscale_crossing_scene(args.size),
        checker_support_scene(args.size),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    split_sets = []
    for scene in scenes:
        result, splits = evaluate_scene(scene, strengths, args.lam, args.mu)
        results.append(result)
        split_sets.append(splits)
        best = result["best"]
        print(
            f"{result['name']}: baseline obj "
            f"{result['baseline']['conditioning_objective']:.5g}; best "
            f"{best['method']}@{best['strength']:g} "
            f"{best['conditioning_objective']:.5g}; gate {result['gate_milliseconds']:.1f} ms"
        )

    method, strength, common = common_fixed_choice(results)
    fixed_label = "baseline" if method == "baseline" else f"{method}@{strength:g}"
    print(f"fixed minimax choice: {fixed_label}")
    for row in common:
        print(
            f"  {row['scene']}: objective {row['relative_objective']:.3f}x, "
            f"texture gain {row['interior_texture_gain']:.3f}, "
            f"contour excess {row['contour_excess_texture_rms']:.3f}, "
            f"allocation AUC {row['texture_over_contour_allocation_auc']:.3f}"
        )

    for scene, splits in zip(scenes, split_sets):
        labels = ["baseline", fixed_label]
        render(scene, splits, labels, OUT / f"{scene['name']}_conditioning.png")

    period_sweep = {}
    for period in (4, 6, 8, 10, 12, 16, 20, 24, 32):
        scene = checker_support_scene(args.size, float(period))
        baseline = first_split(scene["source"], None, lam=args.lam, mu=args.mu)
        conditioned = conditioned_first_split_tsv_tail(
            scene["source"], lam=args.lam, mu=args.mu
        )[:2]
        period_sweep[str(period)] = (
            objective(score_split(*conditioned, scene))
            / objective(score_split(*baseline, scene))
        )
    print("checker-period conditioned/baseline: " + ", ".join(
        f"{period}px {ratio:.3f}x" for period, ratio in period_sweep.items()
    ))

    payload = {
        "equation": "(c I - eta Delta)u = c f - eta div(r0)",
        "strengths_are_offline_validation_not_runtime_scan": True,
        "fixed_choice": {"method": method, "strength": strength, "scenes": common},
        "checker_period_sweep_conditioned_over_baseline": period_sweep,
        "results": results,
    }
    (OUT / "results.json").write_text(json.dumps(payload, indent=2))
    print(f"wrote {OUT / 'results.json'}")


if __name__ == "__main__":
    main()
