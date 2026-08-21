"""Focused falsification battery for continual eikonal noise transport."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .continual_eikonal_noise_transport_2d import (
    denoise_continual_eikonal_noise_transport_2d,
)
from .continual_fabada_eikonal_2d import (
    denoise_complete_moment_residual_posterior_2d,
    denoise_continual_fabada_eikonal_2d,
    denoise_continual_residual_posterior_2d,
)
from .fmmt_certified import denoise_fmmt
from .reflection_consistent_posterior_2d import (
    denoise_reflection_consistent_posterior_2d,
)
from .run_typical_orbit_benchmark import ROIS, _camera, _metrics, _roi
from .sample_series import corrupt


CASES = (
    ("clean", "none", 0.0, 0.0),
    ("uniform_12", "uniform additive", 0.12, 0.0),
    ("gaussian_12", "Gaussian additive", 0.12, 0.0),
    ("laplace_10", "Laplace additive", 0.10, 0.0),
    ("replacement_15", "random-value replacement", 0.0, 0.15),
    ("salt_pepper_15", "salt and pepper", 0.0, 0.15),
    ("mixed_15", "mixed replacement + uniform", 0.12, 0.15),
)


def _save_panel(path: Path, images: dict[str, np.ndarray], scale: int = 1) -> None:
    names = tuple(images)
    height, width = next(iter(images.values())).shape
    label_height = 22
    panel = Image.new("L", (width * len(names), height + label_height), 255)
    draw = ImageDraw.Draw(panel)
    for index, name in enumerate(names):
        tile = Image.fromarray(np.uint8(np.round(
            np.clip(images[name], 0.0, 1.0) * 255.0)))
        panel.paste(tile, (index * width, label_height))
        draw.text((index * width + 3, 4), name, fill=0)
    if scale > 1:
        panel = panel.resize(
            (panel.width * scale, panel.height * scale), Image.Resampling.NEAREST)
    panel.save(path)


def run(size: int, seed: int, output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    truth = _camera(size)
    report: dict[str, object] = {
        "status": "first continual eikonal-noise checkpoint; not promoted",
        "size": int(size),
        "seed": int(seed),
        "cases": {},
    }
    for case, corruption, amount, density in CASES:
        observed = corrupt(
            truth, corruption, amount=amount, density=density, seed=seed)
        methods = {
            "observed": lambda: (observed.copy(), {}),
            "mean3": lambda: (
                ndimage.uniform_filter(observed, 3, mode="reflect"), {}),
            "median3": lambda: (
                ndimage.median_filter(observed, 3, mode="reflect"), {}),
            "rejected_fmmt": lambda: denoise_fmmt(observed),
            "continual_eikonal": lambda: (
                denoise_continual_eikonal_noise_transport_2d(observed)),
            "fabada_eikonal_average": lambda: (
                denoise_continual_fabada_eikonal_2d(observed)),
            "residual_posterior": lambda: (
                denoise_continual_residual_posterior_2d(observed)),
            "complete_moment_posterior": lambda: (
                denoise_complete_moment_residual_posterior_2d(observed)),
            "reflection_posterior": lambda: (
                denoise_reflection_consistent_posterior_2d(observed)),
        }
        images = {"truth": truth}
        case_result: dict[str, object] = {
            "corruption": corruption,
            "amount": amount,
            "density": density,
            "methods": {},
        }
        for name, method in methods.items():
            started = time.perf_counter()
            estimate, diagnostic = method()
            seconds = time.perf_counter() - started
            images[name] = estimate
            case_result["methods"][name] = {
                "seconds": seconds,
                "metrics": _metrics(estimate, truth),
                "accepted_iterations": diagnostic.get("accepted_iterations"),
                "evaluated_iterations": len(diagnostic.get("iterations", ())),
                "status": diagnostic.get("status"),
            }
        _save_panel(output / f"{case}_panel.png", images)
        _save_panel(
            output / f"{case}_tripod.png",
            {name: _roi(image, ROIS["tripod"]) for name, image in images.items()},
            scale=3,
        )
        report["cases"][case] = case_result
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=719)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.size, args.seed, args.out)
    summary = {}
    for case, record in report["cases"].items():
        summary[case] = {
            method: {
                "mse": values["metrics"]["mse"],
                "ssim": values["metrics"]["ssim"],
                "edge": values["metrics"]["strong_edge_projected_retention"],
                "tripod": values["metrics"]["rois"]["tripod"][
                    "strong_edge_projected_retention"],
                "seconds": values["seconds"],
                "iterations": values["accepted_iterations"],
            }
            for method, values in record["methods"].items()
        }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
