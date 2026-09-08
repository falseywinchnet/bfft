"""Matched first battery for DCNT, the reconstructed TV baseline, and FMMT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw

from .dcnt import DCNTResolution, denoise_dcnt, tv_chambolle_reference
from .fmmt_certified import denoise_fmmt
from .run_2d_denoiser_battery import metrics, sources
from .sample_series import corrupt


CASES = (
    ("clean", "none", 0.0, 0.0),
    ("Gaussian 0.10", "Gaussian additive", 0.10, 0.0),
    ("uniform 0.10", "uniform additive", 0.10, 0.0),
    ("replacement 0.25", "random-value replacement", 0.0, 0.25),
    ("mixed 0.10/0.25", "mixed replacement + uniform", 0.10, 0.25),
)


def _panel(images: list[tuple[str, np.ndarray]], output: Path) -> None:
    scale = 4
    label_height = 28
    side = images[0][1].shape[0]
    canvas = Image.new(
        "L", (len(images) * side * scale, side * scale + label_height), 255)
    draw = ImageDraw.Draw(canvas)
    for index, (name, image) in enumerate(images):
        pixels = np.uint8(np.round(np.clip(image, 0.0, 1.0) * 255.0))
        tile = Image.fromarray(pixels).resize(
            (side * scale, side * scale), Image.Resampling.NEAREST)
        left = index * side * scale
        canvas.paste(tile, (left, label_height))
        draw.text((left + 4, 7), name, fill=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def run(size: int, cycles: int, output: Path) -> dict:
    selected = {
        key: value for key, value in sources(size).items()
        if key in ("cameraman", "geometric interfaces", "woven chirps")
    }
    resolution = DCNTResolution(maximum_cycles=cycles)
    rows = []
    panels = []
    for source_name, truth in selected.items():
        for case_name, kind, amount, density in CASES:
            observation = (
                truth.copy() if kind == "none" else corrupt(
                    truth, kind, amount=amount, density=density, seed=9100)
            )
            methods = {
                "observation": lambda: observation.copy(),
                "reconstructed Chambolle TV": lambda: tv_chambolle_reference(
                    observation, weight=0.10),
                "DCNT transport descent": lambda: denoise_dcnt(
                    observation, mode="transport_descent",
                    resolution=resolution)[0],
                "DCNT uncertainty": lambda: denoise_dcnt(
                    observation, mode="uncertainty", resolution=resolution)[0],
                "integrated FMMT control": lambda: denoise_fmmt(observation)[0],
            }
            rendered = [("truth", truth), ("observation", observation)]
            for method_name, estimator in methods.items():
                started = perf_counter()
                estimate = estimator()
                elapsed = perf_counter() - started
                rows.append({
                    "source": source_name,
                    "case": case_name,
                    "method": method_name,
                    "runtime_seconds": elapsed,
                    **metrics(estimate, truth),
                })
                if method_name != "observation":
                    rendered.append((method_name, estimate))
            if source_name == "cameraman" and case_name in (
                    "Gaussian 0.10", "mixed 0.10/0.25"):
                panel_path = output.parent / (
                    f"dcnt_{case_name.replace(' ', '_').replace('/', '_')}.png")
                _panel(rendered, panel_path)
                panels.append(str(panel_path))
    methods = sorted({row["method"] for row in rows})
    summary = {}
    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        summary[method] = {
            key: float(np.mean([row[key] for row in subset]))
            for key in (
                "mse", "ssim", "edge_retention", "variance_ratio",
                "runtime_seconds",
            )
        }
    report = {
        "experiment": "DCNT first target-excluded CONV observer battery",
        "size": int(size),
        "cycles": int(cycles),
        "cases": [case[0] for case in CASES],
        "sources": list(selected),
        "summary": summary,
        "rows": rows,
        "panels": panels,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument(
        "--out", type=Path, default=Path("/tmp/dcnt_first_battery.json"))
    args = parser.parse_args()
    report = run(args.size, args.cycles, args.out)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
