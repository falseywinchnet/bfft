#!/usr/bin/env python3
"""Native one-pass vs conditioned-f_spec pass vs ordinary 64-pass Meyer."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import bfft
from experiments.meyer_first_pass_conditioning import (
    checker_support_scene,
    objective,
)
from experiments.meyer_preconditioning_research import junction_texture_scene
from experiments.meyer_tsv_validation import (
    multiscale_crossing_scene,
    score_split,
    symmetric_support_scene,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "out" / "meyer_native_conditioning"


def timed(call, repeats: int) -> tuple[tuple[np.ndarray, np.ndarray], dict]:
    call()  # warm transforms and plan-owned scratch
    samples = []
    result = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = call()
        samples.append(1000.0 * (time.perf_counter() - started))
    return result, {
        "median_ms": statistics.median(samples),
        "minimum_ms": min(samples),
        "mean_ms": statistics.mean(samples),
        "repeats": repeats,
    }


def render(scene: dict, arms: dict, path: Path) -> None:
    source = np.asarray(scene["source"])
    arrays = [("truth source", source)]
    for name, split in arms.items():
        arrays.extend(((f"{name} cartoon", split[0]), (f"{name} texture", split[1])))
    panels = []
    for label, value in arrays:
        shown = (
            np.clip(127.5 + 2.2 * value, 0.0, 255.0)
            if "texture" in label
            else np.clip(value, 0.0, 255.0)
        )
        image = Image.fromarray(shown.astype(np.uint8), mode="L").convert("RGB")
        panel = Image.new("RGB", (image.width, image.height + 24), "white")
        panel.paste(image, (0, 24))
        ImageDraw.Draw(panel).text((5, 5), label, fill="black")
        panels.append(panel)
    output = Image.new(
        "RGB", (panels[0].width * len(panels), panels[0].height), "white"
    )
    for index, panel in enumerate(panels):
        output.paste(panel, (index * panel.width, 0))
    output.save(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scenes = (
        symmetric_support_scene(256),
        multiscale_crossing_scene(256),
        checker_support_scene(256),
        junction_texture_scene(256),
    )
    report = {"strength": 1.5, "scenes": {}}
    for scene in scenes:
        source = np.ascontiguousarray(scene["source"], dtype=np.float64)
        one = bfft.MeyerPlan(source.shape, passes=1, threads=4, solver=0)
        full = bfft.MeyerPlan(source.shape, passes=64, threads=4, solver=0)
        calls = {
            "ordinary_pass_1": (lambda: one.split(source), 50),
            "conditioned_f_spec_pass_1": (
                lambda: one.split_conditioned_first(source, strength=1.5), 50
            ),
            "finite_virtual_transverse": (
                lambda: one.split_preconditioned(
                    source, strength=1.5, virtual_passes=8, gate_power=8
                ),
                50,
            ),
            "ordinary_pass_64": (lambda: full.split(source), 10),
        }
        arms = {}
        rows = {}
        for name, (call, repeats) in calls.items():
            split, timing = timed(call, repeats)
            arms[name] = split
            scores = score_split(*split, scene)
            rows[name] = {
                "timing": timing,
                "conditioning_objective": objective(scores),
                **scores,
            }
        # Do not collapse texture completeness and contour purity into one
        # scalar ranking. The conditioned pass and pass 64 occupy different
        # sides of that Pareto tradeoff, and the image makes the missing
        # carrier amplitude in the conditioned cartoon unmistakable.
        report["scenes"][scene["name"]] = rows
        render(scene, arms, OUT / f"{scene['name']}.png")
        print(f"\n{scene['name']}")
        for name, row in rows.items():
            print(
                f"  {name:27s} {row['timing']['median_ms']:7.3f} ms  "
                f"gain {row['interior_texture_gain']:.3f}  "
                f"interior error {row['interior_texture_relative_rms_error']:.3f}  "
                f"contour {row['contour_excess_texture_rms']:.3f}  "
                f"AUC {row['texture_over_contour_allocation_auc']:.3f}"
            )
    (OUT / "results.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {OUT / 'results.json'}")


if __name__ == "__main__":
    main()
