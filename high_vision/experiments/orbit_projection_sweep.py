#!/usr/bin/env python3
"""Projection-step and dwell sweep for the orbit-invariant bootstrap."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from poisson_orbit_demo import Camera, run


def cases() -> list[tuple[str, Camera]]:
    common = {
        "size": 32,
        "photons_at_white": 0.75,
        "optimizer_iterations": 400,
        "batch": 128,
        "seed": 7,
    }
    result = []
    for steps in (4, 8, 16, 32, 48):
        result.append((
            f"steps-{steps}",
            Camera(frames=4096, bispectrum_steps=steps, **common),
        ))
    for frames in (1024, 2048, 8192):
        result.append((
            f"frames-{frames}",
            Camera(frames=frames, bispectrum_steps=32, **common),
        ))
    result.append((
        "full-circle-64",
        Camera(
            frames=4096,
            bispectrum_steps=64,
            full_circle_support=True,
            **common,
        ),
    ))
    for iterations in (400, 1400):
        result.append((
            f"poisson-seed-{iterations}",
            Camera(
                frames=4096,
                bispectrum_steps=48,
                poisson_phase_seed=True,
                optimizer_iterations=iterations,
                batch=128,
                seed=7,
                size=32,
                photons_at_white=0.75,
            ),
        ))
    return result


def render(records: list[dict], path: Path) -> None:
    by_name = {record["name"]: record["result"] for record in records}
    step_counts = [4, 8, 16, 32, 48]
    step_results = [by_name[f"steps-{value}"] for value in step_counts]
    frame_counts = [1024, 2048, 4096, 8192]
    frame_results = [
        by_name["steps-32"] if value == 4096
        else by_name[f"frames-{value}"]
        for value in frame_counts
    ]

    def metric(items, name):
        return [
            item["metrics"]["orbit_cross_bispectrum"][name]
            for item in items
        ]

    figure, axes = plt.subplots(2, 2, figsize=(12, 8.2))
    axes[0, 0].plot(
        step_counts, metric(step_results, "psnr_db"), "o-")
    axes[0, 0].set_title("Distinct projection steps at 4,096 frames")
    axes[0, 0].set_ylabel("PSNR (dB)")
    axes[0, 1].plot(
        step_counts, metric(step_results, "ssim"), "o-", color="#4c78a8")
    axes[0, 1].set_title("Structural recovery versus projection support")
    axes[0, 1].set_ylabel("SSIM")
    axes[1, 0].semilogx(
        frame_counts, metric(frame_results, "psnr_db"), "o-",
        base=2)
    axes[1, 0].set_title("Longer photon dwell at 32 steps")
    axes[1, 0].set_ylabel("PSNR (dB)")
    axes[1, 1].loglog(
        frame_counts,
        [item["phase_objective"] for item in frame_results],
        "o-",
        base=2,
        color="#f58518",
    )
    axes[1, 1].set_title("Circle-valued phase residual")
    axes[1, 1].set_ylabel("objective")
    for axis in axes.ravel():
        axis.grid(alpha=0.2)
        axis.set_xlabel(
            "projection steps" if axis in axes[0] else "frames")
    figure.suptitle(
        "BFFT orbit bootstrap: support directions versus photon dwell")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("high_vision/out/orbit_projection_sweep.png"),
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("high_vision/out/orbit_projection_sweep.json"),
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    records = []
    if args.resume and args.json.exists():
        records = json.loads(args.json.read_text()).get("records", [])
    complete = {record["name"]: record for record in records}
    for name, config in cases():
        stored_camera = (
            complete.get(name, {}).get("result", {}).get("camera", {}).copy())
        # Backward-compatible checkpoint after this experimental flag was
        # introduced; false was the behavior of all earlier records.
        stored_camera.setdefault("poisson_phase_seed", False)
        if (
            name in complete
            and stored_camera == asdict(config)
        ):
            print(f"reusing {name}", flush=True)
            continue
        if name in complete:
            records.remove(complete[name])
        print(f"running {name}", flush=True)
        result, _ = run(config)
        records.append({"name": name, "result": result})
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"records": records}, indent=2) + "\n")
        metric = result["metrics"]["orbit_cross_bispectrum"]
        print(
            f"  {metric['psnr_db']:.2f} dB, SSIM {metric['ssim']:.3f}, "
            f"objective {result['phase_objective']:.4g}, "
            f"{result['elapsed_seconds']:.1f}s",
            flush=True,
        )
    order = {name: index for index, (name, _) in enumerate(cases())}
    records.sort(key=lambda record: order[record["name"]])
    args.json.write_text(json.dumps({"records": records}, indent=2) + "\n")
    render(records, args.output)
    print(f"wrote {args.output}")
    print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
