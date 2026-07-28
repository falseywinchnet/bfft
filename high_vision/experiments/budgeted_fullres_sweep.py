#!/usr/bin/env python3
"""Measured 512px budget sweep for bounded low-light registration."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from budgeted_fullres_demo import Budget, run


def cases() -> list[tuple[str, Budget]]:
    baseline = Budget()
    return [
        ("128f", replace(baseline, frames=128)),
        ("one round", replace(baseline, registration_rounds=1)),
        ("baseline", baseline),
        ("0.04 e-", replace(
            baseline, photons_at_white=0.04, registration_group=16)),
        ("0.02 e-", replace(
            baseline, frames=1024, photons_at_white=0.02,
            registration_group=32)),
        ("read 0.15", replace(
            baseline, read_noise_electrons=0.15)),
        ("read 0.30", replace(
            baseline, read_noise_electrons=0.30)),
        ("support 32", replace(
            baseline, shift_radius=32, motion_step=3,
            registration_group=12)),
    ]


def render(records: list[dict], path: Path) -> None:
    labels = [record["name"] for record in records]
    x = np.arange(len(labels))
    unregistered = [
        record["result"]["metrics"]["unregistered"]["psnr_db"]
        for record in records]
    registered = [
        record["result"]["metrics"]["bounded_registered"]["psnr_db"]
        for record in records]
    ssim = [
        record["result"]["metrics"]["bounded_registered"]["ssim"]
        for record in records]
    error = [
        record["result"]["registration_error_pixels"]["median"]
        for record in records]
    milliseconds = [
        record["result"]["milliseconds_per_input_frame_all_passes"]
        for record in records]

    figure, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    width = 0.38
    axes[0, 0].bar(x - width / 2, unregistered, width, label="unregistered")
    axes[0, 0].bar(x + width / 2, registered, width, label="bounded")
    axes[0, 0].set_ylabel("PSNR (dB)")
    axes[0, 0].set_title("Full-resolution reconstruction")
    axes[0, 0].legend()
    axes[0, 1].bar(x, ssim, color="#4c78a8")
    axes[0, 1].set_ylabel("SSIM")
    axes[0, 1].set_title("Registered structural similarity")
    axes[1, 0].bar(x, error, color="#f58518")
    axes[1, 0].set_ylabel("pixels")
    axes[1, 0].set_title("Median registration error after global gauge")
    axes[1, 1].bar(x, milliseconds, color="#54a24b")
    axes[1, 1].set_ylabel("ms / input frame")
    axes[1, 1].set_title("All streaming passes")
    for axis in axes.ravel():
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=28, ha="right")
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("BFFT High Vision: 512² bounded-support budget sweep")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=Path("high_vision/out/budgeted_fullres_sweep.png"))
    parser.add_argument(
        "--json", type=Path,
        default=Path("high_vision/out/budgeted_fullres_sweep.json"))
    parser.add_argument(
        "--resume", action="store_true",
        help="Reuse completed named cases from the JSON checkpoint")
    args = parser.parse_args()
    records = []
    if args.resume and args.json.exists():
        records = json.loads(args.json.read_text()).get("records", [])
    completed = {
        record["name"]: record for record in records
    }
    for name, config in cases():
        stored_budget = (
            completed.get(name, {}).get("result", {}).get("budget", {}).copy())
        stored_budget.setdefault("wiener_ring_width", 2)
        if (name in completed
                and stored_budget == asdict(config)):
            print(f"reusing {name}", flush=True)
            continue
        if name in completed:
            records.remove(completed[name])
        print(f"running {name}: {asdict(config)}", flush=True)
        result, _ = run(config)
        records.append({"name": name, "result": result})
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"records": records}, indent=2) + "\n")
        score = result["metrics"]["bounded_registered"]
        print(
            f"  {score['psnr_db']:.2f} dB, SSIM {score['ssim']:.3f}, "
            f"{result['registration_error_pixels']['median']:.2f}px, "
            f"{result['total_seconds']:.2f}s",
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
