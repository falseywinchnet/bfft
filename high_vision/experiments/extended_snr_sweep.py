#!/usr/bin/env python3
"""Frame-count and projection-circle sweep for low-light fusion."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from budgeted_fullres_demo import (
    Budget,
    best_cyclic_alignment,
    circular_cross_wiener,
    run,
    scores,
)


RING_WIDTHS = (1, 2, 4, 8)


def cases() -> list[tuple[str, Budget]]:
    live = replace(Budget(), registration_rounds=1)
    return [
        ("0.10e 256f", replace(live, frames=256)),
        ("0.10e 512f", replace(live, frames=512)),
        ("0.10e 1024f", replace(live, frames=1024)),
        ("0.10e 2048f", replace(live, frames=2048)),
        ("0.04e 2048f", replace(
            live,
            frames=2048,
            photons_at_white=0.04,
            registration_group=16,
        )),
    ]


def projection_variants(images: dict[str, np.ndarray]) -> list[dict]:
    variants = []
    for width in RING_WIDTHS:
        filtered, first, second, diagnostics = circular_cross_wiener(
            images["registered_even"],
            images["registered_odd"],
            width,
        )
        filtered, _ = best_cyclic_alignment(filtered, images["source"])
        filtered = np.clip(filtered, 0.0, 1.0)
        noise_rms = float(np.sqrt(np.mean((0.5 * (first - second)) ** 2)))
        mean = 0.5 * (first + second)
        signal_variance = max(float(np.var(mean)) - noise_rms ** 2, 1e-15)
        variants.append({
            "ring_width": width,
            "metrics": scores(filtered, images["source"]),
            "noise_rms": noise_rms,
            "snr_db": float(10.0 * np.log10(
                signal_variance / max(noise_rms ** 2, 1e-15))),
            **diagnostics,
        })
    return variants


def render(records: list[dict], path: Path) -> None:
    ordinary = [
        record for record in records
        if record["result"]["budget"]["photons_at_white"] == 0.10
    ]
    frames = np.asarray([
        record["result"]["budget"]["frames"] for record in ordinary])
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))

    raw_psnr = [
        record["result"]["metrics"]["bounded_registered"]["psnr_db"]
        for record in ordinary]
    raw_ssim = [
        record["result"]["metrics"]["bounded_registered"]["ssim"]
        for record in ordinary]
    axes[0, 0].plot(frames, raw_psnr, "o--", label="raw registered")
    axes[0, 1].plot(frames, raw_ssim, "o--", label="raw registered")
    for width in RING_WIDTHS:
        variants = [
            next(item for item in record["projection_variants"]
                 if item["ring_width"] == width)
            for record in ordinary
        ]
        axes[0, 0].plot(
            frames,
            [item["metrics"]["psnr_db"] for item in variants],
            "o-",
            label=f"{width}px circles",
        )
        axes[0, 1].plot(
            frames,
            [item["metrics"]["ssim"] for item in variants],
            "o-",
            label=f"{width}px circles",
        )

    before_noise = [
        record["result"]["split_half_support"]["before"]["noise_rms"]
        for record in ordinary]
    after_noise = [
        record["result"]["split_half_support"]["after"]["noise_rms"]
        for record in ordinary]
    before_snr = [
        record["result"]["split_half_support"]["before"]["snr_db"]
        for record in ordinary]
    after_snr = [
        record["result"]["split_half_support"]["after"]["snr_db"]
        for record in ordinary]
    axes[1, 0].loglog(frames, before_noise, "o--", label="before")
    axes[1, 0].loglog(frames, after_noise, "o-", label="after 2px circles")
    axes[1, 1].plot(frames, before_snr, "o--", label="before")
    axes[1, 1].plot(frames, after_snr, "o-", label="after 2px circles")

    axes[0, 0].set_title("Reconstruction PSNR")
    axes[0, 0].set_ylabel("dB")
    axes[0, 1].set_title("Structural similarity")
    axes[0, 1].set_ylabel("SSIM")
    axes[1, 0].set_title("Split-half measured noise")
    axes[1, 0].set_ylabel("RMS")
    axes[1, 1].set_title("Split-half measured SNR")
    axes[1, 1].set_ylabel("dB")
    for axis in axes.ravel():
        axis.set_xlabel("frames")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle(
        "BFFT High Vision: more photons and more projection-circle support")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("high_vision/out/extended_snr_sweep.png"),
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("high_vision/out/extended_snr_sweep.json"),
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    records = []
    if args.resume and args.json.exists():
        records = json.loads(args.json.read_text()).get("records", [])
    complete = {record["name"]: record for record in records}
    for name, config in cases():
        if (
            name in complete
            and complete[name]["result"].get("budget") == asdict(config)
        ):
            print(f"reusing {name}", flush=True)
            continue
        if name in complete:
            records.remove(complete[name])
        print(f"running {name}", flush=True)
        result, images = run(config)
        variants = projection_variants(images)
        record = {
            "name": name,
            "result": result,
            "projection_variants": variants,
        }
        records.append(record)
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"records": records}, indent=2) + "\n")
        filtered = result["metrics"]["circular_cross_wiener"]
        print(
            f"  {filtered['psnr_db']:.2f} dB, SSIM {filtered['ssim']:.3f}, "
            f"noise -{result['split_half_support']['noise_reduction_db']:.1f}dB",
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
