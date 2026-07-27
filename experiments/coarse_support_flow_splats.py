#!/usr/bin/env python3
"""Use the known-good coarse reconstruction as a carrier for flow splats.

This experiment does not propose that the reference allocator remain in the
final algorithm.  It isolates the compositional claim:

    cheap cartoon-effective support + one-shot scale-time texture splats.

The established receiver-guided model first constructs a coarse carrier.
That reconstruction is frozen.  BFFT pass-volume ellipses then fit only its
remaining target residual.  Circularizing the same splats is the anisotropy
control; uniform R2 residual splats are the placement control.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "viewer"))
sys.path.insert(0, str(ROOT / "experiments"))

import gallery  # noqa: E402
from bfft.vision import SingleStageDecompositionObjective  # noqa: E402
from dual_aperture_support import (  # noqa: E402
    aperture,
    design_matrix,
    score,
    solve_field,
)
from flow_volume_cells import (  # noqa: E402
    emit_population,
    r2_control,
    support_samples,
)
from bfft_flow_stage_geometry import build_flow_volume  # noqa: E402
from receiver_guided_graph import ReceiverGuidedVoronoi  # noqa: E402
from resource_transport_cells import (  # noqa: E402
    ResourceConfig,
    ResourceTransportCells,
)
from transport_voronoi import Config, _fit_rgb  # noqa: E402


def _load_image(path: str | None, gallery_key: str) -> tuple[np.ndarray, str]:
    if path:
        from skimage.io import imread

        resolved = Path(path).expanduser().resolve()
        return imread(resolved), str(resolved)
    return gallery.load(gallery_key), f"gallery:{gallery_key}"


def build_carrier(
    image: np.ndarray,
    side: int,
    initial_cells: int,
    carrier_cells: int,
    kind: str,
    resource_rounds: int,
):
    if kind == "resource":
        model = ResourceTransportCells(
            image,
            ResourceConfig(max_side=side, cells=initial_cells),
        )
        for _ in range(resource_rounds):
            model.step()
        return model
    if kind != "receiver":
        raise ValueError(f"unknown carrier kind {kind!r}")
    model = ReceiverGuidedVoronoi(
        image,
        Config(
            max_side=side,
            initial_cells=initial_cells,
            max_cells=carrier_cells,
            split_batch=24,
            passes=24,
            flow_sweeps=64,
            lam=0.05,
            mu=40.0,
            anisotropy=5.0,
            edge_density=4.0,
            texture_density=3.0,
            edge_barrier=12.0,
            site_reach=1.5,
            allocation_mode="Expected affine gain",
        ),
    )
    model.solve_direct_coupled(4.0, 16.0)
    while len(model.seeds) < carrier_cells:
        before = len(model.seeds)
        model.cfg.split_batch = min(24, carrier_cells - before)
        model.step_direct(True, 4.0, 16.0)
        if len(model.seeds) <= before:
            break
    return model


def _carrier_count(model) -> int:
    if hasattr(model, "seeds"):
        return int(len(model.seeds))
    return int(len(model.centers))


def fit_residual(population, target_lab, carrier_lab, objective):
    height, width = target_lab.shape[:2]
    samples = support_samples(population, height, width)
    weight, dominance, effective = aperture(
        samples, height * width, 1.0)
    design = design_matrix(samples, height * width, weight)
    delta = solve_field(design, target_lab - carrier_lab)
    reconstruction = carrier_lab + delta
    record = score(objective, objective.target_rgb, reconstruction)
    covered = np.asarray(design.getnnz(axis=1) > 0)
    diagnostic = {
        "samples": int(len(samples["rows"])),
        "covered_fraction": float(np.mean(covered)),
        "dominance_mean": float(np.mean(dominance[covered])),
        "effective_median": float(np.median(effective[covered])),
    }
    return record, reconstruction, diagnostic


def _serializable(record):
    return {
        key: float(value)
        for key, value in record.items()
        if key != "rgb"
    }


def save_panel(target, carrier_record, variants, population, output):
    columns = 2 + len(variants)
    fig, axes = plt.subplots(2, columns, figsize=(4 * columns, 8.5))
    displays = [
        ("target", {"rgb": target}),
        ("coarse carrier", carrier_record),
    ] + [(name, record) for name, record, _ in variants]
    for axis, (name, record) in zip(axes[0], displays):
        axis.imshow(record["rgb"])
        title = name
        if "psnr" in record:
            title += f"\n{record['psnr']:.2f} dB  obj {record['objective']:.4g}"
        axis.set_title(title)

    axes[1, 0].imshow(target)
    detail = population.stage > 0
    axes[1, 0].scatter(
        population.centers[detail, 0],
        population.centers[detail, 1],
        c=population.stage[detail],
        s=np.clip(
            population.major[detail] * population.minor[detail], 2, 30),
        cmap="turbo",
        alpha=0.7,
        linewidths=0,
    )
    axes[1, 0].set_title("one-shot flow splats")
    carrier_error = np.sqrt(np.mean(
        (target - carrier_record["rgb"]) ** 2, axis=2))
    axes[1, 1].imshow(carrier_error, cmap="inferno")
    axes[1, 1].set_title("carrier residual offered to splats")
    for axis, (name, record, _) in zip(axes[1, 2:], variants):
        error = np.sqrt(np.mean((target - record["rgb"]) ** 2, axis=2))
        axis.imshow(error, cmap="inferno")
        axis.set_title(f"{name} remaining error")
    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?")
    parser.add_argument("--gallery", default="pikachu")
    parser.add_argument("--side", type=int, default=128)
    parser.add_argument("--initial-cells", type=int, default=180)
    parser.add_argument("--carrier-cells", type=int, default=458)
    parser.add_argument(
        "--carrier",
        choices=("resource", "receiver"),
        default="resource",
    )
    parser.add_argument("--resource-rounds", type=int, default=30)
    parser.add_argument("--flow-cells", type=int, default=300)
    parser.add_argument("--curvature-limit", action="store_true")
    parser.add_argument("--splat-advection", type=float, default=0.0)
    parser.add_argument(
        "--transport-consistency-power",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/out/coarse_support_flow_splats.png",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=ROOT / "experiments/out/coarse_support_flow_splats.json",
    )
    args = parser.parse_args()

    image, source = _load_image(args.image, args.gallery)
    started = time.perf_counter()
    carrier = build_carrier(
        image,
        args.side,
        args.initial_cells,
        args.carrier_cells,
        args.carrier,
        args.resource_rounds,
    )
    carrier_ms = (time.perf_counter() - started) * 1000.0
    objective = SingleStageDecompositionObjective(carrier.rgb)
    carrier_record = score(
        objective, carrier.rgb, carrier.reconstruction)

    volume = build_flow_volume(carrier.rgb, passes=24)
    population, emission = emit_population(
        volume,
        carrier.h,
        carrier.w,
        0,
        args.flow_cells,
        2,
        0.0,
        18.0,
        args.curvature_limit,
        0.0,
        0.0,
        "independent",
        0.0,
        0.0,
        "glass",
        "tangent",
        0.0,
        args.splat_advection,
        args.transport_consistency_power,
    )
    controls = {
        "flow ellipses": population,
        "same sites circular": population.circularized(),
        "uniform residual circles": r2_control(
            len(population.centers), carrier.h, carrier.w, 1.8),
    }
    variants = []
    diagnostics = {}
    for name, control in controls.items():
        record, reconstruction, diagnostic = fit_residual(
            control, carrier.lab, carrier.reconstruction, objective)
        variants.append((name, record, reconstruction))
        diagnostics[name] = diagnostic

    save_panel(
        carrier.rgb, carrier_record, variants, population, args.output)
    report = {
        "source": source,
        "shape": list(carrier.rgb.shape),
        "carrier_kind": args.carrier,
        "carrier_cells": _carrier_count(carrier),
        "flow_cells": int(len(population.centers)),
        "combined_coefficients": int(
            9 * (_carrier_count(carrier) + len(population.centers))),
        "emission": emission,
        "carrier": _serializable(carrier_record),
        "variants": {
            name: _serializable(record)
            for name, record, _ in variants
        },
        "diagnostics": diagnostics,
        "carrier_ms": carrier_ms,
        "total_ms": (time.perf_counter() - started) * 1000.0,
        "output": str(args.output.resolve()),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
