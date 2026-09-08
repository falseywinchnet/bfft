"""Benchmark and visualize the SIMD retained-rank boundary evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np
from PIL import Image, ImageDraw

from .child_volume import (
    AxisAlignedVolume,
    BoundaryFiber,
    ChildVolumeResponseSystem,
    compile_child_volume_rank_blocks,
)
from .light_tensor import LightTensor6
from .native_retained import native_available
from .transport import SphereOccluder


def _boundary(side: int) -> tuple[
    AxisAlignedVolume,
    tuple[BoundaryFiber, ...],
    tuple[BoundaryFiber, ...],
]:
    bounds = AxisAlignedVolume(
        np.array((0.0, -1.0, -1.0)),
        np.array((4.0, 1.0, 1.0)),
    )
    coordinate = np.linspace(-0.8, 0.8, side)
    step = 1.6 / max(side - 1, 1)
    fibers = []
    exits = []
    for y in coordinate:
        for z in coordinate:
            fibers.append(
                BoundaryFiber(
                    np.array((0.0, y, z)),
                    np.array((-1.0, 0.0, 0.0)),
                    np.array((1.0, 0.0, 0.0)),
                    step * step,
                    0.01,
                )
            )
            exits.append(
                BoundaryFiber(
                    np.array((4.0, y, z)),
                    np.array((1.0, 0.0, 0.0)),
                    np.array((1.0, 0.0, 0.0)),
                    step * step,
                    0.01,
                )
            )
    return bounds, tuple(fibers), tuple(exits)


def _modes() -> tuple[LightTensor6, ...]:
    # Three visible basis packets and three polarized/signed ledger channels.
    return (
        LightTensor6(1.0, 620.0, 18.0**2, provenance=1),
        LightTensor6(1.0, 535.0, 18.0**2, provenance=2),
        LightTensor6(1.0, 455.0, 18.0**2, provenance=4),
        LightTensor6(1.0, 580.0, 21.0**2, 22.0, 0.75, 0.20, 8),
        LightTensor6(1.0, 500.0, 24.0**2, 91.0, 0.20, -0.70, 16),
        LightTensor6(1.0, 545.0, 28.0**2, 143.0, 0.35, 0.45, 32),
    )


def _source(side: int) -> np.ndarray:
    coordinate = np.linspace(-0.8, 0.8, side)
    y, z = np.meshgrid(coordinate, coordinate, indexing="ij")
    power = np.zeros((6, side, side), dtype=np.float64)
    power[0] = 1.2 * np.exp(-((y + 0.38) ** 2 + (z + 0.24) ** 2) / 0.12)
    power[1] = 1.0 * np.exp(-((y - 0.30) ** 2 + (z - 0.22) ** 2) / 0.10)
    power[2] = 0.75 * np.exp(-((y + 0.02) ** 2 + (z - 0.43) ** 2) / 0.18)
    power[3] = 0.17 * (0.25 + power[0])
    power[4] = -0.06 * (0.20 + power[1])
    power[5] = 0.11 * (0.20 + power[2])
    return power.reshape(6, side * side)


def _timing(function, repeats: int) -> tuple[float, object]:
    result = function()
    samples = []
    for _ in range(5):
        start = time.perf_counter_ns()
        for _ in range(repeats):
            result = function()
        samples.append((time.perf_counter_ns() - start) / repeats / 1.0e6)
    return statistics.median(samples), result


def _rgb(power: np.ndarray, side: int) -> np.ndarray:
    rgb = np.moveaxis(power[:3].reshape(3, side, side), 0, -1)
    scale = max(float(np.quantile(rgb, 0.995)), 1.0e-15)
    rgb = np.clip(rgb / scale, 0.0, 1.0)
    rgb = np.power(rgb, 1.0 / 2.2)
    return np.asarray(np.rint(255.0 * rgb), dtype=np.uint8)


def _panel(rgb: np.ndarray, title: str, size: int = 320) -> Image.Image:
    image = Image.fromarray(rgb, "RGB").resize((size, size), Image.Resampling.NEAREST)
    panel = Image.new("RGB", (size, size + 34), (19, 20, 24))
    panel.paste(image, (0, 34))
    ImageDraw.Draw(panel).text((10, 9), title, fill=(235, 238, 244))
    return panel


def run(repeats: int, image_path: Path) -> dict[str, object]:
    if not native_available():
        raise RuntimeError(
            "build the native kernel with `make -C "
            "experiments/photonic_transport_field/native retained`"
        )
    modes = _modes()
    descriptions = (
        ("open-field", 32, ()),
        (
            "single-occluder",
            8,
            (SphereOccluder(np.array((2.0, 0.0, 0.0)), 0.36),),
        ),
        (
            "offset-pair",
            8,
            (
                SphereOccluder(np.array((1.7, -0.28, -0.12)), 0.24),
                SphereOccluder(np.array((2.6, 0.31, 0.24)), 0.22),
            ),
        ),
    )
    records = []
    panels = []
    for name, side, occluders in descriptions:
        bounds, ingress, egress = _boundary(side)
        compile_start = time.perf_counter_ns()
        compiled = compile_child_volume_rank_blocks(
            bounds,
            ingress,
            egress,
            (),
            occluders=occluders,
            admissibility=0.5 if occluders else 0.35,
            minimum_fraction=1.0e-7,
        )
        compile_ms = (time.perf_counter_ns() - compile_start) / 1.0e6
        powers = _source(side)
        oracle = ChildVolumeResponseSystem(
            compiled.topology, execution_backend="python"
        )
        native = ChildVolumeResponseSystem(
            compiled.topology, execution_backend="native"
        )
        oracle_ms, oracle_result = _timing(
            lambda: oracle.evaluate_packed(modes, powers), repeats
        )
        native_ms, native_result = _timing(
            lambda: native.evaluate_packed(modes, powers), repeats
        )
        difference = np.abs(oracle_result.powers - native_result.powers)
        denominator = np.maximum(np.abs(oracle_result.powers), 1.0e-300)
        records.append(
            {
                "scene": name,
                "side": side,
                "ingress_fibers": len(ingress),
                "egress_fibers": len(egress),
                "mode_count": len(modes),
                "retained_blocks": len(compiled.topology.blocks),
                "stored_coefficients": compiled.topology.retained_stored_coefficients,
                "equivalent_pair_coefficients": (
                    compiled.topology.retained_expanded_coefficients
                ),
                "geometry_compile_ms": compile_ms,
                "numpy_packed_ms": oracle_ms,
                "native_packed_ms": native_ms,
                "native_speedup": oracle_ms / native_ms,
                "maximum_absolute_error": float(np.max(difference, initial=0.0)),
                "maximum_relative_error": float(
                    np.max(difference / denominator, initial=0.0)
                ),
                "block_applications": native_result.block_applications,
                "gathered_source_coefficients": (
                    native_result.gathered_source_coefficients
                ),
                "scattered_receiver_coefficients": (
                    native_result.scattered_receiver_coefficients
                ),
                "execution_backend": native_result.execution_backend,
                "geometry_diagnostics": compiled.geometry_diagnostics,
            }
        )
        if name == "open-field":
            panels.append(_panel(_rgb(powers, side), "source boundary"))
        panels.append(_panel(_rgb(native_result.powers, side), name))
    montage = Image.new("RGB", (320 * len(panels), 354), (19, 20, 24))
    for index, panel in enumerate(panels):
        montage.paste(panel, (320 * index, 0))
    image_path.parent.mkdir(parents=True, exist_ok=True)
    montage.save(image_path, optimize=True)
    return {
        "equation": "out[mode,receiver] += v[receiver] * medium(mode) * dot(u,power[mode])",
        "pair_matrix_materialized": False,
        "packet_objects_materialized_in_timed_path": False,
        "timing_statistic": "median_of_5_batches_ms_per_evaluation",
        "repeats_per_batch": repeats,
        "scenes": records,
        "visualization": str(image_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/tmp/native_retained_scenes_m4.json"),
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("/tmp/native_retained_scenes_m4.png"),
    )
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    result = run(args.repeats, args.image)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
