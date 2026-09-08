"""Held-out 2x interpolation benchmark over the public ``skimage.data`` catalog.

No interpolated reference image is manufactured.  Every truth patch is cut
directly from a native scikit-image raster.  Its even/even pixels form the
coarse source lattice; MSE is measured on the remaining native pixels.  A
second, explicitly labelled anti-aliased acquisition uses ``positive_restrict``
before reconstruction.  Results retain every raster and patch so aggregate
figures cannot substitute for the paired evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing
from pathlib import Path
import time
from typing import Iterator

import numpy as np

from experiments.self_geometric_harmonic_interpolation import (
    characteristic_transport_resize,
    cosine_hermite_characteristic_resize,
    harmonic_resize,
    lanczos_resize,
    positive_restrict,
)


Array = np.ndarray


# Parameter-free public raster providers in scikit-image 0.26.  Keeping the
# catalog explicit makes additions, aliases, fetch failures, and exclusions
# visible in the result rather than silently changing the benchmark population.
SKIMAGE_DATA_CATALOG = (
    "astronaut",
    "brain",
    "brick",
    "camera",
    "cat",
    "cell",
    "cells3d",
    "checkerboard",
    "chelsea",
    "clock",
    "coffee",
    "coins",
    "colorwheel",
    "eagle",
    "grass",
    "gravel",
    "horse",
    "hubble_deep_field",
    "human_mitosis",
    "immunohistochemistry",
    "kidney",
    "lfw_subset",
    "lily",
    "logo",
    "microaneurysms",
    "moon",
    "nickel_solidification",
    "page",
    "palisades_of_vogt",
    "protein_transport",
    "retina",
    "rocket",
    "shepp_logan_phantom",
    "skin",
    "stereo_motorcycle",
    "text",
    "vortex",
)


def _array_rasters(name: str, value: object) -> Iterator[tuple[str, Array]]:
    """Expand structured images and stacks into named two-dimensional rasters."""

    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from _array_rasters(f"{name}[{index}]", item)
        return
    array = np.asarray(value)
    if array.ndim == 2:
        yield name, array
        return
    if array.ndim >= 3 and array.shape[-1] in (3, 4):
        leading = array.shape[:-3]
        if not leading:
            yield name, array[..., :3]
            return
        for index in np.ndindex(leading):
            yield f"{name}{index}", array[index][..., :3]
        return
    if array.ndim >= 3:
        leading = array.shape[:-2]
        for index in np.ndindex(leading):
            yield f"{name}{index}", array[index]
        return
    raise ValueError(f"{name} is not a 2-D raster or raster stack: {array.shape}")


def _unit_float_image(value: Array) -> tuple[Array, str]:
    """Normalize an intensity raster without changing its spatial samples."""

    array = np.asarray(value)
    if not np.all(np.isfinite(array)):
        raise ValueError("contains non-finite values")
    if np.issubdtype(array.dtype, np.bool_):
        return array.astype(np.float64), "boolean_to_0_1"
    if np.issubdtype(array.dtype, np.integer):
        information = np.iinfo(array.dtype)
        if information.min < 0:
            result = (array.astype(np.float64) - information.min) / (
                information.max - information.min
            )
        else:
            result = array.astype(np.float64) / information.max
        return result, f"dtype_range_{array.dtype}_to_0_1"
    result = array.astype(np.float64)
    low = float(np.min(result))
    high = float(np.max(result))
    if 0.0 <= low and high <= 1.0:
        return result, "native_float_0_1"
    if high == low:
        return np.zeros_like(result), "constant_float_to_0"
    return (result - low) / (high - low), "finite_minmax_to_0_1"


def _patch_side(shape: tuple[int, int], requested: int, scale: int) -> int:
    """Largest ``scale * 2**k + 1`` side admitted by the native raster."""

    maximum = min(shape[0], shape[1], requested)
    intervals = 1
    while scale * (2 * intervals) + 1 <= maximum:
        intervals *= 2
    side = scale * intervals + 1
    if side > maximum:
        raise ValueError(f"raster is too small for a {scale}x nested patch")
    return side


def _patch_origins(
    shape: tuple[int, int], side: int, maximum_count: int
) -> list[tuple[int, int]]:
    """Deterministic spatially stratified patch origins, including boundaries."""

    if maximum_count < 1:
        raise ValueError("maximum patch count must be positive")
    rows = max(1, int(np.floor(np.sqrt(maximum_count))))
    columns = max(1, maximum_count // rows)
    ys = np.rint(np.linspace(0, shape[0] - side, rows)).astype(int)
    xs = np.rint(np.linspace(0, shape[1] - side, columns)).astype(int)
    return [(int(y), int(x)) for y in np.unique(ys) for x in np.unique(xs)][
        :maximum_count
    ]


def _held_out_mask(shape: tuple[int, int], scale: int) -> Array:
    mask = np.ones(shape, dtype=bool)
    mask[::scale, ::scale] = False
    return mask


def _mse(value: Array, truth: Array, mask: Array | None = None) -> float:
    residual = np.asarray(value, dtype=np.float64) - np.asarray(
        truth, dtype=np.float64
    )
    if residual.ndim == 3 and mask is not None:
        residual = residual[mask, :]
    elif mask is not None:
        residual = residual[mask]
    return float(np.mean(residual * residual))


def _global_overshoot(value: Array, source: Array) -> float:
    low = np.min(source, axis=(0, 1), keepdims=True)
    high = np.max(source, axis=(0, 1), keepdims=True)
    return max(
        float(np.max(low - value, initial=0.0)),
        float(np.max(value - high, initial=0.0)),
    )


METHODS = {
    "bilinear": lambda source: harmonic_resize(source, 2, adaptive=False),
    "spatial_characteristic": lambda source: characteristic_transport_resize(
        source, 2
    ),
    "cosine_hermite_transport": lambda source: (
        cosine_hermite_characteristic_resize(source, 2)
    ),
    "lanczos3": lambda source: lanczos_resize(source, 2, radius=3),
}

CANDIDATE_METHOD = "cosine_hermite_transport"


def evaluate_patch(truth: Array) -> dict[str, object]:
    """Evaluate exact-decimation and anti-aliased 2x acquisition protocols."""

    truth = np.asarray(truth, dtype=np.float64)
    exact_source = truth[::2, ::2]
    antialiased_source = positive_restrict(truth, 2)
    held_out = _held_out_mask(truth.shape[:2], 2)
    protocols: dict[str, object] = {}
    for protocol, source in (
        ("exact_sublattice", exact_source),
        ("positive_antialiased", antialiased_source),
    ):
        records: dict[str, object] = {}
        for method, operation in METHODS.items():
            started = time.perf_counter()
            reconstruction = operation(source)
            runtime_seconds = time.perf_counter() - started
            returned = positive_restrict(reconstruction, 2)
            records[method] = {
                "held_out_mse": _mse(reconstruction, truth, held_out),
                "all_pixel_mse": _mse(reconstruction, truth),
                "source_return_mse": _mse(returned, source),
                "maximum_sample_error": float(
                    np.max(np.abs(reconstruction[::2, ::2] - source))
                ),
                "global_source_range_overshoot": _global_overshoot(
                    reconstruction, source
                ),
                "runtime_seconds": runtime_seconds,
            }
        candidate = records[CANDIDATE_METHOD]
        for reference in ("bilinear", "spatial_characteristic", "lanczos3"):
            candidate[f"held_out_mse_delta_vs_{reference}"] = (
                candidate["held_out_mse"] - records[reference]["held_out_mse"]
            )
        protocols[protocol] = records
    return protocols


def _evaluate_record(task: dict[str, object]) -> dict[str, object]:
    """Worker entry point; patch order is preserved by ``Pool.map``."""

    patch = task.pop("patch")
    task["protocols"] = evaluate_patch(patch)
    return task


def _summarize(records: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for protocol in ("exact_sublattice", "positive_antialiased"):
        protocol_summary: dict[str, object] = {}
        for method in METHODS:
            values = np.array([
                record["protocols"][protocol][method]["held_out_mse"]
                for record in records
            ])
            protocol_summary[method] = {
                "raster_patch_count": int(values.size),
                "macro_mean_held_out_mse": float(np.mean(values)),
                "median_held_out_mse": float(np.median(values)),
                "total_held_out_sse": float(np.sum([
                    record["protocols"][protocol][method]["held_out_mse"]
                    * record["held_out_scalar_count"]
                    for record in records
                ])),
                "total_held_out_scalar_count": int(np.sum([
                    record["held_out_scalar_count"] for record in records
                ])),
            }
            protocol_summary[method]["pixel_weighted_held_out_mse"] = (
                protocol_summary[method]["total_held_out_sse"]
                / protocol_summary[method]["total_held_out_scalar_count"]
            )
        candidate_values = np.array([
            record["protocols"][protocol][CANDIDATE_METHOD]["held_out_mse"]
            for record in records
        ])
        for reference in ("bilinear", "spatial_characteristic", "lanczos3"):
            reference_values = np.array([
                record["protocols"][protocol][reference]["held_out_mse"]
                for record in records
            ])
            delta = candidate_values - reference_values
            protocol_summary[f"{CANDIDATE_METHOD}_vs_{reference}"] = {
                "wins": int(np.sum(delta < 0.0)),
                "ties": int(np.sum(delta == 0.0)),
                "losses": int(np.sum(delta > 0.0)),
                "mean_paired_mse_delta": float(np.mean(delta)),
                "median_paired_mse_delta": float(np.median(delta)),
                "delta_quantiles_05_25_50_75_95": [
                    float(value)
                    for value in np.quantile(delta, (0.05, 0.25, 0.5, 0.75, 0.95))
                ],
            }
        providers = sorted({record["provider"] for record in records})
        provider_means = {
            method: np.array([
                np.mean([
                    record["protocols"][protocol][method]["held_out_mse"]
                    for record in records
                    if record["provider"] == provider
                ])
                for provider in providers
            ])
            for method in METHODS
        }
        protocol_summary["provider_balanced"] = {
            "provider_count": len(providers),
            "macro_mean_held_out_mse": {
                method: float(np.mean(values))
                for method, values in provider_means.items()
            },
            f"{CANDIDATE_METHOD}_paired_results": {
                reference: {
                    "wins": int(np.sum(
                        provider_means[CANDIDATE_METHOD]
                        < provider_means[reference]
                    )),
                    "ties": int(np.sum(
                        provider_means[CANDIDATE_METHOD]
                        == provider_means[reference]
                    )),
                    "losses": int(np.sum(
                        provider_means[CANDIDATE_METHOD]
                        > provider_means[reference]
                    )),
                    "mean_paired_mse_delta": float(np.mean(
                        provider_means[CANDIDATE_METHOD]
                        - provider_means[reference]
                    )),
                }
                for reference in (
                    "bilinear", "spatial_characteristic", "lanczos3"
                )
            },
        }
        summary[protocol] = protocol_summary
    return summary


def run_catalog_benchmark(
    *,
    truth_side: int = 65,
    patches_per_raster: int = 1,
    workers: int = 1,
) -> dict[str, object]:
    import skimage
    from skimage import data

    inventory: list[dict[str, object]] = []
    tasks: list[dict[str, object]] = []
    seen_hashes: dict[str, str] = {}
    for provider in SKIMAGE_DATA_CATALOG:
        try:
            value = getattr(data, provider)()
        except Exception as error:
            inventory.append({
                "provider": provider,
                "status": "load_error",
                "detail": f"{type(error).__name__}: {error}",
            })
            continue
        raster_count = 0
        admitted_count = 0
        for raster_name, raw in _array_rasters(provider, value):
            raster_count += 1
            digest = hashlib.sha256(np.ascontiguousarray(raw).view(np.uint8)).hexdigest()
            if digest in seen_hashes:
                inventory.append({
                    "provider": provider,
                    "raster": raster_name,
                    "status": "duplicate_alias",
                    "duplicate_of": seen_hashes[digest],
                })
                continue
            seen_hashes[digest] = raster_name
            try:
                image, normalization = _unit_float_image(raw)
                side = _patch_side(image.shape[:2], truth_side, 2)
                origins = _patch_origins(
                    image.shape[:2], side, patches_per_raster
                )
            except Exception as error:
                inventory.append({
                    "provider": provider,
                    "raster": raster_name,
                    "status": "excluded",
                    "detail": f"{type(error).__name__}: {error}",
                })
                continue
            admitted_count += 1
            inventory.append({
                "provider": provider,
                "raster": raster_name,
                "status": "evaluated",
                "native_shape": list(raw.shape),
                "truth_patch_side": side,
                "patch_count": len(origins),
                "normalization": normalization,
            })
            for patch_index, (y, x) in enumerate(origins):
                patch = image[y : y + side, x : x + side]
                channels = 1 if patch.ndim == 2 else patch.shape[2]
                held_out_pixels = side * side - ((side + 1) // 2) ** 2
                tasks.append({
                    "provider": provider,
                    "raster": raster_name,
                    "patch_index": patch_index,
                    "origin_yx": [y, x],
                    "truth_shape": list(patch.shape),
                    "held_out_scalar_count": held_out_pixels * channels,
                    "patch": np.ascontiguousarray(patch),
                })
        if raster_count == 0:
            inventory.append({
                "provider": provider,
                "status": "excluded",
                "detail": "provider returned no rasters",
            })
    if workers < 1:
        raise ValueError("workers must be positive")
    if workers == 1:
        records = [_evaluate_record(task) for task in tasks]
    else:
        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            records = pool.map(_evaluate_record, tasks)
    return {
        "benchmark": "native-pixel held-out 2x interpolation",
        "skimage_version": skimage.__version__,
        "catalog_provider_count": len(SKIMAGE_DATA_CATALOG),
        "requested_maximum_truth_side": truth_side,
        "patches_per_raster": patches_per_raster,
        "workers": workers,
        "reference_definition": (
            "Native skimage pixels are truth; even/even pixels are the exact "
            "source lattice; held-out MSE excludes source sites."
        ),
        "inventory": inventory,
        "records": records,
        "summary": _summarize(records),
    }


def _write_flat_csv(path: Path, payload: dict[str, object]) -> None:
    fields = [
        "provider", "raster", "patch_index", "origin_y", "origin_x",
        "protocol", "method", "held_out_mse", "all_pixel_mse",
        "source_return_mse", "maximum_sample_error",
        "global_source_range_overshoot",
        "runtime_seconds",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in payload["records"]:
            for protocol, methods in record["protocols"].items():
                for method, metrics in methods.items():
                    writer.writerow({
                        "provider": record["provider"],
                        "raster": record["raster"],
                        "patch_index": record["patch_index"],
                        "origin_y": record["origin_yx"][0],
                        "origin_x": record["origin_yx"][1],
                        "protocol": protocol,
                        "method": method,
                        **{
                            key: metrics[key]
                            for key in fields[7:]
                        },
                    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth-side", type=int, default=65)
    parser.add_argument("--patches-per-raster", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/tmp/skimage_interpolation_benchmark.json"),
    )
    args = parser.parse_args()
    payload = run_catalog_benchmark(
        truth_side=args.truth_side,
        patches_per_raster=args.patches_per_raster,
        workers=args.workers,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    _write_flat_csv(args.out.with_suffix(".csv"), payload)
    print(json.dumps(payload["summary"], indent=2))
    failures = [
        row for row in payload["inventory"]
        if row["status"] in ("load_error", "excluded")
    ]
    print(f"evaluated records: {len(payload['records'])}")
    print(f"load/exclusion records: {len(failures)}")


if __name__ == "__main__":
    main()
