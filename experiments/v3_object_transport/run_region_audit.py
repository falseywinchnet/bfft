#!/usr/bin/env python3
"""Run the evidence-only V3 region-complex audit on control images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from experiments.compound_segment_benchmark import _fit_side, _rgb
from experiments.segmenting_v3 import SegmentingV3Config, build_segmenting_v3
from experiments.v3_object_transport.region_complex import (
    build_region_complex,
    summarize_region_complex,
)
from experiments.v3_object_transport.incidence_bundle import (
    build_incidence_bundle,
    summarize_incidence_bundle,
)
from experiments.v3_object_transport.fused_meyer_evidence import (
    build_fused_meyer_evidence,
)


ROOT = Path(__file__).resolve().parents[2]
ASSET_DIR = ROOT / "experiments" / "v3_object_transport" / "assets"
CONTROL_ASSETS = {
    "pikachu_easy": ASSET_DIR / "pikachu_easy.png",
    "pikachu_hard": ASSET_DIR / "pikachu_hard.png",
    "coffee": ASSET_DIR / "coffee.png",
    "astronaut": ASSET_DIR / "astronaut.png",
    "checker": ASSET_DIR / "checker.png",
    "coins": ASSET_DIR / "coins.png",
}


def _load_image(name: str) -> np.ndarray:
    try:
        path = CONTROL_ASSETS[name]
    except KeyError as error:
        raise ValueError(f"unknown frozen control image {name!r}") from error
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0


def _save_rgb(path: Path, value: np.ndarray) -> None:
    pixels = np.rint(np.clip(value, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(pixels, mode="RGB").save(path)


def _stable_colours(count: int) -> np.ndarray:
    value = np.arange(count, dtype=np.uint32)
    value = value * np.uint32(747796405) + np.uint32(2891336453)
    value = (
        ((value >> ((value >> 28) + 4)) ^ value)
        * np.uint32(277803737)
    )
    value = (value >> 22) ^ value
    return 0.08 + 0.90 * np.column_stack((
        value & 255,
        (value >> 8) & 255,
        (value >> 16) & 255,
    )).astype(np.float64) / 255.0


def _save_labels(path: Path, labels: np.ndarray) -> None:
    value = np.asarray(labels, dtype=np.int32)
    count = int(value.max(initial=-1)) + 1
    _save_rgb(path, _stable_colours(count)[value])


def _save_complex(path: Path, complex_: dict) -> None:
    arrays: dict[str, np.ndarray] = {"labels": complex_["labels"]}
    for section in ("node", "edge", "arc", "ancestry"):
        for name, value in complex_[section].items():
            arrays[f"{section}__{name}"] = np.asarray(value)
    for name, value in complex_["empirical"].items():
        if name == "relation_correlation":
            arrays["empirical__relation_correlation"] = np.asarray(
                value["matrix"], dtype=np.float64)
            arrays["empirical__relation_channel_names"] = np.asarray(
                value["names"], dtype="U64")
        else:
            arrays[f"empirical__{name}"] = np.asarray(value)
    topology = complex_["topology"]
    arrays["topology__shape"] = np.asarray(topology["shape"], dtype=np.int32)
    for section in ("edgel", "arc", "junction"):
        for name, value in topology[section].items():
            arrays[f"topology__{section}__{name}"] = np.asarray(value)
    np.savez_compressed(path, **arrays)


def _save_bundle(path: Path, bundle: dict) -> None:
    arrays = {
        f"{section}__{name}": np.asarray(value)
        for section, fields in bundle.items()
        for name, value in fields.items()
    }
    np.savez_compressed(path, **arrays)


def run_image(
    name: str,
    *,
    side: int,
    topology: str,
    output: Path,
    contrast_invert: bool = False,
) -> dict:
    started = time.perf_counter()
    rgb = _fit_side(_load_image(name), side)
    if contrast_invert:
        # Exact RGB complement: a measurement-equivariance control, not a
        # synthetic redesign or an inference-time normalization.
        rgb = 1.0 - rgb
    result = build_segmenting_v3(
        rgb,
        SegmentingV3Config(
            structural_topology=topology,
            structural_allocation_side=min(side, 512),
            compound_segmentation=True,
            region_posterization=True,
            region_family_fusion=True,
            threads=4,
        ),
    )
    fused_meyer = build_fused_meyer_evidence(result["target_lab"])
    leaf_complex = build_region_complex(
        result, rgb, level="leaves", fused_meyer=fused_meyer)
    compound_complex = build_region_complex(
        result, rgb, level="compounds", fused_meyer=fused_meyer)
    leaf_bundle = build_incidence_bundle(leaf_complex)
    compound_bundle = build_incidence_bundle(compound_complex)
    image_dir = output / name
    image_dir.mkdir(parents=True, exist_ok=True)
    _save_rgb(image_dir / "source.png", rgb)
    _save_rgb(image_dir / "reconstruction.png", result["reconstruction_rgb"])
    _save_labels(image_dir / "structural_regions.png", result["labels"])
    _save_labels(image_dir / "texture_atoms.png", result["texture_labels"])
    _save_labels(
        image_dir / "compound_regions.png",
        result["compound_segmentation"]["labels"],
    )
    _save_labels(
        image_dir / "compound_leaves.png",
        result["compound_segmentation"]["leaf_labels"],
    )
    family = result["region_family_fusion"]
    if bool(family.get("enabled", False)):
        _save_labels(image_dir / "historical_family_control.png", family["labels"])
    np.savez_compressed(
        image_dir / "v3_stages.npz",
        structural_labels=np.asarray(result["labels"], dtype=np.int32),
        texture_labels=np.asarray(result["texture_labels"], dtype=np.int32),
        compound_labels=np.asarray(
            result["compound_segmentation"]["labels"], dtype=np.int32),
        compound_leaf_labels=np.asarray(
            result["compound_segmentation"]["leaf_labels"], dtype=np.int32),
        historical_family_labels=np.asarray(
            family.get("labels", result["compound_segmentation"]["labels"]),
            dtype=np.int32,
        ),
    )
    np.savez_compressed(
        image_dir / "fused_meyer_evidence.npz",
        target=fused_meyer["target"],
        cartoon=fused_meyer["cartoon"],
        texture=fused_meyer["texture"],
        residual=fused_meyer["residual"],
    )
    _save_complex(image_dir / "leaf_region_complex.npz", leaf_complex)
    _save_complex(image_dir / "compound_region_complex.npz", compound_complex)
    _save_bundle(image_dir / "leaf_incidence_bundle.npz", leaf_bundle)
    _save_bundle(image_dir / "compound_incidence_bundle.npz", compound_bundle)
    report = {
        "image": name,
        "shape": list(rgb.shape[:2]),
        "topology": topology,
        "contrast_invert": bool(contrast_invert),
        "model": result["model"],
        "v3": {
            "structural_regions": int(result["labels"].max(initial=-1)) + 1,
            "texture_atoms": int(result["texture_labels"].max(initial=-1)) + 1,
            "compound_regions": int(
                result["compound_segmentation"]["compound_count"]),
            "compound_leaves": int(
                result["compound_segmentation"]["leaf_count"]),
            "historical_family_regions": int(
                family.get("family_count", compound_complex["region_count"])),
            "psnr": float(result["record"]["psnr"]),
            "timing_ms": {
                key: float(value) for key, value in result["timing"].items()
            },
        },
        "fused_meyer": {
            "outer_passes": int(fused_meyer["outer_passes"]),
            "lambda": float(fused_meyer["lambda"]),
            "mu": float(fused_meyer["mu"]),
            "recomposition_max_abs": float(
                fused_meyer["recomposition_max_abs"]),
            "residual_rms": float(fused_meyer["residual_rms"]),
        },
        "leaf_region_complex": summarize_region_complex(leaf_complex),
        "compound_region_complex": summarize_region_complex(compound_complex),
        "leaf_incidence_bundle": summarize_incidence_bundle(leaf_bundle),
        "compound_incidence_bundle": summarize_incidence_bundle(compound_bundle),
        "wall_seconds": time.perf_counter() - started,
    }
    (image_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path("/tmp/v3_object_transport_audit"))
    parser.add_argument("--side", type=int, default=256)
    parser.add_argument(
        "--topology", choices=("half_cartoon", "canonical_v2"),
        default="half_cartoon",
    )
    parser.add_argument(
        "--images", nargs="+",
        default=("pikachu_hard", "coffee", "astronaut", "checker", "coins"),
    )
    parser.add_argument(
        "--contrast-invert", action="store_true",
        help="replace RGB by its exact complement before V3 (control only)",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    reports = []
    for name in args.images:
        print(f"auditing {name}", flush=True)
        report = run_image(
            name, side=args.side, topology=args.topology, output=args.out,
            contrast_invert=args.contrast_invert)
        reports.append(report)
        print(
            f"  {report['v3']['compound_regions']} compound regions, "
            f"{report['leaf_region_complex']['interfaces']} leaf interfaces, "
            f"{report['wall_seconds']:.2f} s",
            flush=True,
        )
    summary = {
        "purpose": (
            "evidence-only V3 region-complex audit; historical family fusion "
            "is a control and no object IDs are inferred"
        ),
        "side": args.side,
        "topology": args.topology,
        "contrast_invert": bool(args.contrast_invert),
        "images": reports,
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
