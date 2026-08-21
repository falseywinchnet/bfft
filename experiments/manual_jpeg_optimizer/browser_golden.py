"""Freeze deterministic native stages for the browser structural port."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .core import (
    JPEGConfig,
    _apply_preprocess_basis,
    _prepare_preprocess_basis,
    block_dct,
    inverse_block_dct,
    quality_table,
    rgb_to_ycc,
    ycc_to_rgb,
)


HERE = Path(__file__).resolve().parent
DEFAULT_FIXTURE = HERE / "fixtures" / "browser_structural_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = np.asarray([
        [[0, 0, 0], [255, 255, 255], [255, 0, 0]],
        [[0, 255, 0], [0, 0, 255], [17, 93, 201]],
        [[254, 128, 1], [64, 192, 129], [11, 37, 73]],
    ], dtype=np.float64)
    y, x = np.mgrid[:8, :8]
    plane = ((17 * x + 31 * y + 3 * x * y + 19) % 256).astype(np.float64)
    sy, sx = np.mgrid[:16, :24]
    structural = np.stack((
        (17 * sx + 7 * sy + 31 * ((sx // 6) % 2)) % 256,
        (5 * sx + 19 * sy + 47 * ((sy // 4) % 2)) % 256,
        (13 * sx + 11 * sy + 29 * (((sx + sy) // 5) % 2)) % 256,
    ), axis=-1).astype(np.float64)
    return rgb, plane, structural


def build_fixture() -> dict[str, Any]:
    rgb, plane, structural = _fixture_inputs()
    ycc = rgb_to_ycc(rgb)
    coefficients = block_dct(plane)[0, 0]
    recovered_plane = inverse_block_dct(coefficients[None, None], plane.shape)
    config = JPEGConfig(
        cartoon_sigma=1.2,
        region_threshold=0.58,
        chroma_projection=0.35,
        luma_texture_shrink=0.12,
        phase_degrees=0.0,
    )
    basis = _prepare_preprocess_basis(structural, config)
    projected = _apply_preprocess_basis(basis, config)
    core_path = HERE / "core.py"
    generator_path = Path(__file__).resolve()
    return {
        "schema": "bfft.manual-jpeg.browser-structural-golden",
        "version": 1,
        "authority": {
            "repository": "bfft",
            "core": "experiments/manual_jpeg_optimizer/core.py",
            "core_sha256": _sha256(core_path),
            "generator": "experiments/manual_jpeg_optimizer/browser_golden.py",
            "generator_sha256": _sha256(generator_path),
        },
        "scope": [
            "rgb_to_ycc", "ycc_to_rgb", "quality_table", "block_dct_8x8",
            "inverse_block_dct_8x8", "gaussian_cartoon_texture",
            "sobel_flat_luma_weight", "connected_ownership_regions",
            "regional_chroma_covariance_projection",
        ],
        "excluded_scope": [
            "jpeg_encoding", "source_quality_inference", "decoded_ssim",
            "decoded_edge_psnr", "spatial_frequency_transport",
            "jpegli_dead_zone_and_trellis",
        ],
        "tolerances": {
            "color_max_abs": 3e-4,
            "dct_max_abs": 1e-9,
            "structural_float_max_abs": 2e-8,
            "labels": "exact_integer",
            "quality_table": "exact_integer",
        },
        "color_case": {
            "shape": list(rgb.shape), "rgb": rgb.tolist(), "ycc": ycc.tolist(),
            "recovered_rgb": ycc_to_rgb(ycc).tolist(),
        },
        "quality_cases": [
            {"quality": q,
             "luma": quality_table(q, False).astype(int).tolist(),
             "chroma": quality_table(q, True).astype(int).tolist()}
            for q in (1, 37, 50, 83, 100)
        ],
        "dct_case": {
            "shape": list(plane.shape), "plane": plane.tolist(),
            "coefficients": coefficients.tolist(),
            "recovered_plane": recovered_plane.tolist(),
        },
        "structural_case": {
            "shape": list(structural.shape), "rgb": structural.tolist(),
            "config": {
                "cartoon_sigma": config.cartoon_sigma,
                "region_threshold": config.region_threshold,
                "chroma_projection": config.chroma_projection,
                "luma_texture_shrink": config.luma_texture_shrink,
                "phase_degrees": config.phase_degrees,
            },
            "cartoon_ycc": basis.cartoon.tolist(),
            "texture_ycc": basis.texture.tolist(),
            "block_labels": basis.block_labels.tolist(),
            "preferred_angle": basis.preferred_angle.tolist(),
            "flat_luma_weight": basis.flat_luma_weight.tolist(),
            "smooth_chroma_texture": basis.smooth_chroma_texture.tolist(),
            "projected_rgb": projected.rgb.tolist(),
        },
    }


def verify_fixture(path: Path = DEFAULT_FIXTURE) -> None:
    stored = json.loads(path.read_text())
    current = build_fixture()
    for key in ("schema", "version", "authority", "scope", "excluded_scope", "tolerances"):
        if stored[key] != current[key]:
            raise AssertionError(f"browser fixture {key} is stale")
    tolerance = current["tolerances"]
    cases = [
        ("color_case.ycc", stored["color_case"]["ycc"], current["color_case"]["ycc"], tolerance["color_max_abs"]),
        ("color_case.recovered_rgb", stored["color_case"]["recovered_rgb"], current["color_case"]["recovered_rgb"], tolerance["color_max_abs"]),
        ("dct_case.coefficients", stored["dct_case"]["coefficients"], current["dct_case"]["coefficients"], tolerance["dct_max_abs"]),
        ("dct_case.recovered_plane", stored["dct_case"]["recovered_plane"], current["dct_case"]["recovered_plane"], tolerance["dct_max_abs"]),
    ]
    for field in ("cartoon_ycc", "texture_ycc", "preferred_angle", "flat_luma_weight", "smooth_chroma_texture", "projected_rgb"):
        cases.append((f"structural_case.{field}", stored["structural_case"][field], current["structural_case"][field], tolerance["structural_float_max_abs"]))
    for name, observed, expected, atol in cases:
        try:
            np.testing.assert_allclose(observed, expected, rtol=0, atol=atol)
        except AssertionError as error:
            raise AssertionError(f"browser fixture {name} changed") from error
    if stored["quality_cases"] != current["quality_cases"]:
        raise AssertionError("browser fixture quality tables changed")
    if stored["structural_case"]["block_labels"] != current["structural_case"]["block_labels"]:
        raise AssertionError("browser fixture ownership labels changed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        verify_fixture(args.output)
        print(f"verified {args.output}")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(build_fixture(), indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

