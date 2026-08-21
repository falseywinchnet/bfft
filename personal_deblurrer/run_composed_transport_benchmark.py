#!/usr/bin/env python3
"""Benchmark one-stage and double-radial consolidated observation transport."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from denoiser.run_2d_denoiser_battery import metrics, sources

from .composed_transport import (
    compose_positive_transports,
    radial_scale_measure,
    refine_consolidated_transport,
)


def _score(image: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    record = metrics(image, truth)
    mse = max(float(record["mse"]), np.finfo(float).tiny)
    return {**record, "psnr": float(-10.0 * math.log10(mse))}


def _mean(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def _radial_alignment(transport) -> float:
    jet = transport.local_moment_jet()
    height, width = transport.shape
    yy, xx = np.mgrid[:height, :width]
    center_x = 0.5 * (width - 1)
    center_y = 0.5 * (height - 1)
    radial = np.stack((xx - center_x, yy - center_y), axis=-1)
    radius = np.linalg.norm(radial, axis=-1)
    radial /= np.maximum(radius[..., None], 1.0)
    alignment = np.abs(np.sum(jet.principal_direction_xy * radial, axis=-1))
    margin = max(int(round(0.08 * min(height, width))), 4)
    supported = (
        jet.supported & (radius >= margin)
        & (yy >= margin) & (yy < height - margin)
        & (xx >= margin) & (xx < width - margin)
    )
    return float(np.mean(alignment[supported]))


def run(size: int, passes: int, output: Path) -> dict[str, object]:
    started = time.perf_counter()
    shape = (int(size), int(size))
    stage_4 = radial_scale_measure(
        shape, fractional_extent=0.04).to_transport(shape)
    stage_5 = radial_scale_measure(
        shape, fractional_extent=0.05).to_transport(shape)
    stage_8 = radial_scale_measure(
        shape, fractional_extent=0.08).to_transport(shape)
    cases = {
        "single_radial_005": ([stage_5], stage_5),
        "double_radial_005_005": (
            [stage_5, stage_5],
            compose_positive_transports(stage_5, stage_5),
        ),
        "double_radial_004_008": (
            [stage_4, stage_8],
            compose_positive_transports(stage_4, stage_8),
        ),
    }
    rows: list[dict[str, object]] = []
    audits: dict[str, object] = {}
    closure_rms: dict[str, float] = {}
    for case, (stages, consolidated) in cases.items():
        maximum_closure = 0.0
        for source, truth in sources(size).items():
            sequential = truth
            for stage in stages:
                sequential = stage.forward(sequential)
            direct = consolidated.forward(truth)
            maximum_closure = max(
                maximum_closure,
                float(np.sqrt(np.mean((direct - sequential) ** 2))),
            )
            result = refine_consolidated_transport(
                sequential,
                consolidated,
                passes=passes,
                ratio_limit=4.0,
            )
            for method, image in (
                ("observation", sequential),
                ("normalized_adjoint_seed", result.adjoint_seed),
                ("consolidated_positive_inverse", result.image),
            ):
                rows.append({
                    "case": case,
                    "source": source,
                    "method": method,
                    **_score(image, truth),
                })
            audits[case] = result.diagnostics
        closure_rms[case] = maximum_closure

    score_keys = (
        "mse", "ssim", "variance_ratio", "central_range_ratio",
        "edge_retention", "mean_bias", "psnr",
    )
    summary: dict[str, object] = {}
    for case, (_, consolidated) in cases.items():
        methods: dict[str, object] = {}
        for method in (
            "observation", "normalized_adjoint_seed",
            "consolidated_positive_inverse",
        ):
            subset = [
                {key: float(row[key]) for key in score_keys}
                for row in rows
                if row["case"] == case and row["method"] == method
            ]
            methods[method] = _mean(subset)
        summary[case] = {
            "methods": methods,
            "contribution_count": consolidated.contribution_count,
            "storage_bytes": consolidated.storage_bytes,
            "sequential_to_consolidated_rms_max": closure_rms[case],
            "mean_intrinsic_radial_direction_alignment": _radial_alignment(
                consolidated),
        }
    result = {
        "experiment": "consolidated_double_radial_observation_transport_v1",
        "scope": (
            "known-measure representation_and_inverse_gate_not_blind_estimation"),
        "operator_decomposition": False,
        "family_classification": False,
        "size": int(size),
        "passes": int(passes),
        "sources": list(sources(size)),
        "summary": summary,
        "audits": audits,
        "rows": rows,
        "wall_seconds": time.perf_counter() - started,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--passes", type=int, default=64)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("personal_deblurrer/composed_transport_results"),
    )
    args = parser.parse_args()
    result = run(args.size, args.passes, args.out)
    for case, record in result["summary"].items():
        print(case)
        for method, score in record["methods"].items():
            print(
                f"  {method:32s} {score['psnr']:7.3f} dB  "
                f"SSIM {score['ssim']:.4f}"
            )
        print(
            f"  closure {record['sequential_to_consolidated_rms_max']:.3e}; "
            f"radial alignment {record['mean_intrinsic_radial_direction_alignment']:.5f}"
        )
    print(f"wall: {result['wall_seconds']:.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
