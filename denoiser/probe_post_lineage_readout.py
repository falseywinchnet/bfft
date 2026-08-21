"""Quality gate for the one-pass post-lineage predictive section."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .fmmt_certified import denoise_fmmt
from .post_lineage_prolongation_2d import (
    denoise_post_lineage_prolongation_2d,
    denoise_post_lineage_residual_2d,
)
from .probe_continuous_tangent_convergence import CONDITIONS
from .run_2d_denoiser_battery import metrics, sources
from .sample_series import corrupt


def run(size: int, selected_sources: tuple[str, ...]) -> dict:
    catalogue = sources(size)
    rows = []
    for source in selected_sources:
        truth = catalogue[source]
        for condition, kind, amount, density in CONDITIONS:
            observation = (
                truth if kind is None else corrupt(
                    truth, kind, amount=amount, density=density, seed=31013)
            )
            estimate, diagnostic = denoise_post_lineage_prolongation_2d(
                observation)
            residual_estimate, residual_diagnostic = (
                denoise_post_lineage_residual_2d(observation))
            rows.append({
                "source": source,
                "condition": condition,
                "observation": metrics(observation, truth),
                "post_lineage_section": metrics(estimate, truth),
                "post_lineage_residual": metrics(residual_estimate, truth),
                "integrated_fmmt": metrics(
                    denoise_fmmt(observation)[0], truth),
                "prolongation_implied_support": float(
                    diagnostic["prolongation"]["implied_support"]),
                "maximum_target_self_lineage": diagnostic[
                    "maximum_target_self_lineage"],
                "returned_residual_energy": residual_diagnostic[
                    "returned_residual_energy"],
            })
    methods = (
        "observation", "post_lineage_section", "post_lineage_residual",
        "integrated_fmmt",
    )
    summary = {
        method: {
            key: float(np.mean([row[method][key] for row in rows]))
            for key in rows[0][method]
        }
        for method in methods
    }
    by_condition = {
        condition: {
            method: {
                key: float(np.mean([
                    row[method][key] for row in rows
                    if row["condition"] == condition
                ]))
                for key in rows[0][method]
            }
            for method in methods
        }
        for condition, *_rest in CONDITIONS
    }
    return {
        "purpose": (
            "test whether the initial transported predictive section is a "
            "structure-preserving one-pass denoising readout"
        ),
        "size": int(size),
        "sources": list(selected_sources),
        "summary": summary,
        "by_condition": by_condition,
        "maximum_target_self_lineage": float(max(
            row["maximum_target_self_lineage"] for row in rows)),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument(
        "--sources",
        default="tapered hair,geometric interfaces,woven chirps",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.size,
        tuple(value.strip() for value in args.sources.split(",") if value.strip()),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "summary": result["summary"],
        "by_condition": result["by_condition"],
    }, indent=2))


if __name__ == "__main__":
    main()
