"""Matched gate for raw versus variance-debiased transported residual loops."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .fmmt_certified import denoise_fmmt
from .post_lineage_prolongation_2d import post_lineage_residual_forms_2d
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
            forms, diagnostic = post_lineage_residual_forms_2d(observation)
            rows.append({
                "source": source,
                "condition": condition,
                "observation": metrics(observation, truth),
                **{name: metrics(value, truth) for name, value in forms.items()},
                "integrated_fmmt": metrics(
                    denoise_fmmt(observation)[0], truth),
                "global_descent_coefficient": diagnostic[
                    "global_descent_coefficient"],
                "mean_authority": diagnostic["mean_authority"],
            })
    methods = (
        "observation", "predictive_base", "maximum_posterior_branch",
        "transported_residual", "covariance_residual", "integrated_fmmt",
    )
    summary = {
        method: {
            key: float(np.mean([row[method][key] for row in rows]))
            for key in rows[0][method]
        }
        for method in methods
    }
    corrupted = [row for row in rows if row["condition"] != "clean"]
    summary_corrupted = {
        method: {
            key: float(np.mean([row[method][key] for row in corrupted]))
            for key in rows[0][method]
        }
        for method in methods
    }
    return {
        "purpose": (
            "separate lineage loop closure from finite-population covariance "
            "authority on one fixed predictive base"
        ),
        "size": int(size),
        "sources": list(selected_sources),
        "summary": summary,
        "summary_corrupted": summary_corrupted,
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
        "summary_corrupted": result["summary_corrupted"],
    }, indent=2))


if __name__ == "__main__":
    main()
