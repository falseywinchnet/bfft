"""Matched gate for transport-coherent characteristic branch selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .branch_posterior_transport_2d import (
    denoise_causal_branch_action_2d,
    denoise_branch_posterior_transport_2d,
)
from .fmmt_certified import denoise_fmmt
from .post_lineage_prolongation_2d import post_lineage_residual_forms_2d
from .probe_continuous_tangent_convergence import CONDITIONS
from .run_2d_denoiser_battery import metrics, sources
from .sample_series import corrupt


def run(size: int, selected_sources: tuple[str, ...], ceiling: int) -> dict:
    catalogue = sources(size)
    rows = []
    for source in selected_sources:
        truth = catalogue[source]
        for condition, kind, amount, density in CONDITIONS:
            observation = (
                truth if kind is None else corrupt(
                    truth, kind, amount=amount, density=density, seed=31013)
            )
            initial = post_lineage_residual_forms_2d(observation)[0][
                "maximum_posterior_branch"]
            transported, diagnostic = denoise_branch_posterior_transport_2d(
                observation, maximum_transports=ceiling)
            causal, causal_diagnostic = denoise_causal_branch_action_2d(
                observation)
            rows.append({
                "source": source,
                "condition": condition,
                "maximum_posterior_branch": metrics(initial, truth),
                "transported_branch": metrics(transported, truth),
                "causal_branch_action": metrics(causal, truth),
                "integrated_fmmt": metrics(
                    denoise_fmmt(observation)[0], truth),
                "accepted_branch_transports": diagnostic[
                    "accepted_branch_transports"],
                "ceiling_hit": diagnostic["branch_transport_ceiling_hit"],
                "causal_branch_change_fraction": causal_diagnostic[
                    "branch_change_fraction"],
            })
    methods = (
        "maximum_posterior_branch", "causal_branch_action",
        "transported_branch", "integrated_fmmt")
    summary = {
        method: {
            key: float(np.mean([row[method][key] for row in rows]))
            for key in rows[0][method]
        }
        for method in methods
    }
    return {
        "purpose": (
            "test whether proper-score Selling transport makes branch identity "
            "coherent without transporting amplitude"
        ),
        "size": int(size),
        "sources": list(selected_sources),
        "transport_ceiling": int(ceiling),
        "summary": summary,
        "ceiling_hits": int(sum(row["ceiling_hit"] for row in rows)),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument(
        "--sources",
        default="tapered hair,geometric interfaces,woven chirps",
    )
    parser.add_argument("--transports", type=int, default=32)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.size,
        tuple(value.strip() for value in args.sources.split(",") if value.strip()),
        args.transports,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "summary": result["summary"],
        "ceiling_hits": result["ceiling_hits"],
        "steps": [
            [row["source"], row["condition"], row["accepted_branch_transports"]]
            for row in result["rows"]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
