"""Gate deblurrer-inspired relative operator closure on unknown image noise."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .cross_chart_transport_closure_2d import (
    denoise_cross_chart_transport_closure_2d,
)
from .fmmt_certified import denoise_fmmt
from .probe_complete_moment_lineage_2d import displacement
from .probe_nuisance_geometry_2d import (
    poisson_observation,
    row_correlated_signal_dependent_observation,
)
from .run_2d_denoiser_battery import metrics, sources
from .sample_series import corrupt


CASES = (
    ("clean", None),
    ("Gaussian 0.10", "gaussian"),
    ("replacement 0.25", "replacement"),
    ("mixed 0.25", "mixed"),
    ("Poisson exposure 16", "poisson"),
    ("row-correlated signal-dependent 0.15", "row"),
)
METHODS = (
    "observation",
    "complete-residual HJ simplex",
    "relative chart closure",
    "source-coverage chart closure",
    "source-coverage closure + FMMT after-pass",
    "integrated FMMT control",
)
METRICS = (
    "mse", "ssim", "variance_ratio", "central_range_ratio",
    "edge_retention", "mean_bias", "observation_displacement_mse",
    "observation_displacement_rms", "observation_displacement_maximum",
    "fraction_moved_over_one_8bit_level",
)


def _observe(truth: np.ndarray, kind: str | None) -> np.ndarray:
    rng = np.random.default_rng(271828)
    if kind is None:
        return truth
    if kind == "gaussian":
        return corrupt(
            truth, "Gaussian additive", amount=0.10, density=0.25,
            seed=271828)
    if kind == "replacement":
        return corrupt(
            truth, "random-value replacement", amount=0.10, density=0.25,
            seed=271828)
    if kind == "mixed":
        return corrupt(
            truth, "mixed replacement + uniform", amount=0.10, density=0.25,
            seed=271828)
    if kind == "poisson":
        return poisson_observation(truth, 16.0, rng)
    return row_correlated_signal_dependent_observation(truth, 0.15, rng)


def run(size: int, phase_count: int, selected: tuple[str, ...]) -> dict:
    catalogue = sources(size)
    rows = []
    for source in selected:
        truth = catalogue[source]
        for condition, kind in CASES:
            observation = _observe(truth, kind)
            closure, closure_diagnostic = (
                denoise_cross_chart_transport_closure_2d(
                    observation,
                    angular_count=4,
                    quantile_count=16,
                    phase_count=phase_count,
                    complete_residual_moment=True,
                ))
            terminal = closure_diagnostic["readouts"]
            source_coverage = terminal[
                "source_coverage_closure_barycenter"]
            fmmt = denoise_fmmt(observation)[0]
            source_then_fmmt = denoise_fmmt(source_coverage)[0]
            estimates = {
                "observation": observation,
                "complete-residual HJ simplex": terminal[
                    "transport_chart_consensus"],
                "relative chart closure": closure,
                "source-coverage chart closure": source_coverage,
                "source-coverage closure + FMMT after-pass": (
                    source_then_fmmt),
                "integrated FMMT control": fmmt,
            }
            rows.append({
                "source": source,
                "condition": condition,
                **{
                    method: {
                        **metrics(estimate, truth),
                        **displacement(estimate, observation),
                    }
                    for method, estimate in estimates.items()
                },
                "closure_diagnostic": closure_diagnostic["closure"],
            })

    def summarize(records: list[dict]) -> dict:
        return {
            method: {
                name: float(np.mean([row[method][name] for row in records]))
                for name in METRICS
            }
            for method in METHODS
        }

    def wins(
        candidate: str, metric: str, maximum: bool = False,
    ) -> dict[str, int]:
        return {
            control: int(sum(
                (row[candidate][metric]
                 > row[control][metric])
                if maximum else
                (row[candidate][metric]
                 < row[control][metric])
                for row in rows
            ))
            for control in (
                "complete-residual HJ simplex", "integrated FMMT control")
        }

    return {
        "purpose": (
            "test whether latent-cancelling disagreement between independent "
            "transport charts is useful uncertainty about the denoising map"
        ),
        "size": size,
        "phase_count": phase_count,
        "sources": list(selected),
        "summary": summarize(rows),
        "by_condition": {
            condition: summarize([
                row for row in rows if row["condition"] == condition])
            for condition, _ in CASES
        },
        "case_wins": {
            candidate: {
                "mse": wins(candidate, "mse"),
                "ssim": wins(candidate, "ssim", maximum=True),
                "edge_retention": wins(
                    candidate, "edge_retention", maximum=True),
            }
            for candidate in (
                "relative chart closure", "source-coverage chart closure",
                "source-coverage closure + FMMT after-pass")
        },
        "closure_state": {
            name: float(np.mean([
                row["closure_diagnostic"][name] for row in rows]))
            for name in (
                "mean_transport_chart_variance",
                "mean_transport_chart_authority",
                "mean_effective_transport_chart_count",
                "mean_observation_displacement_rms",
                "mean_source_coverage_authority",
                "mean_source_noise_scale",
                "mean_source_consensus_gain",
                "mean_source_common_variance",
                "mean_source_coverage_displacement_rms",
            )
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=16)
    parser.add_argument("--phase-count", type=int, default=1)
    parser.add_argument(
        "--sources", default="cameraman,tapered hair,woven chirps")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.size,
        args.phase_count,
        tuple(value.strip() for value in args.sources.split(",") if value.strip()),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "summary": result["summary"],
        "case_wins": result["case_wins"],
        "closure_state": result["closure_state"],
    }, indent=2))


if __name__ == "__main__":
    main()
