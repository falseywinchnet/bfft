"""Gate explicit zero/nonzero residual components on causal HJ transport."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .causal_information_lineage_2d import (
    causal_information_phase_integrated_readouts_2d,
)
from .fmmt_certified import denoise_fmmt
from .probe_complete_moment_lineage_2d import READOUT, displacement
from .probe_nuisance_geometry_2d import (
    poisson_observation,
    row_correlated_signal_dependent_observation,
)
from .run_2d_denoiser_battery import metrics, sources
from .sample_series import corrupt
from .zero_residual_component_2d import (
    phase_integrated_zero_residual_component_family_2d,
)


def run(size: int, phase_count: int, selected: tuple[str, ...]) -> dict:
    catalogue = sources(size)
    cases = (
        ("clean", None),
        ("Gaussian 0.10", "gaussian"),
        ("replacement 0.25", "replacement"),
        ("mixed 0.25", "mixed"),
        ("Poisson exposure 16", "poisson"),
        ("row-correlated signal-dependent 0.15", "row"),
    )
    rows = []
    for source in selected:
        truth = catalogue[source]
        for condition, kind in cases:
            rng = np.random.default_rng(271828)
            if kind is None:
                observation = truth
            elif kind == "gaussian":
                observation = corrupt(
                    truth, "Gaussian additive", amount=0.10, density=0.25,
                    seed=271828)
            elif kind == "replacement":
                observation = corrupt(
                    truth, "random-value replacement", amount=0.10,
                    density=0.25, seed=271828)
            elif kind == "mixed":
                observation = corrupt(
                    truth, "mixed replacement + uniform", amount=0.10,
                    density=0.25, seed=271828)
            elif kind == "poisson":
                observation = poisson_observation(truth, 16.0, rng)
            else:
                observation = row_correlated_signal_dependent_observation(
                    truth, 0.15, rng)
            complete, _complete_diagnostic = (
                causal_information_phase_integrated_readouts_2d(
                    observation,
                    angular_count=4,
                    quantile_count=16,
                    phase_count=phase_count,
                    complete_residual_moment=True,
                ))
            component_family, component_diagnostics = (
                phase_integrated_zero_residual_component_family_2d(
                    observation,
                    angular_count=4,
                    quantile_count=16,
                    phase_count=phase_count,
                    complete_residual_moment=True,
                ))
            mean_component = component_family["mean"]
            complete_component = component_family["complete"]
            self_consistent_component = component_family["self_consistent"]
            transport_component = component_family["transport_uncertain"]
            cavity_component = component_family["observation_cavity"]
            root_component = component_family["root_resolved"]
            mean_component_diagnostic = component_diagnostics["mean"]
            complete_component_diagnostic = component_diagnostics["complete"]
            self_consistent_component_diagnostic = component_diagnostics[
                "self_consistent"]
            transport_component_diagnostic = component_diagnostics[
                "transport_uncertain"]
            cavity_component_diagnostic = component_diagnostics[
                "observation_cavity"]
            root_component_diagnostic = component_diagnostics["root_resolved"]
            fmmt = denoise_fmmt(observation)[0]
            estimates = {
                "observation": observation,
                "complete-residual HJ simplex": complete[READOUT],
                "mean-component barycenter": mean_component[
                    "component_barycenter"],
                "mean-component mode": mean_component["component_mode"],
                "complete-component barycenter": complete_component[
                    "component_barycenter"],
                "complete-component mode": complete_component[
                    "component_mode"],
                "complete terminal mixture": complete_component[
                    "terminal_component_barycenter"],
                "self-consistent component barycenter": (
                    self_consistent_component["component_barycenter"]),
                "self-consistent component mode": (
                    self_consistent_component["component_mode"]),
                "self-consistent terminal mixture": self_consistent_component[
                    "terminal_component_barycenter"],
                "transport-uncertain component barycenter": (
                    transport_component["component_barycenter"]),
                "transport-uncertain component mode": (
                    transport_component["component_mode"]),
                "observation-cavity component barycenter": (
                    cavity_component["component_barycenter"]),
                "observation-cavity component mode": (
                    cavity_component["component_mode"]),
                "observation-cavity terminal mixture": cavity_component[
                    "terminal_component_barycenter"],
                "root-resolved terminal mixture": root_component[
                    "terminal_component_barycenter"],
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
                "mean_component_diagnostic": mean_component_diagnostic,
                "complete_component_diagnostic": complete_component_diagnostic,
                "self_consistent_component_diagnostic": (
                    self_consistent_component_diagnostic),
                "transport_uncertain_component_diagnostic": (
                    transport_component_diagnostic),
                "observation_cavity_component_diagnostic": (
                    cavity_component_diagnostic),
                "root_resolved_component_diagnostic": (
                    root_component_diagnostic),
            })

    methods = (
        "observation",
        "complete-residual HJ simplex",
        "mean-component barycenter",
        "mean-component mode",
        "complete-component barycenter",
        "complete-component mode",
        "complete terminal mixture",
        "self-consistent component barycenter",
        "self-consistent component mode",
        "self-consistent terminal mixture",
        "transport-uncertain component barycenter",
        "transport-uncertain component mode",
        "observation-cavity component barycenter",
        "observation-cavity component mode",
        "observation-cavity terminal mixture",
        "root-resolved terminal mixture",
        "integrated FMMT control",
    )
    names = (
        "mse", "ssim", "variance_ratio", "central_range_ratio",
        "edge_retention", "mean_bias", "observation_displacement_mse",
        "observation_displacement_rms", "observation_displacement_maximum",
        "fraction_moved_over_one_8bit_level",
    )

    def summarize(records: list[dict]) -> dict:
        return {
            method: {
                name: float(np.mean([row[method][name] for row in records]))
                for name in names
            }
            for method in methods
        }

    return {
        "purpose": (
            "test whether an explicit zero-residual causal terminal component "
            "creates meaningful reconstruction motion without a noise label"),
        "size": size,
        "phase_count": phase_count,
        "sources": list(selected),
        "summary": summarize(rows),
        "by_condition": {
            condition: summarize([
                row for row in rows if row["condition"] == condition])
            for condition, _kind in cases
        },
        "component_diagnostic": {
            mode: {
                name: float(np.mean([
                    row[f"{mode}_component_diagnostic"][name]
                    for row in rows]))
                for name in (
                    "mean_nonzero_probability",
                    "mean_zero_component_mass",
                    "mean_zero_component_mode_fraction",
                )
            }
            for mode in (
                "mean", "complete", "self_consistent", "transport_uncertain")
                + ("observation_cavity",)
                + ("root_resolved",)
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
        "by_condition": result["by_condition"],
        "component_diagnostic": result["component_diagnostic"],
    }, indent=2))


if __name__ == "__main__":
    main()
