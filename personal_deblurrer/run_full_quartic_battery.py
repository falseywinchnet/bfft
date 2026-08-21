#!/usr/bin/env python3
"""Measure full fourth-cumulant transport across controlled shape regimes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from denoiser.run_2d_denoiser_battery import sources

from .full_quartic_transport import (
    _covariance_square_root,
    directional_quartic_dictionary,
    estimate_full_quartic_transport,
)
from .quartic_gauge_posterior import solve_quartic_gauge_posterior
from .spatial_transport import SpatialExposureField, SpatialReflectedExposureOperator


def _psnr(candidate: np.ndarray, truth: np.ndarray) -> float:
    error = float(np.mean((candidate - truth) ** 2))
    return float(-10.0 * math.log10(max(error, np.finfo(float).tiny)))


def _field(
    name: str,
    shape: tuple[int, int],
    points: np.ndarray,
    weights: np.ndarray,
) -> SpatialExposureField:
    return SpatialExposureField.from_barycentric_paths(
        name,
        np.zeros((*shape, 2), dtype=np.float64),
        points,
        weights,
        compact_global=True,
    )


def run(size: int = 96, passes: int = 32) -> dict[str, object]:
    truth = sources(int(size))["cameraman"]
    shape = truth.shape
    dictionary, _, labels = directional_quartic_dictionary(8)
    # Each capture retains positive mass.  The null case is the covariance
    # measure; other cases continuously move mass into rotated measures.
    regimes = {
        "null_covariance": (
            ((0, 1.0),),
            ((0, 1.0),),
            ((0, 1.0),),
            ((0, 1.0),),
        ),
        "moderate_directional": (
            ((0, 1.0),),
            ((0, 0.55), (1, 0.45)),
            ((0, 0.55), (3, 0.45)),
            ((0, 0.55), (10, 0.45)),
        ),
        "strong_directional": (
            ((0, 1.0),),
            ((0, 0.10), (1, 0.90)),
            ((0, 0.10), (3, 0.90)),
            ((0, 0.10), (10, 0.90)),
        ),
        "opposed_directional_mixture": (
            ((0, 0.25), (1, 0.75)),
            ((0, 0.25), (5, 0.75)),
            ((0, 0.25), (9, 0.75)),
            ((0, 0.25), (13, 0.75)),
        ),
    }
    covariance = []
    for index in range(4):
        angle = np.deg2rad(17.0 * index)
        cosine = np.cos(angle)
        sine = np.sin(angle)
        rotation = np.asarray(((cosine, -sine), (sine, cosine)))
        covariance.append(
            rotation @ np.diag((1.2 + 0.2 * index, 5.0 + index)) @ rotation.T)
    covariance_array = np.stack(covariance)
    records = []
    for regime_name, capture_specs in regimes.items():
        observations = []
        for capture, specification in enumerate(capture_specs):
            factor = _covariance_square_root(covariance_array[capture])
            point_parts = []
            weight_parts = []
            for component, component_mass in specification:
                standard_points, standard_weights = dictionary[component]
                point_parts.append(standard_points @ factor.T)
                weight_parts.append(component_mass * standard_weights)
            truth_field = _field(
                f"{regime_name}_truth_{capture}",
                shape,
                np.concatenate(point_parts),
                np.concatenate(weight_parts),
            )
            observation = SpatialReflectedExposureOperator(truth_field).forward(
                truth)
            # The noisy regime tests blur-shape evidence in the limited
            # blur/noise overlap without changing the estimator or selecting a
            # separate denoiser.
            if regime_name == "moderate_directional":
                random = np.random.default_rng(1200 + capture)
                observation = np.clip(
                    observation + random.normal(0.0, 0.003, observation.shape),
                    0.0,
                    1.0,
                )
            observations.append(observation)
        estimate = estimate_full_quartic_transport(
            observations,
            covariance_array,
            maximum_frequency=0.14,
        )
        covariance_errors = []
        for capture in range(4):
            tensor_points = estimate.residual_displacements[capture]
            tensor_weights = estimate.residual_weights[capture]
            realized_covariance = np.einsum(
                "k,ki,kj->ij", tensor_weights, tensor_points, tensor_points)
            covariance_errors.append(float(np.max(np.abs(
                realized_covariance - covariance_array[capture]))))
        posterior = solve_quartic_gauge_posterior(
            observations,
            covariance_array,
            estimate=estimate,
            passes=passes,
            descent_method="optimal_positive_line",
        )
        baseline_psnr = _psnr(posterior.covariance_solution.image, truth)
        tensor_psnr = _psnr(posterior.quartic_solution.image, truth)
        posterior_psnr = _psnr(posterior.image, truth)
        records.append({
            "regime": regime_name,
            "capture_dictionary_components": [
                [labels[item[0]] for item in specification]
                for specification in capture_specs
            ],
            "shape_authority": estimate.authority,
            "crossfit_predictive_authority": estimate.diagnostics[
                "crossfit_predictive_authority"],
            "raw_tensor_frobenius": float(np.sqrt(np.sum(
                estimate.raw_standardized_cumulants ** 2))),
            "transported_tensor_frobenius": float(np.sqrt(np.sum(
                estimate.standardized_cumulants ** 2))),
            "baseline_psnr": baseline_psnr,
            "full_tensor_psnr": tensor_psnr,
            "full_tensor_delta_db": tensor_psnr - baseline_psnr,
            "gauge_posterior_psnr": posterior_psnr,
            "gauge_posterior_delta_db": posterior_psnr - baseline_psnr,
            "covariance_posterior_mass": (
                posterior.covariance_posterior_mass),
            "quartic_posterior_mass": posterior.quartic_posterior_mass,
            "image_transport_action": posterior.diagnostics[
                "transport_action"],
            "between_gauge_uncertainty_rms": posterior.diagnostics[
                "between_gauge_uncertainty_rms"],
            "covariance_generated_global_storage_bytes": (
                posterior.diagnostics[
                    "covariance_generated_global_storage_bytes"]),
            "quartic_generated_global_storage_bytes": (
                posterior.diagnostics[
                    "quartic_generated_global_storage_bytes"]),
            "quartic_materialized_bytes_avoided": posterior.diagnostics[
                "quartic_materialized_bytes_avoided"],
            "maximum_covariance_error": max(covariance_errors),
            "minimum_atom_weight": float(min(
                np.min(item) for item in estimate.residual_weights)),
            "maximum_atom_count": estimate.diagnostics["maximum_atom_count"],
            "active_dictionary_components": estimate.diagnostics[
                "active_dictionary_components"],
            "gauge_posterior_entropy": estimate.diagnostics[
                "gauge_posterior_entropy"],
            "gauge_posterior_effective_branch_count": estimate.diagnostics[
                "gauge_posterior_effective_branch_count"],
            "gauge_branch_posterior_mass": estimate.diagnostics[
                "gauge_branch_posterior_mass"],
            "gauge_branch_exact_transfer_rms": estimate.diagnostics[
                "gauge_branch_exact_transfer_rms"],
        })
    return {
        "experiment": "full_quartic_positive_directional_regime_battery_v1",
        "size": int(size),
        "passes": int(passes),
        "regimes": records,
        "method_boundary": (
            "covariances_are_supplied_to_isolate_fourth_cumulant_estimation"),
        "selection_policy": (
            "held_out_predictive_authority_continuously_transports_shape_no_"
            "blur_family_or_capture_selection"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--passes", type=int, default=32)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = run(args.size, args.passes)
    payload = json.dumps(report, indent=2) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
