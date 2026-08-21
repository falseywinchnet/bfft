#!/usr/bin/env python3
"""Audit image-domain evidence for every positive common-K4 gauge branch."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.stats import spearmanr

from denoiser.cross_predictive_transport_2d import (
    heldout_relation_characteristic_measure_2d,
)
from denoiser.run_2d_denoiser_battery import sources

from .full_quartic_transport import (
    _covariance_square_root,
    directional_quartic_dictionary,
    estimate_full_quartic_transport,
)
from .multicapture_transport import _positive_sigma_measure
from .real_capture_evaluation import (
    _fourier_amplification,
    _local_envelope_excursion,
)
from .spatial_consensus import solve_spatial_field_consensus
from .spatial_transport import SpatialExposureField, SpatialReflectedExposureOperator


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


def _psnr(candidate: np.ndarray, truth: np.ndarray) -> float:
    mse = float(np.mean((candidate - truth) ** 2))
    return float(-10.0 * math.log10(max(mse, np.finfo(float).tiny)))


def _realize(
    mixture: np.ndarray,
    covariance: np.ndarray,
    dictionary: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> tuple[np.ndarray, np.ndarray]:
    factor = _covariance_square_root(covariance)
    points = []
    weights = []
    for component_mass, (standard_points, standard_weights) in zip(
        mixture, dictionary
    ):
        if component_mass <= 1e-10:
            continue
        points.append(standard_points @ factor.T)
        weights.append(component_mass * standard_weights)
    point_array = np.concatenate(points)
    weight_array = np.concatenate(weights)
    weight_array /= np.sum(weight_array)
    return point_array, weight_array


def _image_evidence(
    candidate: np.ndarray,
    observations: tuple[np.ndarray, ...],
    predictions: np.ndarray,
) -> dict[str, float]:
    heldout, _ = heldout_relation_characteristic_measure_2d(candidate)
    predictive_rms = float(np.sqrt(np.mean((candidate - heldout) ** 2)))
    gradient_y, gradient_x = np.gradient(candidate)
    gradient_rms = float(np.sqrt(np.mean(
        gradient_x * gradient_x + gradient_y * gradient_y)))
    laplacian_rms = float(np.sqrt(np.mean(
        ndimage.laplace(candidate, mode="reflect") ** 2)))
    closure_rms = float(np.sqrt(np.mean(
        (predictions - np.stack(observations)) ** 2)))
    amplification = _fourier_amplification(candidate, observations)
    return {
        "forward_closure_rms": closure_rms,
        "heldout_characteristic_rms": predictive_rms,
        "gradient_rms": gradient_rms,
        "laplacian_rms": laplacian_rms,
        "heldout_over_gradient": predictive_rms / max(gradient_rms, 1e-12),
        "curvature_over_gradient": laplacian_rms / max(gradient_rms, 1e-12),
        "local_envelope_excursion": float(_local_envelope_excursion(
            candidate, observations)["mean"]),
        "fourier_outer_three_ratio": float(
            amplification["outer_three_mean_ratio"]),
    }


def run(size: int = 64, passes: int = 16) -> dict[str, object]:
    truth = sources(int(size))["cameraman"]
    shape = truth.shape
    dictionary, _, labels = directional_quartic_dictionary(8)
    regimes = {
        "null_covariance": (
            ((0, 1.0),),
            ((0, 1.0),),
            ((0, 1.0),),
            ((0, 1.0),),
        ),
        "moderate_noisy_anchored": (
            ((0, 1.0),),
            ((0, 0.55), (1, 0.45)),
            ((0, 0.55), (3, 0.45)),
            ((0, 0.55), (10, 0.45)),
        ),
        "strong_anchored": (
            ((0, 1.0),),
            ((0, 0.10), (1, 0.90)),
            ((0, 0.10), (3, 0.90)),
            ((0, 0.10), (10, 0.90)),
        ),
        "opposed_unanchored": (
            ((0, 0.25), (1, 0.75)),
            ((0, 0.25), (5, 0.75)),
            ((0, 0.25), (9, 0.75)),
            ((0, 0.25), (13, 0.75)),
        ),
    }
    covariances = []
    for capture in range(4):
        angle = np.deg2rad(17.0 * capture)
        cosine = np.cos(angle)
        sine = np.sin(angle)
        rotation = np.asarray(((cosine, -sine), (sine, cosine)))
        covariances.append(
            rotation @ np.diag((1.2 + 0.2 * capture, 5.0 + capture))
            @ rotation.T)
    covariance_array = np.stack(covariances)
    regime_records = []
    for regime_name, specifications in regimes.items():
        observations = []
        for capture, specification in enumerate(specifications):
            mixture = np.zeros(len(dictionary), dtype=np.float64)
            for component, mass in specification:
                mixture[component] = mass
            points, weights = _realize(
                mixture, covariance_array[capture], dictionary)
            observations.append(SpatialReflectedExposureOperator(_field(
                f"{regime_name}_truth_{capture}", shape, points, weights,
            )).forward(truth))
            if regime_name == "moderate_noisy_anchored":
                random = np.random.default_rng(1200 + capture)
                observations[-1] = np.clip(
                    observations[-1]
                    + random.normal(0.0, 0.003, observations[-1].shape),
                    0.0,
                    1.0,
                )
        observations_tuple = tuple(observations)
        estimate = estimate_full_quartic_transport(
            observations_tuple,
            covariance_array,
            maximum_frequency=0.14,
            evaluate_gauge_catalog=True,
        )
        branch_mixtures = np.asarray(estimate.diagnostics[
            "gauge_branch_dictionary_weights"])
        branch_records = []
        baseline_gauge_image = None
        baseline_gauge_evidence = None
        baseline_gauge_predictions = None
        for branch, capture_mixtures in enumerate(branch_mixtures):
            fields = []
            for capture, mixture in enumerate(capture_mixtures):
                points, weights = _realize(
                    mixture, covariance_array[capture], dictionary)
                fields.append(_field(
                    f"{regime_name}_gauge_{branch}_{capture}",
                    shape,
                    points,
                    weights,
                ))
            solution = solve_spatial_field_consensus(
                observations_tuple,
                fields,
                passes=passes,
                descent_method="optimal_positive_line",
            )
            branch_records.append({
                "branch": branch,
                "label": labels[branch],
                "evaluation_psnr": _psnr(solution.image, truth),
                **_image_evidence(
                    solution.image,
                    observations_tuple,
                    solution.predicted_transport_gauge_observations,
                ),
            })
            if branch == 0:
                baseline_gauge_image = solution.image
                baseline_gauge_evidence = branch_records[-1]
                baseline_gauge_predictions = (
                    solution.predicted_transport_gauge_observations)
        # Include the pure covariance representative as the explicit no-K4
        # gauge reference in the same audit, but never as a winning selector.
        covariance_fields = []
        for capture, covariance in enumerate(covariance_array):
            points, weights = _positive_sigma_measure(covariance)
            covariance_fields.append(_field(
                f"{regime_name}_covariance_{capture}",
                shape,
                points,
                weights,
            ))
        covariance_solution = solve_spatial_field_consensus(
            observations_tuple,
            covariance_fields,
            passes=passes,
            descent_method="optimal_positive_line",
        )
        covariance_record = {
            "branch": -1,
            "label": "pure_covariance_reference",
            "evaluation_psnr": _psnr(covariance_solution.image, truth),
            **_image_evidence(
                covariance_solution.image,
                observations_tuple,
                covariance_solution.predicted_transport_gauge_observations,
            ),
        }
        assert baseline_gauge_image is not None
        assert baseline_gauge_evidence is not None
        assert baseline_gauge_predictions is not None
        closure_ratio = (
            baseline_gauge_evidence["forward_closure_rms"]
            / max(covariance_record["forward_closure_rms"], 1e-12))
        fourier_ratio = (
            baseline_gauge_evidence["fourier_outer_three_ratio"]
            / max(covariance_record["fourier_outer_three_ratio"], 1e-12))
        transport_action = float(
            closure_ratio - 1.0 + abs(math.log(max(fourier_ratio, 1e-12))))
        posterior_temperature = 0.05
        k4_posterior_mass = float(
            1.0 / (1.0 + math.exp(np.clip(
                transport_action / posterior_temperature, -60.0, 60.0))))
        posterior_image = (
            (1.0 - k4_posterior_mass) * covariance_solution.image
            + k4_posterior_mass * baseline_gauge_image)
        posterior_variance = (
            k4_posterior_mass * (1.0 - k4_posterior_mass)
            * (baseline_gauge_image - covariance_solution.image) ** 2)
        posterior_record = {
            "method": (
                "positive_covariance_k4_image_posterior_no_branch_selection"),
            "transport_action": transport_action,
            "posterior_temperature": posterior_temperature,
            "covariance_posterior_mass": 1.0 - k4_posterior_mass,
            "k4_posterior_mass": k4_posterior_mass,
            "evaluation_psnr": _psnr(posterior_image, truth),
            "posterior_uncertainty_rms": float(np.sqrt(np.mean(
                posterior_variance))),
            **_image_evidence(
                posterior_image,
                observations_tuple,
                (1.0 - k4_posterior_mass)
                * covariance_solution.predicted_transport_gauge_observations
                + k4_posterior_mass * baseline_gauge_predictions,
            ),
        }
        all_records = [covariance_record, *branch_records]
        psnr = np.asarray([item["evaluation_psnr"] for item in all_records])
        correlations = {}
        for metric in (
            "forward_closure_rms",
            "heldout_characteristic_rms",
            "heldout_over_gradient",
            "curvature_over_gradient",
            "local_envelope_excursion",
            "fourier_outer_three_ratio",
        ):
            values = np.asarray([item[metric] for item in all_records])
            correlation = spearmanr(values, psnr).statistic
            correlations[metric] = float(correlation)
        regime_records.append({
            "regime": regime_name,
            "shape_authority": estimate.authority,
            "covariance_reference": covariance_record,
            "covariance_k4_posterior": posterior_record,
            "branches": branch_records,
            "intrinsic_metric_spearman_vs_evaluation_psnr": correlations,
        })
    return {
        "experiment": "full_quartic_image_domain_gauge_evidence_v1",
        "size": int(size),
        "passes": int(passes),
        "regimes": regime_records,
        "truth_policy": (
            "truth_used_only_for_evaluation_psnr_and_metric_correlation"),
        "branch_policy": (
            "every_positive_common_gauge_is_reconstructed_no_winner_selected"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--passes", type=int, default=16)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = run(args.size, args.passes)
    payload = json.dumps(report, indent=2) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)
    print(json.dumps({
        item["regime"]: {
            "covariance_psnr": item["covariance_reference"]["evaluation_psnr"],
            "branch_psnr_range": [
                min(branch["evaluation_psnr"] for branch in item["branches"]),
                max(branch["evaluation_psnr"] for branch in item["branches"]),
            ],
            "posterior_psnr": item["covariance_k4_posterior"][
                "evaluation_psnr"],
            "k4_posterior_mass": item["covariance_k4_posterior"][
                "k4_posterior_mass"],
            "transport_action": item["covariance_k4_posterior"][
                "transport_action"],
            "metric_correlations": item[
                "intrinsic_metric_spearman_vs_evaluation_psnr"],
        }
        for item in report["regimes"]
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
