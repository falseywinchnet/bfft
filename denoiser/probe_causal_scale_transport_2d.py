"""Probe scale-local information in the causal Selling filtration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.fft import dctn

from .causal_scale_transport_2d import (
    _heat_state,
    _isotropic_selling_spectrum,
    causal_scale_transport_observation_2d,
)
from .run_2d_denoiser_battery import metrics, sources
from .sample_series import corrupt


def _truth_components(
    truth: np.ndarray,
    times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    spectrum = _isotropic_selling_spectrum(truth.shape)
    coefficient = dctn(truth, type=2, norm="ortho")
    snapshots = np.stack([
        _heat_state(coefficient, spectrum, float(time)) for time in times
    ])
    components = snapshots[:-1] - snapshots[1:]
    return snapshots[-1], components[::-1]


def _oracle_generation(
    observed: np.ndarray,
    truth: np.ndarray,
    confidence: np.ndarray,
    hadamard_pull: np.ndarray,
) -> dict[str, float]:
    observed_flat = observed.reshape(-1)
    truth_flat = truth.reshape(-1)
    observed_action = float(np.dot(observed_flat, observed_flat))
    truth_action = float(np.dot(truth_flat, truth_flat))
    cross_action = float(np.dot(observed_flat, truth_flat))
    correlation = (
        cross_action / np.sqrt(observed_action * truth_action)
        if observed_action > np.finfo(float).tiny
        and truth_action > np.finfo(float).tiny
        else 0.0
    )
    oracle_gain = (
        cross_action / observed_action
        if observed_action > np.finfo(float).tiny else 0.0
    )
    noise = observed - truth
    truth_density = truth * truth
    noise_density = noise * noise
    confidence_flat = confidence.reshape(-1)
    truth_retention = (
        float(np.sum(confidence * truth_density)) / truth_action
        if truth_action > np.finfo(float).tiny else 0.0
    )
    noise_action = float(np.sum(noise_density))
    noise_retention = (
        float(np.sum(confidence * noise_density)) / noise_action
        if noise_action > np.finfo(float).tiny else 0.0
    )
    dominance = truth_density / np.maximum(
        truth_density + noise_density, np.finfo(float).tiny)
    centered_confidence = confidence_flat - float(np.mean(confidence_flat))
    centered_dominance = dominance.reshape(-1) - float(np.mean(dominance))
    correlation_denominator = float(np.linalg.norm(
        centered_confidence) * np.linalg.norm(centered_dominance))
    pull_flat = hadamard_pull.reshape(-1)
    pull_truth_retention = (
        float(np.sum(hadamard_pull * truth_density)) / truth_action
        if truth_action > np.finfo(float).tiny else 0.0
    )
    pull_noise_retention = (
        float(np.sum(hadamard_pull * noise_density)) / noise_action
        if noise_action > np.finfo(float).tiny else 0.0
    )
    centered_pull = pull_flat - float(np.mean(pull_flat))
    pull_correlation_denominator = float(np.linalg.norm(
        centered_pull) * np.linalg.norm(centered_dominance))
    return {
        "oracle_truth_action": truth_action,
        "oracle_noise_action": noise_action,
        "oracle_component_correlation": float(np.clip(
            correlation, -1.0, 1.0)),
        "oracle_scalar_gain": oracle_gain,
        "oracle_bounded_gain": float(np.clip(oracle_gain, 0.0, 1.0)),
        "oracle_truth_confidence_retention": truth_retention,
        "oracle_noise_confidence_retention": noise_retention,
        "oracle_confidence_separation": truth_retention - noise_retention,
        "oracle_local_dominance_correlation": (
            float(np.dot(centered_confidence, centered_dominance))
            / correlation_denominator
            if correlation_denominator > np.finfo(float).tiny else 0.0
        ),
        "oracle_hadamard_truth_retention": pull_truth_retention,
        "oracle_hadamard_noise_retention": pull_noise_retention,
        "oracle_hadamard_separation": (
            pull_truth_retention - pull_noise_retention),
        "oracle_hadamard_dominance_correlation": (
            float(np.dot(centered_pull, centered_dominance))
            / pull_correlation_denominator
            if pull_correlation_denominator > np.finfo(float).tiny else 0.0
        ),
    }


def run(size: int, selected: tuple[str, ...]) -> dict:
    catalogue = sources(size)
    cases = []
    for name in selected:
        truth = catalogue[name]
        cases.extend((
            (name, "clean", truth, truth),
            (
                name,
                "mixed replacement + uniform 0.25",
                truth,
                corrupt(
                    truth,
                    "mixed replacement + uniform",
                    amount=0.10,
                    density=0.25,
                    seed=271828,
                ),
            ),
        ))
    rng = np.random.default_rng(173)
    cases.extend((
        (
            "null",
            "zero-mean Gaussian",
            np.zeros((size, size)),
            rng.normal(size=(size, size)),
        ),
        (
            "null",
            "uniform",
            np.full((size, size), 0.5),
            rng.random((size, size)),
        ),
    ))

    rows = []
    for source, condition, truth, observation in cases:
        readout, residual, diagnostic = (
            causal_scale_transport_observation_2d(observation))
        times = np.asarray(diagnostic["transport_times"])
        truth_coarse, truth_components = _truth_components(truth, times)
        observed_components = np.asarray(
            diagnostic["components_coarse_to_fine"])
        confidence_components = np.asarray(
            diagnostic["confidence_coarse_to_fine"])
        hadamard_components = np.asarray(
            diagnostic["hadamard_pull_coarse_to_fine"])
        generation_rows = []
        for measured, actual, confidence, hadamard_pull, scale_row in zip(
            observed_components,
            truth_components,
            confidence_components,
            hadamard_components,
            diagnostic["generations"],
        ):
            generation_rows.append({
                **scale_row,
                **_oracle_generation(
                    measured, actual, confidence, hadamard_pull),
            })
        coarse_error = float(np.mean(
            (np.asarray(diagnostic["coarse_endpoint"]) - truth_coarse) ** 2))
        rows.append({
            "source": source,
            "condition": condition,
            "observation": metrics(observation, truth),
            "provisional_scale_readout": metrics(readout, truth),
            "hadamard_pull_readout": metrics(
                np.asarray(diagnostic["readouts"]["hadamard_pull"]), truth),
            "spectral_phase_readout": metrics(
                np.asarray(diagnostic["readouts"]["spectral_phase"]), truth),
            "spectral_connection_readout": metrics(
                np.asarray(diagnostic["readouts"]["spectral_connection"]),
                truth,
            ),
            "transport_pull_readout": metrics(
                np.asarray(diagnostic["readouts"]["transport_pull"]), truth),
            "pulled_phase_readout": metrics(
                np.asarray(diagnostic["readouts"]["pulled_phase"]), truth),
            "hadamard_pulled_readout": metrics(
                np.asarray(diagnostic["readouts"]["hadamard_pulled"]), truth),
            "hadamard_eigenprojection_readout": metrics(
                np.asarray(
                    diagnostic["readouts"]["hadamard_eigenprojection"]),
                truth,
            ),
            "phase_susceptibility_readout": metrics(
                np.asarray(
                    diagnostic["readouts"]["phase_susceptibility"]),
                truth,
            ),
            "residual_rms": float(np.sqrt(np.mean(residual * residual))),
            "coarse_endpoint_truth_mse": coarse_error,
            "generation_count": len(generation_rows),
            "mean_retained_component_confidence": diagnostic[
                "mean_retained_component_confidence"],
            "decomposition_maximum_error": diagnostic[
                "decomposition_maximum_error"],
            "endpoint_distance_from_mean": diagnostic[
                "endpoint_distance_from_mean"],
            "generations": generation_rows,
        })
    return {
        "purpose": (
            "measure where directional, isotropic, phase-persistent, and "
            "oracle scene action live along one continuous Selling scale orbit"
        ),
        "size": int(size),
        "sources": list(selected),
        "rows": rows,
        "interpretation_gate": (
            "the decomposition must telescope exactly; a useful confidence "
            "law must separate corrupted from clean generation action without "
            "assuming that every region is informative at every scale"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument(
        "--sources", default="cameraman,tapered hair,woven chirps")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.size,
        tuple(value.strip() for value in args.sources.split(",") if value.strip()),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "rows": [
            {
                "source": row["source"],
                "condition": row["condition"],
                "observation_mse": row["observation"]["mse"],
                "scale_readout_mse": row[
                    "provisional_scale_readout"]["mse"],
                "scale_readout_ssim": row[
                    "provisional_scale_readout"]["ssim"],
                "scale_readout_edge_retention": row[
                    "provisional_scale_readout"]["edge_retention"],
                "hadamard_pull_mse": row["hadamard_pull_readout"]["mse"],
                "hadamard_pull_ssim": row["hadamard_pull_readout"]["ssim"],
                "hadamard_pull_edge_retention": row[
                    "hadamard_pull_readout"]["edge_retention"],
                "spectral_phase_mse": row["spectral_phase_readout"]["mse"],
                "spectral_phase_ssim": row["spectral_phase_readout"]["ssim"],
                "spectral_phase_edge_retention": row[
                    "spectral_phase_readout"]["edge_retention"],
                "spectral_connection_mse": row[
                    "spectral_connection_readout"]["mse"],
                "spectral_connection_ssim": row[
                    "spectral_connection_readout"]["ssim"],
                "spectral_connection_edge_retention": row[
                    "spectral_connection_readout"]["edge_retention"],
                "transport_pull_mse": row["transport_pull_readout"]["mse"],
                "transport_pull_ssim": row[
                    "transport_pull_readout"]["ssim"],
                "transport_pull_edge_retention": row[
                    "transport_pull_readout"]["edge_retention"],
                "pulled_phase_mse": row["pulled_phase_readout"]["mse"],
                "pulled_phase_ssim": row["pulled_phase_readout"]["ssim"],
                "pulled_phase_edge_retention": row[
                    "pulled_phase_readout"]["edge_retention"],
                "hadamard_pulled_mse": row[
                    "hadamard_pulled_readout"]["mse"],
                "hadamard_pulled_ssim": row[
                    "hadamard_pulled_readout"]["ssim"],
                "hadamard_pulled_edge_retention": row[
                    "hadamard_pulled_readout"]["edge_retention"],
                "hadamard_eigenprojection_mse": row[
                    "hadamard_eigenprojection_readout"]["mse"],
                "hadamard_eigenprojection_ssim": row[
                    "hadamard_eigenprojection_readout"]["ssim"],
                "hadamard_eigenprojection_edge_retention": row[
                    "hadamard_eigenprojection_readout"]["edge_retention"],
                "phase_susceptibility_mse": row[
                    "phase_susceptibility_readout"]["mse"],
                "phase_susceptibility_ssim": row[
                    "phase_susceptibility_readout"]["ssim"],
                "phase_susceptibility_edge_retention": row[
                    "phase_susceptibility_readout"]["edge_retention"],
                "mean_confidence": row[
                    "mean_retained_component_confidence"],
                "generation_count": row["generation_count"],
            }
            for row in result["rows"]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
