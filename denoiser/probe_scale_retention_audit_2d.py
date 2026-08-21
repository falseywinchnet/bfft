"""Audit what each continuous-scale transport coordinate retains and loses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.fft import dctn, idctn

from .causal_scale_transport_2d import (
    _heat_state,
    _isotropic_selling_spectrum,
    causal_scale_transport_observation_2d,
)
from .continual_eikonal_noise_transport_2d import _shift_symmetric
from .run_2d_denoiser_battery import metrics, sources
from .sample_series import corrupt


_SCALAR_COORDINATES = {
    "discrete_confidence": "confidence_coarse_to_fine",
    "hadamard_pull": "hadamard_pull_coarse_to_fine",
    "spectral_phase": "spectral_phase_coarse_to_fine",
    "spectral_connection": "spectral_connection_coarse_to_fine",
    "transport_pull": "transport_pull_coarse_to_fine",
    "pulled_phase": "pulled_phase_coarse_to_fine",
    "hadamard_pulled": "hadamard_pulled_coarse_to_fine",
    "phase_susceptibility": "phase_susceptibility_coarse_to_fine",
    "selling_jet_pull": "selling_jet_pull_coarse_to_fine",
    "unresolved_krylov_purity": (
        "unresolved_krylov_purity_coarse_to_fine"),
}


def _filtration_components(
    field: np.ndarray,
    times: np.ndarray,
    spectrum: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    coefficient = dctn(field, type=2, norm="ortho")
    snapshots = np.stack([
        _heat_state(coefficient, spectrum, float(time)) for time in times
    ])
    return snapshots[-1], (snapshots[:-1] - snapshots[1:])[::-1]


def _differential_action(
    field: np.ndarray,
    spectrum: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return value, first-jet, and generator-curvature action densities."""
    gx = 0.5 * (
        _shift_symmetric(field, 0, 1)
        - _shift_symmetric(field, 0, -1))
    gy = 0.5 * (
        _shift_symmetric(field, 1, 0)
        - _shift_symmetric(field, -1, 0))
    laplacian = idctn(
        spectrum * dctn(field, type=2, norm="ortho"),
        type=2,
        norm="ortho",
    )
    return field * field, gx * gx + gy * gy, laplacian * laplacian


def _action_retention(
    coordinate: np.ndarray,
    truth_component: np.ndarray,
    noise_component: np.ndarray,
    spectrum: np.ndarray,
) -> dict[str, float]:
    names = ("value", "first_jet", "curvature")
    truth_action = _differential_action(truth_component, spectrum)
    noise_action = _differential_action(noise_component, spectrum)
    row: dict[str, float] = {}
    for name, truth_density, noise_density in zip(
        names, truth_action, noise_action
    ):
        truth_total = float(np.sum(truth_density))
        noise_total = float(np.sum(noise_density))
        truth_retention = (
            float(np.sum(coordinate * truth_density)) / truth_total
            if truth_total > np.finfo(float).tiny else 0.0
        )
        noise_retention = (
            float(np.sum(coordinate * noise_density)) / noise_total
            if noise_total > np.finfo(float).tiny else 0.0
        )
        row[f"{name}_truth_retention"] = truth_retention
        row[f"{name}_noise_retention"] = noise_retention
        row[f"{name}_separation"] = truth_retention - noise_retention
    return row


def _global_differential_ratio(
    estimate: np.ndarray,
    reference: np.ndarray,
    spectrum: np.ndarray,
) -> dict[str, float]:
    names = ("value", "first_jet", "curvature")
    estimate_action = _differential_action(estimate, spectrum)
    reference_action = _differential_action(reference, spectrum)
    return {
        f"{name}_action_ratio": (
            float(np.sum(measured)) / float(np.sum(actual))
            if float(np.sum(actual)) > np.finfo(float).tiny else 0.0
        )
        for name, measured, actual in zip(
            names, estimate_action, reference_action)
    }


def _coordinate_partition_audit(
    first: np.ndarray,
    second: np.ndarray,
    truth_components: np.ndarray,
    noise_components: np.ndarray,
    spectrum: np.ndarray,
) -> dict[str, Any]:
    """Partition filtration action by agreement and one-sided authority.

    The four coordinates form an exact pointwise partition of unity:
    common support, first-only support, second-only support, and common
    refusal.  This leaves disagreement visible instead of prematurely
    collapsing two dependent witnesses into a fitted scalar posterior.
    """
    common = np.minimum(first, second)
    first_only = np.maximum(first - second, 0.0)
    second_only = np.maximum(second - first, 0.0)
    common_refusal = 1.0 - np.maximum(first, second)
    coordinates = {
        "common_support": common,
        "phase_only": first_only,
        "selling_jet_only": second_only,
        "common_refusal": common_refusal,
    }
    truth_densities = tuple(zip(*[
        _differential_action(component, spectrum)
        for component in truth_components
    ]))
    noise_densities = tuple(zip(*[
        _differential_action(component, spectrum)
        for component in noise_components
    ]))
    action_names = ("value", "first_jet", "curvature")
    result: dict[str, Any] = {}
    for action_name, truth_rows, noise_rows in zip(
        action_names, truth_densities, noise_densities
    ):
        truth_action = np.stack(truth_rows)
        noise_action = np.stack(noise_rows)
        truth_total = float(np.sum(truth_action))
        noise_total = float(np.sum(noise_action))
        result[action_name] = {
            name: {
                "truth_fraction": (
                    float(np.sum(coordinate * truth_action)) / truth_total
                    if truth_total > np.finfo(float).tiny else 0.0
                ),
                "noise_fraction": (
                    float(np.sum(coordinate * noise_action)) / noise_total
                    if noise_total > np.finfo(float).tiny else 0.0
                ),
            }
            for name, coordinate in coordinates.items()
        }
        result[action_name]["partition_error"] = {
            "truth_fraction": (
                abs(1.0 - sum(
                    row["truth_fraction"]
                    for name, row in result[action_name].items()
                    if name != "partition_error"
                ))
                if truth_total > np.finfo(float).tiny else 0.0
            ),
            "noise_fraction": (
                abs(1.0 - sum(
                    row["noise_fraction"]
                    for name, row in result[action_name].items()
                    if name != "partition_error"
                ))
                if noise_total > np.finfo(float).tiny else 0.0
            ),
        }
    return result


def _method_audit(
    name: str,
    coordinate: np.ndarray,
    truth: np.ndarray,
    observation: np.ndarray,
    truth_coarse: np.ndarray,
    truth_components: np.ndarray,
    noise_coarse: np.ndarray,
    noise_components: np.ndarray,
    spectrum: np.ndarray,
    stored_readout: np.ndarray,
) -> dict[str, Any]:
    retained_truth = truth_coarse + np.sum(
        coordinate * truth_components, axis=0)
    retained_noise = noise_coarse + np.sum(
        coordinate * noise_components, axis=0)
    reconstructed = retained_truth + retained_noise
    signal_loss = retained_truth - truth
    total_error = reconstructed - truth
    signal_loss_mse = float(np.mean(signal_loss * signal_loss))
    retained_noise_mse = float(np.mean(retained_noise * retained_noise))
    cross_mse = float(2.0 * np.mean(signal_loss * retained_noise))
    return {
        "method": name,
        "metrics": metrics(reconstructed, truth),
        "signal_loss_mse": signal_loss_mse,
        "retained_noise_mse": retained_noise_mse,
        "signal_noise_cross_mse": cross_mse,
        "mse_decomposition_error": float(abs(
            float(np.mean(total_error * total_error))
            - signal_loss_mse - retained_noise_mse - cross_mse)),
        "readout_reconstruction_maximum_error": float(np.max(np.abs(
            reconstructed - stored_readout))),
        "retained_truth": _global_differential_ratio(
            retained_truth, truth, spectrum),
        "retained_noise": _global_differential_ratio(
            retained_noise, observation - truth, spectrum),
        "lost_truth": _global_differential_ratio(
            truth - retained_truth, truth, spectrum),
    }


def run(size: int, selected: tuple[str, ...]) -> dict[str, Any]:
    catalogue = sources(size)
    rows = []
    for source in selected:
        truth = catalogue[source]
        cases = (
            ("clean", truth),
            (
                "mixed replacement + uniform 0.25",
                corrupt(
                    truth,
                    "mixed replacement + uniform",
                    amount=0.10,
                    density=0.25,
                    seed=271828,
                ),
            ),
        )
        for condition, observation in cases:
            discrete_readout, _residual, diagnostic = (
                causal_scale_transport_observation_2d(observation))
            times = np.asarray(diagnostic["transport_times"])
            spectrum = _isotropic_selling_spectrum(truth.shape)
            truth_coarse, truth_components = _filtration_components(
                truth, times, spectrum)
            noise_coarse, noise_components = _filtration_components(
                observation - truth, times, spectrum)
            stored_readouts = {
                "discrete_confidence": discrete_readout,
                **diagnostic["readouts"],
            }
            method_rows = []
            generation_rows = []
            for method, key in _SCALAR_COORDINATES.items():
                coordinate = np.asarray(diagnostic[key])
                method_rows.append(_method_audit(
                    method,
                    coordinate,
                    truth,
                    observation,
                    truth_coarse,
                    truth_components,
                    noise_coarse,
                    noise_components,
                    spectrum,
                    np.asarray(stored_readouts[method]),
                ))
            for generation, truth_component, noise_component in zip(
                range(truth_components.shape[0]),
                truth_components,
                noise_components,
            ):
                coordinate_rows = {}
                for method, key in _SCALAR_COORDINATES.items():
                    coordinate_rows[method] = _action_retention(
                        np.asarray(diagnostic[key])[generation],
                        truth_component,
                        noise_component,
                        spectrum,
                    )
                generation_rows.append({
                    "ordinal_coarse_to_fine": generation,
                    "scale": diagnostic["generations"][generation],
                    "coordinates": coordinate_rows,
                })
            rows.append({
                "source": source,
                "condition": condition,
                "observation": metrics(observation, truth),
                "methods": method_rows,
                "generations": generation_rows,
                "phase_selling_jet_partition": _coordinate_partition_audit(
                    np.asarray(diagnostic[
                        "phase_susceptibility_coarse_to_fine"]),
                    np.asarray(diagnostic[
                        "selling_jet_pull_coarse_to_fine"]),
                    truth_components,
                    noise_components,
                    spectrum,
                ),
            })
    return {
        "purpose": (
            "decompose every scalar transport readout into retained truth, "
            "lost truth, retained noise, and value/jet/curvature action"
        ),
        "size": int(size),
        "sources": list(selected),
        "rows": rows,
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
                "methods": [
                    {
                        "method": method["method"],
                        "mse": method["metrics"]["mse"],
                        "edge_retention": method["metrics"]["edge_retention"],
                        "signal_loss_mse": method["signal_loss_mse"],
                        "retained_noise_mse": method["retained_noise_mse"],
                        "lost_truth_first_jet": method["lost_truth"][
                            "first_jet_action_ratio"],
                        "retained_noise_first_jet": method["retained_noise"][
                            "first_jet_action_ratio"],
                    }
                    for method in row["methods"]
                ],
                "phase_selling_jet_partition": row[
                    "phase_selling_jet_partition"],
            }
            for row in result["rows"]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
