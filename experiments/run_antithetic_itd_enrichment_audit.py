"""Numerical audit of the claims behind antithetic CONV--ITD enrichment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.antithetic_itd_enrichment import (
    antithetic_average,
    conv_itd_baseline,
    cubature_fourth_moment,
    cubature_second_moment,
    detail_tight_frame,
    first_topology_threshold,
    lift_knot_probe,
    lifted_detail_tight_frame,
    fourth_order_spherical_cubature,
    null_calibrated_correction,
)
from experiments.conv_itd_extrema_ablation import conv_current_extrema


def smooth_operator(matrix: np.ndarray):
    return lambda x: np.tanh(matrix @ x)


def order_fit(epsilon: np.ndarray, error: np.ndarray) -> float:
    valid = (error > 100*np.finfo(float).eps) & np.isfinite(error)
    return float(np.polyfit(np.log(epsilon[valid]), np.log(error[valid]), 1)[0])


def audit(seed: int = 20260830) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    n = 64
    matrix = rng.standard_normal((n, n))/np.sqrt(n)
    signal = rng.standard_normal(n)*.3
    probe = rng.standard_normal(n)
    probe /= np.linalg.norm(probe)
    operator = smooth_operator(matrix)
    epsilon = np.logspace(-1, -5, 17)

    pair_departure = np.array([
        np.linalg.norm(antithetic_average(operator, signal, probe, e)-operator(signal))
        for e in epsilon
    ])
    correction = np.array([
        np.linalg.norm(null_calibrated_correction(operator, signal, probe, e))
        for e in epsilon
    ])
    z = matrix@signal
    direction = matrix@probe
    tangent = np.tanh(z)
    # Half the exact second directional derivative of tanh(Ax).
    second_coefficient = -tangent*(1.0-tangent*tangent)*direction*direction
    remainder = np.array([
        np.linalg.norm(
            null_calibrated_correction(operator, signal, probe, e)
            - e*e*second_coefficient
        )
        for e in epsilon[:9]
    ])

    tau = np.array([0., .8, 1.9, 3.7, 5.0, 8.4, 9.1, 13.6, 18.0])
    frame = detail_tight_frame(tau)
    covariance_eigenvalues = np.linalg.eigvalsh(frame.covariance)
    sites = np.linspace(tau[0], tau[-1], 129)
    raw_lift = np.column_stack([
        lift_knot_probe(tau, frame.probes[:, j], sites)
        for j in range(frame.rank)
    ])
    raw_lift_gram = raw_lift.T@raw_lift
    lifted_frame = lifted_detail_tight_frame(tau, sites)

    coordinate = np.eye(2)
    rotation = np.array([[1.0, -1.0], [1.0, 1.0]])/np.sqrt(2.0)
    quartic_direction = np.array([1.0, 0.0])
    coordinate_fourth = float(np.mean((quartic_direction@coordinate)**4))
    rotated_fourth = float(np.mean((quartic_direction@rotation)**4))
    sphere_rule = fourth_order_spherical_cubature(7)
    sphere_fourth = cubature_fourth_moment(sphere_rule)
    sphere_fourth_target = np.zeros_like(sphere_fourth)
    for i in range(7):
        for j in range(7):
            for k in range(7):
                for ell in range(7):
                    sphere_fourth_target[i, j, k, ell] = (
                        (i == j)*(k == ell)+(i == k)*(j == ell)+(i == ell)*(j == k)
                    )/(7.0*9.0)

    # A quadratic selector is an explicit coherent even-order ghost: the
    # antithetic average retains it while null calibration removes it.
    quadratic = lambda x: x*x
    zero = np.zeros(n)
    ghost_pair = antithetic_average(quadratic, zero, probe, .4)
    ghost_calibrated = null_calibrated_correction(quadratic, zero, probe, .4)

    # Actual CONV--ITD: measure the smooth-regime scaling only below the first
    # sampled-current topology boundary, then show what happens across it.
    t = np.linspace(0.0, 1.0, 192)
    actual_signal = .17*t + .25*np.sin(2*np.pi*2.3*t) + .42*np.sin(2*np.pi*(7*t+5*t*t))
    extrema_tau, _ = conv_current_extrema(actual_signal)
    actual_frame = detail_tight_frame(extrema_tau)
    sample_sites = np.arange(actual_signal.size, dtype=np.float64)
    lifted = np.column_stack([
        lift_knot_probe(extrema_tau, actual_frame.probes[:, j], sample_sites)
        for j in range(actual_frame.rank)
    ])
    lifted /= np.maximum(np.linalg.norm(lifted, axis=0, keepdims=True), 1e-30)
    chosen = lifted[:, 0]
    sampled_current_boundary = first_topology_threshold(actual_signal, chosen)
    actual_epsilon = sampled_current_boundary*np.array([.02, .04, .08, .16, .32, .64, 1.01, 1.5])
    actual_correction = np.array([
        np.linalg.norm(null_calibrated_correction(conv_itd_baseline, actual_signal, chosen, e))
        for e in actual_epsilon
    ])
    actual_events = []
    source_sign = np.sign(np.diff(actual_signal))
    for e in actual_epsilon:
        actual_events.append({
            "epsilon_over_sampled_current_boundary": float(e/sampled_current_boundary),
            "plus_sampled_sign_changes": int(np.sum(np.sign(np.diff(actual_signal+e*chosen)) != source_sign)),
            "minus_sampled_sign_changes": int(np.sum(np.sign(np.diff(actual_signal-e*chosen)) != source_sign)),
            "plus": int(conv_current_extrema(actual_signal+e*chosen)[0].size-2),
            "minus": int(conv_current_extrema(actual_signal-e*chosen)[0].size-2),
        })

    # Known-covariance factorization on the detail subspace.
    a = rng.standard_normal((tau.size, tau.size))
    physical = a@a.T
    restricted = frame.projector@physical@frame.projector
    value, vector = np.linalg.eigh(restricted)
    factor = vector[:, value > 1e-12] * np.sqrt(value[value > 1e-12])

    return {
        "smooth_operator": {
            "paired_departure_order": order_fit(epsilon, pair_departure),
            "null_correction_order": order_fit(epsilon, correction),
            "fourth_order_remainder_fit": order_fit(epsilon[:9], remainder),
            "epsilon": epsilon.tolist(),
            "paired_departure": pair_departure.tolist(),
            "null_correction": correction.tolist(),
            "quadratic_remainder": remainder.tolist(),
        },
        "tight_frame": {
            "rank": frame.rank,
            "orthogonality_error": float(np.linalg.norm(frame.probes.T@frame.probes-np.eye(frame.rank), ord=2)),
            "projector_idempotence_error": float(np.linalg.norm(frame.projector@frame.projector-frame.projector, ord=2)),
            "covariance_nonzero_eigenvalues": covariance_eigenvalues[covariance_eigenvalues > 1e-12].tolist(),
            "target_eigenvalue": 1.0/frame.rank,
            "raw_lift_gram_condition": float(np.linalg.cond(raw_lift_gram)),
            "raw_lift_orthogonality_error": float(np.linalg.norm(raw_lift_gram-np.eye(frame.rank), ord=2)),
            "rewhitened_lift_rank": lifted_frame.rank,
            "rewhitened_lift_orthogonality_error": float(np.linalg.norm(lifted_frame.probes.T@lifted_frame.probes-np.eye(lifted_frame.rank), ord=2)),
            "coordinate_frame_directional_fourth_moment": coordinate_fourth,
            "rotated_frame_directional_fourth_moment": rotated_fourth,
            "fourth_moment_orientation_ratio": coordinate_fourth/rotated_fourth,
            "conclusion": "Second-order tightness does not imply finite-amplitude orientation neutrality.",
        },
        "fourth_order_spherical_cubature": {
            "rank": 7,
            "probe_count": int(sphere_rule.points.shape[1]),
            "weight_sum_error": float(abs(np.sum(sphere_rule.weights)-1.0)),
            "mean_error": float(np.linalg.norm(sphere_rule.points@sphere_rule.weights)),
            "second_moment_error": float(np.linalg.norm(cubature_second_moment(sphere_rule)-np.eye(7)/7.0, ord=2)),
            "fourth_moment_max_error": float(np.max(np.abs(sphere_fourth-sphere_fourth_target))),
            "warning": "Exact fourth-order isotropy does not control sixth and higher responses; the reference rule is exponential in rank.",
        },
        "ghost_control": {
            "antithetic_noise_only_norm": float(np.linalg.norm(ghost_pair)),
            "null_calibrated_noise_only_norm": float(np.linalg.norm(ghost_calibrated)),
        },
        "actual_conv_itd": {
            "extrema_clock_size": int(extrema_tau.size),
            "detail_rank": actual_frame.rank,
            "first_sampled_current_sign_boundary": sampled_current_boundary,
            "epsilon": actual_epsilon.tolist(),
            "correction_norm": actual_correction.tolist(),
            "events": actual_events,
            "warning": "The sampled-current sign boundary is not generally an admitted-CONV extrema boundary. Taylor order is conditional on the actual nonlinear active set.",
        },
        "known_covariance": {
            "restricted_rank": int(factor.shape[1]),
            "factorization_error": float(np.linalg.norm(factor@factor.T-restricted, ord=2)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/tmp/antithetic_itd_enrichment_audit.json"))
    args = parser.parse_args()
    result = audit()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
