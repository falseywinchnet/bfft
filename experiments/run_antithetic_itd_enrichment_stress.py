"""Randomized stress battery for antithetic CONV--ITD assertions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.antithetic_itd_enrichment import (
    conv_itd_baseline,
    detail_tight_frame,
    first_topology_threshold,
    lift_knot_probe,
    lifted_detail_tight_frame,
    null_calibrated_correction,
)
from experiments.conv_itd_extrema_ablation import conv_current_extrema
from experiments.convstar import (
    ordered_sign_ledger,
    project_signed_fibres,
    raw_current_jet_bank,
)


def finite(values: list[float]) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    return result[np.isfinite(result)]


def summary(values: list[float]) -> dict[str, float | int]:
    x = finite(values)
    if not x.size:
        return {"count": 0}
    return {
        "count": int(x.size),
        "minimum": float(np.min(x)),
        "median": float(np.median(x)),
        "maximum": float(np.max(x)),
        "mean": float(np.mean(x)),
    }


def run(trials: int, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    frame_errors: list[float] = []
    lift_conditions: list[float] = []
    whitened_errors: list[float] = []
    random_cov_minimum: list[float] = []
    tight_cov_minimum: list[float] = []
    projection_sum_error: list[float] = []
    projection_sign_error: list[float] = []
    projection_idempotence: list[float] = []

    for _ in range(trials):
        anchors = int(rng.integers(5, 18))
        tau = np.r_[0.0, np.cumsum(rng.uniform(.05, 3.0, anchors-1))]
        frame = detail_tight_frame(tau)
        frame_errors.append(float(np.linalg.norm(
            frame.probes.T@frame.probes-np.eye(frame.rank), ord=2)))
        sites = np.linspace(tau[0], tau[-1], int(rng.integers(64, 160)))
        lifted = lifted_detail_tight_frame(tau, sites)
        whitened_errors.append(float(np.linalg.norm(
            lifted.probes.T@lifted.probes-np.eye(lifted.rank), ord=2)))
        # The condition before rewhitening is reconstructed from the same
        # span by projecting the original knot basis through CONV.
        raw = np.column_stack([
            lift_knot_probe(tau, frame.probes[:, j], sites)
            for j in range(frame.rank)
        ])
        lift_conditions.append(float(np.linalg.cond(raw.T@raw)))

        rank = frame.rank
        random = rng.standard_normal((rank, rank))
        random /= np.linalg.norm(random, axis=0, keepdims=True)
        covariance = random@random.T/rank
        random_cov_minimum.append(float(np.min(np.linalg.eigvalsh(covariance))))
        tight_cov_minimum.append(1.0/rank)

        source = rng.standard_normal((int(rng.integers(20, 100)), int(rng.integers(1, 6))))
        raw_current, delta = raw_current_jet_bank(source)
        signs = ordered_sign_ledger(raw_current, delta)
        proposal = raw_current + rng.normal(0.0, 4.0, raw_current.shape)
        current = project_signed_fibres(proposal, signs, delta)
        projection_sum_error.append(float(np.max(np.abs(np.sum(current, axis=1)-delta))))
        projection_sign_error.append(float(max(0.0, -np.min(signs*current))))
        second = project_signed_fibres(current, signs, delta)
        projection_idempotence.append(float(np.max(np.abs(second-current))))

    actual_slopes: list[float] = []
    admitted_change_ratio: list[float] = []
    boundary_without_admitted_change = 0
    zero_control_error: list[float] = []
    closure_error: list[float] = []
    actual_trials = max(12, trials//4)
    ratio_grid = np.array([1.01, 1.25, 1.5, 2., 3., 4., 6., 8., 12., 16.])
    t = np.linspace(0.0, 1.0, 128)
    for _ in range(actual_trials):
        frequencies = rng.uniform(2.0, 16.0, 3)
        phases = rng.uniform(0.0, 2*np.pi, 3)
        amplitudes = rng.uniform(.08, .5, 3)
        signal = .15*t + sum(
            amplitudes[j]*np.sin(2*np.pi*frequencies[j]*t+phases[j])
            for j in range(3)
        )
        tau, _ = conv_current_extrema(signal)
        if tau.size < 5:
            continue
        lifted = lifted_detail_tight_frame(tau, np.arange(signal.size, dtype=float))
        probe = lifted.probes[:, int(rng.integers(lifted.rank))]
        boundary = first_topology_threshold(signal, probe)
        factors = np.array([.02, .04, .08, .16])
        norms = np.array([
            np.linalg.norm(null_calibrated_correction(
                conv_itd_baseline, signal, probe, boundary*factor))
            for factor in factors
        ])
        if np.all(norms > 1e-13):
            actual_slopes.append(float(np.polyfit(np.log(factors), np.log(norms), 1)[0]))
        original_count = conv_current_extrema(signal)[0].size
        changed = float("inf")
        for ratio in ratio_grid:
            if any(
                conv_current_extrema(signal+branch*ratio*boundary*probe)[0].size != original_count
                for branch in (-1.0, 1.0)
            ):
                changed = float(ratio)
                break
        admitted_change_ratio.append(changed)
        if not np.isfinite(changed):
            boundary_without_admitted_change += 1
        zero_control_error.append(float(np.max(np.abs(
            null_calibrated_correction(conv_itd_baseline, np.zeros_like(signal), probe, .2)
        ))))
        baseline = conv_itd_baseline(signal)
        detail = signal-baseline
        closure_error.append(float(np.max(np.abs(signal-(baseline+detail)))))

    # Scale identification: for a quartic response, the exact noise-conditioned
    # correction is 6 f^2 sigma^2.  Antithetic sigma points reproduce it only
    # when their amplitude equals the external sigma.
    f = 1.7
    sigmas = np.array([.03, .1, .25, .7])
    grid = np.linspace(0.0, 1.0, 10001)
    recovered = []
    for sigma in sigmas:
        target = 6*f*f*sigma*sigma
        response = 6*f*f*grid*grid
        recovered.append(float(grid[np.argmin(np.abs(response-target))]))

    return {
        "trials": trials,
        "fixed_clock_frame_orthogonality_error": summary(frame_errors),
        "raw_lift_condition": summary(lift_conditions),
        "rewhitened_lift_orthogonality_error": summary(whitened_errors),
        "random_equal_count_covariance_minimum_eigenvalue": summary(random_cov_minimum),
        "tight_covariance_minimum_eigenvalue": summary(tight_cov_minimum),
        "current_projection": {
            "endpoint_sum_max_error": summary(projection_sum_error),
            "signed_halfspace_max_violation": summary(projection_sign_error),
            "idempotence_max_error": summary(projection_idempotence),
        },
        "actual_conv_itd": {
            "trials": actual_trials,
            "small_amplitude_log_log_order": summary(actual_slopes),
            "first_admitted_extrema_count_change_over_sampled_boundary": summary(admitted_change_ratio),
            "no_admitted_count_change_through_16x_boundary": boundary_without_admitted_change,
            "zero_signal_control_max_error": summary(zero_control_error),
            "split_closure_max_error": summary(closure_error),
        },
        "amplitude_identification": {
            "external_sigma": sigmas.tolist(),
            "best_epsilon_for_quartic_expectation": recovered,
            "conclusion": "The preferred amplitude follows the external noise scale; symmetry alone cannot identify it.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--out", type=Path, default=Path("/tmp/antithetic_itd_enrichment_stress.json"))
    args = parser.parse_args()
    result = run(args.trials, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
