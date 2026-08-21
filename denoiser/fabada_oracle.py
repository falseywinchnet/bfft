"""Oracle-noise resurrection of the PyITD PFABADA diffusion family.

This is a comparison method, not part of the transport estimator.  It keeps
the useful mathematical residue of FABADA: average every progressively
diffused copy of the observation.  It removes the original repeated-data
Bayesian variance recursion and chi-square weighting, which are invalid once
the candidates are correlated smoothers of the same observation.

The repaired family is the continuous Cesaro heat operator

    A_t = (1/t) integral_0^t exp(-s L) ds,

on the conservative reflected nearest-neighbour line Laplacian.  Candidate
resolution is indexed by integer effective degrees of freedom, trace(A_t),
rather than an iteration count.  Known conditional noise moments provide an
unbiased quadratic-risk estimate for every affine candidate.  Both global
exponential aggregation and the more aggressive point-adaptive PFABADA-style
aggregation are returned for matched falsification.

Implementation ancestry: Joshuah Rainstar's ``pfabada.py`` in PyITD and the
FABADA method of Sanchez-Alarcon and Ascasibar.  The equations below are a new
implementation of the corrected operator/risk form; no original chi-square
or evidence code is carried forward.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np


def _validate_line(value: np.ndarray) -> np.ndarray:
    line = np.asarray(value, dtype=np.float64)
    if line.ndim != 1 or line.size < 3:
        raise ValueError("oracle FABADA expects a one-dimensional line of length >= 3")
    if not np.all(np.isfinite(line)):
        raise ValueError("oracle FABADA input must be finite")
    return line


def reflected_mean_operator_1d(samples: int) -> np.ndarray:
    """Return the symmetric conservative repair of PFABADA's local mean."""
    count = int(samples)
    if count < 3:
        raise ValueError("reflected mean requires at least three samples")
    operator = np.zeros((count, count), dtype=np.float64)
    index = np.arange(1, count - 1)
    operator[index, index - 1] = 1.0 / 3.0
    operator[index, index] = 1.0 / 3.0
    operator[index, index + 1] = 1.0 / 3.0
    # Reflecting the missing neighbour back onto the boundary point preserves
    # constants, total mass, and self-adjointness.  The PyITD special case
    # summed to 2/3 and therefore leaked boundary mass on every iteration.
    operator[0, 0] = 2.0 / 3.0
    operator[0, 1] = 1.0 / 3.0
    operator[-1, -1] = 2.0 / 3.0
    operator[-1, -2] = 1.0 / 3.0
    return operator


def _cesaro_multiplier(action: np.ndarray) -> np.ndarray:
    value = np.asarray(action, dtype=np.float64)
    result = np.ones_like(value)
    informative = np.abs(value) > np.sqrt(np.finfo(float).eps)
    result[informative] = (
        -np.expm1(-value[informative]) / value[informative])
    # Series form avoids cancellation near the constant eigenmode.
    small = ~informative
    if np.any(small):
        x = value[small]
        result[small] = 1.0 - 0.5 * x + x * x / 6.0
    return result


@lru_cache(maxsize=16)
def _oracle_candidate_geometry(samples: int) -> dict[str, np.ndarray]:
    """Resolve one candidate at every integer effective dimension."""
    operator = reflected_mean_operator_1d(samples)
    laplacian = np.eye(samples, dtype=np.float64) - operator
    eigenvalue, eigenvector = np.linalg.eigh(laplacian)
    eigenvalue = np.maximum(eigenvalue, 0.0)

    def effective_dimension(time: float) -> float:
        return float(np.sum(_cesaro_multiplier(time * eigenvalue)))

    targets = np.arange(samples, 0, -1, dtype=np.float64)
    times = np.empty(samples, dtype=np.float64)
    times[0] = 0.0
    lower = 0.0
    for candidate, target in enumerate(targets[1:-1], start=1):
        upper = max(1.0, 2.0 * lower)
        while effective_dimension(upper) > target:
            upper *= 2.0
        lo = lower
        hi = upper
        for _ in range(56):
            midpoint = 0.5 * (lo + hi)
            if effective_dimension(midpoint) > target:
                lo = midpoint
            else:
                hi = midpoint
        times[candidate] = 0.5 * (lo + hi)
        lower = times[candidate]
    times[-1] = np.inf

    multiplier = np.empty((samples, samples), dtype=np.float64)
    for candidate, time in enumerate(times[:-1]):
        multiplier[candidate] = _cesaro_multiplier(time * eigenvalue)
    # Exact infinite-time projection onto the connected graph's constant mode.
    multiplier[-1] = (eigenvalue <= 64.0 * np.finfo(float).eps).astype(float)
    diagonal = multiplier @ (eigenvector * eigenvector).T
    return {
        "effective_dimension": np.sum(multiplier, axis=1),
        "eigenvalue": eigenvalue,
        "eigenvector": eigenvector,
        "multiplier": multiplier,
        "operator_diagonal": diagonal,
        "time": times,
    }


def oracle_corruption_moments_1d(
    clean_reference: np.ndarray,
    corruption: str,
    *,
    amount: float,
    density: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return conditional observation mean and variance for the GUI generator.

    The clean reference is used only because multiplicative, replacement, and
    salt laws are heteroscedastic.  This deliberate oracle access makes the
    comparator stronger than a deployable blind method and is reported.
    Additive clipping at the [0, 1] coordinate boundary is not folded into the
    analytical moments, so the oracle is about the generating noise law rather
    than the realized error.
    """
    truth = _validate_line(clean_reference)
    scale = max(float(amount), 0.0)
    probability = float(np.clip(density, 0.0, 1.0))
    mean = truth.copy()
    variance = np.zeros_like(truth)
    gain = 1.0
    offset = 0.0
    if corruption == "none":
        pass
    elif corruption == "Gaussian additive":
        variance.fill(scale * scale)
    elif corruption == "uniform additive":
        variance.fill(scale * scale / 3.0)
    elif corruption == "Laplace additive":
        variance.fill(2.0 * scale * scale)
    elif corruption == "multiplicative":
        variance = scale * scale * truth * truth
    elif corruption == "random-value replacement":
        gain = 1.0 - probability
        offset = 0.5 * probability
        mean = (1.0 - probability) * truth + 0.5 * probability
        second = (1.0 - probability) * truth * truth + probability / 3.0
        variance = np.maximum(second - mean * mean, 0.0)
    elif corruption == "salt and pepper":
        gain = 1.0 - probability
        offset = 0.5 * probability
        mean = (1.0 - probability) * truth + 0.5 * probability
        second = (1.0 - probability) * truth * truth + 0.5 * probability
        variance = np.maximum(second - mean * mean, 0.0)
    elif corruption == "mixed replacement + uniform":
        gain = 1.0 - probability
        offset = 0.5 * probability
        mean = (1.0 - probability) * truth + 0.5 * probability
        second = (
            (1.0 - probability) * (truth * truth + scale * scale / 3.0)
            + probability / 3.0
        )
        variance = np.maximum(second - mean * mean, 0.0)
    else:
        raise ValueError(f"unknown corruption law {corruption!r}")
    return mean, variance, {
        "corruption": corruption,
        "amount": scale,
        "density": probability,
        "observation_gain": gain,
        "observation_offset": offset,
        "oracle_clean_reference_for_moments": True,
        "clipping_in_moment_law": False,
    }


def denoise_oracle_fabada_1d(
    observation: np.ndarray,
    expected_observation: np.ndarray,
    observation_variance: np.ndarray,
    *,
    clean_reference: np.ndarray | None = None,
    observation_gain: float | None = None,
    observation_offset: float | None = None,
    value_bounds: tuple[float, float] | None = (0.0, 1.0),
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Aggregate the corrected FABADA-Cesaro family by known-noise risk."""
    observed = _validate_line(observation)
    expected = _validate_line(expected_observation)
    variance = _validate_line(observation_variance)
    if expected.shape != observed.shape or variance.shape != observed.shape:
        raise ValueError("oracle moment fields must align with the observation")
    if np.any(variance < 0.0):
        raise ValueError("observation variance must be nonnegative")

    # E[Y|x] is affine in x for every supported catalogue law.  Recover its
    # gain and offset from the supplied conditional mean and clean reference.
    # When no clean reference is supplied the caller has already provided an
    # unbiased expected observation and no debiasing is needed.
    if clean_reference is None:
        centered = observed.copy()
        effective_variance = variance.copy()
        gain = np.ones_like(observed)
        offset = np.zeros_like(observed)
    else:
        truth = _validate_line(clean_reference)
        if truth.shape != observed.shape:
            raise ValueError("clean reference must align with the observation")
        if observation_gain is None or observation_offset is None:
            design = np.column_stack((truth, np.ones_like(truth)))
            coefficient, *_ = np.linalg.lstsq(design, expected, rcond=None)
            scalar_gain = float(coefficient[0])
            scalar_offset = float(coefficient[1])
        else:
            scalar_gain = float(observation_gain)
            scalar_offset = float(observation_offset)
        gain = np.full_like(observed, scalar_gain)
        offset = np.full_like(observed, scalar_offset)
        if abs(scalar_gain) <= np.sqrt(np.finfo(float).eps):
            # At unit replacement density the observation law contains no
            # information about x.  Knowing that fact is fair game for this
            # oracle comparator; returning a statistic of the hidden truth is
            # not.  The bounded uniform law has mean 1/2, so expose the honest
            # informationless estimator instead.
            constant = np.full_like(observed, scalar_offset)
            return {"global": constant, "local": constant}, {
                "state": "oracle FABADA-Cesaro risk aggregation",
                "informationless_observation": True,
                "observation_gain": scalar_gain,
                "observation_offset": scalar_offset,
                "oracle_noise_statistics": True,
                "physical_parameters": "known corruption moments only",
            }
        centered = (observed - offset) / gain
        effective_variance = variance / (gain * gain)

    if float(np.max(effective_variance)) <= np.finfo(float).eps:
        result = centered.copy()
        if value_bounds is not None:
            result = np.clip(result, *value_bounds)
        return {"global": result.copy(), "local": result.copy()}, {
            "state": "oracle FABADA-Cesaro risk aggregation",
            "candidate_count": observed.size,
            "effective_dimension": float(observed.size),
            "oracle_noise_statistics": True,
            "zero_noise_identity": True,
            "physical_parameters": "known corruption moments only",
        }

    geometry = _oracle_candidate_geometry(observed.size)
    eigenvector = geometry["eigenvector"]
    spectral_observation = eigenvector.T @ centered
    candidates = (
        geometry["multiplier"] * spectral_observation[None, :]
    ) @ eigenvector.T
    local_risk = (
        (candidates - centered[None, :]) ** 2
        + 2.0 * geometry["operator_diagonal"] * effective_variance[None, :]
        - effective_variance[None, :]
    )
    global_risk = np.mean(local_risk, axis=1)

    maximum_variance = max(
        float(np.max(effective_variance)), np.finfo(float).eps)
    global_log_weight = (
        -observed.size * global_risk / (4.0 * maximum_variance))
    global_log_weight -= np.max(global_log_weight)
    global_weight = np.exp(global_log_weight)
    global_weight /= np.sum(global_weight)
    global_output = global_weight @ candidates

    variance_floor = max(
        np.finfo(float).eps,
        np.finfo(float).eps * float(np.ptp(centered)) ** 2,
    )
    local_scale = 4.0 * np.maximum(effective_variance, variance_floor)
    local_log_weight = -local_risk / local_scale[None, :]
    local_log_weight -= np.max(local_log_weight, axis=0, keepdims=True)
    local_weight = np.exp(local_log_weight)
    local_weight /= np.sum(local_weight, axis=0, keepdims=True)
    local_output = np.sum(local_weight * candidates, axis=0)
    exact = effective_variance <= variance_floor
    local_output[exact] = centered[exact]

    if value_bounds is not None:
        global_output = np.clip(global_output, *value_bounds)
        local_output = np.clip(local_output, *value_bounds)
    global_entropy = -np.sum(
        global_weight * np.log(np.maximum(global_weight, np.finfo(float).tiny)))
    local_entropy = -np.sum(
        local_weight * np.log(np.maximum(local_weight, np.finfo(float).tiny)),
        axis=0,
    )
    return {"global": global_output, "local": local_output}, {
        "state": "oracle FABADA-Cesaro risk aggregation",
        "implementation_ancestry": "PyITD pfabada.py",
        "repair": (
            "conservative reflected heat; exact Cesaro operator; effective-"
            "dimension catalogue; covariance-risk aggregation"
        ),
        "removed": (
            "reused-data variance recursion, malformed evidence, chi-square "
            "weights, chi-square stopping, iteration-depth setting"
        ),
        "candidate_count": int(candidates.shape[0]),
        "minimum_estimated_global_risk": float(np.min(global_risk)),
        "minimum_risk_effective_dimension": float(
            geometry["effective_dimension"][np.argmin(global_risk)]),
        "global_aggregate_effective_dimension": float(
            global_weight @ geometry["effective_dimension"]),
        "global_weight_entropy": float(global_entropy),
        "mean_local_weight_entropy": float(np.mean(local_entropy)),
        "observation_gain": float(np.mean(gain)),
        "observation_offset": float(np.mean(offset)),
        "mean_effective_noise_variance": float(np.mean(effective_variance)),
        "oracle_noise_statistics": True,
        "point_adaptive_risk_is_comparison_only": True,
        "physical_parameters": "known corruption moments only",
    }


def denoise_oracle_fabada_from_corruption_1d(
    observation: np.ndarray,
    clean_reference: np.ndarray,
    corruption: str,
    *,
    amount: float,
    density: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    expected, variance, moment_diagnostic = oracle_corruption_moments_1d(
        clean_reference,
        corruption,
        amount=amount,
        density=density,
    )
    forms, diagnostic = denoise_oracle_fabada_1d(
        observation,
        expected,
        variance,
        clean_reference=clean_reference,
        observation_gain=float(moment_diagnostic["observation_gain"]),
        observation_offset=float(moment_diagnostic["observation_offset"]),
    )
    diagnostic.update(moment_diagnostic)
    return forms, diagnostic
