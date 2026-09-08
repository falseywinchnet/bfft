"""Conservative support refinement of the transported FMMT posterior.

Integrated FMMT remains the estimator: it transports latent-signal and
residual measures on common ordered fronts and couples them through the
observation equation.  This experiment asks a narrower question after that
posterior has been formed: which detail lost by FMMT is independently
predicted by target-excluded transport?

At each target the three finite CONV* charts propose matched source/posterior
transport corrections.  Their signed barycentre is contracted by its
Rayleigh quotient

    rho = |E[d]|^2 / E[|d|^2].

The resulting common correction is then projected onto the segment from the
FMMT posterior to the observation, so transport can only return observed
energy and can neither invent detail nor overshoot the datum.  Unanimous
witnesses donate their complete common correction, signed disagreement
continuously contracts it, and a zero correction remains the FMMT fixed
point.  There is no noise label, edge threshold, fitted strength, or iteration
count.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .dcnt import target_excluded_conv_family
from .fmmt_certified import denoise_fmmt


Array = np.ndarray


def coherent_support_donation(
    observation: Array,
    posterior: Array,
    observation_family: Array,
    posterior_family: Array | None = None,
) -> tuple[Array, dict[str, Any]]:
    """Return the bounded correction witnessed by three disjoint charts."""

    source = np.asarray(observation, dtype=np.float64)
    base = np.asarray(posterior, dtype=np.float64)
    witnesses = np.asarray(observation_family, dtype=np.float64)
    if source.ndim != 2 or base.shape != source.shape:
        raise ValueError("observation and posterior must be aligned HxW fields")
    if witnesses.shape != (4,) + source.shape:
        raise ValueError("transport family must have shape 4xHxW")
    finite = np.isfinite(witnesses)
    if posterior_family is None:
        posterior_witnesses = np.broadcast_to(base, witnesses.shape)
        coordinate = "transported observation minus target posterior"
    else:
        posterior_witnesses = np.asarray(posterior_family, dtype=np.float64)
        if posterior_witnesses.shape != witnesses.shape:
            raise ValueError("posterior family must align with observation family")
        finite &= np.isfinite(posterior_witnesses)
        coordinate = "matched observation/posterior transport charts"
    population = np.sum(finite, axis=0)
    if not np.all(population == 3):
        raise ValueError("every target requires three target-excluded witnesses")

    residual = source - base
    lower = np.minimum(residual, 0.0)
    upper = np.maximum(residual, 0.0)
    proposed = np.where(finite, witnesses - posterior_witnesses, 0.0)
    mean = np.sum(proposed, axis=0) / population
    second_moment = np.sum(proposed * proposed, axis=0) / population
    coherence = np.divide(
        mean * mean,
        second_moment,
        out=np.zeros_like(mean),
        where=second_moment > 0.0,
    )
    common = coherence * mean
    donation = np.minimum(np.maximum(common, lower), upper)
    estimate = np.clip(base + donation, 0.0, 1.0)
    return np.ascontiguousarray(estimate), {
        "mean_support_coherence": float(np.mean(coherence)),
        "mean_absolute_donation": float(np.mean(np.abs(donation))),
        "donated_action": float(np.mean(donation * donation)),
        "posterior_residual_action": float(np.mean(residual * residual)),
        "maximum_segment_violation": float(np.max(np.maximum(
            lower - donation, donation - upper))),
        "target_excluded_witnesses_per_pixel": 3,
        "correction_coordinate": coordinate,
    }


def denoise_rebuilt_fmmt(
    observation: Array,
    **fmmt_options: Any,
) -> tuple[Array, dict[str, Any]]:
    """Run FMMT and retain only thrice-observed residual support."""

    source = np.asarray(observation, dtype=np.float64)
    posterior, fmmt_diagnostic = denoise_fmmt(source, **fmmt_options)
    return rebuild_fmmt_posterior(source, posterior, fmmt_diagnostic)


def rebuild_fmmt_posterior(
    observation: Array,
    posterior: Array,
    fmmt_diagnostic: dict[str, Any] | None = None,
) -> tuple[Array, dict[str, Any]]:
    """Apply the support closure to an already-computed FMMT posterior."""

    source = np.asarray(observation, dtype=np.float64)
    posterior = np.asarray(posterior, dtype=np.float64)
    if source.ndim != 2 or posterior.shape != source.shape:
        raise ValueError("observation and FMMT posterior must be aligned HxW fields")
    family = target_excluded_conv_family(source)
    posterior_family = target_excluded_conv_family(posterior)
    provisional_donation, donation_diagnostic = coherent_support_donation(
        source, posterior, family, posterior_family)
    donated_1, observer_1 = transport_supported_donation(
        source, posterior, provisional_donation)
    donated_2, observer_2 = transport_supported_donation(
        source, posterior, donated_1)
    estimate, observer_3 = transport_supported_donation(
        source, posterior, donated_2)
    residual = source - estimate
    return estimate, {
        "method": "rebuilt FMMT with matched-chart transport support closure",
        "status": "experimental; canonical FMMT remains unchanged",
        "canonical_fmmt": (
            fmmt_diagnostic if fmmt_diagnostic is not None
            else {"status": "precomputed canonical posterior"}),
        "support_donation": donation_diagnostic,
        "observer_action_trace": [
            observer_1["transport_supported_action"],
            observer_2["transport_supported_action"],
            observer_3["transport_supported_action"],
        ],
        "observer_closure_depth": 3,
        "residual": residual,
        "recomposition_error": float(np.max(np.abs(
            estimate + residual - source))),
    }


def transport_supported_donation(
    observation: Array,
    posterior: Array,
    provisional: Array,
) -> tuple[Array, dict[str, Any]]:
    """Retain only donated residual action that reaches a CONV* observer."""

    source = np.asarray(observation, dtype=np.float64)
    base = np.asarray(posterior, dtype=np.float64)
    candidate = np.asarray(provisional, dtype=np.float64)
    if source.shape != base.shape or candidate.shape != source.shape:
        raise ValueError("donation observer fields must align")
    donation = candidate - base
    family = target_excluded_conv_family(donation)
    finite = np.isfinite(family)
    population = np.sum(finite, axis=0)
    if not np.all(population == 3):
        raise RuntimeError("donation observer requires three witnesses")
    transported = np.sum(np.where(finite, family, 0.0), axis=0) / population
    joint_mean = 0.5 * (donation + transported)
    joint_second_moment = 0.5 * (
        donation * donation + transported * transported)
    coherence = np.divide(
        joint_mean * joint_mean,
        joint_second_moment,
        out=np.zeros_like(joint_mean),
        where=joint_second_moment > 0.0,
    )
    common = coherence * transported
    residual = source - base
    lower = np.minimum(residual, 0.0)
    upper = np.maximum(residual, 0.0)
    admitted = np.minimum(np.maximum(common, lower), upper)
    estimate = np.clip(base + admitted, 0.0, 1.0)
    return np.ascontiguousarray(estimate), {
        "mean_observer_coherence": float(np.mean(coherence)),
        "mean_absolute_provisional_donation": float(
            np.mean(np.abs(donation))),
        "mean_absolute_transport_supported_donation": float(
            np.mean(np.abs(admitted))),
        "transport_supported_action": float(np.mean(admitted * admitted)),
        "target_excluded_witnesses_per_pixel": 3,
    }


__all__ = [
    "coherent_support_donation",
    "denoise_rebuilt_fmmt",
    "rebuild_fmmt_posterior",
    "transport_supported_donation",
]
