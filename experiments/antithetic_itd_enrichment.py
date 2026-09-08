"""Auditable antithetic enrichment primitives for CONV--ITD.

This module is deliberately an experiment, not a change to the canonical
decomposition.  It separates identities that hold for every operator from
claims that require smoothness, a fixed extrema clock, or CONV admission.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from experiments.conv_itd_comparison import conv_irregular
from experiments.conv_itd_extrema_ablation import (
    conv_current_extrema,
    itd_knots_at,
)


Array = np.ndarray
Operator = Callable[[Array], Array]


def itd_knot_matrix(tau: Array) -> Array:
    """Linear ITD knot law for a fixed, strictly ordered extrema clock."""

    x = np.asarray(tau, dtype=np.float64)
    if x.ndim != 1 or x.size < 3 or np.any(np.diff(x) <= 0.0):
        raise ValueError("tau must contain at least three strictly ordered sites")
    n = x.size
    matrix = np.zeros((n, n), dtype=np.float64)
    matrix[0, 0] = matrix[-1, -1] = 1.0
    for k in range(1, n - 1):
        h_minus = x[k] - x[k - 1]
        h_plus = x[k + 1] - x[k]
        total = h_minus + h_plus
        matrix[k, k] = 0.5
        matrix[k, k - 1] = 0.5 * h_plus / total
        matrix[k, k + 1] = 0.5 * h_minus / total
    return matrix


@dataclass(frozen=True)
class TightFrame:
    """Equal-energy orthonormal probes for the range of ``I-K_tau``."""

    probes: Array
    projector: Array
    covariance: Array
    rank: int


@dataclass(frozen=True)
class Cubature:
    """Weighted directional probes stored by column."""

    points: Array
    weights: Array


def detail_tight_frame(tau: Array, rtol: float | None = None) -> TightFrame:
    """Return the isotropic covariance on the fixed-clock detail subspace."""

    detail = np.eye(np.asarray(tau).size) - itd_knot_matrix(tau)
    u, singular, _ = np.linalg.svd(detail, full_matrices=False)
    tolerance = (
        np.finfo(np.float64).eps * max(detail.shape) * singular[0]
        if rtol is None
        else float(rtol) * singular[0]
    )
    rank = int(np.sum(singular > tolerance))
    probes = u[:, :rank]
    projector = probes @ probes.T
    covariance = projector / rank
    return TightFrame(probes, projector, covariance, rank)


def lift_knot_probe(tau: Array, probe: Array, sample_sites: Array) -> Array:
    """Lift one knot-space probe by admitted irregular CONV synthesis."""

    return conv_irregular(
        np.asarray(tau, dtype=np.float64),
        np.asarray(probe, dtype=np.float64),
        np.asarray(sample_sites, dtype=np.float64),
    )


def lifted_detail_tight_frame(tau: Array, sample_sites: Array) -> TightFrame:
    """Whiten the admitted lifts in sample space.

    Individual CONV lifts do not preserve knot-space orthogonality.  The SVD
    here is therefore essential if isotropy is claimed after lifting.
    """

    knot_frame = detail_tight_frame(tau)
    lifted = np.column_stack([
        lift_knot_probe(tau, knot_frame.probes[:, j], sample_sites)
        for j in range(knot_frame.rank)
    ])
    u, singular, _ = np.linalg.svd(lifted, full_matrices=False)
    tolerance = np.finfo(np.float64).eps * max(lifted.shape) * singular[0]
    rank = int(np.sum(singular > tolerance))
    probes = u[:, :rank]
    projector = probes @ probes.T
    return TightFrame(probes, projector, projector/rank, rank)


def fourth_order_spherical_cubature(rank: int) -> Cubature:
    """Bounded symmetric cubature exact through degree four on a sphere.

    The rule mixes the cross polytope (total weight 2/(r+2)) with all
    normalized hypercube vertices (total weight r/(r+2)).  Its second and
    fourth tensors equal those of the uniform unit sphere in R^r.

    This full construction is exponential and is a certificate/reference,
    not a proposed high-dimensional implementation.
    """

    if rank < 1:
        raise ValueError("rank must be positive")
    eye = np.eye(rank)
    cross = np.concatenate((eye, -eye), axis=1)
    integers = np.arange(1 << rank, dtype=np.uint64)[:, None]
    bits = (integers >> np.arange(rank, dtype=np.uint64)[None, :]) & 1
    cube = (2.0*bits.astype(np.float64)-1.0).T/np.sqrt(rank)
    cross_total = 2.0/(rank+2.0)
    cube_total = rank/(rank+2.0)
    weights = np.r_[
        np.full(cross.shape[1], cross_total/cross.shape[1]),
        np.full(cube.shape[1], cube_total/cube.shape[1]),
    ]
    return Cubature(np.concatenate((cross, cube), axis=1), weights)


def cubature_second_moment(rule: Cubature) -> Array:
    return (rule.points*rule.weights[None, :])@rule.points.T


def cubature_fourth_moment(rule: Cubature) -> Array:
    return np.einsum(
        "n,in,jn,kn,ln->ijkl",
        rule.weights,
        rule.points,
        rule.points,
        rule.points,
        rule.points,
        optimize=True,
    )


def antithetic_average(
    operator: Operator, signal: Array, probe: Array, epsilon: float
) -> Array:
    return 0.5 * (
        operator(signal + epsilon * probe)
        + operator(signal - epsilon * probe)
    )


def null_calibrated_correction(
    operator: Operator, signal: Array, probe: Array, epsilon: float
) -> Array:
    """Signal-pair response minus its noise-only even response."""

    zero = np.zeros_like(signal)
    return (
        antithetic_average(operator, signal, probe, epsilon)
        - operator(signal)
        - antithetic_average(operator, zero, probe, epsilon)
        + operator(zero)
    )


def enriched_proposal(
    operator: Operator,
    signal: Array,
    probes: Array,
    epsilon: float | Array,
) -> Array:
    """Equal-weight, null-calibrated antithetic proposal."""

    profiles = np.asarray(probes, dtype=np.float64)
    if profiles.ndim != 2 or profiles.shape[0] != signal.size:
        raise ValueError("probes must have shape (signal.size, probe_count)")
    scale = np.broadcast_to(np.asarray(epsilon, dtype=np.float64), (profiles.shape[1],))
    correction = np.zeros_like(np.asarray(signal, dtype=np.float64))
    for j in range(profiles.shape[1]):
        correction += null_calibrated_correction(
            operator, signal, profiles[:, j], float(scale[j])
        )
    return operator(signal) + correction / profiles.shape[1]


def conv_itd_baseline(signal: Array) -> Array:
    """One fully nonlinear CONV-extrema/CONV-baseline extraction."""

    source = np.asarray(signal, dtype=np.float64)
    tau, value = conv_current_extrema(source)
    if tau.size < 5:
        return np.interp(np.arange(source.size, dtype=np.float64), tau, value)
    knot = itd_knots_at(tau, value)
    return conv_irregular(tau, knot, np.arange(source.size, dtype=np.float64))


def first_topology_threshold(signal: Array, probe: Array) -> float:
    """First amplitude at which either sampled first-difference sign can tie.

    This is an exact boundary for the sampled-current sign sequence.  It is
    not generally a boundary for the extrema of the admitted CONV profile,
    whose derivative is a piecewise quartic polynomial.
    """

    source_delta = np.diff(np.asarray(signal, dtype=np.float64))
    probe_delta = np.diff(np.asarray(probe, dtype=np.float64))
    active = probe_delta != 0.0
    if not np.any(active):
        return float("inf")
    return float(np.min(np.abs(source_delta[active] / probe_delta[active])))


def sign_change_count(sequence: Array) -> int:
    sign = np.sign(np.asarray(sequence, dtype=np.float64))
    sign = sign[sign != 0.0]
    return int(np.sum(sign[1:] != sign[:-1])) if sign.size else 0
