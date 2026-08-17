#!/usr/bin/env python3
"""Executable certificate for an approximate compressed Doob bridge."""

from __future__ import annotations

import math

import numpy as np


def exact_backward_potentials(
    transition: np.ndarray,
    event: np.ndarray,
    horizon: int,
) -> list[np.ndarray]:
    transition = np.asarray(transition, dtype=np.float64)
    potentials = [np.empty(len(event), dtype=np.float64) for _ in range(horizon + 1)]
    potentials[horizon] = np.asarray(event, dtype=np.float64)
    for k in range(horizon - 1, -1, -1):
        potentials[k] = transition @ potentials[k + 1]
    return potentials


def tilted_transition(
    transition: np.ndarray,
    next_potential: np.ndarray,
) -> np.ndarray:
    numerator = np.asarray(transition, dtype=np.float64) * np.asarray(
        next_potential, dtype=np.float64
    )[None, :]
    normalizer = np.sum(numerator, axis=1)
    if np.any(normalizer <= 0.0):
        raise ValueError("tilted transition has a zero normalizer")
    return numerator / normalizer[:, None]


def bridge_endpoint(
    stationary: np.ndarray,
    transition: np.ndarray,
    potentials: list[np.ndarray],
) -> np.ndarray:
    law = np.asarray(stationary, dtype=np.float64).copy()
    law /= np.sum(law)
    for next_potential in potentials[1:]:
        law = law @ tilted_transition(transition, next_potential)
    return law


def max_abs_log2_ratio(numerator: np.ndarray, denominator: np.ndarray) -> float:
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    if np.any(numerator <= 0.0) or np.any(denominator <= 0.0):
        return math.inf
    return float(np.max(np.abs(np.log2(numerator / denominator))))


def certificate_budget(
    stationary: np.ndarray,
    transition: np.ndarray,
    event: np.ndarray,
    exact: list[np.ndarray],
    approximate: list[np.ndarray],
) -> dict[str, float]:
    if len(exact) != len(approximate):
        raise ValueError("potential sequences must have equal length")
    event = np.asarray(event, dtype=bool)
    event_probability = float(np.sum(np.asarray(stationary)[event]))
    initial_flatness = max_abs_log2_ratio(
        exact[0], np.full_like(exact[0], event_probability)
    )
    initial_approximation = max_abs_log2_ratio(exact[0], approximate[0])
    residual = 0.0
    for k in range(len(approximate) - 1):
        evolved = np.asarray(transition) @ approximate[k + 1]
        residual += max_abs_log2_ratio(evolved, approximate[k])
    total = initial_flatness + initial_approximation + residual
    return {
        "initial_committor_flatness_bits": initial_flatness,
        "initial_approximation_error_bits": initial_approximation,
        "accumulated_backward_residual_bits": residual,
        "total_log2_likelihood_budget": total,
    }


def renyi2_log2(target: np.ndarray, proposal: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64)
    proposal = np.asarray(proposal, dtype=np.float64)
    target /= np.sum(target)
    proposal /= np.sum(proposal)
    if np.any((target > 0.0) & (proposal <= 0.0)):
        return math.inf
    return math.log2(float(np.sum(target * target / proposal)))
