"""Explicit zero/nonzero residual components on the causal HJ terminal law."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from .causal_information_lineage_2d import (
    causal_information_lineage_law_2d,
    nested_population_phases,
)
from .witnessed_characteristic_transport_2d import _validate

NonzeroProbabilityMode = Literal[
    "mean", "complete", "self_consistent", "transport_uncertain",
    "observation_cavity", "root_resolved",
]


def zero_residual_component_readouts(
    observation: np.ndarray,
    law: dict[str, np.ndarray],
    *,
    nonzero_probability_mode: NonzeroProbabilityMode = "mean",
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Collide an explicit zero-residual atom with the causal branch law.

    The component base measure is symmetric: half for the zero-residual
    hypothesis and half for the complete nonzero branch fibre.  This is a
    representation measure, not a fitted prior probability.  The inferred
    nonzero component mass is inferred from the order-one causal residual law.
    ``mean`` regards only coherent residual displacement in the order-one law
    as evidence for the nonzero component.  ``complete`` also regards its
    residual dispersion as evidence.  ``self_consistent`` measures coherent
    displacement after the same causal simplex collision that forms the
    terminal branch posterior, eliminating the precursor/posterior mismatch.
    ``transport_uncertain`` adds terminal residual dispersion only in
    proportion to Fisher--Rao/Hellinger disagreement between the local branch
    law and its causally transported ancestry law.  It therefore represents
    uncertainty about transport itself rather than choosing an interpolation.
    ``observation_cavity`` divides the current likelihood out of the causal
    posterior before measuring residual agreement.  Its component evidence is
    therefore supplied by target-free ancestry while reconstruction still
    uses the terminal HJ branch law.
    ``root_resolved`` keeps causal histories separate while taking residual
    moments.  Within-history dispersion may support nuisance; between-history
    dispersion is uncertainty about transport and appears only as competing
    uncertainty in the component probability.
    """
    image = _validate(observation)
    signal = np.asarray(law["signal"], dtype=np.float64)
    residual = np.asarray(law["residual"], dtype=np.float64)
    reference = np.asarray(law["reference_mass"], dtype=np.float64)
    score = np.asarray(law["hj_path_score"], dtype=np.float64)
    order = np.asarray(law["hj_simplex_collision_order"], dtype=np.float64)
    if (
        signal.shape != residual.shape
        or signal.shape != reference.shape
        or signal.shape != score.shape
        or signal.shape[:2] != image.shape
        or order.shape != image.shape
    ):
        raise ValueError("causal terminal fields must align with observation")

    centered_score = score - np.max(score, axis=-1, keepdims=True)
    order_one = reference * np.exp(centered_score)
    order_one /= np.sum(order_one, axis=-1, keepdims=True)
    collision_score = order[..., None] * score
    collision_score -= np.max(collision_score, axis=-1, keepdims=True)
    collision_mass = reference * np.exp(collision_score)
    collision_mass /= np.sum(collision_mass, axis=-1, keepdims=True)
    cavity_surprise = np.zeros(image.shape, dtype=np.float64)
    root_within_variance = np.zeros(image.shape, dtype=np.float64)
    root_between_variance = np.zeros(image.shape, dtype=np.float64)
    if nonzero_probability_mode == "root_resolved":
        root_mass = np.asarray(law["root_mass"], dtype=np.float64)
        if root_mass.ndim != 4 or root_mass.shape[:2] != image.shape:
            raise ValueError("root-resolved branch law must have shape HxWxRxK")
        if root_mass.shape[-1] != signal.shape[-1]:
            raise ValueError("root-resolved branch atoms must align")
        joint_total = np.sum(root_mass, axis=(-2, -1), keepdims=True)
        joint_mass = root_mass / np.maximum(
            joint_total, np.finfo(float).tiny)
        root_probability = np.sum(joint_mass, axis=-1)
        conditional = np.divide(
            joint_mass,
            root_probability[..., None],
            out=np.zeros_like(joint_mass),
            where=root_probability[..., None] > np.finfo(float).tiny,
        )
        root_mean = np.sum(
            conditional * residual[..., None, :], axis=-1)
        residual_mean = np.sum(root_probability * root_mean, axis=-1)
        root_within_variance = np.sum(
            root_probability * np.sum(
                conditional
                * (residual[..., None, :] - root_mean[..., None]) ** 2,
                axis=-1,
            ),
            axis=-1,
        )
        root_between_variance = np.sum(
            root_probability
            * (root_mean - residual_mean[..., None]) ** 2,
            axis=-1,
        )
        residual_variance = root_within_variance + root_between_variance
        statistic_mass = None
    elif nonzero_probability_mode == "observation_cavity":
        causal_mass = np.asarray(law["mass"], dtype=np.float64)
        likelihood = np.asarray(law["likelihood"], dtype=np.float64)
        if causal_mass.shape != signal.shape or likelihood.shape != signal.shape:
            raise ValueError("causal mass and likelihood must align")
        causal_mass = causal_mass / np.sum(
            causal_mass, axis=-1, keepdims=True)
        statistic_mass = causal_mass / np.maximum(
            likelihood, np.finfo(float).tiny)
        statistic_mass /= np.sum(
            statistic_mass, axis=-1, keepdims=True)
        cavity_affinity = np.sum(
            np.sqrt(causal_mass * statistic_mass), axis=-1)
        cavity_surprise = np.clip(1.0 - cavity_affinity, 0.0, 1.0)
    else:
        statistic_mass = (
            collision_mass
            if nonzero_probability_mode in (
                "self_consistent", "transport_uncertain")
            else order_one
        )
    if statistic_mass is not None:
        residual_mean = np.sum(statistic_mass * residual, axis=-1)
        residual_variance = np.sum(
            statistic_mass * (residual - residual_mean[..., None]) ** 2,
            axis=-1,
        )
    scale = max(float(np.max(np.abs(image))), float(np.ptp(image)), 1.0)
    floor = np.finfo(float).eps * scale * scale
    residual_mean_square = residual_mean * residual_mean
    transport_contrast = np.zeros_like(residual_mean)
    if nonzero_probability_mode == "transport_uncertain":
        local_mass = np.asarray(law["local_mass"], dtype=np.float64)
        causal_mass = np.asarray(law["mass"], dtype=np.float64)
        if local_mass.shape != signal.shape or causal_mass.shape != signal.shape:
            raise ValueError("local and causal branch laws must align")
        local_mass = local_mass / np.sum(
            local_mass, axis=-1, keepdims=True)
        causal_mass = causal_mass / np.sum(
            causal_mass, axis=-1, keepdims=True)
        affinity = np.sum(np.sqrt(local_mass * causal_mass), axis=-1)
        transport_contrast = np.clip(1.0 - affinity, 0.0, 1.0)
        nonzero_evidence = (
            residual_mean_square + transport_contrast * residual_variance)
    elif nonzero_probability_mode == "root_resolved":
        nonzero_evidence = residual_mean_square + root_within_variance
    elif nonzero_probability_mode in (
        "mean", "self_consistent", "observation_cavity"):
        nonzero_evidence = residual_mean_square
    elif nonzero_probability_mode == "complete":
        nonzero_evidence = residual_mean_square + residual_variance
    else:
        raise ValueError(
            "nonzero_probability_mode must be 'mean', 'complete', "
            "'self_consistent', 'transport_uncertain', or "
            "'observation_cavity', or 'root_resolved'")
    nonzero_probability = nonzero_evidence / (
        nonzero_evidence + residual_variance + floor)
    nonzero_probability = np.clip(nonzero_probability, 0.0, 1.0)
    zero_probability = 1.0 - nonzero_probability

    # Symmetric component Haar measure: h0=1/2 and hk=(1/2)h_branch,k.
    # Log-density collision avoids underflow and exposes no temperature.
    log_half = -np.log(2.0)
    log_zero_weight = (
        log_half
        + order * (
            np.log(np.maximum(zero_probability, np.finfo(float).tiny))
            - log_half)
    )
    branch_reference = 0.5 * reference
    branch_probability = nonzero_probability[..., None] * order_one
    log_branch_weight = (
        np.log(np.maximum(branch_reference, np.finfo(float).tiny))
        + order[..., None] * (
            np.log(np.maximum(branch_probability, np.finfo(float).tiny))
            - np.log(np.maximum(branch_reference, np.finfo(float).tiny)))
    )
    gauge = np.maximum(
        log_zero_weight,
        np.max(log_branch_weight, axis=-1),
    )
    zero_weight = np.exp(log_zero_weight - gauge)
    branch_weight = np.exp(log_branch_weight - gauge[..., None])
    total = zero_weight + np.sum(branch_weight, axis=-1)
    zero_mass = zero_weight / total
    branch_mass = branch_weight / total[..., None]
    barycenter = (
        zero_mass * image + np.sum(branch_mass * signal, axis=-1))
    terminal_branch_barycenter = np.sum(collision_mass * signal, axis=-1)
    terminal_component_barycenter = (
        (1.0 - nonzero_probability) * image
        + nonzero_probability * terminal_branch_barycenter
    )

    branch_index = np.argmax(branch_mass, axis=-1)
    branch_mode = np.take_along_axis(
        signal, branch_index[..., None], axis=-1)[..., 0]
    maximum_branch_mass = np.max(branch_mass, axis=-1)
    mode = np.where(zero_mass >= maximum_branch_mass, image, branch_mode)
    lower = float(np.min(image))
    upper = float(np.max(image))
    readouts = {
        "component_barycenter": np.clip(barycenter, lower, upper),
        "terminal_component_barycenter": np.clip(
            terminal_component_barycenter, lower, upper),
        "component_mode": np.clip(mode, lower, upper),
        "zero_component_mass": zero_mass,
        "nonzero_probability": nonzero_probability,
    }
    return readouts, {
        "mean_nonzero_probability": float(np.mean(nonzero_probability)),
        "mean_zero_component_mass": float(np.mean(zero_mass)),
        "zero_component_mode_fraction": float(np.mean(
            zero_mass >= maximum_branch_mass)),
        "component_mass_maximum_error": float(np.max(np.abs(
            zero_mass + np.sum(branch_mass, axis=-1) - 1.0))),
        "terminal_component_mass_maximum_error": float(np.max(np.abs(
            (1.0 - nonzero_probability) + nonzero_probability - 1.0))),
        "mean_collision_order": float(np.mean(order)),
        "mean_transport_hellinger_contrast": float(np.mean(
            transport_contrast)),
        "mean_observation_cavity_surprise": float(np.mean(cavity_surprise)),
        "mean_root_within_residual_variance": float(np.mean(
            root_within_variance)),
        "mean_root_between_residual_variance": float(np.mean(
            root_between_variance)),
        "nonzero_probability_mode": nonzero_probability_mode,
    }


def phase_integrated_zero_residual_components_2d(
    observation: np.ndarray,
    *,
    angular_count: int = 4,
    quantile_count: int = 16,
    phase_count: int = 2,
    complete_residual_moment: bool = True,
    nonzero_probability_mode: NonzeroProbabilityMode = "mean",
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Integrate explicit component sections over population phase."""
    results, diagnostics = phase_integrated_zero_residual_component_family_2d(
        observation,
        angular_count=angular_count,
        quantile_count=quantile_count,
        phase_count=phase_count,
        complete_residual_moment=complete_residual_moment,
        nonzero_probability_modes=(nonzero_probability_mode,),
    )
    return results[nonzero_probability_mode], diagnostics[
        nonzero_probability_mode]


def phase_integrated_zero_residual_component_family_2d(
    observation: np.ndarray,
    *,
    angular_count: int = 4,
    quantile_count: int = 16,
    phase_count: int = 2,
    complete_residual_moment: bool = True,
    nonzero_probability_modes: tuple[NonzeroProbabilityMode, ...] = (
        "mean", "complete", "self_consistent", "transport_uncertain",
        "observation_cavity", "root_resolved",
    ),
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, dict[str, Any]],
]:
    """Integrate several component laws over one shared transport solve."""
    image = _validate(observation)
    modes = tuple(nonzero_probability_modes)
    if not modes or len(set(modes)) != len(modes):
        raise ValueError("component family modes must be nonempty and unique")
    phases = nested_population_phases(phase_count)
    sections: dict[str, list[dict[str, np.ndarray]]] = {
        mode: [] for mode in modes}
    diagnostics: dict[str, list[dict[str, Any]]] = {
        mode: [] for mode in modes}
    for phase in phases:
        law, law_diagnostic = causal_information_lineage_law_2d(
            image,
            angular_count=angular_count,
            quantile_count=quantile_count,
            population_phase=phase,
            complete_residual_moment=complete_residual_moment,
        )
        for mode in modes:
            readouts, diagnostic = zero_residual_component_readouts(
                image,
                law,
                nonzero_probability_mode=mode,
            )
            sections[mode].append(readouts)
            diagnostics[mode].append({**diagnostic, "law": law_diagnostic})
    results = {
        mode: {
            name: np.mean(np.stack([
                section[name] for section in sections[mode]]), axis=0)
            for name in sections[mode][0]
        }
        for mode in modes
    }
    summary = {
        mode: {
            "status": (
                "phase-integrated explicit zero/nonzero residual components"),
            "physical_parameters": "none",
            "phase_count": phase_count,
            "phases": phases,
            "complete_residual_moment": complete_residual_moment,
            "nonzero_probability_mode": mode,
            "mean_nonzero_probability": float(np.mean([
                row["mean_nonzero_probability"]
                for row in diagnostics[mode]])),
            "mean_zero_component_mass": float(np.mean([
                row["mean_zero_component_mass"]
                for row in diagnostics[mode]])),
            "mean_zero_component_mode_fraction": float(np.mean([
                row["zero_component_mode_fraction"]
                for row in diagnostics[mode]])),
            "mean_transport_hellinger_contrast": float(np.mean([
                row["mean_transport_hellinger_contrast"]
                for row in diagnostics[mode]])),
            "mean_observation_cavity_surprise": float(np.mean([
                row["mean_observation_cavity_surprise"]
                for row in diagnostics[mode]])),
            "mean_root_within_residual_variance": float(np.mean([
                row["mean_root_within_residual_variance"]
                for row in diagnostics[mode]])),
            "mean_root_between_residual_variance": float(np.mean([
                row["mean_root_between_residual_variance"]
                for row in diagnostics[mode]])),
            "component_mass_maximum_error": float(max(
                row["component_mass_maximum_error"]
                for row in diagnostics[mode])),
            "theory_status": "terminal component experiment; not promoted",
        }
        for mode in modes
    }
    return results, summary
