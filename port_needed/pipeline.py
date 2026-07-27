"""Supported one-decomposition segmentation pipeline used by the viewer."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from bfft.vision import SingleStageDecompositionObjective
from viewer.transport_voronoi import srgb_to_lab

from .allocation_flow import allocate_supports
from .anisotropic_edge_cost import (
    build_edge_costs,
    build_residual_pressure_costs,
)
from .frozen_meyer_geometry import build_frozen_geometry, restrict_geometry
from .density_population import (
    curvature_limited_geometry,
    emit_density_population,
)
from .continuous_eikonal_transport import (
    continuous_first_partition_prepared,
    prepare_continuous_metric,
)
from .first_arrival_site_force import safe_characteristic_site_step
from .fractional_interface_coverage import fractional_interface_coverage
from .hard_region_fit import fit_regions
from .residual_pressure_transport import relax_residual_pressure
from .reverse_residual_flow import reverse_residual_refill
from .soft_support_diffusion import (
    build_soft_support_conductance,
    diffuse_soft_support,
)
from .two_label_transport import (
    hard_partition_with_forest,
    local_hard_partition_with_forest,
)
from .wide_stencil_transport import (
    _metric_fields,
    build_wide_edge_costs,
    walk_wide_two_labels,
)


@dataclass(frozen=True)
class SegmentingConfig:
    allocation_method: str = "causal_density"
    allocation_max_side: int = 512
    tgfd_sweeps: int = 24
    flow_sweeps: int = 24
    threshold: float = 2.5
    metric_extent_threshold: float = 8.0
    metric_strength: float = 1.5
    transport_stencil_radius: int = 1
    minimum_region_pixels: int = 12
    maximum_rounds: int = 24
    safety_cells: int = 32768
    curvature_limited_density: bool = False
    null_evidence_strength: float = 0.5
    boundary_jump_strength: float = 24.0
    interface_coverage_strength: float = 0.4
    soft_support_passes: int = 0
    soft_support_coupling: float = 0.8
    soft_support_colour_percentile: float = 60.0
    branch_bins: int = 64
    characteristic_passes: int = 1
    characteristic_trust_fraction: float = 0.5
    characteristic_core_radius: float = 3.0
    ridge_count: int = 1
    refinement_iterations: int = 0
    refinement_error_ratio: float = 1.5
    refinement_return_distance: float = 8.0
    refinement_detail_gain: float = 1.0
    pressure_passes: int = 0
    pressure_strength: float = 1.0
    pressure_temperature: float = 2.0
    pressure_position_relaxation: float = 0.35
    pressure_capacity_relaxation: float = 0.5
    pressure_metric_gain: float = 4.0
    queue: str = "bucket"
    threads: int = 4


def build_segmenting_representation(
    rgb: np.ndarray,
    config: SegmentingConfig = SegmentingConfig(),
) -> dict:
    started = time.perf_counter()
    geometry = build_frozen_geometry(
        rgb,
        tgfd_sweeps=config.tgfd_sweeps,
        flow_sweeps=config.flow_sweeps,
        null_evidence_strength=config.null_evidence_strength,
        threads=config.threads,
    )
    if config.curvature_limited_density:
        geometry = curvature_limited_geometry(geometry)
    geometry_ms = 1000.0 * (time.perf_counter() - started)

    allocation_geometry = restrict_geometry(
        geometry, config.allocation_max_side)
    allocation_started = time.perf_counter()
    characteristic = None
    causal_allocation = config.allocation_method == "causal_density"
    if causal_allocation:
        if (
            int(config.refinement_iterations) > 0
            or int(config.pressure_passes) > 0
        ):
            raise ValueError(
                "causal density allocation cannot be combined with the "
                "legacy split or soft-pressure controls")
        centers, population = emit_density_population(
            allocation_geometry,
            safety_cells=config.safety_cells,
        )
        allocation_metric = prepare_continuous_metric(
            *_metric_fields(
                allocation_geometry,
                config.metric_strength,
                config.boundary_jump_strength,
            ))
        partition = continuous_first_partition_prepared(
            centers, allocation_metric)
        characteristic_trace = []
        for iteration in range(
            max(int(config.characteristic_passes), 0)
        ):
            centers, partition, diagnostic = safe_characteristic_site_step(
                centers,
                partition,
                allocation_metric,
                allocation_geometry["measure"],
                trust_fraction=config.characteristic_trust_fraction,
                core_radius_px=config.characteristic_core_radius,
            )
            diagnostic["iteration"] = iteration + 1
            characteristic_trace.append(diagnostic)
            if not diagnostic["accepted"]:
                break
        allocation = {
            "centers": centers,
            "cells": len(centers),
            "safety_limit_hit": population["safety_limit_hit"],
            "population": population,
            "transport_model": "continuous_first_arrival",
        }
        trace = []
        characteristic = {
            "initial_centers": (
                characteristic_trace[0]["initial_centers"]
                if characteristic_trace else centers.copy()
            ),
            "final_centers": centers.copy(),
            "trace": characteristic_trace,
            "population": population,
        }

        # One exact full-resolution causal refresh. Restriction emits the
        # population and relaxes its germs; it never classifies final pixels.
        if allocation_geometry["measure"].shape == geometry["measure"].shape:
            forest = partition
            full_metric = allocation_metric
        else:
            full_metric = prepare_continuous_metric(
                *_metric_fields(
                    geometry,
                    config.metric_strength,
                    config.boundary_jump_strength,
                ))
            forest = continuous_first_partition_prepared(
                centers, full_metric)
        labels = forest["labels"]
        full_costs = None
        full_directions = None
    elif config.allocation_method == "legacy_bifurcation":
        _, allocation, trace = allocate_supports(
            allocation_geometry,
            threshold=config.threshold,
            metric_extent_threshold=config.metric_extent_threshold,
            metric_strength=config.metric_strength,
            minimum_region_pixels=config.minimum_region_pixels,
            maximum_rounds=config.maximum_rounds,
            safety_cells=config.safety_cells,
            branch_bins=config.branch_bins,
            transport_queue=config.queue,
            transport_stencil_radius=config.transport_stencil_radius,
        )
        centers = allocation["centers"]

        # One exact full-resolution refresh. Allocation may run on a
        # restricted sample, but every output pixel uses original geometry.
        if int(config.transport_stencil_radius) > 1:
            full_costs, full_directions = build_wide_edge_costs(
                geometry,
                config.metric_strength,
                config.transport_stencil_radius,
            )
        else:
            full_costs = build_edge_costs(
                geometry, config.metric_strength)
            full_directions = None
        forest = (
            walk_wide_two_labels(centers, full_costs, full_directions)
            if full_directions is not None
            else hard_partition_with_forest(centers, full_costs)
        )
        labels = forest["labels"]
    else:
        raise ValueError(
            f"unknown allocation method: {config.allocation_method}")
    allocation_ms = 1000.0 * (time.perf_counter() - allocation_started)

    fit_started = time.perf_counter()
    objective = SingleStageDecompositionObjective(
        rgb,
        passes=config.tgfd_sweeps,
        threads=config.threads,
        solver=1,
    )
    target_lab = srgb_to_lab(rgb)
    has_evolution = (
        int(config.refinement_iterations) > 0
        or int(config.pressure_passes) > 0
    )
    refinement_ridges = 0 if has_evolution else config.ridge_count
    record, reconstruction_lab, ridge = fit_regions(
        labels,
        centers,
        target_lab,
        objective,
        ridge_count=refinement_ridges,
    )
    pressure = None
    if int(config.pressure_passes) > 0:
        initial_pressure_objective = record["objective"]
        pressure_costs, metric_pressure, metric_coherence = (
            build_residual_pressure_costs(
            geometry,
            config.metric_strength,
            objective.last_residual_energy,
            config.pressure_metric_gain,
            )
        )
        centers, reach, forest, pressure_trace, pressure_fields = (
            relax_residual_pressure(
                centers,
                pressure_costs,
                geometry["measure"],
                objective.last_residual_energy * metric_coherence,
                passes=config.pressure_passes,
                pressure_strength=config.pressure_strength,
                temperature=config.pressure_temperature,
                position_relaxation=config.pressure_position_relaxation,
                capacity_relaxation=config.pressure_capacity_relaxation,
            )
        )
        labels = forest["labels"]
        record, reconstruction_lab, ridge = fit_regions(
            labels,
            centers,
            target_lab,
            objective,
            ridge_count=0,
        )
        pressure = {
            **pressure_fields,
            "metric_pressure": metric_pressure,
            "metric_coherence": metric_coherence,
            "reach": reach,
            "trace": pressure_trace,
            "initial_objective": initial_pressure_objective,
            "final_affine_objective": record["objective"],
            "objective_improved": (
                record["objective"] <= initial_pressure_objective),
        }
    refinements = []
    for iteration in range(max(int(config.refinement_iterations), 0)):
        refinement_started = time.perf_counter()
        previous_centers = centers
        previous_labels = labels
        previous_forest = forest
        previous_record = record
        previous_reconstruction_lab = reconstruction_lab
        previous_ridge = ridge
        centers, diagnostic = reverse_residual_refill(
            labels,
            centers,
            forest,
            objective.last_residual_energy,
            geometry["measure"],
            error_ratio_threshold=config.refinement_error_ratio,
            return_distance_threshold=config.refinement_return_distance,
            minimum_region_pixels=config.minimum_region_pixels,
            detail_gain=config.refinement_detail_gain,
            bins=config.branch_bins,
            safety_cells=config.safety_cells,
        )
        diagnostic["iteration"] = iteration + 1
        if diagnostic["split_count"] == 0:
            diagnostic["accepted"] = False
            diagnostic["milliseconds"] = (
                1000.0 * (time.perf_counter() - refinement_started))
            refinements.append(diagnostic)
            break
        centers, forest = local_hard_partition_with_forest(
            centers,
            diagnostic["parent_of_centers"],
            previous_labels,
            full_costs,
        )
        labels = forest["labels"]
        diagnostic["parent_boundary_escape_fraction"] = float(np.mean(
            diagnostic["parent_of_centers"][labels] != previous_labels))
        record, reconstruction_lab, ridge = fit_regions(
            labels,
            centers,
            target_lab,
            objective,
            ridge_count=0,
        )
        if record["objective"] > previous_record["objective"]:
            diagnostic["accepted"] = False
            diagnostic["rejected_objective"] = record["objective"]
            centers = previous_centers
            labels = previous_labels
            forest = previous_forest
            record = previous_record
            reconstruction_lab = previous_reconstruction_lab
            ridge = previous_ridge
            objective.evaluate(previous_record["rgb"])
            diagnostic["milliseconds"] = (
                1000.0 * (time.perf_counter() - refinement_started))
            refinements.append(diagnostic)
            break
        diagnostic["accepted"] = True
        diagnostic["milliseconds"] = (
            1000.0 * (time.perf_counter() - refinement_started))
        refinements.append(diagnostic)
    if has_evolution and int(config.ridge_count) > 0:
        record, reconstruction_lab, ridge = fit_regions(
            labels,
            centers,
            target_lab,
            objective,
            ridge_count=config.ridge_count,
            affine_record=record,
            affine=reconstruction_lab,
        )
    hard_record = record
    interface_coverage = None
    if (
        causal_allocation
        and float(config.interface_coverage_strength) > 0.0
    ):
        proposal_rgb, interface_coverage = fractional_interface_coverage(
            record["rgb"],
            labels,
            forest["distance"],
            full_metric["cardinal_costs"],
            geometry["boundary_confidence"],
            strength=config.interface_coverage_strength,
        )
        proposal_record = objective.evaluate(proposal_rgb)
        proposal_record["rgb"] = proposal_rgb
        interface_coverage["hard_record"] = hard_record
        interface_coverage["proposal_record"] = proposal_record
        interface_coverage["accepted"] = (
            proposal_record["objective"] <= hard_record["objective"])
        if interface_coverage["accepted"]:
            record = proposal_record
            reconstruction_lab = srgb_to_lab(proposal_rgb)
        else:
            objective.evaluate(hard_record["rgb"])
    # Soft support, when enabled, is judged against the best accepted hard
    # readout including its geometric interface coverage.
    hard_record = record
    soft_support = None
    if int(config.soft_support_passes) > 0:
        conductance = build_soft_support_conductance(
            geometry,
            rgb,
            metric_strength=config.metric_strength,
            colour_percentile=config.soft_support_colour_percentile,
        )
        proposal_rgb = np.clip(diffuse_soft_support(
            record["rgb"],
            conductance,
            passes=config.soft_support_passes,
            coupling=config.soft_support_coupling,
        ), 0.0, 1.0)
        proposal_record = objective.evaluate(proposal_rgb)
        # SingleStageDecompositionObjective evaluates scores only.  The
        # established fit/score contract also carries the rendered RGB field,
        # which every viewer and downstream refinement reads from ``record``.
        proposal_record["rgb"] = proposal_rgb
        accepted = (
            proposal_record["objective"] <= hard_record["objective"])
        if accepted:
            record = proposal_record
            reconstruction_lab = srgb_to_lab(proposal_rgb)
        else:
            objective.evaluate(hard_record["rgb"])
        soft_support = {
            "conductance": conductance,
            "passes": int(config.soft_support_passes),
            "coupling": float(config.soft_support_coupling),
            "hard_record": hard_record,
            "proposal_record": proposal_record,
            "accepted": accepted,
        }
    fit_ms = 1000.0 * (time.perf_counter() - fit_started)
    return {
        "geometry": geometry,
        "allocation_geometry": allocation_geometry,
        "labels": labels,
        "centers": centers,
        "trace": trace,
        "record": record,
        "reconstruction_lab": reconstruction_lab,
        "ridge": ridge,
        "soft_support": soft_support,
        "interface_coverage": interface_coverage,
        "refinements": refinements,
        "pressure": pressure,
        "characteristic": characteristic,
        "residual_energy": objective.last_residual_energy,
        "timing": {
            "geometry_ms": geometry_ms,
            "allocation_ms": allocation_ms,
            "fit_ms": fit_ms,
            "total_ms": geometry_ms + allocation_ms + fit_ms,
        },
    }
