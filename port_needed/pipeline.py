"""Supported one-decomposition segmentation pipeline used by the viewer."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from bfft.vision import SingleStageDecompositionObjective
from viewer.transport_voronoi import srgb_to_lab

from .allocation_flow import allocate_supports
from .anisotropic_edge_cost import build_edge_costs
from .frozen_meyer_geometry import build_frozen_geometry, restrict_geometry
from .hard_region_fit import fit_regions
from .two_label_transport import hard_partition


@dataclass(frozen=True)
class SegmentingConfig:
    allocation_max_side: int = 512
    tgfd_sweeps: int = 24
    flow_sweeps: int = 24
    threshold: float = 2.5
    metric_extent_threshold: float = 8.0
    metric_strength: float = 1.5
    minimum_region_pixels: int = 12
    maximum_rounds: int = 24
    safety_cells: int = 8192
    branch_bins: int = 64
    ridge_count: int = 1
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
        threads=config.threads,
    )
    geometry_ms = 1000.0 * (time.perf_counter() - started)

    allocation_geometry = restrict_geometry(
        geometry, config.allocation_max_side)
    allocation_started = time.perf_counter()
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
    )
    centers = allocation["centers"]

    # One exact full-resolution refresh.  Allocation may run on a restricted
    # sample of the frozen support, but every output pixel is classified by
    # the original-resolution metric.
    full_costs = build_edge_costs(geometry, config.metric_strength)
    labels = hard_partition(centers, full_costs, queue=config.queue)
    allocation_ms = 1000.0 * (time.perf_counter() - allocation_started)

    fit_started = time.perf_counter()
    objective = SingleStageDecompositionObjective(
        rgb,
        passes=config.tgfd_sweeps,
        threads=config.threads,
        solver=1,
    )
    record, reconstruction_lab, ridge = fit_regions(
        labels,
        centers,
        srgb_to_lab(rgb),
        objective,
        ridge_count=config.ridge_count,
    )
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
        "timing": {
            "geometry_ms": geometry_ms,
            "allocation_ms": allocation_ms,
            "fit_ms": fit_ms,
            "total_ms": geometry_ms + allocation_ms + fit_ms,
        },
    }
