#!/usr/bin/env python3
"""Emit a complete support population directly from the BFFT pass volume.

This is the first deliberately one-shot control for the flow-volume idea:

* a small circular R2 coat supplies broad, cheap coverage;
* pass-to-pass event amplitude is a continuous emission measure;
* deterministic local stochastic rounding turns that measure into splats;
* the normalized event tensor supplies each splat's axes and orientation;
* one global affine partition-of-unity fit measures the frozen support.

There is no image-space residual loop, owner map, candidate search, ranking,
top-k operation, cell deletion, or geometric refinement.  ``detail_cells`` is
only a matched-complexity research control: internally it becomes one mass
quantum, and every voxel independently emits when its own accumulated flow
crosses that quantum.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from scipy.fft import dctn, idctn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "viewer"))
sys.path.insert(0, str(ROOT / "experiments"))

import gallery  # noqa: E402
from bfft.vision import SingleStageDecompositionObjective  # noqa: E402
from dual_aperture_support import (  # noqa: E402
    aperture,
    design_matrix,
    score,
    solve_field,
    target_layers,
)
from bfft_flow_stage_geometry import build_flow_volume  # noqa: E402
from resource_transport_cells import _r2_sites  # noqa: E402
from transport_voronoi import _fit_rgb, srgb_to_lab  # noqa: E402


@dataclass
class Population:
    centers: np.ndarray
    major: np.ndarray
    minor: np.ndarray
    angle: np.ndarray
    stage: np.ndarray
    base_cells: int

    def circularized(self) -> "Population":
        radius = np.sqrt(self.major * self.minor)
        return Population(
            self.centers.copy(),
            radius,
            radius,
            np.zeros_like(self.angle),
            self.stage.copy(),
            self.base_cells,
        )


def _transport_hinted_substrate(
    volume: dict,
    centers: np.ndarray,
    radius: float,
    height: int,
    width: int,
    density_hint: float,
    shape_hint: float,
    vector_mode: str,
    vector_axis: str,
    transport_hint: float,
    transport_consistency_power: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """One closed-form transport from uniform germs to slow flow density."""
    if len(centers) == 0:
        empty = np.empty(0, dtype=np.float64)
        return centers, empty, empty, empty, {
            "substrate_displacement_rms": 0.0,
            "substrate_ratio_median": 1.0,
        }
    energy = np.asarray(volume["energy"], dtype=np.float64)
    amplitude = np.sqrt(np.maximum(energy, 0.0))
    slow_mass = ndi.gaussian_filter(
        np.sum(amplitude, axis=0),
        sigma=max(0.65 * radius, 1.0),
        mode="reflect",
    )
    normalized = slow_mass / max(float(np.mean(slow_mass)), 1e-30)
    hint = float(np.clip(density_hint, 0.0, 1.0))
    desired_density = (1.0 - hint) + hint * normalized

    # Linearized Monge transport.  If x' = x + grad(phi), the transported
    # density is 1 - Laplacian(phi), so solve Δphi = 1-rho with Neumann
    # boundaries by one orthonormal DCT.
    source = 1.0 - desired_density
    source_hat = dctn(source, type=2, norm="ortho")
    eig_y = 2.0 * np.cos(np.pi * np.arange(height) / height) - 2.0
    eig_x = 2.0 * np.cos(np.pi * np.arange(width) / width) - 2.0
    eigenvalue = eig_y[:, None] + eig_x[None, :]
    potential_hat = np.zeros_like(source_hat)
    nonzero = np.abs(eigenvalue) > 1e-14
    potential_hat[nonzero] = source_hat[nonzero] / eigenvalue[nonzero]
    potential = idctn(potential_hat, type=2, norm="ortho")
    velocity_y, velocity_x = np.gradient(potential)
    coordinates = np.vstack([centers[:, 1], centers[:, 0]])
    dx = ndi.map_coordinates(
        velocity_x, coordinates, order=1, mode="nearest")
    dy = ndi.map_coordinates(
        velocity_y, coordinates, order=1, mode="nearest")
    moved = centers + np.column_stack([dx, dy])
    moved[:, 0] = np.clip(moved[:, 0], 0.0, width - 1.0)
    moved[:, 1] = np.clip(moved[:, 1], 0.0, height - 1.0)
    density_moved = moved.copy()

    # Unlike the Poisson density control, this is an actual path through the
    # decomposition states.  Each germ samples the next state transition at
    # its current location and is carried forward once.
    transport_x = np.asarray(volume["transport_x"], dtype=np.float64)
    transport_y = np.asarray(volume["transport_y"], dtype=np.float64)
    transport_confidence = np.asarray(
        volume["transport_confidence"], dtype=np.float64)
    persistence = np.asarray(
        volume["transport_persistence"], dtype=np.float64)
    transport_displacement = np.zeros_like(moved)
    for stage in range(transport_x.shape[0]):
        coordinates = np.vstack([moved[:, 1], moved[:, 0]])
        confidence = ndi.map_coordinates(
            transport_confidence[stage],
            coordinates,
            order=1,
            mode="nearest",
        )
        if transport_consistency_power > 0.0:
            confidence *= ndi.map_coordinates(
                persistence,
                coordinates,
                order=1,
                mode="nearest",
            ) ** float(transport_consistency_power)
        step_x = ndi.map_coordinates(
            transport_x[stage], coordinates, order=1, mode="nearest")
        step_y = ndi.map_coordinates(
            transport_y[stage], coordinates, order=1, mode="nearest")
        step = float(transport_hint) * confidence[:, None] * np.column_stack([
            step_x, step_y])
        moved += step
        transport_displacement += step
        moved[:, 0] = np.clip(moved[:, 0], 0.0, width - 1.0)
        moved[:, 1] = np.clip(moved[:, 1], 0.0, height - 1.0)

    if vector_mode == "event":
        # Axial averaging avoids assigning an arbitrary sign to a support
        # tangent.  This is retained as the misspecified control.
        angle = np.asarray(volume["angle"], dtype=np.float64)
        coherence = np.asarray(volume["coherence"], dtype=np.float64)
        axial_weight = amplitude * coherence
        axial_cos = ndi.gaussian_filter(
            np.sum(axial_weight * np.cos(2.0 * angle), axis=0),
            sigma=max(0.4 * radius, 1.0),
            mode="reflect",
        )
        axial_sin = ndi.gaussian_filter(
            np.sum(axial_weight * np.sin(2.0 * angle), axis=0),
            sigma=max(0.4 * radius, 1.0),
            mode="reflect",
        )
        total = ndi.gaussian_filter(
            np.sum(axial_weight, axis=0),
            sigma=max(0.4 * radius, 1.0),
            mode="reflect",
        )
        substrate_angle_field = 0.5 * np.arctan2(axial_sin, axial_cos)
        substrate_coherence_field = np.hypot(
            axial_cos, axial_sin) / np.maximum(total, 1e-30)
    elif vector_mode == "glass":
        # The actual signed low-frequency transport hint: the gradient of
        # the final cartoon-to-TV defect, observed at substrate scale.
        glass = np.asarray(volume["defect"], dtype=np.float64)[-1]
        gx = ndi.gaussian_filter(
            ndi.sobel(glass, axis=1, mode="reflect") / 8.0,
            sigma=max(0.4 * radius, 1.0),
            mode="reflect",
        )
        gy = ndi.gaussian_filter(
            ndi.sobel(glass, axis=0, mode="reflect") / 8.0,
            sigma=max(0.4 * radius, 1.0),
            mode="reflect",
        )
        substrate_angle_field = np.arctan2(gy, gx)
        magnitude = np.hypot(gx, gy)
        scale = max(float(np.percentile(magnitude, 95.0)), 1e-30)
        substrate_coherence_field = np.clip(magnitude / scale, 0.0, 1.0)
    else:
        raise ValueError(f"unknown substrate vector mode {vector_mode!r}")
    if vector_axis == "tangent":
        substrate_angle_field = substrate_angle_field + 0.5 * np.pi
    elif vector_axis != "normal":
        raise ValueError(f"unknown substrate vector axis {vector_axis!r}")
    moved_coordinates = np.vstack([moved[:, 1], moved[:, 0]])
    substrate_angle = ndi.map_coordinates(
        substrate_angle_field, moved_coordinates, order=1, mode="nearest")
    substrate_coherence = np.clip(ndi.map_coordinates(
        substrate_coherence_field,
        moved_coordinates,
        order=1,
        mode="nearest",
    ), 0.0, 1.0)
    ratio = 1.0 + max(float(shape_hint), 0.0) * substrate_coherence
    major = radius * np.sqrt(ratio)
    minor = radius / np.sqrt(ratio)
    return moved, major, minor, substrate_angle, {
        "substrate_displacement_rms": float(np.sqrt(np.mean(dx * dx + dy * dy))),
        "substrate_displacement_q95": float(
            np.percentile(np.hypot(dx, dy), 95.0)),
        "substrate_transport_displacement_rms": float(np.sqrt(np.mean(
            np.sum(transport_displacement * transport_displacement, axis=1)))),
        "substrate_density_displacement_rms": float(np.sqrt(np.mean(
            np.sum((density_moved - centers) ** 2, axis=1)))),
        "substrate_ratio_median": float(np.median(ratio)),
        "substrate_density_min": float(np.min(desired_density)),
        "substrate_density_max": float(np.max(desired_density)),
        "substrate_vector_mode": vector_mode,
        "substrate_vector_axis": vector_axis,
    }


def _load_image(path: str | None, gallery_key: str) -> tuple[np.ndarray, str]:
    if path:
        from skimage.io import imread

        resolved = Path(path).expanduser().resolve()
        return imread(resolved), str(resolved)
    return gallery.load(gallery_key), f"gallery:{gallery_key}"


def _hash01(x: np.ndarray, y: np.ndarray, stage: np.ndarray, salt: float) -> np.ndarray:
    """Deterministic spatial dither; not a sequence, search, or ordering."""
    value = np.sin(
        (x + 1.0) * 12.9898
        + (y + 1.0) * 78.233
        + (stage + 1.0) * 37.719
        + salt
    ) * 43758.5453123
    return value - np.floor(value)


def emit_population(
    volume: dict,
    height: int,
    width: int,
    base_cells: int,
    detail_cells: int,
    first_detail_stage: int,
    base_overlap: float,
    max_major: float,
    curvature_limit: bool,
    specificity_power: float,
    specificity_mix: float,
    emission_mode: str,
    substrate_hint: float,
    substrate_shape_hint: float,
    substrate_vector_mode: str,
    substrate_vector_axis: str,
    substrate_transport_hint: float,
    splat_advection: float,
    transport_consistency_power: float,
) -> tuple[Population, dict[str, float]]:
    """Locally quantize flow amplitude into ellipse splats."""
    base_centers = _r2_sites(base_cells, width, height)
    base_radius = math.sqrt(
        base_overlap * height * width /
        (math.pi * max(base_cells, 1)))
    (
        base_centers,
        base_major,
        base_minor,
        base_angle,
        substrate_report,
    ) = _transport_hinted_substrate(
        volume,
        base_centers,
        base_radius,
        height,
        width,
        substrate_hint,
        substrate_shape_hint,
        substrate_vector_mode,
        substrate_vector_axis,
        substrate_transport_hint,
        transport_consistency_power,
    )

    energy = np.asarray(volume["energy"], dtype=np.float64)
    first = max(int(first_detail_stage) - 1, 0)
    amplitude = np.sqrt(np.maximum(energy[first:], 0.0))
    major_key = "major_curvature" if curvature_limit else "major"
    capacity = (
        math.pi
        * np.asarray(volume[major_key], dtype=np.float64)[first:]
        * np.asarray(volume["minor"], dtype=np.float64)[first:]
    )
    # A flow event does not merely carry amplitude; it also states how much
    # image area one locally valid support can explain.  Dividing by that
    # capacity makes specificity itself increase germ density.
    specific_measure = amplitude / np.maximum(
        capacity, 1e-12) ** float(specificity_power)
    amplitude_total = max(float(np.sum(amplitude)), 1e-30)
    specific_total = max(float(np.sum(specific_measure)), 1e-30)
    mix = float(np.clip(specificity_mix, 0.0, 1.0))
    # Preserve a broad-flow population while allowing specificity to add a
    # second emission pressure.  Both measures are normalized before mixing,
    # so the research cell count remains matched.
    emission_measure = (
        (1.0 - mix) * amplitude / amplitude_total
        + mix * specific_measure / specific_total
    )
    quantum = 1.0 / max(int(detail_cells), 1)
    expected = emission_measure / quantum

    stages, yy, xx = np.indices(expected.shape)
    stages = stages + first
    if emission_mode == "causal":
        counts = np.zeros(expected.shape, dtype=np.int16)
        base_y, base_x = np.indices((height, width))
        phase = _hash01(
            base_x, base_y, np.zeros_like(base_x), 0.0) * quantum
        for local_stage in range(expected.shape[0]):
            accumulated = phase + emission_measure[local_stage]
            counts[local_stage] = np.floor(
                accumulated / quantum).astype(np.int16)
            phase = accumulated - counts[local_stage] * quantum
    elif emission_mode == "independent":
        whole = np.floor(expected).astype(np.int16)
        fraction = expected - whole
        accepted_extra = _hash01(xx, yy, stages, 0.0) < fraction
        counts = whole + accepted_extra.astype(np.int16)
    else:
        raise ValueError(f"unknown emission mode {emission_mode!r}")
    selected = np.flatnonzero(counts.ravel() > 0)
    repeats = counts.ravel()[selected]
    flat = np.repeat(selected, repeats)
    stage_index, py, px = np.unravel_index(flat, expected.shape)
    stage_index = stage_index + first

    # Subpixel phase is independent of the emission dither.
    jx = _hash01(px, py, stage_index, 19.19) - 0.5
    jy = _hash01(px, py, stage_index, 47.47) - 0.5
    detail_centers = np.column_stack([
        np.clip(px + jx, 0.0, width - 1.0),
        np.clip(py + jy, 0.0, height - 1.0),
    ])
    unadvected_centers = detail_centers.copy()
    if float(splat_advection) != 0.0 and len(detail_centers):
        transport_x = np.asarray(volume["transport_x"], dtype=np.float64)
        transport_y = np.asarray(volume["transport_y"], dtype=np.float64)
        transport_confidence = np.asarray(
            volume["transport_confidence"], dtype=np.float64)
        persistence = np.asarray(
            volume["transport_persistence"], dtype=np.float64)
        # A cell born in state s is carried by transitions s+1 ... P.
        for transition in range(1, transport_x.shape[0]):
            active = stage_index < transition
            if not np.any(active):
                continue
            coordinates = np.vstack([
                detail_centers[active, 1],
                detail_centers[active, 0],
            ])
            confidence = ndi.map_coordinates(
                transport_confidence[transition],
                coordinates,
                order=1,
                mode="nearest",
            )
            if transport_consistency_power > 0.0:
                confidence *= ndi.map_coordinates(
                    persistence,
                    coordinates,
                    order=1,
                    mode="nearest",
                ) ** float(transport_consistency_power)
            detail_centers[active, 0] += (
                float(splat_advection)
                * confidence
                * ndi.map_coordinates(
                    transport_x[transition],
                    coordinates,
                    order=1,
                    mode="nearest",
                )
            )
            detail_centers[active, 1] += (
                float(splat_advection)
                * confidence
                * ndi.map_coordinates(
                    transport_y[transition],
                    coordinates,
                    order=1,
                    mode="nearest",
                )
            )
            detail_centers[active, 0] = np.clip(
                detail_centers[active, 0], 0.0, width - 1.0)
            detail_centers[active, 1] = np.clip(
                detail_centers[active, 1], 0.0, height - 1.0)
    detail_minor = np.asarray(volume["minor"])[stage_index, py, px]
    major_field = major_key
    detail_major = np.minimum(
        np.asarray(volume[major_field])[stage_index, py, px],
        float(max_major),
    )
    detail_minor = np.clip(detail_minor, 0.85, detail_major)
    detail_angle = np.asarray(volume["angle"])[stage_index, py, px]

    population = Population(
        centers=np.vstack([base_centers, detail_centers]),
        major=np.concatenate([
            base_major,
            detail_major,
        ]),
        minor=np.concatenate([
            base_minor,
            detail_minor,
        ]),
        angle=np.concatenate([
            base_angle,
            detail_angle,
        ]),
        stage=np.concatenate([
            np.zeros(base_cells, dtype=np.int16),
            (stage_index + 1).astype(np.int16),
        ]),
        base_cells=base_cells,
    )
    detail_area = np.pi * detail_major * detail_minor
    report = {
        "mass_quantum": quantum,
        "expected_detail_cells": float(np.sum(expected)),
        "emitted_detail_cells": int(len(detail_centers)),
        "detail_minor_median": float(np.median(detail_minor)),
        "detail_major_median": float(np.median(detail_major)),
        "detail_ratio_median": float(
            np.median(detail_major / np.maximum(detail_minor, 1e-12))),
        "detail_area_sum_per_pixel": float(
            np.sum(detail_area) / (height * width)),
        "curvature_limit": bool(curvature_limit),
        "specificity_power": float(specificity_power),
        "specificity_mix": mix,
        "emission_mode": emission_mode,
        "substrate_hint": float(substrate_hint),
        "substrate_shape_hint": float(substrate_shape_hint),
        "substrate_transport_hint": float(substrate_transport_hint),
        "splat_advection": float(splat_advection),
        "transport_consistency_power": float(
            transport_consistency_power),
        "splat_advection_rms": float(np.sqrt(np.mean(np.sum(
            (detail_centers - unadvected_centers) ** 2, axis=1))))
        if len(detail_centers) else 0.0,
        **substrate_report,
    }
    return population, report


def r2_control(
    cells: int, height: int, width: int, overlap: float,
) -> Population:
    radius = math.sqrt(
        overlap * height * width / (math.pi * max(cells, 1)))
    return Population(
        centers=_r2_sites(cells, width, height),
        major=np.full(cells, radius),
        minor=np.full(cells, radius),
        angle=np.zeros(cells),
        stage=np.zeros(cells, dtype=np.int16),
        base_cells=cells,
    )


def support_samples(
    population: Population,
    height: int,
    width: int,
    power: float = 2.0,
) -> dict[str, np.ndarray | int | float]:
    """Rasterize compact ellipses into the existing exact fit interface."""
    spacing = math.sqrt(height * width / max(len(population.centers), 1))
    rows: list[np.ndarray] = []
    sites: list[np.ndarray] = []
    phis: list[np.ndarray] = []
    bases: list[np.ndarray] = []

    for site, ((cx, cy), major, minor, theta) in enumerate(zip(
        population.centers,
        population.major,
        population.minor,
        population.angle,
    )):
        major = max(float(major), 0.75)
        minor = max(float(minor), 0.75)
        ct, st = math.cos(float(theta)), math.sin(float(theta))
        extent_x = math.sqrt((major * ct) ** 2 + (minor * st) ** 2)
        extent_y = math.sqrt((major * st) ** 2 + (minor * ct) ** 2)
        x0 = max(0, int(math.floor(cx - extent_x)))
        x1 = min(width, int(math.ceil(cx + extent_x)) + 1)
        y0 = max(0, int(math.floor(cy - extent_y)))
        y1 = min(height, int(math.ceil(cy + extent_y)) + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dx = xx - cx
        dy = yy - cy
        along = dx * ct + dy * st
        across = -dx * st + dy * ct
        q = (along / major) ** 2 + (across / minor) ** 2
        visible = q < 1.0
        if not np.any(visible):
            continue
        phi = np.maximum(1.0 - q[visible], 0.0) ** power
        count = int(np.sum(visible))
        rows.append((yy[visible] * width + xx[visible]).astype(np.int32))
        sites.append(np.full(count, site, dtype=np.int32))
        phis.append(phi)
        bases.append(np.column_stack([
            np.ones(count),
            dx[visible] / max(spacing, 1e-12),
            dy[visible] / max(spacing, 1e-12),
        ]))
    return {
        "rows": np.concatenate(rows),
        "sites": np.concatenate(sites),
        "phi": np.concatenate(phis),
        "basis": np.concatenate(bases),
        "cells": len(population.centers),
        "spacing": spacing,
    }


def fit_population(
    population: Population,
    target_lab: np.ndarray,
    objective: SingleStageDecompositionObjective,
    temperature: float = 1.0,
) -> tuple[dict, np.ndarray, dict]:
    height, width = target_lab.shape[:2]
    samples = support_samples(population, height, width)
    weights, dominance, effective = aperture(
        samples, height * width, float(temperature))
    design = design_matrix(samples, height * width, weights)
    uncovered = np.asarray(design.getnnz(axis=1) == 0).reshape(height, width)
    reconstruction = solve_field(design, target_lab)
    record = score(objective, objective.target_rgb, reconstruction)
    covered = ~uncovered.ravel()
    diagnostics = {
        "samples": int(len(samples["rows"])),
        "uncovered_fraction": float(np.mean(uncovered)),
        "dominance_mean": float(np.mean(dominance)),
        "effective_contributors_mean": float(np.mean(effective[covered])),
        "effective_contributors_median": float(np.median(effective[covered])),
        "partition_temperature": float(temperature),
    }
    return record, reconstruction, diagnostics


def fit_layered_population(
    population: Population,
    target_rgb: np.ndarray,
    objective: SingleStageDecompositionObjective,
) -> tuple[dict, np.ndarray, dict]:
    """Use local round supports for cartoon and flow ellipses for texture."""
    target_lab, cartoon_lab, texture_lab = target_layers(target_rgb)
    height, width = target_lab.shape[:2]
    ellipse_samples = support_samples(population, height, width)
    circle_samples = support_samples(
        population.circularized(), height, width)
    ellipse_weight, ellipse_dominance, ellipse_effective = aperture(
        ellipse_samples, height * width, 1.0)
    circle_weight, circle_dominance, circle_effective = aperture(
        circle_samples, height * width, 1.0)
    ellipse_design = design_matrix(
        ellipse_samples, height * width, ellipse_weight)
    circle_design = design_matrix(
        circle_samples, height * width, circle_weight)
    cartoon_field = solve_field(circle_design, cartoon_lab)
    texture_field = solve_field(ellipse_design, texture_lab)
    reconstruction = cartoon_field + texture_field
    record = score(objective, target_rgb, reconstruction)
    diagnostics = {
        "cartoon_dominance_mean": float(np.mean(circle_dominance)),
        "texture_dominance_mean": float(np.mean(ellipse_dominance)),
        "cartoon_effective_median": float(np.median(circle_effective)),
        "texture_effective_median": float(np.median(ellipse_effective)),
    }
    return record, reconstruction, diagnostics


def _serializable(record: dict) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in record.items()
        if key != "rgb"
    }


def save_panel(
    rgb: np.ndarray,
    volume: dict,
    population: Population,
    records: list[tuple[str, dict]],
    output: Path,
) -> None:
    columns = 1 + len(records)
    fig, axes = plt.subplots(2, columns, figsize=(4 * columns, 8.5))
    top = [("target", {"rgb": rgb})] + records
    for axis, (name, record) in zip(axes[0], top):
        axis.imshow(record["rgb"])
        if "psnr" in record:
            axis.set_title(
                f"{name}\n{record['psnr']:.2f} dB  "
                f"obj {record['objective']:.4g}")
        else:
            axis.set_title(name)

    integrated = np.asarray(volume["integrated_energy"])
    axes[1, 0].imshow(np.sqrt(integrated), cmap="magma")
    axes[1, 0].set_title("integrated flow amplitude")
    axes[1, 1].imshow(rgb, alpha=0.28)
    detail = np.arange(len(population.centers)) >= population.base_cells
    scatter = axes[1, 1].scatter(
        population.centers[detail, 0],
        population.centers[detail, 1],
        c=population.stage[detail],
        s=np.clip(population.major[detail] * population.minor[detail], 2, 35),
        cmap="turbo",
        alpha=0.68,
        linewidths=0,
    )
    axes[1, 1].set_title("flow-born splats (colour = pass)")
    fig.colorbar(scatter, ax=axes[1, 1], fraction=0.046)

    for axis, (name, record) in zip(axes[1, 2:], records):
        error = np.sqrt(np.mean((rgb - record["rgb"]) ** 2, axis=2))
        axis.imshow(error, cmap="inferno", vmin=0.0, vmax=np.percentile(error, 99))
        axis.set_title(f"{name} RGB error")
    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?")
    parser.add_argument("--gallery", default="pikachu")
    parser.add_argument("--side", type=int, default=128)
    parser.add_argument("--passes", type=int, default=24)
    parser.add_argument("--base-cells", type=int, default=180)
    parser.add_argument("--detail-cells", type=int, default=800)
    parser.add_argument("--first-detail-stage", type=int, default=2)
    parser.add_argument("--base-overlap", type=float, default=5.0)
    parser.add_argument("--max-major", type=float, default=18.0)
    parser.add_argument(
        "--curvature-limit",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--specificity-power", type=float, default=0.0)
    parser.add_argument("--specificity-mix", type=float, default=1.0)
    parser.add_argument(
        "--emission-mode",
        choices=("independent", "causal"),
        default="independent",
    )
    parser.add_argument("--substrate-hint", type=float, default=0.0)
    parser.add_argument("--substrate-shape-hint", type=float, default=0.0)
    parser.add_argument(
        "--substrate-vector-mode",
        choices=("glass", "event"),
        default="glass",
    )
    parser.add_argument(
        "--substrate-vector-axis",
        choices=("tangent", "normal"),
        default="tangent",
    )
    parser.add_argument("--substrate-transport-hint", type=float, default=0.0)
    parser.add_argument("--splat-advection", type=float, default=0.0)
    parser.add_argument(
        "--transport-consistency-power",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/out/flow_volume_cells.png",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=ROOT / "experiments/out/flow_volume_cells.json",
    )
    args = parser.parse_args()

    image, source = _load_image(args.image, args.gallery)
    rgb = _fit_rgb(image, args.side)
    height, width = rgb.shape[:2]
    started = time.perf_counter()
    volume = build_flow_volume(rgb, passes=args.passes)
    flow_ms = (time.perf_counter() - started) * 1000.0
    population, emission = emit_population(
        volume,
        height,
        width,
        args.base_cells,
        args.detail_cells,
        args.first_detail_stage,
        args.base_overlap,
        args.max_major,
        args.curvature_limit,
        args.specificity_power,
        args.specificity_mix,
        args.emission_mode,
        args.substrate_hint,
        args.substrate_shape_hint,
        args.substrate_vector_mode,
        args.substrate_vector_axis,
        args.substrate_transport_hint,
        args.splat_advection,
        args.transport_consistency_power,
    )
    objective = SingleStageDecompositionObjective(rgb)
    target_lab = srgb_to_lab(rgb)

    variants = {
        "flow ellipses": population,
        "same sites, circular": population.circularized(),
        "uniform R2 circles": r2_control(
            len(population.centers),
            height,
            width,
            args.base_overlap,
        ),
    }
    records = []
    diagnostics = {}
    solve_started = time.perf_counter()
    for name, variant in variants.items():
        record, _, diagnostic = fit_population(
            variant, target_lab, objective)
        records.append((name, record))
        diagnostics[name] = diagnostic
    layered, _, layered_diagnostic = fit_layered_population(
        population, rgb, objective)
    records.append(("round cartoon + flow texture", layered))
    diagnostics["round cartoon + flow texture"] = layered_diagnostic
    solve_ms = (time.perf_counter() - solve_started) * 1000.0

    save_panel(rgb, volume, population, records, args.output)
    report = {
        "source": source,
        "shape": list(rgb.shape),
        "passes": args.passes,
        "first_detail_stage": args.first_detail_stage,
        "base_cells": args.base_cells,
        "requested_detail_cells": args.detail_cells,
        "total_cells": int(len(population.centers)),
        "emission": emission,
        "scores": {
            name: _serializable(record) for name, record in records
        },
        "diagnostics": diagnostics,
        "flow_ms": flow_ms,
        "solve_ms": solve_ms,
        "output": str(args.output.resolve()),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
