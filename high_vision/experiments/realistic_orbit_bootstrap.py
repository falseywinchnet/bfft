#!/usr/bin/env python3
"""Orbit extraction before averaging on the physical moonlight benchmark."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import shift as image_shift
from scipy.sparse import coo_matrix, eye
from scipy.sparse.linalg import spsolve
from skimage import transform
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.registration import phase_cross_correlation

from budgeted_fullres_demo import block_sum
from poisson_orbit_demo import (
    best_cyclic_alignment,
    hermitian_spectrum,
    select_steps,
    signed_frequency,
    solve_phase,
)
from realistic_moonlight_bench import (
    MoonlightCamera,
    capture_stream,
    hdr_scene,
    sensor_maps,
    tone_map,
)


@dataclass(frozen=True)
class OrbitBench:
    grid: int = 64
    frames: int = 4096
    projection_steps: int = 32
    optimizer_iterations: int = 600
    batch: int = 128
    evaluate_registration: bool = False
    phase_solver: str = "factorized"
    evidence_radius: int = 16


def calibrated_thumbnail(
    frame: np.ndarray,
    maps: dict[str, np.ndarray],
    camera: MoonlightCamera,
    grid: int,
) -> np.ndarray:
    """Apply ordinary dark/flat calibration, then preserve mean radiance."""
    expected_fixed = maps["dsnu"] + maps["hot_rate"]
    corrected = (frame - expected_fixed) / np.maximum(maps["prnu"], 1e-4)
    area = (camera.size // grid) ** 2
    return block_sum(corrected, grid).astype(np.float64) / area


def calibrated_sublattice_thumbnails(
    frame: np.ndarray,
    maps: dict[str, np.ndarray],
    camera: MoonlightCamera,
    grid: int,
) -> np.ndarray:
    """Form four disjoint 2×2 sensor sublattices at one thumbnail scale."""
    block_side = camera.size // grid
    if block_side % 2:
        raise ValueError("thumbnail block side must be divisible by two")
    expected_fixed = maps["dsnu"] + maps["hot_rate"]
    corrected = (frame - expected_fixed) / np.maximum(maps["prnu"], 1e-4)
    blocks = corrected.reshape(
        grid, block_side, grid, block_side)
    gatherers = []
    for offset_y, offset_x in ((0, 0), (0, 1), (1, 0), (1, 1)):
        gatherers.append(np.mean(
            blocks[:, offset_y::2, :, offset_x::2],
            axis=(1, 3),
        ))
    return np.asarray(gatherers, dtype=np.float64)


def score(estimate: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    estimate_tone = tone_map(np.clip(estimate, 0.0, 1.0))
    truth_tone = tone_map(truth)
    return {
        "linear_psnr_db": float(peak_signal_noise_ratio(
            truth, np.clip(estimate, 0.0, 1.0), data_range=1.0)),
        "log_psnr_db": float(peak_signal_noise_ratio(
            truth_tone, estimate_tone, data_range=1.0)),
        "log_ssim": float(structural_similarity(
            truth_tone, estimate_tone, data_range=1.0)),
    }


def split_half_radial_support(
    bispectrum_half_sum: np.ndarray,
    grid: int,
    open_sigma: float = 3.0,
    full_sigma: float = 6.0,
    measured_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert independent biphase agreement into an oracle-free ring mask."""
    if bispectrum_half_sum.ndim != 4 or bispectrum_half_sum.shape[0] != 2:
        raise ValueError("expected two [step, y, x] bispectrum witnesses")
    if bispectrum_half_sum.shape[-2:] != (grid, grid):
        raise ValueError("bispectrum grid does not match requested grid")
    if not full_sigma > open_sigma:
        raise ValueError("full_sigma must exceed open_sigma")

    half_phase_delta = np.angle(
        bispectrum_half_sum[0]
        * np.conj(bispectrum_half_sum[1]))
    phase_agreement_samples = np.cos(half_phase_delta)
    frequencies = np.fft.fftfreq(grid) * grid
    support_rings = np.rint(np.hypot(
        frequencies[:, None], frequencies[None, :])).astype(np.int32)
    if measured_mask is None:
        measured_mask = np.ones((grid, grid), dtype=bool)
    elif measured_mask.shape != (grid, grid):
        raise ValueError("measured mask does not match requested grid")
    ring_count = int(np.max(support_rings)) + 1
    ring_agreement = np.zeros(ring_count)
    ring_significance = np.zeros(ring_count)
    ring_gain = np.zeros(ring_count)
    for radius_index in range(ring_count):
        selected = (
            (support_rings == radius_index)
            & measured_mask
        )
        samples = phase_agreement_samples[:, selected].ravel()
        if not len(samples):
            continue
        ring_agreement[radius_index] = float(np.mean(samples))
        # Under the random-phase null, cos(delta phase) has variance 1/2.
        ring_significance[radius_index] = float(
            ring_agreement[radius_index] * np.sqrt(2.0 * len(samples)))
        ring_gain[radius_index] = np.clip(
            (ring_significance[radius_index] - open_sigma)
            / (full_sigma - open_sigma),
            0.0,
            1.0,
        )
    ring_gain[0] = 1.0
    return support_rings, ring_agreement, ring_significance, ring_gain


def poisson_factorial_bispectrum(
    raw_bispectrum: np.ndarray,
    power_without_gaussian: np.ndarray,
    photon_dc: float,
    steps: list[tuple[int, int]],
    block_area: int,
) -> np.ndarray:
    """Remove same-photon collisions from a third-order Fourier moment.

    For a block average Z=C/A, the raw same-frame bispectrum contains the
    desired three-distinct-photon term, three pair collisions, and one triple
    collision. Gaussian power is removed before this correction because read
    noise is not a photon event.
    """
    if raw_bispectrum.shape[0] != len(steps):
        raise ValueError("bispectrum and step counts differ")
    if block_area <= 0:
        raise ValueError("block_area must be positive")
    corrected = np.empty_like(raw_bispectrum)
    for step_index, (sy, sx) in enumerate(steps):
        shifted_power = np.roll(
            np.roll(power_without_gaussian, -sy, axis=0),
            -sx,
            axis=1,
        )
        collisions = (
            power_without_gaussian
            + power_without_gaussian[sy, sx]
            + shifted_power
        )
        corrected[step_index] = (
            raw_bispectrum[step_index]
            - collisions / block_area
            + 2.0 * photon_dc / block_area ** 2
        )
    return corrected


def trimmed_complex_location(
    values: np.ndarray,
    trim: int,
) -> np.ndarray:
    """Fuse independent complex evidence lanes with fixed bounded influence."""
    if values.ndim < 1:
        raise ValueError("values must have a lane axis")
    lanes = values.shape[0]
    if trim < 0 or 2 * trim >= lanes:
        raise ValueError("trim must retain at least one evidence lane")
    if trim == 0:
        return np.mean(values, axis=0)
    real = np.sort(values.real, axis=0)[trim:lanes - trim]
    imag = np.sort(values.imag, axis=0)[trim:lanes - trim]
    return np.mean(real, axis=0) + 1j * np.mean(imag, axis=0)


def distinct_gatherer_product(
    gatherer_spectra: np.ndarray,
    step: tuple[int, int],
    sample_indices: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Average every ordered distinct three-gatherer product in closed form."""
    gatherers = gatherer_spectra.shape[1]
    if gatherers < 3:
        raise ValueError("at least three gatherers are required")
    sy, sx = step
    if sample_indices is None:
        sample_y, sample_x = np.indices(
            gatherer_spectra.shape[-2:])
    else:
        sample_y, sample_x = sample_indices
    gatherer_x = gatherer_spectra[:, :, sample_y, sample_x]
    gatherer_y = gatherer_spectra[:, :, sy, sx].reshape(
        gatherer_spectra.shape[:2]
        + (1,) * (gatherer_x.ndim - 2)
    )
    gatherer_z = np.conj(gatherer_spectra[
        :,
        :,
        (sample_y + sy) % gatherer_spectra.shape[-2],
        (sample_x + sx) % gatherer_spectra.shape[-1],
    ])
    sum_x = np.sum(gatherer_x, axis=1)
    sum_y = np.sum(gatherer_y, axis=1)
    sum_z = np.sum(gatherer_z, axis=1)
    numerator = (
        sum_x * sum_y * sum_z
        - np.sum(gatherer_x * gatherer_y, axis=1) * sum_z
        - np.sum(gatherer_x * gatherer_z, axis=1) * sum_y
        - sum_x * np.sum(gatherer_y * gatherer_z, axis=1)
        + 2.0 * np.sum(
            gatherer_x * gatherer_y * gatherer_z,
            axis=1,
        )
    )
    return numerator / (
        gatherers * (gatherers - 1) * (gatherers - 2))


def solve_phase_marching(
    bispectrum: np.ndarray,
    coherence: np.ndarray,
    steps: list[tuple[int, int]],
    support_gain: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Recover supported phase in one outward, circle-valued fusion pass.

    For q = k + s, the biphase supplies

        phi(q) = phi(k) + phi(s) - arg B_s(k).

    Frequencies are visited from low to high radius. Every candidate whose two
    parents are already known votes as a unit phasor, so redundant projection
    paths fuse without an unwrap, a learning prior, or an optimizer.
    """
    step_count, height, width = bispectrum.shape
    if height != width:
        raise ValueError("marching phase solve currently requires a square grid")
    if step_count != len(steps):
        raise ValueError("bispectrum and step counts differ")
    if support_gain.shape != (height, width):
        raise ValueError("support gain must match the Fourier grid")

    phase = np.zeros((height, width), dtype=np.float64)
    known = np.zeros((height, width), dtype=bool)
    known[0, 0] = True
    # Translation is the two-dimensional linear phase gauge. Pinning the two
    # unit frequencies chooses one orbit representative and supplies no shift.
    for fy, fx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        known[fy % height, fx % width] = True

    frequencies = []
    nyquist = height // 2
    for fy in range(-nyquist + 1, nyquist):
        for fx in range(0, nyquist):
            if fx == 0 and fy < 0:
                continue
            if fy == 0 and fx == 0:
                continue
            iy, ix = fy % height, fx % width
            if support_gain[iy, ix] <= 0.0:
                continue
            frequencies.append((fy * fy + fx * fx, abs(fy) + fx, fy, fx))
    frequencies.sort()

    signed_steps = [
        (signed_frequency(sy, height), signed_frequency(sx, width))
        for sy, sx in steps
    ]
    fusion_count = 0
    maximum_votes = 0
    unsupported_dependencies = 0
    for _, _, qy, qx in frequencies:
        q_index = (qy % height, qx % width)
        if known[q_index]:
            continue
        votes = []
        weights = []
        for step_index, (sy, sx) in enumerate(signed_steps):
            step_grid = (sy % height, sx % width)
            ky, kx = qy - sy, qx - sx
            if not (-nyquist < ky < nyquist and -nyquist < kx < nyquist):
                continue
            k_grid = (ky % height, kx % width)
            if not known[step_grid] or not known[k_grid]:
                continue
            # k=0 or q=s yields the identity B_s(0)=|F(0)F(s)| and carries
            # no phase information.
            if (ky == 0 and kx == 0) or (qy == sy and qx == sx):
                continue
            weight = float(np.clip(
                coherence[step_index, k_grid[0], k_grid[1]], 0.0, 1.0
            )) ** 3
            weight *= float(support_gain[k_grid] * support_gain[step_grid])
            if weight <= 0.0:
                continue
            votes.append(
                phase[k_grid]
                + phase[step_grid]
                - np.angle(bispectrum[step_index, k_grid[0], k_grid[1]])
            )
            weights.append(weight)
        if not votes:
            unsupported_dependencies += 1
            continue
        resultant = np.sum(
            np.asarray(weights) * np.exp(1j * np.asarray(votes)))
        phase[q_index] = float(np.angle(resultant))
        opposite = ((-qy) % height, (-qx) % width)
        phase[opposite] = -phase[q_index]
        known[q_index] = True
        known[opposite] = True
        fusion_count += len(votes)
        maximum_votes = max(maximum_votes, len(votes))

    residual_sum = 0.0
    residual_weight = 0.0
    constraints_used = 0
    for step_index, (sy, sx) in enumerate(steps):
        for ky in range(height):
            for kx in range(width):
                total = ((ky + sy) % height, (kx + sx) % width)
                if not (
                    known[ky, kx]
                    and known[sy, sx]
                    and known[total]
                ):
                    continue
                weight = float(np.clip(
                    coherence[step_index, ky, kx], 0.0, 1.0
                )) ** 3
                if weight <= 0.0:
                    continue
                residual = (
                    phase[ky, kx]
                    + phase[sy, sx]
                    - phase[total]
                    - np.angle(bispectrum[step_index, ky, kx])
                )
                residual_sum += weight * (1.0 - np.cos(residual))
                residual_weight += weight
                constraints_used += 1

    return phase, {
        "phase_solver": "bounded_radial_fusor",
        "optimizer_success": True,
        "optimizer_message": "not used",
        "optimizer_iterations": 0,
        "phase_objective": float(
            residual_sum / max(residual_weight, 1e-12)),
        "constraints_used": constraints_used,
        "phase_fusions": fusion_count,
        "maximum_votes_per_frequency": maximum_votes,
        "unsupported_phase_frequencies": unsupported_dependencies,
        "supported_phase_frequencies": int(np.count_nonzero(known)),
    }


def solve_phase_factorized(
    bispectrum: np.ndarray,
    coherence: np.ndarray,
    steps: list[tuple[int, int]],
    support_gain: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Choose wraps by marching, then reconcile them in one sparse solve."""
    seed, seed_info = solve_phase_marching(
        bispectrum, coherence, steps, support_gain)
    _, height, width = bispectrum.shape

    def canonical(index: tuple[int, int]) -> tuple[tuple[int, int], float]:
        fy = signed_frequency(index[0], height)
        fx = signed_frequency(index[1], width)
        if fx < 0 or (fx == 0 and fy < 0):
            return ((-fy, -fx), -1.0)
        return ((fy, fx), 1.0)

    anchors = {(0, 0), (0, 1), (1, 0)}
    canonical_frequencies = set()
    for iy, ix in zip(*np.nonzero(support_gain > 0.0)):
        frequency, _ = canonical((int(iy), int(ix)))
        if frequency not in anchors:
            canonical_frequencies.add(frequency)
    ordered = sorted(
        canonical_frequencies,
        key=lambda value: (
            value[0] ** 2 + value[1] ** 2,
            abs(value[0]) + abs(value[1]),
            value,
        ),
    )
    variable = {frequency: index for index, frequency in enumerate(ordered)}
    if not ordered:
        return seed, {
            "phase_solver": "march_then_sparse_factorization",
            "optimizer_success": True,
            "optimizer_message": "no free supported phase variables",
            "optimizer_iterations": 0,
            "linear_factorizations": 0,
            "phase_objective": seed_info["phase_objective"],
            "constraints_used": seed_info["constraints_used"],
            "phase_variables": 0,
            "phase_fusions": seed_info["phase_fusions"],
            "maximum_votes_per_frequency": seed_info[
                "maximum_votes_per_frequency"],
            "unsupported_phase_frequencies": seed_info[
                "unsupported_phase_frequencies"],
            "supported_phase_frequencies": seed_info[
                "supported_phase_frequencies"],
        }

    row_indices = []
    column_indices = []
    values = []
    targets = []
    row_weights = []
    row = 0
    for step_index, (sy, sx) in enumerate(steps):
        if support_gain[sy, sx] <= 0.0:
            continue
        for ky, kx in zip(*np.nonzero(support_gain > 0.0)):
            total = ((int(ky) + sy) % height, (int(kx) + sx) % width)
            if support_gain[total] <= 0.0:
                continue
            coefficients: dict[int, float] = {}
            for grid_index, coefficient in (
                ((int(ky), int(kx)), 1.0),
                ((sy, sx), 1.0),
                (total, -1.0),
            ):
                frequency, sign = canonical(grid_index)
                if frequency in anchors:
                    continue
                column = variable[frequency]
                coefficients[column] = (
                    coefficients.get(column, 0.0) + coefficient * sign)
            coefficients = {
                column: coefficient
                for column, coefficient in coefficients.items()
                if abs(coefficient) > 1e-12
            }
            if not coefficients:
                continue
            weight = float(np.clip(
                coherence[step_index, int(ky), int(kx)], 0.0, 1.0
            )) ** 3
            weight *= float(
                support_gain[int(ky), int(kx)]
                * support_gain[sy, sx]
                * support_gain[total]
            )
            if weight <= 0.0:
                continue
            current_left = (
                seed[int(ky), int(kx)]
                + seed[sy, sx]
                - seed[total]
            )
            wrapped_residual = np.angle(np.exp(1j * (
                current_left
                - np.angle(bispectrum[step_index, int(ky), int(kx)])
            )))
            target = current_left - wrapped_residual
            scale = np.sqrt(weight)
            for column, coefficient in coefficients.items():
                row_indices.append(row)
                column_indices.append(column)
                values.append(scale * coefficient)
            targets.append(scale * target)
            row_weights.append(weight)
            row += 1

    matrix = coo_matrix(
        (values, (row_indices, column_indices)),
        shape=(row, len(ordered)),
    ).tocsr()
    target_vector = np.asarray(targets)
    normal = matrix.T @ matrix
    ridge = max(float(np.mean(normal.diagonal())), 1.0) * 1e-10
    solution = spsolve(
        normal + ridge * eye(len(ordered), format="csr"),
        matrix.T @ target_vector,
    )

    phase = np.zeros((height, width), dtype=np.float64)
    for frequency, value in zip(ordered, solution):
        fy, fx = frequency
        phase[fy % height, fx % width] = value
        phase[(-fy) % height, (-fx) % width] = -value

    predicted = matrix @ solution
    objective = float(np.sum(
        np.asarray(row_weights)
        * (1.0 - np.cos(
            (predicted - target_vector)
            / np.sqrt(np.asarray(row_weights))
        ))
    ) / max(float(np.sum(row_weights)), 1e-12))
    return phase, {
        "phase_solver": "march_then_sparse_factorization",
        "optimizer_success": True,
        "optimizer_message": "not used",
        "optimizer_iterations": 0,
        "linear_factorizations": 1,
        "phase_objective": objective,
        "constraints_used": row,
        "phase_variables": len(ordered),
        "phase_fusions": seed_info["phase_fusions"],
        "maximum_votes_per_frequency": seed_info[
            "maximum_votes_per_frequency"],
        "unsupported_phase_frequencies": seed_info[
            "unsupported_phase_frequencies"],
        "supported_phase_frequencies": seed_info[
            "supported_phase_frequencies"],
    }


def summarize_registration(
    group_estimates: list[np.ndarray],
    group_truths: list[list[tuple[int, int]]],
) -> dict[str, float | list[int]]:
    estimates = []
    truths = []
    for estimate, truth_group in zip(group_estimates, group_truths):
        estimates.extend([estimate] * len(truth_group))
        truths.extend(truth_group)
    difference = np.asarray(estimates) - np.asarray(truths)
    gauge = np.rint(np.median(difference, axis=0)).astype(np.int32)
    residual = difference - gauge
    error = np.hypot(residual[:, 0], residual[:, 1])
    return {
        "mean": float(np.mean(error)),
        "median": float(np.median(error)),
        "p90": float(np.quantile(error, 0.9)),
        "global_gauge_yx": [int(gauge[0]), int(gauge[1])],
    }


def registration_cascade(
    mean_reference: np.ndarray,
    orbit_reference: np.ndarray,
    oracle_reference: np.ndarray,
    scene: np.ndarray,
    maps: dict[str, np.ndarray],
    camera: MoonlightCamera,
    orbit_config: OrbitBench,
) -> tuple[dict[str, dict[str, float | list[int]]], np.ndarray]:
    """Use the orbit as a coarse gauge, then build one finer attractor."""
    high_grid = camera.thumbnail
    if camera.size % high_grid:
        raise ValueError("camera size must be divisible by high grid")
    group = []
    low_piles = []
    high_piles = []
    group_truths = []

    def consume(items):
        if not items:
            return
        low_piles.append(np.mean([item[0] for item in items], axis=0))
        high_piles.append(np.mean([item[1] for item in items], axis=0))
        group_truths.append([item[2] for item in items])

    for _, frame, truth_shift in capture_stream(scene, camera, maps):
        group.append((
            calibrated_thumbnail(
                frame, maps, camera, orbit_config.grid),
            calibrated_thumbnail(frame, maps, camera, high_grid),
            truth_shift,
        ))
        if len(group) == camera.registration_group:
            consume(group)
            group.clear()
    consume(group)

    def estimate(
        reference: np.ndarray,
        pile: np.ndarray,
        grid: int,
        radius_full: float,
    ) -> np.ndarray:
        scale = camera.size / grid
        shift, _, _ = phase_cross_correlation(
            reference,
            pile,
            upsample_factor=grid,
            normalization="phase",
        )
        shift = np.clip(
            shift,
            -radius_full / scale - 1.0,
            radius_full / scale + 1.0,
        )
        return shift * scale

    high_mean = np.mean(high_piles, axis=0)
    mean_estimates = [
        estimate(high_mean, pile, high_grid, camera.shift_radius)
        for pile in high_piles
    ]
    oracle_estimates = [
        estimate(
            oracle_reference,
            pile,
            high_grid,
            camera.shift_radius,
        )
        for pile in high_piles
    ]
    coarse_estimates = [
        estimate(
            orbit_reference,
            pile,
            orbit_config.grid,
            camera.shift_radius,
        )
        for pile in low_piles
    ]

    high_scale = camera.size / high_grid
    # The orbit has an arbitrary global translation but a definite phase
    # shape. Upsampling does not invent bandwidth; it only samples its smooth
    # correlation surface on the high-grid displacement lattice.
    high_orbit = transform.resize(
        orbit_reference,
        (high_grid, high_grid),
        order=3,
        anti_aliasing=False,
        preserve_range=True,
    )
    state_radius = int(np.ceil(camera.shift_radius / high_scale))
    state_axis = np.arange(-state_radius, state_radius + 1)
    states = np.asarray([
        (dy, dx) for dy in state_axis for dx in state_axis],
        dtype=np.int32,
    )
    reference_spectrum = np.fft.fft2(high_orbit)
    emissions = []
    for pile in high_piles:
        product = reference_spectrum * np.conj(np.fft.fft2(pile))
        product /= np.maximum(np.abs(product), 1e-10)
        correlation = np.fft.ifft2(product).real
        score_values = np.asarray([
            correlation[dy % high_grid, dx % high_grid]
            for dy, dx in states
        ])
        score_values = (
            score_values - np.mean(score_values)
        ) / max(float(np.std(score_values)), 1e-8)
        emissions.append(score_values)
    emissions = np.asarray(emissions)

    # Brownian camera motion gives a Gaussian transition prior. Its variance
    # is derived from the configured uniform {-step,...,+step} increments,
    # accumulated over one registration group and expressed on the state grid.
    increment_variance = (
        camera.motion_step * (camera.motion_step + 1) / 3.0)
    transition_sigma = max(
        np.sqrt(camera.registration_group * increment_variance) / high_scale,
        0.5,
    )
    delta = states[:, None, :] - states[None, :, :]
    transition = -0.5 * np.sum(delta ** 2, axis=2) / transition_sigma ** 2
    value = emissions[0].copy()
    backpointers = []
    for time_index in range(1, len(emissions)):
        candidates = value[:, None] + transition
        previous = np.argmax(candidates, axis=0)
        value = emissions[time_index] + candidates[previous, np.arange(
            len(states))]
        backpointers.append(previous)
    state_index = int(np.argmax(value))
    path = [state_index]
    for previous in reversed(backpointers):
        state_index = int(previous[state_index])
        path.append(state_index)
    path.reverse()
    viterbi_estimates = [
        states[index].astype(np.float64) * high_scale for index in path
    ]

    reference_sum = np.zeros_like(high_mean)
    reference_support = np.zeros_like(high_mean)
    for pile, coarse in zip(high_piles, coarse_estimates):
        shift_high = coarse / high_scale
        reference_sum += image_shift(
            pile, shift_high, order=1, mode="constant", cval=0.0)
        reference_support += image_shift(
            np.ones_like(pile),
            shift_high,
            order=0,
            mode="constant",
            cval=0.0,
        )
    cascade_reference = reference_sum / np.maximum(
        reference_support, 1e-8)

    cascade_estimates = []
    for pile, coarse in zip(high_piles, coarse_estimates):
        shift_high = coarse / high_scale
        coarse_aligned = image_shift(
            pile, shift_high, order=1, mode="constant", cval=0.0)
        residual = estimate(
            cascade_reference,
            coarse_aligned,
            high_grid,
            radius_full=12.0,
        )
        cascade_estimates.append(coarse + residual)

    return {
        "oracle_group_constant_floor": summarize_registration(
            [
                np.median(np.asarray(truth_group), axis=0)
                for truth_group in group_truths
            ],
            group_truths,
        ),
        "original_oracle_attractor": summarize_registration(
            oracle_estimates, group_truths),
        "false_mean_attractor": summarize_registration(
            mean_estimates, group_truths),
        "orbit_direct": summarize_registration(
            coarse_estimates, group_truths),
        "orbit_viterbi": summarize_registration(
            viterbi_estimates, group_truths),
        "orbit_then_local": summarize_registration(
            cascade_estimates, group_truths),
    }, cascade_reference


def run(
    camera: MoonlightCamera,
    orbit_config: OrbitBench,
) -> tuple[dict, dict[str, np.ndarray]]:
    if camera.size % orbit_config.grid:
        raise ValueError("camera size must be divisible by orbit grid")
    scene = hdr_scene(camera)
    maps = sensor_maps(camera)
    margin = camera.canvas_margin
    truth_full = scene[
        margin:margin + camera.size,
        margin:margin + camera.size,
    ]
    area = (camera.size // orbit_config.grid) ** 2
    truth = block_sum(
        truth_full, orbit_config.grid).astype(np.float64) / area
    high_area = (camera.size // camera.thumbnail) ** 2
    truth_high = block_sum(
        truth_full, camera.thumbnail).astype(np.float64) / high_area
    steps = select_steps(
        orbit_config.grid,
        orbit_config.grid,
        orbit_config.projection_steps,
        full_circle_support=False,
    )
    evidence_frequency = (
        np.fft.fftfreq(orbit_config.grid) * orbit_config.grid)
    evidence_mask = np.hypot(
        evidence_frequency[:, None],
        evidence_frequency[None, :],
    ) <= orbit_config.evidence_radius
    evidence_y, evidence_x = np.nonzero(evidence_mask)
    power_sum = np.zeros(
        (orbit_config.grid, orbit_config.grid), dtype=np.float64)
    observed_sum = np.zeros_like(power_sum)
    bispectrum_sum = np.zeros(
        (len(steps), orbit_config.grid, orbit_config.grid),
        dtype=np.complex128,
    )
    bispectrum_abs_sum = np.zeros_like(bispectrum_sum.real)
    bispectrum_half_sum = np.zeros(
        (2, len(steps), orbit_config.grid, orbit_config.grid),
        dtype=np.complex128,
    )
    cross_bispectrum_sum = np.zeros_like(bispectrum_sum)
    cross_bispectrum_abs_sum = np.zeros_like(bispectrum_abs_sum)
    cross_bispectrum_half_sum = np.zeros_like(bispectrum_half_sum)
    power_half_sum = np.zeros(
        (2, orbit_config.grid, orbit_config.grid),
        dtype=np.float64,
    )
    observed_half_sum = np.zeros(
        (2, orbit_config.grid, orbit_config.grid),
        dtype=np.float64,
    )
    frames_done = 0
    batch_frames = []
    batch_gatherers = []
    batch_parities = []
    started = time.perf_counter()

    def consume(
        frames: list[np.ndarray],
        gatherer_frames: list[np.ndarray],
        parities: list[int],
    ) -> None:
        nonlocal power_sum, observed_sum, bispectrum_sum
        if not frames:
            return
        values = np.asarray(frames, dtype=np.float64)
        gatherer_values = np.asarray(gatherer_frames, dtype=np.float64)
        parity_array = np.asarray(parities)
        observed_sum += np.sum(values, axis=0)
        gatherer_spectra = np.fft.fft2(
            gatherer_values, axes=(-2, -1))
        spectra = np.mean(gatherer_spectra, axis=1)
        spectral_power = np.abs(spectra) ** 2
        power_sum += np.sum(spectral_power, axis=0)
        for parity in (0, 1):
            selected = parity_array == parity
            observed_half_sum[parity] += np.sum(
                values[selected], axis=0)
            power_half_sum[parity] += np.sum(
                spectral_power[selected], axis=0)
        for step_index, (sy, sx) in enumerate(steps):
            total_y = (evidence_y + sy) % orbit_config.grid
            total_x = (evidence_x + sx) % orbit_config.grid
            product = (
                spectra[:, evidence_y, evidence_x]
                * spectra[:, sy, sx, None]
                * np.conj(spectra[:, total_y, total_x])
            )
            bispectrum_sum[
                step_index, evidence_y, evidence_x
            ] += np.sum(product, axis=0)
            bispectrum_abs_sum[
                step_index, evidence_y, evidence_x
            ] += np.sum(np.abs(product), axis=0)

            distinct_product = distinct_gatherer_product(
                gatherer_spectra,
                (sy, sx),
                (evidence_y, evidence_x),
            )
            cross_bispectrum_sum[
                step_index, evidence_y, evidence_x
            ] += np.sum(distinct_product, axis=0)
            cross_bispectrum_abs_sum[
                step_index, evidence_y, evidence_x
            ] += np.sum(np.abs(distinct_product), axis=0)
            for parity in (0, 1):
                selected = parity_array == parity
                bispectrum_half_sum[
                    parity, step_index, evidence_y, evidence_x
                ] += np.sum(product[selected], axis=0)
                cross_bispectrum_half_sum[
                    parity, step_index, evidence_y, evidence_x
                ] += np.sum(distinct_product[selected], axis=0)

    for index, frame, _ in capture_stream(scene, camera, maps):
        gatherers = calibrated_sublattice_thumbnails(
            frame, maps, camera, orbit_config.grid)
        batch_gatherers.append(gatherers)
        batch_frames.append(np.mean(gatherers, axis=0))
        batch_parities.append(index % 2)
        if len(batch_frames) == orbit_config.batch:
            consume(
                batch_frames,
                batch_gatherers,
                batch_parities,
            )
            frames_done += len(batch_frames)
            batch_frames.clear()
            batch_gatherers.clear()
            batch_parities.clear()
    consume(
        batch_frames,
        batch_gatherers,
        batch_parities,
    )
    frames_done += len(batch_frames)

    # Non-normalized FFT white-noise power equals the sum of per-pixel
    # variances. Spatial averaging divides independent sensor variance by the
    # block area. Row/column correlations are intentionally not "oracle
    # subtracted"; they remain a stressor on the axial frequencies.
    mean_signal = float(camera.photons_at_white * np.mean(truth))
    mean_hot = float(np.mean(maps["hot_rate"]))
    independent_sensor_variance = (
        camera.read_noise_electrons ** 2
        + camera.electrons_per_dn ** 2 / 12.0
        + mean_signal
        + camera.dark_electrons
        + mean_hot
    )
    white_noise_floor = (
        orbit_config.grid ** 2 * independent_sensor_variance / area)
    noise_floor_map = np.full(
        (orbit_config.grid, orbit_config.grid),
        white_noise_floor,
        dtype=np.float64,
    )
    block_side = camera.size // orbit_config.grid
    # A row bias is averaged over block_side sensor rows but remains perfectly
    # correlated across every thumbnail column. Its Fourier power therefore
    # lies on kx=0 and scales as grid^3 rather than as grid^2.
    noise_floor_map[:, 0] += (
        orbit_config.grid ** 3
        * camera.row_noise_electrons ** 2
        / block_side
    )
    noise_floor_map[0, :] += (
        orbit_config.grid ** 3
        * camera.column_noise_electrons ** 2
        / block_side
    )
    power_mean = power_sum / frames_done
    power = power_mean - noise_floor_map
    magnitude = np.sqrt(np.maximum(power, 0.0)) / max(
        camera.photons_at_white, 1e-12)
    magnitude[0, 0] = max(
        float(np.sum(observed_sum))
        / (frames_done * camera.photons_at_white),
        0.0,
    )
    base_magnitude = magnitude.copy()
    raw_bispectrum = bispectrum_sum / frames_done
    raw_coherence = np.abs(bispectrum_sum) / np.maximum(
        bispectrum_abs_sum, 1e-12)
    gaussian_noise_floor = np.full(
        (orbit_config.grid, orbit_config.grid),
        orbit_config.grid ** 2
        * (
            camera.read_noise_electrons ** 2
            + camera.electrons_per_dn ** 2 / 12.0
        )
        / area,
        dtype=np.float64,
    )
    gaussian_noise_floor[:, 0] += (
        orbit_config.grid ** 3
        * camera.row_noise_electrons ** 2
        / block_side
    )
    gaussian_noise_floor[0, :] += (
        orbit_config.grid ** 3
        * camera.column_noise_electrons ** 2
        / block_side
    )
    photon_background = camera.dark_electrons + mean_hot
    photon_dc = (
        float(np.sum(observed_sum)) / frames_done
        + orbit_config.grid ** 2 * photon_background
    )
    factorial_bispectrum = poisson_factorial_bispectrum(
        raw_bispectrum,
        power_mean - gaussian_noise_floor,
        photon_dc,
        steps,
        area,
    )
    corrected_half_sum = np.empty_like(bispectrum_half_sum)
    half_counts = np.bincount(
        np.arange(frames_done) % 2, minlength=2)
    for parity in (0, 1):
        count = int(half_counts[parity])
        half_dc = (
            float(np.sum(observed_half_sum[parity])) / count
            + orbit_config.grid ** 2 * photon_background
        )
        corrected_half_sum[parity] = count * poisson_factorial_bispectrum(
            bispectrum_half_sum[parity] / count,
            power_half_sum[parity] / count - gaussian_noise_floor,
            half_dc,
            steps,
            area,
        )
    bispectrum = cross_bispectrum_sum / frames_done
    coherence = (
        np.abs(cross_bispectrum_sum)
        / np.maximum(cross_bispectrum_abs_sum, 1e-12)
    )
    factorial_coherence = (
        np.abs(factorial_bispectrum) * frames_done
        / np.maximum(bispectrum_abs_sum, 1e-12)
    )
    # Degenerate Gaussian contractions and correlated row/column noise live
    # principally on the Fourier axes. Do not let those few bins define phase.
    coherence[:, 0, :] *= 0.05
    coherence[:, :, 0] *= 0.05
    raw_coherence[:, 0, :] *= 0.05
    raw_coherence[:, :, 0] *= 0.05
    factorial_coherence[:, 0, :] *= 0.05
    factorial_coherence[:, :, 0] *= 0.05
    # Two independent temporal witnesses decide which radial phase support is
    # real. Three sigma opens a ring and six sigma fully trusts it, without
    # consulting the clean Cameraman image.
    (
        support_rings,
        ring_agreement,
        ring_significance,
        ring_gain,
    ) = split_half_radial_support(
        cross_bispectrum_half_sum,
        orbit_config.grid,
        measured_mask=evidence_mask,
    )
    support_gain = ring_gain[support_rings]
    phase_started = time.perf_counter()
    if orbit_config.phase_solver == "factorized":
        phase, phase_info = solve_phase_factorized(
            bispectrum,
            coherence,
            steps,
            support_gain,
        )
    elif orbit_config.phase_solver == "marching":
        phase, phase_info = solve_phase_marching(
            bispectrum,
            coherence,
            steps,
            support_gain,
        )
    elif orbit_config.phase_solver == "optimizer":
        phase, phase_info = solve_phase(
            bispectrum,
            coherence,
            steps,
            orbit_config.optimizer_iterations,
        )
        phase_info["phase_solver"] = "nonlinear_optimizer"
    else:
        raise ValueError(
            f"unknown phase solver: {orbit_config.phase_solver}")
    phase_info["phase_solve_seconds"] = (
        time.perf_counter() - phase_started)
    magnitude = base_magnitude * support_gain
    recovered = np.fft.ifft2(
        hermitian_spectrum(magnitude, phase)).real
    recovered, recovered_shift = best_cyclic_alignment(recovered, truth)

    (
        factorial_support_rings,
        factorial_ring_agreement,
        factorial_ring_significance,
        factorial_ring_gain,
    ) = split_half_radial_support(
        corrected_half_sum,
        orbit_config.grid,
        measured_mask=evidence_mask,
    )
    factorial_support_gain = factorial_ring_gain[
        factorial_support_rings]
    factorial_phase, factorial_phase_info = solve_phase_factorized(
        factorial_bispectrum,
        factorial_coherence,
        steps,
        factorial_support_gain,
    )
    factorial_recovered = np.fft.ifft2(hermitian_spectrum(
        base_magnitude * factorial_support_gain,
        factorial_phase,
    )).real
    factorial_recovered, factorial_recovered_shift = best_cyclic_alignment(
        factorial_recovered, truth)

    (
        raw_support_rings,
        raw_ring_agreement,
        raw_ring_significance,
        raw_ring_gain,
    ) = split_half_radial_support(
        bispectrum_half_sum,
        orbit_config.grid,
        measured_mask=evidence_mask,
    )
    raw_support_gain = raw_ring_gain[raw_support_rings]
    raw_phase, raw_phase_info = solve_phase_factorized(
        raw_bispectrum,
        raw_coherence,
        steps,
        raw_support_gain,
    )
    raw_recovered = np.fft.ifft2(hermitian_spectrum(
        base_magnitude * raw_support_gain,
        raw_phase,
    )).real
    raw_recovered, raw_recovered_shift = best_cyclic_alignment(
        raw_recovered, truth)

    mean = observed_sum / (
        frames_done * camera.photons_at_white)
    mean, mean_shift = best_cyclic_alignment(mean, truth)
    registration = None
    cascade_reference = None
    if orbit_config.evaluate_registration:
        registration, cascade_reference = registration_cascade(
            mean,
            recovered,
            truth_high,
            scene,
            maps,
            camera,
            orbit_config,
        )

    source_spectrum = np.fft.fft2(truth)
    recovered_spectrum = np.fft.fft2(recovered)
    cross = recovered_spectrum * np.conj(source_spectrum)
    fy = np.fft.fftfreq(orbit_config.grid) * orbit_config.grid
    fx = np.fft.fftfreq(orbit_config.grid) * orbit_config.grid
    rings = np.rint(np.hypot(fy[:, None], fx[None, :])).astype(np.int32)
    cross_sum = np.bincount(
        rings.ravel(), weights=cross.real.ravel())
    cross_imag = np.bincount(
        rings.ravel(), weights=cross.imag.ravel())
    recovered_power = np.bincount(
        rings.ravel(), weights=np.abs(recovered_spectrum).ravel() ** 2)
    truth_power = np.bincount(
        rings.ravel(), weights=np.abs(source_spectrum).ravel() ** 2)
    coherence_truth = (
        np.hypot(cross_sum, cross_imag)
        / np.sqrt(np.maximum(recovered_power * truth_power, 1e-20))
    )
    elapsed = time.perf_counter() - started
    result = {
        "camera": asdict(camera),
        "orbit": asdict(orbit_config),
        "frames_accumulated": frames_done,
        "elapsed_seconds": elapsed,
        "white_noise_floor": float(white_noise_floor),
        "row_axis_noise_floor": float(noise_floor_map[1, 0]),
        "column_axis_noise_floor": float(noise_floor_map[0, 1]),
        "mean_signal_electrons_per_thumbnail_pixel": float(
            mean_signal * area),
        "recovered_global_shift": list(recovered_shift),
        "mean_global_shift": list(mean_shift),
        "truth_fourier_coherence_by_radius": [
            float(value) for value in coherence_truth
        ],
        "split_half_phase_agreement_by_radius": [
            float(value) for value in ring_agreement
        ],
        "split_half_phase_significance_by_radius": [
            float(value) for value in ring_significance
        ],
        "supported_radial_gain": [
            float(value) for value in ring_gain
        ],
        "factorial_split_half_phase_agreement_by_radius": [
            float(value) for value in factorial_ring_agreement
        ],
        "factorial_split_half_phase_significance_by_radius": [
            float(value) for value in factorial_ring_significance
        ],
        "factorial_supported_radial_gain": [
            float(value) for value in factorial_ring_gain
        ],
        "raw_split_half_phase_agreement_by_radius": [
            float(value) for value in raw_ring_agreement
        ],
        "raw_split_half_phase_significance_by_radius": [
            float(value) for value in raw_ring_significance
        ],
        "raw_supported_radial_gain": [
            float(value) for value in raw_ring_gain
        ],
        "cross_gatherer_split_half_phase_agreement_by_radius": [
            float(value) for value in ring_agreement
        ],
        "cross_gatherer_split_half_phase_significance_by_radius": [
            float(value) for value in ring_significance
        ],
        "cross_gatherer_supported_radial_gain": [
            float(value) for value in ring_gain
        ],
        "registration": registration,
        "third_order_estimator": "four_sublattice_cross",
        "factorial_arithmetic_phase": factorial_phase_info,
        "factorial_arithmetic_global_shift": list(
            factorial_recovered_shift),
        "raw_third_order_phase": raw_phase_info,
        "raw_third_order_global_shift": list(raw_recovered_shift),
        "cross_gatherer_phase": phase_info,
        "cross_gatherer_global_shift": list(recovered_shift),
        **phase_info,
        "metrics": {
            "false_mean": score(mean, truth),
            "raw_third_order_orbit": score(raw_recovered, truth),
            "factorial_arithmetic_orbit": score(
                factorial_recovered, truth),
            "cross_gatherer_orbit": score(recovered, truth),
            "orbit": score(recovered, truth),
        },
    }
    images = {
        "truth": truth,
        "mean": mean,
        "raw_orbit": raw_recovered,
        "factorial_orbit": factorial_recovered,
        "cross_orbit": recovered,
        "orbit": recovered,
        "coherence": np.mean(coherence, axis=0),
        "support": np.fft.fftshift(ring_gain[support_rings]),
        "cross_support": np.fft.fftshift(ring_gain[support_rings]),
        "truth_coherence": coherence_truth,
    }
    return result, images


def render(
    result: dict,
    images: dict[str, np.ndarray],
    path: Path,
) -> None:
    figure, axes = plt.subplots(1, 6, figsize=(21, 4.2))
    panels = (
        ("truth", "Original HDR oracle", None),
        ("mean", "False mean oracle", "false_mean"),
        (
            "raw_orbit",
            "Raw third-order orbit",
            "raw_third_order_orbit",
        ),
        (
            "factorial_orbit",
            "Factorial arithmetic orbit",
            "factorial_arithmetic_orbit",
        ),
        (
            "orbit",
            "Four-sublattice cross orbit",
            "orbit",
        ),
        ("cross_support", "Cross-gatherer supported rings", None),
    )
    for axis, (key, title, metric_key) in zip(axes, panels):
        if key in ("support", "cross_support"):
            axis.imshow(images[key], cmap="magma", vmin=0.0, vmax=1.0)
        else:
            axis.imshow(tone_map(images[key]), cmap="gray", vmin=0.0, vmax=1.0)
        if metric_key:
            metric = result["metrics"][metric_key]
            title += (
                f"\nlog {metric['log_psnr_db']:.2f} dB, "
                f"SSIM {metric['log_ssim']:.3f}")
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle(
        f"Physical moonlight orbit bootstrap: "
        f"{result['frames_accumulated']} frames, "
        f"{result['orbit']['projection_steps']} projection steps, "
        f"{result['orbit']['grid']}×{result['orbit']['grid']}")
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=4096)
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--evidence-radius", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument(
        "--solver",
        choices=("factorized", "marching", "optimizer"),
        default="factorized",
    )
    parser.add_argument("--registration-group", type=int, default=32)
    parser.add_argument("--evaluate-registration", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("high_vision/out/realistic_orbit_bootstrap.png"),
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("high_vision/out/realistic_orbit_bootstrap.json"),
    )
    args = parser.parse_args()
    camera = MoonlightCamera(
        frames=args.frames,
        registration_group=args.registration_group,
    )
    orbit = OrbitBench(
        grid=args.grid,
        frames=args.frames,
        projection_steps=args.steps,
        optimizer_iterations=args.iterations,
        evaluate_registration=args.evaluate_registration,
        phase_solver=args.solver,
        evidence_radius=args.evidence_radius,
    )
    result, images = run(camera, orbit)
    render(result, images, args.output)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"wrote {args.output}")
    print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
