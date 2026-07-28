#!/usr/bin/env python3
"""Independent sublattice orbits with sparse nucleation and gauge consensus."""

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

from closure_transport_consensus import (
    ClosureTransportConfig,
    solve_closure_transport_consensus,
)
from poisson_orbit_demo import (
    best_cyclic_alignment,
    hermitian_spectrum,
    select_steps,
)
from realistic_moonlight_bench import (
    MoonlightCamera,
    capture_stream,
    hdr_scene,
    sensor_maps,
    tone_map,
)
from realistic_orbit_bootstrap import (
    calibrated_sublattice_thumbnails,
    distinct_gatherer_product,
    poisson_factorial_bispectrum,
    score,
    solve_phase_factorized,
    split_half_radial_support,
)


@dataclass(frozen=True)
class ConsensusBench:
    grid: int = 64
    frames: int = 4096
    projection_steps: int = 16
    evidence_radius: int = 12
    nucleus_radius: int = 3
    batch: int = 128
    transport_sweeps: int = 24


def frequency_geometry(
    grid: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frequency = np.fft.fftfreq(grid) * grid
    fy = frequency[:, None] * np.ones((1, grid))
    fx = np.ones((grid, 1)) * frequency[None, :]
    radius = np.rint(np.hypot(fy, fx)).astype(np.int32)
    return fy, fx, radius


def fit_phase_ramp(
    source_phase: np.ndarray,
    reference_phase: np.ndarray,
    mask: np.ndarray,
    weight: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Align one orbit chart to another by its translation gauge."""
    if source_phase.shape != reference_phase.shape or mask.shape != source_phase.shape:
        raise ValueError("phase and mask shapes must agree")
    grid = source_phase.shape[0]
    fy, fx, _ = frequency_geometry(grid)
    selected = mask & ((fy != 0) | (fx != 0))
    if np.count_nonzero(selected) < 3:
        return source_phase.copy(), (0.0, 0.0)
    difference = np.angle(np.exp(
        1j * (source_phase - reference_phase)))
    design = np.column_stack((fy[selected], fx[selected]))
    target = difference[selected]
    if weight is None:
        weights = np.ones_like(target)
    else:
        weights = np.maximum(weight[selected], 1e-8)
    root = np.sqrt(weights)
    coefficient, *_ = np.linalg.lstsq(
        design * root[:, None],
        target * root,
        rcond=None,
    )
    ramp = coefficient[0] * fy + coefficient[1] * fx
    aligned = source_phase - ramp
    shift = (
        float(-coefficient[0] * grid / (2.0 * np.pi)),
        float(-coefficient[1] * grid / (2.0 * np.pi)),
    )
    return aligned, shift


def consensus_radial_support(
    significances: np.ndarray,
    phase_consistency: np.ndarray,
    amplitude_consistency: np.ndarray,
    nucleus_gain: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine independent source friction without forcing equal advancement."""
    source_count, ring_count = significances.shape
    combined_sigma = np.zeros(ring_count)
    gain = np.zeros(ring_count)
    for radius in range(ring_count):
        positive = significances[:, radius] > 0.0
        if np.count_nonzero(positive) >= 2:
            combined_sigma[radius] = (
                np.sum(significances[positive, radius])
                / np.sqrt(np.count_nonzero(positive))
            )
        effective = (
            combined_sigma[radius]
            * phase_consistency[radius]
            * np.sqrt(max(amplitude_consistency[radius], 0.0))
        )
        gain[radius] = np.clip((effective - 3.0) / 3.0, 0.0, 1.0)
    gain[:len(nucleus_gain)] = np.maximum(
        gain[:len(nucleus_gain)], nucleus_gain)
    gain[0] = 1.0
    return combined_sigma, gain


def gaussian_floor(
    camera: MoonlightCamera,
    grid: int,
    sample_area: int,
) -> np.ndarray:
    block_rows = int(np.sqrt(sample_area))
    floor = np.full(
        (grid, grid),
        grid ** 2
        * (
            camera.read_noise_electrons ** 2
            + camera.electrons_per_dn ** 2 / 12.0
        )
        / sample_area,
        dtype=np.float64,
    )
    floor[:, 0] += (
        grid ** 3 * camera.row_noise_electrons ** 2 / block_rows)
    floor[0, :] += (
        grid ** 3 * camera.column_noise_electrons ** 2 / block_rows)
    return floor


def run(
    camera: MoonlightCamera,
    config: ConsensusBench,
) -> tuple[dict, dict[str, np.ndarray]]:
    if camera.size % config.grid:
        raise ValueError("camera size must be divisible by consensus grid")
    if (camera.size // config.grid) % 2:
        raise ValueError("sensor blocks must admit four 2x2 sublattices")
    scene = hdr_scene(camera)
    maps = sensor_maps(camera)
    margin = camera.canvas_margin
    truth_full = scene[
        margin:margin + camera.size,
        margin:margin + camera.size,
    ]
    block_side = camera.size // config.grid
    full_area = block_side ** 2
    source_area = full_area // 4
    truth = truth_full.reshape(
        config.grid, block_side, config.grid, block_side
    ).mean(axis=(1, 3))
    steps = select_steps(
        config.grid,
        config.grid,
        config.projection_steps,
        full_circle_support=False,
    )
    fy, fx, rings = frequency_geometry(config.grid)
    evidence_mask = np.hypot(fy, fx) <= config.evidence_radius
    nucleus_mask = np.hypot(fy, fx) <= config.nucleus_radius
    evidence_y, evidence_x = np.nonzero(evidence_mask)
    nucleus_y, nucleus_x = np.nonzero(nucleus_mask)
    source_count = 4
    step_count = len(steps)
    shape = (source_count, config.grid, config.grid)
    source_sum = np.zeros(shape, dtype=np.float64)
    source_power_sum = np.zeros(shape, dtype=np.float64)
    source_half_sum = np.zeros((2,) + shape, dtype=np.float64)
    source_half_power_sum = np.zeros((2,) + shape, dtype=np.float64)
    source_bispectrum_sum = np.zeros(
        (source_count, step_count, config.grid, config.grid),
        dtype=np.complex128,
    )
    source_bispectrum_abs_sum = np.zeros(
        source_bispectrum_sum.shape, dtype=np.float64)
    source_bispectrum_half_sum = np.zeros(
        (2,) + source_bispectrum_sum.shape,
        dtype=np.complex128,
    )
    transport_bispectrum_sum = np.zeros_like(source_bispectrum_sum)
    transport_bispectrum_abs_sum = np.zeros(
        transport_bispectrum_sum.shape, dtype=np.float64)
    transport_bispectrum_half_sum = np.zeros(
        (2,) + transport_bispectrum_sum.shape,
        dtype=np.complex128,
    )
    transport_counts = np.zeros((source_count, 2), dtype=np.int64)
    nucleus_sum = np.zeros(
        (step_count, config.grid, config.grid), dtype=np.complex128)
    nucleus_abs_sum = np.zeros(nucleus_sum.shape, dtype=np.float64)
    nucleus_half_sum = np.zeros(
        (2,) + nucleus_sum.shape, dtype=np.complex128)
    full_power_sum = np.zeros(
        (config.grid, config.grid), dtype=np.float64)
    full_sum = np.zeros_like(full_power_sum)
    half_counts = np.zeros(2, dtype=np.int64)
    frames_done = 0
    batch_sources: list[np.ndarray] = []
    batch_parity: list[int] = []
    batch_transport_source: list[int] = []
    batch_transport_half: list[int] = []
    started = time.perf_counter()

    def consume(
        items: list[np.ndarray],
        parities: list[int],
        transport_sources: list[int],
        transport_halves: list[int],
    ) -> None:
        nonlocal frames_done
        if not items:
            return
        values = np.asarray(items, dtype=np.float64)
        parity = np.asarray(parities)
        transport_source = np.asarray(transport_sources)
        transport_half = np.asarray(transport_halves)
        for source in range(source_count):
            for half in (0, 1):
                transport_counts[source, half] += int(np.count_nonzero(
                    (transport_source == source)
                    & (transport_half == half)
                ))
        spectra = np.fft.fft2(values, axes=(-2, -1))
        full_values = np.mean(values, axis=1)
        full_spectra = np.mean(spectra, axis=1)
        source_sum[:] += np.sum(values, axis=0)
        source_power_sum[:] += np.sum(np.abs(spectra) ** 2, axis=0)
        full_sum[:] += np.sum(full_values, axis=0)
        full_power_sum[:] += np.sum(np.abs(full_spectra) ** 2, axis=0)
        for half in (0, 1):
            selected = parity == half
            count = int(np.count_nonzero(selected))
            half_counts[half] += count
            source_half_sum[half] += np.sum(values[selected], axis=0)
            source_half_power_sum[half] += np.sum(
                np.abs(spectra[selected]) ** 2, axis=0)
        for step_index, (sy, sx) in enumerate(steps):
            total_y = (evidence_y + sy) % config.grid
            total_x = (evidence_x + sx) % config.grid
            product = (
                spectra[:, :, evidence_y, evidence_x]
                * spectra[:, :, sy, sx, None]
                * np.conj(spectra[:, :, total_y, total_x])
            )
            source_bispectrum_sum[
                :, step_index, evidence_y, evidence_x
            ] += np.sum(product, axis=0)
            source_bispectrum_abs_sum[
                :, step_index, evidence_y, evidence_x
            ] += np.sum(np.abs(product), axis=0)
            for half in (0, 1):
                selected = parity == half
                source_bispectrum_half_sum[
                    half, :, step_index, evidence_y, evidence_x
                ] += np.sum(product[selected], axis=0).T

            transport_product = distinct_gatherer_product(
                spectra,
                (sy, sx),
                (evidence_y, evidence_x),
            )
            for source in range(source_count):
                selected_source = transport_source == source
                transport_bispectrum_sum[
                    source, step_index, evidence_y, evidence_x
                ] += np.sum(transport_product[selected_source], axis=0)
                transport_bispectrum_abs_sum[
                    source, step_index, evidence_y, evidence_x
                ] += np.sum(
                    np.abs(transport_product[selected_source]), axis=0)
                for half in (0, 1):
                    selected = (
                        selected_source & (transport_half == half))
                    transport_bispectrum_half_sum[
                        half, source, step_index, evidence_y, evidence_x
                    ] += np.sum(transport_product[selected], axis=0)

            nucleus_product = distinct_gatherer_product(
                spectra,
                (sy, sx),
                (nucleus_y, nucleus_x),
            )
            nucleus_sum[
                step_index, nucleus_y, nucleus_x
            ] += np.sum(nucleus_product, axis=0)
            nucleus_abs_sum[
                step_index, nucleus_y, nucleus_x
            ] += np.sum(np.abs(nucleus_product), axis=0)
            for half in (0, 1):
                selected = parity == half
                nucleus_half_sum[
                    half, step_index, nucleus_y, nucleus_x
                ] += np.sum(nucleus_product[selected], axis=0)
        frames_done += len(items)

    for index, frame, _ in capture_stream(scene, camera, maps):
        batch_sources.append(calibrated_sublattice_thumbnails(
            frame, maps, camera, config.grid))
        batch_parity.append(index % 2)
        batch_transport_source.append(index % source_count)
        batch_transport_half.append((index // source_count) % 2)
        if len(batch_sources) == config.batch:
            consume(
                batch_sources,
                batch_parity,
                batch_transport_source,
                batch_transport_half,
            )
            batch_sources.clear()
            batch_parity.clear()
            batch_transport_source.clear()
            batch_transport_half.clear()
    consume(
        batch_sources,
        batch_parity,
        batch_transport_source,
        batch_transport_half,
    )

    mean_hot = float(np.mean(maps["hot_rate"]))
    background = camera.dark_electrons + mean_hot
    source_gaussian_floor = gaussian_floor(
        camera, config.grid, source_area)
    source_phases = []
    source_magnitudes = []
    source_ring_gains = []
    source_proposal_gains = []
    source_ring_significance = []
    source_phase_weights = []
    source_phase_info = []
    source_metrics = []
    for source in range(source_count):
        photon_dc = (
            float(np.sum(source_sum[source])) / frames_done
            + config.grid ** 2 * background
        )
        raw_bispectrum = source_bispectrum_sum[source] / frames_done
        corrected = poisson_factorial_bispectrum(
            raw_bispectrum,
            source_power_sum[source] / frames_done
            - source_gaussian_floor,
            photon_dc,
            steps,
            source_area,
        )
        corrected_halves = np.zeros(
            (2,) + corrected.shape, dtype=np.complex128)
        for half in (0, 1):
            count = int(half_counts[half])
            half_dc = (
                float(np.sum(source_half_sum[half, source])) / count
                + config.grid ** 2 * background
            )
            corrected_halves[half] = poisson_factorial_bispectrum(
                source_bispectrum_half_sum[half, source] / count,
                source_half_power_sum[half, source] / count
                - source_gaussian_floor,
                half_dc,
                steps,
                source_area,
            )
        _, _, significance, ring_gain = split_half_radial_support(
            corrected_halves,
            config.grid,
            measured_mask=evidence_mask,
        )
        proposal_gain = np.clip(significance / 3.0, 0.0, 1.0)
        proposal_gain[0] = 1.0
        # A source cannot jump across a radial band where its own two halves
        # disagree. The tentative front may be weak, but it remains connected
        # to the shared nucleus.
        for radius in range(
            config.nucleus_radius + 1, len(proposal_gain)
        ):
            if proposal_gain[radius - 1] <= 0.0:
                proposal_gain[radius] = 0.0
        proposal_support = proposal_gain[rings]
        published_support = ring_gain[rings]
        coherence = (
            np.abs(corrected) * frames_done
            / np.maximum(source_bispectrum_abs_sum[source], 1e-12)
        )
        coherence[:, 0, :] *= 0.05
        coherence[:, :, 0] *= 0.05
        phase, phase_info = solve_phase_factorized(
            corrected, coherence, steps, proposal_support)
        mean_signal = max(
            float(np.sum(source_sum[source]))
            / (frames_done * config.grid ** 2),
            0.0,
        )
        total_floor = source_gaussian_floor + (
            config.grid ** 2 * (mean_signal + background) / source_area)
        source_power = (
            source_power_sum[source] / frames_done - total_floor)
        magnitude = np.sqrt(np.maximum(source_power, 0.0)) / max(
            camera.photons_at_white, 1e-12)
        magnitude[0, 0] = max(
            float(np.sum(source_sum[source]))
            / (frames_done * camera.photons_at_white),
            0.0,
        )
        estimate = np.fft.ifft2(hermitian_spectrum(
            magnitude * published_support, phase)).real
        estimate, _ = best_cyclic_alignment(estimate, truth)
        source_phases.append(phase)
        source_magnitudes.append(magnitude)
        source_ring_gains.append(ring_gain)
        source_proposal_gains.append(proposal_gain)
        source_ring_significance.append(significance)
        source_phase_weights.append(np.mean(coherence, axis=0))
        source_phase_info.append(phase_info)
        source_metrics.append(score(estimate, truth))

    nucleus_bispectrum = nucleus_sum / frames_done
    nucleus_coherence = (
        np.abs(nucleus_sum) / np.maximum(nucleus_abs_sum, 1e-12))
    nucleus_coherence[:, 0, :] *= 0.05
    nucleus_coherence[:, :, 0] *= 0.05
    (
        nucleus_rings,
        nucleus_agreement,
        nucleus_significance,
        nucleus_ring_gain,
    ) = split_half_radial_support(
        nucleus_half_sum,
        config.grid,
        measured_mask=nucleus_mask,
    )
    nucleus_support = nucleus_ring_gain[nucleus_rings]
    nucleus_phase, nucleus_info = solve_phase_factorized(
        nucleus_bispectrum,
        nucleus_coherence,
        steps,
        nucleus_support,
    )

    source_phases_array = np.asarray(source_phases)
    source_magnitudes_array = np.asarray(source_magnitudes)
    source_weights_array = np.asarray(source_phase_weights)
    source_gain_array = np.asarray(source_ring_gains)
    source_proposal_array = np.asarray(source_proposal_gains)
    transport_source_bispectra = []
    transport_source_coherences = []
    transport_source_proposals = []
    transport_source_significance = []
    for source in range(source_count):
        count = int(np.sum(transport_counts[source]))
        if count <= 0:
            raise ValueError("empty full-aperture temporal source")
        source_bispectrum = transport_bispectrum_sum[source] / count
        source_coherence = (
            np.abs(transport_bispectrum_sum[source])
            / np.maximum(
                transport_bispectrum_abs_sum[source], 1e-12)
        )
        source_coherence[:, 0, :] *= 0.05
        source_coherence[:, :, 0] *= 0.05
        _, _, significance, _ = split_half_radial_support(
            transport_bispectrum_half_sum[:, source],
            config.grid,
            measured_mask=evidence_mask,
        )
        proposal = np.clip(significance / 3.0, 0.0, 1.0)
        proposal[0] = 1.0
        nucleus_limit = min(
            config.nucleus_radius + 1,
            len(proposal),
            len(nucleus_ring_gain),
        )
        proposal[:nucleus_limit] = np.maximum(
            proposal[:nucleus_limit],
            nucleus_ring_gain[:nucleus_limit],
        )
        for radius in range(
            config.nucleus_radius + 1, len(proposal)
        ):
            if proposal[radius - 1] <= 0.0:
                proposal[radius] = 0.0
        transport_source_bispectra.append(source_bispectrum)
        transport_source_coherences.append(source_coherence)
        transport_source_proposals.append(proposal)
        transport_source_significance.append(significance)
    aligned_phases = []
    gauge_shifts = []
    overlap = nucleus_support > 0.0
    nucleus_weight = np.mean(nucleus_coherence, axis=0)
    for source in range(source_count):
        source_support = source_proposal_array[source][rings] > 0.0
        aligned, gauge_shift = fit_phase_ramp(
            source_phases_array[source],
            nucleus_phase,
            overlap & source_support,
            source_weights_array[source] * nucleus_weight,
        )
        aligned_phases.append(aligned)
        gauge_shifts.append(gauge_shift)
    aligned_phases = np.asarray(aligned_phases)

    bin_weights = (
        source_weights_array
        * np.asarray([
            source_proposal_array[source][rings]
            for source in range(source_count)
        ])
    )
    phase_vector = np.sum(
        bin_weights * np.exp(1j * aligned_phases), axis=0)
    phase_weight = np.sum(bin_weights, axis=0)
    consensus_phase = np.angle(phase_vector)
    phase_consistency_map = (
        np.abs(phase_vector) / np.maximum(phase_weight, 1e-12))
    consensus_phase[nucleus_support > 0.0] = nucleus_phase[
        nucleus_support > 0.0]

    amplitude_consistency_map = (
        np.sum(source_magnitudes_array, axis=0) ** 2
        / np.maximum(
            source_count * np.sum(
                source_magnitudes_array ** 2, axis=0),
            1e-12,
        )
    )
    ring_count = int(np.max(rings)) + 1
    phase_consistency = np.zeros(ring_count)
    amplitude_consistency = np.zeros(ring_count)
    for radius in range(ring_count):
        selected = rings == radius
        phase_consistency[radius] = float(np.mean(
            phase_consistency_map[selected]))
        amplitude_consistency[radius] = float(np.mean(
            amplitude_consistency_map[selected]))
    significance_array = np.asarray(source_ring_significance)
    combined_sigma, consensus_ring_gain = consensus_radial_support(
        significance_array,
        phase_consistency,
        amplitude_consistency,
        nucleus_ring_gain,
    )
    consensus_support = consensus_ring_gain[rings]
    (
        transport_phase_phasor,
        transport_support,
        transport_info,
        _,
    ) = solve_closure_transport_consensus(
        np.asarray(transport_source_bispectra),
        np.asarray(transport_source_coherences),
        steps,
        np.asarray([
            transport_source_proposals[source][rings]
            for source in range(source_count)
        ]),
        ClosureTransportConfig(sweeps=config.transport_sweeps),
    )
    transport_phase = np.angle(transport_phase_phasor)

    full_mean_signal = max(
        float(np.sum(full_sum))
        / (frames_done * config.grid ** 2),
        0.0,
    )
    full_gaussian_floor = gaussian_floor(
        camera, config.grid, full_area)
    full_total_floor = full_gaussian_floor + (
        config.grid ** 2 * (full_mean_signal + background) / full_area)
    full_power = full_power_sum / frames_done - full_total_floor
    full_magnitude = np.sqrt(np.maximum(full_power, 0.0)) / max(
        camera.photons_at_white, 1e-12)
    full_magnitude[0, 0] = max(
        float(np.sum(full_sum))
        / (frames_done * camera.photons_at_white),
        0.0,
    )
    posthoc_consensus = np.fft.ifft2(hermitian_spectrum(
        full_magnitude * consensus_support,
        consensus_phase,
    )).real
    posthoc_consensus, posthoc_shift = best_cyclic_alignment(
        posthoc_consensus, truth)
    transport_consensus = np.fft.ifft2(hermitian_spectrum(
        full_magnitude * transport_support,
        transport_phase,
    )).real
    transport_consensus, transport_shift = best_cyclic_alignment(
        transport_consensus, truth)
    transport_oracle = np.fft.ifft2(
        np.fft.fft2(truth) * transport_support).real
    transport_oracle, _ = best_cyclic_alignment(
        transport_oracle, truth)
    nucleus_only = np.fft.ifft2(hermitian_spectrum(
        full_magnitude * nucleus_support,
        nucleus_phase,
    )).real
    nucleus_only, _ = best_cyclic_alignment(nucleus_only, truth)
    false_mean = full_sum / (
        frames_done * camera.photons_at_white)
    false_mean, _ = best_cyclic_alignment(false_mean, truth)

    elapsed = time.perf_counter() - started
    result = {
        "camera": asdict(camera),
        "consensus": asdict(config),
        "frames_accumulated": frames_done,
        "elapsed_seconds": elapsed,
        "nucleus_phase": nucleus_info,
        "nucleus_split_half_agreement_by_radius": [
            float(value) for value in nucleus_agreement
        ],
        "nucleus_split_half_significance_by_radius": [
            float(value) for value in nucleus_significance
        ],
        "nucleus_supported_radial_gain": [
            float(value) for value in nucleus_ring_gain
        ],
        "source_split_half_significance_by_radius": [
            [float(value) for value in source]
            for source in significance_array
        ],
        "source_supported_radial_gain": [
            [float(value) for value in source]
            for source in source_gain_array
        ],
        "source_tentative_radial_gain": [
            [float(value) for value in source]
            for source in source_proposal_array
        ],
        "transport_temporal_source_counts": [
            [int(value) for value in source]
            for source in transport_counts
        ],
        "transport_temporal_source_significance_by_radius": [
            [float(value) for value in source]
            for source in transport_source_significance
        ],
        "transport_temporal_source_tentative_gain": [
            [float(value) for value in source]
            for source in transport_source_proposals
        ],
        "source_phase": source_phase_info,
        "source_gauge_shift_yx": [
            [float(value) for value in shift] for shift in gauge_shifts
        ],
        "phase_consistency_by_radius": [
            float(value) for value in phase_consistency
        ],
        "amplitude_consistency_by_radius": [
            float(value) for value in amplitude_consistency
        ],
        "combined_source_significance_by_radius": [
            float(value) for value in combined_sigma
        ],
        "consensus_supported_radial_gain": [
            float(value) for value in consensus_ring_gain
        ],
        "posthoc_consensus_global_shift": list(posthoc_shift),
        "closure_transport_global_shift": list(transport_shift),
        "closure_transport": transport_info,
        "metrics": {
            "false_mean": score(false_mean, truth),
            "nucleus_only": score(nucleus_only, truth),
            "individual_sources": source_metrics,
            "posthoc_phase_consensus": score(
                posthoc_consensus, truth),
            "closure_transport_consensus": score(
                transport_consensus, truth),
            "closure_transport_support_ceiling": score(
                transport_oracle, truth),
            "closure_transport_vs_support_oracle": score(
                transport_consensus, transport_oracle),
        },
    }
    images = {
        "truth": truth,
        "false_mean": false_mean,
        "nucleus": nucleus_only,
        "posthoc_consensus": posthoc_consensus,
        "consensus": transport_consensus,
        "support_oracle": transport_oracle,
        "support": np.fft.fftshift(transport_support),
        "phase_consistency": np.fft.fftshift(phase_consistency_map),
    }
    return result, images


def render(
    result: dict,
    images: dict[str, np.ndarray],
    path: Path,
) -> None:
    figure, axes = plt.subplots(1, 7, figsize=(24, 4.2))
    panels = (
        ("truth", "Original HDR oracle", None),
        ("false_mean", "False mean oracle", "false_mean"),
        ("nucleus", "Sparse cross-source nucleus", "nucleus_only"),
        (
            "posthoc_consensus",
            "Rejected post-hoc phase average",
            "posthoc_phase_consensus",
        ),
        (
            "consensus",
            "Closure-transport consensus",
            "closure_transport_consensus",
        ),
        ("support_oracle", "Oracle at admitted support", None),
        ("support", "Closure-transport support", None),
    )
    for axis, (key, title, metric_key) in zip(axes, panels):
        if key in ("support", "phase_consistency"):
            axis.imshow(images[key], cmap="magma", vmin=0.0, vmax=1.0)
        else:
            axis.imshow(
                tone_map(images[key]), cmap="gray", vmin=0.0, vmax=1.0)
        if metric_key:
            metric = result["metrics"][metric_key]
            title += (
                f"\nlog {metric['log_psnr_db']:.2f} dB, "
                f"SSIM {metric['log_ssim']:.3f}")
        axis.set_title(title)
        axis.axis("off")
    config = result["consensus"]
    figure.suptitle(
        f"Multi-orbit nucleation: {result['frames_accumulated']} frames, "
        f"{config['projection_steps']} steps, "
        f"nucleus r≤{config['nucleus_radius']}, "
        f"discovery r≤{config['evidence_radius']}")
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=4096)
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--evidence-radius", type=int, default=12)
    parser.add_argument("--nucleus-radius", type=int, default=3)
    parser.add_argument("--transport-sweeps", type=int, default=24)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "high_vision/out/realistic_multisource_consensus.png"),
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=Path(
            "high_vision/out/realistic_multisource_consensus.json"),
    )
    args = parser.parse_args()
    camera = MoonlightCamera(frames=args.frames)
    config = ConsensusBench(
        grid=args.grid,
        frames=args.frames,
        projection_steps=args.steps,
        evidence_radius=args.evidence_radius,
        nucleus_radius=args.nucleus_radius,
        transport_sweeps=args.transport_sweeps,
    )
    result, images = run(camera, config)
    render(result, images, args.output)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"wrote {args.output}")
    print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
