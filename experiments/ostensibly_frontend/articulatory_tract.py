"""Deterministic continuous articulatory vocal-tract synthesis.

This is a Python analysis engine derived from the physical construction in
Neil Thapen's Pink Trombone (2017, MIT license): an LF-like glottal source
drives a 44-section Kelly--Lochbaum digital waveguide.  The code is rewritten
for offline, trajectory-controlled experiments rather than copied phone audio.

Copyright (c) 2017 Neil Thapen for the originating Pink Trombone model.
The originating MIT permission notice is retained in ``THIRD_PARTY_NOTICES``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np


TRACT_SECTIONS = 44


@dataclass(frozen=True)
class ArticulatoryFrame:
    """One point on a smooth vocal-production trajectory."""

    time_seconds: float
    f0_hz: float
    intensity: float
    tenseness: float
    tongue_index: float
    tongue_diameter: float
    lip_diameter: float
    constriction_index: float
    constriction_diameter: float
    velum_diameter: float = 0.01


@dataclass(frozen=True)
class ArticulatorySynthesis:
    samples: np.ndarray
    controls: dict[str, np.ndarray]
    tract_diameter: np.ndarray
    sample_rate: int


def _smoothstep(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def interpolate_trajectory(
    frames: tuple[ArticulatoryFrame, ...],
    times: np.ndarray,
) -> dict[str, np.ndarray]:
    """Interpolate articulator knots with zero-slope smooth transitions."""

    if len(frames) < 2:
        raise ValueError("an articulatory trajectory needs at least two frames")
    knots = np.asarray([frame.time_seconds for frame in frames], dtype=np.float64)
    if knots[0] != 0.0 or np.any(np.diff(knots) <= 0.0):
        raise ValueError("trajectory times must start at zero and increase")
    query = np.asarray(times, dtype=np.float64)
    if query.ndim != 1 or query.size == 0 or query[0] < 0.0 or query[-1] > knots[-1]:
        raise ValueError("trajectory query lies outside its knot interval")
    segment = np.clip(np.searchsorted(knots, query, side="right") - 1, 0, len(knots) - 2)
    fraction = (query - knots[segment]) / (knots[segment + 1] - knots[segment])
    weight = _smoothstep(fraction)
    result: dict[str, np.ndarray] = {}
    for descriptor in fields(ArticulatoryFrame):
        if descriptor.name == "time_seconds":
            continue
        values = np.asarray([getattr(frame, descriptor.name) for frame in frames])
        result[descriptor.name] = (
            values[segment] * (1.0 - weight) + values[segment + 1] * weight
        )
    return result


def tract_diameter_from_controls(
    tongue_index: float,
    tongue_diameter: float,
    lip_diameter: float,
    constriction_index: float,
    constriction_diameter: float,
    *,
    section_count: int = TRACT_SECTIONS,
) -> np.ndarray:
    """Map low-dimensional articulator controls to an oral area profile."""

    if section_count < 30:
        raise ValueError("the tract requires at least 30 sections")
    scale = section_count / 44.0
    blade_start = int(np.floor(10 * scale))
    tip_start = int(np.floor(32 * scale))
    lip_start = int(np.floor(39 * scale))
    diameter = np.full(section_count, 1.5, dtype=np.float64)
    diameter[: int(round(7 * scale))] = 0.6
    diameter[int(round(7 * scale)) : int(round(12 * scale))] = 1.1

    indices = np.arange(blade_start, lip_start, dtype=np.float64)
    tongue_index_scaled = tongue_index * scale
    phase = 1.1 * np.pi * (tongue_index_scaled - indices) / max(
        tip_start - blade_start, 1
    )
    fixed = 2.0 + (tongue_diameter - 2.0) / 1.5
    curve = (1.5 - fixed + 1.7) * np.cos(phase)
    diameter[blade_start:lip_start] = 1.5 - curve

    lip_count = max(2, section_count - lip_start)
    lip_weight = _smoothstep(np.linspace(0.0, 1.0, lip_count))
    diameter[lip_start:] = (
        diameter[lip_start:] * (1.0 - lip_weight) + lip_diameter * lip_weight
    )

    center = constriction_index * scale
    width = 10.0 * scale if center < 25.0 * scale else (
        5.0 * scale
        if center >= tip_start
        else (10.0 - 5.0 * (center / scale - 25.0) / max(tip_start / scale - 25.0, 1e-6)) * scale
    )
    location = np.arange(section_count, dtype=np.float64)
    relative = np.maximum(np.abs(location - center) - 0.5, 0.0)
    taper = np.where(
        relative >= width,
        1.0,
        0.5 * (1.0 - np.cos(np.pi * relative / max(width, 1e-6))),
    )
    constricted = constriction_diameter + (diameter - constriction_diameter) * taper
    diameter = np.minimum(diameter, constricted)
    return np.clip(diameter, 0.05, 4.0)


def _lf_source(
    f0_hz: np.ndarray,
    intensity: np.ndarray,
    tenseness: np.ndarray,
    sample_rate: int,
    seed: int,
) -> np.ndarray:
    """Continuous-phase Liljencrants--Fant-style glottal excitation."""

    phase = np.mod(np.cumsum(f0_hz / sample_rate), 1.0)
    rd = np.clip(3.0 * (1.0 - tenseness), 0.5, 2.7)
    ra = -0.01 + 0.048 * rd
    rk = 0.224 + 0.118 * rd
    rg = (rk / 4.0) * (0.5 + 1.2 * rk) / np.maximum(
        0.11 * rd - ra * (0.5 + 1.2 * rk), 1e-6
    )
    ta = ra
    tp = 1.0 / (2.0 * rg)
    te = tp * (1.0 + rk)
    epsilon = 1.0 / np.maximum(ta, 1e-6)
    shift = np.exp(-epsilon * (1.0 - te))
    delta = np.maximum(1.0 - shift, 1e-6)
    rhs = ((shift - 1.0) / epsilon + (1.0 - te) * shift) / delta
    upper = -(-(te - tp) / 2.0 + rhs)
    omega = np.pi / np.maximum(tp, 1e-6)
    sine_te = np.sin(omega * te)
    y = np.maximum(-np.pi * sine_te * upper / np.maximum(2.0 * tp, 1e-6), 1e-6)
    alpha = np.log(y) / np.minimum(tp / 2.0 - te, -1e-6)
    e0 = -1.0 / np.where(
        np.abs(sine_te * np.exp(alpha * te)) > 1e-8,
        sine_te * np.exp(alpha * te),
        -1e-8,
    )
    open_phase = e0 * np.exp(alpha * phase) * np.sin(omega * phase)
    return_phase = (-np.exp(-epsilon * (phase - te)) + shift) / delta
    voiced = np.where(phase > te, return_phase, open_phase)
    loudness = np.power(np.clip(tenseness, 0.0, 1.0), 0.25)
    rng = np.random.default_rng(seed)
    aspiration = rng.normal(size=phase.size) * (
        0.02 * intensity * (1.0 - np.sqrt(np.clip(tenseness, 0.0, 1.0)))
    )
    return voiced * intensity * loudness + aspiration


def synthesize_articulatory_trajectory(
    frames: tuple[ArticulatoryFrame, ...],
    *,
    sample_rate: int = 48_000,
    control_hop: int = 24,
    seed: int = 20260826,
) -> ArticulatorySynthesis:
    """Render a smooth trajectory through a loss-bearing oral waveguide."""

    duration = frames[-1].time_seconds
    sample_count = int(round(duration * sample_rate))
    if sample_count < 2 or control_hop < 1:
        raise ValueError("synthesis geometry is too short")
    sample_times = np.arange(sample_count, dtype=np.float64) / sample_rate
    controls = interpolate_trajectory(frames, sample_times)
    source = _lf_source(
        controls["f0_hz"],
        controls["intensity"],
        controls["tenseness"],
        sample_rate,
        seed,
    )

    control_indices = np.arange(0, sample_count, control_hop, dtype=np.int64)
    if control_indices[-1] != sample_count - 1:
        control_indices = np.r_[control_indices, sample_count - 1]
    profiles = np.stack(
        [
            tract_diameter_from_controls(
                float(controls["tongue_index"][index]),
                float(controls["tongue_diameter"][index]),
                float(controls["lip_diameter"][index]),
                float(controls["constriction_index"][index]),
                float(controls["constriction_diameter"][index]),
            )
            for index in control_indices
        ]
    )
    sample_to_control = np.minimum(
        np.arange(sample_count, dtype=np.int64) // control_hop,
        profiles.shape[0] - 1,
    )

    right = np.zeros(TRACT_SECTIONS, dtype=np.float64)
    left = np.zeros(TRACT_SECTIONS, dtype=np.float64)
    junction_right = np.zeros(TRACT_SECTIONS + 1, dtype=np.float64)
    junction_left = np.zeros(TRACT_SECTIONS + 1, dtype=np.float64)
    output = np.zeros(sample_count, dtype=np.float64)
    for sample_index in range(sample_count):
        diameter = profiles[sample_to_control[sample_index]]
        area = diameter * diameter
        reflection = (area[:-1] - area[1:]) / np.maximum(
            area[:-1] + area[1:], 1e-12
        )
        vocal_output = 0.0
        for _ in range(2):
            junction_right[0] = 0.75 * left[0] + source[sample_index]
            junction_left[-1] = -0.85 * right[-1]
            wave = reflection * (right[:-1] + left[1:])
            junction_right[1:-1] = right[:-1] - wave
            junction_left[1:-1] = left[1:] + wave
            right[:] = np.clip(0.999 * junction_right[:-1], -1.0, 1.0)
            left[:] = np.clip(0.999 * junction_left[1:], -1.0, 1.0)
            vocal_output += right[-1]
        output[sample_index] = 0.125 * vocal_output
    peak = max(float(np.max(np.abs(output))), 1e-12)
    output = 0.95 * output / peak
    return ArticulatorySynthesis(
        samples=output,
        controls=controls,
        tract_diameter=profiles,
        sample_rate=sample_rate,
    )


def initial_all_trajectory(duration_seconds: float = 0.725) -> tuple[ArticulatoryFrame, ...]:
    """A continuous British-like /ɔː/ -> dark-/l/ hypothesis for “all”."""

    if duration_seconds < 0.50:
        raise ValueError("the initial all trajectory needs room for its held vowel")
    scale = duration_seconds / 0.725

    def frame(time: float, **values: float) -> ArticulatoryFrame:
        return ArticulatoryFrame(time_seconds=time * scale, **values)

    vowel = dict(
        f0_hz=138.0,
        intensity=0.92,
        tenseness=0.62,
        tongue_index=17.7,
        tongue_diameter=2.05,
        lip_diameter=0.82,
        constriction_index=38.0,
        constriction_diameter=1.45,
    )
    return (
        frame(0.000, **{**vowel, "intensity": 0.05}),
        frame(0.070, **vowel),
        frame(0.390, **{**vowel, "f0_hz": 136.0}),
        frame(
            0.560,
            f0_hz=132.0,
            intensity=0.88,
            tenseness=0.60,
            tongue_index=14.2,
            tongue_diameter=2.18,
            lip_diameter=1.18,
            constriction_index=38.0,
            constriction_diameter=0.82,
        ),
        frame(
            0.655,
            f0_hz=129.0,
            intensity=0.70,
            tenseness=0.58,
            tongue_index=13.8,
            tongue_diameter=2.12,
            lip_diameter=1.28,
            constriction_index=38.0,
            constriction_diameter=0.78,
        ),
        frame(
            0.725,
            f0_hz=127.0,
            intensity=0.02,
            tenseness=0.55,
            tongue_index=13.8,
            tongue_diameter=2.12,
            lip_diameter=1.28,
            constriction_index=38.0,
            constriction_diameter=0.78,
        ),
    )
