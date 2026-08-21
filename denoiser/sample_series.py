"""Composable 1-D signals and shared 1-D/2-D corruption controls."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


COMPONENTS = (
    "trend",
    "broad bump",
    "smooth step",
    "tone",
    "chirp",
    "damped ripple",
    "pulses",
)

PRESETS: dict[str, tuple[str, ...]] = {
    "mixed transport stress": (
        "trend", "broad bump", "smooth step", "chirp", "damped ripple"),
    "smooth geometry": ("trend", "broad bump", "smooth step"),
    "step + carrier": ("trend", "smooth step", "tone"),
    "chirp packet": ("trend", "broad bump", "chirp"),
    "pulses + drift": ("trend", "pulses"),
    "oscillatory composite": ("tone", "chirp", "damped ripple"),
    "flat negative control": (),
    "custom": (),
}

CORRUPTIONS = (
    "none",
    "Gaussian additive",
    "uniform additive",
    "Laplace additive",
    "salt and pepper",
    "random-value replacement",
    "multiplicative",
    "mixed replacement + uniform",
)


DEFAULT_PARAMETERS = {
    "baseline": 0.24,
    "trend_amplitude": 0.13,
    "bump_amplitude": 0.17,
    "bump_center": 0.24,
    "bump_width": 0.09,
    "step_amplitude": 0.29,
    "step_center": 0.52,
    "step_width": 0.006,
    "tone_amplitude": 0.055,
    "tone_cycles": 13.0,
    "chirp_amplitude": 0.055,
    "chirp_cycles": 16.0,
    "chirp_sweep": 18.0,
    "chirp_start": 0.62,
    "ripple_amplitude": 0.045,
    "ripple_cycles": 44.0,
    "ripple_start": 0.70,
    "ripple_decay": 7.0,
    "pulse_amplitude": 0.17,
    "pulse_width": 0.018,
}


def compose_series(
    size: int,
    enabled: Mapping[str, bool] | set[str] | tuple[str, ...],
    parameters: Mapping[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Compose named continuous components into one bounded test signal."""
    if int(size) < 16:
        raise ValueError("a composite series needs at least 16 samples")
    active = (
        {name for name, selected in enabled.items() if selected}
        if isinstance(enabled, Mapping)
        else set(enabled)
    )
    unknown = active.difference(COMPONENTS)
    if unknown:
        raise ValueError(f"unknown series components: {sorted(unknown)}")
    config = dict(DEFAULT_PARAMETERS)
    if parameters:
        config.update({key: float(value) for key, value in parameters.items()})
    x = np.linspace(0.0, 1.0, int(size), endpoint=False)
    fields: dict[str, np.ndarray] = {}

    if "trend" in active:
        fields["trend"] = config["trend_amplitude"] * x
    if "broad bump" in active:
        width = max(abs(config["bump_width"]), np.finfo(float).eps)
        fields["broad bump"] = config["bump_amplitude"] * np.exp(
            -((x - config["bump_center"]) / width) ** 2)
    if "smooth step" in active:
        width = max(abs(config["step_width"]), np.finfo(float).eps)
        z = np.clip((x - config["step_center"]) / width, -700.0, 700.0)
        fields["smooth step"] = config["step_amplitude"] / (1.0 + np.exp(-z))
    if "tone" in active:
        fields["tone"] = config["tone_amplitude"] * np.sin(
            2.0 * np.pi * config["tone_cycles"] * x)
    if "chirp" in active:
        gate_width = 1.0 / int(size)
        z = np.clip((x - config["chirp_start"]) / gate_width, -700.0, 700.0)
        gate = 1.0 / (1.0 + np.exp(-z))
        phase = 2.0 * np.pi * (
            config["chirp_cycles"] * x + config["chirp_sweep"] * x * x)
        fields["chirp"] = config["chirp_amplitude"] * gate * np.sin(phase)
    if "damped ripple" in active:
        offset = np.maximum(x - config["ripple_start"], 0.0)
        gate_width = 1.0 / int(size)
        z = np.clip((x - config["ripple_start"]) / gate_width, -700.0, 700.0)
        gate = 1.0 / (1.0 + np.exp(-z))
        fields["damped ripple"] = (
            config["ripple_amplitude"]
            * gate
            * np.exp(-config["ripple_decay"] * offset)
            * np.sin(2.0 * np.pi * config["ripple_cycles"] * offset)
        )
    if "pulses" in active:
        width = max(abs(config["pulse_width"]), np.finfo(float).eps)
        fields["pulses"] = config["pulse_amplitude"] * (
            np.exp(-((x - 0.20) / width) ** 2)
            - 0.75 * np.exp(-((x - 0.48) / (1.4 * width)) ** 2)
            + 0.60 * np.exp(-((x - 0.76) / (0.8 * width)) ** 2)
        )

    signal = np.full(x.shape, config["baseline"], dtype=np.float64)
    for component in fields.values():
        signal += component
    return x, np.clip(signal, 0.0, 1.0), fields


def corrupt(
    clean: np.ndarray,
    kind: str,
    *,
    amount: float,
    density: float,
    seed: int,
) -> np.ndarray:
    """Apply the same explicit corruption catalogue to a line or image."""
    value = np.asarray(clean, dtype=np.float64)
    if kind not in CORRUPTIONS:
        raise ValueError(f"unknown corruption: {kind}")
    if kind == "none":
        return value.copy()
    scale = max(float(amount), 0.0)
    probability = float(np.clip(density, 0.0, 1.0))
    rng = np.random.default_rng(int(seed))
    if kind == "Gaussian additive":
        result = value + rng.normal(0.0, scale, value.shape)
    elif kind == "uniform additive":
        result = value + rng.uniform(-scale, scale, value.shape)
    elif kind == "Laplace additive":
        result = value + rng.laplace(0.0, scale, value.shape)
    elif kind == "salt and pepper":
        result = value.copy()
        draw = rng.random(value.shape)
        result[draw < probability / 2.0] = 0.0
        result[(draw >= probability / 2.0) & (draw < probability)] = 1.0
    elif kind == "random-value replacement":
        result = value.copy()
        mask = rng.random(value.shape) < probability
        result[mask] = rng.random(int(np.sum(mask)))
    elif kind == "multiplicative":
        result = value * (1.0 + rng.normal(0.0, scale, value.shape))
    else:
        result = value + rng.uniform(-scale, scale, value.shape)
        mask = rng.random(value.shape) < probability
        result[mask] = rng.random(int(np.sum(mask)))
    return np.clip(result, 0.0, 1.0)

