"""Ownership-field bridge for the fused Jpegli quantizer.

The JLDZ format is deliberately tiny: a 16-byte little-endian header
(``JLDZ``, version, luma block width, luma block height), followed by float32
thresholds shaped ``(3, height, width, 64)`` in natural DCT order.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from .certified_relaxation import _coefficients
from .core import quality_table


def ownership_dead_zone_field(
    ycc: np.ndarray,
    labels: np.ndarray,
    quality: int,
    strength: float = 0.25,
    edge_protection: float = 2.0,
    transported_coefficients: np.ndarray | None = None,
) -> np.ndarray:
    """Build a conservative phase-frustration dead-zone opportunity field.

    Coherent coefficients remain owned by their channel/frequency. Extra
    dead-zone is assigned to weak coefficients in a region whose signed phase
    cancels collectively—the perturbations least able to establish stable
    ownership. DC is immutable.
    """
    coefficients = _coefficients(ycc)
    if labels.size != len(coefficients):
        raise ValueError("labels must contain one value per 8x8 luma block")
    quantizers = np.stack(
        (
            quality_table(quality, False).reshape(64),
            quality_table(quality, True).reshape(64),
            quality_table(quality, True).reshape(64),
        ),
        axis=-1,
    )
    normalized = coefficients / quantizers[None, :, :]
    aligned = normalized
    if transported_coefficients is not None:
        transported = np.asarray(transported_coefficients, dtype=np.float64)
        if transported.shape != coefficients.shape:
            raise ValueError("transported coefficients must match source coefficients")
        aligned = transported / quantizers[None, :, :]
    field = np.zeros_like(normalized, dtype=np.float64)
    flat_labels = labels.ravel()
    for region in range(int(flat_labels.max()) + 1):
        member = np.flatnonzero(flat_labels == region)
        values = aligned[member, 1:, :]
        absolute = np.abs(values)
        coherence = np.abs(np.sum(values, axis=0)) / (
            np.sum(absolute, axis=0) + 1e-9
        )
        regional_scale = np.median(absolute, axis=0) + 0.25
        weak = np.exp(-absolute / regional_scale[None, :, :])
        # The square makes the field selective: moderately coherent phase is
        # retained, while truly frustrated weak constituents receive most of
        # the available dead-zone.
        opportunity = (1.0 - coherence[None, :, :]) ** 2 * weak
        if transported_coefficients is not None:
            source_absolute = np.abs(normalized[member, 1:, :])
            released = np.clip(source_absolute - absolute, 0.0, None) / (
                source_absolute + 0.25
            )
            # Transport is the ownership witness: only constituents whose
            # local claim was actually released receive the full opportunity.
            opportunity *= 0.25 + 0.75 * released
        field[member, 1:, :] = max(float(strength), 0.0) * opportunity
    # Preserve geometric boundaries. The quotient decides who owns a block;
    # this gate decides how reluctant that owner is to surrender coefficients.
    # Low-order luma AC energy is a stable block-edge proxy and avoids deriving
    # boundaries from the same high-frequency noise we intend to remove.
    edge_energy = np.sqrt(np.mean(normalized[:, 1:16, 0] ** 2, axis=1))
    edge_scale = float(np.percentile(edge_energy, 70)) + 1e-9
    edge_gate = np.exp(-max(float(edge_protection), 0.0) * edge_energy / edge_scale)
    field *= edge_gate[:, None, None]
    return field.reshape(labels.shape[0], labels.shape[1], 64, 3).transpose(
        3, 0, 1, 2
    ).astype("<f4")


def write_jldz(path: Path, field: np.ndarray) -> None:
    array = np.asarray(field, dtype="<f4")
    if array.ndim != 4 or array.shape[0] != 3 or array.shape[3] != 64:
        raise ValueError("field must have shape (3, block_height, block_width, 64)")
    if np.any(~np.isfinite(array)) or np.any(array < 0):
        raise ValueError("dead-zone thresholds must be finite and nonnegative")
    path.parent.mkdir(parents=True, exist_ok=True)
    header = struct.pack("<4sIII", b"JLDZ", 1, array.shape[2], array.shape[1])
    path.write_bytes(header + array.tobytes(order="C"))
