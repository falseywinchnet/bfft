"""Literal accounting of source jumps and residuals by hard owner.

Every final pixel should have an owner.  That alone does not mean the
partition represented every source discontinuity: a hard cell can cross a
large source jump, forcing its affine/ridge readout to approximate both sides.
These diagnostics distinguish those two cases without classifying objects or
changing the partition.
"""

from __future__ import annotations

import numpy as np


def residual_ownership_diagnostics(
    source_rgb: np.ndarray,
    labels: np.ndarray,
    residual_energy: np.ndarray,
    centers: np.ndarray | None = None,
) -> dict:
    """Account for residual energy and source jumps under literal ownership."""

    source = np.asarray(source_rgb, dtype=np.float64)
    owner = np.asarray(labels, dtype=np.int32)
    residual = np.maximum(
        np.asarray(residual_energy, dtype=np.float64), 0.0)
    if (
        source.ndim != 3
        or source.shape[2] != 3
        or owner.shape != source.shape[:2]
        or residual.shape != owner.shape
    ):
        raise ValueError("ownership diagnostics require one shared RGB geometry")

    assigned = owner >= 0
    cells = int(np.max(owner, initial=-1)) + 1
    safe_owner = np.where(assigned, owner, 0)
    cell_residual = np.bincount(
        safe_owner[assigned],
        weights=residual[assigned],
        minlength=cells,
    )
    cell_pixels = np.bincount(
        safe_owner[assigned], minlength=cells)
    cell_mean_residual = cell_residual / np.maximum(cell_pixels, 1)

    same_owner_jump = np.zeros(owner.shape, dtype=np.float64)
    interface_jump = np.zeros(owner.shape, dtype=np.float64)

    def accumulate(first_slice, second_slice):
        jump = np.mean(
            np.square(source[first_slice] - source[second_slice]), axis=2)
        valid = assigned[first_slice] & assigned[second_slice]
        same = valid & (owner[first_slice] == owner[second_slice])
        split = valid & ~same
        same_value = 0.5 * jump * same
        split_value = 0.5 * jump * split
        same_owner_jump[first_slice] += same_value
        same_owner_jump[second_slice] += same_value
        interface_jump[first_slice] += split_value
        interface_jump[second_slice] += split_value

    accumulate((slice(None), slice(0, -1)), (slice(None), slice(1, None)))
    accumulate((slice(0, -1), slice(None)), (slice(1, None), slice(None)))

    same_mass = float(np.sum(same_owner_jump))
    interface_mass = float(np.sum(interface_jump))
    jump_mass = same_mass + interface_mass
    result = {
        "assigned": assigned,
        "unowned_pixels": int(np.count_nonzero(~assigned)),
        "assigned_pixels": int(np.count_nonzero(assigned)),
        "cell_pixels": cell_pixels,
        "cell_residual_energy": cell_residual,
        "cell_mean_residual_energy": cell_mean_residual,
        "residual_energy_total": float(np.sum(residual)),
        "assigned_residual_energy": float(np.sum(residual[assigned])),
        "unowned_residual_energy": float(np.sum(residual[~assigned])),
        "same_owner_source_jump": same_owner_jump,
        "interface_source_jump": interface_jump,
        "same_owner_jump_mass": same_mass,
        "interface_jump_mass": interface_mass,
        "same_owner_jump_fraction": same_mass / max(jump_mass, 1e-30),
    }
    if centers is not None:
        sites = np.asarray(centers, dtype=np.float64)
        if sites.ndim != 2 or sites.shape[1] != 2:
            raise ValueError("diagnostic centers must have shape Nx2")
        height, width = owner.shape
        center_x = np.clip(
            sites[:, 0] * width - 0.5, 0.0, width - 1.0)
        center_y = np.clip(
            sites[:, 1] * height - 0.5, 0.0, height - 1.0)
        x0 = np.floor(center_x).astype(np.int32)
        y0 = np.floor(center_y).astype(np.int32)
        x1 = np.minimum(x0 + 1, width - 1)
        y1 = np.minimum(y0 + 1, height - 1)
        samples = np.stack((
            source[y0, x0],
            source[y0, x1],
            source[y1, x0],
            source[y1, x1],
        ), axis=1)
        germ_mean = np.mean(samples, axis=1, keepdims=True)
        germ_jump = np.mean(
            np.square(samples - germ_mean), axis=(1, 2))
        result["germ_source_jump"] = germ_jump
        germ_map = np.zeros(owner.shape, dtype=np.float64)
        germ_map[assigned] = germ_jump[owner[assigned]]
        result["germ_source_jump_map"] = germ_map
        result["germ_source_jump_mean"] = float(np.mean(germ_jump))
        result["germ_source_jump_max"] = float(np.max(
            germ_jump, initial=0.0))
    return result
