import numpy as np

from port_needed.ownership_diagnostics import (
    residual_ownership_diagnostics,
)


def test_distinguishes_unowned_pixels_from_jumps_inside_one_cell():
    source = np.zeros((2, 4, 3), dtype=np.float64)
    source[:, 2:] = 1.0
    labels = np.array([
        [0, 0, 0, 1],
        [0, 0, 0, 1],
    ], dtype=np.int32)
    residual = np.arange(8, dtype=np.float64).reshape(2, 4)

    diagnostic = residual_ownership_diagnostics(
        source,
        labels,
        residual,
        centers=np.array([[0.5, 0.5], [0.875, 0.5]]),
    )

    assert diagnostic["unowned_pixels"] == 0
    assert diagnostic["unowned_residual_energy"] == 0.0
    assert diagnostic["assigned_residual_energy"] == np.sum(residual)
    assert diagnostic["same_owner_jump_mass"] == 2.0
    assert diagnostic["interface_jump_mass"] == 0.0
    assert diagnostic["same_owner_jump_fraction"] == 1.0
    np.testing.assert_allclose(
        diagnostic["cell_residual_energy"],
        [np.sum(residual[:, :3]), np.sum(residual[:, 3:])],
    )
    assert diagnostic["germ_source_jump"].shape == (2,)
    assert diagnostic["germ_source_jump_map"].shape == labels.shape
