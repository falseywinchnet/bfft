"""Deterministic checks for connection-valued orientation diffusion."""

import numpy as np

from .vector_diffusion import diffuse_connection_orientations


def main() -> None:
    directions = np.asarray([
        [0.0, 1.0],
        [0.0, 2.0],
        [1.0, 0.0],
    ])
    radii = np.asarray([1.0, 3.0, 2.0])
    confidence = np.asarray([1.0, 1.0, 0.0])
    flux, info = diffuse_connection_orientations(
        directions,
        radii,
        confidence,
        [np.asarray([0, 1, 2])],
        np.asarray([4]),
    )
    assert np.allclose(flux[0], [0.0, 1.0])
    assert np.allclose(flux[1], [0.0, 3.0])
    assert np.allclose(flux[2], [0.0, 4.0 / 3.0])
    assert info["graph_propagated_fraction"] == 1.0 / 3.0
    assert info["redirected_fraction"] == 1.0 / 3.0
    assert not info["candidate_destinations_materialized"]

    isolated, isolated_info = diffuse_connection_orientations(
        directions,
        radii,
        confidence,
        [],
        np.asarray([], dtype=np.int64),
    )
    assert np.allclose(isolated[0], [0.0, 1.0])
    assert np.allclose(isolated[1], [0.0, 3.0])
    assert np.allclose(isolated[2], [0.0, 0.0])
    assert isolated_info["graph_propagated_fraction"] == 0.0
    print("connection-valued vector diffusion checks: PASS")


if __name__ == "__main__":
    main()
