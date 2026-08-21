from __future__ import annotations

import unittest

import numpy as np

from .multisheet_transport import solve_multisheet_consensus
from .spatial_transport import SpatialExposureField, SpatialReflectedExposureOperator


def _deterministic_field(
    name: str,
    shape: tuple[int, int],
    displacement_xy: tuple[float, float],
) -> SpatialExposureField:
    displacement = np.empty((1, *shape, 2), dtype=np.float64)
    displacement[0, ..., 0] = displacement_xy[0]
    displacement[0, ..., 1] = displacement_xy[1]
    return SpatialExposureField(
        name=name,
        displacements_xy=displacement,
        weights=np.ones((1, *shape), dtype=np.float64),
    )


class MultiSheetTransportTests(unittest.TestCase):
    def test_distinct_appearances_recover_under_known_soft_ownership(self) -> None:
        size = 48
        yy, xx = np.mgrid[:size, :size]
        background = np.clip(
            0.18 + 0.55 * xx / size + 0.12 * np.sin(yy * 0.43), 0.0, 1.0)
        foreground = np.clip(
            0.82 - 0.42 * yy / size + 0.14 * np.cos(xx * 0.71), 0.0, 1.0)
        radius = ((xx - 24.0) / 10.0) ** 2 + ((yy - 24.0) / 13.0) ** 2
        alpha = np.clip(3.0 * (1.0 - radius), 0.0, 1.0)
        identity = _deterministic_field("stationary", (size, size), (0.0, 0.0))
        moving = [
            _deterministic_field("moving_left", (size, size), (-3.0, 0.0)),
            _deterministic_field("moving_right", (size, size), (3.0, 0.0)),
        ]
        observations = []
        ownership = []
        for field in moving:
            operator = SpatialReflectedExposureOperator(field)
            moved_alpha = operator.forward(alpha)
            observations.append(
                (1.0 - moved_alpha) * background
                + moved_alpha * operator.forward(foreground))
            ownership.append(np.stack((1.0 - moved_alpha, moved_alpha), axis=0))
        ownership_array = np.stack(ownership, axis=0)
        reference = np.stack((1.0 - alpha, alpha), axis=0)
        fields = ((identity, moving[0]), (identity, moving[1]))
        result = solve_multisheet_consensus(
            observations,
            fields,
            sensor_ownership=ownership_array,
            reference_ownership=reference,
            passes=80,
        )
        truth = (1.0 - alpha) * background + alpha * foreground
        average = np.mean(observations, axis=0)
        result_mse = float(np.mean((result.image - truth) ** 2))
        average_mse = float(np.mean((average - truth) ** 2))
        self.assertLess(result_mse, 0.25 * average_mse)
        self.assertEqual(
            result.diagnostics["method"],
            "permutation_symmetric_positive_multisheet_transport",
        )
        self.assertTrue(np.all(
            np.diff(result.diagnostics["residual_trace"]) <= 1e-12))
        self.assertNotIn("selected", repr(result.diagnostics).lower())

        permuted = solve_multisheet_consensus(
            observations,
            tuple(tuple(reversed(frame)) for frame in fields),
            sensor_ownership=ownership_array[:, ::-1],
            reference_ownership=reference[::-1],
            passes=80,
        )
        np.testing.assert_allclose(permuted.image, result.image, atol=1e-12)
        np.testing.assert_allclose(
            permuted.sheet_images[::-1], result.sheet_images, atol=1e-12)

    def test_ownership_is_a_positive_simplex(self) -> None:
        shape = (12, 13)
        identity = _deterministic_field("identity", shape, (0.0, 0.0))
        image = np.linspace(0.1, 0.9, shape[0] * shape[1]).reshape(shape)
        supplied = np.stack((
            np.full(shape, 2.0),
            np.full(shape, 3.0),
        ), axis=0)
        result = solve_multisheet_consensus(
            (image, image),
            ((identity, identity), (identity, identity)),
            sensor_ownership=np.stack((supplied, supplied), axis=0),
            passes=2,
        )
        self.assertTrue(np.all(result.sensor_ownership >= 0.0))
        np.testing.assert_allclose(
            np.sum(result.sensor_ownership, axis=1), 1.0, atol=1e-12)
        np.testing.assert_allclose(result.image, image, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
