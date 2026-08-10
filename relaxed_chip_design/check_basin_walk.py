"""Deterministic checks for initial-chart basin probes."""

import numpy as np

from .basin_walk import fit_chart_displacement, spherical_chart_walk


def main() -> int:
    initial = np.asarray([[0.2, 0.3], [0.7, 0.8]], dtype=np.float64)
    target = np.asarray([[0.8, 0.6], [0.4, 0.1]], dtype=np.float64)
    np.testing.assert_allclose(spherical_chart_walk(initial, target, 0.0), initial)
    np.testing.assert_allclose(spherical_chart_walk(initial, target, 1.0), target)

    y_only = spherical_chart_walk(initial, target, 1.0, axes="y")
    np.testing.assert_allclose(y_only[:, 0], initial[:, 0])
    np.testing.assert_allclose(y_only[:, 1], target[:, 1])

    directions = np.column_stack((initial, np.ones(len(initial))))
    fit = fit_chart_displacement(initial, target, directions)
    np.testing.assert_allclose(fit.predicted_displacement, target - initial)
    np.testing.assert_allclose(fit.explained_energy_fraction, 1.0)
    print("initial-chart basin walk checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
