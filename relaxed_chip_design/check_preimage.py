"""Deterministic checks for transport-residual pullback."""

import numpy as np

from .preimage import project_transport_residual, pullback_residual


def main() -> int:
    target = np.asarray([[2.0, 3.0]], dtype=np.float64)
    base = np.zeros_like(target)
    trial = np.asarray([[1.0, 0.0]], dtype=np.float64)
    secant = project_transport_residual(target, base, trial)
    np.testing.assert_allclose(secant.optimal_coefficient, 2.0)
    np.testing.assert_allclose(secant.visible_residual, [[2.0, 0.0]])
    np.testing.assert_allclose(secant.orthogonal_residual, [[0.0, 3.0]])
    np.testing.assert_allclose(secant.visible_energy_fraction, 4.0 / 13.0)
    np.testing.assert_allclose(secant.orthogonal_energy_fraction, 9.0 / 13.0)

    initial = np.asarray([[1.0, 1.0], [3.0, 3.0]])
    fixed = np.asarray([[4.0, 8.0], [8.0, 1.0]])
    observed = np.asarray([[2.0, 2.0], [2.0, 2.0]])
    pulled = pullback_residual(
        initial,
        fixed,
        observed,
        0.5,
        axes="y",
        maximum_step=np.asarray([10.0, 2.0]),
    )
    np.testing.assert_allclose(pulled, [[1.0, 3.0], [3.0, 2.5]])
    print("transport preimage checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
