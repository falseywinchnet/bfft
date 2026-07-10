"""Numerical checks for the Python two-lattice viewer specification."""
from __future__ import annotations

import numpy as np

from two_lattice import MagnitudeFamily, recover_two_lattice


def fixture(length=8192):
    rng = np.random.default_rng(19)
    t = np.arange(length, dtype=np.float64)
    z = (np.exp(2j * np.pi * (0.071 * t + 1.1e-6 * t * t)) +
         0.25 * np.exp(2j * np.pi * 0.193 * t))
    z += 0.02 * (rng.standard_normal(length) +
                 1j * rng.standard_normal(length))
    return z


def test_independent_1024_512_lattices():
    z = fixture()
    long_family = MagnitudeFamily(z, 1024, 128)
    short_family = MagnitudeFamily(z, 512, 32)
    assert np.array_equal(long_family.offsets[:4], [0, 128, 256, 384])
    assert np.array_equal(short_family.offsets[:4], [0, 32, 64, 96])
    assert long_family.indices[-1, -1] < len(z)
    assert short_family.indices[-1, -1] < len(z)


def test_alternating_projection_converges_for_1024_512():
    z = fixture()
    _seed, before = recover_two_lattice(
        z, 1024, 512, iterations=0, seed=7)
    recovered, after = recover_two_lattice(
        z, 1024, 512, iterations=80, seed=7)
    assert np.all(np.isfinite(recovered))
    assert after["long_relative_error"] < 0.02
    assert after["short_relative_error"] < 0.02
    assert after["long_relative_error"] < before["long_relative_error"]
    assert after["short_relative_error"] < before["short_relative_error"]


def test_real_reference_stays_real():
    x = fixture().real
    recovered, _metrics = recover_two_lattice(
        x, 1024, 128, iterations=12, seed=5, real_signal=True)
    assert np.isrealobj(recovered)
    assert np.all(np.isfinite(recovered))


if __name__ == "__main__":
    test_independent_1024_512_lattices()
    test_alternating_projection_converges_for_1024_512()
    test_real_reference_stays_real()
    print("two-lattice reference: PASS")
