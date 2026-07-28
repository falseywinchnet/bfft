"""Checks for finite-D sparse photon flow primitives."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "experiments"))

from finite_d_poisson_flow import (  # noqa: E402
    finite_d_log_weights,
    local_charge_field,
    poisson_thin,
)


def test_poisson_thinning_is_complementary_and_reproducible():
    counts = np.arange(64, dtype=np.uint16).reshape(8, 8)

    first = poisson_thin(counts, 0.4, 123)
    second = poisson_thin(counts, 0.4, 123)

    np.testing.assert_array_equal(first[0] + first[1], counts)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_finite_d_kernel_converges_to_exponential_weights():
    loss = np.linspace(0.0, 8.0, 33)
    expected = finite_d_log_weights(loss, np.inf, 2.5)
    actual = finite_d_log_weights(loss, 1e9, 2.5)

    np.testing.assert_allclose(actual, expected, atol=1e-7)


def test_identical_image_charges_are_a_fixed_point():
    belief = np.arange(64, dtype=np.float64).reshape(8, 8) / 63.0
    charges = np.repeat(belief[None, ...], 4, axis=0)

    result, info = local_charge_field(
        belief,
        charges,
        exposure=2.0,
        dimension=64.0,
        radius=1.0,
        patch=4,
    )

    np.testing.assert_allclose(result, belief, atol=1e-12)
    assert abs(info["mean_effective_charges"] - 4.0) < 1e-12
