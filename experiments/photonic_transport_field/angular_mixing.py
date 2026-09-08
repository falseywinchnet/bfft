"""Direct angular-collision diagnostic, not a replacement scene renderer.

The homogeneous HG collision operator is diagonal in orthonormal spherical
harmonics. Spatial streaming, visibility, and surface reflection are excluded.
No engine roughness value is mapped to HG's g.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import unittest

import numpy as np


def collision_factors(degrees, distance, absorption, scattering, g):
    """Exact homogeneous angular-collision semigroup, per harmonic degree."""
    degrees = np.asarray(degrees)
    if (np.any(degrees < 0) or np.any(degrees != np.floor(degrees))
            or min(distance, absorption, scattering) < 0 or not -1 <= g <= 1):
        raise ValueError("invalid angular collision parameters")
    return np.exp(-distance * (absorption + scattering * (1 - g ** degrees)))


def discarded_observer_bound(light_coefficients, observer_coefficients, omitted):
    """Cauchy-Schwarz certificate in a common orthonormal angular basis.

    A finite coefficient vector certifies only its represented modes. Any
    unrepresented tail needs its own bound; a point-direction delta observer
    is not an L2 observer and cannot use this certificate directly.
    """
    light = np.asarray(light_coefficients)[omitted]
    observer = np.asarray(observer_coefficients)[omitted]
    return float(np.linalg.norm(light) * np.linalg.norm(observer))


class AngularMixingChecks(unittest.TestCase):
    def test_independent_phase_function_quadrature(self):
        mu, weights = np.polynomial.legendre.leggauss(512)
        basis = np.polynomial.legendre.legvander(mu, 12)
        for g in (0., .5, .9):
            # Integral over azimuth absorbs 2*pi of the normalized 3-D HG law.
            density = (1-g*g) / (2 * (1+g*g-2*g*mu) ** 1.5)
            moments = basis.T @ (weights * density)
            np.testing.assert_allclose(moments, g ** np.arange(13), atol=2e-11, rtol=2e-11)

    def test_scattering_preserves_zeroth_mode(self):
        factors = collision_factors(np.arange(9), 17., .0, 2., .6)
        self.assertEqual(factors[0], 1.)
        self.assertLess(factors[1], 2e-6)
        absorbed = collision_factors(np.arange(9), 17., .03, 2., .6)
        self.assertAlmostEqual(absorbed[0], np.exp(-.51))

    def test_semigroup_and_direction_preserving_limit(self):
        degrees = np.arange(9)
        whole = collision_factors(degrees, 7., .02, 1.3, .7)
        pieces = collision_factors(degrees, 3., .02, 1.3, .7) * collision_factors(degrees, 4., .02, 1.3, .7)
        np.testing.assert_allclose(whole, pieces, atol=1e-15)
        np.testing.assert_allclose(collision_factors(degrees, 100., 0., 1., 1.), np.ones(9))
        np.testing.assert_allclose(collision_factors(degrees, 100., 0., 0., .3), np.ones(9))

    def test_observer_bound(self):
        light = np.array([3., -.2, .04, -.01])
        observer = np.array([1., .3, -.5, .1])
        omitted = np.array([False, True, True, True])
        error = abs(float(light[omitted] @ observer[omitted]))
        self.assertLessEqual(error, discarded_observer_bound(light, observer, omitted))
        # A small coefficient can matter when observer gain is large.
        self.assertGreater(discarded_observer_bound(light, observer * 1e4, omitted), 1.)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for g in (0., .5, .9, .99, 1.):
        for depth in (1, 4, 8, 16):
            rows.append(dict(g=g, events=depth, survival=.95**depth,
                relative_dipole=g**depth, relative_degree8=g**(8*depth)))
    result = dict(scope="Controlled homogeneous HG angular mixing; not an engine material model",
        discrete_events=rows,
        conservative_medium_at_optical_depth8=dict(
            unscattered_beam=float(np.exp(-8)),
            intensity=float(collision_factors([0], 8., 0., 1., .5)[0]),
            dipole=float(collision_factors([1], 8., 0., 1., .5)[0])))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
