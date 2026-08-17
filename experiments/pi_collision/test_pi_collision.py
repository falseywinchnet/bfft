import math
import unittest

import mpmath as mp

from pi_collision import (
    closed_weights,
    pi_collision_fused,
    pi_collision_triangle,
    pi_collision_weighted,
    tangent_multiple,
)


class PiCollisionTests(unittest.TestCase):
    def test_tangent_multiple(self):
        mp.mp.dps = 80
        for m in (2, 3, 4, 5, 8, 17):
            x = mp.mpf("0.013")
            self.assertAlmostEqual(
                float(tangent_multiple(x, m)),
                float(mp.tan(m * mp.atan(x))),
                places=15,
            )

    def test_weights_annihilate_powers(self):
        mp.mp.dps = 80
        m, depth = 3, 8
        r = mp.mpf(1) / (m * m)
        weights = closed_weights(m, depth)
        self.assertLess(abs(mp.fsum(weights) - 1), mp.mpf("1e-70"))
        for power in range(1, depth + 1):
            moment = mp.fsum(weights[j] * r ** (j * power) for j in range(depth + 1))
            self.assertLess(abs(moment), mp.mpf("1e-70"))

    def test_three_forms_agree(self):
        mp.mp.dps = 100
        m, depth, dps = 2, 10, 90
        triangle = pi_collision_triangle(m, depth, dps)
        weighted = pi_collision_weighted(m, depth, dps)
        fused = pi_collision_fused(m, depth, dps)
        self.assertLess(abs(triangle - weighted), mp.mpf("1e-75"))
        self.assertLess(abs(triangle - fused), mp.mpf("1e-75"))

    def test_arbitrary_radix(self):
        for m in (2, 3, 5, 8):
            value = pi_collision_fused(m, 12, 100)
            self.assertGreater(-mp.log(abs(value - mp.pi), 2), 100)


if __name__ == "__main__":
    unittest.main()
