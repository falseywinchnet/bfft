import itertools
import random
import unittest

import numpy as np

from twos import (ShiftPlan, compact_forward, compact_inverse, compact_bin,
                  polynomial_bin, diagnostic_complex, pack_orbits,
                  unpack_orbits, phase_shift_orbits, orbit_lane_bits,
                  pack_samples, fold_packed)


class TwosTests(unittest.TestCase):
    def test_modular_arithmetic_including_negative_shifts(self):
        rng = random.Random(57)
        for n in (2, 4, 8, 16, 32):
            p = ShiftPlan(n, 3)
            for _ in range(100):
                x = rng.randrange(-(p.modulus ** 3), p.modulus ** 3)
                self.assertEqual(p.reduce(x), x % p.modulus)
                s = rng.randrange(-4 * p.width, 4 * p.width)
                self.assertEqual(p.shift(x, s), x * pow(2, s, p.modulus) % p.modulus)

    def test_principal_root_even_for_composite_modulus(self):
        # A prime modulus is unnecessary. N=8,b=3 gives 4097=17*241.
        for n, b in ((8, 3), (16, 3), (32, 5)):
            p = ShiftPlan(n, 0, b)
            for k in range(1, n):
                self.assertEqual(sum(pow(2, b * k * j, p.modulus)
                                     for j in range(n)) % p.modulus, 0)

    def test_roundtrip_arbitrary_ring_elements(self):
        rng = random.Random(68)
        for n in (2, 4, 8, 16, 32, 64):
            p = ShiftPlan(n, 1)
            for _ in range(5):
                x = [rng.randrange(p.modulus) for _ in range(n)]
                self.assertEqual(p.fft(p.fft(x), inverse=True), x)

    def check_signal(self, x):
        n = len(x)
        p = ShiftPlan(n, max(map(abs, x)))
        encoded = p.forward(x)
        packets = compact_forward(x)
        self.assertEqual(sum(map(len, packets.values())), n)
        self.assertEqual(compact_inverse(packets), x)
        expected = np.fft.fft(x)
        for k, word in enumerate(encoded):
            coeffs = p.coefficients(word)
            self.assertEqual(coeffs, polynomial_bin(x, k))
            self.assertEqual(coeffs, compact_bin(packets, n, k))
            actual = diagnostic_complex(coeffs, n)
            self.assertLessEqual(abs(actual - expected[k]),
                                 1e-10 * max(1, sum(map(abs, x))))
        self.assertEqual(p.fft(encoded, inverse=True), [p.reduce(v) for v in x])

    def test_basis_and_extreme_signals(self):
        for n in (2, 4, 8, 16, 32, 64):
            self.check_signal([0] * n)
            self.check_signal([32768] * n)
            self.check_signal([-32768] * n)
            self.check_signal([32768 if j & 1 else -32768 for j in range(n)])
            for j in range(n):
                x = [0] * n
                x[j] = 1
                self.check_signal(x)

    def test_every_eight_sample_sign_pattern(self):
        for x in itertools.product((-1, 1), repeat=8):
            self.check_signal(list(x))

    def test_random_ranges(self):
        rng = random.Random(1971)
        for n in (8, 16, 32, 64, 128):
            for bound in (1, 255, 32768, 1 << 60):
                self.check_signal([rng.randint(-bound, bound) for _ in range(n)])

    def test_compact_phase_is_exact_time_shift(self):
        rng = random.Random(90)
        for n in (2, 4, 8, 16, 32, 64, 128):
            x = [rng.randrange(-32768, 32769) for _ in range(n)]
            p = ShiftPlan(n, 32768)
            packets = compact_forward(x)
            packed = pack_orbits(packets, p.lane_bits)
            self.assertEqual(unpack_orbits(packed, p.lane_bits), packets)
            for delay in (-n - 1, -3, 0, 1, 7, n + 3):
                shifted = unpack_orbits(phase_shift_orbits(packed, p.lane_bits, delay),
                                        p.lane_bits)
                expected = [x[(j - delay) % n] for j in range(n)]
                self.assertEqual(shifted, compact_forward(expected))
                self.assertEqual(compact_inverse(shifted), expected)

    def test_guards_and_invalid_inputs(self):
        for n in (0, 1, 3, 12):
            with self.assertRaises(ValueError):
                ShiftPlan(n, 1)
        with self.assertRaises(ValueError):
            ShiftPlan(8, 32768, 8)
        p = ShiftPlan(8, 1)
        for values in ([2] * 8, [1.0] * 8, [True] * 8, [0] * 4):
            with self.assertRaises(ValueError):
                p.forward(values)
        with self.assertRaises(ValueError):
            compact_inverse({1: [0], 2: [1]})

    def test_tighter_orbit_guards_at_extremes(self):
        rng = random.Random(12)
        for n in (2, 4, 8, 16, 32, 64, 128):
            for bound in (0, 1, 32768):
                bits = orbit_lane_bits(n, bound)
                for x in ([bound] * n, [-bound] * n,
                          [rng.choice((-bound, bound)) for _ in range(n)]):
                    packets = compact_forward(x)
                    packed = pack_orbits(packets, bits)
                    self.assertEqual(unpack_orbits(packed, bits), packets)
                    shifted = phase_shift_orbits(packed, bits, -3)
                    self.assertEqual(compact_inverse(unpack_orbits(shifted, bits)),
                                     [x[(j + 3) % n] for j in range(n)])

    def test_packed_folding_without_coefficient_unpacking(self):
        rng = random.Random(27)
        for n in (2, 4, 8, 16, 32, 64, 128, 1024):
            for bound in (0, 1, 255, 32768, 1 << 60):
                plan = ShiftPlan(n, bound)
                for x in ([bound] * n, [-bound] * n,
                          [rng.randint(-bound, bound) for _ in range(n)]):
                    actual = fold_packed(pack_samples(x, plan), plan)
                    self.assertEqual(actual, pack_orbits(compact_forward(x), plan.lane_bits))
                    self.assertEqual(compact_inverse(unpack_orbits(actual, plan.lane_bits)), x)

    def test_small_modulus_does_not_preserve_individual_complex_bins(self):
        # These distinct exact X[1] polynomials collide when evaluated at B=2.
        x = [1, 0, 0, 0, -1, 0, 0, 0]  # X[1] = 2
        y = [0, 1, 0, 0, 0, 0, 0, 0]   # X[1] = zeta
        unsafe = ShiftPlan(8, 0, 1)
        self.assertEqual(unsafe.fft(x)[1], unsafe.fft(y)[1])
        self.assertNotEqual(polynomial_bin(x, 1), polynomial_bin(y, 1))
        safe = ShiftPlan(8, 1)
        self.assertNotEqual(safe.forward(x)[1], safe.forward(y)[1])


if __name__ == '__main__':
    unittest.main()
