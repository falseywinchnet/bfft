import unittest

import numpy as np

from bfft_twiddle_trace import (
    _dif_prefixes,
    _dip_prefixes,
    _dit_prefixes,
    bruun_index,
    build_traces,
    direct_dft_lenses,
    direct_dft_phase,
    execution_order_traces,
    packed_real_dft_matrix,
    reverse_factor_cascade,
)


class TwiddleTraceTest(unittest.TestCase):
    def test_n512_shapes_and_level_counts(self):
        traces = build_traces(512)
        expected_widths = [1, 3, 7, 15, 31, 63, 127]
        for trace in traces:
            self.assertEqual(trace.support.shape, (512, 247))
            self.assertEqual(np.diff(trace.group_edges).tolist(), expected_widths)
            self.assertFalse(trace.support[0].any())
            self.assertFalse(trace.support[-1].any())
            self.assertTrue(trace.support[1:-1].all(axis=1).any() or trace.support[1:-1].any())

        executed = execution_order_traces(traces, 512)
        for logical, actual in zip(traces, executed):
            self.assertEqual(actual.support.shape, logical.support.shape)
            np.testing.assert_array_equal(
                np.sort(actual.angles_over_pi), np.sort(logical.angles_over_pi)
            )
            np.testing.assert_array_equal(
                np.sort(actual.support.sum(axis=0)), np.sort(logical.support.sum(axis=0))
            )
        self.assertEqual(np.count_nonzero(traces[1].support != traces[2].support), 0)
        self.assertGreater(np.count_nonzero(traces[0].support != traces[1].support), 0)

    def test_bruun_leaf_index_is_a_permutation(self):
        for n in (8, 16, 64, 512):
            logn = n.bit_length() - 1
            got = [bruun_index(node, logn) for node in range(1, n // 2)]
            self.assertEqual(sorted(got), list(range(1, n // 2)))

    def test_direct_lens_axes(self):
        phase = direct_dft_phase(512)
        self.assertEqual(phase.shape, (512, 512))
        np.testing.assert_array_equal(phase[0], 0.0)
        self.assertAlmostEqual(phase[1, 1], -2.0 / 512)
        real, imag = direct_dft_lenses(512)
        self.assertEqual(real.shape, (512, 512))
        self.assertEqual(imag.shape, (512, 512))
        np.testing.assert_allclose(real + 1j*imag, np.exp(
            -2j*np.pi*np.arange(512)[:, None]*np.arange(512)[None, :]/512
        ), atol=2e-13)

    def test_reverse_factor_cascades_reconstruct_packed_dft(self):
        n = 512
        final = packed_real_dft_matrix(n)
        for builder in (_dif_prefixes, _dit_prefixes, _dip_prefixes):
            prefixes = builder(n)
            cascade = reverse_factor_cascade(prefixes, n)
            self.assertEqual(cascade[0][0], "standard output")
            np.testing.assert_allclose(cascade[-1][1], final, atol=3e-11)

    def test_supported_power_of_two_sizes(self):
        for n in (8, 16, 32, 64, 128, 256, 1024):
            traces = build_traces(n)
            executed = execution_order_traces(traces, n)
            self.assertTrue(all(a.support.shape == b.support.shape for a, b in zip(traces, executed)))


if __name__ == "__main__":
    unittest.main()
