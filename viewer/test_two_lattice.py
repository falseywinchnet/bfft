"""Numerical checks for the Python two-lattice viewer specification."""
from __future__ import annotations

import numpy as np

from two_lattice import MagnitudeFamily, recover_ladder, recover_two_lattice


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


def test_cpp_combined_kernel_matches_python_specification():
    import iqwaterfall as iqw

    z = fixture()
    shared, _ = iqw.dip_run_complex(
        z, n_steps=1, nb=1024, ns=256)
    expected, _ = recover_two_lattice(
        z, 1024, 256, h_short=128, iterations=1, initial=shared)
    long_family = MagnitudeFamily(z, 1024, 128)
    long_corrected = long_family.project(expected, expected)
    expected += 0.75 * (long_corrected - expected)
    got, _ = iqw.dip_unified(
        z, nb=1024, ns=256, h_short=128, unified_steps=1)
    # Both paths already choose the same global phase. Keep the alignment here
    # so this remains an operator test if the phase-anchor implementation moves.
    phase = np.vdot(got, expected)
    if abs(phase):
        got = got * phase / abs(phase)
    relative = np.max(np.abs(got - expected)) / np.max(np.abs(expected))
    assert relative < 2e-12


def test_cpp_ladder_kernel_matches_python_specification():
    import iqwaterfall as iqw

    z = fixture()
    shared, _ = iqw.dip_run_complex(z, n_steps=1, nb=1024, ns=256)
    rungs = [(4096, 512), (1024, 128), (256, 128)]
    expected, _ = recover_ladder(z, rungs, iterations=1, initial=shared)
    got, _ = iqw.dip_unified_ladder(z, rungs, nb=1024, ns=256,
                                    unified_steps=1)
    phase = np.vdot(got, expected)
    if abs(phase):
        got = got * phase / abs(phase)
    relative = np.max(np.abs(got - expected)) / np.max(np.abs(expected))
    assert relative < 2e-12

    # The pair ABI must remain a special case of the ladder ABI.
    pair, _ = iqw.dip_unified(z, nb=1024, ns=256, h_short=128,
                              unified_steps=1)
    lad2, _ = iqw.dip_unified_ladder(z, [(1024, 128), (256, 128)],
                                     nb=1024, ns=256, unified_steps=1)
    phase = np.vdot(lad2, pair)
    if abs(phase):
        lad2 = lad2 * phase / abs(phase)
    assert np.max(np.abs(lad2 - pair)) / np.max(np.abs(pair)) < 2e-12


def test_cpp_momentum_guard_matches_python_beyond_one_step():
    import iqwaterfall as iqw

    z = fixture()
    shared, _ = iqw.dip_run_complex(z, n_steps=1, nb=1024, ns=256)
    # Deliberately explosive momentum: step two must fall back to latent in
    # both implementations instead of transporting a divergent trial.
    expected, _ = recover_two_lattice(
        z, 1024, 256, h_short=128, iterations=3, beta=1e6,
        initial=shared)
    long_family = MagnitudeFamily(z, 1024, 128)
    correction = long_family.project(expected, expected)
    expected += 0.75 * (correction - expected)
    got, _ = iqw.dip_unified(
        z, nb=1024, ns=256, h_short=128, unified_steps=3, beta=1e6)
    phase = np.vdot(got, expected)
    got *= phase / abs(phase)
    assert np.max(np.abs(got - expected)) / np.max(np.abs(expected)) < 2e-12

    rungs = [(2048, 256), (1024, 128), (256, 128)]
    expected_l, _ = recover_ladder(
        z, rungs, iterations=3, beta=1e6, initial=shared)
    got_l, _ = iqw.dip_unified_ladder(
        z, rungs, nb=1024, ns=256, unified_steps=3, beta=1e6)
    phase = np.vdot(got_l, expected_l)
    got_l *= phase / abs(phase)
    assert np.max(np.abs(got_l - expected_l)) / np.max(
        np.abs(expected_l)) < 2e-12


def test_unified_warm_seed_matches_existing_warm_solver():
    import iqwaterfall as iqw

    z = fixture(12288)
    tile0, tile1 = z[:8192], z[4096:12288]
    _u0, internal0, _ = iqw.dip_unified(
        tile0, nb=1024, ns=256, unified_steps=0,
        return_internal=True)
    shift = 4096 // (1024 // 8)
    warm = np.ascontiguousarray(internal0[:, shift:])
    expected, expected_internal, _ = iqw.dip_run_complex_warm(
        tile1, warm_internal=warm, nb=1024, ns=256, n_steps=1)
    got, got_internal, _ = iqw.dip_unified(
        tile1, nb=1024, ns=256, unified_steps=0,
        warm_internal=warm, return_internal=True)
    phase = np.vdot(got, expected)
    got *= phase / abs(phase)
    assert np.max(np.abs(got - expected)) / np.max(np.abs(expected)) < 2e-12
    assert np.max(np.abs(got_internal - expected_internal)) < 2e-12


def test_cpp_palindromic_finish_matches_python_specification():
    import iqwaterfall as iqw

    z = fixture()
    shared, _ = iqw.dip_run_complex(z, n_steps=1, nb=1024, ns=256)
    expected, _ = recover_two_lattice(
        z, 1024, 256, h_short=128, iterations=1, initial=shared,
        finish_mode="palindromic")
    got, _ = iqw.dip_unified(
        z, nb=1024, ns=256, h_short=128, unified_steps=1,
        finish_mode="palindromic")
    phase = np.vdot(got, expected)
    got *= phase / abs(phase)
    assert np.max(np.abs(got - expected)) / np.max(np.abs(expected)) < 2e-12

    rungs = [(2048, 256), (1024, 128), (256, 128)]
    expected_l, _ = recover_ladder(
        z, rungs, iterations=1, initial=shared,
        finish_mode="palindromic")
    got_l, _ = iqw.dip_unified_ladder(
        z, rungs, nb=1024, ns=256, unified_steps=1,
        finish_mode="palindromic")
    phase = np.vdot(got_l, expected_l)
    got_l *= phase / abs(phase)
    assert np.max(np.abs(got_l - expected_l)) / np.max(
        np.abs(expected_l)) < 2e-12


def test_cursor_anchored_zoom_keeps_cursor_frequency_fixed():
    from iq_waterfall_app import _anchored_zoom, _rung_gain_db

    for q in (0.0, 0.17, 0.5, 0.83, 1.0):
        lo, hi = -0.3, 0.4
        anchor = lo + q * (hi - lo)
        zi = _anchored_zoom(lo, hi, q, 2.0)
        assert abs((zi[0] + q * (zi[1] - zi[0])) - anchor) < 1e-14
        zo = _anchored_zoom(*zi, q, -2.0)
        assert abs(zo[0] - lo) < 1e-14
        assert abs(zo[1] - hi) < 1e-14
    # Exact symmetric-Hann coherent gain, not the asymptotic N ratio.
    assert abs(_rung_gain_db(2048, 1024) -
               20 * np.log10(2047 / 1023)) < 1e-14


def test_seed_only_unified_path_is_the_shared_native_seed():
    import iqwaterfall as iqw

    z = fixture()
    seed, _ = iqw.dip_run_complex(z, n_steps=1, nb=1024, ns=256)
    seed_only, _ = iqw.dip_unified(
        z, nb=1024, ns=256, unified_steps=0)
    phase = np.vdot(seed_only, seed)
    seed_only *= phase / abs(phase)
    assert np.max(np.abs(seed_only - seed)) / np.max(np.abs(seed)) < 2e-12


def test_stream_close_is_a_native_read_barrier():
    import threading
    import time
    from two_lattice import TwoLatticeStream

    class SlowSource:
        num_samples = 32768
        is_complex = True

        def __init__(self):
            self.lock = threading.Lock()
            self.active = 0
            self.max_active = 0

        def read(self, _start, count):
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            time.sleep(0.01)
            with self.lock:
                self.active -= 1
            return np.zeros(count, np.complex128)

    src = SlowSource()
    first = TwoLatticeStream(src, n_long=1024, n_short=256, iterations=1)
    errors = []

    def request_while_switching():
        try:
            for k in range(32):
                first.request((k % 8) * 4096, 4096)
        except Exception as exc:  # the former shutdown/submit RuntimeError
            errors.append(exc)

    requester = threading.Thread(target=request_while_switching)
    requester.start()
    first.close()
    requester.join()
    second = TwoLatticeStream(
        src, n_long=1024, n_short=256, iterations=1, rung_mult=2)
    second.request(0, 4096)
    second.close()
    assert not errors
    assert src.max_active == 1


def test_bilinear_reassignment_conserves_power_and_fills_quantization_gaps():
    import iqwaterfall as iqw

    z = fixture()
    iq = np.empty(2 * len(z), dtype=np.float32)
    iq[0::2] = z.real
    iq[1::2] = z.imag
    engine = iqw.Reassign(1024)
    nearest_db = engine.render_mem(iq, 512, 256, 29).copy()
    engine.set_bilinear(True)
    splat_db = engine.render_mem(iq, 512, 256, 29).copy()
    engine.close()
    nearest = np.power(10.0, nearest_db / 10.0)
    splat = np.power(10.0, splat_db / 10.0)
    assert abs(float(splat.sum() / nearest.sum()) - 1.0) < 2e-6
    assert np.count_nonzero(splat_db > -80.0) > np.count_nonzero(
        nearest_db > -80.0)


if __name__ == "__main__":
    test_independent_1024_512_lattices()
    test_alternating_projection_converges_for_1024_512()
    test_real_reference_stays_real()
    test_cpp_combined_kernel_matches_python_specification()
    test_cpp_ladder_kernel_matches_python_specification()
    test_cpp_momentum_guard_matches_python_beyond_one_step()
    test_unified_warm_seed_matches_existing_warm_solver()
    test_cpp_palindromic_finish_matches_python_specification()
    test_cursor_anchored_zoom_keeps_cursor_frequency_fixed()
    test_seed_only_unified_path_is_the_shared_native_seed()
    test_stream_close_is_a_native_read_barrier()
    test_bilinear_reassignment_conserves_power_and_fills_quantization_gaps()
    print("two-lattice reference: PASS")
