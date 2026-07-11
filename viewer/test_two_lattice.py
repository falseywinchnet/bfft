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
    test_bilinear_reassignment_conserves_power_and_fills_quantization_gaps()
    print("two-lattice reference: PASS")
