import numpy as np

import bfft
from experiments.meyer_first_pass_conditioning import (
    checker_support_scene,
    first_split,
    gates,
    lap_hat,
    predicted_reflection,
    screened,
    tsv_one_forward,
)
from experiments.meyer_preconditioning_research import junction_texture_scene
from experiments.meyer_transverse_route_research import (
    estimate_jump_measure,
    jump_texture_components,
    native_structural_gate,
    paired_one_sided_trace,
    proposed_texture,
    tangent_reservoir_route,
)
from experiments.meyer_tsv_validation import (
    multiscale_crossing_scene,
    score_split,
    symmetric_support_scene,
    tsv_four_direction,
)


def test_python_first_pass_is_the_native_uniform_equation():
    scene = symmetric_support_scene(64)
    source = scene["source"]
    expected = bfft.meyer_split_legacy(
        source, lam=0.05, mu=40.0, passes=1, threads=1, solver=0
    )
    actual = first_split(source, None, lam=0.05, mu=40.0)
    np.testing.assert_allclose(actual[0], expected[0], atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(actual[1], expected[1], atol=2e-12, rtol=2e-12)


def test_one_forward_tsv_is_the_same_operator():
    source = multiscale_crossing_scene(64)["source"]
    expected = tsv_four_direction(
        source, sigma_long=12.0, sigma_width=0.75, radius=12
    )
    np.testing.assert_allclose(tsv_one_forward(source), expected, atol=2e-12)


def test_fixed_tsv_conditioning_reduces_known_truth_first_pass_error():
    for scene in (
        symmetric_support_scene(128),
        checker_support_scene(128),
    ):
        source = scene["source"]
        baseline = first_split(source, None, lam=0.05, mu=40.0)
        gate = gates(source)["tsv_tail"]
        rx, ry = predicted_reflection(source, eta=0.10)
        conditioned = first_split(
            source,
            (1.5 * gate * rx, 1.5 * gate * ry),
            lam=0.05,
            mu=40.0,
        )
        before = score_split(*baseline, scene)
        after = score_split(*conditioned, scene)
        assert after["cartoon_relative_rms_error"] < before[
            "cartoon_relative_rms_error"
        ]
        assert after["texture_relative_rms_error"] < before[
            "texture_relative_rms_error"
        ]
        assert after["contour_excess_texture_rms"] < 0.65 * before[
            "contour_excess_texture_rms"
        ]


def _analytic_tsv(source):
    height, width = source.shape
    wy = 2.0 * np.pi * np.fft.fftfreq(height)[:, None]
    wx = 2.0 * np.pi * np.fft.rfftfreq(width)[None, :]
    source_spectrum = np.fft.rfft2(source)
    total = np.zeros_like(source)
    for dy, dx, theta in (
        (1, 0, 0.0),
        (0, 1, np.pi / 2.0),
        (1, 1, np.pi / 4.0),
        (1, -1, 3.0 * np.pi / 4.0),
    ):
        cosine, sine = np.cos(theta), np.sin(theta)
        along = wx * cosine + wy * sine
        across = -wx * sine + wy * cosine
        gaussian = np.exp(-0.5 * (12.0 * along * along + 0.75 * across * across))
        difference = np.exp(1j * (wx * dx + wy * dy)) - 1.0
        total += np.abs(np.fft.irfft2(
            source_spectrum * difference * gaussian, s=source.shape
        ))
    return total


def test_native_conditioner_matches_independent_spectral_reference():
    scene = symmetric_support_scene(64)
    source = scene["source"]
    tsv = _analytic_tsv(source)
    ratio = tsv / max(1.6 * float(np.mean(tsv)), 1e-12)
    gate = (ratio * ratio / (1.0 + ratio * ratio)) ** 6
    rx, ry = predicted_reflection(source, eta=0.10)
    expected = first_split(
        source, (1.5 * gate * rx, 1.5 * gate * ry), lam=0.05, mu=40.0
    )
    plan = bfft.MeyerPlan(source.shape, passes=1, threads=1, solver=0)
    actual = plan.split_conditioned_first(source, strength=1.5)
    np.testing.assert_allclose(actual[0], expected[0], atol=2e-4, rtol=2e-7)
    np.testing.assert_allclose(actual[1], expected[1], atol=2e-4, rtol=2e-7)


def test_zero_native_conditioning_is_exactly_ordinary_pass_one():
    source = checker_support_scene(64)["source"]
    plan = bfft.MeyerPlan(source.shape, passes=1, threads=1, solver=0)
    expected = plan.split_legacy(source)
    actual = plan.split_conditioned_first(source, strength=0.0)
    np.testing.assert_allclose(actual[0], expected[0], atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(actual[1], expected[1], atol=2e-12, rtol=2e-12)


def test_native_conditioner_is_thread_count_identical():
    source = multiscale_crossing_scene(64)["source"]
    single = bfft.MeyerPlan(
        source.shape, passes=1, threads=1, solver=0
    ).split_conditioned_first(source)
    parallel = bfft.MeyerPlan(
        source.shape, passes=1, threads=4, solver=0
    ).split_conditioned_first(source)
    np.testing.assert_array_equal(single[0], parallel[0])
    np.testing.assert_array_equal(single[1], parallel[1])


def test_arbitrary_size_conditioned_front_end_crops_to_source():
    source = symmetric_support_scene(128)["source"][:93, :117]
    cartoon, texture = bfft.meyer_split_conditioned_first(source, threads=2)
    assert cartoon.shape == source.shape
    assert texture.shape == source.shape
    assert np.isfinite(cartoon).all()
    assert np.isfinite(texture).all()


def test_native_preconditioner_matches_independent_spectral_reference():
    source = symmetric_support_scene(64)["source"]
    tsv = _analytic_tsv(source)
    ratio = tsv / max(1.6 * float(np.mean(tsv)), 1e-12)
    gate = (ratio * ratio / (1.0 + ratio * ratio)) ** 6
    transfer = 0.05 / (0.05 - 0.10 * lap_hat(source.shape))
    virtual_cartoon = np.fft.ifft2(
        np.fft.fft2(source) * transfer ** 8
    ).real
    target = source - (1.0 - gate) ** 8 * (source - virtual_cartoon)
    rx, ry = predicted_reflection(target, eta=0.10)
    proposed_cartoon = screened(
        target, 0.05, 0.10, (1.5 * gate * rx, 1.5 * gate * ry)
    )
    expected_texture, diagnostic = tangent_reservoir_route(
        source - proposed_cartoon, gate, radius=40.0
    )
    expected = (source - expected_texture, expected_texture)
    assert diagnostic["postprojection_maximum"] <= 40.0 * (1.0 + 1e-12)
    assert diagnostic["divergence_change_of_route"] < 1e-9

    actual = bfft.MeyerPlan(
        source.shape, passes=1, threads=1, solver=0
    ).split_preconditioned(source)
    np.testing.assert_allclose(actual[0], expected[0], atol=3e-4, rtol=3e-7)
    np.testing.assert_allclose(actual[1], expected[1], atol=3e-4, rtol=3e-7)
    np.testing.assert_allclose(actual[0] + actual[1], source, atol=3e-12)


def test_native_preconditioner_is_thread_count_identical():
    source = multiscale_crossing_scene(64)["source"]
    single = bfft.MeyerPlan(
        source.shape, passes=1, threads=1, solver=0
    ).split_preconditioned(source)
    parallel = bfft.MeyerPlan(
        source.shape, passes=1, threads=4, solver=0
    ).split_preconditioned(source)
    np.testing.assert_array_equal(single[0], parallel[0])
    np.testing.assert_array_equal(single[1], parallel[1])


def test_arbitrary_size_preconditioned_front_end_crops_to_source():
    source = symmetric_support_scene(128)["source"][:93, :117]
    cartoon, texture = bfft.meyer_split_preconditioned(source, threads=2)
    assert cartoon.shape == source.shape
    assert texture.shape == source.shape
    assert np.isfinite(cartoon).all()
    assert np.isfinite(texture).all()
    np.testing.assert_allclose(cartoon + texture, source, atol=3e-12)


def test_transverse_route_preserves_divergence_and_disk_feasibility():
    scene = multiscale_crossing_scene(64)
    proposed, gate = proposed_texture(
        scene["source"], lam=0.05, strength=1.5,
        virtual_passes=8, gate_power=8,
    )
    texture, diagnostic = tangent_reservoir_route(
        proposed, gate, radius=40.0
    )
    assert np.isfinite(texture).all()
    assert diagnostic["divergence_change_of_route"] < 1e-9
    assert diagnostic["postprojection_maximum"] <= 40.0 * (1.0 + 1e-12)
    assert diagnostic["overload_energy_ratio"] <= 1.0


def test_transverse_route_reduces_junction_disk_readout_loss():
    scene = junction_texture_scene(128)
    proposed, gate = proposed_texture(
        scene["source"], lam=0.05, strength=1.5,
        virtual_passes=8, gate_power=8,
    )
    _texture, diagnostic = tangent_reservoir_route(
        proposed, gate, radius=40.0
    )
    assert diagnostic["disk_readout_loss_ratio"] < 0.90


def test_jump_measure_candidate_improves_two_product_texture_truth_error():
    scene = multiscale_crossing_scene(128)
    source = scene["source"]
    gate = native_structural_gate(source)
    baseline_proposed, _gate = proposed_texture(
        source, lam=0.05, strength=1.5,
        virtual_passes=8, gate_power=8,
    )
    baseline, _ = tangent_reservoir_route(
        baseline_proposed, gate, radius=40.0
    )
    jump_proposed, jump_potential, diagnostic = jump_texture_components(
        source, gate, lam=0.05, virtual_passes=8
    )
    jump_oscillation, _ = tangent_reservoir_route(
        jump_proposed, gate, radius=40.0
    )
    first_cartoon = 0.05 / (0.05 - 0.10 * lap_hat(source.shape))
    jump_boundary = np.fft.ifft2(
        np.fft.fft2(jump_potential) * (1.0 - first_cartoon)
    ).real
    jump = jump_boundary + jump_oscillation
    baseline_error = np.linalg.norm(baseline - scene["texture"])
    jump_error = np.linalg.norm(jump - scene["texture"])
    assert diagnostic["half_threshold"] == 5.0
    assert diagnostic["support_partition"] == "Otsu between-class variance"
    assert 0.0 < diagnostic["support_class_boundary"] < diagnostic[
        "support_high_mean"
    ] < 1.0
    assert jump_error < 0.40 * baseline_error


def test_jump_measure_represents_structure_only_discontinuity_as_texture():
    scene = multiscale_crossing_scene(128)
    # A constant background isolates the authored object discontinuities from
    # the multiscale scene's deliberately nonperiodic affine frame seam.
    source = 100.0 + scene["jump_potential"]
    gate = native_structural_gate(source)
    jump_proposed, jump_potential, _ = jump_texture_components(
        source, gate, lam=0.05, virtual_passes=8
    )
    jump_oscillation, _ = tangent_reservoir_route(
        jump_proposed, gate, radius=40.0
    )
    first_cartoon = 0.05 / (0.05 - 0.10 * lap_hat(source.shape))
    jump_boundary = np.fft.ifft2(
        np.fft.fft2(jump_potential) * (1.0 - first_cartoon)
    ).real
    estimated = jump_boundary + jump_oscillation
    truth = scene["boundary_texture"]
    assert np.linalg.norm(estimated - truth) < 0.05 * np.linalg.norm(truth)


def test_estimated_discontinuity_is_a_conservative_jump_measure():
    scene = multiscale_crossing_scene(128)
    source = scene["source"]
    spectrum, estimated_x, estimated_y, diagnostic = estimate_jump_measure(
        source,
        native_structural_gate(source),
        lam=0.05,
        virtual_passes=8,
    )
    potential = np.fft.ifft2(spectrum).real
    np.testing.assert_allclose(
        estimated_x,
        np.roll(potential, -1, axis=1) - potential,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        estimated_y,
        np.roll(potential, -1, axis=0) - potential,
        atol=2e-12,
    )

    truth = scene["jump_potential"]
    truth_x = np.roll(truth, -1, axis=1) - truth
    truth_y = np.roll(truth, -1, axis=0) - truth
    truth_energy = np.sum(truth_x * truth_x + truth_y * truth_y)
    relative_error = np.sqrt(np.sum(
        (estimated_x - truth_x) ** 2 + (estimated_y - truth_y) ** 2
    ) / truth_energy)
    normal_gain = np.sum(
        estimated_x * truth_x + estimated_y * truth_y
    ) / truth_energy
    assert relative_error < 0.20
    assert normal_gain > 0.94
    assert diagnostic["observed_transverse_energy_fraction"] < 0.01


def test_paired_one_sided_traces_reproduce_step_and_cancel_affine_field():
    y, x = np.mgrid[:64, :64].astype(np.float64)
    value = 2.0 * x - 3.0 * y + 10.0 * (x >= 32)
    adjacent_x, _ = paired_one_sided_trace(value, reproduction_order=1)
    affine_x, affine_y = paired_one_sided_trace(
        value, reproduction_order=2
    )
    cubic_x, cubic_y = paired_one_sided_trace(
        value, reproduction_order=4
    )
    assert adjacent_x[24, 31] == 12.0
    assert affine_x[24, 31] == 10.0
    assert cubic_x[24, 31] == 10.0
    assert affine_y[24, 24] == 0.0
    assert cubic_y[24, 24] == 0.0


def test_native_jump_measure_matches_python_research_operator():
    scene = multiscale_crossing_scene(128)
    source = scene["source"]
    gate = native_structural_gate(source)
    proposed, jump_potential, _ = jump_texture_components(
        source, gate, lam=0.05, virtual_passes=8
    )
    oscillation, _ = tangent_reservoir_route(
        proposed, gate, radius=40.0
    )
    first_cartoon = 0.05 / (0.05 - 0.10 * lap_hat(source.shape))
    boundary = np.fft.ifft2(
        np.fft.fft2(jump_potential) * (1.0 - first_cartoon)
    ).real
    expected_texture = boundary + oscillation
    expected = source - expected_texture, expected_texture
    actual = bfft.MeyerPlan(
        source.shape, passes=64, threads=1, solver=0
    ).split_jump_measure(source)
    np.testing.assert_allclose(actual[0], expected[0], atol=3e-10)
    np.testing.assert_allclose(actual[1], expected[1], atol=3e-10)


def test_default_split_is_jump_measure_and_legacy_remains_explicit():
    source = multiscale_crossing_scene(128)["source"]
    plan = bfft.MeyerPlan(
        source.shape, passes=64, threads=1, solver=0
    )
    default = plan.split(source)
    jump = plan.split_jump_measure(source)
    legacy = plan.split_legacy(source)
    np.testing.assert_array_equal(default[0], jump[0])
    np.testing.assert_array_equal(default[1], jump[1])
    assert np.linalg.norm(default[1] - legacy[1]) > 1.0


def test_native_jump_measure_is_thread_count_identical():
    source = junction_texture_scene(128)["source"]
    single = bfft.MeyerPlan(
        source.shape, threads=1, solver=0
    ).split_jump_measure(source)
    parallel = bfft.MeyerPlan(
        source.shape, threads=4, solver=0
    ).split_jump_measure(source)
    np.testing.assert_array_equal(single[0], parallel[0])
    np.testing.assert_array_equal(single[1], parallel[1])
