import numpy as np

from experiments import cartoon_fourier_angle_terms as angle_terms
from experiments import cartoon_fourier_transport as fourier


def _case():
    g = fourier.tree_experiment._load_image("synthetic", 32)
    trace, fluxes, objectives = fourier.bregman_trace(g, 0.05, 0.10, 16)
    return g, trace[4], fluxes[4], objectives


def test_first_term_is_the_existing_hodge_drop():
    g, current, flux, objectives = _case()
    terms = angle_terms.refreshed_angle_terms(
        current, flux, g, 0.05, objectives, 1
    )
    raw = fourier.routed_flux_proposal(current, flux, g, 0.05)
    expected, alpha = fourier._segment_taylor_drop(
        current, raw, g, 0.05
    )
    assert len(terms) == 1
    assert abs(terms[0].alpha - alpha) < 1e-14
    assert abs(
        terms[0].objective - fourier.objective(expected, g, 0.05)
    ) < 1e-10


def test_requested_divergence_is_outside_the_dual_disk_image():
    g, current, _, _ = _case()
    requested = 0.05 * (current - g)
    _, _, oracle = angle_terms.closest_disk_divergence(
        requested, iterations=4000
    )
    # The convex KKT point is accurate, yet its residual is decisively
    # nonzero: exact active-angle closure at the current primal is impossible.
    assert oracle.fixed_point_error < 1e-7
    assert oracle.relative_residual > 0.20


def test_exact_angle_refresh_terms_saturate_above_the_convex_oracle():
    g, current, flux, objectives = _case()
    sequence = angle_terms.refreshed_angle_terms(
        current, flux, g, 0.05, objectives, 8
    )
    requested = 0.05 * (current - g)
    _, _, oracle = angle_terms.closest_disk_divergence(
        requested, iterations=4000
    )
    assert sequence[-1].divergence_residual > 2.0 * oracle.relative_residual
    # Terms two through eight improve the accepted objective only marginally
    # and do not buy another equivalent Bregman pass.
    assert sequence[-1].equivalent_pass == sequence[0].equivalent_pass
    improvement = sequence[0].objective - sequence[-1].objective
    first_drop = fourier.objective(current, g, 0.05) - sequence[0].objective
    assert improvement < 0.10 * first_drop


def test_fixed_normal_higher_order_series_is_outside_its_radius():
    g, current, flux, _ = _case()
    p0x, p0y = fourier.routed_preflux(current, flux, g, 0.05)
    full = angle_terms.fixed_normal_angle_radius(p0x, p0y)
    stable = angle_terms.fixed_normal_angle_radius(
        p0x, p0y, relative_eigenvalue_cutoff=0.1
    )
    assert full.normal_residual < 1e-9
    assert full.tangent_outside_radius > 0.40
    assert full.tangent_maximum > 20.0
    # Keeping the tangent response inside the square-root series radius
    # discards precisely the weak modes needed to satisfy the normal equation.
    assert stable.tangent_maximum < 1.0
    assert stable.normal_residual > 0.90

    curvature = angle_terms.fixed_normal_curvature_terms(p0x, p0y)
    assert len(curvature) == 4
    assert curvature[1].coefficient_norm > 10.0 * \
        curvature[0].coefficient_norm
    assert curvature[-1].active_capacity_rms > \
        curvature[0].active_capacity_rms


def test_late_convergent_curvature_terms_birth_a_new_active_mask():
    g = fourier.tree_experiment._load_image("synthetic", 32)
    trace, fluxes, _ = fourier.bregman_trace(g, 0.05, 0.10, 24)
    p0x, p0y = fourier.routed_preflux(
        trace[24], fluxes[24], g, 0.05
    )
    radius = angle_terms.fixed_normal_angle_radius(p0x, p0y)
    curvature = angle_terms.fixed_normal_curvature_terms(p0x, p0y)
    assert radius.tangent_maximum < 1.0
    assert curvature[-1].active_capacity_rms < \
        curvature[0].active_capacity_rms
    # The original active equations improve, but newly overloaded pixels
    # outside that mask keep the global field infeasible.
    assert curvature[-1].global_overload > 0.20
    assert curvature[-1].maximum_norm > 1.5
