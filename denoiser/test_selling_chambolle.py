"""Invariants for the Selling-edge transport-Chambolle hypothesis."""

from __future__ import annotations

import unittest

import numpy as np

from .selling_chambolle import (
    _graph_flux_readout,
    _graph_gradient,
    _paired_trace,
    build_selling_flux_graph,
    cross_chart_phase_covector_statistics,
    denoise_cross_chart_phase_chambolle,
    denoise_paired_trace_chambolle,
    denoise_phase_action_chambolle,
    denoise_population_phase_chambolle,
    denoise_selling_chambolle,
    paired_trace_transport_metric,
    transported_noise_authority,
)
from .transport_chambolle import transport_flux_body


class SellingChambolleTests(unittest.TestCase):
    def _body(self):
        yy, xx = np.mgrid[:18, :20]
        image = 0.15 + 0.013 * xx + 0.009 * yy
        image[8, 10] += 0.7
        return image, transport_flux_body(image)

    def test_graph_incidence_is_adjoint(self):
        image, body = self._body()
        graph = build_selling_flux_graph(body)
        rng = np.random.default_rng(4)
        flux = rng.normal(size=graph.edge_count)
        left = float(np.sum(_graph_gradient(image, graph) * flux))
        right = float(np.sum(image * _graph_flux_readout(flux, graph)))
        self.assertAlmostEqual(left, right, places=11)

    def test_every_graph_flux_is_mass_conservative(self):
        _image, body = self._body()
        graph = build_selling_flux_graph(body)
        rng = np.random.default_rng(8)
        readout = _graph_flux_readout(
            rng.normal(size=graph.edge_count), graph)
        self.assertAlmostEqual(float(np.sum(readout)), 0.0, places=12)

    def test_endpoint_capacity_allocation_covers_each_point(self):
        _image, body = self._body()
        graph = build_selling_flux_graph(body)
        incident = np.bincount(
            np.concatenate((graph.first, graph.second)),
            weights=np.concatenate((graph.capacity, graph.capacity)),
            minlength=body.centre.size,
        )
        np.testing.assert_array_less(
            np.abs(body.removable_amplitude).ravel() - 1e-14,
            incident + 1e-14,
        )

    def test_selling_reconstruction_is_binary64_accurate(self):
        _image, body = self._body()
        graph = build_selling_flux_graph(body)
        self.assertLess(graph.selling_reconstruction_error, 1e-10)

    def test_constant_is_exact_identity(self):
        image = np.full((16, 19), 0.42)
        estimate, diagnostic = denoise_selling_chambolle(image)
        np.testing.assert_array_equal(estimate, image)
        self.assertEqual(diagnostic["status"], "transport identity")

    def test_impulse_contracts_and_recomposes(self):
        image = np.full((20, 22), 0.25)
        image[10, 11] = 1.0
        estimate, diagnostic = denoise_selling_chambolle(image)
        self.assertLess(estimate[10, 11], image[10, 11])
        residual = image - estimate
        np.testing.assert_allclose(
            estimate + residual, image, atol=2e-16, rtol=0.0)
        self.assertLessEqual(diagnostic["maximum_capacity_violation"], 1e-14)
        self.assertLess(diagnostic["mass_error"], 1e-12)

    def test_static_inner_objective_is_monotone(self):
        rng = np.random.default_rng(17)
        yy, xx = np.mgrid[:18, :20]
        truth = 0.2 + 0.5 * (xx > 9)
        image = truth + 0.1 * rng.normal(size=truth.shape)
        _estimate, diagnostic = denoise_selling_chambolle(image)
        for cycle in diagnostic["observer_history"]:
            trace = np.asarray(
                cycle["inner"]["contraction_objective_trace"])
            self.assertLessEqual(float(np.max(np.diff(trace))), 1e-10)

    def test_population_phase_is_bounded_and_recomposes(self):
        rng = np.random.default_rng(23)
        image = 0.3 + 0.1 * rng.normal(size=(18, 20))
        estimate, diagnostic = denoise_population_phase_chambolle(image)
        for cycle in diagnostic["observer_history"]:
            self.assertGreaterEqual(cycle["detail_phase"], 0.0)
            self.assertLessEqual(cycle["detail_phase"], 1.0)
        residual = image - estimate
        np.testing.assert_allclose(
            estimate + residual, image, atol=2e-16, rtol=0.0)

    def test_population_phase_constant_is_identity(self):
        image = np.full((16, 18), 0.42)
        estimate, diagnostic = denoise_population_phase_chambolle(image)
        np.testing.assert_array_equal(estimate, image)
        self.assertEqual(diagnostic["status"], "transport identity")

    def test_paired_trace_metric_is_spd_and_affine_blind(self):
        yy, xx = np.mgrid[:18, :20]
        affine = 0.2 + 0.01 * xx - 0.015 * yy
        np.testing.assert_allclose(
            _paired_trace(affine, 1)[:, 1:-2], 0.0,
            atol=2e-16, rtol=0.0)
        np.testing.assert_allclose(
            _paired_trace(affine, 0)[1:-2, :], 0.0,
            atol=2e-16, rtol=0.0)
        metric = paired_trace_transport_metric(affine)
        determinant = (
            metric["metric_xx"] * metric["metric_yy"]
            - metric["metric_xy"] ** 2)
        self.assertGreater(float(np.min(determinant)), 0.0)

    def test_paired_trace_population_recomposes(self):
        rng = np.random.default_rng(29)
        image = 0.4 + 0.08 * rng.normal(size=(18, 20))
        estimate, diagnostic = denoise_paired_trace_chambolle(image)
        residual = image - estimate
        np.testing.assert_allclose(
            estimate + residual, image, atol=2e-16, rtol=0.0)
        self.assertIn("paired-trace", diagnostic["method"])

    def test_transported_noise_authority_is_a_bounded_positive_union(self):
        image, body = self._body()
        authority, _metric, diagnostic = transported_noise_authority(body)
        self.assertEqual(authority.shape, image.shape)
        self.assertTrue(np.all(np.isfinite(authority)))
        self.assertGreaterEqual(float(np.min(authority)), 0.0)
        self.assertLessEqual(float(np.max(authority)), 1.0)
        self.assertLess(
            diagnostic["statistic_transport_row_sum_error"], 1e-12)
        self.assertLess(
            diagnostic["statistic_transport_column_sum_error"], 1e-12)

    def test_phase_action_population_recomposes(self):
        rng = np.random.default_rng(31)
        image = 0.4 + 0.08 * rng.normal(size=(18, 20))
        estimate, diagnostic = denoise_phase_action_chambolle(image)
        residual = image - estimate
        np.testing.assert_allclose(
            estimate + residual, image, atol=2e-16, rtol=0.0)
        self.assertIn("phase-action", diagnostic["method"])
        for cycle in diagnostic["observer_history"]:
            self.assertGreaterEqual(cycle["detail_phase"], 0.0)
            self.assertLessEqual(cycle["detail_phase"], 1.0)

    def test_cross_chart_phase_is_target_excluded_and_psd(self):
        image, _body = self._body()
        numerator, denominator, diagnostic = (
            cross_chart_phase_covector_statistics(image))
        self.assertEqual(numerator.shape, (4,) + image.shape)
        self.assertEqual(denominator.shape, numerator.shape)
        self.assertGreater(diagnostic["distinct_pair_coverage"], 0.85)
        self.assertLessEqual(diagnostic["phase_cauchy_violation"], 1e-14)
        self.assertTrue(np.all(denominator >= 0.0))
        self.assertTrue(np.all(np.abs(numerator) <= denominator + 1e-14))

    def test_cross_chart_phase_action_recomposes(self):
        rng = np.random.default_rng(37)
        image = 0.4 + 0.08 * rng.normal(size=(18, 20))
        estimate, diagnostic = denoise_cross_chart_phase_chambolle(image)
        residual = image - estimate
        np.testing.assert_allclose(
            estimate + residual, image, atol=2e-16, rtol=0.0)
        self.assertIn("distinct-chart", diagnostic["method"])
        self.assertEqual(diagnostic["phase_source"], "distinct charts")
        self.assertTrue(diagnostic[
            "phase_evidence_reused_across_cycles"])


if __name__ == "__main__":
    unittest.main()
