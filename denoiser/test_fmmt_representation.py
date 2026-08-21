"""Estimator-preserving representation tests for the FMMT speed path."""

from __future__ import annotations

import unittest

import numpy as np
from scipy import ndimage
from scipy import sparse
from scipy.sparse.csgraph import dijkstra

from .fmmt_certified import (
    _geo_transport,
    _geo_transport_vectorized,
    _joint_transport,
    _soft_histogram,
    denoise_fmmt,
)


class FMMTRepresentationTests(unittest.TestCase):
    def test_vector_transport_matches_scalar_reference_exactly(self):
        rng = np.random.default_rng(17)
        packets = rng.random((11, 13, 5))
        guide = rng.random((11, 13))
        scale = 0.02 + 0.2 * rng.random((11, 13))
        for alpha in (0.0, 0.2, 3.0):
            reference = _geo_transport(packets, guide, scale, 1.7, alpha)
            vectorized = _geo_transport_vectorized(
                packets, guide, scale, 1.7, alpha)
            np.testing.assert_array_equal(vectorized, reference)

    def test_packet_histogram_filter_matches_per_bin_reference_exactly(self):
        rng = np.random.default_rng(23)
        image = rng.random((17, 19))
        bins = 31
        actual = _soft_histogram(image, bins, -1.0, 1.0, 7)
        position = np.clip((image + 1.0) / 2.0, 0.0, 1.0) * (bins - 1)
        low = np.floor(position).astype(np.int32)
        high = np.minimum(low + 1, bins - 1)
        fraction = position - low
        expected = np.zeros(image.shape + (bins,), np.float32)
        row, column = np.indices(image.shape)
        expected[row, column, low] = 1.0 - fraction
        expected[row, column, high] += fraction
        for channel in range(bins):
            expected[..., channel] = ndimage.uniform_filter(
                expected[..., channel], size=7, mode="reflect")
        expected /= np.maximum(
            expected.sum(axis=-1, keepdims=True), 1e-12)
        np.testing.assert_array_equal(actual, expected)

    def test_front_batch_changes_workspace_not_estimator(self):
        yy, xx = np.mgrid[-1:1:24j, -1:1:24j]
        observation = np.clip(
            0.42 + 0.18 * xx + 0.09 * np.sin(9.0 * yy), 0.0, 1.0)
        small, _ = denoise_fmmt(
            observation, certify_support=False, stride=6, front_batch=2)
        large, diagnostics = denoise_fmmt(
            observation, certify_support=False, stride=6, front_batch=16)
        np.testing.assert_allclose(large, small, atol=5e-16, rtol=0.0)
        self.assertEqual(diagnostics["front_batch"], 16)
        self.assertIn("stage_seconds", diagnostics)

    def test_in_place_joint_packet_transport_matches_split_reference(self):
        graph = sparse.diags(
            [np.ones(11), np.ones(11)], [-1, 1], shape=(12, 12),
            format="csr")
        anchors = np.array([0, 3, 7, 11], dtype=np.int64)
        rng = np.random.default_rng(29)
        fields = [rng.random((12, 5)), rng.random((12, 7))]
        actual, actual_mass = _joint_transport(
            graph, anchors, fields, tau=2.0, batch=3)

        expected = [np.zeros_like(field) for field in fields]
        expected_mass = np.zeros(12)
        for start in range(0, anchors.size, 3):
            ids = anchors[start:start + 3]
            distance = dijkstra(
                graph,
                directed=False,
                indices=ids,
                limit=7.0,
                return_predecessors=False,
            )
            if distance.ndim == 1:
                distance = distance[None, :]
            weight = np.exp(-np.minimum(distance, 700.0) / 2.0)
            weight[~np.isfinite(distance)] = 0.0
            expected_mass += np.sum(weight, axis=0)
            for index, field in enumerate(fields):
                expected[index] += weight.T @ field[ids]
        np.testing.assert_array_equal(actual_mass, expected_mass)
        for value, reference in zip(actual, expected):
            np.testing.assert_allclose(value, reference, atol=2e-15, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
