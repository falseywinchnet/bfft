"""Tests for spectral response heads and their SCL/CFF integrations."""
from __future__ import annotations

import math
import unittest

import torch

from ML_experiment.response_enhanced import (
    SPECTRAL_CFF_MIDDLE,
    SPECTRAL_SCL_MAX,
    SPECTRAL_SCL_MIDDLE,
    make_response_variant,
)
from ML_experiment.spectral_response import (
    SpectralResponse,
    SymmetricPackedEmbedding,
    spectral_response_summary,
)


class SpectralResponseTests(unittest.TestCase):
    def test_packed_embedding_is_symmetric_and_isometric(self):
        embedding = SymmetricPackedEmbedding(5)
        packed = torch.randn(7, embedding.packed_dim)
        matrix = embedding(packed)
        self.assertTrue(torch.allclose(matrix, matrix.mT))
        self.assertTrue(
            torch.allclose(
                packed.square().sum(-1), matrix.square().sum((-2, -1)), atol=1e-5
            )
        )
        self.assertTrue(torch.allclose(embedding.pack(matrix), packed, atol=1e-6))

    def test_initialization_has_gap_and_noncommuting_feature_matrices(self):
        response = SpectralResponse(27, matrix_dim=5)
        base, features = response.coefficient_matrices()
        eigenvalues = torch.linalg.eigvalsh(base)
        self.assertAlmostEqual(float(eigenvalues[2]), 0.0, places=5)
        self.assertGreater(float(eigenvalues[2] - eigenvalues[1]), 0.99)
        self.assertGreater(float(eigenvalues[3] - eigenvalues[2]), 0.99)
        commutator = base @ features[0] - features[0] @ base
        self.assertGreater(float(commutator.norm()), 1e-7)

    def test_response_forward_backward_and_gap_diagnostics(self):
        response = SpectralResponse(9, matrix_dim=5)
        features = torch.randn(4, 12, 9, requires_grad=True)
        output = response(features)
        self.assertEqual(output.shape, (4, 12, 1))
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        summary = spectral_response_summary(response)
        self.assertIsNotNone(summary["spectral_gap_mean"])
        self.assertGreater(summary["spectral_gap_mean"], 0.0)

    def test_network_variants_forward_backward(self):
        for name in (
            SPECTRAL_SCL_MIDDLE,
            SPECTRAL_SCL_MAX,
            SPECTRAL_CFF_MIDDLE,
        ):
            model = make_response_variant(name, 2, 3, 12)
            x = torch.randn(8, 2, requires_grad=True)
            output = model(x)
            self.assertEqual(output.shape, (8, 3))
            output.square().mean().backward()
            self.assertTrue(math.isfinite(float(output.square().mean())))


if __name__ == "__main__":
    unittest.main()
