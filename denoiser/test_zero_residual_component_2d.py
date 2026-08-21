"""Invariants for explicit zero/nonzero causal residual components."""

import unittest

import numpy as np

from .zero_residual_component_2d import zero_residual_component_readouts


class ZeroResidualComponentTests(unittest.TestCase):
    def _law(self, image: np.ndarray) -> dict[str, np.ndarray]:
        signal = np.stack((image - 0.1, image + 0.1), axis=-1)
        residual = image[..., None] - signal
        shape = signal.shape
        return {
            "signal": signal,
            "residual": residual,
            "reference_mass": np.full(shape, 0.5),
            "hj_path_score": np.zeros(shape),
            "hj_simplex_collision_order": np.full(image.shape, 2.0),
            "local_mass": np.full(shape, 0.5),
            "mass": np.full(shape, 0.5),
            "likelihood": np.ones(shape),
            "root_mass": np.full(image.shape + (1, 2), 0.5),
        }

    def test_component_mass_is_conserved_and_readouts_are_bounded(self):
        image = np.linspace(0.2, 0.8, 64).reshape(8, 8)
        readouts, diagnostic = zero_residual_component_readouts(
            image, self._law(image))
        self.assertLess(diagnostic["component_mass_maximum_error"], 1e-14)
        for name in (
            "component_barycenter", "terminal_component_barycenter",
            "component_mode",
        ):
            self.assertGreaterEqual(float(readouts[name].min()), image.min())
            self.assertLessEqual(float(readouts[name].max()), image.max())

    def test_branch_permutation_does_not_change_readout(self):
        image = np.linspace(0.2, 0.8, 64).reshape(8, 8)
        law = self._law(image)
        first, _ = zero_residual_component_readouts(image, law)
        permuted = {
            key: (value[..., ::-1] if value.ndim == 3 else value)
            for key, value in law.items()
        }
        second, _ = zero_residual_component_readouts(image, permuted)
        np.testing.assert_allclose(
            first["component_barycenter"], second["component_barycenter"])
        np.testing.assert_allclose(
            first["component_mode"], second["component_mode"])

    def test_exact_zero_residual_selects_observation(self):
        image = np.full((8, 8), 0.4)
        signal = np.stack((image, image), axis=-1)
        law = {
            "signal": signal,
            "residual": np.zeros_like(signal),
            "reference_mass": np.full_like(signal, 0.5),
            "hj_path_score": np.zeros_like(signal),
            "hj_simplex_collision_order": np.full(image.shape, 2.0),
        }
        readouts, _ = zero_residual_component_readouts(image, law)
        np.testing.assert_array_equal(readouts["component_barycenter"], image)
        np.testing.assert_array_equal(
            readouts["terminal_component_barycenter"], image)
        np.testing.assert_array_equal(readouts["component_mode"], image)

    def test_terminal_component_uses_probability_exactly_once(self):
        image = np.full((8, 8), 0.4)
        law = self._law(image)
        law["hj_path_score"][..., 0] = 0.0
        law["hj_path_score"][..., 1] = -1.0
        readouts, diagnostic = zero_residual_component_readouts(
            image, law, nonzero_probability_mode="complete")
        score = 2.0 * law["hj_path_score"]
        score -= np.max(score, axis=-1, keepdims=True)
        branch = law["reference_mass"] * np.exp(score)
        branch /= np.sum(branch, axis=-1, keepdims=True)
        branch_mean = np.sum(branch * law["signal"], axis=-1)
        probability = readouts["nonzero_probability"]
        expected = (1.0 - probability) * image + probability * branch_mean
        expected = np.clip(expected, float(image.min()), float(image.max()))
        np.testing.assert_allclose(
            readouts["terminal_component_barycenter"], expected)
        self.assertLess(
            diagnostic["terminal_component_mass_maximum_error"], 1e-14)

    def test_complete_evidence_recognizes_zero_mean_residual_dispersion(self):
        image = np.full((8, 8), 0.4)
        law = self._law(image)
        mean_readouts, _ = zero_residual_component_readouts(
            image, law, nonzero_probability_mode="mean")
        complete_readouts, diagnostic = zero_residual_component_readouts(
            image, law, nonzero_probability_mode="complete")
        np.testing.assert_allclose(mean_readouts["nonzero_probability"], 0.0)
        np.testing.assert_allclose(
            complete_readouts["nonzero_probability"], 0.5)
        self.assertEqual(diagnostic["nonzero_probability_mode"], "complete")

    def test_self_consistent_evidence_uses_collided_terminal_law(self):
        image = np.full((8, 8), 0.4)
        law = self._law(image)
        law["hj_path_score"][..., 0] = 0.0
        law["hj_path_score"][..., 1] = -1.0
        precursor, _ = zero_residual_component_readouts(
            image, law, nonzero_probability_mode="mean")
        terminal, diagnostic = zero_residual_component_readouts(
            image, law, nonzero_probability_mode="self_consistent")
        self.assertGreater(
            float(terminal["nonzero_probability"].mean()),
            float(precursor["nonzero_probability"].mean()),
        )
        self.assertEqual(
            diagnostic["nonzero_probability_mode"], "self_consistent")

    def test_invalid_probability_mode_is_rejected(self):
        image = np.linspace(0.2, 0.8, 64).reshape(8, 8)
        with self.assertRaises(ValueError):
            zero_residual_component_readouts(
                image, self._law(image), nonzero_probability_mode="invalid")

    def test_transport_uncertainty_is_fisher_rao_contrast(self):
        image = np.full((8, 8), 0.4)
        law = self._law(image)
        law["hj_path_score"][..., 0] = 0.0
        law["hj_path_score"][..., 1] = -1.0
        coherent, _ = zero_residual_component_readouts(
            image, law, nonzero_probability_mode="self_consistent")
        identical, identical_diagnostic = zero_residual_component_readouts(
            image, law, nonzero_probability_mode="transport_uncertain")
        np.testing.assert_allclose(
            identical["nonzero_probability"],
            coherent["nonzero_probability"],
        )
        self.assertAlmostEqual(
            identical_diagnostic["mean_transport_hellinger_contrast"], 0.0)

        law["local_mass"] = np.stack(
            (np.ones_like(image), np.zeros_like(image)), axis=-1)
        law["mass"] = np.stack(
            (np.zeros_like(image), np.ones_like(image)), axis=-1)
        uncertain, uncertain_diagnostic = zero_residual_component_readouts(
            image, law, nonzero_probability_mode="transport_uncertain")
        self.assertGreater(
            float(uncertain["nonzero_probability"].mean()),
            float(coherent["nonzero_probability"].mean()),
        )
        self.assertAlmostEqual(
            uncertain_diagnostic["mean_transport_hellinger_contrast"], 1.0)

    def test_observation_cavity_divides_out_local_likelihood(self):
        image = np.full((8, 8), 0.4)
        law = self._law(image)
        law["mass"] = np.stack(
            (np.full_like(image, 0.9), np.full_like(image, 0.1)), axis=-1)
        law["likelihood"] = np.stack(
            (np.full_like(image, 9.0), np.ones_like(image)), axis=-1)
        cavity, diagnostic = zero_residual_component_readouts(
            image, law, nonzero_probability_mode="observation_cavity")
        np.testing.assert_allclose(cavity["nonzero_probability"], 0.0)
        self.assertGreater(
            diagnostic["mean_observation_cavity_surprise"], 0.0)

    def test_root_resolved_law_separates_noise_from_transport_uncertainty(self):
        image = np.full((8, 8), 0.4)
        within_law = self._law(image)
        within, within_diagnostic = zero_residual_component_readouts(
            image, within_law, nonzero_probability_mode="root_resolved")

        between_law = self._law(image)
        first = np.stack(
            (np.ones_like(image), np.zeros_like(image)), axis=-1)
        second = np.stack(
            (np.zeros_like(image), np.ones_like(image)), axis=-1)
        between_law["root_mass"] = 0.5 * np.stack(
            (first, second), axis=-2)
        between, between_diagnostic = zero_residual_component_readouts(
            image, between_law, nonzero_probability_mode="root_resolved")
        self.assertGreater(
            float(within["nonzero_probability"].mean()),
            float(between["nonzero_probability"].mean()),
        )
        self.assertGreater(
            within_diagnostic["mean_root_within_residual_variance"], 0.0)
        self.assertGreater(
            between_diagnostic["mean_root_between_residual_variance"], 0.0)


if __name__ == "__main__":
    unittest.main()
