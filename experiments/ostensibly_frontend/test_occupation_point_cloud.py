import unittest

import numpy as np

from experiments.ostensibly_frontend.occupation_point_cloud import (
    affine_marginal_cloud,
    CloudFitConfig,
    PointCloudConfig,
    SupportGeometryConfig,
    fit_phone_cloud,
    jittered_support_distance,
    marginal_copula_cloud,
    occupation_to_point_cloud,
    sliced_wasserstein_distance,
    sliced_wasserstein_projection,
    sliced_wasserstein_projection_distance,
    symmetric_trimmed_chamfer,
)


class OccupationPointCloudTests(unittest.TestCase):
    def test_lift_is_deterministic_and_mass_preserving(self):
        rows, frames = np.mgrid[:20, :12]
        field = np.exp(-0.5 * ((rows - 7.0 - frames / 8.0) / 1.3) ** 2)
        config = PointCloudConfig(row_upsample=3, frame_upsample=5, point_count=4096)
        left = occupation_to_point_cloud(field, config)
        right = occupation_to_point_cloud(field, config)
        self.assertEqual(left.points.shape, (4096, 3))
        self.assertTrue(np.array_equal(left.points, right.points))
        self.assertGreater(left.mass, 0.0)
        self.assertTrue(np.all((left.points[:, 2] >= 0.0) & (left.points[:, 2] <= 1.0)))
        self.assertGreater(np.unique(left.points[:, 0]).size, field.shape[0])

    def test_zero_field_yields_empty_cloud(self):
        cloud = occupation_to_point_cloud(
            np.zeros((8, 6)), PointCloudConfig(point_count=100)
        )
        self.assertEqual(cloud.points.shape, (0, 3))
        self.assertEqual(cloud.mass, 0.0)

    def test_constrained_fit_reduces_synthetic_distance(self):
        frame = np.linspace(0.0, 18.0, 2500)
        row = 48.0 + 8.0 * np.sin(frame / 3.2) + 0.16 * frame
        height = 0.55 + 0.4 * np.cos(frame / 2.3) ** 2
        reference = np.column_stack((row, frame, height))
        moving = reference.copy()
        moving[:, 0] = 1.18 * (reference[:, 0] - 48.0) + 61.0 - 0.45 * (
            reference[:, 1] - 9.0
        )
        moving[:, 1] = 1.27 * (reference[:, 1] - 9.0) + 6.0
        config = CloudFitConfig(
            fit_point_count=1200,
            global_iterations=35,
            population_size=8,
            random_seed=4,
        )
        result = fit_phone_cloud(moving, reference, config)
        before = symmetric_trimmed_chamfer(moving, reference, config)
        after = symmetric_trimmed_chamfer(result.transformed_points, reference, config)
        self.assertLess(after, 0.35 * before)
        self.assertLess(result.postfit_distance, 0.35 * result.prefit_distance)

    def test_sliced_wasserstein_retains_occupation_multiplicity(self):
        low = np.asarray((0.0, 0.0, 0.0))
        high = np.asarray((10.0, 4.0, 0.0))
        left = np.vstack((np.tile(low, (90, 1)), np.tile(high, (10, 1))))
        right = np.vstack((np.tile(low, (10, 1)), np.tile(high, (90, 1))))
        chamfer_config = CloudFitConfig(fit_point_count=100)
        transport_config = CloudFitConfig(
            fit_point_count=100,
            distance_mode="sliced_wasserstein",
        )
        self.assertEqual(symmetric_trimmed_chamfer(left, right, chamfer_config), 0.0)
        self.assertGreater(
            sliced_wasserstein_distance(left, right, transport_config), 1.0
        )

    def test_preprojected_swd_exactly_matches_direct_distance(self):
        rng = np.random.default_rng(19)
        left = rng.normal(size=(93, 3))
        right = rng.normal(size=(137, 3))
        config = CloudFitConfig(
            distance_mode="sliced_wasserstein",
            sliced_projection_count=17,
            row_metric_scale=0.4,
            frame_metric_scale=1.2,
            height_metric_scale=0.7,
        )
        projected = sliced_wasserstein_projection_distance(
            sliced_wasserstein_projection(left, config),
            sliced_wasserstein_projection(right, config),
        )
        self.assertEqual(projected, sliced_wasserstein_distance(left, right, config))

    def test_marginal_copula_is_monotone_invariant_but_retains_dependence(self):
        coordinate = np.linspace(0.01, 0.99, 400)
        height = 0.5 + 0.2 * coordinate
        diagonal = np.column_stack((coordinate, coordinate, height))
        warped = np.column_stack(
            (np.exp(3.0 * coordinate), coordinate**3, height)
        )
        anti_diagonal = np.column_stack((coordinate, 1.0 - coordinate, height))
        left = marginal_copula_cloud(diagonal)
        right = marginal_copula_cloud(warped)
        different = marginal_copula_cloud(anti_diagonal)
        self.assertTrue(np.allclose(left[:, :2], right[:, :2]))
        config = CloudFitConfig(
            distance_mode="sliced_wasserstein",
            row_metric_scale=1.0,
            frame_metric_scale=1.0,
            height_metric_scale=1e-6,
        )
        self.assertLess(sliced_wasserstein_distance(left, right, config), 1e-9)
        self.assertGreater(
            sliced_wasserstein_distance(left, different, config), 0.05
        )

    def test_affine_marginal_gauge_preserves_density_under_global_chart(self):
        coordinate = np.linspace(0.01, 0.99, 400)
        source = np.column_stack(
            (coordinate**2, coordinate**3, np.ones_like(coordinate))
        )
        transformed = source.copy()
        transformed[:, 0] = 2.7 * source[:, 0] + 8.0
        transformed[:, 1] = 1.9 * source[:, 1] - 3.0
        left = affine_marginal_cloud(source)
        right = affine_marginal_cloud(transformed)
        self.assertTrue(np.allclose(left[:, :2], right[:, :2]))
        copula = marginal_copula_cloud(source)
        self.assertGreater(float(np.max(np.abs(left[:, 0] - copula[:, 0]))), 0.1)

    def test_support_distance_discards_multiplicity(self):
        coordinate = np.linspace(0.05, 0.95, 80)
        support = np.column_stack(
            (coordinate, 0.2 + 0.6 * coordinate, np.ones_like(coordinate))
        )
        repeated = np.repeat(support, 7, axis=0)
        config = SupportGeometryConfig(minimum_cell_count=1)
        self.assertEqual(jittered_support_distance(support, repeated, config), 0.0)

    def test_support_distance_prefers_locally_jittered_same_shape(self):
        coordinate = np.linspace(0.08, 0.92, 200)
        base = np.column_stack(
            (
                coordinate,
                0.5 + 0.22 * np.sin(8.0 * coordinate),
                np.ones_like(coordinate),
            )
        )
        base = np.repeat(base, 3, axis=0)
        jittered = base.copy()
        jittered[:, 0] += 1.0 / 256.0
        unrelated = base.copy()
        unrelated[:, 1] = 1.0 - unrelated[:, 1]
        config = SupportGeometryConfig(minimum_cell_count=2)
        self.assertLess(
            jittered_support_distance(base, jittered, config),
            jittered_support_distance(base, unrelated, config),
        )


if __name__ == "__main__":
    unittest.main()
