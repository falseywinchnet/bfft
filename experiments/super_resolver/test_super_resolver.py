import unittest

import numpy as np

from experiments.super_resolver.core import (
    SuperResolutionConfig,
    decimate,
    focus_crop,
    prepare_observation,
    run_eikonal_upscale,
)
from experiments.super_resolver.app import SuperResolverApp


class _TextureDPG:
    def __init__(self):
        self.values = {"sr_display_mode": "Full image"}
        self.updated = []
        self.configured = []

    def get_value(self, tag):
        return self.values[tag]

    def set_value(self, tag, value):
        self.updated.append((tag, value))

    def configure_item(self, tag, **kwargs):
        self.configured.append((tag, kwargs))


class SuperResolverTests(unittest.TestCase):
    def test_point_and_box_decimation_are_explicit(self):
        source = np.arange(8 * 12 * 3, dtype=np.float64).reshape(8, 12, 3)
        source /= source.max()

        point = decimate(source, 2, "Point decimation")
        box = decimate(source, 2, "Box prefilter")

        np.testing.assert_allclose(point, source[::2, ::2])
        np.testing.assert_allclose(
            box, source.reshape(4, 2, 6, 2, 3).mean(axis=(1, 3))
        )

    def test_eikonal_prefilter_builds_a_richer_forward_observation(self):
        source = np.full((32, 40, 3), (0.2, 0.5, 0.8), dtype=np.float64)
        config = SuperResolutionConfig(
            scale=4,
            decimation="Eikonal prefilter",
            anisotropy=1.25,
            tensor_sigma=1.5,
            maximum_side=64,
        )

        prepared = prepare_observation(source, config)

        self.assertEqual(prepared.observed.shape, (8, 10, 3))
        np.testing.assert_allclose(
            prepared.observed,
            np.broadcast_to((0.2, 0.5, 0.8), prepared.observed.shape),
            atol=1e-12,
        )
        self.assertEqual(prepared.forward_anisotropy, 1.25)
        self.assertEqual(prepared.forward_tensor_sigma, 1.5)
        self.assertTrue(prepared.forward_clamp_range)


    def test_constant_survives_four_times_eikonal_upscale(self):
        source = np.full((32, 40, 3), (0.2, 0.5, 0.8), dtype=np.float64)
        config = SuperResolutionConfig(scale=4, maximum_side=64)
        prepared = prepare_observation(source, config)

        result = run_eikonal_upscale(prepared, config)

        np.testing.assert_allclose(result.eikonal, source, atol=1e-12)
        self.assertLess(result.metrics["Eikonal"]["mse"], 1e-24)
        self.assertLess(abs(result.metrics["Eikonal"]["ssim"] - 1.0), 1e-12)
        self.assertEqual(result.views["Fine error difference"].shape, source.shape)


    def test_focus_crop_keeps_size_at_image_boundaries(self):
        image = np.zeros((30, 50, 3), dtype=np.float64)

        upper_left = focus_crop(image, 0.0, 0.0, 16)
        lower_right = focus_crop(image, 1.0, 1.0, 16)

        self.assertEqual(upper_left.shape, (16, 16, 3))
        self.assertEqual(lower_right.shape, (16, 16, 3))


    def test_result_reports_baseline_relative_fine_error(self):
        y, x = np.mgrid[:24, :28]
        grey = 0.5 + 0.35 * np.sin(2.0 * np.pi * x / 7.0) * (y >= 12)
        source = np.repeat(grey[..., None], 3, axis=2)
        config = SuperResolutionConfig(scale=2, maximum_side=64)

        result = run_eikonal_upscale(prepare_observation(source, config), config)

        self.assertEqual(set(result.metrics), {"Lanczos", "Eikonal", "difference"})
        self.assertEqual(
            set(result.metrics["difference"]), {"mse", "ssim", "fine_mse"}
        )
        self.assertTrue(np.isfinite(list(result.metrics["difference"].values())).all())

    def test_display_updates_permanent_texture_without_replacing_it(self):
        dpg = _TextureDPG()
        app = SuperResolverApp(dpg)

        app._replace_texture(1, np.full((20, 30, 3), 0.5))

        self.assertEqual(dpg.updated[0][0], "sr_texture_1")
        self.assertEqual(dpg.configured[0][0], "sr_middle_image")
        self.assertEqual(dpg.configured[0][1]["texture_tag"], "sr_texture_1")
        self.assertAlmostEqual(
            float(app.texture_buffers[1].reshape(460, 460, 4)[0, 0, 3]), 1.0
        )


if __name__ == "__main__":
    unittest.main()
