"""Regression checks for the physical HDR moonlight benchmark."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "experiments"))

from realistic_moonlight_bench import (  # noqa: E402
    MoonlightCamera,
    capture_stream,
    hdr_scene,
    sensor_maps,
    translate_with_mask,
)


def test_hdr_truth_exceeds_an_8_bit_scene():
    config = MoonlightCamera(size=128, canvas_margin=16, frames=2)
    scene = hdr_scene(config)
    stops = np.log2(float(np.max(scene)) / float(np.min(scene)))
    assert stops > 10.0
    assert len(np.unique(scene)) > 4096


def test_translation_has_no_cyclic_wrap():
    image = np.zeros((16, 16), dtype=np.float32)
    image[0, 0] = 1.0
    shifted, support = translate_with_mask(image, (-2, -3))
    assert np.sum(shifted) == 0.0
    assert np.sum(support) == 14 * 13


def test_quantized_sensor_stream_is_reproducible():
    config = MoonlightCamera(
        size=64,
        canvas_margin=8,
        frames=3,
        shift_radius=4,
    )
    scene = hdr_scene(config)
    maps = sensor_maps(config)
    first = list(capture_stream(scene, config, maps))
    second = list(capture_stream(scene, config, maps))
    for a, b in zip(first, second):
        assert a[0] == b[0]
        assert a[2] == b[2]
        np.testing.assert_array_equal(a[1], b[1])
        scaled = (
            a[1] + config.dark_electrons) / config.electrons_per_dn
        np.testing.assert_allclose(scaled, np.rint(scaled))
