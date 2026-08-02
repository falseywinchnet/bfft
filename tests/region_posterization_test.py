import numpy as np

from experiments.region_posterization import (
    build_region_posterization,
    multiscale_region_affinity,
    render_posterization_level,
)


def test_palette_stack_is_nested_and_region_mixtures_are_normalized():
    height, width = 32, 48
    y, x = np.mgrid[:height, :width]
    labels = ((x >= 16).astype(np.int32) + (x >= 32).astype(np.int32))
    image = np.stack((
        0.15 + 0.7 * (x / width),
        0.2 + 0.5 * (y / height),
        0.3 + 0.2 * np.sin((x + y) / 4.0),
    ), axis=2)
    result = build_region_posterization(
        image, labels, max_depth=4, histogram_side=16)

    assert result["region_count"] == 3
    for level in result["levels"]:
        np.testing.assert_allclose(
            np.sum(level["region_mixture"], axis=1), 1.0)
        assert np.all(np.isfinite(
            render_posterization_level(result, level["depth"])))
    for coarse, fine in zip(result["levels"], result["levels"][1:]):
        for family in range(fine["family_count"]):
            bins = fine["bin_family"] == family
            assert np.unique(coarse["bin_family"][bins]).size == 1


def test_multiscale_affinity_tracks_shared_palette_mass():
    image = np.zeros((18, 30, 3), dtype=np.float64)
    labels = np.zeros((18, 30), dtype=np.int32)
    labels[:, 10:20] = 1
    labels[:, 20:] = 2
    image[:, :10] = (0.85, 0.72, 0.18)
    image[:, 10:20] = (0.78, 0.64, 0.12)
    image[:, 20:] = (0.08, 0.12, 0.75)
    # Give the first two regions overlapping luminance phases, equivalent to
    # a local two-color dither observed through its regional average.
    image[::2, :20] *= 0.75
    result = build_region_posterization(
        image, labels, max_depth=4, histogram_side=16)
    affinity = multiscale_region_affinity(
        result, np.array(((0, 1), (0, 2)), dtype=np.int32))

    assert affinity[0] > affinity[1]
    assert affinity[0] > 0.25
