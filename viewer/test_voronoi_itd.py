"""Contracts for the one-shot frozen-Meyer-density Voronoi operator."""

from __future__ import annotations

import numpy as np

from viewer.voronoi_itd import (
    VoronoiITDConfig,
    eikonal_adjacency,
    extract_voronoi_baseline,
    update_knots,
    voronoi_itd,
)


def _curved_test_image(height=72, width=88):
    y, x = np.mgrid[:height, :width]
    edge = np.tanh(
        (x - 0.7 * y - 15.0 - 4.0 * np.sin(y / 9.0)) / 2.5)
    detail = np.sin(2.0 * np.pi * (x + 0.63 * y) / 11.0)
    return 0.48 + 0.23 * edge + 0.07 * detail


def test_lossless_bounded_one_shot_reconstruction():
    image = _curved_test_image()
    result = voronoi_itd(
        image,
        VoronoiITDConfig(levels=12, allocation_max_side=48),
    )
    assert len(result.levels) == 1
    level = result.levels[0]
    assert level.support_measure.shape == image.shape
    assert np.min(level.baseline) >= np.min(image) - 1e-12
    assert np.max(level.baseline) <= np.max(image) + 1e-12
    np.testing.assert_allclose(result.reconstruction, image, atol=2e-14)


def test_constant_image_needs_no_partition():
    image = np.full((40, 52), 0.37)
    result = voronoi_itd(image)
    assert not result.levels
    np.testing.assert_array_equal(result.residual, image)


def test_restricted_allocation_is_prolonged_to_the_source_shape():
    image = _curved_test_image(83, 109)
    level = extract_voronoi_baseline(
        image,
        VoronoiITDConfig(allocation_max_side=32),
    )
    assert level.owner.shape == image.shape
    assert level.eikonal_distance.shape == image.shape
    assert level.baseline.shape == image.shape


def test_interface_graph_contains_exactly_visible_cell_pairs():
    image = _curved_test_image(58, 73)
    level = extract_voronoi_baseline(
        image,
        VoronoiITDConfig(allocation_max_side=40),
    )
    np.testing.assert_array_equal(
        level.delaunay_edges,
        eikonal_adjacency(level.owner),
    )


def test_knot_update_is_a_convex_range_preserving_map():
    values = np.array((0.1, 0.4, 0.8, 0.3))
    centers = np.array((
        (0.1, 0.1), (0.8, 0.1), (0.8, 0.8), (0.1, 0.8)))
    edges = np.array(((0, 1), (1, 2), (2, 3), (3, 0)), dtype=np.int32)
    knots = update_knots(values, centers, edges, alpha=0.65)
    assert np.min(knots) >= np.min(values)
    assert np.max(knots) <= np.max(values)


def test_grayscale_and_rgb_guidance_both_remain_finite():
    image = _curved_test_image(49, 61)
    gray = extract_voronoi_baseline(
        image, VoronoiITDConfig(allocation_max_side=40))
    rgb = np.repeat(image[..., None], 3, axis=2)
    colour = extract_voronoi_baseline(
        image, VoronoiITDConfig(allocation_max_side=40), rgb)
    assert np.all(np.isfinite(gray.baseline))
    assert np.all(np.isfinite(colour.baseline))
