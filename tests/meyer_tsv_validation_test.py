import numpy as np

from experiments.meyer_tsv_validation import (
    _ordering_auc,
    symmetric_support_scene,
    tsv_four_direction,
)


def test_tsv_is_zero_on_a_constant_field():
    value = np.full((32, 32), 17.0)
    assert np.max(np.abs(tsv_four_direction(value, radius=4))) < 1e-12


def test_ordering_auc_has_expected_extremes():
    assert _ordering_auc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 1.0
    assert _ordering_auc(np.array([0.0, 1.0]), np.array([2.0, 3.0])) == 0.0


def test_validation_scene_separates_contour_and_texture_masks():
    scene = symmetric_support_scene(128)
    assert np.count_nonzero(scene["contour"]) > 0
    assert np.count_nonzero(scene["texture_interior"]) > 0
    assert not np.any(scene["contour"] & scene["texture_interior"])
