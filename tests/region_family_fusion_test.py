import numpy as np

from experiments.region_family_fusion import build_region_family_fusion
from experiments.region_posterization import build_region_posterization


def test_terminal_color_siblings_can_join_a_common_geometric_host():
    image = np.full((48, 68, 3), (0.92, 0.94, 0.98), dtype=np.float64)
    labels = np.zeros((48, 68), dtype=np.int32)
    labels[16:43, 12:56] = 1
    image[16:43, 12:56] = (0.93, 0.72, 0.12)
    for tip, transition, left in ((2, 4, 18), (3, 5, 40)):
        labels[7:15, left:left + 8] = tip
        image[7:15, left:left + 8] = (0.04, 0.035, 0.03)
        labels[15:16, left:left + 8] = transition
        image[15:16, left:left + 8] = (0.48, 0.37, 0.08)

    poster = build_region_posterization(
        image, labels, max_depth=5, histogram_side=16,
        labels_are_compact=True)
    result = build_region_family_fusion(poster)
    family = result["region_family"]

    assert family[1] == family[2] == family[3]
    assert family[0] != family[1]
    assert result["host_hyperedges"] >= 1


def test_unbounded_exterior_cannot_adopt_enclosed_color_siblings():
    image = np.full((48, 72, 3), (0.72, 0.57, 0.39), dtype=np.float64)
    labels = np.zeros((48, 72), dtype=np.int32)
    y, x = np.mgrid[:48, :72]
    for region, center_x in ((1, 22), (2, 50)):
        disk = np.square(x - center_x) + np.square(y - 24) <= 8 * 8
        labels[disk] = region
        image[disk] = (0.62, 0.64, 0.66)

    poster = build_region_posterization(
        image, labels, max_depth=5, histogram_side=16,
        labels_are_compact=True)
    result = build_region_family_fusion(poster)
    family = result["region_family"]

    assert family[1] == family[2]
    assert family[0] != family[1]
    assert all(merge["host"] != 0 for merge in result["host_merges"])


def test_single_region_is_a_valid_fixed_point():
    image = np.full((12, 16, 3), (0.3, 0.5, 0.7), dtype=np.float64)
    labels = np.zeros((12, 16), dtype=np.int32)
    poster = build_region_posterization(
        image, labels, max_depth=4, histogram_side=8,
        labels_are_compact=True)
    result = build_region_family_fusion(poster)

    assert result["family_count"] == 1
    assert result["candidate_pairs"] == 0
    np.testing.assert_array_equal(result["labels"], labels)
