import numpy as np

from benchmarks.sad_v3.rate_model import (
    SAD_SITE_BYTES,
    _canonicalize_labels,
    estimate_v3_rate,
)


def test_canonical_label_order_follows_first_raster_appearance():
    labels = np.array([[8, 8, 2], [5, 2, 5]], dtype=np.int32)
    canonical, old_for_new = _canonicalize_labels(labels)

    assert old_for_new.tolist() == [8, 2, 5]
    assert canonical.tolist() == [[0, 0, 1], [2, 1, 2]]


def test_rate_model_uses_parent_map_for_strict_nested_texture():
    structural = np.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.int32)
    texture = np.array([[0, 0, 2, 2], [1, 1, 3, 3]], dtype=np.int32)
    result = {
        "reconstruction_rgb": np.zeros((2, 4, 3), dtype=np.float64),
        "labels": structural,
        "texture_labels": texture,
        "centers": np.zeros((2, 2), dtype=np.float64),
        "texture_centers": np.zeros((4, 2), dtype=np.float64),
    }

    rate = estimate_v3_rate(
        result,
        structural_ridges=1,
        texture_ridges=3,
        graph_phase=True,
    )

    assert rate["topology"]["nested_parent_map"]
    assert rate["topology"]["structural_map_bytes"] == 0
    assert rate["structural_bytes_per_cell"] == 32
    assert rate["texture_bytes_per_cell"] == 68
    assert rate["sad_layered_site_proxy_bytes"] == 6 * SAD_SITE_BYTES
    assert rate["estimated_stream_bpp"] > rate["parameter_only_bpp"]

