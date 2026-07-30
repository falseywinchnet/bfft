import numpy as np
import pytest

from experiments.segmenting_v3 import SegmentingV3Config, build_segmenting_v3


@pytest.mark.parametrize("upgrade_mode", ("boundary_band", "full_map"))
def test_v3_texture_never_changes_cartoon_ownership(upgrade_mode):
    height, width = 48, 60
    y, x = np.mgrid[:height, :width]
    image = np.empty((height, width, 3), dtype=np.float64)
    image[..., 0] = x / width
    image[..., 1] = y / height
    image[..., 2] = 0.25 + 0.2 * ((x // 4 + y // 3) % 2)
    result = build_segmenting_v3(
        image,
        SegmentingV3Config(
            safety_cells=256,
            owner_upgrade_mode=upgrade_mode,
            texture_model="parent_ridges",
            texture_coordinates=2,
            coordinate_geometry="owner_eikonal",
            threads=2,
        ),
    )

    assert result["model"] == (
        "v3_half_cartoon_full_map_owner_upgrade_"
        "parent_ridges_eikonal_paired_local_texture")
    assert result["labels"].shape == (height, width)
    assert result["reconstruction_rgb"].shape == image.shape
    assert np.all(result["labels"] >= 0)
    assert int(np.max(result["labels"])) < len(result["centers"])
    assert np.array_equal(
        np.unique(result["labels"]),
        np.arange(len(result["centers"])),
    )
    assert (
        result["owner_upgrade"]["changed_pixels"]
        <= result["owner_upgrade"]["band_pixels"]
    )
    assert len(result["coordinate_trace"]) == 2
    assert [item["axis"] for item in result["coordinate_trace"]] == [
        "normal", "tangent"]
    assert np.isfinite(result["record"]["psnr"])
    assert np.all(np.isfinite(result["normal_coordinate"]))
    assert np.all(np.isfinite(result["tangent_coordinate"]))


def test_nested_texture_construction_is_parented_then_cleanup_is_flat():
    height, width = 40, 52
    y, x = np.mgrid[:height, :width]
    image = np.stack((
        x / width,
        y / height,
        0.3 + 0.15 * np.sin(2.0 * np.pi * x / 5.0),
    ), axis=2)
    result = build_segmenting_v3(
        image,
        SegmentingV3Config(
            safety_cells=256,
            texture_safety_cells=1024,
            nested_texture_ridges=1,
            threads=2,
        ),
    )

    initial_labels = result["texture_initial_labels"]
    for texture_id in range(len(result["texture_initial_centers"])):
        assert np.unique(
            result["labels"][initial_labels == texture_id]
        ).size == 1

    texture_labels = result["texture_labels"]
    assert np.array_equal(
        np.unique(texture_labels),
        np.arange(len(result["texture_centers"])),
    )
    cleanup = result["texture_cleanup"]
    assert cleanup["enabled"]
    assert (
        cleanup["final_cells"]
        == cleanup["initial_cells"]
        + cleanup["split_count"]
        - cleanup["merge_count"]
    )
    assert cleanup["cross_parent_merge_count"] > 0


def test_canonical_v2_quotient_survives_dense_texture_cleanup():
    height, width = 42, 54
    y, x = np.mgrid[:height, :width]
    image = np.stack((
        x / width,
        y / height,
        0.35 + 0.2 * np.sin(2.0 * np.pi * (x + y) / 7.0),
    ), axis=2)
    result = build_segmenting_v3(
        image,
        SegmentingV3Config(
            structural_topology="canonical_v2",
            structural_allocation_side=24,
            structural_safety_cells=256,
            structural_flow_sweeps=1,
            texture_safety_cells=1024,
            nested_texture_ridges=1,
            texture_cross_structural_merges=False,
            threads=2,
        ),
    )

    assert result["structural_topology"] == "canonical_v2"
    assert result["structural_transport_model"] == "bucket_graph"
    characteristic = result["structural_characteristic"]
    assert characteristic["requested_passes"] == 1
    assert characteristic["effective_passes"] == 0
    assert not characteristic["resolved_core"]
    assert result["model"].startswith(
        "v3_canonical_v2_structural_quotient_")
    assert np.array_equal(
        np.unique(result["labels"]),
        np.arange(len(result["centers"])),
    )
    assert result["texture_cleanup"]["cross_parent_merge_count"] == 0
    assert result["texture_population"]["separable_population"]
    assert result["texture_population"]["missing_parent_seeds"] == 0
    assert result["texture_population"]["surplus_sites"] > 0
    assert np.allclose(
        result["texture_initial_centers"][:len(result["centers"])],
        result["centers"],
    )
    structural_geometry = result["structural_population_geometry"]
    structural_raw = (
        np.asarray(structural_geometry["measure"], dtype=np.float64)
        * float(structural_geometry["implied_cells"])
    )
    full_raw = (
        np.asarray(
            result["cartoon_geometry"]["measure"], dtype=np.float64)
        * float(result["cartoon_geometry"]["implied_cells"])
    )
    assert np.all(structural_raw <= full_raw + 1e-7)
    initial = result["texture_initial_labels"]
    for texture_id in range(len(result["texture_initial_centers"])):
        assert np.unique(result["labels"][initial == texture_id]).size == 1
