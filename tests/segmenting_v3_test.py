import numpy as np
import pytest

from experiments.segmenting_v3 import (
    SegmentingV3Config,
    _cell_frame,
    _centers_after_collapse,
    _graph_unrolled_texture_columns,
    _graph_unrolled_texture_phases,
    _joint_leaf_collapse,
    _paired_metric_split_partition,
    _texture_dirichlet_envelope,
    build_segmenting_v3,
)
from experiments.compound_segment_quotient import (
    _adaptive_labels_at_count,
    build_compound_segment_quotient,
    labels_at_ratio,
)


def test_joint_collapse_preserves_unmerged_transport_centers():
    previous = np.array([
        [0, 0, 1, 1],
        [0, 0, 2, 2],
    ], dtype=np.int32)
    collapsed = np.array([
        [0, 0, 0, 0],
        [0, 0, 1, 1],
    ], dtype=np.int32)
    old_centers = np.array([
        [0.125, 0.25],
        [0.875, 0.25],
        [0.91, 0.83],
    ])
    centers = _centers_after_collapse(
        previous,
        collapsed,
        old_centers,
        np.ones(previous.shape),
    )

    # Cells 0 and 1 were genuinely pooled, so their new center follows their
    # six-pixel union. Cell 2 was only renumbered and must retain its exact
    # canonical site instead of moving to its two-pixel raster centroid.
    np.testing.assert_allclose(centers[0], [5.0 / 12.0, 5.0 / 12.0])
    np.testing.assert_array_equal(centers[1], old_centers[2])


def test_joint_collapse_ignores_unrepresented_transport_sites():
    previous = np.array([
        [0, 0, 2, 2],
        [0, 0, 3, 3],
    ], dtype=np.int32)
    collapsed = np.array([
        [0, 0, 0, 0],
        [0, 0, 1, 1],
    ], dtype=np.int32)
    old_centers = np.array([
        [0.125, 0.25],
        [0.50, 0.50],  # Duplicate transport seed with no owned raster pixel.
        [0.875, 0.25],
        [0.91, 0.83],
    ])
    centers = _centers_after_collapse(
        previous,
        collapsed,
        old_centers,
        np.ones(previous.shape),
    )

    # The dead site is not part of either quotient component. Cells 0 and 2
    # pool, while represented singleton 3 keeps its canonical transport site.
    np.testing.assert_allclose(centers[0], [5.0 / 12.0, 5.0 / 12.0])
    np.testing.assert_array_equal(centers[1], old_centers[3])


def test_compound_segments_split_atom_sides_and_merge_across_atoms():
    height, width = 24, 36
    atom_labels = np.repeat(
        np.arange(3, dtype=np.int32), width // 3,
    )[None, :].repeat(height, axis=0)
    target = np.empty((height, width, 3), dtype=np.float64)
    target[: height // 2] = (0.12, 0.18, 0.24)
    target[height // 2 :] = (0.82, 0.76, 0.68)
    quotient = build_compound_segment_quotient(
        atom_labels,
        target,
        target,
        boundary_confidence=np.zeros((height, width)),
        target_ratio=2.0 / 3.0,
    )

    labels = quotient["labels"]
    assert quotient["atom_count"] == 3
    assert quotient["compound_count"] == 2
    assert np.unique(labels[: height // 2]).size == 1
    assert np.unique(labels[height // 2 :]).size == 1
    assert labels[0, 0] != labels[-1, 0]
    # Every vertical reconstruction atom contributes pixels to both compound
    # segments, while each compound spans all three immutable fit atoms.
    for atom in range(3):
        assert np.unique(labels[atom_labels == atom]).size == 2
    for compound in range(2):
        assert np.unique(atom_labels[labels == compound]).size == 3
    assert not quotient["reconstruction_changed"]
    np.testing.assert_array_equal(
        labels_at_ratio(quotient, 2.0 / 3.0), labels)


def test_parallel_compound_calibration_is_exactly_sequential():
    height, width = 40, 52
    y, x = np.mgrid[:height, :width]
    atoms = ((x // 4) + 13 * (y // 4)).astype(np.int32)
    target = np.stack((
        0.4 + 0.3 * np.sin((x + y) / 3.0),
        0.5 + 0.2 * np.cos((2.0 * x - y) / 5.0),
        0.2 + 0.6 * (x >= width // 2),
    ), axis=2)
    quotient = build_compound_segment_quotient(
        atoms,
        target,
        target,
        boundary_confidence=np.zeros((height, width)),
    )
    arguments = (
        quotient["leaf_labels"],
        quotient["leaf_count"],
        quotient["graph_pairs"],
        quotient["graph_order"],
        quotient["graph_barrier"],
        quotient["leaf_population"],
        quotient["requested_count"],
    )
    sequential = _adaptive_labels_at_count(
        *arguments, parallel_calibration=False)
    parallel = _adaptive_labels_at_count(
        *arguments, parallel_calibration=True)

    np.testing.assert_array_equal(parallel[0], sequential[0])
    assert parallel[1:] == sequential[1:]


def test_cell_frame_can_retain_pre_collapse_coordinate_unit():
    previous = np.array([
        [0, 0, 1, 1],
        [0, 0, 2, 2],
    ], dtype=np.int32)
    collapsed = np.array([
        [0, 0, 0, 0],
        [0, 0, 1, 1],
    ], dtype=np.int32)
    old_centers = np.array([
        [0.125, 0.25],
        [0.875, 0.25],
        [0.91, 0.83],
    ])
    new_centers = _centers_after_collapse(
        previous, collapsed, old_centers, np.ones(previous.shape))
    tensor = (
        np.ones(previous.shape),
        np.zeros(previous.shape),
        np.zeros(previous.shape),
    )
    old_normal, _ = _cell_frame(previous, old_centers, tensor)
    new_normal, _ = _cell_frame(
        collapsed,
        new_centers,
        tensor,
        normalization_cells=len(old_centers),
    )

    singleton = previous == 2
    np.testing.assert_array_equal(
        new_normal.reshape(previous.shape)[singleton],
        old_normal.reshape(previous.shape)[singleton],
    )


def test_frozen_pair_metric_is_the_gauss_chord_bisector():
    labels = np.zeros((9, 9), dtype=np.int32)
    centers = np.array(((0.25, 0.5), (0.75, 0.5)))
    parent = np.array((0, 0), dtype=np.int32)
    # Wild off-chord tensors would curve a pointwise-metric boundary.  The
    # chord itself has an isotropic tensor, so its exact frozen bisector is
    # the vertical half-space between the two sites.
    qxx = np.full(labels.shape, 50.0)
    qxy = np.full(labels.shape, 30.0)
    qyy = np.full(labels.shape, 20.0)
    qxx[4, (3, 5)] = 1.0
    qxy[4, (3, 5)] = 0.0
    qyy[4, (3, 5)] = 1.0
    geometry = {
        "precision_xx": qxx,
        "precision_xy": qxy,
        "precision_yy": qyy,
        "metric_trace_p90": 2.0,
        "max_support_px": 9.0,
    }
    split = _paired_metric_split_partition(
        labels,
        centers,
        parent,
        geometry,
        0.25,
        freeze_pair_metric=True,
    )

    np.testing.assert_array_equal(split[:, :5], 0)
    np.testing.assert_array_equal(split[:, 5:], 1)


def test_graph_phase_unroll_is_deterministic_above_central_difference_fold():
    height, width = 32, 40
    y, x = np.mgrid[:height, :width]
    labels = (
        (y // 8) * (width // 8) + x // 8
    ).astype(np.int32)
    cells = int(np.max(labels)) + 1
    flat = labels.ravel()
    count = np.bincount(flat, minlength=cells)
    centers = np.column_stack((
        np.bincount(
            flat, weights=(x.ravel() + 0.5), minlength=cells)
        / count / width,
        np.bincount(
            flat, weights=(y.ravel() + 0.5), minlength=cells)
        / count / height,
    ))
    signal = np.cos(0.4 * x + 1.8 * y)

    first, diagnostic = _graph_unrolled_texture_phases(
        labels, centers, signal)
    second, repeated = _graph_unrolled_texture_phases(
        labels, centers, signal)

    assert diagnostic == repeated
    assert diagnostic["full_band_one_sided"]
    assert diagnostic["tree_edges"] == cells - 1
    assert diagnostic["median_wave_y"] > np.pi / 2.0
    for first_axis, second_axis in zip(first, second):
        assert np.array_equal(first_axis, second_axis)
        assert np.all(np.isfinite(first_axis))

    columns, column_diagnostic = _graph_unrolled_texture_columns(
        labels, centers, signal)
    assert column_diagnostic["quadrature_complete"]
    assert len(columns) == 4
    np.testing.assert_allclose(
        columns[0] * columns[0] + columns[1] * columns[1],
        1.0,
        atol=2e-15,
    )
    np.testing.assert_allclose(
        columns[2] * columns[2] + columns[3] * columns[3],
        1.0,
        atol=2e-15,
    )


def test_texture_dirichlet_envelope_is_exactly_nonexpansive_per_cell():
    labels = np.zeros((2, 2), dtype=np.int32)
    target = np.zeros((2, 2, 3), dtype=np.float64)
    target[..., 0] = [[0.0, 1.0], [0.0, 1.0]]
    fitted = target.copy()
    fitted[..., 0] = [[-1.0, 2.0], [-1.0, 2.0]]

    bounded, diagnostic = _texture_dirichlet_envelope(
        labels, target, fitted)

    np.testing.assert_allclose(bounded, target)
    assert diagnostic["contracted_cells"] == 1
    assert diagnostic["contracted_pixels"] == 4
    assert diagnostic["minimum_gain"] == pytest.approx(1.0 / 3.0)
    assert diagnostic["after_energy"] <= diagnostic["target_energy"]

    already_bounded, unchanged = _texture_dirichlet_envelope(
        labels, target, target * 0.5)
    assert np.array_equal(already_bounded, target * 0.5)
    assert unchanged["contracted_cells"] == 0


def test_joint_leaf_collapse_unions_only_compatible_one_child_parents():
    height, width = 8, 12
    y, x = np.mgrid[:height, :width]
    structural = (x >= width // 2).astype(np.int32)
    texture = structural.copy()
    smooth = np.stack((
        0.1 + 0.01 * x,
        0.2 + 0.02 * y,
        np.full((height, width), 0.3),
    ), axis=2)
    zero = np.zeros_like(smooth)

    collapsed_structure, collapsed_texture, diagnostic = (
        _joint_leaf_collapse(
            structural,
            texture,
            smooth,
            smooth,
            zero,
            zero,
            penalty=4.0,
            basis_terms=6,
        )
    )
    assert diagnostic["eligible_structural_cells"] == 2
    assert diagnostic["accepted_pairs"] == 1
    assert diagnostic["structural_cells_removed"] == 1
    assert diagnostic["texture_cells_removed"] == 1
    assert np.array_equal(np.unique(collapsed_structure), [0])
    assert np.array_equal(np.unique(collapsed_texture), [0])

    discontinuous = smooth + 2.0 * structural[..., None]
    kept_structure, kept_texture, rejected = _joint_leaf_collapse(
        structural,
        texture,
        discontinuous,
        discontinuous,
        zero,
        zero,
        penalty=4.0,
        basis_terms=6,
    )
    assert rejected["accepted_pairs"] == 0
    assert np.array_equal(kept_structure, structural)
    assert np.array_equal(kept_texture, texture)

    subdivided_texture = np.where(
        structural == 0, (y >= height // 2).astype(np.int32), 2,
    ).astype(np.int32)
    _, _, subdivided = _joint_leaf_collapse(
        structural,
        subdivided_texture,
        smooth,
        smooth,
        zero,
        zero,
        penalty=4.0,
        basis_terms=6,
    )
    assert subdivided["eligible_structural_cells"] == 1
    assert subdivided["candidate_pairs"] == 0


def test_joint_leaf_collapse_does_not_take_transitive_pair_closure():
    height, width = 8, 18
    y, x = np.mgrid[:height, :width]
    structural = np.minimum(x // 6, 2).astype(np.int32)
    texture = structural.copy()
    smooth = np.stack((
        0.1 + 0.01 * x,
        0.2 + 0.02 * y,
        np.full((height, width), 0.3),
    ), axis=2)
    zero = np.zeros_like(smooth)

    collapsed_structure, collapsed_texture, diagnostic = (
        _joint_leaf_collapse(
            structural,
            texture,
            smooth,
            smooth,
            zero,
            zero,
            penalty=4.0,
            basis_terms=6,
        )
    )

    # All three cells lie on one affine surface, so both adjacent pair tests
    # pass. They still do not certify a three-cell component: only the one
    # reciprocal best pair may be contracted in this single noniterative pass.
    assert diagnostic["fit_compatible_pairs"] == 2
    assert diagnostic["accepted_pairs"] == 1
    assert diagnostic["collapse_topology"] == "mutual_best_edge"
    assert np.unique(collapsed_structure).size == 2
    assert np.unique(collapsed_texture).size == 2


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
            nested_texture_ridges=3,
            diagnostic_return_basis=True,
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
    assert cleanup["peak_only_split_count"] >= 0
    assert cleanup["peak_error"].shape == cleanup["mean_error"].shape
    assert result["texture_phase_graph"]["signal"] == (
        "pre_joint_post_cartoon_residual")
    assert result["texture_phase_graph"]["incidence"] == (
        "pre_joint_leaf_quotient")
    assert result["coordinate_trace"][-1]["axis"] == (
        "normal + corner + phase envelope + quadratic frame + "
        "structural trace")
    assert result["texture_active_basis"].shape[2] == 19
    active = result["texture_active_basis"]
    np.testing.assert_allclose(
        active[..., 11:15],
        active[..., 3:7] * active[..., 7:8] * active[..., 9:10],
        atol=2e-15,
    )


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
    construction_centers = result["structural_initial_centers"]
    assert np.allclose(
        result["texture_initial_centers"][:len(construction_centers)],
        construction_centers,
    )
    joint = result["joint_leaf_collapse"]
    assert joint["enabled"]
    assert joint["final_structural_cells"] == len(result["centers"])
    assert joint["final_texture_cells"] == len(result["texture_centers"])
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
