"""Exact invariants for the V3 region-complex audit."""

from __future__ import annotations

import unittest

import numpy as np

from experiments.v3_object_transport.region_complex import (
    build_region_complex,
    summarize_region_complex,
)
from experiments.v3_object_transport.incidence_bundle import (
    build_incidence_bundle,
)
from experiments.v3_object_transport.fused_meyer_evidence import (
    build_fused_meyer_evidence,
)
from experiments.v3_object_transport.connection_bloom import (
    analytical_bloom,
    connection_green_gram,
    connection_heat_gram,
    fit_joint_whitener,
    incidence_topology,
    relation_features,
    region_source_matrix,
    signed_incidence_connection,
)
from experiments.v3_object_transport.contour_transport import (
    build_contour_transport,
    summarize_contour_transport,
)
from experiments.v3_object_transport.relative_enclosure import (
    build_relative_enclosures,
    summarize_relative_enclosures,
)
from experiments.v3_object_transport.participation_algebra import (
    complete_kernel_algebra,
    complete_participation_kernel,
    normalized_linear_kernel,
)
from experiments.v3_object_transport.junction_depth import (
    build_junction_depth,
    summarize_junction_depth,
)
from experiments.v3_object_transport.depth_contour_transport import (
    build_depth_contour_transport,
)
from experiments.v3_object_transport.contour_cycle_nesting import (
    build_contour_cycle_nesting,
)
from experiments.v3_object_transport.compositional_bloom import (
    spectral_exponential_bloom,
    typed_order_two_bloom,
)
from experiments.v3_object_transport.amodal_contour_transport import (
    amodal_pair_residuals,
    build_amodal_transport,
    extract_amodal_ports,
    fit_zero_whitener,
)
from experiments.v3_object_transport.depth_hodge import build_depth_hodge
from experiments.v3_object_transport.support_manifold_transport import (
    build_support_manifold_transport,
)
from experiments.v3_object_transport.proposal_topology_transport import (
    analytical_proposal_bloom,
    build_proposal_connection,
)
from experiments.v3_object_transport.wavelet_leader_evidence import (
    centered_log_leaders,
    region_wavelet_leader_features,
)
from experiments.v3_object_transport.wavelet_split_transport import (
    analytical_split_transport,
    content_connection,
)
from experiments.v3_object_transport.multiscale_proposal_transport import (
    build_multiscale_connection,
    multiscale_point_sources,
    normalized_region_overlap,
    query_multiscale_bloom,
)


def _fixture() -> tuple[np.ndarray, dict]:
    labels = np.array([
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [2, 2, 3, 3],
        [2, 2, 3, 3],
    ], dtype=np.int32)
    structural = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 1, 2, 2],
        [1, 1, 2, 2],
    ], dtype=np.int32)
    rgb = np.zeros((4, 4, 3), dtype=np.float64)
    rgb[:, 2:] = (0.8, 0.2, 0.1)
    target = rgb.copy()
    cartoon = 0.75 * rgb
    texture = target - cartoon
    result = {
        "compound_segmentation": {
            "enabled": True,
            "labels": labels,
            "leaf_labels": labels,
        },
        "labels": structural,
        "target_lab": target,
        "cartoon_lab": cartoon,
        "texture_target_lab": texture,
        "texture_fit_lab": texture.copy(),
        "texture_geometry": {
            "boundary_confidence": np.arange(16).reshape(4, 4) / 15.0,
        },
    }
    return rgb, result


class RegionComplexTests(unittest.TestCase):
    def test_preserves_area_interfaces_and_ancestry(self):
        rgb, result = _fixture()
        complex_ = build_region_complex(result, rgb, level="leaves")
        self.assertEqual(complex_["region_count"], 4)
        self.assertEqual(int(np.sum(complex_["node"]["area"])), 16)
        self.assertEqual(len(complex_["edge"]["first"]), 4)
        self.assertEqual(complex_["topology"]["arc"]["count"], 4)
        self.assertEqual(len(complex_["arc"]["cell_first"]), 4)
        self.assertEqual(complex_["topology"]["junction"]["count"], 1)
        np.testing.assert_array_equal(complex_["edge"]["length"], 2.0)
        self.assertTrue(np.all(
            (complex_["edge"]["boundary"] >= 0.0)
            & (complex_["edge"]["boundary"] <= 1.0)
        ))
        np.testing.assert_array_equal(
            complex_["node"]["structural_support_count"], [1, 1, 1, 1])

        np.testing.assert_array_equal(
            complex_["node"]["structural_dominant"], [0, 0, 1, 2])
        np.testing.assert_allclose(
            complex_["node"]["structural_purity"], 1.0)
        bundle = build_incidence_bundle(complex_)
        self.assertEqual(len(bundle["incidence"]["arc"]), 8)
        self.assertEqual(len(bundle["continuation"]["junction"]), 4)
        incidence = bundle["incidence"]
        half = len(incidence["arc"]) // 2
        np.testing.assert_array_equal(
            incidence["arc"][:half], incidence["arc"][half:])
        np.testing.assert_array_equal(
            incidence["region"][:half], incidence["outside"][half:])
        np.testing.assert_array_equal(
            incidence["outside"][:half], incidence["region"][half:])
        np.testing.assert_allclose(
            incidence["target_transition"][:half],
            -incidence["target_transition"][half:],
        )
        np.testing.assert_allclose(
            incidence["cartoon_transition"][:half],
            -incidence["cartoon_transition"][half:],
        )
        np.testing.assert_allclose(
            incidence["texture_rms_transition"][:half],
            -incidence["texture_rms_transition"][half:],
        )

    def test_proposal_topology_removes_only_self_transport(self):
        from scipy import sparse

        participation = sparse.csr_matrix(np.array([
            [0.0, 0.5, 0.5],
            [1.0, 0.0, 0.0],
        ]))
        weight = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ])
        connection = build_proposal_connection(participation, weight)
        self.assertTrue(np.all(np.diag(connection["adjacency"]) == 0.0))
        self.assertTrue(np.all(connection["adjacency"] >= 0.0))
        np.testing.assert_allclose(
            connection["normalized_connection"],
            connection["normalized_connection"].T,
        )
        bloom = analytical_proposal_bloom(
            connection["normalized_connection"], np.eye(3))
        np.testing.assert_allclose(
            bloom["heat_kernel"], bloom["heat_kernel"].T, atol=1e-12)
        self.assertGreaterEqual(
            np.linalg.eigvalsh(bloom["heat_kernel"]).min(), -1e-12)
        self.assertGreaterEqual(
            np.linalg.eigvalsh(bloom["transported_base_kernel"]).min(),
            -1e-12,
        )

    def test_haar_leaders_are_contrast_complement_invariant(self):
        field = np.arange(64, dtype=np.float64).reshape(8, 8) / 63.0
        direct = centered_log_leaders(field)
        inverted = centered_log_leaders(1.0 - field)
        self.assertEqual(len(direct), 3)
        for first, second in zip(direct, inverted):
            np.testing.assert_allclose(first, second, atol=1e-12)
        labels = np.zeros((8, 8), dtype=np.int32)
        labels[:, 4:] = 1
        features, names = region_wavelet_leader_features(
            labels, {"target": field})
        self.assertEqual(features.shape, (2, 6))
        self.assertEqual(len(names), 6)

    def test_wavelet_split_transport_is_typed_and_permutation_covariant(self):
        proposal = np.array([
            [0.0, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.0],
        ])
        embedding = np.array([
            [1.0, 0.0],
            [0.8, 0.2],
            [0.0, 1.0],
        ])
        base = np.eye(3)
        connection = content_connection(embedding)
        np.testing.assert_allclose(np.diag(connection), 0.0)
        self.assertLessEqual(
            float(np.max(np.abs(np.linalg.eigvalsh(connection)))),
            1.0 + 1e-12,
        )
        direct = analytical_split_transport(proposal, embedding, base)
        permutation = np.array([2, 0, 1])
        inverse = np.argsort(permutation)
        permuted = analytical_split_transport(
            proposal[permutation][:, permutation],
            embedding[permutation], base[permutation][:, permutation])
        for key in ("content_connection", "split_heat_kernel",
                    "transported_base_kernel"):
            np.testing.assert_allclose(
                direct[key], permuted[key][inverse][:, inverse], atol=1e-12)
        self.assertGreaterEqual(
            float(np.min(np.linalg.eigvalsh(
                direct["transported_base_kernel"]))), -1e-12)

    def test_multiscale_overlap_bloom_preserves_exact_scale_topology(self):
        coarse = np.array([[0, 1], [0, 1]], dtype=np.int32)
        fine = np.repeat(np.repeat(coarse, 2, axis=0), 2, axis=1)
        overlap = normalized_region_overlap(coarse, fine, fine.shape)
        np.testing.assert_allclose(overlap.toarray(), np.eye(2))
        proposal = {
            2: np.array([[0.0, 1.0], [1.0, 0.0]]),
            4: np.array([[0.0, 1.0], [1.0, 0.0]]),
        }
        connection = build_multiscale_connection(
            {2: coarse, 4: fine}, proposal)
        np.testing.assert_allclose(
            connection["normalized_connection"].toarray(),
            connection["normalized_connection"].toarray().T,
        )
        points = {
            "left": {"xy": [0.0, 0.0], "instance": "left"},
            "right": {"xy": [1.0, 0.0], "instance": "right"},
        }
        names, sources = multiscale_point_sources(
            connection, {2: coarse, 4: fine}, points)
        query = query_multiscale_bloom(
            connection, sources, {2: np.eye(2), 4: np.eye(2)})
        self.assertEqual(names, ("left", "right"))
        np.testing.assert_allclose(
            np.diag(query["heat_similarity"]), 1.0, atol=1e-12)
        self.assertGreaterEqual(
            np.linalg.eigvalsh(query["heat_similarity"]).min(), -1e-12)

    def test_is_invariant_to_sparse_region_ids(self):
        rgb, result = _fixture()
        baseline = build_region_complex(result, rgb, level="leaves")
        sparse = result.copy()
        sparse["compound_segmentation"] = {
            "enabled": True,
            "labels": np.array([10, 20, 40, 80], dtype=np.int32)[
                result["compound_segmentation"]["labels"]
            ],
            "leaf_labels": np.array([10, 20, 40, 80], dtype=np.int32)[
                result["compound_segmentation"]["leaf_labels"]
            ],
        }
        candidate = build_region_complex(sparse, rgb, level="leaves")
        np.testing.assert_array_equal(candidate["labels"], baseline["labels"])
        for name in (
            "area", "target_mean", "cartoon_mean", "structural_dominant",
        ):
            np.testing.assert_allclose(
                candidate["node"][name], baseline["node"][name])
        np.testing.assert_array_equal(
            candidate["edge"]["first"], baseline["edge"]["first"])
        np.testing.assert_array_equal(
            candidate["edge"]["second"], baseline["edge"]["second"])

    def test_summary_contains_no_object_partition_or_affinity_score(self):
        rgb, result = _fixture()
        summary = summarize_region_complex(build_region_complex(result, rgb))
        self.assertEqual(summary["regions"], 4)
        self.assertEqual(summary["level"], "leaves")
        self.assertEqual(summary["interfaces"], 4)
        self.assertNotIn("object_labels", summary)
        self.assertNotIn("affinity", summary)

    def test_fused_meyer_is_aligned_auxiliary_evidence(self):
        rgb, result = _fixture()
        evidence = build_fused_meyer_evidence(
            result["target_lab"], passes=2)
        np.testing.assert_allclose(
            evidence["cartoon"] + evidence["texture"] + evidence["residual"],
            evidence["target"],
            atol=2e-15,
            rtol=0.0,
        )
        complex_ = build_region_complex(
            result, rgb, level="leaves", fused_meyer=evidence)
        self.assertIn("fused_cartoon_mean", complex_["node"])
        self.assertIn("fused_texture", complex_["arc"])
        bundle = build_incidence_bundle(complex_)
        incidence = bundle["incidence"]
        half = len(incidence["arc"]) // 2
        np.testing.assert_allclose(
            incidence["fused_cartoon_transition"][:half],
            -incidence["fused_cartoon_transition"][half:],
        )

    def test_connection_bloom_is_seed_free_finite_and_centered(self):
        rgb, result = _fixture()
        complex_ = build_region_complex(result, rgb, level="leaves")
        bundle = build_incidence_bundle(complex_)
        raw, names = relation_features(
            complex_, bundle, include_fused=False)
        # A second nonidentical atlas member makes the joint covariance
        # construction explicit without selecting any semantic mode.
        second = raw.copy()
        second[:, 0] += np.linspace(-0.2, 0.2, len(second))
        whitener = fit_joint_whitener((raw, second), names)
        whitened = whitener.transform(raw)
        topology, summary, degree = incidence_topology(complex_, bundle)
        flowed = analytical_bloom(whitened, topology, degree)
        self.assertEqual(flowed.shape, whitened.shape)
        self.assertTrue(np.all(np.isfinite(flowed)))
        self.assertEqual(summary["incidences"], len(bundle["incidence"]["arc"]))
        self.assertEqual(summary["regions"], complex_["region_count"])
        self.assertGreater(summary["total_states"], summary["incidences"])

    def test_outside_shuffle_preserves_topology_but_changes_relations(self):
        rgb, result = _fixture()
        complex_ = build_region_complex(result, rgb, level="leaves")
        bundle = build_incidence_bundle(complex_)
        ordinary, names = relation_features(
            complex_, bundle, include_fused=False)
        shuffled, shuffled_names = relation_features(
            complex_, bundle, include_fused=False,
            shuffled_outside=True, shuffle_key="fixture")
        self.assertEqual(names, shuffled_names)
        self.assertFalse(np.array_equal(ordinary, shuffled))
        first, first_summary, first_degree = incidence_topology(complex_, bundle)
        second, second_summary, second_degree = incidence_topology(complex_, bundle)
        self.assertEqual(first_summary, second_summary)
        np.testing.assert_array_equal(first_degree, second_degree)
        self.assertEqual((first != second).nnz, 0)

    def test_signed_connection_heat_is_symmetric_and_local(self):
        rgb, result = _fixture()
        complex_ = build_region_complex(result, rgb, level="leaves")
        bundle = build_incidence_bundle(complex_)
        raw, names = relation_features(
            complex_, bundle, include_fused=False)
        second = raw + np.linspace(-0.1, 0.1, len(raw))[:, None]
        whitener = fit_joint_whitener((raw, second), names)
        whitened = whitener.transform(raw)
        connection, degree, summary = signed_incidence_connection(
            complex_, bundle, whitened)
        asymmetry = connection - connection.T
        self.assertLessEqual(
            float(np.max(np.abs(asymmetry.data), initial=0.0)), 1e-15)
        self.assertEqual(summary["isolated_incidences"], 0)
        self.assertTrue(np.all(degree > 0.0))
        eigenvalue = np.linalg.eigvalsh(connection.toarray())
        self.assertLessEqual(float(np.max(np.abs(eigenvalue))), 1.0 + 1e-12)
        sources = region_source_matrix(
            bundle["incidence"]["region"], np.arange(4))
        gram, similarity = connection_heat_gram(connection, sources)
        np.testing.assert_allclose(gram, gram.T, atol=1e-14, rtol=0.0)
        np.testing.assert_allclose(
            np.diag(similarity), 1.0, atol=1e-14, rtol=0.0)
        self.assertTrue(np.all(np.isfinite(similarity)))
        green, green_similarity, resistance, solver = connection_green_gram(
            connection, sources)
        np.testing.assert_allclose(green, green.T, atol=1e-11, rtol=0.0)
        np.testing.assert_allclose(
            resistance, resistance.T, atol=1e-11, rtol=0.0)
        np.testing.assert_allclose(
            np.diag(resistance), 0.0, atol=1e-11, rtol=0.0)
        self.assertTrue(np.all(resistance >= 0.0))
        self.assertTrue(np.all(np.isfinite(green_similarity)))
        self.assertIn("projected_null_mode", solver)

    def test_one_sided_contours_preserve_owner_and_link_opposites(self):
        rgb, result = _fixture()
        complex_ = build_region_complex(result, rgb, level="leaves")
        bundle = build_incidence_bundle(complex_)
        contour = build_contour_transport(complex_, bundle)
        component = contour["incidence_component"]
        owner = contour["component_owner"]
        for identifier, region in enumerate(bundle["incidence"]["region"]):
            self.assertEqual(owner[component[identifier]], region)
        kernel = contour["region_kernel"]
        np.testing.assert_allclose(kernel, kernel.T, atol=0.0, rtol=0.0)
        self.assertTrue(np.all(np.linalg.eigvalsh(kernel) >= -1e-12))
        # Regions 1 and 2 meet as opposite states around both region 0 and 3.
        self.assertGreater(kernel[1, 2], 0.0)
        summary = summarize_contour_transport(contour)
        self.assertGreater(summary["multi_arc_components"], 0)

    def test_relative_enclosure_excludes_frame_component(self):
        complex_ = {
            "region_count": 2,
            "arc": {
                "cell_first": np.asarray([0], dtype=np.int32),
                "cell_second": np.asarray([1], dtype=np.int32),
            },
            "node": {
                "touches_frame": np.asarray([True, False]),
                "area": np.asarray([24.0, 1.0]),
            },
        }
        enclosure = build_relative_enclosures(complex_)
        np.testing.assert_array_equal(enclosure["manifold_owner"], [0])
        np.testing.assert_array_equal(enclosure["manifold_member"], [1])
        self.assertEqual(enclosure["region_kernel"][0, 0], 0.0)
        self.assertEqual(enclosure["region_kernel"][1, 1], 1.0)
        summary = summarize_relative_enclosures(enclosure)
        self.assertEqual(summary["bounded_manifolds"], 1)

    def test_complete_participation_algebra_is_positive(self):
        embedding = np.asarray([
            [1.0, 0.0], [0.7, 0.3], [0.0, 1.0],
        ])
        role = normalized_linear_kernel(embedding)
        contour_feature = np.asarray([[1.0], [1.0], [0.0]])
        enclosure_feature = np.asarray([[0.0], [1.0], [1.0]])
        contour = normalized_linear_kernel(contour_feature)
        enclosure = normalized_linear_kernel(enclosure_feature)
        algebra = complete_participation_kernel(role, contour, enclosure)
        complete = algebra["complete"]
        np.testing.assert_allclose(complete, complete.T, atol=0.0, rtol=0.0)
        self.assertGreaterEqual(
            float(np.min(np.linalg.eigvalsh(complete))), -1e-12)
        np.testing.assert_allclose(
            np.diag(complete), 1.0, atol=1e-14, rtol=0.0)
        extended = complete_kernel_algebra({
            "role": role,
            "contour": normalized_linear_kernel(contour_feature),
            "enclosure": normalized_linear_kernel(enclosure_feature),
            "duplicate": role,
        })
        self.assertIn("role_contour_enclosure_duplicate", extended)
        self.assertGreaterEqual(
            float(np.min(np.linalg.eigvalsh(extended["complete"]))), -1e-12)

    def test_t_junction_cap_is_oriented_without_object_label(self):
        labels = np.asarray([[0, 0], [1, 2]], dtype=np.int32)
        rgb = np.zeros((2, 2, 3), dtype=np.float64)
        target = rgb.copy()
        result = {
            "compound_segmentation": {
                "enabled": True, "labels": labels, "leaf_labels": labels,
            },
            "labels": labels,
            "target_lab": target,
            "cartoon_lab": target,
            "texture_target_lab": target,
            "texture_fit_lab": target,
            "texture_geometry": {
                "boundary_confidence": np.zeros((2, 2)),
            },
        }
        complex_ = build_region_complex(result, rgb, level="leaves")
        depth = build_junction_depth(complex_)
        self.assertEqual(len(depth["junction"]), 1)
        self.assertEqual(depth["cap_region"][0], 0)
        self.assertEqual(depth["cap_sector_count"][0], 2)
        self.assertEqual(depth["sector_count"][0], 3)
        np.testing.assert_array_equal(depth["other_region"], [1, 2])
        self.assertEqual(len(depth["cap_arc"]), 2)
        self.assertAlmostEqual(depth["cap_tangent_continuation"][0], 1.0)
        summary = summarize_junction_depth(depth)
        self.assertEqual(summary["classical_t_records"], 1)

    def test_depth_contour_keeps_t_and_focus_as_separate_coordinates(self):
        labels = np.asarray([[0, 0], [1, 2]], dtype=np.int32)
        rgb = np.zeros((2, 2, 3), dtype=np.float64)
        result = {
            "compound_segmentation": {
                "enabled": True, "labels": labels, "leaf_labels": labels,
            },
            "labels": labels,
            "target_lab": rgb,
            "cartoon_lab": rgb,
            "texture_target_lab": rgb,
            "texture_fit_lab": rgb,
            "texture_geometry": {
                "boundary_confidence": np.zeros((2, 2)),
            },
        }
        complex_ = build_region_complex(result, rgb, level="leaves")
        bundle = build_incidence_bundle(complex_)
        contour = build_contour_transport(complex_, bundle)
        depth = build_junction_depth(complex_)
        arc_count = int(complex_["topology"]["arc"]["count"])
        focus = {
            "first_match_margin": np.linspace(-0.2, 0.2, arc_count),
            "reliability": np.ones(arc_count),
        }
        lifted = build_depth_contour_transport(
            complex_, bundle, contour, depth, focus)
        self.assertEqual(
            len(lifted["component_cap_junction_count"]),
            contour["component_count"],
        )
        self.assertTrue(np.any(lifted["component_cap_junction_count"] > 0))
        self.assertTrue(np.all(np.isfinite(
            lifted["component_focus_match_margin"])))

    def test_closed_contour_emits_exact_winding_support(self):
        labels = np.zeros((5, 5), dtype=np.int32)
        labels[1:4, 1:4] = 1
        labels[2, 2] = 2
        rgb = np.zeros((5, 5, 3), dtype=np.float64)
        result = {
            "compound_segmentation": {
                "enabled": True, "labels": labels, "leaf_labels": labels,
            },
            "labels": labels,
            "target_lab": rgb,
            "cartoon_lab": rgb,
            "texture_target_lab": rgb,
            "texture_fit_lab": rgb,
            "texture_geometry": {
                "boundary_confidence": np.zeros((5, 5)),
            },
        }
        complex_ = build_region_complex(result, rgb, level="leaves")
        bundle = build_incidence_bundle(complex_)
        contour = build_contour_transport(complex_, bundle)
        nesting = build_contour_cycle_nesting(complex_, bundle, contour)
        self.assertGreaterEqual(
            nesting["overlap_participation"].shape[0], 2)
        self.assertGreater(nesting["overlap_kernel"][1, 2], 0.0)
        np.testing.assert_allclose(
            nesting["centered_kernel"].toarray(),
            nesting["centered_kernel"].T.toarray())

    def test_compositional_controls_remain_positive_kernels(self):
        first = normalized_linear_kernel(np.asarray([
            [1.0, 0.0], [0.8, 0.2], [0.0, 1.0],
        ]))
        second = normalized_linear_kernel(np.asarray([
            [1.0, 0.0], [0.0, 1.0], [0.7, 0.3],
        ]))
        for kernel in (
            spectral_exponential_bloom(first),
            typed_order_two_bloom((first, second)),
        ):
            np.testing.assert_allclose(kernel, kernel.T, atol=1e-13)
            self.assertGreaterEqual(
                float(np.min(np.linalg.eigvalsh(kernel))), -1e-12)
            np.testing.assert_allclose(np.diag(kernel), 1.0, atol=1e-13)

    def test_amodal_port_keeps_cap_out_of_background_participation(self):
        labels = np.asarray([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 1, 2, 2, 3, 3],
            [1, 1, 2, 2, 3, 3],
            [1, 1, 2, 2, 3, 3],
            [1, 1, 2, 2, 3, 3],
        ], dtype=np.int32)
        rgb = np.zeros((6, 6, 3), dtype=np.float64)
        rgb[labels == 1] = (0.2, 0.3, 0.4)
        rgb[labels == 2] = (0.8, 0.7, 0.6)
        result = {
            "compound_segmentation": {
                "enabled": True, "labels": labels, "leaf_labels": labels,
            },
            "labels": labels,
            "target_lab": rgb,
            "cartoon_lab": rgb,
            "texture_target_lab": np.zeros_like(rgb),
            "texture_fit_lab": np.zeros_like(rgb),
            "texture_geometry": {
                "boundary_confidence": np.zeros((6, 6)),
            },
        }
        complex_ = build_region_complex(result, rgb, level="leaves")
        bundle = build_incidence_bundle(complex_)
        contour = build_contour_transport(complex_, bundle)
        depth = build_junction_depth(complex_)
        ports = extract_amodal_ports(complex_, contour, depth)
        hodge = build_depth_hodge(ports, complex_["region_count"])
        self.assertGreaterEqual(float(hodge["explained_fraction"]), 0.0)
        pair, residual, names = amodal_pair_residuals(
            ports, labels, candidate_mode="contour_delaunay",
            port_depth_agreement=hodge["port_agreement"])
        self.assertGreaterEqual(len(pair["first_port"]), 1)
        whitener = fit_zero_whitener((residual,), names)
        transport = build_amodal_transport(
            pair, residual, whitener, complex_["region_count"])
        cap = int(pair["cap_region_first"][0])
        participant = np.unique(np.concatenate((
            pair["first_left_region"], pair["first_right_region"],
            pair["second_left_region"], pair["second_right_region"],
        )))
        self.assertTrue(np.all(
            transport["region_kernel"][cap, participant] == 0.0))

    def test_support_manifold_transport_is_positive_and_seed_free(self):
        labels = np.zeros((5, 5), dtype=np.int32)
        labels[1:4, 1:4] = 1
        labels[2, 2] = 2
        rgb = np.zeros((5, 5, 3), dtype=np.float64)
        result = {
            "compound_segmentation": {
                "enabled": True, "labels": labels, "leaf_labels": labels,
            },
            "labels": labels,
            "target_lab": rgb,
            "cartoon_lab": rgb,
            "texture_target_lab": rgb,
            "texture_fit_lab": rgb,
            "texture_geometry": {
                "boundary_confidence": np.zeros((5, 5)),
            },
        }
        complex_ = build_region_complex(result, rgb, level="leaves")
        enclosure = build_relative_enclosures(complex_)
        transport = build_support_manifold_transport(complex_, enclosure)
        kernel = transport["region_kernel"]
        np.testing.assert_allclose(kernel, kernel.T, atol=1e-13)
        self.assertGreaterEqual(float(np.min(np.linalg.eigvalsh(kernel))), -1e-12)
        self.assertTrue(np.all(np.isfinite(
            transport["support_manifold_weight"])))


if __name__ == "__main__":
    unittest.main()
