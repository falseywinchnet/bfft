from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from experiments.ostensibly_frontend.word_lattice import (
    PhoneEvidence,
    BoundaryPairEvidence,
    ProvenancePhoneEvidence,
    PronunciationProposalIndex,
    Pronunciation,
    PronunciationIndex,
    acoustic_edit_cost,
    boundary_evidence_from_span_result,
    deletion_keys,
    full_key,
    k_best_phone_sequences,
    parse_cmudict,
    proposal_sequence_audit,
    provenance_edit_cost,
    provenance_evidence_from_context_result,
    rank_pronunciation_oracle,
    relational_sequence_cost,
)
from experiments.ostensibly_frontend.run_synthetic_word_geometry_audit import (
    build_phone_prototype,
    build_word_ensemble,
    build_word_realizations,
    quantile_affine_registration,
    quantile_affine_residual,
    ordered_lane_windows,
    select_coherent_pair_witnesses,
    stitch_pair_witnesses,
)
from experiments.ostensibly_frontend.run_stream_word_lattice import (
    policy_cache_path,
    proposal_channel_union,
    proposal_channels_union,
    selected_proposal_channels,
    stratified_generic_ranks,
)


class WordLatticeTests(unittest.TestCase):
    def test_proposal_channel_union_retains_independent_winners(self) -> None:
        route = [("A",), ("B",), ("C",)]
        generic = [("C",), ("D",), ("A",)]
        self.assertEqual(
            proposal_channel_union(route, generic, 2),
            (("A",), ("B",), ("C",), ("D",)),
        )

    def test_multi_policy_union_and_cache_paths_preserve_provenance(self) -> None:
        self.assertEqual(
            proposal_channels_union(
                [
                    [("A",), ("B",), ("C",)],
                    [("C",), ("D",), ("A",)],
                    [("E",), ("B",)],
                ],
                2,
            ),
            (("A",), ("B",), ("C",), ("D",), ("E",)),
        )
        base = Path("cache.npz")
        self.assertEqual(policy_cache_path(base, "maximum", False), base)
        self.assertEqual(
            policy_cache_path(base, "maximum", True),
            Path("cache.maximum.npz"),
        )
        self.assertEqual(
            selected_proposal_channels(
                19,
                {"unconditional": 3, "maximum": 11, "mixed": None},
                8,
            ),
            ["generic_word:unconditional"],
        )

    def test_generic_ranks_remain_local_to_support_strata(self) -> None:
        rows = [
            (0.10, 5, ("A",)),
            (0.20, 6, ("B",)),
            (0.30, 7, ("C",)),
        ]
        strata = {
            ("A",): (1, 0, 0, 0),
            ("B",): (1, 1, 0, 0),
            ("C",): (1, 0, 0, 0),
        }
        local, global_ranks, ordered = stratified_generic_ranks(rows, strata)
        self.assertEqual(local, {("A",): 1, ("B",): 1, ("C",): 2})
        self.assertEqual(global_ranks[("B",)], 2)
        self.assertEqual(ordered, rows)

    def setUp(self) -> None:
        self.alphabet = ("AH", "B", "K", "T")
        self.pronunciations = (
            Pronunciation("bat", ("B", "AH", "T")),
            Pronunciation("cat", ("K", "AH", "T")),
            Pronunciation("at", ("AH", "T")),
        )
        self.index = PronunciationIndex(self.pronunciations, self.alphabet)

    def test_rolling_deletions_equal_materialized_keys(self) -> None:
        phones = ("B", "AH", "T")
        expected = tuple(
            full_key(phones[:i] + phones[i + 1 :], self.index.atom_codes)
            for i in range(len(phones))
        )
        self.assertEqual(deletion_keys(phones, self.index.atom_codes), expected)

    def test_typed_plans_recall_radius_one(self) -> None:
        ordinal = 0
        for query in (
            ("B", "AH", "T"),
            ("K", "AH", "T"),
            ("B", "T"),
            ("B", "K", "AH", "T"),
        ):
            self.assertIn(ordinal, self.index.query(query).candidates)

    def test_parser_keeps_homophones_and_removes_stress(self) -> None:
        entries = parse_cmudict(
            ["there DH EH1 R", "their DH EH1 R", "bad ZZ1"],
            frozenset(("DH", "EH", "R")),
        )
        self.assertEqual({entry.word for entry in entries}, {"there", "their"})
        self.assertEqual({entry.phones for entry in entries}, {("DH", "EH", "R")})

    def test_grouped_alignment_recovers_fragmented_three_phone_word(self) -> None:
        index = PronunciationProposalIndex(
            (Pronunciation("okay", ("OW", "K", "EY")),)
        )
        evidence = tuple(
            ProvenancePhoneEvidence({"ridge": {phone: 1}}, 8)
            for phone in ("OW", "OW", "K", "K", "EY", "EY")
        )
        boundaries = tuple(
            BoundaryPairEvidence(
                ranks,
                max(ranks.values(), default=128),
            )
            for ranks in (
                {},
                {("OW", "K"): 1},
                {},
                {("K", "EY"): 1},
                {},
            )
        )
        route = index.query_grouped_aligned(
            evidence,
            boundaries,
            maximum_length_delta=3,
            maximum_boundary_rank=8,
        )
        self.assertEqual(route.candidates, frozenset((0,)))
        self.assertEqual(route.minimum_edits[0], 3)

    def test_grouped_alignment_can_retain_a_label_blind_geometry_candidate(self) -> None:
        index = PronunciationProposalIndex(
            (Pronunciation("okay", ("OW", "K", "EY")),)
        )
        evidence = tuple(
            ProvenancePhoneEvidence({"ridge": ranks}, 8)
            for ranks in (
                {"OW": 1},
                {"OW": 2},
                {"T": 1},
                {"AH": 1},
                {"N": 1},
                {"R": 1},
            )
        )
        boundaries = tuple(BoundaryPairEvidence({}, 128) for _ in range(5))
        route = index.query_grouped_aligned(
            evidence,
            boundaries,
            maximum_uncovered_phones=2,
            maximum_uncovered_boundaries=2,
            maximum_length_delta=3,
            maximum_boundary_rank=128,
        )
        self.assertEqual(route.candidates, frozenset((0,)))
        self.assertEqual(route.costs[0].uncovered, 2)

    def test_k_best_sequence_and_acoustic_score(self) -> None:
        evidence = (
            PhoneEvidence({"B": 0.0, "K": 0.8}, 2.0),
            PhoneEvidence({"AH": 0.0}, 2.0),
            PhoneEvidence({"T": 0.0}, 2.0),
        )
        beams = k_best_phone_sequences(evidence, 2)
        self.assertEqual(beams[0][0], ("B", "AH", "T"))
        self.assertLess(
            acoustic_edit_cost(evidence, ("B", "AH", "T")),
            acoustic_edit_cost(evidence, ("K", "AH", "T")),
        )

    def test_provenance_cost_keeps_channels_independent(self) -> None:
        evidence = (
            ProvenancePhoneEvidence(
                {"shape": {"B": 1, "K": 2}, "boundary": {"B": 2}}, 2
            ),
            ProvenancePhoneEvidence({"shape": {"AH": 1}}, 2),
            ProvenancePhoneEvidence({"shape": {"T": 1}}, 2),
        )
        bat = provenance_edit_cost(evidence, ("B", "AH", "T"))
        cat = provenance_edit_cost(evidence, ("K", "AH", "T"))
        self.assertLess(bat, cat)
        self.assertEqual(bat.negative_channel_support, -4)

    def test_provenance_oracle_preserves_homophone_anchor(self) -> None:
        evidence = tuple(
            ProvenancePhoneEvidence({"channel": {phone: 1}}, 3)
            for phone in ("B", "AH", "T")
        )
        entries = self.pronunciations + (
            Pronunciation("batt", ("B", "AH", "T")),
        )
        ranked = rank_pronunciation_oracle(evidence, entries)
        self.assertEqual(ranked[0].phones, ("B", "AH", "T"))
        self.assertEqual(set(ranked[0].words), {"bat", "batt"})

    def test_context_result_lift_uses_ranks_not_distances(self) -> None:
        ranking = [
            {"phone": "B", "distance": 1000.0},
            {"phone": "K", "distance": -50.0},
        ]
        document = {
            field: list(ranking)
            for field in (
                "center_label_ranking",
                "context_label_ranking",
                "relational_label_ranking",
                "support_label_ranking",
            )
        }
        lifted = provenance_evidence_from_context_result(document, 1)
        self.assertEqual(
            lifted.channel_ranks,
            {
                "center": {"B": 1},
                "ordered_context": {"B": 1},
                "boundary_transport": {"B": 1},
                "support_geometry": {"B": 1},
            },
        )

    def test_relational_cost_uses_phone_proposals_as_screen(self) -> None:
        accepted = provenance_edit_cost(
            (
                ProvenancePhoneEvidence({"c": {"B": 2, "K": 1}}, 3),
                ProvenancePhoneEvidence({"c": {"AH": 1}}, 3),
                ProvenancePhoneEvidence({"c": {"T": 1}}, 3),
            ),
            ("B", "AH", "T"),
        )
        boundaries = (
            BoundaryPairEvidence({("B", "AH"): 1}, 4),
            BoundaryPairEvidence({("AH", "T"): 1}, 4),
        )
        correct = relational_sequence_cost(accepted, ("B", "AH", "T"), boundaries)
        rejected = relational_sequence_cost(accepted, ("K", "AH", "T"), boundaries)
        self.assertLess(correct, rejected)

    def test_boundary_atlas_lift_preserves_pair_ranks(self) -> None:
        document = {
            "boundary_pair_rankings": [
                {
                    "ranking": [
                        {"phones": ["B", "AH"], "rank": 1},
                        {"phones": ["K", "AH"], "rank": 2},
                    ]
                }
            ]
        }
        evidence = boundary_evidence_from_span_result(document)
        self.assertEqual(evidence[0].ranks[("K", "AH")], 2)

    def test_positional_proposal_index_matches_exhaustive_coverage(self) -> None:
        evidence = (
            ProvenancePhoneEvidence({"c": {"B": 1, "K": 2}}, 2),
            ProvenancePhoneEvidence({"c": {"AH": 1}}, 2),
            ProvenancePhoneEvidence({"c": {"T": 1}}, 2),
        )
        boundaries = (
            BoundaryPairEvidence({("B", "AH"): 1}, 3),
            BoundaryPairEvidence({("AH", "T"): 1}, 3),
        )
        index = PronunciationProposalIndex(self.pronunciations)
        route = index.query(evidence, boundaries)
        phone_sequences = {
            index.classes[ordinal].phones for ordinal in route.phone_candidates
        }
        boundary_sequences = {
            index.classes[ordinal].phones for ordinal in route.boundary_candidates
        }
        self.assertEqual(phone_sequences, {("B", "AH", "T"), ("K", "AH", "T")})
        self.assertEqual(boundary_sequences, {("B", "AH", "T")})

    def test_positional_proposal_index_recovers_one_uncovered_phone(self) -> None:
        evidence = (
            ProvenancePhoneEvidence({"c": {"B": 1}}, 2),
            ProvenancePhoneEvidence({"c": {"K": 1}}, 2),
            ProvenancePhoneEvidence({"c": {"T": 1}}, 2),
        )
        boundaries = (
            BoundaryPairEvidence({("B", "AH"): 1}, 3),
            BoundaryPairEvidence({("AH", "T"): 1}, 3),
        )
        index = PronunciationProposalIndex(self.pronunciations)
        strict = index.query(evidence, boundaries)
        tolerant = index.query(
            evidence, boundaries, maximum_uncovered_phones=1
        )
        self.assertNotIn(
            ("B", "AH", "T"),
            {index.classes[ordinal].phones for ordinal in strict.boundary_candidates},
        )
        self.assertIn(
            ("B", "AH", "T"),
            {
                index.classes[ordinal].phones
                for ordinal in tolerant.boundary_candidates
            },
        )

    def test_positional_proposal_index_honors_boundary_rank_cap(self) -> None:
        evidence = (
            ProvenancePhoneEvidence({"c": {"B": 1, "K": 2}}, 2),
            ProvenancePhoneEvidence({"c": {"AH": 1}}, 2),
            ProvenancePhoneEvidence({"c": {"T": 1}}, 2),
        )
        boundaries = (
            BoundaryPairEvidence({("K", "AH"): 1, ("B", "AH"): 2}, 3),
            BoundaryPairEvidence({("AH", "T"): 1}, 3),
        )
        index = PronunciationProposalIndex(self.pronunciations)
        route = index.query(evidence, boundaries, maximum_boundary_rank=1)
        self.assertEqual(
            {
                index.classes[ordinal].phones
                for ordinal in route.boundary_candidates
            },
            {("K", "AH", "T")},
        )

    def test_sequence_audit_distinguishes_phone_and_boundary_rejection(self) -> None:
        evidence = (
            ProvenancePhoneEvidence({"c": {"B": 1}}, 2),
            ProvenancePhoneEvidence({"c": {"K": 1}}, 2),
            ProvenancePhoneEvidence({"c": {"T": 1}}, 2),
        )
        boundaries = (
            BoundaryPairEvidence({("K", "AH"): 1}, 3),
            BoundaryPairEvidence({("AH", "T"): 2}, 3),
        )
        audit = proposal_sequence_audit(
            evidence,
            boundaries,
            ("B", "AH", "T"),
            maximum_uncovered_phones=1,
            maximum_boundary_rank=1,
        )
        self.assertEqual(audit.phone_position_admitted, (True, False, True))
        self.assertTrue(audit.phone_stage_admitted)
        self.assertEqual(audit.boundary_ranks, (None, 2))
        self.assertEqual(audit.boundary_position_admitted, (False, False))
        self.assertFalse(audit.boundary_stage_admitted)

    def test_aligned_proposal_route_recovers_one_missing_or_surplus_region(self) -> None:
        evidence = tuple(
            ProvenancePhoneEvidence({"c": {phone: 1}}, 2)
            for phone in ("B", "T")
        )
        boundary = (BoundaryPairEvidence({("B", "T"): 1}, 1),)
        index = PronunciationProposalIndex(self.pronunciations)
        missing = index.query_aligned(
            evidence,
            boundary,
            maximum_uncovered_phones=0,
            maximum_boundary_rank=1,
        )
        self.assertIn(
            ("B", "AH", "T"),
            {index.classes[item].phones for item in missing.candidates},
        )

        surplus_evidence = tuple(
            ProvenancePhoneEvidence({"c": {phone: 1}}, 2)
            for phone in ("B", "K", "AH", "T")
        )
        surplus_boundaries = (
            BoundaryPairEvidence({("B", "K"): 1}, 1),
            BoundaryPairEvidence({("K", "AH"): 1}, 1),
            BoundaryPairEvidence({("AH", "T"): 1}, 1),
        )
        surplus = index.query_aligned(
            surplus_evidence,
            surplus_boundaries,
            maximum_uncovered_phones=0,
            maximum_boundary_rank=1,
        )
        self.assertIn(
            ("B", "AH", "T"),
            {index.classes[item].phones for item in surplus.candidates},
        )

    def test_aligned_route_retains_one_uncertain_boundary_as_cost(self) -> None:
        evidence = tuple(
            ProvenancePhoneEvidence({"c": {phone: 1}}, 2)
            for phone in ("B", "AH", "T")
        )
        boundaries = (
            BoundaryPairEvidence({("B", "AH"): 1}, 2),
            BoundaryPairEvidence({("K", "T"): 1}, 2),
        )
        index = PronunciationProposalIndex(self.pronunciations)
        strict = index.query_aligned(
            evidence,
            boundaries,
            maximum_boundary_rank=1,
            maximum_length_delta=0,
        )
        tolerant = index.query_aligned(
            evidence,
            boundaries,
            maximum_boundary_rank=1,
            maximum_length_delta=0,
            maximum_uncovered_boundaries=1,
        )
        target = next(
            ordinal
            for ordinal, item in enumerate(index.classes)
            if item.phones == ("B", "AH", "T")
        )
        self.assertNotIn(target, strict.candidates)
        self.assertIn(target, tolerant.candidates)
        self.assertEqual(
            tolerant.relational_costs[target].uncovered_transitions, 1
        )

    def test_phone_prototype_equalizes_example_mass(self) -> None:
        frame = np.linspace(0.0, 1.0, 20)
        cloud = np.column_stack((frame, frame, np.ones_like(frame)))
        prototype = build_phone_prototype((cloud,), 4, 10)
        self.assertEqual(prototype.shape, (40, 3))

    def test_word_ensemble_superimposes_complete_realizations(self) -> None:
        frame = np.linspace(0.0, 1.0, 20)
        low = np.column_stack((0.2 + frame, frame, np.ones_like(frame)))
        high = np.column_stack((2.0 + frame, frame, np.ones_like(frame)))
        ensemble = build_word_ensemble(
            ((low,), (high,)), realization_count=4, points_per_slot=10
        )
        realizations = build_word_realizations(
            ((low,), (high,)), realization_count=4, points_per_slot=10
        )
        self.assertEqual(len(realizations), 4)
        self.assertEqual(ensemble.shape, (80, 3))
        for realization in range(4):
            left = ensemble[realization * 20 : realization * 20 + 10]
            right = ensemble[realization * 20 + 10 : realization * 20 + 20]
            self.assertLess(float(np.max(left[:, 0])), float(np.min(right[:, 0])))

    def test_quantile_affine_registration_recovers_known_chart(self) -> None:
        source = np.linspace(-1.0, 1.0, 200)
        target = 2.5 * source + 0.7
        scale, shift = quantile_affine_registration(source, target)
        self.assertAlmostEqual(scale, 2.5, places=10)
        self.assertAlmostEqual(shift, 0.7, places=10)

    def test_coherent_witness_path_rejects_incompatible_local_minima(self) -> None:
        frame = np.linspace(0.0, 1.0, 200)
        affine_family = np.column_stack((frame, frame, np.ones_like(frame)))
        curved_family = np.column_stack((frame**3, frame, np.ones_like(frame)))
        clouds = {
            "a": (affine_family, affine_family),
            "b": (curved_family, curved_family),
            "c": (affine_family, affine_family),
            "d": (curved_family, curved_family),
        }
        options = (
            (("a", 0.0), ("b", 0.2)),
            (("c", 0.2), ("d", 0.0)),
        )
        self.assertEqual(
            select_coherent_pair_witnesses(options, clouds.__getitem__, 0.0),
            ("a", "d"),
        )
        coherent = select_coherent_pair_witnesses(
            options, clouds.__getitem__, coherence_weight=100.0
        )
        self.assertIn(coherent, (("a", "c"), ("b", "d")))
        self.assertLess(
            quantile_affine_residual(
                clouds[coherent[0]][1][:, 0], clouds[coherent[1]][0][:, 0]
            ),
            quantile_affine_residual(clouds["a"][1][:, 0], clouds["d"][0][:, 0]),
        )

    def test_pair_witness_stitching_preserves_phone_lanes(self) -> None:
        frame = np.linspace(0.0, 1.0, 100)
        first = np.column_stack((0.1 + frame, frame, np.ones_like(frame)))
        overlap_a = np.column_stack((0.5 + frame, frame, np.ones_like(frame)))
        overlap_b = np.column_stack((2.0 + 2.0 * frame, frame, np.ones_like(frame)))
        last = np.column_stack((3.0 + 2.0 * frame, frame, np.ones_like(frame)))
        stitched = stitch_pair_witnesses(
            ((first, overlap_a), (overlap_b, last)), 64
        )
        self.assertEqual(stitched.shape, (192, 3))
        for slot in range(3):
            lane = stitched[slot * 64 : (slot + 1) * 64, 1]
            self.assertGreaterEqual(float(np.min(lane)), slot / 3)
            self.assertLessEqual(float(np.max(lane)), (slot + 1) / 3)

    def test_ordered_lane_windows_rebase_only_time(self) -> None:
        frame = (np.arange(30, dtype=np.float64) + 0.5) / 30
        cloud = np.column_stack((frame**2, frame, np.ones_like(frame)))
        windows = ordered_lane_windows(cloud, phone_count=3, window_width=2)
        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0].shape, (20, 3))
        self.assertEqual(windows[1].shape, (20, 3))
        self.assertTrue(np.allclose(windows[0][:, 0], cloud[:20, 0]))
        self.assertGreaterEqual(float(np.min(windows[0][:, 1])), 0.0)
        self.assertLessEqual(float(np.max(windows[0][:, 1])), 1.0)


if __name__ == "__main__":
    unittest.main()
