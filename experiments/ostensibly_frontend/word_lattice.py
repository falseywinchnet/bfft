"""Exact-anchor pronunciation routing and acoustic word-edge scoring.

The history-occupancy address is adapted from the older Kolmogrov filename
retrieval experiment.  It is only a candidate generator.  CMUdict
pronunciations remain the exact anchors, and every proposed pronunciation is
scored again against the complete phone top-k evidence with dynamic
programming.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from itertools import combinations
import re
from typing import Iterable, Mapping, Sequence


FINGERPRINT_MODULUS = (1 << 61) - 1
PHONE_RE = re.compile(r"[0-2]$")
ALTERNATE_RE = re.compile(r"\([0-9]+\)$")
ARPABET_39 = (
    "IY", "IH", "EH", "AE", "AA", "AO", "UH", "UW", "AH", "ER",
    "EY", "AY", "OW", "AW", "OY", "P", "B", "T", "D", "K", "G",
    "CH", "JH", "F", "V", "TH", "DH", "S", "Z", "SH", "ZH", "HH",
    "M", "N", "NG", "L", "R", "W", "Y",
)


@dataclass(frozen=True)
class Pronunciation:
    word: str
    phones: tuple[str, ...]


@dataclass(frozen=True)
class OccupancyView:
    bucket_count: int
    base: int
    multiplier: int
    modulus: int = FINGERPRINT_MODULUS


@dataclass(frozen=True)
class PhoneEvidence:
    costs: Mapping[str, float]
    unseen_cost: float


@dataclass(frozen=True)
class CandidateAudit:
    candidates: frozenset[int]
    exact_anchor: bool
    posting_reads: int


@dataclass(frozen=True)
class ProvenancePhoneEvidence:
    """Per-phone proposal ranks retained independently by evidence channel."""

    channel_ranks: Mapping[str, Mapping[str, int]]
    maximum_admitted_rank: int


@dataclass(frozen=True, order=True)
class SequenceEvidenceCost:
    """Lexicographic word cost; fields are ordered from fatal to tie-breaker."""

    uncovered: int = 0
    edits: int = 0
    worst_rank: int = 0
    rank_sum: int = 0
    negative_channel_support: int = 0

    def combine(self, other: "SequenceEvidenceCost") -> "SequenceEvidenceCost":
        return SequenceEvidenceCost(
            uncovered=self.uncovered + other.uncovered,
            edits=self.edits + other.edits,
            worst_rank=max(self.worst_rank, other.worst_rank),
            rank_sum=self.rank_sum + other.rank_sum,
            negative_channel_support=(
                self.negative_channel_support + other.negative_channel_support
            ),
        )


@dataclass(frozen=True)
class PronunciationClassScore:
    phones: tuple[str, ...]
    words: tuple[str, ...]
    cost: SequenceEvidenceCost


@dataclass(frozen=True)
class BoundaryPairEvidence:
    ranks: Mapping[tuple[str, str], int]
    maximum_rank: int


@dataclass(frozen=True, order=True)
class RelationalSequenceCost:
    """Phone proposals screen first; ordered boundary geometry then ranks."""

    uncovered_phones: int
    uncovered_transitions: int
    edits: int
    worst_transition_rank: int
    transition_rank_sum: int
    worst_phone_rank: int
    phone_rank_sum: int
    negative_channel_support: int


@dataclass(frozen=True)
class PronunciationClass:
    phones: tuple[str, ...]
    words: tuple[str, ...]


@dataclass(frozen=True)
class ProposalRouteAudit:
    phone_candidates: frozenset[int]
    boundary_candidates: frozenset[int]
    phone_posting_reads: int
    boundary_posting_reads: int


@dataclass(frozen=True)
class AlignedProposalRouteAudit:
    candidates: frozenset[int]
    minimum_edits: Mapping[int, int]
    costs: Mapping[int, SequenceEvidenceCost]
    relational_costs: Mapping[int, RelationalSequenceCost]
    alignment_plan_count: int
    phone_posting_reads: int
    boundary_posting_reads: int


@dataclass(frozen=True)
class ProposalSequenceAudit:
    """Explain how one exact phone sequence passes through proposal routing."""

    phone_channel_ranks: tuple[Mapping[str, int], ...]
    phone_position_admitted: tuple[bool, ...]
    admitted_phone_count: int
    phone_stage_admitted: bool
    boundary_ranks: tuple[int | None, ...]
    boundary_position_admitted: tuple[bool, ...]
    boundary_stage_admitted: bool


DEFAULT_VIEWS = (
    OccupancyView(1 << 20, 1_000_003, 1_103_515_245),
    OccupancyView(1 << 20, 1_000_033, 2_654_435_761),
)


CONTEXT_EVIDENCE_CHANNELS = {
    "center": "center_label_ranking",
    "ordered_context": "context_label_ranking",
    "boundary_transport": "relational_label_ranking",
    "support_geometry": "support_label_ranking",
}


def provenance_evidence_from_context_result(
    document: Mapping[str, object],
    top_k_per_channel: int = 8,
) -> ProvenancePhoneEvidence:
    """Lift one context-battery result without fusing its channel distances."""

    if top_k_per_channel < 1:
        raise ValueError("top-k per channel must be positive")
    channel_ranks: dict[str, dict[str, int]] = {}
    for channel, field in CONTEXT_EVIDENCE_CHANNELS.items():
        ranking = document.get(field)
        if not isinstance(ranking, list) or not ranking:
            raise ValueError(f"missing context evidence channel {field!r}")
        channel_ranks[channel] = {
            str(item["phone"]): rank
            for rank, item in enumerate(ranking[:top_k_per_channel], start=1)
        }
    return ProvenancePhoneEvidence(channel_ranks, top_k_per_channel)


def boundary_evidence_from_span_result(
    document: Mapping[str, object],
    atlas_field: str = "boundary_pair_rankings",
) -> tuple[BoundaryPairEvidence, ...]:
    boundaries = document.get(atlas_field)
    if not isinstance(boundaries, list) or not boundaries:
        raise ValueError("span result has no boundary-pair atlas")
    output = []
    for boundary in boundaries:
        ranking = boundary["ranking"]
        ranks = {
            tuple(str(phone) for phone in item["phones"]): int(item["rank"])
            for item in ranking
        }
        output.append(BoundaryPairEvidence(ranks, len(ranking)))
    return tuple(output)


def proposal_sequence_audit(
    evidence: Sequence[ProvenancePhoneEvidence],
    boundaries: Sequence[BoundaryPairEvidence],
    phones: Sequence[str],
    maximum_uncovered_phones: int = 0,
    maximum_boundary_rank: int | None = None,
    maximum_uncovered_boundaries: int = 0,
) -> ProposalSequenceAudit:
    """Return the exact per-position reasons a sequence is routed or rejected."""

    if not evidence or len(phones) != len(evidence):
        raise ValueError("proposal audit needs one evidence item per phone")
    if len(boundaries) != len(phones) - 1:
        raise ValueError("proposal audit needs one boundary between phones")
    if maximum_uncovered_phones < 0:
        raise ValueError("maximum uncovered phones cannot be negative")
    if maximum_boundary_rank is not None and maximum_boundary_rank < 1:
        raise ValueError("maximum boundary rank must be positive")
    if maximum_uncovered_boundaries < 0:
        raise ValueError("maximum uncovered boundaries cannot be negative")

    phone_channel_ranks = tuple(
        {
            channel: int(ranks[phone])
            for channel, ranks in item.channel_ranks.items()
            if phone in ranks
        }
        for phone, item in zip(phones, evidence, strict=True)
    )
    phone_position_admitted = tuple(bool(ranks) for ranks in phone_channel_ranks)
    admitted_phone_count = sum(phone_position_admitted)
    phone_stage_admitted = (
        admitted_phone_count >= len(phones) - maximum_uncovered_phones
    )

    boundary_ranks = tuple(
        item.ranks.get(tuple(pair))
        for item, pair in zip(
            boundaries, zip(phones, phones[1:]), strict=True
        )
    )
    boundary_position_admitted = tuple(
        rank is not None
        and (maximum_boundary_rank is None or rank <= maximum_boundary_rank)
        for rank in boundary_ranks
    )
    return ProposalSequenceAudit(
        phone_channel_ranks=phone_channel_ranks,
        phone_position_admitted=phone_position_admitted,
        admitted_phone_count=admitted_phone_count,
        phone_stage_admitted=phone_stage_admitted,
        boundary_ranks=boundary_ranks,
        boundary_position_admitted=boundary_position_admitted,
        boundary_stage_admitted=(
            phone_stage_admitted
            and sum(not admitted for admitted in boundary_position_admitted)
            <= maximum_uncovered_boundaries
        ),
    )


def relational_sequence_cost(
    phone_cost: SequenceEvidenceCost,
    phones: Sequence[str],
    boundaries: Sequence[BoundaryPairEvidence],
) -> RelationalSequenceCost:
    expected_phone_count = len(boundaries) + 1
    if len(phones) != expected_phone_count:
        maximum = max((item.maximum_rank for item in boundaries), default=1) + 1
        return RelationalSequenceCost(
            uncovered_phones=phone_cost.uncovered,
            uncovered_transitions=len(boundaries),
            edits=phone_cost.edits,
            worst_transition_rank=maximum,
            transition_rank_sum=maximum * len(boundaries),
            worst_phone_rank=phone_cost.worst_rank,
            phone_rank_sum=phone_cost.rank_sum,
            negative_channel_support=phone_cost.negative_channel_support,
        )
    ranks = []
    missing = 0
    for boundary, pair in zip(boundaries, zip(phones, phones[1:]), strict=True):
        rank = boundary.ranks.get(tuple(pair))
        if rank is None:
            missing += 1
            rank = boundary.maximum_rank + 1
        ranks.append(rank)
    return RelationalSequenceCost(
        uncovered_phones=phone_cost.uncovered,
        uncovered_transitions=missing,
        edits=phone_cost.edits,
        worst_transition_rank=max(ranks, default=0),
        transition_rank_sum=sum(ranks),
        worst_phone_rank=phone_cost.worst_rank,
        phone_rank_sum=phone_cost.rank_sum,
        negative_channel_support=phone_cost.negative_channel_support,
    )


def _phone_match_cost(
    evidence: ProvenancePhoneEvidence,
    phone: str,
) -> SequenceEvidenceCost:
    ranks = tuple(
        channel[phone]
        for channel in evidence.channel_ranks.values()
        if phone in channel
    )
    if not ranks:
        rejected_rank = evidence.maximum_admitted_rank + 1
        return SequenceEvidenceCost(
            uncovered=1,
            worst_rank=rejected_rank,
            rank_sum=rejected_rank,
        )
    return SequenceEvidenceCost(
        worst_rank=min(ranks),
        rank_sum=min(ranks),
        negative_channel_support=-len(ranks),
    )


def provenance_edit_cost(
    evidence: Sequence[ProvenancePhoneEvidence],
    pronunciation: Sequence[str],
) -> SequenceEvidenceCost:
    """Align an exact pronunciation anchor in the evidence-cost semiring."""

    if not evidence and not pronunciation:
        return SequenceEvidenceCost()
    maximum_rank = max(
        (item.maximum_admitted_rank for item in evidence), default=1
    )
    gap = SequenceEvidenceCost(
        uncovered=1,
        edits=1,
        worst_rank=maximum_rank + 1,
        rank_sum=maximum_rank + 1,
    )
    previous = [SequenceEvidenceCost()]
    for _ in pronunciation:
        previous.append(previous[-1].combine(gap))
    for observed in evidence:
        current = [previous[0].combine(gap)]
        for column, phone in enumerate(pronunciation, start=1):
            match = previous[column - 1].combine(
                _phone_match_cost(observed, phone)
            )
            insertion = previous[column].combine(gap)
            deletion = current[column - 1].combine(gap)
            current.append(min(match, insertion, deletion))
        previous = current
    return previous[-1]


def rank_pronunciation_oracle(
    evidence: Sequence[ProvenancePhoneEvidence],
    pronunciations: Sequence[Pronunciation],
    maximum_length_delta: int = 1,
) -> tuple[PronunciationClassScore, ...]:
    """Exhaustively score exact dictionary anchors before retrieval compression."""

    if not evidence or maximum_length_delta < 0:
        raise ValueError("oracle needs evidence and a nonnegative length delta")
    grouped: dict[tuple[str, ...], set[str]] = {}
    for entry in pronunciations:
        if abs(len(entry.phones) - len(evidence)) <= maximum_length_delta:
            grouped.setdefault(entry.phones, set()).add(entry.word)
    scores = [
        PronunciationClassScore(
            phones=phones,
            words=tuple(sorted(words)),
            cost=provenance_edit_cost(evidence, phones),
        )
        for phones, words in grouped.items()
    ]
    return tuple(
        sorted(scores, key=lambda item: (item.cost, item.phones, item.words))
    )


def parse_cmudict(
    lines: Iterable[str],
    allowed_phones: frozenset[str],
    max_phones: int = 18,
) -> tuple[Pronunciation, ...]:
    """Parse CMUdict, remove stress, and retain declared phone inventory."""

    entries: set[Pronunciation] = set()
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith(";;;"):
            continue
        fields = line.split()
        if len(fields) < 2:
            continue
        word = ALTERNATE_RE.sub("", fields[0]).lower()
        phones = tuple(PHONE_RE.sub("", phone) for phone in fields[1:])
        if (
            word
            and 0 < len(phones) <= max_phones
            and all(phone in allowed_phones for phone in phones)
        ):
            entries.add(Pronunciation(word, phones))
    return tuple(sorted(entries, key=lambda entry: (entry.phones, entry.word)))


def _atom_values(
    phones: Sequence[str],
    atom_codes: Mapping[str, int],
    modulus: int,
) -> tuple[int, ...]:
    try:
        return tuple(atom_codes[phone] % modulus for phone in phones)
    except KeyError as error:
        raise ValueError(f"unknown phone {error.args[0]!r}") from error


def _cell(fingerprint: int, view: OccupancyView) -> int:
    mixed = fingerprint * view.multiplier % view.modulus
    return mixed * view.bucket_count // view.modulus


def full_key(
    phones: Sequence[str],
    atom_codes: Mapping[str, int],
    views: Sequence[OccupancyView] = DEFAULT_VIEWS,
) -> tuple[int, ...]:
    key = []
    for view in views:
        fingerprint = 0
        power = 1
        for value in _atom_values(phones, atom_codes, view.modulus):
            fingerprint = (fingerprint + value * power) % view.modulus
            power = power * view.base % view.modulus
        key.append(_cell(fingerprint, view))
    return tuple(key)


def deletion_keys(
    phones: Sequence[str],
    atom_codes: Mapping[str, int],
    views: Sequence[OccupancyView] = DEFAULT_VIEWS,
) -> tuple[tuple[int, ...], ...]:
    """Return coupled one-deletion keys using the rolling recurrence."""

    if not phones:
        return ()
    by_history: list[list[int]] = [[] for _ in phones]
    for view in views:
        values = _atom_values(phones, atom_codes, view.modulus)
        descendant = 0
        power = 1
        for value in values[1:]:
            descendant = (descendant + value * power) % view.modulus
            power = power * view.base % view.modulus
        rolling_power = 1
        for history in range(len(values)):
            by_history[history].append(_cell(descendant, view))
            if history + 1 < len(values):
                descendant = (
                    descendant
                    + (values[history] - values[history + 1]) * rolling_power
                ) % view.modulus
                rolling_power = rolling_power * view.base % view.modulus
    return tuple(tuple(key) for key in by_history)


class PronunciationIndex:
    """Two-coordinate typed index over exact pronunciation anchors."""

    def __init__(
        self,
        pronunciations: Sequence[Pronunciation],
        phone_alphabet: Sequence[str],
        views: Sequence[OccupancyView] = DEFAULT_VIEWS,
    ) -> None:
        if not pronunciations:
            raise ValueError("pronunciation index requires at least one anchor")
        self.pronunciations = tuple(pronunciations)
        self.atom_codes = {
            phone: index + 1 for index, phone in enumerate(sorted(phone_alphabet))
        }
        self.views = tuple(views)
        self.exact: dict[tuple[str, ...], set[int]] = {}
        self.full: dict[tuple[int, tuple[int, ...]], set[int]] = {}
        self.history: dict[tuple[int, tuple[int, ...]], set[int]] = {}
        for ordinal, pronunciation in enumerate(self.pronunciations):
            phones = pronunciation.phones
            length = len(phones)
            self.exact.setdefault(phones, set()).add(ordinal)
            key = full_key(phones, self.atom_codes, self.views)
            self.full.setdefault((length, key), set()).add(ordinal)
            for key in set(deletion_keys(phones, self.atom_codes, self.views)):
                self.history.setdefault((length, key), set()).add(ordinal)

    def query(self, phones: Sequence[str]) -> CandidateAudit:
        """Run the exact and three radius-one typed query plans."""

        sequence = tuple(phones)
        length = len(sequence)
        candidates = set(self.exact.get(sequence, ()))
        exact_anchor = bool(candidates)
        history_keys = set(deletion_keys(sequence, self.atom_codes, self.views))
        posting_reads = 0

        # Stored length n: common one-deletion descendant.
        for key in history_keys:
            posting_reads += 1
            candidates.update(self.history.get((length, key), ()))

        # Stored length n+1: delete one stored phone to obtain the full query.
        query_full = full_key(sequence, self.atom_codes, self.views)
        posting_reads += 1
        candidates.update(self.history.get((length + 1, query_full), ()))

        # Stored length n-1: delete one query phone to obtain the stored full key.
        for key in history_keys:
            posting_reads += 1
            candidates.update(self.full.get((length - 1, key), ()))
        return CandidateAudit(frozenset(candidates), exact_anchor, posting_reads)


class PronunciationProposalIndex:
    """Exact positional postings for uncertain phone and boundary proposals."""

    def __init__(self, pronunciations: Sequence[Pronunciation]) -> None:
        if not pronunciations:
            raise ValueError("proposal index requires exact pronunciation anchors")
        grouped: dict[tuple[str, ...], set[str]] = {}
        for entry in pronunciations:
            grouped.setdefault(entry.phones, set()).add(entry.word)
        self.classes = tuple(
            PronunciationClass(phones, tuple(sorted(words)))
            for phones, words in sorted(grouped.items())
        )
        self.by_length: dict[int, set[int]] = {}
        self.position: dict[tuple[int, int, str], set[int]] = {}
        self.boundary: dict[tuple[int, int, str, str], set[int]] = {}
        for ordinal, item in enumerate(self.classes):
            length = len(item.phones)
            self.by_length.setdefault(length, set()).add(ordinal)
            for position, phone in enumerate(item.phones):
                self.position.setdefault((length, position, phone), set()).add(
                    ordinal
                )
            for position, (left, right) in enumerate(
                zip(item.phones, item.phones[1:])
            ):
                self.boundary.setdefault(
                    (length, position, left, right), set()
                ).add(ordinal)

    def query(
        self,
        evidence: Sequence[ProvenancePhoneEvidence],
        boundaries: Sequence[BoundaryPairEvidence],
        maximum_uncovered_phones: int = 0,
        maximum_boundary_rank: int | None = None,
    ) -> ProposalRouteAudit:
        if not evidence or len(boundaries) != len(evidence) - 1:
            raise ValueError("proposal route requires one boundary between phones")
        if maximum_uncovered_phones < 0:
            raise ValueError("maximum uncovered phones cannot be negative")
        if maximum_boundary_rank is not None and maximum_boundary_rank < 1:
            raise ValueError("maximum boundary rank must be positive")
        length = len(evidence)
        phone_reads = 0
        coverage_counts: dict[int, int] = {}
        for position, item in enumerate(evidence):
            accepted = set().union(
                *(set(channel) for channel in item.channel_ranks.values())
            )
            posting = set()
            for phone in accepted:
                phone_reads += 1
                posting.update(self.position.get((length, position, phone), ()))
            for ordinal in posting:
                coverage_counts[ordinal] = coverage_counts.get(ordinal, 0) + 1
        minimum_coverage = max(0, length - maximum_uncovered_phones)
        candidates = {
            ordinal
            for ordinal in self.by_length.get(length, ())
            if coverage_counts.get(ordinal, 0) >= minimum_coverage
        }
        phone_candidates = frozenset(candidates)

        boundary_reads = 0
        for position, item in enumerate(boundaries):
            posting = set()
            for (left, right), rank in item.ranks.items():
                if (
                    maximum_boundary_rank is not None
                    and rank > maximum_boundary_rank
                ):
                    continue
                boundary_reads += 1
                posting.update(
                    self.boundary.get((length, position, left, right), ())
                )
            candidates.intersection_update(posting)
        return ProposalRouteAudit(
            phone_candidates=phone_candidates,
            boundary_candidates=frozenset(candidates),
            phone_posting_reads=phone_reads,
            boundary_posting_reads=boundary_reads,
        )

    def query_aligned(
        self,
        evidence: Sequence[ProvenancePhoneEvidence],
        boundaries: Sequence[BoundaryPairEvidence],
        maximum_uncovered_phones: int = 0,
        maximum_boundary_rank: int | None = None,
        maximum_length_delta: int = 1,
        maximum_uncovered_boundaries: int = 0,
    ) -> AlignedProposalRouteAudit:
        """Route uncertain evidence through exact and one-gap positional plans."""

        if not evidence or len(boundaries) != len(evidence) - 1:
            raise ValueError("aligned route requires one boundary between phones")
        if maximum_uncovered_phones < 0 or maximum_length_delta not in (0, 1):
            raise ValueError("aligned route tolerances are invalid")
        if maximum_boundary_rank is not None and maximum_boundary_rank < 1:
            raise ValueError("maximum boundary rank must be positive")
        if maximum_uncovered_boundaries < 0:
            raise ValueError("aligned boundary tolerance cannot be negative")

        observed_length = len(evidence)
        accepted_phones = [
            set().union(*(set(channel) for channel in item.channel_ranks.values()))
            for item in evidence
        ]
        plans: list[tuple[int, tuple[tuple[int, int], ...]]] = [
            (0, tuple((position, position) for position in range(observed_length)))
        ]
        if maximum_length_delta:
            # Dictionary pronunciation has one phone without an observed region.
            for missing_dictionary_position in range(observed_length + 1):
                plans.append(
                    (
                        1,
                        tuple(
                            (
                                observed,
                                observed
                                if observed < missing_dictionary_position
                                else observed + 1,
                            )
                            for observed in range(observed_length)
                        ),
                    )
                )
            # Observation has one region without a dictionary phone.
            if observed_length > 1:
                for surplus_observed_position in range(observed_length):
                    plans.append(
                        (
                            1,
                            tuple(
                                (
                                    observed,
                                    observed
                                    if observed < surplus_observed_position
                                    else observed - 1,
                                )
                                for observed in range(observed_length)
                                if observed != surplus_observed_position
                            ),
                        )
                    )

        minimum_edits: dict[int, int] = {}
        best_costs: dict[int, SequenceEvidenceCost] = {}
        best_relational_costs: dict[int, RelationalSequenceCost] = {}
        phone_reads = 0
        boundary_reads = 0
        boundary_posting_cache: dict[tuple[int, int, int], set[int]] = {}
        phone_posting_cache: dict[tuple[int, int, int], set[int]] = {}
        for edits, mapping in plans:
            pronunciation_length = observed_length + (
                1 if len(mapping) == observed_length and edits else 0
            )
            if edits and len(mapping) == observed_length - 1:
                pronunciation_length = observed_length - 1
            all_for_length = set(self.by_length.get(pronunciation_length, ()))
            if not all_for_length:
                continue
            candidates = set(all_for_length)
            plan_boundaries = []
            boundary_coverage: dict[int, int] = {}

            phone_coverage: dict[int, int] = {}
            for observed, pronunciation_position in mapping:
                cache_key = (
                    pronunciation_length,
                    pronunciation_position,
                    observed,
                )
                posting = phone_posting_cache.get(cache_key)
                if posting is None:
                    posting = set()
                    for phone in accepted_phones[observed]:
                        phone_reads += 1
                        posting.update(
                            self.position.get(
                                (
                                    pronunciation_length,
                                    pronunciation_position,
                                    phone,
                                ),
                                (),
                            )
                        )
                    posting.intersection_update(all_for_length)
                    phone_posting_cache[cache_key] = posting
                for ordinal in posting:
                    phone_coverage[ordinal] = phone_coverage.get(ordinal, 0) + 1
            required_phone_coverage = max(
                0, len(mapping) - maximum_uncovered_phones
            )
            candidates.intersection_update(
                ordinal
                for ordinal in all_for_length
                if phone_coverage.get(ordinal, 0) >= required_phone_coverage
            )
            if not candidates:
                continue

            for (observed_left, pronunciation_left), (
                observed_right,
                pronunciation_right,
            ) in zip(mapping, mapping[1:]):
                if (
                    observed_right != observed_left + 1
                    or pronunciation_right != pronunciation_left + 1
                ):
                    continue
                plan_boundaries.append((observed_left, pronunciation_left))
                cache_key = (
                    pronunciation_length,
                    pronunciation_left,
                    observed_left,
                )
                posting = boundary_posting_cache.get(cache_key)
                if posting is None:
                    posting = set()
                    for pair, rank in boundaries[observed_left].ranks.items():
                        if (
                            maximum_boundary_rank is not None
                            and rank > maximum_boundary_rank
                        ):
                            continue
                        boundary_reads += 1
                        posting.update(
                            self.boundary.get(
                                (
                                    pronunciation_length,
                                    pronunciation_left,
                                    pair[0],
                                    pair[1],
                                ),
                                (),
                            )
                        )
                    posting.intersection_update(all_for_length)
                    boundary_posting_cache[cache_key] = posting
                for ordinal in posting:
                    boundary_coverage[ordinal] = (
                        boundary_coverage.get(ordinal, 0) + 1
                    )
            required_boundary_coverage = max(
                0, len(plan_boundaries) - maximum_uncovered_boundaries
            )
            candidates.intersection_update(
                ordinal
                for ordinal in all_for_length
                if boundary_coverage.get(ordinal, 0)
                >= required_boundary_coverage
            )
            if not candidates:
                continue
            admitted = set()
            for ordinal in candidates:
                phones = self.classes[ordinal].phones
                missed = 0
                plan_cost = SequenceEvidenceCost()
                if edits:
                    rejected_rank = max(
                        item.maximum_admitted_rank for item in evidence
                    ) + 1
                    plan_cost = SequenceEvidenceCost(
                        uncovered=1,
                        edits=1,
                        worst_rank=rejected_rank,
                        rank_sum=rejected_rank,
                    )
                for observed, pronunciation_position in mapping:
                    phone_reads += 1
                    match_cost = _phone_match_cost(
                        evidence[observed], phones[pronunciation_position]
                    )
                    plan_cost = plan_cost.combine(match_cost)
                    if match_cost.uncovered:
                        missed += match_cost.uncovered
                        if missed > maximum_uncovered_phones:
                            break
                if missed <= maximum_uncovered_phones:
                    admitted.add(ordinal)
                    previous = best_costs.get(ordinal)
                    if previous is None or plan_cost < previous:
                        best_costs[ordinal] = plan_cost
                    transition_ranks = []
                    missing_transitions = 0
                    for observed, pronunciation_position in plan_boundaries:
                        boundary = boundaries[observed]
                        rank = boundary.ranks.get(
                            (
                                phones[pronunciation_position],
                                phones[pronunciation_position + 1],
                            )
                        )
                        if rank is None or (
                            maximum_boundary_rank is not None
                            and rank > maximum_boundary_rank
                        ):
                            missing_transitions += 1
                            rank = boundary.maximum_rank + 1
                        transition_ranks.append(rank)
                    relational_cost = RelationalSequenceCost(
                        uncovered_phones=plan_cost.uncovered,
                        uncovered_transitions=(
                            max(
                                0,
                                pronunciation_length - 1 - len(plan_boundaries),
                            )
                            + missing_transitions
                        ),
                        edits=plan_cost.edits,
                        worst_transition_rank=max(transition_ranks, default=0),
                        transition_rank_sum=sum(transition_ranks),
                        worst_phone_rank=plan_cost.worst_rank,
                        phone_rank_sum=plan_cost.rank_sum,
                        negative_channel_support=plan_cost.negative_channel_support,
                    )
                    previous_relational = best_relational_costs.get(ordinal)
                    if (
                        previous_relational is None
                        or relational_cost < previous_relational
                    ):
                        best_relational_costs[ordinal] = relational_cost
            for ordinal in admitted:
                minimum_edits[ordinal] = min(
                    edits, minimum_edits.get(ordinal, edits)
                )
        return AlignedProposalRouteAudit(
            candidates=frozenset(minimum_edits),
            minimum_edits=minimum_edits,
            costs=best_costs,
            relational_costs=best_relational_costs,
            alignment_plan_count=len(plans),
            phone_posting_reads=phone_reads,
            boundary_posting_reads=boundary_reads,
        )

    def query_grouped_aligned(
        self,
        evidence: Sequence[ProvenancePhoneEvidence],
        boundaries: Sequence[BoundaryPairEvidence],
        maximum_uncovered_phones: int = 0,
        maximum_boundary_rank: int | None = None,
        maximum_length_delta: int = 1,
        maximum_uncovered_boundaries: int = 0,
    ) -> AlignedProposalRouteAudit:
        """Add contiguous region-to-phone partitions to exact/one-gap plans.

        A grouped plan treats adjacent detector fragments as uncertain views of
        one phone for candidate admission.  The caller can still verify the
        complete, unmerged acoustic span geometrically.
        """

        if not evidence or len(boundaries) != len(evidence) - 1:
            raise ValueError("grouped route requires one boundary between regions")
        if maximum_length_delta < 0:
            raise ValueError("grouped route length delta cannot be negative")
        observed_length = len(evidence)
        base = self.query_aligned(
            evidence,
            boundaries,
            maximum_uncovered_phones=maximum_uncovered_phones,
            maximum_boundary_rank=maximum_boundary_rank,
            maximum_length_delta=min(maximum_length_delta, 1),
            maximum_uncovered_boundaries=maximum_uncovered_boundaries,
        )
        minimum_edits = dict(base.minimum_edits)
        best_costs = dict(base.costs)
        best_relational_costs = dict(base.relational_costs)
        alignment_plan_count = base.alignment_plan_count
        phone_reads = base.phone_posting_reads
        boundary_reads = base.boundary_posting_reads

        maximum_reduction = min(
            maximum_length_delta, max(0, observed_length - 1)
        )
        for reduction in range(1, maximum_reduction + 1):
            pronunciation_length = observed_length - reduction
            for cuts in combinations(
                range(1, observed_length), pronunciation_length - 1
            ):
                stops = cuts + (observed_length,)
                starts = (0,) + cuts
                grouped_evidence = []
                for start, stop in zip(starts, stops, strict=True):
                    channel_names = set().union(
                        *(
                            set(evidence[position].channel_ranks)
                            for position in range(start, stop)
                        )
                    )
                    channels = {}
                    for channel in channel_names:
                        ranks = {}
                        for position in range(start, stop):
                            for phone, rank in evidence[position].channel_ranks.get(
                                channel, {}
                            ).items():
                                ranks[phone] = min(rank, ranks.get(phone, rank))
                        channels[channel] = ranks
                    grouped_evidence.append(
                        ProvenancePhoneEvidence(
                            channels,
                            max(
                                evidence[position].maximum_admitted_rank
                                for position in range(start, stop)
                            ),
                        )
                    )
                grouped_boundaries = tuple(
                    boundaries[cut - 1] for cut in cuts
                )
                route = self.query_aligned(
                    grouped_evidence,
                    grouped_boundaries,
                    maximum_uncovered_phones=maximum_uncovered_phones,
                    maximum_boundary_rank=maximum_boundary_rank,
                    maximum_length_delta=0,
                    maximum_uncovered_boundaries=maximum_uncovered_boundaries,
                )
                alignment_plan_count += route.alignment_plan_count
                phone_reads += route.phone_posting_reads
                boundary_reads += route.boundary_posting_reads
                for ordinal in route.candidates:
                    cost = route.costs[ordinal]
                    relational_cost = route.relational_costs[ordinal]
                    previous = (
                        best_relational_costs.get(ordinal),
                        best_costs.get(ordinal),
                    )
                    if previous[0] is None or (
                        relational_cost,
                        cost,
                    ) < previous:
                        best_relational_costs[ordinal] = relational_cost
                        best_costs[ordinal] = cost
                    minimum_edits[ordinal] = min(
                        reduction, minimum_edits.get(ordinal, reduction)
                    )
        return AlignedProposalRouteAudit(
            candidates=frozenset(minimum_edits),
            minimum_edits=minimum_edits,
            costs=best_costs,
            relational_costs=best_relational_costs,
            alignment_plan_count=alignment_plan_count,
            phone_posting_reads=phone_reads,
            boundary_posting_reads=boundary_reads,
        )


def normalized_phone_evidence(
    ranked: Sequence[Mapping[str, object]],
    unseen_margin: float = 1.0,
) -> PhoneEvidence:
    """Convert local raster distances into dimensionless relative costs."""

    if not ranked:
        raise ValueError("phone evidence must contain at least one hypothesis")
    pairs = [(str(item["phone"]), float(item["distance"])) for item in ranked]
    pairs.sort(key=lambda pair: (pair[1], pair[0]))
    minimum = pairs[0][1]
    spread = max(pairs[-1][1] - minimum, abs(minimum) * 0.05, 1e-9)
    costs = {phone: (distance - minimum) / spread for phone, distance in pairs}
    return PhoneEvidence(costs, max(costs.values()) + unseen_margin)


def k_best_phone_sequences(
    evidence: Sequence[PhoneEvidence],
    count: int = 32,
) -> tuple[tuple[tuple[str, ...], float], ...]:
    """Enumerate the bounded lowest-cost Cartesian phone hypotheses."""

    if not evidence or count < 1:
        return ()
    alternatives = [
        sorted(item.costs.items(), key=lambda pair: (pair[1], pair[0]))
        for item in evidence
    ]
    origin = tuple(0 for _ in alternatives)

    def state_cost(state: tuple[int, ...]) -> float:
        return sum(alternatives[i][choice][1] for i, choice in enumerate(state))

    heap = [(state_cost(origin), origin)]
    seen = {origin}
    output = []
    while heap and len(output) < count:
        cost, state = heapq.heappop(heap)
        phones = tuple(
            alternatives[index][choice][0]
            for index, choice in enumerate(state)
        )
        output.append((phones, cost))
        for axis in range(len(state)):
            if state[axis] + 1 >= len(alternatives[axis]):
                continue
            neighbor = list(state)
            neighbor[axis] += 1
            neighbor_tuple = tuple(neighbor)
            if neighbor_tuple not in seen:
                seen.add(neighbor_tuple)
                heapq.heappush(
                    heap, (state_cost(neighbor_tuple), neighbor_tuple)
                )
    return tuple(output)


def acoustic_edit_cost(
    evidence: Sequence[PhoneEvidence],
    pronunciation: Sequence[str],
    insertion_penalty: float = 1.1,
    deletion_penalty: float = 1.1,
) -> float:
    """Score a pronunciation against every top-k phone distribution."""

    rows = len(evidence)
    columns = len(pronunciation)
    previous = [column * deletion_penalty for column in range(columns + 1)]
    for row in range(1, rows + 1):
        current = [row * insertion_penalty] + [0.0] * columns
        observed = evidence[row - 1]
        for column in range(1, columns + 1):
            phone = pronunciation[column - 1]
            match = previous[column - 1] + observed.costs.get(
                phone, observed.unseen_cost
            )
            insertion = previous[column] + insertion_penalty
            deletion = current[column - 1] + deletion_penalty
            current[column] = min(match, insertion, deletion)
        previous = current
    return previous[-1] / max(rows, columns, 1)


def levenshtein_distance(left: Sequence[str], right: Sequence[str]) -> int:
    previous = list(range(len(right) + 1))
    for row, left_phone in enumerate(left, 1):
        current = [row]
        for column, right_phone in enumerate(right, 1):
            current.append(min(
                previous[column] + 1,
                current[column - 1] + 1,
                previous[column - 1] + (left_phone != right_phone),
            ))
        previous = current
    return previous[-1]
