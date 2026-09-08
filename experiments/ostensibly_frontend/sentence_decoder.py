"""Acoustic-only k-best paths through ambiguity-preserving lexical edges."""

from __future__ import annotations

from dataclasses import dataclass
from bisect import bisect_left
from typing import Mapping, Sequence


@dataclass(frozen=True, order=True)
class SentencePathCost:
    total_cost: float
    uncovered_phones: int
    word_count: int


@dataclass(frozen=True)
class SentenceSegment:
    phone0: int
    phone1: int
    seconds0: float
    seconds1: float
    phones: tuple[str, ...]
    words: tuple[str, ...]
    acoustic_score: float
    edge_cost: float
    evidence_channel: str = "route"
    raw_channel_score: float | None = None
    unknown: bool = False


@dataclass(frozen=True)
class SentencePath:
    cost: SentencePathCost
    segments: tuple[SentenceSegment, ...]


def _segment_signature(path: SentencePath) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (segment.phone0, segment.phone1, segment.phones, segment.words)
        for segment in path.segments
    )


def _join_path(
    path: SentencePath,
    segment: SentenceSegment,
) -> SentencePath:
    return SentencePath(
        SentencePathCost(
            total_cost=path.cost.total_cost + segment.edge_cost,
            uncovered_phones=(
                path.cost.uncovered_phones
                + (segment.phone1 - segment.phone0 if segment.unknown else 0)
            ),
            word_count=path.cost.word_count + (0 if segment.unknown else 1),
        ),
        path.segments + (segment,),
    )


def edge_channel_scores(
    classes: Sequence[Mapping[str, object]],
    evidence_channel: str,
    generic_score_mode: str = "per_edge_robust",
    length_distribution: Sequence[float] | None = None,
) -> tuple[tuple[Mapping[str, object], float, float], ...]:
    """Return (class, normalized, raw) scores for one independent channel."""

    if evidence_channel == "route":
        return tuple(
            (item, float(item["score"]), float(item["score"]))
            for item in classes
        )
    prefix = "generic_word:"
    if not evidence_channel.startswith(prefix):
        raise ValueError("unknown sentence evidence channel")
    policy = evidence_channel[len(prefix) :]
    if generic_score_mode not in ("per_edge_robust", "raw", "length_cdf"):
        raise ValueError("unknown generic sentence score mode")
    values = []
    for item in classes:
        channels = item.get("generic_word_channels")
        if not isinstance(channels, Mapping):
            continue
        record = channels.get(policy)
        if not isinstance(record, Mapping) or record.get("distance") is None:
            continue
        values.append((item, float(record["distance"])))
    if not values:
        return ()
    if generic_score_mode == "raw":
        return tuple((item, distance, distance) for item, distance in values)
    if generic_score_mode == "length_cdf":
        if not length_distribution:
            raise ValueError("length-CDF scoring requires a reference distribution")
        ordered = tuple(sorted(float(value) for value in length_distribution))
        count = len(ordered)
        return tuple(
            (
                item,
                (bisect_left(ordered, distance) + 0.5) / count,
                distance,
            )
            for item, distance in values
        )
    distances = sorted(distance for _, distance in values)
    minimum = distances[0]
    middle = len(distances) // 2
    median = (
        distances[middle]
        if len(distances) % 2
        else 0.5 * (distances[middle - 1] + distances[middle])
    )
    scale = max(median - minimum, 1e-12)
    return tuple(
        (item, (distance - minimum) / scale, distance)
        for item, distance in values
    )


def decode_sentence_paths(
    edges: Sequence[Mapping[str, object]],
    phone_count: int,
    path_count: int = 16,
    word_penalty: float = 0.12,
    unknown_phone_penalty: float = 1.5,
    evidence_channel: str = "route",
    generic_score_mode: str = "per_edge_robust",
) -> tuple[SentencePath, ...]:
    """Return globally ranked complete paths without a language model.

    Edge acoustic scores are normalized per observed phone by the word-lattice
    builder.  Multiplying by span length restores additive evidence before the
    small explicit word-count regularizer is applied.
    """

    if phone_count < 1 or path_count < 1:
        raise ValueError("sentence decoding requires phones and positive path count")
    if word_penalty < 0.0 or unknown_phone_penalty <= 0.0:
        raise ValueError("sentence decoder penalties are invalid")

    incoming: list[list[SentenceSegment]] = [[] for _ in range(phone_count + 1)]
    length_distributions: dict[int, list[float]] = {}
    if evidence_channel.startswith("generic_word:") and generic_score_mode == "length_cdf":
        policy = evidence_channel.split(":", 1)[1]
        for edge in edges:
            span = int(edge["phone1"]) - int(edge["phone0"])
            classes = edge.get("ambiguity_classes")
            if not isinstance(classes, list):
                raise ValueError("word edge lacks ambiguity classes")
            for item in classes:
                channels = item.get("generic_word_channels")
                if not isinstance(channels, Mapping):
                    continue
                record = channels.get(policy)
                if isinstance(record, Mapping) and record.get("distance") is not None:
                    length_distributions.setdefault(span, []).append(
                        float(record["distance"])
                    )
    phone_times: dict[int, tuple[float, float]] = {}
    for edge in edges:
        phone0 = int(edge["phone0"])
        phone1 = int(edge["phone1"])
        if not 0 <= phone0 < phone1 <= phone_count:
            raise ValueError("word edge lies outside the phone lattice")
        seconds0 = float(edge["seconds0"])
        seconds1 = float(edge["seconds1"])
        phone_times.setdefault(phone0, (seconds0, seconds1))
        phone_times.setdefault(phone1 - 1, (seconds0, seconds1))
        span = phone1 - phone0
        classes = edge.get("ambiguity_classes")
        if not isinstance(classes, list):
            raise ValueError("word edge lacks ambiguity classes")
        for item, score, raw_score in edge_channel_scores(
            classes,
            evidence_channel,
            generic_score_mode,
            length_distributions.get(span),
        ):
            phones = tuple(str(phone) for phone in item["phones"])
            words = tuple(sorted(str(word) for word in item["words"]))
            if not phones or not words:
                raise ValueError("word ambiguity class cannot be empty")
            incoming[phone1].append(
                SentenceSegment(
                    phone0=phone0,
                    phone1=phone1,
                    seconds0=seconds0,
                    seconds1=seconds1,
                    phones=phones,
                    words=words,
                    acoustic_score=score,
                    edge_cost=score * span + word_penalty,
                    evidence_channel=evidence_channel,
                    raw_channel_score=raw_score,
                )
            )

    states: list[list[SentencePath]] = [[] for _ in range(phone_count + 1)]
    states[0] = [SentencePath(SentencePathCost(0.0, 0, 0), ())]
    for stop in range(1, phone_count + 1):
        candidates: list[SentencePath] = []
        for segment in incoming[stop]:
            for path in states[segment.phone0]:
                candidates.append(_join_path(path, segment))

        seconds0, seconds1 = phone_times.get(
            stop - 1, (float(stop - 1), float(stop))
        )
        unknown = SentenceSegment(
            phone0=stop - 1,
            phone1=stop,
            seconds0=seconds0,
            seconds1=seconds1,
            phones=(),
            words=("<unk>",),
            acoustic_score=unknown_phone_penalty,
            edge_cost=unknown_phone_penalty,
            evidence_channel=evidence_channel,
            raw_channel_score=unknown_phone_penalty,
            unknown=True,
        )
        for path in states[stop - 1]:
            candidates.append(_join_path(path, unknown))

        candidates.sort(key=lambda path: (path.cost, _segment_signature(path)))
        selected = []
        seen = set()
        for path in candidates:
            signature = _segment_signature(path)
            if signature in seen:
                continue
            seen.add(signature)
            selected.append(path)
            if len(selected) == path_count:
                break
        states[stop] = selected
    return tuple(states[phone_count])


def path_display(
    path: SentencePath,
    comma_gap_seconds: float = 0.20,
    period_gap_seconds: float = 0.45,
) -> str:
    """Render a path while retaining homophone ambiguity in braces."""

    if not 0.0 <= comma_gap_seconds <= period_gap_seconds:
        raise ValueError("sentence gap thresholds are invalid")
    output = ""
    previous_stop = None
    for segment in path.segments:
        if previous_stop is not None:
            gap = max(0.0, segment.seconds0 - previous_stop)
            if gap >= period_gap_seconds:
                output += ". "
            elif gap >= comma_gap_seconds:
                output += ", "
            else:
                output += " "
        token = (
            segment.words[0]
            if len(segment.words) == 1
            else "{" + "|".join(segment.words) + "}"
        )
        output += token
        previous_stop = segment.seconds1
    return output
