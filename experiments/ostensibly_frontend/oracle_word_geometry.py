"""Exact dictionary ranking on oracle-bounded phone-rank matrices."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .phone_rank_matrix import PhoneRankMatrix
from .word_lattice import Pronunciation


POLICIES = ("worst_then_sum", "sum_then_worst", "learned_rank_likelihood")


@dataclass(frozen=True)
class PronunciationCatalog:
    phones: tuple[tuple[str, ...], ...]
    words: tuple[tuple[str, ...], ...]
    columns: np.ndarray

    def __post_init__(self) -> None:
        count = len(self.phones)
        length = len(self.phones[0]) if self.phones else 0
        if (
            count < 1
            or length < 1
            or len(self.words) != count
            or self.columns.shape != (count, length)
            or any(len(phones) != length for phones in self.phones)
        ):
            raise ValueError("pronunciation catalog has invalid geometry")


def compile_pronunciation_catalogs(
    pronunciations: Iterable[Pronunciation], labels: tuple[str, ...]
) -> dict[int, PronunciationCatalog]:
    """Group homophone classes by length and pack their label columns."""

    grouped: dict[tuple[str, ...], set[str]] = {}
    for item in pronunciations:
        grouped.setdefault(tuple(item.phones), set()).add(str(item.word))
    label_columns = {label: column for column, label in enumerate(labels)}
    by_length: dict[int, list[tuple[tuple[str, ...], tuple[str, ...]]]] = {}
    for phones, words in grouped.items():
        if any(phone not in label_columns for phone in phones):
            continue
        by_length.setdefault(len(phones), []).append(
            (phones, tuple(sorted(words)))
        )
    output = {}
    for length, rows in by_length.items():
        rows.sort()
        output[length] = PronunciationCatalog(
            phones=tuple(row[0] for row in rows),
            words=tuple(row[1] for row in rows),
            columns=np.asarray(
                [[label_columns[phone] for phone in row[0]] for row in rows],
                dtype=np.int16,
            ),
        )
    return output


def learned_rank_log_likelihood(
    matrix: PhoneRankMatrix,
    *,
    excluded_utterance: str | None = None,
    smoothing: float = 0.5,
) -> np.ndarray:
    """Pooled genuine-versus-impostor rank log likelihoods.

    Each complete query contributes one genuine rank. At every rank not
    occupied by that target, it contributes one impostor label, so the
    impostor population is recovered exactly without inventing samples.
    """

    if smoothing <= 0.0:
        raise ValueError("rank likelihood smoothing must be positive")
    selected = np.ones(matrix.ranks.shape[0], dtype=bool)
    if excluded_utterance is not None:
        selected &= matrix.utterances.astype(str) != excluded_utterance
    rows = np.flatnonzero(selected)
    if not rows.size:
        raise ValueError("rank likelihood has no training queries")
    columns = {str(label): index for index, label in enumerate(matrix.labels)}
    target_ranks = np.asarray(
        [
            matrix.ranks[row, columns[str(matrix.target_phones[row])]]
            for row in rows
        ],
        dtype=np.int64,
    )
    label_count = matrix.labels.size
    counts = np.bincount(target_ranks, minlength=label_count + 1)[1:]
    query_count = rows.size
    genuine = (counts + smoothing) / (query_count + smoothing * label_count)
    impostor_counts = query_count - counts
    impostor = (impostor_counts + smoothing) / (
        query_count * (label_count - 1) + smoothing * label_count
    )
    return np.r_[0.0, np.log(genuine / impostor)]


def query_rank_rows(
    matrix: PhoneRankMatrix,
    *,
    reference_speaker: str,
    query_speaker: str,
    utterance: str,
    ordinal0: int,
    phone_count: int,
) -> np.ndarray:
    """Return consecutive complete rank vectors for one oracle word span."""

    index = matrix.index()
    rows = []
    for ordinal in range(ordinal0, ordinal0 + phone_count):
        key = (reference_speaker, query_speaker, utterance, ordinal)
        if key not in index:
            raise ValueError(f"rank row {key} is absent")
        rows.append(matrix.ranks[index[key]])
    return np.stack(rows).astype(np.int16)


def rank_pronunciation_catalog(
    rank_rows: np.ndarray,
    catalog: PronunciationCatalog,
    *,
    policy: str,
    rank_log_likelihood: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact catalog order and candidate phone-rank matrix."""

    query = np.asarray(rank_rows, dtype=np.int16)
    if (
        query.ndim != 2
        or query.shape[0] != catalog.columns.shape[1]
        or policy not in POLICIES
    ):
        raise ValueError("oracle word ranking has incompatible geometry")
    candidate_ranks = query[
        np.arange(query.shape[0])[None, :], catalog.columns
    ]
    worst = np.max(candidate_ranks, axis=1)
    total = np.sum(candidate_ranks, axis=1)
    tie = np.arange(candidate_ranks.shape[0])
    if policy == "worst_then_sum":
        order = np.lexsort((tie, total, worst))
    elif policy == "sum_then_worst":
        order = np.lexsort((tie, worst, total))
    else:
        likelihood = np.asarray(rank_log_likelihood, dtype=np.float64)
        if likelihood.shape != (query.shape[1] + 1,):
            raise ValueError("learned rank likelihood has incompatible labels")
        evidence = np.sum(likelihood[candidate_ranks], axis=1)
        order = np.lexsort((tie, worst, -evidence))
    return order, candidate_ranks


def target_catalog_rank(order: np.ndarray, target_index: int) -> int:
    positions = np.flatnonzero(np.asarray(order) == target_index)
    if positions.size != 1:
        raise ValueError("target pronunciation is absent or duplicated")
    return int(positions[0] + 1)

