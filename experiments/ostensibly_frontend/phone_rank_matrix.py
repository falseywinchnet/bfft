"""Complete label-rank vectors for frozen cross-speaker phone geometry."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


FORMAT = "ostensibly_phone_rank_matrix_v1"


def complete_rank_vector(
    ranking: Iterable[Mapping[str, object]], labels: tuple[str, ...]
) -> np.ndarray:
    """Pack one complete permutation of label ranks in canonical order."""

    rows = tuple(ranking)
    by_label = {str(row["phone"]): int(row["rank"]) for row in rows}
    if set(by_label) != set(labels) or sorted(by_label.values()) != list(
        range(1, len(labels) + 1)
    ):
        raise ValueError("phone ranking is not a complete label permutation")
    return np.asarray([by_label[label] for label in labels], dtype=np.uint8)


@dataclass(frozen=True)
class PhoneRankMatrix:
    labels: np.ndarray
    reference_speakers: np.ndarray
    query_speakers: np.ndarray
    utterances: np.ndarray
    ordinals: np.ndarray
    target_phones: np.ndarray
    ranks: np.ndarray
    provenance: dict[str, object]

    def __post_init__(self) -> None:
        count = self.ranks.shape[0] if self.ranks.ndim == 2 else -1
        label_count = self.labels.size
        if (
            count < 1
            or label_count < 2
            or self.ranks.shape != (count, label_count)
            or any(
                values.shape != (count,)
                for values in (
                    self.reference_speakers,
                    self.query_speakers,
                    self.utterances,
                    self.ordinals,
                    self.target_phones,
                )
            )
            or len(set(map(str, self.labels))) != label_count
            or np.any(np.sort(self.ranks, axis=1) != np.arange(1, label_count + 1))
        ):
            raise ValueError("phone rank matrix has invalid geometry")

    def index(self) -> dict[tuple[str, str, str, int], int]:
        output = {}
        for row, key in enumerate(
            zip(
                self.reference_speakers,
                self.query_speakers,
                self.utterances,
                self.ordinals,
                strict=True,
            )
        ):
            normalized = (str(key[0]), str(key[1]), str(key[2]), int(key[3]))
            if normalized in output:
                raise ValueError("phone rank matrix contains duplicate queries")
            output[normalized] = row
        return output

    def candidate_ranks(
        self,
        reference_speaker: str,
        query_speaker: str,
        utterance: str,
        ordinal0: int,
        phones: tuple[str, ...],
    ) -> tuple[int, ...]:
        labels = {str(label): column for column, label in enumerate(self.labels)}
        rows = self.index()
        result = []
        for offset, phone in enumerate(phones):
            if phone not in labels:
                raise ValueError(f"phone {phone} is outside the rank inventory")
            key = (
                reference_speaker,
                query_speaker,
                utterance,
                ordinal0 + offset,
            )
            if key not in rows:
                raise ValueError(f"phone query {key} is absent")
            result.append(int(self.ranks[rows[key], labels[phone]]))
        return tuple(result)


def save_phone_rank_matrix(path: Path, matrix: PhoneRankMatrix) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        format=np.asarray(FORMAT),
        labels=matrix.labels,
        reference_speakers=matrix.reference_speakers,
        query_speakers=matrix.query_speakers,
        utterances=matrix.utterances,
        ordinals=matrix.ordinals,
        target_phones=matrix.target_phones,
        ranks=matrix.ranks,
        provenance=np.asarray(json.dumps(matrix.provenance, sort_keys=True)),
    )


def load_phone_rank_matrix(path: Path) -> PhoneRankMatrix:
    with np.load(path, allow_pickle=False) as document:
        if str(document["format"]) != FORMAT:
            raise ValueError("unknown phone rank matrix format")
        return PhoneRankMatrix(
            labels=np.asarray(document["labels"]),
            reference_speakers=np.asarray(document["reference_speakers"]),
            query_speakers=np.asarray(document["query_speakers"]),
            utterances=np.asarray(document["utterances"]),
            ordinals=np.asarray(document["ordinals"], dtype=np.int64),
            target_phones=np.asarray(document["target_phones"]),
            ranks=np.asarray(document["ranks"], dtype=np.uint8),
            provenance=json.loads(str(document["provenance"])),
        )

