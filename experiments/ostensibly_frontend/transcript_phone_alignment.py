"""Align listener-supported transcript phones to uncertain phone regions."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping, Sequence

import numpy as np


TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


@dataclass(frozen=True)
class PhoneProposalAlignment:
    edit_cost: int
    reference_phone_count: int
    proposal_region_count: int
    aligned_count: int
    top1_hits: int
    topk_hits: int
    missing_reference_phones: int
    surplus_proposal_regions: int


@dataclass(frozen=True)
class TranscriptWordPhoneSpan:
    word: str
    phones: tuple[str, ...]
    phone0: int
    phone1: int


def markdown_section(text: str, heading: str) -> str:
    marker = f"## {heading}"
    start = text.find(marker)
    if start < 0:
        raise ValueError(f"missing markdown section {heading!r}")
    start += len(marker)
    stop = text.find("\n## ", start)
    section = text[start:] if stop < 0 else text[start:stop]
    section = re.sub(r"\*\*Speaker\s+\d+:\*\*", " ", section)
    section = re.sub(r"\[[^\]]+\]", " ", section)
    return section.replace("’", "'")


def transcript_words(text: str) -> tuple[str, ...]:
    return tuple(TOKEN_RE.findall(text.lower()))


def transcript_phone_sequence(
    words: Sequence[str],
    pronunciations: Mapping[str, Sequence[Sequence[str]]],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Use the dictionary's deterministic first pronunciation per word."""

    phones = []
    missing = []
    for word in words:
        options = pronunciations.get(word)
        if not options:
            missing.append(word)
            continue
        phones.extend(str(phone) for phone in options[0])
    return tuple(phones), tuple(missing)


def transcript_word_phone_spans(
    words: Sequence[str],
    pronunciations: Mapping[str, Sequence[Sequence[str]]],
) -> tuple[tuple[TranscriptWordPhoneSpan, ...], tuple[str, ...]]:
    spans = []
    missing = []
    offset = 0
    for word in words:
        options = pronunciations.get(word)
        if not options:
            missing.append(word)
            continue
        phones = tuple(str(phone) for phone in options[0])
        spans.append(TranscriptWordPhoneSpan(word, phones, offset, offset + len(phones)))
        offset += len(phones)
    return tuple(spans), tuple(missing)


def align_phone_proposals_trace(
    reference: Sequence[str],
    proposals: Sequence[Sequence[str]],
) -> tuple[PhoneProposalAlignment, tuple[int | None, ...]]:
    """Globally align phones to regions with a top-k-aware unit edit cost."""

    if not reference or not proposals or any(not row for row in proposals):
        raise ValueError("phone alignment requires nonempty references and proposals")
    rows = len(reference)
    columns = len(proposals)
    costs = np.empty((rows + 1, columns + 1), dtype=np.int32)
    trace = np.zeros((rows + 1, columns + 1), dtype=np.uint8)
    costs[:, 0] = np.arange(rows + 1)
    costs[0, :] = np.arange(columns + 1)
    trace[1:, 0] = 1  # reference phone has no proposal region
    trace[0, 1:] = 2  # proposal region has no reference phone
    for row, phone in enumerate(reference, start=1):
        for column, candidates in enumerate(proposals, start=1):
            diagonal = costs[row - 1, column - 1] + (
                phone not in candidates
            )
            deletion = costs[row - 1, column] + 1
            insertion = costs[row, column - 1] + 1
            best = min(diagonal, deletion, insertion)
            costs[row, column] = best
            trace[row, column] = (
                0 if diagonal == best else (1 if deletion == best else 2)
            )

    row, column = rows, columns
    reference_to_proposal: list[int | None] = [None] * rows
    aligned = top1 = topk = missing = surplus = 0
    while row or column:
        operation = trace[row, column]
        if row and column and operation == 0:
            phone = reference[row - 1]
            candidates = proposals[column - 1]
            aligned += 1
            top1 += phone == candidates[0]
            topk += phone in candidates
            reference_to_proposal[row - 1] = column - 1
            row -= 1
            column -= 1
        elif row and (column == 0 or operation == 1):
            missing += 1
            row -= 1
        else:
            surplus += 1
            column -= 1
    return PhoneProposalAlignment(
        edit_cost=int(costs[rows, columns]),
        reference_phone_count=rows,
        proposal_region_count=columns,
        aligned_count=aligned,
        top1_hits=top1,
        topk_hits=topk,
        missing_reference_phones=missing,
        surplus_proposal_regions=surplus,
    ), tuple(reference_to_proposal)


def align_phone_proposals(
    reference: Sequence[str],
    proposals: Sequence[Sequence[str]],
) -> PhoneProposalAlignment:
    return align_phone_proposals_trace(reference, proposals)[0]
