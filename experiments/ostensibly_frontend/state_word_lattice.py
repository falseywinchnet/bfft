"""Exact dictionary composition over a measured state-phone DAG."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .word_lattice import Pronunciation


@dataclass(frozen=True)
class WordPathCost:
    duration_uncovered: int
    state_uncovered: int
    proposal_uncovered: int
    worst_rank: int
    rank_sum: int
    phone_count: int
    worst_duration_rank: int
    duration_rank_sum: int
    worst_state_rank: int
    state_rank_sum: int
    worst_geometry_score: float
    geometry_score_sum: float

    @property
    def mean_rank(self) -> float:
        return self.rank_sum / self.phone_count

    @property
    def mean_geometry_score(self) -> float:
        return self.geometry_score_sum / self.phone_count

    @property
    def objective(self) -> tuple[float | int, ...]:
        return (
            self.proposal_uncovered,
            self.worst_rank,
            self.mean_rank,
            self.worst_geometry_score,
            self.mean_geometry_score,
            self.rank_sum,
        )

    def extend(
        self,
        rank: int,
        geometry_score: float,
        *,
        duration_rank: int | None,
        maximum_duration_rank: int | None,
        state_rank: int | None,
        maximum_state_rank: int | None,
    ) -> "WordPathCost":
        if rank < 1 or not 0.0 <= geometry_score <= 1.0:
            raise ValueError("word path received invalid phone evidence")
        if maximum_duration_rank is not None and (
            duration_rank is None or duration_rank < 1
        ):
            raise ValueError("duration-stratified path lacks duration evidence")
        if maximum_state_rank is not None and (state_rank is None or state_rank < 1):
            raise ValueError("state-stratified path lacks manner evidence")
        duration_failed = bool(
            maximum_duration_rank is not None
            and duration_rank is not None
            and duration_rank > maximum_duration_rank
        )
        state_failed = bool(
            maximum_state_rank is not None
            and state_rank is not None
            and state_rank > maximum_state_rank
        )
        if maximum_duration_rank is not None and maximum_state_rank is not None:
            proposal_failed = duration_failed and state_failed
        elif maximum_duration_rank is not None:
            proposal_failed = duration_failed
        elif maximum_state_rank is not None:
            proposal_failed = state_failed
        else:
            proposal_failed = False
        return WordPathCost(
            duration_uncovered=self.duration_uncovered + int(duration_failed),
            state_uncovered=self.state_uncovered + int(state_failed),
            proposal_uncovered=self.proposal_uncovered + int(proposal_failed),
            worst_rank=max(self.worst_rank, rank),
            rank_sum=self.rank_sum + rank,
            phone_count=self.phone_count + 1,
            worst_duration_rank=max(self.worst_duration_rank, duration_rank or 0),
            duration_rank_sum=self.duration_rank_sum + (duration_rank or 0),
            worst_state_rank=max(self.worst_state_rank, state_rank or 0),
            state_rank_sum=self.state_rank_sum + (state_rank or 0),
            worst_geometry_score=max(self.worst_geometry_score, geometry_score),
            geometry_score_sum=self.geometry_score_sum + geometry_score,
        )


@dataclass(frozen=True)
class WordPathCandidate:
    node0: int
    node1: int
    phones: tuple[str, ...]
    words: tuple[str, ...]
    cost: WordPathCost
    phone_edges: tuple[dict[str, object], ...]

    @property
    def objective(self) -> tuple[object, ...]:
        return (*self.cost.objective, self.phones, self.words)


@dataclass(frozen=True)
class _TrieNode:
    children: Mapping[str, int]
    terminals: tuple[int, ...]


class PronunciationTrie:
    """A compact exact-anchor trie grouped by pronunciation homophony."""

    def __init__(self, pronunciations: Iterable[Pronunciation]) -> None:
        grouped: dict[tuple[str, ...], set[str]] = {}
        for entry in pronunciations:
            grouped.setdefault(entry.phones, set()).add(entry.word)
        if not grouped:
            raise ValueError("pronunciation trie requires exact anchors")
        self.classes = tuple(
            (phones, tuple(sorted(words)))
            for phones, words in sorted(grouped.items())
        )
        children: list[dict[str, int]] = [{}]
        terminals: list[list[int]] = [[]]
        for class_index, (phones, _) in enumerate(self.classes):
            node = 0
            for phone in phones:
                following = children[node].get(phone)
                if following is None:
                    following = len(children)
                    children[node][phone] = following
                    children.append({})
                    terminals.append([])
                node = following
            terminals[node].append(class_index)
        self.nodes = tuple(
            _TrieNode(dict(node_children), tuple(node_terminals))
            for node_children, node_terminals in zip(
                children, terminals, strict=True
            )
        )


@dataclass(frozen=True)
class _PartialPath:
    node: int
    trie_node: int
    cost: WordPathCost
    phone_edges: tuple[dict[str, object], ...]


def _edge_map(crop: Mapping[str, object]) -> dict[int, list[dict[str, object]]]:
    landmarks = tuple(int(value) for value in crop["landmarks"])
    if len(landmarks) < 2:
        raise ValueError("word composition needs state landmarks")
    output: dict[int, list[dict[str, object]]] = {
        node: [] for node in range(len(landmarks) - 1)
    }
    for raw_edge in crop["edges"]:
        edge = dict(raw_edge)
        node0 = int(edge["node0"])
        node1 = int(edge["node1"])
        candidates = edge.get("candidates")
        if (
            node0 not in output
            or not node0 < node1 < len(landmarks)
            or not isinstance(candidates, list)
            or not candidates
        ):
            raise ValueError("phone DAG contains an invalid edge")
        output[node0].append(edge)
    for edges in output.values():
        edges.sort(key=lambda edge: (int(edge["node1"]), int(edge["frame1"])))
    return output


def compose_state_word_lattice(
    crop: Mapping[str, object],
    trie: PronunciationTrie,
    *,
    maximum_phones: int = 18,
    maximum_phone_rank: int | None = None,
    maximum_duration_rank: int | None = None,
    maximum_state_rank: int | None = None,
) -> tuple[WordPathCandidate, ...]:
    """Return the best exact-anchor path for every word class and node span."""

    if maximum_phones < 1 or (
        maximum_phone_rank is not None and maximum_phone_rank < 1
    ) or (maximum_duration_rank is not None and maximum_duration_rank < 1) or (
        maximum_state_rank is not None and maximum_state_rank < 1
    ):
        raise ValueError("word lattice limits are invalid")
    landmarks = tuple(int(value) for value in crop["landmarks"])
    outgoing = _edge_map(crop)
    best_words: dict[tuple[int, int, int], WordPathCandidate] = {}
    for start in range(len(landmarks) - 1):
        current = {
            (start, 0): _PartialPath(
                node=start,
                trie_node=0,
                cost=WordPathCost(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0),
                phone_edges=(),
            )
        }
        for _ in range(maximum_phones):
            following: dict[tuple[int, int], _PartialPath] = {}
            for partial in current.values():
                trie_node = trie.nodes[partial.trie_node]
                for edge in outgoing.get(partial.node, ()):
                    for raw_candidate in edge["candidates"]:
                        candidate = dict(raw_candidate)
                        phone = str(candidate["phone"])
                        child = trie_node.children.get(phone)
                        if child is None:
                            continue
                        rank = int(candidate["rank"])
                        if maximum_phone_rank is not None and rank > maximum_phone_rank:
                            continue
                        duration_rank = (
                            int(candidate["duration_rank"])
                            if "duration_rank" in candidate
                            else None
                        )
                        state_rank = (
                            int(candidate["state_profile_rank"])
                            if "state_profile_rank" in candidate
                            else None
                        )
                        cost = partial.cost.extend(
                            rank,
                            float(candidate["score"]),
                            duration_rank=duration_rank,
                            maximum_duration_rank=maximum_duration_rank,
                            state_rank=state_rank,
                            maximum_state_rank=maximum_state_rank,
                        )
                        phone_edge = {
                            "phone": phone,
                            "rank": rank,
                            "geometry_score": float(candidate["score"]),
                            **(
                                {"duration_rank": duration_rank}
                                if duration_rank is not None
                                else {}
                            ),
                            **(
                                {"state_profile_rank": state_rank}
                                if state_rank is not None
                                else {}
                            ),
                            "node0": int(edge["node0"]),
                            "node1": int(edge["node1"]),
                            "frame0": int(edge["frame0"]),
                            "frame1": int(edge["frame1"]),
                        }
                        path = _PartialPath(
                            node=int(edge["node1"]),
                            trie_node=child,
                            cost=cost,
                            phone_edges=(*partial.phone_edges, phone_edge),
                        )
                        state_key = (path.node, child)
                        incumbent = following.get(state_key)
                        if incumbent is None or (
                            path.cost.objective,
                            tuple(
                                (item["node0"], item["node1"])
                                for item in path.phone_edges
                            ),
                        ) < (
                            incumbent.cost.objective,
                            tuple(
                                (item["node0"], item["node1"])
                                for item in incumbent.phone_edges
                            ),
                        ):
                            following[state_key] = path
            if not following:
                break
            for partial in following.values():
                for class_index in trie.nodes[partial.trie_node].terminals:
                    phones, words = trie.classes[class_index]
                    candidate = WordPathCandidate(
                        node0=start,
                        node1=partial.node,
                        phones=phones,
                        words=words,
                        cost=partial.cost,
                        phone_edges=partial.phone_edges,
                    )
                    key = (start, partial.node, class_index)
                    incumbent = best_words.get(key)
                    if incumbent is None or candidate.objective < incumbent.objective:
                        best_words[key] = candidate
            current = following
    return tuple(
        sorted(
            best_words.values(),
            key=lambda item: (item.node0, item.node1, item.objective),
        )
    )


def word_candidate_to_dict(candidate: WordPathCandidate) -> dict[str, object]:
    return {
        "node0": candidate.node0,
        "node1": candidate.node1,
        "phones": list(candidate.phones),
        "words": list(candidate.words),
        "cost": {
            "duration_uncovered": candidate.cost.duration_uncovered,
            "state_uncovered": candidate.cost.state_uncovered,
            "proposal_uncovered": candidate.cost.proposal_uncovered,
            "worst_rank": candidate.cost.worst_rank,
            "mean_rank": candidate.cost.mean_rank,
            "rank_sum": candidate.cost.rank_sum,
            "phone_count": candidate.cost.phone_count,
            "worst_duration_rank": candidate.cost.worst_duration_rank,
            "mean_duration_rank": (
                candidate.cost.duration_rank_sum / candidate.cost.phone_count
            ),
            "worst_state_rank": candidate.cost.worst_state_rank,
            "mean_state_rank": (
                candidate.cost.state_rank_sum / candidate.cost.phone_count
            ),
            "worst_geometry_score": candidate.cost.worst_geometry_score,
            "mean_geometry_score": candidate.cost.mean_geometry_score,
        },
        "phone_edges": list(candidate.phone_edges),
    }
