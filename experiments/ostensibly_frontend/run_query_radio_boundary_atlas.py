#!/usr/bin/env python3
"""Fuse isolated and adjacent-region occupation atlas phone rankings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .compiled_boundary_atlas import load_boundary_atlas
from .run_occupation_word_span_battery import joint_gauge_chunks, lane_chunks


def endpoint_ranking(ranking: list[dict], endpoint: int) -> list[dict]:
    best = {}
    for item in ranking:
        phone = str(item["phones"][endpoint])
        candidate = {
            "phone": phone,
            "distance": float(item["distance"]),
            "witness": str(item["witness"]),
        }
        current = best.get(phone)
        if current is None or (candidate["distance"], candidate["witness"]) < (
            current["distance"], current["witness"]
        ):
            best[phone] = candidate
    output = sorted(best.values(), key=lambda item: (item["distance"], item["phone"]))
    for rank, item in enumerate(output, start=1):
        item["rank"] = rank
    return output


def fuse_rankings(channels: dict[str, list[dict]], mode: str) -> list[dict]:
    rank_maps = {
        name: {str(item["phone"]): int(item["rank"]) for item in ranking}
        for name, ranking in channels.items()
    }
    phones = sorted(set().union(*(set(ranks) for ranks in rank_maps.values())))
    output = []
    for phone in phones:
        ranks = {name: values[phone] for name, values in rank_maps.items() if phone in values}
        values = list(ranks.values())
        if mode == "mean":
            score = float(np.mean(values))
        elif mode == "conservative":
            score = float(max(values) + 1e-3 * np.mean(values))
        else:
            raise ValueError("unsupported phone rank fusion")
        output.append(
            {
                "phone": phone,
                "distance": score,
                "rank_score": score,
                "channel_ranks": ranks,
            }
        )
    output.sort(key=lambda item: (item["rank_score"], item["phone"]))
    for rank, item in enumerate(output, start=1):
        item["rank"] = rank
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("region_clouds", type=Path)
    parser.add_argument("center_lattice", type=Path)
    parser.add_argument("boundary_atlas", type=Path)
    parser.add_argument("--points-per-phone", type=int, default=2048)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--report-count", type=int, default=39)
    parser.add_argument("--retain-boundary-count", type=int, default=128)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    center = json.loads(args.center_lattice.read_text())
    with np.load(args.region_clouds) as document:
        clouds = np.asarray(document["clouds"], dtype=np.float64)
    if clouds.shape[0] != len(center["phones"]):
        raise ValueError("region cloud and center lattice counts disagree")
    atlas = load_boundary_atlas(args.boundary_atlas)
    left_channels = [None] * clouds.shape[0]
    right_channels = [None] * clouds.shape[0]
    boundary_rankings = []
    started = perf_counter()
    for index in range(clouds.shape[0] - 1):
        pair = lane_chunks(
            joint_gauge_chunks(
                (clouds[index], clouds[index + 1]), args.points_per_phone
            )
        )
        ranking = atlas.rank(pair, chunk_size=args.chunk_size)
        boundary_rankings.append(
            {
                "boundary_index": index,
                "ranking": ranking[: args.retain_boundary_count],
            }
        )
        right_channels[index] = endpoint_ranking(ranking, 0)
        left_channels[index + 1] = endpoint_ranking(ranking, 1)
        if (index + 1) % 50 == 0:
            print(f"queried {index + 1}/{clouds.shape[0] - 1} boundaries", flush=True)
    query_seconds = perf_counter() - started

    rows = []
    for index, center_row in enumerate(center["phones"]):
        center_ranking = [
            {**item, "rank": rank}
            for rank, item in enumerate(center_row["top5"], start=1)
        ]
        boundary_channels = {}
        if left_channels[index] is not None:
            boundary_channels["left_boundary"] = left_channels[index]
        if right_channels[index] is not None:
            boundary_channels["right_boundary"] = right_channels[index]
        boundary_mean = fuse_rankings(boundary_channels, "mean")
        all_channels = {"center": center_ranking, **boundary_channels}
        mean_rank = fuse_rankings(all_channels, "mean")
        conservative = fuse_rankings(all_channels, "conservative")
        rows.append(
            {
                **{key: value for key, value in center_row.items() if key != "top5"},
                "center_top5": center_ranking[: args.report_count],
                "boundary_top5": boundary_mean[: args.report_count],
                "mean_rank_top5": mean_rank[: args.report_count],
                "conservative_top5": conservative[: args.report_count],
                "top5": mean_rank[: args.report_count],
            }
        )
    result = {
        "method": "isolated_plus_adjacent_occupation_atlas_rank_fusion",
        "selection_status": "diagnostic fusion variants retained independently",
        "center_lattice": str(args.center_lattice),
        "boundary_atlas": str(args.boundary_atlas),
        "phone_count": len(rows),
        "boundary_query_seconds": query_seconds,
        "joint_boundary_pair_swd_rankings": boundary_rankings,
        "phones": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.out), "boundary_query_seconds": query_seconds}, indent=2))


if __name__ == "__main__":
    main()
