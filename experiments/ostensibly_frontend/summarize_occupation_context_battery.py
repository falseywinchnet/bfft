#!/usr/bin/env python3
"""Aggregate fixed cross-speaker occupation-context battery results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median


RANK_FIELDS = {
    "center": "center_only_target_rank",
    "ordered_context": "context_target_rank",
    "boundary_transport": "relational_target_rank",
    "support_geometry": "support_target_rank",
    "conservative_fusion": "conservative_fusion_target_rank",
}

EXACT_FIELDS = {
    "ordered_context": "corresponding_context_instance_rank",
    "boundary_transport": "corresponding_relational_instance_rank",
    "support_geometry": "corresponding_support_instance_rank",
    "conservative_fusion": "corresponding_conservative_fusion_instance_rank",
}


def summarize_results(documents: list[dict]) -> dict:
    if not documents:
        raise ValueError("at least one context battery is required")
    queries = []
    for document in documents:
        summary = document["summary"]
        target = summary["target_phone"]
        query = {
            "target_phone": target,
            "label_ranks": {
                channel: summary[field] for channel, field in RANK_FIELDS.items()
            },
            "exact_instance_ranks": {
                channel: summary[field] for channel, field in EXACT_FIELDS.items()
            },
            "proposal_shortlists": {},
        }
        for name, proposals in document["proposal_shortlists"].items():
            phones = [item["phone"] for item in proposals]
            query["proposal_shortlists"][name] = {
                "target_retained": target in phones,
                "unique_label_count": len(phones),
            }
        queries.append(query)

    channel_summary = {}
    for channel in RANK_FIELDS:
        values = [query["label_ranks"][channel] for query in queries]
        channel_summary[channel] = {
            "median_label_rank": float(median(values)),
            "mean_label_rank": sum(values) / len(values),
            "top_5_hits": sum(value <= 5 for value in values),
            "top_10_hits": sum(value <= 10 for value in values),
        }
        if channel in EXACT_FIELDS:
            exact = [query["exact_instance_ranks"][channel] for query in queries]
            channel_summary[channel]["median_exact_instance_rank"] = float(
                median(exact)
            )

    proposal_summary = {}
    for name in queries[0]["proposal_shortlists"]:
        values = [query["proposal_shortlists"][name] for query in queries]
        proposal_summary[name] = {
            "target_hits": sum(item["target_retained"] for item in values),
            "median_unique_label_count": float(
                median(item["unique_label_count"] for item in values)
            ),
        }
    return {
        "method": "multi_query_cross_speaker_context_audit",
        "query_count": len(queries),
        "channel_summary": channel_summary,
        "proposal_summary": proposal_summary,
        "queries": queries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    documents = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    result = summarize_results(documents)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": result, "output": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
