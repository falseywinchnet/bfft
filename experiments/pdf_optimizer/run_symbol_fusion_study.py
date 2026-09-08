#!/usr/bin/env python3
"""Recursively fuse pages into terminal-byte-winning shared JBIG2 dictionaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time

import numpy as np

from pdf_optimizer.symbol_fusion import (
    bitmap_profile,
    encode_verified_group,
    signature_match,
)


def _pages(value: str) -> list[int]:
    result = []
    for part in value.split(","):
        if "-" in part:
            first, last = map(int, part.split("-", 1))
            result.extend(range(first, last + 1))
        elif part:
            result.append(int(part))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bitmap-dir", type=Path, required=True)
    parser.add_argument("--generic-dir", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--pages", default="1-468")
    parser.add_argument("--threshold", type=float, default=0.94)
    parser.add_argument("--high-entropy-quantile", type=float, default=0.85)
    parser.add_argument("--minimum-match", type=float, default=0.75)
    parser.add_argument("--frontier", type=int, default=8)
    parser.add_argument("--max-clusters", type=int, default=12)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    pages = _pages(args.pages)
    started = time.monotonic()
    profiles = {
        page: bitmap_profile(
            page,
            args.bitmap_dir / f"{page:04d}.pbm",
            (args.generic_dir / f"{page:04d}.jb2").stat().st_size,
        )
        for page in pages
    }
    entropy_ceiling = float(np.quantile([profile.entropy for profile in profiles.values()], args.high_entropy_quantile))
    report = {
        "threshold": args.threshold,
        "entropy_ceiling": entropy_ceiling,
        "independent": [],
        "clusters": [],
    }
    with tempfile.TemporaryDirectory(prefix="jbig2-fusion-") as directory:
        work = Path(directory)
        independent_cost = {}
        for page in pages:
            measurement = encode_verified_group(
                (page,), bitmap_dir=args.bitmap_dir, encoder=args.encoder,
                decoder=args.decoder, threshold=args.threshold, work=work,
            )
            generic = profiles[page].generic_bytes
            cost = min(generic, measurement.total_bytes)
            independent_cost[page] = cost
            report["independent"].append({
                "page": page, "entropy": profiles[page].entropy,
                "generic_bytes": generic, "symbol_bytes": measurement.total_bytes,
                "selected_bytes": cost, "symbol_exact": measurement.pixel_exact,
            })

        available = {page for page in pages if profiles[page].entropy <= entropy_ceiling}
        for cluster_index in range(args.max_clusters):
            if len(available) < 2:
                break
            ordered = sorted(available, key=lambda page: profiles[page].entropy)
            seed = ordered[len(ordered) // 2]
            members = (seed,)
            cluster_cost = independent_cost[seed]
            cluster_signature = profiles[seed].signature.copy()
            steps = []
            while True:
                ranked = sorted(
                    (
                        (signature_match(cluster_signature, profiles[page].signature), page)
                        for page in available if page not in members
                    ),
                    reverse=True,
                )
                frontier = [(match, page) for match, page in ranked if match >= args.minimum_match][:args.frontier]
                proposals = []
                for match, page in frontier:
                    proposal_pages = tuple(sorted((*members, page)))
                    measurement = encode_verified_group(
                        proposal_pages, bitmap_dir=args.bitmap_dir, encoder=args.encoder,
                        decoder=args.decoder, threshold=args.threshold, work=work,
                    )
                    gain = cluster_cost + independent_cost[page] - measurement.total_bytes
                    proposals.append((gain, match, page, measurement))
                if not proposals:
                    break
                gain, match, page, winner = max(proposals, key=lambda item: (item[0], item[1]))
                if gain <= 0 or not winner.pixel_exact:
                    break
                members = winner.pages
                cluster_cost = winner.total_bytes
                weights = np.array([profiles[item].generic_bytes for item in members], dtype=np.float64)
                cluster_signature = np.average(
                    np.stack([profiles[item].signature for item in members]), axis=0, weights=weights
                )
                norm = np.linalg.norm(cluster_signature)
                if norm:
                    cluster_signature /= norm
                steps.append({"added_page": page, "match": match, "gain_bytes": gain,
                              "cluster_bytes": cluster_cost, "members": members})
            if len(members) > 1:
                available.difference_update(members)
                baseline = sum(independent_cost[page] for page in members)
                report["clusters"].append({
                    "seed": seed, "members": members, "baseline_bytes": baseline,
                    "cluster_bytes": cluster_cost, "gain_bytes": baseline - cluster_cost,
                    "steps": steps,
                })
                print(json.dumps(report["clusters"][-1]), flush=True)
            else:
                available.remove(seed)

    independent_total = sum(independent_cost.values())
    fusion_gain = sum(cluster["gain_bytes"] for cluster in report["clusters"])
    report["generic_total_bytes"] = sum(profile.generic_bytes for profile in profiles.values())
    report["independent_total_bytes"] = independent_total
    report["fusion_total_bytes"] = independent_total - fusion_gain
    report["fusion_gain_bytes"] = fusion_gain
    report["total_seconds"] = time.monotonic() - started
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "generic_total_bytes", "independent_total_bytes", "fusion_total_bytes",
        "fusion_gain_bytes", "total_seconds"
    )}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
