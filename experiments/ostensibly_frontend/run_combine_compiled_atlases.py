#!/usr/bin/env python3
"""Concatenate compatible compiled phone or boundary atlases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .compiled_boundary_atlas import (
    CompiledBoundaryAtlas,
    load_boundary_atlas,
    save_boundary_atlas,
)
from .compiled_phone_atlas import (
    CompiledPhoneAtlas,
    load_phone_atlas,
    save_phone_atlas,
)


def compatible(atlases) -> None:
    first = atlases[0]
    for atlas in atlases[1:]:
        if (
            atlas.quantile_count != first.quantile_count
            or atlas.projection_count != first.projection_count
            or atlas.projection_seed != first.projection_seed
            or atlas.metric_scales != first.metric_scales
        ):
            raise ValueError("compiled atlases use incompatible projection bases")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("phone", "boundary"))
    parser.add_argument("atlases", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.kind == "phone":
        atlases = [load_phone_atlas(path) for path in args.atlases]
        compatible(atlases)
        first = atlases[0]
        combined = CompiledPhoneAtlas(
            labels=np.concatenate([atlas.labels for atlas in atlases]),
            witnesses=np.concatenate([atlas.witnesses for atlas in atlases]),
            projections=np.concatenate([atlas.projections for atlas in atlases]),
            quantile_count=first.quantile_count,
            projection_count=first.projection_count,
            projection_seed=first.projection_seed,
            metric_scales=first.metric_scales,
            provenance={
                "sources": [dict(atlas.provenance or {}) for atlas in atlases]
            },
        )
        save_phone_atlas(args.out, combined)
        labels = combined.labels
    else:
        atlases = [load_boundary_atlas(path) for path in args.atlases]
        compatible(atlases)
        first = atlases[0]
        combined = CompiledBoundaryAtlas(
            phones=np.concatenate([atlas.phones for atlas in atlases]),
            witnesses=np.concatenate([atlas.witnesses for atlas in atlases]),
            projections=np.concatenate([atlas.projections for atlas in atlases]),
            quantile_count=first.quantile_count,
            projection_count=first.projection_count,
            projection_seed=first.projection_seed,
            metric_scales=first.metric_scales,
            provenance={
                "sources": [dict(atlas.provenance or {}) for atlas in atlases]
            },
        )
        save_boundary_atlas(args.out, combined)
        labels = np.asarray(["-".join(pair) for pair in combined.phones])
    print(
        json.dumps(
            {
                "output": str(args.out),
                "kind": args.kind,
                "source_count": len(atlases),
                "occurrence_count": combined.projections.shape[0],
                "label_count": len(set(labels.tolist())),
                "size_bytes": args.out.stat().st_size,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
