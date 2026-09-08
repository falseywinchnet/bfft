#!/usr/bin/env python3
"""Estimate one unlabeled radio-to-atlas physical row conversion."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .physical_interval_geometry import (
    combine_physical_interval_atlases,
    load_physical_interval_atlas,
    physical_interval_signature,
)
from .population_interval_alignment import estimate_population_row_multiplier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("region_clouds", type=Path)
    parser.add_argument("physical_atlas", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    atlas = combine_physical_interval_atlases(
        load_physical_interval_atlas(path) for path in args.physical_atlas
    )
    with np.load(args.region_clouds, allow_pickle=False) as document:
        clouds = np.asarray(document["clouds"], dtype=np.float64)
    surfaces = np.stack(
        [
            physical_interval_signature(
                cloud,
                atlas.time_bins,
                atlas.row_quantiles,
                atlas.row_scale,
            )[0]
            for cloud in clouds
        ]
    )
    alignment = estimate_population_row_multiplier(atlas.surfaces, surfaces)
    result = {
        "method": "unlabeled_log_wasserstein_adjacent_interval_alignment",
        "selection_status": "no radio phone or transcript labels consumed",
        "region_clouds": str(args.region_clouds),
        "physical_atlases": [str(path) for path in args.physical_atlas],
        "radio_cloud_count": int(clouds.shape[0]),
        **asdict(alignment),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
