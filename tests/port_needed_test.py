"""Contracts for the native vision port queue."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "viewer"))
sys.path.insert(0, str(ROOT / "experiments"))

from port_needed.anisotropic_edge_cost import build_edge_costs
from port_needed.frozen_meyer_geometry import build_frozen_geometry
from port_needed.two_label_transport import walk_two_labels


def _fixture(height=32, width=40):
    yy, xx = np.mgrid[:height, :width]
    rgb = np.stack((
        0.2 + 0.7 * (xx >= width // 2),
        0.15 + 0.6 * yy / max(height - 1, 1),
        0.5 + 0.25 * np.sin(0.4 * xx),
    ), axis=2)
    return np.clip(rgb, 0.0, 1.0)


def test_frozen_geometry_uses_one_target_decomposition():
    geometry = build_frozen_geometry(
        _fixture(), tgfd_sweeps=4, flow_sweeps=4, threads=1)
    assert geometry["target_decompositions"] == 1.0
    assert geometry["measure"].dtype == np.float32
    assert np.isclose(np.sum(geometry["measure"]), 1.0, atol=1e-6)


def test_monotone_bucket_matches_heap_distances():
    geometry = build_frozen_geometry(
        _fixture(), tgfd_sweeps=4, flow_sweeps=4, threads=1)
    costs = build_edge_costs(geometry, 1.5)
    centers = np.array(((0.2, 0.2), (0.8, 0.25), (0.5, 0.8)))
    heap = walk_two_labels(centers, costs, queue="heap")
    bucket = walk_two_labels(centers, costs, queue="bucket")
    assert np.allclose(heap[2], bucket[2], atol=1e-9, rtol=0.0)
    assert np.allclose(heap[3], bucket[3], atol=1e-9, rtol=0.0)


if __name__ == "__main__":
    test_frozen_geometry_uses_one_target_decomposition()
    test_monotone_bucket_matches_heap_distances()
    print("port-needed contracts: ok")
