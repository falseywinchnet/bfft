import numpy as np

from experiments.transport_graph_scattering import (
    scattering_anchor_field,
    transport_graph_scattering,
)


def _cycle_graph(director_phase: float = 0.0) -> dict:
    theta = np.linspace(0.0, np.pi, 4, endpoint=False) + director_phase
    anisotropy = 0.5
    qxx = 1.0 + anisotropy * np.cos(2.0 * theta)
    qxy = anisotropy * np.sin(2.0 * theta)
    qyy = 1.0 - anisotropy * np.cos(2.0 * theta)
    return {
        "graph": {
            "cells": 4,
            "area": np.ones(4),
            "node_energy": np.array([1.0, 2.0, 1.0, 2.0]),
            "node_qxx": qxx,
            "node_qxy": qxy,
            "node_qyy": qyy,
            "edge": {
                "first": np.array([0, 1, 2, 3], dtype=np.int32),
                "second": np.array([1, 2, 3, 0], dtype=np.int32),
                "length": np.ones(4),
            },
        },
    }


def test_scattering_is_invariant_to_global_director_rotation():
    first = transport_graph_scattering(
        _cycle_graph(0.0), scales=3)["descriptor"]
    rotated = transport_graph_scattering(
        _cycle_graph(0.37), scales=3)["descriptor"]
    assert np.allclose(first, rotated, atol=1e-12)


def test_scattering_anchor_is_exact_at_anchor():
    result = transport_graph_scattering(_cycle_graph(), scales=3)
    field = scattering_anchor_field(result, 2, maximum_scale=1)
    assert field.shape == (4,)
    assert field[2] == 1.0
    assert result["column_scale"].shape == (result["feature_count"],)
    assert result["column_scale"].max() == 3
