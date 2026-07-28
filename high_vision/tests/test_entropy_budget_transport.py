"""Checks for the entropy-budgeted registration operator."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "experiments"))

from entropy_budget_transport import (  # noqa: E402
    EntropyBench,
    StreamingEntropyTransport,
    cross_predictive_evidence,
    entropy_project_scores,
)
from sparse_ray_transport import RayBench  # noqa: E402


def test_entropy_projection_hits_requested_geometric_support():
    rng = np.random.default_rng(17)
    scores = rng.normal(size=(12, 31))

    probability, info = entropy_project_scores(scores, 8.0)

    np.testing.assert_allclose(
        probability.sum(axis=1), 1.0, atol=1e-12)
    assert abs(info["effective_shifts"] - 8.0) < 1e-8
    assert abs(info["entropy_residual_nats"]) < 1e-8


def test_full_entropy_budget_is_exactly_uniform():
    scores = np.arange(35, dtype=np.float64).reshape(5, 7)

    probability, info = entropy_project_scores(scores, 7.0)

    np.testing.assert_allclose(probability, 1.0 / 7.0)
    assert info["inverse_temperature"] == 0.0
    assert np.isinf(info["temperature"])


def test_cross_predictive_evidence_is_bidirectionally_symmetric():
    rng = np.random.default_rng(23)
    first = rng.normal(size=(6, 19))
    second = rng.normal(size=(6, 19))
    budgets = (2.0, 4.0, 8.0, 16.0)

    forward = cross_predictive_evidence(
        first,
        second,
        budgets,
        pixels=64,
        bisection_steps=30,
    )
    reverse = cross_predictive_evidence(
        second,
        first,
        budgets,
        pixels=64,
        bisection_steps=30,
    )

    np.testing.assert_allclose(forward, reverse, atol=1e-14)


def test_streaming_operator_resets_on_discontinuity():
    config = RayBench(
        grid=16,
        frames=12,
        batch=4,
        shift_radius=4,
        photons_at_white=0.08,
    )
    entropy = EntropyBench(
        budgets=(2.0, 4.0, 8.0, 16.0, 32.0),
        bisection_steps=16,
        robustness_seeds=1,
    )
    stream = StreamingEntropyTransport(config, entropy)
    rng = np.random.default_rng(41)
    counts = rng.poisson(
        0.04, size=(12, 16, 16)).astype(np.uint16)

    _, initialized = stream.push(counts[:4])
    assert initialized["initialized"]
    _, advanced = stream.push(counts[4:8])
    assert not advanced["initialized"]
    assert stream.support == 8
    assert stream.support_changes <= 1

    _, restarted = stream.push(counts[8:], discontinuity=True)
    assert restarted["initialized"]
    assert stream.support == 4
    assert stream.selected_budget == 32.0
    np.testing.assert_array_equal(stream.evidence_state, 0.0)
