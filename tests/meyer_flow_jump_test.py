#!/usr/bin/env python3
"""Contract gates for the coupled finite-flow Meyer jump."""

from __future__ import annotations

import unittest

import numpy as np

import bfft
from experiments.meyer_deep_jump_causal_audit import (
    pure_carrier_scene,
    pure_edge_scene,
)
from experiments.meyer_semismooth_state_jump import ReducedMeyerMap


def python_oracle(source: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model = ReducedMeyerMap(source, 0.05, 40.0)
    state = model.initial()
    for _ in range(1, 4):
        state = model.step(state)
    for _ in range(5):
        state, _ = model.finite_horizon_jump(
            state, krylov_steps=2, horizon=10
        )
        state = model.step(model.step(state))
    texture = source - state.u - state.w
    return source - texture, texture


class MeyerFlowJumpTest(unittest.TestCase):
    def test_native_matches_independent_semismooth_oracle(self):
        source = pure_edge_scene(32)["source"]
        native = bfft.MeyerPlan(
            source.shape, lam=0.05, mu=40.0, passes=64, threads=1
        ).split_flow_jump(source)
        oracle = python_oracle(source)
        for actual, expected in zip(native, oracle):
            np.testing.assert_allclose(
                actual, expected, rtol=2e-13, atol=3e-12
            )

    def test_recomposition_translation_and_thread_invariance(self):
        source = pure_carrier_scene(64)["source"]
        one = bfft.MeyerPlan(
            source.shape, lam=0.05, mu=40.0, passes=64, threads=1
        ).split_flow_jump(source)
        four = bfft.MeyerPlan(
            source.shape, lam=0.05, mu=40.0, passes=64, threads=4
        ).split_flow_jump(source)
        for left, right in zip(one, four):
            np.testing.assert_array_equal(left, right)
        np.testing.assert_allclose(source, one[0] + one[1], atol=3e-14)

        shift = (9, 13)
        moved = bfft.MeyerPlan(
            source.shape, lam=0.05, mu=40.0, passes=64, threads=1
        ).split_flow_jump(np.roll(source, shift, axis=(0, 1)))
        for actual, expected in zip(moved, one):
            np.testing.assert_allclose(
                actual,
                np.roll(expected, shift, axis=(0, 1)),
                rtol=3e-13,
                atol=5e-12,
            )

    def test_fixed_schedule_beats_hard_jump_and_equal_cost_fused(self):
        for scene in (pure_edge_scene(64), pure_carrier_scene(64)):
            source = scene["source"]
            target = bfft.MeyerPlan(
                source.shape, lam=0.05, mu=40.0, passes=64, threads=1
            ).split_legacy(source)[1]
            flow = bfft.MeyerPlan(
                source.shape, lam=0.05, mu=40.0, passes=1, threads=1
            ).split_flow_jump(source)[1]
            hard = bfft.MeyerPlan(
                source.shape, lam=0.05, mu=40.0, passes=1, threads=1
            ).split_jump_measure(source, virtual_passes=12)[1]
            # Exact residual + two Arnoldi tangents + two settling passes per
            # jump: 4 + 5*(1+2+2) = 29 pass-equivalents.
            equal_cost = bfft.MeyerPlan(
                source.shape, lam=0.05, mu=40.0, passes=29, threads=1
            ).split_legacy(source)[1]
            flow_error = np.linalg.norm(flow - target)
            self.assertLess(flow_error, np.linalg.norm(hard - target))
            self.assertLess(flow_error, np.linalg.norm(equal_cost - target))

    def test_facr_rejects_the_spectral_flow_chart(self):
        source = pure_edge_scene(64)["source"]
        plan = bfft.MeyerPlan(
            source.shape, lam=0.05, mu=40.0,
            passes=64, threads=1, solver=1,
        )
        # A power-of-two periodic shape retains the full spectral path under
        # solver 1, so use a genuinely swept dimension.
        source = pure_edge_scene(64)["source"][:, :63]
        facr = bfft.MeyerPlan(
            source.shape, lam=0.05, mu=40.0,
            passes=64, threads=1, solver=1,
        )
        with self.assertRaises(RuntimeError):
            facr.split_flow_jump(source)
        del plan


if __name__ == "__main__":
    unittest.main(verbosity=2)
