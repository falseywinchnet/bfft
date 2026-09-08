#!/usr/bin/env python3
"""Regression gates for default Meyer splitting and the old hard control."""

from __future__ import annotations

import unittest

import numpy as np

import bfft


def pure_edge(size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.mgrid[:size, :size].astype(np.float64)
    smooth = 92.0 + 10.0 * x / size
    circle = (x - 0.28 * size) ** 2 + (y - 0.31 * size) ** 2 < (
        0.15 * size
    ) ** 2
    rectangle = (
        (x > 0.51 * size) & (x < 0.86 * size)
        & (y > 0.20 * size) & (y < 0.82 * size)
    )
    jump = 78.0 * circle + 53.0 * rectangle
    return smooth + jump, jump


def pure_carrier(size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.mgrid[:size, :size].astype(np.float64)
    smooth = 96.0 + 8.0 * x / size - 5.0 * y / size
    distance = np.minimum.reduce((
        x - 0.16 * size, 0.87 * size - x,
        y - 0.18 * size, 0.84 * size - y,
    ))
    taper = np.clip((distance - 4.0) / 18.0, 0.0, 1.0)
    carrier = 18.0 * taper * (
        np.cos(2.0 * np.pi * (x + 0.35 * y) / 17.0 + 0.21)
        + 0.55 * np.cos(2.0 * np.pi * (x - 1.4 * y) / 7.0 - 0.43)
    )
    return smooth + carrier, carrier


class MeyerDeepJumpContractTest(unittest.TestCase):
    def test_default_is_the_validated_quality_flow_schedule(self):
        source, _ = pure_carrier()
        plan = bfft.MeyerPlan(
            source.shape, lam=0.05, mu=40.0, passes=1, threads=1
        )
        default = plan.split(source)
        explicit = plan.split_flow_jump(
            source, prefix_passes=4, horizon=10,
            settle_passes=2, jump_count=5,
        )
        for left, right in zip(default, explicit):
            np.testing.assert_array_equal(left, right)

    def test_fixed_operator_is_translation_covariant_and_exact(self):
        source, _ = pure_edge(64)
        plan = bfft.MeyerPlan(
            source.shape, lam=0.05, mu=40.0, passes=1, threads=1
        )
        cartoon, texture = plan.split(source)
        shift = (7, 11)
        moved_cartoon, moved_texture = plan.split(
            np.roll(source, shift, axis=(0, 1))
        )
        np.testing.assert_allclose(
            moved_cartoon,
            np.roll(cartoon, shift, axis=(0, 1)),
            rtol=2e-13,
            atol=8e-13,
        )
        np.testing.assert_allclose(
            moved_texture,
            np.roll(texture, shift, axis=(0, 1)),
            rtol=2e-13,
            atol=8e-13,
        )
        np.testing.assert_allclose(
            source, cartoon + texture, rtol=0.0, atol=3e-14
        )

    def test_pure_edge_tracks_fused64_better_than_the_hard_control(self):
        source, _jump = pure_edge()
        plan = bfft.MeyerPlan(
            source.shape, lam=0.05, mu=40.0, passes=1, threads=1
        )
        _cartoon, texture = plan.split(source)
        fused = bfft.MeyerPlan(
            source.shape, lam=0.05, mu=40.0, passes=64, threads=1
        ).split_legacy(source)[1]
        hard = plan.split_jump_measure(source, virtual_passes=12)[1]
        self.assertLess(
            np.linalg.norm(texture - fused), np.linalg.norm(hard - fused)
        )

    def test_finite_flow_retains_pure_carrier(self):
        source, carrier = pure_carrier()
        plan = bfft.MeyerPlan(
            source.shape, lam=0.05, mu=40.0, passes=1, threads=1
        )
        cartoon, texture = plan.split(source)
        gain = float(
            np.sum(texture * carrier) / np.sum(carrier * carrier)
        )
        self.assertGreater(gain, 0.85)
        np.testing.assert_allclose(
            source, cartoon + texture, rtol=0.0, atol=3e-14
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
