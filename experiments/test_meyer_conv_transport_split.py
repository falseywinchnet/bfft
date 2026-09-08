"""Invariants for the CONV-raised determinant-one Meyer transport state."""

from __future__ import annotations

import unittest

import numpy as np

from experiments.meyer_conv_transport_split import conv_raised_transport_metric


class MeyerConvTransportSplitTests(unittest.TestCase):
    def test_constant_has_identity_metric(self) -> None:
        metric=conv_raised_transport_metric(np.full((32,33),17.0))
        self.assertLess(np.max(np.abs(metric["metric_xx"]-1.0)),1e-14)
        self.assertLess(np.max(np.abs(metric["metric_xy"])),1e-14)
        self.assertLess(np.max(np.abs(metric["metric_yy"]-1.0)),1e-14)

    def test_metric_is_determinant_one_and_bounded(self) -> None:
        rng=np.random.default_rng(77)
        image=rng.standard_normal((32,35))
        metric=conv_raised_transport_metric(image)
        xx=np.asarray(metric["metric_xx"])
        xy=np.asarray(metric["metric_xy"])
        yy=np.asarray(metric["metric_yy"])
        determinant=xx*yy-xy*xy
        self.assertLess(np.max(np.abs(determinant-1.0)),2e-14)
        self.assertGreater(np.min(xx),0.0)
        self.assertLessEqual(metric["metric_condition_maximum"],np.e**2+1e-12)
        self.assertEqual(metric["chart_recomposition_count_minimum"],3)
        self.assertEqual(metric["chart_recomposition_count_maximum"],3)


if __name__=="__main__":unittest.main()
