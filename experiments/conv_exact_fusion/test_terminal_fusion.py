import unittest
import numpy as np
from experiments.conv_admission_band.core import Operator
from experiments.conv_exact_fusion.terminal_fusion import FusedTerminal

class TerminalFusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base=Operator('conv')
        cls.variants=[FusedTerminal(),FusedTerminal(preblend=True)]
    def test_rectangular_channels_and_anchor_phases(self):
        rng=np.random.default_rng(606)
        for channels in (None,1,3,4,5,7):
            shape=(7,9) if channels is None else (7,9,channels)
            source=rng.normal(size=shape).astype(np.float32)
            for target in ((7,9),(13,17),(29,41),(7,35),(31,9),(1,1)):
                expected=self.base.synthesize(source,target)
                for variant in self.variants:
                    actual=variant.synthesize(source,target)
                    np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))
    def test_zeros_steps_quantization_and_dynamic_range(self):
        rng=np.random.default_rng(670)
        sources=[np.zeros((9,11),np.float32),np.full((9,11),-0.0,np.float32),
                 np.ones((9,11),np.float32),rng.integers(-2,3,size=(9,11)).astype(np.float32),
                 (rng.normal(size=(9,11))*10**rng.uniform(-15,15,size=(9,11))).astype(np.float32)]
        for source in sources:
            expected=self.base.synthesize(source,(65,83))
            for variant in self.variants:
                actual=variant.synthesize(source,(65,83))
                np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))

class ProductionTerminalTests(unittest.TestCase):
    def test_integrated_wrapper_and_workspace_fallback(self):
        from experiments.conv_exact_fusion.study_production_terminal import ProductionOperator
        from experiments.conv_distilled_core import conv_two_order_synthesis, nodal_current_geometry
        base=ProductionOperator('base');fast=ProductionOperator('fused')
        rng=np.random.default_rng(510)
        for source in (rng.normal(size=(13,9)).astype(np.float32),
                       rng.normal(size=(9,13,5)).astype(np.float32).transpose(1,0,2)):
            for shape in ((51,73),(13,9)):
                expected=base.synthesize(source,shape)
                actual=fast.synthesize(source,shape)
                np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))
                eta=nodal_current_geometry(source)[3]
                fallback=conv_two_order_synthesis(source,shape,eta,workspace_megabytes=0)
                np.testing.assert_array_equal(fallback.view(np.uint32),expected.view(np.uint32))

if __name__=='__main__':unittest.main()
