import unittest
import numpy as np
from experiments.conv_exact_fusion.core import operator,reference_source
from experiments.conv_admission_band.core import Operator,ROOT

class FusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base=operator('base');cls.fast=operator('rolling')

    def equal_bits(self,a,b):
        np.testing.assert_array_equal(a.view(np.uint32),b.view(np.uint32))

    def test_ties_zeros_and_extreme_dynamic_ranges(self):
        rng=np.random.default_rng(6309)
        for n in (5,9,33,129):
            for x in [rng.integers(-2,3,size=(n,31)).astype(np.float32),
                      rng.normal(size=(n,31)).astype(np.float32),
                      (rng.normal(size=(n,31))*10**rng.uniform(-15,15,size=(n,31))).astype(np.float32),
                      np.zeros((n,31),np.float32)]:
                self.equal_bits(self.base.profile(x),self.fast.profile(x))

    def test_complete_reduction_and_two_order_synthesis(self):
        rng=np.random.default_rng(519)
        for x in [rng.random((37,29),dtype=np.float32),rng.random((23,31,3),dtype=np.float32)]:
            for shape in [(9,7),(97,113)]:self.equal_bits(self.base.resize(x,shape),self.fast.resize(x,shape))

    def test_guarded_projection_remains_equivalent(self):
        other=operator('projection');rng=np.random.default_rng(835)
        x=(rng.normal(size=(65,31))*10**rng.uniform(-15,15,size=(65,31))).astype(np.float32)
        self.equal_bits(self.base.profile(x),other.profile(x))

    def test_production_is_promoted_rolling_implementation(self):
        source=(ROOT/'standalone_conv_resize_demo/native/conv_native.c').read_text()
        self.assertIn('CONV_LEDGER_COST_BEGIN',source)
        self.assertNotIn('CONV_LEDGER_COST_BEGIN',reference_source(source))
        actual=Operator('conv');x=np.random.default_rng(615).random((65,37),dtype=np.float32)
        self.equal_bits(actual.profile(x),self.base.profile(x))

if __name__=='__main__':unittest.main()
