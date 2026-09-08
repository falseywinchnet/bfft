import math
import unittest
import numpy as np
from experiments.conv_admission_band.core import Operator,VARIANTS,subdivide,word,subsequence,jet_bank
from experiments.conv_distilled_core import distilled_conv_resize
from experiments.convstar import ordered_sign_ledger,raw_current_jet_bank

class RefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ops={name:Operator(name) for name in VARIANTS}

    def test_reference_is_actual_demo(self):
        y=np.random.default_rng(31).random((23,19)).astype(np.float32)
        for shape in [(47,39),(9,7)]:
            np.testing.assert_array_equal(self.ops['conv'].resize(y,shape),distilled_conv_resize(y,shape))

    def test_refined_jet_moments(self):
        from fractions import Fraction
        from experiments.convstar import FUSED_CURRENT_KERNELS
        np.testing.assert_allclose(np.array(jet_bank(2),dtype=float),FUSED_CURRENT_KERNELS,atol=0,rtol=0)
        for r in (3,4):
            bank=jet_bank(r)
            for k in range(2*r+1):
                currents=[sum(a*Fraction(x)**k for a,x in zip(row,range(-r,r+2))) for row in bank]
                self.assertEqual(sum(currents),int(k>0))
                self.assertEqual(5*currents[0],int(k==1))
                self.assertEqual(5*currents[4],k)
                self.assertEqual(20*(currents[1]-currents[0]),2*int(k==2))
                self.assertEqual(20*(currents[4]-currents[3]),k*(k-1))

    def test_subdivision_preserves_polynomial(self):
        a=np.random.default_rng(7).normal(size=5)
        for depth in (1,2,3):
            refined=subdivide(a,depth)
            for index,b in enumerate(refined):
                for t in (.0,.13,.57,1.):
                    u=(index+t)/len(refined)
                    basis=lambda v:np.array([math.comb(4,k)*v**k*(1-v)**(4-k) for k in range(5)])
                    self.assertAlmostEqual(a@basis(u),b@basis(t),places=14)

    def test_stationary_inflection_retained(self):
        a=np.array([.25,.0,-1/12,.0,.25],dtype=np.float32)/5
        for name in ['uniform1','word1','word2','ray1']:
            c=self.ops[name].fibre(a,np.ones(5),float(a.sum(dtype=np.float64)))
            np.testing.assert_allclose(c,a,atol=1e-8,rtol=0)
        self.assertGreater(np.max(abs(self.ops['conv'].fibre(a,np.ones(5),float(a.sum()))-a)),.01)

    def test_fibre_mass_and_actual_topology(self):
        rng=np.random.default_rng(391)
        u=np.linspace(0,1,4097)
        basis=np.array([math.comb(4,k)*u**k*(1-u)**(4-k) for k in range(5)])
        for _ in range(150):
            a=rng.normal(size=5).astype(np.float32)
            signs=rng.choice([-1,1],size=5).astype(np.int8)
            delta=float(rng.uniform(-1,1))
            if np.all(signs==signs[0]):delta=signs[0]*abs(delta)
            for name,op in self.ops.items():
                if name=='raw':continue
                c=op.fibre(a,signs,delta)
                self.assertLess(abs(c.sum(dtype=float)-delta),2e-6)
                self.assertTrue(subsequence(word(c@basis,tol=1e-7),word(signs)),(name,a,signs,c))

    def test_line_variation_and_cardinality(self):
        rng=np.random.default_rng(17)
        lines=np.column_stack([rng.normal(size=33),np.arange(33),np.zeros(33),np.sin(np.arange(33)*2.3)])
        for name,op in self.ops.items():
            if name=='raw':continue
            z=op.axis(lines,1025,0)
            np.testing.assert_array_equal(z[::32],lines.astype(np.float32))
            for k in range(4):
                self.assertLessEqual(len(word(np.diff(z[:,k]),1e-6)),len(word(np.diff(lines[:,k]),1e-6)))

    def test_refined_jet_native_profile(self):
        for radius in (3,4):
            raw=Operator('raw',radius);admitted=Operator('conv',radius)
            rng=np.random.default_rng(172);lines=rng.normal(size=(33,5)).astype(np.float32)
            profile=raw.profile(lines)
            bank=np.array(jet_bank(radius),dtype=float)
            for i in range(radius,33-radius-1):
                expected=bank@lines[i-radius:i+radius+2].astype(float)
                np.testing.assert_allclose(profile[i],expected,atol=5e-7,rtol=1e-6)
            z=admitted.axis(lines,1025,0)
            np.testing.assert_array_equal(z[::32],lines)
            for k in range(5):
                self.assertLessEqual(len(word(np.diff(z[:,k]),1e-6)),len(word(np.diff(lines[:,k]),1e-6)))

if __name__=='__main__':unittest.main()
