import unittest
from fractions import Fraction as F
import math
import numpy as np
from experiments.conv_fixed_orders.core import bank,CONFIGS,FixedOrder
from experiments.conv_admission_band.core import jet_bank,Operator

class FixedOrderTests(unittest.TestCase):
    def test_six_five_is_existing_exact_bank(self):
        self.assertEqual(bank(6,5),jet_bank(2))
    def test_raw_mass_and_cardinality(self):
        for taps,d in CONFIGS.values():
            r=(taps-2)//2
            for cell in [None,*range(r)]:
                a=bank(taps,d,cell)
                left=r if cell is None else cell
                self.assertEqual([sum(row[k] for row in a) for k in range(taps)],
                                 [F(int(k==left+1)-int(k==left)) for k in range(taps)])
    def test_interior_reproduction_and_basin_integrals(self):
        for taps,d in CONFIGS.values():
            r=(taps-2)//2;a=bank(taps,d)
            for u in (F(1,4),F(1,2),F(3,4)):
                basis=[F(math.comb(d,j))*u**j*(1-u)**(d-j) for j in range(d+1)]
                tails=[sum(basis[k+1:]) for k in range(d)]
                weights=[F(int(j==r))+sum(tails[k]*a[k][j] for k in range(d)) for j in range(taps)]
                for power in range(min(d,2*r)+1):
                    self.assertEqual(sum(weights[j]*F(j-r)**power for j in range(taps)),u**power)
            op=FixedOrder(taps,d,raw=True)
            x=np.arange(65,dtype=np.float32)/64
            for power in range(1,min(d,2*r)+1):
                z=op.axis(x**power,9,0,basin=True)
                left=(np.arange(1,8)-.5)/8;right=(np.arange(1,8)+.5)/8
                expected=(right**(power+1)-left**(power+1))/((power+1)*(right-left))
                np.testing.assert_allclose(z[1:-1],expected,atol=2e-6,rtol=0)

    def test_native_invariants(self):
        rng=np.random.default_rng(807)
        for taps,d in CONFIGS.values():
            op=FixedOrder(taps,d)
            small=np.arange(taps,dtype=np.float32)
            np.testing.assert_allclose(op.axis(small,(taps-1)*8+1,0),np.arange((taps-1)*8+1)/8,atol=3e-6,rtol=0)
            for x in (np.zeros((17,7),np.float32),np.ones((17,7),np.float32),rng.normal(size=(17,7)).astype(np.float32)):
                cur=op.profile(x)
                self.assertTrue(np.all(np.isfinite(cur)))
                np.testing.assert_allclose(cur.sum(axis=1),np.diff(x,axis=0),atol=2e-6,rtol=0)
                out=op.axis(x,129,0)
                np.testing.assert_array_equal(out[::8],x)
            ramp=np.repeat(np.arange(17,dtype=np.float32)[:,None],7,axis=1)
            np.testing.assert_allclose(op.axis(ramp,129,0)[:,0],np.arange(129)/8,atol=5e-6,rtol=0)
    def test_control_matches_version1(self):
        rng=np.random.default_rng(184)
        x=rng.normal(size=(33,29)).astype(np.float32)
        a=Operator('conv');b=FixedOrder(6,5)
        # Boundary FIR reassociation is explicitly not a bitwise claim.
        np.testing.assert_allclose(a.profile(x),b.profile(x),atol=3e-6,rtol=0)
        np.testing.assert_allclose(a.synthesize(x,(129,113)),b.synthesize(x,(129,113)),atol=4e-6,rtol=0)

if __name__=='__main__':unittest.main()
