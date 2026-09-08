import unittest
import numpy as np
from experiments.conv_compatible_current import compatible_proposal,admitted,joint_admitted,profile_range
from experiments.convstar import synthesize_polyphase,raw_current_jet_bank,ordered_sign_ledger
from experiments.conv_functional_current import minimum


class CompatibleCurrentTests(unittest.TestCase):
    def test_polynomial_reproduction_through_degree_five(self):
        nodes=np.arange(17,dtype=float)
        query=np.linspace(0,16,129)
        for k in range(6):
            y=((nodes-8)/8)**k
            a,d=compatible_proposal(y)
            z=synthesize_polyphase(y[:,None],a,8)[:,0]
            np.testing.assert_allclose(z,((query-8)/8)**k,atol=5e-12)
            np.testing.assert_allclose(a.sum(axis=1)[:,0],d[:,0],atol=1e-13)

    def test_proposal_compatible_jets_and_reflection(self):
        y=np.random.default_rng(873).normal(size=19)
        a,d=compatible_proposal(y)
        np.testing.assert_allclose(a[:-1,4],a[1:,0],atol=1e-13)
        np.testing.assert_allclose(a[:-1,4]-a[:-1,3],a[1:,1]-a[1:,0],atol=1e-13)
        np.testing.assert_allclose(a[:-1,4]-2*a[:-1,3]+a[:-1,2],
                                   a[1:,2]-2*a[1:,1]+a[1:,0],atol=1e-12)
        np.testing.assert_allclose(a[:-1,4]-3*a[:-1,3]+3*a[:-1,2]-a[:-1,1],
                                   a[1:,3]-3*a[1:,2]+3*a[1:,1]-a[1:,0],atol=1e-12)
        b,_=compatible_proposal(y[::-1])
        np.testing.assert_allclose(b,-a[::-1,::-1],atol=1e-11)

    def test_admitted_mass_topology_and_affine_covariance(self):
        y=np.tanh((np.arange(17)-8.3)/.8)
        c,a,s=admitted(y)
        np.testing.assert_allclose(c.sum(axis=1)[:,0],np.diff(y),atol=1e-11)
        for i in range(16):
            self.assertGreaterEqual(minimum(c[i,:,0])[0],-1e-10)
        z=synthesize_polyphase(y[:,None],c,8)[:,0]
        np.testing.assert_array_equal(z[::8],y)
        transformed=admitted(-2*y+7)[0]
        # Near double contacts the floating-point exchange oracle is less
        # accurate in coefficients than in the integrated curve.
        np.testing.assert_allclose(transformed,-2*c,atol=5e-6)

    def test_compact_ledger_is_reused(self):
        y=np.random.default_rng(71).normal(size=15)
        c,a,s=admitted(y)
        raw,d=raw_current_jet_bank(y[:,None])
        np.testing.assert_array_equal(s,ordered_sign_ledger(raw,d))
        mixed=np.any(s!=s[:,0:1,:],axis=1)[:,0]
        self.assertTrue(np.all(s[mixed]*c[mixed]>=-1e-12))

    def test_joint_affine(self):
        y=np.arange(9,dtype=float)*.7+.3
        for continuity in (1,2):
            c=joint_admitted(y,continuity)
            np.testing.assert_allclose(c,.7/5,atol=1e-7)

    def test_continuous_support_range(self):
        nodes=np.arange(17,dtype=float)
        for y in (np.sin(2*np.pi*(.4*nodes+.1)),
                  ((nodes>5)&(nodes<11)).astype(float)):
            c,a,s=admitted(y,bounded=True)
            np.testing.assert_allclose(c.sum(axis=1)[:,0],np.diff(y),atol=1e-11)
            for i in range(16):
                low,high=profile_range(c[i,:,0])
                support=y[max(0,i-2):min(len(y),i+4)]
                self.assertGreaterEqual(low+y[i],support.min()-1e-10)
                self.assertLessEqual(high+y[i],support.max()+1e-10)

    def test_envelope_does_not_increase_line_excursion(self):
        x=np.arange(25,dtype=float)
        for y in (np.sin(2*np.pi*(.4*x+.17)),
                  np.sin(2*np.pi*(.03*x+.19*x*x/24+.17)),
                  ((x>8)&(x<15)).astype(float)):
            base=admitted(y,proposal='compact')[0]
            new=admitted(y,envelope=True)[0]
            def limits(c):
                rr=[np.array(profile_range(c[i,:,0]))+y[i] for i in range(len(y)-1)]
                return np.min(rr),np.max(rr)
            old_min,old_max=limits(base);new_min,new_max=limits(new)
            self.assertGreaterEqual(new_min,old_min-1e-10)
            self.assertLessEqual(new_max,old_max+1e-10)

    def test_tiny_mass_with_large_proposal(self):
        from experiments.conv_functional_current import admit_cell,SLOPE_GRAM
        a=np.array([.3,-.8,.6,-.4,.3])
        a[-1]-=a.sum()-1e-12
        c=admit_cell(a,1e-12,1,SLOPE_GRAM,True)
        self.assertAlmostEqual(c.sum()/1e-12,1,places=8)
        self.assertGreaterEqual(minimum(c)[0],-1e-22)


if __name__=='__main__':
    unittest.main()
