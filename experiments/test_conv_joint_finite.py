import unittest
from unittest.mock import patch
import numpy as np
from scipy import sparse
from experiments.conv_joint_finite import admit,finite_capacity


class FiniteJointTests(unittest.TestCase):
    def test_overlapping_entity_capacity_preserves_affine_constraints(self):
        rng=np.random.default_rng(982)
        a=rng.normal(size=(41,12));base=rng.normal(size=12)
        lower=a@base-rng.uniform(0,.5,41);target=base+rng.normal(size=12)
        out,d=finite_capacity(sparse.csr_matrix(a),lower,base,target,np.repeat(np.arange(4),3))
        self.assertTrue(d['accepted']);self.assertGreaterEqual((a@out-lower).min(),-1e-12)

    def test_infeasible_baseline_is_not_silently_repaired(self):
        a=sparse.eye(2,format='csr');lower=np.zeros(2)
        out,d=finite_capacity(a,lower,np.array([-1.,1.]),np.array([-2.,1.]),np.arange(2))
        self.assertIsNone(out);self.assertFalse(d['accepted'])
        out,d=finite_capacity(a,lower,np.array([-1.,1.]),np.ones(2),np.arange(2))
        np.testing.assert_array_equal(out,np.ones(2));self.assertTrue(d['candidate_passed'])

    def test_direct_shared_jet_invariants_without_solves(self):
        y,x=np.mgrid[:7,:7];source=np.sin(1.3*x+.8*y)
        with patch('numpy.linalg.solve',side_effect=AssertionError('linear solve forbidden')), \
             patch('scipy.optimize.minimize',side_effect=AssertionError('optimization forbidden')), \
             patch('experiments.conv_joint_potential.qp_admm',side_effect=AssertionError('QP forbidden')):
            p,d=admit(source);pt,dt=admit(source.T);pr,dr=admit(source[:,::-1])
        self.assertTrue(d['accepted']);self.assertTrue(dt['accepted']);self.assertTrue(dr['accepted'])
        np.testing.assert_array_equal(p[::5,::5],source)
        np.testing.assert_allclose(p,pt.T,atol=1e-12)
        np.testing.assert_allclose(p,pr[:,::-1],atol=1e-12)
        np.testing.assert_allclose(2*p[:,5:-1:5],p[:,4:-2:5]+p[:,6::5],atol=1e-12)
        np.testing.assert_allclose(p[:,3:-3:5]-2*p[:,4:-2:5]+2*p[:,6::5]-p[:,7::5],0,atol=1e-12)
        np.testing.assert_allclose(2*p[5:-1:5],p[4:-2:5]+p[6::5],atol=1e-12)
        np.testing.assert_allclose(p[3:-3:5]-2*p[4:-2:5]+2*p[6::5]-p[7::5],0,atol=1e-12)
        self.assertGreaterEqual(d['minimum_admitted_margin'],-1e-11)

    def test_affine_field_passes_exactly(self):
        y,x=np.mgrid[:6,:7];source=.3*x+.7*y+2
        p,d=admit(source)
        self.assertTrue(d['candidate_passed'])
        yy,xx=np.mgrid[:26,:31]/5
        np.testing.assert_allclose(p,.3*xx+.7*yy+2,atol=2e-12)


if __name__=='__main__':
    unittest.main()
