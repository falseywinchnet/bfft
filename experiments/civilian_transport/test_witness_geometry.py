import unittest
import numpy as np
from .witness_geometry import supported_subspace, supported_ray, clean_gain


class WitnessGeometryTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(19)
        self.base=self.rng.normal(size=(12,3))
        self.d=self.rng.normal(size=(12,3,4))
        self.w=np.arange(8)
        self.truth=self.rng.normal(size=(12,3))
        self.y=self.truth[self.w]+.35*self.rng.normal(size=(8,3))

    def test_uniform_bound_on_confidence_event(self):
        r=supported_subspace(self.base,self.d,self.w,self.y,.35)
        noise=((self.y-self.truth[self.w])/.35).reshape(-1)
        self.assertLess(np.linalg.norm(r['witness_basis'].T @ noise),r['radius'])
        gain=clean_gain(self.base,r['mean'],self.truth,self.w)
        self.assertGreaterEqual(gain+1e-10,r['lower_gain'])
        # Check additional arbitrary corrections, beyond the selected one.
        for _ in range(20):
            z=self.rng.normal(size=r['rank'])
            delta=np.einsum('qdr,r->qd',r['transport'],z)
            lower=.35**2*(2*z @ r['projected']-z @ z-2*r['radius']*np.linalg.norm(z))
            gain=clean_gain(self.base,self.base+delta,self.truth,self.w)
            self.assertGreaterEqual(gain+1e-10,lower)

    def test_basis_invariance(self):
        a=supported_subspace(self.base,self.d,self.w,self.y,.35)
        invertible=np.eye(4)+.1*self.rng.normal(size=(4,4))
        b=supported_subspace(self.base,self.d @ invertible,self.w,self.y,.35)
        np.testing.assert_allclose(a['mean'],b['mean'],atol=2e-12)
        self.assertAlmostEqual(a['lower_gain'],b['lower_gain'],places=10)

    def test_spatial_equivariance(self):
        r,_=np.linalg.qr(self.rng.normal(size=(3,3)))
        shift=np.array([30.,-20.,10.])
        a=supported_subspace(self.base,self.d,self.w,self.y,.35)
        transformed=np.einsum('qdk,de->qek',self.d,r)
        b=supported_subspace(self.base @ r+shift,transformed,self.w,self.y @ r+shift,.35)
        np.testing.assert_allclose(b['mean'],a['mean'] @ r+shift,atol=2e-11)
        self.assertAlmostEqual(a['lower_gain'],b['lower_gain'],places=9)

    def test_zero_and_rank_deficient_fields(self):
        a=supported_subspace(self.base,np.zeros_like(self.d),self.w,self.y,.35)
        self.assertEqual(a['rank'],0)
        np.testing.assert_array_equal(a['mean'],self.base)
        d=np.stack([self.d[...,0],2*self.d[...,0]],axis=-1)
        a=supported_subspace(self.base,d,self.w,self.y,.35)
        self.assertEqual(a['rank'],1)
        self.assertLess(a['endpoint_null_response'],1e-12)

    def test_future_unidentifiability_is_not_removed(self):
        d=np.zeros((12,3,1)); d[-1,0,0]=100.
        a=supported_subspace(self.base,d,self.w,self.y,.35)
        self.assertEqual(a['rank'],0)
        self.assertEqual(a['endpoint_null_response'],100.)
        # No supported correction does not imply a zero true future correction.
        np.testing.assert_array_equal(a['mean'],self.base)

    def test_ray_bound_for_all_scales(self):
        direction=self.d[...,0]
        r=supported_ray(self.base,direction,self.w,self.y,.35)
        true_slope=float(np.sum(direction[self.w]*(self.truth[self.w]-self.base[self.w])))
        self.assertGreaterEqual(true_slope,r['score']-r['uncertainty'])
        self.assertGreaterEqual(clean_gain(self.base,r['mean'],self.truth,self.w)+1e-10,r['lower_gain'])

    def test_training_directions_do_not_read_witnesses(self):
        from .witness_study import fixture, propose, WITNESS
        case=fixture('helix',90123,.35,'ordinary')
        a=case['observations'].copy()
        b=a.copy();b[WITNESS]+=1000.;b[61:]=np.nan
        aa=propose(a,.35);bb=propose(b,.35)
        for x,y in zip(aa,bb):
            np.testing.assert_array_equal(x,y)
        np.testing.assert_allclose(aa[0]+aa[1].sum(axis=-1),aa[2],atol=1e-12)

    def test_indistinguishable_motion_sensor_pair(self):
        from .witness_study import fixture
        a=fixture('figure8',90234,.35,'maneuver')
        b=fixture('figure8',90234,.35,'sensor_drift')
        np.testing.assert_array_equal(a['observations'],b['observations'])
        self.assertGreater(np.linalg.norm(a['truth'][-1]-b['truth'][-1]),6.4)
        self.assertTrue(a['witness_contract'])
        self.assertFalse(b['witness_contract'])

    def test_exact_null_certificate_rate(self):
        # Fixed training-defined rank-4 directions and clean residual zero.
        # This acceptance event is exactly chi-square rank 4 > its 95% quantile.
        accepted=0
        base=np.zeros((8,3)); directions=self.d[:8]
        for _ in range(2000):
            y=self.rng.normal(size=(8,3))
            r=supported_subspace(base,directions,np.arange(8),y,1.)
            accepted+=r['certified']
        self.assertTrue(.035 < accepted/2000 < .065)


if __name__=='__main__':
    unittest.main()
