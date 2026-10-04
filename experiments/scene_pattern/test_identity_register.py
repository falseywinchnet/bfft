import unittest
import numpy as np
from scipy import ndimage
from .identity_register import signature,trace_peaks,ring_delays,verify,rerank
from .radial import unwind

class IdentityRegisterTests(unittest.TestCase):
    def test_magnitude_retrieval_retains_angular_phase_for_verification(self):
        rng=np.random.default_rng(22);a=rng.normal(size=(16,48,4));b=np.roll(a,9,axis=1)
        np.testing.assert_allclose(signature(a),signature(b),atol=1e-14)
        peaks=trace_peaks(a,b,.1);self.assertAlmostEqual(peaks[0]['angle'],9*2*np.pi/48,places=3)
        self.assertGreater(peaks[0]['correlation'],.99);self.assertEqual(len(peaks),3)
    def test_retrieval_is_serializable_and_does_not_need_old_tracks(self):
        import json
        from .identity_study import retrieve
        rng=np.random.default_rng(19);a=rng.normal(size=(16,48,4))
        traces=np.array([a,np.roll(a,7,axis=1),rng.normal(size=a.shape)])
        nodes=np.array([[0,0],[1,0],[2,0]]);rows=retrieve(traces,nodes,np.ones(3)*16)
        json.dumps(rows,allow_nan=False)
        first=min((r for r in rows if r['a']==[0,0]),key=lambda r:r['distance'])
        self.assertEqual(first['b'],[1,0]);self.assertLess(first['distance'],1e-12)
    def test_radial_shift_is_scale_without_radius_wrap(self):
        rng=np.random.default_rng(31);a=rng.normal(size=(16,48,4));b=np.zeros_like(a)
        b[2:]=np.roll(a[:-2],5,axis=1)
        peak=trace_peaks(a,b,.1)[0]
        self.assertAlmostEqual(peak['scale'],np.exp(.2),places=2)
        self.assertAlmostEqual(peak['angle'],5*2*np.pi/48,places=2)
    def test_finite_ringing_delay_has_correct_direction(self):
        rng=np.random.default_rng(21);a=ndimage.gaussian_filter(rng.normal(size=(81,81)),1.)
        b=ndimage.shift(a,(2.4,-3.2),order=3,mode='reflect')
        peaks=ring_delays(a,b);np.testing.assert_allclose(peaks[0]['shift'],[-3.2,2.4],atol=.2)
    def test_affine_identity_proposals_recover_translation_rotation_and_skew(self):
        rng=np.random.default_rng(93);image=ndimage.gaussian_filter(rng.normal(size=(150,170,3)),(1.1,1.1,0))*.2+[.5,.03,.02]
        a=np.array([[1.08,-.25],[.19,.93]]);p=np.array([80.,70.]);q=np.array([82.,72.]);yy,xx=np.indices(image.shape[:2]);coords=(np.stack([xx,yy],axis=-1)-q)@np.linalg.inv(a).T+p
        moving=np.stack([ndimage.map_coordinates(image[...,k],[coords[...,1],coords[...,0]],order=3,mode='reflect') for k in range(3)],axis=-1)
        angle=.2;initial=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        rows=verify(image,moving,image[...,0],moving[...,0],p,q+[3.,-2.],[initial])
        good=[r for r in rows if r['accepted']];self.assertTrue(good)
        best=max(good,key=lambda r:r['train']);np.testing.assert_allclose(best['q'],q,atol=.35);np.testing.assert_allclose(best['A'],a,atol=.06)
    def test_repeated_identity_stays_ambiguous_and_third_view_can_revise_leader(self):
        def edge(a,b,p,q,score=.96):return dict(a=a,b=b,p=p,site_q=q,q=q,A=np.eye(2).tolist(),validation=score,reverse_validation=score,accepted=True)
        # Two indistinguishable repeated regions in capture 1. A slight pairwise
        # preference initially selects site 1; independent image edges favor site 0.
        edges=[edge([0,0],[1,0],[0,0],[10,0]),edge([0,0],[1,1],[0,0],[30,0],.97)]
        initial=rerank(edges);self.assertLess(initial[0]['relative_weight'],initial[1]['relative_weight'])
        self.assertLess(max(r['relative_weight'] for r in initial),.6)
        edges += [edge([0,0],[2,0],[0,0],[20,0]),edge([2,0],[1,0],[20,0],[10,0])]
        revised=rerank(edges);self.assertGreater(revised[0]['relative_weight'],revised[1]['relative_weight']);self.assertEqual(revised[0]['third_capture_support'],[2])
        # Remove the evidence and the old leader returns: there is no frozen union.
        again=rerank(edges[:2]);self.assertEqual(initial,again)
    def test_distortion_start_multiplicity_does_not_inflate_identity_weight(self):
        def e(site):return dict(a=[0,0],b=[1,site],p=[0,0],site_q=[site*20,0],q=[site*20,0],A=np.eye(2).tolist(),validation=.96,reverse_validation=.96,accepted=True)
        base=rerank([e(0),e(1)]);repeated=rerank([e(0),e(0),e(0),e(1)])
        self.assertAlmostEqual(base[0]['identity_weight'],repeated[0]['identity_weight'])
        self.assertAlmostEqual(base[1]['relative_weight'],repeated[-1]['relative_weight'])
    def test_duplicate_third_paths_do_not_inflate_independent_support(self):
        def e(a,b,p,q):return dict(a=a,b=b,p=p,site_q=q,q=q,A=np.eye(2).tolist(),validation=.96,reverse_validation=.96,accepted=True)
        edges=[e([0,0],[1,0],[0,0],[10,0]),e([0,0],[2,0],[0,0],[20,0]),e([2,0],[1,0],[20,0],[10,0])]
        out=rerank(edges+edges[1:]);self.assertEqual(out[0]['third_capture_support'],[2])

if __name__=='__main__':unittest.main()
