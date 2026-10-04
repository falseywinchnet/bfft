import unittest
import numpy as np
from scipy import ndimage
from .distortion_basis import map_basis,geometry,fit_models
from .joint_distortion import supported_extension

class DistortionBasisTests(unittest.TestCase):
    def test_projective_matches_homogeneous_coordinates(self):
        p=np.array([3,-2,4,-3,2,1,8,-5.]);c=np.array([45,60.]);points=np.array([[20,30],[75,30],[30,85],[60,60.]])
        h=np.eye(3);h[:2,:2]+=p[2:6].reshape(2,2)/50;h[:2,2]=p[:2];h[2,:2]=p[6:]/2500
        q=np.c_[points-c,np.ones(len(points))]@h.T
        np.testing.assert_allclose(map_basis(points,p,c,'projective'),c+q[:,:2]/q[:,2:],atol=1e-12)
    def test_all_bases_contain_identity(self):
        q=np.array([[1,2],[20,30],[80,75.]])
        for model,n in [('affine',6),('projective',8),('quadratic',12)]:
            np.testing.assert_allclose(map_basis(q,np.zeros(n),[45,45],model),q)
            self.assertAlmostEqual(geometry(q,np.zeros(n),[45,45],model)['min_jacobian'],1.)
    def test_extension_never_evaluates_unsupported_polynomial(self):
        raw=np.full((100,100,2),1e9);support=np.zeros((100,100),bool);support[40:60,40:60]=True;raw[support]=[3,-2]
        result=supported_extension(raw,support,12)
        np.testing.assert_array_equal(result[support],raw[support])
        self.assertLessEqual(np.max(np.linalg.norm(result,axis=-1)),np.sqrt(13)+1e-12)
        self.assertLess(np.linalg.norm(result[0,0]),1e-8)
    def test_known_projective_distortion_is_recovered(self):
        shape=(80,90);rng=np.random.default_rng(71);a=ndimage.gaussian_filter(rng.random((*shape,3)),(1.3,1.3,0))
        y,x=np.indices(shape);center=np.array([45,40.]);points=np.stack((x,y),-1);p=np.array([3,-2,2,-1,1,2,9,-7.])
        h=np.eye(3);h[:2,:2]+=p[2:6].reshape(2,2)/50;h[:2,2]=p[:2];h[2,:2]=p[6:]/2500
        q=np.concatenate((points-center,np.ones((*shape,1))),-1)@np.linalg.inv(h).T;q=center+q[...,:2]/q[...,2:]
        b=np.stack([ndimage.map_coordinates(a[...,k],[q[...,1],q[...,0]],order=3,mode='reflect') for k in range(3)],-1)
        regions=[(x>15)&(x<75)&(y>15)&(y<39),(x>15)&(x<75)&(y>=39)&(y<65)]
        r=fit_models(a,b,np.ones(shape),np.ones(shape),regions)
        self.assertEqual(r['status'],'accepted');t=r['selected']
        probes=np.array([[25,25],[65,25],[25,55],[65,55.]])
        error=np.linalg.norm(map_basis(probes,np.asarray(t['parameters']),r['center'],t['model'])-map_basis(probes,p,center,'projective'),axis=-1)
        self.assertLess(float(error.max()),.8)
        self.assertLess(np.mean(t['reporting']),np.mean(r['before_reporting'])*.3)
    def test_harmonic_continuation_preserves_evidence_and_converges(self):
        from .joint_distortion import harmonic_extension
        support=np.zeros((50,60),bool);support[18:32,22:38]=True
        raw=np.zeros((50,60,2));raw[support]=[3,-2]
        extension,residuals=harmonic_extension(raw,support,12)
        np.testing.assert_array_equal(extension[support],raw[support])
        self.assertLess(max(residuals),1.1e-6)
        self.assertLessEqual(extension[...,0].max(),3+1e-6)
        self.assertGreaterEqual(extension[...,0].min(),-1e-6)
        np.testing.assert_array_equal(extension[[0,-1]],0)
    def test_bounded_continuation_leaves_other_regions_exactly_fixed(self):
        from .joint_distortion import harmonic_extension
        support=np.zeros((60,70),bool);support[25:35,30:40]=True
        allowed=np.zeros_like(support);allowed[15:45,20:50]=True
        raw=np.zeros((60,70,2));raw[support]=[2,-1]
        extension,residuals=harmonic_extension(raw,support,16,allowed=allowed)
        np.testing.assert_array_equal(extension[~allowed],0)
        np.testing.assert_array_equal(extension[support],raw[support])
        self.assertLess(max(residuals),1.1e-6)
    def test_partial_capture_preserves_known_shift_and_minimum_overlap(self):
        shape=(75,100);rng=np.random.default_rng(23);a=ndimage.gaussian_filter(rng.random((*shape,3)),(1.3,1.3,0))
        y,x=np.indices(shape);b=np.stack([ndimage.map_coordinates(a[...,k],[y,x+10],order=1,mode='reflect') for k in range(3)],-1)
        weight=(x>=25).astype(float);b*=weight[...,None]
        regions=[(x>25)&(x<90)&(y>12)&(y<35),(x>25)&(x<90)&(y>=35)&(y<64)]
        r=fit_models(a,b,np.ones(shape),weight,regions,allow_visibility_change=True)
        self.assertEqual(r['status'],'accepted');t=r['selected'];q=np.array([[45,25],[75,25],[45,50],[75,50.]])
        error=np.linalg.norm(map_basis(q,np.array(t['parameters']),r['center'],t['model'])-(q+[-10,0]),axis=1)
        self.assertLess(error.max(),1.)
        self.assertGreaterEqual(min(t['coverage']),.75)
