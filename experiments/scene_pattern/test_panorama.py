import unittest
import numpy as np
from .panorama import camera_chain,world_rays

class PanoramaTests(unittest.TestCase):
    def test_anchor_and_roundtrip(self):
        h=np.array([[1,.02,-.3],[.03,1,.1],[.1,.02,1.]])
        pairs=[{'matrices':{'final':h.tolist()}}]
        k,chain=camera_chain(pairs,1)
        np.testing.assert_allclose(chain[1],np.eye(3),atol=1e-12)
        pixels=np.array([[.1,.2],[.8,.7],[.5,.5]])
        rays=world_rays(chain[0],pixels,k)
        projected=rays@(k@chain[0]).T
        np.testing.assert_allclose(projected[:,:2]/projected[:,2:],pixels,atol=1e-12)
    def test_rotation_projection_is_proper(self):
        h=np.array([[.8,.1,-.2],[0,1.1,.1],[.1,0,1.]])
        _,chain=camera_chain([{'matrices':{'final':h.tolist()}}],0,True)
        np.testing.assert_allclose(chain[1].T@chain[1],np.eye(3),atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(chain[1]),1)

class SeamTests(unittest.TestCase):
    def test_disjoint_sources_and_constant_overlap(self):
        from .seams import seam_mosaic
        stack=np.full((2,24,40,3),.4,np.float32)
        weights=np.zeros((2,24,40));weights[0,:,:28]=1;weights[1,:,12:]=1
        result,labels=seam_mosaic(stack,weights,step=2)
        np.testing.assert_allclose(result,.4,atol=1e-6)
        self.assertTrue(np.all(labels[:,:12]==0));self.assertTrue(np.all(labels[:,28:]==1))
    def test_unsupported_pixels_remain_empty(self):
        from .seams import seam_mosaic
        stack=np.ones((2,20,20,3),np.float32);weights=np.zeros((2,20,20));weights[0,3:17,3:17]=1
        result,_=seam_mosaic(stack,weights,step=2)
        self.assertTrue(np.all(result[weights.sum(0)==0]==0))
        self.assertTrue(np.all(np.isfinite(result)))

class RegionalRayTests(unittest.TestCase):
    def test_known_rotation_from_region_correspondences(self):
        k,_=camera_chain([],0)
        t=.23;r=np.array([[np.cos(t),0,np.sin(t)],[0,1,0],[-np.sin(t),0,np.cos(t)]])
        x,y=np.meshgrid(np.linspace(.15,.75,5),np.linspace(.2,.8,4));a=np.c_[x.ravel(),y.ravel()]
        h=k@r@np.linalg.inv(k);mapped=np.c_[a,np.ones(len(a))]@h.T;b=mapped[:,:2]/mapped[:,2:]
        pair={'matrices':{'final':np.eye(3).tolist()},'seed':{'reference_points':a.tolist(),'moving_points':b.tolist()}}
        _,chain=camera_chain([pair],0,True)
        np.testing.assert_allclose(chain[1],r,atol=1e-12)
