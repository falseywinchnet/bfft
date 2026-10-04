import unittest
import numpy as np
from .fusion import anchor_matrices,atlas_grid,fuse,linearize,encode
from .core import map_points

class FusionTests(unittest.TestCase):
    def test_anchor_direction(self):
        hs=[np.eye(3),np.array([[1.2,.1,.2],[0,.9,-.1],[.01,.03,1.]])]
        anchored=anchor_matrices(hs,1)
        p=np.array([[.2,.3],[.7,.8]])
        np.testing.assert_allclose(map_points(anchored[1],p),p,atol=1e-12)
        np.testing.assert_allclose(map_points(hs[1],map_points(anchored[0],p)),p,atol=1e-12)

    def test_constant_and_missing_support(self):
        stack=np.full((3,8,9,3),.4,dtype=np.float32)
        weights=np.ones((3,8,9));weights[:,0]=0
        stack[2]=.99;weights[2]=0
        fused,mean,fractions,raw,_=fuse(stack,weights)
        np.testing.assert_allclose(fused[1:],.4,atol=1e-6)
        np.testing.assert_allclose(fractions[:,1:].sum(0),1,atol=1e-12)
        self.assertTrue(np.all(fractions[:,0]==0))
        self.assertTrue(np.all(np.isfinite(fused)))
        np.testing.assert_allclose(raw,0,atol=1e-6)

    def test_outlier_does_not_hide_disagreement(self):
        stack=np.full((5,8,9,3),.4,dtype=np.float32);stack[4]=.9
        fused,mean,fractions,raw,robust=fuse(stack,np.ones((5,8,9)))
        self.assertLess(float(np.abs(fused-.4).max()),.001)
        self.assertGreater(float(np.mean(mean-fused)),.1)
        self.assertTrue(np.all(raw>robust*5))

    def test_atlas_aspect_and_horizon(self):
        q,lo,hi=atlas_grid([np.eye(3)],100)
        self.assertEqual(q.shape,(80,100,2))
        np.testing.assert_allclose(q[-1,-1],[1,1])
        h=np.array([[1,0,0],[0,1,0],[2,0,-1.]])
        with self.assertRaises(ValueError):atlas_grid([h],100)

    def test_linear_light_roundtrip(self):
        x=np.linspace(0,1,100)
        np.testing.assert_allclose(encode(linearize(x)),x,atol=1e-14)

if __name__=='__main__':unittest.main()
