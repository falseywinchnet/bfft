import unittest
import numpy as np
from scipy import ndimage
from .core import texture_coordinates,map_points,warp,fractional_delay,ringing

class GeometryTests(unittest.TestCase):
    def test_matrix_direction(self):
        h=np.array([[1.2,.1,.04],[-.07,.9,.03],[.13,-.08,1.]])
        p=np.random.default_rng(7).uniform(.1,.9,(60,2))
        np.testing.assert_allclose(map_points(np.linalg.inv(h),map_points(h,p)),p,atol=1e-14)

    def test_conv_cardinality_and_affine_plane(self):
        y,x=np.indices((19,23));field=.2+.009*x+.006*y
        identity,mask=warp(field,np.eye(3),method='conv')
        np.testing.assert_allclose(identity,field,atol=2e-6)
        h=np.array([[.8,0,.08],[0,.85,.06],[0,0,1.]])
        result,mask=warp(field,h,method='conv')
        expected=.2+.009*(.8*x+.08*22)+.006*(.85*y+.06*18)
        np.testing.assert_allclose(result,expected,atol=2e-6)

    def test_hue_wrap_and_achromatic_limit(self):
        angles=np.array([np.pi-1e-5,-np.pi+1e-5])
        lab=np.zeros((1,2,3));lab[...,0]=.5
        lab[0,:,1]=.1*np.cos(angles);lab[0,:,2]=.1*np.sin(angles)
        cartoon=lab[:,::-1].copy()
        coords=texture_coordinates(lab,cartoon,lab-cartoon)
        self.assertLess(np.max(abs(coords[...,2:])),3e-6)
        lab[...,1:]=0
        self.assertEqual(float(np.max(abs(texture_coordinates(lab,cartoon,lab-cartoon)[...,2:]))),0.)

    def test_fractional_delay_sign_and_accuracy(self):
        rng=np.random.default_rng(8)
        ref=ndimage.gaussian_filter(rng.normal(size=(128,128)),1.2)
        moved=np.fft.ifftn(ndimage.fourier_shift(np.fft.fftn(ref),[.7,-1.35])).real
        for mode in ('phase','ringing'):
            shift,_=fractional_delay(ref,moved,mode)
            np.testing.assert_allclose(shift,[1.35,-.7],atol=.08)
        self.assertGreater(np.max(abs(ringing(ref))),0)

if __name__=='__main__':unittest.main()
