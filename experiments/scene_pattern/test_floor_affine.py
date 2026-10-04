import unittest
import numpy as np
from scipy.ndimage import gaussian_filter,map_coordinates
from .floor_affine import affine_points,fit_floor,floor_fields

class FloorAffineTests(unittest.TestCase):
    def test_symmetric_split_is_exact_inside_plane(self):
        p=np.array([-6,2,2,-2,1.5,2.]);center=np.array([64,64.]);polygon=[(20,20),(108,20),(108,108),(20,108)]
        fields,mask,h=floor_fields((128,128),p,center,polygon)
        y,x=np.indices(mask.shape);q=np.stack((x,y),-1)
        a=q+fields[0];b=q+fields[1]
        np.testing.assert_allclose((a@h[:2,:2].T+h[:2,2])[mask],b[mask],atol=1e-10)
    def test_synthetic_affine_recovery(self):
        shape=(128,128);rng=np.random.default_rng(123)
        a=gaussian_filter(rng.random(shape),1)*.8+.1
        yy,xx=np.indices(shape);q=np.stack((xx,yy),-1);center=np.array([65,65.]);p=np.array([-5,2,2,-2,1.5,1.5])
        A=np.eye(2)+p[2:].reshape(2,2)/50;off=p[:2]-(A-np.eye(2))@center
        inverse=(q-off)@np.linalg.inv(A).T
        b=map_coordinates(a,[inverse[...,1],inverse[...,0]],order=3,mode='reflect')
        aa=np.repeat(a[...,None],3,-1).astype(np.float32);bb=np.repeat(b[...,None],3,-1).astype(np.float32)
        polygon=[(25,25),(105,25),(105,105),(25,105)]
        estimated,c,r=fit_floor(aa,bb,np.ones(shape),np.ones(shape),polygon)
        points=np.array([[40,40],[90,40],[90,90],[40,90],[65,65.]])
        difference=affine_points(points,estimated,c)-affine_points(points,p,center)
        self.assertLess(float(np.max(np.linalg.norm(difference,axis=1))),.7)
        self.assertLess(r['after_test'],r['before_test']*.5)
    def test_identity_does_not_move_floor(self):
        f,mask,h=floor_fields((40,40),np.zeros(6),np.array([20,20]),[(10,10),(30,10),(30,30),(10,30)])
        np.testing.assert_allclose(f,0,atol=1e-12)
