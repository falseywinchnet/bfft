import unittest
import numpy as np
from .radial import unwind,trace_similarity


def field(size,scale=1.,angle=0.):
    y,x=np.indices((size,size));x=x-(size-1)/2;y=y-(size-1)/2
    r=np.maximum(np.hypot(x,y)/scale,.01);theta=np.arctan2(y,x)-np.deg2rad(angle)
    result=np.zeros_like(r)
    for radius,t,width in [(11,.2,.25),(19,-1.1,.35),(31,2.3,.25),(47,-2.6,.3)]:
        delta=np.arctan2(np.sin(theta-t),np.cos(theta-t))
        result+=np.exp(-.5*(np.log(r/radius)/.18)**2-.5*(delta/width)**2)
    return result

class RadialTests(unittest.TestCase):
    def test_centered_scale_rotation(self):
        a,valid,meta=unwind(field(161),(80,80),r_min=7,r_max=65)
        for scale,angle in [(1.,0.),(1.16,23.),(.88,-31.)]:
            b,_,_=unwind(field(161,scale,angle),(80,80),r_min=7,r_max=65)
            estimate=trace_similarity(a,b,meta['log_radius_step'])
            self.assertAlmostEqual(estimate['scale'],scale,delta=.018)
            self.assertAlmostEqual(estimate['angle_degrees'],angle,delta=1.)
        self.assertTrue(valid.all())

    def test_center_error_is_visible(self):
        a,_,meta=unwind(field(161),(80,80),r_min=7,r_max=65)
        b,_,_=unwind(field(161,1.16,23),(80,80),r_min=7,r_max=65)
        off,_,_=unwind(field(161,1.16,23),(88,75),r_min=7,r_max=65)
        correct=trace_similarity(a,b,meta['log_radius_step'])
        wrong=trace_similarity(a,off,meta['log_radius_step'])
        self.assertLess(wrong['correlation'],correct['correlation']-.1)

    def test_outside_image_is_explicit(self):
        _,valid,_=unwind(np.ones((50,50)),(2,2),r_max=30)
        self.assertFalse(valid.all())

if __name__=='__main__':unittest.main()
