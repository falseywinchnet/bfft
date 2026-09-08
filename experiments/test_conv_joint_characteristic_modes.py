import unittest
from unittest.mock import patch
import numpy as np
from experiments.conv_joint_characteristic_modes import compile_mode,compile_radial
from experiments.conv_joint_ridge_modes import evaluate,evaluate_jet,direction_arc


class CharacteristicModeTests(unittest.TestCase):
    def test_direction_intersection_is_finite_and_rejects_zero_only_cone(self):
        c,h=direction_arc([[1,0],[0,1]])
        self.assertAlmostEqual(c,np.pi/4);self.assertAlmostEqual(h,np.pi/4)
        self.assertIsNone(direction_arc([[1,0],[-1,0],[0,1],[0,-1]]))

    def test_affine_and_shared_scalar_c2_without_solves(self):
        y,x=np.mgrid[:9,:9];source=.37*x+np.sqrt(2)*y-2
        with patch('numpy.linalg.solve',side_effect=AssertionError('solve forbidden')), \
             patch('scipy.linalg.solve',side_effect=AssertionError('solve forbidden')), \
             patch('scipy.optimize.minimize',side_effect=AssertionError('optimizer forbidden')), \
             patch('experiments.conv_joint_potential.qp_admm',side_effect=AssertionError('QP forbidden')):
            m,d=compile_mode(source)
            q=np.linspace(0,8,31);yy,xx=np.meshgrid(q,q,indexing='ij')
            value,g,h=evaluate_jet(m,xx,yy)
        self.assertTrue(d['accepted']);self.assertTrue(d['affine_witness'])
        np.testing.assert_allclose(value,.37*xx+np.sqrt(2)*yy-2,atol=2e-12)
        np.testing.assert_allclose(g[...,0],.37,atol=2e-11)
        np.testing.assert_allclose(g[...,1],np.sqrt(2),atol=2e-11)
        p=m['controls'];width=np.diff(m['knots'])
        left=5*(p[:,1]-p[:,0])/width;right=5*(p[:,-1]-p[:,-2])/width
        np.testing.assert_allclose(left[1:],right[:-1],atol=2e-11)
        np.testing.assert_allclose(20*(p[:,2]-2*p[:,1]+p[:,0])/width**2,0,atol=2e-8)
        np.testing.assert_allclose(20*(p[:,-1]-2*p[:,-2]+p[:,-3])/width**2,0,atol=2e-8)

    def test_ridge_cardinality_symmetry_covariance_and_range(self):
        y,x=np.mgrid[:13,:13];source=.5*(1+np.tanh((x-.4*y-3.8)*1.5))
        m,d=compile_mode(source);mt,dt=compile_mode(source.T)
        mr,dr=compile_mode(source[:,::-1]);ma,da=compile_mode(3-2*source)
        self.assertTrue(all(q['accepted'] for q in (d,dt,dr,da)))
        np.testing.assert_allclose(evaluate(m,x,y),source,atol=2e-13)
        rng=np.random.default_rng(32);xq=rng.uniform(0,12,200);yq=rng.uniform(0,12,200)
        z=evaluate(m,xq,yq)
        np.testing.assert_allclose(z,evaluate(mt,yq,xq),atol=3e-12)
        np.testing.assert_allclose(z,evaluate(mr,12-xq,yq),atol=3e-12)
        np.testing.assert_allclose(3-2*z,evaluate(ma,xq,yq),atol=3e-12)
        for a,b,value in zip(xq,yq,z):
            cx=int(a);cy=int(b);support=source[max(0,cy-2):cy+4,max(0,cx-2):cx+4]
            self.assertGreaterEqual(value,support.min()-2e-12)
            self.assertLessEqual(value,support.max()+2e-12)

    def test_radial_curvature_companion_is_required(self):
        y,x=np.mgrid[:9,:9];source=(x-4.)**2+(y-4.)**2
        m,d=compile_radial(source);self.assertTrue(d['accepted'],d)
        value,gradient,hessian=evaluate_jet(m,5.5,5.2)
        self.assertAlmostEqual(value,1.5**2+1.2**2,places=11)
        np.testing.assert_allclose(gradient,[3,2.4],atol=2e-11)
        np.testing.assert_allclose(hessian,2*np.eye(2),atol=2e-10)

    def test_analytic_gradient_is_path_independent(self):
        y,x=np.mgrid[:9,:9];source=.5*(1+np.tanh((np.hypot(x-4,y-4)-2.3)*2))
        m,d=compile_radial(source);self.assertTrue(d['accepted'],d)
        gauss,weight=np.polynomial.legendre.leggauss(5)
        def integral(a,b):
            a=np.asarray(a);b=np.asarray(b);v=b-a;c=a-m['center'];s=m['polarity']
            A=s*(v@v);B=2*s*(c@v);C=s*(c@c)
            cuts=[0.,1.]
            for knot in m['knots']:
                discriminant=B*B-4*A*(C-knot)
                if discriminant>=0:
                    roots=[(-B-np.sqrt(discriminant))/(2*A),(-B+np.sqrt(discriminant))/(2*A)]
                    cuts.extend(r for r in roots if 0<r<1)
            cuts=np.unique(cuts);total=0.
            for lo,hi in zip(cuts[:-1],cuts[1:]):
                t=(lo+hi)/2+(hi-lo)*gauss/2;points=a+t[:,None]*v
                total+=(hi-lo)/2*np.dot(weight,evaluate(m,points[:,0],points[:,1],True)@v)
            return total
        a=[.3,.8];b=[7.6,.8];c=[7.6,7.1];d=[.3,7.1]
        self.assertAlmostEqual(integral(a,b)+integral(b,c)+integral(c,d)+integral(d,a),0,places=11)
        self.assertAlmostEqual(integral(a,b),float(evaluate(m,*b)-evaluate(m,*a)),places=11)

    def test_unsupported_geometry_is_rejected(self):
        y,x=np.mgrid[:9,:9]
        for source in (np.sin(x)+np.cos(1.3*y),np.hypot((x-4)/3,(y-4)/2)):
            m,d=compile_mode(source)
            self.assertIsNone(m);self.assertFalse(d['accepted'])

    def test_zero_amplitude_does_not_impose_an_unsupported_direction(self):
        y,x=np.mgrid[:13,:13]/12
        source=.5*(1+np.tanh(32*(x+2**-.5*(y-.5)-.48)))
        m,d=compile_mode(source)
        self.assertTrue(d['accepted'],d)
        self.assertFalse(d['direction_only_cone_compatible'])
        self.assertGreaterEqual(d['current_certificate_margin'],0.)
        self.assertLess(d['cardinality_error'],2e-13)


if __name__=='__main__':
    unittest.main()
