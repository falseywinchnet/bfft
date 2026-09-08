import unittest
from unittest.mock import patch
import numpy as np
from scipy.ndimage import minimum_filter,maximum_filter
from experiments.conv_budget_modes import cone_bank,compile_mode,resize,profile,sample_profile
from experiments.conv_joint_ridge_modes import support_normals,compile_profile,evaluate_profile
from experiments.probe_conv_compatible_2d import resize as original


class BudgetModesTests(unittest.TestCase):
    def test_batched_cones_against_original_membership(self):
        rng=np.random.default_rng(491);y,x=np.mgrid[:9,:11]
        cases=[rng.normal(size=x.shape),.3*x+.7*y,np.hypot(x-5,y-4),
               (x+.7*y>7).astype(float),np.maximum(x-4,0),np.zeros(x.shape),
               (x-5.)**2+1e-13*(x-5)*y,(x-5.)**2-1e-13*(x-5)*y]
        angles=np.linspace(0,2*np.pi,91);vectors=np.stack((np.cos(angles),np.sin(angles)),axis=1)
        for z in cases:
            z=(z-z.min())/max(1.,np.ptp(z));bank=cone_bank(z)
            for j in range(8):
                for i in range(10):
                    ns=support_normals(z,j,i)
                    expected=np.ones(len(vectors),bool) if not ns else np.min(np.array(ns)@vectors.T,axis=0)>=-1e-11
                    actual=np.min(bank[j,i]@vectors.T,axis=0)>=-1e-11
                    self.assertTrue(np.all(~actual|expected),(j,i,z,bank[j,i],ns))

    def test_profile_polynomial_reproduction_and_shared_c2(self):
        rng=np.random.default_rng(149);t=np.sort(rng.uniform(0,5,80))
        f=.1*t+.02*t*t+.001*t**3;model=profile(t,f)
        q=np.linspace(t[0],t[-1],201)
        np.testing.assert_allclose(sample_profile(model,q),.1*q+.02*q*q+.001*q**3,atol=2e-12)
        p=model['power'];h=np.diff(model['knots'])
        right=(p[:,1]+2*p[:,2]+3*p[:,3]+4*p[:,4]+5*p[:,5])/h
        right2=(2*p[:,2]+6*p[:,3]+12*p[:,4]+20*p[:,5])/h**2
        np.testing.assert_allclose(right[:-1],p[1:,1]/h[1:],atol=2e-10)
        np.testing.assert_allclose(right2[:-1],2*p[1:,2]/h[1:]**2,atol=2e-8)
        self.assertGreaterEqual(model['minimum_derivative'],-1e-11)
        for _ in range(20):
            f=np.sort(rng.uniform(0,1,80));model=profile(t,f)
            self.assertGreaterEqual(model['minimum_derivative'],-1e-9)
            value=sample_profile(model,np.linspace(t[0],t[-1],3000))
            self.assertGreaterEqual(np.diff(value).min(),-1e-12)

    def test_binary_compression_preserves_original_complete_profile(self):
        t=np.linspace(0,1,1001);f=(t>.431).astype(float);model=profile(t,f)
        ref,_=compile_profile(t,f,128*np.finfo(float).eps);ref.update(offset=0.,scale=1.)
        self.assertLessEqual(len(model['knots']),4)
        q=np.linspace(0,1,10003)
        np.testing.assert_allclose(sample_profile(model,q),evaluate_profile(ref,q),atol=3e-15)

    def test_full_support_filter_indices(self):
        z=np.random.default_rng(71).normal(size=(9,11))
        low=minimum_filter(z,size=6,origin=-1,mode='nearest');high=maximum_filter(z,size=6,origin=-1,mode='nearest')
        for j in range(8):
            for i in range(10):
                support=z[max(0,j-2):j+4,max(0,i-2):i+4]
                self.assertEqual(low[j,i],support.min());self.assertEqual(high[j,i],support.max())

    def test_complete_operator_no_solves_cardinality_and_fallback_identity(self):
        y,x=np.mgrid[:17,:17]/16
        fields=[.5*(1+np.tanh(20*(x+.7*y-.86))),.5*(1+np.tanh(35*(np.hypot(x-.5,y-.5)-.27))),
                (x+.7*y>.86).astype(float),(np.hypot(x-.5,y-.5)<.27).astype(float),
                np.sin(x*25)+np.cos(y*19),np.ones(x.shape)]
        with patch('numpy.linalg.solve',side_effect=AssertionError('forbidden')),patch('scipy.optimize.minimize',side_effect=AssertionError('forbidden')):
            for source in fields:
                result,diag=resize(source,3,True)
                np.testing.assert_allclose(result[::3,::3],source,atol=2e-12)
                if not diag['accepted']:np.testing.assert_array_equal(result,original(source,3,'CONV'))
                else:
                    self.assertGreaterEqual(result.min(),source.min()-2e-12)
                    self.assertLessEqual(result.max(),source.max()+2e-12)

    def test_accepted_modes_obey_original_dense_current_certificate(self):
        y,x=np.mgrid[:13,:13];xn=x/12;yn=y/12;accepted=0
        for source in (.5*(1+np.tanh(32*(xn+2**-.5*(yn-.5)-.48))),
                       .5*(1+np.tanh(25*(np.hypot(xn-.5,yn-.5)-.23)))):
            fast,d=compile_mode(source)
            if fast is None:continue
            accepted+=1;z=(source-source.min())/np.ptp(source)
            for j in range(12):
                for i in range(12):
                    yy,xx=np.meshgrid(j+np.linspace(0,1,5),i+np.linspace(0,1,5),indexing='ij')
                    if fast['chart']=='ridge':
                        n=fast['direction'];phase=n[0]*xx+n[1]*yy;gphase=n
                    else:
                        s=fast['polarity'];cx,cy=fast['center'];phase=s*((xx-cx)**2+(yy-cy)**2)
                        gphase=2*s*np.stack((xx-cx,yy-cy),axis=-1)
                    knots=fast['knots'];idx=np.clip(np.searchsorted(knots,phase,side='right')-1,0,len(knots)-2)
                    h=knots[idx+1]-knots[idx];u=np.clip((phase-knots[idx])/h,0,1);p=fast['power'][idx]
                    first=(p[...,1]+u*(2*p[...,2]+u*(3*p[...,3]+u*(4*p[...,4]+u*5*p[...,5]))))/h
                    first=np.where((phase<knots[0])|(phase>knots[-1]),0,first)
                    g=(first[...,None]*gphase).reshape(-1,2);normals=support_normals(z,j,i)
                    if normals:self.assertGreaterEqual(float(np.min(g@np.array(normals).T)),-2e-11)
        self.assertEqual(accepted,2)

    def test_exact_axial_and_biquadratic_paths(self):
        y,x=np.mgrid[:9,:11]
        for source in (np.sin(x),np.cos(y),.1*x*x+.2*x*y+.3*y*y):
            for scale in (2,3,6):
                z,d=resize(source,scale,True)
                np.testing.assert_array_equal(z,original(source,scale,'CONV'))
                self.assertFalse(d['accepted'])

    def test_admitted_ridge_and_radial_covariance(self):
        y,x=np.mgrid[:17,:17]/16
        for source in (.5*(1+np.tanh(20*(x+.7*y-.86))),.5*(1+np.tanh(35*(np.hypot(x-.5,y-.5)-.27)))):
            base,d=resize(source,3,True)
            self.assertTrue(d['accepted'])
            trans,dt=resize(source.T,3,True);reflected,dr=resize(source[:,::-1],3,True)
            affine,da=resize(3-2*source,3,True)
            self.assertTrue(dt['accepted'] and dr['accepted'] and da['accepted'])
            np.testing.assert_allclose(base,trans.T,atol=2e-11)
            np.testing.assert_allclose(base,reflected[:,::-1],atol=2e-11)
            np.testing.assert_allclose(3-2*base,affine,atol=2e-11)

    def test_color_path_is_exact_existing_conv(self):
        source=np.random.default_rng(88).uniform(size=(9,11,3))
        from experiments.convstar import convstar_resize
        result,d=resize(source,3,True)
        np.testing.assert_array_equal(result,convstar_resize(source,3))
        self.assertFalse(d['accepted'])


if __name__=='__main__':unittest.main()
