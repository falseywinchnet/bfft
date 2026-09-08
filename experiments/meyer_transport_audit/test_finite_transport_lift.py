import unittest
import numpy as np
from .model import ReducedMeyerMap,State,project_disk
from .finite_transport_lift import secant_matrix,apply_matrix,lifted_step,increment_transfer

class FiniteLiftTests(unittest.TestCase):
    def test_exact_finite_projection_and_contraction(self):
        rng=np.random.default_rng(201)
        for radius in [.1,1.,10.]:
            t=rng.normal(size=(2,1000))*radius;h=rng.normal(size=(2,1000))*radius*3
            # Include interior, exterior, antipodal, unchanged and boundary cases.
            t[:,:5]=np.array([[0,1,2,1,-2],[0,0,0,0,0]])*radius
            h[:,:5]=np.array([[0,1,-4,-1,5],[0,0,0,0,0]])*radius
            B=secant_matrix(*t,*h,radius);actual=apply_matrix(B,*h)
            p=project_disk(*t,radius);pn=project_disk(*(t+h),radius)
            np.testing.assert_allclose(actual,np.asarray(pn)-np.asarray(p),atol=3e-14,rtol=1e-10)
            xx,xy,yy=B;rad=np.sqrt((xx-yy)**2+4*xy**2)
            self.assertGreaterEqual(np.min((xx+yy-rad)/2),-1e-13)
            self.assertLessEqual(np.max((xx+yy+rad)/2),1+1e-13)

    def test_arbitrary_finite_complete_state_difference(self):
        rng=np.random.default_rng(202);m=ReducedMeyerMap(rng.uniform(0,255,(12,16)),.05,40)
        for _ in range(3):
            z=State(*(rng.normal(size=m.shape)*20 for _ in range(6)))
            h=State(*(rng.normal(size=m.shape)*20 for _ in range(6)))
            zn,hn=lifted_step(m,z,h)
            expected=m.subtract(m.step(zn),m.step(z))
            np.testing.assert_allclose(m.pack(hn),m.pack(expected),atol=1e-11,rtol=1e-9)

    def test_lift_reproduces_128_ordinary_steps(self):
        rng=np.random.default_rng(203);m=ReducedMeyerMap(rng.uniform(0,255,(16,16)),.05,40)
        z=m.initial();ordinary=z;h=m.subtract(m.step(z),z)
        for _ in range(128):
            z,h=lifted_step(m,z,h);ordinary=m.step(ordinary)
            np.testing.assert_allclose(m.pack(z),m.pack(ordinary),atol=1e-8,rtol=1e-8)

    def test_transport_change_includes_changing_transfer(self):
        rng=np.random.default_rng(204);m=ReducedMeyerMap(rng.uniform(0,255,(12,16)),.05,40)
        z=m.initial();h=m.subtract(m.step(z),z)
        for _ in range(4):
            zn,hn=lifted_step(m,z,h);znn,hnn=lifted_step(m,zn,hn)
            B0=(secant_matrix(z.tux,z.tuy,h.tux,h.tuy,m.ru),secant_matrix(z.twx,z.twy,h.twx,h.twy,m.rw))
            B1=(secant_matrix(zn.tux,zn.tuy,hn.tux,hn.tuy,m.ru),secant_matrix(zn.twx,zn.twy,hn.twx,hn.twy,m.rw))
            a=m.subtract(hn,h)
            carried=increment_transfer(m,a,*B1)
            change=m.subtract(increment_transfer(m,h,*B1),increment_transfer(m,h,*B0))
            predicted=State(*(x+y for x,y in zip(carried.fields(),change.fields())))
            np.testing.assert_allclose(m.pack(predicted),m.pack(m.subtract(hnn,hn)),atol=1e-11,rtol=1e-9)
            z,h=zn,hn

if __name__=='__main__':unittest.main()
