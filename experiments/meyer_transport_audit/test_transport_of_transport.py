import unittest
import numpy as np
from unittest.mock import patch
from .model import ReducedMeyerMap,State
from .transport_of_transport import CarriedTransport,FullMemoryTransport,pullback

class CarriedTests(unittest.TestCase):
    def test_pullback_exact_on_arbitrary_states(self):
        rng=np.random.default_rng(84);m=ReducedMeyerMap(rng.uniform(0,255,(12,16)),.05,40)
        for _ in range(5):
            z=State(*(rng.normal(size=m.shape)*30 for _ in range(6)))
            q=State(*(rng.normal(size=m.shape) for _ in range(6)))
            linear,V,c=pullback(m,q)
            t=np.stack([z.tux,z.tuy,z.twx,z.twy])
            from .model import project_disk
            p=np.stack([*project_disk(t[0],t[1],m.ru),*project_disk(t[2],t[3],m.rw)])
            actual=float(m.pack(q)@m.pack(m.step(z)))
            predicted=float(linear@m.pack(z)+np.sum(V*p)+c)
            self.assertAlmostEqual(actual,predicted,places=9)

    def test_reduced_map_is_exact_orthogonal_projection(self):
        rng=np.random.default_rng(85);m=ReducedMeyerMap(rng.uniform(0,255,(12,16)),.05,40)
        z=m.initial();r=CarriedTransport(m,z,4)
        for scale in [0,1,10,1000]:
            y=rng.normal(size=r.rank)*scale
            expected=r.Q.T@(m.pack(m.step(r.reconstruct(y)))-m.pack(z))
            np.testing.assert_allclose(r.reduced_map(y),expected,atol=3e-11,rtol=1e-10)

    def test_carries_observed_transport_exactly_and_inner_has_no_fft(self):
        rng=np.random.default_rng(86);m=ReducedMeyerMap(rng.uniform(0,255,(12,16)),.05,40)
        z=m.initial();r=CarriedTransport(m,z,4);y=np.zeros(r.rank);expected=z
        for _ in range(4):
            y=r.reduced_map(y);expected=m.step(expected)
            np.testing.assert_allclose(m.pack(r.reconstruct(y)),m.pack(expected),atol=2e-11,rtol=1e-10)
        with patch.object(np.fft,'fft2',side_effect=AssertionError('unexpected FFT')):
            with patch.object(np.fft,'ifft2',side_effect=AssertionError('unexpected FFT')):
                r.rollout(16,True)

    def test_constant_state_rank_zero(self):
        m=ReducedMeyerMap(np.full((12,16),90.),.05,40);z=m.initial();r=CarriedTransport(m,z,2)
        np.testing.assert_allclose(m.pack(r.rollout(16,True)),m.pack(z),atol=1e-12)

    def test_full_memory_primal_projection_and_capacity(self):
        rng=np.random.default_rng(87);m=ReducedMeyerMap(rng.uniform(0,255,(12,16)),.05,40)
        r=FullMemoryTransport(m,m.initial(),4)
        y=r.ytrain.copy();b=r.btrain.copy()
        for _ in range(12):
            actual=m.step(r.reconstruct(y,b));yn,bn=r.advance(y,b)
            x=np.concatenate([actual.u.ravel(),actual.w.ravel()])
            np.testing.assert_allclose(yn,r.Q.T@(x-r.x0),atol=2e-11,rtol=1e-10)
            self.assertLessEqual(np.max(np.hypot(bn[0],bn[1])),m.ru+1e-12)
            self.assertLessEqual(np.max(np.hypot(bn[2],bn[3])),m.rw+1e-12)
            y,b=yn,bn

if __name__=='__main__':unittest.main()
