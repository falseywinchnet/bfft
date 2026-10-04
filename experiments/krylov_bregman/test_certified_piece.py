import unittest
import numpy as np
from .certified_piece import HuberMirror,discover_piece,reduced_path


class CertifiedPieceTests(unittest.TestCase):
    def test_block_doubling_matches_recurrence_including_jordan_drift(self):
        for H in [np.array([[1.,.1],[0.,1.]]),np.array([[.9,.1],[.1,.7]])]:
            source=np.array([.3,.7]);C=reduced_path(H,source,130)
            c=np.zeros(2)
            for j in range(130):
                np.testing.assert_allclose(C[:,j],c,atol=1e-10)
                c=source+H@c

    def test_analytic_action_and_global_nonexpansivity(self):
        rng=np.random.default_rng(81);model=HuberMirror(rng.normal(size=(25,8)),rng.normal(size=25))
        for _ in range(20):
            x,y=rng.normal(size=(2,8));h=rng.normal(size=8)
            self.assertLessEqual(np.linalg.norm(model.step(x)-model.step(y)),np.linalg.norm(x-y)+1e-12)
            eps=1e-6;fd=(model.step(x+eps*h)-model.step(x-eps*h))/(2*eps)
            np.testing.assert_allclose(fd,model.linearize(x).action(h),atol=1e-9)

    def test_drift_stops_at_first_event_and_matches_all_steps(self):
        B=np.array([[1.,.2],[.3,1.],[.6,.5]])
        model=HuberMirror(B,B@np.array([50.,80.]))
        z=model.initial;result=discover_piece(model,z,maximum_horizon=4096)
        self.assertTrue(result.accepted);self.assertEqual(result.depth,1)
        self.assertGreater(result.horizon,50)
        actual=z.copy()
        for _ in range(result.horizon):actual=model.step(actual)
        np.testing.assert_allclose(result.state,actual,atol=2e-11)
        self.assertTrue(np.any(np.abs(B@result.state-model.b)<1.))

    def test_affine_piece_certificate_bounds_actual_nonlinear_trajectory(self):
        rng=np.random.default_rng(82)
        accepted=0
        for _ in range(15):
            B=rng.normal(size=(20,6))*.02
            model=HuberMirror(B,rng.normal(size=20)*.1)
            # alpha well inside the stability limit gives long valid pieces.
            model.alpha*=.03
            z=rng.normal(size=6)*.1
            d=discover_piece(model,z,maximum_horizon=64,tolerance=.02)
            if d.accepted:
                accepted+=1
                actual=z.copy()
                for _ in range(d.horizon):actual=model.step(actual)
                self.assertLessEqual(np.linalg.norm(actual-d.state),d.error_bound+3e-13)
        self.assertGreater(accepted,0)

    def test_anderson_cannot_accelerate_exact_constant_residual_piece(self):
        from .core import Anderson
        model=HuberMirror(np.ones((1,1)),np.array([1000.]),alpha=1.)
        aa=Anderson(4);z=np.zeros(1)
        for _ in range(400):z=aa.advance(z,model.step(z))
        np.testing.assert_array_equal(z,[400.])
        d=discover_piece(model,np.zeros(1),maximum_horizon=4096)
        self.assertEqual(d.horizon,1000)
        self.assertEqual(d.calls+d.actions,2)
        np.testing.assert_array_equal(d.state,[1000.])

    def test_boundary_input_output_index(self):
        model=HuberMirror(np.ones((1,1)),np.array([10.]),alpha=1.)
        d=discover_piece(model,np.zeros(1),maximum_horizon=100)
        self.assertEqual(d.horizon,10) # inputs 0,...,9; at input9 residual=-1.
        np.testing.assert_allclose(d.state,[10.])

    def test_quadratic_mirror_coordinate_equivalence(self):
        rng=np.random.default_rng(83);A=rng.normal(size=(20,8));g=np.geomspace(1,50,8)
        b=rng.normal(size=20);model=HuberMirror(A/np.sqrt(g),b,mirror_diagonal=g)
        x=rng.normal(size=8);t=np.sqrt(g)*x
        xp=x-model.alpha*(A.T@np.clip(A@x-b,-1,1))/g
        np.testing.assert_allclose(model.primal(model.step(t)),xp,atol=1e-14)
