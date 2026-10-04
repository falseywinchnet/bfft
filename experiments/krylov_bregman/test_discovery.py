import unittest
import itertools
import numpy as np
from .core import Linearization, arnoldi
from .discovery import esp, rank_witness, arnoldi_stream, discover


class DiscoveryTests(unittest.TestCase):
    def test_esp_matches_explicit_squarefree_monomials(self):
        x=np.array([.2,.4,.7,1.1,1.4])
        for s in range(7):
            self.assertAlmostEqual(esp(x,s),sum(np.prod(t) for t in itertools.combinations(x,s)))

    def test_esp_tail_bound_retains_conditioning_factor(self):
        values=np.array([.6,.25,.1,.04,.01])
        for rank in [1,2,3,4]:
            self.assertLessEqual(values[rank:].sum(),esp(values,rank+1)/np.prod(values[:rank])+1e-15)

    def test_rank_witness_is_orthogonally_invariant(self):
        rng=np.random.default_rng(8)
        A=rng.normal(size=(9,2))@rng.normal(size=(2,6))
        Q=np.linalg.qr(rng.normal(size=(9,9)))[0]
        P=np.linalg.qr(rng.normal(size=(6,6)))[0]
        self.assertLess(rank_witness(A,2)['esp'],1e-28)
        self.assertGreater(rank_witness(A,1)['esp'],.001)
        self.assertAlmostEqual(rank_witness(A,1)['esp'],rank_witness(Q@A@P,1)['esp'])

    def test_incremental_arnoldi_equals_independent_prefixes(self):
        rng=np.random.default_rng(9); A=rng.normal(size=(20,20))*.15; r=rng.normal(size=20)
        count=[0]
        def action(v):count[0]+=1;return A@v
        for b,_ in arnoldi_stream(action,r,8):
            self.assertEqual(count[0],b.actions)
            independent=arnoldi(lambda v:A@v,r,b.actions)
            np.testing.assert_allclose(b.displacement(23),independent.displacement(23),atol=1e-12)

    def test_discovery_finds_finite_drift_without_stationary_inverse(self):
        class Drift:
            def __init__(self):self.calls=0;self.actions=0
            def step(self,z):self.calls+=1;return z+np.arange(1,len(z)+1)
            def linearize(self,z):
                def action(v):self.actions+=1;return v
                return Linearization(self.step(z),action)
        model=Drift();z=np.zeros(8); result=discover(model,z)
        self.assertTrue(result.accepted);self.assertEqual(result.depth,1)
        self.assertEqual(result.horizon,64)
        self.assertEqual(model.calls,result.map_calls);self.assertEqual(model.actions,result.tangent_actions)
        np.testing.assert_allclose(result.candidate,64*np.arange(1,9),atol=1e-11)

    def test_sparse_samples_are_not_a_path_certificate(self):
        # All queried points lie beyond the hidden event, yet the true orbit
        # reaches it. Even algebraic tangent closure + exact probes can fail.
        class Hidden:
            def step(self,z):
                bump=np.maximum(0.,1-4*(z-2)**2)**3
                return z+1+.5*bump
            def linearize(self,z):return Linearization(self.step(z),lambda v:v)
        model=Hidden();z=np.zeros(1);result=discover(model,z)
        self.assertTrue(result.accepted)
        actual=z.copy()
        for _ in range(result.horizon):actual=model.step(actual)
        self.assertGreater(np.linalg.norm(actual-result.candidate),.4)

    def test_polynomial_lift_closes_while_state_jacobian_does_not(self):
        a,b,c=.97,.92,.15
        T=np.array([[1,0,0,0],[0,a,0,0],[0,0,b,c],[0,0,0,a*a]])
        def lift(z):u,v=z;return np.array([1,u,v,u*u])
        def step(z):u,v=z;return np.array([a*u,b*v+c*u*u])
        z=np.array([.7,-.3]);x=z.copy()
        for _ in range(80):x=step(x)
        np.testing.assert_allclose(np.linalg.matrix_power(T,80)@lift(z),lift(x),atol=2e-15)
        J=np.array([[a,0],[2*c*z[0],b]])
        basis=arnoldi(lambda h:J@h,step(z)-z,2)
        self.assertLess(np.linalg.norm(basis.remainder),1e-14)
        self.assertGreater(np.linalg.norm(z+basis.displacement(80)-x),.01)

    def test_polynomial_lift_discovered_from_unisolvent_probes(self):
        # Coefficients are withheld; least-squares identification is a classic
        # finite-dictionary Koopman control, not a proposed new algorithm.
        rng=np.random.default_rng(10);states=rng.uniform(-1,1,(20,2))
        def phi(z):u,v=z;return np.array([1,u,v,u*u])
        def step(z):u,v=z;return np.array([.97*u,.92*v+.15*u*u])
        X=np.column_stack([phi(z) for z in states])
        Y=np.column_stack([phi(step(z)) for z in states])
        T=np.linalg.lstsq(X.T,Y.T,rcond=None)[0].T
        self.assertEqual(np.linalg.matrix_rank(X),4)
        for z in rng.uniform(-2,2,(10,2)):
            np.testing.assert_allclose(T@phi(z),phi(step(z)),atol=5e-15)
