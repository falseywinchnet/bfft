import unittest
import numpy as np
from .core import arnoldi, finite_flow, shadow_diagnostic, Linearization, metric_norm, Anderson
from .problems import (QuadraticMirror, EntropicSeparable, EntropicSimplex,
                       ADMMLasso, EntropicPrimal, EntropicBoundary, ADMMShadow)


class Contracts(unittest.TestCase):
    def test_analytic_actions_against_independent_differences(self):
        rng = np.random.default_rng(117)
        for cls in [QuadraticMirror, EntropicSeparable, EntropicSimplex, ADMMLasso, EntropicPrimal, ADMMShadow]:
            model = cls(n=18, seed=17)
            state = model.initial.copy()
            for _ in range(7):
                state = model.step(state)
            h = rng.normal(size=len(state)); h /= np.linalg.norm(h)
            lin = model.linearize(state)
            np.testing.assert_allclose(lin.next_state, model.step(state), atol=1e-13)
            eps = 1e-6
            fd = (model.step(state + eps*h) - model.step(state - eps*h))/(2*eps)
            np.testing.assert_allclose(lin.action(h), fd, atol=3e-8, rtol=3e-7,
                                       err_msg=cls.name)

    def test_mirror_jacobian_self_adjoint_at_nonstationary_points(self):
        rng = np.random.default_rng(118)
        for cls in [QuadraticMirror, EntropicSeparable, EntropicSimplex]:
            model = cls(n=20, seed=18)
            z = rng.normal(size=model.dimension)*.3
            lin = model.linearize(z)
            h, v = rng.normal(size=(2, len(z)))
            self.assertGreater(metric_norm(h, lin.metric), 0)
            self.assertAlmostEqual(float(h @ lin.metric(lin.action(v))),
                                   float(lin.action(h) @ lin.metric(v)), places=11)

    def test_natural_arnoldi_has_symmetric_projection(self):
        for cls in [QuadraticMirror, EntropicSimplex]:
            model = cls(n=24, seed=19)
            _, basis, _ = finite_flow(model, model.initial, depth=7, horizon=20)
            np.testing.assert_allclose(basis.H, basis.H.T, atol=5e-13)
            gram = basis.Q.T @ np.column_stack([basis.metric(q) for q in basis.Q.T])
            np.testing.assert_allclose(gram, np.eye(basis.actions), atol=5e-13)

    def test_affine_polynomial_reproduction_up_to_basis_degree(self):
        rng = np.random.default_rng(120)
        A = rng.normal(size=(30,30))*.1
        r = rng.normal(size=30)
        basis = arnoldi(lambda x: A @ x, r, 6)
        expected = np.zeros(30)
        for horizon in range(1, 8):
            expected = r + A @ expected
            if horizon <= 6:
                np.testing.assert_allclose(basis.displacement(horizon), expected,
                                           atol=2e-13, rtol=2e-13)
            else:
                self.assertGreater(np.linalg.norm(basis.displacement(horizon)-expected),1e-7)

    def test_closed_subspace_long_horizon_and_unit_eigenvalue(self):
        A = np.array([[1., .5], [0., 1.]])
        r = np.array([0., 1.])  # (I-A) delta=r has no solution.
        basis = arnoldi(lambda h: A @ h, r, 4)
        self.assertEqual(basis.actions,2)
        for m in [1,2,10,100]:
            expected = np.array([m*(m-1)*.25, float(m)])
            np.testing.assert_allclose(basis.displacement(m),expected,atol=1e-10)
        self.assertLess(np.linalg.norm(basis.remainder),1e-13)

    def test_metric_coordinate_covariance(self):
        rng=np.random.default_rng(121)
        A=rng.normal(size=(18,18))*.1
        r=rng.normal(size=18)
        S=np.exp(rng.uniform(-3,3,18))
        original=arnoldi(lambda h:A@h,r,5)
        transformed=arnoldi(lambda h:S*(A@(h/S)),S*r,5,lambda h:h/(S*S))
        np.testing.assert_allclose(transformed.displacement(17)/S,
                                   original.displacement(17),atol=2e-12,rtol=2e-12)

    def test_shadow_defect_decomposition_and_lipschitz_bound(self):
        class Smooth:
            def step(self,z): return .7*np.tanh(z)+.1
            def linearize(self,z):
                return Linearization(self.step(z), lambda h:.7/(np.cosh(z)**2)*h)
        model=Smooth();z=np.linspace(-1,1,12)
        _,b,lin=finite_flow(model,z,3,12)
        record=shadow_diagnostic(model,z,b,lin,12)
        self.assertLess(max(x['decomposition_error'] for x in record['rows']),3e-15)
        bound=sum(.7**(11-i)*x['defect'] for i,x in enumerate(record['rows']))
        self.assertLessEqual(record['terminal_error'],bound+2e-14)

    def test_entropy_coordinates_are_exact_for_ordinary_iteration(self):
        dual=EntropicSeparable(20,9);primal=EntropicPrimal(20,9)
        y=dual.initial;x=primal.initial
        for _ in range(100):
            y=dual.step(y);x=primal.step(x)
        np.testing.assert_allclose(np.exp(y),x,atol=2e-13,rtol=2e-13)
        self.assertAlmostEqual(dual.gap(y),primal.gap(x),places=12)

    def test_simplex_feasibility_and_reduced_chart(self):
        model=EntropicSimplex(21,4)
        for y in [model.initial,np.linspace(-100,100,20)]:
            x=model.primal(y)
            self.assertGreater(np.min(x),0)
            self.assertAlmostEqual(float(x.sum()),1.,places=14)
        target=model.logstar[:-1]-model.logstar[-1]
        np.testing.assert_allclose(model.step(target),target,atol=1e-13)
        self.assertLess(model.gap(target),1e-13)

    def test_admm_known_kkt_point_and_necessary_memory(self):
        model=ADMMLasso(24,8)
        target=np.concatenate((model.xstar,model.lam/model.rho*model.subgradient))
        np.testing.assert_allclose(model.step(target),target,atol=1e-13)
        self.assertLess(model.gap(target),1e-13)
        first=model.step(model.initial)
        amputated=first.copy();amputated[model.n:]=0
        self.assertGreater(np.linalg.norm(model.step(first)-model.step(amputated)),.01)

    def test_scalar_curvature_can_defeat_cost_adjusted_flow(self):
        class Quartic:
            def step(self,z):return z-.3*z**3
            def linearize(self,z):return Linearization(self.step(z),lambda h:(1-.9*z*z)*h)
        model=Quartic();z=np.array([1.])
        accelerated,_,_=finite_flow(model,z,depth=1,horizon=16)
        accelerated=model.step(accelerated)  # one map + one tangent + one map
        base=z.copy()
        for _ in range(3):base=model.step(base)
        self.assertGreater(float(accelerated[0]**4),float(base[0]**4))

    def test_zero_residual(self):
        model=QuadraticMirror(12,1)
        state,basis,_=finite_flow(model,model.target,depth=4,horizon=100)
        np.testing.assert_array_equal(state,model.target)
        self.assertEqual(basis.actions,0)

    def test_quadratic_finite_horizon_cg_energy_identity_and_bound(self):
        condition=50.
        model=QuadraticMirror(28,2,condition=condition)
        z=model.initial.copy()
        q=1-model.alpha/condition
        theta=(np.sqrt(condition)-1)/(np.sqrt(condition)+1)
        lin=model.linearize(z)
        def W(h):return lin.metric(h-lin.action(h))
        for k in [1,2,4,8]:
            for m in [1,8,64]:
                candidate,basis,_=finite_flow(model,z,k,m)
                rhs=np.zeros(basis.actions);rhs[0]=basis.beta
                cinf=np.linalg.solve(np.eye(basis.actions)-basis.H,rhs)
                infinite=z+basis.Q@cinf
                ei=infinite-model.target
                tail=candidate-infinite
                actual=candidate-model.target
                self.assertAlmostEqual(float(ei@W(tail)),0.,places=11)
                self.assertAlmostEqual(float(actual@W(actual)),
                                       float(ei@W(ei)+tail@W(tail)),places=11)
                bound=q**(2*m)+(1-q**(2*m))*min(1.,4*theta**(2*k))
                self.assertLessEqual(model.gap(candidate)/model.gap(z),bound+1e-11)

    def test_boundary_mirror_descent_has_exact_rank_one_finite_flow(self):
        model=EntropicBoundary(20,7)
        z=model.initial
        candidate,basis,_=finite_flow(model,z,depth=4,horizon=500)
        self.assertEqual(basis.actions,1)
        np.testing.assert_allclose(candidate,z+500*model.r,atol=2e-10)
        self.assertGreater(np.linalg.norm(model.r),1.)
        self.assertAlmostEqual(float(basis.H[0,0]),1.,places=13)
        self.assertLess(model.gap(candidate),1e-10)

    def test_anderson_constant_residual_does_not_fit_roundoff(self):
        model=EntropicBoundary(32,2);aa=Anderson(4)
        z=model.initial.copy()
        for _ in range(300):
            z=aa.advance(z,model.step(z))
        np.testing.assert_allclose(z,300*model.r,atol=2e-2,rtol=2e-4)

    def test_admm_shadow_is_exact_ordinary_state_not_amputated_memory(self):
        for rho in [.001,.1,1.,10.]:
            shadow=ADMMShadow(24,seed=6,rho=rho)
            s=shadow.initial.copy();z=shadow.full.initial.copy()
            for _ in range(100):
                s=shadow.step(s);z=shadow.full.step(z)
                np.testing.assert_allclose(shadow.decode(s),z,atol=2e-11,rtol=2e-11)
            self.assertAlmostEqual(shadow.gap(s),shadow.full.gap(z),places=10)

    def test_shadow_tangent_is_nonexpansive_and_lift_has_same_metric(self):
        rng=np.random.default_rng(123)
        model=ADMMShadow(24,seed=2,rho=1.)
        s=rng.normal(size=24)
        lin=model.linearize(s)
        J=np.column_stack([lin.action(h) for h in np.eye(24)])
        self.assertLessEqual(np.linalg.norm(J,2),1.+1e-12)
        D=np.diag(np.abs(s)>model.full.lam/model.full.rho)
        L=np.vstack((D,np.eye(24)-D))
        np.testing.assert_array_equal(L.T@L,np.eye(24))


if __name__ == '__main__': unittest.main()
