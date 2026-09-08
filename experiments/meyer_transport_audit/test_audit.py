import unittest
import numpy as np
from .model import ReducedMeyerMap,State,grad,div,project_disk
from .run_audit import finite_jump
from .certificate import certificate,tv

class AuditTests(unittest.TestCase):
    def test_adjoint_sign(self):
        rng=np.random.default_rng(4);u=rng.normal(size=(12,16));x=rng.normal(size=u.shape);y=rng.normal(size=u.shape)
        gx,gy=grad(u)
        self.assertAlmostEqual(float(np.sum(gx*x+gy*y)),float(-np.sum(u*div(x,y))),places=11)

    def test_tangent_matches_finite_difference_away_from_boundary(self):
        rng=np.random.default_rng(5);m=ReducedMeyerMap(rng.normal(size=(12,16)),.05,40)
        z=State(*(rng.normal(size=m.shape) for _ in range(6)))
        h=State(*(rng.normal(size=m.shape) for _ in range(6)));eps=1e-6
        fd=(m.pack(m.step(m.add_scaled(z,h,eps)))-m.pack(m.step(m.add_scaled(z,h,-eps))))/(2*eps)
        np.testing.assert_allclose(fd,m.pack(m.tangent(z,h)),atol=2e-9,rtol=2e-8)

    def test_one_pass_polynomial(self):
        m=ReducedMeyerMap(np.arange(256).reshape(16,16),.05,40);z=m.initial()
        a,_=finite_jump(m,z,2,1)
        np.testing.assert_allclose(m.pack(a),m.pack(m.step(z)),atol=1e-12)

    def test_certificate_decomposes_into_nonnegative_terms(self):
        rng=np.random.default_rng(6);m=ReducedMeyerMap(rng.normal(size=(16,16))*30+90,.05,40)
        z=State(*(rng.normal(size=m.shape)*20 for _ in range(6)))
        ux,uy=project_disk(z.tux,z.tuy,m.ru);wx,wy=project_disk(z.twx,z.twy,m.rw)
        q=-m.etau*div(ux,uy);v=-(m.etaw/m.cw)*div(wx,wy)
        terms=[tv(z.u)-float(np.sum(q*z.u)),m.mu*tv(q)-float(np.sum(q*v)),
               .5*m.lam*float(np.sum((m.image-z.u-v-q/m.lam)**2))]
        self.assertGreaterEqual(min(terms),-1e-9)
        self.assertAlmostEqual(sum(terms),certificate(m,z)['gap'],places=7)

    def test_constant_exact_certificate(self):
        m=ReducedMeyerMap(np.full((16,16),93.),.05,40)
        self.assertLess(certificate(m,m.initial())['gap'],1e-20)

    def test_boundary_fixed_derivative_remainder_is_linear(self):
        from .model import project_disk_derivative
        for eps in [1e-2,1e-4,1e-6]:
            x=np.array([1.]);zero=np.array([0.]);h=np.array([eps])
            p,_=project_disk(x+h,zero,1.)
            derivative,_=project_disk_derivative(x,zero,h,zero,1.)
            self.assertAlmostEqual(float(abs(p-x-derivative)[0]),eps,places=14)

if __name__=='__main__': unittest.main()
