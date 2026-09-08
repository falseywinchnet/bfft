import unittest
import numpy as np
from experiments.conv_joint_potential import (
    proposal, admit, evaluate, build_problem, local_gradient_gram, tensor_certificate,
    _directional_derivative_coefficients,
)


class JointPotentialTests(unittest.TestCase):
    def test_subdivision_admits_positive_current_with_negative_coefficients(self):
        # F'(u)=(u-.5)^2+.001: the motivating stationary-inflection shape.
        i=np.arange(6,dtype=float)
        controls=.251*i/5-.5*i*(i-1)/20+i*(i-1)*(i-2)/180
        potential=np.tile(controls,(6,1))
        derivative=np.array(_directional_derivative_coefficients(1,0)).reshape(36,36)
        current=derivative@potential.ravel()
        self.assertLess((tensor_certificate(0)@current).min(),-.04)
        self.assertGreater((tensor_certificate(1)@current).min(),.000999999)
        u=np.linspace(0,1,43)
        np.testing.assert_allclose(evaluate(potential,u,.37,'x'),(u-.5)**2+.001,atol=1e-14)

    def test_tensor_polynomial_and_gradient_reproduction(self):
        y,x=np.mgrid[:7,:8]/7
        source=x**3*y**2+.2*x*y-.1*y
        lattice=proposal(source)
        v,u=np.meshgrid(np.linspace(0,6,23),np.linspace(0,7,29),indexing='ij')
        np.testing.assert_allclose(evaluate(lattice,u,v),(u/7)**3*(v/7)**2+.2*u*v/49-.1*v/7,atol=2e-12)
        np.testing.assert_allclose(evaluate(lattice,u,v,'x'),(3*(u/7)**2*(v/7)**2+.2*v/7)/7,atol=2e-12)
        np.testing.assert_allclose(evaluate(lattice,u,v,'y'),(2*(u/7)**3*(v/7)+.2*u/7-.1)/7,atol=2e-12)

    def test_exact_gradient_objective(self):
        p=np.random.default_rng(47).normal(size=(6,6))
        nodes,weights=np.polynomial.legendre.leggauss(7)
        nodes=(nodes+1)/2;weights=weights/2
        yy,xx=np.meshgrid(nodes,nodes,indexing='ij')
        integral=np.sum(weights[:,None]*weights[None,:]*(evaluate(p,xx,yy,'x')**2+evaluate(p,xx,yy,'y')**2))
        self.assertAlmostEqual(integral,p.ravel()@local_gradient_gram()@p.ravel(),places=11)

    def test_shared_traces_range_and_query_independence(self):
        y,x=np.mgrid[:7,:7]
        source=np.tanh((x+.6*y-4.3)/.8)
        p,diag=admit(source,maxiter=60000)
        self.assertTrue(diag['converged'],diag)
        self.assertLess(diag['maximum_violation'],2e-6)
        np.testing.assert_array_equal(p[::5,::5],source)
        # C0 is structural; these are the two normal C1 trace identities.
        np.testing.assert_allclose(2*p[:,5:-1:5],p[:,4:-2:5]+p[:,6::5],atol=1e-12)
        np.testing.assert_allclose(2*p[5:-1:5],p[4:-2:5]+p[6::5],atol=1e-12)
        for cy in range(6):
            for cx in range(6):
                bounds=source[max(0,cy-2):min(7,cy+4),max(0,cx-2):min(7,cx+4)]
                certificate=tensor_certificate(1)@p[5*cy:5*cy+6,5*cx:5*cx+6].ravel()
                self.assertGreaterEqual(certificate.min(),bounds.min()-4e-6)
                self.assertLessEqual(certificate.max(),bounds.max()+4e-6)
        xx=np.array([.11,1.4,3.7,5.999]);yy=np.array([2.7,4.2,.6,1.1])
        a=evaluate(p,xx,yy)
        b=np.array([evaluate(p,u,v) for u,v in zip(xx,yy)])
        np.testing.assert_allclose(a,b,atol=1e-14)
        # Integrate the admitted gradient along two different grid-aligned paths.
        q,w=np.polynomial.legendre.leggauss(4);q=(q+1)/2;w=w/2
        def horizontal(y):
            return sum(np.dot(w,evaluate(p,i+q,np.full(4,y),'x')) for i in range(6))
        def vertical(x):
            return sum(np.dot(w,evaluate(p,np.full(4,x),i+q,'y')) for i in range(6))
        self.assertAlmostEqual(horizontal(0)+vertical(6),vertical(0)+horizontal(6),places=11)
        self.assertAlmostEqual(horizontal(0)+vertical(6),source[-1,-1]-source[0,0],places=11)

    def test_transpose_and_affine_covariance(self):
        y,x=np.mgrid[:6,:7]
        source=np.sin(.9*x+.7*y)
        p,d=admit(source,tol=1e-9,maxiter=60000)
        pt,dt=admit(source.T,tol=1e-9,maxiter=60000)
        pc,dc=admit(3-2*source,tol=1e-9,maxiter=60000)
        self.assertTrue(all(z['converged'] for z in (d,dt,dc)))
        np.testing.assert_allclose(p,pt.T,atol=2e-7)
        np.testing.assert_allclose(pc,3-2*p,atol=4e-7)

    def test_affine_feasible_without_admission(self):
        y,x=np.mgrid[:6,:7]
        p,d=admit(.7*x-.2*y+4)
        self.assertEqual(d['iterations'],0)
        yy,xx=np.meshgrid(np.linspace(0,5,23),np.linspace(0,6,29),indexing='ij')
        np.testing.assert_allclose(evaluate(p,xx,yy),.7*xx-.2*yy+4,atol=2e-12)

    def test_c2_and_natural_quadratic_reproduction(self):
        y,x=np.mgrid[:6,:7]
        source=.02*x*x+.03*y*y+.04*x*y
        p,d=admit(source,continuity=2,boundary='natural',cones=False)
        self.assertTrue(d['converged'],d)
        yy,xx=np.meshgrid(np.linspace(0,5,23),np.linspace(0,6,29),indexing='ij')
        np.testing.assert_allclose(evaluate(p,xx,yy),.02*xx*xx+.03*yy*yy+.04*xx*yy,atol=1e-11)
        p,d=admit(np.sin(.9*x+.7*y),continuity=2,boundary='natural')
        self.assertTrue(d['converged'],d)
        np.testing.assert_allclose(p[:,3:-3:5]-2*p[:,4:-2:5]+2*p[:,6::5]-p[:,7::5],0,atol=2e-12)
        np.testing.assert_allclose(p[3:-3:5]-2*p[4:-2:5]+2*p[6::5]-p[7::5],0,atol=2e-12)


if __name__=='__main__':
    unittest.main()
