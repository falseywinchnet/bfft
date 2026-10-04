import unittest
import sympy as s

from experiments.conv_dual_pair.derive_two_thirds import (
    families,matrices,equations,z,
)
from experiments.conv_dual_pair.audit_two_thirds import (
    original_analysis,canonical_polynomial_defects,
)


class TwoThirdsTests(unittest.TestCase):
    def test_original_banks_are_exact_quintic_basin_integrals(self):
        x=s.Symbol('x')
        y={i:s.Symbol(f'y{i}') for i in range(-3,6)}
        def polynomial(i):
            m=lambda j:(y[j-2]-8*y[j-1]+8*y[j+1]-y[j+2])/12
            q=lambda j:(-y[j+2]+16*y[j+1]-30*y[j]+16*y[j-1]-y[j-2])/12
            p=[y[i],y[i]+m(i)/5,y[i]+2*m(i)/5+q(i)/20,
               y[i+1]-2*m(i+1)/5+q(i+1)/20,y[i+1]-m(i+1)/5,y[i+1]]
            return sum(s.binomial(5,j)*x**j*(1-x)**(5-j)*p[j] for j in range(6))
        A=original_analysis()
        for phase,center in enumerate((s.Rational(0),s.Rational(3,2))):
            lo,hi=center-s.Rational(3,4),center+s.Rational(3,4)
            integral=0
            for cell in range(int(s.floor(lo)),int(s.ceiling(hi))):
                integral+=s.integrate(polynomial(cell),(x,max(lo,cell)-cell,min(hi,cell+1)-cell))*s.Rational(2,3)
            integral=s.expand(integral)
            for k in y:
                self.assertEqual(integral.coeff(y[k]),s.expand(A[phase,k%3]).coeff(z,k//3))

    def test_seven_tap_centered_pair_is_infeasible(self):
        h,g,p=families(5)
        self.assertEqual(list(map(len,g)),[7,7,7])
        A,G=matrices(h,g)
        basis=s.groebner(equations(A,G),*p)
        self.assertEqual([v.as_expr() for v in basis.polys],[1])

    def test_next_algebraic_pair_cannot_conserve_mass(self):
        h,g,p=families(6)
        A,G=matrices(h,g)
        eq=equations(A,G)+[sum(A[:,0]).subs(z,1)-s.Rational(2,3)]
        basis=s.groebner(eq,*p)
        self.assertEqual([v.as_expr() for v in basis.polys],[1])

    def test_canonical_dual_fails_affine_reproduction(self):
        self.assertEqual(canonical_polynomial_defects(original_analysis(),1),
                         [[0,0,0],[0,s.Rational(29973,655360),-s.Rational(29973,655360)]])

    def test_original_analysis_has_full_rank_on_unit_circle(self):
        A=original_analysis()
        determinant=s.factor((A*A.subs(z,1/z).T).det())
        numerator=s.fraction(determinant)[0]
        coefficients=s.Poly(numerator,z).all_coeffs()
        center=coefficients[4]
        # Palindromic Laurent polynomial: constant term dominates the sum of
        # every oscillatory term, so the determinant is strictly positive.
        self.assertGreater(center,sum(abs(v) for j,v in enumerate(coefficients) if j!=4))

    def test_loss_one_incompatible_with_quartic_and_mass_in_this_analysis_family(self):
        h,g,p=families()
        A,_=matrices(h,g)
        kernel=A.row(0).cross(A.row(1))
        equations=[s.factor(sum(s.expand(kernel[j]).coeff(z,l)*s.Rational(j-3*l)**k
                   for j in range(3) for l in range(-2,3))) for k in (1,3,4)]
        equations.append(sum(A[:,0]).subs(z,1)-s.Rational(2,3))
        basis=s.groebner(equations,*p[:2])
        self.assertEqual([v.as_expr() for v in basis.polys],[1])


if __name__=='__main__':unittest.main()
