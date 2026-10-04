"""Independent interior quadrature checks of finite analytic edge moments."""
from math import comb
import unittest
import mpmath as mp
import numpy as np
from experiments.conv_warp.projective_bank_proof import quintic_bank


def interior_bank(vertices, gamma, alpha, beta):
    nodes,weights=np.polynomial.legendre.leggauss(40)
    u=(nodes[:,None]+1)/2
    v=(nodes[None,:]+1)/2
    result=np.zeros((6,6))
    p0=np.array(vertices[0],dtype=float)
    for index in range(1,len(vertices)-1):
        p1,p2=np.array(vertices[index],dtype=float),np.array(vertices[index+1],dtype=float)
        dx,dy=p1-p0,p2-p0
        x=p0[0]+u*dx[0]+(1-u)*v*dy[0]
        y=p0[1]+u*dx[1]+(1-u)*v*dy[1]
        jac=abs(dx[0]*dy[1]-dx[1]*dy[0])*(1-u)
        w=weights[:,None]*weights[None,:]/4*jac/(gamma+alpha*x+beta*y)**3
        for i in range(6):
            for j in range(6):
                result[i,j]+=np.sum(w*comb(5,i)*x**i*(1-x)**(5-i)*comb(5,j)*y**j*(1-y)**(5-j))
    return result


class ProjectiveBankProof(unittest.TestCase):
    def test_all_36_moments_against_independent_interior_quadrature(self):
        polygon=[('0.1','0.2'),('0.8','0.1'),('0.9','0.7'),('0.3','0.9')]
        # Positive and negative coordinate Jacobians, axis exchange, nearly
        # affine denominators, and the exactly affine polynomial branch.
        maps=[('1','.4','-.2'),('1','-.3','.1'),('1','.1','.5'),
              ('1','0','.4'),('1','.0001','-.00002'),('1','0','0')]
        with mp.workdps(100):
            for gamma,alpha,beta in maps:
                with self.subTest(alpha=alpha,beta=beta):
                    bank=quintic_bank(polygon,gamma,alpha,beta)
                    numerical=interior_bank(polygon,float(gamma),float(alpha),float(beta))
                    np.testing.assert_allclose(np.array(bank,dtype=float),numerical,atol=2e-14,rtol=2e-12)
                    self.assertTrue(all(value>0 for row in bank for value in row))

    def test_constant_denominator_edges_and_orientation(self):
        triangle=[('0','0'),('1','0'),('0','1')]
        with mp.workdps(80):
            bank=quintic_bank(triangle,'1','.5','.5')
            reverse=quintic_bank(list(reversed(triangle)),'1','.5','.5')
            self.assertEqual(bank,reverse)
            np.testing.assert_allclose(np.array(bank,dtype=float),interior_bank(triangle,1,.5,.5),atol=2e-14,rtol=2e-12)


if __name__ == '__main__':
    unittest.main()
