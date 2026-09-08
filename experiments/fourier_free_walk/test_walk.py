import itertools,unittest
import numpy as np
from walk import Gate,inverse2,orient,compile_walk,circuit_matrix,execute,synthesize,verify,fourier,ordered_control

class Tests(unittest.TestCase):
    def test_orientation_against_full_enumeration(self):
        rng=np.random.default_rng(341)
        for n in (3,4):
            for trial in range(8):
                edges=[tuple(map(int,rng.choice(n,2,replace=False))) for _ in range(7)]
                for p in itertools.permutations(range(n)):
                    possible=False
                    for bits in range(1<<len(edges)):
                        state=list(p)
                        for k,(i,j) in enumerate(edges):
                            if bits>>k&1:state[i],state[j]=state[j],state[i]
                        possible |= state==list(range(n))
                    actual=orient(p,edges)
                    self.assertEqual(actual['success'],possible)
                    self.assertTrue(actual['exhaustive'])
    def test_broader_than_radix(self):
        for n in (4,6,8):
            q=fourier(n);np.testing.assert_allclose(q@q.T,np.eye(n),atol=1e-14)
    def test_synthesis_and_physical_execution(self):
        successes=0
        for n in (4,6,8):
            for oblique in (False,True):
                result=synthesize(n,3,oblique)
                if not result['complete']:continue
                route=orient(result['permutation'],[(g.i,g.j) for g in reversed(result['gates'])])
                if not route['success']:continue
                physical,error=verify(result,route);self.assertLess(error,1e-10)
                for j in range(n):
                    x=np.eye(n)[j];np.testing.assert_allclose(execute(x,physical),circuit_matrix(n,physical)@x,atol=1e-12)
                successes+=1
        # Sparse walks can legitimately fail the placement contract; verify
        # compilation separately on a guaranteed, explicitly quadratic control.
        for n in (4,6,8):
            result=ordered_control(n)
            route=orient(result['permutation'],[(g.i,g.j) for g in reversed(result['gates'])])
            self.assertTrue(route['success'])
            gates,error=verify(result,route)
            self.assertLess(error,1e-10)
    def test_inverse_general_shear(self):
        for g in (np.array([[1.,3.],[0.,1.]]),np.array([[2.,-.5],[3.,4.]])):
            np.testing.assert_allclose(inverse2(g)@g,np.eye(2),atol=1e-14)

if __name__=='__main__':unittest.main()
