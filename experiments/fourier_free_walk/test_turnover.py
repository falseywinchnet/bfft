import itertools,unittest
import numpy as np
from walk import Gate
from turnover import rotation,euler,product,simplify,search

class Tests(unittest.TestCase):
    def test_all_euler_charts(self):
        rng=np.random.default_rng(75)
        for count in (2,3):
            for trial in range(100):
                word=[]
                for _ in range(count):
                    i,j=map(int,rng.choice(3,2,replace=False));theta=rng.uniform(-np.pi,np.pi)
                    word.append(Gate(i,j,rotation(np.cos(theta),np.sin(theta))))
                for wires in itertools.permutations(range(3)):
                    actual=euler(word,list(wires))
                    np.testing.assert_allclose(product(actual,list(wires)),product(word,list(wires)),atol=2e-12)
    def test_degenerate_charts(self):
        for theta in (0,np.pi,-np.pi,np.pi/2):
            word=[Gate(0,1,rotation(np.cos(theta),np.sin(theta)))]
            for wires in itertools.permutations(range(3)):
                np.testing.assert_allclose(product(euler(word,list(wires)),list(wires)),product(word,list(wires)),atol=1e-12)
    def test_commuting_fusion(self):
        a=Gate(0,1,rotation(.6,.8));b=Gate(2,3,rotation(.8,.6));c=Gate(0,1,a.g.T)
        word=[a,b,c];simplified=simplify(word)
        self.assertEqual(len(simplified),1)
        np.testing.assert_allclose(product(word,list(range(4))),product(simplified,list(range(4))),atol=1e-14)
    def test_full_walk(self):
        for n in (4,6,8):
            result=search(n,9,300)
            self.assertLess(result['max_matrix_error'],1e-10)
            self.assertTrue(result['route']['success'])

if __name__=='__main__':unittest.main()
