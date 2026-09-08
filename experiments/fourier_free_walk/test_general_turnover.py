import itertools,unittest
import numpy as np
from walk import Gate
from turnover import product,search
from general_turnover import general

class Tests(unittest.TestCase):
    def test_general_oblique_charts(self):
        rng=np.random.default_rng(804);accepted=0
        for trial in range(100):
            word=[]
            for i,j in ((0,1),(1,2),(0,1)):
                g=rng.normal(size=(2,2))
                while abs(np.linalg.det(g))<.3:g=rng.normal(size=(2,2))
                word.append(Gate(i,j,g))
            for wires in itertools.permutations(range(3)):
                replacement=general(word,list(wires))
                if replacement is None:continue
                np.testing.assert_allclose(product(word,list(wires)),product(replacement,list(wires)),atol=1e-9,rtol=1e-9);accepted+=1
        self.assertGreater(accepted,500)
    def test_chart_singularity_is_explicit(self):
        self.assertIsNone(general([Gate(0,1,np.eye(2))],[0,1,2]))
    def test_complete_general_walk(self):
        result=search(8,4,600,general_frames=True)
        self.assertGreater(result['general_frame_moves'],0)
        self.assertLess(result['max_matrix_error'],1e-9)

if __name__=='__main__':unittest.main()
