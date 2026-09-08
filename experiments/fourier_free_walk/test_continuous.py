import itertools,unittest
import numpy as np
from walk import Gate,inverse2
from turnover import product
from general_turnover import general

class Tests(unittest.TestCase):
    def test_arbitrary_mixer_excursion_preserves_transfer(self):
        rng=np.random.default_rng(322);accepted=0
        for trial in range(40):
            old=Gate(0,1,np.array([[1.,rng.uniform(-2,2)],[0.,1.]]))
            theta=rng.uniform(-2,2)
            h=np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]]) if trial%2 else np.array([[1.,theta],[0.,1.]])
            bridge=Gate(1,2,h)
            for wires in itertools.permutations(range(3)):
                changed=general([old,bridge],list(wires))
                if changed is None:continue
                changed.append(Gate(bridge.i,bridge.j,inverse2(h)))
                np.testing.assert_allclose(product(changed,list(range(3))),product([old],list(range(3))),atol=1e-10)
                accepted+=1
        self.assertGreater(accepted,40)

if __name__=='__main__':unittest.main()
