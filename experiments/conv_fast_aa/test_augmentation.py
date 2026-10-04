import unittest
import numpy as np
from .augmentation import representations
from experiments.convstar import synthesize_polyphase


class Augmentation(unittest.TestCase):
    def test_fused_identity_and_nodes(self):
        a=np.random.default_rng(90).random((39,7));base,residual,raw,current=representations(a)
        np.testing.assert_allclose(residual[::8],0,atol=1e-15)
        for lam in (0,.25,.5,1):
            np.testing.assert_allclose(synthesize_polyphase(a,current+lam*(raw-current),8),base+lam*residual,atol=2e-15)

    def test_unseen_line_has_no_residual(self):
        # A line entirely between observed source points gives no source current.
        base,residual,_,_=representations(np.zeros(21))
        np.testing.assert_array_equal(base,0);np.testing.assert_array_equal(residual,0)


if __name__=='__main__':unittest.main()
