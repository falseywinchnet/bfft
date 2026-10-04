import copy
import unittest
import numpy as np
from .surface_observability import infinity_error
from .surface_jets import project


class InfiniteDepthTests(unittest.TestCase):
    def fixture(self,baseline):
        camera=dict(shape=[600,800],focal=500.,projection='perspective');poses={str(i):dict(rotation=np.eye(3).tolist(),translation=[-c,0.,0.],center=[c,0.,0.]) for i,c in enumerate([0.,baseline,1.])}
        xyz=[0.,0.,6.];obs=[dict(capture=i,xy=project([xyz],camera,poses[str(i)])[0].tolist(),affine=np.eye(2).tolist()) for i in range(3)]
        return dict(reference_capture=0,holdout_capture=2,radius=3.,observations=obs),[camera]*3,poses

    def test_colocated_training_views_cannot_bound_depth(self):
        jet,cameras,poses=self.fixture(0.)
        self.assertLess(infinity_error(jet,cameras,poses),1e-10)
        bad=copy.deepcopy(jet);bad['observations'][-1]['xy']=[9000,9000]
        self.assertEqual(infinity_error(jet,cameras,poses),infinity_error(bad,cameras,poses))

    def test_displaced_training_view_rejects_infinity(self):
        jet,cameras,poses=self.fixture(1.)
        self.assertGreater(infinity_error(jet,cameras,poses),80.)


if __name__=='__main__':unittest.main()
