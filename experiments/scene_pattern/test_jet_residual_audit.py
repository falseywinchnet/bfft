import copy
import unittest
import numpy as np
from .jet_residual_audit import audit


class ResidualAuditTests(unittest.TestCase):
    def test_translation_and_neighborhood_distortion_remain_separate(self):
        camera=dict(shape=[101,101],focal=100.,projection='perspective')
        jet=dict(track=0,center=[0.,0,5.],radius=3.,tangents=[[.05,0,0],[0,.05,0]],holdout_capture=0,accepted=False,
                 observations=[dict(capture=0,xy=[53.,50.],affine=np.eye(2).tolist())])
        pose=dict(poses={'0':dict(rotation=np.eye(3).tolist(),translation=[0.,0,0])},surface_jets=[jet])
        translation=audit(pose,[camera])['observations'][0]
        self.assertAlmostEqual(translation['center_pixels'],3.)
        self.assertAlmostEqual(translation['differential_rms_pixels'],0.)
        warped=copy.deepcopy(pose);obs=warped['surface_jets'][0]['observations'][0]
        obs['xy']=[50.,50.];obs['affine'][0][0]=1.5
        distortion=audit(warped,[camera])['observations'][0]
        self.assertAlmostEqual(distortion['center_pixels'],0.)
        self.assertGreater(distortion['differential_rms_pixels'],1.)


if __name__=='__main__':unittest.main()
