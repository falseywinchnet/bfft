import copy
import unittest
import numpy as np
from .point_support_audit import fit_training_point
from .surface_jets import project


class PointPredictionControlTests(unittest.TestCase):
    def test_fixed_camera_control_never_uses_withheld_ray_or_initial_point(self):
        c=dict(shape=[600,800],focal=500.,projection='perspective');cameras=[c]*3
        poses={str(i):dict(rotation=np.eye(3).tolist(),translation=[-float(i),0,0],center=[float(i),0,0]) for i in range(3)}
        x=np.array([.2,.3,6.]);obs=[dict(capture=i,xy=project([x],c,poses[str(i)])[0].tolist(),training=i<2) for i in range(3)]
        a,_=fit_training_point(obs,cameras,poses);np.testing.assert_allclose(a,x,atol=1e-10)
        bad=copy.deepcopy(obs);bad[-1]['xy']=[5000,8000];b,_=fit_training_point(bad,cameras,poses);np.testing.assert_array_equal(a,b)


if __name__=='__main__':unittest.main()
