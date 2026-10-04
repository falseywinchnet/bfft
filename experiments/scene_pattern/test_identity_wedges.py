import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from .identity_wedges import directed, wedges
from .identity_pose_graph import loops


class IdentityWedgeTests(unittest.TestCase):
    def test_disagreeing_initial_pose_loop_does_not_freeze_joint_proposals(self):
        pose=dict(R=np.eye(3),t=np.array([-1.,0,0]))
        wrong=dict(R=Rotation.from_rotvec([0,.2,0]).as_matrix(),t=pose['t'])
        pairs={(0,1):[pose],(1,2):[pose],(0,2):[wrong]}
        self.assertEqual(loops(pairs),[])
        hypotheses=directed(pairs)
        self.assertEqual(set(wedges(hypotheses,set(hypotheses))),{(0,1,2),(1,0,2),(2,0,1)})

    def test_third_image_map_can_check_a_wedge_without_a_third_camera_model(self):
        pose=dict(R=np.eye(3),t=np.array([-1.,0,0]))
        hypotheses=directed({(4,8):[pose],(4,9):[pose]})
        self.assertEqual(wedges(hypotheses,{(4,8),(4,9),(8,9)}),[(4,8,9)])
        self.assertEqual(wedges(hypotheses,{(4,8),(4,9)}),[])

    def test_reversed_origin_uses_the_inverse_relative_pose(self):
        R=Rotation.from_rotvec([.2,-.4,.1]).as_matrix();t=np.array([.3,-.2,1.])
        pairs=directed({(2,7):[dict(R=R,t=t)]});p=pairs[7,2][0]
        x=np.array([.1,.3,4.]);np.testing.assert_allclose(p['R']@(R@x+t)+p['t'],x,atol=1e-14)


if __name__=='__main__':unittest.main()
