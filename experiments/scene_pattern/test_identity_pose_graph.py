import copy,unittest
import numpy as np
from scipy.spatial.transform import Rotation
from .identity_pose_graph import loops,fit_scale
from .identity_geometry import roles
from .multiview_geometry import unit

class IdentityPoseGraphTests(unittest.TestCase):
    def test_straight_walk_directions_remain_possible_but_scale_is_underdetermined(self):
        p=dict(R=np.eye(3),t=np.array([-1.,0,0]));pairs={(0,1):[p],(1,2):[p],(0,2):[p]}
        result=loops(pairs);self.assertEqual(len(result),1);self.assertTrue(result[0]['direction_compatible']);self.assertEqual(result[0]['direction_rank'],1)
    def test_rotation_loop_rejects_incompatible_pair_hypotheses(self):
        p=dict(R=np.eye(3),t=np.array([-1.,0,0]));wrong=dict(R=Rotation.from_rotvec([0,.3,0]).as_matrix(),t=p['t'])
        self.assertEqual(loops({(0,1):[p],(1,2):[p],(0,2):[wrong]}),[])
    def test_shared_scale_selects_coherent_identity_and_predicts_unused_regions(self):
        rng=np.random.default_rng(4);ac=dict(R=Rotation.from_rotvec([.01,-.1,.03]).as_matrix(),t=unit(np.array([-.5,-.8,.04])));candidates=[]
        for site in range(120):
            X=rng.uniform([-3,-2,5],[3,2,12]);p=X[:2]/X[2]*400+[399.5,299.5];train,held,_=roles([dict(p=p)],0)
            for s in (1.7,rng.uniform(.3,3.5)):
                candidates.append(dict(X=X,ray=unit(ac['R']@X+s*ac['t']),scale=s,site=site,p=p,training=bool(train[0]),held=bool(held[0]),appearance=.98))
        fits=fit_scale(candidates,ac,dict(focal=400));self.assertTrue(fits);self.assertTrue(fits[0]['accepted']);self.assertAlmostEqual(fits[0]['scale'],1.7,places=8)
        self.assertGreater(fits[0]['check_count'],5);self.assertLess(fits[0]['check_median'],1e-9)
        poisoned=copy.deepcopy(candidates)
        for c in poisoned:
            if not c['training']:c['ray']=unit(rng.normal(size=3))
        other=fit_scale(poisoned,ac,dict(focal=400));self.assertAlmostEqual(other[0]['scale'],fits[0]['scale'],places=12);self.assertFalse(other[0]['accepted'])

if __name__=='__main__':unittest.main()
