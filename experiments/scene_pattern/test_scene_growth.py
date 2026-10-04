import copy
import unittest
import numpy as np
from .scene_growth import grow


class GrowthTests(unittest.TestCase):
    def test_refreshed_structure_places_new_camera_and_keeps_holdout_gate(self):
        rng=np.random.default_rng(552);points=rng.uniform([-2,-1,5],[2,1,10],(100,3));centers=np.array([[0.,0,0],[1.,0,0],[.2,.4,.1],[1.3,-.2,.1]])
        cameras=[dict(shape=[600,800],focal=500.,projection='perspective')]*4;tracks=[]
        for k,p in enumerate(points):
            obs=[]
            for i,c in enumerate(centers):
                q=p-c;obs.append(dict(capture=i,site=4*k+i,xy=(q[:2]/q[2]*500+[399.5,299.5]).tolist()))
            tracks.append(dict(id=k,observations=obs))
        poses={str(i):dict(rotation=np.eye(3).tolist(),translation=(-c).tolist(),center=c.tolist(),role='heldout_pose') for i,c in enumerate(centers[:3])}
        e=dict(cameras=cameras,tracks=tracks);pose=dict(accepted=True,seed=[0,1],poses=poses,points=[])
        result=grow(e,pose);self.assertEqual(len(result['poses']),4);np.testing.assert_allclose(result['poses']['3']['center'],centers[3],atol=1e-6)
        poisoned=copy.deepcopy(e)
        for t in poisoned['tracks']:
            o=t['observations'][3]
            if o['site']%7==0:o['xy']=[5.,5.]
        rejected=grow(poisoned,pose);self.assertEqual(len(rejected['poses']),3)


if __name__=='__main__':unittest.main()
