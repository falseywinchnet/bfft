import copy
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from .submap_alignment import similarity,align,transfer_pose


class SubmapTests(unittest.TestCase):
    def test_similarity_and_camera_transfer_preserve_rays(self):
        rng=np.random.default_rng(617);x=rng.normal(size=(30,3));r=Rotation.from_rotvec([.1,-.3,.2]).as_matrix();s=2.3;t=np.array([3.,-1,2]);y=s*x@r.T+t
        ss,rr,tt=similarity(x,y);self.assertAlmostEqual(ss,s);np.testing.assert_allclose(rr,r,atol=1e-10);np.testing.assert_allclose(tt,t,atol=1e-10)
        pose=dict(rotation=np.eye(3).tolist(),translation=[-1.,0,0],center=[1.,0,0]);new=transfer_pose(pose,dict(scale=s,rotation=r,translation=t))
        np.testing.assert_allclose(y@np.asarray(new['rotation']).T+new['translation'],s*(x+[-1.,0,0]),atol=1e-10)

    def test_shared_points_validate_frame_and_poisoned_holdout_rejects(self):
        rng=np.random.default_rng(55);x=rng.uniform([-2,-1,5],[2,1,12],(90,3));r=Rotation.from_rotvec([.1,.2,-.1]).as_matrix();s=1.7;t=np.array([1.,2,-.5]);local=(x-t)@r/s
        target_poses={};source_poses={};tracks=[];camera=dict(shape=[600,800],focal=500.,projection='perspective')
        for i,c in enumerate(([0.,0,0],[1.,0,0],[.4,.3,0])):
            target_poses[str(i)]=dict(rotation=np.eye(3).tolist(),translation=(-np.asarray(c)).tolist(),center=c)
            source_poses[str(i)]=dict(rotation=r.tolist(),translation=((t-c)/s).tolist(),center=((np.asarray(c)-t)@r/s).tolist())
        for k,p in enumerate(x):
            obs=[]
            for i,c in enumerate(([0.,0,0],[1.,0,0],[.4,.3,0])):
                q=p-c;obs.append(dict(capture=i,site=k,xy=(q[:2]/q[2]*500+[399.5,299.5]).tolist()))
            tracks.append(dict(id=k,observations=obs))
        source=dict(poses=source_poses,points=[dict(track=k,xyz=p.tolist(),promotion='third_view_consistent') for k,p in enumerate(local)])
        target=dict(poses=target_poses,points=[dict(track=k,xyz=p.tolist(),promotion='third_view_consistent') for k,p in enumerate(x)]);e=dict(cameras=[camera]*3,tracks=tracks)
        result=align(source,target,e);self.assertTrue(result['accepted']);self.assertAlmostEqual(result['scale'],s,places=6)
        poisoned=copy.deepcopy(e)
        for track in poisoned['tracks']:
            if track['id']%7==0:
                for o in track['observations']:o['xy']=[10.,10.]
        bad=align(source,target,poisoned);self.assertFalse(bad['accepted']);np.testing.assert_array_equal(bad['translation'],result['translation'])


if __name__=='__main__':unittest.main()
