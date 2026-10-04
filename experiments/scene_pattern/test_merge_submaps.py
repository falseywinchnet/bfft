import copy
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .merge_submaps import merge


class MergeTests(unittest.TestCase):
    def test_frame_join_adds_camera_and_rejects_conflicting_camera(self):
        rng=np.random.default_rng(712);points=rng.uniform([-2,-1,5],[2,1,10],(70,3));centers=np.array([[0.,0,0],[1.,0,0],[.4,.3,0],[1.3,-.2,.1]])
        r=Rotation.from_rotvec([.1,-.2,.1]).as_matrix();scale=2.;offset=np.array([2.,-1,.5]);local=(points-offset)@r/scale
        cameras=[dict(shape=[600,800],focal=500.,projection='perspective')]*4;tracks=[]
        for tid,x in enumerate(points):
            obs=[]
            for i,c in enumerate(centers):
                q=x-c;obs.append(dict(capture=i,site=tid,xy=(q[:2]/q[2]*500+[399.5,299.5]).tolist()))
            tracks.append(dict(id=tid,observations=obs))
        target=dict(accepted=True,seed=[0,1],poses={str(i):dict(rotation=np.eye(3).tolist(),translation=(-c).tolist(),center=c.tolist(),role='heldout_pose') for i,c in enumerate(centers[:3])},
                    points=[dict(track=k,xyz=x.tolist(),promotion='third_view_consistent') for k,x in enumerate(points)])
        source=dict(accepted=True,seed=[0,1],poses={str(i):dict(rotation=r.tolist(),translation=((offset-c)/scale).tolist(),center=((c-offset)@r/scale).tolist(),role='heldout_pose') for i,c in enumerate(centers)},
                    points=[dict(track=k,xyz=x.tolist(),promotion='third_view_consistent') for k,x in enumerate(local)])
        e=dict(cameras=cameras,tracks=tracks)
        with tempfile.TemporaryDirectory() as path:
            folder=Path(path);(folder/'submaps.json').write_text(json.dumps(dict(submaps=[dict(accepted=True,seed=[0,1],cameras=[0,1,2,3],promotion=dict(third_view_consistent=70))],observed_cameras=[0,1,2,3])))
            (folder/'000-001.json').write_text(json.dumps(source));good=merge(e,target,folder)
            self.assertEqual(len(good['poses']),4);np.testing.assert_allclose(good['poses']['3']['center'],centers[3],atol=1e-6)
            self.assertEqual(good['poses']['0'],target['poses']['0'])
            calibrated_target=copy.deepcopy(target);calibrated_target['camera_models']=copy.deepcopy(cameras)
            calibrated_source=copy.deepcopy(source);calibrated_source['camera_models']=copy.deepcopy(cameras)
            nominal=copy.deepcopy(e)
            for camera in nominal['cameras']:camera['focal']=250.
            (folder/'000-001.json').write_text(json.dumps(calibrated_source))
            calibrated=merge(nominal,calibrated_target,folder)
            self.assertEqual(len(calibrated['poses']),4);self.assertEqual(len(calibrated['points']),70)
            self.assertEqual(calibrated['camera_models'][3]['focal'],500.)
            bad=copy.deepcopy(source);bad['poses']['3']['translation'][0]+=20
            (folder/'000-001.json').write_text(json.dumps(bad));rejected=merge(e,target,folder)
            self.assertEqual(len(rejected['poses']),3);self.assertEqual(rejected['submap_merge']['attempts'][0]['rejected'],'new_cameras_conflict_with_existing_points')


if __name__=='__main__':unittest.main()
