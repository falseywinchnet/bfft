import unittest
import numpy as np
from .joint_submap import validate,initialize_union


class JointMapGateTests(unittest.TestCase):
    def test_existing_camera_regression_cannot_be_hidden_by_new_coverage(self):
        points=[dict(track=i,xyz=[i*.03,.1,5.]) for i in range(35)];cameras=[dict(shape=[101,101],focal=100.,projection='perspective')]*3
        poses={str(i):dict(rotation=np.eye(3).tolist(),translation=[-i,0.,0.],center=[i,0.,0.]) for i in range(3)}
        tracks=[]
        for p in points:
            obs=[]
            for i in range(3):
                q=np.array(p['xyz'])+poses[str(i)]['translation'];obs.append(dict(capture=i,site=p['track'],xy=(q[:2]/q[2]*100+50).tolist()))
            tracks.append(dict(id=p['track'],observations=obs))
        e=dict(cameras=cameras,tracks=tracks);before=dict(poses={k:v for k,v in poses.items() if k!='2'},points=points);candidate=dict(poses=poses,points=points)
        self.assertTrue(validate(e,before,candidate)['accepted'])
        candidate['poses']['0']=dict(poses['0'],translation=[1.,0,0])
        self.assertFalse(validate(e,before,candidate)['accepted'])

    def test_joint_fit_recovers_new_camera_without_damaging_old_views(self):
        import copy
        from scipy.spatial.transform import Rotation
        from .scene_bundle import refine
        rng=np.random.default_rng(201);xyz=rng.uniform([-2,-1,5],[2,1,10],(80,3))
        centers=np.array([[0.,0,0],[1.,0,0],[.3,.4,.1],[1.3,-.2,.2]])
        models=[dict(shape=[600,800],focal=500.,projection='perspective')]*4
        poses={str(i):dict(rotation=np.eye(3).tolist(),translation=(-c).tolist(),center=c.tolist(),role='heldout_pose') for i,c in enumerate(centers)}
        tracks=[]
        for k,x in enumerate(xyz):
            obs=[]
            for i,c in enumerate(centers):
                q=x-c;obs.append(dict(capture=i,site=4*k+i,xy=(q[:2]/q[2]*500+[399.5,299.5]).tolist()))
            tracks.append(dict(id=k,observations=obs))
        evidence=dict(cameras=models,tracks=tracks)
        before=dict(accepted=True,seed=[0,1],poses={i:p for i,p in poses.items() if i!='3'},points=[dict(track=k,xyz=x.tolist()) for k,x in enumerate(xyz)])
        initial=copy.deepcopy(before);initial['poses']['3']=dict(poses['3'],rotation=Rotation.from_rotvec([.01,-.02,.015]).as_matrix().tolist(),translation=[-1.27,.21,-.18])
        candidate,receipt=refine(evidence,initial,max_evaluations=80)
        self.assertTrue(receipt['applied']);self.assertTrue(validate(evidence,before,candidate)['accepted'])
        np.testing.assert_allclose(candidate['poses']['3']['center'],centers[3],atol=.003)

    def test_union_keeps_shared_camera_single_and_does_not_replace_old_points(self):
        p=dict(rotation=np.eye(3).tolist(),translation=[0.,0,0],center=[0.,0,0])
        base=dict(poses={'0':p},points=[dict(track=0,xyz=[0.,0,3.])])
        source=dict(poses={'0':dict(p,translation=[9.,0,0]),'1':p},points=[dict(track=0,xyz=[9.,9,9],promotion='third_view_consistent'),dict(track=1,xyz=[1.,0,3.],promotion='third_view_consistent')])
        relation=dict(scale=2.,rotation=np.eye(3).tolist(),translation=[1.,0,0]);out=initialize_union(base,source,relation)
        self.assertEqual(out['poses']['0'],base['poses']['0']);self.assertEqual(out['points'][0],base['points'][0]);self.assertEqual(out['points'][1]['xyz'],[3.,0.,6.])


class TrainingPointRefreshTests(unittest.TestCase):
    def test_refresh_uses_final_cameras_but_never_withheld_rays(self):
        import copy
        from .joint_submap import refresh_training_points
        xyz=np.array([.2,.3,5.]);cameras=[dict(shape=[101,101],focal=100.,projection='perspective')]*3
        poses={str(i):dict(rotation=np.eye(3).tolist(),translation=[-i,0.,0.],center=[i,0.,0.]) for i in range(3)}
        obs=[]
        for i in range(3):
            q=xyz+poses[str(i)]['translation'];obs.append(dict(capture=i,site=7 if i==2 else 1,xy=(q[:2]/q[2]*100+50).tolist()))
        evidence=dict(cameras=cameras,tracks=[dict(id=0,observations=obs)])
        pose=dict(poses=poses,points=[dict(track=0,xyz=[9.,9,9])])
        result=refresh_training_points(evidence,pose);np.testing.assert_allclose(result['points'][0]['xyz'],xyz,atol=1e-10)
        poisoned=copy.deepcopy(evidence);poisoned['tracks'][0]['observations'][2]['xy']=[0.,0.]
        other=refresh_training_points(poisoned,pose);self.assertEqual(result['points'],other['points'])


if __name__=='__main__':unittest.main()
