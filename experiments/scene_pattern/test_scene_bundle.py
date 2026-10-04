import copy
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from .scene_bundle import refine


class BundleTests(unittest.TestCase):
    def test_cropped_panorama_basis_predicts_unused_pixels(self):
        rng=np.random.default_rng(663);points=rng.uniform([-6,-2,4],[6,2,12],(70,3));centers=np.array([[0.,0,0],[1.,0,0],[.4,.3,.1],[1.3,-.2,.2]])
        shape=[500,1200];f=260.;tracks=[];poses={};models=[dict(shape=shape,focal=f,projection='cylindrical') for _ in centers]
        for k,p in enumerate(points):
            obs=[]
            for i,c in enumerate(centers):
                q=p-c;fx=f*(1+.025*i);fy=f*(1-.02*i);cy=249.5+12*(i-1)
                xy=[np.arctan2(q[0],q[2])*fx+599.5,q[1]/np.hypot(q[0],q[2])*fy+cy]
                obs.append(dict(capture=i,site=4*k+i,xy=xy))
            tracks.append(dict(id=k,observations=obs))
        for i,c in enumerate(centers):poses[str(i)]=dict(rotation=np.eye(3).tolist(),translation=(-c).tolist(),center=c.tolist(),role='heldout_pose')
        e=dict(cameras=models,tracks=tracks);pose=dict(accepted=True,seed=[0,1],poses=poses,points=[dict(track=k,xyz=p.tolist()) for k,p in enumerate(points)])
        result,receipt=refine(e,pose,max_evaluations=100,calibrate_panoramas=True)
        self.assertTrue(receipt['applied']);self.assertLess(receipt['after']['holdout_median_pixels'],receipt['before']['holdout_median_pixels']*.15)
        self.assertIn('camera_models',result)

    def test_joint_fit_improves_unused_observations_and_preserves_gauge(self):
        rng=np.random.default_rng(332);points=rng.uniform([-2,-1,5],[2,1,10],(60,3))
        centers=np.array([[0.,0,0],[1.,0,0],[.3,.3,.1],[1.3,.2,-.1]])
        tracks=[];poses={};shape=[600,800];f=500.
        for k,p in enumerate(points):
            obs=[]
            for i,c in enumerate(centers):
                q=p-c;obs.append(dict(capture=i,site=k*4+i,xy=(q[:2]/q[2]*f+[399.5,299.5]).tolist()))
            tracks.append(dict(id=k,observations=obs))
        for i,c in enumerate(centers):
            r=np.eye(3) if i==0 else Rotation.from_rotvec(rng.normal(size=3)*.008).as_matrix()
            t=-c if i==0 else -c+rng.normal(size=3)*.02
            if i==1:t=t/np.linalg.norm(t)
            poses[str(i)]=dict(rotation=r.tolist(),translation=t.tolist(),center=(-r.T@t).tolist(),role='seed_gauge' if i==0 else 'heldout_pose')
        e=dict(cameras=[dict(shape=shape,focal=f,projection='perspective') for _ in centers],tracks=tracks)
        pose=dict(accepted=True,seed=[0,1],poses=poses,points=[dict(track=k,xyz=(p+rng.normal(size=3)*.04).tolist()) for k,p in enumerate(points)])
        result,receipt=refine(e,pose,max_evaluations=80)
        self.assertTrue(receipt['applied']);self.assertLess(receipt['after']['holdout_median_degrees'],receipt['before']['holdout_median_degrees']*.05)
        np.testing.assert_array_equal(result['poses']['0']['rotation'],np.eye(3))
        np.testing.assert_array_equal(result['poses']['0']['translation'],[0.,0,0])
        self.assertAlmostEqual(np.linalg.norm(result['poses']['1']['translation']),1.)
        # Held-out pixel values cannot steer optimizer trajectories or training
        # residuals. They can only accept/reject its one completed candidate.
        poisoned=copy.deepcopy(e)
        for track in poisoned['tracks']:
            for o in track['observations']:
                if o['site']%7==0:o['xy']=[20.,20.]
        _,bad=refine(poisoned,pose,max_evaluations=80)
        self.assertEqual(bad['evaluations'],receipt['evaluations'])
        self.assertAlmostEqual(bad['after']['train_median_degrees'],receipt['after']['train_median_degrees'],places=9)


if __name__=='__main__':unittest.main()
