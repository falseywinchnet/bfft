import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from .multiview_geometry import bearings,unit,rotation_fit,rotation_errors,essential_fit,essential_errors,essential_poses,hypotheses,triangulate

from .scene_evidence import cycle_components

class MultiViewTests(unittest.TestCase):
    def scene(self,n=100):
        rng=np.random.default_rng(3);x=rng.uniform([-3,-2,5],[3,2,14],(n,3));r=Rotation.from_rotvec([.04,.18,-.03]).as_matrix();t=np.array([-1.,.05,.1]);return x,r,t,unit(x),unit(x@r.T+t)
    def test_bridge_does_not_inherit_cycle_support(self):
        self.assertEqual(cycle_components({0,1,2,3},{(0,1),(1,2),(0,2),(2,3)}),[[0,1,2]])
    def test_each_projection_produces_unit_bearings(self):
        for model in ('perspective','cylindrical','equirectangular'):
            d=bearings([[0,0],[400,100],[799,199]],[200,800],160,model);np.testing.assert_allclose(np.linalg.norm(d,axis=1),1)
        self.assertLess(bearings([[0,100]],[200,800],160,'cylindrical')[0,2],0)
    def test_crop_and_anisotropic_resize_preserve_panorama_bearings(self):
        xy=np.array([[30.,20.],[380,80],[740,150]])
        original=bearings(xy,[200,800],160,'cylindrical')
        crop=np.array([17.,11.]);scale=np.array([1.3,.8])
        transformed=bearings((xy-crop)*scale,[142,998],160,'cylindrical',focal_x=160*scale[0],focal_y=160*scale[1],principal_point=(np.array([399.5,99.5])-crop)*scale)
        np.testing.assert_allclose(transformed,original,atol=1e-12)
    def test_distinct_origins_recover_pose_and_positive_depth(self):
        x,r,t,a,b=self.scene();e=essential_fit(a,b);self.assertLess(essential_errors(a,b,e).max(),1e-6)
        p=essential_poses(e,a,b)[0];self.assertGreater(p['positive_fraction'],.99);np.testing.assert_allclose(p['rotation'],r,atol=1e-6);np.testing.assert_allclose(p['translation'],unit(t),atol=1e-6)
        np.testing.assert_allclose(p['points'],x/np.linalg.norm(t),atol=1e-5)
    def test_rotation_cannot_explain_translation_at_multiple_depths(self):
        _,r,t,a,b=self.scene();fit=rotation_fit(a,b);self.assertGreater(np.median(rotation_errors(a,b,fit)),.01)
        np.testing.assert_allclose(rotation_fit(a,a@r.T),r,atol=1e-8)
    def test_near_points_have_more_parallax_for_same_baseline(self):
        a=unit(np.array([[1,0,4],[3,0,12.]]));r=np.eye(3);t=np.array([-1.,0,0]);x=np.array([[1,0,4],[3,0,12.]]);b=unit(x+t);_,_,_,_,angles=triangulate(a,b,r,t);self.assertGreater(angles[0],angles[1]*2)
    def test_alternatives_do_not_multiply_support(self):
        _,_,_,a,b=self.scene(70);rng=np.random.default_rng(71);aa=np.repeat(a,2,axis=0);bb=np.empty_like(aa);bb[::2]=b;bb[1::2]=unit(rng.normal(size=b.shape));sa=np.repeat(np.arange(70),2);sb=np.ravel(np.c_[np.arange(70),np.arange(70)+100])
        result=hypotheses(aa,bb,sa,sb,trials=180);e=result['essential'];self.assertGreater(e['train_count']+e['heldout_count'],50);self.assertLessEqual(e['train_count']+e['heldout_count'],70);self.assertGreater(e['heldout_count'],5)

class PoseAndTrackTests(unittest.TestCase):
    def test_pose_recovery_uses_separate_origin_and_unused_tracks(self):
        from .scene_pose import fit_pose
        rng=np.random.default_rng(83);points=rng.uniform([-3,-2,5],[3,2,13],(100,3));r=Rotation.from_rotvec([.1,-.2,.03]).as_matrix();t=np.array([1.5,.2,-.1]);b=unit(points@r.T+t)
        result=fit_pose(points,b,np.arange(len(points)),trials=80)
        self.assertTrue(result['accepted']);np.testing.assert_allclose(result['center'],-r.T@t,atol=1e-6);self.assertGreater(result['holdout'],10)
    def test_holdout_failure_cannot_be_hidden_by_training_fit(self):
        from .scene_pose import fit_pose
        rng=np.random.default_rng(84);points=rng.uniform([-3,-2,5],[3,2,13],(100,3));b=unit(points+[1.,0,0]);ids=np.arange(100);b[ids%7==0]=unit(rng.normal(size=(sum(ids%7==0),3)))
        result=fit_pose(points,b,ids,trials=80)
        self.assertIsNotNone(result);self.assertFalse(result['accepted']);self.assertGreater(result['train'],70)
    def test_tracks_keep_conflicts_and_multiple_panorama_associations(self):
        from .scene_evidence import build_tracks,associations
        features=[dict(sites=np.array([[10.,10.],[20.,20.]])) for _ in range(4)]
        def edge(a,b,sa=0,sb=0):
            return dict(a=a,b=b,status='translation_supported',selected_model='essential',candidates=[[sa,sb,.01,0]],models=dict(essential=dict(train_inliers=[0],heldout_inliers=[],heldout_count=10)))
        edges=[edge(0,1),edge(1,2),edge(0,2),edge(2,3),edge(0,2,1,0)]
        tracks,conflicts,_=build_tracks(edges,features);self.assertGreater(len(conflicts),0)
        records=[dict(panoramic=True),dict(panoramic=True),dict(panoramic=False),dict(panoramic=False)];a=associations(records,tracks)
        self.assertEqual(len(a[0]['candidates']),2);self.assertEqual(sum(x['cycle_supported_tracks'] for x in a[1]['candidates']),0)



class ReconstructionTests(unittest.TestCase):
    def test_three_camera_prediction_and_poisoned_holdout(self):
        from .scene_pose import reconstruct,skew
        rng=np.random.default_rng(18);points=rng.uniform([-2,-1,4],[2,1,11],(90,3));shape=[600,800];f=500.
        centers=[np.array([0.,0,0]),np.array([1.,0,0]),np.array([.3,.3,0])]
        tracks=[]
        for tid,p in enumerate(points):
            observations=[]
            for i,c in enumerate(centers):
                q=p-c;xy=q[:2]/q[2]*f+[399.5,299.5];observations.append(dict(capture=i,site=tid,xy=xy.tolist()))
            tracks.append(dict(id=tid,observations=observations,cycle_components=[[0,1,2]],independent_cycles=1))
        pose=dict(rotation=np.eye(3).tolist(),translation=[-1.,0,0])
        model=dict(heldout_count=12,poses=[pose]);edge=dict(a=0,b=1,status='translation_supported',selected_model='essential',models=dict(essential=model),candidates=[[i,i,.01,0] for i in range(90)])
        e=dict(cameras=[dict(shape=shape,focal=f,projection='perspective') for _ in centers],records=[dict(panoramic=True) for _ in centers],tracks=tracks,edges=[edge])
        result=reconstruct(e);self.assertTrue(result['accepted']);np.testing.assert_allclose(result['poses']['2']['center'],centers[2],atol=1e-5)
        from .scene_promotion import audit
        self.assertEqual(audit(e,result)['promotion_counts']['third_view_consistent'],90)
        # Stage-two photo seeds use the same geometry gates; the default stays
        # panorama-first, and enabling photos cannot rescue poisoned holdouts.
        for record in e['records']:record['panoramic']=False
        self.assertFalse(reconstruct(e)['accepted'])
        self.assertTrue(reconstruct(e,allow_photo_seeds=True)['accepted'])
        # The two-view seed and all training observations stay exact.
        for track in tracks:
            if track['id']%7==0:track['observations'][2]['xy']=(rng.uniform([0,0],[799,599])).tolist()
        poisoned=reconstruct(e,allow_photo_seeds=True);self.assertFalse(poisoned['accepted'])



class PlanarCameraTests(unittest.TestCase):
    def test_wall_points_determine_camera_without_full_rank_3d_dlt(self):
        from .scene_pose import planar_pose,pose_errors,fit_pose
        rng=np.random.default_rng(67);points=np.c_[rng.uniform(-2,2,100),rng.uniform(-1,1,100),np.full(100,6.)];r=Rotation.from_rotvec([.1,.3,-.07]).as_matrix();t=np.array([1.2,.4,-.3]);b=unit(points@r.T+t)
        choices=planar_pose(points,b);best=min(choices,key=lambda p:np.mean(pose_errors(points,b,*p)));np.testing.assert_allclose(best[0],r,atol=1e-8);np.testing.assert_allclose(best[1],t,atol=1e-8)
        result=fit_pose(points,b,np.arange(100),trials=40);self.assertTrue(result['accepted']);self.assertLess(result['holdout_median_degrees'],1e-6)

if __name__=='__main__':unittest.main()
