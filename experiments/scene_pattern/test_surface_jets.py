import copy
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from .surface_jets import fit_jet,project,spatial_holdout


class JetTests(unittest.TestCase):
    def test_spatial_holdout_does_not_starve_later_cameras(self):
        counts={1:0,2:0,3:0}
        for y in range(14,100,8):
            for x in range(14,100,8):
                obs=[dict(capture=i,xy=[x,y],affine=np.eye(2).tolist()) for i in range(4)]
                first=spatial_holdout(obs);self.assertEqual(first,spatial_holdout(obs));self.assertEqual(first[0]['capture'],0)
                counts[first[-1]['capture']]+=1
        self.assertTrue(all(n>20 for n in counts.values()))
    def example(self):
        camera=dict(shape=[600,800],focal=500.,projection='perspective')
        cameras=[camera]*4;poses={}
        for i,c in enumerate(([0.,0,0],[1.,.1,0],[.2,.5,.1],[-.7,.2,-.1])):
            r=Rotation.from_rotvec(np.array([.01,-.025,.005])*i).as_matrix();poses[i]=dict(rotation=r,translation=-r@c,center=np.array(c))
        normal=np.array([.2,-.1,1.]);p=np.array([450.,330.])
        def surface(q):
            ray=np.r_[(q-[399.5,299.5])/500.,1.];return ray*6/(ray@normal)
        x=surface(p);eps=.001;tangent=np.array([(surface(p+np.eye(2)[i]*eps)-surface(p-np.eye(2)[i]*eps))/(2*eps) for i in range(2)])
        observations=[]
        for i in range(4):
            xy=project([x],camera,poses[i])[0]
            j=np.stack([(project([surface(p+np.eye(2)[k]*eps)],camera,poses[i])[0]-project([surface(p-np.eye(2)[k]*eps)],camera,poses[i])[0])/(2*eps) for k in range(2)],axis=1)
            observations.append(dict(capture=i,xy=xy.tolist(),affine=j.tolist()))
        return observations,cameras,poses,x,tangent

    def test_recovers_local_surface_position_and_tangents(self):
        obs,cameras,poses,x,tangent=self.example();result=fit_jet(obs,cameras,poses,x+[.1,.05,.2])
        self.assertTrue(result['accepted']);np.testing.assert_allclose(result['center'],x,atol=.001)
        np.testing.assert_allclose(result['tangents'],tangent,atol=.00002)
        self.assertLess(result['validation']['maximum_pixels'],.02)

    def test_wrong_unused_distortion_rejects_without_steering_surface(self):
        obs,cameras,poses,x,_=self.example();good=fit_jet(obs,cameras,poses,x)
        bad=copy.deepcopy(obs);bad[-1]['affine']=(np.asarray(bad[-1]['affine'])*3).tolist()
        result=fit_jet(bad,cameras,poses,x)
        self.assertFalse(result['accepted']);np.testing.assert_array_equal(result['center'],good['center'])
        np.testing.assert_array_equal(result['tangents'],good['tangents'])

    def test_same_origin_cannot_create_depth(self):
        obs,cameras,poses,x,_=self.example()
        for i in poses:poses[i]=dict(rotation=np.eye(3),translation=np.zeros(3),center=np.zeros(3));obs[i]['xy']=obs[0]['xy'];obs[i]['affine']=np.eye(2).tolist()
        result=fit_jet(obs,cameras,poses,x)
        self.assertFalse(result['accepted']);self.assertEqual(result['reason'],'unresolved_depth_or_tangent')


if __name__=='__main__':unittest.main()
