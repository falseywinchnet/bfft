"""Synthetic calibration and exclusion checks for differential reconstruction."""
import copy
import unittest
import numpy as np
from .surface_jets import fit_jet, project, image_rays
from .jet_bundle import refine_jets


class JetCalibrationTests(unittest.TestCase):
    def test_panorama_calibration_predicts_unused_neighborhoods(self):
        cameras=[dict(shape=[500,1200],focal=260.,projection='cylindrical') for _ in range(4)]
        truth=[];poses={}
        for i,c in enumerate(([0.,0,0],[1.,0,0],[.3,.5,.1],[1.2,-.4,.2])):
            truth.append(dict(cameras[i],focal_x=260*(1+.015*i),focal_y=260*(1-.01*i),principal_point=[599.5,249.5+2*i]))
            poses[i]=dict(rotation=np.eye(3),translation=-np.array(c),center=np.array(c))
        rows=[];points=[]
        for k in range(48):
            xy=np.array([435+(k%8)*45.,190+(k//8)*23.]);normal=np.array([.3*np.sin(k),.2*np.cos(k),1.]);distance=5+.06*k
            def surface(q):
                ray=image_rays(np.array([q]),truth[0])[0];return ray*distance/(normal@ray)
            x=surface(xy);hold=1+k%3;obs=[]
            for i in [0]+[j for j in (1,2,3) if j!=hold]+[hold]:
                eps=.001
                jac=np.stack([(project([surface(xy+np.eye(2)[axis]*eps)],truth[i],poses[i])[0]-project([surface(xy-np.eye(2)[axis]*eps)],truth[i],poses[i])[0])/(2*eps) for axis in range(2)],axis=1)
                obs.append(dict(capture=i,xy=project([x],truth[i],poses[i])[0].tolist(),affine=jac.tolist()))
            row=fit_jet(obs,truth,poses,x);rows.append(dict(track=k,**row));points.append(dict(track=k,xyz=x.tolist()))
        pose=dict(seed=[0,1],poses={str(i):{k:v.tolist() for k,v in p.items()} for i,p in poses.items()},points=points)
        packet=dict(results=rows)
        result,receipt,candidate=refine_jets(packet,pose,cameras,max_evaluations=250,calibrate_panoramas=True)
        self.assertTrue(receipt['applied']);self.assertLess(receipt['after']['holdout_median_pixels'],receipt['before']['holdout_median_pixels']*.3)
        self.assertIn('camera_models',result)
        np.testing.assert_array_equal(result['poses']['0']['rotation'],np.eye(3))
        self.assertAlmostEqual(np.linalg.norm(result['poses']['1']['translation']),1.)
        poisoned=copy.deepcopy(packet)
        for row in poisoned['results']:
            if 'observations' in row:row['observations'][-1]['affine'][0][0]+=1.
        _,bad,other=refine_jets(poisoned,pose,cameras,max_evaluations=250,calibrate_panoramas=True)
        self.assertEqual(receipt['evaluations'],bad['evaluations'])
        self.assertEqual(candidate['camera_models'],other['camera_models'])
        for i in candidate['poses']:np.testing.assert_array_equal(candidate['poses'][i]['translation'],other['poses'][i]['translation'])


if __name__=='__main__':unittest.main()
