import copy
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from .surface_jets import fit_jet,project
from .jet_bundle import refine_jets


class JetBundleTests(unittest.TestCase):
    def run_surface_case(self,coherent=False,retain_points=False):
        rng=np.random.default_rng(164);camera=dict(shape=[600,800],focal=500.,projection='perspective');cameras=[camera]*4
        centers=[np.array(c) for c in ([0.,0,0],[1.,0,0],[.3,.4,0],[1.2,-.3,.2])];truth={};initial={}
        for i,c in enumerate(centers):
            truth[i]=dict(rotation=np.eye(3),translation=-c,center=c)
            r=np.eye(3) if i==0 else Rotation.from_rotvec(rng.normal(size=3)*.001).as_matrix()
            t=-c if i==0 else -c+rng.normal(size=3)*.003
            if i==1:t=t/np.linalg.norm(t)
            initial[i]=dict(rotation=r,translation=t,center=-r.T@t)
        jets=[];points=[]
        for k in range(24):
            spacing=6 if coherent else 20
            p=np.array([340.+(k%6)*spacing,250.+(k//6)*spacing]);normal=np.array([.1,.05,1.]) if coherent else np.array([.2*np.sin(k),.1*np.cos(k),1.]);d=6 if coherent else 6+.02*k
            def surface(q):
                ray=np.r_[(q-[399.5,299.5])/500.,1.];return ray*d/(normal@ray)
            x=surface(p);obs=[];hold=1+k%3;order=[0]+[i for i in (1,2,3) if i!=hold]+[hold]
            for i in order:
                eps=.001;j=np.stack([(project([surface(p+np.eye(2)[v]*eps)],camera,truth[i])[0]-project([surface(p-np.eye(2)[v]*eps)],camera,truth[i])[0])/(2*eps) for v in range(2)],axis=1)
                obs.append(dict(capture=i,xy=project([x],camera,truth[i])[0].tolist(),affine=j.tolist()))
            result=fit_jet(obs,cameras,initial,x);jets.append(dict(track=k,**result));points.append(dict(track=k,xyz=x.tolist()))
        pose=dict(seed=[0,1],poses={str(i):{k:v.tolist() for k,v in p.items()} for i,p in initial.items()},points=points)
        evidence=None
        if retain_points:
            tracks=[dict(id=j['track'],observations=[dict(capture=o['capture'],xy=o['xy'],site=1000+j['track']) for o in j['observations']]) for j in jets]
            for k in range(24,40):
                xyz=np.array([-.7+(k%4)*.4,-.5+(k//4-6)*.3,5.+.15*(k%5)])
                points.append(dict(track=k,xyz=xyz.tolist()))
                tracks.append(dict(id=k,observations=[dict(capture=i,xy=project([xyz],camera,truth[i])[0].tolist(),site=k*5+i) for i in range(4)]))
            pose['points']=points;evidence=dict(tracks=tracks)
        packet=dict(results=jets);result,receipt,candidate=refine_jets(packet,pose,cameras,max_evaluations=200,coherent_surfaces=coherent,protect_cameras=coherent,point_evidence=evidence)
        self.assertTrue(receipt['applied']);self.assertLess(receipt['after']['holdout_median_pixels'],receipt['before']['holdout_median_pixels']*.2)
        np.testing.assert_array_equal(result['poses']['0']['rotation'],np.eye(3));self.assertAlmostEqual(np.linalg.norm(result['poses']['1']['translation']),1.)
        if retain_points:self.assertGreater(receipt['point_support']['points'],10)
        if coherent:self.assertGreater(receipt['coherence_before']['edges'],10)
        poisoned=copy.deepcopy(packet)
        for row in poisoned['results']:row['observations'][-1]['xy'][0]+=20
        _,bad,rejected=refine_jets(poisoned,pose,cameras,max_evaluations=200,coherent_surfaces=coherent,protect_cameras=coherent,point_evidence=evidence)
        self.assertEqual(bad['evaluations'],receipt['evaluations'])
        for i in pose['poses']:np.testing.assert_array_equal(candidate['poses'][i]['translation'],rejected['poses'][i]['translation'])

    def test_joint_surface_constraint_predicts_withheld_views(self):self.run_surface_case()

    def test_coherence_optimizer_predicts_withheld_views_without_reading_them(self):self.run_surface_case(coherent=True)

    def test_surface_fit_retains_remaining_point_support(self):self.run_surface_case(coherent=True,retain_points=True)


if __name__=='__main__':unittest.main()
