import unittest,copy
import numpy as np
from scipy.spatial.transform import Rotation
from .identity_joint import fit
from .multiview_geometry import unit


def fixture(n=90):
    rng=np.random.default_rng(5);camera=dict(shape=[600,800],focal=400.,projection='perspective');poses=[dict(R=Rotation.from_rotvec([.05,.12,-.04]).as_matrix(),t=unit(np.array([-1.,.05,.03]))),dict(R=Rotation.from_rotvec([-.02,.25,.05]).as_matrix(),t=unit(np.array([-.3,-.8,.04]))*1.6)];fields=[[],[]]
    for site in range(n):
        X=rng.uniform([-3,-2,5],[3,2,12]);normal=unit(np.r_[rng.uniform(-.2,.2,2),1.]);p=X[:2]/X[2]*400+[399.5,299.5]
        for field,pose in zip(fields,poses):
            H=pose['R']+np.outer(pose['t'],normal)/(normal@X)
            def mapped(pixel):
                q=H@np.r_[(pixel-[399.5,299.5])/400,1.];return q[:2]/q[2]*400+[399.5,299.5]
            eps=.001;A=np.column_stack([(mapped(p+np.eye(2)[j]*eps)-mapped(p-np.eye(2)[j]*eps))/(2*eps) for j in range(2)]);q=mapped(p)
            field.append(dict(p=p,q=q,A=A,training=.96,site=site));field.append(dict(p=p,q=q+[25.,17.],A=A,training=.98,site=site))
    return fields,[camera]*3,poses

class JointIdentityTests(unittest.TestCase):
    def test_joint_camera_depth_and_identity_fit_predicts_reserved_regions(self):
        fields,cameras,poses=fixture();ab=copy.deepcopy(poses[0]);ac=copy.deepcopy(poses[1]);ab['R']=Rotation.from_rotvec([.003,-.002,0]).as_matrix()@ab['R'];ac['t']*=1.03
        result=fit(*fields,cameras,ab,ac,maximum=70,rounds=2)
        self.assertTrue(result['accepted'],result['checks']);self.assertGreater(result['check_count'],10)
        np.testing.assert_allclose(result['poses'][2]['center'],-poses[1]['R'].T@poses[1]['t'],atol=.03)
        self.assertTrue(all(s['selected_b']==0 and s['selected_c']==0 for s in result['surfaces']))

    def test_held_third_view_cannot_steer_joint_cameras(self):
        from .identity_geometry import roles
        fields,cameras,poses=fixture();baseline=fit(*fields,cameras,*poses,maximum=40,rounds=2)
        changed=copy.deepcopy(fields);training,held,_=roles(changed[1],0)
        for i in np.flatnonzero(~training):changed[1][i]['q']=changed[1][i]['q']+[70.,-50.]
        poisoned=fit(*changed,cameras,*poses,maximum=40,rounds=2)
        for a,b in zip(baseline['poses'],poisoned['poses']):
            np.testing.assert_allclose(a['rotation'],b['rotation'],atol=1e-12)
            np.testing.assert_allclose(a['translation'],b['translation'],atol=1e-12)
        self.assertFalse(poisoned['accepted'])

if __name__=='__main__':unittest.main()
