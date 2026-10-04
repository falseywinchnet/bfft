import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from .affine_camera_geometry import measurements,essential_linear,homography_linear,homography_poses,epi_error,skew,refine_essential
from .multiview_geometry import essential_poses,unit


def fixture(n=18,plane=False):
    rng=np.random.default_rng(34);camera=dict(shape=[600,800],focal=400.,projection='perspective');R=Rotation.from_rotvec([.05,.14,-.025]).as_matrix();t=np.array([-1.,.04,.13]);rows=[]
    for i in range(n):
        p=np.r_[rng.uniform([-2.5,-1.7],[2.5,1.7]),7. if plane else rng.uniform(5,12)]
        # A local plane can have its own depth and slope; all share camera motion.
        normal=np.array([0.,0,1.]) if plane else unit(np.r_[rng.uniform(-.3,.3,2),1.]);H=R+np.outer(t,normal)/(normal@p)
        xy=p[:2]/p[2]*400+[399.5,299.5]
        def mapped(x):
            ray=np.r_[(x-[399.5,299.5])/400,1.];q=H@ray;return q[:2]/q[2]*400+[399.5,299.5]
        eps=.001;A=np.column_stack([(mapped(xy+np.eye(2)[j]*eps)-mapped(xy-np.eye(2)[j]*eps))/(2*eps) for j in range(2)])
        rows.append(dict(p=xy,q=mapped(xy),A=A,training=.99,validation=.99,site=i,root=i))
    return rows,[camera,camera],R,t

class AffineCameraGeometryTests(unittest.TestCase):
    def test_three_affine_regions_recover_general_relative_pose(self):
        rows,cameras,R,t=fixture();m=measurements(rows,cameras);E=essential_linear(m,[0,1,2]);self.assertLess(epi_error(m,E).max(),1e-5)
        pose=essential_poses(E,m['a'],m['b'])[0];np.testing.assert_allclose(pose['rotation'],R,atol=1e-6);np.testing.assert_allclose(pose['translation'],unit(t),atol=1e-6)
    def test_cylindrical_ray_derivatives_preserve_relative_geometry(self):
        rows,cameras,R,t=fixture()
        def cylindrical(x):
            xx,yy=(x-[399.5,299.5])/400
            return np.array([400*np.arctan(xx)+399.5,400*yy/np.sqrt(1+xx*xx)+299.5])
        def jac(x):
            eps=.001;return np.column_stack([(cylindrical(x+np.eye(2)[j]*eps)-cylindrical(x-np.eye(2)[j]*eps))/(2*eps) for j in range(2)])
        for row in rows:
            row['A']=jac(row['q'])@row['A']@np.linalg.inv(jac(row['p']))
            row['p']=cylindrical(row['p']);row['q']=cylindrical(row['q'])
        cameras=[{**c,'projection':'cylindrical'} for c in cameras]
        m=measurements(rows,cameras);E=essential_linear(m,[0,1,2]);self.assertLess(epi_error(m,E).max(),1e-5)
        pose=essential_poses(E,m['a'],m['b'])[0];np.testing.assert_allclose(pose['rotation'],R,atol=1e-6)
    def test_planar_initialization_retains_pose_ambiguity(self):
        rows,cameras,R,t=fixture(10,plane=True);m=measurements(rows,cameras);H=homography_linear(m,[0,1,2]);poses=homography_poses(H)
        self.assertEqual(len(poses),4);errors=[np.linalg.norm(p['rotation']-R)+np.linalg.norm(p['translation']-unit(t)) for p in poses];self.assertLess(min(errors),1e-6)
        self.assertTrue(all(np.linalg.det(p['rotation'])>.999 for p in poses))
    def test_wrong_local_distortion_is_visible_even_with_correct_centers(self):
        rows,cameras,R,t=fixture();truth=measurements(rows,cameras);E=skew(unit(t))@R;self.assertLess(epi_error(truth,E).max(),1e-5)
        for row in rows:row['A']=row['A']+np.array([[.25,.2],[-.3,.1]])
        wrong=measurements(rows,cameras);self.assertGreater(np.median(epi_error(wrong,E)),.4)

if __name__=='__main__':unittest.main()
