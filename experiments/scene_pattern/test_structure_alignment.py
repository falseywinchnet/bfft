import unittest
import numpy as np
from .structure_alignment import compose_field,jacobian_min

class StructureAlignmentTests(unittest.TestCase):
    def test_composition_matches_analytic_affine(self):
        y,x=np.indices((60,70));q=np.stack((x,y),-1).astype(float)
        a=np.array([[.03,-.02],[.04,.01]]);old=q@a.T+[1,-2]
        delta=np.broadcast_to([3.,-1.],q.shape)
        expected=delta+(q+delta)@a.T+[1,-2]
        np.testing.assert_allclose(compose_field(old,delta)[5:-5,5:-5],expected[5:-5,5:-5],atol=1e-12)
    def test_identity_and_fold_detection(self):
        y,x=np.indices((30,30));zero=np.zeros((30,30,2));np.testing.assert_array_equal(compose_field(zero,zero),zero)
        self.assertEqual(jacobian_min(zero),1.)
        zero[...,0]=-2*x;self.assertEqual(jacobian_min(zero),-1.)
    def test_broad_color_features_recover_known_displacement(self):
        from scipy.ndimage import gaussian_filter,map_coordinates
        from .floor_affine import fit_floor,affine_points
        rng=np.random.default_rng(17);shape=(100,100)
        a=gaussian_filter(rng.random((*shape,3)),(2,2,0))
        y,x=np.indices(shape);shift=np.array([5.,-3.])
        b=np.stack([map_coordinates(a[...,k],[y-shift[1],x-shift[0]],order=1,mode='reflect') for k in range(3)],-1)
        polygon=[(25,25),(75,25),(75,75),(25,75)]
        p,c,r=fit_floor(a,b,np.ones(shape),np.ones(shape),polygon,structural=True)
        q=np.array([[30,30],[70,30],[70,70],[30,70.]])
        self.assertLess(np.max(np.linalg.norm(affine_points(q,p,c)-q-shift,axis=1)),.5)
        self.assertLess(r['after_test'],r['before_test']*.15)
    def test_polygon_survives_saved_fit_json(self):
        import json
        from .structure_alignment import polygon_mask
        polygon=[(4,4),(20,4),(20,20),(4,20)]
        np.testing.assert_array_equal(polygon_mask((30,30),polygon),polygon_mask((30,30),json.loads(json.dumps(polygon))))
