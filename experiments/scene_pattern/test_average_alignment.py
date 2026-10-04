import unittest
import itertools
import numpy as np
from scipy.ndimage import gaussian_filter,shift
from .average_alignment import synchronize,center_edge

class GraphAlignmentTests(unittest.TestCase):
    def test_all_pair_gauge_and_cycle_closure(self):
        expected=np.array([[2.,-1.],[-3.,4.],[1.,-3.]])
        edges=list(itertools.combinations(range(3),2));values=np.asarray([expected[j]-expected[i] for i,j in edges])
        coordinates,residual,groups=synchronize(3,edges,values,np.ones(3))
        np.testing.assert_allclose(coordinates,expected,atol=1e-12);np.testing.assert_allclose(residual,0,atol=1e-12)
        self.assertEqual(groups[0]['cycle_rank'],1)
    def test_disconnected_components_have_separate_gauges(self):
        c,r,g=synchronize(5,[(0,1),(2,3)],[[4,2],[-2,6]],[1,1])
        np.testing.assert_allclose(c,[[ -2,-1],[2,1],[1,-3],[-1,3],[0,0]],atol=1e-12)
        self.assertTrue(all(x['cycle_rank']==0 for x in g))
    def test_permutation_only_relabels_coordinates(self):
        truth=np.array([[1.,2.],[3.,-2.],[-4.,0.]])
        order=np.array([2,0,1]);edges=list(itertools.combinations(range(3),2))
        c,_,_=synchronize(3,edges,[truth[order[j]]-truth[order[i]] for i,j in edges],[1]*3)
        np.testing.assert_allclose(c,truth[order],atol=1e-12)
    def test_fractional_observation_displacement_sign(self):
        a=gaussian_filter(np.random.default_rng(21).random((64,64)),.8)
        b=shift(a,(-1.3,2.4),order=3,mode='reflect')
        result=center_edge(a,b)
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result['shift'],[2.4,-1.3],atol=.2)
    def test_inconsistent_triangle_has_nonzero_residual(self):
        _,r,_=synchronize(3,[(0,1),(1,2),(0,2)],[[1,0],[1,0],[4,0]],[1,1,1])
        self.assertGreater(np.linalg.norm(r),.1)
