import unittest
import numpy as np
from .paper_images import CASES,truth,basis,point,area,true_area,lanczos_weights,area_weights,flat_controls
from .study import atlas,evaluate_joint_atlas
from experiments.conv_synthetic_evidence import _scene

class PaperImageTests(unittest.TestCase):
 def test_original_paper_fields(self):
  y,x=np.mgrid[0:17,0:17]/16
  for k,p in CASES.items():np.testing.assert_allclose(truth(k,x,y),_scene(k,17,**p),atol=2e-15)
 def test_batched_point_matches_reference(self):
  P=np.random.default_rng(8).uniform(size=(81,81));q=np.linspace(0,16,7);y,x=np.meshgrid(q,q,indexing='ij')
  np.testing.assert_allclose(point(P,7),evaluate_joint_atlas(atlas(P),x,y),atol=8e-16)
 def test_exact_area_of_linear_field(self):
  y,x=np.mgrid[:81,:81]/5;P=x+2*y
  for side in (9,17,33,65):
   q=np.linspace(0,16,side);edge=np.r_[0,(q[:-1]+q[1:])/2,16];mid=(edge[1:]+edge[:-1])/2
   np.testing.assert_allclose(area(P,side),mid[None,:]+2*mid[:,None],atol=4e-14)
 def test_integrals_preserve_constant(self):
  for side in (9,33):
   np.testing.assert_allclose(area(np.ones((81,81)),side),1,atol=2e-15)
   np.testing.assert_allclose(area_weights(side,lanczos_weights).sum(axis=1),1,atol=2e-15)
 def test_truth_quadrature(self):
  for k in CASES:np.testing.assert_allclose(true_area(k,33,8),true_area(k,33,12),atol=1e-11,rtol=0)
 def test_full_degree_atlas_integral_under_quadrature_refinement(self):
  P=np.random.default_rng(21).normal(size=(81,81))
  for side in (65,129):
   W=area_weights(side,basis,8)
   np.testing.assert_allclose(area(P,side),W@flat_controls(P)@W.T,atol=2e-15,rtol=0)

if __name__=='__main__':unittest.main()
