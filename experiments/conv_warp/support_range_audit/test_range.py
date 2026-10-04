import unittest
import numpy as np
from experiments.conv_warp.bounded_sharpening.study import half_controls
from .study import constraints,evaluation

class RangeTests(unittest.TestCase):
 def test_bounded_polynomial_can_have_outside_controls(self):
  # 4t(1-t), degree-elevated to quintic; exact range [0,1].
  p=np.array([0.,.8,1.2,1.2,.8,0.]);w=np.array([1,5,10,10,5,1])/32
  self.assertAlmostEqual(p@w,1.)
  self.assertAlmostEqual(np.clip(p,0,1)@w,.875)
  self.assertLessEqual((half_controls(5)@p).max(),1.+2e-16)
  self.assertGreaterEqual((half_controls(5)@p).min(),0.)
 def test_legitimate_peak_exceeds_all_source_samples(self):
  f=lambda x:.9-.1*(x-2.5)**2
  source=f(np.arange(6));self.assertAlmostEqual(source.max(),.875)
  self.assertAlmostEqual(f(2.5),.9)
  sign=np.sign(np.diff(source));sign=sign[sign!=0]
  self.assertEqual(np.sum(sign[1:]!=sign[:-1]),1)
 def test_second_order_extremum_obstruction(self):
  kappa=3.
  for h in (1.,.5,.25,.125):
   f=lambda x:1.-kappa*x*x/2
   self.assertAlmostEqual(f(0)-f(h/2),kappa*h*h/8)
 def test_refined_certificate_reconstructs_constant(self):
  source=np.full((5,5),.4);listed,A,R,lo,hi,_,_=constraints(source)
  P=np.full((21,21),.4)
  np.testing.assert_allclose(R@P.ravel(),.4,atol=2e-16,rtol=0)
  self.assertGreaterEqual(np.min(R@P.ravel()-lo),-2e-16)
  np.testing.assert_allclose(evaluation(P,17),.4,atol=3e-16,rtol=0)

if __name__=='__main__':unittest.main()
