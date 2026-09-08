import unittest
from fractions import Fraction as F
from math import comb
import numpy as np
from experiments.conv_admission_band.core import Operator
from experiments.conv_downstream.core import Downstream,NAMES,grams,weights,transformed

class DownstreamTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  try:cls.ops={k:Downstream(k) for k in NAMES}
  except Exception as e:
   if hasattr(e,'stderr'):print(e.stderr.decode())
   raise
 def test_gram(self):
  g,v=grams()
  for i in range(5):
   self.assertEqual(sum(g[i]),5)
   self.assertEqual(sum(v[i]),0)
   for j in range(5):
    self.assertEqual(g[i][j],F(25*comb(4,i)*comb(4,j),9*comb(8,i+j)))
    self.assertEqual(v[i][j],v[4-i][4-j])
  self.assertEqual(weights('slope'),[F(35,18),F(10,9),F(1),F(10,9),F(35,18)])
 def test_profile_mass_cardinality(self):
  rng=np.random.default_rng(81)
  for name,op in self.ops.items():
   for n in (5,9,33):
    x=rng.normal(size=(n,7)).astype(np.float32)
    c=op.profile(x)
    self.assertTrue(np.isfinite(c).all(),name)
    np.testing.assert_allclose(c.sum(axis=1),np.diff(x,axis=0),atol=2e-6,err_msg=name)
    np.testing.assert_allclose(op.axis(x,(n-1)*4+1,0)[::4],x,atol=2e-6,err_msg=name)
    np.testing.assert_allclose(op.axis(np.ones_like(x),min(n,7),0,True),1,atol=2e-6,err_msg=name)
 def test_weighted_kkt(self):
  rng=np.random.default_rng(14)
  for name,kind in [('projection_diag','slope'),('projection_value','value')]:
   op=self.ops[name];w=np.array(weights(kind),float)
   for t in range(1000):
    a=rng.normal(size=5).astype(np.float32);s=rng.choice([-1,1],5).astype(np.int8)
    feasible=s*rng.exponential(size=5);delta=np.float32(feasible.sum())
    c=op.fibre(a,s,delta).astype(float)
    self.assertTrue(np.isfinite(c).all(),(name,a,s,delta))
    self.assertLess(abs(c.sum()-delta),2e-6)
    self.assertGreaterEqual(min(c*s),-2e-6)
    active=abs(c)>1e-6;lam=np.mean(w[active]*(a[active]-c[active]))
    np.testing.assert_allclose(w[active]*(a[active]-c[active]),lam,atol=2e-5)
    self.assertTrue(np.all(s[~active]*(w[~active]*a[~active]-lam)<=2e-5))
 def test_primitive_same_average(self):
  rng=np.random.default_rng(8)
  for n in (5,17,65):
   x=rng.normal(size=(n,5)).astype(np.float32)
   for m in (1,2,3,min(n,9)):
    np.testing.assert_allclose(self.ops['basin_primitive'].axis(x,m,0,True),self.ops['v1'].axis(x,m,0,True),atol=2e-6)
 def test_full_ledger_direct_cost(self):
  direct=Operator('conv',source_transform=lambda _:transformed('ledger_full').replace('if(second_cost-best_cost<=256*DBL_EPSILON*energy)','if(1)'))
  rng=np.random.default_rng(95)
  for n in (5,9,33,129):
   x=rng.normal(size=(n,31)).astype(np.float32)
   np.testing.assert_array_equal(direct.profile(x),self.ops['ledger_full'].profile(x))
 def test_hat_integral_conservation(self):
  x=np.random.default_rng(4).normal(size=(65,7)).astype(np.float32)
  whole=self.ops['v1'].axis(x,1,0,True)[0]
  for name in ('v1','basin_primitive','basin_hat'):
   for m in (2,5,17,65):
    z=self.ops[name].axis(x,m,0,True)
    q=np.ones(m);q[[0,-1]]=.5;q/=m-1
    np.testing.assert_allclose(q@z,whole,atol=2e-6)
 def test_blend_transpose(self):
  x=np.random.default_rng(3).normal(size=(7,9)).astype(np.float32)
  for name in ('v1','blend_equal','blend_amplitude'):
   op=self.ops[name]
   np.testing.assert_allclose(op.synthesize(x,(25,33)),op.synthesize(x.T,(33,25)).T,atol=2e-6)

if __name__=='__main__':unittest.main()
