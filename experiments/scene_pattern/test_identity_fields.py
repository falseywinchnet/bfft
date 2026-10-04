import unittest
import numpy as np
from scipy import ndimage
from .identity_fields import grow_images,distinct

class IdentityFieldTests(unittest.TestCase):
    def test_conflicting_targets_are_not_averaged(self):
        rows=[dict(p=np.array([20.,20]),q=np.array(q),A=np.eye(2),training=s) for q,s in [([30,20],.99),([30.1,20],.9),([55,20],.95)]]
        kept=distinct(rows);self.assertEqual(len(kept),2);np.testing.assert_array_equal(kept[0]['q'],[30,20]);np.testing.assert_array_equal(kept[1]['q'],[55,20])
    def test_image_growth_recovers_neighbors_and_rejects_false_seed(self):
        rng=np.random.default_rng(63);a=ndimage.gaussian_filter(rng.normal(size=(100,120,3)),(1.1,1.1,0))*.2+[.5,.03,.02]
        b=ndimage.shift(a,(2,4,0),order=1,mode='reflect');p=np.array([52.,52.]);seed=dict(p=p,q=p+[4,2],A=np.eye(2),training=.99,root=0)
        rows,history=grow_images(a,b,[seed],rounds=2);self.assertGreater(len(rows),15)
        for r in rows:np.testing.assert_allclose(r['q']-r['p'],[4,2],atol=.25)
        wrong=ndimage.gaussian_filter(rng.normal(size=a.shape),(1.1,1.1,0))*.2+[.5,.03,.02]
        rows,_=grow_images(a,wrong,[seed],rounds=1);self.assertEqual(rows,[])

if __name__=='__main__':unittest.main()
