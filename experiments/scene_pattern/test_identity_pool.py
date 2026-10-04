import unittest
import numpy as np
from scipy.ndimage import gaussian_filter, shift
from .identity_pool import canonicalize, physical_site, retain, reverify


class IdentityPoolTests(unittest.TestCase):
    def test_physical_keys_and_transport_do_not_depend_on_input_lattice(self):
        A=np.array([[1.1,.2],[-.1,.9]])
        p=np.array([30.4,40.2]); q=np.array([60.,50.])
        a=canonicalize(dict(p=p,q=q,A=A,origins=['one']))
        b=canonicalize(dict(p=[30,40],q=q+A@(np.array([30,40])-p),A=A,origins=['two']))
        self.assertEqual(a['site'],b['site'])
        np.testing.assert_allclose(a['q'],b['q'])
        self.assertEqual(len({physical_site([x,y]) for x in range(0,100,2) for y in range(0,100,2)}),2500)

    def test_competing_identity_kept_equivalent_origins_combined_without_average(self):
        rows=[dict(p=np.array([30,40]),q=np.array(q),A=np.eye(2),site=100,
                   training=t,origins=[name])
              for q,t,name in [([60,40],.95,'dense'),([60.2,40],.9,'ring'),([90,40],.85,'other')]]
        bank=retain(rows)
        self.assertEqual(len(bank),2)
        self.assertEqual(bank[0]['origins'],['dense','ring'])
        np.testing.assert_array_equal(bank[0]['q'],[60,40])
        np.testing.assert_array_equal(bank[1]['q'],[90,40])

    def test_original_image_verification_rejects_false_proposal(self):
        rng=np.random.default_rng(547)
        a=gaussian_filter(rng.normal(size=(100,140,3)),(1,1,0))*.2+[.5,.02,.01]
        b=shift(a,(2,4,0),order=1,mode='reflect')
        proposals=[dict(p=[50,50],q=q,A=np.eye(2),origins=[name])
                   for q,name in [([54,52],'true'),([100,50],'false')]]
        rows,checks=reverify(a,b,proposals)
        self.assertEqual([r['passed'] for r in checks],[True,False])
        self.assertEqual(len(rows),1)
        np.testing.assert_allclose(rows[0]['q'],[54,52],atol=.1)


if __name__=='__main__':unittest.main()
