import unittest
import numpy as np
from scipy.ndimage import gaussian_filter,shift
from .registration_audit import field_evidence,contrast_error
from .structure_alignment import jacobian_min

class RegistrationAuditTests(unittest.TestCase):
    def test_smooth_translation_can_be_entirely_unsupported(self):
        delta=np.zeros((50,50,2));delta[...,0]=3
        measured=np.zeros((50,50),bool);measured[:10]=True
        inspected=np.zeros((50,50),bool);inspected[30:40]=True
        self.assertEqual(jacobian_min(delta),1.)
        r=field_evidence(delta,measured,inspected)
        self.assertEqual(r['measured_pixels'],0)
        self.assertEqual(r['extrapolated_pixels'],500)
        self.assertEqual(r['extrapolated_max_displacement'],3.)
    def test_region_validation_detects_cross_region_tradeoff(self):
        rng=np.random.default_rng(9);a=gaussian_filter(rng.random((100,120)),1)
        original=a.copy();original[:45]=shift(a[:45],[0,5],mode='reflect')
        candidate=a.copy();candidate[55:]=shift(a[55:],[0,-5],mode='reflect')
        rgb=lambda x:np.repeat(x[...,None],3,-1)
        upper=np.zeros(a.shape,bool);upper[10:35,10:-10]=True
        lower=np.zeros(a.shape,bool);lower[65:90,10:-10]=True
        self.assertLess(contrast_error(rgb(a),rgb(candidate),upper),contrast_error(rgb(a),rgb(original),upper)*.1)
        self.assertGreater(contrast_error(rgb(a),rgb(candidate),lower),contrast_error(rgb(a),rgb(original),lower)+.01)
    def test_no_support_is_unknown(self):
        a=np.ones((20,20,3))*.5
        self.assertIsNone(contrast_error(a,a,np.zeros((20,20),bool)))
    def test_region_gate_cannot_trade_away_another_region(self):
        from .registration_audit import nonregressing_regions
        self.assertFalse(nonregressing_regions([dict(before=1.,after=.1),dict(before=1.,after=1.3)]))
        self.assertTrue(nonregressing_regions([dict(before=1.,after=.7),dict(before=1.,after=.9)]))
        self.assertFalse(nonregressing_regions([dict(before=None,after=None)]))
        self.assertFalse(nonregressing_regions([]))
