import json
import tempfile
import unittest
from pathlib import Path
from . import test_spatial_surface_study as fixtures
from .spatial_surface_study import initialize
from .jet_bundle import refine_jets
from .full_surface_field import attach_predictions


class FullSurfaceFieldTests(unittest.TestCase):
    def test_all_view_fit_does_not_claim_prediction_and_scores_attach_afterward(self):
        packet,pose,cameras=fixtures.SpatialSurfaceTests().fixture();marked=packet['results']
        for j in marked:
            for o in j['observations']:o.update(training=True,validation=False)
        prepared,_=initialize(marked,pose,cameras)
        _,receipt,candidate=refine_jets(prepared,pose,cameras,coherent_surfaces=True,freeze_cameras=True,max_evaluations=100)
        self.assertFalse(receipt['prediction_evaluated']);self.assertFalse(receipt['applied']);self.assertIsNone(receipt['after']['holdout_median_pixels'])
        self.assertEqual(receipt['accepted_jets_after'],0);self.assertGreater(receipt['training_supported_jets_after'],50)
        json.dumps(candidate,allow_nan=False)
        geometry=[j['center'] for j in candidate['surface_jets']]
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)
            for fold in range(4):
                rows=[dict(track=j['track'],capture=fold,resolved=True,passed=j['track']!=0 or fold!=2) for j in candidate['surface_jets']]
                (p/f'{fold}-coherent.predictions.json').write_text(json.dumps(rows))
            attach_predictions(candidate,p)
        self.assertEqual(geometry,[j['center'] for j in candidate['surface_jets']]);self.assertFalse(candidate['surface_jets'][0]['accepted']);self.assertTrue(all(j['accepted'] for j in candidate['surface_jets'][1:]))


if __name__=='__main__':unittest.main()
