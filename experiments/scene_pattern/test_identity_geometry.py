import copy,unittest
import numpy as np
from .test_affine_camera_geometry import fixture
from .identity_geometry import fit_pair,roles

class IdentityGeometryTests(unittest.TestCase):
    def test_geometry_competes_with_wrong_image_identity_and_retains_alternatives(self):
        rows,cameras,R,t=fixture(65);rng=np.random.default_rng(81);allrows=[]
        for row in rows:
            truth=copy.deepcopy(row);truth['training']=.96;allrows.append(truth)
            wrong=copy.deepcopy(row);wrong['q']=wrong['q']+rng.uniform([25,20],[65,60]);wrong['training']=.98;allrows.append(wrong)
        result=fit_pair(allrows,cameras,trials=110)
        self.assertGreater(result.get('accepted_models',0),0)
        changed=[u for u in result['updates'] if u['changed']];self.assertGreater(len(changed),40)
        self.assertTrue(all(u['after']%2==0 for u in changed));self.assertEqual(len(allrows),130)
    def test_training_hypothesis_archive_is_refitted_even_if_previously_rejected(self):
        from .affine_camera_geometry import skew
        from .multiview_geometry import unit
        rows,cameras,R,t=fixture(65);prior=dict(kind='essential',matrix=(skew(unit(t))@R).tolist(),accepted=False)
        result=fit_pair(rows,cameras,trials=0,prior_models=[prior])
        self.assertEqual(len(result['models']),1);self.assertTrue(result['models'][0]['accepted'])
        self.assertEqual(result['models'][0]['origin'],'replayed_training_hypothesis')
    def test_pair_refinement_does_not_erase_the_starting_camera_branch(self):
        from scipy.spatial.transform import Rotation
        from .affine_camera_geometry import skew
        from .multiview_geometry import unit
        rows,cameras,R,t=fixture(65);E=skew(unit(t))@Rotation.from_rotvec([.001,0,0]).as_matrix()@R
        result=fit_pair(rows,cameras,trials=0,prior_models=[dict(kind='essential',matrix=E.tolist())])
        stages={m['stage']:m for m in result['models']};self.assertIn('proposal',stages);self.assertIn('pair_refined',stages)
        np.testing.assert_allclose(stages['proposal']['matrix'],E,atol=1e-15)
        self.assertLess(stages['pair_refined']['train_median'],stages['proposal']['train_median'])
    def test_reserved_coordinates_cannot_choose_or_refine_geometry(self):
        rows,cameras,R,t=fixture(55);split=roles(rows,0);base=fit_pair(rows,cameras,trials=30,split=split)
        other=copy.deepcopy(rows);rng=np.random.default_rng(57)
        for k in np.flatnonzero(~split[0]):other[k]['q']=rng.uniform([20,20],[780,580]);other[k]['A']=rng.uniform(-2,2,(2,2))
        poisoned=fit_pair(other,cameras,trials=30,split=split)
        self.assertEqual(len(base['models']),len(poisoned['models']))
        for a,b in zip(base['models'],poisoned['models']):np.testing.assert_allclose(a['matrix'],b['matrix'],atol=1e-12)
        self.assertFalse(any(m['accepted'] for m in poisoned['models']))

if __name__=='__main__':unittest.main()
