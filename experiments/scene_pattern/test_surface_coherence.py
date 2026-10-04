import copy
import unittest
import numpy as np
from .surface_coherence import neighbors,operator


class SurfaceCoherenceTests(unittest.TestCase):
    def fixture(self):
        jets=[]
        for k in range(3):
            xy=np.array([10.+k*6,12.]);j=np.array([[1.1,.2],[.1,.9]])
            jets.append(dict(track=k,center=[xy[0]*.1,xy[1]*.1,5.],tangents=[[.1,0,0],[0,.1,0]],radius=3.,holdout_capture=2,
                observations=[dict(capture=0,xy=xy.tolist(),affine=np.eye(2).tolist()),dict(capture=1,xy=(j@xy+[3.,1]).tolist(),affine=j.tolist()),dict(capture=2,xy=(xy+[5.,2]).tolist(),affine=np.eye(2).tolist())]))
        cameras=[dict(focal=100.)]*3;poses={str(i):dict(center=[float(i),0,0]) for i in range(3)}
        return jets,cameras,poses

    def test_flat_surface_is_integrable_and_fold_is_detected(self):
        jets,cameras,poses=self.fixture();edges=neighbors(jets);self.assertGreater(len(edges),0);matrix=operator(jets,edges,cameras,poses)
        values=np.array([np.r_[j['center'],np.asarray(j['tangents']).ravel()*j['radius']] for j in jets]);np.testing.assert_allclose(matrix@values.ravel(),0,atol=1e-12)
        values[1,2]+=2;self.assertGreater(np.linalg.norm(matrix@values.ravel()),10.)

    def test_holdout_values_cannot_select_links_and_depth_edges_do_not_link(self):
        jets,_,_=self.fixture();original=neighbors(jets);poisoned=copy.deepcopy(jets)
        for j in poisoned:j['observations'][-1]['xy']=[1000.,2000.];j['observations'][-1]['affine']=[[9.,0],[0,9.]]
        self.assertEqual(original,neighbors(poisoned))
        discontinuous=copy.deepcopy(jets)
        discontinuous[1]['observations'][1]['xy'][0]+=20
        self.assertFalse(any(1 in (e['a'],e['b']) for e in neighbors(discontinuous)))

    def test_reference_parameterization_does_not_change_surface_constraint(self):
        jets,cameras,poses=self.fixture();edges=neighbors(jets);values=np.array([np.r_[j['center'],np.asarray(j['tangents']).ravel()*j['radius']] for j in jets]);before=operator(jets,edges,cameras,poses)@values.ravel()
        change=np.array([[2.,.1],[0.,.5]]);other=copy.deepcopy(jets)
        for j in other:
            j['tangents']=(np.linalg.inv(change).T@np.asarray(j['tangents'])).tolist()
            for o in j['observations']:o['affine']=(np.asarray(o['affine'])@np.linalg.inv(change)).tolist()
        values=np.array([np.r_[j['center'],np.asarray(j['tangents']).ravel()*j['radius']] for j in other]);after=operator(other,neighbors(other),cameras,poses)@values.ravel()
        np.testing.assert_allclose(after,before,atol=1e-10)


if __name__=='__main__':unittest.main()
