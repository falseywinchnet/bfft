import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from .submap_graph import round_trip,components


class MapGraphTests(unittest.TestCase):
    def test_round_trip_is_scale_and_origin_covariant_and_rejects_drift(self):
        rng=np.random.default_rng(10);x=rng.normal(size=(80,3));r=Rotation.from_rotvec([.1,.3,-.2]).as_matrix();s=2.;t=np.array([3.,-4,1.])
        f=dict(scale=s,rotation=r.tolist(),translation=t.tolist());b=dict(scale=1/s,rotation=r.T.tolist(),translation=(-r.T@t/s).tolist())
        self.assertTrue(round_trip(f,b,x)['accepted'])
        # Wrong scale can appear acceptable near the chart origin; test the
        # whole cloud with a centered, scale-relative normalization.
        b['scale']*=1.2
        self.assertFalse(round_trip(f,b,x)['accepted'])
        self.assertFalse(round_trip(f,b,x+100)['accepted'])

    def test_only_checked_edges_connect_maps_through_neighbors(self):
        edges=[dict(a=0,b=1,accepted=True),dict(a=1,b=2,accepted=True),dict(a=2,b=3,accepted=False)]
        self.assertEqual(components(4,edges),[[0,1,2],[3]])


if __name__=='__main__':unittest.main()
