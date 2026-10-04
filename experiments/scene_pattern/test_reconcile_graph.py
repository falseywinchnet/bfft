import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .reconcile_graph import reconcile


class NeighborReconciliationTests(unittest.TestCase):
    def test_neighbor_chain_recovers_map_without_direct_seed_overlap(self):
        rng=np.random.default_rng(224);points=rng.uniform([-2,-1,5],[2,1,10],(160,3));centers=np.array([[0.,0,0],[1.,0,0],[.3,.4,0],[1.5,-.2,.1],[2.,.3,.2]])
        groups=[([0,1,2],range(80)),([1,2,3],range(40,120)),([2,3,4],range(80,160))];tracks=[]
        for tid,x in enumerate(points):
            ids=sorted({i for cameras,span in groups if tid in span for i in cameras});obs=[]
            for i in ids:
                q=x-centers[i];obs.append(dict(capture=i,site=tid,xy=(q[:2]/q[2]*500+[399.5,299.5]).tolist()))
            tracks.append(dict(id=tid,observations=obs))
        e=dict(cameras=[dict(shape=[600,800],focal=500.,projection='perspective')]*5,tracks=tracks)
        with tempfile.TemporaryDirectory() as path:
            folder=Path(path);rows=[]
            for k,(ids,span) in enumerate(groups):
                r=Rotation.from_rotvec(np.array([.1,-.2,.05])*k).as_matrix();s=1+k
                # Keep each seed's first camera at its own origin.
                origin=centers[ids[0]];local=(points-origin)@r/s
                pose=dict(accepted=True,seed=ids[:2],poses={str(i):dict(role='heldout_pose',rotation=r.tolist(),translation=((origin-centers[i])/s).tolist(),center=((centers[i]-origin)@r/s).tolist()) for i in ids},
                          points=[dict(track=tid,xyz=local[tid].tolist(),promotion='third_view_consistent') for tid in span])
                (folder/f'{ids[0]:03d}-{ids[1]:03d}.json').write_text(json.dumps(pose));rows.append(dict(accepted=True,seed=ids[:2],cameras=ids,promotion=dict(third_view_consistent=80)))
            (folder/'submaps.json').write_text(json.dumps(dict(submaps=rows,observed_cameras=list(range(5)))))
            result=reconcile(e,folder,rows)
            self.assertEqual(len(result['poses']),5)
            self.assertEqual(len(result['submap_merge']['merged_seeds']),2)
            np.testing.assert_allclose(result['poses']['4']['center'],centers[4],atol=1e-5)
            self.assertGreaterEqual(len(result['points']),150)


if __name__=='__main__':unittest.main()
