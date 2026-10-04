import unittest
import numpy as np
from .point_support import prepare
from .surface_jets import project


class PointSupportTests(unittest.TestCase):
    def test_surface_holdout_sites_and_unsupported_depth_are_excluded(self):
        cameras=[dict(shape=[600,800],focal=500.,projection='perspective')]*3
        poses={str(i):dict(rotation=np.eye(3).tolist(),center=[float(i),0.,0.],translation=[-float(i),0.,0.]) for i in range(3)}
        point=[0.,0.,6.];tracks=[]
        for k in range(3):
            obs=[dict(capture=i,site=11+i,xy=project([point],cameras[i],poses[str(i)])[0].tolist()) for i in range(3)]
            if k==2:
                for o in obs:o['site']=7
            tracks.append(dict(id=k,observations=obs))
        pose=dict(poses=poses,points=[dict(track=k,xyz=point) for k in range(3)])
        support=prepare(dict(tracks=tracks),pose,[dict(track=0,holdout_capture=2)],cameras)
        self.assertEqual([p['track'] for p in support['points']],[1])
        self.assertEqual(support['unsupported_tracks'],[2])
        self.assertTrue(all(o['capture']!=2 for o in support['observations'] if o['training']))
        self.assertEqual(sum(not o['training'] for o in support['observations']),1)


if __name__=='__main__':unittest.main()
