import unittest
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np
from .scene_surface import local_faces,build
from .surface_jets import project


class SurfaceTests(unittest.TestCase):
    def test_local_wall_samples_form_surface(self):
        xy=np.array([[0.,0.],[8,0],[0,8],[8,8]])
        xyz=np.c_[xy*.01,np.ones(4)*4]
        self.assertEqual(len(local_faces(xy,xyz,np.zeros(3))),2)

    def test_no_triangle_bridges_image_gap_or_depth_discontinuity(self):
        xy=np.array([[0.,0.],[8,0],[0,8],[8,8]])
        xyz=np.c_[xy*.01,np.array([4,4,40,40])]
        self.assertEqual(len(local_faces(xy,xyz,np.zeros(3))),0)
        xyz[:,2]=4
        self.assertEqual(len(local_faces(xy*10,xyz,np.zeros(3))),0)

    def test_unverified_estimates_have_a_separate_geometry_buffer(self):
        camera=dict(shape=[20,20],focal=10.,projection='perspective');poses={str(i):dict(rotation=np.eye(3).tolist(),translation=[-float(i),0,0],center=[float(i),0,0],role='seed') for i in range(3)}
        center=[0.,0.,6.];obs=[dict(capture=i,site=1,xy=project([center],camera,poses[str(i)])[0].tolist(),affine=np.eye(2).tolist()) for i in range(3)]
        evidence=dict(cameras=[camera]*3,tracks=[dict(id=0,observations=obs)],records=[dict(name=f'{i}.png',panoramic=False) for i in range(3)])
        pose=dict(accepted=True,poses=poses,points=[dict(track=0,xyz=center,promotion='third_view_consistent')])
        jet=dict(track=0,center=center,tangents=[[.6,0,0],[0,.6,0]],radius=3.,reference_capture=0,observations=obs,accepted=False,training_supported=True,prediction_passed=False)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for i in range(3):Image.new('RGB',(20,20),(80,140,90)).save(root/f'{i}.png')
            packet=build(evidence,pose,root,root/'out',jets=[jet],trial=True,training_estimates=True)
            self.assertEqual(packet['surface_vertices'],0);self.assertEqual(packet['estimate_vertices'],6)
            self.assertEqual(packet['summary']['prediction_checked_patches'],0)
            self.assertEqual((root/'out/estimates.bin').stat().st_size,6*5*4)
            jet.update(accepted=True,prediction_passed=True)
            packet=build(evidence,pose,root,root/'checked',jets=[jet],trial=True,training_estimates=True)
            self.assertEqual(packet['surface_vertices'],6);self.assertEqual(packet['estimate_vertices'],0)


if __name__=='__main__':unittest.main()
