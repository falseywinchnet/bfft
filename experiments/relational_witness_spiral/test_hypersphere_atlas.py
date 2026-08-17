import importlib.util
from pathlib import Path
import tempfile
import unittest
import torch
from PIL import Image

HERE=Path(__file__).parent; SPEC=importlib.util.spec_from_file_location('atlas',HERE/'run_hypersphere_atlas.py'); M=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(M)

class AtlasTests(unittest.TestCase):
    def test_layer_starts_dense_with_live_atlas_gradient(self):
        torch.manual_seed(2); layer=M.HypersphereAtlasLinear(5,7,'eikonal'); x=torch.randn(9,5)
        expected=layer.base(x).detach(); actual=layer(x); torch.testing.assert_close(actual,expected)
        actual.square().mean().backward(); self.assertGreater(float(layer.atlas_scale.grad.abs()),0.)
    def test_all_modes_are_finite(self):
        x=torch.randn(4,6)
        for mode in ['identity','isotropic','eikonal','shuffled']:
            y=M.HypersphereAtlasLinear(6,8,mode)(x); self.assertTrue(torch.isfinite(y).all())
    def test_decision_raster_uses_cartesian_y_orientation(self):
        class PositiveY(torch.nn.Module):
            def forward(self,x): return torch.stack((-x[:,1],x[:,1]),1)
        points=torch.tensor([[-1.,-1.],[1.,1.]]); labels=torch.tensor([0,1])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'orientation.png'
            M.draw_scatter(path,PositiveY(),points,labels,points,labels,'orientation',size=240)
            image=Image.open(path)
            top=image.getpixel((120,55)); bottom=image.getpixel((120,185))
        self.assertGreater(top[0],top[2]); self.assertGreater(bottom[2],bottom[0])

if __name__=='__main__': unittest.main()
