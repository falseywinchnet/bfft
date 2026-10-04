import unittest
import numpy as np
from .geometry_reference import pixel_area,cross,render_reference


class GeometryChecks(unittest.TestCase):
    def test_partition_and_shared_edge(self):
        triangles=np.array([[[1.,1.],[4.,1.],[4.,4.]],[[1.,1.],[4.,4.],[1.,4.]]])
        image,overlap=render_reference(triangles,6,6)
        np.testing.assert_array_equal(image[1:4,1:4],1)
        self.assertEqual(image.sum(),9)
        self.assertEqual(overlap,2)
        a=pixel_area(triangles[0],2,2);b=pixel_area(triangles[1],2,2)
        self.assertAlmostEqual(a+b,1.)
        self.assertAlmostEqual(a+b-a*b,.75)  # Alpha-over is not coverage addition.

    def test_random_mass_and_conservative_support(self):
        rng=np.random.default_rng(17)
        for _ in range(60):
            tri=rng.uniform(1,8,(3,2))
            area2=cross(tri[1]-tri[0],tri[2]-tri[0])
            if area2<0:tri=tri[[0,2,1]];area2=-area2
            if area2<1e-6:continue
            image,_=render_reference([tri],10,10)
            self.assertAlmostEqual(image.sum(),area2/2,places=11)
            edges=np.roll(tri,-1,axis=0)-tri;radii=np.abs(edges).sum(axis=1)/2
            scale=1+3*radii.max()/area2;centre=tri.mean(0)
            enlarged=centre+scale*(tri-centre)
            for corner in ((-.5,-.5),(-.5,.5),(.5,-.5),(.5,.5)):
                for p in tri+corner:
                    for a,b in zip(enlarged,np.roll(enlarged,-1,axis=0)):
                        self.assertGreaterEqual(cross(b-a,p-a),-1e-8)

    def test_subpixel_sliver(self):
        tri=np.array([[1.1,3.11],[7.1,3.11],[4.1,3.19]])
        image,_=render_reference([tri],10,10)
        self.assertAlmostEqual(image.sum(),.24,places=12)
        self.assertLess(image.max(),.08)


if __name__=='__main__':unittest.main()
