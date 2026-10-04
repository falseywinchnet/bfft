import unittest
import numpy as np
from scipy import ndimage
from .region_transport import fit_affine_regions,transport

class RegionalTransportTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(91);self.image=ndimage.gaussian_filter(rng.normal(size=(128,160,3)),(1.1,1.1,0))*.2+[.5,.03,.02]
    def test_affine_recovers_shift_skew_scale_without_equating_centroids(self):
        a=np.array([[1.05,.12],[-.05,.94]]);t=np.array([4.,-2.]);yy,xx=np.indices(self.image.shape[:2]);q=(np.stack([xx,yy],axis=-1)-t)@np.linalg.inv(a).T
        moving=np.stack([ndimage.map_coordinates(self.image[...,k],[q[...,1],q[...,0]],order=3,mode='reflect') for k in range(3)],axis=-1)
        pa=np.array([[50.,50],[80.,60],[100.,70]]);truth=pa@a.T+t;pb=truth+[1.4,-1.1]
        result=fit_affine_regions(self.image,moving,pa,pb,np.tile(np.eye(2),(3,1,1)),iterations=24)
        np.testing.assert_allclose(result['center'],truth,atol=.25);np.testing.assert_allclose(result['affine'],np.tile(a,(3,1,1)),atol=.035);self.assertGreater(result['validation_correlation'].min(),.95)
    def test_texture_line_reports_aperture_uncertainty(self):
        yy,xx=np.indices((128,160));plane=.5+.15*np.sin(xx/3);im=np.stack([plane,plane*.1,plane*.05],axis=-1);pa=np.array([[64.,64.]])
        result=fit_affine_regions(im,im,pa,pa)
        self.assertLess(result['information_ratio'][0],1e-5)
    def test_unrelated_images_cannot_promote_regions(self):
        rng=np.random.default_rng(18);other=ndimage.gaussian_filter(rng.normal(size=self.image.shape),(1.1,1.1,0))*.2+[.5,.03,.02]
        pa=np.array([[50.,50],[80.,60],[100.,70]])
        result=transport(self.image,other,pa,pa)
        self.assertFalse(result['accepted'].any())


class ContinuousCycleTests(unittest.TestCase):
    def test_centroid_offsets_do_not_change_the_transported_point(self):
        from .transport_tracks import close_triangle,synchronize
        a=np.array([[1.1,.08],[-.04,.9]]);ta=np.array([4.,-1.]);c=np.array([[.92,-.1],[.06,1.05]]);tc=np.array([-2.,3.]);pa=np.array([40.,40.]);pb=a@pa+ta+[1.2,-1.8];pc=c@pa+tc+[-1.3,.7]
        bc=c@np.linalg.inv(a);tbc=tc-bc@ta
        def link(i,j,p,qsite,mat,t):return dict(a=(i,0),b=(j,0),p=p,site_q=qsite,q=mat@p+t,A=mat,correlation=.99)
        ab=link(0,1,pa,pb,a,ta);ac=link(0,2,pa,pc,c,tc);bcl=link(1,2,pb,pc,bc,tbc)
        error,jac=close_triangle(ab,bcl,ac);self.assertLess(error,1e-10);self.assertLess(jac,1e-10)
        xy,diag=synchronize([(0,0),(1,0),(2,0)],[ab,bcl,ac]);np.testing.assert_allclose(xy[1,0],a@pa+ta,atol=1e-10);np.testing.assert_allclose(xy[2,0],c@pa+tc,atol=1e-10)
        bcl['q']=bcl['q']+[4.,0];self.assertGreater(close_triangle(ab,bcl,ac)[0],3)

class FlatSupportTests(unittest.TestCase):
    def test_constant_and_black_regions_do_not_break_batched_fitting(self):
        rng=np.random.default_rng(29);im=ndimage.gaussian_filter(rng.random((100,100,3)),(1,1,0));flat=np.zeros_like(im);flat[20:80,20:80]=[.6,.02,.01]
        p=np.array([[15.,15.],[35.,35.],[65.,65.]])
        result=fit_affine_regions(im,flat,p,p)
        self.assertTrue(np.isfinite(result['center']).all());self.assertFalse(np.any(result['validation_correlation']>.73))



class FusionSupportTests(unittest.TestCase):
    def test_equal_support_average_improves_known_transport(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from .transport_fusion import render,linear_image
        from .run import save
        rng=np.random.default_rng(39);rgb=ndimage.gaussian_filter(rng.random((80,100,3)),(1,1,0))*.7+.1
        with TemporaryDirectory() as temp:
            root=Path(temp);records=[];obs=[];links=[];p=np.array([50.,40.])
            for i,t in enumerate(([0,0],[4,0],[-3,2])):
                t=np.array(t,float);name=f'{i}.png';records.append(dict(name=name,panoramic=True));shifted=ndimage.shift(rgb,(t[1],t[0],0),order=1,mode='nearest');save(shifted,root/name)
                obs.append(dict(capture=i,site=0,xy=(p+t).tolist(),centroid=(p+t+(np.array([2.,-1]) if i else 0)).tolist()))
                if i:links.append(dict(a=[0,0],b=[i,0],affine=np.eye(2).tolist()))
            e=dict(records=records,cameras=[dict(shape=[80,100])]*3,tracks=[dict(id=0,observations=obs,transport_links=links)])
            result=render(e,root,root/'out',0,width=100);self.assertGreater(result['dispersion_reduction'],.999);self.assertGreater(result['three_view_fraction'],0);self.assertLess(result['any_support_fraction'],.1);linear_image.cache_clear()

class DenseFieldTests(unittest.TestCase):
    def test_affine_field_interpolation_and_missing_support(self):
        from .dense_transport import Field
        p=np.array([[0.,0.],[8.,0.],[0.,8.],[8.,8.]]);a=np.array([[1.2,.1],[-.1,.9]]);t=np.array([5.,-2.]);f=Field(p,p@a.T+t,np.tile(a,(4,1,1)))
        query=np.array([[4.,4.],[3.,3.],[100.,100.]]);q,j,valid=f.at(query)
        np.testing.assert_allclose(q[:2],query[:2]@a.T+t,atol=1e-10);self.assertEqual(valid.tolist(),[True,True,False])
    def test_source_set_cannot_combine_incompatible_triangle_branches(self):
        from .dense_scene import largest_clique
        clique=largest_clique((1,2,3,4),((2,),(1,),(4,),(3,)))
        self.assertIn(clique,((1,2),(3,4)));self.assertEqual(len(clique),2)

class AttachmentSupportTests(unittest.TestCase):
    def test_one_way_or_small_triangle_support_cannot_attach_a_capture(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from .attach_fields import supported_graph, reachable
        with TemporaryDirectory() as temp:
            folder=Path(temp)
            rows=[dict(a=0,b=1),dict(a=1,b=2),dict(a=2,b=3)]
            for a,b,forward,reverse in ((0,1,15,15),(1,2,15,11),(2,3,20,20)):
                for x,y,n in ((a,b,forward),(b,a,reverse)):
                    np.savez(folder/f'{x:03d}-{y:03d}.npz',third_view_support=np.r_[np.ones(n),np.zeros(20-n)])
            graph=supported_graph(folder,rows)
            self.assertEqual(reachable(graph,{0}),{0,1})
            self.assertEqual(reachable(graph,{2}),{2,3})
            np.savez(folder/'002-001.npz',third_view_support=np.ones(12))
            self.assertEqual(reachable(supported_graph(folder,rows),{0}),{0,1,2,3})

if __name__=='__main__':unittest.main()
