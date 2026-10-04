import unittest
import numpy as np
from .sweep_features import match
from .sweep_graph import components
from .sweep_mesh import lattice,evaluate,solve,minimum_area_ratio,admissible_step,panorama_initialization,periodic_panorama_initialization,unwrap_points
from .sweep_fusion import raster_map,attach
from .sweep_verify import verify_edge
from .sweep_dense import track
from scipy import ndimage

class SweepTests(unittest.TestCase):
    def test_known_affine_region_correspondences_with_outliers(self):
        rng=np.random.default_rng(48);p=rng.uniform([20,20],[340,190],(100,2));d=rng.normal(size=(100,75))
        q=p@np.array([[.93,.04],[-.02,1.05]]).T+[15,9];db=d+rng.normal(0,.003,d.shape);db[-20:]=rng.normal(size=(20,75))
        a=dict(points=p,descriptors=d,shape=[220,380]);b=dict(points=q,descriptors=db,shape=[240,400]);r=match(a,b)
        self.assertTrue(r['accepted']);self.assertGreater(r['inliers'],70);self.assertLess(r['error'],1e-5)
    def test_ambiguous_repeated_descriptors_rejected(self):
        a=dict(points=np.random.default_rng(1).uniform(0,100,(40,2)),descriptors=np.ones((40,75)),shape=[120,120])
        self.assertFalse(match(a,a)['accepted'])
    def test_mesh_interpolation_preserves_affine(self):
        p,_=lattice([160,320]);m=p@np.array([[1.2,.1],[-.2,.9]]).T+[3,8];q=np.array([[0,0],[319,159],[100.2,98.1]])
        np.testing.assert_allclose(evaluate(m,q,[160,320]),q@np.array([[1.2,.1],[-.2,.9]]).T+[3,8],atol=1e-6)
    def test_joint_relative_alignment_and_fixed_stage(self):
        shape=[160,320];p,_=lattice(shape);q=np.array([(x,y) for y in (30,60,90,120) for x in (40,100,160,220,280)],float)
        e=dict(a=0,b=1,a_points=q.tolist(),b_points=(q+[12,0]).tolist())
        result,receipt=solve([shape,shape],[e],{0:p,1:p.copy()},{0},iterations=3)
        np.testing.assert_array_equal(result[0],p)
        error=np.linalg.norm(evaluate(result[0],q,shape)-evaluate(result[1],q+[12,0],shape),axis=1)
        self.assertLess(np.median(error),.3);self.assertGreater(receipt['minimum_area_ratio'],.1)
    def test_raster_coordinate_direction(self):
        shape=[40,60];p,_=lattice(shape);mapped=raster_map(p+[10,5],shape,np.zeros(2),1.,(55,80))
        np.testing.assert_allclose(mapped[15,30],[20,10],atol=1e-5);self.assertTrue(np.isnan(mapped[0,0]).all())
    def test_nonaffine_mesh_uses_renderer_triangle_basis(self):
        shape=[81,81];p,_=lattice(shape);m=p.copy();m[1,1]+=[3,4]
        # First cell's lower triangle excludes its opposite (1,1) vertex.
        q=np.array([[4.,5.]])
        np.testing.assert_allclose(evaluate(m,q,shape),q,atol=1e-6)
        inverse=raster_map(m,shape,np.zeros(2),1.,(90,90))
        selected=np.array([[23.,24.],[35.,35.],[45.,45.]])
        back=evaluate(m,inverse[selected[:,1].astype(int),selected[:,0].astype(int)],shape)
        np.testing.assert_allclose(back,selected,atol=1e-4)
    def test_disconnected_images_are_explicit(self):
        self.assertEqual(components(4,[dict(a=0,b=1,accepted=True),dict(a=2,b=3,accepted=False)]),[[0,1],[2],[3]])
        p,_=lattice([80,100]);base,missing=attach([[80,100],[80,100]],{0:p},[])
        self.assertEqual(missing,[1]);self.assertEqual(set(base),{0})
    def test_local_fold_gate_keeps_distant_safe_update(self):
        p,_=lattice([160,320]);proposal=p.copy();proposal[1,1]=[-150,-100];proposal[-1,-1]+=[2,3]
        safe,fraction=admissible_step(p,p,proposal)
        self.assertGreaterEqual(minimum_area_ratio(p,safe),.1)
        np.testing.assert_allclose(safe[-1,-1],proposal[-1,-1])
    def test_global_panorama_translation_gauge(self):
        p=np.array([[30.,30.],[80,40],[100,70],[40,100]])
        edges=[dict(a=0,b=1,a_points=p.tolist(),b_points=(p+[12,4]).tolist(),inliers=40,error=0)]
        meshes=panorama_initialization([[160,320],[160,320]],edges,0)
        np.testing.assert_allclose(evaluate(meshes[0],p,[160,320]),evaluate(meshes[1],p+[12,4],[160,320]),atol=1e-6)
    def test_signed_structure_verification_accepts_real_translation(self):
        rng=np.random.default_rng(41);a=ndimage.gaussian_filter(rng.normal(size=(160,240)),1.)
        b=ndimage.shift(a,[3,5]);p=np.array([(x,y) for y in (30,60,90,120) for x in (30,70,110,150,190)],float)
        e=dict(a=0,b=1,inliers=len(p),a_points=p.tolist(),b_points=(p+[5,3]).tolist(),matrix=[[1,0,5],[0,1,3],[0,0,1]],accepted=True)
        result=verify_edge(e,a,b);self.assertTrue(result['accepted']);self.assertGreater(result['verification_median'],.99)
        unrelated=ndimage.gaussian_filter(rng.normal(size=a.shape),1.)
        self.assertFalse(verify_edge(e,a,unrelated)['accepted'])
    def test_periodic_scene_wrap_and_inference(self):
        p=np.array([[40.,30.],[80,50],[140,80],[180,100]])
        positions=[0,80,160,240,320];edges=[]
        for a in range(5):
            for b in range(a+1,5):
                delta=(positions[b]-positions[a]+200)%400-200
                edges.append(dict(a=a,b=b,a_points=p.tolist(),b_points=(p-[delta,0]).tolist(),inliers=40,error=0))
        _,period,receipt=periodic_panorama_initialization([[160,320]]*5,edges,0)
        self.assertTrue(receipt['accepted']);np.testing.assert_allclose(period,[400,0],atol=1)
        unwrapped=unwrap_points([[390.,40.],[10.,40.]],np.array([400.,0.]))
        self.assertLess(np.ptp(unwrapped[:,0]),21)
    def test_periodic_mesh_constraints_do_not_pull_across_seam(self):
        shape=[160,320];p,_=lattice(shape);q=np.array([(x,y) for y in (30,60,90,120) for x in (40,100,160,220)],float)
        e=dict(a=0,b=1,a_points=q.tolist(),b_points=(q+[40,0]).tolist())
        result,_=solve([shape,shape],[e],{0:p,1:p+[360,0]},{0},iterations=2,period=np.array([400.,0.]))
        np.testing.assert_allclose(result[1],p+[360,0],atol=1e-4)
    def test_bidirectional_texture_tracks_known_shift(self):
        rng=np.random.default_rng(104);a=ndimage.gaussian_filter(rng.uniform(.1,.9,(96,144,3)),(1.1,1.1,0));b=ndimage.shift(a,[0,3,0],mode='reflect')
        yy,xx=np.indices(a.shape[:2]);inverse=np.stack([xx,yy],-1).astype(float);q=np.array([(x,y) for y in (24,40,56,72) for x in (24,44,64,84,104,124)],float)
        ids,pb,receipt=track(a,b,np.ones(a.shape[:2]),np.ones(a.shape[:2]),q,inverse)
        self.assertGreater(len(ids),12);self.assertLess(np.median(np.linalg.norm(pb-q[ids]-[3,0],axis=1)),.7)
    def test_fold_is_detected(self):
        p,_=lattice([80,100]);self.assertLess(minimum_area_ratio(p,p*[1,-1]),0)

if __name__=='__main__':unittest.main()
