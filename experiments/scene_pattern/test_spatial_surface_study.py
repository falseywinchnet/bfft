import copy
import unittest
import numpy as np
from .spatial_surface_study import roles,initialize,mark_observation,tile_fold,FOLDS,TILE,predict,stats
from .surface_jets import project
from .jet_bundle import refine_jets


class SpatialSurfaceTests(unittest.TestCase):
    def fixture(self):
        camera=dict(shape=[200,300],focal=200.,projection='perspective');cameras=[camera]*4
        centers=[[0.,0,0],[0.,0,0],[1.,0,0],[1.2,.5,0]];poses={str(i):dict(rotation=np.eye(3).tolist(),translation=(-np.array(c)).tolist(),center=c) for i,c in enumerate(centers)}
        jets=[];points=[]
        def surface(q):
            ray=np.r_[(q-[149.5,99.5])/200.,1.];return ray*6/(np.array([.25,-.15,1.])@ray)
        for k in range(64):
            xy=np.array([90.+(k%8)*6,65.+(k//8)*6]);world=surface(xy);obs=[]
            for i in range(4):
                eps=.001;affine=np.stack([(project([surface(xy+np.eye(2)[axis]*eps)],camera,poses[str(i)])[0]-project([surface(xy-np.eye(2)[axis]*eps)],camera,poses[str(i)])[0])/(2*eps) for axis in range(2)],axis=1)
                obs.append(dict(capture=i,xy=project([world],camera,poses[str(i)])[0].tolist(),affine=affine.tolist()))
            jets.append(dict(track=k,radius=3.,reference_capture=0,observations=obs));points.append(dict(track=k,xyz=world.tolist()))
        return dict(results=jets),dict(seed=[0,2],poses=poses,points=points),cameras

    def test_every_center_reserved_once_and_footprint_guard_cannot_train(self):
        packet,_,_=self.fixture()
        for j in packet['results']:
            for o in j['observations']:
                marked=[mark_observation(o,j['radius'],fold) for fold in range(FOLDS)]
                self.assertEqual(sum(r['validation'] for r in marked),1)
                for fold,r in enumerate(marked):
                    if not r['training']:continue
                    for dx in np.linspace(-3,3,7):
                        for dy in np.linspace(-3,3,7):
                            q=np.asarray(o['xy'])+np.asarray(o['affine'])@[dx,dy]
                            self.assertNotEqual(tile_fold(o['capture'],*np.floor(q/TILE).astype(int)),fold)

    def test_fixed_camera_fit_and_initialization_do_not_read_reserved_values(self):
        packet,pose,cameras=self.fixture();marked=roles(packet,0);prepared,receipt=initialize(marked,pose,cameras)
        _,fit,a=refine_jets(prepared,pose,cameras,freeze_cameras=True,coherent_surfaces=True,max_evaluations=150)
        self.assertGreater(fit['coherence_before']['edges'],10)
        for i in pose['poses']:np.testing.assert_array_equal(a['poses'][i]['translation'],pose['poses'][i]['translation'])
        poison=copy.deepcopy(marked)
        for j in poison:
            for o in j['observations']:
                if not o['training']:o['xy']=[4000.,5000.];o['affine']=[[8.,2.],[1.,9.]]
        other,other_receipt=initialize(poison,pose,cameras)
        self.assertEqual(receipt,other_receipt)
        for x,y in zip(prepared['results'],other['results']):
            np.testing.assert_array_equal(x['center'],y['center']);np.testing.assert_array_equal(x['tangents'],y['tangents'])
        _,bad,b=refine_jets(other,pose,cameras,freeze_cameras=True,coherent_surfaces=True,max_evaluations=150)
        self.assertEqual(fit['evaluations'],bad['evaluations'])
        for x,y in zip(a['surface_jets'],b['surface_jets']):np.testing.assert_array_equal(x['center'],y['center'])
        self.assertLess(fit['after']['holdout_median_pixels'],.05)

    def test_displaced_neighbors_supply_depth_to_colocated_view_patch(self):
        packet,pose,cameras=self.fixture();marked=copy.deepcopy(packet['results'])
        for j in marked:
            for o in j['observations']:
                held=j['track'] in (27,28,35,36) and o['capture'] in (2,3)
                o.update(training=not held,validation=held)
        prepared,initialization=initialize(marked,pose,cameras)
        self.assertEqual(initialization['range_initializations'],4)
        errors=[]
        for coherent in (False,True):
            _,_,candidate=refine_jets(prepared,pose,cameras,freeze_cameras=True,coherent_surfaces=coherent,max_evaluations=150)
            errors.append(stats(predict(marked,candidate,cameras))['median_pixels'])
        self.assertGreater(errors[0],.1)
        self.assertLess(errors[1],errors[0]*.1)


if __name__=='__main__':unittest.main()
