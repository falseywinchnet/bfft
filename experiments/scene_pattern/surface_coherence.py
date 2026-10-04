"""Training-only neighborhood compatibility and discrete surface integrability."""
from collections import defaultdict
from .surface_jets import is_training
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix


def neighbors(jets,maximum_distance=12.,maximum_degree=6):
    """Link only regions whose measured transports agree in two training views."""
    observations=[];views=defaultdict(list)
    for k,j in enumerate(jets):
        obs={o['capture']:o for o in j['observations'] if is_training(j,o)}
        observations.append(obs)
        for i,o in obs.items():views[i].append((k,o['xy']))
    candidates=set()
    for rows in views.values():
        xy=np.asarray([r[1] for r in rows])
        for a,b in cKDTree(xy).query_pairs(maximum_distance):candidates.add(tuple(sorted((rows[a][0],rows[b][0]))))
    edges=[]
    for a,b in sorted(candidates):
        oa,ob=observations[a],observations[b];common=sorted(oa.keys()&ob.keys())
        if len(common)<2:continue
        choices=[]
        for reference in common:
            delta=np.asarray(ob[reference]['xy'])-oa[reference]['xy'];distance=float(np.linalg.norm(delta))
            if not 2.<=distance<=maximum_distance:continue
            ia=np.linalg.inv(np.asarray(oa[reference]['affine']));ib=np.linalg.inv(np.asarray(ob[reference]['affine']));errors=[];changes=[]
            for i in common:
                da=np.asarray(oa[i]['affine'])@ia;db=np.asarray(ob[i]['affine'])@ib;actual=np.asarray(ob[i]['xy'])-oa[i]['xy']
                errors.extend([np.linalg.norm(da@delta-actual),np.linalg.norm(db@delta-actual)])
                changes.append(np.linalg.norm(da-db)/max(np.linalg.norm((da+db)/2),1e-9))
            if max(errors)>1.5 or max(changes)>.15:continue
            choices.append((max(errors),distance,reference,delta,ia,ib))
        if not choices:continue
        error,distance,reference,delta,ia,ib=min(choices,key=lambda r:r[:3])
        edges.append(dict(a=a,b=b,reference=reference,distance=distance,transport_error=float(error),
                          delta_a=(ia@delta).tolist(),delta_b=(ib@delta).tolist(),inverse_a=ia.tolist(),inverse_b=ib.tolist(),common_training_cameras=common))
    degrees=np.zeros(len(jets),int);retained=[]
    for e in sorted(edges,key=lambda e:(e['transport_error'],e['distance'],e['a'],e['b'])):
        a,b=e['a'],e['b']
        if degrees[a]>=maximum_degree or degrees[b]>=maximum_degree:continue
        degrees[a]+=1;degrees[b]+=1;retained.append(e)
    return retained


def operator(jets,edges,cameras,poses):
    """Trapezoidal position compatibility plus transported tangent continuity.

The unknown per jet is [center, radius*tangent_u, radius*tangent_v]. The frozen
world-to-feature normalization uses only training views and initial geometry.
"""
    scales=[]
    for j in jets:
        scales.append(np.median([np.linalg.norm(np.asarray(j['center'])-poses[str(o['capture'])]['center'])/cameras[o['capture']]['focal'] for o in j['observations'] if is_training(j,o)]))
    rr=[];cc=[];vv=[]
    def add(row,jet,block,dimension,value):
        rr.append(row);cc.append(jet*9+block*3+dimension);vv.append(value)
    for k,e in enumerate(edges):
        a,b=e['a'],e['b'];scale=max(np.sqrt(scales[a]*scales[b]),1e-9)
        da=np.asarray(e['delta_a'])/jets[a]['radius'];db=np.asarray(e['delta_b'])/jets[b]['radius']
        # Xb-Xa = integral of the local tangent field along the image edge.
        for dim in range(3):
            row=9*k+dim;add(row,a,0,dim,-1/scale);add(row,b,0,dim,1/scale)
            for axis in range(2):add(row,a,axis+1,dim,-.5*da[axis]/scale);add(row,b,axis+1,dim,-.5*db[axis]/scale)
        # Compare derivatives in a common image chart, not raw reference bases.
        ia=np.asarray(e['inverse_a']).T/jets[a]['radius'];ib=np.asarray(e['inverse_b']).T/jets[b]['radius'];length=min(e['distance'],3.)
        for axis in range(2):
            for dim in range(3):
                row=9*k+3+3*axis+dim
                for src in range(2):add(row,a,src+1,dim,ia[axis,src]*length/scale);add(row,b,src+1,dim,-ib[axis,src]*length/scale)
    return coo_matrix((vv,(rr,cc)),shape=(9*len(edges),9*len(jets))).tocsr()


def summary(jets,edges,matrix,values):
    error=(matrix@values.ravel()).reshape(-1,9)
    return dict(edges=len(edges),connected_patches=len({e[k] for e in edges for k in ('a','b')}),
                position_median_feature_equivalent=float(np.median(np.linalg.norm(error[:,:3],axis=1))) if len(edges) else None,
                tangent_median_feature_equivalent=float(np.median(np.linalg.norm(error[:,3:],axis=1))) if len(edges) else None,
                policy='At least two common training cameras; local transport error <=1.5 pixels and relative Jacobian change <=0.15; 2–12 pixel neighbors, degree <=6. Robust trapezoidal position and transported tangent residuals; no held-out observation enters links or normalization.')
