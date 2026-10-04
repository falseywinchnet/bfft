"""Conservative textured surface fragments from multi-view-consistent points.

Triangles interpolate supported samples; they are not independently measured
depth at every interior pixel. Gaps and depth discontinuities are kept open.
"""
import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial import Delaunay, QhullError


def local_faces(xy, xyz, center, maximum_edge=24., maximum_depth_ratio=1.5):
    if len(xy)<3:return np.empty((0,3),int)
    try:faces=Delaunay(xy).simplices
    except QhullError:return np.empty((0,3),int)
    image_edges=xy[faces]-np.roll(xy[faces],1,axis=1)
    depth=np.linalg.norm(xyz[faces]-center,axis=2)
    area=np.linalg.norm(np.cross(xyz[faces[:,1]]-xyz[faces[:,0]],xyz[faces[:,2]]-xyz[faces[:,0]]),axis=1)
    valid=(np.linalg.norm(image_edges,axis=2).max(1)<=maximum_edge)
    valid&=depth.max(1)/np.maximum(depth.min(1),1e-12)<=maximum_depth_ratio
    valid&=area>1e-12
    return faces[valid]


def build(evidence, pose, data, out, jets=None, trial=False, training_estimates=False):
    out.mkdir(parents=True,exist_ok=True)
    lookup={t['id']:t for t in evidence['tracks']}
    points=[p for p in pose['points'] if p['promotion']=='third_view_consistent']
    xyz=np.asarray([p['xyz'] for p in points]); cameras=sorted(map(int,pose['poses']))
    consistent_tracks={p['track'] for p in points}
    def selected(j):return j['accepted'] or (training_estimates and j.get('training_supported',False))
    if jets is not None:
        # Point-pool deduplication may retain another sample of the same small
        # image neighborhood. Absence from that pool is not a failed jet audit.
        from .scene_promotion import audit
        check=audit(evidence,dict(accepted=True,poses=pose['poses'],camera_models=pose.get('camera_models',evidence['cameras']),
                                 points=[dict(track=j['track'],xyz=j['center']) for j in jets if selected(j)]))
        consistent_tracks={p['track'] for p in check['points'] if p['promotion']=='third_view_consistent'}
    accepted_jets=[j for j in jets if selected(j) and j['track'] in consistent_tracks] if jets is not None else None
    if accepted_jets is not None:
        centers={j['track']:j['center'] for j in accepted_jets}
        xyz=np.asarray([centers.get(p['track'],p['xyz']) for p in points])
    atlas_width=4096;placements={};images={};x=y=row_height=0
    for i in cameras:
        im=Image.open(data/evidence['records'][i]['name']).convert('RGB');im.thumbnail((1024,1024))
        w,h=im.size
        if x+w+4>atlas_width:x=0;y+=row_height;row_height=0
        placements[i]=(x+2,y+2,w,h);images[i]=im;x+=w+4;row_height=max(row_height,h+4)
    atlas_height=2**math.ceil(math.log2(max(2,y+row_height)))
    atlas=Image.new('RGB',(atlas_width,atlas_height))
    for i,im in images.items():atlas.paste(im,placements[i][:2])
    atlas.save(out/'texture.jpg',quality=94)
    def uv(i,xy):
        h,w=evidence['cameras'][i]['shape'];x,y,iw,ih=placements[i]
        return (np.asarray(xy)/[w-1,h-1]*[iw-1,ih-1]+[x+.5,y+.5])/[atlas_width,atlas_height]
    faces={};point_uv=[None]*len(points);counts={}
    for i in cameras:
        rows=[]
        for k,p in enumerate(points):
            obs=next((o for o in lookup[p['track']]['observations'] if o['capture']==i),None)
            if obs is not None:rows.append((k,obs['xy']))
        if not rows:continue
        ids=np.array([r[0] for r in rows]);xy=np.asarray([r[1] for r in rows]);tex=uv(i,xy)
        for k,t in zip(ids,tex):
            if point_uv[k] is None:point_uv[k]=t.tolist()
        center=np.asarray(pose['poses'][str(i)]['center'])
        local=local_faces(xy,xyz[ids],center);counts[i]=len(local)
        for face in local:
            vertices=ids[face];key=tuple(sorted(map(int,vertices)))
            a,b,c=xy[face];ab=b-a;ac=c-a;area=abs(float(ab[0]*ac[1]-ab[1]*ac[0]))
            sh=evidence['cameras'][i]['shape'];_,_,iw,ih=placements[i]
            score=area*iw*ih/(sh[0]*sh[1])
            # Exact common triangles get one texture, selected by available
            # image sampling area. Different local triangulations remain fragments.
            if key not in faces or score>faces[key]['score']:
                faces[key]=dict(vertices=vertices.tolist(),uv=tex[face].tolist(),source=i,score=score)
    triangles=list(faces.values());mesh=np.asarray([np.r_[xyz[k],t] for f in triangles for k,t in zip(f['vertices'],f['uv'])],dtype='<f4')
    estimates=[]
    if accepted_jets is not None:
        mesh=[];counts={}
        corners=np.array([[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.]])
        for j in accepted_jets:
            i=j['reference_capture'];xy=next(o['xy'] for o in j['observations'] if o['capture']==i)
            offsets=corners*j['radius'];world=np.asarray(j['center'])+offsets@np.asarray(j['tangents']);tex=uv(i,np.asarray(xy)+offsets)
            destination=estimates if training_estimates and not j.get('prediction_passed',False) else mesh
            for face in ((0,1,2),(0,2,3)):
                destination.extend(np.r_[world[k],tex[k]] for k in face)
            counts[i]=counts.get(i,0)+2
        mesh=np.asarray(mesh,dtype='<f4').reshape(-1,5)
    estimates=np.asarray(estimates,dtype='<f4').reshape(-1,5);estimates.tofile(out/'estimates.bin')
    mesh.tofile(out/'surface.bin')
    point_data=np.asarray([np.r_[p,t] for p,t in zip(xyz,point_uv)],dtype='<f4');point_data.tofile(out/'points.bin')
    target=np.median(xyz,axis=0);distance=np.linalg.norm(xyz-target,axis=1)
    packet=dict(displayed_geometry='candidate' if trial else 'retained',cameras=[dict(id=i,name=evidence['records'][i]['name'],panoramic=evidence['records'][i]['panoramic'],**pose['poses'][str(i)]) for i in cameras],
        summary=dict(cameras=len(cameras),photographs=sum(not evidence['records'][i]['panoramic'] for i in cameras),points=len(points),triangles=(len(mesh)+len(estimates))//3,
                     captures_total=len(evidence['records']),surface_sources=len({f['source'] for f in triangles}) if jets is None else len(counts)),
        target=target.tolist(),radius=float(max(np.quantile(distance,.9),.1)),extent=float(max(distance.max(),1.)),
        surface_vertices=len(mesh),estimate_vertices=len(estimates),point_vertices=len(point_data),atlas_shape=[atlas_height,atlas_width],faces_by_camera=counts,
        policy=dict(maximum_image_edge_pixels=24,maximum_vertex_range_ratio=1.5,source='only all-available-view-consistent points',
                    rendering='depth-tested textured local triangles; no hole filling'),
        limitations='Partial nominal-calibration geometry. Triangles interpolate sparse measured points; interior surfaces and visibility are not independently validated. Local fragments can overlap or disagree. Unresolved captures and regions remain absent.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if accepted_jets is not None:
        packet['summary'].update(surface_patches=len(accepted_jets),tested_patches=sum('validation' in j for j in jets),
                                 rejected_by_center_audit=sum(selected(j) and j['track'] not in consistent_tracks for j in jets))
        packet['policy']=dict(source='locally inferred position and two tangents; patch predicts its withheld camera within 1.5 pixels and center passes the all-available-view audit',
                             rendering='depth-tested texture over the tested 3-pixel reference radius; no connections across patch gaps')
        packet['limitations']='Only accepted differential surface patches are textured. Other point hypotheses remain visible as context. Sparse patch support does not establish a continuous surface or global visibility; camera geometry remains conditional.'
    if training_estimates:
        packet['summary'].update(training_estimates=True,prediction_checked_patches=sum(j.get('prediction_passed',False) for j in accepted_jets))
        packet['policy']=dict(source='All-view fitting support, with separate spatially blocked prediction evidence',rendering='Checked patches and unverified estimates are separate buffers; no connections across patch gaps')
        packet['limitations']='All-view experimental surface field. Unverified surface estimates fit their observations but do not pass every blocked prediction. Toggle them separately. Camera initialization, correspondence discovery and depth remain conditional; this is not a complete recovered scene.'
    if 'jet_bundle' in pose:packet['refinement']=pose['jet_bundle']
    (out/'scene.json').write_text(json.dumps(packet,indent=2));shutil.copyfile(Path(__file__).with_name('surface-viewer.html'),out/'index.html')
    return packet


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--pose',type=Path,required=True)
    p.add_argument('--training-estimates',action='store_true');p.add_argument('--trial',action='store_true');p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--jets',type=Path);args=p.parse_args()
    jets=None
    if args.jets:
        packet=json.loads(args.jets.read_text());jets=packet.get('results',packet.get('surface_jets'))
        if jets is None:raise ValueError('No differential surface hypotheses in jets input')
    if args.training_estimates and jets is None:raise ValueError('Training estimate display requires a surface field')
    e=json.loads(args.evidence.read_text());pose=json.loads(args.pose.read_text());result=build(e,pose,args.data,args.out,jets=jets,trial=args.trial,training_estimates=args.training_estimates)
    if args.jets:result['jets_sha256']=hashlib.sha256(args.jets.read_bytes()).hexdigest()
    result['evidence_sha256']=hashlib.sha256(args.evidence.read_bytes()).hexdigest();result['pose_sha256']=hashlib.sha256(args.pose.read_bytes()).hexdigest()
    (args.out/'scene.json').write_text(json.dumps(result,indent=2));print(json.dumps(result['summary']))


if __name__=='__main__':main()
