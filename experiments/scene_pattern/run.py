"""Run real-image alignment and only then open evaluation homographies."""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from .core import analyze,register,map_points,grid,warp

HERE=Path(__file__).resolve().parent

def image(path):return np.asarray(Image.open(path).convert('RGB'),dtype=np.float32)/255

def save(array,path):Image.fromarray(np.uint8(np.clip(array,0,1)*255+.5)).save(path)

def reference_matrix(data,i,j):
    # Oxford maps are in pixel coordinates with a 1-based evaluation convention.
    hi=np.eye(3) if i==1 else np.loadtxt(data/f'H1to{i}p')
    hj=np.eye(3) if j==1 else np.loadtxt(data/f'H1to{j}p')
    w,h=Image.open(data/f'img{i}.ppm').size
    wi,he=Image.open(data/f'img{j}.ppm').size
    ni=np.array([[w-1,0,1],[0,h-1,1],[0,0,1.]])
    nj=np.array([[wi-1,0,1],[0,he-1,1],[0,0,1.]])
    return np.linalg.inv(nj)@hj@np.linalg.inv(hi)@ni

def error(h,truth,data,j):
    q=grid((32,40)).reshape(-1,2);expected=map_points(truth,q)
    mask=np.all((expected>.02)&(expected<.98),axis=1)
    w,hh=Image.open(data/f'img{j}.ppm').size
    distances=np.linalg.norm((map_points(h,q[mask])-expected[mask])*[w-1,hh-1],axis=1)
    return {'median_px':float(np.median(distances)),'p95_px':float(np.quantile(distances,.95)),
            'mean_px':float(np.mean(distances)),'evaluated_points':int(mask.sum()),'within_3px_fraction':float(np.mean(distances<=3))}

def labels_preview(labels):
    rng=np.random.default_rng(18);colors=rng.uniform(.12,.95,(int(labels.max())+1,3))
    return colors[labels]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,default=HERE/'data/graf')
    parser.add_argument('--out',type=Path,default=Path('/tmp/scene_pattern'))
    parser.add_argument('--count',type=int,default=6);args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();scenes=[];photos=[];results=[]
    for i in range(1,args.count+1):
        rgb=image(args.data/f'img{i}.ppm');photos.append(rgb)
        scene=analyze(rgb);scenes.append(scene)
        save(rgb,args.out/f'photo{i}.jpg')
        save(labels_preview(scene.labels['lightness']),args.out/f'regions{i}.png')
        save(np.clip(.5+scene.texture*4,0,1),args.out/f'texture{i}.png')
        print(json.dumps({'analyzed':i,**scene.receipt}),flush=True)
    transforms=[np.eye(3)]
    for i in range(1,args.count):
        print(f'registering {i} -> {i+1}',flush=True)
        h,receipt=register(scenes[i-1],scenes[i])
        # Ground truth first enters here, after the image-only estimate is final.
        truth=reference_matrix(args.data,i,i+1)
        receipt['evaluation']={k:error(np.array(v),truth,args.data,i+1) for k,v in receipt['matrices'].items()}
        receipt.update(reference=i,moving=i+1)
        results.append(receipt);transforms.append(h@transforms[-1])
        (args.out/'pairs.json').write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps({'estimated_pair':[i,i+1],'seconds':receipt['seconds'],'evaluation':receipt['evaluation']}),flush=True)
        aligned,valid=warp(photos[i],h,photos[i-1].shape[:2],method='conv')
        save(np.where(valid[...,None],aligned,.08),args.out/f'aligned{i+1}.jpg')
        # Checkerboard and difference make bad geometry visible.
        y,x=np.indices(valid.shape);checker=((x//40+y//40)%2)==0
        check=np.where(checker[...,None],photos[i-1],aligned)
        save(np.where(valid[...,None],check,.08),args.out/f'checker{i+1}.jpg')
        save(np.where(valid[...,None],np.abs(photos[i-1]-aligned)*3,0),args.out/f'difference{i+1}.jpg')
        print(json.dumps({'pair':[i,i+1],'seconds':receipt['seconds'],'evaluation':receipt['evaluation'],'ringing':receipt['ringing']['accepted']}),flush=True)
        # Save partial progress so a later difficult pair cannot erase evidence.
        (args.out/'pairs.json').write_text(json.dumps(results,indent=2)+'\n')
    total=time.perf_counter()-start
    receipt={'dataset':'Oxford affine Graffiti, six real photographs of a planar scene','ground_truth_used_for_fit':False,
        'scope':'Planar-view Python prototype. No claim of 3D visibility recovery or realtime matching.',
        'source_sha256':{str(p.relative_to(HERE.parent.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'core.py',HERE/'radial.py',HERE/'run.py',HERE.parent.parent/'standalone_conv_resize_demo/native/conv_native.c']},
        'python':platform.python_version(),'machine':platform.machine(),'total_seconds':total,
        'analysis':[s.receipt for s in scenes],'pairs':results,'global_matrices':[h.tolist() for h in transforms],
        'input_sha256':{f'img{i}.ppm':hashlib.sha256((args.data/f'img{i}.ppm').read_bytes()).hexdigest() for i in range(1,args.count+1)}}
    (args.out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'complete':str(args.out),'seconds':total}),flush=True)

if __name__=='__main__':main()
