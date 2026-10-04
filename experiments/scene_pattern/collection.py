"""Image-only registration of an arbitrary EXIF-oriented photo collection."""
import argparse,json,time,pickle
from pathlib import Path
import numpy as np
from .core import analyze,register,warp,map_points,grid
from .run import image,save


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    paths=sorted(args.data.glob('IMG_*.png'));start=time.perf_counter();scenes=[];pairs=[]
    for path in paths:
        s=analyze(image(path));scenes.append(s);print(json.dumps({'analyzed':path.name,'regions':len(s.centers)}),flush=True)
    with (args.out/'scenes.pkl').open('wb') as f:pickle.dump(scenes,f)
    for i in range(len(paths)-1):
        h,r=register(scenes[i],scenes[i+1]);r.update(reference=paths[i].name,moving=paths[i+1].name)
        pairs.append(r)
        moved,valid=warp(scenes[i+1].rgb,h,scenes[i].rgb.shape[:2],method='linear')
        save(np.where(valid[...,None],.5*moved+.5*scenes[i].rgb,scenes[i].rgb*.25),args.out/f'join{i+1:02}.jpg')
        print(json.dumps({'pair':i+1,'score':r['final_appearance_score'],'seed_inliers':r['seed'].get('inliers',0),'h':h.tolist()}),flush=True)
        (args.out/'registration.json').write_text(json.dumps(dict(files=[p.name for p in paths],pairs=pairs,seconds=time.perf_counter()-start),indent=2))
if __name__=='__main__':main()
