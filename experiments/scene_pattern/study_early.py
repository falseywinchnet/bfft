"""Test the proposed order: coarse regions -> delay/trace -> geometric anneal.

The three branches share an image-only region seed. Reference maps are read
only after the branches finish. No image pixels are written by this study.
"""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
from .core import analyze,region_seed,fit_geometry,appearance_score,warp,refine_delays,refine_radial
from .run import image,reference_matrix,error,HERE


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('/tmp/scene_pattern_early'));p.add_argument('--data',type=Path,default=HERE/'data/graf');args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    scenes=[analyze(image(args.data/f'img{i}.ppm')) for i in range(1,7)]
    records=[]
    for i,(a,b) in enumerate(zip(scenes[:-1],scenes[1:]),1):
        initial,regional=region_seed(a,b)
        # A coarse image-only fit supplies a common correspondence basin.
        initial,coarse=fit_geometry(a,b,initial,side=80,max_nfev=65)
        warped=warp(b.texture[...,0],initial,a.texture.shape[:2],method='conv')
        start=time.perf_counter();phase,pr=refine_delays(a,b,initial,'phase',warped);pt=time.perf_counter()-start
        start=time.perf_counter();ring,rr=refine_delays(a,b,initial,'ringing',warped);rt=time.perf_counter()-start
        start=time.perf_counter();radial,ur=refine_radial(a,b,initial,warped);ut=time.perf_counter()-start
        branches={}
        for name,h in [('plain',initial),('phase',phase),('ringing',ring),('radial',radial)]:
            start=time.perf_counter();stages=[]
            for side in (160,256):
                proposal,stage=fit_geometry(a,b,h,side=side,max_nfev=50)
                if appearance_score(a,b,proposal)<appearance_score(a,b,h):h=proposal
                stages.append(stage)
            branches[name]={'matrix':h.tolist(),'seconds':time.perf_counter()-start,'stages':stages}
        truth=reference_matrix(args.data,i,i+1)
        for branch in branches.values():branch['evaluation']=error(np.array(branch['matrix']),truth,args.data,i+1)
        result={'pair':[i,i+1],'coarse':error(initial,truth,args.data,i+1),'branches':branches,
                'delay_seconds':{'phase':pt,'ringing':rt,'radial':ut},'phase':pr,'ringing':rr,'radial':ur}
        records.append(result);(args.out/'receipt.json').write_text(json.dumps(records,indent=2)+'\n')
        print(json.dumps({'pair':[i,i+1],'coarse':result['coarse'],'branches':{k:v['evaluation'] for k,v in branches.items()},'accepted':{'phase':pr['accepted'],'ringing':rr['accepted'],'radial':ur['accepted']}}),flush=True)

if __name__=='__main__':main()
