"""Attach individual captures to frozen panorama meshes and render both stages."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
from .sweep_fusion import attach,render
from .sweep_mesh import solve


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--features',type=Path,required=True);p.add_argument('--graph',type=Path,required=True);p.add_argument('--scene',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    g=json.loads(args.graph.read_text());r=json.loads((args.scene/'receipt.json').read_text());records=g['records'];shapes=[np.load(args.features/(Path(row['name']).stem+'.npz'))['shape'] for row in records];period=np.array(r['period']['period']);edges=[e for e in g['edges'] if e['accepted']]
    with np.load(args.scene/'panorama-meshes.npz') as f:pano={int(k):f[k] for k in f.files}
    base,missing=attach(shapes,pano,edges,period=period);final,diag=solve(shapes,edges,base,set(pano),period=period);assert all(np.array_equal(final[i],pano[i]) for i in pano)
    r.update(stage2=diag,stage1_images=sorted(pano),stage2_images=sorted(set(final)-set(pano)),unresolved=missing,panoramas_only=False,policy=g['policy'])
    r['render']=render(args.data,records,shapes,final,edges,diag,args.out,period=period)
    r['seconds']=time.perf_counter()-start;r['source_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('sweep*.py')};r['frozen_panorama_receipt_sha256']=hashlib.sha256((args.scene/'receipt.json').read_bytes()).hexdigest()
    np.savez_compressed(args.out/'panorama-meshes.npz',**{str(i):m for i,m in pano.items()});np.savez_compressed(args.out/'meshes.npz',**{str(i):m for i,m in final.items()});(args.out/'receipt.json').write_text(json.dumps(r,indent=2));(args.out/'graph.json').write_text(json.dumps(g,indent=2));print(json.dumps({'attached':len(final),'missing':missing,'seconds':r['seconds']}),flush=True)
if __name__=='__main__':main()
