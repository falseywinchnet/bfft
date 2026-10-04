"""Local reviewer for all inputs, stage outputs and source provenance."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--data',type=Path,required=True);args=p.parse_args();out=args.run/'scene';r=json.loads((out/'receipt.json').read_text());g=json.loads((args.run/'graph'/'graph.json').read_text())
    metrics={m['index']:m for m in r['render']['images']};records=[]
    (out/'thumbnails').mkdir(exist_ok=True);(out/'layers').mkdir(exist_ok=True)
    for i,row in enumerate(r['records']):
        with Image.open(args.data/row['name']) as im:
            im.thumbnail((320,220));im.save(out/'thumbnails'/f'{i}.jpg',quality=85)
        if i in metrics:
            with np.load(out/f'layer-{i:03}.npz') as f:rgb=f['rgb'];mask=f['weight']>0
            rgba=np.concatenate([np.uint8(np.clip(rgb,0,1)*255),np.uint8(mask[...,None])*255],axis=-1)
            Image.fromarray(rgba).save(out/'layers'/f'{i}.png')
        records.append(dict(index=i,name=row['original_name'],panoramic=row['panoramic'],registered=i in metrics,**metrics.get(i,{})))
    with np.load(out/'scene.npz') as f:
        selected=f['preferred_source'];encoded=np.zeros((*selected.shape,3),np.uint8);encoded[...,0]=np.where(selected>=0,selected+1,0);Image.fromarray(encoded).save(out/'source-index.png')
    payload=dict(records=records,stage1=len(r['stage1_images']),stage2=len(r['stage2_images']),unresolved=r['unresolved'],anchor=r['anchor'],
        edges=sum(e['accepted'] for e in g['edges']),tested_edges=len(g['edges']),stage1_edges=sum(e['accepted'] and e['stage']==1 for e in g['edges']),
        render=r['render'],seconds=r['seconds'],graph_seconds=g['seconds'],
        stage1_residual=float(np.median([e['median'] for e in r['stage1']['residuals']])),stage2_residual=float(np.median([e['median'] for e in r['stage2']['residuals']])),
        minimum_area_ratio=r['stage2']['minimum_area_ratio'],limitations=r['limitations'])
    (out/'viewer-data.json').write_text(json.dumps(payload,indent=2));shutil.copyfile(Path(__file__).with_name('sweep-viewer.html'),out/'index.html')
    print(json.dumps({k:v for k,v in payload.items() if k not in ('records','render')}))
if __name__=='__main__':main()
