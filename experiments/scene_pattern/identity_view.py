"""Local image-registration reviewer; no fused scene or station claims."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from PIL import Image

def build(study,data,features,out):
    out.mkdir(parents=True,exist_ok=True);(out/'images').mkdir(exist_ok=True)
    records=json.loads((data/'manifest.json').read_text());summary=json.loads((study/'summary.json').read_text());edges=json.loads((study/'identities.json').read_text())
    for i,r in enumerate(records):
        with np.load(features/(Path(r['name']).stem+'.npz')) as f:rgb=f['rgb']
        Image.fromarray(np.uint8(np.clip(rgb,0,1)*255)).save(out/'images'/f'{i}.jpg',quality=94)
    (out/'identity-data.json').write_text(json.dumps(dict(records=records,summary=summary,edges=edges)))
    shutil.copyfile(Path(__file__).with_name('identity-viewer.html'),out/'index.html')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--features',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();build(a.study,a.data,a.features,a.out)
