"""Build and verify the local laptop-only comparison."""
import argparse,json,hashlib,shutil
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from .run import HERE
from .laptop_alignment import domain

def main():
    ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);args=ap.parse_args();p=args.directory;r=json.loads((p/'receipt.json').read_text())
    before=np.asarray(Image.open(p/'before.png'));after=np.asarray(Image.open(p/'after.png'));_,_,allowed=domain(before.shape[:2])
    difference=int(np.abs(before.astype(int)-after.astype(int))[~allowed].max());r['maximum_pixel_difference_outside_laptop_domain']=difference
    if difference:raise ValueError('Pixels changed outside the laptop domain')
    n=sum(m['pixels'] for m in r['metrics']);a=sum(m['before']*m['pixels'] for m in r['metrics'])/n;b=sum(m['after']*m['pixels'] for m in r['metrics'])/n
    r['automation_status']='supervised_diagnostic_not_final_method'
    r['manual_dependencies']=['laptop region','reference capture','protected floor region','partial-view experiment selection']
    r['aggregate_contrast']=dict(before=a,after=b,reduction=1-b/a,weighting='Common-support pixel count per source/reference pair')
    im=Image.new('RGB',(962,340),'#0b1015');draw=ImageDraw.Draw(im)
    for j,name in enumerate(['before','after']):
        crop=Image.open(p/f'laptop-{name}.png').resize((471,297));im.paste(crop,(j*481,30));draw.text((j*481+4,5),'Previous joint alignment' if j==0 else 'Laptop refinement',fill='white')
    im.save(p/'comparison.png');r['renderer_source_hashes']={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ['laptop_alignment.py','joint_distortion.py','distortion_basis.py','registration_audit.py','average_alignment.py','core.py']}
    for name in ['laptop-partial','laptop-absolute']:
        source=HERE/'out'/name/'receipt.json'
        if source.exists():shutil.copy2(source,p/(name+'-audit.json'))
    (p/'receipt.json').write_text(json.dumps(r,indent=2));(p/'index.html').write_text((HERE/'laptop-viewer.html').read_text())
    print(json.dumps(dict(aggregate=r['aggregate_contrast'],metrics=r['metrics'],seconds=r['seconds'],minimum_jacobian=r['minimum_jacobian'],outside_pixel_difference=difference),indent=2))
if __name__=='__main__':main()
