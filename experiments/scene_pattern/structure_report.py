"""Build the local structure comparison and preserve source fingerprints."""
import argparse,hashlib,json
from pathlib import Path
from PIL import Image,ImageDraw
from .run import HERE

def main():
    ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);args=ap.parse_args();p=args.directory
    r=json.loads((p/'receipt.json').read_text());im=Image.new('RGB',(990,680),'#0b1015');draw=ImageDraw.Draw(im)
    for k,g in enumerate(r['regions']):
        count=sum(m['pixels'] for m in g['metrics'])
        a=sum(m['before']*m['pixels'] for m in g['metrics'])/count;b=sum(m['after']*m['pixels'] for m in g['metrics'])/count
        g['aggregate_contrast']=dict(before=a,after=b,reduction=1-b/a,weighting='Common-support pixel count per source-to-reference comparison')
        for j,n in enumerate(['before','after']):
            crop=Image.open(p/(g['name']+'-'+n+'.png'));scale=min(315/crop.width,295/crop.height)
            crop=crop.resize((round(crop.width*scale),round(crop.height*scale)))
            im.paste(crop,(k*330,j*340+28));draw.text((k*330+3,j*340+5),g['name'].capitalize()+' - '+n,fill='white')
    im.save(p/'comparison.png')
    r['source_hashes']={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ['structure_alignment.py','floor_affine.py','average_alignment.py','core.py']}
    (p/'receipt.json').write_text(json.dumps(r,indent=2))
    (p/'index.html').write_text((HERE/'structure-viewer.html').read_text())
    print(json.dumps({g['name']:g['aggregate_contrast'] for g in r['regions']},indent=2))
if __name__=='__main__':main()
