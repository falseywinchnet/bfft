"""Build a local joint-basis comparison and fingerprint the renderer sources."""
import argparse,json,hashlib
from pathlib import Path
from PIL import Image,ImageDraw
from .run import HERE

def main():
 ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);args=ap.parse_args();p=args.directory;r=json.loads((p/'receipt.json').read_text())
 im=Image.new('RGB',(856,550),'#0b1015');draw=ImageDraw.Draw(im)
 for j,name in enumerate(['before','after']):
  a=Image.open(p/f'lamp-{name}.png').resize((418,510));im.paste(a,(j*428,30));draw.text((j*428+5,6),'Previous alignment' if j==0 else 'Accepted joint distortion',fill='white')
 im.save(p/'comparison.png');r['renderer_source_hashes']={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ['joint_distortion.py','distortion_basis.py','registration_audit.py','average_alignment.py','core.py']}
 (p/'receipt.json').write_text(json.dumps(r,indent=2));(p/'index.html').write_text((HERE/'joint-viewer.html').read_text())
 print(json.dumps(dict(seconds=r['seconds'],minimum_jacobian=r['minimum_jacobian'],metrics=r['metrics'],models=[dict(view=v['view'],applied=v['applied'],model=v['selected']['model'] if v.get('selected') and v['applied'] else None,status=v['status'],continuation_residuals=v.get('continuation_residuals')) for v in r['views']]),indent=2))
if __name__=='__main__':main()
