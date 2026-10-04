"""Private collection import and reproducible two-stage sweep registration.

No object annotations or room-diagnostic fields enter this runner.
"""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps


def prepare(source, out):
    out.mkdir(parents=True, exist_ok=True)
    records=[]
    for path in sorted(source.iterdir()):
        if path.suffix.lower() not in ('.jpg','.jpeg','.png'): continue
        with Image.open(path) as im:
            exif=im.getexif(); rgb=ImageOps.exif_transpose(im).convert('RGB')
            size=rgb.size; panoramic=max(size)/min(size)>2.4
            rgb.thumbnail((2048,2048) if panoramic else (1000,1000))
            Image.frombytes('RGB',rgb.size,rgb.tobytes()).save(out/(path.stem+'.jpg'),quality=94)
        records.append(dict(name=path.stem+'.jpg', original_name=path.name,
            original_size=size,working_size=rgb.size,panoramic=panoramic,
            timestamp=exif.get(306),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        print(json.dumps({'prepared':path.name,'panoramic':panoramic}),flush=True)
    (out/'manifest.json').write_text(json.dumps(records,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path);p.add_argument('--data',type=Path,required=True)
    args=p.parse_args();prepare(args.source,args.data)
if __name__=='__main__':main()
