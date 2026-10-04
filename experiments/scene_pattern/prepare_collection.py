"""Make EXIF-oriented, metadata-free working copies; never modify originals."""
import argparse,json,hashlib
from pathlib import Path
from PIL import Image,ImageOps,ImageDraw


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--side',type=int,default=1200);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    paths=sorted(p for p in args.source.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png'))
    sheet=Image.new('RGB',(1000,280*((len(paths)+3)//4)),'#17202a');draw=ImageDraw.Draw(sheet);manifest=[]
    for i,path in enumerate(paths):
        rgb=ImageOps.exif_transpose(Image.open(path)).convert('RGB');original_size=rgb.size;rgb.thumbnail((args.side,args.side))
        # New image object prevents EXIF/IPTC metadata following the working copy.
        clean=Image.frombytes('RGB',rgb.size,rgb.tobytes());clean.save(args.out/(path.stem+'.png'))
        thumb=ImageOps.contain(clean,(245,240));x=i%4*250;y=i//4*280;sheet.paste(thumb,(x,y+25));draw.text((x+5,y+5),path.name,fill='white')
        manifest.append(dict(name=path.name,original_size=original_size,working_size=rgb.size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    sheet.save(args.out/'contact.jpg');(args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
