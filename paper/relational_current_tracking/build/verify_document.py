from pathlib import Path
import re,json
import pypdfium2 as pdfium
from pypdf import PdfReader
from PIL import Image,ImageDraw
root=Path(__file__).resolve().parents[1]
p=root/'build'
reader=PdfReader(p/'main.pdf')
text='\n'.join(x.extract_text() for x in reader.pages)
assert len(reader.pages)>=9
assert '??' not in text
for term in ['Exact Markov Elimination','Theorem 1','TABLE X','Robust IMM','6144','37.2','12.55']:
    assert term in text,term
log=(p/'main.log').read_text()
assert 'Overfull' not in log
assert not re.search(r'(Citation|Reference).*undefined|There were undefined|LaTeX Error|Missing character',log)
source=(root/'main.tex').read_text()
refs=set(re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',source))
labels=set(re.findall(r'\\label\{([^}]+)\}',source));assert not refs-labels
cites={v for x in re.findall(r'\\cite\{([^}]+)\}',source) for v in x.split(',')}
bibs=set(re.findall(r'\\bibitem\{([^}]+)\}',source));assert cites==bibs
for name in re.findall(r'\\inputrows\{([^}]+)\}',source):assert (root/name).is_file()
for name in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',source):assert (root/'figures'/name).is_file()
doc=pdfium.PdfDocument(p/'main.pdf');thumbs=[]
for i,page in enumerate(doc):
    im=page.render(scale=1.5).to_pil();im.save(p/f'page-{i+1:02d}.png')
    im.thumbnail((425,550));can=Image.new('RGB',(445,580),'#dddddd');can.paste(im,((445-im.width)//2,20));ImageDraw.Draw(can).text((10,560),str(i+1),fill='black');thumbs.append(can)
sheet=Image.new('RGB',(1335,580*((len(thumbs)+2)//3)),'white')
for i,im in enumerate(thumbs):sheet.paste(im,((i%3)*445,(i//3)*580))
sheet.save(p/'contact.png')
report=dict(pages=len(reader.pages),references=len(bibs),figures=3,tables=10,unresolved_references=0,overfull_boxes=0,missing_characters=0,compilation='Tectonic 0.17.0 / XeTeX',visual_review='Page images generated for inspection')
(p/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
