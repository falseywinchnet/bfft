"""Bundle the dependency-free mechanism as one embeddable/offline HTML file."""
from pathlib import Path
p=Path(__file__).resolve().parent
s=(p/'page.html').read_text()
for token,name in [('/*__STYLE__*/','style.css'),('/*__MECHANICS__*/','mechanics.js'),('/*__ASSEMBLY__*/','assembly.js'),('/*__APP__*/','app.js')]:
    assert s.count(token)==1
    s=s.replace(token,(p/name).read_text())
(p/'index.html').write_text(s)
print(p/'index.html')
