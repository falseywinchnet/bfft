"""Copy common input geometry and the independent oracle, with byte provenance."""
import argparse,hashlib,json,shutil,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('packing',type=Path);args=p.parse_args()
root=Path(__file__).resolve().parent;fixtures=root/'fixtures';fixtures.mkdir(exist_ok=True)
provenance={}
def copy(source,destination):
    destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,destination)
    provenance[str(destination.relative_to(root))]={'source':str(source.relative_to(args.packing)),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
for count in [32,64,128,256,512]:
    source=args.packing/'results'/('count%d'%count)
    if not source.exists() and count==512:source=args.packing/'results'/'interrupted512'
    for seed in [1,7,19]:
        for suffix in ['json','txt']:copy(source/('seed%d.%s'%(seed,suffix)),fixtures/('count%d'%count)/('seed%d.%s'%(seed,suffix)))
for source in (args.packing/'results'/'controls'/'scenes').glob('*'):
    if source.suffix in ('.json','.txt'):copy(source,fixtures/'controls'/source.name)
for name in ['score.mjs','test_score.mjs']:copy(args.packing/name,root/'scoring'/name)
# The narrow visual reconstruction is a separate diagnostic, not a scaling point.
(fixtures/'demo').mkdir(exist_ok=True)
subprocess.run(['node',str(args.packing/'scene.mjs'),'--count','32','--seed','1','--mode','demo','--out',str(fixtures/'demo'/'seed1.json')],check=True)
for source in (fixtures/'demo').glob('*'):
    provenance[str(source.relative_to(root))]={'source':'scene.mjs --count 32 --seed 1 --mode demo','sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
(fixtures/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
