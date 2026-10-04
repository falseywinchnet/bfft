"""Validate receipts and deterministic trajectories without treating physical failures as passes."""
import gzip,hashlib,json,math
from pathlib import Path
root=Path(__file__).resolve().parent;audits=[]
for folder in sorted((root/'results').glob('count*')):
 if not(folder/'manifest.json').exists():continue
 for name,expected in json.load(open(folder/'manifest.json')).items():
  contents=(folder/name).read_bytes();assert len(contents)==expected['bytes'] and hashlib.sha256(contents).hexdigest()==expected['sha256'],str(folder/name)
 records=json.load(open(folder/'progress.json'))['records'];assert len(records)==72,(folder,len(records))
 hashes={};worst_mass=0;worst_quat=0;warnings=[]
 for r in records:
  if r['status']!='ok':continue
  scene=json.load(open(folder/(r['scene']+'.json')));data=json.load(gzip.open(folder/(r['scene']+'-'+r['label']+'-r%d.json.gz'%r['repeat'])))
  assert len(data['samples'])==round(scene['seconds']*60)
  assert abs(data['samples'][-1]['t']-scene['seconds'])<1e-8
  worst_mass=max(worst_mass,r['metrics']['maxMassRelativeError']);assert r['metrics']['maxMassRelativeError']<1e-5
  for sample in data['samples']:
   if 'poses' not in sample:continue
   assert len(sample['poses'])==scene['count']
   for i,pose in enumerate(sample['poses']):
    assert (pose is None)==(scene['bodies'][i]['release']>=sample['t']-1e-9)
    if pose is not None:
     assert all(isinstance(v,(float,int)) and math.isfinite(v) for v in pose)
     worst_quat=max(worst_quat,abs(sum(v*v for v in pose[3:])-1))
  assert worst_quat<1e-4
  digest=hashlib.sha256(json.dumps(data['samples'],sort_keys=True,separators=(',',':')).encode()).hexdigest();key=r['scene']+'-'+r['label'];hashes.setdefault(key,set()).add(digest)
  if sum(data.get('warnings',[])) or data.get('guards',0):warnings.append(key)
 nondeterministic=[key for key,value in hashes.items() if len(value)>1]
 assert not nondeterministic,nondeterministic
 audits.append(dict(count=int(folder.name[5:]),records=len(records),completed=sum(r['status']=='ok' for r in records),deterministicSceneConfigurations=len(hashes),maxRelativeMassDifference=worst_mass,maxQuaternionNormSquaredError=worst_quat,warningRuns=warnings,receiptHashesVerified=True))
(root/'audits.json').write_text(json.dumps(audits,indent=2)+'\n');print(json.dumps(audits,indent=2))
