"""Create independent physical checks from the common exported geometry."""
import copy,json,sys
from pathlib import Path
base=json.load(open(sys.argv[1]));out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
def save(name,s):
 s['name']=name;s['seed']=0;s['count']=len(s['bodies']);s['releaseEnd']=0
 (out/(name+'.json')).write_text(json.dumps(s,separators=(',',':')))
 rows=['PACK1',' '.join(map(str,[s['density'],s['friction'],*s['gravity'],s['seconds'],s['sampleRate'],s['geometryRate'],len(s['shapes'])]))]
 for sh in s['shapes']:
  rows.append(str(len(sh['hulls'])))
  for h in sh['hulls']:rows.append(str(len(h['vertices'])));rows.extend(' '.join(map(str,v)) for v in h['vertices'])
 rows.append(str(len(s['walls'])));rows.extend(' '.join(map(str,[*w['p'],*w['size']])) for w in s['walls'])
 rows.append(str(len(s.get('fixedBodies',[]))));rows.extend(' '.join(map(str,[b['shape'],*b['p'],*b['q']])) for b in s.get('fixedBodies',[]))
 rows.append(str(s['count']));rows.extend(' '.join(map(str,[b['shape'],b['release'],*b['p'],*b['q'],*b['v'],*b['w']])) for b in s['bodies'])
 (out/(name+'.txt')).write_text('\n'.join(rows)+'\n')
def scene(z,velocity,seconds):
 s=copy.deepcopy(base);s.update(walls=[],fixedBodies=[],inner=1000,height=1000,seconds=seconds,sampleRate=480,geometryRate=480,bodies=[dict(shape=5,release=0,p=[0,0,z],q=[1,0,0,0],v=[0,0,velocity],w=[0,0,0])]);return s
s=scene(100,0,.5);s['oracle']={'kind':'freefall','expectedZ':100-.5*9.81*.5**2};save('freefall',s)
cup=base['shapes'][0];minimum=min(v[2] for h in cup['hulls'] for v in h['vertices']);bottomtop=max(v[2] for v in cup['hulls'][0]['vertices'])-minimum;ballmin=min(v[2] for h in base['shapes'][5]['hulls'] for v in h['vertices'])
s=scene(.25,0,3);s['fixedBodies']=[dict(shape=0,p=[0,0,-minimum],q=[1,0,0,0])];s['oracle']={'kind':'cavity','bottomTop':bottomtop,'expectedMinimumCentreZ':bottomtop-ballmin,'rimZ':max(v[2] for h in cup['hulls'] for v in h['vertices'])-minimum};save('cavity',s)
for speed in [1,10,30]:
 s=scene(.5,-speed,2);s['walls']=[dict(p=[0,0,.15],size=[.3,.3,.004])];s['oracle']={'kind':'thin-slab','topZ':.152,'expectedMinimumCentreZ':.152-ballmin,'impactSpeed':speed};save('slab-%d'%speed,s)
