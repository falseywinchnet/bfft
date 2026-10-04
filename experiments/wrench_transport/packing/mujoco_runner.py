import argparse,json,time
import mujoco
import numpy as np
p=argparse.ArgumentParser();p.add_argument('scene');p.add_argument('out');p.add_argument('--hz',type=int,default=120);p.add_argument('--solref',type=float,default=.02);a=p.parse_args()
s=json.load(open(a.scene));setup=time.perf_counter();hz=a.hz
xml=['<mujoco model="compound-packing"><compiler inertiafromgeom="false"/><option timestep="%.15g" gravity="%s" cone="elliptic" solver="Newton"><flag multiccd="enable"/></option>'%(1/hz,' '.join(map(str,s['gravity']))),'<default><geom friction=".5 0 0" condim="3" solref="%g 1"/></default><asset>'%a.solref]
for k,sh in enumerate(s['shapes']):
 for h,part in enumerate(sh['hulls']):xml.append('<mesh name="m%d_%d" vertex="%s"/>'%(k,h,' '.join(str(v) for pt in part['vertices'] for v in pt)))
xml+=['</asset><worldbody><geom name="floor" type="plane" size="5 5 .1"/>']
for i,w in enumerate(s['walls']):xml.append('<geom type="box" pos="%s" size="%s"/>'%(' '.join(map(str,w['p'])),' '.join(str(v/2) for v in w['size'])))
for i,b in enumerate(s.get('fixedBodies',[])):
 xml.append('<body pos="%s" quat="%s">'%(' '.join(map(str,b['p'])),' '.join(map(str,b['q']))))
 for h in range(len(s['shapes'][b['shape']]['hulls'])):xml.append('<geom type="mesh" mesh="m%d_%d"/>'%(b['shape'],h))
 xml.append('</body>')
for i,b in enumerate(s['bodies']):
 sh=s['shapes'][b['shape']];I=sh['inertia'];xml.append('<body name="b%d" pos="0 0 %g" gravcomp="1"><freejoint/><inertial pos="0 0 0" mass="%.15g" fullinertia="%s"/>'%(i,10+i*.2,sh['mass'],' '.join(str(I[j]) for j in [0,4,8,1,2,5])))
 for h in range(len(sh['hulls'])):xml.append('<geom name="g%d_%d" type="mesh" mesh="m%d_%d" contype="1" conaffinity="1"/>'%(i,h,b['shape'],h))
 xml.append('</body>')
xml.append('</worldbody></mujoco>');model=mujoco.MjModel.from_xml_string(''.join(xml));d=mujoco.MjData(model)
ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'b%d'%i) for i in range(s['count'])]
for bid in ids:
 startg=model.body_geomadr[bid];endg=startg+model.body_geomnum[bid];model.geom_contype[startg:endg]=0;model.geom_conaffinity[startg:endg]=0
active=np.zeros(s['count'],dtype=bool);radii=np.array([s['shapes'][b['shape']]['radius'] for b in s['bodies']]);samples=[];times=[];late=0;late_steps=0;activation=0
setup_seconds=time.perf_counter()-setup
for f in range(round(s['seconds']*hz)):
 start=time.perf_counter()
 for i,b in enumerate(s['bodies']):
  if not active[i] and b['release']<=f/hz+1e-9:
   active[i]=True;bid=ids[i];model.body_gravcomp[bid]=0;startg=model.body_geomadr[bid];endg=startg+model.body_geomnum[bid];model.geom_contype[startg:endg]=1;model.geom_conaffinity[startg:endg]=1
   d.qpos[i*7:i*7+7]=b['p']+b['q'];d.qvel[i*6:i*6+6]=b['v']+b['w']
 activation+=time.perf_counter()-start
 start=time.perf_counter();mujoco.mj_step(model,d);cost=time.perf_counter()-start;times.append(cost*1000)
 if f/hz>=s['releaseEnd']-1e-9:late+=cost;late_steps+=1
 if (f+1)%max(1,hz//s['sampleRate'])==0:
  vel=d.qvel.reshape(-1,6);speed=np.linalg.norm(vel[:,:3],axis=1)+radii*np.linalg.norm(vel[:,3:],axis=1);row={'t':(f+1)/hz,'maxSpeed':float(speed[active].max()) if active.any() else 0,'contacts':int(d.ncon)}
  if (f+1)%max(1,hz//s['geometryRate'])==0 or f+1==round(s['seconds']*hz):row['poses']=[d.qpos[i*7:i*7+7].tolist() if active[i] else None for i in range(s['count'])]
  samples.append(row)
result={'engine':'mujoco '+mujoco.__version__,'hz':hz,'solref':a.solref,'setupSeconds':setup_seconds,'activationSeconds':activation,'masses':[next((float(model.body_mass[ids[i]]) for i,b in enumerate(s['bodies']) if b['shape']==k),None) for k in range(len(s['shapes']))],'samples':samples,'wallSeconds':sum(times)/1000,'activeSeconds':late,'activeSteps':late_steps,'timing':{k:float(np.quantile(times,q)) for k,q in [('p50Ms',.5),('p95Ms',.95),('p99Ms',.99),('maxMs',1)]},'warnings':[int(w.number) for w in d.warning]}
json.dump(result,open(a.out,'w'),separators=(',',':'));print(result['engine'],s['count'],hz,result['wallSeconds'],flush=True)
