"""Sequential fixed-configuration runs; timing excludes scoring and serialization."""
import argparse,gzip,hashlib,json,os,platform,re,subprocess,time,signal
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--count',type=int,default=32);p.add_argument('--seeds',default='1,7,19');p.add_argument('--repeats',type=int,default=3);p.add_argument('--output',required=True);p.add_argument('--controls',action='store_true');p.add_argument('--demo',action='store_true');p.add_argument('--timeout',type=int,default=180);a=p.parse_args()
root=Path(__file__).resolve().parent;out=Path(a.output);out.mkdir(parents=True,exist_ok=True);node='/opt/homebrew/bin/node';native='/tmp/wrench_packing/packing_bench';modules='/Users/joshuahkuttenkuler/Developer/CodexBuilds/wrench_external'
configs=[('wrench60','wrench',60,0),('wrench120','wrench',120,0),('rapier60','rapier',60,4),('rapier240','rapier',240,8),('mujoco120','mujoco',120,.02),('mujoco480','mujoco',480,.005)]
records=[];failed_configs=set()
def call(cmd,timeout=None):
 process=subprocess.Popen([str(x) for x in cmd],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
 try: stdout,stderr=process.communicate(timeout=timeout)
 except subprocess.TimeoutExpired:
  os.killpg(process.pid,signal.SIGKILL);process.communicate();raise
 if process.returncode:raise subprocess.CalledProcessError(process.returncode,cmd,stdout,stderr)
 return subprocess.CompletedProcess(cmd,process.returncode,stdout,stderr)
scenes=[]
if a.controls:
 base=out/'base.json';call([node,root/'scene.mjs','--count','32','--seed','1','--out',base]);call(['python3',root/'controls.py',base,out/'scenes']);scenes=sorted((out/'scenes').glob('*.json'));configs += [('wrench480','wrench',480,0)];repeats=0
else:
 for seed in map(int,a.seeds.split(',')):
  scene=out/('seed%d.json'%seed);call([node,root/'scene.mjs','--count',a.count,'--seed',seed,'--out',scene,*(['--mode','demo'] if a.demo else [])]);scenes.append(scene)
 repeats=a.repeats
for scene in scenes:
 data=json.loads(scene.read_text());name=scene.stem
 for repeat in range(repeats+1):
  offset=(repeat+data['seed'])%len(configs);order=configs[offset:]+configs[:offset]
  for label,engine,hz,setting in order:
   key='%s-%s-r%d'%(name,label,repeat);raw=out/(key+'.json');scored=out/(key+'-score.json');log=out/(key+'.log');record=dict(scene=name,count=data['count'],seed=data['seed'],repeat=repeat,warmup=(repeat==0 and not a.controls),label=label,status='failed')
   if engine=='wrench':cmd=[native,scene.with_suffix('.txt'),raw,hz]
   elif engine=='rapier':cmd=[node,root/'rapier_runner.mjs',scene,raw,hz,setting,modules]
   else:cmd=['python3',root/'mujoco_runner.py',scene,raw,'--hz',hz,'--solref',setting]
   if (name,label) in failed_configs:
    record.update(status='not-run',reason='earlier run of this scene/configuration failed or exceeded the time budget');records.append(record);continue
   start=time.monotonic()
   try:
    run=call(['/usr/bin/time','-l',*cmd],a.timeout);log.write_text(run.stdout+'\n'+run.stderr);record['processSeconds']=time.monotonic()-start
    rss=re.search(r'(\d+)\s+maximum resident set size',run.stderr);record['peakRssBytes']=int(rss.group(1)) if rss else None
    call([node,root/'score.mjs',scene,raw,scored],a.timeout);metrics=json.loads(scored.read_text());record.update(status='ok',metrics=metrics)
    if a.controls:
     result=json.loads(raw.read_text());poses=result['samples'][-1]['poses'];z=poses[0][2];oracle=data['oracle'];record['oracle']=oracle;record['finalCentreZ']=z
     if oracle['kind']=='freefall':record['positionErrorMm']=1000*abs(z-oracle['expectedZ'])
     elif oracle['kind']=='cavity':record['insideCavity']=oracle['bottomTop']-.002<z<oracle['rimZ']-.005
     else:record['caughtBySlab']=z>oracle['topZ']-.001
    with raw.open('rb') as source,gzip.open(str(raw)+'.gz','wb',compresslevel=6) as dest:
     import shutil;shutil.copyfileobj(source,dest)
    raw.unlink()
   except subprocess.TimeoutExpired as e:
    record.update(status='timeout',timeoutSeconds=a.timeout,processSeconds=time.monotonic()-start);log.write_text('TIMEOUT\n'+str(e))
   except Exception as e:
    record['error']=str(e)
    if isinstance(e,subprocess.CalledProcessError):log.write_text((e.stdout or '')+'\n'+(e.stderr or ''))
   if record['status']!='ok':failed_configs.add((name,label))
   records.append(record);(out/'progress.json').write_text(json.dumps(dict(platform=platform.platform(),records=records),indent=2))
   m=record.get('metrics',{});print(key,record['status'],'step_s',m.get('wallSeconds'),'final_mm',m.get('final',{}).get('overlapMm'),'motion_mm',m.get('maxSampledOverlapMm'),flush=True)
(out/'progress.json').write_text(json.dumps(dict(platform=platform.platform(),records=records),indent=2))
(out/'sources.json').write_text(json.dumps({str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in root.rglob('*') if f.is_file() and 'results' not in f.parts and '__pycache__' not in f.parts},indent=2))
manifest={str(f.relative_to(out)):dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in out.rglob('*') if f.is_file() and f.name!='manifest.json'}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
