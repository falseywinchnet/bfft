"""Sequential native runs, preserved failures, and independently scored poses.

Run on the Mini. Identical scene + complete trajectory hashes may reuse geometry
scores; every run retains its own timing and raw state record.
"""
import argparse,gzip,hashlib,json,os,platform,re,shutil,signal,subprocess,time,tarfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--counts',default='32,64,128,256,512');p.add_argument('--seeds',default='1,7,19');p.add_argument('--repeats',type=int,default=3);p.add_argument('--output',required=True);p.add_argument('--modes',default='adaptive');p.add_argument('--controls',action='store_true');p.add_argument('--demo',action='store_true');p.add_argument('--trace',action='store_true');p.add_argument('--timeout',type=int,default=180);args=p.parse_args()
root=Path(__file__).resolve().parent;out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
if (out/'progress.json').exists():raise SystemExit('Use a fresh output directory; existing results will not be overwritten.')
records=[];failed=set();geometry_cache={};node='/opt/homebrew/bin/node';binary='/tmp/obligation_dynamics/obligation_bench'
source_hashes={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in root.rglob('*') if f.is_file() and not any(x in f.parts for x in ['results','build','__pycache__'])}
for f in sorted((root.parent/'wrench_transport'/'phys').glob('*')):
    if f.is_file() and f.suffix in ('.cpp','.hpp','.txt'):source_hashes['../wrench_transport/phys/'+f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
source_hashes['binary:obligation_bench']=hashlib.sha256(Path(binary).read_bytes()).hexdigest()
(out/'sources.json').write_text(json.dumps(source_hashes,indent=2)+'\n')
with tarfile.open(out/'source-snapshot.tar.gz','w:gz') as archive:
    for relative in source_hashes:
        if relative.startswith('binary:'):continue
        path=(root/relative).resolve()
        archive.add(path,arcname=str(path.relative_to(root.parent)),recursive=False)
def command(cmd,timeout):
    process=subprocess.Popen(list(map(str,cmd)),stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
    try:stdout,stderr=process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGKILL);process.communicate();raise
    if process.returncode:raise subprocess.CalledProcessError(process.returncode,cmd,stdout,stderr)
    return stdout,stderr
def progress():
    (out/'progress.json').write_text(json.dumps({'platform':platform.platform(),'records':records},indent=2)+'\n')
cases=[]
if args.controls:
    cases=[('controls-'+f.stem,f) for f in sorted((root/'fixtures'/'controls').glob('*.json'))]
elif args.demo:cases=[('demo-seed1',root/'fixtures'/'demo'/'seed1.json')]
else:
    for count in map(int,args.counts.split(',')):
        for seed in map(int,args.seeds.split(',')):cases.append(('count%d-seed%d'%(count,seed),root/'fixtures'/('count%d'%count)/('seed%d.json'%seed)))
repeats=0 if args.controls else args.repeats
for case,scene_path in cases:
    scene=json.loads(scene_path.read_text());scene_hash=hashlib.sha256(scene_path.read_bytes()).hexdigest()
    for repeat in range(repeats+1):
        modes=args.modes.split(',');offset=(scene['seed']+repeat)%len(modes);modes=modes[offset:]+modes[:offset]
        for mode in modes:
            if mode not in ('adaptive','awake'):raise ValueError(mode)
            key='%s-%s-r%d'%(case,mode,repeat);raw=out/(key+'.json');score_path=out/(key+'-score.json');log=out/(key+'.log')
            record={'case':case,'count':scene['count'],'seed':scene['seed'],'mode':mode,'repeat':repeat,'warmup':repeat==0 and not args.controls,'status':'failed'}
            if (case,mode) in failed:
                record.update(status='not-run',reason='earlier attempt of this scene/configuration failed or exceeded its budget');records.append(record);progress();continue
            hz=480 if args.controls else 60;extra=['--no-equilibrium'] if mode=='awake' else []
            if args.trace:extra.append('--trace')
            stage='simulation';start=time.monotonic()
            try:
                stdout,stderr=command(['/usr/bin/time','-l',binary,scene_path.with_suffix('.txt'),raw,hz,*extra],args.timeout)
                log.write_text(stdout+'\n'+stderr);record['processSeconds']=time.monotonic()-start
                rss=re.search(r'(\d+)\s+maximum resident set size',stderr);record['peakRssBytes']=int(rss.group(1)) if rss else None
                data=json.loads(raw.read_text());trajectory_hash=hashlib.sha256(json.dumps(data['samples'],sort_keys=True,separators=(',',':')).encode()).hexdigest();cache_key=(scene_hash,trajectory_hash)
                record['trajectorySha256']=trajectory_hash;stage='geometry scoring'
                if cache_key in geometry_cache:
                    metrics=json.loads(json.dumps(geometry_cache[cache_key][1]));metrics.update(wallSeconds=data['wallSeconds'],activeSeconds=data['activeSeconds'],activeMeanMs=data['activeSeconds']/data['activeSteps']*1000,setupSeconds=data['setupSeconds'],activationSeconds=data['activationSeconds'],timing=data['timing'],guards=data['guards'])
                    record['geometryReusedFrom']=geometry_cache[cache_key][0];score_path.write_text(json.dumps(metrics,indent=2)+'\n')
                else:
                    command([node,root/'scoring'/'score.mjs',scene_path,raw,score_path],args.timeout);metrics=json.loads(score_path.read_text());geometry_cache[cache_key]=(key,metrics)
                record.update(status='ok',metrics=metrics,work=data['work'],equilibriumBodies=sum(data['bodyEquilibrium']),maxEquilibriumResidual=max((r for q,r in zip(data['bodyEquilibrium'],data['forceResiduals']) if q),default=0))
                with raw.open('rb') as source,gzip.open(str(raw)+'.gz','wb',compresslevel=6) as target:shutil.copyfileobj(source,target)
                raw.unlink()
            except subprocess.TimeoutExpired as error:
                record.update(status='timeout',stage=stage,timeoutSeconds=args.timeout,processSeconds=time.monotonic()-start);log.write_text('TIMEOUT '+stage+'\n'+str(error))
            except Exception as error:
                record.update(error=str(error),stage=stage)
                if isinstance(error,subprocess.CalledProcessError):log.write_text((error.stdout or '')+'\n'+(error.stderr or ''))
            if record['status']!='ok':failed.add((case,mode))
            records.append(record);progress();m=record.get('metrics',{})
            print(key,record['status'],'compute',m.get('wallSeconds'),'motion_mm',m.get('maxSampledOverlapMm'),'final_mm',m.get('final',{}).get('overlapMm'),'escaped',m.get('maxEscaped'),'certified',record.get('equilibriumBodies'),flush=True)
manifest={str(f.relative_to(out)):{'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in out.rglob('*') if f.is_file() and f.name!='manifest.json'}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
