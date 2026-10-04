#!/usr/bin/env python3
"""Compare the finished native engine with fixed MuJoCo/Rapier configurations."""
import argparse, json, platform, statistics, subprocess, sys
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--native', required=True)
p.add_argument('--node', default='node')
p.add_argument('--modules', required=True)
p.add_argument('--output', required=True)
p.add_argument('--repeats', type=int, default=3)
a=p.parse_args(); root=Path(__file__).resolve().parent; out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
seeds=[1,7,19]; records=[]
for seed in seeds:
    directory=out/f'seed{seed}'; directory.mkdir(exist_ok=True)
    scene=directory/'scene.json'; text=directory/'scene.txt'
    subprocess.run([a.node,str(root/'container.mjs'),'--seed',str(seed),'--export',str(scene),'--export-text',str(text)],check=True,capture_output=True)
    common=[str(scene)]
    configs={
      'wrench60':[a.native,str(text),'{out}','60','8'],
      'wrench120':[a.native,str(text),'{out}','120','8'],
      'rapier60':[a.node,str(root/'external/run_rapier.mjs'),str(scene),'{out}','--steps-per-sample','1','--iterations','4','--modules',a.modules],
      'rapier60scaled':[a.node,str(root/'external/run_rapier.mjs'),str(scene),'{out}','--steps-per-sample','1','--iterations','4','--length-unit','.05','--modules',a.modules],
      'rapier240scaled':[a.node,str(root/'external/run_rapier.mjs'),str(scene),'{out}','--steps-per-sample','4','--iterations','8','--length-unit','.05','--modules',a.modules],
      'mujoco60':[sys.executable,str(root/'external/run_mujoco.py'),str(scene),'{out}','--steps-per-sample','1','--multiccd'],
      'mujoco120':[sys.executable,str(root/'external/run_mujoco.py'),str(scene),'{out}','--steps-per-sample','2','--multiccd'],
      'mujoco480stiff':[sys.executable,str(root/'external/run_mujoco.py'),str(scene),'{out}','--steps-per-sample','8','--multiccd','--solref','.005'],
    }
    for repeat in range(a.repeats+1):
      labels=list(configs); offset=repeat%len(labels); labels=labels[offset:]+labels[:offset]
      paths=[]
      for label in labels:
        result=directory/f'{label}-{repeat}.json'; cmd=[str(result) if x=='{out}' else x for x in configs[label]]
        process=subprocess.run(cmd,check=True,text=True,capture_output=True)
        paths.append(result)
        print(f'seed {seed}, repeat {repeat}, {label}: '+process.stdout.strip(),flush=True)
      report=directory/f'report-{repeat}.json'
      process=subprocess.run([a.node,str(root/'container_report.mjs'),'--seed',str(seed),'--out',str(report),*[str(x) for x in paths]],check=True,text=True,capture_output=True)
      (directory/f'report-{repeat}.md').write_text(process.stdout)
      if repeat:
        scores=json.loads(report.read_text())['scores']
        for label,score in zip(labels,scores): records.append(dict(seed=seed,repeat=repeat,label=label,**score))
      (out/'progress.json').write_text(json.dumps(dict(records=records),indent=2))
summary=[]
for label in configs:
    rows=[r for r in records if r['label']==label]
    summary.append(dict(label=label,engine=rows[0]['engine'],config=rows[0]['config'],medianWallSeconds=statistics.median(r['wallSeconds'] for r in rows),meanWallSeconds=statistics.mean(r['wallSeconds'] for r in rows),maxOverlapMm=max(r['deepestOverlapMm'] for r in rows),medianOverlapMm=statistics.median(r['deepestOverlapMm'] for r in rows),settledSeeds=len(set(r['seed'] for r in rows if r['settledAt']>=0)),maxLateSpeed=max(r['lastSecondMaxSpeed'] for r in rows),escaped=max(r['escaped'] for r in rows)))
payload=dict(platform=platform.platform(),seeds=seeds,repeats=a.repeats,warmups=1,seconds=8,records=records,summary=summary)
(out/'comparison.json').write_text(json.dumps(payload,indent=2)+'\n')
print(json.dumps(summary,indent=2),flush=True)
