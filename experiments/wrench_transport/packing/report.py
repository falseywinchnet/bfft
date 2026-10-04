"""Aggregate saved runs without changing measurements; make paired error/cost plots."""
import gzip,json,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
names={'wrench60':'Wrench · 60 Hz','wrench120':'Wrench · 120 Hz','wrench480':'Wrench · 480 Hz','rapier60':'Rapier · 60 Hz / 4 iterations','rapier240':'Rapier · 240 Hz / 8 iterations','mujoco120':'MuJoCo · 120 Hz','mujoco480':'MuJoCo · 480 Hz / stiff'}
colors={'wrench60':'#126878','wrench120':'#52a59b','rapier60':'#be873b','rapier240':'#d6af6b','mujoco120':'#80649c','mujoco480':'#ad92c3'}
rows=[];counts=[];records=[]
for folder in sorted((root/'results').glob('count*'),key=lambda p:int(p.name[5:])):
 if not (folder/'manifest.json').exists():continue
 count=int(folder.name[5:]);counts.append(count);allrecords=json.loads((folder/'progress.json').read_text())['records'];records+=allrecords
 for label in colors:
  chosen=[r for r in allrecords if r['label']==label and not r['warmup']];good=[r for r in chosen if r['status']=='ok'];m=[r['metrics'] for r in good]
  work=[]
  if label.startswith('wrench'):
   for r in good:
    path=folder/(r['scene']+'-'+label+'-r%d.json.gz'%r['repeat'])
    work.append(json.load(gzip.open(path)).get('work',{}))
  row=dict(count=count,label=label,name=names[label],completed=len(good),planned=9,statuses={status:sum(r['status']==status for r in chosen) for status in ['ok','timeout','failed','not-run']},timeoutSeeds=sorted({r['seed'] for r in allrecords if r['label']==label and r['status']=='timeout'}))
  if m:
   row.update(activeMeanMs=statistics.median(r['activeMeanMs'] for r in m),activeSeconds=statistics.median(r['activeSeconds'] for r in m),wallSeconds=statistics.median(r['wallSeconds'] for r in m),p95Ms=statistics.median(r['timing']['p95Ms'] for r in m),p99Ms=statistics.median(r['timing']['p99Ms'] for r in m),worstStepMs=max(r['timing']['maxMs'] for r in m),finalOverlapMm=max(r['final']['overlapMm'] for r in m),motionOverlapMm=max(r['maxSampledOverlapMm'] for r in m),settledSeeds=len({r['seed'] for r in m if r['settledAt'] is not None}),maxEscaped=max(r['maxEscaped'] for r in m),finalEscaped=max(r['final']['escaped'] for r in m),maxLateSpeed=max(r['lastSecondMaxSpeed'] for r in m),maxMassRelativeError=max(r['maxMassRelativeError'] for r in m),maxGuards=max((r['guards'] or 0) for r in m),maxWarningCount=max(sum(r['warnings'] or []) for r in m),peakRssBytes=max((r['peakRssBytes'] or 0) for r in good),parts=m[0]['parts'],maxUnconverged=max((r.get('unconverged',0) for r in work),default=0),activeSecondsMin=min(r['activeSeconds'] for r in m),activeSecondsMax=max(r['activeSeconds'] for r in m))
  else:row.update({k:None for k in ['activeMeanMs','activeSeconds','wallSeconds','p95Ms','p99Ms','worstStepMs','finalOverlapMm','motionOverlapMm','settledSeeds','maxEscaped','finalEscaped','maxLateSpeed','maxMassRelativeError','maxGuards','maxWarningCount','peakRssBytes','parts']})
  rows.append(row)
controls=[];cr=root/'results'/'controls';control_records=json.loads((cr/'progress.json').read_text())['records']
for label in [*colors,'wrench480']:
 subset=[r for r in control_records if r['label']==label];entry=dict(label=label,name=names[label],slab={})
 for r in subset:
  if r['scene']=='cavity':entry['cavity']=r['insideCavity']
  elif r['scene']=='freefall':entry['freefallMm']=r['positionErrorMm']
  else:
   data=json.load(gzip.open(cr/(r['scene']+'-'+label+'-r0.json.gz')));scene=json.load(open(cr/'scenes'/(r['scene']+'.json')));slab=scene['walls'][0];mid=slab['p'][2];radius=scene['shapes'][scene['bodies'][0]['shape']]['radius'];previous=scene['bodies'][0]['p'];tunneled=False;events=[]
   for sample in data['samples']:
    pose=sample['poses'][0]
    if previous[2]>mid>=pose[2]:
     u=(previous[2]-mid)/(previous[2]-pose[2]);xy=[previous[k]+u*(pose[k]-previous[k]) for k in range(2)];interior=all(abs(xy[k]-slab['p'][k])<slab['size'][k]/2-radius for k in range(2));events.append(dict(t=sample['t'],xy=xy,interior=interior));tunneled|=interior
    previous=pose
   speed=int(r['scene'].split('-')[1]);entry['slab'][speed]=tunneled;entry.setdefault('crossings',{})[speed]=events
 controls.append(entry)
summary=dict(schema='compound-packing-summary-v1',counts=counts,complete=counts==[32,64,128,256,512],rows=rows,controls=controls,protocol='../PROTOCOL.md')
(root/'viewer'/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#f1f4ed','axes.facecolor':'#f1f4ed','axes.labelcolor':'#21484e','text.color':'#21484e','xtick.color':'#466269','ytick.color':'#466269','axes.edgecolor':'#a0b5b3','grid.color':'#cad6d0'})
fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
fields=[('activeMeanMs','Mean simulation step after all releases (ms)'),('activeSeconds','Compute for eight all-active seconds (s)'),('finalOverlapMm','Worst final penetration (mm)'),('motionOverlapMm','Worst sampled motion penetration (mm)')]
for ax,(field,title) in zip(axs.flat,fields):
 for label,color in colors.items():
  data=[r for r in rows if r['label']==label and r['completed']==9]
  if data:
   ax.plot([r['count'] for r in data],[max(r[field],1e-7) for r in data],'-',label=names[label],color=color,lw=1.7)
   for r in data:ax.plot(r['count'],max(r[field],1e-7),marker='x' if r['maxEscaped'] or r['maxWarningCount'] or r['maxGuards'] else 'o',color=color,ms=6,mew=1.5)
   if field=='activeSeconds':ax.fill_between([r['count'] for r in data],[r['activeSecondsMin'] for r in data],[r['activeSecondsMax'] for r in data],color=color,alpha=.08)
 ax.set_title(title,loc='left',fontweight='bold',pad=12);ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_xticks([32,64,128,256,512],[32,64,128,256,512]);ax.set_xlim(28,580);ax.grid(True,alpha=.6);ax.set_xlabel('Dynamic objects')
axs[0,1].axhline(8,color='#8b9896',ls=':',lw=1);axs[0,1].text(34,8*1.12,'8 s = real-time limit',fontsize=8,color='#687c7e')
handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False,fontsize=9)
fig.suptitle('Hollow-object packing: cost and collision error together',fontweight='bold',fontsize=17,x=.02,ha='left')
fig.savefig(root/'viewer'/'scaling.png',dpi=160);fig.savefig(root/'viewer'/'scaling.svg');plt.close(fig)
lines=['# Compound packing: measured results','', 'This report is generated from saved runs. See `PROTOCOL.md` for source attribution, scene construction and measurement definitions.','', '| Bodies | Configuration | Completed / planned | Active mean step (ms) | Active compute (s) | Final overlap (mm) | Motion overlap (mm) | Settled seeds | Max escaped |','|---:|---|---:|---:|---:|---:|---:|---:|---:|']
def fmt(x):return '—' if x is None else '%.5g'%x
for r in rows:lines.append('| %d | %s | %d / 9 | %s | %s | %s | %s | %s / 3 | %s |'%(r['count'],r['name'],r['completed'],fmt(r['activeMeanMs']),fmt(r['activeSeconds']),fmt(r['finalOverlapMm']),fmt(r['motionOverlapMm']),fmt(r['settledSeeds']),fmt(r['maxEscaped'])))
lines+=['','A cross marker in the plots means at least one body escaped or an engine reported numerical instability at that size. Its timing is not successful-packing throughput. Compute shading spans the measured minimum and maximum; lines use medians. Only complete nine-run points are plotted.','', '## Controlled checks','','“Tunneled” means a centre trajectory crosses the slab midplane inside its footprint, with an additional body-radius margin from every edge. Every solver step is exported. Exits over an edge are retained separately.','', '| Configuration | Cup cavity | 1 m/s slab | 10 m/s slab | 30 m/s slab | Free-fall error (mm) |','|---|---|---|---|---|---:|']
for r in controls:lines.append('| %s | %s | %s | %s | %s | %.5g |'%(r['name'],'passed' if r['cavity'] else 'failed',*['tunneled' if r['slab'][s] else 'caught / edge exit' for s in [1,10,30]],r['freefallMm']))
lines+=['','Free fall is measured for 0.5 seconds against the analytic centre-of-mass trajectory. Passing these controlled checks does not establish universal collision accuracy.','', '## Runtime limits','','A run has a 180-second process budget. Scoring has a separate 180-second budget. A failed/timed-out scene/configuration is not retried in its measured repetitions; those records remain explicitly not run. Partial results are not counted as completed simulations.','']
(root/'RESULTS.md').write_text('\n'.join(lines)+'\n');print('Reported counts:',counts)
