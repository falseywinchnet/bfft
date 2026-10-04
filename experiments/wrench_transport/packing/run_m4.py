"""Run whole sizes sequentially on the selected Mini and verify copied receipts.

The mirror is read-only source. Result files live under our unique /tmp root.
Completed remote size directories are removed only after local hash validation.
"""
import argparse,hashlib,json,shlex,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--counts',default='64,128,256,512');p.add_argument('--sync',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parent;repo=root.parents[2];host=subprocess.check_output(['/Users/ultimussecundai/.local/bin/m4host'],text=True).strip();mirror='/Users/joshuahkuttenkuler/Developer/CodexBuilds/'+repo.name+'-'+hashlib.sha256(str(repo).encode()).hexdigest()[:12]
if a.sync:subprocess.run(['/Users/ultimussecundai/.local/bin/m4build','--sync-only'],cwd=repo,check=True)
for count in map(int,a.counts.split(',')):
 remote='/tmp/wrench-packing-count%d'%count;command='cd %s && python3 experiments/wrench_transport/packing/run_study.py --count %d --output %s'%(shlex.quote(mirror),count,shlex.quote(remote))
 print('BEGIN COUNT',count,flush=True);subprocess.run(['ssh',host,command],check=True)
 dest=root/'results'/('count%d'%count);dest.mkdir(parents=True,exist_ok=True);subprocess.run(['scp','-qr',host+':'+remote+'/.',str(dest)+'/'],check=True)
 manifest=json.loads((dest/'manifest.json').read_text())
 for name,expected in manifest.items():
  b=(dest/name).read_bytes();assert len(b)==expected['bytes'] and hashlib.sha256(b).hexdigest()==expected['sha256'],name
 print('VERIFIED LOCAL COUNT',count,len(manifest),'files',flush=True)
 # This exact directory was created above by this run and every file was verified.
 subprocess.run(['ssh',host,'rm -r -- '+shlex.quote(remote)],check=True)
