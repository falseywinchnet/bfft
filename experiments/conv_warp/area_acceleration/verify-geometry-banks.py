"""Validate compiled WASM weights against the paper's rational/log edge oracle."""
import argparse, hashlib, json, math, sys
from pathlib import Path
import mpmath as mp
p=argparse.ArgumentParser();p.add_argument('--bfft',type=Path,default=Path.home()/'bfft');p.add_argument('--input',type=Path,default=Path(__file__).parent/'results/geometry-banks.json');p.add_argument('--output',type=Path,default=Path(__file__).parent/'results/geometry-bank-check.json');args=p.parse_args()
sys.path.insert(0,str(args.bfft))
from experiments.conv_warp.projective_bank_proof import quintic_bank
mp.mp.dps=220
packet=json.loads(args.input.read_text());rules=json.loads((Path(__file__).parent/'triangle-rules.json').read_text())['rules']
# The positive rule exactly integrates B_i^5 B_j^5 times the first K+1
# terms of (1+z)^-3. Both exact integration and cubature of the remainder
# are bounded using partition of unity, without inspecting any controls.
threshold_checks=[]
for k,cutoff in enumerate(packet['cutoffs']):
 r=mp.mpf(str(cutoff));d=1-r
 relative=r**(k+1)*(r*(1+r)/d**3+(2*k+5)*r/d**2+(k+2)*(k+3)/d)*(1+r)**3
 assert relative<=mp.mpf('1e-9')
 rule=rules[k];assert all(u>=0 and v>=0 and u+v<=1+1e-15 and w>0 for u,v,w in rule['nodes'])
 max_moment=0
 for a in range(rule['degree']+1):
  for b in range(rule['degree']+1-a):
   exact=mp.factorial(a)*mp.factorial(b)/mp.factorial(a+b+2)
   actual=mp.fsum(mp.mpf(w)*mp.mpf(u)**a*mp.mpf(v)**b for u,v,w in rule['nodes'])
   max_moment=max(max_moment,float(abs(actual-exact)/exact))
 assert max_moment<3e-14
 threshold_checks.append({'degree':10+k,'radius':cutoff,'relativeOperatorBound':float(relative),'ruleMomentRelativeError':max_moment})
checks=[]
for case in packet['cases']:
 exact=quintic_bank(case['vertices'],case['gamma'],case['alpha'],case['beta'])
 reference=[exact[i][j] for j in range(6) for i in range(6)]
 mass=mp.fsum(reference)
 relative=float(mp.fsum(abs(mp.mpf(w)-e) for w,e in zip(case['weights'],reference))/mass)
 assert min(case['weights'])>=0
 assert relative<1e-9,(case['ti'],case['r'],relative)
 # An adversarial bounded control vector selects precisely the positive
 # weight defects. L1 controls every possible patch in [-1,1].
 checks.append({'triangle':case['ti'],'radius':case['r'],'relativeL1WeightError':relative,'relativeMassError':float(abs(sum(map(mp.mpf,case['weights']))-mass)/mass),'subdivisions':case['counts'][4],'ruleCounts':case['counts'][5:12]})
result={'precisionDecimalDigits':mp.mp.dps,'oracle':'BFFT experiments/conv_warp/projective_bank_proof.py:quintic_bank','oracleSHA256':hashlib.sha256((args.bfft/'experiments/conv_warp/projective_bank_proof.py').read_bytes()).hexdigest(),'inputSHA256':hashlib.sha256(args.input.read_bytes()).hexdigest(),'thresholds':threshold_checks,'cases':checks,'worstRelativeL1WeightError':max(c['relativeL1WeightError'] for c in checks)}
args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'cases':len(checks),'worstRelativeL1WeightError':result['worstRelativeL1WeightError']}))
