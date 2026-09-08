"""Keep the existing defect-gated update; randomize only its judgment."""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.meyer_transport_audit.model import ReducedMeyerMap
from experiments.meyer_transport_audit.defect_relaxation import DefectRelaxation,scenes
from experiments.meyer_transport_audit.decision_noise import best_gap


def alignment(previous,extra):
    dot=previous[0]*extra[0]+previous[1]*extra[1]
    norm=np.hypot(*previous)*np.hypot(*extra)
    return np.clip(np.divide(dot,norm,out=np.zeros_like(dot),where=norm>0),-1.,1.)


class NoisyDefectGate(DefectRelaxation):
    def __init__(self,m,mode,seed=11):
        super().__init__(m,'aligned_refresh')
        self.mode=mode;self.rng=np.random.default_rng(seed)
        self.threshold=0.;self.changed=0;self.compared=0

    def step(self,z):
        if self.mode=='shared':self.threshold=self.rng.uniform(-1.,1.)
        return super().step(z)

    def admit_defect(self,previous,extra):
        deterministic=super().admit_defect(previous,extra)
        if self.mode=='deterministic':return deterministic
        c=alignment(previous,extra)
        threshold=self.threshold if self.mode=='shared' else self.rng.uniform(-1.,1.,size=c.shape)
        return c>=threshold


def run(n,steps,seeds,out,ramp_only):
    sources={'ramp1d':255.*np.arange(n)[None,:]/n} if ramp_only else scenes(n,3029)
    result={'size':n,'steps':steps,'seeds':seeds,'cases':{},
            'scope':'The existing defect-gated candidates and updates are fixed. Only the alignment decision is randomized. c=cos(beta,e), threshold uniform[-1,1], accept iff c>=threshold. Zero-vector confidence=0. Shared threshold per pass for both branches, or independent thresholds per site and branch. Best feasible incoming/projected bound for all methods.'}
    for name,f in sources.items():
        m=ReducedMeyerMap(f,.05,40);case={}
        for mode in ['deterministic','shared','local']:
            reps=[]
            for seed in seeds if mode!='deterministic' else seeds[:1]:
                op=NoisyDefectGate(m,mode,seed);z=m.initial();elapsed=0.;rows=[]
                for k in range(2,steps+1):
                    start=time.perf_counter();z=op.step(z);elapsed+=time.perf_counter()-start
                    if k%8==0:rows.append({'pass':k,'seconds':elapsed,'gap':best_gap(m,z)})
                reps.append({'seed':seed,'rows':rows})
            case[mode]=reps
        result['cases'][name]=case
        Path(out).write_text(json.dumps(result,indent=2))
        print(name,{mode:[{'first_1e-3':next((r['pass'] for r in rep['rows'] if r['gap']<=1e-3),None),'final_gap':rep['rows'][-1]['gap']} for rep in reps] for mode,reps in case.items()},flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--size',type=int,default=128);p.add_argument('--steps',type=int,default=1024);p.add_argument('--seeds',default='11,29,47');p.add_argument('--ramp-only',action='store_true');p.add_argument('--out',required=True)
    a=p.parse_args();run(a.size,a.steps,[int(s) for s in a.seeds.split(',')],a.out,a.ramp_only)
