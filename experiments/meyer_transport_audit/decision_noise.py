"""Randomness selects exact memory candidates; no noise is added to fields."""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.meyer_transport_audit.model import ReducedMeyerMap,State,grad,project_disk
from experiments.meyer_transport_audit.defect_relaxation import DefectRelaxation,scenes
from experiments.meyer_transport_audit.diagnose_ramp_refresh import terms


class DecisionNoise:
    def __init__(self,m,mode,seed=11):
        self.m=m;self.mode=mode;self.rng=np.random.default_rng(seed)
        self.base=DefectRelaxation(m,'ordinary');self.k=0;self.memory=None
        self.parity=np.indices(m.shape).sum(axis=0)%2

    def step(self,z):
        self.k+=1;m=self.m
        n=self.base.step(z);gu,gw=grad(n.u),grad(n.w)
        p=[(n.tux-gu[0],n.tuy-gu[1]),(n.twx-gw[0],n.twy-gw[1])]
        q=[project_disk(n.tux,n.tuy,m.ru),project_disk(n.twx,n.twy,m.rw)]
        if self.mode=='alternating': masks=[self.k%2==0]*2
        elif self.mode=='checker': masks=[(self.parity+self.k)%2==0]*2
        elif self.mode=='random_global': masks=[bool(self.rng.integers(2))]*2
        elif self.mode=='random_branch': masks=list(self.rng.integers(2,size=2).astype(bool))
        elif self.mode=='random_site': masks=[self.rng.integers(2,size=m.shape).astype(bool)]*2
        else: masks=list(self.rng.integers(2,size=(2,*m.shape)).astype(bool))
        if self.mode=='random_gated':
            if self.memory is None:
                a,b=grad(z.u),grad(z.w)
                self.memory=[(z.tux-a[0],z.tuy-a[1]),(z.twx-b[0],z.twy-b[1])]
            masks=[mask&(((pi[0]-bi[0])*(qi[0]-pi[0])+(pi[1]-bi[1])*(qi[1]-pi[1]))>=0)
                   for mask,pi,qi,bi in zip(masks,p,q,self.memory)]
        selected=[tuple(np.where(mask,qi,pi) for qi,pi in zip(qb,pb)) for mask,pb,qb in zip(masks,p,q)]
        self.memory=selected
        return State(n.u,n.w,gu[0]+selected[0][0],gu[1]+selected[0][1],gw[0]+selected[1][0],gw[1]+selected[1][1])


def best_gap(m,z):
    a,b=terms(m,z),terms(m,z,True)
    return min(a['primal'],b['primal'])-max(a['dual'],b['dual'])


def run(n,steps,seeds,ramp_only,out):
    modes=['ordinary','refresh','aligned_refresh','alternating','checker','random_global','random_branch','random_site','random_independent','random_gated']
    sources={'ramp1d':255.*np.arange(n)[None,:]/n} if ramp_only else scenes(n,3029)
    result={'size':n,'max_passes':steps,'seeds':seeds,'cases':{},'note':'Best incoming/projected feasible bounds for every method. All update and RNG costs timed; shared diagnostics excluded. Sample interval 8. Fair coin schedules and matched deterministic controls.'}
    for lam,mu in ([(.02,20),(.05,40)] if ramp_only else [(.05,40)]):
        for name,f in sources.items():
            m=ReducedMeyerMap(f,lam,mu);case={}
            for mode in modes:
                repetitions=[]
                for seed in seeds if mode.startswith('random') else seeds[:1]:
                    op=DefectRelaxation(m,mode) if mode in ['ordinary','refresh','aligned_refresh'] else DecisionNoise(m,mode,seed)
                    z=m.initial();elapsed=0.;rows=[]
                    for k in range(2,steps+1):
                        start=time.perf_counter();z=op.step(z);elapsed+=time.perf_counter()-start
                        if k%8==0:rows.append({'pass':k,'seconds':elapsed,'gap':best_gap(m,z)})
                    repetitions.append({'seed':seed,'rows':rows})
                case[mode]=repetitions
            result['cases'][f'{name}_lambda{lam}_mu{mu}']=case
            Path(out).write_text(json.dumps(result,indent=2))
            print(name,lam,{mode:[next((r['pass'] for r in rep['rows'] if r['gap']<=1e-4),None) for rep in reps] for mode,reps in case.items()},flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--size',type=int,default=128);p.add_argument('--steps',type=int,default=512);p.add_argument('--ramp-only',action='store_true');p.add_argument('--seeds',default='11,29,47');p.add_argument('--out',required=True)
    a=p.parse_args();run(a.size,a.steps,[int(x) for x in a.seeds.split(',')],a.ramp_only,a.out)
