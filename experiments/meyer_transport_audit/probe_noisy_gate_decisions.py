"""Untimed decision census, including symmetry loss from local decisions."""
from pathlib import Path
import argparse,json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.meyer_transport_audit.model import ReducedMeyerMap
from experiments.meyer_transport_audit.defect_relaxation import scenes
from experiments.meyer_transport_audit.noisy_defect_gate import NoisyDefectGate,alignment,best_gap


class CensusGate(NoisyDefectGate):
    def __init__(self,*args):
        super().__init__(*args)
        self.active=0;self.ambiguous=0;self.flips=0;self.energy=0.;self.flip_energy=0.
    def admit_defect(self,previous,extra):
        mask=super().admit_defect(previous,extra)
        base=previous[0]*extra[0]+previous[1]*extra[1]>=0
        c=alignment(previous,extra);energy=extra[0]**2+extra[1]**2
        active=energy>1e-24
        self.active+=int(np.count_nonzero(active))
        self.ambiguous+=int(np.count_nonzero(active&(np.abs(c)<1-1e-8)))
        self.flips+=int(np.count_nonzero(active&(mask!=base)))
        self.energy+=float(np.sum(energy));self.flip_energy+=float(np.sum(energy[mask!=base]))
        return mask


def run():
    result={}
    for name,lam,mu in [('ramp',.02,20),('crossing',.05,40),('noise',.05,40)]:
        f=scenes(128,3029)[name];m=ReducedMeyerMap(f,lam,mu);case={}
        for mode in ['deterministic','shared','local']:
            op=CensusGate(m,mode,11);z=m.initial();first=None
            for k in range(2,513):
                z=op.step(z)
                if k%8==0 and first is None and best_gap(m,z)<=1e-4:first=k
            case[mode]={'active_decisions':op.active,'ambiguous_fraction':op.ambiguous/max(op.active,1),
                        'flip_fraction':op.flips/max(op.active,1),'flip_energy_fraction':op.flip_energy/max(op.energy,1e-300),
                        'final_gap':best_gap(m,z),'first_1e-4':first,
                        'u_row_variation_rms':float(np.sqrt(np.mean((z.u-z.u.mean(axis=0,keepdims=True))**2))),
                        'w_row_variation_rms':float(np.sqrt(np.mean((z.w-z.w.mean(axis=0,keepdims=True))**2)))}
        result[f'{name}_lambda{lam}_mu{mu}']=case
        print(name,case,flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    Path(a.out).write_text(json.dumps(run(),indent=2))
