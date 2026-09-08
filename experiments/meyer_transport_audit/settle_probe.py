"""Ablate two final ordinary chart refreshes after current-state relaxation."""
from pathlib import Path
import sys,json,argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.meyer_transport_audit.native_audit import Native,library
from experiments.meyer_transport_audit.model import ReducedMeyerMap
from experiments.meyer_transport_audit.certificate import certificate
from experiments.benchmark_meyer_flow_jump import benchmark_scene

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True)
    ap.add_argument('--arrays',default='/tmp/meyer_validate256.npz');a=ap.parse_args()
    lib=library('/tmp/meyer_audit_settle.dylib');f=benchmark_scene(32);p=Native(lib,32,1);m=ReducedMeyerMap(f,.05,40)
    p.split(f,4,24,1.75);z=m.initial()
    for k in range(1,24):
        z=m.step(z) if k<4 or k>=22 else m.unpack(m.pack(z)+1.75*m.pack(m.residual(z)))
    np.testing.assert_allclose(p.state(),np.array(z.fields()),rtol=3e-12,atol=3e-11);p.close()
    data=np.load(a.arrays);report={}
    for key in data.files:
        if not key.endswith('_source'):continue
        f=np.ascontiguousarray(data[key]);n=len(f);p=Native(lib,n,1);m=ReducedMeyerMap(f,.05,40);rows={}
        ref=p.split(f,0,64)
        for mode in [3,4]:
            for k in [24,32,48,64]:
                v=p.split(f,mode,k,1.75);z=m.unpack(p.state().ravel())
                rows[f'mode{mode}_{k}']={**certificate(m,z),'residual_rms':m.residual_norm(z),
                    'error64':float(np.linalg.norm(v-ref)/np.linalg.norm(ref))}
        report[key]=rows;p.close()
    Path(a.out).write_text(json.dumps(report,indent=2));print('Two-refresh oracle and ablation complete',flush=True)

if __name__=='__main__':main()
