"""Focused repeated validation, including feasible primal-dual gap.

The common alpha=1.75 was selected during the exploratory native screen;
the four-pass prefix is inherited from the paper. This is repeated
measurement, not a held-out image study. All timing samples are retained.
"""
from pathlib import Path
import sys,json,time,argparse,platform
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.meyer_transport_audit.native_audit import Native,library
from experiments.meyer_transport_audit.model import ReducedMeyerMap
from experiments.meyer_transport_audit.certificate import certificate
from experiments.benchmark_meyer_flow_jump import benchmark_scene

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--size',type=int,default=256);ap.add_argument('--threads',type=int,default=1)
    ap.add_argument('--repeats',type=int,default=15);ap.add_argument('--out',required=True)
    ap.add_argument('--lib',default='/tmp/meyer_audit.dylib');ap.add_argument('--mode',type=int,default=2)
    ap.add_argument('--reference',type=int,default=4096)
    ap.add_argument('--load-note',default='Other pre-existing CPU-heavy jobs active; randomized interleaving, no isolation.')
    a=ap.parse_args()
    lib=library(a.lib);archive=np.load(ROOT/'paper/fast_meyer_bregman/results/arrays.npz')
    from scipy.ndimage import zoom
    scenes={key[:-7]:np.ascontiguousarray(zoom(archive[key].astype(float),(a.size/archive[key].shape[0],a.size/archive[key].shape[1]),order=1)) for key in archive.files if key.endswith('_source')}
    scenes['crossing']=benchmark_scene(a.size)
    configs={'ordinary32':(0,32,1.,10,5),'ordinary40':(0,40,1.,10,5),'ordinary48':(0,48,1.,10,5),
             'ordinary64':(0,64,1.,10,5),'quality':(1,29,1.,10,5),'fast':(1,19,1.,18,3),
             'relax1.75_24':(2,24,1.75,10,5),'relax1.75_32':(2,32,1.75,10,5),
             'relax1.75_48':(2,48,1.75,10,5),'relax1.75_64':(2,64,1.75,10,5)}
    report={'size':a.size,'threads':a.threads,'repeats':a.repeats,'cpu_load':a.load_note,'scenes':{}}
    configs={key:((a.mode,)+c[1:] if c[0]==2 else c) for key,c in configs.items()}
    report['relaxation_native_mode']=a.mode
    arrays={};rng=np.random.default_rng(29)
    for name,f in scenes.items():
        p=Native(lib,a.size,a.threads);m=ReducedMeyerMap(f,.05,40);refs={};refcert={}
        for k in sorted(set([64,2048,a.reference])):
            refs[k]=p.split(f,0,k);refcert[k]=certificate(m,m.unpack(p.state().ravel()))
        times={key:[] for key in configs};outputs={};certs={}
        for repeat in range(a.repeats+2):
            for key in rng.permutation(list(configs)):
                start=time.perf_counter_ns();v=p.split(f,*configs[key]);ms=(time.perf_counter_ns()-start)/1e6
                if repeat>=2:times[key].append(ms)
                if repeat==a.repeats+1:
                    z=m.unpack(p.state().ravel());outputs[key]=v
                    certs[key]={**certificate(m,z),'residual_rms':m.residual_norm(z)}
        rows={key:{'median_ms':float(np.median(times[key])),'samples_ms':times[key],**certs[key],
                   **{'error'+str(k):float(np.linalg.norm(v-ref)/max(np.linalg.norm(ref),1e-30)) for k,ref in refs.items()}} for key,v in outputs.items()}
        report['scenes'][name]={'methods':rows,'reference_certificates':refcert,'reference_drift':float(np.linalg.norm(refs[a.reference]-refs[2048])/np.linalg.norm(refs[a.reference]))}
        arrays[name+'_source']=f;arrays[name+'_reference'+str(a.reference)]=refs[a.reference]
        for key,v in outputs.items():arrays[name+'_'+key]=v
        Path(a.out).write_text(json.dumps(report,indent=2));np.savez_compressed(str(Path(a.out).with_suffix('.npz')),**arrays)
        print(name, {key:{k:rows[key][k] for k in ['median_ms','error64','error'+str(a.reference),'relative_gap']} for key in ['quality','ordinary64','relax1.75_32','relax1.75_64']},flush=True)
        p.close()

if __name__=='__main__':main()
