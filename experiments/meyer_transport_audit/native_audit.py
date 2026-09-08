"""Native wall-time frontier including all six-state relaxation work."""
from pathlib import Path
import sys,ctypes,json,time,argparse,platform
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.meyer_transport_audit.model import ReducedMeyerMap
from experiments.benchmark_meyer_flow_jump import benchmark_scene
import bfft

class Native:
    def __init__(self,lib,n,threads=8,lam=.05,mu=40):
        self.lib=lib;self.n=n;self.p=lib.audit_create(n,threads,lam,mu)
        if not self.p: raise RuntimeError('native plan creation failed')
    def split(self,image,mode=0,passes=64,alpha=1.,horizon=10,jumps=5):
        out=np.empty_like(image)
        self.lib.audit_split(self.p,image,out,mode,passes,alpha,horizon,jumps)
        return out
    def state(self):
        z=np.empty((6,self.n,self.n));self.lib.audit_state(self.p,z);return z
    def close(self): self.lib.audit_destroy(self.p)

def library(path):
    lib=ctypes.CDLL(path);arr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    lib.audit_create.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_double,ctypes.c_double];lib.audit_create.restype=ctypes.c_void_p
    lib.audit_destroy.argtypes=[ctypes.c_void_p]
    lib.audit_split.argtypes=[ctypes.c_void_p,arr,arr,ctypes.c_int,ctypes.c_int,ctypes.c_double,ctypes.c_int,ctypes.c_int]
    lib.audit_state.argtypes=[ctypes.c_void_p,arr]
    return lib

def test(lib):
    f=benchmark_scene(32);m=ReducedMeyerMap(f,.05,40)
    for alpha in [1.,1.5,1.75,1.8]:
        p=Native(lib,32,1);p.split(f,2,19,alpha)
        z=m.initial()
        for k in range(1,19):
            z=m.step(z) if k<4 else m.unpack(m.pack(z)+alpha*m.pack(m.residual(z)))
        np.testing.assert_allclose(p.state(),np.array(z.fields()),atol=3e-11,rtol=3e-12)
        p.split(f,3,19,alpha)
        np.testing.assert_allclose(p.state(),np.array(z.fields()),atol=3e-11,rtol=3e-12)
        p.close()
    one=Native(lib,32,1);four=Native(lib,32,4)
    a=one.split(f,2,32,1.5);b=four.split(f,2,32,1.5)
    np.testing.assert_array_equal(a,b)
    np.testing.assert_allclose(one.split(np.ascontiguousarray(np.roll(f,(3,7),(0,1))),2,32,1.5),np.roll(a,(3,7),(0,1)),atol=1e-10)
    np.testing.assert_allclose(one.split(f,0,64),bfft.MeyerPlan(f.shape,threads=1).split_legacy(f)[1],atol=1e-11)
    for mode in [3,4]:
        a=one.split(f,mode,32,1.75);b=four.split(f,mode,32,1.75)
        np.testing.assert_array_equal(a,b)
        np.testing.assert_allclose(one.split(np.ascontiguousarray(np.roll(f,(3,7),(0,1))),mode,32,1.75),np.roll(a,(3,7),(0,1)),atol=1e-10)
    one.close();four.close()
    print('Native six-field oracle, thread identity, translation and library equivalence: PASS',flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--lib',default='/tmp/meyer_audit.dylib');ap.add_argument('--out',required=True)
    ap.add_argument('--repeats',type=int,default=7);ap.add_argument('--size',type=int,default=256);a=ap.parse_args()
    lib=library(a.lib);test(lib)
    old=np.load(ROOT/'paper/fast_meyer_bregman/results/arrays.npz')
    scenes={key[:-7]:old[key] for key in old.files if key.endswith('_source')}
    scenes['crossing']=benchmark_scene(a.size)
    from scipy.ndimage import zoom
    scenes={key:np.ascontiguousarray(zoom(f.astype(np.float64),(a.size/f.shape[0],a.size/f.shape[1]),order=1)) for key,f in scenes.items()}
    configs={f'ordinary{k}':(0,k,1.,10,5) for k in [16,19,24,29,32,36,40,48,56,64,96,128]}
    configs.update({'fast':(1,19,1.,18,3),'quality':(1,29,1.,10,5)})
    for alpha in [1.25,1.5,1.75,1.9]:
        for k in [19,24,29,32,40,48,64,96,128]: configs[f'relax{alpha}_{k}']=(2,k,alpha,10,5)
    result={'size':a.size,'repeats':a.repeats,'threads':8,'platform':platform.platform(),'tests':'passed','scenes':{}}
    arrays={};rng=np.random.default_rng(42)
    for name,f in scenes.items():
        p=Native(lib,a.size);model=ReducedMeyerMap(f,.05,40)
        refs={k:p.split(f,0,k) for k in [64,1024,2048]}
        samples={k:[] for k in configs};outputs={};states={}
        for repeat in range(a.repeats+2):
            for key in rng.permutation(list(configs)):
                start=time.perf_counter_ns();v=p.split(f,*configs[key]);ms=(time.perf_counter_ns()-start)/1e6
                if repeat>=2:samples[key].append(ms)
                if repeat==a.repeats+1: outputs[key]=v;states[key]=p.state()
        rows={}
        for key,v in outputs.items():
            z=model.unpack(states[key].ravel())
            rows[key]={'median_ms':float(np.median(samples[key])),'samples_ms':samples[key],
                       'residual_rms':model.residual_norm(z),
                       **{'error'+str(k):float(np.linalg.norm(v-ref)/max(np.linalg.norm(ref),1e-30)) for k,ref in refs.items()}}
        result['scenes'][name]={'reference_drift':float(np.linalg.norm(refs[2048]-refs[1024])/np.linalg.norm(refs[2048])),'methods':rows}
        arrays[name+'_source']=f
        for key in ['ordinary64','quality','relax1.5_32','relax1.75_32']: arrays[name+'_'+key]=outputs[key]
        arrays[name+'_reference2048']=refs[2048]
        Path(a.out).write_text(json.dumps(result,indent=2));np.savez_compressed(str(Path(a.out).with_suffix('.npz')),**arrays)
        print(name,json.dumps({k:rows[k] for k in ['ordinary32','quality','relax1.5_32','relax1.75_32','ordinary64']}),flush=True)
        p.close()

if __name__=='__main__':main()
