"""Same native proposal, ledger, basin integration and two-order synthesis.

Only the fibre admission is replaced in generated /tmp native copies. No
production source edits, image classification, fitting, or new solver.
"""
import ctypes
import hashlib
import os
from pathlib import Path
import platform
import subprocess
import sys
from fractions import Fraction
from math import comb
import numpy as np
from experiments.conv_distilled_core import nodal_current_geometry, q1_order_blend

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
VARIANTS={'conv':(0,0),'raw':(-1,0),'uniform1':(1,1),
          'word0':(2,0),'word1':(2,1),'word2':(2,2),'ray1':(3,1)}
FP=ctypes.POINTER(ctypes.c_float)
SP=ctypes.POINTER(ctypes.c_int8)

def jet_bank(radius):
    """Unique symmetric centered finite differences of order 2*radius.

    Derive currents algebraically from the same quintic endpoint two-jet.
    No fitted coefficients or selected frequency samples.
    """
    F=Fraction;r=radius
    first={0:F(0)};second={}
    for k in range(1,r+1):
        first[k]=F((-1)**(k+1)*comb(2*r,r-k),k*comb(2*r,r))
        first[-k]=-first[k]
        second[k]=second[-k]=2*first[k]/k
    second[0]=-2*sum(second[k] for k in range(1,r+1))
    bank=[[] for _ in range(5)]
    for k in range(-r,r+2):
        ml=first.get(k,0);mr=first.get(k-1,0)
        ql=second.get(k,0);qr=second.get(k-1,0)
        values=[ml/5,ml/5+ql/20,F(int(k==1)-int(k==0))-F(2,5)*(ml+mr)+(qr-ql)/20,
                mr/5-qr/20,mr/5]
        for row,value in zip(bank,values):row.append(F(value))
    return bank

class Operator:
    def __init__(self,name,radius=2,source_transform=None):
        mode,depth=VARIANTS[name];self.name=name;self.radius=radius
        source=(ROOT/'standalone_conv_resize_demo/native/conv_native.c').read_text()
        if source_transform is not None:source=source_transform(source)
        if radius!=2:
            bank=jet_bank(radius);taps=2*radius+2
            begin=source.index('static const float CURRENT_FIR[5][6]')
            end=source.index('\n};',begin)+3
            rows=[','.join(f'{v.numerator}.0f/{v.denominator}.0f' for v in row) for row in bank]
            source=source[:begin]+f'static const float CURRENT_FIR[5][{taps}] = {{\n'+',\n'.join('{'+row+'}' for row in rows)+'\n};'+source[end:]
            begin=source.index('static void raw_currents_worker(');end=source.index('\nstatic void raw_currents(',begin)
            worker=source[begin:end]
            assert 'cell >= 2 && cell < n - 3' in worker
            worker=worker.replace('cell >= 2 && cell < n - 3',f'cell >= {radius} && cell < n - {radius+1}')
            worker=worker.replace('tap < 6',f'tap < {taps}').replace('cell - 2 + tap',f'cell - {radius} + tap')
            source=source[:begin]+worker+source[end:]
        marker='typedef struct {\n    const float *x;\n    int n;\n    int lanes;\n    float *current;\n} projection_context;'
        assert source.count(marker)==1
        extension=(HERE/'refinement.h').read_text()
        source=source.replace(marker,extension+'\n'+marker)
        call='            project_fibre(a, signs, delta, c);'
        assert source.count(call)==1
        if mode>0:source=source.replace(call,'            refined_fibre(a, signs, delta, c);')
        if mode<0:
            start='static void ordered_projection(const float *x, int n, int lanes, float *current) {'
            assert source.count(start)==1
            source=source.replace(start,start+'\n    return; /* RAW diagnostic: no admission. */')
        source=f'#define ADMISSION_MODE {mode}\n#define ADMISSION_DEPTH {depth}\n'+source
        digest=hashlib.sha256((source+platform.machine()).encode()).hexdigest()[:16]
        build=Path('/tmp/conv_admission_native');build.mkdir(exist_ok=True)
        cpath=build/(digest+'.c');libpath=build/(digest+('.dylib' if sys.platform=='darwin' else '.so'))
        if not libpath.exists():
            cpath.write_text(source)
            flags=['-dynamiclib'] if sys.platform=='darwin' else ['-shared','-fPIC']
            subprocess.run([os.environ.get('CC','cc'),'-std=c11','-O3','-DNDEBUG',
                            '-mcpu=native','-fvisibility=hidden','-pthread',*flags,
                            str(cpath),'-lm','-o',str(libpath)],check=True,capture_output=True)
        self.lib=ctypes.CDLL(str(libpath))
        for name in ['conv_resize_lines_f32','conv_basin_average_lines_f32']:
            f=getattr(self.lib,name);f.argtypes=[FP,ctypes.c_int,ctypes.c_int,FP,ctypes.c_int];f.restype=ctypes.c_int
        self.lib.conv_prepare_profile_f32.argtypes=[FP,ctypes.c_int,ctypes.c_int,FP]
        self.lib.conv_prepare_profile_f32.restype=ctypes.c_int
        self.lib.conv_audit_fibre.argtypes=[FP,SP,ctypes.c_float,FP]
        self.lib.conv_audit_fibre.restype=None

    def axis(self,values,target,axis,basin=False):
        field=np.asarray(values,dtype=np.float32);moved=np.moveaxis(field,axis,0)
        n=moved.shape[0];lines=np.ascontiguousarray(moved.reshape(n,-1))
        out=np.empty((target,lines.shape[1]),dtype=np.float32)
        f=self.lib.conv_basin_average_lines_f32 if basin else self.lib.conv_resize_lines_f32
        status=f(lines.ctypes.data_as(FP),n,lines.shape[1],out.ctypes.data_as(FP),int(target))
        if status:raise RuntimeError(status)
        return np.moveaxis(out.reshape((target,)+moved.shape[1:]),0,axis)

    def profile(self,lines):
        lines=np.ascontiguousarray(lines,dtype=np.float32)
        if lines.ndim==1:lines=lines[:,None]
        n,lanes=lines.shape;current=np.empty((n-1,5,lanes),dtype=np.float32)
        assert self.lib.conv_prepare_profile_f32(lines.ctypes.data_as(FP),n,lanes,current.ctypes.data_as(FP))==0
        return current

    def fibre(self,a,signs,delta):
        a=np.ascontiguousarray(a,dtype=np.float32);signs=np.ascontiguousarray(signs,dtype=np.int8)
        c=np.empty(5,dtype=np.float32)
        self.lib.conv_audit_fibre(a.ctypes.data_as(FP),signs.ctypes.data_as(SP),delta,c.ctypes.data_as(FP))
        return c

    def synthesize(self,source,shape):
        forward=self.axis(self.axis(source,shape[1],1),shape[0],0)
        reverse=self.axis(self.axis(source,shape[0],0),shape[1],1)
        beta=nodal_current_geometry(source)[3].astype(np.float32)
        return q1_order_blend(beta,forward,reverse)

    def reduce(self,source,shape):
        z=np.asarray(source,dtype=np.float32)
        if shape[0]<z.shape[0]:z=self.axis(z,shape[0],0,True)
        if shape[1]<z.shape[1]:z=self.axis(z,shape[1],1,True)
        return z

    def resize(self,source,shape):
        z=self.reduce(source,shape)
        return z if z.shape[:2]==tuple(shape) else self.synthesize(z,shape)


def subdivide(a,depth):
    """Independent float64 de Casteljau oracle; final axis contains coefficients."""
    a=np.asarray(a,dtype=float)
    if depth==0:return a[...,None,:]
    row=a.copy();left=[row[...,0]];right=[row[...,-1]]
    for _ in range(4):
        row=(row[...,:-1]+row[...,1:])/2
        left.append(row[...,0]);right.append(row[...,-1])
    return np.concatenate([subdivide(np.stack(left,-1),depth-1),
                           subdivide(np.stack(right[::-1],-1),depth-1)],axis=-2)

def word(a,tol=0):
    out=[]
    for v in np.asarray(a).flat:
        if abs(v)<=tol:continue
        s=1 if v>0 else -1
        if not out or out[-1]!=s:out.append(s)
    return out

def subsequence(a,b):
    it=iter(b)
    return all(any(x==y for y in it) for x in a)
