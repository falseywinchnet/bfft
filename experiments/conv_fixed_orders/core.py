"""Fixed-order CONV family. No runtime selection between variants.

Coefficients are exact derivatives of Lagrange basis polynomials, composed
with endpoint Hermite-to-Bernstein identities. No coefficient fitting/solve.
"""
from fractions import Fraction as F
from functools import lru_cache
from math import comb, factorial
from pathlib import Path
import ctypes, hashlib, os, platform, re, subprocess, sys
import numpy as np
from experiments.conv_exact_fusion.polynomial_proof import poly, mul, scale
from experiments.conv_admission_band.core import Operator as Legacy, FP
from experiments.conv_distilled_core import q1_order_blend

HERE=Path(__file__).resolve().parent
CONFIGS={'4c3':(4,3),'4c5':(4,5),'6c3':(6,3),'6c5':(6,5),'8c5':(8,5),'8c7':(8,7)}

def derivative(nodes,at,order):
    return _derivative(tuple(nodes),at,order)

@lru_cache(maxsize=2048)
def _derivative(nodes,at,order):
    out={}
    for k in nodes:
        p=poly([1]);den=F(1)
        for j in nodes:
            if j!=k:p=mul(p,poly([at-j,1]));den*=k-j
        out[k]=p[order]*factorial(order)/den if order<len(p) else F(0)
    return out

@lru_cache(maxsize=2048)
def node_jet(n,i,r,order):
    reach=min(i,n-1-i,r)
    if reach and 2*reach+1>order:
        nodes=list(range(i-reach,i+reach+1))
    else:
        count=order+2
        start=max(0,min(n-count,i-count//2))
        nodes=list(range(start,start+count))
    return derivative(nodes,i,order)

def bank(taps,degree,cell=None):
    r=(taps-2)//2;q=(degree-1)//2
    interior=cell is None
    n=4*taps;cell=taps if interior else cell
    indices=list(range(cell-r,cell+r+2)) if interior else list(range(taps))
    controls=[]
    for k in range(degree+1):
        right=k>q;j=degree-k if right else k;node=cell+int(right)
        row={node:F(1)}
        for order in range(1,j+1):
            factor=F(comb(j,order)*factorial(degree-order),factorial(degree))
            if right:factor*=(-1)**order
            for index,w in node_jet(n,node,r,order).items():
                row[index]=row.get(index,F(0))+factor*w
        controls.append(row)
    return [[controls[k+1].get(i,F(0))-controls[k].get(i,F(0)) for i in indices]
            for k in range(degree)]

def cint_change(s,degree):
    return re.sub(r'(?<![\w.])(10|5|4)(u?)(?![\w.])',
                  lambda m:str({'10':2*degree,'5':degree,'4':degree-1}[m[1]])+m[2],s)

def source_for(taps,degree,raw=False):
    s=(HERE/'v1_native.c').read_text();r=(taps-2)//2
    pre=s[:s.index('static const float CURRENT_FIR')]
    pool=s[s.index('typedef void (*parallel_function)'):s.index('static inline float first_jet')]
    # The original projection and causal ledger, parameterized by current count.
    a=s.index('static inline void compare_swap');b=s.index('static void tail_weights',a)
    admission=s[a:b]
    sort_start=admission.index('static inline void sort_five');sort_end=admission.index('static void simplex_five')
    if degree!=5:
        # Fixed adjacent comparator network, not a selected sort strategy.
        pairs=[]
        for phase in range(degree):
            for j in range(phase%2,degree-1,2):pairs.append((j,j+1))
        replacement='static inline void sort_five(float value[5]) { SORT_NETWORK }\n\n'
        admission=admission[:sort_start]+replacement+admission[sort_end:]
    # Change only current-count literals; lane/SIMD widths are absent here.
    admission=cint_change(admission,degree)
    if degree!=5:
        admission=admission.replace('SORT_NETWORK',''.join(f'compare_swap(value+{a},value+{b});' for a,b in pairs))
    # Existing exceptional floating-point bisection is diagnostic failure,
    # never a construction in this finite family.
    a=admission.index('    if (!found) {');b=admission.index('    double mass = 0.0;\n    int best = -1;',a)
    admission=admission[:a]+f'    if (!found) {{ for(int k=0;k<{degree};++k)c[k]=NAN; return; }}\n'+admission[b:]
    if raw:
        a=admission.index('static void ordered_projection(')
        admission=admission[:a]+admission[a:].replace('{','{ return;',1)
    def arr(rows):return '{'+','.join('{'+','.join(f'{x.numerator}.0f/{x.denominator}.0f' for x in row)+'}' for row in rows)+'}'
    tables=f'static const float BANK[{degree}][{taps}]='+arr(bank(taps,degree))+';\n'
    tables+=f'static const float LEFT[{r}][{degree}][{taps}]={{'+','.join(arr(bank(taps,degree,i)) for i in range(r))+'};\n'
    worker=(HERE/'raw_worker.inc').read_text().replace('@T@',str(taps)).replace('@D@',str(degree)).replace('@R@',str(r))
    if degree==5:
        a=s.index('static void tail_weights');b=s.index('typedef struct',a);tails=s[a:b]
    else:
        terms=[]
        for k in range(degree+1):terms.append(str(float(comb(degree,k)))+'*'+'*'.join(['u']*k+['v']*(degree-k)))
        tails=f'''static void tail_weights(double u,float tail[{degree}]){{
 const double v=1-u,b[{degree+1}]={{ {','.join(terms)} }};double sum=0;
 for(int k={degree};k>=1;--k){{sum+=b[k];tail[k-1]=(float)sum;}}
}}\n'''
    a=s.rfind('typedef struct',0,s.index('static void synthesis_worker'))
    b=s.index('typedef struct',s.index('API int conv_resize_lines_f32'))
    synthesis=s[a:b]
    # Do not replace vector widths of 4 in synthesis.
    synthesis=re.sub(r'(?<![\w.])5(u?)(?![\w.])',lambda m:str(degree)+m[1],synthesis)
    a=s.rfind('typedef struct',0,s.index('static void basin_average_worker'))
    b=s.index('/* Double-precision realization',a)
    basin=s[a:b]
    basin=basin.replace('{0.0f,0.0f,0.0f,0.0f,0.0f}','{0}')
    basin=re.sub(r'(?<![\w.])5(u?)(?![\w.])',lambda m:str(degree)+m[1],basin)
    quadrature=(degree+1)//2
    if quadrature!=3:
        nodes,weights=np.polynomial.legendre.leggauss(quadrature)
        basin=re.sub(r'const double nodes\[3\]=\{[^}]*\};',f'const double nodes[{quadrature}]={{'+','.join(format(x,'.17g') for x in nodes)+'};',basin)
        basin=re.sub(r'const double weights\[3\]=\{[^}]*\};',f'const double weights[{quadrature}]={{'+','.join(format(x,'.17g') for x in weights)+'};',basin)
        basin=basin.replace('quadrature<3',f'quadrature<{quadrature}')
    api=f'''API int conv_prepare_profile_f32(const float*x,int n,int lanes,float*c){{
 if(n<{taps})return -1;raw_currents(x,n,lanes,c);ordered_projection(x,n,lanes,c);return 0;}}
API int conv_audit_fibre(const float*a,const int8_t*sg,float delta,float*c){{project_fibre(a,sg,delta,c);return 0;}}
'''
    synthesis=synthesis.replace(f'n < {degree}',f'n < {taps}')
    basin=basin.replace(f'n<{degree}',f'n<{taps}')
    return pre+pool+tables+worker+admission+tails+synthesis+basin+api

class FixedOrder(Legacy):
    def __init__(self,taps,degree,raw=False):
        self.taps=taps;self.degree=degree;self.radius=(taps-2)//2
        source=source_for(taps,degree,raw)
        digest=hashlib.sha256((source+platform.machine()).encode()).hexdigest()[:16]
        build=Path('/tmp/conv_fixed_orders_native');build.mkdir(exist_ok=True)
        c=build/(digest+'.c');lib=build/(digest+'.dylib')
        if not lib.exists():
            c.write_text(source)
            subprocess.run(['cc','-std=c11','-O3','-mcpu=native','-dynamiclib','-pthread',str(c),'-lm','-o',str(lib)],check=True,capture_output=True)
        self.lib=ctypes.CDLL(str(lib))
        for name in ['conv_resize_lines_f32','conv_basin_average_lines_f32']:
            f=getattr(self.lib,name);f.argtypes=[FP,ctypes.c_int,ctypes.c_int,FP,ctypes.c_int];f.restype=ctypes.c_int
        self.lib.conv_prepare_profile_f32.argtypes=[FP,ctypes.c_int,ctypes.c_int,FP]
        self.lib.conv_prepare_profile_f32.restype=ctypes.c_int
    def profile(self,lines):
        lines=np.ascontiguousarray(lines,dtype=np.float32)
        if lines.ndim==1:lines=lines[:,None]
        n,lanes=lines.shape;c=np.empty((n-1,self.degree,lanes),np.float32)
        assert self.lib.conv_prepare_profile_f32(lines.ctypes.data_as(FP),n,lanes,c.ctypes.data_as(FP))==0
        return c
    def axis(self,values,target,axis,basin=False):
        if np.shape(values)[axis]<self.taps:raise ValueError('source axis shorter than fixed support')
        return super().axis(values,target,axis,basin)
    def geometry(self,source):
        field=np.asarray(source,dtype=float)
        if field.ndim==2:field=field[...,None]
        gradients=[]
        for axis in (1,0):
            y=np.moveaxis(field,axis,0);g=np.zeros_like(y)
            r=self.radius;n=len(y)
            for j,w in derivative(list(range(-r,r+1)),0,1).items():
                g[r:n-r]+=float(w)*y[r+j:n-r+j]
            for i in list(range(r))+list(range(n-r,n)):
                for j,w in node_jet(n,i,r,1).items():g[i]+=float(w)*y[j]
            gradients.append(np.moveaxis(g,0,axis))
        xx=np.sum(gradients[0]**2,axis=2);yy=np.sum(gradients[1]**2,axis=2)
        return np.divide(yy,xx+yy,out=np.full_like(xx,.5),where=xx+yy>0).astype(np.float32)
    def synthesize(self,source,shape):
        a=self.axis(self.axis(source,shape[1],1),shape[0],0)
        b=self.axis(self.axis(source,shape[0],0),shape[1],1)
        return q1_order_blend(self.geometry(source),a,b)
