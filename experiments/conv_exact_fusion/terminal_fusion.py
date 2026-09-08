"""Fuse both terminal reconstructions and Q1 blending, with original profiles."""
import ctypes
from pathlib import Path
import numpy as np
from experiments.conv_admission_band.core import Operator,FP,nodal_current_geometry

class FusedTerminal(Operator):
    def __init__(self,preblend=False):
        extension=(Path(__file__).with_name('terminal_preblend.h') if preblend else Path(__file__).with_suffix('.h')).read_text()
        def transform(source):
            if '/* CONV_TERMINAL_FUSION_BEGIN */' in source:
                start=source.index('/* CONV_TERMINAL_FUSION_BEGIN */')
                end=source.index('/* CONV_TERMINAL_FUSION_END */',start)+len('/* CONV_TERMINAL_FUSION_END */')
                source=source[:start]+source[end:]
            return source+'\n'+extension
        super().__init__('conv',source_transform=transform)
        self.lib.conv_fused_terminal.argtypes=[FP]*5+[ctypes.c_int]*5+[FP]
        self.lib.conv_fused_terminal.restype=ctypes.c_int
    def synthesize(self,source,shape):
        source=np.asarray(source,dtype=np.float32)
        h,w=source.shape[:2];oh,ow=shape[:2]
        c=1 if source.ndim==2 else source.shape[2]
        x=np.ascontiguousarray(self.axis(source,ow,1).reshape(h,ow*c))
        y=np.ascontiguousarray(np.moveaxis(self.axis(source,oh,0),1,0).reshape(w,oh*c))
        cy=self.profile(x);cx=self.profile(y)
        eta=np.ascontiguousarray(nodal_current_geometry(source)[3],dtype=np.float32)
        out=np.empty((oh,ow) if source.ndim==2 else (oh,ow,c),dtype=np.float32)
        args=[v.ctypes.data_as(FP) for v in (x,y,cy,cx,eta)]
        status=self.lib.conv_fused_terminal(*args,h,w,oh,ow,c,out.ctypes.data_as(FP))
        if status:raise RuntimeError(status)
        return out

if __name__=='__main__':
    import argparse
    from experiments.conv_exact_fusion import study
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--repeats',type=int,default=31)
    p.add_argument('--methods',default='base,fused,preblend')
    a=p.parse_args()
    study.operator=lambda name:Operator('conv') if name=='base' else FusedTerminal(preblend=name=='preblend')
    study.run(a.out,a.repeats,a.methods)
