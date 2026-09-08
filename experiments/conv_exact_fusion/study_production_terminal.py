"""Time the actual demo wrappers, including their copies and basin reduction."""
from experiments.conv_admission_band.core import Operator
from experiments.conv_distilled_core import (
    distilled_conv_synthesis, nodal_current_geometry, reverse_conv_resize,
    conv_resize, q1_order_blend, _basin_average_axis,
)
import numpy as np

class ProductionOperator(Operator):
    def __init__(self,name):
        super().__init__('conv');self.variant=name
    def synthesize(self,source,shape):
        if self.variant=='fused':return distilled_conv_synthesis(source,shape)
        forward=conv_resize(source,shape)
        reverse=reverse_conv_resize(source,shape)
        eta=nodal_current_geometry(source)[3].astype(np.float32)
        return q1_order_blend(eta,forward,reverse)
    def reduce(self,source,shape):
        z=np.asarray(source,dtype=np.float32)
        if shape[0]<z.shape[0]:z=_basin_average_axis(z,shape[0],0)
        if shape[1]<z.shape[1]:z=_basin_average_axis(z,shape[1],1)
        return z

if __name__=='__main__':
    import argparse
    from experiments.conv_exact_fusion import study
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--repeats',type=int,default=31)
    a=p.parse_args();study.operator=ProductionOperator
    study.run(a.out,a.repeats,'base,fused')
