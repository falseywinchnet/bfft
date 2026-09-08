"""Fixed principled downstream variants of the frozen CONV Version 1.0.
No runtime policy selects among these experiments.
"""
from fractions import Fraction as F
from pathlib import Path
from math import comb
import numpy as np
from experiments.conv_admission_band.core import Operator
from experiments.conv_distilled_core import nodal_current_geometry,q1_order_blend
from experiments.conv_projection_metric_certificate import matrices

HERE=Path(__file__).resolve().parent
FROZEN=HERE.parent/'conv_fixed_orders/v1_native.c'
NAMES=['v1','ledger_diag','projection_diag','joint_diag','ledger_full',
       'projection_value','blend_equal','blend_amplitude','basin_primitive','basin_hat']

def grams():
    value,slope=matrices()
    # Mass-zero errors: tau_i minus their mean u is a reversal-symmetric gauge.
    centered=[[value[i][j]-sum(value[i])/5-sum(row[j] for row in value)/5
               +sum(map(sum,value))/25 for j in range(5)] for i in range(5)]
    return slope,centered

def weights(kind):
    g=grams()[kind=='value'];w=[g[i][i] for i in range(5)]
    return [x/w[2] for x in w]

def number(x):return f'{x.numerator}.0/{x.denominator}.0'

def transformed(name):
    source=FROZEN.read_text()
    if name in ('ledger_diag','joint_diag','ledger_full'):
        start=source.index('            /* CONV_LEDGER_COST_BEGIN */')
        end=source.index('            /* CONV_LEDGER_COST_END */',start)+len('            /* CONV_LEDGER_COST_END */')
        # A fixed finite cost on the original eligible boundaries. No new
        # proposals, region catalog, or stopping rule is introduced.
        ledger=(HERE/'ledger.inc').read_text()
        if name=='ledger_full':
            matrix=grams()[0]
            ledger=ledger.replace('@GRAM@','{'+','.join('{'+','.join(number(v) for v in row)+'}' for row in matrix)+'}')
        else:
            w=weights('slope')
            ledger=source[start:end]
            ledger=ledger.replace('double increments[10]', 'static const double ledger_weight[5]={'+','.join(number(v) for v in w)+'};\n            double increments[10]')
            ledger=ledger.replace('square=value*value','square=ledger_weight[k]*value*value')
            ledger=ledger.replace('cost += (double)value * value','cost += ledger_weight[k]*(double)value * value')
        source=source[:start]+ledger+source[end:]
    if name in ('projection_diag','joint_diag','projection_value'):
        declaration='static void project_fibre(const float a[5], const int8_t sign[5], float delta, float c[5]) {'
        source=source.replace(declaration,declaration.replace('project_fibre(','project_fibre_v1('),1)
        marker='typedef struct {\n    const float *x;\n    int n;\n    int lanes;\n    float *current;\n} projection_context;'
        w=weights('value' if name=='projection_value' else 'slope')
        code=(HERE/'weighted_projection.inc').read_text().replace('@WEIGHTS@','{'+','.join(number(x) for x in w)+'}').replace('@INVERSE@','{'+','.join(number(1/x) for x in w)+'}')
        source=source.replace(marker,code+'\n'+marker)
    if name=='basin_primitive':
        marker='static void basin_average_worker('
        source=source.replace(marker,(HERE/'primitive.inc').read_text()+'\n'+marker,1)
        start=source.index('            for(int quadrature=0;quadrature<3;++quadrature)',source.index(marker))
        end=source.index('            int lane=0;',start)
        code='''            double pleft[5],pright[5];
            primitive_tails(segment_left-cell,pleft);
            primitive_tails(segment_right-cell,pright);
            for(int k=0;k<5;++k)current_factor[k]=(float)((pright[k]-pleft[k])/width);
'''
        source=source[:start]+code+source[end:]
    if name=='basin_hat':
        start=source.index('static void basin_average_worker(')
        end=source.index('API int conv_basin_average_lines_f32',start)
        source=source[:start]+(HERE/'hat_basin.inc').read_text()+'\n'+source[end:]
    return source

class Downstream(Operator):
    def __init__(self,name):
        self.variant=name
        super().__init__('conv',source_transform=lambda _:transformed(name))
    def synthesize(self,source,shape):
        forward=self.axis(self.axis(source,shape[1],1),shape[0],0)
        reverse=self.axis(self.axis(source,shape[0],0),shape[1],1)
        if self.variant=='blend_equal':
            beta=np.full(np.shape(source)[:2],.5,dtype=np.float32)
        elif self.variant=='blend_amplitude':
            xx,_,yy,_=nodal_current_geometry(source)
            x=np.sqrt(xx);y=np.sqrt(yy)
            beta=np.divide(y,x+y,out=np.full_like(x,.5),where=x+y>0).astype(np.float32)
        else:beta=nodal_current_geometry(source)[3].astype(np.float32)
        return q1_order_blend(beta,forward,reverse)
