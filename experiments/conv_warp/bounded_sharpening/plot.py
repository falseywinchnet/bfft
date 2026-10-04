"""Render the measured unit-strength, derivative-bounded line proposal."""
from pathlib import Path
import numpy as np
from scipy import sparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .study import (finite_joint_control_nets,_global_lattice_from_atlas,
 constraints,derivative_rows,refined_derivative_rows,concentration,admit,atlas,evaluate_joint_atlas)

x=np.arange(9);line=np.maximum(0,np.minimum(x+.5,5.085)-np.maximum(x-.5,4.085));source=np.tile(line,(9,1))
control,_=finite_joint_control_nets(source);P=_global_lattice_from_atlas(control[...,None])[...,0]
A,b=constraints(source);D=derivative_rows(P);bound=2*np.max(abs(D@P.ravel()));D=refined_derivative_rows(P)
A=sparse.vstack([A,D,-D],format='csr');b=np.concatenate([b,np.full(2*D.shape[0],-bound)])
Q,_=admit(P,concentration(P,source),A,b)
q=np.linspace(2,7,1501)
base=evaluate_joint_atlas(atlas(P),q,np.full_like(q,4));sharp=evaluate_joint_atlas(atlas(Q),q,np.full_like(q,4))
nodes,weights=np.polynomial.legendre.leggauss(4)
def pixels(V):
 result=[]
 for i in x[1:-1]:
  z=np.concatenate([i-.25+.25*nodes,i+.25+.25*nodes]);w=np.tile(weights*.25,2)
  result.append(evaluate_joint_atlas(atlas(V),z,np.full_like(z,4))@w)
 return np.array(result)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
fig,(a,b)=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
fig.suptitle('Quintic current concentration with range and derivative bounds',fontsize=15)
a.plot(q,base,label='Admitted base',color='#64748b',linewidth=2)
a.plot(q,sharp,label='Concentrated current',color='#007f80',linewidth=2)
a.scatter(x,line,color='#172d3a',s=26,zorder=4,label='Fixed source values')
a.set(xlim=(2.6,6.5),ylim=(-.02,.65),xlabel='Source coordinate',ylabel='Normalized contrast',title='Continuous source potential');a.legend(frameon=False,fontsize=9)
w=.24;positions=x[1:-1]
b.bar(positions-w,pixels(P),width=w,label='Base area integral',color='#94a3b8')
b.bar(positions,pixels(Q),width=w,label='Concentrated area integral',color='#007f80')
b.bar(positions+w,line[1:-1],width=w,label='Original pixel coverage',color='#dba74c')
b.set(xlim=(2.5,6.5),ylim=(0,.65),xlabel='Target pixel centre (unit spacing)',title='Exact integration over target pixels');b.legend(frameon=False,fontsize=9)
for ax in (a,b):
 ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
out=Path('output/support_geometry/conv_bounded_sharpening');out.mkdir(parents=True,exist_ok=True)
fig.savefig(out/'bounded-concentration.png',dpi=180);fig.savefig(out/'bounded-concentration.pdf')
