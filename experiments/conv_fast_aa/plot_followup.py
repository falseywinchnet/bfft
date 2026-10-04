"""Render retained follow-up measurements; performs no benchmark work."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('output/support_geometry/conv_fast_aa')
fig,axs=plt.subplots(1,3,figsize=(16,4.8),layout='constrained')
names=['single','duplicate','overlap','depth_crossing','shared_edges','thin_crossing','affine_shading','four_layers']
for folder,label,mode in [('visibility_v1','Initial exact kernel','fast'),('visibility_v3','Queued boundary pixels','sparse'),('visibility_v4','Queue + full-occluder reduction','sparse')]:
    rec=json.loads((root/folder/'visibility_timing.json').read_text())['records']
    vals=[r['gpu_ms'] for r in rec if r['width']==1920 and (r.get('mode')==mode if 'mode' in r else r['fast_math'])]
    axs[0].plot(range(8),vals,'o-',label=label)
axs[0].set_xticks(range(8),['single','duplicate','overlap','depth\ncrossing','shared\nedges','thin\ncrossing','affine\ncolor','four\nlayers'],rotation=45,ha='right')
axs[0].set_yscale('log');axs[0].set_ylabel('1080p GPU milliseconds (log scale)');axs[0].set_title('Exact visibility: faster, still costly');axs[0].legend(fontsize=7)
aug=json.loads((root/'augmentation_v1/augmentation.json').read_text())
for weight in [0,.5,1.]:
    rows=[r for r in aug['records'] if r['kind']=='sine' and r['weight']==weight]
    fs=sorted(set(r['frequency'] for r in rows));mse=[np.mean([r['mse'] for r in rows if r['frequency']==f]) for f in fs]
    axs[1].plot(fs,mse,label=f'Raw residual weight {weight:g}',linestyle={0:'-',.5:'--',1.:':'}[weight])
axs[1].set_xlabel('Sine frequency (cycles / source pixel)');axs[1].set_ylabel('Mean squared reconstruction error');axs[1].set_title('Near Nyquist: restoring raw current fails');axs[1].legend(fontsize=8)
profiles=np.load(root/'augmentation_v1/augmentation_profiles.npz');x=profiles['x'];i=19*16
for weight in [0,.5,1.]:axs[2].plot(x,profiles['base'][:,i]+weight*profiles['residual'][:,i],label=f'Weight {weight:g}')
axs[2].plot(x,profiles['target'][:,i],color='k',linestyle=':',label='True step');axs[2].set_xlim(61,68);axs[2].set_title('Residual restoration introduces ringing');axs[2].set_xlabel('Source pixel position');axs[2].legend(fontsize=8)
for ax in axs:ax.grid(alpha=.2)
fig.suptitle('Apple M4 follow-up • measured GPU execution and fixed algebraic augmentation',fontsize=14)
fig.savefig(root/'followup_summary.png',dpi=160);plt.close(fig)
ref=np.load(root/'visibility_validated/visibility_reference.npy');gpu=np.fromfile(root/'visibility_validated/visibility_sparse.bin',np.float32).reshape(len(ref),32,32,4)[...,:3]
fig,axs=plt.subplots(3,4,figsize=(11,8),layout='constrained')
for col,i in enumerate([1,3,5,7]):
 axs[0,col].imshow(np.clip(ref[i],0,1),interpolation='nearest');axs[0,col].set_title(names[i].replace('_',' '))
 axs[1,col].imshow(np.clip(gpu[i],0,1),interpolation='nearest')
 axs[2,col].imshow(np.max(abs(gpu[i]-ref[i]),axis=2),vmin=0,vmax=2e-6,cmap='magma',interpolation='nearest')
 for row in range(3):axs[row,col].set_xticks([]);axs[row,col].set_yticks([])
for row,label in enumerate(['Independent oracle','Metal sparse integration','Absolute RGB error\n0–2 × 10⁻⁶']):axs[row,0].set_ylabel(label)
fig.suptitle('Visible color-area integration • thin lines retain their geometric coverage')
fig.savefig(root/'visibility_followup.png',dpi=160)
