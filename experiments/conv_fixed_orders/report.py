"""Render retained fixed-order measurements; no recomputation of the operator."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS={'v1':'Version 1.0 (6 / 5)','4c3':'4 source / 3 current','4c5':'4 source / 5 current',
        '6c3':'6 source / 3 current','6c5':'6 / 5 control','8c5':'8 source / 5 current','8c7':'8 source / 7 current'}
COLORS={'v1':'#222222','4c3':'#d18420','4c5':'#bcab62','6c3':'#7383b5','6c5':'#555555','8c5':'#268d83','8c7':'#ab4479'}

def render(root):
    root=Path(root);p=json.loads((root/'results.json').read_text());a=np.load(root/'arrays.npz')
    names=['4c3','4c5','6c3','6c5','8c5','8c7'];times=p['timing'][0]['methods'];base=times['v1']['median']
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    for k in names:
        r=p['spectra'][k];axes[0,0].plot(r['f'],r['gain'],color=COLORS[k],label=LABELS[k],linewidth=2)
    axes[0,0].set(xlim=(.4,1.65),ylim=(-.04,1.07),xlabel='Frequency / source Nyquist',ylabel='Coherent gain',title='Raw proposal: upper passband / stopband transition')
    axes[0,0].axvline(1,color='gray',lw=.7,ls=':');axes[0,0].legend(fontsize=8)
    for k in ['v1','4c3','8c5','8c7']:
        rows=[r for r in p['carriers'] if r['method']==k and r['gain'] is not None]
        fs=sorted(set(r['frequency'] for r in rows));ys=[np.mean([r['gain'] for r in rows if r['frequency']==f]) for f in fs]
        axes[0,1].plot(fs,ys,color=COLORS[k],label=LABELS[k],linewidth=2)
    axes[0,1].set(xlim=(.4,1.65),ylim=(-.04,1.1),xlabel='Frequency / source Nyquist',ylabel='Fundamental gain',title='Actual admission: two source phases averaged')
    axes[0,1].legend(fontsize=8)
    axes[0,1].annotate('Nyquist phase ambiguity',xy=(1,.78),xytext=(1.13,.88),fontsize=8,arrowprops={'arrowstyle':'->','lw':.8})
    values=[times[k]['median']/base for k in names]
    bars=axes[1,0].bar(range(len(names)),values,color=[COLORS[k] for k in names])
    axes[1,0].axhline(1,color='black',ls='--',lw=1)
    axes[1,0].bar_label(bars,labels=[f'{v:.2f}×' for v in values],padding=3)
    axes[1,0].set(xticks=range(len(names)),xticklabels=names,ylabel='Time / Version 1.0',title='Native cameraman 512 → 64 → 512, four workers',ylim=(0,max(values)*1.18))
    images=['camera','text','brick','coins','grass','moon']
    data=[]
    for k in names:
        row=[]
        for im in images:
            baseline=next(r['roundtrip']['mse'] for r in p['images'] if r['image']==im and r['method']=='v1')
            value=next(r['roundtrip']['mse'] for r in p['images'] if r['image']==im and r['method']==k)
            row.append(100*(baseline-value)/baseline)
        data.append(row)
    vmax=max(.1,np.max(np.abs(data)))
    axes[1,1].imshow(data,cmap='RdYlGn',vmin=-vmax,vmax=vmax,aspect='auto')
    axes[1,1].set(xticks=range(6),xticklabels=images,yticks=range(len(names)),yticklabels=names,title='Round-trip MSE improvement (%) — positive is better')
    for i,row in enumerate(data):
        for j,v in enumerate(row):axes[1,1].text(j,i,f'{v:+.2f}',ha='center',va='center',fontsize=9)
    fig.suptitle('Fixed CONV orders: source support and downstream degree',fontsize=15)
    fig.savefig(root/'comparison.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(2,5,figsize=(15,6.7),layout='constrained')
    for j,k in enumerate(['original','v1','4c3','8c5','8c7']):
        z=a[k];ax[0,j].imshow(z,cmap='gray',vmin=0,vmax=1);ax[1,j].imshow(z[80:240,140:300],cmap='gray',vmin=0,vmax=1)
        ax[0,j].set_title('Original' if k=='original' else LABELS[k],fontsize=10)
        for axis in ax[:,j]:axis.axis('off')
    fig.suptitle('Cameraman: full 8× basin reduction and 8× reconstruction\nDisplay limits only; measurements use unclipped arrays',fontsize=12)
    fig.savefig(root/'cameraman.png',dpi=170);plt.close(fig)
    lines=['# Fixed-order measured results','',f"Native four-worker, 31-repeat medians. Version 1.0 complete round trip: {base*1000:.4f} ms.",'',
           '| Method | Raw 90–10 width | Width change | Gain at 0.9 Nyquist | Round-trip time | Synthesis time | Camera MSE improvement | Worst thin-strip excursion |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    baseline_width=p['spectra']['6c5']['width90_10'];ups=next(r['methods'] for r in p['timing'] if r['case']=='camera_synthesis')
    base_mse=next(r['roundtrip']['mse'] for r in p['images'] if r['image']=='camera' and r['method']=='v1')
    for k in ['v1']+names:
        sp=p['spectra']['6c5' if k=='v1' else k]
        mse=next(r['roundtrip']['mse'] for r in p['images'] if r['image']=='camera' and r['method']==k)
        exc=max(r['excursion'] for r in p['edges'] if r['method']==k and r['kind']=='strip')
        lines.append(f"| {k} | {sp['width90_10']:.5f} | {100*(sp['width90_10']/baseline_width-1):+.2f}% | {sp['gain_at_0_9']:.5f} | {times[k]['median']/base:.3f}× | {ups[k]['median']/ups['v1']['median']:.3f}× | {100*(base_mse-mse)/base_mse:+.3f}% | {exc:.5f} |")
    lines+=['','Raw width is not a universal transfer width for the nonlinear admitted operator. See the admitted-carrier curves, orientation probes, and excursions alongside it.',
            '','The generated 6c5 control and Version 1.0 have the same mathematical bank; its timing difference measures implementation overhead. No variant is automatically selected at runtime.']
    (root/'SUMMARY.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');render(p.parse_args().root)
