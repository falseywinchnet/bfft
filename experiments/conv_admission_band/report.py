"""Render saved scientific evidence; never alter measured arrays."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/support_geometry/conv_admission_band'

def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d=json.loads((OUT/'conv_admission_band_full/results.json').read_text())
    j=json.loads((OUT/'conv_jet_headroom/results.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    colors={'conv':'#263e64','raw':'#b65b45','word2':'#008879'}
    for name,label in [('conv','CONV'),('raw','Admission removed'),('word2','Refined certificate')]:
        rows=[x for x in d['roundtrip_waves'] if x['angle']==0 and x['method']==name]
        axes[0].plot([x['frequency'] for x in rows],[x['gain'] for x in rows],
                     label=label,color=colors[name],linestyle='-' if name=='conv' else '--',
                     linewidth=3 if name=='conv' else 1.4)
    axes[0].set_title('8× round trip: admission is not the droop\nAll three curves overlap')
    axes[0].set_xlabel('Frequency / reduced-lattice axial Nyquist');axes[0].set_ylabel('Retained fundamental amplitude')
    axes[0].legend(fontsize=9);axes[0].set_ylim(0,1.05)
    for name,label in [('conv','Existing fourth-order jet'),('jet6','Sixth-order jet'),('jet8','Eighth-order jet')]:
        rows=[x for x in j['waves'] if x['angle']==0 and x['method']==name]
        fs=sorted({x['frequency'] for x in rows})
        axes[1].plot(fs,[np.mean([x['gain'] for x in rows if x['frequency']==f]) for f in fs],marker='o',label=label)
    axes[1].set_title('Same admission, more accurate derivative bank\n2-D enlargement of axial carriers')
    axes[1].set_xlabel('Frequency / source axial Nyquist');axes[1].legend(fontsize=9);axes[1].set_ylim(0,1.05)
    names=['camera','text','brick','coins','grass','moon'];position=np.arange(len(names))
    for index,method in enumerate(['jet6','jet8']):
        values=[]
        for name in names:
            rows={x['method']:x for x in j['natural'] if x['source']==name}
            values.append(100*(1-rows[method]['both']['mse']/rows['conv']['both']['mse']))
        axes[2].bar(position+(.18 if index else -.18),values,width=.35,label=method)
    axes[2].axhline(0,color='black',linewidth=.8);axes[2].set_xticks(position,names,rotation=30)
    axes[2].set_ylabel('MSE reduction relative to CONV (%)');axes[2].set_title('Natural-image 8× down/up validation\nPositive = improvement; negative = regression');axes[2].legend()
    for ax in axes:ax.grid(axis='y',alpha=.18)
    fig.savefig(OUT/'bandwidth_findings.png',dpi=170);plt.close(fig)
    a=np.load(OUT/'conv_admission_band_full/arrays.npz');b=np.load(OUT/'conv_jet_headroom/arrays.npz')
    images=[a['original'],a['conv'],a['uniform1'],b['jet8']]
    titles=['Original · 512 × 512','CONV · 512 → 64 → 512',
            'Refined admission · same round trip\n0.0086% MSE improvement',
            'Eighth-order proposal · same round trip\n0.2624% MSE improvement; more excursion']
    fig,axes=plt.subplots(1,4,figsize=(17,4.7),constrained_layout=True)
    for ax,z,title in zip(axes,images,titles):
        ax.imshow(z,cmap='gray',vmin=0,vmax=1,interpolation='nearest');ax.set_title(title,fontsize=10);ax.axis('off')
    fig.savefig(OUT/'cameraman_8x.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(17,4.9),constrained_layout=True)
    for ax,z,title in zip(axes,images,titles):
        ax.imshow(z[70:310,120:330],cmap='gray',vmin=0,vmax=1,interpolation='nearest');ax.set_title(title,fontsize=10);ax.axis('off')
    fig.savefig(OUT/'cameraman_8x_detail.png',dpi=170);plt.close(fig)
    # A common display range is used; numerical metrics always use floats.
    for name,z in [('original',images[0]),('conv',images[1]),('refined_admission',images[2]),('eighth_order',images[3])]:
        from PIL import Image
        Image.fromarray(np.rint(255*np.clip(z,0,1)).astype(np.uint8)).save(OUT/(name+'.png'))

if __name__=='__main__':main()
