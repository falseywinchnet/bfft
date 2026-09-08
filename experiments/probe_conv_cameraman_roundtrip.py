"""Cameraman 512 -> 128 -> 512: demo CONV versus budgeted characteristic path.

The experimental mode has no downsampler. Both arms use demo CONV analysis.
Its fixed-scale wrapper is extended ONLY to arbitrary endpoint query locations,
with the same Python CONV currents on fallback. Demo two-order and experimental
single-order fallback are explicitly distinguished in the saved record.
"""
import argparse,json,math,time
from pathlib import Path
import numpy as np
from PIL import Image
from experiments.conv_budget_modes import compile_mode,evaluate
from experiments.convstar import raw_current_jet_bank,ordered_sign_ledger,project_signed_fibres,convstar_resize
from experiments.conv_distilled_core import distilled_conv_resize
from standalone_conv_resize_demo.backend import conv_resize as native_single


def arbitrary_single_order(source,shape):
    z=np.asarray(source,dtype=float)
    for axis in (1,0):
        lines=np.moveaxis(z,axis,0);old=lines.shape;lines=lines.reshape(old[0],-1)
        raw,delta=raw_current_jet_bank(lines)
        current=project_signed_fibres(raw,ordered_sign_ledger(raw,delta),delta)
        query=np.linspace(0,len(lines)-1,shape[axis]);idx=np.minimum(query.astype(int),len(lines)-2);u=query-idx
        bern=np.stack([math.comb(5,k)*u**k*(1-u)**(5-k) for k in range(6)],axis=1)
        tails=np.cumsum(bern[:,1:][:,::-1],axis=1)[:,::-1]
        fine=lines[idx]+np.einsum('qkc,qk->qc',current[idx],tails)
        z=np.moveaxis(fine.reshape((len(query),)+old[1:]),0,axis)
    return z


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    path=Path('personal_deblurrer/source_assets/v3_skimage/camera.png')
    source=np.asarray(Image.open(path).convert('L'),dtype=np.float32)/255
    assert source.shape==(512,512),source.shape
    small=distilled_conv_resize(source,(128,128))
    baseline=distilled_conv_resize(small,source.shape)
    model,diag=compile_mode(small)
    if model is None:
        candidate=arbitrary_single_order(small,source.shape)
    else:
        q=np.linspace(0,127,512);candidate=evaluate(model,q[None,:],q[:,None])
    single=native_single(small,source.shape)
    # Verify arbitrary-query extension against the existing integer wrapper.
    nested=arbitrary_single_order(small,(509,509));old=convstar_resize(small,4)
    extension_error=float(np.max(abs(nested-old)))
    assert extension_error<1e-12
    assert np.isfinite(baseline).all() and np.isfinite(candidate).all()
    def metrics(z):
        mse=float(np.mean((z.astype(float)-source)**2))
        return dict(mse=mse,psnr_db=-10*math.log10(mse),minimum=float(z.min()),maximum=float(z.max()),
                    below_zero_pixels=int(np.sum(z<0)),above_one_pixels=int(np.sum(z>1)),
                    outside_source_range_pixels=int(np.sum((z<source.min())|(z>source.max()))))
    result=dict(source=str(path),source_shape=list(source.shape),reduced_shape=list(small.shape),
                baseline='image-demo distilled CONV: basin analysis + two-order tensor-weighted synthesis',
                new='same basin analysis; budget mode on scalar reduced image; original single-order Python CONV fallback',
                admission=diag,conv=metrics(baseline),new_approach=metrics(candidate),
                maximum_difference=float(np.max(abs(candidate-baseline))),
                candidate_vs_native_single_max=float(np.max(abs(candidate-single))),
                arbitrary_query_vs_existing_wrapper_max=extension_error,
                equal_demo_fallback_would_be_identical=model is None,
                clipping='float arrays and metrics unclipped; PNGs clipped to [0,1] only for display')
    np.savez_compressed(out/'arrays.npz',original=source,coarse=small,conv=baseline,new=candidate,native_single=single)
    for name,z in [('original',source),('coarse',small),('conv',baseline),('new',candidate)]:
        Image.fromarray(np.rint(np.clip(z,0,1)*255).astype(np.uint8)).save(out/(name+'.png'))
    (out/'metrics.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


def plot(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out);a=np.load(out/'arrays.npz')
    titles=['Original — 512 × 512','CONV demo — 512 → 128 → 512','New approach — 512 → 128 → 512\nGeometric mode rejected; single-order fallback']
    fig,axes=plt.subplots(1,3,figsize=(15.4,5.6),constrained_layout=True)
    for ax,key,title in zip(axes,['original','conv','new'],titles):
        ax.imshow(a[key],cmap='gray',vmin=0,vmax=1,interpolation='nearest');ax.set_title(title,fontsize=11);ax.axis('off')
    fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)
    # Identical pixel crop around head, camera and sky: no interpolating display filter.
    crop=np.s_[80:240,140:320]
    fig,axes=plt.subplots(1,3,figsize=(15.4,5.3),constrained_layout=True)
    for ax,key,title in zip(axes,['original','conv','new'],titles):
        ax.imshow(a[key][crop],cmap='gray',vmin=0,vmax=1,interpolation='nearest');ax.set_title(title,fontsize=11);ax.axis('off')
    fig.suptitle('Same crop in all three: head, camera and high-contrast sky boundary',fontsize=13)
    fig.savefig(out/'detail.png',dpi=150);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='/tmp/conv_cameraman_roundtrip');p.add_argument('--plot-only',action='store_true');a=p.parse_args()
    plot(a.out) if a.plot_only else run(a.out)
