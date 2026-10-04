"""Fixed raw-minus-admitted current augmentation, without extra source analysis."""
import argparse
import json
from pathlib import Path
import numpy as np
from experiments.convstar import raw_current_jet_bank,ordered_sign_ledger,project_signed_fibres,synthesize_polyphase


def representations(source,scale=8):
    source=np.asarray(source,dtype=float)
    if source.ndim==1:source=source[:,None]
    raw,delta=raw_current_jet_bank(source)
    admitted=project_signed_fibres(raw,ordered_sign_ledger(raw,delta),delta)
    base=synthesize_polyphase(source,admitted,scale)
    residual=synthesize_polyphase(np.zeros_like(source),raw-admitted,scale)
    return base,residual,raw,admitted


def run(out):
    out.mkdir(parents=True,exist_ok=True);n=129;scale=8;x=np.arange(n);xx=np.arange((n-1)*scale+1)/scale
    frequencies=np.linspace(.05,.5,19);phases=(np.arange(16)+.5)/16
    columns=[];truth=[];metadata=[]
    for f in frequencies:
        for phase in phases:
            columns.append(.5+.5*np.sin(2*np.pi*(f*x+phase)))
            truth.append(.5+.5*np.sin(2*np.pi*(f*xx+phase)))
            metadata.append({'kind':'sine','frequency':float(f),'phase':float(phase)})
    for kind in ('step','strip'):
        for phase in phases:
            for width in ((1.,) if kind=='step' else (.35,.75,1.5)):
                centre=64+phase
                fn=(lambda t:(t>=centre).astype(float)) if kind=='step' else (lambda t:(abs(t-centre)<width/2).astype(float))
                columns.append(fn(x));truth.append(fn(xx));metadata.append({'kind':kind,'width':width,'phase':float(phase)})
    source=np.array(columns).T;target=np.array(truth).T;base,residual,raw,admitted=representations(source,scale)
    interior=slice(4*scale,-4*scale);records=[]
    for lam in (0,.25,.5,.75,1.):
        value=base+lam*residual
        for i,meta in enumerate(metadata):
            v=value[interior,i];truth_i=target[interior,i]
            records.append(dict(meta,weight=lam,mse=float(np.mean((v-truth_i)**2)),range_excursion=float(max(0,-v.min(),v.max()-1)),peak_to_peak=float(np.ptp(v))))
    fused_error=max(float(np.max(abs(synthesize_polyphase(source,admitted+lam*(raw-admitted),scale)-(base+lam*residual)))) for lam in (.25,.5,.75,1.))
    # A direct nonnegative reconstruction can reproduce this high-frequency
    # alternating signal. Ringing-free is not synonymous with band-limited.
    alternating=(x%2).astype(float);alt_base,_,_,_=representations(alternating,scale)
    summary={'scope':'1-D synthesis experiment; all phase/frequency records retained. Point-sampled targets, no unseen phase information supplied. Raw currents and admission already exist in CONV; residual construction adds no analysis. Lambda>0 changes the admitted law and may ring. No lambda promoted.',
        'fused_vs_separate_max_error':fused_error,'residual_at_source_nodes_max':float(np.max(abs(residual[::scale]))),
        'alternating_nyquist':{'source_peak_to_peak':float(np.ptp(alternating)),'base_peak_to_peak':float(np.ptp(alt_base[interior])),'range_excursion':float(max(0,-alt_base.min(),alt_base.max()-1))},'records':records}
    (out/'augmentation.json').write_text(json.dumps(summary,indent=2)+'\n')
    np.savez_compressed(out/'augmentation_profiles.npz',x=xx,source=source,base=base,residual=residual,target=target)
    print('Fused error',fused_error,'nodal residual',summary['residual_at_source_nodes_max'],flush=True)
    for f in (.4,.45,.5):
        print('frequency',f,{lam:float(np.mean([r['mse'] for r in records if r['kind']=='sine' and abs(r['frequency']-f)<1e-8 and r['weight']==lam])) for lam in (0,.25,.5,.75,1.)},flush=True)
    print('step excursions',{lam:max(r['range_excursion'] for r in records if r['kind']=='step' and r['weight']==lam) for lam in (0,.25,.5,.75,1.)},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.out)
