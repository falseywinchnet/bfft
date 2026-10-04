"""Independent analytic similarity and center-offset census for the annular trace."""
from pathlib import Path
import argparse,json
import numpy as np
from .radial import unwind,trace_similarity
from .test_radial import field


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=Path('/tmp/scene_pattern_radial'));args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ref=field(161);moving=field(161,1.16,23)
    trace,_,meta=unwind(ref,(80,80),r_min=7,r_max=65)
    rows=[]
    for offset in (0,1,2,4,8):
        other,_,_=unwind(moving,(80+offset,80-.625*offset),r_min=7,r_max=65)
        rows.append({'center_offset_px':[offset,-.625*offset],**trace_similarity(trace,other,meta['log_radius_step'])})
    correct,_,_=unwind(moving,(80,80),r_min=7,r_max=65)
    fig,ax=plt.subplots(2,3,figsize=(13,7),facecolor='#eef1ed')
    ax[0,0].imshow(ref,cmap='magma');ax[0,0].set_title('Analytic reference')
    ax[0,1].imshow(moving,cmap='magma');ax[0,1].set_title('Scale 1.16 · rotation +23°')
    ax[0,2].plot([r['center_offset_px'][0] for r in rows],[r['correlation'] for r in rows],'o-',color='#316941');ax[0,2].set(xlabel='Center offset x (pixels)',ylabel='Trace correlation',title='Center error reduces agreement')
    for a,data,title in [(ax[1,0],trace,'Reference unwound trace'),(ax[1,1],correct,'Moving unwound trace')]:
        a.imshow(data,origin='lower',aspect='auto',extent=(0,360,np.log(7),np.log(65)),cmap='magma');a.set(xlabel='Angle (degrees)',ylabel='log radius',title=title)
    ax[1,2].axis('off');r=rows[0];ax[1,2].text(0,.8,f"Recovered scale: {r['scale']:.4f}\nRecovered angle: {r['angle_degrees']:.2f}°\n\nAngle wraps. Radius does not.\nAngular phase is retained.\nA correct center is an assumption.",va='top',fontsize=12)
    fig.tight_layout();fig.savefig(args.out/'radial-trace.png',dpi=150);plt.close(fig)
    (args.out/'receipt.json').write_text(json.dumps({'truth':{'scale':1.16,'angle_degrees':23},'center_census':rows},indent=2)+'\n')
    print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
