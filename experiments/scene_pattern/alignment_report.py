"""Attach applied-field diagnostics to a frozen average-alignment receipt."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);args=parser.parse_args()
    path=args.directory/'receipt.json';r=json.loads(path.read_text())
    with np.load(args.directory/'fields.npz') as state:
        fields=state['fields'];weights=state['masks'];gradients=[np.gradient(f,axis=(0,1)) for f in fields]
        for chart in r['records']:
            x,y=chart['center'];chart['applied_centers']=[fields[i,y,x].tolist() for i in chart['views']]
            chart['applied_gradient_xy']=[np.column_stack((gradients[i][1][y,x],gradients[i][0][y,x])).tolist() for i in chart['views']]
            total=np.zeros((2,2));mass=0
            for component in chart['components']:
                ids=component['vertices'];positions=np.asarray(chart['centers'])[ids]
                w=np.array([weights[chart['views'][j],y,x] for j in ids]);center=np.sum(positions*w[:,None],0)/max(w.sum(),1e-20)
                d=positions-center;total+=(d*w[:,None]).T@d;mass+=w.sum()
            chart['weighted_within_component_covariance']=(total/max(mass,1e-20)).tolist()
    r['report_notes']='Applied centers and gradient are sampled from the frozen, shrunk, interpolated field. Weighted within-component covariance uses native comparison weights; disconnected-group offsets are unidentifiable, so between-component spread is omitted.'
    r['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('average_alignment.py'),Path('personal_deblurrer/circles.py'),Path('personal_deblurrer/multicapture_transport.py'),Path('personal_deblurrer/decomposition.py')]}
    path.write_text(json.dumps(r,indent=2)+'\n')
if __name__=='__main__':main()
