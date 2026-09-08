"""A short, fully recorded observation of one periodic 64-sample edge."""
from pathlib import Path
import sys,json,time,argparse
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.meyer_transport_audit.model import ReducedMeyerMap,grad,rms
from experiments.meyer_transport_audit.certificate import certificate
from experiments.meyer_transport_audit.intermediate_chart import labels

def run():
    x=np.arange(64)[None,:];f=50.+150.*(x>32)
    m=ReducedMeyerMap(f,.05,40);z=m.initial();frames=[];states=[];events=[]
    start=time.perf_counter()
    previous=None
    for k in range(1,321):
        if k>1:z=m.step(z)
        state=m.pack(z).reshape(6,64);states.append(state.copy())
        label=(labels(z.tux,m.ru).ravel(),labels(z.twx,m.rw).ravel())
        if previous is not None:
            for b in [0,1]:
                for i in np.flatnonzero(label[b]!=previous[b]):
                    events.append({'pass':k,'branch':'u' if b==0 else 'w','sample':int(i),
                                   'from':int(previous[b][i]),'to':int(label[b][i])})
        previous=label
        gap=certificate(m,z)
        frames.append({'pass':k,'gap':gap['gap_per_pixel'],
                       'u':z.u.ravel().tolist(),'w':z.w.ravel().tolist(),
                       'tu':z.tux.ravel().tolist(),'tw':z.twx.ravel().tolist()})
    elapsed=time.perf_counter()-start
    e=next(e for e in events if e['pass']>64)
    i=e['sample'];field=2 if e['branch']=='u' else 4
    rows=[]
    for k in [64,128,192,212,213,214,215,240,280,320]:
        s=states[k-1];nexts=states[k] if k<320 else None
        primal=s[0] if field==2 else s[1]
        b=s[field]-(np.roll(primal,-1)-primal)
        rows.append({'pass':k,'event_field_value':float(s[field,i]),
                     'event_incoming_b':float(b[i]),'event_primal_gradient':float(np.roll(primal,-1)[i]-primal[i]),
                     'event_field_increment':float(nexts[field,i]-s[field,i]) if nexts is not None else None,
                     'gap_per_pixel':frames[k-1]['gap'],
                     'u_change_rms_next':rms(nexts[0]-s[0]) if nexts is not None else None})
    return {'source':f.ravel().tolist(),'lambda':.05,'mu':40,'u_radius':m.ru,'w_radius':m.rw,
            'seconds_including_diagnostics':elapsed,'first_event_after64':e,
            'event_inspection':rows,'events':events,'frames':frames},np.asarray(states)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    r,s=run();Path(a.out+'.json').write_text(json.dumps(r,separators=(',',':')))
    np.savez_compressed(a.out+'.npz',states=s)
    print(json.dumps({k:v for k,v in r.items() if k not in ['frames','source','events']},indent=2))
