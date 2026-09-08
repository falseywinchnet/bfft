"""Fixed natural-boundary joint potential on different sample counts/fields."""
import json
from pathlib import Path
import numpy as np
from experiments.conv_joint_potential import admit,evaluate,run
from experiments.probe_conv_compatible_2d import resize


def metrics(z,truth,source,mask):
    dz=np.array(np.gradient(z,edge_order=2));dt=np.array(np.gradient(truth,edge_order=2))
    return {'mse':float(np.mean((z[mask]-truth[mask])**2)),
            'full_mse':float(np.mean((z-truth)**2)),
            'discrete_gradient_mse':float(np.mean(np.sum((dz-dt)**2,axis=0)[mask])),
            'excursion':float(max(0,z.max()-source.max(),source.min()-z.min())),
            'cardinality_error':float(np.max(abs(z[::3,::3]-source)))}


def validate(out):
    n=13;scale=3
    yy,xx=np.meshgrid(np.linspace(0,1,n),np.linspace(0,1,n),indexing='ij')
    v,u=np.meshgrid(np.linspace(0,1,37),np.linspace(0,1,37),indexing='ij')
    mask=(u>.2)&(u<.8)&(v>.2)&(v<.8)
    fields=[]
    for fx,fy in ((2.7,1.3),(3.8,2.1),(4.9,-2.8)):
        for phase in (.19,.57):
            fields.append((f'wave {fx} {fy} {phase}','wave',
                           lambda x,y,fx=fx,fy=fy,p=phase:.5+.4*np.sin(2*np.pi*(fx*x+fy*y+p))))
    for slope,width in ((-.4,18),(.35,30),(1.1,42)):
        fields.append((f'sigmoid {slope} {width}','sigmoid',
                       lambda x,y,s=slope,w=width:.5*(1+np.tanh(w*(x+s*(y-.5)-.48)))))
    for curvature in (1.7,2.2,2.8):
        fields.append((f'chirp {curvature}','chirp',
                       lambda x,y,c=curvature:.5+.4*np.sin(2*np.pi*(.4*x+c*x*x+1.3*y*y+.37))))
    rows=[];images={}
    for name,family,f in fields:
        source=f(xx,yy);truth=f(u,v)
        p,d=admit(source,boundary='natural',maxiter=120000)
        joint=evaluate(p,12*u,12*v)
        images[name]={'source':source.tolist(),'truth':truth.tolist()}
        for method in ('CONV','Lanczos-3','joint natural'):
            z=joint if method=='joint natural' else resize(source,scale,method)
            row={'case':name,'family':family,'method':method,**metrics(z,truth,source,mask)}
            if method=='joint natural':
                row.update(d)
            rows.append(row);images[name][method]=z.tolist()
            print(json.dumps(row),flush=True)
        Path(out).write_text(json.dumps({'n':n,'scale':scale,'rows':rows,'images':images},indent=2)+'\n')


if __name__=='__main__':
    root=Path('/tmp/conv_joint_final');root.mkdir(exist_ok=True)
    run(root/'natural.json',boundary='natural',maxiter=120000)
    validate(root/'validation.json')
