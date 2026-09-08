"""Matched Cartesian-factor test of the new current on analytic 2-D fields."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from experiments.conv_compatible_current import admitted
from experiments.convstar import refine_lines,synthesize_polyphase
from experiments.conv_functional_current import lanczos


def resize(y,scale,method):
    z=np.asarray(y,dtype=float)
    for axis in (1,0):
        lines=np.moveaxis(z,axis,0)
        shape=lines.shape
        lines=lines.reshape(shape[0],-1)
        if method=='CONV':
            fine=refine_lines(lines,scale)
        elif method=='Lanczos-3':
            query=np.linspace(0,len(lines)-1,(len(lines)-1)*scale+1)
            fine=np.stack([lanczos(lines[:,ch],query) for ch in range(lines.shape[1])],axis=1)
        else:
            c=admitted(lines,bounded=(method=='compatible bounded'),
                       envelope=(method=='compatible envelope'))[0]
            fine=synthesize_polyphase(lines,c,scale)
        z=np.moveaxis(fine.reshape((len(fine),)+shape[1:]),0,axis)
    return z


def run(out):
    n=17;scale=3
    src=np.linspace(0,1,n);tar=np.linspace(0,1,(n-1)*scale+1)
    sy,sx=np.meshgrid(src,src,indexing='ij');ty,tx=np.meshgrid(tar,tar,indexing='ij')
    fields=[('diagonal wave',lambda x,y:.5+.4*np.sin(2*np.pi*(5*x+3*y+.13))),
            ('crossing waves',lambda x,y:.5+.2*np.sin(2*np.pi*(5*x+2*y+.17))+.2*np.cos(2*np.pi*(2*x-5*y+.31))),
            ('diagonal sigmoid',lambda x,y:.5*(1+np.tanh((x+.7*y-.86)*20))),
            ('curved sigmoid',lambda x,y:.5*(1+np.tanh((np.hypot(x-.5,y-.5)-.27)*35))),
            ('diagonal step',lambda x,y:(x+.7*y>.86).astype(float)),
            ('disk',lambda x,y:(np.hypot(x-.5,y-.5)<.27).astype(float)),
            ('chirped field',lambda x,y:.5+.4*np.sin(2*np.pi*(x+3*x*x+2*y*y+.23))),
            ('edge and texture',lambda x,y:.4*(1+np.tanh((x+.4*y-.73)*25))+.08*np.sin(2*np.pi*(5*x-3*y+.13)))]
    methods=('CONV','compatible functional','compatible bounded','compatible envelope','Lanczos-3')
    mask=(tx>.2)&(tx<.8)&(ty>.2)&(ty<.8)
    rows=[];images={}
    for name,f in fields:
        source=f(sx,sy);truth=f(tx,ty)
        images[name]={'source':source.tolist(),'truth':truth.tolist()}
        for method in methods:
            start=time.perf_counter();z=resize(source,scale,method)
            row={'case':name,'method':method,
                 'mse':float(np.mean((z[mask]-truth[mask])**2)),
                 'full_mse':float(np.mean((z-truth)**2)),
                 'excursion':float(max(0,z.max()-source.max(),source.min()-z.min())),
                 'cardinality_error':float(np.max(abs(z[::scale,::scale]-source))),
                 'seconds':time.perf_counter()-start}
            rows.append(row);images[name][method]=z.tolist()
            print(json.dumps(row),flush=True)
    Path(out).write_text(json.dumps({'n':n,'scale':scale,'rows':rows,'images':images},indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='/tmp/conv_compatible_2d.json')
    run(p.parse_args().out)
