"""Repeat the same complete native timing gate at another worker count."""
import argparse,json,os,time
from pathlib import Path
import numpy as np
from PIL import Image
from experiments.conv_fixed_orders.core import CONFIGS,FixedOrder,Legacy,HERE

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--repeats',type=int,default=31);args=p.parse_args()
    frozen=(HERE/'v1_native.c').read_text()
    ops={'v1':Legacy('conv',source_transform=lambda _:frozen)}
    ops.update({k:FixedOrder(*v) for k,v in CONFIGS.items()})
    src=np.asarray(Image.open('personal_deblurrer/source_assets/v3_skimage/camera.png'),dtype=np.float32)/255
    coarse=ops['v1'].reduce(src,(64,64));names=list(ops)
    result={'threads':os.getenv('CONV_NATIVE_THREADS'),'timing':[]}
    jobs={'camera_roundtrip':lambda op:op.synthesize(op.reduce(src,(64,64)),(512,512)),
          'camera_synthesis':lambda op:op.synthesize(coarse,(512,512)),
          'camera_reduction':lambda op:op.reduce(src,(64,64))}
    for label,call in jobs.items():
        times={k:[] for k in names}
        for k in names:call(ops[k])
        for rep in range(args.repeats):
            for k in (names if rep%2==0 else names[::-1]):
                t=time.perf_counter();call(ops[k]);times[k].append(time.perf_counter()-t)
        result['timing'].append({'case':label,'methods':{k:{'median':float(np.median(v)),'seconds':v} for k,v in times.items()}})
        print(label,{k:round(np.median(v)/np.median(times['v1']),3) for k,v in times.items()},flush=True)
    Path(args.out).write_text(json.dumps(result,indent=2)+'\n')
