"""Independent NumPy check of explicit native input/output binary records."""
import argparse
import json
import struct
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument("dump")
parser.add_argument("--out",required=True)
args=parser.parse_args()
results=[]
methods=("dit","dif","stockham","packet","cancelled","fused")
with open(args.dump,"rb") as f:
    while True:
        header=f.read(4)
        if not header:break
        n,=struct.unpack("<i",header)
        data=f.read(16*n*7)
        assert len(data)==16*n*7
        rows=np.frombuffer(data,dtype="<c16").reshape(7,n)
        expected=np.fft.fft(rows[0])
        for method,actual in zip(methods,rows[1:]):
            err=float(np.max(np.abs(actual-expected)))
            rel=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
            assert err<2e-10 and rel<3e-14,(n,method,err,rel)
            results.append({"N":n,"method":method,"max_absolute_error":err,"relative_l2_error":rel})
assert len(results)==48,"Expected all eight sizes and all six methods"
with open(args.out,"w") as f:
    json.dump({"oracle":"numpy.fft.fft","numpy_version":np.__version__,"results":results},f,indent=2)
    f.write("\n")
print("Independent native oracle passed",len(results),"cases; max absolute error",
      max(x["max_absolute_error"] for x in results))
