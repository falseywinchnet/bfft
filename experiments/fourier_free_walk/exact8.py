"""Exactly certified saved walk: eight physical real slots in/out.

Input slots contain x0..x7. Output slots contain
DC, Re1, Im1, Re2, Im2, Re3, Im3, Nyquist.
20 additions/subtractions, 2 nontrivial multiplications, 3 signed exchanges.
The signed exchanges remain explicit work; this is not a new FFT claim.
"""
import math
HALF_ROOT2=math.sqrt(.5)


def transform_inplace(v):
    if len(v)!=8:raise ValueError('exact8 requires eight physical slots')
    a,b=v[1],v[7];v[1]=b-a;v[7]=a+b
    a,b=v[2],v[6];v[2]=a+b;v[6]=a-b
    a,b=v[5],v[7];v[5]=b;v[7]=-a
    a,b=v[1],v[4];v[1]=b;v[4]=-a
    a,b=v[2],v[3];v[2]=-b;v[3]=-a
    a,b=v[2],v[7];v[2]=b-a;v[7]=a+b
    a,b=v[5],v[7];v[5]=-HALF_ROOT2*(a+b);v[7]=b-a
    a,b=v[2],v[4];v[2]=a+b;v[4]=a-b
    a,b=v[0],v[1];v[0]=a+b;v[1]=a-b
    a,b=v[2],v[6];r=-HALF_ROOT2*a;v[2]=r-b;v[6]=r+b
    a,b=v[0],v[3];v[0]=a-b;v[3]=a+b
    a,b=v[0],v[7];v[0]=a-b;v[7]=a+b
    a,b=v[1],v[5];v[1]=a-b;v[5]=a+b
    return v


if __name__=='__main__':
    import argparse,json
    p=argparse.ArgumentParser();p.add_argument('samples',nargs=8,type=float);a=p.parse_args()
    print(json.dumps(transform_inplace(a.samples)))
