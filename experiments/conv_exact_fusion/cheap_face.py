"""Isolated native local-face certificate, against the promoted ledger."""
from pathlib import Path
from experiments.conv_admission_band.core import Operator

HERE = Path(__file__).resolve().parent

def transform(source):
    marker='    float ordered[5];memcpy(ordered,a,sizeof(ordered));sort_five(ordered);'
    assert source.count(marker)==1
    return source.replace(marker,(HERE/'cheap_face.h').read_text()+'\n'+marker)

def reverse_transform(source):
    start=source.index('    double prefix=0.0,theta=0.0;',source.index('static void simplex_five'))
    end=source.index('    double mass=0.0;',start)
    return source[:start]+(HERE/'reverse_face.h').read_text()+source[end:]

def operator(name):
    if name == 'base': return Operator('conv')
    if name == 'reverse': return Operator('conv',source_transform=reverse_transform)
    if name == 'face': return Operator('conv',source_transform=transform)
    raise ValueError(name)

if __name__ == '__main__':
    import argparse
    from experiments.conv_exact_fusion import study
    p=argparse.ArgumentParser()
    p.add_argument('--out',required=True)
    p.add_argument('--methods',default='base,face')
    p.add_argument('--repeats',type=int,default=31)
    a=p.parse_args()
    study.operator=operator
    study.run(a.out,a.repeats,a.methods)
