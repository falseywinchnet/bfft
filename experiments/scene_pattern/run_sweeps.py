"""Checked collection runner; preserves each stage's receipts on failure."""
import argparse,subprocess,sys,shutil
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--features',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--reuse-graph',type=Path);args=p.parse_args()
    subprocess.run([sys.executable,'-m','unittest','experiments.scene_pattern.test_sweeps','-v'],check=True)
    if args.reuse_graph:
        shutil.copytree(args.reuse_graph.parent,args.out/'graph',dirs_exist_ok=True)
    else:
        subprocess.run([sys.executable,'-m','experiments.scene_pattern.sweep_graph','--data',str(args.data),'--features',str(args.features),'--out',str(args.out/'graph')],check=True)
    subprocess.run([sys.executable,'-m','experiments.scene_pattern.sweep_fusion','--data',str(args.data),'--features',str(args.features),'--graph',str(args.out/'graph'/'graph.json'),'--out',str(args.out/'scene')],check=True)
if __name__=='__main__':main()
