"""Focused follow-up; every target is the same saved analytic battery."""
from pathlib import Path
from experiments.conv_joint_potential import run


if __name__=='__main__':
    root=Path('/tmp/conv_joint_study');root.mkdir(exist_ok=True)
    run(root/'curved_retry.json',cases=['curved sigmoid'],maxiter=120000)
    run(root/'natural.json',boundary='natural',maxiter=60000)
    for depth,continuity in ((0,1),(1,0),(2,1)):
        run(root/f'ablation_d{depth}_c{continuity}.json',depth=depth,continuity=continuity,
            cases=['diagonal wave','chirped field'],maxiter=60000)
