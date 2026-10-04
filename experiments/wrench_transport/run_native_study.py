#!/usr/bin/env python3
"""Matched native comparisons; timing surrounds World::step only, sleep is off."""
import argparse, json, platform, statistics, subprocess, tempfile
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--binary', action='append', required=True, help='label=/absolute/path')
p.add_argument('--node', default='node')
p.add_argument('--output', required=True)
p.add_argument('--repeats', type=int, default=7)
a = p.parse_args()
root = Path(__file__).resolve().parent
binaries = dict(item.split('=', 1) for item in a.binary)
seeds = [1, 7, 19, 101, 271, 314159, 8675309]
records = []
with tempfile.TemporaryDirectory(prefix='wrench-study-') as directory:
    directory = Path(directory)
    for seed in seeds:
        scene = directory / 'scene.txt'
        subprocess.run([a.node, str(root / 'container.mjs'), '--seed', str(seed), '--export-text', str(scene)], check=True, capture_output=True)
        reference = None
        for repeat in range(a.repeats + 1):
            # Rotate order to avoid systematically favoring the second executable.
            labels = list(binaries)
            labels = labels[repeat % len(labels):] + labels[:repeat % len(labels)]
            for label in labels:
                output = directory / 'record.json'
                process = subprocess.run([binaries[label], str(scene), str(output), '60', '8'], check=True, capture_output=True, text=True)
                result = json.loads(output.read_text())
                physical = {key: result[key] for key in ['trajectoryHash', 'finalPoses', 'samples', 'floorForce', 'work']}
                if reference is None:
                    reference = physical
                if physical != reference:
                    raise RuntimeError(f'physics mismatch: seed={seed} repeat={repeat} binary={label}')
                if repeat:
                    records.append(dict(seed=seed, repeat=repeat, binary=label, timing=result['timing'], worstStepMs=result['worstStepMs'], trajectoryHash=result['trajectoryHash'], work=result['work'], stdout=process.stdout))
        print(f'seed {seed}: all trajectories and work counters identical', flush=True)
summary = []
for label in binaries:
    selected = [r for r in records if r['binary'] == label]
    summary.append(dict(binary=label, meanMs=statistics.mean(r['timing']['meanMs'] for r in selected), medianMeanMs=statistics.median(r['timing']['meanMs'] for r in selected)))
payload = dict(platform=platform.platform(), machine=platform.machine(), seeds=seeds, repeats=a.repeats, warmupRunsPerSeed=1, hertz=60, seconds=8, sleeping=False, trajectoriesIdentical=True, summary=summary, records=records)
Path(a.output).write_text(json.dumps(payload, indent=2) + '\n')
print(json.dumps(summary, indent=2))
