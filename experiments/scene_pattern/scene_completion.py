"""Extend physical-point observations through agreeing measured transport paths.

This changes geometry evidence only. It does not grant a renderer permission to
warp a photograph through a long chain, or equate nearby independent points.
"""
import argparse
import copy
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
from .dense_transport import Field
from .dense_scene import largest_clique


def consensus(votes, radius=1.5):
    """One vote per source; at least two mutually agreeing, majority-supported paths."""
    if len(votes) < 2:
        return None
    if len({v[0] for v in votes}) != len(votes):
        raise ValueError('A source cannot multiply votes')
    xy = np.asarray([v[1] for v in votes])
    distance = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    vertices = tuple(range(len(votes)))
    adjacency = tuple(tuple(j for j in vertices if i != j and distance[i, j] <= radius)
                      for i in vertices)
    chosen = largest_clique(vertices, adjacency)
    if len(chosen) < 2 or len(chosen) <= len(votes) / 2:
        return None
    center = xy[list(chosen)].mean(0)
    return dict(xy=center.tolist(), sources=[int(votes[i][0]) for i in chosen],
                spread=float(np.max(np.linalg.norm(xy[list(chosen)] - center, axis=1))),
                rejected_paths=len(votes)-len(chosen))


def complete(scene, folder):
    start = time.perf_counter()
    meta = json.loads((folder / 'dense.json').read_text())
    if meta['records'] != scene['records']:
        raise ValueError('Source captures differ')
    fields = {}
    for pair in meta['pairs']:
        for a, b in ((pair['a'], pair['b']), (pair['b'], pair['a'])):
            with np.load(folder / f'{a:03d}-{b:03d}.npz') as row:
                good = row['valid'] & (row['third_view_support'] > 0)
                fields[a, b] = Field(row['grid'].reshape(-1, 2)[good],
                                     row['target'][good], row['affine'][good])
    # Only original measured observations propose extensions. Newly completed
    # observations cannot amplify their own support during this pass.
    by_camera = defaultdict(list)
    original = []
    for k, track in enumerate(scene['tracks']):
        original.append({o['capture'] for o in track['observations']})
        for observation in track['observations']:
            by_camera[observation['capture']].append((k, observation['xy']))
    votes = defaultdict(list)
    attempted = 0
    for (a, b), field in sorted(fields.items()):
        rows = [(k, xy) for k, xy in by_camera[a] if b not in original[k]]
        if not rows:
            continue
        p = np.asarray([xy for k, xy in rows])
        q, j, valid = field.at(p)
        back, jb, reverse_valid = fields[b, a].at(q)
        valid &= reverse_valid & (np.linalg.norm(back-p, axis=1) <= 1.5)
        valid &= np.linalg.norm(jb @ j - np.eye(2), axis=(1, 2)) <= .6
        attempted += len(rows)
        for i in np.flatnonzero(valid):
            votes[rows[i][0], b].append((a, q[i]))
    result = copy.deepcopy(scene)
    counts = defaultdict(int)
    candidates = 0
    for (k, target), proposal in sorted(votes.items()):
        if len(proposal) < 2:
            continue
        candidates += 1
        accepted = consensus(proposal)
        if accepted is None:
            continue
        result['tracks'][k]['observations'].append(dict(
            capture=target, xy=accepted['xy'], centroid=accepted['xy'],
            site=-1, completion=accepted))
        counts[target] += 1
    result['kind'] = 'geometry_observations_completed_from_regional_transport'
    result['completion'] = dict(added_observations=sum(counts.values()),
        additions_by_capture=dict(sorted(counts.items())), attempted_paths=attempted,
        multi_path_candidates=candidates, seconds=time.perf_counter()-start,
        policy=dict(minimum_independent_sources=2, maximum_spread_diameter=1.5,
                    maximum_roundtrip=1.5, maximum_jacobian_loop=.6,
                    source_observations='original only; no recursive self-support',
                    competing_paths='strict majority and pairwise agreement'),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        transport_sha256=hashlib.sha256((folder/'dense.json').read_bytes()).hexdigest())
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--fields', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    scene = json.loads(args.scene.read_text())
    result = complete(scene, args.fields)
    result['completion']['scene_sha256'] = hashlib.sha256(args.scene.read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, separators=(',', ':')))
    print(json.dumps(result['completion']), flush=True)


if __name__ == '__main__':
    main()
