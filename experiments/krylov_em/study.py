import argparse, json, time
import numpy as np
from .rl import Blur, RL, solve


def scene(n, seed):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:n, :n]/n
    img = 0.2+np.zeros((n, n))
    for _ in range(12):  # points, disks and bars: classic RL stress content
        cy, cx, r = rng.random(3); r = 0.02+0.08*r
        img += rng.random()*3*((yy-cy)**2+(xx-cx)**2 < r*r)
    for _ in range(20):
        img[rng.integers(n), rng.integers(n)] += 40*rng.random()
    img[n//3:n//3+2, n//5:4*n//5] += 2
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', type=int, default=128)
    ap.add_argument('--sigmas', default='1.5,3')
    ap.add_argument('--counts', default='100,10000')
    ap.add_argument('--ref-iters', type=int, default=3000)
    ap.add_argument('--seeds', type=int, default=2)
    ap.add_argument('--configs', default='4:16,4:64,8:64,8:256')
    ap.add_argument('--taus', default='0.02,0.1')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(); rows = []
    for seed in range(a.seeds):
        for sigma in map(float, a.sigmas.split(',')):
            for counts in map(float, a.counts.split(',')):
                rng = np.random.default_rng(100+seed)
                truth = scene(a.size, seed); truth *= counts/truth.mean()
                bl = Blur(truth.shape, sigma); bg = 0.05*counts
                y = rng.poisson(bl(truth.ravel())+bg).astype(float)
                m = RL(y, bl, bg)
                # Target = objective ordinary RL reaches after ref-iters sweeps.
                m.reset_counts(); z = m.initial.copy(); t = time.perf_counter()
                fs = []
                for k in range(a.ref_iters):
                    z = m.step(z)
                    if k % 8 == 7: fs.append(m.objective(z))
                t_ord = time.perf_counter()-t; target = fs[-1]
                case = dict(seed=seed, sigma=sigma, counts=counts, target=target,
                            ordinary_seconds=t_ord, ordinary_work=a.ref_iters, runs=[])
                runs = [('anderson', dict(depth=4)), ('anderson_unguarded', dict(depth=4, guard=False))]
                runs += [(f'krylov_d{d}_h{h}', dict(depth=int(d), horizon=int(h)))
                         for d, h in (c.split(':') for c in a.configs.split(','))]
                runs += [(f'certified_d8_t{t}', dict(depth=8, tau=float(t))) for t in a.taus.split(',')]
                for name, kw in runs:
                    meth = name.split('_')[0]
                    _, r = solve(m, meth, target, budget=a.ref_iters*2, **kw)
                    r['name'] = name; r['speedup'] = t_ord/r['seconds'] if r['status'] == 'target' else None
                    r.pop('trace'); case['runs'].append(r)
                    print(seed, sigma, counts, name, r['status'], r['calls']+r['actions'],
                          r['accepted'], r['rejected'], round(r['seconds'], 3), r['speedup'], flush=True)
                rows.append(case)
    json.dump(rows, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
