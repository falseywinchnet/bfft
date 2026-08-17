#!/usr/bin/env python3
"""Hard-holdout double-spiral test for relational witness operators.

This is deliberately NumPy-only so the full CPU experiment runs on the M4 Mini
without a machine-learning runtime.  The candidate layers contain no learned
coordinate-wise filter bank.  Several fixed views of the current activation are
formed, relational evidence is pooled coordinate by coordinate, and one small
response rule shared by every coordinate generates the realized operator.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np


def spiral_points(count: int, lo: float, hi: float, seed: int, noise: float = 0.5):
    """Notebook geometry, restricted to a normalized phase interval [lo, hi]."""
    rng = np.random.default_rng(seed)
    phase = 8.0 * math.pi * np.sqrt(rng.uniform(lo * lo, hi * hi, (count, 1)))
    nx = (2.0 * rng.random((count, 1)) - 1.0) * noise
    ny = (2.0 * rng.random((count, 1)) - 1.0) * noise
    a = np.concatenate([0.5 * (np.sin(phase) * phase + nx),
                        0.5 * (np.cos(phase) * phase + ny)], axis=1)
    x = np.concatenate([a, -a], axis=0).astype(np.float64)
    y = np.concatenate([np.zeros(count, dtype=np.int64),
                        np.ones(count, dtype=np.int64)])
    u = np.concatenate([phase[:, 0], phase[:, 0]]) / (8.0 * math.pi)
    return x, y, u


class Parameter:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float64)
        self.grad = np.zeros_like(self.value)
        self.m = np.zeros_like(self.value)
        self.v = np.zeros_like(self.value)


def glorot(rng, fan_in, fan_out):
    bound = math.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-bound, bound, (fan_in, fan_out))


class Linear:
    def __init__(self, n_in, n_out, rng):
        self.w = Parameter(glorot(rng, n_in, n_out))
        self.b = Parameter(np.zeros(n_out))
        self.cache = None

    def parameters(self): return [self.w, self.b]

    def forward(self, x):
        self.cache = x
        return x @ self.w.value + self.b.value

    def backward(self, dy):
        self.w.grad += self.cache.T @ dy
        self.b.grad += dy.sum(axis=0)
        return dy @ self.w.value.T


class GELU:
    def __init__(self): self.cache = None

    def forward(self, x):
        self.cache = x
        c = math.sqrt(2.0 / math.pi)
        return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x**3)))

    def backward(self, dy):
        x = self.cache
        c = math.sqrt(2.0 / math.pi)
        z = c * (x + 0.044715 * x**3)
        t = np.tanh(z)
        derivative = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t*t) * c * (1.0 + 3.0 * 0.044715 * x*x)
        return dy * derivative


def fixed_views(n_in, n_out, views, seed):
    """Deterministic semi-orthogonal views; these are buffers, not parameters."""
    rng = np.random.default_rng(seed)
    result = []
    for _ in range(views):
        raw = rng.normal(size=(max(n_in, n_out), max(n_in, n_out)))
        q, _ = np.linalg.qr(raw)
        result.append(q[:n_in, :n_out] * math.sqrt(max(n_out / n_in, 1.0)))
    return np.stack(result)


class WitnessLinear:
    """A fixed multiview scaffold with either static or observation-made mixing."""
    def __init__(self, n_in, n_out, rng, mode="relational", views=3, response=12, seed=0):
        self.mode = mode
        self.views = views
        self.p = fixed_views(n_in, n_out, views, 7000 + seed + 31*n_in + n_out)
        self.cache = None
        if mode == "static":
            self.w1 = self.b1 = self.w2 = self.b2 = None
        else:
            self.w1 = Parameter(glorot(rng, 3, response))
            self.b1 = Parameter(np.zeros(response))
            self.w2 = Parameter(glorot(rng, response, views))
            self.b2 = Parameter(np.zeros(views))

    def parameters(self):
        if self.mode == "static": return []
        return [self.w1, self.b1, self.w2, self.b2]

    def forward(self, x):
        q = np.einsum("bi,vio->bov", x, self.p)
        if self.mode == "static":
            alpha = np.full_like(q, 1.0 / self.views)
            self.cache = (x, q, alpha, None, None, None)
            return (alpha * q).sum(axis=2)
        z = np.tanh(q)
        if self.mode == "relational":
            stats = np.stack([
                z.mean(axis=2),
                (z[:, :, 0]*z[:, :, 1] + z[:, :, 1]*z[:, :, 2] + z[:, :, 2]*z[:, :, 0]) / 3.0,
                z.prod(axis=2),
            ], axis=2)
        elif self.mode == "marginal":
            stats = np.stack([z.mean(axis=2), (z*z).mean(axis=2),
                              np.sqrt(z*z + 1e-4).mean(axis=2)], axis=2)
        else:
            raise ValueError(self.mode)
        hidden = np.tanh(stats @ self.w1.value + self.b1.value)
        logits = hidden @ self.w2.value + self.b2.value
        logits -= logits.max(axis=2, keepdims=True)
        alpha = np.exp(logits)
        alpha /= alpha.sum(axis=2, keepdims=True)
        self.cache = (x, q, alpha, z, stats, hidden)
        return (alpha * q).sum(axis=2)

    def backward(self, dy):
        x, q, alpha, z, stats, hidden = self.cache
        dq = dy[:, :, None] * alpha
        if self.mode != "static":
            dalpha = dy[:, :, None] * q
            dlogits = alpha * (dalpha - (dalpha * alpha).sum(axis=2, keepdims=True))
            flat_dlogits = dlogits.reshape(-1, self.views)
            flat_hidden = hidden.reshape(-1, hidden.shape[-1])
            self.w2.grad += flat_hidden.T @ flat_dlogits
            self.b2.grad += flat_dlogits.sum(axis=0)
            dhidden = dlogits @ self.w2.value.T
            da = dhidden * (1.0 - hidden*hidden)
            flat_da = da.reshape(-1, da.shape[-1])
            flat_stats = stats.reshape(-1, 3)
            self.w1.grad += flat_stats.T @ flat_da
            self.b1.grad += flat_da.sum(axis=0)
            ds = da @ self.w1.value.T
            dz = np.zeros_like(z)
            if self.mode == "relational":
                dz += ds[:, :, 0, None] / 3.0
                dz[:, :, 0] += ds[:, :, 1] * (z[:, :, 1] + z[:, :, 2]) / 3.0
                dz[:, :, 1] += ds[:, :, 1] * (z[:, :, 0] + z[:, :, 2]) / 3.0
                dz[:, :, 2] += ds[:, :, 1] * (z[:, :, 0] + z[:, :, 1]) / 3.0
                dz[:, :, 0] += ds[:, :, 2] * z[:, :, 1] * z[:, :, 2]
                dz[:, :, 1] += ds[:, :, 2] * z[:, :, 0] * z[:, :, 2]
                dz[:, :, 2] += ds[:, :, 2] * z[:, :, 0] * z[:, :, 1]
            else:
                dz += ds[:, :, 0, None] / self.views
                dz += ds[:, :, 1, None] * (2.0*z / self.views)
                dz += ds[:, :, 2, None] * (z / np.sqrt(z*z + 1e-4) / self.views)
            dq += dz * (1.0 - z*z)
        return np.einsum("bov,vio->bi", dq, self.p)


class Model:
    def __init__(self, kind, width, seed):
        rng = np.random.default_rng(100 + seed)
        self.kind = kind
        self.layers = [Linear(2, width, rng), GELU()]
        if kind == "reference_linear":
            self.layers += [Linear(width, 2*width, rng), GELU(), Linear(2*width, width, rng)]
        else:
            mode = kind.removeprefix("witness_")
            self.layers += [WitnessLinear(width, 2*width, rng, mode, seed=seed), GELU(),
                            WitnessLinear(2*width, width, rng, mode, seed=seed+1)]
        self.layers += [Linear(width, 2, rng)]

    def parameters(self):
        return [p for layer in self.layers for p in getattr(layer, "parameters", lambda: [])()]

    def forward(self, x):
        for layer in self.layers: x = layer.forward(x)
        return x

    def backward(self, dy):
        for layer in reversed(self.layers): dy = layer.backward(dy)


def cross_entropy(logits, y):
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted); probs /= probs.sum(axis=1, keepdims=True)
    loss = -np.log(probs[np.arange(len(y)), y] + 1e-12).mean()
    probs[np.arange(len(y)), y] -= 1.0
    return loss, probs / len(y)


def accuracy(model, x, y, batch=2048):
    predictions = []
    for start in range(0, len(x), batch):
        predictions.append(model.forward(x[start:start+batch]).argmax(axis=1))
    return float((np.concatenate(predictions) == y).mean())


def adam_step(parameters, step, lr, weight_decay=1e-4):
    b1, b2 = 0.9, 0.999
    for p in parameters:
        g = p.grad + weight_decay * p.value
        p.m = b1*p.m + (1-b1)*g
        p.v = b2*p.v + (1-b2)*(g*g)
        p.value -= lr * (p.m/(1-b1**step)) / (np.sqrt(p.v/(1-b2**step)) + 1e-8)
        p.grad.fill(0.0)


def train(kind, width, seed, train_fraction, steps, batch, lr):
    x, y, _ = spiral_points(1600, 0.015, train_fraction, 1000 + seed)
    perm = np.random.default_rng(2000 + seed).permutation(len(x))
    nval = len(x) // 5
    va, tr = perm[:nval], perm[nval:]
    model = Model(kind, width, seed)
    params = model.parameters()
    rng = np.random.default_rng(3000 + seed)
    best = None
    for step in range(1, steps + 1):
        ix = rng.choice(tr, size=batch, replace=True)
        logits = model.forward(x[ix])
        loss, grad = cross_entropy(logits, y[ix])
        model.backward(grad)
        adam_step(params, step, lr)
        if step % 100 == 0 or step == steps:
            val = accuracy(model, x[va], y[va])
            if best is None or val > best[0]:
                best = (val, [p.value.copy() for p in params], loss)
    for p, value in zip(params, best[1]): p.value[...] = value
    return model, best[0], best[2], sum(p.value.size for p in params)


def tail_profile(model, train_fraction, seed, bins=20, per_bin=250):
    rows = []
    for j in range(bins):
        lo = train_fraction + (1-train_fraction)*j/bins
        hi = train_fraction + (1-train_fraction)*(j+1)/bins
        x, y, _ = spiral_points(per_bin, lo, hi, 90000 + 1000*seed + j)
        rows.append(accuracy(model, x, y))
    threshold = 0.80
    survival = 0
    for value in rows:
        if value < threshold: break
        survival += 1
    return rows, survival, float(np.mean(rows[:5])), float(np.mean(rows))


def write_svg(summary, path):
    models = sorted({r["model"] for r in summary})
    fractions = sorted({r["train_fraction"] for r in summary}, reverse=True)
    colors = {"reference_linear":"#222222", "witness_static":"#9b59b6",
              "witness_marginal":"#e67e22", "witness_relational":"#168aad"}
    w, h = 920, 430
    pieces = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
              '<rect width="100%" height="100%" fill="#fbfaf7"/>',
              '<text x="28" y="30" font-family="sans-serif" font-size="20">Contiguous spiral extrapolation from the training frontier</text>']
    for panel, frac in enumerate(fractions):
        x0 = 50 + panel*450; y0 = 55; pw=380; ph=300
        pieces += [f'<text x="{x0}" y="{y0+15}" font-family="sans-serif" font-size="14">train fraction {frac:.2f}</text>',
                   f'<line x1="{x0}" y1="{y0+ph}" x2="{x0+pw}" y2="{y0+ph}" stroke="#777"/>',
                   f'<line x1="{x0}" y1="{y0+30}" x2="{x0}" y2="{y0+ph}" stroke="#777"/>']
        for tick in [0,.25,.5,.75,1]:
            yy=y0+ph-tick*(ph-30)
            pieces += [f'<line x1="{x0}" y1="{yy}" x2="{x0+pw}" y2="{yy}" stroke="#ddd"/>',
                       f'<text x="{x0-35}" y="{yy+4}" font-family="sans-serif" font-size="11">{tick:.2f}</text>']
        for model in models:
            records=[r for r in summary if r["train_fraction"]==frac and r["model"]==model]
            values=np.mean([r["tail_bins"] for r in records], axis=0)
            pts=" ".join(f'{x0+i*pw/(len(values)-1):.1f},{y0+ph-v*(ph-30):.1f}' for i,v in enumerate(values))
            pieces.append(f'<polyline points="{pts}" fill="none" stroke="{colors[model]}" stroke-width="2"/>')
    for i, model in enumerate(models):
        pieces += [f'<line x1="{55+i*210}" y1="395" x2="{85+i*210}" y2="395" stroke="{colors[model]}" stroke-width="3"/>',
                   f'<text x="{92+i*210}" y="400" font-family="sans-serif" font-size="12">{model}</text>']
    pieces.append('</svg>')
    path.write_text("\n".join(pieces))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("experiments/relational_witness_spiral/results"))
    ap.add_argument("--steps", type=int, default=1800)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--width", type=int, default=24)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    records = []
    kinds = ["reference_linear", "witness_static", "witness_marginal", "witness_relational"]
    for fraction in [0.50, 0.30]:
        for seed in range(args.seeds):
            for kind in kinds:
                model, val, loss, parameters = train(kind, args.width, seed, fraction,
                                                     args.steps, args.batch, args.lr)
                bins, survival, frontier5, tail = tail_profile(model, fraction, seed)
                record = {"model":kind, "seed":seed, "train_fraction":fraction,
                          "parameters":parameters, "val_accuracy":val, "loss":loss,
                          "survival_bins_at_80pct":survival, "frontier5_accuracy":frontier5,
                          "tail_accuracy":tail, "tail_bins":bins}
                records.append(record)
                print(json.dumps(record), flush=True)
    with (args.out / "runs.json").open("w") as f:
        json.dump({"runtime_seconds":time.time()-start, "config":vars(args) | {"out":str(args.out)},
                   "runs":records}, f, indent=2)
    fields=["model","train_fraction","parameters","val_mean","val_std","frontier5_mean",
            "frontier5_std","tail_mean","survival_mean","survival_max"]
    summary=[]
    for fraction in [0.50,0.30]:
        for kind in kinds:
            rs=[r for r in records if r["model"]==kind and r["train_fraction"]==fraction]
            row={"model":kind,"train_fraction":fraction,"parameters":rs[0]["parameters"],
                 "val_mean":np.mean([r["val_accuracy"] for r in rs]),
                 "val_std":np.std([r["val_accuracy"] for r in rs]),
                 "frontier5_mean":np.mean([r["frontier5_accuracy"] for r in rs]),
                 "frontier5_std":np.std([r["frontier5_accuracy"] for r in rs]),
                 "tail_mean":np.mean([r["tail_accuracy"] for r in rs]),
                 "survival_mean":np.mean([r["survival_bins_at_80pct"] for r in rs]),
                 "survival_max":int(np.max([r["survival_bins_at_80pct"] for r in rs]))}
            summary.append(row)
    with (args.out / "summary.csv").open("w", newline="") as f:
        writer=csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(summary)
    write_svg(records, args.out / "tail_survival.svg")
    print(json.dumps({"runtime_seconds":time.time()-start,"summary":summary}, indent=2))


if __name__ == "__main__": main()
