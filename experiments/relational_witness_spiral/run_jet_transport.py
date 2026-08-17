#!/usr/bin/env python3
"""Learn local representation and operator jets from observed spiral secants.

The model receives no radius, phase, ordering, Fourier features, or held-out
samples.  Its nonlinear path is an unbounded input-conditioned linear map.
Optional auxiliary losses teach two directional derivatives: input displacement
to representation displacement, and representation displacement to operator
coefficient displacement.  The transport mode also asks those local changes to
compose across observed triples.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F

from run_hypersphere_atlas import accuracy, draw_scatter, spiral_points, tail_profile

torch.set_default_dtype(torch.float64)
torch.set_num_threads(8)


class JetLinear(nn.Module):
    """A dense map plus a learned, unbounded field of low-rank linear maps."""

    def __init__(self, n_in: int, n_out: int, latent: int = 6,
                 charts: int = 6, branch_rank: int = 2):
        super().__init__()
        self.n_in, self.latent, self.charts = n_in, latent, charts
        self.base = nn.Linear(n_in, n_out)
        self.representation = nn.Sequential(
            nn.Linear(n_in, latent), nn.SiLU(), nn.Linear(latent, latent)
        )
        self.coefficients = nn.Sequential(
            nn.Linear(latent, 2 * latent), nn.SiLU(), nn.Linear(2 * latent, charts)
        )
        # C_x(x)[dx] predicts dz; C_q(z)[dz] predicts the operator change dq.
        self.input_connection = nn.Sequential(
            nn.Linear(n_in, 2 * latent), nn.SiLU(),
            nn.Linear(2 * latent, latent * n_in),
        )
        self.operator_connection = nn.Sequential(
            nn.Linear(latent, 2 * latent), nn.SiLU(),
            nn.Linear(2 * latent, charts * latent),
        )
        self.left = nn.Parameter(torch.empty(charts, n_out, branch_rank))
        self.right = nn.Parameter(torch.empty(charts, branch_rank, n_in))
        self.representation_origin = nn.Parameter(torch.zeros(latent))
        self.operator_origin = nn.Parameter(torch.zeros(charts))
        self.jet_scale = nn.Parameter(torch.tensor(-2.0))
        nn.init.normal_(self.left, std=0.10)
        nn.init.normal_(self.right, std=0.10)

    def _integrate_connections(self, x: torch.Tensor):
        # Three-point Gauss-Legendre quadrature on the generic straight path
        # from the learned origin. No task coordinate or continuation direction
        # enters this construction.
        nodes = x.new_tensor([.1127016653792583, .5, .8872983346207417])
        weights = x.new_tensor([5/18, 8/18, 5/18])
        z_delta = torch.zeros(len(x), self.latent, dtype=x.dtype, device=x.device)
        for node, weight in zip(nodes, weights):
            cx = self.input_connection(node * x).view(-1, self.latent, self.n_in)
            z_delta = z_delta + weight * torch.einsum("bli,bi->bl", cx, x)
        z = self.representation_origin + z_delta
        q_delta = torch.zeros(len(x), self.charts, dtype=x.dtype, device=x.device)
        for node, weight in zip(nodes, weights):
            cq = self.operator_connection(node * z).view(-1, self.charts, self.latent)
            q_delta = q_delta + weight * torch.einsum("bcl,bl->bc", cq, z)
        return z, self.operator_origin + q_delta

    def forward(self, x: torch.Tensor, diagnostics: bool = False, integrated: bool = False):
        if integrated:
            z, q = self._integrate_connections(x)
        else:
            z = self.representation(x)
            q = self.coefficients(z)  # deliberately unsaturated
        vx = torch.einsum("cri,bi->bcr", self.right, x)
        correction = torch.einsum("bc,cor,bcr->bo", q, self.left, vx)
        correction = correction * F.softplus(self.jet_scale) / math.sqrt(self.charts)
        y = self.base(x) + correction
        if not diagnostics:
            return y
        cx = self.input_connection(x).view(-1, self.latent, self.n_in)
        cq = self.operator_connection(z).view(-1, self.charts, self.latent)
        return y, {"x": x, "z": z, "q": q, "cx": cx, "cq": cq}


class JetNetwork(nn.Module):
    def __init__(self, width: int = 24, integrated: bool = False):
        super().__init__()
        self.integrated = integrated
        self.embed = nn.Linear(2, width)
        self.up = JetLinear(width, 2 * width)
        self.down = JetLinear(2 * width, width)
        self.out = nn.Linear(width, 2)

    def forward(self, x: torch.Tensor, diagnostics: bool = False):
        h = self.embed(x)
        if diagnostics:
            h, d1 = self.up(h, True, self.integrated)
            h = F.gelu(h)
            h, d2 = self.down(h, True, self.integrated)
            return self.out(h), (d1, d2)
        h = F.gelu(self.up(h, integrated=self.integrated))
        return self.out(self.down(h, integrated=self.integrated))


class DenseNetwork(nn.Module):
    def __init__(self, width: int = 24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, width), nn.Linear(width, 2 * width), nn.GELU(),
            nn.Linear(2 * width, width), nn.Linear(width, 2),
        )

    def forward(self, x: torch.Tensor, diagnostics: bool = False):
        return (self.net(x), ()) if diagnostics else self.net(x)


def observed_neighbor_triples(x: torch.Tensor, train_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Make local triples using only distances among observed inputs."""
    observed = x[train_indices]
    distances = torch.cdist(observed, observed)
    distances.fill_diagonal_(float("inf"))
    nearest = distances.topk(4, largest=False).indices
    j_local = nearest[:, 0]
    k_local = torch.empty_like(j_local)
    for i in range(len(observed)):
        candidates = nearest[j_local[i]]
        valid = candidates[candidates != i]
        k_local[i] = valid[0] if len(valid) else nearest[i, 1]
    return train_indices[j_local], train_indices[k_local]


def _relative_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    scale = target.detach().square().mean().sqrt().clamp_min(0.05)
    return F.smooth_l1_loss(pred / scale, target / scale)


def jet_relation_loss(di: dict, dj: dict, dk: dict, composition: bool) -> torch.Tensor:
    dx_ij, dx_jk = dj["x"] - di["x"], dk["x"] - dj["x"]
    dz_ij, dz_jk = dj["z"] - di["z"], dk["z"] - dj["z"]
    dq_ij, dq_jk = dj["q"] - di["q"], dk["q"] - dj["q"]

    pred_z_ij = torch.einsum("bli,bi->bl", .5 * (di["cx"] + dj["cx"]), dx_ij)
    pred_z_jk = torch.einsum("bli,bi->bl", .5 * (dj["cx"] + dk["cx"]), dx_jk)
    pred_q_ij = torch.einsum("bcl,bl->bc", .5 * (di["cq"] + dj["cq"]), dz_ij)
    pred_q_jk = torch.einsum("bcl,bl->bc", .5 * (dj["cq"] + dk["cq"]), dz_jk)
    loss = (_relative_loss(pred_z_ij, dz_ij) + _relative_loss(pred_z_jk, dz_jk)
            + _relative_loss(pred_q_ij, dq_ij) + _relative_loss(pred_q_jk, dq_jk))
    if composition:
        dx_ik, dz_ik, dq_ik = dk["x"] - di["x"], dk["z"] - di["z"], dk["q"] - di["q"]
        pred_z_ik = torch.einsum("bli,bi->bl", .5 * (di["cx"] + dk["cx"]), dx_ik)
        pred_q_ik = torch.einsum("bcl,bl->bc", .5 * (di["cq"] + dk["cq"]), dz_ik)
        loss = loss + _relative_loss(pred_z_ik, dz_ik) + _relative_loss(pred_q_ik, dq_ik)
        loss = loss + .5 * _relative_loss(pred_z_ik, pred_z_ij + pred_z_jk)
        loss = loss + .5 * _relative_loss(pred_q_ik, pred_q_ij + pred_q_jk)
    return loss


def train(kind: str, seed: int, fraction: float, steps: int, batch: int,
          lr: float, width: int, relation_weight: float):
    torch.manual_seed(100 + seed)
    x, y = spiral_points(1600, .015, fraction, 1000 + seed)
    perm = torch.randperm(len(x), generator=torch.Generator().manual_seed(2000 + seed))
    va, tr = perm[:len(x) // 5], perm[len(x) // 5:]
    integrated = kind in {"jet_transport", "jet_shuffled"}
    model = DenseNetwork(width) if kind == "dense" else JetNetwork(width, integrated)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    neighbor_j, neighbor_k = observed_neighbor_triples(x, tr)
    gen = torch.Generator().manual_seed(3000 + seed)
    best, history = None, []

    for step in range(1, steps + 1):
        local = torch.randint(len(tr), (batch,), generator=gen)
        ii, jj, kk = tr[local], neighbor_j[local], neighbor_k[local]
        if kind == "jet_shuffled":
            roll = torch.randperm(batch, generator=gen)
            jj, kk = jj[roll], kk[roll]
        opt.zero_grad()
        if kind == "dense":
            logits = model(x[ii]); relation = torch.zeros(())
        else:
            logits_all, diagnostics = model(torch.cat([x[ii], x[jj], x[kk]]), True)
            logits = logits_all[:batch]
            relation = torch.zeros(())
            if kind != "jet_direct":
                for d in diagnostics:
                    di = {name: value[:batch] for name, value in d.items()}
                    dj = {name: value[batch:2 * batch] for name, value in d.items()}
                    dk = {name: value[2 * batch:] for name, value in d.items()}
                    relation = relation + jet_relation_loss(di, dj, dk, kind != "jet_secant")
        classification = F.cross_entropy(logits, y[ii])
        loss = classification + relation_weight * relation
        loss.backward(); opt.step()
        if step % 100 == 0 or step == steps:
            val = accuracy(model, x[va], y[va])
            _, _, front, _ = tail_profile(model, fraction, seed, bins=5, per_bin=100)
            history.append({"step": step, "loss": float(loss), "classification": float(classification),
                            "relation": float(relation), "val": val, "frontier5": front})
            score = val - .01 * float(classification)
            if best is None or score > best[0]:
                best = (score, {k: v.detach().clone() for k, v in model.state_dict().items()},
                        float(classification), float(relation))
    model.load_state_dict(best[1])
    return model, accuracy(model, x[va], y[va]), best[2], best[3], sum(p.numel() for p in model.parameters()), history


COLORS = {"dense": (40, 40, 40), "jet_direct": (127, 86, 163),
          "jet_secant": (226, 133, 38), "jet_transport": (14, 132, 164),
          "jet_shuffled": (194, 55, 47)}


def survival_plot(path: Path, curves: dict[str, np.ndarray]):
    width, height = 1000, 620; left, top, right, bottom = 85, 45, 25, 75
    image = Image.new("RGB", (width, height), (249, 248, 244)); draw = ImageDraw.Draw(image)
    draw.rectangle((left, top, width-right, height-bottom), outline=(70, 70, 70))
    for value in [0, .25, .5, .75, 1]:
        yy = height-bottom-value*(height-bottom-top)
        draw.line((left, yy, width-right, yy), fill=(220, 220, 215))
        draw.text((45, yy-7), f"{value:.2f}", fill=(50, 50, 50))
    threshold_y = height-bottom-.8*(height-bottom-top)
    draw.line((left, threshold_y, width-right, threshold_y), fill=(100, 100, 100), width=2)
    draw.text((width-right-105, threshold_y-18), "80% survival", fill=(80, 80, 80))
    draw.text((left, 12), "Accuracy by distance beyond the observed frontier", fill=(20, 20, 20))
    draw.text((430, height-35), "held-out bin", fill=(20, 20, 20))
    for index in [0, 4, 9, 14, 19]:
        xx = left+index/19*(width-right-left)
        draw.line((xx, height-bottom, xx, height-bottom+5), fill=(50, 50, 50))
        draw.text((xx-5, height-bottom+9), str(index+1), fill=(50, 50, 50))
    for index, (name, values) in enumerate(curves.items()):
        points = [(left+i/(len(values)-1)*(width-right-left), height-bottom-v*(height-bottom-top))
                  for i, v in enumerate(values)]
        draw.line(points, fill=COLORS[name], width=4)
        legend_x, legend_y = left+10+(index%3)*245, top+10+(index//3)*23
        draw.line((legend_x, legend_y+6, legend_x+28, legend_y+6), fill=COLORS[name], width=4)
        draw.text((legend_x+35, legend_y), name, fill=COLORS[name])
    image.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("experiments/relational_witness_spiral/results_jet_transport"))
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--batch", type=int, default=192)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--relation-weight", type=float, default=.08)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    kinds = ["dense", "jet_direct", "jet_secant", "jet_transport", "jet_shuffled"]
    runs, representatives, histories = [], {}, {}; started = time.time()
    for fraction in [.5, .3]:
        for seed in range(args.seeds):
            for kind in kinds:
                model, val, cls, rel, params, history = train(kind, seed, fraction, args.steps,
                                                               args.batch, args.lr, args.width,
                                                               args.relation_weight)
                bins, survival, frontier, tail = tail_profile(model, fraction, seed)
                row = {"model": kind, "seed": seed, "train_fraction": fraction,
                       "parameters": params, "val_accuracy": val, "classification_loss": cls,
                       "relation_loss": rel, "survival_bins_at_80pct": survival,
                       "frontier5_accuracy": frontier, "tail_accuracy": tail, "tail_bins": bins}
                runs.append(row); print(json.dumps(row), flush=True)
                # Long remote runs checkpoint outside the mirrored tree when
                # --out points there; a later repository sync cannot erase it.
                with (args.out / "runs.partial.json").open("w") as handle:
                    json.dump({"elapsed_seconds": time.time()-started, "runs": runs}, handle, indent=2)
                if fraction == .5 and seed == 0:
                    representatives[kind], histories[kind] = model, history
    summary = []
    for fraction in [.5, .3]:
        for kind in kinds:
            selected = [r for r in runs if r["model"] == kind and r["train_fraction"] == fraction]
            summary.append({"model": kind, "train_fraction": fraction,
                            "parameters": selected[0]["parameters"],
                            "val_mean": float(np.mean([r["val_accuracy"] for r in selected])),
                            "frontier5_mean": float(np.mean([r["frontier5_accuracy"] for r in selected])),
                            "frontier5_std": float(np.std([r["frontier5_accuracy"] for r in selected])),
                            "tail_mean": float(np.mean([r["tail_accuracy"] for r in selected])),
                            "survival_mean": float(np.mean([r["survival_bins_at_80pct"] for r in selected])),
                            "survival_max": int(np.max([r["survival_bins_at_80pct"] for r in selected]))})
    with (args.out / "runs.json").open("w") as handle:
        json.dump({"runtime_seconds": time.time()-started, "runs": runs}, handle, indent=2)
    with (args.out / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0].keys()); writer.writeheader(); writer.writerows(summary)
    train_x, train_y = spiral_points(600, .015, .5, 12)
    hold_x, hold_y = spiral_points(400, .5, 1, 13)
    for kind, model in representatives.items():
        draw_scatter(args.out / f"decision_{kind}.png", model, train_x, train_y, hold_x, hold_y,
                     f"{kind}: 50% spiral holdout")
    curves = {kind: np.mean([r["tail_bins"] for r in runs if r["model"] == kind
                             and r["train_fraction"] == .5], axis=0) for kind in kinds}
    survival_plot(args.out / "survival_50pct.png", curves)
    print(json.dumps({"runtime_seconds": time.time()-started, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
