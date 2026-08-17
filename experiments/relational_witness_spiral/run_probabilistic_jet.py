#!/usr/bin/env python3
"""Posterior mixture over learned connection hypotheses on double spirals.

Each hypothesis is an integrated JetNetwork trained on a bootstrap of observed
relation triples. A disjoint set of observed triples supplies predictive
evidence for checkpointing and posterior weights. No held-out spiral sample is
used until final evaluation. The test-only oracle diagnoses whether failures
come from the hypothesis basis or from posterior selection.
"""

from __future__ import annotations

import argparse
import copy
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
from run_jet_transport import JetNetwork, jet_relation_loss, observed_neighbor_triples

torch.set_default_dtype(torch.float64)
torch.set_num_threads(8)


class JetPosterior(nn.Module):
    def __init__(self, hypotheses: list[JetNetwork], weights: torch.Tensor):
        super().__init__()
        self.hypotheses = nn.ModuleList(hypotheses)
        self.register_buffer("weights", weights / weights.sum())

    def forward(self, x: torch.Tensor):
        logits = torch.stack([model(x) for model in self.hypotheses], dim=0)
        return torch.einsum("h,hbc->bc", self.weights, logits)


def split_relation_anchors(count: int, seed: int, evidence_fraction: float = .2):
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(count, generator=generator)
    cut = max(1, int(count * evidence_fraction))
    return order[cut:], order[:cut]


def _slice(diagnostics: tuple[dict, ...], start: int, stop: int):
    return tuple({name: value[start:stop] for name, value in layer.items()} for layer in diagnostics)


def relation_loss_for_indices(model: JetNetwork, x: torch.Tensor, anchors: torch.Tensor,
                              neighbor_j: torch.Tensor, neighbor_k: torch.Tensor,
                              batch: int = 256, gradients: bool = True):
    losses = []
    context = torch.enable_grad() if gradients else torch.no_grad()
    with context:
        for local in anchors.split(batch):
            ii, jj, kk = local, neighbor_j[local], neighbor_k[local]
            _, diagnostics = model(torch.cat([x[ii], x[jj], x[kk]]), True)
            size = len(local); loss = torch.zeros((), dtype=x.dtype)
            for layer in diagnostics:
                di = {n: v[:size] for n, v in layer.items()}
                dj = {n: v[size:2*size] for n, v in layer.items()}
                dk = {n: v[2*size:] for n, v in layer.items()}
                loss = loss + jet_relation_loss(di, dj, dk, True)
            losses.append((loss, size))
    return sum(loss * size for loss, size in losses) / sum(size for _, size in losses)


def train_hypothesis(x, y, tr, val_x, val_y, neighbor_j, neighbor_k, fit_anchors,
                     evidence_anchors, seed, steps, batch, lr, width,
                     relation_weight, checkpoint_interval):
    torch.manual_seed(seed)
    model = JetNetwork(width, integrated=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed + 10000)
    # A fixed bootstrap is the hypothesis's private relational experience.
    bootstrap = fit_anchors[torch.randint(len(fit_anchors), (len(fit_anchors),), generator=generator)]
    best, history = None, []
    for step in range(1, steps + 1):
        classification_indices = tr[torch.randint(len(tr), (batch,), generator=generator)]
        relation_indices = bootstrap[torch.randint(len(bootstrap), (batch,), generator=generator)]
        optimizer.zero_grad()
        classification = F.cross_entropy(model(x[classification_indices]), y[classification_indices])
        relation = relation_loss_for_indices(model, x, relation_indices, neighbor_j, neighbor_k,
                                             batch=batch, gradients=True)
        loss = classification + relation_weight * relation
        loss.backward(); optimizer.step()
        if step % checkpoint_interval == 0 or step == steps:
            val = accuracy(model, val_x, val_y)
            evidence = float(relation_loss_for_indices(model, x, evidence_anchors,
                                                       neighbor_j, neighbor_k, gradients=False))
            history.append({"step": step, "validation": val, "evidence_loss": evidence,
                            "classification_loss": float(classification), "fit_relation_loss": float(relation)})
            if val >= .98 and (best is None or evidence < best[0]):
                best = (evidence, copy.deepcopy(model.state_dict()), step, val)
    if best is None:
        last = history[-1]
        best = (last["evidence_loss"], copy.deepcopy(model.state_dict()), last["step"], last["validation"])
    model.load_state_dict(best[1])
    return model, {"evidence_loss": best[0], "checkpoint_step": best[2],
                   "validation": best[3], "history": history}


def posterior_weights(evidence_losses: list[float], temperature: float):
    losses = torch.tensor(evidence_losses)
    return torch.softmax(-(losses - losses.min()) / temperature, dim=0)


def evaluate_candidate(model, fraction, seed):
    bins, survival, frontier, tail = tail_profile(model, fraction, seed)
    return {"survival_bins_at_80pct": survival, "frontier5_accuracy": frontier,
            "first_bin_accuracy": bins[0], "tail_accuracy": tail, "tail_bins": bins}


COLORS = {"uniform": (112, 78, 160), "posterior": (5, 132, 159),
          "permuted_posterior": (198, 55, 46), "best_hypothesis_oracle": (35, 35, 35)}


def plot_survival(path: Path, curves: dict[str, np.ndarray]):
    width, height = 1000, 620; left, top, right, bottom = 85, 45, 25, 75
    image = Image.new("RGB", (width, height), (249, 248, 244)); draw = ImageDraw.Draw(image)
    draw.rectangle((left, top, width-right, height-bottom), outline=(65, 65, 65))
    for value in [0, .25, .5, .75, 1]:
        yy = height-bottom-value*(height-bottom-top)
        draw.line((left, yy, width-right, yy), fill=(220, 220, 215))
        draw.text((45, yy-7), f"{value:.2f}", fill=(50, 50, 50))
    yy = height-bottom-.8*(height-bottom-top)
    draw.line((left, yy, width-right, yy), fill=(90, 90, 90), width=2)
    draw.text((width-right-105, yy-18), "80% survival", fill=(70, 70, 70))
    draw.text((left, 12), "Posterior connection hypotheses beyond the observed frontier", fill=(20, 20, 20))
    for index in [0, 4, 9, 14, 19]:
        xx = left+index/19*(width-right-left)
        draw.text((xx-5, height-bottom+10), str(index+1), fill=(50, 50, 50))
    for index, (name, values) in enumerate(curves.items()):
        points = [(left+i/(len(values)-1)*(width-right-left), height-bottom-v*(height-bottom-top))
                  for i, v in enumerate(values)]
        draw.line(points, fill=COLORS[name], width=4)
        lx, ly = left+10+(index%2)*330, top+10+(index//2)*23
        draw.line((lx, ly+6, lx+28, ly+6), fill=COLORS[name], width=4)
        draw.text((lx+35, ly), name, fill=COLORS[name])
    draw.text((430, height-35), "held-out bin", fill=(20, 20, 20))
    image.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("experiments/relational_witness_spiral/results_probabilistic_jet"))
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--hypotheses", type=int, default=5)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--batch", type=int, default=160)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--relation-weight", type=float, default=.08)
    parser.add_argument("--posterior-temperature", type=float, default=.02)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    runs, hypothesis_records, representatives = [], [], {}; started = time.time()
    for fraction in [.5, .3]:
        for seed in range(args.seeds):
            torch.manual_seed(100 + seed)
            x, y = spiral_points(1600, .015, fraction, 1000 + seed)
            order = torch.randperm(len(x), generator=torch.Generator().manual_seed(2000 + seed))
            va, tr = order[:len(x)//5], order[len(x)//5:]
            local_x = x[tr]
            j_global, k_global = observed_neighbor_triples(x, tr)
            # Convert global neighbors to local observed-array indices.
            inverse = torch.full((len(x),), -1, dtype=torch.long); inverse[tr] = torch.arange(len(tr))
            neighbor_j, neighbor_k = inverse[j_global], inverse[k_global]
            fit_anchors, evidence_anchors = split_relation_anchors(len(tr), 7000 + seed, .2)
            hypotheses, metadata = [], []
            for hypothesis in range(args.hypotheses):
                model, record = train_hypothesis(local_x, y[tr], torch.arange(len(tr)),
                    x[va], y[va], neighbor_j, neighbor_k, fit_anchors,
                    evidence_anchors, 50000 + 1000*seed + hypothesis, args.steps,
                    args.batch, args.lr, args.width, args.relation_weight,
                    args.checkpoint_interval)
                # Validation indices are global, so score explicitly here.
                record["validation"] = accuracy(model, x[va], y[va])
                hypotheses.append(model); metadata.append(record)
            losses = [record["evidence_loss"] for record in metadata]
            weights = posterior_weights(losses, args.posterior_temperature)
            candidate_weights = {"uniform": torch.ones(args.hypotheses)/args.hypotheses,
                                 "posterior": weights,
                                 "permuted_posterior": torch.roll(weights, 1)}
            candidate_results = {}
            for name, candidate_weight in candidate_weights.items():
                mixture = JetPosterior(hypotheses, candidate_weight)
                result = evaluate_candidate(mixture, fraction, seed)
                row = {"model": name, "seed": seed, "train_fraction": fraction,
                       "parameters": sum(p.numel() for p in mixture.parameters()),
                       "posterior_weights": candidate_weight.tolist(),
                       "evidence_losses": losses, **result}
                runs.append(row); candidate_results[name] = (mixture, result)
            individual = [evaluate_candidate(model, fraction, seed) for model in hypotheses]
            oracle_index = max(range(len(individual)), key=lambda h: (individual[h]["survival_bins_at_80pct"],
                                                                       individual[h]["first_bin_accuracy"],
                                                                       individual[h]["frontier5_accuracy"]))
            oracle = {"model": "best_hypothesis_oracle", "seed": seed, "train_fraction": fraction,
                      "hypothesis": oracle_index, "selection_uses_test_data": True,
                      "evidence_losses": losses, **individual[oracle_index]}
            runs.append(oracle)
            hypothesis_records.append({"seed": seed, "train_fraction": fraction,
                                       "fit_anchor_count": len(fit_anchors),
                                       "evidence_anchor_count": len(evidence_anchors),
                                       "hypotheses": metadata, "individual_test_results": individual,
                                       "posterior_weights": weights.tolist()})
            print(json.dumps({"seed": seed, "train_fraction": fraction, "evidence_losses": losses,
                              "posterior_weights": weights.tolist(),
                              "candidates": {k: v[1] for k, v in candidate_results.items()},
                              "oracle": oracle}), flush=True)
            with (args.out / "runs.partial.json").open("w") as handle:
                json.dump({"runs": runs, "hypothesis_records": hypothesis_records}, handle, indent=2)
            if fraction == .5 and seed == 0:
                representatives = {name: value[0] for name, value in candidate_results.items()}
                representatives["best_hypothesis_oracle"] = JetPosterior(hypotheses,
                    F.one_hot(torch.tensor(oracle_index), args.hypotheses).to(torch.float64))
    summary = []
    for fraction in [.5, .3]:
        for name in ["uniform", "posterior", "permuted_posterior", "best_hypothesis_oracle"]:
            selected = [row for row in runs if row["model"] == name and row["train_fraction"] == fraction]
            summary.append({"model": name, "train_fraction": fraction,
                            "first_bin_mean": float(np.mean([r["first_bin_accuracy"] for r in selected])),
                            "frontier5_mean": float(np.mean([r["frontier5_accuracy"] for r in selected])),
                            "frontier5_std": float(np.std([r["frontier5_accuracy"] for r in selected])),
                            "tail_mean": float(np.mean([r["tail_accuracy"] for r in selected])),
                            "survival_mean": float(np.mean([r["survival_bins_at_80pct"] for r in selected])),
                            "survival_max": int(np.max([r["survival_bins_at_80pct"] for r in selected]))})
    with (args.out / "runs.json").open("w") as handle:
        json.dump({"runtime_seconds": time.time()-started, "configuration": vars(args),
                   "runs": runs, "hypothesis_records": hypothesis_records}, handle, indent=2, default=str)
    with (args.out / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0].keys()); writer.writeheader(); writer.writerows(summary)
    train_x, train_y = spiral_points(600, .015, .5, 12); hold_x, hold_y = spiral_points(400, .5, 1, 13)
    for name, model in representatives.items():
        draw_scatter(args.out / f"decision_{name}.png", model, train_x, train_y, hold_x, hold_y,
                     f"{name}: posterior jet, 50% holdout")
    curves = {name: np.mean([row["tail_bins"] for row in runs if row["model"] == name
                             and row["train_fraction"] == .5], axis=0)
              for name in ["uniform", "posterior", "permuted_posterior", "best_hypothesis_oracle"]}
    plot_survival(args.out / "survival_50pct.png", curves)
    print(json.dumps({"runtime_seconds": time.time()-started, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
