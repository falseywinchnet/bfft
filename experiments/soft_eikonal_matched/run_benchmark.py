#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from metrics import evaluate, jacobian_variability, soft_diagnostics, tail_metrics
from models import make_mlp_pair, parameter_count
from tasks import TASK_BUILDERS


torch.set_num_threads(8)


def train(model, task, seed, steps, batch, lr, evaluate_every):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(90000 + seed); history = []; best = None; started = time.perf_counter()
    for step in range(1, steps + 1):
        index = torch.randint(len(task.x_train), (batch,), generator=generator)
        optimizer.zero_grad(set_to_none=True); output = model(task.x_train[index])
        if task.kind == "classification": loss = F.cross_entropy(output, task.y_train[index])
        else: loss = F.mse_loss(output, task.y_train[index])
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5); optimizer.step()
        if step == 1 or step % evaluate_every == 0 or step == steps:
            metrics = evaluate(model, task, task.x_val, task.y_val); row = {"step": step, "loss": float(loss), **metrics}; history.append(row)
            if best is None or metrics["score"] > best[0]:
                best = (metrics["score"], {key:value.detach().clone() for key,value in model.state_dict().items()}, step)
    model.load_state_dict(best[1]); return history, time.perf_counter() - started, best[2]


def threshold_step(history, threshold):
    for row in history:
        if row["score"] >= threshold: return row["step"]
    return None


def curve_auc(history, steps):
    return float(np.trapz([row["score"] for row in history], [row["step"] for row in history]) / steps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/tmp/soft_eikonal_matched"))
    parser.add_argument("--tasks", default=",".join(TASK_BUILDERS))
    parser.add_argument("--widths", default="16,36")
    parser.add_argument("--seeds", type=int, default=3); parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch", type=int, default=256); parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--eval-every", type=int, default=25); parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    partial = args.out / "runs.partial.json"; runs = json.loads(partial.read_text())["runs"] if args.resume and partial.exists() else []
    done = {(row["task"],row["width"],row["seed"],row["model"]) for row in runs}; started = time.perf_counter()
    for task_name in args.tasks.split(","):
        for seed in range(args.seeds):
            task = TASK_BUILDERS[task_name](seed)
            for width in map(int, args.widths.split(",")):
                torch.manual_seed(10000 + seed); mlp, soft = make_mlp_pair(task.input_dim, task.output_dim, width)
                for name, model in (("ordinary_mlp",mlp),("soft_eikonal",soft)):
                    if (task_name,width,seed,name) in done: continue
                    history, seconds, best_step = train(model, task, seed, args.steps, args.batch, args.lr, args.eval_every)
                    row = {"task":task_name,"kind":task.kind,"input_dim":task.input_dim,"output_dim":task.output_dim,
                           "width":width,"seed":seed,"model":name,"parameters":parameter_count(model),
                           "effective_affine_dof":None,
                           "seconds":seconds,"best_step":best_step,"learning_auc":curve_auc(history,args.steps),
                           "steps_to_80":threshold_step(history,.8),"steps_to_90":threshold_step(history,.9),
                           **evaluate(model,task),**tail_metrics(model,task),"history":history}
                    variability, rank = jacobian_variability(model, task.x_val)
                    row.update({"jacobian_variability":variability,"jacobian_change_rank":rank})
                    if name == "soft_eikonal": row.update(soft_diagnostics(model,task))
                    runs.append(row); partial.write_text(json.dumps({"runs":runs},indent=2)); print(json.dumps({k:v for k,v in row.items() if k not in {"history","tail_bins"}}),flush=True)
    summary=[]
    for key in sorted({(r["task"],r["width"],r["model"]) for r in runs}):
        selected=[r for r in runs if (r["task"],r["width"],r["model"])==key]
        def avg(name):
            values=[r.get(name) for r in selected if r.get(name) is not None]
            return float(np.mean(values)) if values else None
        summary.append({"task":key[0],"width":key[1],"model":key[2],"parameters":selected[0]["parameters"],
                        "validation_score":avg("matched_score") if key[2]=="soft_eikonal" else float(np.mean([
                            max(point["score"] for point in run["history"]) for run in selected])),
                        "effective_affine_dof":selected[0].get("effective_affine_dof"),
                        "score":avg("score"),"learning_auc":avg("learning_auc"),"seconds":avg("seconds"),
                        "steps_to_80":avg("steps_to_80"),"steps_to_90":avg("steps_to_90"),"accuracy":avg("accuracy"),
                        "normalized_mse":avg("normalized_mse"),"tail_score":avg("tail_score"),"tail_survival":avg("tail_survival"),
                        "jacobian_variability":avg("jacobian_variability"),"base_only_drop":avg("base_only_drop"),
                        "uniform_drop":avg("uniform_drop"),"mismatched_drop":avg("mismatched_drop"),
                        "up_allocation_variation":avg("up_allocation_variation"),"down_allocation_variation":avg("down_allocation_variation"),
                        "up_correction_ratio":avg("up_correction_ratio"),"down_correction_ratio":avg("down_correction_ratio")})
    payload={"configuration":{**vars(args),"out":str(args.out)},"runtime_seconds":time.perf_counter()-started,"runs":runs,"summary":summary}
    (args.out/"results.json").write_text(json.dumps(payload,indent=2))
    with (args.out/"summary.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=summary[0].keys());writer.writeheader();writer.writerows(summary)
    print(json.dumps({"complete":True,"runs":len(runs)},indent=2))


if __name__ == "__main__": main()
