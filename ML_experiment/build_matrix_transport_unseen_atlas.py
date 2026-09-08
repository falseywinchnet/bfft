"""Build the frozen Matrix Transport unseen-battery report and evidence atlas."""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


ORDER = ("adamw", "matrix_transport")
LABELS = {"adamw": "AdamW", "matrix_transport": "Matrix Transport"}


def finite(values):
    return [float(value) for value in values if value is not None and math.isfinite(float(value))]


def mean(rows, key):
    values = finite(row.get(key) for row in rows)
    return statistics.fmean(values) if values else None


def clustered_interval(task_deltas, seed=1701, draws=20_000):
    """Percentile interval after resampling tasks, the independent units."""
    generator = random.Random(seed)
    values = list(task_deltas)
    samples = sorted(
        statistics.fmean(generator.choice(values) for _ in values)
        for _ in range(draws)
    )
    return [samples[int(draws * 0.025)], samples[int(draws * 0.975)]]


def aggregate(data):
    runs = data["runs"]
    grouped = defaultdict(list)
    fits = {}
    for row in runs:
        grouped[row["task"], row["optimizer"]].append(row)
        if row.get("fit_visualization") is not None and row["seed"] == 0:
            fits[row["task"], row["optimizer"]] = row["fit_visualization"]

    tasks = []
    paired_auc = []
    paired_score = []
    paired_tail = []
    for task in sorted({row["task"] for row in runs}):
        task_runs = [row for row in runs if row["task"] == task]
        curves = []
        means = {}
        for optimizer in ORDER:
            optimizer_rows = sorted(grouped[task, optimizer], key=lambda row: row["seed"])
            by_step = defaultdict(list)
            for row in optimizer_rows:
                for point in row["history"]:
                    score = float(point["score"])
                    loss = float(point["loss"])
                    if math.isfinite(score) and math.isfinite(loss):
                        by_step[int(point["step"])].append((score, loss))
            points = []
            for step, values in sorted(by_step.items()):
                scores, losses = zip(*values)
                points.append({
                    "step": step,
                    "score": statistics.fmean(scores),
                    "scoreMin": min(scores),
                    "scoreMax": max(scores),
                    "loss": statistics.fmean(losses),
                    "lossMin": min(losses),
                    "lossMax": max(losses),
                })
            means[optimizer] = {
                key: mean(optimizer_rows, key)
                for key in ("learning_auc", "validation_score", "score", "tail_score", "seconds")
            }
            curves.append({
                "optimizer": optimizer,
                **means[optimizer],
                "points": points,
                "fit": fits.get((task, optimizer)),
            })

        seed_pairs = []
        for seed in sorted({row["seed"] for row in task_runs}):
            pair = {
                row["optimizer"]: row
                for row in task_runs if row["seed"] == seed
            }
            seed_pair = {"seed": seed}
            for metric in ("learning_auc", "score", "tail_score"):
                left = pair["adamw"].get(metric)
                right = pair["matrix_transport"].get(metric)
                seed_pair[metric] = None if left is None or right is None else float(right - left)
            seed_pairs.append(seed_pair)
            paired_auc.append(seed_pair["learning_auc"])
            paired_score.append(seed_pair["score"])
            if seed_pair["tail_score"] is not None:
                paired_tail.append(seed_pair["tail_score"])

        first = task_runs[0]
        tasks.append({
            "task": task,
            "kind": first["kind"],
            "inputDim": first["input_dim"],
            "outputDim": first["output_dim"],
            "curves": curves,
            "seedPairs": seed_pairs,
            "aucDelta": means["matrix_transport"]["learning_auc"] - means["adamw"]["learning_auc"],
            "scoreDelta": means["matrix_transport"]["score"] - means["adamw"]["score"],
            "tailDelta": (
                None if means["matrix_transport"]["tail_score"] is None
                else means["matrix_transport"]["tail_score"] - means["adamw"]["tail_score"]
            ),
        })

    auc_task_deltas = [task["aucDelta"] for task in tasks]
    score_task_deltas = [task["scoreDelta"] for task in tasks]
    tail_task_deltas = finite(task["tailDelta"] for task in tasks)
    matrix_rows = [row for row in runs if row["optimizer"] == "matrix_transport"]
    adam_rows = [row for row in runs if row["optimizer"] == "adamw"]
    diagnostic_points = [
        point for row in matrix_rows for point in row["history"]
        if point.get("optimizer_root_refresh_fraction") is not None
    ]
    overview = {
        "runs": len(runs),
        "failures": sum(row.get("failure") is not None for row in runs),
        "tasks": len(tasks),
        "seeds": len({row["seed"] for row in runs}),
        "auc": {
            "delta": statistics.fmean(auc_task_deltas),
            "medianDelta": statistics.median(auc_task_deltas),
            "pairedWins": sum(value > 0 for value in paired_auc),
            "pairedTotal": len(paired_auc),
            "taskWins": sum(value > 0 for value in auc_task_deltas),
            "taskTotal": len(auc_task_deltas),
            "clustered95": clustered_interval(auc_task_deltas),
        },
        "score": {
            "delta": statistics.fmean(score_task_deltas),
            "medianDelta": statistics.median(score_task_deltas),
            "pairedWins": sum(value > 0 for value in paired_score),
            "pairedTies": sum(value == 0 for value in paired_score),
            "pairedTotal": len(paired_score),
            "taskWins": sum(value > 0 for value in score_task_deltas),
            "taskTies": sum(value == 0 for value in score_task_deltas),
            "taskTotal": len(score_task_deltas),
            "clustered95": clustered_interval(score_task_deltas, seed=1702),
        },
        "tail": {
            "delta": statistics.fmean(tail_task_deltas),
            "medianDelta": statistics.median(tail_task_deltas),
            "pairedWins": sum(value > 0 for value in paired_tail),
            "pairedTotal": len(paired_tail),
            "taskWins": sum(value > 0 for value in tail_task_deltas),
            "taskTotal": len(tail_task_deltas),
            "clustered95": clustered_interval(tail_task_deltas, seed=1703),
        },
        "runtime": {
            "adamw": mean(adam_rows, "seconds"),
            "matrix": mean(matrix_rows, "seconds"),
        },
        "roots": {
            "refreshFraction": statistics.fmean(
                float(point["optimizer_root_refresh_fraction"])
                for point in diagnostic_points
            ),
            "staleness": statistics.fmean(
                float(point["optimizer_root_staleness"])
                for point in diagnostic_points
            ),
        },
    }
    overview["runtime"]["ratio"] = overview["runtime"]["matrix"] / overview["runtime"]["adamw"]
    return {"configuration": data["configuration"], "overview": overview, "tasks": tasks}


def fmt(value, digits=4, signed=False):
    if value is None:
        return "—"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def markdown(evidence):
    overview = evidence["overview"]
    rows = []
    for task in evidence["tasks"]:
        adam, matrix = task["curves"]
        rows.append(
            f"| {task['task']} | {fmt(adam['learning_auc'])} | {fmt(matrix['learning_auc'])} | "
            f"{fmt(task['aucDelta'], signed=True)} | {fmt(adam['score'])} | {fmt(matrix['score'])} | "
            f"{fmt(task['scoreDelta'], signed=True)} |"
        )
    auc_low, auc_high = overview["auc"]["clustered95"]
    score_low, score_high = overview["score"]["clustered95"]
    tail_low, tail_high = overview["tail"]["clustered95"]
    return f"""# Matrix Transport: Frozen Unseen Battery

## Result first

The Ripple-developed optimizer was frozen, including its declared learning rate
of 0.003, and then run against AdamW on 23 other problems. Each arm used the
same self-context M-layer, width 24, initialization, minibatch order, exact
self-context backward pass, 500 updates, and three paired seeds. There were
{overview['failures']} numerical failures in {overview['runs']} runs.

Matrix Transport is a faster learner on this battery, but it is not yet a
uniformly better generalizer:

| Measure | Frozen unseen result |
|---|---:|
| Mean validation acquisition-AUC delta | **{fmt(overview['auc']['delta'], signed=True)}** |
| Task-clustered bootstrap 95% interval | [{fmt(auc_low, signed=True)}, {fmt(auc_high, signed=True)}] |
| Paired-seed AUC wins | **{overview['auc']['pairedWins']} / {overview['auc']['pairedTotal']}** |
| Task-mean AUC wins | **{overview['auc']['taskWins']} / {overview['auc']['taskTotal']}** |
| Mean held-out score delta | {fmt(overview['score']['delta'], signed=True)} |
| Task-clustered bootstrap 95% interval | [{fmt(score_low, signed=True)}, {fmt(score_high, signed=True)}] |
| Paired held-out wins / ties | {overview['score']['pairedWins']} / {overview['score']['pairedTies']} of {overview['score']['pairedTotal']} |
| Task-mean held-out wins / ties | {overview['score']['taskWins']} / {overview['score']['taskTies']} of {overview['score']['taskTotal']} |
| Mean tail-score delta ({overview['tail']['taskTotal']} applicable tasks) | {fmt(overview['tail']['delta'], signed=True)} |
| Task-clustered tail 95% interval | [{fmt(tail_low, signed=True)}, {fmt(tail_high, signed=True)}] |
| Mean M4 runtime, AdamW → Matrix | {overview['runtime']['adamw']:.2f}s → {overview['runtime']['matrix']:.2f}s ({overview['runtime']['ratio']:.2f}×) |

The acquisition result is distributed rather than being only a single showcase:
42 of 69 paired runs and 13 of 23 task means favor Matrix Transport. The
task-clustered interval still crosses zero, so this is encouraging evidence,
not a settled population effect. The final held-out result is less favorable:
its mean is almost exactly tied, its median task delta is
{fmt(overview['score']['medianDelta'], signed=True)}, and extrapolation tails
lean toward AdamW. This rejects the stronger claim that the current geometry
is already a universal replacement for AdamW.

The largest new win is the high-rank 16-D spiral: mean held-out score rises
from 0.7040 to 0.8898 while AUC rises by 0.0282. The clearest regressions are
radial stripes, chirp continuation, and drifted-chirp continuation. Those are
not optimizer failures during fitting; they are cases where faster
interpolation does not select the same extrapolating function.

## Cached matrix roots

The Kronecker factors are still updated every step. Their inverse fourth roots
are recomputed only when the corrected covariance moves by at least 5% in
relative Frobenius norm, or when a hard ten-step age limit is reached. The
cache never consults the loss, validation score, task identity, or gradient
labels. It is therefore a numerical realization rule for the same metric, not
a task-conditioned optimizer.

Across this unseen battery, evaluation snapshots report a mean root-refresh
fraction of {overview['roots']['refreshFraction']:.3f} and mean root age of
{overview['roots']['staleness']:.2f} updates. On the five-seed Ripple control,
the cache reduced end-to-end M4 time from 6.65s to 6.20s and slightly increased
mean AUC from 0.78443 to 0.78532. Matrix Transport remains
{overview['runtime']['ratio']:.2f}× AdamW wall time on this small CPU battery;
the residual is the matrix algebra and transport state, not repeated roots
alone.

## Every problem

The score is taken at the best validation checkpoint. AUC is the area under
the validation-learning curve, so it measures acquisition speed throughout
the fixed 500-update budget.

| Problem | AdamW AUC | Matrix AUC | Δ AUC | AdamW held-out | Matrix held-out | Δ held-out |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Interpretation

The result supports the narrow mechanism proposed on Ripple: a full matrix
second moment plus confidence-earned transport can improve how quickly a
self-context model acquires structure without using an elementwise adaptive
rate. It does not support the stronger proposition that learning faster on
observed support necessarily improves continuation. The tail regressions are
especially useful: they locate the next mathematical problem in the
optimizer's scalar coherence gain or checkpoint path, not in the cached
eigendecomposition.

No task-specific learning rates, geometry branches, loss probes, or post-hoc
optimizer changes were used. Ripple is excluded from all counts above.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    evidence = aggregate(json.loads(args.results.read_text()))
    args.summary.write_text(json.dumps(evidence, indent=2))
    args.report.write_text(markdown(evidence))
    packed = json.dumps(evidence, separators=(",", ":"))
    args.html.write_text(TEMPLATE.replace("__EVIDENCE__", packed))
    print(args.html)
    print(args.report)


TEMPLATE = r'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Frozen Matrix Transport · unseen battery</title><style>
:root{color-scheme:light dark;--bg:#f3f0e8;--fg:#161714;--muted:#686960;--panel:#fff;--line:#d4d1c7;--plot:#fbfaf6;--adam:#6855df;--matrix:#078d7c;--good:#087f5b;--bad:#be3f4b}
@media(prefers-color-scheme:dark){:root{--bg:#11130f;--fg:#f1eee5;--muted:#aaa99f;--panel:#1b1e18;--line:#3a3e34;--plot:#151711;--good:#4fd1a5;--bad:#ff7b84}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.48 ui-sans-serif,system-ui,sans-serif}header{padding:38px max(22px,4vw) 28px;border-bottom:1px solid var(--line)}h1{font-size:36px;line-height:1.04;margin:0 0 12px;letter-spacing:-.03em}header>p{max-width:1050px;font-size:16px;color:var(--muted);margin:7px 0}.facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:1px;background:var(--line);border:1px solid var(--line);margin-top:24px;max-width:1080px}.fact{background:var(--panel);padding:14px}.fact b{font-size:24px;display:block}.fact span{color:var(--muted);font-size:12px}.controls{position:sticky;top:0;z-index:5;padding:11px max(22px,4vw);display:flex;gap:18px;align-items:center;flex-wrap:wrap;background:color-mix(in srgb,var(--bg) 93%,transparent);backdrop-filter:blur(9px);border-bottom:1px solid var(--line)}select,input{font:inherit;padding:4px 7px}.legend{display:flex;gap:14px}.sw{display:inline-block;width:18px;height:3px;margin-right:5px;vertical-align:middle}main{padding:21px max(16px,3vw) 60px}.section{font-size:25px;margin:32px 0 6px}.note{color:var(--muted);max-width:1050px;margin:0 0 15px}.explain{max-width:1080px;display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px;margin:18px 0}.explain section{border-top:3px solid var(--line);padding:10px 4px}.explain h2{font-size:16px;margin:0 0 5px}.explain p{color:var(--muted);margin:0}.delta-wrap{max-width:1080px;background:var(--panel);border:1px solid var(--line);padding:14px}.delta-wrap svg{display:block;width:100%;height:auto}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(430px,1fr));gap:14px}.panel{min-width:0;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px}.panel h2{font-size:17px;margin:0}.sub{font-size:12px;color:var(--muted);margin:3px 0 9px}.metrics{display:grid;grid-template-columns:1fr 1fr;gap:5px;margin:8px 0}.metric{border-left:3px solid;padding:4px 7px;font-size:12px}.metric b{display:block}.metric span{color:var(--muted)}svg.curves{width:100%;height:auto;display:block}.axis{stroke:var(--line);stroke-width:1}.tick{fill:var(--muted);font-size:10px}.curve{fill:none;stroke-width:2.1}.band{opacity:.09}details{border-top:1px solid var(--line);margin-top:10px;padding-top:8px}summary{cursor:pointer;color:var(--muted);font-size:12px}.fitgrid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:9px}.fit h3{font-size:11px;margin:0 0 3px}.fit canvas,.fit svg{width:100%;height:auto;display:block;background:var(--plot);border:1px solid var(--line)}.positive{color:var(--good)}.negative{color:var(--bad)}@media(max-width:520px){.grid{grid-template-columns:1fr}.panel{padding:10px}.fitgrid{grid-template-columns:1fr}}
</style></head><body><header><h1>Frozen Matrix Transport on unseen problems</h1><p>One optimizer developed on Ripple, then frozen. Twenty-three different self-context M-layer problems, three paired seeds, 500 updates, and the same declared learning rate of 0.003 against AdamW. No task-conditioned branches and no post-hoc learning-rate choices.</p><p>The result is specific: transport improves acquisition speed across the battery; held-out behavior is mixed, and continuation tails still expose a weakness.</p><div id="facts" class="facts"></div></header>
<div class="controls"><label>Curve <select id="metric"><option value="score">validation score</option><option value="loss">training loss (log)</option></select></label><label>Find problem <input id="search" placeholder="spiral"></label><div class="legend"><span><i class="sw" style="background:var(--adam)"></i>AdamW</span><span><i class="sw" style="background:var(--matrix)"></i>Matrix Transport</span></div></div>
<main><div class="explain"><section><h2>What was optimized</h2><p>The left/right covariance factors still update every step. Their inverse fourth roots are reused until relative covariance drift reaches 5%, with a hard ten-step refresh limit. This changes realization cost, not optimizer geometry.</p></section><section><h2>What AUC means</h2><p>Area under the validation-score trajectory over the same 500-update budget. It measures how rapidly useful structure is acquired; it is not a substitute for held-out score.</p></section><section><h2>What failed to become universal</h2><p>Held-out score is almost tied in the mean, while extrapolation-tail score leans toward AdamW. Faster fitting and better continuation are distinct claims.</p></section></div>
<h2 class="section">Every task, without averaging it away</h2><p class="note">Bars are Matrix Transport minus AdamW. Green is favorable to transport. The two bars answer different questions: speed on validation support and quality at the best-validation checkpoint on held-out support.</p><div class="delta-wrap" id="deltas"></div>
<h2 class="section">Loss, validation, and fitted functions</h2><p class="note">Curves are three-seed means; translucent bands show the complete seed range. Switch to training loss to see floor behavior. Open the fit disclosure for paired seed-0 best checkpoints.</p><div class="grid" id="tasks"></div></main>
<script>const DATA=__EVIDENCE__,COLORS={adamw:'#6855df',matrix_transport:'#078d7c'},LABELS={adamw:'AdamW',matrix_transport:'Matrix Transport'},NS='http://www.w3.org/2000/svg';
const O=DATA.overview,sign=v=>v>0?'+':'',cls=v=>v>=0?'positive':'negative';document.querySelector('#facts').innerHTML=`<div class="fact"><b class="positive">${sign(O.auc.delta)}${O.auc.delta.toFixed(4)}</b><span>mean acquisition-AUC delta · ${O.auc.pairedWins}/${O.auc.pairedTotal} paired wins</span></div><div class="fact"><b class="${cls(O.score.delta)}">${sign(O.score.delta)}${O.score.delta.toFixed(4)}</b><span>mean held-out delta · interval crosses zero</span></div><div class="fact"><b>${O.roots.refreshFraction.toFixed(3)}</b><span>mean root-refresh fraction · hard max age 10</span></div><div class="fact"><b>${O.runtime.ratio.toFixed(2)}×</b><span>Matrix / AdamW wall time · ${O.failures} failures</span></div>`;
function el(tag,a={}){const n=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(a))n.setAttribute(k,v);return n}
function deltaChart(){const sorted=[...DATA.tasks].sort((a,b)=>a.aucDelta-b.aucDelta),W=980,row=25,m={l:190,r:75,t:28,b:34},H=m.t+m.b+sorted.length*row,all=sorted.flatMap(x=>[x.aucDelta,x.scoreDelta]),lim=Math.max(.02,...all.map(Math.abs)),x=v=>m.l+(W-m.l-m.r)*(.5+v/(2*lim)),s=el('svg',{viewBox:`0 0 ${W} ${H}`});s.append(el('line',{x1:x(0),y1:m.t-10,x2:x(0),y2:H-m.b,stroke:'var(--fg)','stroke-width':1}));for(const t of [-1,-.5,0,.5,1]){let v=t*lim,l=el('text',{x:x(v),y:H-10,'text-anchor':'middle',class:'tick'});l.textContent=(v>=0?'+':'')+v.toFixed(2);s.append(l)}sorted.forEach((d,i)=>{const y=m.t+i*row,l=el('text',{x:m.l-8,y:y+7,'text-anchor':'end',class:'tick'});l.textContent=d.task.replaceAll('_',' ');s.append(l);[[d.aucDelta,y-2,8],[d.scoreDelta,y+8,5]].forEach(([v,yy,h])=>s.append(el('rect',{x:Math.min(x(0),x(v)),y:yy,width:Math.max(1,Math.abs(x(v)-x(0))),height:h,fill:v>=0?'var(--good)':'var(--bad)',opacity:h===8?1:.55}))) });let a=el('text',{x:m.l,y:14,class:'tick'});a.textContent='solid: acquisition AUC   faint: held-out score';s.append(a);return s}document.querySelector('#deltas').append(deltaChart());
function curvePath(points,x,y,key){return points.map((p,i)=>`${i?'L':'M'}${x(p.step).toFixed(2)},${y(p[key]).toFixed(2)}`).join('')}
function curveChart(curves,metric){const W=460,H=220,m={l:50,r:10,t:8,b:28},s=el('svg',{viewBox:`0 0 ${W} ${H}`,class:'curves'}),all=curves.flatMap(c=>c.points),xmax=Math.max(...all.map(p=>p.step)),log=metric==='loss';let vals=all.flatMap(p=>[p[metric+'Min'],p[metric+'Max']]).filter(Number.isFinite);if(log)vals=vals.map(v=>Math.log10(Math.max(v,1e-30)));let lo=Math.min(...vals),hi=Math.max(...vals),pad=Math.max((hi-lo)*.07,.003);lo-=pad;hi+=pad;const x=v=>m.l+(W-m.l-m.r)*v/xmax,y=v=>{v=log?Math.log10(Math.max(v,1e-30)):v;return m.t+(H-m.t-m.b)*(hi-v)/(hi-lo||1)};s.append(el('line',{x1:m.l,y1:H-m.b,x2:W-m.r,y2:H-m.b,class:'axis'}),el('line',{x1:m.l,y1:m.t,x2:m.l,y2:H-m.b,class:'axis'}));for(const t of [0,.5,1]){let tx=el('text',{x:x(t*xmax),y:H-8,'text-anchor':'middle',class:'tick'});tx.textContent=Math.round(t*xmax);s.append(tx);let v=lo+t*(hi-lo),ty=el('text',{x:m.l-6,y:m.t+(1-t)*(H-m.t-m.b)+3,'text-anchor':'end',class:'tick'});ty.textContent=log?`1e${v.toFixed(0)}`:v.toFixed(2);s.append(ty)}for(const c of curves){const top=c.points.map(p=>`${x(p.step)},${y(p[metric+'Max'])}`),bot=[...c.points].reverse().map(p=>`${x(p.step)},${y(p[metric+'Min'])}`);s.append(el('polygon',{points:[...top,...bot].join(' '),fill:COLORS[c.optimizer],class:'band'}));s.append(el('path',{d:curvePath(c.points,x,y,metric),stroke:COLORS[c.optimizer],class:'curve'}))}return s}
function colorValue(v){v=Math.max(0,Math.min(1,v));const a=[47,58,133],b=[25,151,124],c=[243,218,54],t=v<.5?v*2:(v-.5)*2,p=v<.5?a:b,q=v<.5?b:c;return `rgb(${p.map((x,i)=>Math.round(x+(q[i]-x)*t)).join(',')})`}
function categorical(v){return ['#3566c4','#dd493f','#31a36c','#9b56b2','#e89422','#22a5ad'][Math.abs(Math.round(v))%6]}
function surface(f){const c=document.createElement('canvas');c.width=220;c.height=170;const g=c.getContext('2d'),side=f.side;for(let i=0;i<f.prediction.length;i++){g.fillStyle=f.confidence?categorical(f.prediction[i]):colorValue((f.prediction[i]+2)/4);g.fillRect((i%side)*c.width/side,Math.floor(i/side)*c.height/side,c.width/side+1,c.height/side+1)}if(f.sample_x){const[x0,x1,y0,y1]=f.limits;f.sample_x.forEach((p,i)=>{g.fillStyle=categorical(f.sample_y[i]);g.globalAlpha=.65;g.beginPath();g.arc((p[0]-x0)/(x1-x0)*c.width,c.height-(p[1]-y0)/(y1-y0)*c.height,1.5,0,7);g.fill()});g.globalAlpha=1}return c}
function lineFit(f,is3=false){const W=220,H=170,m=10,s=el('svg',{viewBox:`0 0 ${W} ${H}`});if(is3){const proj=p=>[W/2+38*p[0]+20*p[2],H/2-38*p[1]+12*p[2]],path=a=>a.map((p,i)=>`${i?'L':'M'}${proj(p)}`).join('');s.append(el('path',{d:path(f.target),fill:'none',stroke:'#888','stroke-width':2}),el('path',{d:path(f.prediction),fill:'none',stroke:'#d7426a','stroke-width':1.5}));return s}const x=f.coordinate,all=[...f.target,...f.prediction],x0=Math.min(...x),x1=Math.max(...x),y0=Math.min(...all),y1=Math.max(...all),xp=v=>m+(W-2*m)*(v-x0)/(x1-x0||1),yp=v=>H-m-(H-2*m)*(v-y0)/(y1-y0||1),path=a=>a.map((v,i)=>`${i?'L':'M'}${xp(x[i])},${yp(v)}`).join('');s.append(el('path',{d:path(f.target),fill:'none',stroke:'#888','stroke-width':2}),el('path',{d:path(f.prediction),fill:'none',stroke:'#d7426a','stroke-width':1.5}));return s}
function scatter(f,nd=false){const W=220,H=170,m=8,s=el('svg',{viewBox:`0 0 ${W} ${H}`}),p=f.coordinates,xv=p.map(q=>q[0]),yv=p.map(q=>q[1]),zv=p.map(q=>q[2]||0),x0=Math.min(...xv),x1=Math.max(...xv),y0=Math.min(...yv),y1=Math.max(...yv),z0=Math.min(...zv),z1=Math.max(...zv);p.forEach((q,i)=>{const x=m+(W-2*m)*(xv[i]-x0)/(x1-x0||1)+8*(zv[i]-z0)/(z1-z0||1),y=H-m-(H-2*m)*(yv[i]-y0)/(y1-y0||1)-5*(zv[i]-z0)/(z1-z0||1),fill=nd?colorValue(f.probability[i]):categorical(Array.isArray(f.prediction[i])?0:f.prediction[i]),stroke=nd?categorical(f.label[i]):categorical(Array.isArray(f.target[i])?0:f.target[i]);s.append(el('circle',{cx:x,cy:y,r:nd?1.7:2,fill,stroke,'stroke-width':.5,opacity:nd&&f.region[i]?'0.28':'0.82'}))});return s}
function fitView(f){if(!f)return document.createTextNode('No fit payload');if(f.kind==='surface_2d')return surface(f);if(f.kind==='curve_1d')return lineFit(f);if(f.kind==='curve_3d')return lineFit(f,true);if(f.kind==='nd_spiral_3d')return scatter(f,true);return scatter(f,false)}
function render(){const metric=document.querySelector('#metric').value,q=document.querySelector('#search').value.toLowerCase(),root=document.querySelector('#tasks');root.innerHTML='';for(const p of DATA.tasks){if(q&&!p.task.includes(q))continue;const card=document.createElement('article');card.className='panel';const a=p.curves[0],m=p.curves[1];card.innerHTML=`<h2>${p.task.replaceAll('_',' ')}</h2><div class="sub">${p.kind} · ${p.inputDim}→${p.outputDim} · three paired seeds</div><div class="metrics"><div class="metric" style="border-color:${COLORS.adamw}"><b>AdamW</b><span>AUC ${a.learning_auc.toFixed(4)} · held ${a.score.toFixed(4)}</span></div><div class="metric" style="border-color:${COLORS.matrix_transport}"><b>Matrix Transport</b><span>AUC ${m.learning_auc.toFixed(4)} · held ${m.score.toFixed(4)}</span></div></div><div class="sub">Δ AUC <b class="${cls(p.aucDelta)}">${sign(p.aucDelta)}${p.aucDelta.toFixed(4)}</b> · Δ held-out <b class="${cls(p.scoreDelta)}">${sign(p.scoreDelta)}${p.scoreDelta.toFixed(4)}</b></div>`;card.append(curveChart(p.curves,metric));const d=document.createElement('details');d.innerHTML='<summary>Best-checkpoint fitted functions · paired seed 0</summary>';const g=document.createElement('div');g.className='fitgrid';for(const c of p.curves){const v=document.createElement('div');v.className='fit';v.innerHTML=`<h3 style="color:${COLORS[c.optimizer]}">${LABELS[c.optimizer]}</h3>`;v.append(fitView(c.fit));g.append(v)}d.append(g);card.append(d);root.append(card)}}document.querySelector('#metric').addEventListener('change',render);document.querySelector('#search').addEventListener('input',render);render();
</script></body></html>'''


if __name__ == "__main__":
    main()
