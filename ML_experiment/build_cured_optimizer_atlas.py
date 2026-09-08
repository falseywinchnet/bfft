"""Build the four-arm source-aware self-context optimizer atlas."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


OPTIMIZER_ORDER = (
    "anchor_cured",
    "lepton_transport_cured",
    "adamw",
    "adamw_transport",
)
OPERATOR_ORDER = (*OPTIMIZER_ORDER, "muon")
OPERATOR_NAMES = {
    "anchor": "anchor_cured",
    "lepton_transport_guarded": "lepton_transport_cured",
    "adamw": "adamw",
    "adamw_transport": "adamw_transport",
    "muon": "muon",
}


def finite_mean(values):
    values = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.fmean(values) if values else float("nan")


def aggregate(rows):
    grouped = defaultdict(list)
    fits = {}
    task_meta = {}
    for row in rows:
        key = row["task"], row["optimizer"]
        grouped[key].append(row)
        task_meta[row["task"]] = {
            "kind": row["kind"],
            "inputDim": row["input_dim"],
            "outputDim": row["output_dim"],
        }
        if row.get("fit_visualization") is not None:
            fits[key] = row["fit_visualization"]

    tasks = defaultdict(list)
    overall = defaultdict(lambda: {
        "auc": [], "validation": [], "test": [], "seconds": [],
        "bestStep": [], "failures": 0, "aucWins": 0, "testWins": 0,
    })
    for (task, optimizer), optimizer_rows in grouped.items():
        by_step = defaultdict(list)
        context_scales = []
        for row in optimizer_rows:
            for point in row["history"]:
                score, loss = float(point["score"]), float(point["loss"])
                if math.isfinite(score) and math.isfinite(loss):
                    by_step[int(point["step"])].append((score, loss))
                if "context_chart_scale" in point:
                    context_scales.append(float(point["context_chart_scale"]))
        points = []
        for step, values in sorted(by_step.items()):
            scores, losses = zip(*values)
            points.append({
                "step": step,
                "score": statistics.fmean(scores),
                "scoreMin": min(scores), "scoreMax": max(scores),
                "loss": statistics.fmean(losses),
                "lossMin": min(losses), "lossMax": max(losses),
            })
        auc_value = finite_mean(row["learning_auc"] for row in optimizer_rows)
        validation = finite_mean(row["validation_score"] for row in optimizer_rows)
        test = finite_mean(row["score"] for row in optimizer_rows)
        seconds = finite_mean(row["seconds"] for row in optimizer_rows)
        best_step = finite_mean(row["best_step"] for row in optimizer_rows)
        tail = finite_mean(
            row["tail_score"] for row in optimizer_rows if "tail_score" in row
        )
        failures = sum(row.get("failure") is not None for row in optimizer_rows)
        summary = {
            "optimizer": optimizer,
            "auc": auc_value,
            "validation": validation,
            "test": test,
            "tail": tail,
            "seconds": seconds,
            "bestStep": best_step,
            "failures": failures,
            "contextMinScale": min(context_scales, default=1.0),
            "contextMeanScale": finite_mean(context_scales) if context_scales else 1.0,
            "points": points,
            "fit": fits.get((task, optimizer)),
        }
        tasks[task].append(summary)
        destination = overall[optimizer]
        for key, value in (
            ("auc", auc_value), ("validation", validation), ("test", test),
            ("seconds", seconds), ("bestStep", best_step),
        ):
            destination[key].append(value)
        destination["failures"] += failures

    panels = []
    for task, curves in sorted(tasks.items()):
        curves.sort(key=lambda curve: OPTIMIZER_ORDER.index(curve["optimizer"]))
        max_auc = max(curve["auc"] for curve in curves)
        max_test = max(curve["test"] for curve in curves)
        for curve in curves:
            if curve["auc"] == max_auc:
                overall[curve["optimizer"]]["aucWins"] += 1
            if curve["test"] == max_test:
                overall[curve["optimizer"]]["testWins"] += 1
        panels.append({"task": task, **task_meta[task], "curves": curves})

    leaderboard = []
    for optimizer in OPTIMIZER_ORDER:
        values = overall[optimizer]
        leaderboard.append({
            "optimizer": optimizer,
            "auc": finite_mean(values["auc"]),
            "validation": finite_mean(values["validation"]),
            "test": finite_mean(values["test"]),
            "seconds": finite_mean(values["seconds"]),
            "bestStep": finite_mean(values["bestStep"]),
            "failures": values["failures"],
            "aucWins": values["aucWins"],
            "testWins": values["testWins"],
        })
    return panels, leaderboard


def aggregate_operator(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[OPERATOR_NAMES[row["optimizer"]]].append(row)
    curves = []
    for optimizer in OPERATOR_ORDER:
        optimizer_rows = grouped[optimizer]
        by_step = defaultdict(list)
        for row in optimizer_rows:
            for point in row["history"]:
                by_step[int(point["step"])].append((
                    float(point["relative_loss"]),
                    float(point["operator_error"]),
                ))
        points = []
        for step, values in sorted(by_step.items()):
            losses, errors = zip(*values)
            points.append({
                "step": step,
                "loss": statistics.fmean(losses),
                "lossMin": min(losses), "lossMax": max(losses),
                "error": statistics.fmean(errors),
                "errorMin": min(errors), "errorMax": max(errors),
            })
        curves.append({
            "optimizer": optimizer,
            "points": points,
            "finalLoss": finite_mean(
                row["final_relative_loss"] for row in optimizer_rows
            ),
            "finalError": finite_mean(
                row["final_operator_error"] for row in optimizer_rows
            ),
            "toLoss1e2": finite_mean(
                row["steps_to_loss_1e-2"] for row in optimizer_rows
                if row["steps_to_loss_1e-2"] is not None
            ),
        })
    return curves


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--operator", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.results.read_text())
    panels, leaderboard = aggregate(payload["runs"])
    evidence = {
        "order": list(OPTIMIZER_ORDER),
        "configuration": payload["configuration"],
        "tasks": panels,
        "leaderboard": leaderboard,
        "operator": (
            aggregate_operator(json.loads(args.operator.read_text())["runs"])
            if args.operator else []
        ),
    }
    document = TEMPLATE.replace(
        "__EVIDENCE__", json.dumps(evidence, separators=(",", ":"))
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(document)
    print(args.out)


TEMPLATE = r'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Transport under self-context — optimizer atlas</title><style>
:root{color-scheme:light dark;--bg:#f2eee5;--ink:#171713;--muted:#66645d;--paper:#fffdf8;--line:#d5cdbc;--plot:#faf8f1;--anchor:#c66a25;--lepton:#008f7c;--adam:#6857d5;--tadam:#ca4168;--muon:#087e8b}
@media(prefers-color-scheme:dark){:root{--bg:#11130f;--ink:#f3efe3;--muted:#aaa79d;--paper:#1a1d17;--line:#393d33;--plot:#141610;--anchor:#ef9a52;--lepton:#42c7af;--adam:#9c8df0;--tadam:#ee7898;--muon:#47b6c0}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 ui-sans-serif,system-ui,sans-serif}header,main{max-width:1640px;margin:auto;padding-left:max(20px,3vw);padding-right:max(20px,3vw)}header{padding-top:46px;padding-bottom:32px;border-bottom:1px solid var(--line)}h1{font:700 clamp(34px,5vw,68px)/.98 ui-serif,Georgia,serif;letter-spacing:-.035em;max-width:1080px;margin:0 0 22px}header p{font-size:17px;max-width:1020px;margin:9px 0;color:var(--muted)}.equation{font:16px/1.7 ui-monospace,SFMono-Regular,monospace;color:var(--ink);background:var(--paper);border-left:4px solid var(--anchor);padding:14px 18px;margin:24px 0;overflow:auto}.guide{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:22px;margin-top:28px}.guide h2{font-size:13px;text-transform:uppercase;letter-spacing:.08em;margin:0 0 5px}.guide p{font-size:14px;margin:0}.sticky{position:sticky;top:0;z-index:5;background:color-mix(in srgb,var(--bg) 94%,transparent);backdrop-filter:blur(10px);border-bottom:1px solid var(--line);padding:10px max(20px,3vw);display:flex;gap:18px;align-items:center}.legend{display:flex;gap:14px;flex-wrap:wrap;flex:1}.legend span{white-space:nowrap}.swatch{display:inline-block;width:20px;height:3px;margin-right:6px;vertical-align:middle}.sticky input{font:inherit;padding:5px 8px;border:1px solid var(--line);background:var(--paper);color:var(--ink);width:190px}main{padding-top:34px;padding-bottom:80px}.section-title{font:700 29px/1.1 ui-serif,Georgia,serif;margin:0 0 8px}.section-note{color:var(--muted);max-width:1000px;margin:0 0 20px}.leader{width:100%;border-collapse:collapse;background:var(--paper);margin:0 0 42px}.leader th,.leader td{padding:9px 11px;border-bottom:1px solid var(--line);text-align:right;font-variant-numeric:tabular-nums}.leader th:first-child,.leader td:first-child{text-align:left}.tasks{display:grid;gap:28px}.task{background:var(--paper);border:1px solid var(--line);padding:20px;min-width:0}.task-head{display:flex;align-items:baseline;justify-content:space-between;gap:16px;margin-bottom:13px}.task h2{font:700 23px/1.1 ui-serif,Georgia,serif;margin:0}.meta{color:var(--muted);font-size:12px}.plots{display:grid;grid-template-columns:1fr 1fr;gap:12px}.plot-block h3,.fit h3{font-size:12px;margin:0 0 5px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}svg.chart{width:100%;height:auto;display:block;background:var(--plot);border:1px solid var(--line)}.axis{stroke:var(--line);stroke-width:1}.tick{fill:var(--muted);font-size:10px}.curve{fill:none;stroke-width:2.2}.band{opacity:.09}.metrics{width:100%;border-collapse:collapse;margin:14px 0 18px}.metrics th,.metrics td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:right;font-size:12px;font-variant-numeric:tabular-nums}.metrics th:first-child,.metrics td:first-child{text-align:left}.fits{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}.fit{min-width:0}.fit canvas,.fit svg{display:block;width:100%;height:auto;aspect-ratio:4/3;background:var(--plot);border:1px solid var(--line);touch-action:none}.fit-note{font-size:11px;color:var(--muted);margin-top:4px}.empty{color:var(--muted);padding:28px 0}.winner{font-weight:750}
@media(max-width:980px){.fits{grid-template-columns:1fr 1fr}.guide{grid-template-columns:1fr}.plots{grid-template-columns:1fr}}
@media(max-width:560px){header,main{padding-left:12px;padding-right:12px}.sticky{padding-left:12px;padding-right:12px;align-items:flex-start;flex-direction:column;gap:7px}.task{padding:12px}.fits{grid-template-columns:1fr}.leader{font-size:11px}.leader th,.leader td{padding:6px 4px}.hide-mobile{display:none}}
#leaderboard{max-width:100%;overflow-x:auto}.task{overflow-x:auto}
@media(max-width:560px){.leader{min-width:650px}.metrics{min-width:620px}}
</style></head><body><header><h1>Transport under self-context</h1>
<p>Four optimizers face the same self-context model, initialization, minibatch stream, width, training budget, and source-aware backward rule. There is no task detector and no optimizer chosen after seeing geometry.</p>
<p>The forward model is unchanged. Backpropagation distinguishes the loss gradient with the current chart held fixed from feedback caused by moving that chart. The normalization Jacobian is made locally nonexpansive, then chart feedback is restrained within each parameter tensor without clipping its coordinates.</p>
<div class="equation">g<sub>fixed</sub> = D<sub>θ</sub>L(θ,c) &nbsp;&nbsp; g<sub>chart</sub> = D<sub>θ</sub>L(θ,c(θ)) − g<sub>fixed</sub><br>α<sub>p</sub> = min(1, ‖g<sub>fixed,p</sub>‖ / ‖g<sub>chart,p</sub>‖) &nbsp;&nbsp; g<sub>step,p</sub> = g<sub>fixed,p</sub> + α<sub>p</sub>g<sub>chart,p</sub></div>
<div class="guide"><div><h2>Learning</h2><p>Validation score and batch loss are shown together. Lines are three-seed means; translucent envelopes span all seeds.</p></div><div><h2>Generalization</h2><p>Validation selects the checkpoint. Held-out and tail columns evaluate different samples, often including continuation beyond the trained support.</p></div><div><h2>Fit</h2><p>Every reconstruction is the paired seed-0 best-validation checkpoint. Drag an N-D spiral to rotate its three-dimensional PCA reconstruction.</p></div></div></header>
<div class="sticky"><div id="legend" class="legend"></div><label>Find problem <input id="search" placeholder="spiral, ripple…"></label></div>
<main><h2 class="section-title">Battery summary</h2><p class="section-note">Task-averaged scores are orientation, not a substitute for the per-problem curves. AUC wins measure learning speed; held-out wins measure the selected checkpoint away from its validation sample.</p><div id="leaderboard"></div>
<h2 class="section-title">Muon-specialized challenge: Standing Population</h2><p class="section-note">This synthetic operator-recovery witness keeps a full rotated covariance visible. It was designed to expose Muon's polar geometry. Relative objective measures fitting; relative operator error asks whether the hidden linear map itself was recovered. Muon is shown here as the intended geometric control, not as a task-conditioned member of the self-context family.</p><div id="operator"></div>
<h2 class="section-title">Every problem</h2><p class="section-note">No cards are suppressed because a method failed to learn. Those negative controls define the battery's limits.</p><div id="tasks" class="tasks"></div></main>
<script>const DATA=__EVIDENCE__;
const COLORS={anchor_cured:'var(--anchor)',lepton_transport_cured:'var(--lepton)',adamw:'var(--adam)',adamw_transport:'var(--tadam)',muon:'var(--muon)'};
const LABELS={anchor_cured:'Anchor',lepton_transport_cured:'Transport-Lepton',adamw:'AdamW',adamw_transport:'Transported AdamW',muon:'Muon'};
const NS='http://www.w3.org/2000/svg';
function node(tag,attrs={}){const n=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,v);return n}
function fmt(v,d=3){return Number.isFinite(v)?v.toFixed(d):'—'}
function chart(curves,key,log){const W=620,H=245,m={l:58,r:12,t:10,b:30},s=node('svg',{viewBox:`0 0 ${W} ${H}`,class:'chart','aria-label':key==='score'?'validation score by step':'training loss by step'}),all=curves.flatMap(c=>c.points),xmax=Math.max(...all.map(p=>p.step)),raw=all.flatMap(p=>[p[key+'Min'],p[key+'Max']]).filter(Number.isFinite),vals=log?raw.map(v=>Math.log10(Math.max(v,1e-30))):raw;let lo=Math.min(...vals),hi=Math.max(...vals),pad=Math.max((hi-lo)*.06,.002);lo-=pad;hi+=pad;const xp=v=>m.l+(W-m.l-m.r)*v/xmax,yp=v=>{const z=log?Math.log10(Math.max(v,1e-30)):v;return m.t+(H-m.t-m.b)*(hi-z)/(hi-lo||1)};s.append(node('line',{x1:m.l,y1:H-m.b,x2:W-m.r,y2:H-m.b,class:'axis'}),node('line',{x1:m.l,y1:m.t,x2:m.l,y2:H-m.b,class:'axis'}));for(const t of [0,.25,.5,.75,1]){if(innerWidth<520&&t%0.5!==0)continue;const tx=node('text',{x:xp(t*xmax),y:H-9,'text-anchor':'middle',class:'tick'});tx.textContent=Math.round(t*xmax);s.append(tx)}for(const t of [0,.5,1]){const value=lo+t*(hi-lo),ty=node('text',{x:m.l-7,y:m.t+(1-t)*(H-m.t-m.b)+3,'text-anchor':'end',class:'tick'});ty.textContent=log?`10^${value.toFixed(1)}`:value.toFixed(2);s.append(ty)}for(const c of curves){const top=c.points.map(p=>`${xp(p.step)},${yp(p[key+'Max'])}`),bottom=[...c.points].reverse().map(p=>`${xp(p.step)},${yp(p[key+'Min'])}`);s.append(node('polygon',{points:[...top,...bottom].join(' '),fill:COLORS[c.optimizer],class:'band'}));const d=c.points.map((p,i)=>`${i?'L':'M'}${xp(p.step).toFixed(2)},${yp(p[key]).toFixed(2)}`).join('');s.append(node('path',{d,stroke:COLORS[c.optimizer],class:'curve'}))}return s}
function classColor(i){return['#2878c8','#dc4b43','#25a270','#9a55b5','#e58b22','#1d9ca8'][i%6]}
function valueColor(v){v=Math.max(0,Math.min(1,v));const a=[45,55,140],b=[26,163,132],c=[245,218,54],p=v<.5?a:b,q=v<.5?b:c,t=v<.5?v*2:(v-.5)*2;return `rgb(${p.map((x,i)=>Math.round(x+(q[i]-x)*t)).join(',')})`}
function surface(fit){const c=document.createElement('canvas');c.width=360;c.height=270;const g=c.getContext('2d'),side=fit.side,isClass=Boolean(fit.confidence);if(!isClass&&fit.target){const all=[...fit.target,...fit.prediction],lo=Math.min(...all),hi=Math.max(...all),half=c.width/2,draw=(values,offset)=>{const cw=half/side,ch=c.height/side;for(let i=0;i<values.length;i++){g.fillStyle=valueColor((values[i]-lo)/(hi-lo||1));g.fillRect(offset+(i%side)*cw,Math.floor(i/side)*ch,cw+1,ch+1)}};draw(fit.target,0);draw(fit.prediction,half);g.fillStyle='rgba(255,255,255,.82)';g.fillRect(4,4,48,17);g.fillRect(half+4,4,64,17);g.fillStyle='#171713';g.font='11px system-ui';g.fillText('target',9,16);g.fillText('prediction',half+9,16);return c}const cw=c.width/side,ch=c.height/side;for(let i=0;i<fit.prediction.length;i++){g.fillStyle=classColor(fit.prediction[i]);g.fillRect((i%side)*cw,Math.floor(i/side)*ch,cw+1,ch+1)}if(fit.sample_x){const [xmin,xmax,ymin,ymax]=fit.limits;for(let i=0;i<fit.sample_x.length;i++){const p=fit.sample_x[i];g.fillStyle=classColor(fit.sample_y[i]);g.globalAlpha=.65;g.beginPath();g.arc((p[0]-xmin)/(xmax-xmin)*c.width,c.height-(p[1]-ymin)/(ymax-ymin)*c.height,1.7,0,Math.PI*2);g.fill()}g.globalAlpha=1}return c}
function lines(fit,spatial=false){const W=320,H=240,m=15,s=node('svg',{viewBox:`0 0 ${W} ${H}`}),target=fit.target,pred=fit.prediction;if(spatial){const project=p=>[W/2+55*p[0]+27*p[2],H/2-55*p[1]+17*p[2]],path=a=>a.map((p,i)=>`${i?'L':'M'}${project(p).join(',')}`).join('');s.append(node('path',{d:path(target),fill:'none',stroke:'#85857e','stroke-width':2.4}),node('path',{d:path(pred),fill:'none',stroke:'var(--tadam)','stroke-width':2}));return s}const x=fit.coordinate,all=[...target,...pred],xmin=Math.min(...x),xmax=Math.max(...x),ymin=Math.min(...all),ymax=Math.max(...all),xp=v=>m+(W-2*m)*(v-xmin)/(xmax-xmin||1),yp=v=>H-m-(H-2*m)*(v-ymin)/(ymax-ymin||1),path=a=>a.map((v,i)=>`${i?'L':'M'}${xp(x[i])},${yp(v)}`).join('');s.append(node('path',{d:path(target),fill:'none',stroke:'#85857e','stroke-width':2.4}),node('path',{d:path(pred),fill:'none',stroke:'var(--tadam)','stroke-width':2}));return s}
function orbit(fit){const c=document.createElement('canvas');c.width=360;c.height=270;const g=c.getContext('2d'),points=fit.coordinates;let yaw=-.65,pitch=.35,drag=false,last=null;const extent=Math.max(...points.flatMap(p=>p.map(Math.abs)),1);function draw(){g.clearRect(0,0,c.width,c.height);const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch),projected=points.map((p,i)=>{const x=cy*p[0]+sy*p[2],z=-sy*p[0]+cy*p[2],y=cp*p[1]-sp*z,d=sp*p[1]+cp*z;return{x:c.width/2+150*x/extent,y:c.height/2-150*y/extent,d,i}}).sort((a,b)=>a.d-b.d);for(const p of projected){const i=p.i,unseen=fit.region&&fit.region[i],prob=fit.probability?fit.probability[i]:.5,label=fit.label?fit.label[i]:0;g.globalAlpha=unseen?.25:.82;g.fillStyle=valueColor(prob);g.strokeStyle=classColor(label);g.lineWidth=.8;g.beginPath();g.arc(p.x,p.y,unseen?1.8:2.5,0,Math.PI*2);g.fill();g.stroke()}g.globalAlpha=1}c.addEventListener('pointerdown',e=>{drag=true;last=[e.clientX,e.clientY];c.setPointerCapture(e.pointerId)});c.addEventListener('pointermove',e=>{if(!drag)return;yaw+=(e.clientX-last[0])*.012;pitch=Math.max(-1.35,Math.min(1.35,pitch+(e.clientY-last[1])*.009));last=[e.clientX,e.clientY];draw()});c.addEventListener('pointerup',()=>drag=false);draw();return c}
function scatter(fit){const c=document.createElement('canvas');c.width=360;c.height=270;const g=c.getContext('2d'),p=fit.coordinates,x=p.map(q=>q[0]),y=p.map(q=>q[1]),xmin=Math.min(...x),xmax=Math.max(...x),ymin=Math.min(...y),ymax=Math.max(...y),scalar=v=>Array.isArray(v)?v[0]:v,pred=fit.prediction.map(scalar),target=fit.target.map(scalar),all=[...pred,...target],lo=Math.min(...all),hi=Math.max(...all),continuous=Array.isArray(fit.prediction[0]);for(let i=0;i<p.length;i++){const cx=10+(c.width-20)*(x[i]-xmin)/(xmax-xmin||1),cy=c.height-10-(c.height-20)*(y[i]-ymin)/(ymax-ymin||1);g.globalAlpha=.72;g.fillStyle=continuous?valueColor((pred[i]-lo)/(hi-lo||1)):classColor(pred[i]);g.strokeStyle=continuous?valueColor((target[i]-lo)/(hi-lo||1)):classColor(target[i]);g.lineWidth=.8;g.beginPath();g.arc(cx,cy,2.4,0,Math.PI*2);g.fill();g.stroke()}g.globalAlpha=1;return c}
function fitVisual(fit){if(!fit)return null;if(fit.kind==='surface_2d')return surface(fit);if(fit.kind==='curve_1d')return lines(fit);if(fit.kind==='curve_3d')return lines(fit,true);if(fit.kind==='nd_spiral_3d')return orbit(fit);return scatter(fit)}
function fitNote(fit){if(!fit)return'No fit payload';if(fit.kind==='nd_spiral_3d')return'Fill: predicted probability · outline: true class · faint: continuation · drag to rotate';if(fit.kind==='surface_2d')return fit.confidence?'Field: predicted class · dots: true samples':'Left: target · right: prediction · shared color scale';if(fit.kind==='projected_fit')return'Fill: prediction · outline: target · shared continuous scale';return'Gray: target · colored line: prediction'}
function leaderboard(){const root=document.querySelector('#leaderboard'),table=document.createElement('table');table.className='leader';table.innerHTML='<thead><tr><th>optimizer</th><th>AUC wins</th><th>held-out wins</th><th>mean AUC</th><th>mean validation</th><th>mean held-out</th><th class="hide-mobile">mean best step</th><th class="hide-mobile">seconds / run</th><th>failures</th></tr></thead>';const body=document.createElement('tbody');for(const r of DATA.leaderboard){const tr=document.createElement('tr');tr.innerHTML=`<td style="color:${COLORS[r.optimizer]};font-weight:700">${LABELS[r.optimizer]}</td><td>${r.aucWins}</td><td>${r.testWins}</td><td>${fmt(r.auc,4)}</td><td>${fmt(r.validation,4)}</td><td>${fmt(r.test,4)}</td><td class="hide-mobile">${fmt(r.bestStep,0)}</td><td class="hide-mobile">${fmt(r.seconds,2)}</td><td>${r.failures}</td>`;body.append(tr)}table.append(body);root.append(table)}
function renderOperator(){const root=document.querySelector('#operator');if(!DATA.operator.length){root.textContent='Operator witness not supplied';return}const card=document.createElement('article');card.className='task';const plots=document.createElement('div');plots.className='plots';for(const [key,title]of[['loss','relative objective · log scale'],['error','relative operator error · log scale']]){const block=document.createElement('div');block.className='plot-block';block.innerHTML=`<h3>${title}</h3>`;block.append(chart(DATA.operator,key,true));plots.append(block)}card.append(plots);const table=document.createElement('table');table.className='metrics';table.innerHTML='<thead><tr><th>optimizer</th><th>final relative objective</th><th>final operator error</th><th>steps to objective 10⁻²</th></tr></thead>';const body=document.createElement('tbody');for(const c of DATA.operator){const tr=document.createElement('tr');tr.innerHTML=`<td style="color:${COLORS[c.optimizer]};font-weight:700">${LABELS[c.optimizer]}</td><td>${fmt(c.finalLoss,6)}</td><td>${fmt(c.finalError,4)}</td><td>${fmt(c.toLoss1e2,0)}</td>`;body.append(tr)}table.append(body);card.append(table);root.append(card)}
function taskCard(panel){const card=document.createElement('article');card.className='task';card.dataset.name=panel.task;const head=document.createElement('div');head.className='task-head';head.innerHTML=`<h2>${panel.task.replaceAll('_',' ')}</h2><div class="meta">${panel.kind} · ${panel.inputDim}D → ${panel.outputDim}D · 3 seeds</div>`;card.append(head);const plots=document.createElement('div');plots.className='plots';for(const [key,title,log]of[['score','validation score',false],['loss','training loss · log scale',true]]){const block=document.createElement('div');block.className='plot-block';block.innerHTML=`<h3>${title}</h3>`;block.append(chart(panel.curves,key,log));plots.append(block)}card.append(plots);const bestA=Math.max(...panel.curves.map(c=>c.auc)),bestT=Math.max(...panel.curves.map(c=>c.test)),table=document.createElement('table');table.className='metrics';table.innerHTML='<thead><tr><th>optimizer</th><th>learning AUC</th><th>validation</th><th>held-out</th><th>tail</th><th>best step</th><th class="hide-mobile">sec</th><th class="hide-mobile">min chart scale</th></tr></thead>';const body=document.createElement('tbody');for(const c of panel.curves){const tr=document.createElement('tr');tr.innerHTML=`<td style="color:${COLORS[c.optimizer]};font-weight:700">${LABELS[c.optimizer]}</td><td class="${c.auc===bestA?'winner':''}">${fmt(c.auc,4)}</td><td>${fmt(c.validation,4)}</td><td class="${c.test===bestT?'winner':''}">${fmt(c.test,4)}</td><td>${fmt(c.tail,4)}</td><td>${fmt(c.bestStep,0)}</td><td class="hide-mobile">${fmt(c.seconds,2)}</td><td class="hide-mobile">${fmt(c.contextMinScale,3)}</td>`;body.append(tr)}table.append(body);card.append(table);const fits=document.createElement('div');fits.className='fits';for(const c of panel.curves){const box=document.createElement('div');box.className='fit';box.innerHTML=`<h3 style="color:${COLORS[c.optimizer]}">${LABELS[c.optimizer]} · fitted result</h3>`;const visual=fitVisual(c.fit);if(visual)box.append(visual);else{const empty=document.createElement('div');empty.className='empty';empty.textContent='No fit payload';box.append(empty)}const note=document.createElement('div');note.className='fit-note';note.textContent=c.fit&&c.fit.kind==='nd_spiral_3d'?'Fill: predicted probability · outline: true class · faint: continuation · drag to rotate':'target/reference in gray where applicable';box.append(note);fits.append(box)}card.append(fits);return card}
function renderTasks(){const q=document.querySelector('#search').value.trim().toLowerCase(),root=document.querySelector('#tasks');root.innerHTML='';for(const panel of DATA.tasks){if(q&&!panel.task.includes(q))continue;const card=taskCard(panel);card.querySelectorAll('.fit-note').forEach((note,index)=>note.textContent=fitNote(panel.curves[index].fit));root.append(card)}}
document.querySelector('#legend').innerHTML=DATA.order.map(name=>`<span><i class="swatch" style="background:${COLORS[name]}"></i>${LABELS[name]}</span>`).join('');document.querySelector('#search').addEventListener('input',renderTasks);leaderboard();renderOperator();renderTasks();
</script></body></html>'''


if __name__ == "__main__":
    main()
