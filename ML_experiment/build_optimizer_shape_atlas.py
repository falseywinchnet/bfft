"""Build a standalone all-problem curve atlas for the six-arm battery."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


ORDER = (
    "adamw", "sgd", "anchor", "anchor_restrained_momentum", "muon",
    "lepton_transport",
)


def _aggregate_neural(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["task"], row["variant"], row["optimizer"]].append(row)
    panels = defaultdict(list)
    for (task, variant, optimizer), optimizer_rows in grouped.items():
        by_step = defaultdict(list)
        for row in optimizer_rows:
            for point in row["history"]:
                score = float(point["score"])
                loss = float(point["loss"])
                if math.isfinite(score) and math.isfinite(loss):
                    by_step[int(point["step"])].append((score, loss))
        points = []
        for step, values in sorted(by_step.items()):
            scores = [value[0] for value in values]
            losses = [value[1] for value in values]
            points.append({
                "step": step,
                "score": statistics.fmean(scores),
                "scoreMin": min(scores),
                "scoreMax": max(scores),
                "loss": statistics.fmean(losses),
                "lossMin": min(losses),
                "lossMax": max(losses),
            })
        panels[task, variant].append({
            "optimizer": optimizer,
            "auc": statistics.fmean(float(row["learning_auc"]) for row in optimizer_rows),
            "points": points,
        })
    return [
        {
            "task": task,
            "variant": variant,
            "curves": sorted(curves, key=lambda row: ORDER.index(row["optimizer"])),
        }
        for (task, variant), curves in sorted(panels.items())
    ]


def _aggregate_operator(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["protocol"], row["scenario"], row["optimizer"]].append(row)
    panels = defaultdict(list)
    for (protocol, scenario, optimizer), optimizer_rows in grouped.items():
        by_step = defaultdict(list)
        for row in optimizer_rows:
            for point in row["history"]:
                by_step[int(point["step"])].append((
                    float(point["relative_loss"]),
                    float(point["operator_error"]),
                ))
        points = [
            {
                "step": step,
                "loss": statistics.fmean(value[0] for value in values),
                "error": statistics.fmean(value[1] for value in values),
            }
            for step, values in sorted(by_step.items())
        ]
        panels[protocol, scenario].append({"optimizer": optimizer, "points": points})
    return [
        {
            "protocol": protocol,
            "scenario": scenario,
            "curves": sorted(curves, key=lambda row: ORDER.index(row["optimizer"])),
        }
        for (protocol, scenario), curves in sorted(panels.items())
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("neural", type=Path)
    parser.add_argument("operator", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    neural = json.loads(args.neural.read_text())
    operator = json.loads(args.operator.read_text())
    evidence = {
        "order": list(ORDER),
        "neural": _aggregate_neural(neural["runs"]),
        "operator": _aggregate_operator(operator["runs"]),
    }
    packed = json.dumps(evidence, separators=(",", ":"))
    html = TEMPLATE.replace("__EVIDENCE__", packed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html)


TEMPLATE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Optimizer geometry battery atlas</title>
<style>
:root{color-scheme:light dark;--bg:#f7f5ef;--fg:#181816;--muted:#69675f;--panel:#fff;--line:#d8d4c9}
@media(prefers-color-scheme:dark){:root{--bg:#131411;--fg:#f2efe5;--muted:#aaa79d;--panel:#1c1e1a;--line:#3b3e36}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 ui-sans-serif,system-ui,sans-serif}
header{padding:28px max(22px,4vw) 18px;border-bottom:1px solid var(--line)}h1{font-size:30px;margin:0 0 8px}p{max-width:950px;margin:6px 0;color:var(--muted)}
.controls{position:sticky;top:0;z-index:4;background:color-mix(in srgb,var(--bg) 92%,transparent);backdrop-filter:blur(8px);padding:12px max(22px,4vw);display:flex;gap:18px;flex-wrap:wrap;border-bottom:1px solid var(--line)}
button,input,select{font:inherit}.legend{display:flex;gap:12px;flex-wrap:wrap}.sw{display:inline-block;width:18px;height:3px;margin-right:5px;vertical-align:middle}
main{padding:20px max(18px,3vw) 50px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(390px,1fr));gap:14px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:13px;min-width:0}
.panel h2{font-size:15px;margin:0}.sub{font-size:12px;color:var(--muted);margin:2px 0 8px}.winner{font-weight:700}svg{width:100%;height:auto;display:block}.axis{stroke:var(--line);stroke-width:1}.tick{fill:var(--muted);font-size:10px}.curve{fill:none;stroke-width:2}.band{opacity:.08}.section{font-size:22px;margin:30px 0 12px}
@media(max-width:520px){.grid{grid-template-columns:1fr}.panel{padding:9px}}
</style></head><body>
<header><h1>Optimizer geometry battery</h1><p>Every panel contains the same paired initialization and minibatch order for AdamW, SGD, Anchor, restrained-momentum Anchor, Muon, and Transport-Lepton. Lines are three-seed means; translucent bands are the full seed range. Use training loss when a validation score conceals a recognizable failure mode.</p></header>
<div class="controls"><label>Metric <select id="metric"><option value="score">validation score</option><option value="loss">training loss (log)</option></select></label><label>Architecture <select id="variant"><option value="all">both</option><option value="ordinary_mlp">ordinary MLP</option><option value="self_context">self-context</option></select></label><label>Find task <input id="search" placeholder="pinwheel"></label><div class="legend" id="legend"></div></div>
<main><h2 class="section">All 48 neural problem × architecture curves</h2><div id="neural" class="grid"></div><h2 class="section">Anisotropic operator-recovery witness</h2><p>Standing uses the neural-battery learning rates. Operator-native exposes unit matrix steps and separately declared comparator rates. Dashed vertical position is the four-update complete block sweep.</p><div id="operator" class="grid"></div></main>
<script>
const DATA=__EVIDENCE__;
const COLORS={adamw:'#6c5ce7',sgd:'#ff5a5f',anchor:'#f39c12',anchor_restrained_momentum:'#c46f00',muon:'#087e8b',lepton_transport:'#00a878'};
const LABELS={adamw:'AdamW',sgd:'SGD',anchor:'Anchor',anchor_restrained_momentum:'Restrained-momentum Anchor',muon:'Muon',lepton_transport:'Transport-Lepton'};
const NS='http://www.w3.org/2000/svg';
document.querySelector('#legend').innerHTML=DATA.order.map(n=>`<span><i class="sw" style="background:${COLORS[n]}"></i>${LABELS[n]}</span>`).join('');
function el(tag,attrs={}){const n=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,v);return n}
function path(points,x,y,key){return points.map((p,i)=>`${i?'L':'M'}${x(p.step).toFixed(2)},${y(p[key]).toFixed(2)}`).join('')}
function chart(curves,metric,bands=true){const W=430,H=205,m={l:45,r:10,t:8,b:27};const svg=el('svg',{viewBox:`0 0 ${W} ${H}`});const all=curves.flatMap(c=>c.points);const xmax=Math.max(...all.map(p=>p.step));let vals=all.flatMap(p=>bands?[p[metric+'Min'],p[metric+'Max']]:[p[metric]]).filter(Number.isFinite);const log=metric==='loss'||metric==='error';if(log)vals=vals.map(v=>Math.log10(Math.max(v,1e-30)));let lo=Math.min(...vals),hi=Math.max(...vals);if(!log){const pad=Math.max((hi-lo)*.08,.005);lo-=pad;hi+=pad}else{lo-=.15;hi+=.15}const x=v=>m.l+(W-m.l-m.r)*v/xmax;const y=v=>{v=log?Math.log10(Math.max(v,1e-30)):v;return m.t+(H-m.t-m.b)*(hi-v)/(hi-lo||1)};
svg.append(el('line',{x1:m.l,y1:H-m.b,x2:W-m.r,y2:H-m.b,class:'axis'}));svg.append(el('line',{x1:m.l,y1:m.t,x2:m.l,y2:H-m.b,class:'axis'}));
for(const t of [0,.5,1]){const xv=t*xmax,tx=el('text',{x:x(xv),y:H-8,'text-anchor':'middle',class:'tick'});tx.textContent=Math.round(xv);svg.append(tx);const vv=lo+t*(hi-lo),ty=el('text',{x:m.l-6,y:m.t+(1-t)*(H-m.t-m.b)+3,'text-anchor':'end',class:'tick'});ty.textContent=log?`1e${vv.toFixed(0)}`:vv.toFixed(2);svg.append(ty)}
for(const c of curves){if(bands){const top=c.points.map(p=>`${x(p.step)},${y(p[metric+'Max'])}`);const bot=[...c.points].reverse().map(p=>`${x(p.step)},${y(p[metric+'Min'])}`);svg.append(el('polygon',{points:[...top,...bot].join(' '),fill:COLORS[c.optimizer],class:'band'}))}svg.append(el('path',{d:path(c.points,x,y,metric),stroke:COLORS[c.optimizer],class:'curve'}))}return svg}
function render(){const metric=document.querySelector('#metric').value,variant=document.querySelector('#variant').value,q=document.querySelector('#search').value.toLowerCase();const root=document.querySelector('#neural');root.innerHTML='';for(const p of DATA.neural){if(variant!=='all'&&p.variant!==variant||q&&!p.task.includes(q))continue;const winner=[...p.curves].sort((a,b)=>b.auc-a.auc)[0];const card=document.createElement('article');card.className='panel';card.innerHTML=`<h2>${p.task.replaceAll('_',' ')}</h2><div class="sub">${p.variant.replaceAll('_',' ')} · AUC winner: <span class="winner" style="color:${COLORS[winner.optimizer]}">${LABELS[winner.optimizer]} ${winner.auc.toFixed(4)}</span></div>`;card.append(chart(p.curves,metric,true));root.append(card)}}
function renderOperator(){const root=document.querySelector('#operator');for(const p of DATA.operator){for(const metric of ['loss','error']){const card=document.createElement('article');card.className='panel';card.innerHTML=`<h2>${p.protocol.replaceAll('_',' ')} · ${p.scenario.replaceAll('_',' ')}</h2><div class="sub">${metric==='loss'?'relative objective':'relative operator error'} · log scale</div>`;card.append(chart(p.curves,metric,false));root.append(card)}}}
for(const id of ['metric','variant','search'])document.querySelector('#'+id).addEventListener(id==='search'?'input':'change',render);render();renderOperator();
</script></body></html>'''


if __name__ == "__main__":
    main()
