"""Build the self-context optimizer research atlas with fitted-function views."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


NEURAL_ORDER = (
    "adamw", "adamw_coherence", "adamw_turn", "anchor",
    "lepton_transport_guarded",
)
OPERATOR_ORDER = (*NEURAL_ORDER, "muon")


def aggregate_neural(rows):
    grouped = defaultdict(list)
    fits = {}
    for row in rows:
        key = row["task"], row["optimizer"]
        grouped[key].append(row)
        if row.get("fit_visualization") is not None:
            fits[key] = row["fit_visualization"]
    panels = defaultdict(list)
    for (task, optimizer), optimizer_rows in grouped.items():
        by_step = defaultdict(list)
        for row in optimizer_rows:
            for point in row["history"]:
                score, loss = float(point["score"]), float(point["loss"])
                if math.isfinite(score) and math.isfinite(loss):
                    by_step[int(point["step"])].append((score, loss))
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
        panels[task].append({
            "optimizer": optimizer,
            "auc": statistics.fmean(float(row["learning_auc"]) for row in optimizer_rows),
            "validation": statistics.fmean(float(row["validation_score"]) for row in optimizer_rows),
            "test": statistics.fmean(float(row["score"]) for row in optimizer_rows),
            "points": points,
            "fit": fits.get((task, optimizer)),
        })
    return [
        {"task": task, "curves": sorted(curves, key=lambda x: NEURAL_ORDER.index(x["optimizer"]))}
        for task, curves in sorted(panels.items())
    ]


def aggregate_operator(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["scenario"], row["optimizer"]].append(row)
    panels = defaultdict(list)
    for (scenario, optimizer), optimizer_rows in grouped.items():
        by_step = defaultdict(list)
        for row in optimizer_rows:
            for point in row["history"]:
                by_step[int(point["step"])].append((
                    float(point["relative_loss"]), float(point["operator_error"])
                ))
        points = [{
            "step": step,
            "loss": statistics.fmean(value[0] for value in values),
            "error": statistics.fmean(value[1] for value in values),
        } for step, values in sorted(by_step.items())]
        panels[scenario].append({"optimizer": optimizer, "points": points})
    return [{
        "scenario": scenario,
        "curves": sorted(curves, key=lambda x: OPERATOR_ORDER.index(x["optimizer"])),
    } for scenario, curves in sorted(panels.items())]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("neural", type=Path)
    parser.add_argument("operator", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    neural = json.loads(args.neural.read_text())
    operator = json.loads(args.operator.read_text())
    evidence = {
        "neuralOrder": list(NEURAL_ORDER),
        "operatorOrder": list(OPERATOR_ORDER),
        "neural": aggregate_neural(neural["runs"]),
        "operator": aggregate_operator(operator["runs"]),
    }
    document = TEMPLATE.replace("__EVIDENCE__", json.dumps(evidence, separators=(",", ":")))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(document)
    print(args.out)


TEMPLATE = r'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Self-context optimizer atlas</title><style>
:root{color-scheme:light dark;--bg:#f4f1e8;--fg:#171713;--muted:#69675e;--panel:#fff;--line:#d7d1c3;--plot:#fbfaf6}
@media(prefers-color-scheme:dark){:root{--bg:#11130f;--fg:#f2efe4;--muted:#aaa79d;--panel:#1b1e18;--line:#3a3e34;--plot:#151711}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 ui-sans-serif,system-ui,sans-serif}
header{padding:34px max(22px,4vw) 22px;border-bottom:1px solid var(--line)}h1{font-size:34px;line-height:1.08;margin:0 0 10px}header p{max-width:1050px;color:var(--muted);font-size:16px}
.controls{position:sticky;top:0;z-index:5;padding:11px max(22px,4vw);display:flex;gap:18px;align-items:center;flex-wrap:wrap;background:color-mix(in srgb,var(--bg) 93%,transparent);backdrop-filter:blur(9px);border-bottom:1px solid var(--line)}
select,input{font:inherit;padding:4px 7px}.legend{display:flex;gap:11px;flex-wrap:wrap}.sw{display:inline-block;width:16px;height:3px;margin-right:5px;vertical-align:middle}
main{padding:18px max(16px,3vw) 60px}.section{font-size:24px;margin:30px 0 6px}.section-note{color:var(--muted);max-width:1000px;margin:0 0 15px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(410px,1fr));gap:14px}.panel{min-width:0;background:var(--panel);border:1px solid var(--line);border-radius:11px;padding:14px}.panel h2{font-size:18px;margin:0}.sub{font-size:12px;color:var(--muted);margin:3px 0 8px}.winner{font-weight:750}
svg.curves{width:100%;height:auto;display:block}.axis{stroke:var(--line);stroke-width:1}.tick{fill:var(--muted);font-size:10px}.curve{fill:none;stroke-width:2}.band{opacity:.08}
details{border-top:1px solid var(--line);margin-top:10px;padding-top:8px}summary{cursor:pointer;color:var(--muted);font-size:12px}.fitgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:8px;margin-top:9px}.fit{min-width:0}.fit h3{font-size:11px;margin:0 0 3px}.fit canvas,.fit svg{width:100%;height:auto;display:block;background:var(--plot);border:1px solid var(--line)}
.operator{display:grid;grid-template-columns:repeat(auto-fit,minmax(430px,1fr));gap:14px}.operator .panel{padding-bottom:16px}
@media(max-width:520px){.grid,.operator{grid-template-columns:1fr}.panel{padding:10px}.fitgrid{grid-template-columns:1fr 1fr}}
</style></head><body><header><h1>Self-context optimizer atlas</h1>
<p>Paired initialization and minibatch order across AdamW, Coherence-Adam, Turn-Adam, Anchor, and guarded Transport-Lepton. Curves are three-seed means with full seed ranges. Every fitted-function panel is the best validation checkpoint from the paired seed-0 run. Muon is restricted to the anisotropic operator witness.</p></header>
<div class="controls"><label>Curve <select id="metric"><option value="score">validation score</option><option value="loss">training loss (log)</option></select></label><label>Find task <input id="search" placeholder="ripple"></label><div id="legend" class="legend"></div></div>
<main><h2 class="section">Self-context neural battery</h2><p class="section-note">Open a card's fitted-function section to inspect the learned surface, curve, or projected manifold. For N-D spirals, fill is predicted class probability; marker outline is the true class and faint points are unseen continuation.</p><div id="neural" class="grid"></div>
<h2 class="section">Anisotropic operator-recovery witness</h2><p class="section-note">Population keeps the full rotated covariance visible. Block sweep exposes one rank-deficient support at a time. Muon appears only here.</p><div id="operator" class="operator"></div></main>
<script>const DATA=__EVIDENCE__;
const COLORS={adamw:'#6956e8',adamw_coherence:'#d7426a',adamw_turn:'#e88916',anchor:'#8d5a1f',lepton_transport_guarded:'#00a878',muon:'#087e8b'};
const LABELS={adamw:'AdamW',adamw_coherence:'Coherence-Adam',adamw_turn:'Turn-Adam',anchor:'Anchor',lepton_transport_guarded:'Guarded Transport-Lepton',muon:'Muon'};
const NS='http://www.w3.org/2000/svg';
document.querySelector('#legend').innerHTML=DATA.operatorOrder.map(n=>`<span><i class="sw" style="background:${COLORS[n]}"></i>${LABELS[n]}</span>`).join('');
function el(tag,a={}){const n=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(a))n.setAttribute(k,v);return n}
function curvePath(points,x,y,key){return points.map((p,i)=>`${i?'L':'M'}${x(p.step).toFixed(2)},${y(p[key]).toFixed(2)}`).join('')}
function curveChart(curves,metric,bands=true){const W=460,H=220,m={l:50,r:10,t:8,b:28},s=el('svg',{viewBox:`0 0 ${W} ${H}`,class:'curves'}),all=curves.flatMap(c=>c.points),xmax=Math.max(...all.map(p=>p.step)),log=metric!=='score';let vals=all.flatMap(p=>bands?[p[metric+'Min'],p[metric+'Max']]:[p[metric]]).filter(Number.isFinite);if(log)vals=vals.map(v=>Math.log10(Math.max(v,1e-30)));let lo=Math.min(...vals),hi=Math.max(...vals);const pad=Math.max((hi-lo)*.07,.003);lo-=pad;hi+=pad;const x=v=>m.l+(W-m.l-m.r)*v/xmax,y=v=>{v=log?Math.log10(Math.max(v,1e-30)):v;return m.t+(H-m.t-m.b)*(hi-v)/(hi-lo||1)};s.append(el('line',{x1:m.l,y1:H-m.b,x2:W-m.r,y2:H-m.b,class:'axis'}),el('line',{x1:m.l,y1:m.t,x2:m.l,y2:H-m.b,class:'axis'}));for(const t of [0,.5,1]){let tx=el('text',{x:x(t*xmax),y:H-8,'text-anchor':'middle',class:'tick'});tx.textContent=Math.round(t*xmax);s.append(tx);let vv=lo+t*(hi-lo),ty=el('text',{x:m.l-6,y:m.t+(1-t)*(H-m.t-m.b)+3,'text-anchor':'end',class:'tick'});ty.textContent=log?`1e${vv.toFixed(0)}`:vv.toFixed(2);s.append(ty)}for(const c of curves){if(bands){const top=c.points.map(p=>`${x(p.step)},${y(p[metric+'Max'])}`),bot=[...c.points].reverse().map(p=>`${x(p.step)},${y(p[metric+'Min'])}`);s.append(el('polygon',{points:[...top,...bot].join(' '),fill:COLORS[c.optimizer],class:'band'}))}s.append(el('path',{d:curvePath(c.points,x,y,metric),stroke:COLORS[c.optimizer],class:'curve'}))}return s}
function colorValue(v){v=Math.max(0,Math.min(1,v));const a=[40,47,135],b=[34,168,132],c=[252,231,37],t=v<.5?v*2:(v-.5)*2,p=v<.5?a:b,q=v<.5?b:c;return `rgb(${p.map((x,i)=>Math.round(x+(q[i]-x)*t)).join(',')})`}
function categorical(v,confidence=1){const palette=['#3566c4','#dd493f','#31a36c','#9b56b2','#e89422','#22a5ad'];const h=palette[v%palette.length];return h}
function surfaceCanvas(fit){const c=document.createElement('canvas');c.width=220;c.height=170;const g=c.getContext('2d'),side=fit.side,cell=c.width/side;for(let i=0;i<fit.prediction.length;i++){g.fillStyle=fit.kind==='surface_2d'&&fit.confidence?categorical(fit.prediction[i],fit.confidence[i]):colorValue((fit.prediction[i]+2)/4);g.fillRect((i%side)*cell,Math.floor(i/side)*c.height/side,cell+1,c.height/side+1)}if(fit.sample_x){const [xmin,xmax,ymin,ymax]=fit.limits;for(let i=0;i<fit.sample_x.length;i++){const p=fit.sample_x[i];g.fillStyle=categorical(fit.sample_y[i]);g.globalAlpha=.65;g.beginPath();g.arc((p[0]-xmin)/(xmax-xmin)*c.width,c.height-(p[1]-ymin)/(ymax-ymin)*c.height,1.6,0,Math.PI*2);g.fill()}g.globalAlpha=1}return c}
function lineSvg(fit,is3=false){const W=220,H=170,m=12,s=el('svg',{viewBox:`0 0 ${W} ${H}`}),target=fit.target,pred=fit.prediction;if(is3){const project=p=>[W/2+38*p[0]+20*p[2],H/2-38*p[1]+12*p[2]],path=a=>a.map((p,i)=>`${i?'L':'M'}${project(p).join(',')}`).join('');s.append(el('path',{d:path(target),fill:'none',stroke:'#8b8b84','stroke-width':2}),el('path',{d:path(pred),fill:'none',stroke:'#d7426a','stroke-width':1.6}));return s}const x=fit.coordinate,all=[...target,...pred],xmin=Math.min(...x),xmax=Math.max(...x),ymin=Math.min(...all),ymax=Math.max(...all),xp=v=>m+(W-2*m)*(v-xmin)/(xmax-xmin||1),yp=v=>H-m-(H-2*m)*(v-ymin)/(ymax-ymin||1),path=a=>a.map((v,i)=>`${i?'L':'M'}${xp(x[i])},${yp(v)}`).join('');s.append(el('path',{d:path(target),fill:'none',stroke:'#8b8b84','stroke-width':2}),el('path',{d:path(pred),fill:'none',stroke:'#d7426a','stroke-width':1.6}));return s}
function scatterSvg(fit,nd=false){const W=220,H=170,m=8,s=el('svg',{viewBox:`0 0 ${W} ${H}`}),p=fit.coordinates,xv=p.map(q=>q[0]),yv=p.map(q=>q[1]),zv=p.map(q=>q[2]||0),xmin=Math.min(...xv),xmax=Math.max(...xv),ymin=Math.min(...yv),ymax=Math.max(...yv),zmin=Math.min(...zv),zmax=Math.max(...zv);for(let i=0;i<p.length;i++){const x=m+(W-2*m)*(xv[i]-xmin)/(xmax-xmin||1)+8*(zv[i]-zmin)/(zmax-zmin||1),y=H-m-(H-2*m)*(yv[i]-ymin)/(ymax-ymin||1)-5*(zv[i]-zmin)/(zmax-zmin||1),pred=nd?fit.probability[i]:fit.prediction[i],fill=nd?colorValue(pred):categorical(Array.isArray(pred)?0:pred),stroke=nd?categorical(fit.label[i]):categorical(Array.isArray(fit.target[i])?0:fit.target[i]);s.append(el('circle',{cx:x,cy:y,r:nd?1.8:2,fill,stroke,'stroke-width':.5,opacity:nd&&fit.region[i]?'0.28':'0.82'}))}return s}
function fitView(fit){if(!fit)return document.createTextNode('No fit payload');if(fit.kind==='surface_2d')return surfaceCanvas(fit);if(fit.kind==='curve_1d')return lineSvg(fit);if(fit.kind==='curve_3d')return lineSvg(fit,true);if(fit.kind==='nd_spiral_3d')return scatterSvg(fit,true);return scatterSvg(fit,false)}
function fitGallery(curves){const grid=document.createElement('div');grid.className='fitgrid';for(const c of curves){const d=document.createElement('div');d.className='fit';d.innerHTML=`<h3 style="color:${COLORS[c.optimizer]}">${LABELS[c.optimizer]} · seed 0 best</h3>`;d.append(fitView(c.fit));grid.append(d)}return grid}
function render(){const metric=document.querySelector('#metric').value,q=document.querySelector('#search').value.toLowerCase(),root=document.querySelector('#neural');root.innerHTML='';for(const p of DATA.neural){if(q&&!p.task.includes(q))continue;const winner=[...p.curves].sort((a,b)=>b.auc-a.auc)[0],card=document.createElement('article');card.className='panel';card.innerHTML=`<h2>${p.task.replaceAll('_',' ')}</h2><div class="sub">AUC winner: <span class="winner" style="color:${COLORS[winner.optimizer]}">${LABELS[winner.optimizer]} ${winner.auc.toFixed(4)}</span> · best validation ${winner.validation.toFixed(4)} · held-out ${winner.test.toFixed(4)}</div>`;card.append(curveChart(p.curves,metric,true));const details=document.createElement('details');details.innerHTML='<summary>Best-checkpoint fitted function · paired seed 0</summary>';details.append(fitGallery(p.curves));card.append(details);root.append(card)}}
function renderOperator(){const root=document.querySelector('#operator');for(const p of DATA.operator){for(const metric of ['loss','error']){const card=document.createElement('article');card.className='panel';card.innerHTML=`<h2>${p.scenario.replaceAll('_',' ')}</h2><div class="sub">${metric==='loss'?'relative objective':'relative operator error'} · log scale</div>`;card.append(curveChart(p.curves,metric,false));root.append(card)}}}
document.querySelector('#metric').addEventListener('change',render);document.querySelector('#search').addEventListener('input',render);render();renderOperator();
</script></body></html>'''


if __name__ == "__main__":
    main()
