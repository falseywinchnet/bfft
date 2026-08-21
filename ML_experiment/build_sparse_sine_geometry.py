#!/usr/bin/env python3
"""Build the progressively thinned sine fitted-curve diagnosis."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def read(directory):
    return json.loads((ROOT / directory / "results.json").read_text())["runs"]


def pick(source, model, mode, seed=0):
    return next(
        row for row in source
        if row["model"] == model and row["mode"] == mode and row["seed"] == seed
    )


def compact(row, stride=3):
    probe = row["probe"]
    return {
        "x": probe["x"][::stride],
        "truth": probe["truth"][::stride],
        "prediction": probe["prediction"][::stride],
        "r2": row["r2"],
        "sparse": row["sparse_tail_r2"],
        "minimum": row["minimum_segment_r2"],
        "extrapolation": row["extrapolation_r2"],
        "segments": [segment["r2"] for segment in row["segment_metrics"]],
        "history": row["history"],
        "parameters": row["parameters"],
        "seconds": row["seconds"],
    }


def data():
    self_modes = read("results_sparse_sine_self_modes")
    operator = read("results_sparse_sine_operator_confirm")
    operator_best = read("results_sparse_sine_operator_best_curve")
    linear = read("results_sparse_sine_linear_control")
    cff = read("results_sparse_sine_cff")
    mlp = read("results_sparse_sine_mlp")
    selected = {
        "self_empirical": pick(self_modes, "self_context", "empirical"),
        "self_hermite": pick(self_modes, "self_context", "hermite"),
        "cff_hermite": pick(cff, "cff", "hermite"),
        "operator_empirical": pick(operator, "operator_sphere", "empirical"),
        "operator_linear": pick(linear, "operator_sphere", "linear"),
        "operator_hermite": pick(operator_best, "operator_sphere", "hermite"),
        "mlp_hermite": pick(mlp, "mlp_26k", "hermite"),
    }
    names = {
        "self_empirical": "Self-context · empirical pairs",
        "self_hermite": "Self-context · Hermite transport",
        "cff_hermite": "CFF · Hermite transport",
        "operator_empirical": "Operator sphere · empirical pairs",
        "operator_linear": "Operator sphere · linear intervals",
        "operator_hermite": "Operator sphere · Hermite intervals",
        "mlp_hermite": "Dense MLP · Hermite intervals",
    }
    return {
        "counts": [768, 768, 512, 384, 256, 128, 64, 32, 16, 8],
        "names": names,
        "models": {name: compact(row) for name, row in selected.items()},
    }


HTML = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Sparse sine · geometry acquisition</title><style>
:root{color-scheme:light dark;--bg:#f5f3ee;--surface:#fffefa;--ink:#1d2730;--muted:#69747c;--line:#d9d4c9;--red:#d45135;--blue:#356aa8;--green:#16806a;--violet:#7b58a5;--gold:#ad7c1c;--cyan:#17869f;--pink:#a84f79}
@media(prefers-color-scheme:dark){:root{--bg:#121619;--surface:#191f23;--ink:#edf1f3;--muted:#a9b2b8;--line:#354047;--red:#f07b61;--blue:#73a8ea;--green:#51c9a8;--violet:#b99ae2;--gold:#e5b95a;--cyan:#65bfd2;--pink:#df8bb3}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:32px 28px 80px}.eyebrow{text-transform:uppercase;letter-spacing:.14em;font-size:11px;color:var(--red);font-weight:750}h1{font-size:clamp(32px,4.2vw,60px);line-height:.98;letter-spacing:-.045em;max-width:1080px;margin:10px 0 14px}.dek{font-size:17px;line-height:1.55;color:var(--muted);max-width:1000px;margin:0 0 26px}.facts{display:grid;grid-template-columns:repeat(4,1fr);border:1px solid var(--line);background:var(--surface);margin-bottom:46px}.fact{padding:18px;border-right:1px solid var(--line)}.fact:last-child{border:0}.fact b{display:block;font-size:27px;letter-spacing:-.04em}.fact span{display:block;color:var(--muted);font-size:12px;margin-top:4px;line-height:1.35}section{margin-top:48px}h2{font-size:24px;letter-spacing:-.025em;margin:0 0 6px}.note{color:var(--muted);font-size:14px;margin:0 0 16px}.legend{display:flex;flex-wrap:wrap;gap:7px 16px;margin:10px 0 14px}.legend button{border:0;background:transparent;color:var(--ink);padding:2px 0;font:inherit;font-size:12px;cursor:pointer;opacity:.38}.legend button[aria-pressed=true]{opacity:1}.swatch{display:inline-block;width:16px;height:3px;margin-right:6px;vertical-align:middle}.chart{width:100%;height:auto;background:var(--surface);border:1px solid var(--line)}.main-chart{height:430px}.small-chart{height:330px}.grid{stroke:var(--line);stroke-width:.7}.tick,.axis-title,.label{fill:var(--muted);font-size:11px}.series{fill:none;stroke-width:1.65}.truth{stroke:var(--ink);stroke-width:2}.boundary{stroke:var(--red);stroke-width:1.5;stroke-dasharray:5 4}.two{display:grid;grid-template-columns:1.2fr 1fr;gap:22px}.panel{min-width:0;border-top:1px solid var(--line);padding-top:13px}.panel h3{font-size:14px;margin:0 0 10px}.segments{display:grid;grid-template-columns:repeat(10,minmax(0,1fr));gap:4px}.segment{min-width:0}.segment .count{font-size:10px;color:var(--muted);height:28px;display:flex;align-items:end;justify-content:center}.segment .density{height:90px;border-bottom:1px solid var(--line);display:flex;align-items:end}.segment .density i{display:block;width:100%;background:var(--blue)}.segment label{display:block;text-align:center;font-size:10px;color:var(--muted);margin-top:5px}.heat{display:grid;grid-template-columns:minmax(175px,1.5fr) repeat(10,minmax(34px,1fr));gap:3px;align-items:center;font-size:11px}.heat .name{padding-right:8px}.cell{height:30px;display:flex;align-items:center;justify-content:center;font-variant-numeric:tabular-nums;background:color-mix(in srgb,var(--red) calc((1 - var(--v))*55%),var(--green) calc(var(--v)*55%));color:var(--ink)}.head{text-align:center;color:var(--muted);font-size:10px}.finding{border-left:3px solid var(--red);padding:3px 0 3px 14px;line-height:1.5;margin:14px 0}.finding strong{color:var(--red)}footer{border-top:1px solid var(--line);margin-top:54px;padding-top:15px;color:var(--muted);font-size:12px}
@media(max-width:900px){.facts{grid-template-columns:1fr 1fr}.fact:nth-child(2){border-right:0}.fact:nth-child(-n+2){border-bottom:1px solid var(--line)}.two{grid-template-columns:1fr}.heat{grid-template-columns:minmax(140px,1.5fr) repeat(10,minmax(28px,1fr));font-size:10px}.main-chart{height:360px}}
@media(max-width:560px){main{padding:24px 14px 60px}.facts{grid-template-columns:1fr}.fact{border-right:0;border-bottom:1px solid var(--line)!important}.main-chart{height:310px}.heat{grid-template-columns:120px repeat(10,32px);overflow-x:auto}.segments{grid-template-columns:repeat(5,1fr)}}
</style></head><body><main><div class="eyebrow">Sparse observation experiment · M4 CPU</div><h1>The missing tail is first a measure problem, then a geometry problem.</h1><p class="dek">The observed function is one ordinary sine wave repeated for ten periods. Sampling falls from 768 points per period to eight. Empirical training learns the dense head and abandons the observed tail. Local interval transport restores equal geometric support; curvature-aware Hermite transport plus the operator sphere reconstructs every observed period.</p><div class="facts"><div class="fact"><b>96×</b><span>density drop within the observed interval</span></div><div class="fact"><b>0.967</b><span>uniform observed R² · operator sphere + Hermite</span></div><div class="fact"><b>0.942</b><span>worst observed-period R², including eight-point period</span></div><div class="fact"><b>still no</b><span>genuine extrapolation beyond the final observation</span></div></div>
<section><h2>Fitted function: observed support and genuine continuation</h2><p class="note">The dashed boundary at x = 1 separates progressively thinned observations from five completely unseen periods.</p><div class="legend" id="legend"></div><svg id="mainPlot" class="chart main-chart" role="img" aria-label="Fitted sine curves over observed and extrapolation intervals"></svg></section>
<section><h2>The observation measure collapses while the geometry does not</h2><p class="note">Counts are per complete observed period. Evaluation gives every period equal coordinate measure.</p><div class="segments" id="density"></div></section>
<section><h2>Period-by-period retention</h2><p class="note">R² inside each observed period. Red cells are geometry the model failed to retain.</p><div class="heat" id="heat"></div></section>
<section><h2>Acquisition behavior</h2><div class="two"><div class="panel"><h3>Worst observed-period R² during training</h3><svg id="history" class="chart small-chart" role="img" aria-label="Minimum segment R squared over optimizer steps"></svg></div><div><div class="finding"><strong>Empirical pairs:</strong> even the operator sphere fits the dense head and gives approximately zero R² in the sparse periods.</div><div class="finding"><strong>Linear interval filling:</strong> reaches 0.712 worst-period R². Correcting density alone helps, but it blurs curvature.</div><div class="finding"><strong>Hermite interval transport:</strong> reaches 0.942 worst-period R² without sine or frequency features. The interval carries value and locally estimated tangent.</div><div class="finding"><strong>Extrapolation remains separate:</strong> all generic pointwise models leave the learned chart once x exceeds the final observation. The red-tail failure is now isolated from sparse acquisition.</div></div></div></section>
<footer>Ten observed periods · segment counts 768, 768, 512, 384, 256, 128, 64, 32, 16, 8 · five fully unseen periods · LELU models · no Fourier or sinusoidal features.</footer>
</main><script>
const data=__DATA__;const keys=['self_empirical','self_hermite','cff_hermite','operator_empirical','operator_linear','operator_hermite','mlp_hermite'];const colors=['var(--blue)','var(--violet)','var(--cyan)','var(--red)','var(--gold)','var(--green)','var(--pink)'];const shown=new Set(['self_empirical','operator_empirical','operator_linear','operator_hermite']);const ns='http://www.w3.org/2000/svg';const el=(tag,a={})=>{const n=document.createElementNS(ns,tag);Object.entries(a).forEach(([k,v])=>n.setAttribute(k,v));return n};const path=(xs,ys,x,y)=>ys.map((v,i)=>(i?'L':'M')+x(xs[i])+','+y(v)).join(' ');const legend=document.getElementById('legend');keys.forEach((k,i)=>{const b=document.createElement('button');b.type='button';b.setAttribute('aria-pressed',shown.has(k));b.innerHTML=`<span class="swatch" style="background:${colors[i]}"></span>${data.names[k]}`;b.onclick=()=>{shown.has(k)?shown.delete(k):shown.add(k);b.setAttribute('aria-pressed',shown.has(k));drawMain()};legend.append(b)});
function axes(svg,{left,right,top,bottom,xScale,yScale,xTicks,yTicks,xLabel,yLabel}){xTicks.forEach(v=>{const x=xScale(v);svg.append(el('line',{x1:x,x2:x,y1:top,y2:bottom,class:'grid'}));const t=el('text',{x,y:bottom+18,class:'tick','text-anchor':'middle'});t.textContent=v;svg.append(t)});yTicks.forEach(v=>{const y=yScale(v);svg.append(el('line',{x1:left,x2:right,y1:y,y2:y,class:'grid'}));const t=el('text',{x:left-8,y:y+4,class:'tick','text-anchor':'end'});t.textContent=v;svg.append(t)});const xt=el('text',{x:(left+right)/2,y:bottom+40,class:'axis-title','text-anchor':'middle'});xt.textContent=xLabel;svg.append(xt);const yt=el('text',{x:15,y:(top+bottom)/2,class:'axis-title',transform:`rotate(-90 15 ${(top+bottom)/2})`,'text-anchor':'middle'});yt.textContent=yLabel;svg.append(yt)}
function drawMain(){const svg=document.getElementById('mainPlot');svg.innerHTML='';svg.setAttribute('viewBox','0 0 1000 430');const left=60,right=982,top=22,bottom=375,x=v=>left+v/1.5*(right-left),y=v=>bottom-(v+1.35)/2.7*(bottom-top);axes(svg,{left,right,top,bottom,xScale:x,yScale:y,xTicks:[0,.25,.5,.75,1,1.25,1.5],yTicks:[-1,0,1],xLabel:'normalized coordinate x',yLabel:'physical output'});const ref=data.models.operator_hermite;svg.append(el('path',{d:path(ref.x,ref.truth,x,y),class:'series truth'}));keys.forEach((k,i)=>{if(shown.has(k))svg.append(el('path',{d:path(data.models[k].x,data.models[k].prediction,x,y),class:'series',stroke:colors[i]}))});svg.append(el('line',{x1:x(1),x2:x(1),y1:top,y2:bottom,class:'boundary'}));const a=el('text',{x:x(1)-6,y:top+15,class:'label','text-anchor':'end'});a.textContent='observed';svg.append(a);const b=el('text',{x:x(1)+6,y:top+15,class:'label'});b.textContent='unseen';svg.append(b)}drawMain();
const density=document.getElementById('density'),maximum=Math.max(...data.counts);data.counts.forEach((count,i)=>{const d=document.createElement('div');d.className='segment';d.innerHTML=`<div class="count">${count}</div><div class="density"><i style="height:${100*Math.log1p(count)/Math.log1p(maximum)}%"></i></div><label>period ${i+1}</label>`;density.append(d)});
const heatKeys=['self_empirical','self_hermite','cff_hermite','operator_empirical','operator_linear','operator_hermite','mlp_hermite'],heat=document.getElementById('heat');heat.append(document.createElement('div'));for(let i=0;i<10;i++){const h=document.createElement('div');h.className='head';h.textContent=i+1;heat.append(h)}heatKeys.forEach(k=>{const n=document.createElement('div');n.className='name';n.textContent=data.names[k];heat.append(n);data.models[k].segments.forEach(v=>{const c=document.createElement('div');c.className='cell';c.style.setProperty('--v',Math.max(0,Math.min(1,v)));c.textContent=v.toFixed(2);heat.append(c)})});
function drawHistory(){const svg=document.getElementById('history');svg.setAttribute('viewBox','0 0 680 330');const left=56,right=665,top=20,bottom=280,x=v=>left+v/1000*(right-left),y=v=>bottom-(Math.max(-.3,Math.min(1,v))+.3)/1.3*(bottom-top);axes(svg,{left,right,top,bottom,xScale:x,yScale:y,xTicks:[0,250,500,750,1000],yTicks:[0,.5,1],xLabel:'optimizer step',yLabel:'worst observed-period R²'});['self_empirical','self_hermite','operator_empirical','operator_linear','operator_hermite'].forEach((k,i)=>{const h=data.models[k].history;svg.append(el('path',{d:h.map((p,j)=>(j?'L':'M')+x(p.step)+','+y(p.minimum_segment_r2)).join(' '),class:'series',stroke:colors[keys.indexOf(k)]}))})}drawHistory();
</script></body></html>'''


def main():
    output = ROOT / "sparse_sine_geometry.html"
    output.write_text(HTML.replace("__DATA__", json.dumps(data(), separators=(",", ":"))))
    print(f"{output} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
