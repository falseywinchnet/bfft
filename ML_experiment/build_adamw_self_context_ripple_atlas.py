#!/usr/bin/env python3
"""Build a self-contained evidence atlas for the paired Ripple experiment."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results_adamw_self_context_ripple" / "results.json"
OUTPUT = ROOT / "adamw_self_context_ripple.html"


HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AdamW × self-context — Ripple</title>
<style>
:root{--ink:#17202a;--muted:#5e6b76;--paper:#f4f0e8;--panel:#fffdf8;--line:#d5cfc3;--blue:#1667a8;--orange:#c45620;--green:#1c7a5a;--red:#a93535}
*{box-sizing:border-box} body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.5 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif} main{max-width:1440px;margin:auto;padding:42px 34px 80px} h1{font:700 clamp(38px,6vw,82px)/.95 ui-serif,Georgia,serif;letter-spacing:-.045em;margin:10px 0 20px;max-width:1100px} h2{font:650 28px/1.1 ui-serif,Georgia,serif;margin:0 0 14px} h3{font-size:14px;text-transform:uppercase;letter-spacing:.11em;margin:0 0 10px}.kicker{color:var(--orange);font-weight:750;letter-spacing:.14em;text-transform:uppercase}.lede{font-size:20px;max-width:980px;color:#33404a}.verdict{border-left:5px solid var(--orange);padding:18px 22px;background:#fff8ee;max-width:1120px;font-size:18px}.grid{display:grid;gap:18px}.metrics{grid-template-columns:repeat(4,minmax(0,1fr));margin:28px 0}.metric,.panel{background:var(--panel);border:1px solid var(--line);box-shadow:0 8px 26px #3d322314}.metric{padding:18px}.metric strong{font:700 30px/1 ui-serif,Georgia,serif;display:block}.metric span{color:var(--muted)}.two{grid-template-columns:repeat(2,minmax(0,1fr));margin:18px 0}.three{grid-template-columns:repeat(3,minmax(0,1fr));margin:18px 0}.panel{padding:20px;min-width:0}.chart{width:100%;height:330px;display:block}.surface{width:100%;aspect-ratio:1;display:block;image-rendering:auto}.legend{display:flex;gap:20px;flex-wrap:wrap;color:var(--muted);margin:4px 0 12px}.swatch{display:inline-block;width:18px;height:3px;margin-right:7px;vertical-align:middle}.blue{background:var(--blue)}.orange{background:var(--orange)}.green{background:var(--green)}.red{background:var(--red)}.note{color:var(--muted);font-size:13px}.math{font:16px/1.7 ui-monospace,SFMono-Regular,Menlo,monospace;background:#f0ece3;border:1px solid var(--line);padding:15px 17px;overflow:auto;white-space:pre-wrap}.mechanism{grid-template-columns:1fr 1.15fr;margin:28px 0}.flow{display:grid;grid-template-columns:1fr auto 1fr auto 1fr;align-items:center;gap:10px}.node{padding:16px 12px;border:1px solid var(--line);background:#fff;text-align:center;min-height:104px;display:grid;place-content:center}.arrow{font-size:25px;color:var(--orange)} table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums} th,td{padding:9px 10px;border-bottom:1px solid var(--line);text-align:right} th:first-child,td:first-child{text-align:left} th{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.06em}.winner{font-weight:750;color:var(--blue)} .section{margin-top:50px}.callout{background:#eaf3f7;border:1px solid #b9d2df;padding:18px 20px}.smallcaps{text-transform:uppercase;letter-spacing:.09em;font-size:12px;font-weight:750}.footer{margin-top:55px;padding-top:18px;border-top:1px solid var(--line);color:var(--muted)}
@media(max-width:900px){main{padding:25px 16px 60px}.metrics,.two,.three,.mechanism{grid-template-columns:1fr}.flow{grid-template-columns:1fr}.arrow{transform:rotate(90deg);text-align:center}.chart{height:280px}}
</style>
</head>
<body><main>
<div class="kicker">Paired optimizer experiment · Ripple · M4</div>
<h1>Separating self-context from AdamW’s memory made it safer—and slower.</h1>
<p class="lede">This is AdamW against AdamW, not an optimizer zoo. Both arms see the same self-context model, initial weights, minibatches, learning rate, weight decay, and global clip. The only intervention is whether chart motion is allowed to write Adam’s persistent moments.</p>
<p class="verdict"><strong>Result:</strong> ordinary AdamW remains the winner. Source separation prevents context feedback from impersonating a persistent optimization direction, but on Ripple that feedback is useful enough that removing it from the moments costs <strong>0.0258 mean score-AUC</strong> and about <strong>40 iterations to 0.80</strong>. This is a clean negative result: the mechanism works as specified, but the specification over-restrains useful curvature information.</p>

<div class="grid metrics">
 <div class="metric"><span>Mean score-AUC</span><strong>0.7713</strong><span>AdamW · source-aware 0.7456</span></div>
 <div class="metric"><span>Median steps to 0.80</span><strong>345</strong><span>AdamW · source-aware 385</span></div>
 <div class="metric"><span>Mean best validation</span><strong>0.9536</strong><span>AdamW · source-aware 0.9302</span></div>
 <div class="metric"><span>Reached 0.95</span><strong>4 / 5</strong><span>AdamW · source-aware 1 / 5</span></div>
</div>

<section class="section"><h2>The curves that decide it</h2>
<div class="legend"><span><i class="swatch blue"></i>ordinary AdamW</span><span><i class="swatch orange"></i>source-aware AdamW</span><span>faint lines = individual seeds; heavy line = five-seed mean</span></div>
<div class="grid two">
 <article class="panel"><h3>Observed minibatch loss</h3><canvas id="loss" class="chart"></canvas><p class="note">Log scale. This is the actual training loss from the sampled minibatch at every fifth step—not an endpoint score.</p></article>
 <article class="panel"><h3>Validation normalized MSE</h3><canvas id="val" class="chart"></canvas><p class="note">Log scale over the complete validation set. Lower is better. Ordinary AdamW opens the clearer gap after roughly 300 steps.</p></article>
</div></section>

<section class="section"><h2>What “handle self-context correctly” meant here</h2>
<div class="grid mechanism">
 <article class="panel"><div class="flow">
  <div class="node"><strong>Frozen chart</strong><span>g<sub>f</sub><br>persistent learning decision</span></div><div class="arrow">→</div>
  <div class="node"><strong>Adam memory</strong><span>m, v updated only by g<sub>f</sub></span></div><div class="arrow">→</div>
  <div class="node"><strong>Certified request</strong><span>chart correction added only now</span></div>
 </div></article>
 <article class="panel"><div class="math">g_live = g_f + g_c
m_t = β₁m_{t−1} + (1−β₁)g_f
v_t = β₂v_{t−1} + (1−β₂)g_f²
u_f = m̂_t / (√v̂_t + ε)
u_c = g_c / (√v̂_t + ε)
α = min(1, ‖u_f‖ / ‖u_c‖)
u = Π_{⟨u,g_f⟩≥0}(u_f + αu_c)
θ ← (1−ηλ)θ − ηu</div></article>
</div>
<p class="callout">The important distinction is temporal authority. The context Jacobian is not deleted: its current correction is applied. It simply cannot accumulate its own momentum or elementwise second-moment history. The correction is capped after Adam preconditioning as well as during backprop, because a diagonal denominator can re-amplify a raw gradient that was harmless before preconditioning. Finally, any combined request that would reverse the live frozen-chart decision is minimally projected to its non-reversing boundary.</p>
</section>

<section class="section"><h2>The restraint was active, but it was not fighting an explosion</h2>
<div class="grid two">
 <article class="panel"><h3>Context size before and after Adam’s metric</h3><canvas id="ratios" class="chart"></canvas><p class="note">Green: raw ‖g<sub>c</sub>‖/‖g<sub>f</sub>‖ from backprop. Orange: ‖u<sub>c</sub>‖/‖u<sub>f</sub>‖ after the diagonal metric. The metric magnifies context unevenly even when its raw norm is small.</p></article>
 <article class="panel"><h3>Certificate behavior</h3><canvas id="certificate" class="chart"></canvas><p class="note">Request scale, alignment with the current frozen decision, and fraction of parameter tensors requiring the anti-reversal projection.</p></article>
</div>
<p>The raw chart gradient is generally only a few percent of the frozen gradient late in training. Yet after Adam’s elementwise metric, the chart request can be a much larger fraction of the fixed request. The source-aware rule therefore does real work. The failure is not numerical instability; it is that the moving chart carries useful coordinate-conditioned information on Ripple, and treating all of it as disposable current-step feedback throws away some of AdamW’s advantage.</p>
</section>

<section class="section"><h2>Fitted Ripple surface at each seed-0 best checkpoint</h2>
<div class="grid three">
 <article class="panel"><h3>Target</h3><canvas id="target" class="surface"></canvas></article>
 <article class="panel"><h3>Ordinary AdamW</h3><canvas id="fitAdam" class="surface"></canvas><p class="note">Best validation score 0.9799 at step 730.</p></article>
 <article class="panel"><h3>Source-aware AdamW</h3><canvas id="fitSource" class="surface"></canvas><p class="note">Best validation score 0.9351 at step 740.</p></article>
</div>
<div class="grid two">
 <article class="panel"><h3>|error| · ordinary AdamW</h3><canvas id="errAdam" class="surface"></canvas></article>
 <article class="panel"><h3>|error| · source-aware AdamW</h3><canvas id="errSource" class="surface"></canvas></article>
</div></section>

<section class="section"><h2>Every paired seed</h2><div class="panel"><table id="seedTable"><thead><tr><th>Seed / arm</th><th>Score-AUC</th><th>Best score</th><th>Best step</th><th>Score @ 750</th><th>Step to .80</th><th>Step to .90</th><th>Step to .95</th></tr></thead><tbody></tbody></table></div></section>

<section class="section"><h2>What this tells us next</h2>
<p>The original diagnosis—self-context has two causal gradient sources and the optimizer should be able to tell them apart—is supported mechanically, but the blunt temporal policy is not. “Context must not own momentum” is too strong. Ripple says the chart motion contains repeatable information worth remembering.</p>
<p>The next principled object is therefore not a weaker cap or a tuned learning rate. It is a <strong>coupled, non-elementwise second moment</strong> that preserves the identities of fixed decision and chart motion while estimating their covariance. AdamW currently wins by mixing them before squaring; this experiment loses by refusing the chart any memory. A better rule would retain a small block metric over the two source channels—fixed, chart, and their cross-term—so repeated compatible chart motion can earn authority while incoherent chart motion cannot manufacture elementwise sign normalization.</p>
<p class="note">Protocol: five paired seeds, 750 steps, width 24, batch 256, η=0.003, weight decay 10⁻⁴, global gradient norm clip 5, evaluation every five steps. No learning-rate sweep, task-conditioned branch, validation feedback, or selected-seed exclusion. All ten runs completed without numerical failure.</p>
</section>
<div class="footer">Generated from <code>results_adamw_self_context_ripple/results.json</code>. The page contains the complete plotted histories and seed-0 surface grids.</div>
</main>
<script>const DATA=__DATA__;
const COLORS={adamw:'#1667a8',adamw_source_aware:'#c45620'};
function setup(canvas){const d=devicePixelRatio||1,r=canvas.getBoundingClientRect();canvas.width=Math.round(r.width*d);canvas.height=Math.round(r.height*d);const c=canvas.getContext('2d');c.setTransform(d,0,0,d,0,0);return [c,r.width,r.height]}
function seriesPlot(id,arms,selectors,options={}){const canvas=document.getElementById(id),[c,w,h]=setup(canvas),pad={l:58,r:15,t:15,b:38};c.clearRect(0,0,w,h);let points=[];arms.forEach(a=>DATA.runs.filter(r=>r.arm===a).forEach(r=>r.history.forEach(x=>selectors.forEach(s=>{const v=s.get(x);if(Number.isFinite(v)&&v>=(options.log?0: -Infinity))points.push([x.step,v])}))));const xs=points.map(p=>p[0]),raw=points.map(p=>p[1]),ys=options.log?raw.filter(v=>v>0).map(Math.log10):raw;let xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=options.ymin??Math.min(...ys),ymax=options.ymax??Math.max(...ys);if(ymax===ymin)ymax+=1;const X=x=>pad.l+(x-xmin)/(xmax-xmin)*(w-pad.l-pad.r),Y=y=>{const z=options.log?Math.log10(Math.max(y,1e-12)):y;return h-pad.b-(z-ymin)/(ymax-ymin)*(h-pad.t-pad.b)};c.strokeStyle='#cfc8bc';c.lineWidth=1;c.beginPath();c.moveTo(pad.l,pad.t);c.lineTo(pad.l,h-pad.b);c.lineTo(w-pad.r,h-pad.b);c.stroke();c.fillStyle='#66717a';c.font='11px system-ui';c.textAlign='center';for(let i=0;i<=5;i++){let x=xmin+(xmax-xmin)*i/5;c.fillText(Math.round(x),X(x),h-15)}c.textAlign='right';for(let i=0;i<=4;i++){let z=ymin+(ymax-ymin)*i/4,v=options.log?10**z:z;c.fillText(v<.01?v.toExponential(1):v.toFixed(v<1?2:1),pad.l-7,h-pad.b-(i/4)*(h-pad.t-pad.b)+4)}c.save();c.translate(14,h/2);c.rotate(-Math.PI/2);c.textAlign='center';c.fillText(options.ylabel||'',0,0);c.restore();
 arms.forEach((arm,ai)=>{const runs=DATA.runs.filter(r=>r.arm===arm);selectors.forEach((sel,si)=>{runs.forEach(run=>{c.strokeStyle=sel.color||COLORS[arm];c.globalAlpha=.13;c.lineWidth=1;c.beginPath();run.history.forEach((row,j)=>{const y=sel.get(row);if(!Number.isFinite(y)||options.log&&y<=0)return;const fn=j?'lineTo':'moveTo';c[fn](X(row.step),Y(y))});c.stroke()});const steps=runs[0].history.map(x=>x.step);c.globalAlpha=1;c.strokeStyle=sel.color||COLORS[arm];c.lineWidth=sel.width||2.8;c.setLineDash(sel.dash||[]);c.beginPath();steps.forEach((step,j)=>{const vals=runs.map(r=>sel.get(r.history[j])).filter(Number.isFinite);const y=vals.reduce((a,b)=>a+b,0)/vals.length;c[j?'lineTo':'moveTo'](X(step),Y(y))});c.stroke();c.setLineDash([])})});c.globalAlpha=1}
function heat(id,values,min,max){const canvas=document.getElementById(id),[c,w,h]=setup(canvas),side=Math.round(Math.sqrt(values.length)),cellW=w/side,cellH=h/side;function color(v){let t=Math.max(0,Math.min(1,(v-min)/(max-min||1)));const stops=[[18,38,68],[24,107,143],[49,161,123],[211,191,77],[197,70,42]],p=t*(stops.length-1),i=Math.min(stops.length-2,Math.floor(p)),q=p-i,a=stops[i],b=stops[i+1];return `rgb(${a.map((x,k)=>Math.round(x+(b[k]-x)*q)).join(',')})`}values.forEach((v,i)=>{c.fillStyle=color(v);c.fillRect((i%side)*cellW,Math.floor(i/side)*cellH,Math.ceil(cellW)+1,Math.ceil(cellH)+1)})}
const fits={};DATA.runs.filter(r=>r.seed===0).forEach(r=>fits[r.arm]=r.fit_visualization);const target=fits.adamw.target,pa=fits.adamw.prediction,ps=fits.adamw_source_aware.prediction,all=target.concat(pa,ps),lo=Math.min(...all),hi=Math.max(...all);heat('target',target,lo,hi);heat('fitAdam',pa,lo,hi);heat('fitSource',ps,lo,hi);const ea=target.map((v,i)=>Math.abs(v-pa[i])),es=target.map((v,i)=>Math.abs(v-ps[i])),emax=Math.max(...ea,...es);heat('errAdam',ea,0,emax);heat('errSource',es,0,emax);
seriesPlot('loss',['adamw','adamw_source_aware'],[{get:r=>r.loss}],{log:true,ylabel:'minibatch MSE'});seriesPlot('val',['adamw','adamw_source_aware'],[{get:r=>r.normalized_mse}],{log:true,ylabel:'validation MSE'});seriesPlot('ratios',['adamw_source_aware'],[{get:r=>r.context_to_fixed_ratio,color:'#1c7a5a'},{get:r=>r.optimizer_context_request_ratio,color:'#c45620'}],{log:true,ylabel:'ratio'});seriesPlot('certificate',['adamw_source_aware'],[{get:r=>r.optimizer_context_request_scale,color:'#c45620'},{get:r=>r.optimizer_fixed_alignment,color:'#1667a8'},{get:r=>r.optimizer_reversal_fraction,color:'#a93535',dash:[5,4]}],{ymin:0,ymax:1,ylabel:'fraction / cosine'});
const tbody=document.querySelector('#seedTable tbody');for(let seed=0;seed<5;seed++)for(const arm of ['adamw','adamw_source_aware']){const r=DATA.runs.find(x=>x.seed===seed&&x.arm===arm),tr=document.createElement('tr'),label=arm==='adamw'?'AdamW':'AdamW · source-aware',other=DATA.runs.find(x=>x.seed===seed&&x.arm!==arm);const win=r.auc>other.auc?'winner':'';tr.innerHTML=`<td>${seed} · ${label}</td><td class="${win}">${r.auc.toFixed(4)}</td><td>${r.best_score.toFixed(4)}</td><td>${r.best_step}</td><td>${r.final_score.toFixed(4)}</td>${['0.8','0.9','0.95'].map(x=>`<td>${r.thresholds[x]??'—'}</td>`).join('')}`;tbody.appendChild(tr)}
addEventListener('resize',()=>location.reload());</script></body></html>'''


def main():
    payload = json.loads(RESULTS.read_text())
    OUTPUT.write_text(HTML.replace("__DATA__", json.dumps(payload, separators=(",", ":"))))
    print(OUTPUT)


if __name__ == "__main__":
    main()
