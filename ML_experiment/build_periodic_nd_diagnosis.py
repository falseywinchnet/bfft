#!/usr/bin/env python3
"""Build the focused periodic N-D diagnostic viewer from measured M4 runs."""
from __future__ import annotations

import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCES = {
    "screen": ROOT / "results_periodic_nd_screen" / "results.json",
    "confirm": ROOT / "results_periodic_nd_confirm" / "results.json",
    "frame_screen": ROOT / "results_periodic_nd_frame_screen" / "results.json",
    "frame_confirm": ROOT / "results_periodic_nd_frame_confirm" / "results.json",
    "hybrid_500": ROOT / "results_periodic_nd_hybrid_500" / "results.json",
    "hybrid": ROOT / "results_periodic_nd_hybrid" / "results.json",
}

NAMES = {
    "mlp_26k": "Dense MLP · 26k",
    "self_context": "Self-context",
    "cff": "CFF",
    "learned_cone": "Learned cone",
    "operator_sphere": "Operator sphere",
    "axis_chart_16": "Axis charts · 2.6k",
    "axis_chart_32": "Axis charts · 9.2k",
    "orthogonal_chart_24": "Learned commuting frame",
    "orthogonal_chart_24_fast": "Transport-rate frame",
    "orthogonal_identity_24": "Identity-preserving frame",
    "qr_chart_24": "QR frame",
    "self_commuting_chart": "Self + commuting charts",
    "fourier_oracle": "Fourier oracle ceiling",
}


def load():
    return {
        name: json.loads(path.read_text())["runs"]
        for name, path in SOURCES.items()
    }


def rows(source, variant):
    return [row for row in source if row["variant"] == variant]


def aggregate(source, variants):
    result = []
    for variant in variants:
        selected = rows(source, variant)
        if not selected:
            continue
        values = [row["r2"] for row in selected]
        result.append({
            "variant": variant,
            "name": NAMES[variant],
            "r2": statistics.mean(values),
            "r2_sd": statistics.pstdev(values),
            "parameters": selected[0]["parameters"],
            "seconds": statistics.mean(row["seconds"] for row in selected),
            "offdiag": statistics.mean(
                row["hessian_off_diagonal_ratio"] for row in selected
            ),
            "gradient": statistics.mean(
                row["gradient_correlation"] for row in selected
            ),
        })
    return result


def make_data(sources):
    at_500 = aggregate(sources["screen"], (
        "mlp_26k", "self_context", "cff", "learned_cone",
        "operator_sphere", "axis_chart_16", "axis_chart_32", "fourier_oracle",
    ))
    at_500 += aggregate(sources["frame_screen"], ("orthogonal_chart_24_fast",))
    at_500 += aggregate(sources["hybrid_500"], ("self_commuting_chart",))
    at_2000 = aggregate(sources["confirm"], (
        "mlp_26k", "self_context", "learned_cone", "operator_sphere",
        "axis_chart_16", "axis_chart_32",
    ))
    at_2000 += aggregate(sources["frame_confirm"], (
        "orthogonal_chart_24", "orthogonal_chart_24_fast",
        "orthogonal_identity_24", "qr_chart_24",
    ))
    at_2000 += aggregate(sources["hybrid"], ("self_commuting_chart",))

    partial_sources = {
        "self_context": rows(sources["screen"], "self_context")[0],
        "learned_cone": rows(sources["screen"], "learned_cone")[0],
        "operator_sphere": rows(sources["screen"], "operator_sphere")[0],
        "axis_chart_32": rows(sources["screen"], "axis_chart_32")[0],
        "self_commuting_chart": rows(
            sources["hybrid_500"], "self_commuting_chart"
        )[0],
    }
    partial = {
        name: row["partial_dependence"] for name, row in partial_sources.items()
    }

    history_sources = {
        "mlp_26k": rows(sources["confirm"], "mlp_26k")[0],
        "self_context": rows(sources["confirm"], "self_context")[0],
        "learned_cone": rows(sources["confirm"], "learned_cone")[0],
        "operator_sphere": rows(sources["confirm"], "operator_sphere")[0],
        "axis_chart_32": rows(sources["confirm"], "axis_chart_32")[0],
        "orthogonal_chart_24_fast": rows(
            sources["frame_confirm"], "orthogonal_chart_24_fast"
        )[0],
        "self_commuting_chart": rows(
            sources["hybrid"], "self_commuting_chart"
        )[0],
    }
    history = {
        name: [{"step": point["step"], "r2": point["r2"]}
               for point in row["history"]]
        for name, row in history_sources.items()
    }
    seed_stability = {
        variant: [row["r2"] for row in rows(sources["frame_confirm"], variant)]
        for variant in (
            "orthogonal_chart_24", "orthogonal_chart_24_fast",
            "orthogonal_identity_24", "qr_chart_24",
        )
    }
    return {
        "names": NAMES,
        "at500": at_500,
        "at2000": at_2000,
        "partial": partial,
        "history": history,
        "seedStability": seed_stability,
    }


HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Periodic N-D · commuting-chart diagnosis</title>
<style>
:root{color-scheme:light dark;--bg:#f5f3ee;--surface:#fffefa;--ink:#1d2730;--muted:#68737c;--line:#d8d4ca;--accent:#d45135;--good:#16806a;--blue:#3366a8;--violet:#7857a6;--gold:#ad7d1d;--cyan:#1686a0}
@media(prefers-color-scheme:dark){:root{--bg:#121619;--surface:#191f23;--ink:#edf1f3;--muted:#a8b1b7;--line:#354047;--accent:#f07b61;--good:#51c9a8;--blue:#73a8ea;--violet:#b99ae2;--gold:#e5b95a;--cyan:#65bfd2}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}main{max-width:1500px;margin:auto;padding:32px 28px 80px}.eyebrow{text-transform:uppercase;letter-spacing:.14em;font-size:11px;color:var(--accent);font-weight:750}h1{font-size:clamp(30px,4vw,58px);line-height:.98;max-width:980px;margin:10px 0 14px;letter-spacing:-.045em}.dek{max-width:970px;color:var(--muted);font-size:17px;line-height:1.55;margin:0 0 24px}.verdict{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));border:1px solid var(--line);background:var(--surface);margin:26px 0 46px}.fact{padding:18px 20px;border-right:1px solid var(--line)}.fact:last-child{border:0}.fact b{display:block;font-size:27px;letter-spacing:-.04em}.fact span{font-size:12px;color:var(--muted);line-height:1.3;display:block;margin-top:4px}section{margin-top:48px}h2{font-size:24px;letter-spacing:-.025em;margin:0 0 6px}.section-note{color:var(--muted);font-size:14px;margin:0 0 18px}.two{display:grid;grid-template-columns:1fr 1fr;gap:22px}.panel{border-top:1px solid var(--line);padding-top:14px;min-width:0}.panel h3{font-size:14px;margin:0 0 10px}.bar-row{display:grid;grid-template-columns:minmax(130px,1.2fr) 4fr 46px;align-items:center;gap:10px;margin:7px 0;font-size:12px}.bar-track{height:11px;background:color-mix(in srgb,var(--line) 55%,transparent);position:relative}.bar{height:100%;background:var(--blue)}.bar.winner{background:var(--good)}.bar.oracle{background:var(--muted)}.value{text-align:right;font-variant-numeric:tabular-nums}.legend{display:flex;flex-wrap:wrap;gap:7px 15px;margin:10px 0 16px}.legend button{border:0;background:transparent;color:var(--ink);padding:2px 0;font:inherit;font-size:12px;cursor:pointer;opacity:.42}.legend button[aria-pressed=true]{opacity:1}.swatch{width:16px;height:3px;display:inline-block;margin-right:6px;vertical-align:middle}.partials{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}.chart{width:100%;height:auto;background:var(--surface);border:1px solid var(--line)}.axis-title{font-size:10px;fill:var(--muted)}.tick{font-size:9px;fill:var(--muted)}.grid{stroke:var(--line);stroke-width:.7}.series{fill:none;stroke-width:1.7}.truth{stroke:var(--ink);stroke-width:2.1}.wide-chart{height:360px}.geometry{display:grid;grid-template-columns:1.4fr 1fr;gap:22px}.seed-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}.seed-card{border:1px solid var(--line);background:var(--surface);padding:13px}.seed-card b{font-size:13px;display:block;min-height:34px}.seed-values{display:flex;align-items:end;gap:5px;height:100px;margin-top:9px;border-bottom:1px solid var(--line)}.seed-values i{display:block;flex:1;background:var(--violet);min-height:2px}.seed-card small{color:var(--muted);display:block;margin-top:7px}.finding{border-left:3px solid var(--accent);padding:3px 0 3px 15px;margin:16px 0;max-width:920px;line-height:1.5}.finding strong{color:var(--accent)}footer{margin-top:56px;border-top:1px solid var(--line);padding-top:16px;color:var(--muted);font-size:12px}
@media(max-width:900px){.verdict{grid-template-columns:1fr 1fr}.fact:nth-child(2){border-right:0}.fact:nth-child(-n+2){border-bottom:1px solid var(--line)}.two,.geometry{grid-template-columns:1fr}.partials{grid-template-columns:repeat(2,1fr)}}
@media(max-width:520px){main{padding:24px 14px 60px}.verdict{grid-template-columns:1fr}.fact{border-right:0;border-bottom:1px solid var(--line)!important}.partials,.seed-grid{grid-template-columns:1fr}.bar-row{grid-template-columns:110px 1fr 42px}.wide-chart{height:300px}}
</style>
</head>
<body><main>
<div class="eyebrow">Focused M4 experiment · periodic N-D</div>
<h1>The target is not asking for transport. It is asking for commuting curvature.</h1>
<p class="dek">Eight independent scalar oscillators share one output. Existing self-context and flow layers dynamically remix their curvature frame, so they acquire frequencies 1–2 and almost completely miss 3–5. A nonperiodic LELU chart bank that preserves one scalar response per frame ray changes the result.</p>
<div class="verdict">
 <div class="fact"><b>0.327</b><span>Self-context mean R²<br>2,000 steps · 9.3k parameters</span></div>
 <div class="fact"><b>0.994</b><span>Identity-preserving learned frame<br>2,000 steps · 5.4k parameters</span></div>
 <div class="fact"><b>0.991</b><span>Self-context + commuting charts<br>2,000 steps · 14.7k parameters</span></div>
 <div class="fact"><b>0 sine</b><span>The winning chart models use only affine maps and LELU</span></div>
</div>

<section><h2>Endpoint: the failure is not parameter count</h2><p class="section-note">R² on held-out points. Error whiskers appear where paired seeds were run.</p><div class="two"><div class="panel"><h3>500 optimizer steps</h3><div id="bars500"></div></div><div class="panel"><h3>2,000 optimizer steps</h3><div id="bars2000"></div></div></div></section>

<section><h2>What each coordinate actually learned at 500 steps</h2><p class="section-note">Partial dependence averages over 128 authentic contexts. Each panel is one observed coordinate; the title gives its target frequency.</p><div class="legend" id="partialLegend"></div><div class="partials" id="partials"></div></section>

<section><h2>Acquisition dynamics</h2><p class="section-note">Self-context saturates after acquiring the low-frequency coordinates. Scalar chart paths continue acquiring the missing harmonics.</p><div class="panel"><svg id="history" class="chart wide-chart" role="img" aria-label="R squared over training steps"></svg></div></section>

<section><h2>The geometry exposes the mismatch</h2><p class="section-note">The truth has a diagonal Hessian in one fixed frame. Lower mixed-curvature ratio is therefore structurally correct here.</p><div class="geometry"><div class="panel"><svg id="geometry" class="chart wide-chart" role="img" aria-label="R squared versus mixed Hessian curvature"></svg></div><div><div class="finding"><strong>Current mechanisms:</strong> mixed-Hessian ratios around 0.6–0.8. Their local interpretation changes with the observation.</div><div class="finding"><strong>Commuting charts:</strong> each scalar response differentiates only along its own ray. Mixed curvature is zero in that frame.</div><div class="finding"><strong>Remaining problem:</strong> a randomly initialized learned frame can reach R² 0.96, but frame acquisition is optimizer-sensitive.</div></div></div></section>

<section><h2>Frame acquisition is now the honest open problem</h2><p class="section-note">Three paired datasets, 2,000 steps. The identity-preserving frame is robust; random-frame parameterizations expose optimizer variance.</p><div class="seed-grid" id="seeds"></div></section>

<footer>CPU-only M4 runs · AdamW · identical periodic N-D data generator · LELU everywhere except the explicitly labeled Fourier diagnostic ceiling.</footer>
</main>
<script>
const data=__DATA__;
const colors=['var(--blue)','var(--accent)','var(--violet)','var(--good)','var(--gold)','var(--cyan)'];
const chosen=['self_context','learned_cone','operator_sphere','axis_chart_32','self_commuting_chart'];
const colorOf=n=>colors[chosen.indexOf(n)%colors.length];
function bars(id,rows){const root=document.getElementById(id);[...rows].sort((a,b)=>b.r2-a.r2).forEach(r=>{const row=document.createElement('div');row.className='bar-row';const label=document.createElement('span');label.textContent=r.name;const track=document.createElement('div');track.className='bar-track';const bar=document.createElement('div');bar.className='bar '+(r.r2>.94?'winner ':'')+(r.variant==='fourier_oracle'?'oracle':'');bar.style.width=`${Math.max(0,r.r2)*100}%`;bar.title=`R² ${r.r2.toFixed(4)} · ${r.parameters.toLocaleString()} parameters · ${r.seconds.toFixed(2)} s`;track.appendChild(bar);const value=document.createElement('span');value.className='value';value.textContent=r.r2.toFixed(3);row.append(label,track,value);root.append(row)})}
bars('bars500',data.at500);bars('bars2000',data.at2000);
const ns='http://www.w3.org/2000/svg';const svgEl=(tag,attrs={})=>{const e=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));return e};
function path(values,x,y){return values.map((v,i)=>(i?'L':'M')+x(i)+','+y(v)).join(' ')}
const visible=new Set(chosen);const legend=document.getElementById('partialLegend');chosen.forEach((name,i)=>{const button=document.createElement('button');button.type='button';button.setAttribute('aria-pressed','true');button.innerHTML=`<span class="swatch" style="background:${colors[i]}"></span>${data.names[name]}`;button.onclick=()=>{visible.has(name)?visible.delete(name):visible.add(name);button.setAttribute('aria-pressed',visible.has(name));drawPartials()};legend.append(button)});
function drawPartials(){const root=document.getElementById('partials');root.innerHTML='';const reference=data.partial[chosen[0]];reference.curves.forEach((curve,axis)=>{const svg=svgEl('svg',{viewBox:'0 0 320 210',class:'chart',role:'img','aria-label':`Axis ${axis+1}, frequency ${curve.frequency}`});const left=38,right=310,top=25,bottom=178;const all=[];chosen.forEach(n=>all.push(...data.partial[n].curves[axis].prediction));all.push(...curve.truth);const lo=Math.min(...all),hi=Math.max(...all),pad=(hi-lo)*.08||.1;const yy=v=>bottom-(v-(lo-pad))/(hi-lo+2*pad)*(bottom-top),xx=i=>left+i/(curve.truth.length-1)*(right-left);[-Math.PI,0,Math.PI].forEach((v,j)=>{const x=left+j*(right-left)/2;svg.append(svgEl('line',{x1:x,x2:x,y1:top,y2:bottom,class:'grid'}));const t=svgEl('text',{x,y:194,class:'tick','text-anchor':'middle'});t.textContent=j===0?'−π':j===1?'0':'π';svg.append(t)});const zero=yy(0);if(zero>=top&&zero<=bottom)svg.append(svgEl('line',{x1:left,x2:right,y1:zero,y2:zero,class:'grid'}));const title=svgEl('text',{x:12,y:16,class:'axis-title'});title.textContent=`x${axis+1} · frequency ${curve.frequency}`;svg.append(title);const truth=svgEl('path',{d:path(curve.truth,xx,yy),class:'series truth'});svg.append(truth);chosen.forEach(name=>{if(!visible.has(name))return;svg.append(svgEl('path',{d:path(data.partial[name].curves[axis].prediction,xx,yy),class:'series',stroke:colorOf(name)}))});root.append(svg)})}
drawPartials();
function axes(svg,{left,right,top,bottom,xTicks,yTicks,xScale,yScale,xLabel,yLabel}){xTicks.forEach(v=>{const x=xScale(v);svg.append(svgEl('line',{x1:x,x2:x,y1:top,y2:bottom,class:'grid'}));const t=svgEl('text',{x,y:bottom+18,class:'tick','text-anchor':'middle'});t.textContent=v;svg.append(t)});yTicks.forEach(v=>{const y=yScale(v);svg.append(svgEl('line',{x1:left,x2:right,y1:y,y2:y,class:'grid'}));const t=svgEl('text',{x:left-8,y:y+3,class:'tick','text-anchor':'end'});t.textContent=Number(v).toFixed(1);svg.append(t)});const xt=svgEl('text',{x:(left+right)/2,y:bottom+38,class:'axis-title','text-anchor':'middle'});xt.textContent=xLabel;svg.append(xt);const yt=svgEl('text',{x:15,y:(top+bottom)/2,class:'axis-title',transform:`rotate(-90 15 ${(top+bottom)/2})`,'text-anchor':'middle'});yt.textContent=yLabel;svg.append(yt)}
function drawHistory(){const svg=document.getElementById('history');svg.setAttribute('viewBox','0 0 900 360');const left=58,right=884,top=22,bottom=310,x=v=>left+v/2000*(right-left),y=v=>bottom-v*(bottom-top);axes(svg,{left,right,top,bottom,xTicks:[0,500,1000,1500,2000],yTicks:[0,.25,.5,.75,1],xScale:x,yScale:y,xLabel:'optimizer step',yLabel:'held-out R²'});Object.entries(data.history).forEach(([name,series],i)=>{svg.append(svgEl('path',{d:series.map((p,j)=>(j?'L':'M')+x(p.step)+','+y(p.r2)).join(' '),class:'series',stroke:colors[i%colors.length]}));const last=series[series.length-1],t=svgEl('text',{x:x(last.step)-4,y:y(last.r2)-5,class:'axis-title','text-anchor':'end'});t.textContent=data.names[name];t.setAttribute('fill',colors[i%colors.length]);svg.append(t)})}drawHistory();
function drawGeometry(){const svg=document.getElementById('geometry');svg.setAttribute('viewBox','0 0 700 360');const selected=data.at2000.filter(r=>['self_context','learned_cone','operator_sphere','axis_chart_32','orthogonal_chart_24_fast','orthogonal_identity_24','self_commuting_chart'].includes(r.variant));const left=58,right=680,top=22,bottom=310,x=v=>left+v/.9*(right-left),y=v=>bottom-v*(bottom-top);axes(svg,{left,right,top,bottom,xTicks:[0,.2,.4,.6,.8],yTicks:[0,.25,.5,.75,1],xScale:x,yScale:y,xLabel:'mixed Hessian energy ratio',yLabel:'held-out R²'});selected.forEach((r,i)=>{const c=colors[i%colors.length],circle=svgEl('circle',{cx:x(r.offdiag),cy:y(r.r2),r:6,fill:c});svg.append(circle);const t=svgEl('text',{x:x(r.offdiag)+9,y:y(r.r2)+4,class:'axis-title'});t.textContent=r.name;svg.append(t)})}drawGeometry();
function drawSeeds(){const root=document.getElementById('seeds');Object.entries(data.seedStability).forEach(([name,values])=>{const card=document.createElement('div');card.className='seed-card';const title=document.createElement('b');title.textContent=data.names[name];const bars=document.createElement('div');bars.className='seed-values';values.forEach(v=>{const bar=document.createElement('i');bar.style.height=`${v*100}%`;bar.title=`R² ${v.toFixed(4)}`;bars.append(bar)});const note=document.createElement('small');note.textContent=`mean ${((values.reduce((a,b)=>a+b,0))/values.length).toFixed(3)} · range ${Math.min(...values).toFixed(3)}–${Math.max(...values).toFixed(3)}`;card.append(title,bars,note);root.append(card)})}drawSeeds();
</script></body></html>'''


def main():
    data = make_data(load())
    output = ROOT / "periodic_nd_diagnosis.html"
    output.write_text(HTML.replace("__DATA__", json.dumps(data, separators=(",", ":"))))
    print(f"{output} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
