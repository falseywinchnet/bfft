#!/usr/bin/env python3
"""Build the compact interactive visualization used in the experiment report."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/soft_eikonal_instructive/results_confirm/results.json"
SUMMARY = ROOT / "experiments/soft_eikonal_instructive/results_confirm/summary.json"
PROBES = ROOT / "experiments/soft_eikonal_instructive/results_confirm/probes.json"
RADIAL_PROBES = ROOT / "experiments/soft_eikonal_instructive/results_confirm/radial_probes.json"
OUT = Path("/Users/ultimussecundai/.codex/visualizations/2026/08/16/01a00875-c9dc-7e01-8e13-444500eb3edf/soft-eikonal-instructive.html")


def compact_probe(row):
    result = {k: row[k] for k in ("task", "variant", "type")}
    if row["type"] == "curve":
        take = range(0, len(row["x"]), 4)
        result.update(x=[round(row["x"][i], 4) for i in take],
                      truth=[round(row["truth"][i], 5) for i in take],
                      prediction=[round(row["prediction"][i], 5) for i in take],
                      train_limit=row["train_limit"])
    else:
        old = row["size"]; indices = list(range(0, old, 2)); size = len(indices)
        flat = lambda source: [round(source[y * old + x], 4) for y in indices for x in indices]
        result.update(size=size, limits=row["limits"], field=flat(row["field"]), truth=flat(row["truth"]))
    return result


def compact_data():
    results = json.loads(RESULTS.read_text())["runs"]
    summary = json.loads(SUMMARY.read_text())
    probes = (json.loads(PROBES.read_text())["probes"]
              + json.loads(RADIAL_PROBES.read_text())["probes"])
    keep = {"soft_eikonal", "self_context", "temperature_hard", "secant_relational",
            "garnish_instructive", "paired_zero"}
    histories = {}
    for row in results:
        if row["variant"] not in keep:
            continue
        key = f"{row['task']}|{row['variant']}"
        histories.setdefault(key, []).append(row["history"])
    averaged = {}
    for key, seed_histories in histories.items():
        averaged[key] = [{"step": points[0]["step"],
                          "score": round(sum(p["score"] for p in points) / len(points), 5)}
                         for points in zip(*seed_histories)]
    task_rows = [{k: row[k] for k in ("task", "variant", "validation_score", "score",
                                                   "tail_score", "learning_auc", "seconds")}
                 for row in summary["by_task"] if row["variant"] in keep]
    return {"tasks": summary["tasks"], "byTask": task_rows, "histories": averaged,
            "probes": [compact_probe(row) for row in probes if row["variant"] in keep]}


TEMPLATE = r'''<div class="sei-root">
  <style>
    .sei-root{--bg:var(--color-background-primary,#faf9f6);--panel:var(--color-background-secondary,#fff);--text:var(--color-text-primary,#20211f);--muted:var(--color-text-secondary,#6c706a);--line:var(--color-border-primary,#d8dad5);--blue:var(--color-accent-blue,#2474d2);--cyan:var(--color-accent-cyan,#1b9aaa);--orange:var(--color-accent-orange,#e57a2d);--red:var(--color-accent-red,#ce4a50);--green:var(--color-accent-green,#2d9b68);font:14px/1.42 ui-sans-serif,system-ui,-apple-system,sans-serif;color:var(--text);background:var(--bg);border:1px solid var(--line);border-radius:16px;padding:20px;box-sizing:border-box;max-width:1180px;margin:auto}.sei-root *{box-sizing:border-box}.sei-root h2{font-size:23px;line-height:1.12;margin:0 0 6px}.sei-root h3{font-size:14px;margin:0}.sei-kicker{font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--blue);font-weight:750}.sei-sub{color:var(--muted);max-width:850px;margin:0}.sei-chips{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:18px 0}.sei-chip,.sei-panel{background:var(--panel);border:1px solid var(--line);border-radius:12px}.sei-chip{padding:12px}.sei-num{display:block;font-size:23px;font-weight:760;letter-spacing:-.03em}.sei-label{font-size:11px;color:var(--muted)}.sei-controls{display:flex;gap:10px;align-items:end;flex-wrap:wrap;margin:12px 0}.sei-controls label{font-size:11px;color:var(--muted);display:grid;gap:4px}.sei-controls select{color:var(--text);background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:7px 28px 7px 9px}.sei-buttons{display:flex;gap:6px;flex-wrap:wrap}.sei-buttons button{color:var(--muted);background:var(--panel);border:1px solid var(--line);border-radius:999px;padding:7px 10px;cursor:pointer}.sei-buttons button.active{color:var(--panel);background:var(--blue);border-color:var(--blue)}.sei-grid{display:grid;grid-template-columns:1.05fr .95fr;gap:12px}.sei-panel{padding:14px;min-width:0}.sei-panel-head{display:flex;justify-content:space-between;align-items:baseline;gap:8px;margin-bottom:8px}.sei-note{font-size:11px;color:var(--muted)}.sei-plots{display:grid;grid-template-columns:1fr 1fr;gap:10px}.sei-plot-title{text-align:center;color:var(--muted);font-size:11px;margin-bottom:2px}.sei-root svg{display:block;width:100%;height:auto;overflow:visible}.sei-axis text{fill:var(--muted);font-size:10px}.sei-axis path,.sei-axis line{stroke:var(--line)}.sei-legend{display:flex;gap:12px;flex-wrap:wrap;margin-top:6px;font-size:11px;color:var(--muted)}.sei-dot{width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:4px}.sei-delta{margin-top:12px}.sei-drow{display:grid;grid-template-columns:116px 1fr 48px;gap:8px;align-items:center;margin:7px 0;font-size:12px}.sei-track{height:8px;background:var(--bg);border:1px solid var(--line);border-radius:99px;overflow:hidden}.sei-bar{height:100%;border-radius:99px}.sei-pos{background:var(--green)}.sei-neg{background:var(--red)}.sei-callout{border-left:3px solid var(--orange);padding:8px 10px;margin-top:10px;background:color-mix(in srgb,var(--orange) 8%,var(--panel));font-size:12px}.sei-tooltip{position:fixed;pointer-events:none;display:none;background:var(--text);color:var(--panel);padding:6px 8px;border-radius:6px;font-size:11px;z-index:30}.sei-foot{font-size:11px;color:var(--muted);margin-top:10px}.sei-root :focus-visible{outline:2px solid var(--blue);outline-offset:2px}@media(prefers-color-scheme:dark){.sei-root{--bg:var(--color-background-primary,#171916);--panel:var(--color-background-secondary,#22241f);--text:var(--color-text-primary,#f3f3ef);--muted:var(--color-text-secondary,#afb3aa);--line:var(--color-border-primary,#42463e)}}@media(max-width:760px){.sei-chips{grid-template-columns:1fr 1fr}.sei-grid{grid-template-columns:1fr}.sei-root{padding:14px}.sei-drow{grid-template-columns:96px 1fr 44px}}@media(max-width:420px){.sei-chips{grid-template-columns:1fr}.sei-plots{grid-template-columns:1fr}.sei-root h2{font-size:20px}}
    .sei-root .sei-buttons button.active{color:var(--color-text-on-accent,#fff)}
  </style>
  <div class="sei-kicker">270 exact-budget M4 CPU runs</div>
  <h2>What changes when Eikonal allocation gets a private first guess?</h2>
  <p class="sei-sub">Self-context reallocates after a parameter-free, model-generated backprojection. It learns faster everywhere, but it does not manufacture evidence for unseen checkerboard or spiral continuation.</p>
  <div class="sei-chips">
    <div class="sei-chip"><span class="sei-num">6 / 6</span><span class="sei-label">learning-AUC wins vs unchanged Eikonal</span></div>
    <div class="sei-chip"><span class="sei-num">+.079</span><span class="sei-label">radial-stripes score</span></div>
    <div class="sei-chip"><span class="sei-num">+.099</span><span class="sei-label">multiscale continuation score</span></div>
    <div class="sei-chip"><span class="sei-num">same</span><span class="sei-label">7,741–7,814 parameters by task</span></div>
  </div>
  <div class="sei-controls">
    <label>Problem<select class="sei-task"></select></label>
    <div><div class="sei-note" style="margin-bottom:4px">Prediction variant</div><div class="sei-buttons"></div></div>
  </div>
  <div class="sei-grid">
    <section class="sei-panel">
      <div class="sei-panel-head"><h3>Geometry: truth vs fitted solution</h3><span class="sei-note sei-geometry-note"></span></div>
      <div class="sei-plots"><div><div class="sei-plot-title">Truth</div><svg class="sei-truth" viewBox="0 0 300 230" role="img"></svg></div><div><div class="sei-plot-title">Prediction</div><svg class="sei-pred" viewBox="0 0 300 230" role="img"></svg></div></div>
      <div class="sei-callout"></div>
    </section>
    <section class="sei-panel">
      <div class="sei-panel-head"><h3>Acquisition trajectory</h3><span class="sei-note">mean of 3 seeds · 800 steps</span></div>
      <svg class="sei-learning" viewBox="0 0 430 260" role="img" aria-label="Learning curve comparison"></svg>
      <div class="sei-legend"></div>
      <div class="sei-delta"><h3>Self-context minus unchanged Eikonal</h3><div class="sei-drows"></div></div>
    </section>
  </div>
  <p class="sei-foot">Classification uses mean class recall; regression uses 1 / (1 + normalized MSE). For checkerboard, the field includes the unobserved exterior. For 1-D tasks, the shaded center is the observed interval [−3, 3]. Axes preserve mathematical y-up orientation.</p>
  <div class="sei-tooltip"></div>
  <script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
  <script>
  (()=>{const root=document.currentScript.closest('.sei-root'),data=__DATA__;
  const labels={soft_eikonal:'Eikonal',self_context:'Self-context',temperature_hard:'Hard gate',secant_relational:'Relational secant',garnish_instructive:'Garnish',paired_zero:'Paired-zero'};
  const order=Object.keys(labels),taskLabels={checkerboard:'Checkerboard',localized_steps_1d:'Localized steps 1-D',multiscale_1d:'Multiscale 1-D',radial_stripes:'Radial stripes',ripple:'Ripple',spiral:'Spiral'};
  const css=getComputedStyle(root),colors={soft_eikonal:css.getPropertyValue('--blue').trim(),self_context:css.getPropertyValue('--green').trim(),temperature_hard:css.getPropertyValue('--orange').trim(),secant_relational:css.getPropertyValue('--cyan').trim(),garnish_instructive:css.getPropertyValue('--red').trim(),paired_zero:css.getPropertyValue('--muted').trim()};
  let task='radial_stripes',variant='self_context'; const select=d3.select(root).select('.sei-task'); select.selectAll('option').data(['radial_stripes','ripple','multiscale_1d','localized_steps_1d','checkerboard']).join('option').attr('value',d=>d).text(d=>taskLabels[d]);select.property('value',task).on('change',e=>{task=e.target.value;render()});
  const buttons=d3.select(root).select('.sei-buttons').selectAll('button').data(order).join('button').text(d=>labels[d]).on('click',(e,d)=>{variant=d;render()});
  const tooltip=d3.select(root).select('.sei-tooltip');
  function probe(which){return data.probes.find(d=>d.task===task&&d.variant===which)} function row(which){return data.byTask.find(d=>d.task===task&&d.variant===which)}
  function field(svg,node,key){svg.selectAll('*').remove();const n=node.size,[xmin,xmax,ymin,ymax]=node.limits,m={l:28,r:7,t:5,b:22},w=300-m.l-m.r,h=230-m.t-m.b,x=d3.scaleLinear([xmin,xmax],[m.l,m.l+w]),y=d3.scaleLinear([ymin,ymax],[m.t+h,m.t]);const vals=node[key],extent=d3.extent(vals),isClass=task==='checkerboard'||task==='radial_stripes',lo=css.getPropertyValue('--blue').trim(),hi=css.getPropertyValue('--orange').trim(),mid=css.getPropertyValue('--panel').trim();const color=isClass?d3.scaleLinear([0,.5,1],[lo,mid,hi]):d3.scaleDiverging([extent[0],0,extent[1]],d3.interpolateRdBu).clamp(true);const cw=w/n,ch=h/n;svg.append('g').selectAll('rect').data(vals).join('rect').attr('x',(d,i)=>m.l+(i%n)*cw).attr('y',(d,i)=>m.t+h-(Math.floor(i/n)+1)*ch).attr('width',cw+.4).attr('height',ch+.4).attr('fill',d=>color(d)).on('pointermove',(e,d)=>tooltip.style('display','block').style('left',(e.clientX+10)+'px').style('top',(e.clientY+10)+'px').text(d.toFixed(3))).on('pointerleave',()=>tooltip.style('display','none'));svg.append('g').attr('class','sei-axis').attr('transform',`translate(0,${m.t+h})`).call(d3.axisBottom(x).ticks(4));svg.append('g').attr('class','sei-axis').attr('transform',`translate(${m.l},0)`).call(d3.axisLeft(y).ticks(4));}
  function curve(svg,node,key){svg.selectAll('*').remove();const m={l:35,r:7,t:5,b:22},w=300-m.l-m.r,h=230-m.t-m.b,x=d3.scaleLinear(d3.extent(node.x),[m.l,m.l+w]),all=node.truth.concat(node.prediction),y=d3.scaleLinear(d3.extent(all),[m.t+h,m.t]).nice();svg.append('rect').attr('x',x(-3)).attr('y',m.t).attr('width',x(3)-x(-3)).attr('height',h).attr('fill',css.getPropertyValue('--blue').trim()).attr('opacity',.07);[-3,3].forEach(v=>svg.append('line').attr('x1',x(v)).attr('x2',x(v)).attr('y1',m.t).attr('y2',m.t+h).attr('stroke',css.getPropertyValue('--line').trim()).attr('stroke-dasharray','4 3'));const line=d3.line().x((d,i)=>x(node.x[i])).y(d=>y(d));svg.append('path').datum(node[key]).attr('fill','none').attr('stroke',key==='truth'?css.getPropertyValue('--text').trim():colors[variant]).attr('stroke-width',2).attr('d',line);svg.append('g').attr('class','sei-axis').attr('transform',`translate(0,${m.t+h})`).call(d3.axisBottom(x).ticks(5));svg.append('g').attr('class','sei-axis').attr('transform',`translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5));}
  function geometry(){const p=probe(variant),truth=d3.select(root).select('.sei-truth'),pred=d3.select(root).select('.sei-pred'); if(p.type==='field'){field(truth,p,'truth');field(pred,p,'field')}else{curve(truth,p,'truth');curve(pred,p,'prediction')} const r=row(variant),b=row('soft_eikonal');d3.select(root).select('.sei-geometry-note').text(`${labels[variant]} · score ${r.score.toFixed(3)}`);let msg=task==='checkerboard'?'The inner fit is excellent, but the exterior does not continue the checker rule. No variant earns a strict tail-survival bin.':task==='radial_stripes'?'Self-context turns the uncertain rings into a stable radial atlas: .911 ± .008 versus .831 ± .091.':task==='multiscale_1d'?'Continuation remains imperfect, but self-context and relational secants bend toward the withheld tails instead of immediately diverging.':task==='localized_steps_1d'?'The base already fits this piecewise rule; relational secants preserve its tails best.':'At width 36, all Eikonal forms resolve the ripple; self-context reaches it sooner.';d3.select(root).select('.sei-callout').text(msg)}
  function learning(){const svg=d3.select(root).select('.sei-learning');svg.selectAll('*').remove();const m={l:42,r:10,t:8,b:28},w=430-m.l-m.r,h=260-m.t-m.b,x=d3.scaleLinear([0,800],[m.l,m.l+w]),series=order.map(v=>({v,pts:data.histories[task+'|'+v]})),y=d3.scaleLinear([Math.max(0,d3.min(series,s=>d3.min(s.pts,d=>d.score))-.04),1],[m.t+h,m.t]);svg.append('g').attr('class','sei-axis').attr('transform',`translate(0,${m.t+h})`).call(d3.axisBottom(x).ticks(5));svg.append('g').attr('class','sei-axis').attr('transform',`translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5));const line=d3.line().x(d=>x(d.step)).y(d=>y(d.score));svg.append('g').selectAll('path').data(series).join('path').attr('d',d=>line(d.pts)).attr('fill','none').attr('stroke',d=>colors[d.v]).attr('stroke-width',d=>d.v==='self_context'?3:1.6).attr('opacity',d=>['soft_eikonal','self_context'].includes(d.v)?1:.66);const legend=d3.select(root).select('.sei-legend').selectAll('span').data(order).join('span');legend.html(d=>`<i class="sei-dot" style="background:${colors[d]}"></i>${labels[d]}`)}
  function deltas(){const s=row('self_context'),b=row('soft_eikonal'),items=[['Learning AUC',s.learning_auc-b.learning_auc],['Held-out score',s.score-b.score],['Tail score',s.tail_score-b.tail_score]];const scale=d3.scaleLinear([0,.12],[0,100]);const rows=d3.select(root).select('.sei-drows').selectAll('.sei-drow').data(items).join('div').attr('class','sei-drow');rows.html(d=>`<span>${d[0]}</span><span class="sei-track"><i class="sei-bar ${d[1]>=0?'sei-pos':'sei-neg'}" style="display:block;width:${scale(Math.abs(d[1]))}%"></i></span><b>${d3.format('+.3f')(d[1])}</b>`)}
  function render(){buttons.classed('active',d=>d===variant);geometry();learning();deltas()} render();
  })();</script>
</div>'''


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(TEMPLATE.replace("__DATA__", json.dumps(compact_data(), separators=(",", ":"))))
    print(OUT)
    print(OUT.stat().st_size)


if __name__ == "__main__":
    main()
