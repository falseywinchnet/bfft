#!/usr/bin/env python3
"""Build the gradient-zonotope battery and long-horizon sparse-sine viewer."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BATTERY = ROOT / "ML_experiment/results_gradient_zonotope_battery/results.json"
CONTROL = ROOT / "ML_experiment/results_gradient_zonotope_battery/compute_control.json"
SPARSE = ROOT / "ML_experiment/results_sparse_sine_gradient_zonotope"
INLINE = Path(
    "/Users/ultimussecundai/.codex/visualizations/2026/08/16/"
    "01a00875-c9dc-7e01-8e13-444500eb3edf/"
    "gradient-zonotope-battery.html"
)
STANDALONE = ROOT / "ML_experiment/gradient_zonotope_battery.html"


def load_data():
    battery = json.loads(BATTERY.read_text())["runs"]
    control = json.loads(CONTROL.read_text())["runs"]
    grouped = {}
    for row in battery:
        grouped.setdefault(row["task"], {})[row["variant"]] = row
    compute = {row["task"]: row for row in control}
    tasks = []
    for task, variants in grouped.items():
        baseline = variants["self_context"]
        cold = variants["transport_cold"]
        warm = variants["transport_warm"]
        tasks.append({
            "task": task,
            "cold_delta": cold["score"] - compute[task]["score"],
            "warm_delta": warm["score"] - compute[task]["score"],
            "cold_auc": cold["learning_auc"] - baseline["learning_auc"],
            "warm_auc": warm["learning_auc"] - baseline["learning_auc"],
            "control": compute[task]["score"],
            "cold": cold["score"],
            "warm": warm["score"],
        })
    tasks.sort(key=lambda row: max(row["cold_delta"], row["warm_delta"]), reverse=True)

    baseline = json.loads((SPARSE / "baseline_seed2.json").read_text())["runs"][0]
    cold = json.loads((SPARSE / "cold_seed2.json").read_text())["runs"][0]
    warm = json.loads((SPARSE / "warm_seed2.json").read_text())["runs"][0]
    probe = baseline["probe"]
    keep = range(0, len(probe["x"]), 8)
    curve = [{
        "x": probe["x"][index],
        "truth": probe["truth"][index],
        "baseline": probe["prediction"][index],
        "cold": cold["probe"]["prediction"][index],
        "warm": warm["probe"]["prediction"][index],
    } for index in keep]
    return {"tasks": tasks, "curve": curve}


FRAGMENT = r'''
<div id="gradient-zonotope-battery-v1">
  <h2>Optimizer-transported gradient zonotope</h2>
  <div class="legend" aria-label="Series legend"></div>
  <section class="battery-grid">
    <div class="plot"><h3>Final score versus 1,200-gradient self-context</h3><svg id="gz-final"></svg></div>
    <div class="plot"><h3>Learning AUC versus 400-step self-context</h3><svg id="gz-auc"></svg></div>
  </section>
  <section class="curve-grid">
    <div class="plot"><h3>Sparse sine · twenty unseen periods · output clipped at ±3</h3><svg id="gz-curve-linear"></svg></div>
    <div class="plot"><h3>Sparse sine far tail · symmetric-log output scale</h3><svg id="gz-curve-log"></svg></div>
  </section>
  <div class="tooltip" role="tooltip"></div>
</div>
<style>
#gradient-zonotope-battery-v1 { color: var(--foreground); font-family: ui-sans-serif, system-ui, sans-serif; position: relative; }
#gradient-zonotope-battery-v1 h2 { font-size: 17px; margin: 0 0 5px; font-weight: 650; }
#gradient-zonotope-battery-v1 h3 { font-size: 12px; margin: 0 0 3px 154px; font-weight: 650; }
#gradient-zonotope-battery-v1 .legend { display: flex; flex-wrap: wrap; gap: 4px 15px; margin: 0 0 8px 154px; }
#gradient-zonotope-battery-v1 .legend button { appearance: none; border: 0; background: transparent; color: var(--foreground); padding: 2px 0; font: inherit; font-size: 12px; cursor: pointer; }
#gradient-zonotope-battery-v1 .legend button[aria-pressed="false"] { opacity: .35; }
#gradient-zonotope-battery-v1 .swatch { display: inline-block; width: 18px; height: 3px; margin-right: 5px; vertical-align: middle; }
#gradient-zonotope-battery-v1 .battery-grid, #gradient-zonotope-battery-v1 .curve-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
#gradient-zonotope-battery-v1 .curve-grid { margin-top: 12px; }
#gradient-zonotope-battery-v1 svg { width: 100%; display: block; overflow: visible; }
#gradient-zonotope-battery-v1 text { fill: var(--foreground); font-size: 12px; }
#gradient-zonotope-battery-v1 .axis path, #gradient-zonotope-battery-v1 .axis line { stroke: var(--border); }
#gradient-zonotope-battery-v1 rect[data-chart-frame] { fill: transparent; stroke: var(--border); }
#gradient-zonotope-battery-v1 .tooltip { position: absolute; pointer-events: none; opacity: 0; background: var(--popover); color: var(--popover-foreground); border: 1px solid var(--border); padding: 6px 8px; font-size: 12px; z-index: 10; }
@media (max-width: 700px) {
  #gradient-zonotope-battery-v1 .battery-grid, #gradient-zonotope-battery-v1 .curve-grid { grid-template-columns: 1fr; }
  #gradient-zonotope-battery-v1 h3, #gradient-zonotope-battery-v1 .legend { margin-left: 110px; }
}
</style>
<script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
<script>
(() => {
  const root = document.getElementById("gradient-zonotope-battery-v1");
  const payload = __DATA__;
  const tooltip = d3.select(root).select(".tooltip");
  const definitions = [
    {key:"cold", label:"Cold transported", color:"var(--viz-series-2)"},
    {key:"warm", label:"Warm transported", color:"var(--viz-series-4)"},
    {key:"baseline", label:"Self-context", color:"var(--viz-series-1)"},
    {key:"truth", label:"Truth", color:"var(--foreground)", dash:"5 3"}
  ];
  const enabled = new Map(definitions.map(d => [d.key, true]));
  const redraw = [];
  const legend = d3.select(root).select(".legend");
  definitions.forEach(d => {
    const button = legend.append("button").attr("type","button").attr("aria-pressed","true");
    button.append("span").attr("class","swatch").style("background",d.color);
    button.append("span").text(d.label);
    button.on("click", () => {
      enabled.set(d.key, !enabled.get(d.key));
      button.attr("aria-pressed", String(enabled.get(d.key)));
      redraw.forEach(draw => draw());
    });
  });

  function deltaChart(selector, coldKey, warmKey, axisTitle) {
    const svg = d3.select(root).select(selector);
    const holder = svg.node().parentElement;
    const draw = () => {
      const outerWidth = Math.max(260, holder.getBoundingClientRect().width);
      const height = 620, margin = {top:8,right:outerWidth<420?10:18,bottom:46,left:outerWidth<420?110:154};
      const width = outerWidth-margin.left-margin.right, innerHeight=height-margin.top-margin.bottom;
      svg.attr("viewBox",`0 0 ${outerWidth} ${height}`).selectAll("*").remove();
      const g=svg.append("g").attr("transform",`translate(${margin.left},${margin.top})`);
      const visible=[];
      payload.tasks.forEach(d => {
        if(enabled.get("cold")) visible.push(d[coldKey]);
        if(enabled.get("warm")) visible.push(d[warmKey]);
      });
      visible.push(0);
      const extent=d3.extent(visible), pad=Math.max(.005,(extent[1]-extent[0])*.08);
      const x=d3.scaleLinear().domain([extent[0]-pad,extent[1]+pad]).range([0,width]);
      const y=d3.scaleBand().domain(payload.tasks.map(d=>d.task)).range([0,innerHeight]).padding(.25);
      g.append("rect").attr("data-chart-frame","").attr("width",width).attr("height",innerHeight);
      g.append("line").attr("x1",x(0)).attr("x2",x(0)).attr("y2",innerHeight).attr("stroke","var(--foreground)").attr("opacity",.45);
      g.append("g").attr("class","axis").call(d3.axisLeft(y).tickSize(0));
      g.append("g").attr("class","axis").attr("transform",`translate(0,${innerHeight})`).call(d3.axisBottom(x).ticks(4).tickFormat(d3.format("+.2f")));
      const specs=[{key:"cold",field:coldKey,color:"var(--viz-series-2)",dy:-3},{key:"warm",field:warmKey,color:"var(--viz-series-4)",dy:3}];
      specs.forEach(s => {
        if(!enabled.get(s.key)) return;
        g.selectAll(`circle.${s.key}-${coldKey}`).data(payload.tasks).enter().append("circle")
          .attr("cx",d=>x(d[s.field])).attr("cy",d=>y(d.task)+y.bandwidth()/2+s.dy).attr("r",4).attr("fill",s.color)
          .append("title").text(d=>`${d.task}: ${d[s.field]>=0?"+":""}${d[s.field].toFixed(4)}`);
      });
      svg.append("text").attr("class","axis-title").attr("data-axis","x").attr("x",margin.left+width/2).attr("y",height-5).attr("text-anchor","middle").text(axisTitle);
      svg.append("text").attr("class","axis-title").attr("data-axis","y").attr("transform","rotate(-90)").attr("x",-(margin.top+innerHeight/2)).attr("y",13).attr("text-anchor","middle").text("problem");
    };
    new ResizeObserver(draw).observe(holder); redraw.push(draw); draw();
  }
  deltaChart("#gz-final","cold_delta","warm_delta","score difference · positive beats 1,200-gradient control");
  deltaChart("#gz-auc","cold_auc","warm_auc","learning-AUC difference · positive learns sooner per accepted step");

  function curveChart(selector, farOnly, symlog) {
    const svg=d3.select(root).select(selector), holder=svg.node().parentElement;
    const draw=()=>{
      const outerWidth=Math.max(260,holder.getBoundingClientRect().width),height=300,margin={top:8,right:10,bottom:46,left:outerWidth<420?52:64};
      const width=outerWidth-margin.left-margin.right,innerHeight=height-margin.top-margin.bottom;
      svg.attr("viewBox",`0 0 ${outerWidth} ${height}`).selectAll("*").remove();
      const g=svg.append("g").attr("transform",`translate(${margin.left},${margin.top})`);
      const data=payload.curve.filter(d=>!farOnly||d.x>=1);
      const x=d3.scaleLinear().domain(farOnly?[1,3]:[0,3]).range([0,width]);
      let y;
      if(symlog){
        const values=[]; data.forEach(d=>definitions.forEach(s=>values.push(d[s.key])));
        const extent=d3.extent(values),bound=Math.max(Math.abs(extent[0]),Math.abs(extent[1]),1.1);
        y=d3.scaleSymlog().constant(1).domain([-bound,bound]).range([innerHeight,0]);
      }else y=d3.scaleLinear().domain([-3,3]).range([innerHeight,0]);
      g.append("rect").attr("data-chart-frame","").attr("width",width).attr("height",innerHeight);
      if(!farOnly) g.append("rect").attr("x",x(1)).attr("width",x(3)-x(1)).attr("height",innerHeight).attr("fill","var(--border)").attr("opacity",.10);
      g.append("line").attr("x1",x(1)).attr("x2",x(1)).attr("y2",innerHeight).attr("stroke","var(--foreground)").attr("stroke-dasharray","3 3").attr("opacity",.55);
      g.append("g").attr("class","axis").attr("transform",`translate(0,${innerHeight})`).call(d3.axisBottom(x).ticks(4));
      g.append("g").attr("class","axis").call(d3.axisLeft(y).ticks(5));
      const paths=new Map();
      definitions.forEach(s=>paths.set(s.key,g.append("path").datum(data).attr("fill","none").attr("stroke",s.color).attr("stroke-width",s.key==="truth"?1.3:1.7).attr("stroke-dasharray",s.dash||null)));
      const line=key=>d3.line().x(d=>x(d.x)).y(d=>y(symlog?d[key]:Math.max(-3,Math.min(3,d[key]))));
      const guide=g.append("line").attr("data-chart-hover-guide","").attr("y2",innerHeight).attr("stroke","var(--foreground)").attr("opacity",0);
      const markers=new Map(definitions.map(s=>[s.key,g.append("circle").attr("data-chart-hover-marker","").attr("r",3).attr("fill",s.color).attr("opacity",0)]));
      g.append("rect").attr("data-chart-hit","").attr("data-chart-hover-overlay","cross-series").attr("width",width).attr("height",innerHeight).attr("fill","transparent")
        .on("pointermove",event=>{
          const px=d3.pointer(event)[0],xv=x.invert(px),i=d3.bisector(d=>d.x).center(data,xv),d=data[i];
          guide.attr("x1",px).attr("x2",px).attr("opacity",.5); const rows=[];
          definitions.filter(s=>enabled.get(s.key)).forEach(s=>{markers.get(s.key).attr("cx",x(d.x)).attr("cy",y(symlog?d[s.key]:Math.max(-3,Math.min(3,d[s.key])))).attr("opacity",1);rows.push(`${s.label}: ${d[s.key].toFixed(3)}`)});
          const box=root.getBoundingClientRect();tooltip.style("opacity",1).style("left",`${event.clientX-box.left+10}px`).style("top",`${event.clientY-box.top+10}px`).html(`x ${d.x.toFixed(3)}<br>${rows.join("<br>")}`);
        }).on("pointerleave",()=>{guide.attr("opacity",0);markers.forEach(m=>m.attr("opacity",0));tooltip.style("opacity",0)});
      definitions.forEach(s=>paths.get(s.key).attr("d",line(s.key)).style("display",enabled.get(s.key)?null:"none"));
      svg.append("text").attr("class","axis-title").attr("data-axis","x").attr("x",margin.left+width/2).attr("y",height-5).attr("text-anchor","middle").text("coordinate x · observation ends at 1");
      svg.append("text").attr("class","axis-title").attr("data-axis","y").attr("transform","rotate(-90)").attr("x",-(margin.top+innerHeight/2)).attr("y",13).attr("text-anchor","middle").text(symlog?"output · symlog":"output · clipped");
    };
    new ResizeObserver(draw).observe(holder);redraw.push(draw);draw();
  }
  curveChart("#gz-curve-linear",false,false);
  curveChart("#gz-curve-log",true,true);
})();
</script>
'''


def main():
    data = json.dumps(load_data(), separators=(",", ":"))
    fragment = FRAGMENT.replace("__DATA__", data)
    INLINE.parent.mkdir(parents=True, exist_ok=True)
    INLINE.write_text(fragment)
    STANDALONE.write_text(
        "<!doctype html><html><head><meta charset=\"utf-8\"><meta name=\"viewport\" "
        "content=\"width=device-width,initial-scale=1\"><title>Gradient zonotope battery</title>"
        "<style>:root{--foreground:#171717;--border:#c8c8c8;--popover:#fff;"
        "--popover-foreground:#171717;--viz-series-1:#2878b5;--viz-series-2:#e07a24;"
        "--viz-series-3:#6a9f58;--viz-series-4:#9b59b6;--viz-series-5:#c84f55;"
        "--viz-series-6:#4f9d9d}body{margin:24px;max-width:1600px}</style></head><body>"
        + fragment + "</body></html>"
    )
    print(INLINE)
    print(STANDALONE)


if __name__ == "__main__":
    main()
