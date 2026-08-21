#!/usr/bin/env python3
"""Build the compact in-conversation witnessed-descent comparison."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "ML_experiment/results_sparse_sine_witness_full/results.json"
WITNESS = (
    ROOT / "ML_experiment/results_sparse_sine_witness_resident/seeds01.json",
    ROOT / "ML_experiment/results_sparse_sine_witness_resident/seed2.json",
)
OUT = Path(
    "/Users/ultimussecundai/.codex/visualizations/2026/08/16/"
    "01a00875-c9dc-7e01-8e13-444500eb3edf/"
    "sparse-sine-witness-descent.html"
)


def build_data():
    baseline = json.loads(BASE.read_text())["runs"]
    witnessed = []
    for path in WITNESS:
        witnessed.extend(json.loads(path.read_text())["runs"])
    lookup = {(row["method"], row["seed"]): row for row in baseline + witnessed}
    seeds = []
    for seed in range(3):
        path = lookup[("baseline_path_matched", seed)]
        compute = lookup[("baseline_compute_matched", seed)]
        witness = lookup[("zonotopic_witness", seed)]
        source = path["probe"]
        keep = range(0, len(source["x"]), 4)
        curve = [{
            "x": source["x"][index],
            "truth": source["truth"][index],
            "path": path["probe"]["prediction"][index],
            "compute": compute["probe"]["prediction"][index],
            "witness": witness["probe"]["prediction"][index],
        } for index in keep]
        seeds.append({
            "seed": seed,
            "curve": curve,
            "metrics": {
                "path": {
                    "minimum": path["minimum_segment_r2"],
                    "tail": path["sparse_tail_r2"],
                    "seconds": path["seconds"],
                    "gradients": path["gradient_evaluations"],
                },
                "compute": {
                    "minimum": compute["minimum_segment_r2"],
                    "tail": compute["sparse_tail_r2"],
                    "seconds": compute["seconds"],
                    "gradients": compute["gradient_evaluations"],
                },
                "witness": {
                    "minimum": witness["minimum_segment_r2"],
                    "tail": witness["sparse_tail_r2"],
                    "seconds": witness["seconds"],
                    "gradients": witness["gradient_evaluations"],
                },
            },
        })
    return {"seeds": seeds}


FRAGMENT = r'''
<div id="sparse-sine-witness-v1">
  <h2>Resident witness descent on the 96× thinned sine</h2>
  <div class="legend" aria-label="Series legend"></div>
  <div class="plots"></div>
  <div class="metric-plots"></div>
  <div class="tooltip" role="tooltip"></div>
</div>
<style>
#sparse-sine-witness-v1 { color: var(--foreground); font-family: ui-sans-serif, system-ui, sans-serif; }
#sparse-sine-witness-v1 h2 { font-size: 17px; font-weight: 650; margin: 0 0 6px; }
#sparse-sine-witness-v1 .legend { display: flex; flex-wrap: wrap; gap: 5px 15px; margin: 0 0 8px 58px; }
#sparse-sine-witness-v1 .legend button { appearance: none; background: transparent; border: 0; color: var(--foreground); padding: 2px 0; font: inherit; font-size: 12px; cursor: pointer; }
#sparse-sine-witness-v1 .legend button[aria-pressed="false"] { opacity: .38; }
#sparse-sine-witness-v1 .swatch { display: inline-block; width: 18px; height: 3px; margin-right: 5px; vertical-align: middle; }
#sparse-sine-witness-v1 .plots { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }
#sparse-sine-witness-v1 .metric-plots { display: grid; grid-template-columns: 1.5fr 1fr; gap: 14px; margin-top: 12px; }
#sparse-sine-witness-v1 .panel-title { font-size: 12px; font-weight: 650; margin: 0 0 2px 58px; }
#sparse-sine-witness-v1 svg { width: 100%; display: block; overflow: visible; }
#sparse-sine-witness-v1 text { fill: var(--foreground); font-size: 12px; }
#sparse-sine-witness-v1 .axis path, #sparse-sine-witness-v1 .axis line { stroke: var(--border); }
#sparse-sine-witness-v1 rect[data-chart-frame] { fill: transparent; stroke: var(--border); }
#sparse-sine-witness-v1 .tooltip { position: absolute; pointer-events: none; opacity: 0; background: var(--popover); color: var(--popover-foreground); border: 1px solid var(--border); padding: 6px 8px; font-size: 12px; z-index: 20; }
@media (max-width: 700px) {
  #sparse-sine-witness-v1 .plots, #sparse-sine-witness-v1 .metric-plots { grid-template-columns: 1fr; }
  #sparse-sine-witness-v1 .legend { margin-left: 52px; }
}
</style>
<script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
<script>
(() => {
  const root = document.getElementById("sparse-sine-witness-v1");
  const payload = __DATA__;
  const series = [
    {key:"truth", label:"Truth", color:"var(--foreground)", dash:"5 3"},
    {key:"path", label:"Self-context · 1k gradients", color:"var(--viz-series-1)"},
    {key:"compute", label:"Self-context · 3k gradients", color:"var(--viz-series-2)"},
    {key:"witness", label:"Resident witness · 3k gradients", color:"var(--viz-series-4)"}
  ];
  const enabled = new Map(series.map(s => [s.key, true]));
  const tooltip = d3.select(root).select(".tooltip");
  const renderers = [];

  const legend = d3.select(root).select(".legend");
  series.forEach(s => {
    const button = legend.append("button").attr("type", "button").attr("aria-pressed", "true");
    button.append("span").attr("class", "swatch").style("background", s.color);
    button.append("span").text(s.label);
    button.on("click", () => {
      enabled.set(s.key, !enabled.get(s.key));
      button.attr("aria-pressed", String(enabled.get(s.key)));
      renderers.forEach(draw => draw());
    });
  });

  function curvePanel(seedData) {
    const holder = d3.select(root).select(".plots").append("div");
    holder.append("div").attr("class", "panel-title").text(`Seed ${seedData.seed} · observed support and continuation`);
    const svg = holder.append("svg").attr("viewBox", "0 0 360 250");
    const margin = {top:8, right:10, bottom:43, left:58};
    const width = 360 - margin.left - margin.right, height = 250 - margin.top - margin.bottom;
    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);
    const x = d3.scaleLinear().domain([0, 1.5]).range([0, width]);
    const y = d3.scaleLinear().domain([-3, 3]).range([height, 0]);
    g.append("rect").attr("data-chart-frame", "").attr("width", width).attr("height", height);
    g.append("rect").attr("x", x(1)).attr("width", x(1.5)-x(1)).attr("height", height).attr("fill", "var(--border)").attr("opacity", .12);
    g.append("line").attr("x1", x(1)).attr("x2", x(1)).attr("y2", height).attr("stroke", "var(--foreground)").attr("stroke-dasharray", "3 3").attr("opacity", .55);
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${height})`).call(d3.axisBottom(x).ticks(4));
    g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(5));
    svg.append("text").attr("class", "axis-title").attr("data-axis", "x").attr("x", margin.left + width/2).attr("y", 245).attr("text-anchor", "middle").text("coordinate x · shaded region is unseen");
    svg.append("text").attr("class", "axis-title").attr("data-axis", "y").attr("transform", "rotate(-90)").attr("x", -(margin.top + height/2)).attr("y", 14).attr("text-anchor", "middle").text("output · clipped at ±3");
    const line = key => d3.line().x(d => x(d.x)).y(d => y(Math.max(-3, Math.min(3, d[key]))));
    const paths = new Map();
    series.forEach(s => paths.set(s.key, g.append("path").datum(seedData.curve).attr("fill", "none").attr("stroke", s.color).attr("stroke-width", s.key === "truth" ? 1.4 : 1.7).attr("stroke-dasharray", s.dash || null)));
    const guide = g.append("line").attr("data-chart-hover-guide", "").attr("y2", height).attr("stroke", "var(--foreground)").attr("opacity", 0);
    const markers = new Map(series.map(s => [s.key, g.append("circle").attr("data-chart-hover-marker", "").attr("r", 3).attr("fill", s.color).attr("opacity", 0)]));
    g.append("rect").attr("data-chart-hit", "").attr("data-chart-hover-overlay", "cross-series").attr("width", width).attr("height", height).attr("fill", "transparent")
      .on("pointermove", event => {
        const px = d3.pointer(event)[0], xv = x.invert(px);
        guide.attr("x1", px).attr("x2", px).attr("opacity", .5);
        const rows = [];
        series.filter(s => enabled.get(s.key)).forEach(s => {
          const values = seedData.curve, i = d3.bisector(d => d.x).center(values, xv), d = values[i];
          markers.get(s.key).attr("cx", x(d.x)).attr("cy", y(Math.max(-3, Math.min(3, d[s.key])))).attr("opacity", 1);
          rows.push(`${s.label}: ${d[s.key].toFixed(3)}`);
        });
        const box = root.getBoundingClientRect();
        tooltip.style("opacity", 1).style("left", `${event.clientX-box.left+10}px`).style("top", `${event.clientY-box.top+10}px`).html(`x ${xv.toFixed(3)}<br>${rows.join("<br>")}`);
      }).on("pointerleave", () => { guide.attr("opacity", 0); markers.forEach(m => m.attr("opacity", 0)); tooltip.style("opacity", 0); });
    const draw = () => series.forEach(s => paths.get(s.key).attr("d", line(s.key)).style("display", enabled.get(s.key) ? null : "none"));
    renderers.push(draw); draw();
  }

  payload.seeds.forEach(curvePanel);

  function metricPanel(metric, title, domain, format) {
    const holder = d3.select(root).select(".metric-plots").append("div");
    holder.append("div").attr("class", "panel-title").text(title);
    const svg = holder.append("svg").attr("viewBox", "0 0 470 230");
    const margin = {top:8, right:16, bottom:48, left:58};
    const width = 470-margin.left-margin.right, height = 230-margin.top-margin.bottom;
    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);
    const methods = ["path","compute","witness"];
    const labels = {path:"Self 1k",compute:"Self 3k",witness:"Witness 3k"};
    const colors = {path:"var(--viz-series-1)",compute:"var(--viz-series-2)",witness:"var(--viz-series-4)"};
    const x0 = d3.scaleBand().domain(methods).range([0,width]).padding(.2);
    const x1 = d3.scalePoint().domain([0,1,2]).range([12,x0.bandwidth()-12]);
    const y = d3.scaleLinear().domain(domain).nice().range([height,0]);
    g.append("rect").attr("data-chart-frame", "").attr("width", width).attr("height", height);
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${height})`).call(d3.axisBottom(x0).tickFormat(d => labels[d]));
    g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(5).tickFormat(format));
    methods.forEach(method => {
      const points = payload.seeds.map(seed => ({seed:seed.seed, value:seed.metrics[method][metric]}));
      g.append("path").datum(points).attr("fill", "none").attr("stroke", colors[method]).attr("stroke-width", 1.4).attr("opacity", .45)
        .attr("d", d3.line().x(d => x0(method)+x1(d.seed)).y(d => y(d.value)));
      g.selectAll(`circle.${method}-${metric}`).data(points).enter().append("circle")
        .attr("cx", d => x0(method)+x1(d.seed)).attr("cy", d => y(d.value)).attr("r", 4).attr("fill", colors[method])
        .append("title").text(d => `seed ${d.seed}: ${format(d.value)}`);
    });
    svg.append("text").attr("class", "axis-title").attr("data-axis", "x").attr("x", margin.left+width/2).attr("y", 226).attr("text-anchor", "middle").text("training rule · number is gradient evaluations");
    svg.append("text").attr("class", "axis-title").attr("data-axis", "y").attr("transform", "rotate(-90)").attr("x", -(margin.top+height/2)).attr("y", 14).attr("text-anchor", "middle").text(metric === "minimum" ? "worst observed-period R²" : "wall time (seconds)");
  }
  metricPanel("minimum", "Worst observed period: basin robustness", [0.60, 1.01], d3.format(".2f"));
  metricPanel("seconds", "Cost of selecting one surviving trajectory", [0, 58], d => `${d.toFixed(0)}s`);
})();
</script>
'''


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(FRAGMENT.replace("__DATA__", json.dumps(build_data(), separators=(",", ":"))))
    print(OUT)


if __name__ == "__main__":
    main()
