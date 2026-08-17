#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUMMARY = json.loads((HERE / "results_confirm/summary.json").read_text())
PROBES = json.loads((HERE / "results_confirm/probes.json").read_text())["probes"]
TARGET = Path("/Users/ultimussecundai/.codex/visualizations/2026/08/16/01a00875-c9dc-7e01-8e13-444500eb3edf/self-context-superset.html")
ADDONS = {"self_context_hard", "self_context_iterated", "self_context_uncertainty", "self_context_secant", "self_context_chart"}


def field(row):
    old = row["size"]; indices = list(range(0, old, 2)); flat = lambda key: [row[key][y * old + x] for y in indices for x in indices]
    return {"variant": row["variant"], "size": len(indices), "truth": flat("truth"), "prediction": flat("prediction")}


def curve(row):
    take = range(0, len(row["x"]), 10)
    return {"variant": row["variant"], "x": [round(row["x"][i], 4) for i in take],
            "truth": [round(row["truth"][i], 5) for i in take],
            "prediction": [round(row["prediction"][i], 5) for i in take]}


def main():
    overall = [{k: row[k] for k in ("variant", "validation_score_delta", "score_delta", "tail_score_delta", "learning_auc_delta")}
               for row in SUMMARY["overall"] if row["variant"] in ADDONS]
    by_task = [{"task": row["task"], "tail_score_delta": row["tail_score_delta"]}
               for row in SUMMARY["by_task"] if row["variant"] == "self_context_chart"]
    spiral = [field(row) for row in PROBES if row["task"] == "spiral"]
    multiscale = [curve(row) for row in PROBES if row["task"] == "multiscale_1d"]
    data = json.dumps({"overall": overall, "byTask": by_task, "spiral": spiral, "multiscale": multiscale}, separators=(",", ":"))
    source = TARGET.read_text(); source = source.replace("__DATA__", data)
    TARGET.write_text(source); print(TARGET); print(TARGET.stat().st_size)


if __name__ == "__main__": main()
