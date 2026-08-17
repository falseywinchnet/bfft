#!/usr/bin/env python3
"""Derive evidence-MAP and evidence/frontier diagnostics from saved runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw


def correlation(xs, ys):
    mx, my = sum(xs)/len(xs), sum(ys)/len(ys)
    covariance = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    denominator = math.sqrt(sum((x-mx)**2 for x in xs) * sum((y-my)**2 for y in ys))
    return covariance/denominator if denominator else float("nan")


def mean(values):
    return sum(values)/len(values)


def evidence_plot(path, points_by_fraction):
    width, height = 900, 620; left, top, right, bottom = 95, 50, 30, 80
    image = Image.new("RGB", (width, height), (249, 248, 244)); draw = ImageDraw.Draw(image)
    all_x = [math.log10(p[0]) for points in points_by_fraction.values() for p in points]
    xlo, xhi = min(all_x)-.08, max(all_x)+.08
    draw.rectangle((left, top, width-right, height-bottom), outline=(60, 60, 60))
    for value in [0, .2, .4, .6, .8, 1]:
        yy = height-bottom-value*(height-bottom-top)
        draw.line((left, yy, width-right, yy), fill=(222, 222, 217))
        draw.text((55, yy-7), f"{value:.1f}", fill=(45, 45, 45))
    colors = {.5: (8, 128, 156), .3: (225, 127, 35)}
    for fraction, points in points_by_fraction.items():
        transformed = [(math.log10(loss), frontier) for loss, frontier in points]
        for x, y in transformed:
            xx = left+(x-xlo)/(xhi-xlo)*(width-right-left)
            yy = height-bottom-y*(height-bottom-top)
            draw.ellipse((xx-5, yy-5, xx+5, yy+5), fill=colors[fraction])
        mx = mean([p[0] for p in transformed]); my = mean([p[1] for p in transformed])
        covariance = sum((x-mx)*(y-my) for x, y in transformed)
        variance = sum((x-mx)**2 for x, _ in transformed)
        slope = covariance/variance if variance else 0; intercept = my-slope*mx
        y0, y1 = intercept+slope*xlo, intercept+slope*xhi
        draw.line((left, height-bottom-y0*(height-bottom-top), width-right,
                   height-bottom-y1*(height-bottom-top)), fill=colors[fraction], width=4)
    draw.text((left, 15), "Does observed relational evidence rank frontier hypotheses?", fill=(20, 20, 20))
    draw.text((340, height-35), "log10 held-relation loss (lower is better)", fill=(20, 20, 20))
    draw.text((8, top), "first-bin accuracy", fill=(20, 20, 20))
    draw.rectangle((left+12, top+12, left+24, top+24), fill=colors[.5]); draw.text((left+31, top+10), "50% observed", fill=colors[.5])
    draw.rectangle((left+150, top+12, left+162, top+24), fill=colors[.3]); draw.text((left+169, top+10), "30% observed", fill=colors[.3])
    image.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    payload = json.loads((args.results / "runs.json").read_text())
    groups, points = [], {.5: [], .3: []}
    for record in payload["hypothesis_records"]:
        losses = [item["evidence_loss"] for item in record["hypotheses"]]
        tests = record["individual_test_results"]
        first = [item["first_bin_accuracy"] for item in tests]
        map_index = min(range(len(losses)), key=losses.__getitem__)
        oracle_index = max(range(len(tests)), key=lambda i: (tests[i]["survival_bins_at_80pct"],
                                                              tests[i]["first_bin_accuracy"],
                                                              tests[i]["frontier5_accuracy"]))
        fraction = float(record["train_fraction"])
        points[fraction].extend(zip(losses, first))
        groups.append({"train_fraction": fraction, "seed": record["seed"],
                       "map_index": map_index, "oracle_index": oracle_index,
                       "map_matches_oracle": map_index == oracle_index,
                       "within_seed_loss_frontier_correlation": correlation(losses, first),
                       "map_test_result": tests[map_index], "oracle_test_result": tests[oracle_index]})
    summary = {}
    for fraction in [.5, .3]:
        selected = [group for group in groups if group["train_fraction"] == fraction]
        summary[str(fraction)] = {
            "map_first_bin_mean": mean([g["map_test_result"]["first_bin_accuracy"] for g in selected]),
            "map_frontier5_mean": mean([g["map_test_result"]["frontier5_accuracy"] for g in selected]),
            "map_survival": [g["map_test_result"]["survival_bins_at_80pct"] for g in selected],
            "map_matches_oracle_count": sum(g["map_matches_oracle"] for g in selected),
            "mean_within_seed_loss_frontier_correlation": mean([g["within_seed_loss_frontier_correlation"] for g in selected]),
            "pooled_loss_frontier_correlation": correlation([p[0] for p in points[fraction]],
                                                               [p[1] for p in points[fraction]]),
        }
    (args.results / "posterior_analysis.json").write_text(json.dumps({"groups": groups, "summary": summary}, indent=2))
    evidence_plot(args.results / "evidence_vs_frontier.png", points)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
