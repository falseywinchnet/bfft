#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def shrink_field(plot, target=41):
    source = plot["resolution"]
    picks = [round(index * (source - 1) / (target - 1)) for index in range(target)]
    indices = [row * source + column for row in picks for column in picks]
    return {"limits": plot["limits"], "resolution": target,
            **{name: [round(plot[name][index], 5) for index in indices]
               for name in ("truth", "mlp", "soft")}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("analysis", type=Path); parser.add_argument("probes", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    analysis = json.loads(args.analysis.read_text())
    probes = json.loads(args.probes.read_text())["tasks"]
    compact = {
        "comparisons": analysis["comparisons"],
        "curves": {name: {key: [round(item, 5) for item in value[::2]]
                           if isinstance(value, list) and len(value) > 100 else value
                           for key, value in probes[name]["plot"].items()}
                   for name in ("multiscale_1d", "chirp_1d", "localized_steps_1d", "fourier_mix_1d")},
        "fields": {name: shrink_field(probes[name]["plot"]) for name in ("spiral", "checkerboard")},
    }
    template = Path(__file__).with_name("visualization.template.html").read_text()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(template.replace("__DATA__", json.dumps(compact, separators=(",", ":"))))
    print(json.dumps({"out": str(args.out), "bytes": args.out.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
