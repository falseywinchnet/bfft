"""Summarize the saved native boundary comparison; no transport runs locally."""
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def main():
    folder = Path(__file__).with_name("child_boundary_m4")
    rows = json.loads((folder / "child_boundary_screen.json").read_text())["frames"]
    summary = []
    fig, axes = plt.subplots(4, 3, figsize=(13, 13), layout="constrained")
    for row, name in enumerate(dict.fromkeys(r["scene"] for r in rows)):
        old = np.asarray(Image.open(folder / f"child_boundary_{name}_0.ppm")).astype(float)
        new = np.asarray(Image.open(folder / f"child_boundary_{name}_1.ppm")).astype(float)
        difference = np.abs(new - old)
        record = dict(scene=name, mean_abs_channel_difference_255=float(difference.mean()),
                      max_channel_difference_255=float(difference.max()),
                      changed_pixels=int(np.any(difference > 0, axis=2).sum()))
        for mode in (0, 1):
            selected = [r for r in rows if r["scene"] == name and r["mode"] == mode]
            record["on" if mode else "off"] = dict(
                cold_frame_ms=selected[0]["render_ms"],
                warm_median_ms=statistics.median(r["render_ms"] for r in selected[1:]),
                build_ms=selected[0]["build_ms"], quadrature=selected[0]["quadrature"],
                secondary=selected[0]["secondary"],
                resident_unit_responses=selected[-1]["resident_unit_responses"],
                payload_bytes_excluding_map_overhead=selected[-1]["unit_payload_bytes"],
                warm_new_unit_responses=sum(r["new_unit_responses"] for r in selected[1:]))
        summary.append(record)
        titles = ("Existing native route", "Child boundary route", "Max channel difference (0–32/255)")
        for col, values in enumerate((old / 255, new / 255, difference.max(axis=2))):
            ax = axes[row, col]
            ax.imshow(values, **(dict(cmap="inferno", vmin=0, vmax=32) if col == 2 else {}))
            ax.axis("off")
            ax.set_title(f"{name}\n{titles[col]}", fontsize=10)
    fig.savefig(folder / "comparison.png", dpi=150)
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
