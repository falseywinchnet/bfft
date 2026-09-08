"""Render canonical intensity fingerprints from a direct-vowel inventory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inventory", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.inventory.read_text(encoding="utf-8"))
    labels = list(data["symbols"])
    voices = list(data["voices"])
    records = {
        (record["voice"], record["label"]): record
        for record in data["templates"]
    }
    fig, axes = plt.subplots(
        len(voices), len(labels), figsize=(2.0 * len(labels), 2.8 * len(voices)),
        constrained_layout=True,
    )
    for row, voice in enumerate(voices):
        for column, label in enumerate(labels):
            axis = axes[row, column]
            fingerprint = records[(voice, label)]["fingerprint"]
            intensity = np.asarray(fingerprint["intensity"]).reshape(64, 32)
            axis.imshow(intensity, origin="upper", aspect="auto", cmap="magma")
            axis.set_title(f"{label}\nanchor {fingerprint['anchor_row']}", fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])
            if column == 0:
                axis.set_ylabel(voice)
    fig.suptitle("Direct Apple-phoneme IRODFT/Meyer fingerprints", fontsize=16)
    fig.savefig(args.out, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
