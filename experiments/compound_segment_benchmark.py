"""Same-budget benchmark for compound versus reconstruction-cell IDs."""

from __future__ import annotations

import argparse

import numpy as np

from bfft.effects import srgb_to_lab
from experiments.segmenting_v3 import SegmentingV3Config, build_segmenting_v3
from port_needed.fast_image_ops import resize
from viewer import gallery


def _rgb(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image, dtype=np.float64)
    if value.ndim == 2:
        value = np.repeat(value[..., None], 3, axis=2)
    value = value[..., :3]
    if float(np.max(value, initial=0.0)) > 1.5:
        value = value / 255.0
    return np.clip(value, 0.0, 1.0)


def _fit_side(image: np.ndarray, side: int) -> np.ndarray:
    value = _rgb(image)
    height, width = value.shape[:2]
    scale = min(1.0, float(side) / max(height, width))
    shape = (max(round(height * scale), 16), max(round(width * scale), 16))
    return value if shape == (height, width) else resize(
        value, shape, order=1, anti_aliasing=True)


def _boundary_energy(labels: np.ndarray, lab: np.ndarray) -> tuple[int, float]:
    horizontal = labels[:, 1:] != labels[:, :-1]
    vertical = labels[1:] != labels[:-1]
    h_energy = np.sum((lab[:, 1:] - lab[:, :-1]) ** 2, axis=2)
    v_energy = np.sum((lab[1:] - lab[:-1]) ** 2, axis=2)
    count = int(np.count_nonzero(horizontal) + np.count_nonzero(vertical))
    total = float(np.sum(h_energy[horizontal]) + np.sum(v_energy[vertical]))
    return count, total / max(count, 1)


def _truth_score(labels: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    predicted = np.asarray(labels, dtype=np.int32).ravel()
    target = np.asarray(truth, dtype=np.int32).ravel()
    predicted_count = int(np.max(predicted)) + 1
    target_count = int(np.max(target)) + 1
    key = predicted.astype(np.int64) * target_count + target
    overlap = np.bincount(key, minlength=predicted_count * target_count)
    overlap = overlap.reshape(predicted_count, target_count)
    undersegmentation = float(
        (np.sum(overlap) - np.sum(np.max(overlap, axis=1))) / len(target)
    )
    horizontal_truth = truth[:, 1:] != truth[:, :-1]
    vertical_truth = truth[1:] != truth[:-1]
    horizontal_prediction = labels[:, 1:] != labels[:, :-1]
    vertical_prediction = labels[1:] != labels[:-1]
    truth_edges = np.count_nonzero(horizontal_truth) + np.count_nonzero(
        vertical_truth)
    recalled = np.count_nonzero(
        horizontal_truth & horizontal_prediction) + np.count_nonzero(
        vertical_truth & vertical_prediction)
    return undersegmentation, float(recalled / max(truth_edges, 1))


def _compound_action(atoms: np.ndarray, compounds: np.ndarray) -> tuple[int, int]:
    atom_count = int(np.max(atoms)) + 1
    compound_count = int(np.max(compounds)) + 1
    pair = np.unique(
        atoms.astype(np.int64).ravel() * compound_count + compounds.ravel())
    atom = pair // compound_count
    compound = pair % compound_count
    split_atoms = int(np.count_nonzero(
        np.bincount(atom, minlength=atom_count) > 1))
    multi_atom_compounds = int(np.count_nonzero(
        np.bincount(compound, minlength=compound_count) > 1))
    return split_atoms, multi_atom_compounds


def benchmark_image(
    name: str,
    image: np.ndarray,
    *,
    side: int,
    truth: np.ndarray | None = None,
) -> dict:
    rgb = _fit_side(image, side)
    if truth is not None and truth.shape != rgb.shape[:2]:
        truth = resize(
            np.asarray(truth, dtype=np.float64),
            rgb.shape[:2],
            order=0,
            anti_aliasing=False,
        ).astype(np.int32)
    result = build_segmenting_v3(
        rgb,
        SegmentingV3Config(
            structural_topology="canonical_v2",
            structural_allocation_side=min(side, 512),
            structural_flow_sweeps=1,
            compound_segmentation=True,
            threads=4,
        ),
    )
    atom = result["texture_labels"]
    quotient = result["compound_segmentation"]
    compound = quotient["labels"]
    lab = srgb_to_lab(rgb)
    atom_boundary = _boundary_energy(atom, lab)
    compound_boundary = _boundary_energy(compound, lab)
    split_atoms, multi_atom = _compound_action(atom, compound)
    row = {
        "name": name,
        "shape": rgb.shape[:2],
        "atoms": int(np.max(atom)) + 1,
        "compounds": int(np.max(compound)) + 1,
        "leaves": int(quotient["leaf_count"]),
        "atom_boundary_px": atom_boundary[0],
        "compound_boundary_px": compound_boundary[0],
        "atom_boundary_energy": atom_boundary[1],
        "compound_boundary_energy": compound_boundary[1],
        "split_atoms": split_atoms,
        "multi_atom_compounds": multi_atom,
        "compound_ms": float(quotient["milliseconds"]),
    }
    if truth is not None:
        row["atom_underseg"], row["atom_recall"] = _truth_score(atom, truth)
        row["compound_underseg"], row["compound_recall"] = _truth_score(
            compound, truth)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=int, default=384)
    args = parser.parse_args()
    truth_loaders = {
        "seg_curve": gallery._seg_curve,
        "seg_quads": gallery._seg_quads,
        "seg_freq": gallery._seg_frequency,
        "seg_orient": gallery._seg_orientation,
    }
    rows = []
    for name, loader in truth_loaders.items():
        image, truth = loader(args.side)
        rows.append(benchmark_image(
            name, image, side=args.side, truth=truth))
    for name in ("page", "checker", "rig", "camera", "coffee", "pikachu", "golden_gate"):
        rows.append(benchmark_image(
            name, gallery.load(name), side=args.side))
    for row in rows:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

