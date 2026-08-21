"""Measure relational motion between source and posterized color charts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from posterizer.core import PosterizerConfig, _perceptual_importance, _rgba_to_lab_alpha


def _weighted_mean(value: np.ndarray, weight: np.ndarray) -> float:
    return float(np.sum(weight * value) / max(float(np.sum(weight)), 1e-15))


def _weighted_correlation(
    first: np.ndarray, second: np.ndarray, weight: np.ndarray
) -> float:
    mean_first = _weighted_mean(first, weight)
    mean_second = _weighted_mean(second, weight)
    left = first - mean_first
    right = second - mean_second
    covariance = _weighted_mean(left * right, weight)
    variance = _weighted_mean(left * left, weight) * _weighted_mean(
        right * right, weight
    )
    return float(covariance / np.sqrt(max(variance, 1e-30)))


def _measure_relations(
    source: np.ndarray,
    target: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    importance: np.ndarray,
) -> dict[str, float]:
    source_delta = source[first] - source[second]
    target_delta = target[first] - target[second]
    source_distance = np.linalg.norm(source_delta, axis=1)
    target_distance = np.linalg.norm(target_delta, axis=1)
    weight = np.sqrt(importance[first] * importance[second])
    source_energy = _weighted_mean(source_distance * source_distance, weight)
    intrinsic_error = target_distance - source_distance
    coordinate_error = target_delta - source_delta
    stress = np.sqrt(
        _weighted_mean(intrinsic_error * intrinsic_error, weight)
        / max(source_energy, 1e-30)
    )
    oriented_strain = np.sqrt(
        _weighted_mean(np.sum(coordinate_error * coordinate_error, axis=1), weight)
        / max(source_energy, 1e-30)
    )
    scale = _weighted_mean(source_distance * target_distance, weight) / max(
        source_energy, 1e-30
    )
    relation_energy = weight * source_distance * source_distance
    collapse = float(np.sum(
        relation_energy[target_distance < 0.25 * source_distance]
    ) / max(float(np.sum(relation_energy)), 1e-30))
    active = (source_distance > 1e-5) & (target_distance > 1e-5)
    cosine = np.einsum(
        "ij,ij->i", source_delta[active], target_delta[active]
    ) / (source_distance[active] * target_distance[active])
    alignment_weight = relation_energy[active]
    return {
        "chart_stress": float(stress),
        "oriented_chart_strain": float(oriented_strain),
        "distance_correlation": _weighted_correlation(
            source_distance, target_distance, weight
        ),
        "distance_scale": float(scale),
        "collapsed_relation_energy": collapse,
        "direction_alignment": _weighted_mean(cosine, alignment_weight)
        if len(cosine)
        else 1.0,
        "relations": int(len(first)),
    }


def _global_pairs(
    visible: np.ndarray, count: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    active = np.flatnonzero(visible)
    random = np.random.default_rng(seed)
    first = random.choice(active, size=count, replace=True)
    second = random.choice(active, size=count, replace=True)
    equal = first == second
    while np.any(equal):
        second[equal] = random.choice(active, size=int(np.sum(equal)), replace=True)
        equal = first == second
    return first, second


def _local_pairs(
    visible: np.ndarray, limit: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    height, width = visible.shape
    index = np.arange(height * width, dtype=np.int64).reshape(height, width)
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for radius in (1, 2, 4, 8, 16, 32):
        for dy, dx in ((0, radius), (radius, 0), (radius, radius), (radius, -radius)):
            if abs(dy) >= height or abs(dx) >= width:
                continue
            y0 = slice(max(0, -dy), min(height, height - dy))
            y1 = slice(max(0, dy), min(height, height + dy))
            x0 = slice(max(0, -dx), min(width, width - dx))
            x1 = slice(max(0, dx), min(width, width + dx))
            valid = visible[y0, x0] & visible[y1, x1]
            pairs.append((index[y0, x0][valid], index[y1, x1][valid]))
    first = np.concatenate([pair[0] for pair in pairs])
    second = np.concatenate([pair[1] for pair in pairs])
    if len(first) > limit:
        random = np.random.default_rng(seed)
        chosen = random.choice(len(first), size=limit, replace=False)
        first = first[chosen]
        second = second[chosen]
    return first, second


def chart_distance(
    source_rgba: np.ndarray,
    target_rgba: np.ndarray,
    *,
    pairs: int = 250_000,
) -> dict[str, dict[str, dict[str, float]]]:
    if source_rgba.shape != target_rgba.shape:
        raise ValueError("source and target must have identical raster dimensions")
    source_lab = _rgba_to_lab_alpha(source_rgba)
    target_lab = _rgba_to_lab_alpha(target_rgba)
    visible = (source_rgba[..., 3] > 4) & (target_rgba[..., 3] > 4)
    importance = _perceptual_importance(source_lab, visible, PosterizerConfig())
    flat_importance = importance.ravel()
    global_first, global_second = _global_pairs(visible.ravel(), pairs, 7351)
    local_first, local_second = _local_pairs(visible, pairs, 9277)
    result: dict[str, dict[str, dict[str, float]]] = {}
    for sigma in (0.0, 1.5, 4.0):
        if sigma:
            source_field = ndimage.gaussian_filter(
                source_lab[..., :3], sigma=(sigma, sigma, 0.0), mode="reflect"
            )
            target_field = ndimage.gaussian_filter(
                target_lab[..., :3], sigma=(sigma, sigma, 0.0), mode="reflect"
            )
        else:
            source_field = source_lab[..., :3]
            target_field = target_lab[..., :3]
        source_flat = source_field.reshape(-1, 3)
        target_flat = target_field.reshape(-1, 3)
        result[f"sigma_{sigma:g}"] = {
            "global": _measure_relations(
                source_flat,
                target_flat,
                global_first,
                global_second,
                flat_importance,
            ),
            "local_multiscale": _measure_relations(
                source_flat,
                target_flat,
                local_first,
                local_second,
                flat_importance,
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("targets", nargs="+", type=Path)
    parser.add_argument("--pairs", type=int, default=250_000)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    source = np.asarray(Image.open(args.source).convert("RGBA"))
    report = {}
    for path in args.targets:
        target = np.asarray(Image.open(path).convert("RGBA"))
        report[str(path)] = chart_distance(source, target, pairs=args.pairs)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.out:
        args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
