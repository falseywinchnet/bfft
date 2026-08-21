"""Relational assembly between bounded manifolds and spatial supports.

Every bounded relative-complement manifold is treated as a soft part candidate
and every V3 region simultaneously acts as a possible support.  Their relation
uses only two dimensionless observables: covariance-normalized displacement
and symmetric area commensurability.  No support, manifold, orientation, or
object count is selected.
"""

from __future__ import annotations

import numpy as np


def _region_second_moments(complex_: dict) -> dict[str, np.ndarray]:
    labels = np.asarray(complex_["labels"], dtype=np.int32)
    node = complex_["node"]
    count = int(complex_["region_count"])
    area = np.asarray(node["area"], dtype=np.float64)
    yy, xx = np.indices(labels.shape, dtype=np.float64)
    dx = xx - node["centroid_x"][labels]
    dy = yy - node["centroid_y"][labels]
    flat = labels.ravel()
    # Pixel centers represent unit square apertures, contributing variance
    # 1/12 on both axes.  This is a raster measurement, not regularization.
    covariance_xx = (
        np.bincount(flat, weights=(dx * dx).ravel(), minlength=count) / area
        + 1.0 / 12.0
    )
    covariance_xy = np.bincount(
        flat, weights=(dx * dy).ravel(), minlength=count) / area
    covariance_yy = (
        np.bincount(flat, weights=(dy * dy).ravel(), minlength=count) / area
        + 1.0 / 12.0
    )
    determinant = covariance_xx * covariance_yy - covariance_xy ** 2
    return {
        "covariance_xx": covariance_xx,
        "covariance_xy": covariance_xy,
        "covariance_yy": covariance_yy,
        "inverse_xx": covariance_yy / determinant,
        "inverse_xy": -covariance_xy / determinant,
        "inverse_yy": covariance_xx / determinant,
    }


def build_support_manifold_transport(
    complex_: dict,
    enclosure: dict,
    *,
    include_centeredness: bool = True,
    include_scale: bool = True,
) -> dict[str, np.ndarray]:
    """Emit every support/manifold assembly proposal and its PSD region Gram."""
    node = complex_["node"]
    area = np.asarray(node["area"], dtype=np.float64)
    centroid_x = np.asarray(node["centroid_x"], dtype=np.float64)
    centroid_y = np.asarray(node["centroid_y"], dtype=np.float64)
    region_count = int(complex_["region_count"])
    participation = enclosure["participation"].tocsr()
    manifold_area = np.asarray(enclosure["manifold_area"], dtype=np.float64)
    moment = _region_second_moments(complex_)
    manifold_count = participation.shape[0]
    manifold_centroid_x = np.asarray(participation @ centroid_x).ravel()
    manifold_centroid_y = np.asarray(participation @ centroid_y).ravel()
    dx = manifold_centroid_x[:, None] - centroid_x[None, :]
    dy = manifold_centroid_y[:, None] - centroid_y[None, :]
    quadratic = (
        moment["inverse_xx"][None, :] * dx * dx
        + 2.0 * moment["inverse_xy"][None, :] * dx * dy
        + moment["inverse_yy"][None, :] * dy * dy
    )
    centeredness = (
        1.0 / np.sqrt(1.0 + np.maximum(quadratic, 0.0))
        if include_centeredness
        else np.ones((manifold_count, region_count), dtype=np.float64)
    )
    scale = (
        2.0 * manifold_area[:, None] * area[None, :]
        / (manifold_area[:, None] ** 2 + area[None, :] ** 2)
        if include_scale
        else np.ones((manifold_count, region_count), dtype=np.float64)
    )
    weight = centeredness * scale

    # Each proposal feature is sqrt(w_ms) * (e_s + p_m). Summing all Grams
    # factorizes into one diagonal, two cross terms, and one sparse weighted
    # manifold Gram. This is identical to explicit enumeration but scales to
    # the 384-pixel control atlas.
    cross = np.asarray(participation.T @ weight)
    manifold_gram = (
        participation.T
        @ participation.multiply(np.sum(weight, axis=1)[:, None])
    ).toarray()
    kernel = np.diag(np.sum(weight, axis=0)) + cross + cross.T + manifold_gram
    norm = np.sqrt(np.maximum(np.diag(kernel), 0.0))
    denominator = norm[:, None] * norm[None, :]
    normalized = np.divide(
        kernel, denominator, out=np.zeros_like(kernel),
        where=denominator > 1e-30)
    return {
        **moment,
        "manifold_centroid_x": manifold_centroid_x,
        "manifold_centroid_y": manifold_centroid_y,
        "support_manifold_weight": weight,
        "region_kernel": np.clip(
            0.5 * (normalized + normalized.T), 0.0, 1.0),
    }


def summarize_support_manifold_transport(transport: dict) -> dict:
    weight = transport["support_manifold_weight"]
    return {
        "assembly_proposals": int(weight.size),
        "weight_quantiles": [
            float(value) for value in np.quantile(
                weight, (0.0, 0.25, 0.5, 0.75, 1.0))
        ] if weight.size else [0.0] * 5,
    }
