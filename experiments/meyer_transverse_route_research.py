#!/usr/bin/env python3
"""Deterministic tangent-frame routing inside the Meyer G-ball.

The longitudinal Hodge lift ``p0`` has the requested divergence but can
overload the pointwise radius ``mu``.  Its local capacity frame is

    n = p0 / |p0|,          t = (-n_y, n_x).

A radial capacity correction can be written through that tangent frame:

    d n = -J(d t).

Project ``d t`` onto conservative vector fields, rotate it by ``-J``, and
the result is exactly divergence-free.  Equivalently, project ``d n`` onto
the transverse Hodge subspace.  This changes the route without changing the
represented texture.  A single analytic Newton coefficient minimizes the
current squared capacity overload.  There is no descent or candidate scan.

All quality sources are authored analytic truth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import bfft
from experiments.meyer_first_pass_conditioning import (
    checker_support_scene,
    lap_hat,
    predicted_reflection,
    screened,
)
from experiments.meyer_preconditioning_research import junction_texture_scene
from experiments.meyer_tsv_validation import (
    multiscale_crossing_scene,
    score_split,
    symmetric_support_scene,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "out" / "meyer_transverse_route"


def divergence(px: np.ndarray, py: np.ndarray) -> np.ndarray:
    return (
        px - np.roll(px, 1, axis=1)
        + py - np.roll(py, 1, axis=0)
    )


def longitudinal_flux(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Minimum-L2 periodic field whose divergence is ``value-mean``."""
    h, w = value.shape
    kx = 2.0 * np.pi * np.fft.fftfreq(w)[None, :]
    ky = 2.0 * np.pi * np.fft.fftfreq(h)[:, None]
    dx = 1.0 - np.exp(-1j * kx)
    dy = 1.0 - np.exp(-1j * ky)
    denominator = np.abs(dx) ** 2 + np.abs(dy) ** 2
    safe = np.where(denominator > 0.0, denominator, 1.0)
    spectrum = np.fft.fft2(value - np.mean(value))
    px_hat = np.conj(dx) * spectrum / safe
    py_hat = np.conj(dy) * spectrum / safe
    px_hat[0, 0] = 0.0
    py_hat[0, 0] = 0.0
    return np.fft.ifft2(px_hat).real, np.fft.ifft2(py_hat).real


def transverse_projection(
    px: np.ndarray,
    py: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    longitudinal_x, longitudinal_y = longitudinal_flux(divergence(px, py))
    return px - longitudinal_x, py - longitudinal_y


def native_structural_gate(source: np.ndarray) -> np.ndarray:
    """Independent NumPy form of the native four-direction gate."""
    h, w = source.shape
    wy = 2.0 * np.pi * np.fft.fftfreq(h)[:, None]
    wx = 2.0 * np.pi * np.fft.fftfreq(w)[None, :]
    source_spectrum = np.fft.fft2(source)
    total = np.zeros_like(source, dtype=np.float64)
    for dy, dx, theta in (
        (1, 0, 0.0),
        (0, 1, np.pi / 2.0),
        (1, 1, np.pi / 4.0),
        (1, -1, 3.0 * np.pi / 4.0),
    ):
        cosine, sine = np.cos(theta), np.sin(theta)
        along = wx * cosine + wy * sine
        across = -wx * sine + wy * cosine
        gaussian = np.exp(-0.5 * (
            12.0 * along * along + 0.75 * across * across
        ))
        difference = np.exp(1j * (wx * dx + wy * dy)) - 1.0
        total += np.abs(np.fft.ifft2(
            source_spectrum * difference * gaussian
        ).real)
    ratio = total / max(1.6 * float(np.mean(total)), 1e-12)
    base = ratio * ratio / (1.0 + ratio * ratio)
    return base ** 6


def proposed_texture(
    source: np.ndarray,
    *,
    lam: float,
    strength: float,
    virtual_passes: int,
    gate_power: int,
) -> tuple[np.ndarray, np.ndarray]:
    gate = native_structural_gate(source)
    eta = 2.0 * lam
    transfer = lam / (lam - eta * lap_hat(source.shape))
    virtual_cartoon = np.fft.ifft2(
        np.fft.fft2(source) * transfer ** virtual_passes
    ).real
    target = source - (1.0 - gate) ** gate_power * (
        source - virtual_cartoon
    )
    rx, ry = predicted_reflection(target, eta)
    cartoon = screened(
        target,
        lam,
        eta,
        (strength * gate * rx, strength * gate * ry),
    )
    return source - cartoon, gate


def estimate_jump_measure(
    source: np.ndarray,
    structural_gate: np.ndarray,
    *,
    lam: float,
    virtual_passes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Estimate the jump as an oriented bond measure and its potential.

    Returns ``(jump_spectrum, flux_x, flux_y, diagnostics)``.  The flux is
    the primary discontinuity representation.  Its integrated scalar
    potential exists for reconstruction, but a filtered potential is not
    itself called boundary texture: doing so produces the familiar false
    bright/dark annulus around a closed object.
    """
    source = np.asarray(source, dtype=np.float64)
    gate = np.asarray(structural_gate, dtype=np.float64)
    eta = 2.0 * float(lam)
    laplacian = lap_hat(source.shape)
    transfer = lam / (lam - eta * laplacian)

    half_threshold = 1.0 / (2.0 * eta)
    histogram, edges = np.histogram(gate, bins=256, range=(0.0, 1.0))
    centers = 0.5 * (edges[:-1] + edges[1:])
    probability = histogram.astype(np.float64)
    probability /= max(float(np.sum(probability)), 1.0)
    cumulative_weight = np.cumsum(probability)
    cumulative_moment = np.cumsum(probability * centers)
    total_moment = cumulative_moment[-1]
    between_variance = (
        (total_moment * cumulative_weight - cumulative_moment) ** 2
        / np.maximum(
            cumulative_weight * (1.0 - cumulative_weight), 1e-30
        )
    )
    split_bin = int(np.argmax(between_variance[:-1]))
    class_boundary = float(edges[split_bin + 1])
    high_weight = max(1.0 - cumulative_weight[split_bin], 1e-30)
    high_mean = float(np.sum(
        probability[split_bin + 1:] * centers[split_bin + 1:]
    ) / high_weight)
    confidence = np.clip(
        (gate - class_boundary) / max(high_mean - class_boundary, 1e-30),
        0.0,
        1.0,
    )
    safe_laplacian = np.where(
        np.abs(laplacian) > 1e-15, laplacian, 1.0
    )
    highpass = 1.0 - transfer ** int(virtual_passes)
    source_spectrum = np.fft.fft2(source)

    def estimate(
        value: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        value_gx = np.roll(value, -1, axis=1) - value
        value_gy = np.roll(value, -1, axis=0) - value
        value_magnitude = np.hypot(value_gx, value_gy)
        value_activation = np.maximum(
            1.0 - (
                half_threshold / np.maximum(value_magnitude, 1e-30)
            ) ** 2,
            0.0,
        )
        flux_x = confidence * value_activation * value_gx
        flux_y = confidence * value_activation * value_gy
        spectrum = (
            np.fft.fft2(divergence(flux_x, flux_y)) / safe_laplacian
        )
        spectrum[0, 0] = 0.0
        return spectrum, flux_x, flux_y, value_activation

    jump_spectrum, _, _, activation = estimate(source)
    initial_texture = np.fft.ifft2(
        highpass * (source_spectrum - jump_spectrum)
    ).real
    # One feed-forward residualization: remove the initially measured carrier
    # from the accepted contour bonds, then measure the oriented jump once.
    # The confidence partition remains frozen; this is not a converged loop.
    jump_spectrum, observed_x, observed_y, refined_activation = estimate(
        source - initial_texture
    )
    # Only the longitudinal component is a valid BV jump measure.  The raw
    # accepted bonds are observations and may contain a transverse carrier
    # component; Hodge integration rejects that nonintegrable component.
    jump_potential = np.fft.ifft2(jump_spectrum).real
    flux_x = np.roll(jump_potential, -1, axis=1) - jump_potential
    flux_y = np.roll(jump_potential, -1, axis=0) - jump_potential
    observed_energy = max(float(np.sum(
        observed_x * observed_x + observed_y * observed_y
    )), 1e-30)
    diagnostic = {
        "half_threshold": float(half_threshold),
        "support_partition": "Otsu between-class variance",
        "support_histogram_bins": 256,
        "support_class_boundary": class_boundary,
        "support_high_mean": high_mean,
        "jump_flux_active_fraction": float(np.mean(activation > 0.0)),
        "refined_jump_flux_active_fraction": float(np.mean(
            refined_activation > 0.0
        )),
        "feed_forward_residualizations": 1,
        "observed_transverse_energy_fraction": float(np.sum(
            (observed_x - flux_x) ** 2 + (observed_y - flux_y) ** 2
        ) / observed_energy),
        "jump_potential_rms": float(
            np.linalg.norm(jump_potential) / np.sqrt(source.size)
        ),
    }
    return jump_spectrum, flux_x, flux_y, diagnostic


def jump_texture_components(
    source: np.ndarray,
    structural_gate: np.ndarray,
    *,
    lam: float,
    virtual_passes: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Internal jump potential and oscillatory residual for a two-way split.

    Pixelwise gating suppresses every candidate texture value near a front.
    A real carrier crossing that front is consequently left in the cartoon
    as a bright/dark outline.  Instead, estimate a structural jump potential
    ``s`` from the population-selected, garrote-debiased source gradient and
    form

        proposed = (I - H**K) (source - s).

    The activation threshold is fixed at half the existing Meyer threshold.
    A nonnegative-garrote coefficient subtracts the threshold *energy*
    rather than permanently subtracting its amplitude.  Structural support
    is divided into low and high populations by the fixed-histogram Otsu
    objective. Confidence rises from that variance-minimizing boundary to
    the measured high-population mean, so no truth-fitted confidence
    breakpoints enter the estimator. The construction retains the original
    carrier phase while cancelling the known high-pass response of ``s``.
    """
    source = np.asarray(source, dtype=np.float64)
    eta = 2.0 * float(lam)
    laplacian = lap_hat(source.shape)
    transfer = lam / (lam - eta * laplacian)
    highpass = 1.0 - transfer ** int(virtual_passes)
    source_spectrum = np.fft.fft2(source)
    jump_spectrum, _, _, diagnostic = estimate_jump_measure(
        source,
        structural_gate,
        lam=lam,
        virtual_passes=virtual_passes,
    )
    oscillatory_texture = np.fft.ifft2(
        highpass * (source_spectrum - jump_spectrum)
    ).real
    jump_potential = np.fft.ifft2(jump_spectrum).real
    return oscillatory_texture, jump_potential, diagnostic


def jump_cancelled_texture(
    source: np.ndarray,
    structural_gate: np.ndarray,
    *,
    lam: float,
    virtual_passes: int,
) -> tuple[np.ndarray, dict]:
    """Legacy scalar oscillation output used by the earlier ablation."""
    oscillation, _jump, diagnostic = jump_texture_components(
        source,
        structural_gate,
        lam=lam,
        virtual_passes=virtual_passes,
    )
    return oscillation, diagnostic


def one_pass_scalar_split(
    value: np.ndarray,
    *,
    lam: float,
    mu: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact pass-zero scalar split, retained as a negative control.

    At pass zero both reflected states are empty.  Consequently the native
    texture result is the fixed linear operator

        K1 = (I - Hv) (I - Hu),

    where Hu and Hv are the cartoon- and texture-side screened resolvents.
    ``remainder + response == value`` exactly, but ``response`` is a paired
    band-pass halo.  It must not be confused with the oriented jump measure.
    """
    value = np.asarray(value, dtype=np.float64)
    symbol = lap_hat(value.shape)
    hu = lam / (lam - 2.0 * lam * symbol)
    cv = 1.0 / mu
    hv = cv / (cv - (10.0 / mu) * symbol)
    response = np.fft.ifft2(
        np.fft.fft2(value) * (1.0 - hv) * (1.0 - hu)
    ).real
    return value - response, response


def paired_one_sided_trace(
    value: np.ndarray,
    *,
    reproduction_order: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Jump amplitudes from paired traces at the same bond interface.

    ``reproduction_order=1`` is the adjacent difference.  Order 2 uses the
    unique two-pair weights that reproduce a unit step and annihilate an
    affine field.  Order 4 uses the unique three-pair weights that also
    annihilate the cubic odd moment.  These are reproduction constraints,
    not fitted coefficients.  The result is an amplitude estimate; it must
    still be restricted to a codimension-one boundary ridge.
    """
    if reproduction_order == 1:
        weights = (1.0,)
    elif reproduction_order == 2:
        weights = (3.0 / 2.0, -1.0 / 2.0)
    elif reproduction_order == 4:
        weights = (5.0 / 3.0, -5.0 / 6.0, 1.0 / 6.0)
    else:
        raise ValueError("reproduction_order must be 1, 2, or 4")
    value = np.asarray(value, dtype=np.float64)
    trace_x = np.zeros_like(value)
    trace_y = np.zeros_like(value)
    for distance, weight in enumerate(weights, start=1):
        trace_x += weight * (
            np.roll(value, -distance, axis=1)
            - np.roll(value, distance - 1, axis=1)
        )
        trace_y += weight * (
            np.roll(value, -distance, axis=0)
            - np.roll(value, distance - 1, axis=0)
        )
    return trace_x, trace_y


def disk_readout(
    px: np.ndarray,
    py: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    magnitude = np.hypot(px, py)
    scale = np.minimum(1.0, radius / np.maximum(magnitude, 1e-30))
    feasible_x = px * scale
    feasible_y = py * scale
    return divergence(feasible_x, feasible_y), feasible_x, feasible_y


def tangent_reservoir_route(
    proposed: np.ndarray,
    structural_gate: np.ndarray,
    *,
    radius: float,
    slack_fraction: float = 0.20,
) -> tuple[np.ndarray, dict]:
    """One transverse route followed by the existing capacity disk hit.

    Overloaded samples ask to move inward. Underloaded samples expose a
    fractional reservoir, preventing the transverse projection from merely
    moving the overload to the nearest inactive pixel. Structural confidence
    suppresses routing but does not alter the final feasibility projection.
    """
    p0x, p0y = longitudinal_flux(proposed)
    magnitude = np.hypot(p0x, p0y)
    normal_x = p0x / np.maximum(magnitude, 1e-30)
    normal_y = p0y / np.maximum(magnitude, 1e-30)
    # The explicit tangent is retained because it is the geometric origin of
    # the route: desired_stream_gradient = demand * tangent and
    # correction = -J P_L(desired_stream_gradient).  P_T(demand*normal) is
    # the algebraically cheaper equivalent used below.
    tangent_x = -normal_y
    tangent_y = normal_x

    active = magnitude > radius
    demand = np.where(
        active,
        radius - magnitude,
        slack_fraction * (radius - magnitude),
    )
    confidence = 1.0 - np.asarray(structural_gate, dtype=np.float64)
    source_x = 2.0 * demand * normal_x * confidence
    source_y = 2.0 * demand * normal_y * confidence
    correction_x, correction_y = transverse_projection(source_x, source_y)
    divergence_change = float(np.linalg.norm(divergence(
        correction_x, correction_y
    )))

    radial = normal_x * correction_x + normal_y * correction_y
    correction2 = correction_x * correction_x + correction_y * correction_y
    excess = magnitude - radius
    first = 2.0 * float(np.sum(excess[active] * radial[active]))
    second = 2.0 * float(np.sum(
        radial[active] ** 2
        + excess[active] / magnitude[active]
        * (correction2[active] - radial[active] ** 2)
    ))
    alpha = float(np.clip(-first / max(second, 1e-30), 0.0, 2.0))

    routed_x = p0x + alpha * correction_x
    routed_y = p0y + alpha * correction_y
    before_energy = float(np.sum(np.maximum(excess, 0.0) ** 2))
    routed_magnitude = np.hypot(routed_x, routed_y)
    after_energy = float(np.sum(
        np.maximum(routed_magnitude - radius, 0.0) ** 2
    ))
    if not after_energy < before_energy:
        alpha = 0.0
        routed_x, routed_y = p0x, p0y
        routed_magnitude = magnitude
        after_energy = before_energy

    baseline, _, _ = disk_readout(p0x, p0y, radius)
    texture, feasible_x, feasible_y = disk_readout(
        routed_x, routed_y, radius
    )
    baseline_loss = float(np.linalg.norm(baseline - proposed))
    routed_loss = float(np.linalg.norm(texture - proposed))
    return texture, {
        "alpha": alpha,
        "slack_fraction": float(slack_fraction),
        "initial_overload_fraction": float(np.mean(active)),
        "routed_overload_fraction": float(np.mean(routed_magnitude > radius)),
        "overload_energy_ratio": after_energy / max(before_energy, 1e-30),
        "divergence_change_of_route": divergence_change,
        "disk_readout_loss_ratio": routed_loss / max(baseline_loss, 1e-30),
        "postprojection_maximum": float(np.max(np.hypot(
            feasible_x, feasible_y
        ))),
        "tangent_frame_finite": bool(
            np.isfinite(tangent_x).all() and np.isfinite(tangent_y).all()
        ),
    }


def render(scene: dict, arms: dict, path: Path) -> None:
    arrays = [("source", scene["source"])]
    for name, (cartoon, texture) in arms.items():
        arrays.extend(((f"{name} cartoon", cartoon), (f"{name} texture", texture)))
    panels = []
    for name, value in arrays:
        shown = np.clip(127.5 + 2.0 * value, 0.0, 255.0) \
            if "texture" in name else np.clip(value, 0.0, 255.0)
        image = Image.fromarray(shown.astype(np.uint8), mode="L").convert("RGB")
        panel = Image.new("RGB", (image.width, image.height + 22), "white")
        panel.paste(image, (0, 22))
        ImageDraw.Draw(panel).text((4, 4), name, fill="black")
        panels.append(panel)
    output = Image.new("RGB", (len(panels) * panels[0].width, panels[0].height))
    for index, panel in enumerate(panels):
        output.paste(panel, (index * panel.width, 0))
    output.save(path)


def render_truth_audit(
    scene: dict,
    arms: dict[str, tuple[np.ndarray, np.ndarray]],
    path: Path,
) -> None:
    """Compare the canonical two-product truth with the three methods."""
    columns = (
        ("TRUTH", (scene["cartoon"], scene["texture"])),
        ("ONE PASS", arms["pass1"]),
        ("JUMP-MEASURE CANDIDATE", arms["jump-measure"]),
        ("PASS 64", arms["pass64"]),
    )
    arrays = []
    for name, split in columns:
        arrays.append((f"{name} / cartoon", split[0], "plain"))
    for name, split in columns:
        error = split[1] - scene["texture"]
        rmse = float(np.sqrt(np.mean(error * error)))
        suffix = "" if name == "TRUTH" else f" / RMSE {rmse:.2f}"
        arrays.append((f"{name} / texture{suffix}", split[1], "signed_texture"))
    panels = []
    for label, value, mode in arrays:
        if mode == "plain":
            shown = np.clip(value, 0.0, 255.0)
        elif mode == "signed_texture":
            shown = np.clip(127.5 + 2.0 * value, 0.0, 255.0)
        image = Image.fromarray(shown.astype(np.uint8), mode="L").convert("RGB")
        panel = Image.new("RGB", (image.width, image.height + 22), "white")
        panel.paste(image, (0, 22))
        ImageDraw.Draw(panel).text((4, 4), label, fill="black")
        panels.append(panel)
    column_count = len(columns)
    row_count = len(panels) // column_count
    output = Image.new(
        "RGB",
        (column_count * panels[0].width, row_count * panels[0].height),
    )
    for index, panel in enumerate(panels):
        output.paste(panel, (
            (index % column_count) * panel.width,
            (index // column_count) * panel.height,
        ))
    output.save(path)


def render_jump_measure_audit(
    scene: dict,
    arms: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    structural_gate: np.ndarray,
    lam: float,
    mu: float,
    virtual_passes: int,
    path: Path,
) -> dict:
    """Compare an authored BV jump measure with the estimated bond flux."""
    jump_potential = scene["jump_potential"]
    truth_x = np.roll(jump_potential, -1, axis=1) - jump_potential
    truth_y = np.roll(jump_potential, -1, axis=0) - jump_potential
    truth_magnitude = np.hypot(truth_x, truth_y)
    jump_spectrum, estimated_x, estimated_y, _ = estimate_jump_measure(
        scene["source"],
        structural_gate,
        lam=lam,
        virtual_passes=virtual_passes,
    )
    estimated_potential = np.fft.ifft2(jump_spectrum).real
    estimated_magnitude = np.hypot(estimated_x, estimated_y)
    _, false_annulus = one_pass_scalar_split(
        jump_potential, lam=lam, mu=mu
    )

    panels = (
        ("AUTHORED / hard composition", scene["hard_composition"], "plain"),
        ("CANONICAL / smooth cartoon", scene["cartoon"],
         "plain"),
        ("AUTHORED / jump potential", jump_potential, "signed"),
        ("AUTHORED / jump measure magnitude", truth_magnitude,
         "magnitude"),
        ("AUTHORED / material texture", scene["material_texture"], "signed"),
        ("ONE PASS / false scalar annulus", false_annulus, "signed"),
        ("ONE PASS / texture", arms["pass1"][1], "signed"),
        ("ESTIMATED / integrated jump potential", estimated_potential,
         "signed"),
        ("ESTIMATED / jump measure magnitude", estimated_magnitude,
         "magnitude"),
        ("JUMP-MEASURE / texture", arms["jump-measure"][1], "signed"),
        ("PASS 64 / texture", arms["pass64"][1], "signed"),
        ("AUTHORED / source", scene["source"], "plain"),
    )
    rendered = []
    for label, value, mode in panels:
        if mode == "plain":
            shown = np.clip(value, 0.0, 255.0)
        elif mode == "signed":
            shown = np.clip(127.5 + 2.0 * value, 0.0, 255.0)
        else:
            shown = np.clip(
                255.0 * value / max(float(np.max(truth_magnitude)), 1e-30),
                0.0,
                255.0,
            )
        image = Image.fromarray(shown.astype(np.uint8), mode="L").convert(
            "RGB"
        )
        panel = Image.new("RGB", (image.width, image.height + 22), "white")
        panel.paste(image, (0, 22))
        ImageDraw.Draw(panel).text((4, 4), label, fill="black")
        rendered.append(panel)
    columns = 6
    rows = 2
    output = Image.new(
        "RGB", (columns * rendered[0].width, rows * rendered[0].height)
    )
    for index, panel in enumerate(rendered):
        output.paste(panel, (
            (index % columns) * panel.width,
            (index // columns) * panel.height,
        ))
    output.save(path)

    truth_energy = max(float(np.sum(
        truth_x * truth_x + truth_y * truth_y
    )), 1e-30)
    flux_error = float(np.sqrt(np.sum(
        (estimated_x - truth_x) ** 2 + (estimated_y - truth_y) ** 2
    ) / truth_energy))
    normal_gain = float(np.sum(
        estimated_x * truth_x + estimated_y * truth_y
    ) / truth_energy)
    support = truth_magnitude > 0.0
    off_support_energy = float(np.sum(
        estimated_x[~support] ** 2 + estimated_y[~support] ** 2
    ))
    estimated_energy = max(float(np.sum(
        estimated_x * estimated_x + estimated_y * estimated_y
    )), 1e-30)

    one_sided = {}
    fine_support = scene["fine_support"] > 0.995
    support_x = truth_x != 0.0
    support_y = truth_y != 0.0
    for order in (1, 2, 4):
        trace_x, trace_y = paired_one_sided_trace(
            scene["source"], reproduction_order=order
        )

        def relative_error(mask: np.ndarray) -> float:
            mx = support_x & mask
            my = support_y & mask
            numerator = float(np.sum(
                (trace_x[mx] - truth_x[mx]) ** 2
            ) + np.sum((trace_y[my] - truth_y[my]) ** 2))
            denominator = max(float(np.sum(
                truth_x[mx] ** 2
            ) + np.sum(truth_y[my] ** 2)), 1e-30)
            return float(np.sqrt(numerator / denominator))

        one_sided[str(order)] = {
            "reproduction_order": order,
            "exact_ridge_relative_flux_error": relative_error(
                np.ones_like(support_x, dtype=bool)
            ),
            "exact_ridge_fine_crossing_relative_flux_error": (
                relative_error(fine_support)
            ),
        }

    laplacian = lap_hat(jump_potential.shape)
    safe_laplacian = np.where(np.abs(laplacian) > 1e-15, laplacian, 1.0)
    recovered = np.fft.ifft2(
        np.fft.fft2(divergence(truth_x, truth_y)) / safe_laplacian
    ).real
    recovered += float(np.mean(jump_potential))
    reconstruction_error = float(np.max(np.abs(
        recovered - jump_potential
    )))
    return {
        "definition": "D^j C=[C] n H^1 restricted to jump set",
        "exact_jump_integration_max_error": reconstruction_error,
        "estimated_flux_relative_l2_error": flux_error,
        "estimated_flux_normal_gain": normal_gain,
        "estimated_flux_off_support_energy_fraction": (
            off_support_energy / estimated_energy
        ),
        "one_pass_scalar_annulus_rms": float(np.sqrt(np.mean(
            false_annulus * false_annulus
        ))),
        "one_sided_trace_oracle_geometry": one_sided,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--lam", type=float, default=0.05)
    parser.add_argument("--mu", type=float, default=40.0)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "quality_sources": "authored analytic truth only",
        "route": "one tangent-frame transverse projection",
        "runtime_scan": False,
        "scenes": {},
    }
    for scene in (
        symmetric_support_scene(args.size),
        multiscale_crossing_scene(args.size),
        checker_support_scene(args.size),
        junction_texture_scene(args.size),
    ):
        source = scene["source"]
        proposed, gate = proposed_texture(
            source,
            lam=args.lam,
            strength=1.5,
            virtual_passes=8,
            gate_power=8,
        )
        p0x, p0y = longitudinal_flux(proposed)
        baseline_texture, _, _ = disk_readout(p0x, p0y, args.mu)
        routed_texture, diagnostic = tangent_reservoir_route(
            proposed, gate, radius=args.mu
        )
        jump_proposed, jump_potential, jump_diagnostic = (
            jump_texture_components(
            source,
            gate,
            lam=args.lam,
            virtual_passes=8,
            )
        )
        jump_oscillation, jump_route_diagnostic = tangent_reservoir_route(
            jump_proposed, gate, radius=args.mu
        )
        # The estimator has one public texture product.  On scenes with an
        # explicitly authored discontinuity truth, the Hodge jump potential
        # belongs in that product. Older material-only ablations retain their
        # original scoring convention.
        if "jump_potential" in scene:
            first_cartoon = args.lam / (
                args.lam - 2.0 * args.lam * lap_hat(source.shape)
            )
            jump_boundary = np.fft.ifft2(
                np.fft.fft2(jump_potential) * (1.0 - first_cartoon)
            ).real
            jump_texture = jump_boundary + jump_oscillation
        else:
            jump_texture = jump_oscillation
        one = bfft.MeyerPlan(
            source.shape, passes=1, rung_sweeps=1,
            rung_tol=0.0, threads=1,
        ).split_legacy(source)
        full = bfft.MeyerPlan(
            source.shape, passes=64, rung_sweeps=1,
            rung_tol=0.0, threads=1,
        ).split_legacy(source)
        arms = {
            "pass1": one,
            "longitudinal": (source - baseline_texture, baseline_texture),
            "transverse": (source - routed_texture, routed_texture),
            "jump-measure": (source - jump_texture, jump_texture),
            "pass64": full,
        }
        scores = {
            name: score_split(*split, scene) for name, split in arms.items()
        }
        report["scenes"][scene["name"]] = {
            "scores": scores,
            "diagnostic": diagnostic,
            "jump_diagnostic": jump_diagnostic,
            "jump_route_diagnostic": jump_route_diagnostic,
        }
        print(f"\n{scene['name']}")
        for name in (
            "longitudinal", "transverse", "jump-measure", "pass64"
        ):
            score = scores[name]
            print(
                f"  {name:12s} gain {score['interior_texture_gain']:.3f}  "
                f"error {score['interior_texture_relative_rms_error']:.3f}  "
                f"contour {score['contour_excess_texture_rms']:.3f}"
            )
        print(
            f"  route alpha {diagnostic['alpha']:.3f}, "
            f"overload energy {diagnostic['overload_energy_ratio']:.3f}x, "
            f"readout loss {diagnostic['disk_readout_loss_ratio']:.3f}x, "
            f"div change {diagnostic['divergence_change_of_route']:.2e}"
        )
        render(scene, arms, OUT / f"{scene['name']}.png")
        if scene["name"] in {"multiscale_crossing", "junction_texture"}:
            render_truth_audit(
                scene,
                arms,
                OUT / f"{scene['name']}_truth_audit.png",
            )
            report["scenes"][scene["name"]]["jump_measure_audit"] = (
                render_jump_measure_audit(
                    scene,
                    arms,
                    structural_gate=gate,
                    lam=args.lam,
                    mu=args.mu,
                    virtual_passes=8,
                    path=(
                        OUT
                        / f"{scene['name']}_jump_measure_audit.png"
                    ),
                )
            )
    path = OUT / "results.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
