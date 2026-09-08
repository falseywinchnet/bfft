"""Meyer G-flux decomposition in a frozen Eikonal SPD transport state.

The ordinary Meyer texture ball uses v=div(p), |p|_2<=mu.  Here the flux is
written p=M^{-1/2}q with |q|_2<=mu, where M is the source-derived Eikonal
metric.  Optimizing in q keeps the constraint a Euclidean disk and gives an
exact metric dual ball p^T M p<=mu^2 without per-pixel root solves.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bfft  # noqa: E402
from denoiser.continual_eikonal_noise_transport_2d import (  # noqa: E402
    continual_transport_metric,
    directional_noise_witnesses,
)


OUT = ROOT / "experiments" / "out" / "meyer_eikonal_transport_split"


def grad(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gx = np.roll(field, -1, axis=1) - field
    gy = np.roll(field, -1, axis=0) - field
    return gx, gy


def div(px: np.ndarray, py: np.ndarray) -> np.ndarray:
    return px - np.roll(px, 1, axis=1) + py - np.roll(py, 1, axis=0)


def inverse_sqrt_metric(metric: dict[str, np.ndarray | float]) -> tuple[np.ndarray, ...]:
    """Return the symmetric inverse square root of a 2x2 SPD field."""

    a = np.asarray(metric["metric_xx"], dtype=np.float64)
    b = np.asarray(metric["metric_xy"], dtype=np.float64)
    c = np.asarray(metric["metric_yy"], dtype=np.float64)
    angle = 0.5 * np.arctan2(2.0*b, a-c)
    cosine, sine = np.cos(angle), np.sin(angle)
    gap = np.hypot(a-c, 2.0*b)
    high = np.maximum(0.5*(a+c+gap), np.finfo(float).tiny)
    low = np.maximum(0.5*(a+c-gap), np.finfo(float).tiny)
    ih, il = 1.0/np.sqrt(high), 1.0/np.sqrt(low)
    lxx = ih*cosine*cosine + il*sine*sine
    lxy = (ih-il)*cosine*sine
    lyy = ih*sine*sine + il*cosine*cosine
    return lxx, lxy, lyy


def metric_rof(
    source: np.ndarray,
    fidelity: float,
    inverse_sqrt: tuple[np.ndarray, np.ndarray, np.ndarray],
    q_state: tuple[np.ndarray, np.ndarray] | None,
    iterations: int,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray], dict[str, float]]:
    """Resolve the Riemannian ROF dual by projected gradient in whitened flux."""

    g = np.asarray(source, dtype=np.float64)
    lxx, lxy, lyy = inverse_sqrt
    qx, qy = (
        (np.zeros_like(g), np.zeros_like(g))
        if q_state is None else (q_state[0].copy(), q_state[1].copy())
    )
    inverse_xx = lxx*lxx+lxy*lxy
    inverse_xy = lxy*(lxx+lyy)
    inverse_yy = lxy*lxy+lyy*lyy
    inverse_high = 0.5*(
        inverse_xx+inverse_yy
        + np.hypot(inverse_xx-inverse_yy,2.0*inverse_xy)
    )
    metric_lipschitz_scale = max(float(np.max(inverse_high)),1.0)
    step = float(fidelity)/(8.0*metric_lipschitz_scale)
    previous_objective = np.inf
    maximum_increase = 0.0
    for _ in range(int(iterations)):
        px = lxx*qx + lxy*qy
        py = lxy*qx + lyy*qy
        u = g + div(px, py)/float(fidelity)
        gx, gy = grad(u)
        # L is symmetric: q descent is q + step L grad(u).
        candidate_x = qx + step*(lxx*gx + lxy*gy)
        candidate_y = qy + step*(lxy*gx + lyy*gy)
        norm = np.hypot(candidate_x, candidate_y)
        scale = 1.0/np.maximum(norm, 1.0)
        qx, qy = candidate_x*scale, candidate_y*scale
        px = lxx*qx + lxy*qy
        py = lxy*qx + lyy*qy
        u = g + div(px, py)/float(fidelity)
        objective = 0.5*float(np.vdot(u, u).real)
        if np.isfinite(previous_objective):
            maximum_increase = max(maximum_increase, objective-previous_objective)
        previous_objective = objective
    return u, (qx, qy), {
        "dual_objective": previous_objective,
        "maximum_objective_increase": maximum_increase,
        "maximum_whitened_flux_norm": float(np.max(np.hypot(qx, qy))),
        "metric_lipschitz_scale": metric_lipschitz_scale,
    }


def metric_meyer(
    image: np.ndarray,
    inverse_sqrt: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    texture_inverse_sqrt: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    lam: float = 0.05,
    mu: float = 40.0,
    outer_iterations: int = 96,
    inner_iterations: int = 8,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Warm-interleaved Meyer alternation in one frozen transport metric."""

    f = np.asarray(image, dtype=np.float64)
    texture_geometry = (
        inverse_sqrt if texture_inverse_sqrt is None else texture_inverse_sqrt
    )
    u = np.zeros_like(f)
    v = np.zeros_like(f)
    state_u = state_v = None
    last_delta = np.inf
    maximum_increase = 0.0
    flux_norm = 0.0
    for outer in range(int(outer_iterations)):
        u_new, state_u, diagnostic_u = metric_rof(
            f-v, lam, inverse_sqrt, state_u, inner_iterations)
        remainder = f-u_new
        smooth, state_v, diagnostic_v = metric_rof(
            remainder, 1.0/mu, texture_geometry, state_v, inner_iterations)
        v_new = remainder-smooth
        last_delta = float(np.linalg.norm(u_new-u)/max(np.linalg.norm(u_new), 1e-30))
        u, v = u_new, v_new
        maximum_increase = max(
            maximum_increase,
            diagnostic_u["maximum_objective_increase"],
            diagnostic_v["maximum_objective_increase"],
        )
        flux_norm = max(
            flux_norm,
            diagnostic_u["maximum_whitened_flux_norm"],
            diagnostic_v["maximum_whitened_flux_norm"],
        )
    # Assign the model residual to cartoon so the public split is exact.
    cartoon = f-v
    return cartoon, v, {
        "outer_iterations": int(outer_iterations),
        "inner_iterations": int(inner_iterations),
        "last_relative_cartoon_change": last_delta,
        "maximum_inner_objective_increase": maximum_increase,
        "maximum_whitened_flux_norm": flux_norm,
        "recomposition_linf": float(np.max(np.abs(f-cartoon-v))),
    }


def scenes(size: int = 128) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    yy, xx = np.mgrid[:size, :size].astype(np.float64)
    x, y = (xx+0.5)/size, (yy+0.5)/size
    cartoon = 42.0 + 74.0*x + 28.0*(y > 0.62-0.25*x)
    disk = (x-0.67)**2+(y-0.34)**2 < 0.16**2
    cartoon = cartoon + 62.0*disk
    window = ((x-0.46)/0.35)**2+((y-0.58)/0.27)**2 < 1.0
    coherent = 18.0*np.sin(2*np.pi*(18*x+7*y))*window
    chirp = 16.0*np.sin(2*np.pi*(3*x+18*x*x))*(x > 0.12)*(x < 0.88)
    rng = np.random.default_rng(4051)
    noise = 10.0*rng.standard_normal((size, size))
    return {
        "coherent oscillation": (cartoon+coherent, cartoon, coherent),
        "quadratic chirp": (cartoon+chirp, cartoon, chirp),
        "white noise": (cartoon+noise, cartoon, noise),
        "mixed": (cartoon+coherent+0.55*chirp+0.45*noise,
                  cartoon, coherent+0.55*chirp+0.45*noise),
    }


def relative_mse(actual: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean((actual-truth)**2)/max(np.mean(truth**2), 1e-30))


def ranged(array: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(array)), float(np.max(array))
    return np.clip((array-lo)/max(hi-lo, 1e-30), 0, 1)


def signed(array: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(0.5+0.5*np.asarray(array)/max(scale, 1e-30), 0, 1)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = scenes()
    identity = (np.ones((128,128)), np.zeros((128,128)), np.ones((128,128)))
    records: dict[str, dict[str, float | dict[str, float]]] = {}
    outputs: dict[str, np.ndarray] = {}
    rows = []
    for name, (image, cartoon_truth, texture_truth) in cases.items():
        plan = bfft.MeyerPlan(image.shape, lam=0.05, mu=40.0, passes=96, threads=1)
        ordinary_cartoon, ordinary_texture = plan.split_legacy(image)
        _centre, variance, _radius, _witness = directional_noise_witnesses(
            image, ordinary_cartoon)
        metric = continual_transport_metric(ordinary_cartoon, variance)
        eikonal_cartoon, eikonal_texture, eikonal_diag = metric_meyer(
            image, inverse_sqrt_metric(metric))
        identity_cartoon, identity_texture, identity_diag = metric_meyer(
            image, identity)
        hybrid_cartoon, hybrid_texture, hybrid_diag = metric_meyer(
            image, identity, texture_inverse_sqrt=inverse_sqrt_metric(metric))
        record = {
            "ordinary_cartoon_relative_mse": relative_mse(ordinary_cartoon, cartoon_truth),
            "ordinary_texture_relative_mse": relative_mse(ordinary_texture, texture_truth),
            "identity_cartoon_relative_mse": relative_mse(identity_cartoon, cartoon_truth),
            "identity_texture_relative_mse": relative_mse(identity_texture, texture_truth),
            "eikonal_cartoon_relative_mse": relative_mse(eikonal_cartoon, cartoon_truth),
            "eikonal_texture_relative_mse": relative_mse(eikonal_texture, texture_truth),
            "hybrid_cartoon_relative_mse": relative_mse(hybrid_cartoon, cartoon_truth),
            "hybrid_texture_relative_mse": relative_mse(hybrid_texture, texture_truth),
            "ordinary_recomposition_linf": float(np.max(np.abs(image-ordinary_cartoon-ordinary_texture))),
            "metric_determinant_minimum": float(metric["metric_determinant_minimum"]),
            "metric_condition_p90": float(metric["metric_condition_p90"]),
            "identity_diagnostic": identity_diag,
            "eikonal_diagnostic": eikonal_diag,
            "hybrid_diagnostic": hybrid_diag,
        }
        records[name] = record
        key = name.replace(" ", "_")
        for label, value in (
            ("image", image), ("cartoon_truth", cartoon_truth),
            ("texture_truth", texture_truth),
            ("ordinary_cartoon", ordinary_cartoon),
            ("ordinary_texture", ordinary_texture),
            ("identity_cartoon", identity_cartoon),
            ("identity_texture", identity_texture),
            ("eikonal_cartoon", eikonal_cartoon),
            ("eikonal_texture", eikonal_texture),
            ("hybrid_cartoon", hybrid_cartoon),
            ("hybrid_texture", hybrid_texture),
        ):
            outputs[f"{key}_{label}"] = value.astype(np.float32)
        rows.append((name, image, cartoon_truth, texture_truth,
                     ordinary_cartoon, ordinary_texture,
                     eikonal_cartoon, eikonal_texture,
                     hybrid_cartoon, hybrid_texture))

    (OUT/"metrics.json").write_text(json.dumps(records, indent=2)+"\n")
    np.savez_compressed(OUT/"meyer_eikonal_split.npz", **outputs)
    cell, label_width, header = 128, 135, 30
    canvas = Image.new("RGB", (label_width+9*cell, header+len(rows)*cell), "white")
    draw = ImageDraw.Draw(canvas)
    titles = ("source", "truth cartoon", "truth texture", "ordinary cartoon",
              "ordinary texture", "Eikonal-both cartoon", "Eikonal-both texture",
              "Eikonal-flux cartoon", "Eikonal-flux texture")
    for j,title in enumerate(titles):
        draw.text((label_width+j*cell+4,8), title, fill="black")
    for i,row in enumerate(rows):
        name,*planes=row
        y0=header+i*cell
        draw.text((4,y0+8),name,fill="black")
        scale=max(float(np.max(np.abs(planes[2]))),1.0)
        panels=(ranged(planes[0]),ranged(planes[1]),signed(planes[2],scale),
                ranged(planes[3]),signed(planes[4],scale),
                ranged(planes[5]),signed(planes[6],scale),
                ranged(planes[7]),signed(planes[8],scale))
        for j,panel in enumerate(panels):
            gray=np.uint8(panel*255)
            canvas.paste(Image.fromarray(gray).convert("RGB"),(label_width+j*cell,y0))
    canvas.save(OUT/"meyer_eikonal_comparison.png")


if __name__ == "__main__":
    main()
