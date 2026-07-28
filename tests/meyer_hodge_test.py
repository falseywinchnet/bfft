#!/usr/bin/env python3
"""One-shot Fourier/Hodge ROF accelerator integration checks.

Run from the repository root:

    python tests/meyer_hodge_test.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import bfft  # noqa: E402
import cartoon_fourier_transport as prototype  # noqa: E402


def scene(n=128, seed=5):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:n, :n].astype(np.float64)
    image = 45.0 + 125.0 * x / n
    image += 38.0 * (y > n / 2)
    image += 21.0 * np.cos(2.0 * np.pi * (x + y) / 7.0)
    image += 13.0 * np.cos(2.0 * np.pi * y / 5.0) * (x < n / 2)
    return image + 2.0 * rng.standard_normal((n, n))


def relative_l2(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def main():
    ok = True
    image = scene()
    c, eta = 0.05, 0.10
    plan = bfft.MeyerPlan(image.shape, threads=1)

    # The native one-shot must reproduce the independent NumPy derivation
    # before any subsequent Split-Bregman continuation can obscure it.
    trace, flux, _ = prototype.bregman_trace(image, c, eta, 4)
    raw = prototype.routed_flux_proposal(trace[4], flux[4], image, c)
    expected = prototype.make_drop(raw, trace[4], image, c).u
    actual = plan.rof(
        image, c, eta=eta, sweeps=4, tol=0.0, hodge_after=4
    )
    error = float(np.max(np.abs(actual - expected)) / np.max(np.abs(image)))
    equivalent = (
        error < 1e-12
        and plan.last_rof_sweeps == 4
        and plan.last_rof_hodge_applied
    )
    print(f"1. native/prototype relative max error {error:.3e}:",
          "ok" if equivalent else "FAIL")
    ok &= equivalent

    # At a fixed early budget the accepted closure must move both the exact
    # isotropic objective and the answer toward a deeply converged solve.
    reference = plan.rof(image, c, eta=eta, sweeps=1024, tol=0.0)
    plain8 = plan.rof(image, c, eta=eta, sweeps=8, tol=0.0)
    accelerated8 = plan.rof(
        image, c, eta=eta, sweeps=8, tol=0.0, hodge_after=4
    )
    plain_objective = prototype.objective(plain8, image, c)
    accelerated_objective = prototype.objective(accelerated8, image, c)
    plain_error = relative_l2(plain8, reference)
    accelerated_error = relative_l2(accelerated8, reference)
    effective = (
        accelerated_objective < plain_objective
        and accelerated_error < plain_error
    )
    print("2. fixed 8-sweep budget:")
    print(f"   objective {plain_objective:.9e} -> "
          f"{accelerated_objective:.9e}")
    print(f"   reference error {plain_error:.3e} -> "
          f"{accelerated_error:.3e}:",
          "ok" if effective else "FAIL")
    ok &= effective

    # Continued iteration must remain on the same high-precision target.
    plain256 = plan.rof(image, c, eta=eta, sweeps=256, tol=0.0)
    accelerated256 = plan.rof(
        image, c, eta=eta, sweeps=256, tol=0.0, hodge_after=4
    )
    plain_deep_error = relative_l2(plain256, reference)
    accelerated_deep_error = relative_l2(accelerated256, reference)
    same_target = accelerated_deep_error <= 1.05 * plain_deep_error
    print(f"3. deep reference error {plain_deep_error:.3e} vs "
          f"{accelerated_deep_error:.3e}:",
          "ok" if same_target else "FAIL")
    ok &= same_target

    # The public arbitrary-size shim must preserve its normal pad/crop policy.
    arbitrary = image[3:74, 9:109]
    shim = bfft.rof(
        arbitrary, c=c, eta=eta, sweeps=8, tol=0.0, hodge_after=4
    )
    shim_ok = shim.shape == arbitrary.shape and np.isfinite(shim).all()
    print("4. arbitrary-size accelerated ROF:",
          "ok" if shim_ok else "FAIL")
    ok &= shim_ok

    # FACR/Neumann do not own the full periodic two-axis projector.
    unsupported = False
    try:
        facr = bfft.MeyerPlan((95, 128), solver=1)
        facr.rof(np.ones((95, 128)), c, sweeps=8, hodge_after=4)
    except RuntimeError:
        unsupported = True
    print("5. FACR rejection:", "ok" if unsupported else "FAIL")
    ok &= unsupported

    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
