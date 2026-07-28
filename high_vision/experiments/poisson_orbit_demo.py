#!/usr/bin/env python3
"""Low-light reconstruction without choosing a registration per frame.

Each observation is an independently Poisson-sampled, randomly translated
copy of a small skimage scene.  The photons are thinned into three independent
virtual gatherers.  Cross-bispectra of those gatherers are translation
invariant and have no same-photon self product, so they can be accumulated
without estimating any frame's shift.

This is a deliberately small falsification rig, not a production night-vision
algorithm.  It compares:

* one gain-expanded photon frame;
* an unregistered average;
* an average using unrelated ("fraudulent") shift guesses;
* a reconstruction from power and thinned cross-bispectrum alone.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.sparse.linalg import LinearOperator, cg
from skimage import data, transform
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


@dataclass(frozen=True)
class Camera:
    size: int = 48
    frames: int = 4096
    photons_at_white: float = 0.75
    dark_electrons: float = 0.002
    read_noise_electrons: float = 0.0
    gatherers: int = 3
    seed: int = 7
    batch: int = 128
    bispectrum_steps: int = 20
    full_circle_support: bool = False
    poisson_phase_seed: bool = False
    optimizer_iterations: int = 180


def source_image(size: int) -> np.ndarray:
    image = data.camera().astype(np.float64) / 255.0
    image = transform.resize(
        image, (size, size), anti_aliasing=True, preserve_range=True)
    # Retain a small black pedestal as a camera would; it prevents the
    # synthetic cyclic boundary from becoming an unrealistically perfect cue.
    return np.clip(0.02 + 0.96 * image, 0.0, 1.0)


def signed_frequency(index: int, length: int) -> int:
    return index if index <= length // 2 else index - length


def select_steps(
    height: int,
    width: int,
    count: int,
    full_circle_support: bool = False,
) -> list[tuple[int, int]]:
    """Choose a fixed radial stencil without consulting the source.

    Full-circle mode retains antipodal witnesses. Although their noiseless
    biphases are Hermitian-related, the thinned finite-photon cross moments are
    distinct supported measurements and should not be discarded a priori.
    """
    candidates = []
    for y in range(height):
        fy = signed_frequency(y, height)
        for x in range(width):
            fx = signed_frequency(x, width)
            radius2 = fx * fx + fy * fy
            if radius2 == 0 or radius2 > 7 * 7:
                continue
            angle = math.atan2(fy, fx)
            candidates.append((radius2, abs(angle), angle, y, x))
    candidates.sort()
    steps = [(0, 1), (1, 0)]
    if count <= len(steps):
        return steps[:count]
    for _, _, _, y, x in candidates:
        step = (y, x)
        opposite = ((-y) % height, (-x) % width)
        if step in steps:
            continue
        if not full_circle_support and opposite in steps:
            continue
        steps.append(step)
        if len(steps) >= count:
            break
    return steps


def phase_seed(bispectrum: np.ndarray) -> np.ndarray:
    """A path-integrated seed using the two unit-step biphases."""
    _, height, width = bispectrum.shape
    phase = np.zeros((height, width), dtype=np.float64)
    # Gauge: phase at DC and at the two unit frequencies is zero.  This merely
    # chooses one global cyclic translation of the answer.
    for y in range(1, height):
        k = (y - 1, 0)
        phase[y, 0] = phase[y - 1, 0] - np.angle(
            bispectrum[1, k[0], k[1]])
    for y in range(height):
        for x in range(1, width):
            k = (y, x - 1)
            phase[y, x] = phase[y, x - 1] - np.angle(
                bispectrum[0, k[0], k[1]])
    return phase


def solve_phase(
    bispectrum: np.ndarray,
    coherence: np.ndarray,
    steps: list[tuple[int, int]],
    iterations: int,
    initial_phase: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Fit all biphase constraints without introducing frame shifts."""
    step_count, height, width = bispectrum.shape
    assert step_count == len(steps)
    count = height * width
    grid = np.arange(count, dtype=np.int64).reshape(height, width)
    left = []
    step_index = []
    total = []
    target = []
    weight = []
    for s, (sy, sx) in enumerate(steps):
        shifted = np.roll(np.roll(grid, -sy, axis=0), -sx, axis=1)
        left.append(grid.ravel())
        step_index.append(np.full(count, grid[sy, sx], dtype=np.int64))
        total.append(shifted.ravel())
        target.append(np.angle(bispectrum[s]).ravel())
        weight.append(coherence[s].ravel())
    left = np.concatenate(left)
    step_index = np.concatenate(step_index)
    total = np.concatenate(total)
    target = np.concatenate(target)
    weight = np.concatenate(weight)
    # Discard constraints whose cross moment did not rise above its empirical
    # fluctuation floor. A soft cubic weight retains graded evidence.
    floor = np.quantile(weight, 0.35)
    keep = weight > max(floor, 0.015)
    left, step_index, total, target, weight = (
        value[keep] for value in
        (left, step_index, total, target, weight))
    weight = np.minimum(weight, 1.0) ** 3
    weight /= max(float(np.mean(weight)), 1e-12)

    if initial_phase is None:
        seed = phase_seed(bispectrum[:2]).ravel()
    else:
        if initial_phase.shape != (height, width):
            raise ValueError("initial phase shape does not match bispectrum")
        seed = np.asarray(initial_phase, dtype=np.float64).ravel().copy()
    seed -= seed[0]
    # Bispectrum determines an image only up to cyclic translation. Pin DC and
    # the two unit-frequency phases to remove that flat two-dimensional gauge
    # from the numerical solve; this does not supply registration information.
    fixed = np.zeros(count, dtype=bool)
    fixed[[0, 1, width]] = True
    free = ~fixed

    def objective(phase_free: np.ndarray) -> tuple[float, np.ndarray]:
        phase = np.zeros(count, dtype=np.float64)
        phase[free] = phase_free
        residual = phase[left] + phase[step_index] - phase[total] - target
        value = np.sum(weight * (1.0 - np.cos(residual))) / len(residual)
        force = weight * np.sin(residual) / len(residual)
        gradient = np.zeros(count, dtype=np.float64)
        np.add.at(gradient, left, force)
        np.add.at(gradient, step_index, force)
        np.add.at(gradient, total, -force)
        return float(value), gradient[free]

    result = minimize(
        objective,
        seed[free],
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": iterations, "ftol": 1e-8, "gtol": 1e-7},
    )
    phase = np.zeros(count, dtype=np.float64)
    phase[free] = result.x
    phase = phase.reshape(height, width)
    return phase, {
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_iterations": int(result.nit),
        "phase_objective": float(result.fun),
        "constraints_used": int(len(weight)),
    }


def solve_phase_poisson(
    bispectrum: np.ndarray,
    coherence: np.ndarray,
    iterations: int,
) -> tuple[np.ndarray, dict]:
    """Integrate two wrapped biphase-gradient planes in nearly linear time.

    For the unit frequency ex,

        arg B(k, ex) = phi(k) + phi(ex) - phi(k + ex).

    Choosing the unidentifiable translation gauge phi(ex)=0 makes
    -arg B(k, ex) the wrapped forward x-gradient of Fourier phase. The ey
    relation supplies y. A coherence-weighted Poisson solve integrates both
    fields globally, so path noise does not accumulate across rows.
    """
    if bispectrum.shape[0] < 2:
        raise ValueError("Poisson phase integration needs ex and ey moments")
    _, height, width = bispectrum.shape
    target_x = -np.angle(bispectrum[0])
    target_y = -np.angle(bispectrum[1])
    weight_x = np.clip(coherence[0], 0.0, 1.0) ** 2
    weight_y = np.clip(coherence[1], 0.0, 1.0) ** 2
    floor = max(float(np.quantile(
        np.concatenate((weight_x.ravel(), weight_y.ravel())), 0.25)), 1e-5)
    weight_x = np.where(weight_x >= floor, weight_x, 0.0)
    weight_y = np.where(weight_y >= floor, weight_y, 0.0)
    scale = max(float(np.mean(weight_x + weight_y)), 1e-8)
    weight_x /= scale
    weight_y /= scale
    ridge = 1e-6

    def forward_x(field):
        return np.roll(field, -1, axis=1) - field

    def forward_y(field):
        return np.roll(field, -1, axis=0) - field

    def adjoint_x(field):
        return np.roll(field, 1, axis=1) - field

    def adjoint_y(field):
        return np.roll(field, 1, axis=0) - field

    def apply(flat):
        field = flat.reshape(height, width)
        result = (
            adjoint_x(weight_x * forward_x(field))
            + adjoint_y(weight_y * forward_y(field))
            + ridge * field
        )
        return result.ravel()

    operator = LinearOperator(
        (height * width, height * width), matvec=apply, dtype=np.float64)
    phase = np.zeros((height, width), dtype=np.float64)
    total_cg_iterations = 0
    cg_status = 0
    # Rewrap after each solve. This is the standard least-squares phase
    # unwrapping correction and handles occasional 2pi branch inconsistencies.
    for _ in range(3):
        residual_x = np.angle(np.exp(
            1j * (target_x - forward_x(phase))))
        residual_y = np.angle(np.exp(
            1j * (target_y - forward_y(phase))))
        right = (
            adjoint_x(weight_x * residual_x)
            + adjoint_y(weight_y * residual_y)
        ).ravel()
        counter = [0]

        def counted(_):
            counter[0] += 1

        correction, status = cg(
            operator, right, rtol=2e-5, atol=0.0,
            maxiter=iterations, callback=counted)
        total_cg_iterations += counter[0]
        cg_status = max(cg_status, int(status))
        phase += correction.reshape(height, width)
        phase -= phase[0, 0]

    residual_x = np.angle(np.exp(
        1j * (forward_x(phase) - target_x)))
    residual_y = np.angle(np.exp(
        1j * (forward_y(phase) - target_y)))
    objective = float(np.mean(
        weight_x * residual_x ** 2 + weight_y * residual_y ** 2))
    return phase, {
        "phase_solver": "weighted_poisson",
        "optimizer_success": cg_status == 0,
        "optimizer_message": (
            "converged" if cg_status == 0 else f"CG status {cg_status}"),
        "optimizer_iterations": total_cg_iterations,
        "phase_objective": objective,
        "constraints_used": int(np.count_nonzero(weight_x)
                                + np.count_nonzero(weight_y)),
    }


def hermitian_spectrum(magnitude: np.ndarray, phase: np.ndarray) -> np.ndarray:
    spectrum = magnitude * np.exp(1j * phase)
    height, width = spectrum.shape
    opposite = spectrum[
        (-np.arange(height)) % height][:, (-np.arange(width)) % width]
    spectrum = 0.5 * (spectrum + np.conj(opposite))
    spectrum[0, 0] = abs(spectrum[0, 0])
    return spectrum


def best_cyclic_alignment(
    estimate: np.ndarray, reference: np.ndarray
) -> tuple[np.ndarray, tuple[int, int]]:
    correlation = np.fft.ifft2(
        np.fft.fft2(reference) * np.conj(np.fft.fft2(estimate))).real
    shift = np.unravel_index(np.argmax(correlation), correlation.shape)
    aligned = np.roll(estimate, shift, axis=(0, 1))
    return aligned, (int(shift[0]), int(shift[1]))


def metrics(estimate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    return {
        "psnr_db": float(peak_signal_noise_ratio(
            reference, estimate, data_range=1.0)),
        "ssim": float(structural_similarity(
            reference, estimate, data_range=1.0)),
    }


def run(config: Camera) -> tuple[dict, dict[str, np.ndarray]]:
    if config.gatherers != 3:
        raise ValueError("the cross-bispectrum demo requires three gatherers")
    rng = np.random.default_rng(config.seed)
    source = source_image(config.size)
    height, width = source.shape
    photon_rate = config.photons_at_white * source
    dark_per_gatherer = config.dark_electrons / config.gatherers
    # The power estimator uses the full Poisson frame with its exact white
    # shot-noise floor removed. Bispectrum uses independent thinned witnesses.
    power_sum = np.zeros_like(source)
    observed_sum = np.zeros_like(source)
    fraudulent_sum = np.zeros_like(source)
    first_frame = None
    bispectrum_sum = None
    bispectrum_abs_sum = None
    frames_done = 0

    steps = select_steps(
        height,
        width,
        config.bispectrum_steps,
        config.full_circle_support,
    )

    started = time.perf_counter()
    while frames_done < config.frames:
        batch = min(config.batch, config.frames - frames_done)
        shifts_y = rng.integers(0, height, size=batch)
        shifts_x = rng.integers(0, width, size=batch)
        gatherers = np.empty(
            (3, batch, height, width), dtype=np.float64)
        for frame in range(batch):
            shifted_rate = np.roll(
                photon_rate, (shifts_y[frame], shifts_x[frame]), axis=(0, 1))
            for gatherer in range(3):
                sample = rng.poisson(
                    (shifted_rate + config.dark_electrons) / 3.0)
                if config.read_noise_electrons:
                    sample = sample + rng.normal(
                        0.0,
                        config.read_noise_electrons / math.sqrt(3.0),
                        sample.shape,
                    )
                gatherers[gatherer, frame] = sample - dark_per_gatherer

        full = np.sum(gatherers, axis=0)
        if first_frame is None:
            first_frame = full[0].copy()
        observed_sum += np.sum(full, axis=0)
        fake_y = rng.integers(0, height, size=batch)
        fake_x = rng.integers(0, width, size=batch)
        for frame in range(batch):
            fraudulent_sum += np.roll(
                full[frame], (-fake_y[frame], -fake_x[frame]), axis=(0, 1))

        spectra = np.fft.fft2(gatherers, axes=(-2, -1))
        full_spectrum = np.sum(spectra, axis=0)
        power_sum += np.sum(np.abs(full_spectrum) ** 2, axis=0)
        if bispectrum_sum is None:
            shape = (len(steps), height, width)
            bispectrum_sum = np.zeros(shape, dtype=np.complex128)
            bispectrum_abs_sum = np.zeros(shape, dtype=np.float64)
        permutations = (
            (0, 1, 2), (1, 0, 2), (0, 2, 1),
            (2, 0, 1), (1, 2, 0), (2, 1, 0),
        )
        for s, (sy, sx) in enumerate(steps):
            accumulated = np.zeros((batch, height, width), np.complex128)
            accumulated_abs = np.zeros(
                (batch, height, width), np.float64)
            for a, b, c in permutations:
                product = (
                    spectra[a]
                    * spectra[b, :, sy, sx, None, None]
                    * np.conj(np.roll(
                        np.roll(spectra[c], -sy, axis=1),
                        -sx, axis=2))
                )
                accumulated += product
                accumulated_abs += np.abs(product)
            accumulated /= len(permutations)
            accumulated_abs /= len(permutations)
            bispectrum_sum[s] += np.sum(accumulated, axis=0)
            bispectrum_abs_sum[s] += np.sum(accumulated_abs, axis=0)
        frames_done += batch

    observed_mean = observed_sum / (
        config.frames * config.photons_at_white)
    fraudulent_mean = fraudulent_sum / (
        config.frames * config.photons_at_white)
    # E|FFT(Y)|^2 = |FFT(lambda)|^2 + sum(lambda) for Poisson Y.
    average_total_electrons = float(np.sum(
        np.maximum(first_frame, 0.0)))  # diagnostic only
    # Estimate the Poisson white floor from the observed DC count. The only
    # supplied camera calibration is the known dark-current expectation.
    expected_shot_floor = max(
        float(np.sum(observed_sum)) / config.frames
        + height * width * config.dark_electrons,
        0.0,
    )
    expected_read_floor = (
        height * width * config.read_noise_electrons ** 2)
    power = power_sum / config.frames - (
        expected_shot_floor + expected_read_floor)
    magnitude = np.sqrt(np.maximum(power.real, 0.0)) / max(
        config.photons_at_white, 1e-12)
    magnitude[0, 0] = max(
        float(np.sum(observed_sum))
        / (config.frames * config.photons_at_white),
        0.0,
    )

    bispectrum = bispectrum_sum / config.frames
    coherence = np.abs(bispectrum_sum) / np.maximum(
        bispectrum_abs_sum, 1e-12)
    initial_phase = None
    seed_info = {"phase_seed": "path_integrated"}
    if config.poisson_phase_seed:
        initial_phase, poisson_info = solve_phase_poisson(
            bispectrum[:2],
            coherence[:2],
            min(config.optimizer_iterations, 120),
        )
        # Remove the two-dimensional translation gauge so the unit-frequency
        # phases agree with the nonlinear solver's fixed gauge.
        fy = np.asarray([
            signed_frequency(y, height) for y in range(height)])
        fx = np.asarray([
            signed_frequency(x, width) for x in range(width)])
        initial_phase -= (
            fy[:, None] * initial_phase[1, 0]
            + fx[None, :] * initial_phase[0, 1]
        )
        seed_info = {
            "phase_seed": "weighted_poisson_then_circle",
            "poisson_seed_objective": poisson_info["phase_objective"],
            "poisson_seed_iterations": poisson_info["optimizer_iterations"],
        }
    phase, phase_info = solve_phase(
        bispectrum,
        coherence,
        steps,
        config.optimizer_iterations,
        initial_phase=initial_phase,
    )
    orbit = np.fft.ifft2(hermitian_spectrum(magnitude, phase)).real
    orbit, recovered_shift = best_cyclic_alignment(orbit, source)
    orbit = np.clip(orbit, 0.0, 1.0)

    single = np.clip(first_frame / config.photons_at_white, 0.0, 1.0)
    # Unregistered and fraudulent means should approach a flat cyclic average.
    observed_mean = np.clip(observed_mean, 0.0, 1.0)
    fraudulent_mean = np.clip(fraudulent_mean, 0.0, 1.0)
    elapsed = time.perf_counter() - started
    result = {
        "camera": asdict(config),
        "elapsed_seconds": elapsed,
        "mean_electrons_per_frame": float(np.sum(
            photon_rate + config.dark_electrons)),
        "nonzero_fraction_first_frame": float(np.mean(first_frame > 0)),
        "diagnostic_first_frame_electrons": average_total_electrons,
        "steps": [[int(y), int(x)] for y, x in steps],
        "recovered_global_shift": list(recovered_shift),
        **seed_info,
        **phase_info,
        "metrics": {
            "single": metrics(single, source),
            "unregistered_mean": metrics(observed_mean, source),
            "fraudulent_mean": metrics(fraudulent_mean, source),
            "orbit_cross_bispectrum": metrics(orbit, source),
        },
    }
    images = {
        "source": source,
        "single": single,
        "unregistered": observed_mean,
        "fraudulent": fraudulent_mean,
        "orbit": orbit,
        "coherence": np.mean(coherence, axis=0),
    }
    return result, images


def render(images: dict[str, np.ndarray], result: dict, path: Path) -> None:
    panels = (
        ("source", "Ground truth", None),
        ("single", "One photon frame", "single"),
        ("unregistered", "Unregistered mean", "unregistered_mean"),
        ("fraudulent", "Fraudulent-shift mean", "fraudulent_mean"),
        ("orbit", "Orbit cross-bispectrum", "orbit_cross_bispectrum"),
        ("coherence", "Mean invariant coherence", None),
    )
    figure, axes = plt.subplots(2, 3, figsize=(12, 8.6))
    for axis, (key, title, metric_key) in zip(axes.ravel(), panels):
        image = images[key]
        axis.imshow(image, cmap="magma" if key == "coherence" else "gray",
                    vmin=0.0, vmax=1.0)
        if metric_key is not None:
            score = result["metrics"][metric_key]
            title += f"\n{score['psnr_db']:.2f} dB, SSIM {score['ssim']:.3f}"
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle(
        f"{result['camera']['frames']} frames, "
        f"{result['camera']['photons_at_white']:.3g} photon/white pixel/frame, "
        "no registration estimates",
        fontsize=13,
    )
    figure.subplots_adjust(
        left=0.025, right=0.985, bottom=0.025, top=0.89,
        wspace=0.12, hspace=0.17)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=48)
    parser.add_argument("--frames", type=int, default=4096)
    parser.add_argument("--photons", type=float, default=0.75)
    parser.add_argument("--dark", type=float, default=0.002)
    parser.add_argument("--read-noise", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument(
        "--full-circle-support",
        action="store_true",
        help="Retain antipodal projection steps for comparison")
    parser.add_argument(
        "--poisson-seed",
        action="store_true",
        help="Use the two-plane Poisson solution only as a circle-fit seed")
    parser.add_argument("--iterations", type=int, default=220)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output", type=Path,
        default=Path("high_vision/out/poisson_orbit_demo.png"))
    parser.add_argument(
        "--json", type=Path,
        default=Path("high_vision/out/poisson_orbit_demo.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Camera(
        size=args.size,
        frames=args.frames,
        photons_at_white=args.photons,
        dark_electrons=args.dark,
        read_noise_electrons=args.read_noise,
        bispectrum_steps=args.steps,
        full_circle_support=args.full_circle_support,
        poisson_phase_seed=args.poisson_seed,
        optimizer_iterations=args.iterations,
        seed=args.seed,
    )
    result, images = run(config)
    render(images, result, args.output)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"wrote {args.output}")
    print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
