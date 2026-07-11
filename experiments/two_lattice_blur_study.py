#!/usr/bin/env python3
"""The two §4c measurements: complementary blur and soft-weight seams.

A. Complementary blur (no source sharp anywhere).  Two captures of one
   ground-truth scene through disk PSFs of different radii.  Disk OTFs have
   zeros at different frequencies, so at each (patch, frequency) at least
   one capture retains energy — per-coefficient consensus across captures
   covers what each blur destroyed.  Beyond its first OTF zero a capture's
   phase is sign-flipped, so pixel-domain fusion is structurally unable to
   use that band; magnitude projection + positivity re-retrieves phase.
   Configurations (weights ladder, all measured quantities):
     per-patch softmax weights  (the MFF configuration; blind to frequency)
     per-coefficient softmax    (frequency-resolved reliability)
     max-consensus target       (elementwise max magnitude, hard w=1)
   against capture/average baselines, PSNR to ground truth.

B. Negative control: identical blurs.  No complementary information exists;
   an honest method must not hallucinate sharpness (PSNR ~= capture).

C. Soft-weight seam.  Spatially-varying focus with a wide transition band
   where weights are necessarily ambiguous.  Measure the projection stage's
   gain over the weighted-average seed as a function of position: the gain
   should concentrate in the seam.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE / "out"

from two_lattice_mff import PatchFamily  # noqa: E402

SCRATCH = Path("/private/tmp/claude-501/-Users-quentinkuttenkuler-bfft/"
               "7ec9c357-2c5f-45c7-83cc-25520967d9ba/scratchpad")


def ground_truth(n=256):
    """Sharp real texture: the in-focus shell center of MFFW3/4 source 1."""
    import matplotlib.image as mpimg
    img = mpimg.imread(str(SCRATCH / "mff_image1.png")).astype(np.float64)
    return img[60:60 + n, 100:100 + n, :3].mean(axis=2)


def disk_kernel(radius):
    r = int(np.ceil(radius))
    y, x = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float64)
    k = (x ** 2 + y ** 2 <= radius ** 2 + 1e-9).astype(np.float64)
    return k / k.sum()


def convolve_reflect(img, k):
    r = k.shape[0] // 2
    pad = np.pad(img, r, mode="reflect")
    kk = np.zeros_like(pad)
    kk[:k.shape[0], :k.shape[1]] = k
    kk = np.roll(kk, (-r, -r), axis=(0, 1))      # kernel center at origin
    out = np.fft.irfft2(np.fft.rfft2(pad) * np.fft.rfft2(kk), s=pad.shape)
    return out[r:r + img.shape[0], r:r + img.shape[1]]


def psnr(u, ref):
    rng = ref.max() - ref.min()
    return 10 * np.log10(rng ** 2 / np.mean((u - ref) ** 2))


def coeff_weights(fams, gamma=2.0):
    m = np.stack([f.target for f in fams]) ** gamma
    return m / (m.sum(axis=0, keepdims=True) + 1e-30)


def patch_weights(fams, gamma=2.0, low_r=2.0):
    p = fams[0].patch
    f = np.fft.fftfreq(p) * p
    rr = np.sqrt(f[:, None] ** 2 + f[None, :] ** 2)
    high = (rr > low_r)[None]
    e = np.stack([np.sum((f_.target ** 2) * high, axis=(1, 2))
                  for f_ in fams]) ** gamma
    return e / (e.sum(axis=0, keepdims=True) + 1e-30)


def project_coeff_weighted(fam, u, w_coeff):
    S = fam.analyze(u)
    radial = fam.target * S / (np.abs(S) + 1e-12)
    return fam.synthesize((1.0 - w_coeff) * S + w_coeff * radial, u)


def run(caps, mode, iters=24, beta=0.7, gamma=2.0, seed_img=None):
    fams = [PatchFamily(c) for c in caps]
    if mode == "max":
        consensus = PatchFamily(caps[0])
        consensus.target = np.maximum.reduce([f.target for f in fams])
        fams_run, wc = [consensus], [np.ones_like(consensus.target)]
    elif mode == "coeff":
        wc = list(coeff_weights(fams, gamma))
        fams_run = fams
    else:                                             # per-patch (MFF config)
        wp = patch_weights(fams, gamma)
        wc = [np.broadcast_to(wp[i][:, None, None], fams[i].target.shape)
              for i in range(len(fams))]
        fams_run = fams
    latent = (sum(caps) / len(caps)) if seed_img is None else seed_img.copy()
    prev = latent.copy()
    for _ in range(iters):
        trial = latent + beta * (latent - prev)
        prev = latent
        out = trial
        for fam, w in zip(fams_run, wc):
            out = project_coeff_weighted(fam, out, w)
        latent = np.clip(out, 0.0, None)
    return latent


def part_a():
    print("=== A. complementary blur: disk r=3 vs r=5, noise 0.002 ===")
    gt = ground_truth()
    rng = np.random.default_rng(9)
    caps = [convolve_reflect(gt, disk_kernel(r)) + 0.002 *
            rng.standard_normal(gt.shape) for r in (3.0, 5.0)]
    rows = [("capture r=3", caps[0]), ("capture r=5", caps[1]),
            ("pixel average", sum(caps) / 2)]
    for mode, label in (("patch", "proj per-patch w"),
                        ("coeff", "proj per-coeff w"),
                        ("max", "proj max-consensus")):
        rows.append((label, run(caps, mode)))
    print(f"{'config':>22s} {'PSNR dB':>8s}")
    for name, im in rows:
        print(f"{name:>22s} {psnr(im, gt):8.2f}")
    it_row = ["  max-consensus vs iterations:"]
    for k in (0, 4, 16, 40):
        it_row.append(f"it{k}={psnr(run(caps, 'max', iters=k), gt):.2f}")
    print(" ".join(it_row) + " dB")
    return gt, caps, rows


def part_a2():
    """Strong complementarity: orthogonal motion blurs.

    A horizontal box blur destroys high horizontal frequencies but keeps
    every vertical frequency untouched, and vice versa.  Together the two
    captures cover the whole frequency plane except the high-fx AND high-fy
    corner.  If projection value = spectral complementarity, fusion must
    decisively beat BOTH captures here (unlike the disk case)."""
    print("\n=== A2. orthogonal motion blurs (11-px h-box vs v-box) ===")
    gt = ground_truth()
    rng = np.random.default_rng(9)
    kh = np.ones((1, 11)) / 11.0
    kv = np.ones((11, 1)) / 11.0

    def conv2(img, k):
        ph, pw = k.shape[0] // 2, k.shape[1] // 2
        pad = np.pad(img, ((ph, ph), (pw, pw)), mode="reflect")
        kk = np.zeros_like(pad)
        kk[:k.shape[0], :k.shape[1]] = k
        kk = np.roll(kk, (-ph, -pw), axis=(0, 1))
        out = np.fft.irfft2(np.fft.rfft2(pad) * np.fft.rfft2(kk),
                            s=pad.shape)
        return out[ph:ph + img.shape[0], pw:pw + img.shape[1]]

    caps = [conv2(gt, kh) + 0.002 * rng.standard_normal(gt.shape),
            conv2(gt, kv) + 0.002 * rng.standard_normal(gt.shape)]
    rows = [("capture h-blur", caps[0]), ("capture v-blur", caps[1]),
            ("pixel average", sum(caps) / 2)]
    for mode, label in (("patch", "proj per-patch w"),
                        ("coeff", "proj per-coeff w"),
                        ("max", "proj max-consensus")):
        rows.append((label, run(caps, mode, iters=40)))
    print(f"{'config':>22s} {'PSNR dB':>8s}")
    for name, im in rows:
        print(f"{name:>22s} {psnr(im, gt):8.2f}")
    it_row = ["  max-consensus vs iterations:"]
    for k in (0, 4, 16, 40):
        it_row.append(f"it{k}={psnr(run(caps, 'max', iters=k), gt):.2f}")
    print(" ".join(it_row) + " dB")
    return rows


def part_b():
    print("\n=== B. negative control: identical blurs r=4 (no joint info) ===")
    gt = ground_truth()
    rng = np.random.default_rng(9)
    caps = [convolve_reflect(gt, disk_kernel(4.0)) + 0.002 *
            rng.standard_normal(gt.shape) for _ in range(2)]
    print(f"  capture PSNR {psnr(caps[0], gt):.2f} dB | "
          f"max-consensus fused {psnr(run(caps, 'max'), gt):.2f} dB "
          f"(must not exceed capture by much)")


def part_c():
    print("\n=== C. soft-weight seam: spatially varying focus, wide transition ===")
    gt = ground_truth()
    n = gt.shape[1]
    x = np.arange(n)
    blend = 1.0 / (1.0 + np.exp(-(x - n / 2) / 20.0))    # 0=left, 1=right
    rng = np.random.default_rng(9)
    blurred = convolve_reflect(gt, disk_kernel(4.0))
    cap_a = gt * (1 - blend)[None, :] + blurred * blend[None, :]   # sharp left
    cap_b = gt * blend[None, :] + blurred * (1 - blend)[None, :]   # sharp right
    cap_a = cap_a + 0.002 * rng.standard_normal(gt.shape)
    cap_b = cap_b + 0.002 * rng.standard_normal(gt.shape)
    caps = [cap_a, cap_b]
    fams = [PatchFamily(c) for c in caps]
    for gamma in (1.0, 2.0, 4.0):
        wp = patch_weights(fams, gamma)
        maps = []
        for i in (0, 1):
            acc = np.zeros_like(gt)
            np.add.at(acc, (fams[i].iy, fams[i].ix),
                      wp[i][:, None, None] * fams[i].win[None] ** 2)
            maps.append(acc / np.maximum(fams[i].den, 1e-8))
        seed = maps[0] * cap_a + maps[1] * cap_b
        fused = run(caps, "patch", gamma=gamma, seed_img=seed)
        col_err_seed = np.sqrt(np.mean((seed - gt) ** 2, axis=0))
        col_err_fused = np.sqrt(np.mean((fused - gt) ** 2, axis=0))
        seam = slice(n // 2 - 24, n // 2 + 24)
        away = np.r_[32:n // 2 - 40, n // 2 + 40:n - 32]
        print(f"  gamma={gamma:.0f}: PSNR seed={psnr(seed, gt):6.2f} "
              f"fused={psnr(fused, gt):6.2f} | seam RMSE "
              f"seed={col_err_seed[seam].mean():.4f} "
              f"fused={col_err_fused[seam].mean():.4f} | away "
              f"seed={col_err_seed[away].mean():.4f} "
              f"fused={col_err_fused[away].mean():.4f}")


def save_panel(gt, caps, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True, exist_ok=True)
    panels = [("ground truth", gt)] + rows
    fig, axes = plt.subplots(2, 4, figsize=(14.5, 7.6))
    y0, y1, x0, x1 = 96, 208, 64, 176            # zoomed crop
    for ax, (name, im) in zip(axes.ravel(), panels[:8]):
        ax.imshow(im[y0:y1, x0:x1], cmap="gray",
                  vmin=gt.min(), vmax=gt.max())
        label = name if name == "ground truth" else \
            f"{name} ({psnr(im, gt):.1f} dB)"
        ax.set_title(label, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT / "two_lattice_blur_study.png", dpi=130)
    print(OUT / "two_lattice_blur_study.png")


if __name__ == "__main__":
    gt, caps, rows = part_a()
    rows_a2 = part_a2()
    part_b()
    part_c()
    save_panel(gt, caps, rows_a2)
