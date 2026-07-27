# Null, jump, and interface refinement

Date: 2026-07-27

This round separated three visible failures which initially looked like one
support-fusion problem.

## Controls

- Pikachu: `/Users/quentinkuttenkuler/Downloads/25.png`, 475×475.
- Photographic sky control: `skimage.data.rocket()`, 640×427.
- One frozen Meyer split, curvature population, one characteristic pass, one
  measured ridge, and 16 soft-support passes.
- The unchanged objective combines RGB, single-stage cartoon, and
  single-stage texture error.

## 1. Weak-detail local null

Cross-scale gradient agreement is positive structural evidence. The squared
gradient disagreement supplies a local null field. Suppression is multiplied
by a weak-evidence gate, so strong isotropic texture is not rejected merely
because it is fine.

At strength 0.5:

| Image | Control cells | New cells | Control PSNR | New PSNR | Control objective | New objective |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pikachu | 1,194 | 1,030 | 27.862 | 27.700 | 2.259e-3 | 2.339e-3 |
| Rocket | 8,465 | 6,374 | 30.069 | 30.013 | 2.329e-3 | 2.150e-3 |

The rocket's upper third retained 67.5% of its former weak-detail confidence,
while the textured lower third retained 85.3%. This is the intended
spatially selective fusion. The first attempted form multiplied all support
by persistence confidence; it reduced Pikachu to 560 cells and 24.90 dB and
was rejected as indiscriminate.

## 2. Decisive-edge jump action

The unchanged target supplies a rank-one discontinuity tensor. Its confidence
uses a fixed OKLab scale and fourth-order onset. The tensor changes transport
action but not population: crossing the normal becomes expensive while travel
along the contour remains available.

At jump action 24:

| Image | Control PSNR | New PSNR | Control objective | New objective |
| --- | ---: | ---: | ---: | ---: |
| Pikachu | 27.862 | 27.916 | 2.259e-3 | 2.231e-3 |
| Rocket | 30.069 | 30.068 | 2.329e-3 | 2.329e-3 |

The earlier scene-relative confidence activated on ordinary photographic
texture and was rejected. Fixed-scale fourth-order onset preserved the
Pikachu improvement while becoming neutral on the photographic control.

## 3. Fractional interface coverage

At a lattice edge where two accepted fronts collide, their actions and the
incident edge cost analytically locate the equality crossing. Only the pixel
footprint cut by that crossing receives fractional colour coverage. No second
owner is propagated or ranked. The resulting readout is accepted only when
the full decomposition objective improves.

At strength 0.4, Pikachu improved to 27.974 dB and the proposal was accepted.
The rocket proposal was rejected and the hard result was retained. Strength
1.0 was rejected on Pikachu; this negative established that collision
coverage is a small rasterization correction, not another soft-support pass.

## Combined default

The viewer defaults are null strength 0.5, jump action 24, and interface
coverage 0.4. On Pikachu the combined result uses 1,030 cells and reaches
27.841 dB before any manual tuning. On rocket it uses 6,374 cells, reaches
30.009 dB, and improves the combined objective from 2.329e-3 to 2.151e-3.

The three diagnostic fields are independently visible in the canonical
viewer. This matters: a lower sky population must not conceal a crossing
error, and interface antialiasing must not conceal a misspecified partition.
