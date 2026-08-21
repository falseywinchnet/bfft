# Blind Fourier-circle flow-atlas estimation

## Complementary connections, not model selection

The dense estimator provides a spatial connection `f_dense(p)`. It is strong
on translation, affine shear, rotation, and smooth local deformation, but a
robust local basin can agree with itself while missing a second appearance.
Woven chirps exposed exactly that failure: brightness and derivative
estimators disagreed by only 0.013 pixels even though both flows were wrong.

Fourier circles retain phase transport discarded by that local basin. Seven
smooth annuli each produce a phase-correlation displacement with positive
weight

```text
joint spectral energy * (1 - separated-peak ambiguity)^2.
```

Their barycenter gives a global connection `f_G`; no annulus or peak wins. A
second construction applies the same circle law on overlapping reflected
windows. For chart `k`, residual improvement and within-chart circle coherence
induce a positive Gaussian spatial mass `w_k(p)`. Transporting all chart
vectors gives

```text
f_L(p) = sum_k w_k(p) v_k / sum_k w_k(p).
```

Static charts remain zero-vector anchors. Their atlas variance is recorded,
not collapsed into a camera label. This local field is nonzero on the curved
rotation control even when `f_G = [0, 0]`.

## One positive tensor measure

Global and local paths are coordinates of one transport atlas:

```text
F(p, lambda, mu) = lambda [(1 - mu) f_G + mu f_L(p)],
0 <= lambda <= 1,  0 <= mu <= 1.
```

Scale uses Lobatto support `{0, 1/2, 1}` and atlas coordinate uses its two
positive endpoint cells. The shared zero path is deduplicated, leaving five
operator plans. These are quadrature points, not five scene layers. Each point
carries a distinct exchangeable appearance through `multisheet_transport.py`.
Cross-predictive scale mass is equal across the three samples; the rejected
cell-width prior over-weighted half-motion and weakened the larger woven case.

The atlas prior is continuous. Eighth-power global/local motion mass and
Fourier-circle coherence set a soft center on `mu`; a positive Gaussian
density plus a nonzero floor supplies both atlas endpoints. Forward and reverse
cross-prediction evaluate every support flow, convert residual gaps into a soft
simplex measure, and pull both sensor measures into one symmetric reference
gauge. There is no argmax, winning chart, foreground label, or selected layer
count.

## Continuous authority

The dense common gauge remains authoritative unless independent transport
evidence requires the appearance atlas. Let `r_G` and `r_L` be dense
disagreement with the global and local circle connections, each normalized by
its own motion scale, and let `r = sqrt(r_G r_L)`. The global authority retains
the successful translational law

```text
a_G = gate_12(r_G; 0.12) * exp(-(q_G / 0.22)^4).
```

Local atlas authority is a product of continuous evidence: a twentieth-power
gate on `r`, local/global motion-mass locality, sparse chart observability,
per-pixel positive observability, and within-chart spectral coherence. Sparse
observability matters because a pervasive locally coherent field is evidence
for one smooth connection, whereas partial local support is the signature of
an independently moving appearance. Finally,

```text
a_circle = 1 - (1 - a_G)(1 - a_L)
a = 1 - (1 - a_fold)(1 - a_circle)
x = (1 - a) x_dense + a x_atlas.
```

All gates are smooth positive transports. They regulate constraint authority;
they do not choose a reconstruction family. Fold/non-closure pressure remains
the complementary Eikonal authority.

## Falsifications retained

- Cycle-derived frame rejection reduced the moderate control by 0.96 dB.
- Flow particles carrying one shared appearance reduced both disocclusion
  controls.
- Matched-adjoint blind ownership lost 1.376 dB on average to dense flow.
- Raw motion-coordinate ownership lost 1.334 dB.
- Brightness-versus-derivative cross-prediction was smallest on woven chirps;
  both local families occupied the same wrong basin.
- The selected global-only circle fiber failed curved motion: mean/worst
  changes from dense were -0.398/-1.959 dB and only 4/12 trials improved.
- Directly averaging global and local vector fields regressed both straight
  layered motion and smooth deformation. Their separate positive atlas charts
  are therefore representation, not optional decoration.
- The unconditional selected atlas loses 3.709 dB on average and 10.985 dB in
  the worst smooth-deformation trial. Authority is essential.

The rejected ledgers remain checked in beside the selected artifacts.

## Straight layered-motion checkpoint

| Moving-layer case | Dense flow | Unified flow atlas | SSIM |
|---|---:|---:|---:|
| Moderate disocclusion | 28.994 dB | 36.250 dB | 0.9793 |
| Larger disocclusion | 25.617 dB | 33.323 dB | 0.9772 |

All 12 trials improve. Mean/minimum gains over dense are 7.481/4.254 dB;
mean/minimum gains over averaging are 11.336/3.869 dB. Both exceed the former
global-only checkpoint. With radiometric transport active, the M4 battery takes
5.67 seconds. Selected artifact SHA-256:
`65d486328f4e363f99819658f7f3dfedaf4304fec1b721e64321e8ff7553bb80`.

## Curved layered-motion checkpoint

The control rotates one independently moving occluding appearance about the
image center over a static background, so displacement direction varies over
the raster.

| Curved case | Dense flow | Unified flow atlas | SSIM |
|---|---:|---:|---:|
| Moderate curvature | 30.645 dB | 32.137 dB | 0.9772 |
| Larger curvature | 28.595 dB | 28.967 dB | 0.9394 |

All 12 trials improve over dense. Mean/minimum gains are 0.932/0.050 dB. The
M4 battery takes 5.83 seconds. Selected artifact SHA-256:
`771a7112be4e7b35511c439cd785197370df29dffa8fe1de52af43e907b90d46`.
The rejected global-only ledger SHA-256 is
`902a3b4b4de22015bd7e69bde7b1b6dbeb77776746871ff65317763488c8eb22`.

## Smooth-deformation preservation checkpoint

On translation plus affine shear, smooth local deformation, and exposure
mixing, the raw atlas loses 3.710 dB on average. The unified output retains
33.462 dB / 0.9702 SSIM with mean/worst changes from dense of only
-0.000022/-0.000064 dB. Selected artifact SHA-256:
`803619d332b463681c49edbd67e8bc8adc6866f8087c24db9a35f56f37175f7a`.

The auxiliary appearance descent remains one warm exact sweep. ABI v5 batches
all exchangeable positive operators without moving estimation or authority
out of auditable Python. The Dear PyGui Pair A/B action runs this unified atlas
path. `RADIOMETRIC_TRANSPORT.md` records the promoted exposure/clipping and
accelerated rolling-shutter checkpoints. Broad real-capture evidence, shared
rolling-shutter gauge calibration, lens aberration, and turbulence remain open
gates.
