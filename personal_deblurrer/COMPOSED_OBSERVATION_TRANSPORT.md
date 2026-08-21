# One observation transport, including double radial blur

## The unknown-image contract

An unknown raster is not first classified into an outer operator and an inner
operator.  It supplies evidence for one positive conditional measure

```text
y(p) = integral x(q) K(p, dq),
K(p, image domain) = 1.
```

A deterministic warp is the one-atom limit of `K`. A spatially constant cloud
is convolution. A varying cloud contains lens distortion, camera exposure,
radial mixing, and sensor footprints without changing reconstruction law.
These descriptions are generator controls, not inference labels.

This is stricter than merely running several inverse stages. If two transports
occurred, they consolidate by the Chapman--Kolmogorov law

```text
K_21(p, dq) = integral K_1(r, dq) K_2(p, dr).
```

`composed_transport.PositiveObservationTransport` applies that law directly to
the exact reflected/bilinear row measures. Therefore

```text
T_21 x = T_2(T_1 x)
```

closes to floating precision, including the interpolation footprints created
by both observations. Its adjoint is the exact transpose scatter. The inverse
receives only `T_21`; it does not receive the factorization.

## Why radial composition is unusually clean

An atom of radial scale transport about center `c` has the affine source map

```text
q = c + s(p-c) = s p + (1-s)c.
```

Two atoms with scales `s1` and `s2` compose to a single atom with scale
`s1 s2`. In log scale,

```text
u = log s,
u21 = u1 + u2.
```

Thus a twice-applied radial exposure is not a new blur kind. Its positive
log-scale measure is the convolution of the two constituent measures. The
same statement extends to affine-map measures by matrix multiplication; the
radial case is commutative because every matrix is a scalar multiple of the
identity.

The local displacement covariance of the consolidated row measure has its
principal direction along `p-c`. `local_moment_jet()` obtains that direction
from the measure itself and abstains where its variance vanishes. It does not
compare radial, line, Gaussian, or warp templates.

## What the present gate proves—and does not prove

The synthetic double-radial gate answers a representation question: if the
correct positive observation measure is available, can two sequential radial
transports be expressed and inverted as one? The answer is yes. The 64x64
unit test moves its structural fixture from 31.29 dB to 44.74 dB, while the
full composed forward residual is `2.41e-4` and the input hash is unchanged.

The six-source M4 Mini battery at 96x96 and 64 positive-line passes gives:

| Consolidated measure | Observation | One-transport inverse | Ray alignment | Composition closure |
| --- | ---: | ---: | ---: | ---: |
| radial 0.05 | 30.816 dB / 0.9131 | 53.919 dB / 0.9872 | 0.99968 | exact zero |
| radial 0.05 then 0.05 | 28.041 dB / 0.8455 | 34.931 dB / 0.9346 | 0.99968 | exact zero |
| radial 0.04 then 0.08 | 27.019 dB / 0.8157 | 39.651 dB / 0.9556 | 0.99994 | exact zero |

The unequal double transport is easier to invert than the equal double despite
having the same number of generating stages. Conditioning belongs to the
consolidated characteristic measure, not to a stage count.

The affine gather/scatter is matrix-free. At 96x96, the single and equal-double
operators retain only 168 bytes of operator state; the unequal double retains
336 bytes. The same state sizes hold at 1024x1024. Per-call coordinate scratch
scales with image area, but the composed row table is never materialized. This
avoids the gigabyte-scale plan that a naive Cartesian product of two bilinear
row tables would create.

This is not yet blind recovery. A single observed image does not uniquely
factor into an arbitrary latent image and an arbitrary row-stochastic `K`.
The intended blind continuation is therefore not a larger catalogue. It is a
direct estimate of the local characteristic action of `K` in the Eikonal and
Fourier-circle basis, coupled to one latent reconstruction. In the radial
case, the sought field should expose convergent local characteristic rays and
an additive log-scale action. The image should accumulate authority in that
direction because it closes the observation transport, not because a radial
candidate won a contest.

For a local row measure `K_p`, define

```text
phi_p(f) = integral exp(-2 pi i f dot (p-q)) K(p,dq),
A_p(f) = -log |phi_p(f)|^2.
```

Composition adds characteristic action in the transported chart. The blind
unknown is therefore the smooth positive action field `A_p`, with its
direction supplied by `grad_f A_p`; “radial,” “line,” and “resampling” are not
state variables. Scene power is a nuisance term and must be removed by
crossfit/circle closure with an explicit abstention where one image supplies no
separating evidence. This is the next estimator gate.

Container metadata, when present, can still constrain a known forward audit.
It must not create a mandatory “outer pullback” branch for an unknown image.
Quantization and compression remain uncertainty/support loss in the same
observation contract until their transport is independently known.

## Executable controls

- `test_composed_transport.py` checks exact composition, exact adjointness,
  log-scale addition, intrinsic radial direction, immutability, identity, and
  double-radial recovery.
- `run_composed_transport_benchmark.py` measures single, equal-double, and
  unequal-double radial exposure on the six denoiser scaffold sources. Its
  ledger explicitly labels the result as a known-measure representation and
  inverse gate, not blind estimation.
- The Dear PyGui workbench exposes `Radial scale exposure` and
  `Double radial exposure` only under **Blur image**, the synthetic generator
  role. **Use as-is for deblurring** supplies no such operator label.
- `OBSERVATION_ANOMALIES.md` extends this closure to displaced centers,
  rotation, shear, ghost mass, astigmatic scale, and interval-censored sensor
  samples.

Wronski's article is a useful fixed-convolution comparison, especially its
discussion of positive spreading, quantization, gamma space, and inverse gain:
[Removing blur from images](https://bartwronski.com/2022/05/26/removing-blur-from-images-deconvolution-and-using-optimized-simple-filters/).
