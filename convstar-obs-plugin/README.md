# DIP-CONV Gaussian Comb Transport for OBS

Version 0.5 is a single serious same-shape video operator. The illustrative
"support toy" and quantum-metaphor presets from 0.3/0.4 are no longer exposed.
The filter now uses the repository's real SIMD DIP Fourier walk, an exact
Gaussian translation characteristic, frequency-conditioned inverse transport,
and the proven local CONV* factor-order coordinate. The shader only samples the
native frame through the C++-computed transport field and applies the resulting
OKLCH action; it does not invent the effect.

The displayed OBS filter name is **DIP-CONV Gaussian Comb Transport**. Its
internal source id remains `convstar_support_toys` so existing OBS scenes update
in place.

## 1. Finite DIP representation

For a one-dimensional line of length `N=eq`, a DIP level is the finite Zak
state

```text
Z[d,j] = sum_(r=0)^(e-1) x[j+r q] exp(-2 pi i d r/e).
```

Each `d` is therefore a frequency comb `k = d (mod e)`, not a metaphorical
packet. The C++ implementation applies the repository's depth-first
`DIP_RFFT_kernel` on every row, then constructs each complex column transform
from two real DIP transforms. The inverse performs the exact reverse
construction. On Apple Silicon the active walk is the explicit 128-bit NEON
backend; the same kernel selects SSE2 or AVX2/FMA on matching hosts.

The OBS staging and transport rasters are the rectangular power-of-two DIP
lattice itself. At the 720-long-side default this is 512x256. Higher settings
can select up to 1024x512; the native video is always rendered at its original
dimensions. This avoids a redundant analysis-to-DIP resample and five-field
DIP-to-analysis expansion; the final native shader performs the sole required
bilinear continuation.

## 2. Harmonic-comb decomposition

The two-dimensional spectrum is partitioned into folded residue classes with
comb denominator `e=8`:

```text
C_ab = {(kx,ky): fold(kx mod e)=a, fold(ky mod e)=b}, 0<=a,b<=e/2.
```

This gives 25 real-compatible two-dimensional combs. Within each comb, adjacent
teeth are separated by exactly `e`. Their normalized complex correlations are

```text
R_x = sum X[kx,ky] conj(X[kx-e,ky]),
gamma_x = |R_x| / sqrt(sum|X|^2 sum|X_previous|^2),
```

and analogously for `y`. By the Fourier shift theorem, the correlation phase
is a translation estimate:

```text
mu_x = -arg(R_x) Nx/(2 pi e),
mu_y = -arg(R_y) Ny/(2 pi e).
```

For adjacent teeth, a Gaussian translation gives
`gamma=exp[-(2 pi e sigma/N)^2/2]`; therefore the displacement standard
deviation is recovered by moment inversion,
`sigma=N sqrt(-2 log gamma)/(2 pi e)`. Only a finite-lattice ceiling and a
quarter-pixel numerical floor are imposed. The operator exchanges the two means
and covariance axes in normalized comb coordinates before returning to pixel
units. No object, edge, face, or texture class is selected.

## 3. Exact Gaussian superposition

Let the transposed displacement of a comb be
`V ~ N(mu,Sigma)`, with the measured diagonal covariance. Translation by a
deterministic `v` acts on a Fourier coefficient by the unitary character
`exp(-i omega.v)`. A coherent superposition over uncertain translations is
therefore not approximated by samples; its expectation is the Gaussian
characteristic function

```text
G(omega) = E exp(-i omega.V)
         = exp(-i omega.mu - 1/2 omega^T Sigma omega).
```

This identity is both the uncertainty law and the fast implementation. Because
`Sigma` is diagonal, `G=G_x G_y`. Phase and exponential factors are evaluated
once per comb and axis (about 13,000 evaluations at 512x256), then multiplied
to all teeth. The former five-state cubature required roughly 655,000 complex
phase evaluations per frame and was removed.

The user strength `s` increases transport with frequency independently on both
axes:

```text
s_j(rho_j) = s [0.08 + 1.42 rho_j^1.35],
omega_j    = 2 pi s_j k_j/N_j.
```

Thus high-frequency combs are genuinely more trans-shifted, while DC is fixed.

## 4. Frequency-conditioned partial inverse

After the forward Gaussian transport, low frequencies are partially reversed
in transport space:

```text
r(rho) = r0 (1-rho)^1.7,
G_dagger = conj(G) / (|G|^2 + lambda),
X_out = [(1-r)I + r G_dagger] G X.
```

The regularizer is

```text
lambda = 1e-8 + 0.08(1-gamma)(1-|G|)^2(0.2+0.8 rho^2).
```

It vanishes quadratically in the well-conditioned unitary limit. This detail is
essential: a fixed Tikhonov floor created a new low-frequency amplitude residual
even while cancelling phase. The invariant test caught that defect. Where the
Gaussian has erased phase information, the uncertainty-dependent term prevents
unstable inversion. Consequently low combs return close to their input while
uncertain high combs remain visibly transposed.

## 5. CONV* transport admission

The reconstructed target and uncertainty spectra are inverse-transformed. A
regularized brightness-constancy solve converts their residual into a local
transport vector. On the same DIP lattice, source first jets provide the exact
CONV* nodal factor-order coordinate

```text
eta = Dy^2/(Dx^2+Dy^2),
```

with `eta=1/2` at zero current. The field expanded to the OBS analysis raster is
the cardinal bilinear continuation `I_h eta`. This is the local compilation
proved in `../output/pdf/convstar_insert.tex`: it is the unique continuous
cellwise bilinear extension of the formal admission trace, rather than an
extra content rule.

At each site, `eta` convexly admits the two Cartesian readings of the
phase-comb-transposed transport. Hue is the admitted transport phase, chroma is
the log-amplitude loss induced by Gaussian uncertainty, and lightness is the
transport residual. All are formed in C++ and applied in OKLCH coordinates.

## 6. Invariants and measured path

`dip-comb-transport-test` checks:

- DIP forward/inverse identity (about `1.5e-16` RMS at 128x128);
- greater response at high harmonic frequency;
- actual reduction of the low-frequency residual as reversal increases;
- horizontal/vertical CONV* admission near 0/1;
- finite bounded transport, uncertainty, residual, and circular phase fields.

The final standalone Release benchmark on the nearby M4 Mini measured 3.32 ms
for the direct 512x256 path and 6.99 ms for direct 1024x512. The 512x256 split
was 0.23 ms forward DIP, 1.58 ms comb transport, 0.50 ms for both inverse DIPs,
and 0.97 ms for field construction. The high-detail split was 0.54, 3.21, 1.01,
and 2.14 ms respectively. The identity/frequency/admission battery runs in the
same executable. AddressSanitizer and UndefinedBehaviorSanitizer also complete
the 1024x512 path without a report.

On the local Apple A18 Pro, OBS 32.2.1 and OpenGL, the complete filter is much
more variable because GPU staging, OKLCH conversion, texture upload, and the
host's thermal scheduling surround the operator. A repeated 1280x720 smoke run
reported 30.4 ms/filter frame; other runs under load were slower. The selected
DIP backend was `neon-128`. These are measurements, not cross-machine promises.

## 7. Controls

- **Transport amount**: dry/wet scale for geometry and OKLCH action.
- **Frequency trans-shift**: strength `s` in the frequency law, 0 to 3.
- **DIP low-frequency reversal**: `r0` in the conditioned inverse, 0 to 1.
- **OKLCH phase/chroma gain**: scales phase-derived hue and uncertainty chroma.
- **OKLCH residual-lightness gain**: scales the transported lightness residual.
- **Analysis long side**: 360, 480, 540, 720, 900, 1080, 1440, or native.
- **CPU threads**: 1 to 8.

## 8. Build, test, and install

```sh
cmake -S convstar-obs-plugin -B build-convstar-obs \
  -DOBS_SOURCE_DIR=/private/tmp/obs-studio-32.2.1-full \
  -DSIMDE_SOURCE_DIR=/private/tmp/simde-0.8.2-full \
  -DOBS_APP=/Applications/OBS.app \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-convstar-obs --parallel
./build-convstar-obs/dip-comb-transport-test
```

Install `build-convstar-obs/convstar-support.plugin` in
`~/Library/Application Support/obs-studio/plugins/`, restart OBS, and add
**DIP-CONV Gaussian Comb Transport** in the source's **Effect Filters** panel.

The smoke executable accepts
`MODULE CAPTURE MODE WIDTH HEIGHT ANALYSIS_LONG_SIDE TRANS_SHIFT PHASE_GAIN LIGHTNESS_GAIN BACKEND REVERSE`.
`MODE` is retained only for compatibility and version 0.5 always selects the
serious operator. Use `opengl` as `BACKEND` on this OBS installation.

## Local derivation sources

- `../notes/dip_phase_packet_design.md`
- `../notes/dip_twisted_walk.md`
- `../notes/dip_zak_fusion.md`
- `../paper/on_bruun_revisited/sections/10_dip.tex`
- `../output/pdf/convstar_insert.tex`
- `../output/pdf/conv_witnessed_transport.tex`
- `../experiments/CONV_PRIOR_WORK_ARCHAEOLOGY.md`
