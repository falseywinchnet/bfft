# For Sol: cheap viewer improvements from the ladder work (2026-07-11)

Context: the aperture-ladder project (notes/two_lattice_superresolution.md
S5) shipped `iqw_dip_unified_ladder` + the "slow rung" viewer combo
(off/2x/4x).  Measured facts you can rely on: the upward rung costs
+0.5 ms/tile at NB=1024 and +0.06 ms at NB=4096 (effectively free); a
packet pair at dt=1200 with a 70-degree relative-phase error is
unrecoverable by {NB, NB/4} and recovered to 0.0 degrees by the 2x rung;
C++ == Python spec to 2e-12 (`test_two_lattice.py`, ladder test included).

Items below are ordered by value/effort.  1-3 are small; 4 is the one real
feature and the reason the rung currently ships opt-in.

## 1. Momentum guard parity in the C++ unified loops (tiny)

`two_lattice.recover_two_lattice` / `recover_ladder` guard the momentum
trial (`not isfinite` or `max|trial| > 50*rms` -> fall back to latent).
`iqw_dip_unified` and `iqw_dip_unified_ladder` do NOT.  Inert at
unified_steps=1 (trial == latent), but any user raising steps loses the
spec's divergence protection.  Add the same two checks inside both C++
step loops so the parity tests stay honest at steps > 1.

## 2. Surface the rung state in the UI text (tiny)

`sr_geometry` and `recon_status` don't mention the active rung.  Append
e.g. "| rung 2048" when `app.sr_rung > 0` (rung length =
`min(sr_rung * sr_nb, TwoLatticeStream.tile)`), so A/B comparisons are
legible in screenshots.

## 3. Warm-seed the unified path across tiles (small, measured motivation)

The seed is load-bearing (notes S2: one iteration polishes only near the
seed).  `dip_run_complex_warm` already chains PGHI phase across tiles but
the unified path always cold-seeds.  Plumb `n_warm/warm_internal` through
`iqw_dip_unified(_ladder)` the same way.  Expected win: fewer seed
artifacts at tile boundaries at zero added FFT cost; the OLA striping
history (README) suggests boundary seeds are where PGHI hurts.

## 4. Ladder-aware readout (the real feature)

Measured structural fact: the NB-scale reassignment readout CANNOT display
what only >NB windows see — its frames never contain both members of a
slow pair, so the rung's recovered structure lives in the latent only.
Until the readout consults the rung, the "slow rung" combo improves the
waveform but not the picture, which is why it defaults off.

Minimal correct design (display-only, no solver changes):
- Render the recovered latent through `Reassign(n)` for each rung n in the
  active ladder (bilinear splat on), on the same display grid.  Cost: one
  extra reassignment pass per rung over the visible span — the 2x/4x rungs
  have few frames.
- Merge per display cell by MAX POWER across rungs.  Reassignment is
  sparse/concentrated, so max is conservative (no energy invention) and
  lets each scale claim the cells it localizes best: slow beats appear
  from the long rung, transients from the short one.  (Per-cell scale
  ownership — picking the rung whose window matches the local structure —
  is the principled upgrade; max is the honest v1.)
- Label the mode "Super-resolution (ladder readout)" so the two-aperture
  product stays inspectable unchanged.
- Validation: extend `validate_modes.py` with a slow-pair fixture (two
  packets, dt ~ 1.2*NB, correlated phase): the beat interference band is
  visible in the ladder readout iff the rung is on.  Without the rung the
  image must be UNCHANGED (readout of the same latent at the same scales).

## Non-items (already done, don't redo)

- Display transfer function (Amplitude auto / percentile) and
  peak-preserving resampling: shipped 2026-07-10.
- Ladder ABI, wrapper, spec, parity tests, viewer combo: shipped.
- DR/RAAR splitting upgrades: measured null (notes S4e) — the residual is
  information-limited, not algorithmic.  Do not spend time there.
- Absolute reliability gate in the solver: conceptually right, measured
  invisible at current noise; parked.
