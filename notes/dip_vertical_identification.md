# Vertical identification: the superresolution problem inside the DIP walk

Status 2026-07-06 (late): theorems validated to machine precision, censuses
measured, closed-block engine locally contracting.  Executable record:
`experiments/dip_vertical_identification.py` (scipy-free, 4.5 s).
This note supersedes the *framing* of `notes/aperture_ladder_design.md` (whose
FAST solver remains the reference phenomenon but lives outside the walk) and
retracts the load-bearing claim of `notes/gauge_dof_bounds.md` (see §0).

## 0. Corrections owned

1. The aperture-ladder solver used round-trip record FFTs + scipy L-BFGS.  It
   discovers the phenomenon; it is not the system.  The system must live in
   the walk: state = paused level states, updates = walk stages + banded row
   ops + radial projections + adjoints.  This note builds exactly that.
2. The gauge/vortex DOF analysis identified defects HORIZONTALLY (along the
   lattice of one level).  The walk's own algebra says phase information is
   carried VERTICALLY — between levels — and the horizontal census, however
   internally correct, does not bound the problem the vertical solver works
   on.  Retracted as a solver foundation.
3. A tempting vertical claim is FALSE and was caught by validation:
   contiguous-block spectra do NOT decouple into record-bin residue classes
   (rect crops leak by Dirichlet tails; the comb and block factorizations were
   conflated).  Exact attachment is comb-side and per-frame (T3' below).

## 1. The theorems (all machine-precision, real data)

**T1 (polarization ladder).**  One DIF butterfly p = a + wb, q = a − wb obeys
|p|² − |q|² = 4 Re(ā·wb).  So level-(t+1) magnitudes measure the RELATIVE
PHASES of level-t pairs: the magnitude profile down the tower is a complete
hierarchy of relative-phase measurements at all dyadic lags, and the hardness
of phase retrieval is exactly the LEVEL GAPS between magnitude-observed rungs.
Per butterfly one sign bit (of sin) stays discrete.  Census on the record:
35% of butterfly energy has |cos| > 0.9 (sign irrelevant), 18% has |cos| < 0.3
(sign-critical) — the honest discrete content, vertical version.

**T2 (windows are taps, never divisions).**  Periodic Hann is an exact twisted
3-tap row operator at EVERY level:
Zak(h·u)[d,j] = 0.5 Z[d,j] − 0.25 e^{2πij/n} Z[d−1,j] − 0.25 e^{−2πij/n} Z[d+1,j],
a per-column circulant with symbol 0.5 − 0.5cos(2πm/e − θ_j) — exactly one
zero (the window's edge null).  Validated 1e-16 at levels 0/3/5/8.  np.hanning
(symmetric) is detuned by 1/(n−1) bin and not finitely banded: production must
use periodic Hann (identical resolution).

**T3' (apertures attach in the state).**  With hop = q* = 32, EVERY short
frame inside a long frame sits at a q*-aligned offset, and
Zak_s(short) = M_δ · Zak_b(long L5) per column, M_δ = F₄·crop_δ·F₃₂⁻¹ (a 4×32
Dirichlet band); Hann via T2 in the short tower.  Validated 3e-16 for all 29
offsets.  Consequence: the ENTIRE multi-aperture observation set attaches to
one paused state per long frame — no time-domain round trip exists in the
formulation, only walk stages and banded row ops.

**T5 (per-row Pauli bank).**  The suffix from L5 restricted to one row d is a
unitary twisted 32-DFT W_d onto the endpoint comb {k ≡ d mod 32} (nesting).
Per frame the vertical problem is a bank of 32 dimension-32 PAULI PROBLEMS
(recover a row from |row| and |W_d row|), coupled across rows only by the
short banded crops and across frames by the comb overlap bus.
Census: the Pauli pair alone pins **32.8%** of row energy (171/225 rows
ambiguous) — this NUMBER is what the short band and the bus must supply, and
it is the vertical replacement for the horizontal DOF census.

## 2. The optimization target (declared)

Unknown, per long frame: the rect-frame level-5 state Z ∈ C^{32×32} with the
real-frame row symmetry Z[−d] = conj(Z[d]) (≈ 1024 real DOF/frame).
Observations, all attached in-state:
    y_long        = |suffix(tap3_b(Z))|            (513 informative mags)
    y_short[δ]    = |suffix_s(tap3_s(M_δ Z))|      (29 × 65 mags)
plus the cross-frame comb bus (exact overlap identity, hop | q).
~2.3 : 1 over-determined, all operators = walk stages + banded row taps.

## 3. The engine (closed blocks; measured)

Sweep = sum over rungs of ADJOINT RESIDUAL PULLBACKS: radial residual at the
rung, pulled back through pure adjoints (suffixᴴ = scaled inverse stages,
tap3ᴴ, M_δᴴ), scale = operator norms (nothing tuned), then the Hermitian row
projection.  This is the per-frame, in-state form of the paused-L5 "teacher
law" in `zak_dip_l5_noscipy_lbfgs.py` — with the short apertures coupled
through M_δ in the state instead of through OLA synthesis.

Measured:
- **Local contraction** around truth: rel err 0.100 → 0.064 in 60 sweeps,
  strictly monotone, every frame tested.  The vertical constraint set is
  locally attracting and the closed-block sweep is a valid local solver.
- **Rejected with reason**: Tikhonov-inverse pullbacks (invert tap3/suffix
  then average).  They are BIASED at truth — λ/(λ²+reg) ≠ 1/λ at moderate λ —
  so the iteration's fixed point is not the solution (measured divergence /
  0.65 stall).  Pullbacks must be adjoints, exactly as the teacher law does.
- **Cold start does not reach the basin** (rel err ~1.0).  Seeding is the
  open problem — consistent with every other measurement today (the ladder's
  PGHI seed, codex's seed, the L5 file's seed all carry the basin).

## 4. Where the seed should come from (the research direction)

T1 says the missing information between rungs is relative phases recoverable
from magnitude LADDERS, with sparse sign bits.  The short observations are a
band-limited proxy for a mid-rung magnitude field (T3': |suffix_s(tap3_s(M_δ
Z))| is a banded image of Z).  The seeding problem is therefore a bank of
small gap-sync problems: per row-band, recover the level-5 phases from the
endpoint comb magnitudes (32/row) plus the short band magnitudes (that row's
share of 29×65), using the polarization structure — dimension ≤ 32×3 per
band, over-determined ~2:1, with L-BFGS-free small-block iterations (power/
mini-GL per band) and the per-butterfly sign bits as the only discrete
choices (18% of energy).  This replaces PGHI (whose frequency-law is the weak
link on polyphony) with a law-free, in-walk seed.  Not yet built — it is the
next concrete step, and the only missing piece between the theorems and a
complete in-walk system.

## 5. Relation to prior artifacts

- `zak_dip_l5_noscipy_lbfgs.py` (user/Fable): the teacher law = our sweep's
  adjoint pullback with the record OLA as the coupling bus and L-BFGS memory
  on top.  Compatible: curvature memory can be added to our per-frame sweeps
  unchanged (the geometry is fixed); the in-state short attachment (T3')
  removes its per-iteration synthesis.
- `aperture_ladder.py` FAST (0.9820 @ 162 ms): remains the reference
  phenomenon and the quality target for the in-walk system.
- `gauge_dof_bounds.md`: horizontal analysis; retracted as solver foundation
  (kept for the record-consensus closed form and the rigidity certificate
  methodology, which transfer).

## 6. Next steps (ranked)

1. Build the §4 band-sync seed (small closed blocks, in-walk); measure basin
   reach; combined with §3 sweeps this is the complete system candidate.
2. Add curvature memory (their ComplexLBFGS, scipy-free) to the §3 sweeps —
   the teacher's measured accelerator — and the cross-frame comb bus.
3. Wide aperture: T3' generalizes (1024 inside 4096 at q*-aligned offsets);
   attach and measure the 3-aperture in-walk system.
4. Sign-bit search: 18% of butterfly energy is sign-critical; enumerate per
   band (the discrete set is small and local) instead of global restarts.
5. Benchmark protocol parity: periodic-vs-symmetric Hann (T2) — regenerate
   reference targets with periodic Hann so the in-walk system's numbers are
   apples-to-apples with the ladder FAST reference.
