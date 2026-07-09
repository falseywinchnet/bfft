# Analytical bounds on the multi-aperture phase-gauge problem

Purpose: not a solver — a *tight enumeration of the degrees of freedom* of the
problem the solver works on, with every count either proved or measured on the
benchmark (first second of daveandsimon.wav @ 8192 Hz).  This is the formal
continuation of the closed-form line of attack (the IQ-balancer maxim: build
the object whose correction is algebraic) and of the sibling model's reduction
of the short-aperture phase field to row gauges, which visibly lost structure.
Reproduce every number: `.venv/bin/python experiments/gauge_dof_analysis.py`.
Depends on: `notes/aperture_ladder_design.md` (record consensus = OLA, equal-q
coupling, phase laws), `experiments/aperture_ladder.py`.

## 0. The result in one line

    8192 real samples  →  16,445 short-aperture cell phases  →  ~785 continuous
    gauge constants (time-law reduction, PROVEN sufficient to +276 dB)  →  0
    continuous DOF after cross-aperture (Jacobian rank-full, nullity 0)  →
    ~4,320 sparse integer vortex charges (in {−2..+2}, local to zeros).

The solver's actual problem is **not** an 8192-dimensional nonconvex phase
retrieval.  It is a *locally rigid, ~100:1 over-determined, ~785-dimensional
linear gauge solve*, plus a sparse integer topology of a few thousand local
±1 defects.  Only the last is a genuine search.

## 1. Setup and the exact blocks (recap, made precise)

Apertures a give fixed linear analyses A_a x of one latent record x, observed
only in magnitude y_a = |A_a x|.  Two facts are exactly closed-form
(aperture_ladder_design.md, re-stated as the algebra here needs them):

- **Record consensus.** For fixed phases c_a = y_a e^{iφ_a}, the least-squares
  record is x(φ) = C⁻¹ Σ_a A_a* c_a with C = Σ_a A_a*A_a.  For windowed unitary
  STFT, C is the samplewise window-coverage diagonal, so x(φ) is overlap-add /
  coverage division — an *evaluation*, not an optimization.  Eliminating x, the
  only compatibility left is (I − A C⁻¹ A*) c = 0: the true unknowns are phases
  *modulo the range of the analysis frame*, not record samples.

- **Phase is a gauge field.** s = log|V| and φ are Re/Im of the analytic
  (Bargmann) transform, so on the active support Ω the phase is a 0-cochain
  whose coboundary is estimated by the window laws:
      time edges (reliable):    ∂φ/∂τ = 2π w + (1/λ) ∂s/∂w
      freq edges (unreliable):  ∂φ/∂w = −λ ∂s/∂τ,     λ = 0.25645 n² (Hann).

## 2. The Hodge / DOF decomposition (the declaration)

Let K be the cell 2-complex of Ω (nodes = active cells, edges = 4-neighbour
adjacencies present in Ω, faces = unit plaquettes fully in Ω).  The phase field,
given **trusted time edges only**, decomposes as:

- **H⁰(K) = ℝ^{b0}** — one free additive constant per *time-connected
  component*.  A time edge never changes the frequency bin, so a time-component
  is a maximal horizontal run in one row: a **ridge segment**.  b0 = number of
  ridge segments = the continuous gauge DOF.
- **Integer vortex charges** — one integer per plaquette where the wrapped curl
  of the phase gradient is nonzero (a spectrogram zero threaded by the loop).
  These are the *integrability obstruction*: fix them (branch cuts) and the
  field integrates; they are the branch-cut / defect search space.
- Everything else is **determined** by integrating the trusted edges.

The row-gauge reduction (one constant per frequency row) is the special case
b0 → #rows.  It under-counts because rows break at amplitude holes; a per-row
constant forces phase continuity across a hole where the true phase is
independent — this is the visible "missing pieces."

Cross-aperture magnitudes then fix the b0 constants.  Writing the gauges
u_p = e^{iθ_p}, the short field is linear, c_short(u) = Σ_p u_p b_p, so the
record x(u) = Σ_p u_p x_p is linear and every other-aperture coefficient is
(A_a x(u))_l = Σ_p v_{l,p} u_p.  The magnitudes give
      |Σ_p v_{l,p} u_p|² = y_l²   ⇔   trace(G_l U) = y_l²,  U = u u*, diag(U)=1,
an **angular synchronization / rank-1 Hermitian recovery** — closed-form
recoverable by the top eigenvector of a connection Laplacian (the analogue of
the IQ balancer's "construct the statistic that exposes the correction").

## 3. Measured census (long aperture n=1024, h=128; 29,241 cells)

| thr  | \|Ω\|  | E_t   | E_f   | b0_time | b0_full | b1     | vortices | E%   |
|------|--------|-------|-------|---------|---------|--------|----------|------|
| 1e-1 |    905 |   604 |   470 |   **301** |    126 |    295 |     129  | 60.2 |
| 1e-2 | 14,727 |13,177 |12,752 |  **1,550** |     87 | 11,289 |   4,320  | 99.6 |
| 1e-3 | 20,658 |20,139 |20,457 |    519  |      3 | 19,941 |   7,905  |100.0 |

Read `b0_time` as the honest continuous DOF: **hundreds to ~1,500 ridge
segments**, never the ~65 rows a per-row gauge assumes.  `b0_full` (trusting
frequency edges too) collapses to <130 because mainlobes connect vertically —
but that direction is the unreliable law, so it is not a *safe* reduction.  The
b1 explosion at 1e-2 is the vertical connections closing loops around zeros;
each such loop that carries circulation is a vortex.

## 4. Sufficiency and mechanism (short aperture n=128, Fable's; 16,445 cells)

Reconstruct the record from (per-component constant + time integration), oracle
anchors, to isolate the segmentation DOF from law error:

| gradient used  | gauge         | SNR       | corr   |
|----------------|---------------|-----------|--------|
| oracle phase   | —             | +313.5 dB | —      |
| TRUE time-diff | per-row (65)  | +276.5 dB | 0.9975 |
| TRUE time-diff | per-seg (785) | +276.3 dB | 0.9975 |
| LAW  time-diff | per-row (65)  |  −0.8 dB  | 0.2285 |
| LAW  time-diff | per-seg (785) |  −0.9 dB  | 0.2256 |

Two theorems fall out of this table:

1. **Sufficiency.** The b0 constants + time integration reproduce the record to
   +276 dB.  The continuous content of the phase problem is *exactly* the gauge
   constants; capacity is not the limiter (even 65 constants suffice *given* the
   true differences).  So reducing to gauges loses nothing in principle.

2. **The binding constraint is law reliability, not DOF count.** With the *law's*
   gradient the reconstruction collapses to 0.22 regardless of segmentation,
   because the short-aperture time-law is 0.50 rad/edge (vs long 0.19 rad/edge)
   and random-walks past π within a handful of hops.  Segmentation's real role
   is to **re-anchor after every amplitude hole**, bounding law-error
   accumulation to the segment length — but the anchors themselves cannot come
   from the law (per-segment LAW integration is still 0.22).  They must come
   from cross-aperture data.  This is exactly why Fable's row-gauge, seeded and
   integrated from a single aperture's law, "looked right but had pieces
   missing": right subspace, undetermined and law-poisoned coordinates.

## 5. Rigidity certificate (the decisive bound)

At the true gauge (reference phase = truth, θ = 0), the Jacobian of the
cross-aperture constraint map |A_a x(θ)|² w.r.t. the b0 gauge angles:

    gauge unknowns b0 = 872;  cross-aperture constraints = 95,232  (~109 : 1;
    ~33 : 1 counting only active constraints)
    Jacobian rank = 872 / 872,  nullity = 0,  cond ≈ 2.5e5.

**Rank-full, nullity zero.** Once reduced to gauges, the cross-aperture
magnitudes pin *every* constant — including the global phase, because a real
record has no global-phase symmetry (only the discrete x → −x).  The true
solution is locally isolated and massively over-determined, so the reduced
problem is exactly the kind the IQ maxim wants: a rigid algebraic solve, not a
descent.  The honest caveat is the condition number ≈ 2.5e5: rigid everywhere,
but *ill-conditioned in the weakest-signal gauges* — precisely the segments
near spectrogram zeros where the vortices sit.

## 6. The irreducible core (what remains a search)

Inside the active support at 1e-2: **4,320 vortices**, charges
{−2: 276, −1: 2737, +1: 1264, +2: 43} — integer, sparse, local to zeros.  This
is the *entire* non-closed-form content of the problem.  Not the 8192 samples,
not the 16,445 cell phases, not the 785 gauges: a few thousand ±1 branch-cut
choices localized at magnitude minima.

## 7. Equal-q column decomposition (where the search factorizes)

Pausing every aperture at q* = 32 makes cross-scale coupling
Z_small[:,j] = M Z_big[:,j] **32 independent column blocks** (design doc S2.2),
per-column rank ≤ e_small (short e=4, long e=32, wide e=128).  So the gauge
solve and the vortex search are not global: they decompose into 32 column
problems of bounded row-rank before record consensus.  The wide aperture adds
frequency ROWS (domain extension), the short owns fine-time transport — the
same anti-blur structure that makes >2-aperture fusion consistent.

## 8. Consequences for the engine (stated, not built)

The bound says the production update is: (i) segment Ω on the reliable time
graph → b0 components [closed form: connected components]; (ii) integrate the
time-law within each segment → fixed component fields b_p [closed form]; (iii)
solve the b0 gauges by the over-determined angular-synchronization eigenproblem
against long+wide magnitudes [closed form: top eigenvector / weighted LS];
(iv) synthesize the record by OLA [closed form]; (v) update the sparse vortex
charges where residual plaquette circulation is nonzero [the only search, local
integer].  Iterate (v)→(i) a *very small* number of times.  Every continuous
block is exact; iteration count is bounded by topology updates, not by
convergence of a descent.

## 9. Open analytical questions (ranked)

1. **Conditioning of the synchronization operator.** cond ≈ 2.5e5 concentrates
   in weak-signal gauges.  Bound it in terms of segment SNR and inter-aperture
   overlap; it predicts which gauges need a vortex/branch-cut decision vs a
   direct solve.
2. **Vortex charge conservation.** Net charge is nonzero on a bounded window
   (defects exit through the boundary).  A discrete Gauss law (Σ charge =
   boundary phase winding) would turn the 4,320-defect search into a
   constrained, far smaller one — the real prize.
3. **b0 as a function of polyphony and window.** b0 = ridge-segment count scales
   with #components × #onsets; formalize against the signal model to predict the
   gauge dimension a priori (and hence the eigen-solve size).
4. **Which aperture minimizes b0·cond.** The seed aperture should minimize gauge
   count times ill-conditioning, not just have the densest frames; §4/§5 suggest
   a longer aperture (fewer, better-conditioned gauges) may beat the short one
   the ladder currently seeds from.
5. **Exactness of equal-q coupling under Hann.** S7 coupling is stated for rect
   frames; fold the window into M (a diagonal in comb coordinates) to make the
   32-column factorization exact on the observed families.
