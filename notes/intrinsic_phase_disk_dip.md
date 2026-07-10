# Intrinsic Phase-Disk DIP

Status: exact packet algebra and Python reference validated, 2026-07-09.
The fast joint time/frequency factorization is the remaining kernel problem.

Companions:

- `experiments/intrinsic_dip_frontier.py`
- `experiments/intrinsic_dip_branchbound.py`
- `experiments/intrinsic_dip_boundary.py`
- `experiments/dip_numba.py`

## 1. Target

For complex input `x[n]`, emit per Fourier bin

    C(k,tau_k) = sum_{n<tau_k} x[n] exp(-2 pi i k n/N)

where `tau_k` maximizes `|C|^2/tau`.  `arg C` is the sine/cosine correlation
phase.  Support must be packet metadata produced inside the walk; an external
endpoint recurrence is not part of the transform.

## 2. Exact support frontier

A local time packet A at frequency omega carries its total phasor `Z_A` and
prefix set `P_A={(tau,C_A(tau))}`. Adjacent packets combine as

    rho  = exp(-i omega |A|)
    Z_AB = Z_A + rho Z_B
    P_AB = P_A union {(|A|+tau, Z_A + rho z) : (tau,z) in P_B}.

This is exact set-valued closure.  Its worst-case width is linear, so it is a
definition, not yet a fast algorithm.

The bounded-frontier experiment keeps candidates that win under possible
future complex translations, rather than merely keeping the current top score:

    q_j(a,L) = |a+z_j|^2/(L+tau_j).

At N=256, envelope width M=16 gave median score ratio 1.0 on tones, bursts,
chirps, crossings, and noise; chirps/crossings were exact in every active bin.
M=32 made all tested active cases exact except neighboring burst endpoints
whose score ratio remained essentially one.  The result supports an adaptive
frontier, but does not prove constant exact width.

## 3. Fixed-size certified phase disk

For selection, a packet need not retain every prefix.  It can retain a disk
`D(c,r)` known to contain all local prefix phasors.  If packets A and B are
adjacent, rotate B into A's coordinates and form

    D1 = D(c_A, r_A)
    D2 = D(Z_A + rho c_B, r_B).

The smallest disk enclosing D1 and D2 is a parent bound. For centers separated
by `d=|c2-c1|`:

- if one disk contains the other, keep the containing disk;
- otherwise `R=(d+r1+r2)/2` and
  `c=c1 + (R-r1)(c2-c1)/d`.

This is constant-size compositional state and is certified even though repeated
pairwise enclosure need not be the globally minimal disk.

With a preceding phasor `a`, every unresolved endpoint in a packet obeys

    |C| <= |a+c| + r
    score <= (|a+c|+r)^2/tau_min.

If that upper bound cannot beat the incumbent, the entire support packet is
discarded. Otherwise it splits. This produces the exact global optimum: no
heuristic endpoint window and no DDC polish.

At N=1024, among activity-gated bins, the exact selector visited a median of
only 2 endpoint leaves:

| signal | median / p90 / max leaves | median joint support nodes |
|---|---:|---:|
| tone | 2 / 2 / 3 | 61 |
| burst | 2 / 4 / 6 | 83 |
| chirp | 2 / 3 / 5 | 51 |
| noise outliers | 2 / 2 / 2 | 65 |

At N=2048 the median stayed 2--3 leaves and roughly 59--77 nodes.  Ambiguity
work did not grow with N in these signals.  Building a complete disk tree per
bin would still be quadratic; production must create/tighten disks lazily from
the linear packet tower. A cheaper frequency-independent energy disk is always
available:

    |partial block| <= sqrt(block_length) ||x_block||_2.

It is looser but requires no per-frequency preprocessing.

### Frontier-seeded certification

A width-16 phase-frontier beam is an effective incumbent generator even when
its residual disk is too loose to certify the answer by itself.  Feeding its
winner into exact disk descent at N=256 reduced median opened nodes from
51→49 (tone), 53→51 (burst), 37→33 (chirp), 41→33 (crossing), and 53→39
(noise outlier). Median endpoint leaves fell to zero in every active test: the
disk walk only proved the beam winner, rather than discovering another leaf.

Residual candidates are retained as certified disks separated by dyadic tau
band and phase cone. A single disk was too coarse for transient signals; the
banded/phase-cone form improved certification but did not eliminate fallback.
The correct product is therefore anytime: beam first, certified descent only
for unresolved packets. It never silently accepts an uncertified beam result.

## 4. The Zak boundary theorem

At DIP level t, let `e=2^t`, `q=N/e`, and write

    tau = a q + b,  0 <= b < q.

Since time is `n=j+r q`, the prefix mask is

    n < tau  iff  r<a, or (r=a and j<b).

Therefore a natural leading edge is exactly `a` complete time-comb rows plus
one partial row in the finite Zak lattice.  It is a two-height staircase, not
an arbitrary mask. Its level-t Zak state is

    B_t^tau[delta,j]
      = sum_{r<a} x[j+r q] exp(-2 pi i delta r/e)
        + 1_{j<b} x[j+a q] exp(-2 pi i delta a/e).

This was validated against `dip_partial_forward(masked_x,t,N)` and finishing
from every paused level, with errors around 1e-12 at N=256.

When the walk advances one level (`q' = q/2`), support metadata closes as

    a' = 2a + floor(b/q')
    b' = b mod q'.

One boundary bit is promoted from b into a per DIP level, synchronized with the
frequency-bit refinement.  This is the intrinsic support type that Claude's
search never used.

Selected phase-disk endpoints were reified as `(a,b)` paused Zak states and
finished through the normal DIP. Their selected-bin correlations matched the
direct prefix correlation to about 2e-12. There is no special FCT egress.

## 4b. Adversarial separation from the heuristic FCT

`experiments/ipd_vs_fct_adversarial.py` generates deterministic noisy
multi-burst signals at N=128. Across 200 trials, the checked-in heuristic FCT
produced a non-DC bin with

    chosen tau = 128
    true tau   = 12
    chosen score / global score = 0.012936.

The exact phase-disk reference returned tau=12 and the direct correlation to
machine precision. The distinction between support search and intrinsic
certified support is therefore operational, not terminological.

## 5. Candidate transform

Call the emerging object **Intrinsic Phase-Disk DIP (IPD-DIP)**.

A joint packet contains:

    frequency phase type: (d,e) or complex-DIP row delta
    support type:         boundary interval in (a,b)
    linear state:         total phasor / paused Zak lane
    nonlinear bound:      phase disk or energy cone
    incumbent:            (score,tau,C)
    ambiguity flag

The walk proceeds as follows:

1. Ordinary DIP cells transport total phasors and resolve frequency bits.
2. In lockstep, the boundary automaton promotes one support bit.
3. A support child whose certified disk/cone cannot beat the incumbent dies.
4. A decisive child becomes egress metadata without further support work.
5. Only ambiguous joint time-frequency packets split; these are the adaptive
   work count and a useful uncertainty observable in their own right.
6. The winning paused Zak lane continues through the ordinary DIP finish.

The desired cost is `O(N log N + Q)`, where Q is opened ambiguous joint
packets. The experiments show small Q on structured signals. This complexity
is not yet proved for the shipped DIP because contiguous time-block totals must
be supplied to the joint packets without materializing every fractional
frequency channel. The honest worst case remains O(N^2).

## 6. Remaining kernel theorem

The remaining problem is not support selection. It is exact/shared transport
of the total phasors needed when an `(a,b)` support packet splits.

Equivalent views:

- a joint dyadic recursion over a contiguous time interval and a dyadic
  fractional-frequency packet;
- a contextual leading-edge transform along the Zak column axis;
- a butterfly-factorization problem for the Fourier submatrix restricted to
  ambiguous support packets.

The DFT/ODFT one-level identity is insufficient because recursive splitting
introduces quarter-, eighth-, and finer-bin channels. Viable next paths are:

1. use the DIP diagonal phase packet as an exact fractional-frequency type and
   derive the matching contiguous-boundary shear;
2. use a controlled-rank Chebyshev/Taylor phase packet with certified remainder
   bounds (a Fourier butterfly/FMM formulation);
3. initially accept O(N log^2 N) linear totals while validating that Q and the
   nonlinear support work stay small, then factor the linear tower separately.

The per-channel primitive is now established. For

    F_N^alpha[p] = sum_n x[n] exp(-2 pi i (p+alpha)n/N),

even/odd splitting gives children with the **same alpha** and parent twiddle
`exp(-2 pi i (p+alpha)/N)`. Thus arbitrary dyadic `alpha=delta/e` is one
twisted DIP/FFT cell family with an offset angle; it needs no input pre-twist.
`experiments/twisted_fractional_dip.py` validates direct sums and the global
`(block,delta/e)` channel identity to machine precision.

### Fractional-channel cost audit

For a dyadic time block of length L, set `e=N/L` and write global bins
`k=p e+delta`. All bins with fixed delta are one length-L fractional transform
with offset `alpha=delta/e`. Caching opened packets by `(block,delta)` therefore
shares work across coarse-frequency p exactly; BODFT is the alpha=1/2 member.

At N=1024, using the certified energy-bound openings and a hybrid choice of
direct dots, local fractional FFTs, or a zero-padded full FFT per block, the
estimated linear work was:

| signal | work / (N log2 N) | work / N^2 |
|---|---:|---:|
| tone | 20.3 | 0.199 |
| burst | 10.0 | 0.098 |
| chirp | 39.6 | 0.386 |
| noise outliers | 2.1 | 0.021 |

An exact pruned twisted butterfly was then implemented: requested coarse bins
share child residues recursively, so one bin costs O(L), all bins recover the
ordinary length-L FFT, and partial requests interpolate without heuristics.
Measured complex-butterfly counts relative to one N-point FFT were 20.0×
(tone), 11.4× (burst), 39.8× (chirp), and 3.8× (noise outliers). This confirms
the hybrid estimate and provides the actual production-shaped recursion.

This is already far below a DDC bank for sparse/transient content, but the
chirp is not yet FFT-like. `O(N log N + Q)` remains the target cost contingent
on a better shared fractional-packet factorization; it is not a current theorem.

The third path is the safest implementation sequence: the new mathematics is
already testable without pretending the last factorization has been solved.

## 7. Shared transport results (2026-07-09, second pass)

Companions: `experiments/fractional_channel_sharing.py`,
`experiments/spectral_gate_selector.py`.

### 7a. Refined demand accounting

The earlier audit charged every *visited* node.  The walk only consumes a
block total when a non-pruned internal node hands `before + V(left)` to its
right child, plus leaves and dyadic seed prefixes.  Recounted this way the
honest baseline at N=1024 is 20.5x (chirp), 6.4x (burst), 2.1x (noise), not
40/11/4x.

### 7b. Exact cross-scale sharing (demand-closure DP)

In global-phase coordinates the DIF identity

    V([lo,lo+2L), k) = V([lo,lo+L), k) + V([lo+L,lo+2L), k)

is one add per (parent, k).  A demanded channel is either evaluated DIRECTly
(pruned twisted butterfly) or SYNTHesized from its children, which then
inherit the parent's bin set; inherited sets along a root path are nested, so
the min-cost assignment is a small memoized DP.  At N=1024 it gives
burst 6.4->3.7x, noise 2.1->0.6x, chirp 20.5->18.2x.  Chirp cost concentrates
at L=64..16 where every active bin demands nearly every block: the
Cauchy-Schwarz bound cannot prune off-resonance blocks.

### 7c. Certified spectral gate

One ordinary FFT per dyadic block gives an exact Dirichlet representation
with a shared numerator: with local fractional frequency nu = kL/N,

    |V(B,k)| <= (|sin pi nu|/L) Sum_p |G[p]| / |sin(pi(nu-p)/L)|,

evaluated as a few exact near terms plus dyadic distance rings over circular
prefix sums of |G| (O(1)-ish per query).  Prefix-max closes exactly:

    maxpre(B) <= max(maxpre(B_L), Q(B_L,k) + maxpre(B_R)).

Property-tested as a true upper bound; the pyramid is built lazily and
amortizes across bins.  Chirp DP transport falls 18.2->12.2x plus ~2x lazy
pyramid, but fine-scale demand barely moves.

### 7d. The Fresnel demand law (measured obstruction)

Chirp demand per active bin under magnitude-only bounds grows superlinearly
in log N (energy-walk nodes/bin 92/119/150/195 at N=512..4096, ~N^0.36),
because near bin k's resonance the score plateau holds ~sqrt(N/rate)
magnitude-indistinguishable endpoints.  The phase-disk walk stays
essentially logarithmic (41/47/54/63).  Consequence: **no amount of
magnitude-bound transport sharing reaches O(N log N + Q) on chirps — the
demand itself is the floor.  Phase-aware bound state must be transported.**
This answers which stronger type is required: phase, not more support
resolution.

### 7e. Two-regime hybrid (exact, end-to-end)

Coarse nodes (L > Lf=64) prune with the O(1) magnitude gate; a fine node
entered after the magnitude gate fails builds its exact local disk subtree
once (3L, descendants free) and prunes with `|before+c|+r`.  All bounds are
certified, every bin asserts the true global optimum (validated on the
adversarial multi-burst class; see `tests/ipd_test.py`).  End-to-end ops
(pyramid + envelope queries + coarse V transport DP + fine disk builds)
versus a full DDC bank:

| N | chirp hybrid /FFT | DDC /FFT |
|---:|---:|---:|
| 1024 | 48.8 | 31.6 |
| 2048 | 59.5 | 55.5 |
| 4096 | 69.0 | 97.7 |

The hybrid grows ~ +10 per doubling (log-linear); DDC compounds ~1.75x.  The
exact certified transform is subquadratic on chirps and beats the DDC bank
beyond N~2048.  Dominant remaining term: per-bin fine disk builds (~28-31x),
which share nothing across bins.

### 7f. The isolated next problem

Fine disks are the only object still costing O(L) per (block,bin) pair.
Candidate factorization: a banded frequency-grid disk pyramid per block —
at sub-block length l keep disks only on a frequency grid of spacing L/l
(l reps per sub-block, so linear per scale), answer off-grid queries with a
certified inflation `l1(sub-block) * 2 pi |dk| l / L`, and refine grids up
the DIF recursion.  Per block this is ~L log^2 L amortized across every bin
that touches the block, replacing 3L per bin.  Untested; it is the disk
analog of the twisted butterfly and is now the single remaining kernel
problem.

## 8. Frequency-grid disk store (2026-07-09, third pass)

Companion: `experiments/grid_disk_pyramid.py`; regression in
`tests/ipd_test.py`.

Solved 7f, with one simplification over the sketch: a **single grid**
(spacing `dk = N/(Lf G)` bins) instead of per-scale grids.  Disk trees are
built **exactly** at grid frequencies — every tree node combines at the same
frequency, so the build carries zero inflation — and all slack enters once
at query time through the translation lemma: for a sub-block S with global
center n0 and offset `dw = 2 pi (k-k_g)/N`,

    |C_k(tau) - e^{-i dw n0} C_{k_g}(tau)| <= M1(S) |dw|,
    M1(S) = Sum_{n in S} |x[n]| |n-n0|   (O(1) from two prefix-sum tables),

so the grid disk rotated by `e^{-i dw n0}` and inflated by `M1 |dw|`
certifiably contains every prefix at k (containment property-tested to
6e-13).  Worst-case query inflation is `M1(S) pi/(Lf G)`, roughly
`l1 pi/(4G)` at the top scale; even G=4 left the walk's node count intact
(60.3 vs 58.5 per bin on chirp).  The per-scale-grid variant in 7f is
strictly worse here: refining grids up the recursion re-snaps children and
accumulates `~l1 pi log(L)/(4G)` of build inflation, while the single grid
pays the bound once.

The sharing factor per build is `dk` bins, so the scheme pays nothing at
`N ~ Lf G` and improves linearly beyond it — measured exactly so (chirp,
G=4, Lf=64; fine-build term of the end-to-end cost):

| N | builds/bin | fine builds /FFT | total /FFT | DDC /FFT |
|---:|---:|---:|---:|---:|
| 1024 | 1.3 | 7.6 | 43.7 | 31.6 |
| 2048 | 0.8 | 4.4 | 47.9 | 55.5 |
| 4096 | 0.5 | 2.2 | 54.7 | 97.7 |
| 8192 | 0.3 | 1.2 | 67.4 | 173.1 |

(G=16 at N=1024 has dk=1 — zero sharing — and reproduces the unshared
hybrid's 27.6x exactly, a good sanity check.)  Accounting is active-bin
only, consistent with the whole series; the walk asserts the certified
global optimum for every bin, including the adversarial multi-burst class.

The fine-disk term has gone from dominant (30.6x at N=4096) to negligible
(1-4x) and shrinks relatively as N grows.  What now governs the total is
the coarse magnitude-gated walk itself: nodes/bin grow ~N^0.3 (60/74/91/114
at N=1024..8192) versus the pure disk walk's ~log N (47/54/63), and the
V-transport DP and envelope-query terms track that node count.  The
remaining gap to `O(N log N + Q)` is therefore no longer transport of any
kind — it is extending phase-aware pruning power to coarse scales, where
grid disks are ineffective because the translation inflation
`l1(S) pi/(4G)` grows linearly with block length while resonant disk radii
do not.  A coarse phase bound that beats the Dirichlet envelope without
O(L)-per-bin state is the next (and different) problem.
