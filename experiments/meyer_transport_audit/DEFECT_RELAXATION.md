# Refreshing the compatibility defect at the updated primal state

Follow-up: `RAMP_REFRESH_DIAGNOSIS.md` identifies both real alternating
feedback and substantial certificate-stage sensitivity in the ramp. The
timings below remain measurements of the stated next-projection convention;
they do not use the best feasible bound already available across stages.

September 5, 2026. This experiment implements the request to carry coupled
relaxation and its compatibility defects without introducing additional
global solvers or fitted step factors. It finds a useful but nonuniform
improvement from a second local memory projection at the new primal state.
Direction gates using current and previous defects improve some trajectories;
the simplest refresh has the best aggregate time-to-certificate here.

## Construction

For either branch, recover the feasible incoming memory b=t-Dx. The ordinary
pass computes p=Pi(Dx+b), advances the coupled primal pair, then stores
t+=Dx++p. Keep that entire primal computation. Before storing its endpoint,
compute one additional local projection

    q = Pi(Dx+ + p),
    b+ = q,  t+ = Dx+ + q.                            (1)

The two branches both use their own newly computed primal field. There is
no inner iteration or additional screened response. The original two
forward and two inverse FFTs per pass are retained. This is one extra
pointwise memory-relaxation stage, not a claim that extra computation is free.

The current compatibility defect and the new stage's defect are

    beta = p-b = Dx-shrink(Dx+b),
    e = q-p = Dx+-shrink(Dx++p).                       (2)

The directional variant chooses q at a site when <beta,e> >= 0 and p
otherwise. It lets a continuing local defect receive the extra relaxation,
while using the ordinary endpoint where the two stage defects disagree.
Both choices are feasible. A second variant compares e against the previous
pass's actual memory change instead, carrying that vector explicitly; its
first pass uses the ordinary endpoint because no previous witness exists.
No fitted threshold, scalar relaxation coefficient, or line search is used.
The zero dot-product boundary is a floating-point sign test, not an interval
certificate of direction under roundoff.

The final implementation carries b explicitly for gated variants, avoiding
the repeated reconstruction t-Dx. Current-direction gating adds four stored
arrays in 2-D; previous-direction gating also carries four memory-change
arrays. These are redundant with part of the original Markov history, but
useful computationally. The operator object belongs to one sequential
trajectory and must be recreated for a new trajectory.

## What is formally preserved

Memory feasibility and the exact graph t=Dx+b are preserved by construction.
At a feasible p, firm nonexpansiveness of projection gives, with g=Dx+,

    <g,q-p> >= ||q-p||^2.                            (3)

The extra memory motion thus has nonnegative work against the *updated*
primal gradient. This local fact does not imply global primal-dual gap
descent. The directional gate also gives

    ||beta+e||^2 >= ||beta||^2 + ||e||^2

at admitted sites. That is directional agreement, not a contraction claim.

The ungated refresh has exactly the same feasible fixed points as the
ordinary map. At a fixed primal x, let G(b)=Pi(b+Dx). Ordinary fixed memory
satisfies b=G(b); refresh fixed memory satisfies b=G(G(b)). Since G is
firmly nonexpansive, setting p=G(b) and assuming G(p)=b gives

    ||p-b||^2 <= <p-b,b-p> = -||p-b||^2,

so p=b. The converse is immediate. At a fixed point of either gated rule,
each site chooses G(b) or G^2(b), and the same argument applies sitewise.
Once p=b, the unchanged primal equations are exactly the ordinary fixed-point
equations. This fixed-point equivalence is not a global convergence theorem.

## Experiments that were rejected

The first screen at size 64 and 512 passes compared ordinary, reversed and
alternating primal schedules, algebraically fused primal responses, refresh,
and fused refresh. Changing only the primal schedule gave little consistent
improvement. The fused response uses a direct 2-by-2 Fourier formula with
the same transform count; it remains an experimental control.

Next, carrying the actual primal-gradient displacement into either the
projected or unprojected memory was tested:

    b+ = Pi(p + D(x+-x)), or b+ = Pi(t + D(x+-x)).

Both failed badly, including on the carrier. At (.05,40), the 64-sample
edge gap after 512 passes was approximately 130.39 or 80.64 versus 2.12e-5
for ordinary transport. The source and objective were unchanged. Preserving
memory feasibility and fixed points did not preserve convergence behavior.

A matched extra-projection control used the old primal gradient:

    q_stale = Pi(Dx+p).

At size 128 and 1024 passes it left large gaps on most inputs. For example,
at (.05,40) the crossing gap was 0.8282, compared with 2.661e-4 for ordinary
and 5.470e-5 for the fresh-state refresh. This distinguishes simply doing
more local projection work from updating memory at the appropriate stage.
It is evidence for stage dependence, not an attribution or novelty claim.

## Certificate and cost protocol

The baseline reuses its already computed projections and matches model.py;
it does not pay for the original oracle's duplicate projection evaluations.
All methods use the same NumPy Fourier backend. Update timings include
projections, gradient stencils, history, dot products and allocations.
Common certificate sampling is excluded from all update timings.

The 64-size screen uses five sources and three parameter pairs:
(.02,20), (.05,40), (.1,80), 512 passes, noise seed 983. The 128-size
validation uses 1024 passes and samples the feasible primal-dual gap per
sample every eight passes. Four sources are 128-square; the edge is 1x128.
Method order is shuffled per case. An uncached validation used noise seed
1987, and the final explicitly retained-memory validation used seed 3029.
Only the noise source changes with the seed. These are exploratory timings,
not repeated native throughput measurements or a claim of universal speedup.

Final retained-memory validation, 15 cases:

| Gap/sample target | Both ordinary and candidate reach | Plain refresh median speedup | Current-defect gate median speedup | Previous-defect gate median speedup |
|---|---:|---:|---:|---:|
| 1e-2 | 14 | 1.137x | 1.126x | 1.118x |
| 1e-3 | 12 | 1.318x | 1.150x | 1.158x |
| 1e-4 | 9 | 1.235x | 1.200x | 1.123x |

Speedup is ordinary update time divided by candidate update time at the
first sampled certified target. At 1e-3, all three candidates win time on
9 of the 12 paired cases; each reaches one additional case that ordinary
does not reach by 1024. Two cases are reached by neither. At 1e-4 each
candidate reaches two additional cases; four are reached by neither.
There are no ordinary-only target hits at these three thresholds in this
validation. Missing targets are reported explicitly and excluded from the
paired medians, rather than counted as zero or infinite speedups.

Examples at target 1e-3:

| Source and parameters | Ordinary passes | Refresh passes | Refresh speedup |
|---|---:|---:|---:|
| crossing, (.02,20) | 520 | 288 | 1.561x |
| noise, (.02,20) | 472 | 288 | 1.407x |
| carrier, (.02,20) | 184 | 96 | 1.758x |
| crossing, (.05,40) | 632 | 424 | 1.312x |
| ramp, (.02,20) | 88 | 128 | 0.563x |
| ramp, (.05,40) | 152 | 192 | 0.725x |
| noise, (.05,40) | 216 | 200 | 0.902x |

The regressions matter. The gates reduce several iteration regressions but
do not universally repay their overhead. Explicit older history has not
shown an aggregate advantage over the within-pass defect comparison. The
strongest supported result is the fresh local memory stage itself.

## The original shoulder and the longer tail

On the original 1x64 edge at (.05,40), the first negative shoulder gradient
after pass 64 occurs at pass 214 for ordinary and 206 for both refresh and
the current-defect gate. The sampled 1e-4 gap is reached at pass 432 versus
368. At 4096 passes all three gaps are below 7e-13.

On the harder 1x128 edge at (.1,80), the shoulder release occurs at pass
1112 for ordinary, 984 for refresh and 986 for the current-defect gate.
The sampled 1e-5 gap is reached at pass 1952 versus 1448 for both candidates.
At 4096 all four tested methods have gap below 3e-12. This checks the
convergence tail on these two examples; it does not prove convergence on
arbitrary images or establish a global monotonicity guarantee.

## Implementation, checks and reproduction

`defect_relaxation.py` contains all controls and candidates; production code
is unchanged. `test_defect_relaxation.py` has four test groups checking
ordinary-oracle equivalence, the exact transform budget, nonconstant-source
fixed points, memory feasibility, fused-response equations, retained-memory
consistency and the local projection-work inequality.

```sh
/Users/ultimussecundai/.local/bin/m4build -- env \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 -m unittest experiments.meyer_transport_audit.test_defect_relaxation -v

/Users/ultimussecundai/.local/bin/m4build -- env \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 experiments/meyer_transport_audit/validate_defect_relaxation.py \
  --size 128 --steps 1024 --seed 3029 \
  --methods ordinary,refresh,aligned_refresh,remembered_refresh \
  --out /tmp/meyer_defect_validation128_cached.json

/Users/ultimussecundai/.local/bin/m4build -- env \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 experiments/meyer_transport_audit/probe_defect_edge.py \
  --out /tmp/meyer_defect_edge_tail.json
```

Copy every /tmp JSON back immediately with m4host. Analyze the copied
validation with `analyze_defect_relaxation.py INPUT --out SUMMARY` locally.
The records are `meyer_defect_screen.json`, `meyer_defect_transport_screen.json`,
`meyer_defect_history64.json`, `meyer_defect_validation128_uncached.json`,
`meyer_defect_validation128_cached.json`, `meyer_defect_acquisition128_cached.json`,
and `meyer_defect_edge_tail.json`.
