# Exact CONV* ledger fusion

September 5, 2026. The rolling ledger-cost recurrence is promoted into
`standalone_conv_resize_demo/native/conv_native.c`. The interpolation kernel,
proposal, admissible set, projection objective, boundary search interval,
first-minimum tie rule, and two-order reconstruction are unchanged.

## Algebraic reduction

At a witnessed sign change, let s be the left sign and -s the right sign.
For a boundary b the existing ledger cost is

\[
 C(b)=\sum_{j<b} a_j^2\,1_{s a_j<0}
      +\sum_{j\ge b} a_j^2\,1_{-s a_j<0}.
\]

Moving the boundary one position changes only one term:

\[
 \boxed{C(b+1)=C(b)-s a_b|a_b|.}
\]

The implementation computes the first cost and signed squared increments
once, then walks the same permitted boundary positions using addition.
The prior implementation could repeat ten term checks at each of nine
positions; the ordinary new path uses ten initial term evaluations and
at most eight cost updates. This is reuse of an existing finite calculation,
not a new model-selection stage or an iterative solver.

An intermediate prefix/suffix implementation also works:
C(b)=P_left(b)+S_right(b). The rolling form is cheaper in the measured cases.
The complete asymptotic complexity remains linear in image size.

## Floating arithmetic and ties

Reassociation can alter nearly equal costs. The new code tracks the best and
second-best values and the local squared-current energy E. If their gap is
at most 64*DBL_EPSILON*E, or E is nonfinite, it executes the old cost loop
with its original order and tie rule. This numerical ambiguity guard does
not modify a geometric threshold or relax admission.

For finite binary32 inputs the squared values fit in binary64 without
overflow or underflow. There are at most ten initial terms and eight signed
updates. Standard bounded-roundoff accounting gives an absolute error
bounded by a small multiple of binary64 unit roundoff times E; the guard
exceeds the combined comparison uncertainty of both accumulation paths.
Ambiguous cases therefore retain the original computation. The measured
bit comparisons below supplement this arithmetic argument.

## Measured complete-pipeline time

M4 Mini, native C -O3 -mcpu=native, same Python wrapper, allocations, basin
analysis, nodal geometry, and both synthesis orders included. These are
31-repeat warmed medians with alternating method order. Times exclude native
compilation. The four-worker study preceded the addition of the nonfinite
energy guard; all its inputs are finite. The one-worker study includes the
guard and explicitly compares uint32 views of the output buffers, including
signed zeros.

| Workload | Previous, four workers | Fused, four workers | Time reduction | Time reduction, one worker |
|---|---:|---:|---:|---:|
| Cameraman 512 -> 64 -> 512 | 6.825 ms | 5.696 ms | 16.54% | 16.03% |
| Text 8x round trip | 2.338 ms | 2.048 ms | 12.44% | 13.47% |
| Brick 8x round trip | 6.054 ms | 5.150 ms | 14.93% | 16.01% |
| Grass 8x round trip | 7.220 ms | 6.009 ms | 16.77% | 17.19% |
| Coins 8x round trip | 3.318 ms | 2.867 ms | 13.59% | 15.59% |
| Cameraman 64 -> 256 | 1.080 ms | 0.960 ms | 11.13% | 13.83% |
| Cameraman 64 -> 512 | 1.884 ms | 1.671 ms | 11.34% | 13.91% |
| Cameraman 64 -> 1024 | 4.308 ms | 3.850 ms | 10.63% | 13.30% |
| Noise 64 -> 512 | 2.189 ms | 1.839 ms | 16.01% | 19.84% |
| RGB noise 64 -> 512 | 6.055 ms | 5.085 ms | 16.03% | 20.35% |

The result is approximately 1.12–1.20x faster with four workers and
1.16–1.26x faster with one worker on these workloads. This is an empirical
range, not a universal speed guarantee. Mathematical work is reduced only
at ledger transitions; flat or small workloads need not receive these gains.

## Equality and regression checks

Each study compares 20 source-profile banks and ten complete image workloads
across three implementations. Cases include quantized ties, constants,
ramps, steps, random signals, source lengths 5/9/33/129, amplitude scales
1e-20 through 1e20, and large dynamic ranges within a single profile bank.
All comparisons passed. The one-worker run explicitly checks raw uint32
buffer equality. No pixel changes, MSE changes, or display clipping are
introduced by the promoted code on these tests.

After promotion, four focused fusion tests plus the existing native backend
and distilled-CONV suites passed on the Mini: 39 tests in total. The extra
legacy Python CONV* suite initially could not import matplotlib on the Mini;
all eight tests subsequently passed in the existing local .venv-jpeg
environment, giving 47 passing tests across the two environments. No
dependency installation is part of this change.

The pre-fusion backend SHA256 was
`17a5b35898b2a2a15e57e0ef154d2f9a5d7b53243627542278fd8bf4a0e68188`.
`original_ledger.c.inc` and the marked production block allow the experiment
to reconstruct the exact original backend source after promotion, so its
baseline does not silently become the optimized implementation.

## Other attempted reductions

The projection onto the mass hyperplane has the explicit form

\[
 \lambda=(\mathbf1^Ta-\delta)/5,\qquad c=a-\lambda\mathbf1.
\]

If c satisfies the signed constraints, it is also the signed-fibre optimum.
An early-exit implementation with guards preserving binary32 rounding passed
the equality checks, but added roughly 1–4% to total time in the first screen.
The guard and branching overhead outweighed the saved work. It was not
promoted. Streaming ledger signs directly into projection also gave no
consistent improvement and was not promoted.

## Larger mathematical fusion pathway

For each axis write T_d=R_d+E_d, where R_d is the raw linear quintic and E_d
the nonlinear admission correction. Since admission preserves interval mass,
the correction has zero value at the interval endpoints. The raw tensor
operators commute in exact arithmetic: R_y R_x=R_x R_y=R_xy.

Consequently,

\[
 T_{xy}Y=R_{xy}Y+C_{xy},\quad
 C_{xy}=R_yE_xY+E_y(T_xY),
\]
\[
 T_{yx}Y=R_{xy}Y+C_{yx},\quad
 C_{yx}=R_xE_yY+E_x(T_yY),
\]

and the existing blend equals

\[
 \boxed{T^*Y=R_{xy}Y+C_{xy}+\beta(C_{yx}-C_{xy}).}
\]

This identity was already recognized in the older CONV conversation. It is
a valid remaining compilation pathway, not a newly discovered operator.
Only the admission corrections cause the order difference. Sharing the raw
backbone and transporting correction supports could save further work.
However, a naive expansion can compute more filters: second-stage admission
still depends on each complete first-stage result. The identity alone does
not establish a speedup or allow either nonlinear branch to be discarded.
No such speed claim or production rewrite is included here.

## Proof scope

The earlier six-tap half-sample kernel is unique under its degree-five
polynomial reproduction constraints. Neither that uniqueness nor the need
to pay a tradeoff for a different objective proves global optimality of all
CONV reconstructions. The promoted ledger recurrence needs no broader
optimality claim: it computes the same existing cost and decision with
less repeated arithmetic.

## Reproduction

From the repository, use m4build with CONV_NATIVE_THREADS=1 or 4 and
OPENBLAS_NUM_THREADS=1, VECLIB_MAXIMUM_THREADS=1:

```
python3 -m experiments.conv_exact_fusion.study --methods base,ledger,rolling --repeats 31 --out /tmp/conv_exact_fusion.json
```

Copy the JSON immediately into `output/support_geometry/conv_exact_fusion/`.
The retained records are `conv_exact_fusion_rolling4.json` and
`conv_exact_fusion_rolling1.json`; `conv_exact_fusion4.json` contains the
first projection/prefix/stream ablation.

## Larger fusion proof follow-up

See [LARGER_FUSION_PROOFS.md](LARGER_FUSION_PROOFS.md) for the exact four-coordinate
mass-null correction, transverse coefficient transport, certified polynomial
admission reuse, direct projection/synthesis stencil, and conditional
bicquintic commutator factorization. Seven rational tests pass; thirty profiles
also match the independent existing float64 implementation. The small 2-D
certificate probe establishes nontrivial exact reuse, but its natural-image
interval coverage and eighth-cell granularity do not establish useful 8x
amortization. No additional native optimization is promoted by those proofs.
