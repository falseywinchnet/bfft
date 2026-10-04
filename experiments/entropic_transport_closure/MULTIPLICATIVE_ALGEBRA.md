# Multiplicative request algebra

The follow-up implements measured kernel action on the mixed-product algebra
of observed log-request directions. It preserves every ordinary full-vector
Sinkhorn request. It does not replace the trajectory with an optimizer.

## Construction and scope

For positive requests x = a exp(sum_j s_j w_j), retain an orthonormal basis Q
built by repeated pointwise multiplication by w_j, including mixed products.
Measure K diag(a) Q once at acquisition, counting every nonconstant column as
one product. The constant response comes from the current live K a query.
Future full requests are projected onto Q; reuse requires a componentwise
relative residual enclosure, positivity, and a 1e-8 relative interval-width
gate. Failed admission evaluates the ordinary product. Live carriers learn
2 or 4 directions by SVD of the last eight observed log requests and rebuild
after 64 misses. No future requests enter that learning.

If the generators take J distinct joint value patterns, their function algebra
has dimension J. It includes exponentials and nonzero reciprocals on those
patterns. Thus its measured kernel action suffices for all requests in the
family, not merely infinitesimal changes. For d binary generators, J <= 2^d.
This is exact-arithmetic closure; numerical generation and reuse are checked
separately. Generic continuous generators need not have a small exact algebra:
even one generator with n distinct values generates all n coordinate functions
through interpolation. Approximation can still be cheap for restricted ranges.

The implementation's floating-point allowance is an engineering estimate,
not directed-rounding interval certification. Reported bounds and tests audit
it numerically. The admission is based on full input reconstruction, not the
reported Frobenius multiplication-closure defect alone.

## Controlled families

`results/theory/algebra_family64_corrected.json` contains two seeds, n=128,
256 requests, rank budget 64, and three timing repeats. Directions are either
standardized binary fields or standardized Gaussian fields; total log-request
RMS coefficient scale is held at 1.4, not maximum log range. The supplied-family
arm receives the defined generators; the learned arm receives only live queries.
Both include acquisition costs. Entries below count kernel-vector equivalents.

| Family | Directions | Supplied mixed | Learned mixed | Supplied separate powers |
|---|---:|---:|---:|---:|
| Binary | 1 | 2 / 2 | 9 / 9 | 2 / 2 |
| Binary | 2 | 4 / 4 | 11 / 11 | 258 / 258 |
| Binary | 4 | 16 / 16 | 23 / 27 | 260 / 260 |
| Gaussian | 1 | 64 / 64 | 109 / 71 | 30 / 28 |
| Gaussian | 2 | 279 / 266 | 264 / 260 | 314 / 307 |
| Gaussian | 4 | 319 / 319 | 319 / 319 | 319 / 319 |

Ordinary evaluation costs 256. Supplied binary families reuse all 255 requests
after acquisition; learned binary families reuse all 248 after eight observations.
The learned binary rank can exceed the minimal pattern rank because numerical
SVD generators and basis stopping are not exact symbolic pattern detection.
One continuous direction admits all 255 supplied-family requests. Two continuous
directions admit only 40 / 53; four admit none. Separate powers stop numerically
earlier in one dimension than the recursively orthogonalized mixed construction;
its cheaper one-direction count is not a different exact algebra.

Mixed products are essential in the binary multidirection control. The separate
powers implementation explicitly forms original univariate powers before
orthogonalization, preventing orthogonalized cross-direction combinations from
silently introducing mixed terms. The initial files `algebra_family64_initial_axis.json`
and `multiplicative_algebra_first.json` predate this correction; their axis
comparisons are superseded and must not be used as valid ablations.

Maximum observed Hilbert output error across the corrected family study was
1.13e-8, and maximum bound excess was 9.46e-14. These are measured errors, not
machine-certified bounds. No small-n wall-clock speedup is demonstrated: for
seed zero ordinary evaluation took approximately 0.6 ms, while mixed-family
runs took 6.2–12.5 ms including setup and admission.

## Actual difficult trajectories

`results/theory/algebra_actual1024.json` uses the two original difficult sources,
n=1024, epsilon=.001, 256 complete ordinary passes, rank budget 64, and three
repeats on the M4 CPU with one BLAS thread. There are 512 ordinary products.

| Method | Products, seeds 0 / 1 | Reuses | Median ms, seeds 0 / 1 |
|---|---:|---:|---:|
| Ordinary | 512 / 512 | 0 / 0 | 73.38 / 61.34 |
| Existing response cache | 442 / 433 | 70 / 79 | 76.69 / 72.90 |
| Separate powers, 2 directions | 988 / 984 | 0 / 0 | 167.03 / 162.10 |
| Mixed, 2 directions | 1016 / 1016 | 0 / 0 | 215.95 / 216.55 |
| Mixed, 4 directions | 1015 / 1012 | 1 / 4 | 222.74 / 223.65 |

New carrier timings include retrospective rejection diagnostics; they are not
optimized production timings. Acquisition and product counts already establish
failure to save primitive work here. Largest new-carrier terminal Hilbert error
was 2.13e-10; most updates simply fall back to the ordinary product.

After each rejection, the already-required full product audits the rejected
prediction, without extra kernel calls and without affecting admission or
learning. In the two-direction mixed arm, none of 496 rejected predictions
per seed had componentwise relative output error <=1e-8; median errors were
7.05 and 5.65. Four directions left only 2 of 495 and 1 of 492 rejected candidates
within that threshold. Thus an overly conservative input certificate is not the
principal failure in these runs. The limited learned algebra does not represent
the future requests accurately across the frozen-anchor range. These diagnostics
do not separately identify generator insufficiency versus polynomial truncation.

## Conclusion

Finite joint-pattern closure is implemented and experimentally verified. It
explains a real success of multiplicative representation. The present learned,
truncated algebra does not supply cheap general closure for actual difficult
Sinkhorn trajectories. It is an experimental carrier, not a promoted accelerator.
Gaussianity alone is not a universal smooth cost law: joint-pattern complexity,
range, tolerance, acquisition, and family coverage all matter.

## Reproduction

Run on the Mini through m4build, with OPENBLAS_NUM_THREADS=1 and
VECLIB_MAXIMUM_THREADS=1:

```sh
python3 -m unittest experiments.entropic_transport_closure.test_multiplicative_algebra -v
python3 -m experiments.entropic_transport_closure.probe_algebra_family --budget 64 --repeats 3 --out /tmp/algebra_family64_corrected.json
python3 -m experiments.entropic_transport_closure.probe_multiplicative_algebra --size 1024 --budget 64 --eps .001 --repeats 3 --out /tmp/algebra_actual1024.json
```

Copy both JSON files back immediately. Four structural tests cover binary closure,
the necessity of mixed products, correctness of the separate-powers control,
and the live ordinary trajectory bound.
