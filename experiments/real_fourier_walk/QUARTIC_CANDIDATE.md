# Bounded quartic packet FFT with selectable cross-parent regrouping

Native follow-up: [factorization and scope](NATIVE_FACTORIZATION.md),
[NEON benchmark results](NATIVE_RESULTS.md). The cheap normalized factorization
exposes ordinary Bruun arithmetic with selectable packet routing; the quartic
coefficient-frame construction alone does not establish a new arithmetic FFT.

2026-09-05. `quartic_fft.py` is an executable real-input FFT candidate obtained
by keeping a bounded number of factors in each packet. It computes the ordinary
DFT in O(N log N) real arithmetic, with N real degrees of freedom in the active
packet frontier. Unlike the small-factor Newton chain, it does not need a
quadratic number of exchanges. Unlike a single quadratic Bruun packet tree,
it allows cross-parent pairings at each split. This is a candidate construction,
not a claim of novelty over the full literature or of competitive native speed.

## State and input

Define q_theta(y)=y²-2 cos(theta)y+1 for 0<theta<pi, and let the special
ridge factor be q_R(y)=y²-1. A packet has geometry

    Q(y)=q_a(y) q_b(y),  y=t^h,

and stores the real residue

    r(t)=r0(t)+t^h r1(t)+t^(2h) r2(t)+t^(3h) r3(t),
    degree(rj)<h.

Its four coefficient streams contain 4h real scalars. The state represents
the original input polynomial modulo Q(t^h). The initial packet is

    q_R(y) q_(pi/2)(y)=(y²-1)(y²+1)=y^4-1,
    h=N/4.

Thus ordinary real samples are already its coefficient streams. Input needs
no inverse transform or enlarged complex representation.

## Descent and freely selectable local pairing

When h is even, write z=t^(h/2). Each quadratic in y=z² has two real
quadratic factors in z:

    q_theta(z²)=q_(theta/2)(z) q_(pi-theta/2)(z),
    q_R(z²)=q_R(z) q_(pi/2)(z).

The two old factors therefore produce FOUR child quadratics q0,q1,q2,q3.
There are three ways to form two new quartic packets:

    (q0*q1, q2*q3)   preserve the two old parents
    (q0*q2, q1*q3)   cross them one way
    (q0*q3, q1*q2)   cross them the other way.

Each choice partitions the same roots and is exactly admissible. Forming the
two new residues is just reduction of the parent's eight coefficient streams
(in z, each of width h/2) modulo the chosen two monic quartics.

The production experimental reducer performs four monic elimination steps,
each affecting four lower coefficient streams. It uses real multiply/subtract
operations only. There is no dense inverse, complex Fourier subroutine, or
reconstruction of the entire input polynomial. The local carry is a fixed
4-by-8 map for each destination, independent of N.

After every split, packet polynomial degree remains FOUR while h halves.
This is the bounded-complexity property that the earlier complex torus/carry
family did not possess. It follows from keeping two quadratics together and
regrouping their children, rather than letting arbitrary root-set size grow.

The factor choice can be supplied by a callback at each visited node. A full
binary plan has N/4-1 internal nodes, each with three pairing choices. The
prototype follows directed refinement; it does not make arbitrary global
chart motion a logarithmic-length operation. The Newton exchange separately
supplies reversible local chart motion. These are related capabilities, not
identical claims.

## Endpoint and actual complexity

At h=1, each packet has four real coefficients. Reduce it to each of its two
quadratic factors. For q_theta, residue u+vt gives

    Re X = u+v cos(theta),   -Im X = v sin(theta).

The ridge residue gives DC=u+v and Nyquist=u-v. The prototype writes an N-real
packed output in natural bin order. There is no separate global sorting pass;
the terminal stores are indexed scatters, whose movement is not claimed free.

Each level does constant work per real coordinate, and there are log2(N)-2
splitting levels. Default stack operations are O(1) per packet, so the actual
default Python implementation has O(N log N) algorithmic work, not just a
formal arithmetic factorization. Data storage is O(N), including output and
scratch. The N-coordinate frontier statistic excludes temporary copies and
the separate output array; it is not an in-place-storage claim.

For the unspecialized reducer, each splitting level uses 4N real multiply
operations and 4N subtractions. Including leaf polynomial reductions and
readout, the literal arithmetic schedule uses

    multiplies = 4N(log2 N-2) + 3N - 2
    adds/subtracts = 4N(log2 N-2) + (5/2)N + 1.

Geometry construction, cosine calls, indices and allocation are additional
O(N) work in this prototype. Factors containing zero or +/-1 coefficients
have not been specialized away. The constant is substantially higher than
the normalized Bruun control. This is a fast transform asymptotically; native
speed superiority is not established.

## Path selection and conditioning

The implementation compares three policies:

- `sibling`: preserve old parent groups.
- `random`: independently choose one of the three pairings at each node.
- `separated`: keep the ridge paired with its quarter-turn companion and,
  for ordinary factors, maximize the smaller separation between paired cosine
  parameters. This is a conditioning heuristic, not an empirical speed model.

The separated policy selects cross-parent regroupings at 4083 of 4095 splits
in the N=16384 test. It therefore does not simply execute the sibling grouping
under a new label. Its minimum observed within-packet cosine gap is about
0.8665 in that case, versus 0.000767 for sibling grouping. These are measured
finite-size properties, not a general numerical stability theorem.

At N=16384 on the seeded real input:

| Policy | Cross-parent regroupings | Relative L2 error | Maximum absolute error |
|---|---:|---:|---:|
| Sibling | 0 | 1.56e-14 | 1.01e-10 |
| Random | 2711 | 2.33e-13 | 1.49e-9 |
| Separated | 4083 | 1.42e-14 | 4.24e-11 |

The separated route is much more accurate than the unrestricted natural-order
Newton chain. It remains less accurate than the normalized quadratic Bruun
control on its separate input battery. No matched speed measurement is made.

## Exact and numerical verification

`test_quartic.py` passes the actual reducer SymPy object arrays and compares
every output coefficient with independent symbolic polynomial remainders.
It checks all three pairings for the ridge example and for two generic parent
angles, with child cosine parameters C,D left symbolic: 96 basis-remainder
identities. Parent/child polynomial product identities are checked too.

At N=16 there are 3^3=27 complete pairing plans. Every plan is tested on all
16 input basis vectors, for 432 full-transform checks. The size sweep covers
three policies at N=8 through N=16384. Complex NumPy FFT is used only as an
independent test oracle; the transform implementation never uses complex data.

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 experiments/real_fourier_walk/test_quartic.py \
  --out /tmp/real_quartic_fft.json
```

Copy the JSON back immediately using `m4host`; results are retained as
`quartic_results.json`.

## Remaining research question

There is now a bounded real packet family with an executable O(N log N)
Fourier endpoint and locally selectable cross-parent refinement. The next
step is to factor its fixed real transition into a better-conditioned small
frame and fewer elementary operations while preserving its regrouping choices.
It must then be compared to the actual real BFFT kernels. Establishing whether
this construction is a new algorithmic family requires a careful comparison
with generalized Bruun/polynomial-factorization FFTs; the different tested
groupings alone do not prove novelty.
