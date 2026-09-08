# A real Fourier state space with arbitrary admissible transitions

2026-09-05. This experiment asks about freedom of traversal before speed.
Its state is a tuple of real polynomial residues. It never promotes an
N-real-input signal to N independent complex numbers, calls an FFT to move,
or forms a dense change-of-basis inverse. Exact algebraic arithmetic is used
to test identities; this is a mathematical oracle, not a production kernel.

The polynomial/Chinese remainder foundation is established algebraic signal
processing, including real Bruun transforms. See Voronenko and Püschel,
*Algebraic Signal Processing Theory: Cooley–Tukey Type Algorithms for Real
DFTs* (2009), Sections II–IV:
https://spiral.ece.cmu.edu/pub-spiral/pubfile/jrft_131.pdf
The investigation here uses that foundation to test arbitrary regrouping,
changing spectral locations, composition, and collisions. No novelty priority
is asserted for these underlying identities.

## The represented object and the state

An input x represents a real polynomial

    f(t) = sum(j=0..N-1) x_j t^j,  degree(f)<N.

A chart gamma consists of pairwise coprime monic real polynomials p_i,
with sum(degree(p_i))=N. Its state is

    r_i = f mod p_i,    degree(r_i)<degree(p_i).

There are exactly N real coordinates, even when some p_i have nonreal roots.
The individual r_i are real polynomials. Algebraic numbers in the exact
implementation require more than one machine word each; N counts real
degrees of freedom, not the storage bytes of the symbolic oracle.

The chart, its chosen monomial bases, and its residues completely specify
the represented f. The state does not retain input samples, a full spectrum,
or a path history. The implementation caches geometry-only polynomial
inverses; cache contents are not signal state and are not needed for semantics.

## Fixed-root packet space

For an even-length real DFT, the irreducible real factors of t^N-1 are

    t-1, t+1,
    t² - 2 cos(2πk/N)t + 1,  1<=k<N/2.

Each quadratic represents a conjugate pair using two real coordinates.
A packet can contain ANY subset of these factors. A partition of the factors
is a chart; its state consists of residues modulo the packet products.

This already exceeds a single radix tree: incompatible groupings are legal
states, and one can split, merge, or change groupings at any step. For example,
at N=8, with factor labels 0,1,2,3,4, one can go from

    {0,1}|{2,3,4}  to  {0,2}|{1,3}|{4}

without forming the full input polynomial or a complete singleton spectrum.
`Atlas.move` works through intersections of old/new packets, reducing each
source only to the required intersection and merging into each target.

For merging two residues r mod A and s mod B, use

    r_new = r + A * (((s-r) * inverse(A mod B)) mod B).

The inverse is a polynomial Bézout inverse, computed by extended Euclid on
known chart geometry. It is not an inverse FFT, signal-dependent solver,
or an arbitrary dense matrix inverse.

At N=8 there are Bell(5)=52 partition charts, with 160 undirected elementary
split/merge edges. All are connected. The exact test checks each directed
edge on all eight input basis vectors: 2560 basis-edge identities.
Consequently each edge is verified as a linear operator on arbitrary input.
The theoretical CRT argument below establishes path composition beyond this
finite enumeration.

## Moving the spectral locations, not just the partitions

The chart polynomials need not remain factors of t^N-1. They can change,
provided they remain pairwise coprime and keep total degree N. This permits
moving a conjugate pair on the unit circle, off-circle charts, repeated-root
packets, and the half-bin ODFT endpoint. It is a broader interpolation space
containing Fourier charts, not a claim that every chart is a uniform Fourier grid.

The following is an explicit transition law, not merely R_new R_old^-1.
For the current chart p_i define geometry

    P = product_i p_i
    M_i = P/p_i
    u_i = inverse(M_i mod p_i).

For each current residue form its local correction

    a_i = (u_i r_i) mod p_i.

For each target polynomial q_j, compute directly

    s_j = [sum_i (M_i mod q_j) (a_i mod q_j)] mod q_j.

All quantities are real. Each a_i has its source packet's degree bound.
The implementation reduces the contributions directly into target packets;
it does not construct the coefficients of sum_i M_i a_i as an intermediate.
If a target factor is unchanged, its residue can be retained directly.

Why the rule works: sum_i M_i a_i has degree<N and agrees with r_i modulo
every p_i. CRT uniqueness therefore identifies that sum with the represented
f. Reducing its contributions modulo q_j yields f mod q_j. This proof works
even when product(q_j) differs from product(p_i).

This freedom is global-state freedom. The exact test moving just one quadratic
to t²-(2/3)t+1 shows that its new residue depends on ALL five old FFT packets
at N=8. The other four target residues remain unchanged. An arbitrary move
exists, but it is generally not determined by the moved packet alone.
No assertion of sparsity or economical execution is made here.

## Composition, closed loops, and memory

For admissible charts alpha, beta, gamma,

    T_beta->gamma T_alpha->beta = T_alpha->gamma.

Both sides represent the same unique degree-<N real polynomial in gamma's
coordinates. In particular T_alpha->alpha is identity and a closed chart
loop closes exactly. No path history is needed. Here bases are explicitly
fixed monomial bases; there is no hidden rotation of an unspecified frame.
Additional independently chosen real frames would have to be part of the
chart metadata before comparing endpoints.

The graph of fixed-root partitions is discrete and enumerable. The broader
chart space also includes continuously parameterized polynomials. Dijkstra
can search a finite chosen graph once edges and weights are supplied; this
experiment deliberately supplies no performance objective or speed weights.
Its reachability calculation is an unweighted graph check.

## Why jets appear at a collision

Two SEPARATE chart factors may not share a root. At that point their residue
constraints cease to determine N independent coordinates. This is an actual
admissibility boundary, not a failure of a particular traversal implementation.

A repeated root INSIDE one packet is different:

    f mod (t-a)^d

retains the first d Taylor/derivative coordinates of f at a (after the known
change from monomial to shifted monomial basis). In particular a double root
retains f(a) and f'(a). Two coincident value samples alone would lose one of
those coordinates; the residue packet keeps it.

The exact test traverses t²+lambda at lambda=-1/4,0,+1/4 while the other
packet is (t-2)^6. The factors stay coprime along this entire interval. The
moving packet's roots pass from real to a double root to a conjugate pair,
while its two real residue coordinates remain valid. This witness goes off
the unit circle; it does not prove a continuous unit-circle-only passage from
FFT to ODFT. The direct discrete FFT-to-ODFT chart transition is tested separately.

## Real Fourier and ODFT readout

At an endpoint quadratic t²-2 cos(theta)t+1, let the residue be u+vt. Then

    Re = u+v cos(theta),   -Im = v sin(theta).

Those two real expressions provide the conventional negative-exponent DFT
pair without running complex arithmetic. Linear t-1/t+1 packets yield DC
and Nyquist. The ODFT endpoint instead uses N/2 quadratics with angles
(2k+1)π/N, factors of t^N+1. It has N real coordinates and no separate DC or
Nyquist packets. Both endpoints are representations of the same f.

## Evidence and reproduction

- `partition_results.json`: all 52 N=8 states and 320 directed edges, 2560
  exact basis checks; 12 arbitrary repartitions each at N=8 and N=16, exact
  Fourier readout and return to the input chart.
- `moving_results.json`: 56 exact basis transitions at N=8 through moving
  nodes, confluent packets, ODFT and FFT; exact endpoint/loop identities;
  explicit cross-packet collision rejection and all-source dependency test.
- `atlas.py`, `moving.py`: state and transition implementations.
- `run.py`, `test_moving.py`: executable certificates, with direct polynomial
  remainder and real cosine/sine endpoint oracles kept outside transport.

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 experiments/real_fourier_atlas/run.py \
  --out /tmp/real_fourier_atlas.json
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 experiments/real_fourier_atlas/test_moving.py \
  --out /tmp/real_fourier_moving.json
```

Copy both JSON files back immediately using the host selected by `m4host`.
The Mini's existing SymPy 1.13.3 performs exact arithmetic in real algebraic
number fields. No software installation is required.

## What is established and what remains open

An explicit, entirely real, history-independent state space supports arbitrary
admissible regrouping and changing polynomial charts. The test paths can turn,
return, cross internal root collisions while preserving jets, and reach either
FFT or ODFT. This is a constructive answer to a specified traversal question.

It is not a universal characterization of every real Fourier representation,
a novelty certificate, or a new fast FFT. The next structural question is
which useful arbitrary moves can be described using a small neighborhood of
packets, and what extra local state would make that possible without adding
independent signal degrees of freedom. The one-pair move already demonstrates
why a local phase label alone is insufficient in these residue coordinates.
