# DIP2: decimation on the Bruun phase axis

Status: exact research reference, not a production kernel. Companion code:
`experiments/dip2_phase_axis.py`; regression:
`tests/dip2_phase_axis_test.py`.

## What changed at the core

The shipped DIP is a dyadic Zak walk. At level `e`, its packet is a time comb
with one folded residue row `d`; a subtree ends on the frequency comb
`{j*e +/- d}`. Its present boundary policy makes it look like half DIF and
half DIT, but changing that policy or fusing its cells does not create a new
walk.

DIP2 discards that packet invariant. Start with the real polynomial

    P(z) = sum_n x[n] z^n,             z = exp(-i theta),

and use Bruun's own real phase coordinate

    t = z + z^-1 = 2 cos(theta).

Pair the time coefficients `n` and `N-n`. There are unique real polynomials
`A` and `B` such that, on the Nth roots,

    P(z) = A(t) + (z - z^-1) B(t),

with

    A(t) = x[0] + x[N/2] T_(N/2)(t/2)
           + sum_(n=1)^(N/2-1) (x[n]+x[N-n]) T_n(t/2),

    B(t) = 1/2 sum_(n=1)^(N/2-1) (x[n]-x[N-n]) U_(n-1)(t/2).

Therefore at `theta_k = 2*pi*k/N`, entirely in real arithmetic,

    X[k] = A(2 cos(theta_k)) - 2 i sin(theta_k) B(2 cos(theta_k)).

The public complex pair is formed only at egress. `(A(t_k), B(t_k))` is a
Bruun residue coordinate: the quadratic
`z^2 - 2 cos(theta_k) z + 1` becomes the linear factor `t-t_k`.

The DIP2 walk is a balanced subproduct/remainder tree over the ordered phase
nodes `t_0, ..., t_(N/2)`. Every node owns a consecutive interval of k, and
every child is obtained by reducing the two real parent polynomials modulo
the product for its half interval. Its depth-first leaf order is already
`0,1,...,N/2`. In contrast to the old DIP, every subtree is a contiguous
frequency band rather than a pair of combs. Partial-band egress is native.

## What the executable probe establishes

- The stable Chebyshev-series oracle agrees with `numpy.fft.rfft` through
  roundoff (tested through N=1024).
- The literal real remainder tree agrees for N<=32. It is deliberately
  refused above 32 because monomial phase polynomials become catastrophically
  ill-conditioned; pretending otherwise would conceal the main problem.
- The tree has natural leaf order and contiguous-bin descendants at every
  node.
- The consecutive-node product polynomials are dense (some internal moduli
  have coefficient density 1.0).
- Every q-bin interval block of the Chebyshev evaluation matrix has row rank
  q. A phase twist that only rescales/rephases rows or columns preserves that
  rank. Consequently an exact independent q-bin packet cannot be compressed
  to a fixed two-real-number Bruun cell.

That last point is the current obstruction. The new ordering is not free:
the old comb ancestry is precisely what makes its factors sparse. With naive
long division the phase-interval tree is quadratic. Generic fast multipoint
evaluation is possible, but obtains its speed from fast polynomial products;
using an FFT inside DIP2 would not constitute a new FFT walk. On these nodes,
the two transforms are also recognizable as a DCT-I and DST-I pair. Merely
calling existing fast DCT/DST machinery would recover a known FFT
factorization and would not preserve the interval-remainder walk as a new
sparse core.

## What would make DIP2 succeed

The remaining target is precise: find a stable phase-local basis in which a
consecutive Chebyshev interval reduction has exact structured complexity
`O(q log q)` or better and composes between parent and children without a
global transpose. A diagonal phase twist is insufficient; the basis change
must be wider than a single `(a,b)` residue yet structured enough that the
total work remains `O(N log N)`.

Useful next probes are:

1. Express parent-to-child reductions in centered Chebyshev bases and search
   for exact Toeplitz/Hankel, displacement-rank, or butterfly structure.
2. Determine whether paired sibling reductions share enough work to reduce a
   dense pair to a sparse-plus-low-rank map.
3. Benchmark pruned phase intervals only after such a structure is exact;
   approximate low-rank Fourier butterflies are interesting, but they are a
   different product from an exact Bruun FFT.

## The separate conjugate-pair walk

`experiments/conjugate_pair_split_walk.py` preserves the complex walk explored
alongside DIP2. It is exact, has one `N/2` child and two `N/4` children, and
uses the `4n+1` / `4n-1` conjugate twiddle pair. The `4n-1` rotation changes
the leaves-up ingest permutation: at N=1024 its total adjacent-index travel is
about 0.851 times ordinary bit reversal.

That is a real resorting difference, but it is not presently a novelty claim.
The recurrence is the standard conjugate-pair split-radix FFT; published
depth-first implementations explicitly discuss its complicated input
scrambling. What may still be new is a better scheduling/layout consequence
inside this library, not the factorization itself. Keep it separate from
DIP2, and require a graph or transport result beyond the known conjugate-pair
family before presenting it as a novel FFT walk.
