# A cost witness that does not assume an FFT route

The search needs lower bounds that survive arbitrary choices of intermediate
basis and physical wire order. The following simple witness applies to exact
linear constant-coefficient circuits. It is not a new claimed complexity theorem.

Let K be the field generated over Q by the circuit's constants, including the
Fourier constants. Pick any Q-linear functional ell: K -> Q with ell(1)=0.
Apply ell entrywise to the current matrix of coefficients of input samples in
all live values. Initially this projected matrix has rank zero.

- Addition, subtraction, rational multiplication, copying, and permutation cannot
  increase its row rank: projected rows undergo rational linear combinations.
- A scalar multiplication by a nonrational constant replaces or appends one
  coefficient row, and can increase projected rank by at most one.

Consequently every such circuit computing F requires at least rank(ell(F))
nonrational scalar multiplications. This argument permits extra live wires and
arbitrary intermediate bases. It does not apply unchanged to nonlinear circuits,
and it is not a bound on NEON instructions, total arithmetic, memory traffic,
or latency. Rational multiplication and all placement are deliberately free in
this particular bound.

For power-of-two N, use c=cos(2*pi/N), degree d=N/4, with

    T_d(c)=0.

The polynomial 2*T_d(y/2) is monic Eisenstein at 2 for d>1, proving the stated
number-field degree. The code checks this algebraic condition. Express the
ordinary unnormalized real DFT as

    F=A0+c*A1+...+c^(d-1)*A_(d-1),

with rational matrices Ar. Choosing ell(c^r) gives an explicit rational matrix
whose nonzero minors certify a lower bound. The functional can be extended
Q-linearly to a larger constant field if the circuit uses other constants.

The saved witnesses establish:

| N | Certified lower bound on nonrational scalar multiplications |
|---:|---:|
| 4 | 0 |
| 8 | 2 |
| 16 | 8 |
| 32 | 22 |
| 64 | 52 |

These are lower bounds, not claims of attainable complete algorithms at those
counts. The eight-sample exact witness attains its bound of two, while still
paying for its additions and signed exchanges.

`certificates/projection_rank.json` records the functional weights and the row/
column indices of a nonzero minor modulo the prime 1,000,003. Modular elimination
is exact integer arithmetic. A nonzero determinant modulo this prime proves that
the rational determinant is nonzero. Modular rank can be LOWER than rational
rank, which is why the result is stated as a bound, not an exact rank assertion.

The first direct rational-elimination attempt became slow at N=64. It was stopped
and replaced with this smaller exact witness; no floating rank estimate was used.

Reproduce with:

    /Users/ultimussecundai/.local/bin/m4build -- python3 experiments/fourier_free_walk/projection_rank.py --sizes 4,8,16,32,64 --out /tmp/fourier_projection_rank.json

Copy the /tmp JSON back immediately with m4host. Tests reconstruct the exact
small Fourier matrices from the field coefficients, verify a witness determinant,
and check explicitly that modular rank is only a lower bound.

For a partial linear walk, max(0, rank(ell(F))-rank(ell(B_current))) is therefore
an admissible lower bound on remaining nonrational scalar multiplications. It
can guide a future shortest-path search without selecting a radix grammar.
It does not detect misplaced values; the separate physical-contact reachability
check addresses a different obligation. The present implementation certifies
these endpoint bounds; it does not yet run Dijkstra over continuous bases.
