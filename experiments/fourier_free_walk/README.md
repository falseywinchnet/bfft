# Walks through unrestricted real Fourier coordinates

2026-09-05. This experiment reopens the intermediate representation and the
pair-interaction sequence, following the user's correction that the quartic
prototype preserved Bruun's paths and attached destination placement.

The current outcome is a verified search apparatus and several small executable
walks. It is not a newly established FFT family, a speed claim, or an exhaustive
search of continuous real-coordinate space. No production BFFT kernel is called
by a synthesized walk or used as its factorization template.

## State and actual endpoint

A state has N real values on N physical wires. During offline synthesis the
meaning of those values is a real basis matrix B: v=B*x. Any invertible real
basis is admissible in the model. Elementary two-wire GL(2,R) changes generate
GL(N,R), so this state model is not restricted to roots, real/imaginary partners,
radix stages, orthogonal states, fixed frequency locations, or Bruun packets.
The finite searches below exercise subsets of that continuous space.

Only the planner holds matrices. The executor holds the N real values and,
for each cell, reads two physical wires and writes those same two wires. The
final physical order is

    DC, Re X[1], Im X[1], ..., Re X[N/2-1], Im X[N/2-1], Nyquist.

These are the ordinary negative-exponent DFT's independent real components.
There is no final gather or scatter. Standalone signed exchanges inside a walk
are still counted as transport cells; their presence is never called free.
An input copy in the generic Python interpreter just initializes the work state.
`exact8.transform_inplace` also permits direct mutation of the supplied array.

## Degrees of freedom actually exercised

1. **Arbitrary pair selection and current-basis angles.** The sparse search visits
   all row pairs and generates every distinct zero-making direction furnished by
   that pair's current rows. It does not use a table of Fourier twiddles. It also
   offers independently selected, oblique pairs of such directions. The first
   greedy screen nevertheless selected only orthogonal cells: making a freedom
   admissible did not mean the objective would exercise it.
2. **Output orientation of each existing cell.** A separate finite state search
   explores keeping or exchanging the two computed outputs. Future-contact cones
   prune impossible assignments. A solution can absorb placement into the same
   two stores; a failed search is not repaired by renaming the output. Search is
   exact when its frontier cap is not reached, and reports pruning otherwise.
3. **Changing interaction topology and angles together.** On three wires, proper
   orthogonal products can change among six Euler charts. Two cells may expand
   to three before later fusing or cancelling. Disjoint cells can commute.
4. **General, nonorthogonal three-wire chart changes.** The direct rank-one
   construction below factors a generic GL(3) product into a different three-cell
   arrangement. It permits oblique frames and reflections, rather than merely
   varying angles in a fixed rotation network. Degenerate charts are explicitly
   declined; this is a chart admission rule, not an impossibility claim.
5. **An independent scale for every intermediate wire value.** Endpoint scales
   are fixed to one. Admissible scale equations make selected coefficients ±1;
   all other coefficients and every cell's reads/writes remain in the ledger.
6. **Continuous excursions with freely chosen parameters.** An arbitrary-angle
   rotation or arbitrary-strength shear can recruit any third wire. Its inverse
   closes the excursion, with intervening cells refactored before continuing.
   The added work is charged. The 15-run screen accepted 711 such excursions,
   along with 5,003 general-frame moves; saved endpoints remain valid. Their
   parameters are not selected from a Fourier twiddle catalog.
7. **Nonmonotone walks.** The topology search accepts some temporarily more costly
   moves. The joint search applies nonorthogonal chart changes and intermediate
   scale changes in the same loop, rather than optimizing them only in isolation.

## Why three-wire GL changes are legitimate

For a three-wire transfer, partition its matrix as

    M = [ m00  v ]
        [  u   L ].

Where d = v adj(L) u is nonzero, set a=det(L)/d. Then

    K = L - a*u*v

has determinant zero. Choose a nonzero pivot and factor K=w*z. Consequently

    M = diag(1,[u,w]) * diag([[m00,1],[1,a]],1) * diag(1,[v;z]).

The three factors act on wire pairs (1,2), (0,1), (1,2), in reverse execution
order. Permuting the three physical wire labels supplies other charts. This
changes intermediate dependencies and coefficients while preserving M itself.
No signal-dependent fit, FFT, or dense numerical inverse constructs the step.
The determinant/rank-one construction and all nine resulting matrix entries
are symbolically certified in `certificate.py`.

The same certificate verifies a nonorthogonal shear turnover. In execution
order A(x), B(y), A(z), where A=I+xE01 and B=I+yE12, a chart with x+z != 0 gives

    B(zy/(x+z)), A(x+z), B(xy/(x+z)).

That formula is retained as a further valid move; the current joint search uses
the more general rank-one GL(3) move, not this separate shear specialization.

## Frame freedom and the cycle obstruction

Assign a positive scale exp(lambda_v) to each intermediate wire-value vertex.
A scalar coefficient a on an edge becomes

    a_new = a * exp(lambda_out - lambda_in).

Making its magnitude one imposes a difference equation on its two vertex
potentials. A weighted union-find admits compatible equations and detects
conflicting cycles. Boundary potentials are fixed, so no normalization cost is
silently moved to the input or output. Local gauge changes telescope exactly.

For a dense two-by-two cell [[a,b],[c,d]], the ratio a*d/(b*c) is unchanged by
independent row/column scales. More generally, products with alternating
orientation around closed dependency cycles constrain simultaneous unitization.
This is the concrete obstruction to independently making every multiplication
cheap. Topology changes alter the graph in which those constraints are expressed.
The joint search therefore investigates topology and frame choices together.

The admission orders are sampled; weighted union-find is not a proof of globally
optimal gauge selection. A numerical scale cap rejects extreme proposed frames.
Numerical simplification can remove cells within 2e-12 of identity; the complete
retained operator is then checked against the endpoint. All whole-transform
matrix checks use the actual retained coefficients rather
than silently replacing near-zero entries with zeros.

## Observed results

- The initial 30 sparse-support searches all reduced the Fourier matrix to a
  signed monomial form. Every one failed its output-orientation feasibility
  check, with no frontier pruning. They are rejected as unplaced paths.
- An ordered QR/Givens control establishes a valid endpoint, but mostly remains
  quadratic under the tested rewrites. Its records are in `qr_turnovers/`.
- A second control starts from a sparse walk and explicitly pays for its residual
  permutation with signed quarter-turn cells. Those cells participate in the
  algebraic rewrites; they are not exempt from cost or called absorption.
- In that search, one N=6 path shrank from 12 cells to 8, with signed exchange
  cells falling from 5 to 1. A saved N=8 path shrank from 15 cells to 13, with
  exchange cells falling from 5 to 2. Placement work was reduced, not abolished.
- The joint nonorthogonal/frame search also improves paths, but has not produced
  a competitive scaling law. For example, N=16 seed 0 decreases from 41 to 40
  cells and retains 11 signed exchange cells. Larger orthogonal searches at
  N=24 and N=32 did not reduce their best starting cell counts in this screen.

All saved completed walks finish in the specified physical order. The recorded
matrix errors are at roughly 1e-16 to 2e-14 for the displayed sizes. This alone
is not an exactness proof: exact certificates were additionally recovered for
N=4, N=6 and N=8, and the actual hand-simplified `exact8.py` is tested symbolically on
all eight input basis vectors.

The exact eight-sample program uses **20 additions/subtractions, two general
multiplications, and three signed exchange cells**, with no final placement.
Its arithmetic count is a familiar small-RFFT count; it is a verified witness
of the search machinery, not a novelty claim. Earlier JSON operation counts
are conservative per-cell estimates and miss some shared products and factoring
of equal coefficients within a row. They are not NEON instruction counts or
benchmark measurements. Every recorded cell still counts a pair read/write.

## Validation and use

Run the test suite on the M4 Mini:

    /Users/ultimussecundai/.local/bin/m4build -- python3 -m unittest discover -s experiments/fourier_free_walk -p 'test_*.py' -v

The suite checks orientation reachability against full enumeration, 1,200 random
Euler-chart reconstructions plus degenerate charts, generic oblique GL charts,
commuting fusion, cycle consistency, complete matrix reconstruction, and the
actual exact8 program, continuous mixer excursions, and exact projection-rank
witnesses. Each saved full matrix check covers every input basis
vector, not only a selected signal.

Run all main screens and immediately copy their artifacts back with:

    experiments/fourier_free_walk/run.sh

Results from this reproduction go to `reproduced/`; the initial discovery
records remain in `initial/`, `sparse_turnovers/`, `gauges/`, `joint/`, and `continuous/`.
For the earlier QR control, use `run_turnover.py --start-kind qr` through m4build.

Run the exact standalone eight-sample witness:

    python3 experiments/fourier_free_walk/exact8.py 1 2 3 4 5 6 7 8

It returns

    [36, -4, 9.656854..., -4, 4, -4, 1.656854..., -4].

See also [the basis-independent projection-rank certificates](PROJECTION_RANK.md).

## Scope and related work

This is a broader search, not an exhaustive exploration of all possible FFT
algorithms. The explored sizes are finite; oblique annihilator candidates use
an explicit six-direction shortlist; the rewrite searches are stochastic and
local; the cost weights are heuristic. No O(N log N) bound is proved for the
ordered synthesized family, and no new native speed result is claimed.

Givens-factor search and scaling are established ideas. Relevant primary work:

- Frerix and Bruna, *Approximating Orthogonal Matrices with Effective Givens
  Factorization* (2019): https://proceedings.mlr.press/v97/frerix19a.html
- Johnson and Frigo, *A Modified Split-Radix FFT With Fewer Arithmetic Operations*
  (2007): https://math.mit.edu/~stevenj/papers/JohnsonFr07.pdf
- SPIRAL's explicit factorization-rule search:
  https://www.spiral.net/doc/usermanual/infrastructure/breakdown.html

These precedents are not a reason to restrict the admissible walk to their
particular factorizations. They do prevent calling the underlying rotation,
scaling, or synthesis tools new inventions. The open target here is a useful
joint route through basis, topology, and frame freedom that finishes ordered
without retaining the current transport overhead.
