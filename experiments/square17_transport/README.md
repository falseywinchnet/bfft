# Seventeen squares as a BFFT configuration transport

This experiment applies the repository's Night Vision, segmenting-v3,
continuous-eikonal, chip-unrelaxation, and Bruun/DIP ideas to packing
seventeen unit squares in the smallest possible square.  It does not implement
the construction or certification method from the motivating post.

## Correct numerical statement

Mira's `4.468292` is a **strict lower bound**, not a packing:

```text
4.468292 < s(17) <= 4.675530093604551...
```

The upper endpoint is the verified construction in `reference_chart.py`.
Therefore a run exactly at `4.468292` cannot emit a valid packing.  The
computational goal is to lower the upper endpoint while respecting the lower
one.

Clearance is not the evolved objective.  A valid terminal chart will
necessarily have positive clearance at inactive contacts, so clearance remains
useful as an independent terminal SAT audit.  The error in the earlier
experiments was using clearance or Euclidean pose residual as the generating
energy.

## Intended BFFT state

Let `P_L` be the pose space of one unit square on the floor plan.  The lifted
state is a positive amplitude over the full seventeen-body configuration
space

```text
X_L = P_L x ... x P_L    (17 factors).
```

Each discrete complete configuration is an orthogonal coordinate ket.
Distinct configurations are consequently all at distance `sqrt(2)` before
relaxation.  This is the equidistant lift; it is not a collection of Euclidean
trial charts placed at the vertices of a simplex.

Evolution uses physical polygon intersection area

```text
V(q) = sum_{i<j} area(square(q_i) intersect square(q_j))
     + wall area.
```

The kinetic step is a positive floor-plan transport applied along every
particle axis.  The potential-sandwiched imaginary-time operator is positive,
and its normalized action is a Birkhoff/Banach contraction in Hilbert's
projective metric.  Each annealing stage is relaxed to its Perron fixed point.
Only after this relaxation is one joint configuration measured into Euclidean
centers and square phases.  That chart is never fed back into the transport.

The full derivation is in `TRANSPORT_FORMULATION.md`.

## Exact controls now implemented

### Labelled binary product

`configuration_transport.py` carries all `2^17 = 131072` states of a
two-pose-per-square alphabet.  At side `4.468292` it measured:

```text
observed projective contraction ratio    0.363291
discrete global overlap-area minimum     4.7195767841
measured overlap-area energy             4.7195767841
terminal state equals enumerated minimum yes
```

The terminal chart overlaps badly because two generic poses per square are
not a useful basis.  The result nevertheless verifies that the contraction
reaches the global minimum of its complete product space, not a Euclidean
basin.

### Permutation-quotiented occupation basis

The labelled tensor product contains `17!` physically redundant permutations.
`occupation_transport.py` removes them.  A state is a 17-occupied subset of
`M` floor-plan pose modes.  Kinetic transport is the Johnson-graph exchange
walk; its discounted resolvent is evaluated by the literal Banach iteration

```text
y[n+1] = b + gamma P y[n],    gamma < 1.
```

For `M = 22`, the exact Hilbert space has only

```text
binomial(22, 17) = 26334
```

eigenbasis configurations, so no tensor approximation is needed.

The reference-control alphabet contains the 17 verified poses plus five
distractors.  The transport finds the global zero-area subset:

```text
projective contraction ratio             0.245410
outer fixed-point residual                5.80e-13
inner Banach residual                     3.94e-12
physical overlap-area energy              7.55e-15
terminal SAT overlap residual             6.38e-16
selected modes                            0..16
```

The complete record and rendering are
`results/occupation_reference_control.json` and
`results/occupation_reference_control.svg`.

With all reference modes removed, eight deterministic 22-mode floor-plan
bases were tested at the Mira lower-bound side.  Every transport measured the
enumerated global minimum of its basis.  The best was seed zero:

```text
projective contraction ratio             0.254987
global/measured overlap-area energy       4.9292426111
outer fixed-point residual                4.36e-13
inner Banach residual                     3.95e-12
terminal SAT overlap residual             2.24845
```

This is a negative representation result, not an optimization failure and not
a packing bound: a source-free alphabet of 22 point poses is far too sparse.
Its record is `results/occupation_lower_bound_seed0.json`.

### Bruun/DIP packet chart with exact GDL transport

`spectral_pose_transport.py` gives every persistent square identity a
normalized Bruun/DIP--Zak packet chart of simultaneous `(x, y, theta)`
displacements.  With 16 packets per identity the implicit equidistant basis
has

```text
16^17 = 295147905179352825856
```

complete configurations.  Physical polygon-overlap area factors pairwise on
the packet identities.  The measured interaction graph has 29 edges and
minimum-fill induced treewidth three at the verified chart.

`exact_packet_transport.py` applies the generalized distributive law to that
factor graph.  It obtains the exact global packet minimum with a largest table
of only `16^4 = 65536` energies.  A single backward pass emits the achieving
packet for every identity.  This is the zero-temperature endpoint of the
positive transport; there is no SVD truncation, Euclidean search, or clearance
objective.

The reference calibration selected packet zero for every square:

```text
side                                      4.675530093604551
implicit configurations                   295147905179352825856
exact/physical area energy                7.99e-15
terminal SAT overlap residual             6.38e-16
induced treewidth                         3
maximum exact table entries               65536
```

The authoritative control is
`results/exact_packet_reference_control.json`.

At side `4.67`, 576 exact packet charts varied DIP rung, U(1) identity phase,
translation scale, and rotation scale.  The best one had physical area
`3.27283e-4` and SAT residual `0.0166438`.  A fixed coarse-to-fine packet
preimage lowered these to `7.97881e-5` and `0.00877099`, but did not emit a
packing.  Transporting the independent feasible side-`4.8` topology down to
`4.67` was much worse: best area `0.0644684`, SAT residual `0.232863`.

The frontier test used 576 further exact charts at side `4.675500`, only
`3.00936e-5` below the verified upper construction.  Its best first pass was:

```text
physical area energy                      9.24464e-9
terminal SAT overlap residual             9.40570e-5
worst penetration                         4.63144e-5
```

`frontier_preimage.py` then ran 200 deterministic exact GDL restriction rungs.
The final measurement was still infeasible:

```text
physical area energy                      2.39644e-9
terminal SAT overlap residual             4.83255e-5
worst penetration                         2.00597e-5
```

The complete trace is
`results/exact_packet_frontier_46755.json`.  It is a negative result, not a
new upper bound.  The pinned penetration is consistent with the independent
nonnegative local shrink stress of the reference contact chart.

## Compression result

`tensor_transport.py` and `spectral_pose_transport.py` also tested a
conventional matrix-product state over the
labelled `17^17 = 827240261886336764177` product basis.  Rank 8 through 32
discarded 37% to 86% of the squared Schmidt weight at individual gates and
failed even the reference control.  The identity-specific spectral train also
discarded 26% to 65% at individual gates.  Both are retained as rejected
compression ablations.  The exact low-treewidth GDL path replaces them.

## Earlier rejected ablations

- `floorplan_banach.py` has a valid 17-scalar inner contraction, but alternates
  it with Euclidean polygon-gradient descent.  It therefore remains a local
  optimizer and is not the proposed method.
- `lifted_equilibrium.py` uses Gibbs SAT branches, posterior lanes,
  temperature, and Euclidean terminal optimization.  Those mechanisms violate
  the hard first-arrival and identity-preserving transport learned in v3 and
  Night Vision.
- `chip_transport.py`, `global_transport_search.py`, `topology_search.py`, and
  `stress_cut_search.py` are historical local/topological controls.
- The feasible side-`4.8` chart in `results/lifted_480_measured.json` remains a
  valid upper construction, but it is not evidence for the desired algorithm.

The known `4.675530...` chart has 39 active incidences, physical active rank
36, and a nonnegative local shrink stress with equilibrium residual
`7.56e-16`.  This is only a first-order local Farkas certificate; it is not a
global lower-bound proof.

## What this establishes and what remains

The experiment now separates three questions:

1. The exact positive configuration transport reaches global minima in its
   finite basis.
2. Bruun/DIP packet pose capacity can be solved exactly because the physical
   factor graph has low treewidth.
3. The known contact topology remains blocked below `4.675530...`, and the
   tested independent `4.8` topology does not approach it.

The missing object is therefore not a better local minimizer or more packet
resolution around these two charts.  It is a source-free continuous initial
chart whose achieving causal structure can change the contact topology before
the exact GDL restriction.  That chart must preserve persistent square
identity and the first-arrival inverse; it cannot be a population of decoded
Euclidean candidates.

No decoded clearance, candidate placement, or Euclidean improvement signal is
allowed back into the solve.

## Reproduce on the M4 Mini

From the repository root:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  /Users/joshuahkuttenkuler/Developer/CodexBuilds/.venv-bfft/bin/python \
  experiments/square17_transport/test_square17_transport.py

/Users/ultimussecundai/.local/bin/m4build -- \
  /Users/joshuahkuttenkuler/Developer/CodexBuilds/.venv-bfft/bin/python \
  experiments/square17_transport/occupation_transport.py \
  --alphabet reference_control \
  --output experiments/square17_transport/results/occupation_reference_control.json \
  --svg experiments/square17_transport/results/occupation_reference_control.svg

/Users/ultimussecundai/.local/bin/m4build -- \
  /Users/joshuahkuttenkuler/Developer/CodexBuilds/.venv-bfft/bin/python \
  experiments/square17_transport/occupation_transport.py \
  --side 4.468292 \
  --alphabet low_discrepancy \
  --seed 0 \
  --output experiments/square17_transport/results/occupation_lower_bound_seed0.json \
  --svg experiments/square17_transport/results/occupation_lower_bound_seed0.svg

/Users/ultimussecundai/.local/bin/m4build -- \
  /Users/joshuahkuttenkuler/Developer/CodexBuilds/.venv-bfft/bin/python \
  experiments/square17_transport/exact_packet_transport.py \
  --side 4.675530093604551 \
  --packets 16 \
  --dip-level 2 \
  --translation-radius 0.06 \
  --phase-radius 0.06 \
  --output experiments/square17_transport/results/exact_packet_reference_control.json \
  --svg experiments/square17_transport/results/exact_packet_reference_control.svg

/Users/ultimussecundai/.local/bin/m4build -- \
  /Users/joshuahkuttenkuler/Developer/CodexBuilds/.venv-bfft/bin/python \
  experiments/square17_transport/frontier_preimage.py \
  --side 4.6755 \
  --fine-rungs 160 \
  --output experiments/square17_transport/results/exact_packet_frontier_46755.json \
  --svg experiments/square17_transport/results/exact_packet_frontier_46755.svg
```

The MacBook checkout remains authoritative.  Generated Mini artifacts must be
copied back immediately after the run.
