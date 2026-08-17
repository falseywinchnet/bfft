# Relational witness spiral experiment

This project tests the surviving hypothesis from the living-filter experiments:
an operator may need agreement among several views of a state, rather than a
geometry manufactured from one undifferentiated state.

The completed findings and the rejection criteria are recorded in
`RESULTS.md`. Both proposed witness mechanisms fail contiguous continuation.

`run_hypersphere_atlas.py` is the structure-agnostic transport follow-up. Its
charts and spherical geometry are learned only from hidden activations, and its
full dense residual prevents loss of arbitrary linear structure. It produces
the requested dataset, decision-region, training, and holdout-survival plots.

`run_jet_transport.py` removes the atlas correction's bounded-range defect and
tests derivative learning directly. Its learned nonlinear path is an
input-conditioned low-rank linear operator whose magnitude is unsaturated.
Observed input secants train an explicit representation connection; represented
secants train an explicit operator connection; and observed triples train
composition. The shuffled-secant control has identical capacity but destroys
the proposed relational signal.

In `jet_transport` and `jet_shuffled`, the connection is not merely an
auxiliary predictor: three-point quadrature integrates it to construct both the
representation and the input-conditioned operator coefficients used by the
forward pass. The completed result is negative for contiguous continuation and
is documented in `RESULTS.md`; the connection carrier changes the immediate
frontier, but nearest-neighbor secants do not select the correct continuation.

`run_probabilistic_jet.py` keeps five independently initialized, bootstrap-
trained integrated connections. Twenty percent of the observed relation
triples are excluded from gradient fitting and used as predictive evidence for
checkpoint selection and posterior weighting. Uniform and permuted-posterior
mixtures are matched controls. A test-only oracle reports whether any generated
hypothesis continued correctly, but is never used for training or selection.
`analyze_probabilistic_jet.py` derives the evidence-MAP diagnostic and plots
held-relation likelihood against immediate-frontier accuracy from saved runs.

`run_associative_shells.py` removes explicit pairs, triples, hypotheses, and
scene encoders. Each context observation makes one rank-one write into a shared
key/value memory. Repeated nonlinear application of that single operator
creates implicit pair, triple, and higher Newton shells. Episodic context and
query sets are disjoint, so a query label is never present in the memory used
to predict it. Its matched shuffled control permutes complete values against
keys, preserving both marginals while destroying coherent observation writes.

`run_projective_quotient.py` learns non-positional structural labels from the
inner scene. Its covariance operator moves learned subspaces; normalized
projection energy assigns context observations and queries to those roles.
Each role owns an affine law fitted from its assigned observations, and there
is no direct coordinate-to-class head. Diagonal-covariance, shuffled-placement,
positional-cell, and single-global-affine controls isolate the mechanism.

`run_projective_transition.py` makes affine structure act between those roles.
Randomly oriented half-space occlusions inside the observed scene provide
generic continuation episodes. Context adjacency induces a directed Markov
operator on projective labels, and a query must anchor to the visible scene and
traverse learned powers of that operator. Direct-role, identity-transition,
and shuffled-edge controls distinguish labeling from transport.

`run_banach_eikonal_sieve.py` treats finite views only as quadrature particles
on a continuous operator-valued circle. A tanh-bounded periodic potential from
the observed context transports their density by an analytic eikonal velocity.
The induced operator acts inside every hidden layer. Frozen density, direct
potential reweighting, and broken differential transport are matched controls.

The data geometry is the double spiral from the M-layer notebook, but none of
that notebook's models, optimizers, or training code is used. The full phase
range is split at 50% and 30% training frontiers. Twenty consecutive held-out
bins are evaluated, and the primary metric is contiguous survival from the
frontier (accuracy at least 80%), not aggregate held-out accuracy.

Four tiny embed/operator/GELU/operator/unembed models are compared:

- `reference_linear`: ordinary learned dense operators.
- `witness_static`: fixed independent views, averaged uniformly.
- `witness_marginal`: a shared response rule sees marginal view statistics.
- `witness_relational`: the same response capacity sees pair and triple view
  products and generates a sample-specific mixture of the fixed views.

The candidate has no learned coordinate-specific filter bank. Its learned
object is one response rule shared across all output coordinates.

`run_learned_subspace.py` is the less biased follow-up. Each substitute layer
strictly contains an ordinary dense linear map. Its extra branches are learned
low-rank maps, not fixed geometric directions. `subspace_static` is a purely
linear factorized control; `subspace_marginal` conditions branch weights on
individual branch energy; and `subspace_gram` conditions them on the complete
mutual Gram matrix. Run it with the M4's existing CPU Torch installation:

```sh
/Users/ultimussecundai/.local/bin/m4build -- env \
  PYTHONPATH=/Users/joshuahkuttenkuler/Library/Python/3.9/lib/python/site-packages \
  python3 experiments/relational_witness_spiral/run_learned_subspace.py
```

Run the complete experiment on the M4 Mini from the repository root:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 experiments/relational_witness_spiral/run_experiment.py
```

Run the tests on the M4 Mini with:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 experiments/relational_witness_spiral/test_relational_witness_spiral.py
```
