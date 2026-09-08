# Projected-signature transport restraint for SGD

This directory asks one narrow question: can the state-transfer mechanism
behind BFFT's fused Meyer descent accelerate plain Euclidean SGD while every
instantaneous update remains a restraint of the current gradient?

## The operator under test

`SignatureTransportSGD` carries one bounded vector per local parameter cell
(a weight-matrix row, or a whole vector parameter). It uses the reduced
Split-Bregman state algebra from `experiments/cartoon_state_algebra.py`:

```text
q_k     = project(t_k, unit_ball)
t_{k+1} = normalize(g_{k+1}) + q_k
```

`q_k` is only an observer. The optimizer subtracts from the current gradient
a fraction of its component along the departure between the current request
and the carried signature. It never adds `q_k` to the update, never carries
a velocity, and asserts `||g_effective|| <= ||g||`.

`SIG-scalar` applies the exact same total norm reduction uniformly.
`SIG-shape` applies it only to the incompatible component. Their comparison
therefore separates shape transport from ordinary learning-rate damping.

The earlier `ResponseTransportSGD` same-batch secant arm is retained as a
negative result. Its response-axis projection was neutral on a deterministic
Rosenbrock oracle and should not be tuned further.

## Run

```sh
python -m pytest -q experiments/sgd_transport_restraint/test_optimizer.py
```

Run the experiment:

```sh
PYTHONPATH=. python experiments/sgd_transport_restraint/run_signature_oracle.py \
  --dimensions 16 --steps 20000 --lr 0.002

PYTHONPATH=. python experiments/sgd_transport_restraint/run_signature_mlp.py \
  --condition 10000 --steps 3000 --lr 0.03

PYTHONPATH=. python experiments/sgd_transport_restraint/run_anchor_mlp.py \
  --condition 10000 --steps 500 --lr 1.0 --trust-radius 0.02
```

See `ANALYSIS.md` for the closed-form geometry and `RESULTS.md` for the fixed-
setting iteration measurements. `ANCHOR.md` derives the dynamic LR controller
and records the contrast screen that preceded the full optimizer battery.
