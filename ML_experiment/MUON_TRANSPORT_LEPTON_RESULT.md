# Transport-Muon and Lepton: first shape result

Date: 2026-08-29  
Status: mechanism implemented; 60-run paired screen and 8-run exact-iteration confirmation complete

## Result

Transport helps Muon, but it helps the smooth Bregman continuation of Muon
more consistently and by a larger mean acquisition margin.

The decisive paired result is:

| comparison | paired runs | acquisition-AUC wins | mean AUC delta | endpoint-acquisition wins | mean endpoint delta |
|---|---:|---:|---:|---:|---:|
| Transport-Muon over Muon | 12 | 10 | +0.000944 | 9 | +0.001217 |
| Lepton over Muon | 12 | 7 | -0.000032 | 5 | +0.000529 |
| Transport-Lepton over Muon | 12 | 11 | +0.002088 | 9 | +0.003394 |
| **Transport-Lepton over Lepton** | **12** | **12** | **+0.002119** | **10** | **+0.002865** |

Transport's mean AUC benefit is about `2.25x` larger in Lepton's Bregman dual
state than in Muon's hard-polar momentum state.

At the ten-update measurement grid, Transport-Lepton's median speedup to
untransported Lepton's final acquisition target is `1.0x`; the grid is too
coarse because several final targets are reached only near the budget edge.
An eight-run per-update follow-up on self-context pinwheel resolves that
ambiguity.  Transport-Lepton reaches `0.90` 11 and 17 updates earlier than
Lepton, and exactly 21 updates earlier than ordinary Muon in both seeds.  This
is a real iteration reduction, although not yet an iteration collapse.

## 1. The five mechanisms compared

All runs used paired initialization and minibatch order.

### Muon

```text
M_t = beta M_(t-1) + (1-beta) G_t,
D_t = NS(M_t) approximately polar(M_t).           (1)
```

The exact polar map replaces every supported singular value by one.

### Transport-Muon

```text
Mbar_t = T[r_(t-1) -> q_t](M_(t-1)),
M_t    = beta Mbar_t + (1-beta) G_t,
D_t    = NS(M_t).                                 (2)
```

Each complete parameter matrix is one cell.  `T` is Anchor's minimum
norm-preserving spherical rotation in the flattened matrix space.  `q_t` is
the spherical midpoint of the carried and current normalized gradient
signatures.  Muon's Newton--Schulz readout is otherwise unchanged.

### Lepton

Normalize each matrix gradient before inserting it into the discounted dual
state:

```text
Ghat_t = G_t / max(||G_t||_F,epsilon_0),
Y_t    = beta Y_(t-1) + (1-beta) Ghat_t.           (3)
```

For singular decomposition

```text
Y_t = U diag(sigma_i) V^T,
```

Lepton applies

```text
D_t = grad h_epsilon*(Y_t)
    = U diag(
        sigma_i / sqrt(sigma_i^2 + epsilon^2)
      ) V^T.                                      (4)
```

The dual potential is the smooth spectral function

```text
h_epsilon*(Y)
  = sum_i sqrt(sigma_i(Y)^2 + epsilon^2).          (5)
```

Its conjugate is the matrix-ball potential

```text
h_epsilon(D)
  = -epsilon sum_i sqrt(1-sigma_i(D)^2),
  ||D||_2 < 1,                                    (6)
```

up to additive constants and the rectangular null-space convention.  This is
a genuine smooth Bregman dual map on the interior of the spectral ball.

As `epsilon -> 0`, every nonzero response in (4) approaches one and Lepton
approaches exact Muon.  At finite `epsilon`, weak singular modes remain weak.
This is restraint encoded as geometry rather than multiplied onto the final
update.

### Transport-Lepton

Transport is applied to the dual state before the mirror readout:

```text
Ybar_t = T[r_(t-1) -> q_t](Y_(t-1)),
Y_t    = beta Ybar_t + (1-beta) Ghat_t,
D_t    = grad h_epsilon*(Y_t).                    (7)
```

This ordering is essential.  The historical request is re-expressed in the
current observed frame while it is still a dual object.  The Bregman map then
constructs the new primal displacement.

### Anchor

Standing Anchor remains the external mechanism reference.  It transports a
slow Euclidean request, reads it through the faster lead--lag state, applies a
rank-one departure contraction, and enforces a local relative-step
certificate.  It is not LR-matched to the matrix methods: its standing
ceiling remains `1.0`, versus `0.003` for the three RMS-matched matrix arms.
The purpose of including it is to preserve the shape target, not to declare a
tuning-free optimizer winner from this screen.

## 2. Experimental boundary

The screen crossed

- tasks: pinwheel, complex 3-D spiral, sparse sine 1-D;
- models: ordinary MLP and self-context;
- seeds: 0 and 1;
- width: 16;
- training budget: 200 updates;
- acquisition measurement: every 10 updates;
- batch size: 256;
- matrix learning rate: 0.003;
- Anchor ceiling: 1.0;
- Lepton smoothing: `epsilon=0.05`;
- momentum: 0.95 with Nesterov readout;
- hidden matrix routing: matrix optimizer;
- input/output matrices, biases, gains, and nonmatrix parameters: AdamW.

There were 60 complete runs and no numerical failures.

The sparse-sine problem remained largely unlearned by every arm at this width
and budget.  Its tiny positive deltas are consistency evidence only, not a
claim of solving that task.  The complex-spiral acquisition score also does
not equal held-out regression quality; this screen is about trajectory shape
and iterations, not a replacement for the full battery's winner rule.

## 3. Exact iteration follow-up

The strongest screen result, self-context pinwheel, was rerun with validation
acquisition measured after every update rather than every ten updates.

| seed | Muon to 0.80 | Transport-Muon | Lepton | Transport-Lepton |
|---:|---:|---:|---:|---:|
| 0 | 55 | 62 | 64 | **62** |
| 1 | 32 | **32** | 37 | 33 |

| seed | Muon to 0.90 | Transport-Muon | Lepton | Transport-Lepton |
|---:|---:|---:|---:|---:|
| 0 | 182 | 165 | 172 | **161** |
| 1 | 177 | 161 | 173 | **156** |

At the more meaningful `0.90` threshold:

- Transport-Lepton saves 11 updates versus Lepton in seed 0: `6.4%` fewer;
- it saves 17 updates in seed 1: `9.8%` fewer;
- it saves 21 updates versus Muon in both seeds: about `11.5%` and `11.9%`
  fewer;
- Transport-Muon also saves 17 and 16 updates versus Muon, confirming that
  transport itself is active rather than Lepton merely receiving a favorable
  smoothing constant.

Transport-Lepton's exact-grid AUC gains over Lepton are `+0.00675` and
`+0.00641`.  Endpoint acquisition rises from `0.91958` to `0.92819` and from
`0.92035` to `0.92973`.

The `0.80` threshold exposes a useful boundary: early in training, transport
can be neutral or slightly slower because it preserves historical frame
commitment.  Its advantage appears during the tighter approach, where the
untransported optimizer begins paying correction debt.  This is precisely
the regime the Anchor restraint hypothesis is intended to improve.

## 4. Pair-level result

Mean paired deltas relative to ordinary Muon:

| task | model | Transport-Muon AUC | Lepton AUC | Transport-Lepton AUC | Transport-Lepton endpoint |
|---|---|---:|---:|---:|---:|
| pinwheel | ordinary | +0.000471 | -0.001519 | +0.001499 | -0.002280 |
| pinwheel | self-context | +0.004943 | -0.001838 | **+0.004895** | **+0.012241** |
| complex spiral | ordinary | +0.000195 | +0.000773 | **+0.001221** | +0.000296 |
| complex spiral | self-context | -0.000526 | +0.002314 | **+0.004045** | **+0.006550** |
| sparse sine | ordinary | +0.000146 | -0.000075 | +0.000190 | +0.000115 |
| sparse sine | self-context | +0.000434 | +0.000156 | **+0.000676** | **+0.003438** |

Direct Transport-Lepton minus Lepton AUC deltas are positive in every pair:

```text
pinwheel ordinary       +0.003017
pinwheel self-context   +0.006733
complex ordinary        +0.000448
complex self-context    +0.001731
sparse ordinary         +0.000266
sparse self-context     +0.000521
```

The largest effect is the self-context pinwheel, where the network itself has
a moving internal context geometry.  The second-largest is the self-context
complex spiral.  That ordering supports, but does not prove, the hypothesis
that transport becomes more valuable when both optimizer state and model
state live in moving frames.

## 5. Why Bregman state benefits more from transport

Muon's hard polar map is a quotient:

```text
U diag(sigma_i) V^T -> U V^T.                    (8)
```

It destroys radial information in every supported singular mode.  Transport
can improve the singular subspaces supplied to that quotient, but much of the
history's graded confidence is discarded immediately afterward.

Lepton keeps a smooth singular response:

```text
sigma_i -> sigma_i / sqrt(sigma_i^2+epsilon^2).  (9)
```

Transported orientation and carried confidence therefore survive together.
The dual state says both *where* historical motion points and *how strongly*
each singular mode is supported.  The nonlinear mirror map turns that joint
state into a bounded primal action.

This gives a sharper version of the transport rule:

> Transport is most useful before a nonlinear primal readout, while the
> historical state still retains the information that the readout needs.

BFFT's Meyer acceleration follows the same ordering.  It carries exact
Bregman/constraint state forward, then reconstructs the next primal solve.  It
does not first collapse the state to a final primal displacement and attempt
to transport only that collapsed output.

## 6. Consequence for Anchor

Anchor currently has

```text
transported slow request -> fast lead--lag readout
                         -> Euclidean departure contraction
                         -> trust certificate.                  (10)
```

The Lepton result suggests a more coherent eventual form:

```text
transported dual request -> restraint-shaped mirror map
                         -> primal update
                         -> trust certificate.                  (11)
```

Formally, let `h_t` be the local restraint geometry and let the carried primal
request be `v_(t-1)`.  A moving-potential Anchor would use

```text
Ybar_t = T_t[grad h_(t-1)(v_(t-1))],
Y_t    = beta Ybar_t + (1-beta) g_t,
v_t    = grad h_t*(Y_t).                           (12)
```

The restraint is now the Hessian/shape of `h_t`, not a post-readout mask.
Where reversal evidence is certain, `h_t` contracts that mode.  Where the
restraint is uncertain, the potential relaxes toward the hard spectral ball
and permits momentum to price the gambit.

Equation (12) also explains why momentum need not be elementwise.  `Y_t` can
be a complete matrix covector; `h_t*` couples its singular modes; transport
acts before that coupled decision is converted into parameter motion.

## 7. What has and has not been learned

Established in this screen:

1. norm-preserving full-matrix transport is mechanically compatible with
   Muon;
2. it produces small positive mean acquisition changes at fixed LR;
3. the same transport applied to Lepton's dual state produces a larger mean
   effect;
4. Transport-Lepton beats Lepton on AUC in all 12 paired runs;
5. smooth spectral restraint can coexist with momentum without becoming
   elementwise;
6. the best location for transport is before the primal mirror readout.

Not established:

1. that any new arm beats tuned Muon over a broad task battery;
2. that the gains survive wider models, longer budgets, or more seeds;
3. that `epsilon=0.05` is optimal or transferable;
4. that the current flattened-matrix spherical transport is the correct
   connection for the Bregman metric;
5. that the present gains are large enough to matter outside an
   iteration-focused research setting;
6. that endpoint generalization improves whenever acquisition improves.

## 8. Next hard push

The next useful experiment is not a generic LR sweep.  It is a shape assay
with a mode-dependent `epsilon`:

```text
epsilon_i = epsilon_min
            + confidence_i * reversal_i * epsilon_span.        (13)
```

Use the negative eigenspaces of

```text
sym(polar(G_t)^T polar(Ybar_t)),
sym(polar(G_t) polar(Ybar_t)^T)                  (14)
```

to locate input/output action modes in which momentum reverses the current
decision.  High-confidence reversal increases `epsilon_i` and restrains that
mode.  Uncertainty keeps `epsilon_i` near `epsilon_min`, preserving the
momentum gambit.  This realizes the proposed deformable hidden manifold
without an elementwise variance estimate.

The measurement must remain iteration-based:

- time to the same sustained acquisition target;
- overshoot and later correction distance;
- whether restrained singular modes predict future reversal;
- whether transport gain increases with model-frame motion;
- final held-out quality as a non-regression constraint.

## Reproduce

Run the five arms:

```sh
python3 -m ML_experiment.run_benchmark \
  --out /tmp/muon_shape_assay \
  --tasks pinwheel,complex_spiral_3d,sparse_sine_1d \
  --variants ordinary_mlp,self_context \
  --widths 16 --seeds 2 --steps 200 --batch 256 --eval-every 10 \
  --optimizers anchor,muon,muon_transport,lepton,lepton_transport \
  --optimizer-lrs \
anchor=1.0,muon=0.003,muon_transport=0.003,lepton=0.003,lepton_transport=0.003
```

Summarize:

```sh
python3 -m ML_experiment.analyze_muon_shape_assay \
  /tmp/muon_shape_assay/results.json
```

## Conclusion

The user prediction survives the first paired test: Bregman Lepton benefits
more from transport than hard-polar Muon does.  The reason is not mystical.
Lepton has a graded dual state worth transporting; Muon immediately quotients
away much of that grade information.

The consequential lesson for Anchor is that restraint should become the
shape of a mirror map applied after transport.  Momentum then remains a
coupled matrix request, reversal can be restrained by mode, uncertainty can
locally reopen the ball, and the final trust certificate can remain as the
outer safety boundary.
