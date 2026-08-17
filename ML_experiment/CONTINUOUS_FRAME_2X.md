# Continuous frame flow: doubled-capacity matched benchmark

## Experiment

The complete 23-problem battery compares four plotted objects in a fixed order:

1. ground truth;
2. an ordinary encode-expand-LELU-contract-decode MLP;
3. self-context;
4. self-context with continuous full-space frame flow, trained with AdamW.

The three learned models are exactly parameter-matched within every task. Width
38 gives 8,465–9,074 trainable parameters depending on task I/O, approximately
twice the width-24 parameter budget. Every run uses 500 AdamW steps, batch 256,
learning rate `3e-3`, two paired seeds, and CPU Torch on the M4 Mini. The run
contains 138 fits and 69 seed-0 fitted-function probes.

## Aggregate result

| Model | Validation | Held-out score | Tail score | Learning AUC | Seconds / fit |
|---|---:|---:|---:|---:|---:|
| Vanilla LELU MLP | .7941 | .5852 | .5748 | .7306 | .40 |
| Self-context | .9116 | .6720 | .6469 | .8517 | 4.29 |
| **Continuous frame flow (AdamW)** | **.9257** | **.6762** | **.6613** | **.8635** | 16.56 |

Relative to self-context, frame flow gains `.00418` held-out score, `.01439`
tail score, and `.01178` learning AUC. It wins acquisition on 17 of 23 tasks,
tail retention on 13, and held-out endpoint on 9. Its acquisition advantage is
meaningful on 11 tasks and its meaningful held-out wins and losses are 3 and 6.

## What doubled capacity changed

Against the width-24 run, doubling the approximate parameter budget changes
aggregate held-out/tail/AUC as follows:

| Model | Held-out | Tail | Learning AUC |
|---|---:|---:|---:|
| Vanilla LELU MLP | +.0080 | +.0140 | +.0157 |
| Self-context | +.0069 | +.0008 | +.0082 |
| Continuous frame flow | -.0033 | +.0103 | +.0087 |

Capacity helps acquisition and tails, but does not monotonically improve frame
flow's final average. That is further evidence that the limiting object is the
optimization path through the transported frame state rather than raw function
capacity alone.

The decisive larger-model endpoint win is radial stripes: frame flow reaches
`.8586` mean held-out score versus `.5518` for self-context, a `+.3068` gain,
and gains `+.1250` in learning AUC. Polynomial drifted chirp also gains `+.1357`
at the endpoint. Conversely, self-context retains the stronger mean endpoint on
chirp (`-.2027` frame-flow delta), multiscale 1-D (`-.0792`), and high-rank N-D
spiral (`-.0565`), even though frame flow still acquires all three faster.

This split is visible rather than hidden in aggregate MSE: the atlas plots show
where transported acquisition finds global structure early and where its final
state loses local fidelity or lands in a worse seed-dependent basin.

The standalone graphical report is `continuous_frame_2x.html`. Raw benchmark,
paired-seed summary, and full visual probes are in
`results_continuous_frame_2x/`.
