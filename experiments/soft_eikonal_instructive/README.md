# Soft Eikonal instructive-compartment screen

This experiment compares parameter-identical variants of the soft Eikonal pool.
The biological paper motivates spatially segregated, cell-specific instructive
streams; it does not imply that cortex omits feedback or backpropagation.

Variants:

- `soft_eikonal`: unchanged control;
- `self_context`: a first allocation reconstructs a parameter-free contextual
  guess that augments the latent input before final allocation;
- `garnish_instructive`: two perturbed views compute an averaged error
  derivative that instructs the true-input stream, which receives no direct
  target loss;
- `paired_zero`: a truly double-width input and double output trained on paired
  and unpaired samples, always evaluated as `(x, 0)`;
- `secant_relational`: direct loss plus prediction-difference supervision;
- `allocation_secant`: local second-difference regularization of the learned
  filter allocation;
- hard and soft allocation temperatures;
- the exact-budget ordinary LELU MLP control.

All architectures exactly match the unchanged soft Eikonal parameter budget.

`run_visual_probes.py` exports selected fitted fields and 1-D curves for visual
inspection after the numerical screen.
