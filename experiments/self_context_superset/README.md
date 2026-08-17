# Self-context Eikonal superset

Round two promotes parameter-free self-context to the Eikonal baseline and
compares it with an exact-budget ordinary LELU MLP plus five modifications:

- harder allocation (`temperature=.55`);
- a second anchored context refinement;
- uncertainty-gated context strength;
- output secant supervision;
- local allocation-chart curvature regularization.

The task catalog is the union of the 19-problem matched study with low-rank and
high-rank 16-D spirals and a rotated 16-D hypercube checker: 22 problems total.
No model receives a task-specific basis or unseen-support information.
