# Analytical-twin direction and archived Chambolle candidate

The user has now set Chambolle aside and requested other problems. The ROF
construction below is an archived candidate, not the active implementation
target. The measurements do not prove Chambolle optimal; they did not establish
a compelling advantage for our proposed representation. Preserve the broader
objective of an analytical representation followed by a single correction,
with all construction and verification costs included. Screen other real
library problems before selecting a new implementation target.

The user's explicit screening requirement is now problems KNOWN to take
thousands of iterations. An expensive update, a high maximum-iteration setting,
or a convenient exact special case is not qualifying evidence. Require actual
published runs or documented convergence behavior for a specified method and
problem regime; distinguish successful convergence from capped/nonconverged
runs and reconstruction-quality stopping from numerical optimality. The initial
covariance-barycenter and Gaussian-process suggestions have not met this gate.
Check stronger existing methods before interpreting the long run as an open
acceleration opportunity.

The user rejected modest iteration-preserving speedups as the objective.
The requested direction is an analytical twin of the original problem:
reach a strong approximation to its floor through our analytical engine,
then obtain the library's numerical solution with a single correction solve.
The target is a much sharper descent, potentially orders of magnitude faster,
not a twofold replay of the library's iteration. Prior finite-pass and adaptive
ports in this directory are diagnostic evidence, not the active design target.

Use the same image-domain ROF problem as scikit-image:

    J(u) = 0.5 ||u-f||^2 + lambda sum_i ||(G u)_i||_2.

G has the library's forward differences and zero terminal components. Its
floor is unique because the data term is strongly convex. A dual feasible
field p, with ||p_i||<=lambda and zero terminal components, gives

    d = G.T p
    D(p) = -0.5 ||d||^2 - <f,d> <= J* <= J(u).

For the library readout u=f+d, the primal-dual gap equals
lambda*TV(u)+<u,d>. This provides an objective-floor certificate independent
of a stopping-energy change or a long trajectory used as an exact oracle.
The library stopping quantity is ||d||^2+lambda*TV(u), divided by pixel count;
it is not the same expression as J(u).

For an analytical candidate, the optimality conditions can instead be written

    u-f + lambda G.T n = 0,
    n_i = (G u)_i / ||(G u)_i|| if (G u)_i != 0,
    ||n_i|| <= 1 otherwise.

These define the analytical twin's target independently of Chambolle's update
rule. A single correction solve is the final stage only; it does not supply
missing analytical geometry for free. Flat regions and nonzero-gradient
normal directions must be represented consistently. An arbitrary frozen
quadratic surrogate, repeated IRLS/Newton iterations, or generic replacement
optimizer is not by itself the requested analytical-floor construction.

The phrase "80% of floor" still needs a declared metric: fraction of
objective decrease, distance to the limiting reconstruction, or reduction
of remaining optimality gap. Measure all three where possible rather than
silently redefining the user's target. Charge analytical construction,
correction, and certification in any claimed 1000x elapsed-time gain.

The first direct Chambolle count uses 512x512 camera, noise sigma=.1, seed 0,
weight=.1. Default stopping takes 7 iterations; 80% of the available objective
decrease already takes 3. A contemporaneous primal-dual gap <=1e-5 of the
available objective decrease takes 5331 iterations. The late floor is bracketed
within .012 objective units after 10000 iterations. Thus default-call speed,
early objective progress, and accurate approach to the floor are different
benchmark contracts. See `chambolle_floor.py` and the retained JSON records.
