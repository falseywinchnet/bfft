# Signed spectral--polarization transport foundation

## Implemented scope

The first experiment implements and tests three coupled objects:

1. a six-coordinate continuous light packet;
2. a signed positive/correction ledger with physical terminal reconciliation;
3. a registered sparse transport graph with exact incremental correction after
   link insertion, deletion, or gain change.

The implementation is in `light_tensor.py`, its executable invariants are in
`test_light_tensor.py`, and `run_light_tensor_experiment.py` produces the
measured dogfood record.

## The six physical coordinates

One packet is

\[
\ell=(w,\mu_\lambda,\sigma_\lambda^2,\theta,p,\chi).
\]

- \(w\) is signed integrated radiant power. Positive packets add radiance;
  negative packets carry a correction that removes previously registered
  radiance.
- \(\mu_\lambda\) and \(\sigma_\lambda^2\) are the centre and shape of one
  Gaussian spectral component.
- \(\theta\in[0,360)\) is the central polarization orientation in degrees.
- \(p\in[0,1]\) is the linear peakedness around that orientation.
- \(\chi\in[-1,1]\) is signed chirality. Its sign is handedness; its magnitude
  is the retained circular polarization concentration.

The physical realizability constraint for the represented distribution is the
Poincare ball

\[
p^2+\chi^2\leq 1.
\]

Source and geometry labels are registration metadata. They are not physical
light coordinates and therefore do not make this a seven-dimensional light
state. A non-Gaussian spectrum is a sparse mixture of six-coordinate packets;
it is not silently collapsed into one packet.

The sign belongs to the measure. It does not permit negative emitted light.
At a terminal or a boundary that destroys the transported representation, the
positive and negative measures are reconciled wavelength by wavelength. Any
negative remainder is recorded as an unmet correction and clamped to zero.
For terminal physical verification only, the stored state can be mapped to
\((Q/I,U/I,V/I)=(p\cos2\theta,p\sin2\theta,\chi)\) and projected onto the
physical polarization cone. These Stokes values are derived; they are not the
transport representation. Thus linear transport can use signed algebra
without allowing an observer or surface to receive an unphysical radiance
state.

This is a Jordan decomposition of the transported measure:

\[
\nu=\nu^+-\nu^-,\qquad \nu^+,\nu^-\geq0.
\]

The purple control uses one broad positive spectral packet and one narrow
negative green packet. It removes green participation while retaining more of
the red and blue sensor response. No negative light reaches the terminal.

## Continuous spectra and closure

Two spectral parameters are sufficient to *store one Gaussian component*.
They are not sufficient to determine the output of an arbitrary
wavelength-dependent material as another Gaussian. A notch, line spectrum,
sharp absorption edge, or dispersive branch creates skew, kurtosis, or
multiple lobes.

The experiment evaluates the continuous incident Gaussian on a deterministic
local wavelength chart, applies the material transmission, and fits Gaussian
mixtures of increasing rank. Acceptance is based on relative spectral
\(L^1\) error, not only moment agreement. Each retained component still has
only centre and variance. This gives the required refinement law:

\[
\text{one packet if closed};\qquad
\text{bifurcate the spectral mixture if the material creates unresolved shape}.
\]

The initial skew/kurtosis splitting heuristic was rejected: on the dogfood
notch it reduced moment residual while increasing spectral \(L^1\) error.
The retained density-tested mixture is the correction produced by the
experiment.

## Polarization feed-forward

The transported representation is exactly the central orientation,
peakedness, and signed chirality requested here.

- Unpolarized light is \((\theta,p,\chi)=(0,0,0)\).
- A linear filter at angle \(\phi\) extinguishes the incident population at its
  incoming port and selects/generates a transmitted mode family with
  \(\theta'=\phi\), \(p'=1\),
  and \(\chi'=0\). Its surviving power follows the generalized Malus fraction

  \[
  a_L=\frac12\left(1+p\cos 2(\theta-\phi)\right).
  \]

- A circular filter with handedness \(h\in\{-1,+1\}\) extinguishes the incident
  population and selects/generates the transmitted family
  \((\theta',p',\chi')=(0,0,h)\). Its surviving fraction is

  \[
  a_C=\frac12(1+h\chi).
  \]

- A partially polarized field keeps \(p\) and \(|\chi|\) inside their
  continuous ranges. The chirality sign carries handedness.

At a diffuse recipient surface, the incident family is extinguished and the
surface generates a diffuse outgoing family with
\((\theta,p,\chi)=(0,0,0)\). The incident packet itself is never rewritten.

Inside a medium, every polarization-mode coordinate is invariant. Extinction
and diffusive mode loss are one probabilistic extinction equation:

\[
w_m(d)=w_m(0)\exp\left[-\rho d
\left(\sigma_t+\kappa_Lp_m+\kappa_C|\chi_m|\right)\right],
\]

\[
(\theta_m(d),p_m(d),\chi_m(d))=
(\theta_m(0),p_m(0),\chi_m(0)).
\]

The field's aggregate mode-distribution concentration appears to change only
because the weights of its unchanged mode families decay at different rates.
The dogfood material sets
\(\kappa_C<\kappa_L\), so the circular family retains more population than the
linear family. Arbitrary Mueller interactions are supported transitively as a
boundary ledger. The incident family is retained verbatim, its incoming
population is extinguished, and a distinct outgoing family is
selected/generated. Stokes coordinates are derived for the single local
matrix multiplication and the outgoing distribution is immediately
represented again as \((\theta,p,\chi)\). Stokes coordinates are never stored
or marched. Matrix composition and sequential boundary generation are
numerically identical; this is coordinate arithmetic for the boundary, not an
assertion that the incident polarization changed.

## Antimatter transport and registration

Let the old solved field obey

\[
L=T_{\rm old}L+E.
\]

After geometry moves and its local links are rebuilt, the new field obeys

\[
L'=T_{\rm new}L'+E.
\]

For \(\Delta L=L'-L\), subtraction gives

\[
\boxed{
\Delta L=(I-T_{\rm new})^{-1}
         (T_{\rm new}-T_{\rm old})L
}
\]

when emission is unchanged. A changed emission adds \(\Delta E\) inside the
parentheses. This is the exact mathematical form of antimatter transport.

- A deleted or weakened link injects negative registered light at its former
  receiver.
- An inserted or strengthened link injects positive registered light.
- The ordinary residual march propagates both signs through all descendants.
- If a feedback loop returns to an earlier node, the correction follows it
  automatically. No special backward invalidation recursion is required.
- New family members are reached through the new operator.
- A disconnected branch is never visited.

Every packet carries a source-provenance bitset. Every edge carries the label
of the geometry that owns it. The prototype computes the forward affected
closure from changed edge receivers, including strongly connected feedback
regions. A production million-object register should replace arbitrary Python
integers with interned compressed bitmaps or provenance-DAG handles, but the
semantics are already explicit.

## M4 dogfood measurements

All nine focused invariants and the seven established transport controls pass
on the M4 Mini. The complete experiment took 94.553 ms in the Mini's system
Python/NumPy runtime.

The signed spectral ledger subtracts 0.15 units of narrow green participation
from one broad positive packet. Its terminal has zero unmet negative power.
Relative sensor transmission is `(red, green, blue) = (0.8481, 0.6771,
0.9495)`, so green is selectively removed rather than replacing the field with
an RGB label.

For a narrow absorption notch, one centre/variance packet has 27.34% relative
spectral L1 error. The accepted six-packet mixture has 5.30% error. This is a
measured rank increase caused by the material law.

The polarization control sends unit-power linear and circular mode families
through the same volume. Their stored states remain exactly `(31, 1, 0)` and
`(0, 0, 1)`. The linear family retains 0.1620 power; the circular family
retains 0.4317. When each is mixed with an unpolarized family, aggregate
mode-distribution concentration changes from 0.5 to 0.2210 for the linear
mixture and 0.4305 for the circular mixture solely through unequal mode
extinction. No constituent family changes its polarization. The diffuse
terminal extinguishes its incident family and generates a new isotropic family
with degree zero.

The localized 64-node registration control changes one branch containing a
feedback loop. Full recomputation needs 101 edge applications; signed
correction needs 67. Its maximum dense spectral-Stokes discrepancy from a full
solve is `1.05e-16`, and the disconnected source branch retains its original
provenance.

The repository-geometry control uses the existing centroid visibility solver:
eight facing surface patches change from 32 open links to 24 links after a
spherical occluder is inserted. Signed correction remains correct to
`3.78e-15`, but its affected closure is the entire strongly connected graph
and costs 156 edge applications versus 84 for full remarching. The retained
planner therefore selects `full-remarch`. Antimatter transport is an exact
update law, not a promise that every topology change is local.

## Boundary volumes: next implemented boundary

An invisible object volume should be a cached child transport domain, but it
cannot be a one-way light sink. A refractive, reflective, luminescent, or
subsurface object changes the parent scene. The correct interface is a two-port
boundary response:

\[
L_{\partial V}^{\rm out}
=\mathcal R_V[L_{\partial V}^{\rm in}]+E_V.
\]

The parent registers incoming directional regions on \(\partial V\). The child
domain turns them into internal boundary emitters, solves its triangles,
normal/displacement fields, and participating media, and returns only the
compressed outgoing boundary field. The child's internal geometry is absent
from the parent BVH. If neither its boundary input nor its internal state
changes, the child solve is not invoked.

This is a Schur-complement/domain-decomposition boundary, not visible proxy
geometry. A box, sphere, or other analytic hull is only a chart for the
boundary field. It does not approximate the object's visible shape.

For participating media the registered coefficients must include extinction
\(\sigma_t\), absorption \(\sigma_a\), scattering \(\sigma_s\), and a
directional phase function. Density and a scalar diffusivity alone cannot
produce a trustworthy coherent beam inside a prism. Diffusion is the
many-scattering closure; collimated and low-scattering transport must remain a
directional spectral field. The next experiment will implement this two-port
volume response and use antimatter residuals when its cached response changes.
