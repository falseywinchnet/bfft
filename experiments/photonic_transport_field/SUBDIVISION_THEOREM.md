# Ray-induced light-field subdivision theorem

## 1. The object being subdivided

The scene is never tiled in advance. It remains an analytic first-hit oracle
over objects, materials, and bounding volumes. Subdivision acts on a light
bundle

```text
D = X x Omega,
```

where `X` is an already illuminated source footprint and `Omega` is an
outgoing directional cone. Its geometry is the first-hit map

```text
H(x,omega) = first scene intersection of the ray (x,omega).
```

The image `H(D)` is a recipient footprint created by light. A surface region
that no surviving bundle reaches never becomes a transport cell.

## 2. Acceptance certificate

Suppose every ray in `D` reaches the same smooth receiver chart, has the same
visibility and material state, lies away from a tangent or first-hit exchange,
and admits a differentiable chart map `phi = chart o H` with finite bounds.
Let `g(z)` be RGB power density on bundle coordinate `z=(x,omega)`, including
source radiance, angular measure, transmittance, and receiver response. For a
representative `z_c`, define

```text
osc_D(g) = sup_{z in D} |g(z) - g(z_c)|.
```

Then replacing the complete bundle integral by one transported footprint
obeys

```text
| integral_D g(z) dz - measure(D) g(z_c) |
    <= measure(D) osc_D(g).                       (1)
```

If `g` is Lipschitz on `D` with constant `Lambda_D`, then

```text
osc_D(g) <= Lambda_D diameter(D),                 (2)
```

so a sufficient acceptance rule is

```text
measure(D) Lambda_D diameter(D) <= epsilon_D.     (3)
```

Equations (1)-(3), plus the first-hit continuity certificate, are the
subdivision theorem. They concern the geometry of the light field, not a
surface-pair kernel over a pre-existing mesh.

## 3. Proof

For every `z` in `D`, the definition of oscillation gives

```text
|g(z)-g(z_c)| <= osc_D(g).
```

Integrating both sides and applying the triangle inequality gives (1).
Lipschitz continuity gives

```text
|g(z)-g(z_c)| <= Lambda_D |z-z_c|
                <= Lambda_D diameter(D),
```

which proves (2) and therefore (3). QED.

The inequality is not the difficult part. The essential geometric work is
certifying that `H` is continuous on the bundle. The native prototype uses
center, corner, edge-midpoint, and deterministic interior probe rays. A
production compiler replaces this empirical certificate with beam/BVH bounds
that prove same-hit and wholly blocked cases.

## 4. Boundary bifurcation

If probe rays disagree about first-hit object, miss state, material, or chart
orientation, continuity is not assumed. The bundle bifurcates along the source
or angular coordinate producing the greatest projected disagreement. Only
children intersecting that disagreement continue to refine.

For piecewise smooth scene geometry, coherent interior bundles cover a
full-dimensional optical region. Uncertain bundles collapse around the
codimension-one preimage of silhouettes, occlusion boundaries, corners, and
material transitions. At angular scale `h`, interior work therefore follows
optical-region count while finest work follows boundary measure, rather than a
uniform surface-area tessellation.

## 5. Energy extinction

Let `P_D` be the RGB energy owned by a bundle and `U_D` a conservative upper
bound on the fraction that any descendant can deliver. The bundle and its
complete undiscovered recipient subtree are removed when

```text
|P_D|_1 U_D <= tau_D.                             (4)
```

All removed bounds debit one cumulative error ledger. Under Lambertian
reflection the next outgoing energy is no greater than `rho_max` times the
received energy. If `rho_max < 1`, omission `delta` at one generation can
affect all later generations by at most

```text
rho_max delta / (1-rho_max).                      (5)
```

Thus active light-field size shrinks with diffusion. There is no fixed
scene-wide transport graph.

## 6. Projected footprints and overlap

An accepted bundle stores its receiver-chart footprint `phi(D)`, transported
RGB density, generating bundle, and error certificate. Overlapping footprints
are summed as an optical arrangement. They do not require the underlying
surface to be subdivided first.

For a diffuse material, accumulated incident irradiance creates

```text
L_out(x,omega) = albedo(x) / pi * E_in(x).
```

Every energized arrangement region is therefore the source footprint of a new
hemispherical bundle generation. Regions with compatible material, radiance,
and outgoing angular state may merge; regions separated by an optical boundary
may not.

## 7. Finite-resolution termination

Away from first-hit discontinuities, dyadic subdivision drives
`diameter(D)` and the bound in (3) to zero. Boundary cells stop when their
projected uncertainty is below the requested display scale or when (4)
extinguishes them. Consequently the construction terminates at finite error
and image resolution while retaining large smooth light-field footprints.

Subdivision conserves bundle measure and power:

```text
sum measure(child) = measure(parent),
sum power(child)   = power(parent).
```

This—not conservation of a prebuilt surface mesh—is the invariant required by
the implementation.
