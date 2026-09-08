# Confirmed results before refresh and interval experiments

Scope: retain the original ordinary Meyer transport. No refresh, defect gate,
randomized decision, new relaxation factor, or interval advance is promoted.

## Integration decision

The native split path already contains the confirmed implementation savings:
six spatial state fields, retained primal spectra, streamed reflected
divergences, and the fused lower-triangular pair of screened inverses. A
subsequent ordinary pass uses two forward and two inverse 2-D transforms.
Source initialization is a distinct first pass.

The audit establishes stronger contracts for that implementation; it does
not establish an additional production speedup. Integrate the contracts and
their regression commands. Keep the exact finite-increment construction as
an independent mathematical oracle. Its equivalence to ordinary evolution
does not make its execution cheaper. No runtime algorithm changes are made
by this integration.

## Exact state and phase contracts

For either branch, with its corresponding disk radius, write

    b = t - D x,  p = Pi(t),  d = t - p.

An ordinary step obeys

    p - b = D x - d,
    t_next - t = D(x_next - x) + (p - b).

These separate primal movement from memory movement. A small change in the
image alone cannot certify that the full transport state has settled.
The incoming b lies in its capacity disk along ordinary trajectories with
the prescribed initialization; this is not an unconditional property of
arbitrary extrapolated states.

The complete vector memory is necessary. Its divergence-free component may
have zero immediate divergence but affect the next nonlinear projection,
whose divergence need not vanish. Retaining only a scalar potential or
divergence is therefore not an exact state reduction.

The two local disk-projection remainders drive the complete nonlinear
remainder through the same triangular screened transport. This locates the
source of approximation error without changing the algorithm. In two
dimensions the exterior disk projection remains curved even when the
inside/outside mask is unchanged; a fixed mask does not make the map affine.

The exact finite secant lift satisfies

    Pi(t+h) - Pi(t) = B(t,h) h,  0 <= B <= I,
    h_next = T(z+h) - T(z),  z_next = z+h.

Initializing h=T(z)-z reproduces ordinary evolution. This is a finite
increment identity, not permission to freeze B over multiple steps. The
local disk contraction also does not by itself prove contraction of the
entire coupled map.

## Performance interpretation

Preserve the existing finite-flow API as an existing approximation option;
this audit does not newly certify its schedules as equal-accuracy speedups.
Distance to the 64-pass texture is a reference-agreement measurement, not
distance to an exact minimizer. Extra diagnostic transforms and stored
fields must be charged when measuring any proposed implementation.

The paper's corrected adjoint sign, explicit initialization, and separation
of transport construction from its small numerical approximation already
incorporate the corresponding earlier audit findings.

## Focused regressions

Run from the authoritative checkout on the M4 Mini:

```sh
/Users/ultimussecundai/.local/bin/m4build -- python3 -m unittest \
  experiments.meyer_transport_audit.test_intermediate_state.IntermediateTests.test_within_pass_and_remainder_identities \
  experiments.meyer_transport_audit.test_intermediate_state.IntermediateTests.test_transverse_field_can_change_projected_divergence \
  experiments.meyer_transport_audit.test_finite_transport_lift
```

These six tests cover the ordinary phase identities, nonlinear remainder,
transverse memory, disk secants, arbitrary finite state differences, a
128-step equivalence trajectory, and changing transfer. They verify the
mathematical oracle; they do not replace native implementation tests.

Validation on September 7, 2026: all six passed on the M4 Mini system Python
in 0.110 seconds. The broad checkout sync was stopped after a prolonged
transfer; the authoritative Python sources were copied to a temporary Mini
directory for this run. The unused benchmark-scene import was moved inside
the diagnostic sweep function so these NumPy-only identity tests do not
require a compiled BFFT library merely to import their oracle.

Supporting derivations: [intermediate state](INTERMEDIATE_STATE.md) and
[finite transport lift](TRANSPORT_OF_TRANSPORT.md).
