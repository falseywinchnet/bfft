# Toward an intrinsic support transform

Date: 2026-07-09

Update: the phase-frontier idea below produced a fixed-size certified phase-disk
bound and a closed Zak boundary type.  The current result and measurements are
in `notes/intrinsic_phase_disk_dip.md`; this file remains the starting sketch.

## The distinction

The current `FctPlan` does not make support emerge from a butterfly type.  It
uses dyadic spectra to guess a neighborhood, then explicitly walks candidate
endpoints for each bin.  That is a support-search accelerator.

The desired object should carry time support as part of a phase packet and
resolve it while packets combine.  The output support is then egress metadata,
not the result of a separate DDC-like scan.

## Exact set-valued closure

For a fixed frequency `omega`, let a block A carry its total phasor

    Z_A = sum_{t in A} x[t] exp(-i omega t)

and the set of every normalized prefix candidate

    P_A = {(tau, C_A(tau)) : 1 <= tau <= |A|}.

For adjacent blocks `A|B`, with B represented in its local coordinates, the
exact combine is

    Z_AB = Z_A + exp(-i omega |A|) Z_B

    P_AB = P_A union
           {( |A|+tau,
              Z_A + exp(-i omega |A|) C_B(tau) ) : (tau,C_B) in P_B}.

Egress chooses the member maximizing `|C|^2/tau` and reports its phase.  This is
an honest phase-space type: support is carried through combination and the
winner falls out at egress.

It also exposes the obstruction.  In the worst case every prefix can become
the winner after a future complex translation.  There is no fixed-size exact
summary of `P_A`: the frontier can have O(|A|) members.  Across all frequency
bins, an exact arbitrary-support transform therefore has no established
O(N log N) fixed-width butterfly.  Any claim of such a walk must prove a
frontier bound or exploit additional signal assumptions.

## Correct pressure

For a continuous endpoint model, with `C' = dC/dtau`, the correlation score is

    S(tau) = |C|^2 / tau

and

    S'(tau) = [2 tau Re(conj(C) C') - |C|^2] / tau^2.

The stationary condition is therefore

    2 tau Re(conj(C) C') = |C|^2.

`Im(conj(C) C')` is phase pressure: it is proportional to the derivative of
`arg C`.  Setting it to zero locates stationary phase, not maximum normalized
correlation.  The earlier conversation conflated these two objectives.

## Practical intrinsic approximations

Three honest paths remain.

1. **Typed bounded frontier.** Each packet carries total phasor plus M candidate
   `(tau,C)` states selected by score, endpoint coverage, and angular diversity.
   Combine frontiers by the exact set law, then prune back to M. Complexity is
   O(M N log N); approximation is explicit and inspectable.

2. **Branch-and-bound DIP.** Each support packet carries a current best and an
   upper bound for all unresolved descendants.  A simple safe bound for a
   remaining block B is

       |C_future| <= |C_now| + sqrt(|B|) ||x_B||_2.

   Phase-cone information in a DIP packet can tighten it. Descend only when the
   bound can beat the incumbent. Work becomes signal-adaptive; worst case stays
   quadratic, but no external endpoint polish is disguised as a transform.

3. **Native dyadic-support transform.** Restrict allowed supports to dyadic
   packet boundaries. Then scale/support is a genuine finite type and all
   candidates live in a multiresolution phase space. Fractional endpoint jets
   can refine locally afterward, explicitly as optional interpolation rather
   than as the definition of the transform.

The next prototype should compare these three against the brute-force manifold
using score ratio, support error, frontier width, and operations.  The bounded
frontier is the most direct test of whether real signals keep the phase-space
type small enough to be useful.
