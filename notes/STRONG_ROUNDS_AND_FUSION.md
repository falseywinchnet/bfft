# Strong receiver rounds and marching fusion

Date: 2026-07-26

## Direction one: a harder round

`ReceiverGuidedVoronoi.receiver_trust_step` proposes site reach and two global
log-overlap variables together:

- additive reach on the current measured co-ownership graph;
- cartoon partition temperature;
- texture partition temperature.

Reach uses the existing sparse graph Gauss-Newton model.  The overlap
variables use their exact sigmoid derivatives and a 2x2 Gauss-Newton system.
The joint nonlinear proposal is judged by the measured RGB + one-stage
cartoon + one-stage texture objective.  A failed full proposal receives
exactly one half-step retry.  Accepted state is retained directly; rejected
state is restored from the already solved baseline fields.

Five-image control, 96 pixels on the long side, 300 cells, three outer
rounds:

| image | old objective | joint-trust objective | old proposals | trust proposals | old time | trust time |
|---|---:|---:|---:|---:|---:|---:|
| Pikachu | 0.004999 | **0.004254** | 15 | 3 | 0.330 s | **0.109 s** |
| Cameraman | **0.002410** | 0.002504 | 15 | 5 | 0.337 s | **0.148 s** |
| Chelsea | 0.001380 | **0.001359** | 15 | 3 | 0.267 s | **0.080 s** |
| Coins | 0.002195 | **0.002107** | 15 | 3 | 0.278 s | **0.088 s** |
| Astronaut | 0.006300 | **0.005705** | 15 | 3 | 0.315 s | **0.110 s** |

This is a successful faster round, not a universal replacement.  Cameraman
shows that the old five-level search can still find a slightly better scalar
reach update.  The viewer therefore exposes the new joint round alongside
the established control.

The overlap variables moved materially rather than acting as decoration.
After three rounds, common final values were roughly cartoon 1.4 and texture
5.6 versus the starting 4 and 16.

## Direction two: reversible marching fusion

`experiments/marching_fusion.py` contracts only edges that exist in the
current measured owner/runner graph.

For group affine jet `z`, every member cell jet is obtained by an exact frame
transport:

    x = P z

The constrained field solve is therefore:

    G_group = P.T G P
    h_group = P.T h

No pixels, sites, ownership regions, or boundaries are deleted.  The first
stage only ties compatible jets.  This makes a proposed internal-boundary
removal reversible and gives it an exact objective measurement.

Compatibility uses:

- prediction mismatch on pixels jointly supported by the two groups;
- shared partition mass;
- BFFT edge strength;
- texture activity;
- axial frame alignment where texture is active.

One non-overlapping low-score front is proposed.  If it exceeds the global
objective or PSNR loss budget, one half-front retry is allowed.  There is no
candidate-graph enumeration.

Strict control, 128 pixels on the long side, 700 cells, objective-loss budget
0.25%, PSNR-loss budget 0.03 dB:

| image | cells | soft groups | tied boundaries | PSNR before | PSNR after |
|---|---:|---:|---:|---:|---:|
| Pikachu | 700 | 522 | 178 | 26.361 | 26.353 |
| Cameraman | 700 | 644 | 56 | 29.086 | 29.060 |

At 96/300 under the same budget, Pikachu tied 77 boundaries, Cameraman 47,
Astronaut 12, and Chelsea/Coins tied none.  Refusing to fuse is the correct
outcome when the measured support does not match.

## Viewer

The high-performance viewer now initializes `ReceiverGuidedVoronoi` and
exposes:

- **Joint reach + overlap round**
- **March soft fusion**
- **Soft fusion groups** visualization
- objective and PSNR loss budgets for fusion

Any subsequent subdivision or trust update invalidates the fusion overlay,
because it no longer represents the currently fitted constrained space.

