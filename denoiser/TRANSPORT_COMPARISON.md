# Repository-wide support-mechanism comparison

This is a comparative reading of mechanisms, not a proposal to combine the
algorithms. FMMT remains the only denoiser implemented here.

## Principles already established in the repository

### 1. Transport must carry the state that makes the relation true

The audio denoising study found that keeping phase outside the decomposition
was much worse than decomposing a phase-carrying representation. The specific
`irfft` layout was a null against real/imaginary STFT, so the lesson is not a
preferred coordinate transform. It is that support cannot be inferred from a
magnitude quotient after relational phase has been discarded.

FMMT already respects this better than a scalar filter: signal and residual
are full empirical packets transported on shared fronts. The remaining
violation is upstream, where support birth is still represented by scalar
confidence. The next hair-edge state must carry tangent orientation and
two-sided contrast, not just edge magnitude.

### 2. Support is a coordinate system, not necessarily a good picture

The representation-residual codec's strongest Cameraman result uses a crude
12-cell base. Its value is the deterministic ordering it gives the exact
residual; visual approximation alone would have selected a richer and larger
state. It also proves that storing a support raster alongside a residual
duplicates information.

For this denoiser, a support state should be evaluated by how it changes the
FMMT posterior and by whether its state can later be serialized compactly. A
beautiful support visualization is insufficient. The future representation
should encode generators and transported moments, never another dense bitmap
as normative state.

### 3. Conservation must be literal

The JPEG spatial/frequency ownership solver transports positive and negative
constituent mass through a Laplacian resolvent and verifies that the divergence
equals coefficient displacement. The Meyer decomposition verifies exact
cartoon/texture recomposition. The transport-cell allocator accounts for
accepted mass and interface flux.

The new FMMT support step therefore uses antisymmetric face flux. Its total
bootstrap mass is conserved to floating-point accuracy and its diagnostics
report the error. This does **not** mean denoising conserves the observed pixel
sum after the posterior; it means the provisional support evolution cannot
manufacture or delete its own scalar state invisibly.

### 4. Geometry must be causal and direction-continuous

The continuous-support transport work falsified eight-neighbor Dijkstra as an
approximation to a strongly anisotropic Riemannian metric: it solves a
polygonal local geometry. Reduced-basis simplex Hopf-Lax updates repair this by
choosing a continuous barycentric direction inside a locally causal cone.

The supplied FMMT still transports both packets over an eight-neighbor graph.
The current experiment does not hide that. Its support equation is continuous
in state, but the later front geometry remains jagged. A correct replacement
must preserve FMMT's shared full measures; the existing first-label allocator
cannot simply be dropped in.

### 5. Observation agreement authorizes reach; it does not create truth

The Fourier-shell experiment shows that correctly located metric transport
helps within observed coherent regions, while blind-wedge extrapolation fails
as connection uncertainty rises. More propagation cannot repair an uncertain
connection. The integrated FMMT checkpoint makes the analogous move in image
space: independent lanes authorize support before it becomes hereditary.

The current uplift keeps that principle and removes finite acceptance bands.
Entropy and residual participation limit authority; scale-space lane agreement
limits support. Long transport will eventually need a local confidence/action
field rather than the current scalar global authority.

### 6. Evidence must not rewrite the geometry that supplied it

The object-support and relation-forensics work repeatedly freezes canonical
transport geometry before pooling appearance, focus, or relational evidence.
It records several failures caused by confusing representation evidence with
physical boundaries or object identity.

FMMT's support witness is therefore read only from the unchanged observation
and provisional chart. Clean truth, final reconstruction, and the benchmark
winner never feed the runtime support equation. Future hair diagnostics must
remain evaluation-only; otherwise the edge score would become a hidden edge
oracle.

### 7. Different confidences answer different questions

The object work separates core altitude from saddle margin, material barriers
from occlusion order, and blur magnitude from observation confidence. The
supplied FMMT correctly separates support admission from eikonal barrier
admission, but uses the same scalar evidence for both roles.

The continuous branch retains two outputs—mobility and barrier gate—but both
still derive from one scalar density. The next oriented state should expose at
least:

- support existence;
- tangent continuation confidence;
- two-sided contrast confidence;
- observation authority;
- permitted crossing action.

They may share measurements, but they must not be collapsed into one meaning.

### 8. Topology outranks a better scalar weight

The transport-object audit found disconnected ownership hidden inside a site
ID. No barrier reweighting could recover a silhouette absent from the graph.
The compressor likewise found that soft boundary displacement was the wrong
state because errors included both location and side appearance.

This bears directly on Cameraman hair. If a tapered strand disappears from the
provisional topology, merely increasing scalar support at nearby pixels cannot
reconstruct its continuation. The needed state is a line measure with tangent
and two-sided appearance, allowed to continue through weakening contrast while
paying curvature and contradiction action.

## Algorithm-by-algorithm disposition

| Repository work | Support mechanism worth inspecting | Adopt now | Defer | Reject as a denoiser substitution |
|---|---|:---:|:---:|:---:|
| integrated FMMT checkpoint | witness-before-geometry; shared signal/residual fronts | yes | — | — |
| continuous eikonal transport | reduced local metric basis and simplex Hopf-Lax | — | yes, until shared-measure front exists | first-label ownership API |
| Fourier-shell Eikonal | cross-observation connection confidence and bounded reach | authority principle | local confidence field | Fourier circles as an image prior |
| optimized Meyer jump measure | oriented Hodge jump support and exact recomposition | mechanism audit | oriented line state | cartoon layer as FMMT posterior |
| Meyer harmonic G-ball relaxation | demodulate by measured phase before local diffusion | mechanism audit | tangent-frame edge packets | JPEG-artifact objective |
| representation-residual codec | compact deterministic support as residual coordinate | representation criterion | serialization design | rate as denoising objective |
| JPEG ownership transport | literal conserved constituent flux and inverse accounting | conservation audit | signed packet accounting | quantization/rate branches |
| transport-object support | literal topology before weights; separate confidence roles | topology audit | strand-fragment topology | object labels in denoising |
| relation/focus forensics | freeze geometry before pooling new evidence | evidence hygiene | hair-edge forensic views | feedback into support birth |
| soft-Eikonal learned models | matched-budget ablations and allocation diagnostics | evaluation discipline | possible learned falsification control | learned task model in this blind analytic branch |
| audio denoising study | keep phase inside the supported representation | state principle | 1-D oriented packet study | audio metrics or STFT coordinate as image law |

## Consequences for this experiment

The immediate implementation deliberately stops at the part supported by all
of these principles:

- support is measured independently from the unchanged observation;
- scale is integrated rather than selected from a catalogue;
- support evolution is conservative and budget-stopped;
- 1-D and 2-D use the same law;
- all numerical resolution is reported;
- the supplied FMMT posterior remains intact after support birth;
- failed edge variants remain documented.

It deliberately does **not** yet claim:

- continuous-direction FMMT fronts;
- recovered Cameraman hair;
- a compact codec for the support state;
- C++ readiness;
- superiority over the supplied integrated checkpoint.

Those are the next experiment's falsifiable obligations, not conclusions.

