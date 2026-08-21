# Observer-lens denoising: rejected geometry, surviving bookkeeping

> **Correction:** the optical/deblurring reading of "virtual lens" is
> rejected. The observer is an explanatory compressed sensor over scene
> support. Noise lives in the scene, not in an unknown lens. The active
> experiment is `compressed_eikonal_observer_2d.py`; the material below is
> retained only for its exact residual bookkeeping and as a falsified nearby
> interpretation.

## Correction of the initial interpretation

The proposed observer is not a bank of candidate observations. It is a
virtual lens: one evolving generative transport that absorbs coherent scene
structure into an observer state, leaves transport-incoherent content in a
residual coordinate, and possesses an inverse that renders the structural
state back into scene coordinates.

The earlier relative-chart closure is still useful evidence about uncertainty
between transport maps, but it is not this decomposition. The nearby
`observer_transport_extraction_2d.py` screened-normal experiment is retained
as the explicit rejected interpretation: it compares chart predictions but
does not learn an absorbing lens state.

## Minimal exact lens

The V3 Hopf--Lax forest already stores a sparse lens topology: every accepted
pixel has either one causal parent or a parent simplex with barycentric
fraction `t`, and the complete acceptance order is retained. For a scalar
state the parent prediction is

\[
p_c=(1-t)s_p+t s_q,
\qquad d_c=s_c-p_c.
\]

The innovation is not discarded. The parent state receives the Euclidean
least-squares lifting update

\[
\begin{bmatrix}s_p'\\s_q'\end{bmatrix}
=
\begin{bmatrix}s_p\\s_q\end{bmatrix}
+\frac{d_c}{1+(1-t)^2+t^2}
\begin{bmatrix}1-t\\t\end{bmatrix}.
\]

Processing children in reverse causal order transports predictable structure
toward the observer roots. Every nonroot coordinate becomes an exact detail
coefficient. The inverse subtracts the same update and then reconstructs
`s_c=d_c+p_c`. The transform is therefore exactly reversible before any
denoising decision.

The first residual operator is a screened Dirichlet resolvent on the causal
forest. It acts only on detail coefficients; roots are a zero-detail boundary.
Forward lifting inversion recomposes the result. This is structurally
different from smoothing the raster.

## First result and falsification boundary

The scalar lifting transform reconstructs arbitrary inputs to machine
precision and sends constants entirely into observer/root state. However, its
parent simplex does not reproduce a general affine field because the accepted
child is off the opposing parent edge. Consequently edges, hair, and phase
remain in the detail coordinates and the residual smoother erases them.

The first uplift parallel-transports the terminal full jet from the parent
simplex to the child. With an exact supplied jet, every affine image is
absorbed with zero detail. With the current estimated HJ jet it improves the
scalar lens on Cameraman and tapered hair but not on woven phase. The estimated
jet is still a local posterior field; it is not yet a graph-unrolled
connection phase like V3's relational texture state.

The experiment establishes the architecture but rejects both current
endpoints as denoisers:

1. backward eikonal lifting is the analysis/absorption map;
2. retained detail makes the map exactly invertible;
3. denoising belongs in observer detail space;
4. scalar parent prediction is insufficient;
5. a local jet is insufficient for distributed phase;
6. the next lens state must carry graph-unrolled affine jet, orientation, and
   phase through the lifting steps before its residual can be interpreted as
   noise.

This is the direct algorithmic analogue of a learned encoder/decoder: the
research question is whether the lens state and its evolution can remain a
small continuous transport program.

## Replacement: direct support observation

Let the V3 transport measure emit sparse germs and let continuous eikonal
first arrival define their supported measurement cells. If `B_S` contains the
affine moment basis carried by that support, the observer takes one direct
compressed measurement

\[
e_S=B_S(B_S^TB_S)^+B_S^Ty,
\qquad r_S=y-e_S.
\]

Then `B_S^T r_S = 0`: the residual is precisely what this observer cannot
explain. A later observation remeasures support on that residual and creates a
new sensor; it does not descend on the old coefficients. The first unresolved
issue is selection bias because the same scene currently emits `S` and is
measured by `B_S`. Until a transport-valid cross-measure removes that bias,
the extraction is a theorem probe rather than a denoiser.
