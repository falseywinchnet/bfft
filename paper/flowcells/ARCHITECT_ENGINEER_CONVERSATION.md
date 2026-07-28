# The Architecture of FlowCells

## A distilled conversation between an architect and an engineer

This document reconstructs the complete working conversation as a dialogue between two roles:

- **The Architect** holds the visual intuition, the desired behavior, the biological analogies, and the authority to reject a result that is mathematically respectable but perceptually wrong.
- **The Engineer** translates those intuitions into objectives, algorithms, diagnostics, implementations, and experiments.

It is not a verbatim transcript. It is a careful chronological distillation of the actual exchange, preserving what changed, why it changed, which results caused surprise, and where a line of reasoning was explicitly redirected or abandoned.

The markers have specific meanings:

- **Architect’s recognition** — the user sees a structural truth in an image, behavior, or failure.
- **Engineer’s spark** — a result or observation suddenly reorganizes the technical picture.
- **Redirect** — the architect stops a line of reasoning and changes the problem definition.
- **Double-take** — a result is not merely good or bad; it exposes something neither party could comfortably dismiss.

---

## I. The effect arrives before the representation

**Architect:** We have a single-stage cartoon decomposition algorithm and a Python viewer that Claude worked on. I want it in OBS as a real-time effect. The webcam should feed through it at 240p and 30 frames per second. OBS cannot run the current form directly, so we need a plugin or bridge into BFFT, enough optimization to make it real time, and controls for texture enhancement and cartoon enhancement. Go hunting.

**Engineer:** Then the first task is not to redesign the mathematics. It is to make the existing decomposition observable in motion. We need a reliable real-time path, an effect interface, and parameters that can be adjusted while watching a live scene.

**Architect:** I added it. It is working.

That changed the nature of the project immediately. The decomposition was no longer a still-image operator judged one frame at a time. It had become a material.

**Architect’s recognition:** In the supplemental PDF, look at the first column. Compare our result with the slow standard Gilles method. Our method reaches the smooth analytical state he had sought, but the *difference* between the two methods has a beautiful wave structure. I want to display that difference by itself as a deformation field on a rendered 3D texture—chrome, liquid, something reflective.

**Engineer:** The residual is not merely error. It can be interpreted as geometry: a height field, a normal field, or a refractive displacement. The real-time decomposition has produced a second effect almost accidentally.

**Architect:** And are we sure about twelve versus twenty-four iterations? I dimly remember Claude saying that only one TV solve was needed to get from our state to his.

This was an early statement of a principle that would return much later:

> Do not repeat an expensive process merely because the original derivation was iterative. If the topology of the intermediate state tells us what the converged operation is doing, perhaps one correctly constructed pass is enough.

The live flow mode appeared. The image looked metallic and fluid, with strong transported contours. It was doing in real time something whose comparison method required roughly thirty seconds per frame.

**Architect — double-take:** Really, really cool. The first mode still needs texture/cartoon tuning, but the flow mode is extraordinary. This is a real-time algorithm where the current comparison takes about thirty seconds per frame.

The user then discovered a stranger operating point.

**Architect:** If cartoon and texture are both set to zero, the effect behaves very oddly. Do not touch it. Everything is red, yet somehow not red. Then I apply an OBS color-correction filter: gamma 2, contrast 2.2, brightness \(-0.33\), saturation 1.1, hue shift 180 degrees, a green color addition, and white multiplication. Look at the latest video in Movies. I recorded my phone screen through it. This is wild.

Without the correction, the image was dark red, facial structure was preserved in an uncanny way, and ordinary illumination had become a sparse, inferential geometry.

**Architect — double-take:** It is not a trivial effect. It looks demonic. What if this is the true reality?

The joke carried a serious observation: the operator was exposing a latent organization of contrast and support that did not behave like a conventional filter.

**Engineer:** The important fact is not the color cast. It is that the field preserves structural distinctions after the ordinary appearance has collapsed. The result suggests that the decomposition may be useful as a guide for a representation, not only as an endpoint effect.

**Architect:** This algorithm is going to be very useful for AI.

---

## II. From filtering to an explicit image representation

The architect introduced an older experiment: a viewer that filled an image with colored Voronoi-like regions. The original concept had been to seed centroids with blue noise, let cells skew and reshape toward local gradients, subtract their contribution, and repeat recursively.

**Architect:** The cartoon form gives coarse truth. It can define initial centroids. Those regions can be subdivided and sharpened using anisotropy until they approach the texture form. Recovery of the image would happen by generating centroids on cartoon bounds and orienting them to recapture texture shape as gradients.

He also pointed to Lucky Iyinbor’s Soft Anisotropic Diagrams, or SAD. The old prototype had been visually interesting but unstable and difficult to direct.

**Architect:** Transport flow from the BFFT decomposition should guide how the Voronoi cells reshape. Cartoon determines origin and region. Texture determines behavior. And OKLCH is excellent. Let us import the mathematics, build a DearPyGui viewer, and create a new decomposer on top of our decomposition.

The first implementation descended, but the result was noisy, streaked, and poorly allocated.

**Architect:** Let us stop and work on the formal mathematics. The algorithm clearly needs to be thought about.

The original intuition was restated more sharply:

1. Blue noise gives a non-clumped initial population.
2. Cartoon structure says where broad regions exist.
3. Texture says where shape should become narrow and directional.
4. Cells should approximate sub-objects, be subtracted, and allow another layer to resolve what remains.
5. Edge handling and empty-space filling are the hard parts.

**Architect:** Texture should influence the very shape and gradient of the cell—where it runs.

Then he supplied a perceptual story rather than an equation.

**Architect:** Sometimes, immediately after waking, I have watched vision assemble. A wall appeared halftoned and then resolved. Furniture expanded into three-dimensional structure from the floor in under half a second. In darkness, a small amount of light seemed to flicker—not physically, but because my representation was trying different geometries. There were Fourier-like aspects. When people stare at a mirror without moving their eyes, details fade because the visual system cannot continually register a fixed signal without motion. I configured the BFFT cartoon filter to look like that. It resembles cartoon plus the difference field: texture erodes, and then detail is polled from another layer.

**Engineer:** Then the representation should not be treated as a static tessellation placed directly on intensity. It should be treated as a hypothesis over support: a coarse state first, with detail entering only where the evidence remains coherent enough to justify a more specific geometry.

**Architect:** Code it. I will look.

---

## III. The first major failure: edge worship

The new algorithm fixated on object boundaries. Site density accumulated around the Pikachu outline, eyes, and tail. Broad interior areas remained poorly described. Even after many iterations, the claw was wrong.

**Architect’s recognition:** Nearly every algorithm designed so far makes the same mistake. Nothing makes it go away. The sites wind up on the edges. The error image is dominated by contours, but the reconstruction does not smoothly allocate cells where they are needed. The original method at least knew how to use detail where necessary and remain coarse elsewhere.

The engineer traced the allocation logic.

**Engineer’s spark:** The error image is honest; the allocator is interpreting it incorrectly.

Two concrete mistakes were found:

1. Initial blue-noise sampling squared the density, heavily over-rewarding edge energy.
2. Recursive splitting ranked cells by average error and multiplied candidate value by edge density again.

A tiny boundary cell could therefore beat a large under-resolved region forever. Persistent edge mismatch was being mistaken for reducible detail.

The replacement allocator used:

- genuinely uniform blue noise by default;
- robust **integrated** error rather than average error;
- a coverage and clearance reward;
- a penalty for crowding the same contour;
- measured exchanges at the cell ceiling;
- rejection of exchanges that did not reduce reconstruction error;
- an explicit allocation-pressure view separate from raw error.

On the Pikachu crop, strong-edge site occupancy fell from roughly 76 percent under the old initialization to 12.5 percent initially and 4.2 percent after refinement. At 480 cells, PSNR continued to improve after reaching the budget.

**Architect:** This round was our best.

This became the first stable reference. It was important not only because it scored better, but because it taught a durable distinction:

> Error is evidence about what is wrong. Allocation pressure is a claim about what can be improved.

---

## IV. A biological hypothesis—and its rejection

The architect proposed three cooperating cell types.

**Architect:** Coarse cartoon cells should fill a region, overlap smoothly, and align in color except at edges. Texture cells should be seeded according to entropy density: more entropy, tighter seeding, stronger anisotropy. A third pressure mechanism should make cartoon and texture populations inhibit each other, but only within TV-flow regions. Cells nucleate and flood the geometry. If a region is textured, texture cells appear; otherwise cartoon cells appear. Does this have formal support?

**Engineer:** It resembles a coupled point process or competing reaction-diffusion population, but the important part is not the biological language. The formal question is whether local evidence can determine both cell density and cell type without a global ranked allocator.

The implementation did not work. It stayed below 18 dB.

**Architect:** Nature does not work computationally or competitively. Local gradient consistency should determine what kind of cell appears.

The hypothesis changed:

- local consistency controls anisotropic propensity;
- residual error after subtracting a cell controls new seeding;
- cells may be seeded underneath existing ones, blue-noise style, to fill holes;
- they spawn underneath but fill on top.

The score still failed.

**Redirect:** TV should only determine the boundary of nucleation. Texture must remain the actual boundary of the represented region.

The score stayed near 18 dB.

**Redirect:** Go back. The approach around 6:02 a.m. was better. Return to basics.

This was a decisive intervention. The architect did not permit the biological metaphor to survive merely because it was elegant. If the result washed out structure or could not exceed the earlier reference, the metaphor was wrong or attached at the wrong level.

---

## V. Residual memory and recursive perception

After a break, the architect returned with two ideas.

**Architect:** First, the cartoon decomposition is cheap enough to reuse. If cartoon and texture are applied at half strength repeatedly to their own outputs, what happens? This resembles motion extraction by integrating frame differences: static parts fade to gray and tiny changes remain. It also resembles staring into a mirror until fixed detail disappears.

Second:

**Architect:** What if the Voronoi geometry descends against the decomposition discrepancy, not against raw pixel error? The target should be the error between the decomposition of the reconstruction and the decomposition of the source. Fine blue-noise seeds should appear where detail is fine. Broad underlying cells should nucleate on flow-stage boundaries, coat the subtext of the scene, remain diffuse and irregular, and not be anisotropic.

The experiment exposed a striking residual field.

**Architect — double-take:** The carried collective residual shows exactly where the coarse structure is in the entire scene.

He corrected himself immediately:

**Architect’s recognition:** No—it shows where detail builds *first*.

On the cameraman image, the residual concentrated on the eyes and camera stand. Everything else was less focused.

**Architect:** The finer-grained the collective residual seed, the more anisotropic the item that should appear around it. The coarser and gentler the residual, the more it should become flowing texture. Then recurse and improve.

The non-loop result looked promising.

**Architect:** This looks great. Now try the loop.

The loop produced the gray, eroded perceptual effect rather than a good decomposition.

**Redirect:** No. Whatever this loop is, no. Use the non-loop residual only to nucleate a better decomposition.

The engineer attempted a coupled objective:

- recursive residual as one signal;
- direct cartoon mismatch and texture mismatch as separate signals;
- coarse cells optimized against the cartoon decomposition of the combined reconstruction;
- fine cells layered above coarse cells;
- possible flow discrepancy as an additional discipline.

The result remained washed out.

**Architect:** Something is terribly wrong. It still has the recursive decomposition-to-gray effect everywhere.

**Architect:** What is this “births” machinery? This is not an algorithm we want.

**Redirect:** Blue noise. Spend error.

When automatic spending still failed to reach acceptable detail:

**Redirect:** Return to the manual, round-by-round seeding approach. It was slower to operate but at least each action was interpretable.

When image loading broke:

**Architect:** Restore the ability to open images. Also, 27 dB is poor; the older form did better. Save the useful knowledge—residual error and compositional discrepancy—but restore the genuinely good allocator.

The engineer initially restored too recent a version.

**Redirect:** This is not the last good one. Go further back.

The architect supplied the exact earlier explanation of the best round: integrated error, coverage, clearance, edge-crowding penalty, and measured exchanges.

That restored a methodological discipline:

> New signals could be retained as diagnostics without being allowed to corrupt the known-good allocation mechanism.

---

## VI. One-stage memory and a richer objective

The architect reduced the memory hypothesis.

**Architect:** Residual memory from one stage is enough. It is only needed to reveal the high-focus regions that nucleate the scene. The other recursive signals do not matter.

He expanded the evaluation criterion:

**Architect:** Do not optimize only RGB MSE. Include the MSE between one-stage cartoon decompositions and the MSE between one-stage texture decompositions. Try multiple supported ideas. Record each one separately when it fails. Aim to understand why.

He also offered a mathematical analogy from a signed quadratic optimization problem:

- lift scalar signs to unit vectors;
- solve the continuous relaxation exactly;
- round back through Gaussian hyperplanes;
- understand the gap introduced by discrete ownership.

The exact spin construction was not itself an image algorithm. Its value was epistemic:

> A difficult discrete allocation may be better understood by solving the continuous support problem first, then asking what is lost when it is rounded into hard regions.

Two additional model analyses were introduced. One studied the code and experimental ledger. Another supplied ideas in `notes/vision_segmentation_ideas.md`. The architect raised the budget, selected `25.png`—the simple Pikachu image—as a control, and authorized long experiments and compiled acceleration.

**Architect:** Pikachu has simple geometry. It tests whether the algorithm can allocate domains ideally. Take your time. Use Numba or C where appropriate. Fast experiments matter because they let us ask more questions.

The result was, in the architect’s judgment, the best so far.

**Architect’s recognition:** The graph is misspecified. Perhaps descent is not only transporting flow; perhaps it transports the graph itself. The main image is imperfect, but the cell image approaches optimality. Error is diffuse, which is the best kind of error.

---

## VII. The graph was already inside the renderer

A parallel investigation began from a powerful premise:

> The coupled fit’s own Hessian is the cell graph.

The renderer’s partition of unity determines which cells jointly explain each pixel. The normal matrix therefore contains a nonzero block for cell pair \((i,j)\) precisely when those cells jointly contribute to a pixel. The graph need not be guessed from Euclidean neighbors, blue-noise proximity, or a hand-built signed adjacency.

The engineer’s analysis found:

- exact production-equivalent predecessor-tracking Dijkstra;
- an envelope-theorem derivative of reconstruction error with respect to every transport edge;
- a reverse pass over the Dijkstra predecessor forest;
- additive site-weight descent for a geodesic power diagram;
- measured ridge enrichment derived from residual sign statistics.

The free per-pixel field improved PSNR, but site-weight descent did better with no extra per-pixel parameters. Measured ridge enrichment improved the fit further and left a more diffuse residual than adding an equivalent number of cells.

**Engineer’s spark:** The “reach” was wrong, not the recipe. BFFT feature tuning could recover only part of the gain because the important variable was the geodesic power weight—the transported domain itself.

The architect connected this to SAD.

**Architect:** Study Lucky’s algorithm. Do not implement it. Understand what it models and how it relates to ours. It takes roughly two thousand iterations. Optimally, we want one. Its segmentation is interesting because it is diffusion-nucleated.

Then he asked a rendering question:

**Architect:** In ray tracing, what is the opposite direction? How does Unreal Engine grow an apparently diffused graph outward from the first available light and detail?

The Unreal reference was likely MegaLights: a method that spends a limited set of important light samples, propagates useful information, and avoids exhaustively evaluating every possible interaction.

The analogy was not “use a ray tracer.” It was:

> Begin from causally available evidence. Propagate only what can affect the result. Let support grow outwards along a geometry that already encodes relevance.

---

## VIII. The method becomes a system

The architect requested two simultaneous directions:

1. Merge the strongest Claude work into the main viewer so it could be used immediately.
2. Fork new experiments into separate files so speculative ideas could be evaluated without destabilizing the reference.

Then the focus shifted to high-resolution operation and native acceleration.

**Architect:** We need full-resolution images. We need optimized C++ side ports. Claude found real waste in repeated scoring and repeated decomposition of a target that never changes.

One particularly important optimization treated normal-equation construction as finite-element assembly:

- each pixel contributes a rank-one update to two cell blocks;
- the cell-pair sparsity pattern is fixed by co-ownership;
- the design matrix is unnecessary;
- block contributions can be scatter-added directly into a fixed CSR structure;
- cartoon and texture systems can share the same ownership pattern.

This removed repeated COO construction, sorting, sparse matrix multiplication, and redundant target decomposition.

**Engineer’s spark:** The co-ownership graph used by the mathematical model and the sparse pattern needed by the optimized solver are the same object.

Several unfinished investigations were inherited after another model’s usage was cut off: solver reuse, selected inversion, and FFT opportunities. The architect asked Codex agents to take over.

One solver-reuse line began patterning work by candidates.

**Redirect:** Stop it. Candidate-pattern reuse has the smell of exhaustive iteration. Redirect it.

The architect clarified two desired directions:

1. Find more optimal algebraic forms so each iteration can be pushed much harder and faster.
2. Allow compatible supports to fuse while the representation evolves, rather than applying a stapled-on merge pass.

The high-resolution pipeline remained unexpectedly slow.

**Architect:** Trace why. Find anything that blooms as \(O(n^2)\) or \(O(n^3)\). Examine initialization onward. Do modest SIMD bookkeeping—no manual intrinsic theater, just make sure the code compiles into sane loops. Move the canonical viewer to `viewer/segmenting_veroni_viewer.py`. Organize the several viewer families and explain which one to run.

This was also when the project became a library rather than an experiment collection. Algorithms were to be isolated, named, tested, and eventually ported into BFFT’s C++ core.

---

## IX. Against ranked allocation

The architect rejected top-\(k\) allocation explicitly.

**Redirect:** Belay top-\(k\).

Then he rejected the deeper worldview beneath it.

**Architect:** The method needs to be reworked from the top down. On difficult images it does not degrade gracefully; it collapses into crud. We seed only 180 sites over a 1400-by-700 image and initialization takes fifteen seconds. Adding five hundred cells takes seven seconds, and later iterations get slower. Anything involving search, sorting, or top-\(k\) is not what nature does.

He articulated a different mechanics:

- a site should not be deleted;
- it should grow toward the domain invited by transport;
- faster directional growth should naturally produce stronger anisotropy;
- a cell’s existence inhibits others because it consumes the same available support;
- there is no referee and no ranked queue;
- nobody owns pixels;
- cells consume available energy.

**Architect:** Competition is starvation of a resource that competitors want.

The glass-like transport-flow state from the original OBS effect returned as a possible anisotropy guide.

**Architect:** If that glass field lives as a gradient toward the target, perhaps it already tells us how the support wants to stretch.

At that moment, the research line was interrupted by a production failure.

**Architect:** Something caused a memory overrun in the OBS package that runs the BFFT effect. Spin out a worker, rebuild it, reinstall it for OBS, and find the leak.

**Engineer:** The real-time effect and the transport-cell research share mathematics, but they must not share an uncontrolled lifetime. I will isolate the OBS investigation, repair the package, and keep the experimental state intact.

The memory leak was handled in parallel, the OBS package was rebuilt and reinstalled, and the main conversation returned to transport cells.

**Architect:** Good. Resume the transport-cell experiment. Put the new method in a copy of the viewer first so I can play with it before it enters the canonical viewer.

That request established another practical rule: speculative mathematics had to arrive in an inspectable viewer before it was promoted into the main application.

The engineer continued to speak in terms of offspring and bifurcation.

**Redirect:** Offspring should not reasonably exist. The required density and bifurcation rate are inferable from the transport itself.

When bifurcation was reintroduced as allocation splitting:

**Redirect:** That is still population smuggling. Cell counts should be determined directly on the transport.

This became the central constraint of the eventual FlowCells population law:

> Derive a local support measure. Integrate it. Place the resulting population deterministically. Do not generate candidates and decide which deserve to live.

---

## X. Population from the transport substrate

The architect asked the engineer to build the complete method.

**Architect:** Use the transport itself as the substrate. Take us through the win.

The resulting system was very good at fitting detail, but its visualization did not expose the cells clearly. SAD’s “site IDs” display became the reference for what a meaningful support visualization might show.

SAD’s site-ID image is not simply a hard label map. Deterministic colors are assigned to sites and blended using the same soft ownership weights as the representation. Continuous color therefore reveals shared support and uncertainty.

**Architect:** Being able to see the cells is essential. The site-ID view is a meaningful part of the representation.

He raised smooth AP coverage. Most visible defects disappeared.

**Architect’s recognition:** Why?

The answer was uncomfortable.

**Architect — double-take:** There it is. Massive flaw. None of the cells are occupying their own space.

The apparently successful reconstruction had been leaning on broad overlap rather than correctly placed domains.

The desired process was reformulated:

- preseeded density centroids migrate;
- orientation and filling sharpen;
- the representation converges to content-aligned regions during transport optimization;
- cooperation is allowed only insofar as boundaries evolve together;
- broad canopy filling may be useful, but only if it does not conceal bad support.

Then canopy itself was questioned.

**Architect:** The right process may be soft-to-hard Wasserstein transport. Seed at the point of uncertainty, then divide down the transport. We are not bifurcating the target; we are bifurcating the allocation.

But even “bifurcation” had to be interpreted carefully. The architect was not asking for a recursive birth heuristic. He was asking whether a continuous mass distribution could harden into the correct number and shape of domains without a ranked population process.

The original JavaScript prototype remained an uncomfortable control.

**Architect:** This method feels slow. At full resolution it needs around eight thousand items, while the JavaScript form is fast. Pikachu should optimistically need fewer than one thousand cells, with only a small number of extremely sharp supports at the edges. Instead, I see anisotropy appearing in the white space.

**Architect’s recognition:** The newer method may fit detail well while misunderstanding where its expressive budget belongs. A favorable SNR cannot settle that question.

This forced the engineer to compare the causal flow of the new method with the known-good reference: where sites began, where they ended, which supports became elongated, and whether the transport geometry was explaining detail or merely spending enough overlapping kernels to reproduce it.

---

## XI. One decomposition, not a changed target

The architect imposed another decisive constraint.

**Architect:** We only need one BFFT pass. Any later BFFT pass is operating on a changed target. The original cartoon method has a multiscale ladder that can segment geometry automatically, but I have not extracted it cheaply enough for real time.

This clarified the role of the decomposition:

- it is a frozen measurement of the source;
- its cartoon, texture, glass, and flow signals may be reused;
- it must not be recursively recomputed until the target becomes something else;
- representation optimization should occur under this frozen geometry.

The engineer investigated why the transport did not produce a few clean, large sites over Pikachu’s white background.

The next result was a genuine advance.

**Architect — double-take:** This is incredible. Wherever the source image itself has flaws, the representation shows them, but that is not a flaw in the method. The result is blurry in places, but the image is clearly segmented.

He identified the next task:

> Determine optimal density and the criterion for localized higher density. Once the broad segmentation is correct, tiny regions that genuinely need detail can receive it cleanly.

The runtime still contained several global actions, around 6.8 seconds total. The architect asked whether any could become a one-step descent, following the same logic that had produced the one-shot Meyer method.

Then came the aspect-ratio question.

**Architect:** How does SAD control aspect ratio? Does our support contain a surrogate that can discipline the longest slivers so the maximum aspect ratio approaches the ninetieth percentile?

The engineer added a one-step extension based on support curvature and local topology.

---

## XII. Topology changes are information

The architect returned to the underlying Meyer decomposition.

**Architect:** You said topology changes between global walks. The same thing happens to \(g\) across split-Bregman iterations. Recognizing that led to the fast Meyer one-shot method. Knowledge of *how* the topology changes tells us a lot.

The instruction was to study the cartoon method itself—not merely use its output.

**Engineer’s spark:** An iteration is sometimes expensive only because it rediscovers a topology that the previous state already predicts. If we can characterize where the topology is stable and where it changes, the exact update may be local.

The architect accepted exact local topology refresh.

**Architect:** Exact local refresh is fine. What is bad is an algorithm that has to nitpick its way through candidates. Perhaps the signal telling us where to refill also falls out of transport.

The known-good pieces were merged upward into the main segmenting viewer. Each important algorithm was separated into `port_needed/` so it could be inspected and translated to C++ independently. The faster Meyer implementation was pulled in to support large images.

The full-resolution Golden Gate image ran. The representation was coarse in fine-detail areas, but the broad image was correctly segmented.

**Architect’s recognition:** This is an iteration. Within each cell, the next question is whether further resolution is deserved. I think we reverse the flow: ask where the generated target fails to account for source energy. That error says, “I deserve more cells.” Then perform a forward operation. The major win is that we are now segmenting correctly, so every later decision becomes clean.

This was not a return to recursive births. It was a statement about direction:

1. Forward transport establishes a causal support partition.
2. Reverse transport measures unaccounted energy through that support.
3. A new forward evaluation alters density or geometry where the support law itself says the representation is insufficient.

---

## XIII. The tripod and the sin of preferred directions

On the cameraman image, one tripod leg developed a strong elongated support. The other two did not.

**Architect:** Beautiful. But the tripod has three legs and only one is optimized. Do you see the flaw?

The first proposed response involved recursive support analysis and PCA-like direction extraction.

**Redirect:** That is recursion. Even PCA is sinful. We should not need either. The sliver basis should evolve organically.

An experiment was tried, but the architect suspected the analysis was looking in the wrong place.

**Architect:** Ask a simpler diagnostic question. In a controlled setting, do only shapes aligned with the primary image axes acquire this elongated support? If so, what causes that?

The experiment exposed directional bias in the propagation stencil or local update.

**Architect’s recognition:** I see it. Now go deeper. Why do we need enumerated directions at all? The support should follow the direction demanded by energy. We want infinitely many directions; nature does not manually enumerate them.

This was the conceptual route to continuous-direction anisotropic first arrival:

- represent local support as a positive tensor rather than a finite direction label;
- derive accepted directions from the metric itself;
- use a lattice-basis-reduced or equivalent continuous-direction local solver;
- separate tensor estimation from propagation;
- let orientation emerge as the characteristic of the eikonal solution.

The architect also rejected the “runner-up” model.

**Architect:** It is first come, first served. Transport may ultimately tell us who wins. Runners-up should not exist. Stopping short is an energy problem.

**Engineer’s spark:** The correct second quantity is not a second owner. It is the arrival geometry at the interface: the cost gap, collision time, and causal boundary between first arrivals.

This led to a hard-region model with optional **fractional interface coverage** derived from the collision of arrival fronts, rather than a softmax over two owners. A pixel could remain causally claimed by one site while its boundary coverage was estimated continuously.

The architect authorized the mathematical exploration, then asked for step-size safety and integration into the main viewer.

---

## XIV. Tightening before C++

Claude’s review of the Python ports was introduced. Some suggestions referred to older versions already surpassed; others identified useful local improvements.

**Architect:** Examine every algorithm. Tighten the mathematics before committing to native ports.

The engineer reviewed:

- frozen Meyer geometry;
- density population;
- wide-stencil and continuous eikonal transport;
- reverse first-arrival force;
- hard-region fitting;
- fractional interface coverage;
- soft-support diffusion;
- the single-stage RGB/cartoon/texture objective.

The resulting canonical structure became:

1. One frozen Meyer/BFFT decomposition.
2. Amplitude-normalized support tensors.
3. A separate decisive-boundary tensor.
4. Density from the determinant of the support measure.
5. Curvature correction for support that is locally rank one.
6. Deterministic low-discrepancy population.
7. Continuous-direction anisotropic first arrival.
8. A reverse causal support force.
9. One topology-safe characteristic step.
10. Exact remarch.
11. Conditioned affine cell readout with an optional measured ridge.
12. Fractional collision coverage.
13. Transport-gated soft-support diffusion.
14. Acceptance or rejection under a fixed RGB/cartoon/texture objective.

No top-\(k\), no candidate queue, no offspring, no deletion, no runner-up ownership.

---

## XV. Soft support, fused regions, and the cheek ring

The architect compared the method with SAD visually.

**Architect:** In SAD, continuous color inside a region means a weighted blend of several site colors. Territory is shared. Each site has an oriented spatial kernel, and pixels accumulate normalized contributions from the kernels that reach them.

He wanted compatible cells to lose unnecessary boundaries:

**Architect:** If adjacent cells have little meaningful difference at their boundary, they should be able to fuse into a homogeneous soup. This lets broad regions grow by dropping region identity. We already have nearly all the signals needed to decide this.

Then he pointed to a highly specific defect.

**Architect:** Look at the ring around Pikachu’s cheek patch, then at the paw. The paw is fine. The cheek ring is not. Why?

The difference suggested that not all boundaries were equivalent:

- the paw had high-confidence local geometry and multiple converging support cues;
- the cheek patch was a smooth closed contour, locally rank one along any small arc;
- a purely tangent-oriented support tensor could produce a halo or ring because it lacked evidence for the contour’s curvature over its support horizon.

**Engineer’s spark:** Curvature is not another preferred direction. It is a correction to how long a locally coherent tangent remains valid.

The curvature-horizon correction limited how far a slender support could extend before the contour turned away. It addressed the cheek halo without destroying correctly elongated support along genuinely straight structures.

Soft-support diffusion was also made transport-gated and objective-gated. It could remove internal boundaries in broad smooth regions, but only when the fixed decomposition objective improved. Otherwise it was rejected.

---

## XVI. Native acceleration without solver theater

A viewer bug appeared because some results no longer stored an `rgb` key where the older viewer expected it. That was repaired.

The architect then gave the signal to compile and wire the tight loops into C++.

**Architect:** Make the viewer faster.

During optimization, the engineer considered dense linear solves and Cholesky-based routes.

**Redirect:** Even `linalg.solve` is too expensive, and Cholesky is a name I never want to see in these files.

This was not a prohibition against linear algebra. It was a demand that the representation exploit its own structure instead of hiding the problem inside a generic factorization.

The native ports focused on:

- direct accumulation of local moments;
- fixed-size conditioned fits;
- streaming graph operations;
- contiguous native buffers;
- elimination of repeated decompositions;
- reuse of frozen geometry;
- removal of unnecessary Python materialization.

A large image then crashed during texture upload because the RGBA buffer retained the dimensions of a previous image. The input had 338,650 pixels while the existing texture buffer held only 225,625.

The dynamic-texture path was corrected so image-size changes recreated or resized the backing texture rather than broadcasting into stale storage.

---

## XVII. Five seconds, fifteen thousand cells, and remaining defects

The high-resolution Golden Gate image ran with approximately:

- 15,000 cells;
- 27 dB PSNR;
- stage times around 300 ms, 2.4 s, and 1.873 s;
- roughly five seconds total.

The architect compared this with SAD’s much larger cell count and longer optimization.

**Architect — double-take:** This is very good work. The sky is more fused than before. We need only fifteen thousand cells and roughly five seconds. SAD uses about 128,000 cells and around sixty seconds for a comparable reported PSNR.

He did not turn the comparison into a universal superiority claim. He used it to identify the next research questions.

First:

**Architect’s recognition:** The sky is still segmented too confidently. More domains could probably fuse.

Second:

**Architect:** Look beside Pikachu’s tail, where black meets the white border. Look above the head to the right, where the edge becomes jagged. There are small defects.

These were interpreted as distinct remaining problems:

- **overconfident partitioning in smooth support**, suggesting a better support-validity or fusion law;
- **subpixel interface defects at high-contrast boundaries**, suggesting more accurate collision coverage or local topology refresh;
- **localized jaggedness**, suggesting mismatch between continuous transport geometry and rasterized final support.

The source control image was confirmed:

**Architect:** The Pikachu source is `25.png`.

---

## XVIII. What the architect repeatedly protected

Across the entire conversation, the architect’s redirects protected a remarkably consistent set of principles.

### 1. Preserve strange but informative behavior

When cartoon and texture were both zero, the output became bizarre and visually uncanny. The instruction was not to normalize it away. Unexpected states were treated as evidence.

### 2. Judge support, not only score

A result could have good SNR and still learn detail poorly. Diffuse error was preferable to structured error. A correct broad segmentation with blur was more valuable than a sharp reconstruction built from the wrong domains.

### 3. Do not confuse visible error with useful allocation

Edges naturally retain residual energy. That does not mean another edge site can reduce it. The best early allocator separated raw error from allocation pressure.

### 4. Do not smuggle population through heuristics

Births, offspring, repeated splitting, candidate queues, and top-\(k\) selection were repeatedly rejected. The final population had to fall out of a local support measure.

### 5. One frozen decomposition means one epistemic frame

Recomputing the decomposition on a changed reconstruction changed the question. The geometry of the source had to remain fixed while the representation descended beneath it.

### 6. Do not manually enumerate what should emerge continuously

PCA axes, image-axis stencils, and hand-supported direction sets were rejected when the tripod exposed their bias. Anisotropy had to emerge from a continuous tensor and its transport characteristics.

### 7. Do not introduce a second owner when the boundary is the real object

First arrival determines causal territory. Fractional coverage belongs at the collision interface, not in a runner-up ownership fiction.

### 8. Let local topology changes carry information

Topology refresh was acceptable when exact and local. Iterative nitpicking was not. The project repeatedly looked for the one-shot operation concealed inside an apparently iterative process.

### 9. Preserve the known-good reference

When speculative biology, recursive memory, or coupled population rules failed, the architect repeatedly demanded a return to the precise earlier algorithm that worked. Useful diagnostics were saved; failed mechanisms were not allowed to survive by momentum.

---

## XIX. What the engineer discovered

The engineer’s most important realizations were not isolated optimizations. Each changed the ontology of the method.

### Spark 1: The edge error was honest; the allocator was wrong

This separated residual measurement from reducible opportunity and produced the first genuinely stable refinement law.

### Spark 2: The renderer already defines the graph

The coupled fit’s Hessian and co-support pattern were not implementation details. They were the exact interaction graph of the representation.

### Spark 3: Reach, not feature tuning, controlled the missed gain

Geodesic site weights changed the transported domain itself. Retuning a fixed BFFT metric could not recover the same improvement.

### Spark 4: Residual memory identifies where detail appears first

The one-stage residual was valuable as focus evidence, not as a recursively reapplied target.

### Spark 5: The decomposition is frozen geometry

Cartoon, texture, and glass are not three output images to be blended arbitrarily. They are complementary measurements used to construct support, density, and crossing action.

### Spark 6: Curvature is a horizon correction

The cheek ring showed that a tangent can be locally correct and globally invalid. Curvature determines how far that tangent is trustworthy; it need not introduce a new discrete direction.

### Spark 7: First-arrival collision replaces runner-up ownership

Subpixel boundary coverage can be estimated from the meeting of causal fronts while preserving a hard first-arrival region.

### Spark 8: Soft fusion should be proposed, measured, and refused when wrong

Soft support is not a global blur knob. It is a transport-gated proposal accepted only when RGB, cartoon, and texture objectives jointly improve.

### Spark 9: The graph and the optimized assembly pattern are the same object

The mathematical support graph could directly organize efficient block assembly and native execution.

### Spark 10: Iterative topology may conceal a one-step characteristic update

The fast Meyer experience suggested the same strategy for cell motion: infer a safe action from the current topology, take one characteristic step, and perform an exact remarch.

---

## XX. The method that emerged

The conversation ultimately produced a coherent method rather than a pile of accepted experiments.

The method begins with one frozen single-stage Meyer/BFFT decomposition. From the returned cartoon, texture, defect, and transport-like channels, it constructs an amplitude-normalized local support tensor. A separate boundary tensor records decisive crossings without turning all texture into walls.

The determinant of the support tensor defines a local population measure. Curvature expands that measure where a locally rank-one tangent cannot safely cover a turning contour. Integrating the measure determines the population directly. Deterministic low-discrepancy phase places sites without candidate generation, sorting, deletion, or recursive birth.

The support tensor and boundary action form a continuous anisotropic metric. A first-arrival solver propagates every site through that metric. Territory is causal: the first accepted arrival claims it. Direction is not selected from a small catalog; it emerges from the continuous metric and local eikonal characteristic.

A reverse pass through the causal transport graph measures how source energy acts on the current sites. A topology-safe step moves the sites along this force, followed by an exact remarch. This is the permitted descent: one informed action under known topology, not thousands of blind perturbations.

Within the resulting regions, conditioned affine fields reconstruct color in a perceptual space. Where measured residual statistics justify it, a bounded ridge enriches the local basis. At interfaces, arrival-front collision estimates fractional coverage without introducing a second owner.

Finally, transport-gated diffusion proposes a soft partition of unity. Broad compatible cells can lose internal identity and fuse; decisive boundaries remain expensive to cross. The proposal is accepted only if a fixed objective improves:

- RGB reconstruction error;
- error between the single-stage cartoon decompositions of source and reconstruction;
- error between the single-stage texture decompositions of source and reconstruction.

The result is FlowCells: a representation derived from one decomposition, with population, anisotropy, territory, refinement, and fusion all disciplined by the same frozen support geometry.

---

## XXI. The paper

After the viewer, native ports, full-resolution experiments, failure ledger, and formal support model had converged into a coherent story, the architect asked for a professional report.

**Architect:** Use LaTeX and compile a PDF. Explain the methodology from the bottom up and tell how and why we do everything. Write it so a SIGGRAPH, NeurIPS, or related audience will understand where the method came from. Do not disclaim state of the art and do not claim it. Credit OpenAI Codex Sydney, Claude Opus 5, and Joshuah Rainstar. Credit Lucky Iyinbor’s exploration as the inspiration without which we would not have taken up the challenge.

The engineer reviewed the authorship and editorial specifications in `template_harness`, surveyed relevant segmentation and adaptive image-representation work from the prior decade, audited the canonical implementation, generated scientific figures from the running code, and wrote:

> **FlowCells: Single-Decomposition Transport Geometry for Adaptive Image Regions**

The final paper presents the construction in dependency order rather than discovery order. The failed births, recursive washout, candidate allocators, runner-up ownership, and axis-biased support do not appear as components. They survive in the project history because they explain why the final constraints matter.

That difference is the point of this distilled conversation.

The paper explains **what the method is**.

The conversation explains **what had to be recognized, rejected, and seen twice before the method could become itself**.

---

## Epilogue: the working compact

**Architect:** I will keep telling you when the image is lying.

**Engineer:** I will keep turning those objections into quantities we can inspect.

**Architect:** Do not protect an algorithm because it is clever.

**Engineer:** Then every stage must expose the evidence it uses, the support it produces, and the objective that accepts it.

**Architect:** Do not make nature rank candidates.

**Engineer:** Then population must be a measure, direction must be a characteristic, and territory must be an arrival event.

**Architect:** Do not lose the strange effects. They are often where the representation shows itself.

**Engineer:** Then the viewer is part of the mathematics.

**Architect:** And when the support is finally correct?

**Engineer:** Detail becomes a refinement problem instead of a rescue operation.
