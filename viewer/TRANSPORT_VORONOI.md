# BFFT Transport Voronoi Laboratory

This experiment reconstructs an image as a hierarchy of oriented cells. It is
not a port of Soft Anisotropic Diagrams (SAD): BFFT supplies the geometry
before the reconstruction is fitted.

## Model

1. A BFFT Meyer split produces cartoon and texture lightness.
2. One accurate cartoon-side ROF solve produces the TGFD outer-map defect
   `cartoon - ROF(image - texture, lambda)`. This is the fine state-to-state
   wave field visible in the visual supplement's TGFD/Gilles difference.
3. Cartoon boundaries and the defect field define crossing barriers.
4. A structure tensor blends those normals with texture gradients into a
   spatial Riemannian metric.
5. Uniform deterministic blue noise establishes a coarse underlayer, with
   TV-flow basins deciding where germs originate.
6. Multi-label graph propagation forms additively weighted geodesic power
   cells. Local gradient consistency continuously controls anisotropic
   propensity; it does not choose a competing cell species.
7. New cells germinate at blue-noise-spaced maxima of the reconstruction
   residual. Each child records the parent ownership map at birth and can
   never propagate outside that inherited footprint: it is born underneath
   its parent geometry and fills residual holes on top.
8. Each cell fits a bounded two-dimensional detail patch in OKLab over a
   persistent color-cartoon base. This is the Cartesian, hue-wrap-safe form
   of OKLCH.
9. Robust integrated residual and geodesic clearance decide where children
   are added. Hierarchical cells are not exchanged after birth, because doing
   so would sever their inherited support.
10. Rendering optionally blends the nearest two cells with a soft partition of
   unity. Set ownership softness to zero for hard cells.

In short: cartoon supplies the coarse underlayer, BFFT flow locates
nucleation, texture bounds and shapes propagation, and residual holes
germinate new children which fill on top.

## Run

From the repository root:

```bash
.venv/bin/python viewer/transport_voronoi_app.py
```

Choose a gallery image or load a file, press **Initialize**, then use **One
subdivision** or **Run continuously**. The right-panel selector exposes the
cartoon, texture, TV flow, site density, ownership cells, error, and fitted
reconstruction.

The cyan stroke on a residual child is its local texture tangent. Gold points
are the coarse underlayer and magenta points are descendants. The application
stops automatically at the maximum-cell
ceiling; raise that control and run again to continue the hierarchy.

## Controls worth exploring first

- **cartoon edge density**: allocates more sites around coarse BFFT bounds.
- **texture density**: raises germination density where the texture residual
  has energy.
- **initial content bias**: blends content weighting into the initial uniform
  blue-noise layout. Zero is intentionally the default.
- **anisotropy**: elongates cells continuously where local gradients agree.
- **density anisotropy**: strengthens that directional metric in dense
  content without creating a direction where tensor coherence is absent.
- **gradient consistency influence**: controls how strongly agreement among
  local gradients contributes to the displayed texture propensity.
- **flow TV sweeps**: controls the accuracy of the one-step outer-map
  projection used to form the transport defect.
- **cartoon crossing barrier**: raises the geodesic cost of cutting through a
  coarse boundary.
- **site reach**: supplies the additive power weight independently of shape.
- **ownership softness**: blends the two nearest local color planes.
- **centroid yield**: lets sites drift toward density and residual error after
  each subdivision.
- **split batch**: sets how quickly the hierarchy descends.
- **detail precision**, **Fade detail**, and **Refresh detail** expose the
  texture evidence gate without changing the persistent cartoon.

Initialization controls require another Initialize. Cell policy controls
apply to the current hierarchy on its next step.

The **Allocation pressure** view is the allocator's residual-density,
coverage-aware germination field. **Hierarchy cells** shows the coarse
underlayer and descendants; **Flow territories** shows the TV nucleation
basins, not ownership walls. Allocation pressure is intentionally different from raw
**Error**: persistent one-pixel edge mismatch must not consume the entire
cell budget.
