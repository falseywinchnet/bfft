# Extinguishing opaque source support before scene traversal

The night city's original native lighting preparation took 800.314 seconds.
A source-chart extinction path reduces that to approximately one minute while
reproducing the entire captured illumination screen byte for byte.

The main expense was not the intensity of the car and street point sources.
Those are evaluated during pinhole capture. It was the static area-light
irradiance preparation: millions of receiver queries generated hundreds of
millions of source intervals, most of which were already hidden behind opaque
geometry.

## Changed termination point

The renderer already projects finite blocker polygons onto the rectangular
source chart and partitions it into source rows and intervals. Previously, each
interval still discovered a complete receiver-to-source surface sequence through
the scene BVH, then evaluated its Gaussian ordinates before returning zero for
opaque occlusion.

The new path retains an opaque blocker's row span while constructing the same
cuts. For an interval inside that span, it uses the named primitive as a zero
witness. It checks the midpoint and every actual Gaussian ordinate using the
existing finite intersection predicate, ray-origin offset and evaluation limit.
If the primitive blocks all of them, the complete source interval contributes
exactly zero. No source-program discovery, later geometry walk, transmission
arithmetic, or kernel evaluation is needed for that interval.

This is not a distance cutoff or an intensity threshold. Source radiance,
geometry, cosine tests, source partitions, quadrature nodes, atlas resolution,
refinement tests and nonzero calculations remain unchanged. Accumulation retains
the same nonzero terms in the same order.

The shortcut requires an all-opaque candidate set and no affine medium-source
input or response request. Mixed dielectric candidate sets, unproven intervals,
curved blockers without a planar witness, and numerical membership failures use
the existing path. This deliberately conservative implementation checks the
actual ordinates instead of assuming perfect floating-point polygon membership.

## Verification

- 2,177 seeded city receiver queries, including facade edges and corners, return
  exactly the same RGB triples with extinction enabled and disabled.
- A mixed glass/opaque fixture retains the original path and result.
- A sphere fixture checks conic and grazing-row fallback behavior.
- Four native scenes (`standard`, `aperture-canyon`, `mirror-relay`, and
  `occlusion-garden`) have identical direct, bounce and outgoing node radiance,
  coupling matrices, atlas lookup tables, coarse/refined irradiance values and
  refinement decisions with the feature on and off.
- The complete 44,089,372-byte serialized night screen, including its header,
  is byte-identical to the original 800-second bake. Its clear-frame PPM is also
  identical. The frozen screen-plus-mip checksum remains
  `2970080582719551191`.

The first complete accelerated run made 2,930,492 source-integral calls and
visited 413,926,251 eligible source intervals. It extinguished 383,483,907 of
those intervals before scene traversal. It performed 1,150,457,809 small
primitive-certificate checks; 30,442,344 intervals retained the original path.
The complete-bake statistics are from the serial preparation thread; parallel
capture worker counters are not included in them.

The first measured accelerated preparation was 63.0905 seconds, a 12.69× speedup
and a 92.12% time reduction against the original 800.314-second preparation.
Pinhole capture remained approximately 5.85 seconds. The focused query comparison
was 740.91 ms versus 63.51 ms, with zero unequal queries and zero maximum error.
These are measured CPU runs, not universal speed guarantees.

The source partition and irradiance-atlas construction still take time. This
change eliminates proven-zero ray walks; it does not claim to eliminate all
precomputation cost.

## Use and reproduce

The night-screen `bake` mode enables `source_interval_extinction`. Its
`bake-legacy` mode selects the old path for a controlled comparison. Other native
renderer entry points retain their prior default behavior.

```sh
experiments/photonic_transport_field/city_rain/run_source_extinction.sh
```

The wrapper builds and runs the focused and native regression comparisons on
the host selected by `m4host`, performs the accelerated night bake, and copies
results back to `output/source_extinction/` immediately. The original benchmark
metadata is retained there as `reference_bake.json`. Re-running the standard
`run_night_screen.sh` now uses the accelerated preparation automatically.
