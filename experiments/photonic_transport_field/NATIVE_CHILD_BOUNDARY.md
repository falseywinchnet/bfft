# Native child boundaries: ownership implemented, compactness gate failed

**Historical exact-state cache result.** The subsequent shared-law route removes
query-state registration and is now the default response backend when child
boundaries are enabled. See [OBSERVER_REGISTRATION.md](OBSERVER_REGISTRATION.md).
Reproduce the old cache explicitly with `--child-response exact`.

The native prototype routes the prism, thin sheet, and participating jelly through
owned boundary responses. It does **not** solve the million-operation rendering
problem. It remains opt-in with `--child-boundaries on`; the executable defaults
to `off`. Promoting the current exact-state response cache would introduce a large
memory cost and regress the most expensive scene.

## Implemented boundary route

`native/child_boundary_response.hpp` contains immutable child definitions with
private face copies and material response data. The parent stops at the first
boundary. The child returns weighted external ports and, for the jelly, a linear
internal-source coefficient. Only those external ports enter the parent optical
queue. Internal reflections query only the child's owned geometry. Convex prism
exits use supporting planes, avoiding the finite-face offset failure at grazing
wedges. Total internal reflection uses reflection coefficient one.

The native camera, straight source-light queries, beam traversal, spectral
classifier, and visible-path classifier dispatch through that interface. Source
programs record boundary role as well as primitive identity so reusing a program
cannot mistake an egress for an ingress. The source-light model remains the
existing straight visibility/attenuation model; this work does not introduce a
refracted area-light integral. The existing beam approximation continues to use
its transmitted-branch convention. The camera evaluates all returned exits;
classification retains the existing reflection/transmission convention.

In child mode, the hidden jelly core has no parent transport node. The parent
retained field instead contains the jelly boundary mode. The child's copied
internal material supplies its reduced diffuse response. Stable asset IDs remain
in the scene registry; the hidden core is non-intersectable and excluded from the
active parent transport graph. This is the current homogeneous-child model,
not a general compiler for arbitrary nested private geometry.

## Response reuse and its failure to stay compact

For the prism and jelly, the prototype caches a unit boundary response for each
exact position, direction, spectral coordinate, channel, boundary role, and
material attenuation convention. Power, observer identity, and parent depth are
absent from the key. Internal-source intensity is applied after the cached
response is read. Thin sheets have a direct two-term formula and need no march.

The unit response uses cutoff `1e-12` and a 128-interaction numerical ceiling,
independent of parent population. Unresolved unit weight is retained as a
reported quantity. It is not a universal radiance-error certificate. A cache hit
performs no child march. Sharded locks ensure that concurrent requests compile
one copy. Geometry/material changes replace the affected child definition;
unchanged children and intensity-only changes retain their unit responses.

This meets the narrow reuse test but **fails the intended compact field
representation**. Most boundary states are distinct. The cache stores those
individual answers, so the first aperture-canyon frame creates 1,380,677 new
responses in addition to 879 beam responses. Repeating that exact frame creates
zero new responses, but retains 1,381,556 entries and 1,082,370,368 bytes of payload.
That payload count excludes hash-table and allocator overhead. The complete
four-scene probe reached 1,332,412,416 bytes maximum resident size on the M4.
The exact-state cache has no eviction limit and must not be treated as a scalable
production representation or as continuous region compression.

## M4 screen

All scenes use 800 × 600, the existing visible-edge renderer, terminal error 1,
linear scene traversal, and the retained transport backend. There are 24 renders:
four scenes, both routes, three repeats with alternating order. Scene/beam/field
construction occurs once per route and is reported separately. Every render
constructs its camera-dependent field; child unit responses persist between
repeats. “Warm” below is the median of two repeats, not a confidence interval.

| Scene | Existing first / warm frame | Child first / warm frame | Retained child payload |
|---|---:|---:|---:|
| standard | 206.4 / 204.7 ms | 251.1 / 177.6 ms | 236.5 MB |
| aperture-canyon | 1190.0 / 1183.6 ms | 1549.3 / 1305.2 ms | 1082.4 MB |
| mirror-relay | 835.8 / 824.1 ms | 1029.2 / 758.4 ms | 908.1 MB |
| occlusion-garden | 447.2 / 449.8 ms | 460.2 / 381.8 ms | 302.1 MB |

Construction, before the first render:

| Scene | Existing construction | Child construction |
|---|---:|---:|
| standard | 287.8 ms | 298.3 ms |
| aperture-canyon | 1557.9 ms | 1574.2 ms |
| mirror-relay | 1407.9 ms | 1426.9 ms |
| occlusion-garden | 1550.0 ms | 1575.1 ms |

In aperture-canyon, parent secondary queries fall from 1,556,936 to 1,357,935.
Source quadrature rises from 16,186,047 to 19,572,874. A warm frame performs zero
new child marches and is still slower. The remaining source integrals dominate;
ownership and exact-state caching do not remove those integrals.

An earlier uncached version is preserved in `child_boundary_m4/uncached_boundary_screen.json`.
It avoided the large cache but repeated local marches. An earlier all-exit
classifier screen is also retained as diagnostic history. Neither is the final
cached result in `child_boundary_screen.json`.

## Image interpretation

This is a functional comparison, not a bit-equivalence benchmark. The new route
changes total internal reflection, the child numerical budget, and the jelly's
boundary-mode illumination. The largest changes are visible around the prism,
its reflected image, and the jelly. Mean absolute channel differences are
0.535–0.780 on the 0–255 scale, but maximum differences reach 180–195; the mean
must not be used to hide those local changes. The comparison does not establish
that the complete new image is physically more accurate.

See `child_boundary_m4/comparison.png`, the eight saved PPMs, and `summary.json`.

## Verification

The focused optimized and ASan/UBSan checks pass:

- all five prism boundary faces; outward egress and unit energy accounting;
- independent finite-face transport against the owned supporting-plane response;
- two-sided thin-sheet response;
- 600 straight source transfers and source-program agreement (maximum error zero);
- boundary-role reversal under program reuse;
- independence from mutation of the source geometry registry after compilation;
- exclusion of the hidden core from parent transport and inclusion of the boundary mode;
- intensity/parent-budget reuse without another child march;
- one compilation under concurrent requests;
- affected-child invalidation, unchanged-child retention, and incremental versus
  cold boundary-field agreement.

The existing native scene, update, camera, boundary, and source regressions also
pass. The existing source regression covers 185,700 comparisons, 45,266 zero
certificates, and 67,479 cone tests. These existing regressions use the legacy
fixture configuration; the focused tests exercise child mode explicitly.

## Reproduce

From the authoritative checkout:

```sh
experiments/photonic_transport_field/run_child_boundary.sh
.venv-jpeg/bin/python -m experiments.photonic_transport_field.plot_child_boundary
```

The script selects the M4 host through `m4build`, builds fresh native code, runs
regressions and the screen, and copies `/tmp` results back. The focused Make target
is `make -C experiments/photonic_transport_field/native child-boundary-test`.
A native render can select the prototype with `--child-boundaries on` and the
existing route with `--child-boundaries off`.

The outstanding requirement is a compact continuous boundary/source response
that shares a law across varying states. This prototype establishes the native
ownership interface and demonstrates why per-state response retention is
insufficient. It is not the completed compact-field solution and is not promoted.
