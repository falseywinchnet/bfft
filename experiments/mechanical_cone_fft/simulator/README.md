# The Fourier Machine — standalone mechanical FFT simulator

Open **[index.html](index.html)** in a browser. It is self-contained: no packages,
CDNs, fonts, images, network calls, backend or build step are required to use it.
It can also be served as a normal static page.

The mathematical completion is in
[Exact finite-motion completion](../../../notes/mechanical_fft_finite_linkage.md).
The model preserves the original eight-sample, three-stage, 44-bar Bruun
factorization from `../certificate.py`.

## Interaction

- Eight fixed-pivot input rockers at the top, each in [-1,1]. Drag the white endpoint vertically, or focus it and use arrow keys, Home/End or Page Up/Down. The two endpoint pins drive the positive and negative rails.
- Wave, impulse, alternating, constant and zero presets.
- The assembly runs vertically through 44 bars and 60 rigid branched transfer yokes. Eight moving differential rulers at the bottom measure the Fourier coordinates; the five complex coefficient cards follow below.
- Select any bar in the drawing or the accessible selector to inspect its
  sliding pins, fixed tap, length, rotation and weighted displacement.
- Selecting a bar highlights its upstream transmission path. **Show all**
  restores the entire network.
- **Slow motion** slows the input movement. It does not independently ease
  intermediate bars, so all constraints remain satisfied during animation.
- **Travel the wave** moves a periodic waveform through the sample controls;
  the output phasors rotate accordingly. Manual input stops the demo.
- The full network fits the available width. **Enlarge joints** enables a detailed horizontally scrollable view. Presets, inspector and bin readouts reflow for phones. Reduced-motion preference
  removes the automatic transition easing; wave playback is opt-in.

The bottom readout and spectrum cards show the ordinary unnormalized FFT.
The first and last bins are real. The three omitted bins are the conjugate
mirror of bins 3,2,1.

## Put it on the website

Copy `index.html` into the site's static assets, for example as
`fourier-machine.html`. It can be a standalone page or an iframe:

```html
<iframe
  src="/fourier-machine.html"
  title="Interactive mechanical Fourier transform"
  width="100%"
  height="1500"
  loading="lazy"
  style="display:block;border:0;border-radius:8px"
></iframe>
```

Set the containing page's iframe height to fit its layout; the simulator also
supports ordinary internal scrolling. No publishing or site configuration was
changed by creating this file.

For a site with a restrictive CSP, host `style.css`, `mechanics.js`, `assembly.js` and `app.js`
separately and replace the four inline blocks in `page.html` with ordinary
stylesheet/script references. Do not weaken the site's CSP just to accept the
self-contained bundle.

## Source and reproduction

- `mechanics.js`: signed Bruun stages, cone lift, kinematic gauge, binary-bar
  construction, exact finite rigid geometry and calibrated output readout.
- `assembly.js`: connected yokes, fixed-pivot input rockers, shared pass-through members, finite slot geometry and moving differential rulers.
- `app.js`: controls, animation of prescribed inputs, SVG linkage and output
  display. The direct DFT is only an independent check.
- `page.html` and `style.css`: page source.
- `build.py`: combines these sources into the single `index.html`.
- `test_mechanics.cjs`: model tests using only Node's standard library.
- `test_assembly.cjs`: connected pin/slot, rigid-yoke, rocker and physical ruler certificates.
- `validation.json` and `assembly_validation.json`: recorded numerical test results.

```sh
node experiments/mechanical_cone_fft/simulator/test_mechanics.cjs
node experiments/mechanical_cone_fft/simulator/test_assembly.cjs
python3 experiments/mechanical_cone_fft/simulator/build.py
```

The model tests are tiny and can run locally. The original NumPy research
certificate can be run on the Mini:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 experiments/mechanical_cone_fft/certificate.py
```

## What is physically modeled

Each bar is a rigid unit-length member. A vertical guide fixes its lateral
centre; endpoint pins slide in horizontal slots on vertical shuttles. A fixed
material tap drives its own vertically moving shuttle through a horizontal
slot. At input travels a,b, its pose is determined by the fixed-length
constraint. Its tap height, divided by the travel scale, gives the output.

Each transfer is now an explicitly connected rigid yoke: the source slot,
vertical stem, horizontal crosspiece, branches and receiving slots all translate
by exactly the same amount. All interpolation bars use the same physical travel
scale. The 12 copy operations reuse an existing yoke; they introduce no gap or
unconnected receiver. Source and destination pins stay in their slots throughout
motion. The outer frame and crossbeams support the vertical guide bearings.

The front view projects 60 distinct yoke depth planes. Ringed axle pins connect
front-plane bars to the appropriate yoke slots. A projected crossing is not a
joint. This gives an explicit connected kinematic topology; it is not a
finite-thickness collision certificate for the long axles and layered members.

At the bottom, the gold scale is rigidly attached to the negative output yoke;
the cyan pointer is rigidly attached to the positive output yoke. Their relative
height, with the fixed gauge calibration, supplies the signed FFT coordinate.

The simulator is quasistatic. It does not claim to integrate masses, springs,
damping or physical settling times. Friction, compliance, inertia, tolerances
and fabrication remain outside this exact kinematic model.

## Checks performed

1,266 numerical input cases pass, including every sign-box corner. The maximum
absolute FFT error is approximately 3.86e-15. Bar length error is at most
2.22e-16, and fixed tap fraction error at most 1.11e-16. Intermediate stages,
Parseval energy, stochastic row sums, rail bounds and the [16,24,4] bar census
are independently checked. The original NumPy certificate also passes on M4.

Browser verification covered desktop and phone layouts, impulse and alternating
presets, keyboard input at both slider limits, selected-bar inspection, endpoint
slot bounds at full rotation, and wave playback. No browser console errors were
reported during these checks.

The connected assembly additionally passes 1,257 cases (all 256 corners,
1,000 seeded random inputs, and zero). Maximum pin/slot vertical mismatch and
rigid-yoke segment-length error are each 1.14e-13 SVG units. The displayed
physical ruler displacement recovers the direct DFT within 5.69e-14, including
coordinate subtraction at the bottom of the page. Input rocker lengths and
fixed pivots, slot clearances, and every source-to-receiver attachment are checked.
