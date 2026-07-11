try the douglas-rachford upgrade on the cross-hatch region

Douglas–Rachford on the cross-hatch: the two sets are the max-consensus magnitude variety (nonconvex, projector = radial + OLA) and positivity (convex, projector = clip). Instead of alternating projections, DR iterates through reflections — x ← x + P_C(2P_M(x)−x) − P_M(x) — which is what lets it walk out of the AP fixed-point traps where the doubly-destroyed band's phase is stuck. I'll also try RAAR (the relaxed variant), and measure the cross-hatch where it lives: the spectral band where both motion-blur OTFs are small:


Created a file, ran 3 commands, edited 2 files

Recalled a memory
The Douglas–Rachford upgrade is measured, and the answer is a decisive — and unusually clean — null that resolved into something better. Full record in §4e, error panels at two_lattice_dr.png.

DR does not fix the cross-hatch, because there is nothing to fix algorithmically. DR (100 iterations), RAAR at β=0.75 and 0.9, DR-then-AP-polish, every variant lands within +0.07 dB of plain alternating projections (26.30 → 26.37 at best), with the dead-band error unchanged. There is no stagnation to escape — AP was already sitting at the information-limited point.

The diagnosis fell out of the band-resolved metric: with τ=0.3, the band where both box OTFs are small covers 74.5% of the frequency plane, and its relative error is 1.04 — which decomposes as 1.00 (content entirely missing: neither capture measured it) plus ~0.04 (a small Rayleigh-max bias from taking the max of two noise-level magnitudes). Meanwhile the live band is essentially solved (0.0017). The hatch is the structured absence of an unmeasured band, textured by the sharp boundary of the coverage region — not enforced garbage, not a local trap.

I also tested the theory-completing idea the null suggested: an absolute reliability gate (t²/(t² + (κ·floor)²), noise floor estimated from the outer frequency shell — softmax weights say whom to trust, the gate says whether to trust anyone). Conceptually it's the right final piece of the weighted-projection framework, and it goes in the note as such — but empirically it's invisible here, because the bias it removes is only that 0.04.

The constructive resolution, measured: add measurement, not machinery. A third capture with 45° motion blur covers part of the corner both others kill — original-dead-band error 1.041 → 0.813, PSNR 26.30 → 26.41, using the identical AP operator that all the fancy splitting couldn't improve. Only new information moved the number. And the shape of the fix is telling: angular blur diversity samples the frequency plane like tomographic projections, so the capture-design rule is coverage of the joint dead zone — which is exactly the audio ladder's third-aperture argument wearing image clothes.

So the session's law survives intact and gets sharper: operators polish toward what the constraints know; complementarity decides what they know; and when a residual refuses to move under better operators, that refusal is itself the measurement — it's telling you where your captures are jointly blind.

if these additional capture angles are also necessary to get high level SR in, for example, the waterfall domain, we should note the next series of research tasks- in stft, in waterfall, in image experiments- and add that as the bold next project to work on.

The premise deserves a measurement before it becomes a charter. The audio analog of "capture angles" is precise: a window of length N pins a packet pair's relative phase only if the pair shares an analysis cell — Δt ≲ N and Δf ≲ 1/N. Two windows cover two slices of the Δt·Δf ≲ 1 hyperbola; pairs in between — separated too long for the short window, too far apart in frequency for the long — are the waterfall's dead zone, and the intermediate aperture is the diagonal blur. Let me verify with the gauge probe:



^

the next chapter in this viewer saga. 6/10/26. until we meet again?
