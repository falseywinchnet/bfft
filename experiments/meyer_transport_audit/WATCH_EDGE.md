# Watching one synthetic edge

`watch_edge.py` records the ordinary six-field recurrence on a periodic 1 × 64
signal: 50 through sample 32, then 200, with lambda = 0.05 and mu = 40.
Periodicity supplies a second edge at the wrap. Pass 1 is `initial()`;
the recording includes 319 subsequent steps. Recording the 320 states and
diagnostics took 0.06244 seconds on the M4 Mini, excluding initialization.
The interactive replay slows passes 64–280 to five seconds.

The first projection-pattern change after pass 64 is at pass 214, in the u
branch at sample 33. This site describes a shoulder beside the source edge:
the source values at samples 33 and 34 are both 200. Its retained field obeys
`t = Du + incoming_b`. The incoming field remains 10 while the shoulder's
gradient declines from 29.05078 at pass 64 to 0.24050 at pass 212.
The projected field `p = clip(t, -10, 10)` remains 10 throughout this interval.
At pass 214, t reaches 9.85411 and the projection enters its interior.
The gap per sample drops from 0.11259 at pass 213 to 0.08389 at pass 214;
subsequent changes need not be monotone (0.08477 at pass 215).

Let h_k = z_(k+1) - z_k and a_k = h_(k+1) - h_k, using all six fields.
At pass 212, RMS(h) = 0.035673 and ||a|| / ||h|| = 0.000277565.
Thus the transport is nearly steady while remaining nonzero. A small second
difference is not a convergence certificate. In this case the recurrence
keeps removing a shoulder under the same saturated response until the
projection changes regime. This interpretation concerns the whole coupled
recurrence; the tracked site makes its progress visible but is not an
independent scalar solver.

A diagnostic constant-rate estimate of the next clipping boundary, using all
128 scalar projection sites, predicted pass 213.64 from pass 128; the actual
event was pass 214. The same estimate at pass 64 incorrectly predicted a w
event near pass 193. These observations suggest examining regime lifetime
from carried transport, but do not certify a jump or establish acceleration.
No extrapolation, factors, or altered solver were used for this recording.

`meyer_watch_edge.json` contains the full-precision profiles, gaps, projection
events and selected measurements. `meyer_watch_edge.npz` contains the complete
320 × 6 × 64 state array, ordered u, w, tux, tuy, twx, twy. The model and gap
definitions are in `model.py` and `certificate.py`.

Reproduce on the Mini:

```sh
/Users/ultimussecundai/.local/bin/m4build -- \
  python3 experiments/meyer_transport_audit/watch_edge.py \
  --out /tmp/meyer_watch_edge
```

Copy both `/tmp/meyer_watch_edge.json` and `/tmp/meyer_watch_edge.npz` back
immediately using the host selected by `m4host`.
