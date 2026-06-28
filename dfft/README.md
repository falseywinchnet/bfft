# DFFT: dragon BFFT workspace

DFFT is a full experimental copy of the BFFT repository.  It exists so the
real-to-complex and complex-to-real double paths can be redesigned around
explicit boundary ownership without destabilizing the clean BFFT baseline at the
repository root.

Start with [`DRAGON_BOUNDARIES.md`](DRAGON_BOUNDARIES.md).  That file names the
front-to-back pathway boundaries, the control surfaces DFFT should take away
from implicit compiler/codegen behavior, and the benchmark loop required before
any dragon mechanism graduates.

The root BFFT remains the reference implementation.  The `dfft/` subtree is the
place for invasive scheduling, arena, block-cursor, lane-pack, and queue-control
experiments.
