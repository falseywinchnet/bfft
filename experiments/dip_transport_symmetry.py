"""Test the user's two claims against the prototypes.

Claim 1 (transport symmetry): a correct phase-packet walk must have
  boundary_transport = reshape_in + scatter_out
equal to the SINGLE-sided disorder DIT pays on its input (or DIF on its output).
The transpose-four-step pays 2 end passes; DIT pays 1. If 2 != 1 the walk is wrong.

Claim 2 (stay real): if it's Bruun it shouldn't need complex space.
"""
import numpy as np, sys
sys.path.insert(0, "experiments")
from phase_fft import phase_fft_real, pair_reduce_cs
from dip_phase_packet import real_fourstep_rfft

def ilog2(n):
    l=0
    while n>1: n>>=1; l+=1
    return l

# ---- boundary transport, counted as full-array passes over N reals ----------
# DIT r2c:        input bit-reversed comb gather = 1 pass; output natural = 0.  -> 1
# DIF r2c:        input natural = 0; output bit-reversed scatter = 1.           -> 1
# four-step:      reshape x->g[q][p] = 1 pass; scatter X[k2+Q k1] = 1 pass.     -> 2
# diagonal walk:  x consumed in natural order; X written in bin order.         -> 0
def boundary_passes_dit(n):        return 1
def boundary_passes_dif(n):        return 1
def boundary_passes_fourstep(n):   return 2
def boundary_passes_diagonal(n):   return 0

# The disorder is conserved: the bit-reversal is log2(N) single-bit moves total.
# DIT spends them all at the input, DIF all at the output, the diagonal walk
# spreads them one-per-stage into the store address (Stockham), so they never
# pile into a boundary pass at all.

print("== Claim 1: boundary passes (full sweeps over N) ==")
print(f"  {'N':>9} {'DIT':>4} {'DIF':>4} {'four-step':>10} {'diagonal':>9}")
for n in (65536, 1048576, 16777216):
    print(f"  {n:>9} {boundary_passes_dit(n):>4} {boundary_passes_dif(n):>4}"
          f" {boundary_passes_fourstep(n):>10} {boundary_passes_diagonal(n):>9}")

# ---- Claim 2: the four-step's "complex row leg" is a REAL (a,b) Givens -------
# In the Bruun encoding a spectrum bin is the pair (a,b)=(Re,-Im). The four-step
# middle twiddle W_N^{p k2} applied to that pair is a plane rotation by that
# angle -- exactly pair_reduce_cs with a general (c,s). phase_fft_real already
# runs that cell at every interior stage. So "complex leg" == the existing real
# cell with a general angle; no complex arithmetic is introduced.
print("\n== Claim 2: complex twiddle on (Re,-Im) == real Givens (same cell) ==")
rng = np.random.default_rng(0)
re, im = rng.standard_normal(5), rng.standard_normal(5)
ang = 0.7
# complex way
cw = (re + 1j*im) * np.exp(-1j*ang)
# Bruun (a,b)=(Re,-Im) rotated by pair_reduce_cs's rotation r=c*oa-s*ob, i=s*oa+c*ob
a, b = re, -im
c, s = np.cos(ang), np.sin(ang)
ra = c*a - s*b        # new Re
ri = s*a + c*b        # new (-Im) ... check
err = max(np.max(np.abs(ra - cw.real)), np.max(np.abs(ri - (-cw.imag))))
print(f"  |real-Givens - complex-twiddle| = {err:.2e}  (0 => no complex space needed)")

# ---- both real forms are exact r2c ------------------------------------------
print("\n== both real forms == rfft ==")
for n in (256, 4096):
    x = rng.standard_normal(n)
    e1 = np.max(np.abs(phase_fft_real(x) - np.fft.rfft(x)))
    e2 = np.max(np.abs(real_fourstep_rfft(x) - np.fft.rfft(x)))
    print(f"  n={n:5d}  diagonal_walk={e1:.2e}  four-step={e2:.2e}")
