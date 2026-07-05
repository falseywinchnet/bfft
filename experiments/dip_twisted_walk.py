"""The twisted DIP: helical phase-packet walks on the (time, frequency) torus.

MISSION (2026-07-05): the DIP's dyadic descent (d,e) -> (d,2e),(e-d,2e) walks
straight across the phase toroid (radial only). Find the legal SPIRALS: walks
that also drift the rank/frame coordinate per stage, changing the global
packet order (bracketing, mirror-merge geometry) while preserving the sparse
real Bruun cell.

The candidate twists are the admissible shear gauge family already proven for
the COMPLEX walk (experiments/phase_fft.py): frame shear sigma_t at level
e = 2^t, admissible iff sigma * (n/e)^2 == 0 (mod n) -- the freedom diamond
2^min(t, m-t). This file answers, numerically and combinatorially:

  Q1  which shears survive the REAL (Bruun residue-pair) fold?  The fold
      pairs conjugate packets; in the sheared frame conjugation maps
      (kappa, j) -> (-kappa + 2*sigma*j (mod e), j): the partner is
      j-INDEPENDENT (span-local fold possible) iff 2*sigma == 0 (mod e),
      i.e. sigma in {0, e/2}.  Conjecture: the fold-legal spiral is
      quantized to HALF-TURNS -- one bit per level.
  Q2  a half-turn at level e relabels the folded diagonal d -> e/2 - d ...
      which permutes the SAME child set {d, e-d}: the twist collapses to a
      per-level (or per-node) SIBLING ORIENTATION choice in the span walk.
      Verify: twisted walks are exact for every orientation word (same
      cells, relabeled placement).
  Q3  what do spirals BUY?  Search orientation words for: leaf-order
      clustering of low-frequency bins (band segregation), emission travel,
      mirrored sibling geometry.  Also PROVE the invariant: the leaf SET of
      any subtree is the comb {j*e +/- d} in every legal gauge (bands are
      unreachable by ancestry; only the ORDER moves).
  Q4  the mirror-merge dividend that needs no twist at all: sibling spans
      (d, 2e) and (e-d, 2e) have complementary angles -- cos/sin SWAP.
      Half of all twiddle loads are redundant (the DIT's mirrored-lane
      sharing, present in the DIP tree for free).
"""
import numpy as np

# ============================================================ Q1: fold scan
def packet_state(x, t, sigma):
    """Sheared packet state A^sigma_t[kappa, j] for real input x (plain DFT
    packets sheared per phase_fft.py's verified monomial law)."""
    n = len(x)
    e = 1 << t
    q = n // e
    g = x.reshape(e, q)                       # g[r, j] = x[j + r*q]
    A = np.fft.fft(g, axis=0)                 # plain: A[k, j], k in [0, e)
    c = (sigma * q * q) // n                  # admissibility integer
    out = np.empty_like(A)
    for j in range(q):
        for kappa in range(e):
            k = (kappa - sigma * j - c * (e // 2)) % e
            out[kappa, j] = np.exp(1j * np.pi * sigma * j * j / n) * A[k, j]
    return out

def conj_partner_map(x, t, sigma):
    """For each (kappa, j) find (kappa', j') with A[kappa', j'] ~= conj(A[kappa, j])
    up to a unit phase. Returns True iff the partner is row-local (same j) and
    j-independent in kappa (a span-local fold exists)."""
    A = packet_state(x, t, sigma)
    e, q = A.shape
    kmap = np.full((e, q), -1)
    for j in range(q):
        for kappa in range(e):
            tgt = np.conj(A[kappa, j])
            # partner must be in the same column for a span-local fold
            col = A[:, j]
            mags = np.abs(np.abs(col) - np.abs(tgt))
            cand = np.where(mags < 1e-9 * (1 + np.abs(tgt)))[0]
            hit = -1
            for kp in cand:
                if abs(tgt) < 1e-12:
                    hit = kp; break
                ph = col[kp] / tgt
                if abs(abs(ph) - 1) < 1e-9:
                    hit = kp; break
            kmap[kappa, j] = hit
    if (kmap < 0).any():
        return False, kmap
    # j-independence: kappa' must not depend on j
    return bool((kmap == kmap[:, :1]).all()), kmap

def scan_fold_legal(n=64):
    m = n.bit_length() - 1
    rng = np.random.default_rng(3)
    x = rng.standard_normal(n)
    print(f"== Q1: fold-legal shears per level (n={n}) ==")
    print("   level e | admissible diamond | fold-legal sigmas")
    for t in range(1, m):
        e = 1 << t
        q = n // e
        step = max(1, n // (q * q))           # admissibility: sigma multiple of step
        legal = []
        for sigma in range(0, e, step):
            ok, _ = conj_partner_map(x, t, sigma)
            if ok:
                legal.append(sigma)
        print(f"   e={e:4d}   | multiples of {step:4d} | {legal}   (predict {{0, {e//2}}} ∩ admissible)")

# ==================================== Q2/Q3: twisted span walk, leaf orders
def leaves(d, e, w, orient):
    """Leaf bins of span (d, e) with w leaves under orientation function
    orient(d, e) in {0, 1}: 0 = (lo=d, hi=e-d), 1 = swapped placement."""
    if w == 1:
        return [d]
    lo = leaves(d, 2 * e, w // 2, orient)
    hi = leaves(e - d, 2 * e, w // 2, orient)
    return lo + hi if orient(d, e) == 0 else hi + lo

def full_leaf_order(n, orient):
    """Leaf (bin) order of the whole forward walk: driver order
    [ridge spawns...] [B2] [B1] [B3] as in the kernel, level-8 seed."""
    q = n // 8
    out = []
    # ridge chain: dc/ny handled separately; spawns bins (e/2, 2e) with w = q2
    def ridge(qq, e):
        if qq == 1:
            return
        ridge(qq // 2, 2 * e)
        out.extend(leaves(e // 2, 2 * e, qq // 2, orient))
    # NOTE: kernel driver order is ridge, B2, B1, B3; ridge recursion emits
    # deepest spawn first here -- order within ridge does not affect scores.
    ridge(q, 8)
    for d in (2, 1, 3):
        out.extend(leaves(d, 8, q, orient))
    return out

def comb_invariance_check(n=256):
    """Q3 invariant: the leaf SET of every span is {j*e +/- d} regardless of
    orientation. Bands are unreachable; only order moves."""
    rng = np.random.default_rng(0)
    ok = True
    for trial in range(20):
        bits = rng.integers(0, 2, size=1 << 16)
        orient = lambda d, e: int(bits[(d * 2654435761 + e) % len(bits)])
        for (d, e, w) in ((1, 8, n // 8), (3, 8, n // 8), (2, 16, n // 16)):
            got = sorted(leaves(d, e, w, orient))
            want = sorted(set([(r + 1) // 2 * e + d if r % 2 == 0 else (r + 1) // 2 * e - d
                               for r in range(w)]))
            if got != want:
                ok = False
    print(f"== Q3 invariant: subtree leaf sets are the combs {{j*e +/- d}} for all orientations: {ok} ==")

def score_order(order, n):
    """Scores: (a) travel = mean |gap| between consecutive emitted bins;
    (b) low-band clustering: number of contiguous position-intervals holding
    all bins < n/8 (lower = more segregated); (c) longest monotone run."""
    arr = np.array(order)
    travel = float(np.mean(np.abs(np.diff(arr))))
    low = arr < (n // 8)
    intervals = int(np.sum(np.diff(low.astype(int)) == 1) + (1 if low[0] else 0))
    runs, cur, best = 1, 1, 1
    for i in range(1, len(arr)):
        cur = cur + 1 if arr[i] > arr[i - 1] else 1
        best = max(best, cur)
    return travel, intervals, best

def search_orientation_words(n=1024):
    """Global per-level words (one bit per level): exhaustive.  The word IS
    the spiral: bit t = half-turn of the frame at level 2^t."""
    m = n.bit_length() - 1
    depth = m - 3 + 1                     # levels 8..n
    print(f"\n== Q3: exhaustive per-LEVEL spiral words, n={n} ({depth} bits) ==")
    base = None
    results = []
    for word in range(1 << depth):
        orient = lambda d, e, w=word: (w >> (e.bit_length() - 4)) & 1
        order = full_leaf_order(n, orient)
        tr, iv, run = score_order(order, n)
        results.append((tr, iv, run, word))
        if word == 0:
            base = (tr, iv, run)
    results.sort()
    print(f"   untwisted (word 0): travel={base[0]:8.1f}  low-band intervals={base[1]:3d}  best run={base[2]}")
    print("   best by travel:")
    for tr, iv, run, word in results[:4]:
        phi = bin(word)[2:].zfill(depth)[::-1]
        print(f"     word={phi} (LSB=level 8): travel={tr:8.1f}  intervals={iv:3d}  run={run}")
    ivs = sorted(results, key=lambda r: (r[1], r[0]))
    print("   best by low-band clustering:")
    for tr, iv, run, word in ivs[:4]:
        phi = bin(word)[2:].zfill(depth)[::-1]
        print(f"     word={phi}: intervals={iv:3d}  travel={tr:8.1f}  run={run}")
    return results

def search_per_node(n=1024):
    """Per-NODE orientation, chosen greedily to minimize emission travel:
    at each node, place the child whose leaf set continues closest to the
    previous emitted bin. (The DIF's heapopt orientation DP, DIP edition.)"""
    print(f"\n== Q3: greedy per-NODE orientation, n={n} ==")
    def emit(d, e, w, prev):
        if w == 1:
            return [d]
        lo_first = leaves(d, 2 * e, w // 2, lambda *_: 0)
        hi_first = leaves(e - d, 2 * e, w // 2, lambda *_: 0)
        # choose orientation minimizing the jump from prev
        if abs(lo_first[0] - prev) <= abs(hi_first[0] - prev):
            a = emit(d, 2 * e, w // 2, prev)
            b = emit(e - d, 2 * e, w // 2, a[-1])
        else:
            a = emit(e - d, 2 * e, w // 2, prev)
            b = emit(d, 2 * e, w // 2, a[-1])
        return a + b
    q = n // 8
    out = []
    def ridge(qq, e):
        if qq == 1:
            return
        ridge(qq // 2, 2 * e)
        out.extend(emit(e // 2, 2 * e, qq // 2, out[-1] if out else 0))
    ridge(q, 8)
    for d in (2, 1, 3):
        out.extend(emit(d, 8, q, out[-1] if out else 0))
    tr, iv, run = score_order(out, n)
    print(f"   greedy per-node: travel={tr:8.1f}  low-band intervals={iv:3d}  best run={run}")
    return out

# ======================================= Q2: numeric exactness of the twist
def pair_reduce_cs(ea, eb, oa, ob, c, s):
    r = c * oa - s * ob
    i = s * oa + c * ob
    return ea + r, eb + i, ea - r, i - eb

def twisted_rfft(x, orient):
    """In-place-style span walk with per-node orientation: identical cells,
    swapped child placement. Bins extracted by tracking labels."""
    n = len(x)
    q = n // 8
    # level-8 seed (plain DFT of stride-q combs, folded)
    g = x.reshape(8, q)
    S = np.fft.fft(g, axis=0)
    spans = {}
    spans['dc'] = S[0].real.copy()
    spans['ny'] = S[4].real.copy()
    for d in (1, 2, 3):
        spans[d] = (S[d].real.copy(), -S[d].imag.copy())
    X = np.zeros(n // 2 + 1, complex)

    def descend(a, b, d, e):
        w = len(a)
        if w == 1:
            X[d] = a[0] - 1j * b[0]
            return
        w2 = w // 2
        th = np.pi * d / e
        c, s = np.cos(th), np.sin(th)
        la, lb, ha, hb = pair_reduce_cs(a[:w2], b[:w2], a[w2:], b[w2:], c, s)
        if orient(d, e) == 0:
            descend(la, lb, d, 2 * e, ), descend(ha, hb, e - d, 2 * e)
        else:
            descend(ha, hb, e - d, 2 * e), descend(la, lb, d, 2 * e)

    def descend_ridge(dc, ny, e):
        qq = len(dc)
        if qq == 1:
            X[0] = dc[0]
            X[n // 2] = ny[0]
            return
        q2 = qq // 2
        ndc = dc[:q2] + dc[q2:]
        nny = dc[:q2] - dc[q2:]
        descend_ridge(ndc, nny, 2 * e)
        descend(ny[:q2].copy(), ny[q2:].copy(), e // 2, 2 * e)

    descend_ridge(spans['dc'], spans['ny'], 8)
    for d in (1, 2, 3):
        descend(spans[d][0], spans[d][1], d, 8)
    return X

def exactness_scan(n=1024, words=8):
    rng = np.random.default_rng(1)
    x = rng.standard_normal(n)
    ref = np.fft.rfft(x)
    print(f"\n== Q2: twisted-walk exactness, n={n} (random orientation words) ==")
    for trial in range(words):
        bits = rng.integers(0, 2, size=1 << 16)
        orient = lambda d, e: int(bits[(d * 2654435761 + e * 40503) % len(bits)])
        err = np.max(np.abs(twisted_rfft(x, orient) - ref))
        print(f"   word {trial}: max err = {err:.2e}")

# ================================ Q4: the free mirror twiddle-sharing check
def mirror_twiddle_check(n=4096):
    print(f"\n== Q4: sibling spans share swapped twiddles (n={n}) ==")
    ok = True
    for e in (8, 16, 64, 512):
        for d in range(1, e // 2):
            tl = np.pi * d / (2 * e)                # theta(d, 2e)
            th = np.pi * (e - d) / (2 * e)          # theta(e-d, 2e)
            if not (abs(np.cos(th) - np.sin(tl)) < 1e-15 and
                    abs(np.sin(th) - np.cos(tl)) < 1e-15):
                ok = False
    print(f"   cos(theta(e-d,2e)) == sin(theta(d,2e)) and vice versa: {ok}")
    print("   => half the twiddle table loads in span8/cell4 are redundant")
    print("      (the DIT's mirrored-lane sharing, present in the DIP for free).")

if __name__ == "__main__":
    scan_fold_legal(64)
    comb_invariance_check(256)
    exactness_scan(1024)
    search_orientation_words(1024)
    search_per_node(1024)
    mirror_twiddle_check()
