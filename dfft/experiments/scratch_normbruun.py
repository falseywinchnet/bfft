"""Normalized-basis Bruun forward, mirroring src/detail/bruun_kernel.hpp.

Pure-numpy reference for the SAME normalized basis the SIMD kernel uses, so the
radix-p codelets can feed into this exact recursion. Forward is condition-1
(uniform sqrt(2)/level scaling) => FFT-grade, unlike the polydiv references.
"""
import numpy as np

def _bitrev(r, t):
    out = 0
    for _ in range(t):
        out = (out << 1) | (r & 1); r >>= 1
    return out

def _graydecode(g):
    s = 1
    while s < 32:
        g ^= g >> s; s <<= 1
    return g

def bruun_idx(m, L):
    t = m.bit_length() - 1
    r = m ^ (1 << t)
    return (2 * _graydecode(_bitrev(r, t)) + 1) << ((L - 2) - t)

def build_tables(N):
    L = N.bit_length() - 1
    half = N // 2
    C = np.zeros(half)
    IDX = np.zeros(half, dtype=int)
    for m in range(1, half):
        IDX[m] = bruun_idx(m, L)
    if N >= 4:
        C[1] = np.sqrt(0.5)
    def s_tw(m):
        flip = 1 if m > 1 else 0
        return C[m ^ flip]
    m = 1
    while 2 * m < half:
        c = C[m]; s = s_tw(m)
        ce = np.sqrt(0.5 * (1.0 + c)); se = s / (2.0 * ce)
        C[2*m] = ce
        if 2*m + 1 < half:
            C[2*m + 1] = se
        m += 1
    return C, IDX, L, s_tw

def _binomial_fwd(v, h):
    a = v[:h].copy(); b = v[h:2*h].copy()
    v[:h] = a + b; v[h:2*h] = a - b

def _norm_q_fwd(v, off, q, c, s):
    A0 = v[off:off+q].copy(); B0 = v[off+q:off+2*q].copy()
    A1 = v[off+2*q:off+3*q].copy(); B1 = v[off+3*q:off+4*q].copy()
    R = c*B0 - s*B1; I = s*B0 + c*B1
    v[off:off+q] = A0 + R
    v[off+q:off+2*q] = A1 + I
    v[off+2*q:off+3*q] = A0 - R
    v[off+3*q:off+4*q] = -A1 + I

def rfft_normbruun(x):
    x = np.asarray(x, dtype=np.float64)
    N = x.shape[0]
    C, IDX, L, s_tw = build_tables(N)
    v = x.copy()
    for jj in range(0, L - 1):
        s = N >> jj; h = s >> 1; q = s >> 2; m_end = 1 << jj
        _binomial_fwd(v, h)
        for m in range(1, m_end):
            _norm_q_fwd(v, m*s, q, C[m], s_tw(m))
    X = np.zeros(N//2 + 1, dtype=np.complex128)
    X[0] = v[0] + v[1]
    X[N//2] = v[0] - v[1]
    for m in range(1, N//2):
        X[IDX[m]] = v[2*m] - 1j * v[2*m+1]
    return X

if __name__ == "__main__":
    print(f"{'N':>7} {'normbruun_vs_numpy':>20}")
    for Lp in range(2, 16):
        N = 1 << Lp
        x = np.random.default_rng(N).standard_normal(N)
        e = np.abs(rfft_normbruun(x) - np.fft.rfft(x)).max() / max(np.abs(np.fft.rfft(x)).max(),1)
        print(f"{N:>7} {e:>20.2e}  {'OK' if e<1e-13 else 'FAIL'}")
