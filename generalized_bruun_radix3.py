"""Generalized Bruun radix-3 split, correctness-first reference."""
import numpy as np

def _split_quarters(r):
    m = len(r); q = m // 4
    return r[:q], r[q:2*q], r[2*q:3*q], r[3*q:4*q]

def _quad_child_remainder(r, child_angle):
    r0, r1, r2, r3 = _split_quarters(r)
    c = np.cos(child_angle)
    lo = r0 - r2 - 2.0 * c * r3
    hi = r1 + 2.0 * c * r2 + (4.0 * c * c - 1.0) * r3
    return np.concatenate([lo, hi])

def _eval_quad_factor(r, m, theta, N, X):
    if m == 2:
        a = r[0]; b = r[1]; z = np.exp(-1j * theta)
        k = int(round(theta * N / (2.0 * np.pi))) % N
        X[k] = a + b * z
        if k != 0 and (2*k) % N != 0:
            X[(-k) % N] = np.conj(X[k])
        return
    left = theta / 2.0; right = np.pi - theta / 2.0
    _eval_quad_factor(_quad_child_remainder(r, left), m // 2, left, N, X)
    _eval_quad_factor(_quad_child_remainder(r, right), m // 2, right, N, X)

def fft_radix3_power2_tail_real(x):
    x = np.asarray(x, dtype=np.float64); N = x.size
    assert N % 3 == 0
    M = N // 3
    assert M >= 1 and (M & (M - 1)) == 0
    a = x[:M]; b = x[M:2*M]; c = x[2*M:3*M]
    X = np.zeros(N, dtype=np.complex128)
    s = a + b + c
    X[0::3] = np.fft.fft(s)
    if M == 1:
        r = np.array([a[0] - c[0], b[0] - c[0]], dtype=np.float64)
        _eval_quad_factor(r, 2, 2.0*np.pi/3.0, N, X)
    else:
        r = np.concatenate([a - c, b - c])
        _eval_quad_factor(r, 2*M, 2.0*np.pi/3.0, N, X)
    return X

def _max_rel(a, b):
    return np.max(np.abs(a-b)) / max(1.0, np.max(np.abs(b)))

if __name__ == "__main__":
    rng = np.random.default_rng(123)
    print("N,M,max_rel_err")
    for a in range(0, 13):
        M = 1 << a; N = 3 * M; worst = 0.0
        for _ in range(5):
            x = rng.standard_normal(N)
            worst = max(worst, _max_rel(fft_radix3_power2_tail_real(x), np.fft.fft(x)))
        print(f"{N},{M},{worst:.3e}")
