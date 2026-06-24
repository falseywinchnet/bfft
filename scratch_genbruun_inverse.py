"""Inverse of the generalized Bruun (arbitrary-N) real FFT -- Python reference.

Reverses the forward schedule of scratch_genbruun_exact.py op by op (postorder:
gather children/leaves from the spectrum, combine up to the parent residue).
Each step is the inverse of a condition-1 forward step:

  pow2 minus-node   -> numpy.fft.irfft (C++ will use bruun::RFFT::inverse)
  rooted cascade    -> reverse butterflies: A0=(L+R)/2, R=(L-R)/2, A1=(L1-R1)/2,
                       I=(L1+R1)/2, [B0;B1]=[c s; -s c][R;I]
  minus-split proj  -> p-point real inverse DFT across blocks (scale 2/p, 1/p DC)
  bruun-odd proj    -> p-point complex inverse DFT across blocks (scale 1/p)
"""
import numpy as np
from fractions import Fraction as Fr
from scratch_genbruun_exact import _cos, _sin, _is_pow2, _odd_prime


def irfft_gen(Xhalf, N):
    Xhalf = np.asarray(Xhalf, dtype=np.complex128)
    # full Hermitian spectrum for gathering
    Xf = np.zeros(N, dtype=np.complex128)
    Xf[:N // 2 + 1] = Xhalf
    for k in range(1, (N + 1) // 2):
        Xf[N - k] = np.conj(Xhalf[k])

    def gather(f):                       # leaf value cd at bin round(N*f)
        k = int(round(N * float(f % 1))) % N
        return Xf[k]

    # inverse rooted cascade: return residue (length d) for node angle f
    def inv_nb(f, d):
        if d == 2:
            v = gather(f)
            return np.array([v.real, -v.imag])
        q = d // 4
        cL = inv_nb(f / 2, d // 2)
        cR = inv_nb(Fr(1, 2) - f / 2, d // 2)
        c = _cos(f / 2); s = _sin(f / 2)
        A0 = 0.5 * (cL[:q] + cR[:q]); R = 0.5 * (cL[:q] - cR[:q])
        A1 = 0.5 * (cL[q:] - cR[q:]); I = 0.5 * (cL[q:] + cR[q:])
        B0 = c * R + s * I; B1 = -s * R + c * I
        return np.concatenate([A0, B0, A1, B1])

    # inverse Bruun node (cascade frame): return (lo, hi) each length M
    def inv_bruun(M, f):
        if M == 1:
            v = gather(f)
            return np.array([v.real]), np.array([-v.imag])
        if _is_pow2(M):
            seed = inv_nb(f, 2 * M)
            return seed[:M], seed[M:]
        p = _odd_prime(M); Mp = M // p
        lo = np.zeros(M); hi = np.zeros(M)
        for t in range(p):
            phi = (f + t) / p
            clo, chi = inv_bruun(Mp, phi)
            for i in range(p):
                cc = _cos(i * phi); ss = _sin(i * phi)   # cos/sin(2*pi*i*phi)
                # A_i = (1/p) sum_t (clo cos + chi sin); B_i = (1/p) sum_t (chi cos - clo sin)
                lo[i * Mp:(i + 1) * Mp] += (clo * cc + chi * ss) / p
                hi[i * Mp:(i + 1) * Mp] += (chi * cc - clo * ss) / p
        return lo, hi

    # inverse minus-node: return residue (length D)
    def inv_minus(D, sigma):
        if _is_pow2(D):
            if D == 1:
                return np.array([Xf[0].real])
            Xloc = np.array([Xf[(sigma * k) % N] for k in range(D // 2 + 1)], dtype=np.complex128)
            return np.fft.irfft(Xloc, D)
        p = _odd_prime(D); M = D // p
        s = inv_minus(M, sigma * p)                  # DC block-sum = DFT[0]
        los = []; his = []
        for j in range(1, (p - 1) // 2 + 1):
            g = Fr(j, p)
            if M == 1:
                v = gather(g); los.append(np.array([v.real])); his.append(np.array([-v.imag]))
            else:
                lo, hi = inv_bruun(M, g); los.append(lo); his.append(hi)
        # R_i = (1/p)[ s + 2 sum_j (lo_j cos(2pi ij/p) + hi_j sin(2pi ij/p)) ]
        r = np.zeros(D)
        for i in range(p):
            acc = s.copy()
            for jx, j in enumerate(range(1, (p - 1) // 2 + 1)):
                cc = _cos(Fr(i * j, p)); ss = _sin(Fr(i * j, p))
                acc = acc + 2.0 * (los[jx] * cc + his[jx] * ss)
            r[i * M:(i + 1) * M] = acc / p
        return r

    return inv_minus(N, 1)


if __name__ == "__main__":
    import sympy
    from scratch_genbruun_exact import rfft_gen_exact
    print(f"{'N':>6} {'factors':>16} {'roundtrip':>11} {'vs np.irfft':>12}")
    worst_rt = worst_np = 0.0
    for N in [3,5,7,9,15,27,45,75,127,225,257,509,521,1021,1024,1920,2187,3000,6075,10125]:
        rng = np.random.default_rng(N)
        x = rng.standard_normal(N)
        Xh = rfft_gen_exact(x)
        xr = irfft_gen(Xh, N)
        rt = np.abs(xr - x).max() / max(np.abs(x).max(), 1)
        # inverse vs numpy on an independent Hermitian spectrum
        Xind = np.fft.rfft(rng.standard_normal(N))
        e_np = np.abs(irfft_gen(Xind, N) - np.fft.irfft(Xind, N)).max() / max(np.abs(np.fft.irfft(Xind, N)).max(), 1)
        worst_rt = max(worst_rt, rt); worst_np = max(worst_np, e_np)
        f = "OK" if rt < 1e-11 and e_np < 1e-11 else "FAIL"
        print(f"{N:>6} {str(sympy.factorint(N)):>16} {rt:>11.2e} {e_np:>12.2e}  {f}")
    print(f"WORST roundtrip={worst_rt:.2e}  vs_np={worst_np:.2e}")
