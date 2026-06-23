#pragma once

// MIT joshuah.rainstar@gmail.com 2026
// Internal paired radix-4 half-shift (BDCT) kernel.
//
// This kernel computes the native half-bin-shifted transform
//
//     H[k] = sum_{n=0}^{N-1} x[n] * exp(-2*pi*i*(k+1/2)*n/N),   k = 0..N-1
//
// for real input x. It is the same Bruun-style decimation logic used by the
// BFFT real kernel, specialized to the half-shifted spectrum. Because the input
// is real, H obeys the conjugate symmetry H[N-1-k] = conj(H[k]); the kernel
// therefore emits only the independent lower half H[0..N/2-1] as a contiguous
// packed complex spectrum. There is no DC/Nyquist special case and no final
// reorder.
//
// The decimation is radix-4. For N = 4M, split samples by n = 4q+r. Each child
// is again a length-M half-shift transform of real data, so each child spectrum
// obeys A_r[M-1-k] = conj(A_r[k]). Pairing each output position k with its
// partner kp = M-1-k inside the radix-4 combine lets the two partners share the
// three child twiddle multiplies (t, t^2, t^3), the same way the Bruun real
// kernel pairs conjugate residues. That halves the twiddle multiplies of the
// naive radix-4 half-shift combine.
//
// A length-N DCT-IV is recovered from a length-2N half-shift of the zero-padded
// input followed by a single output rotation (see dctiv_plan_t below).
//
// Precision: float and double only (matching the BFFT real kernel). The kernel
// reuses the heap_array, complex_t/complex_f32_t and SIMD-backend machinery from
// the Bruun real kernel; only the decimation arithmetic is new. The forward
// double combine has an explicit 128-bit SoA SIMD specialization that runs on
// the NEON and SSE2 backends through the shared Bruun vector macros; the scalar
// path stays the exact reference for every backend, precision and tail.

#include "bruun_kernel.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884
#endif

namespace bdct {

// Reuse the tested primitives from the Bruun kernel rather than redefining them.
using bruun::heap_array;
using bruun::complex_t;
using bruun::complex_f32_t;
using bruun::is_power2;
using bruun::ilog2_pow2;

// ---------------------------------------------------------------------------
// Small complex helpers, generic over complex_t (double) and complex_f32_t
// (float). These mirror the scalar tail arithmetic of the Bruun norm-split
// butterflies, kept as named inlines so the combine reads like the algebra it
// implements.
// ---------------------------------------------------------------------------

template <class CT>
static inline CT cadd(CT a, CT b) { return CT{a.re + b.re, a.im + b.im}; }
template <class CT>
static inline CT csub(CT a, CT b) { return CT{a.re - b.re, a.im - b.im}; }
template <class CT>
static inline CT cmul(CT a, CT b) {
    return CT{a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}
template <class CT, class R>
static inline CT cmul_r(CT a, R s) { return CT{a.re * s, a.im * s}; }
template <class CT>
static inline CT cconj(CT a) { return CT{a.re, -a.im}; }
// i * a   (rotate by +90 degrees)
template <class CT>
static inline CT cmuli(CT a) { return CT{-a.im, a.re}; }

// ---------------------------------------------------------------------------
// Forward double combine, 128-bit SoA SIMD specialization.
//
// Processes two consecutive output positions k, k+1 per iteration. Children and
// twiddles are loaded as complex pairs and transposed into real/imag lane
// registers (SoA) so the three per-position twiddle multiplies are pure lanewise
// arithmetic. The partner positions kp = M-1-k and M-2-k are written by simple
// address arithmetic, so no vector reversal is needed: lane 0 stores to kp, lane
// 1 to kp-1. Returns the (even) count of positions handled; the caller's scalar
// loop finishes any tail. Reused unchanged on SSE2 and NEON through the shared
// Bruun V2 macros.
// ---------------------------------------------------------------------------
#if BRUUN_LEVEL >= 1
static inline int bdct_combine_fwd_simd_f64(const complex_t* RESTRICT tab,
                                            const complex_t* RESTRICT c0,
                                            const complex_t* RESTRICT c1,
                                            const complex_t* RESTRICT c2,
                                            const complex_t* RESTRICT c3,
                                            complex_t* RESTRICT out,
                                            int M, int half) {
    const bruun_v2 zero = V2_SET1(0.0);
    int k = 0;
    for (; k + 2 <= half; k += 2) {
        const bruun_v2 ta = V2_LD(&tab[k].re);
        const bruun_v2 tb = V2_LD(&tab[k + 1].re);
        const bruun_v2 tre = V2_UNPLO(ta, tb);   // (t_k.re,   t_{k+1}.re)
        const bruun_v2 tim = V2_UNPHI(ta, tb);   // (t_k.im,   t_{k+1}.im)

        const bruun_v2 t2re = V2_SUB(V2_MUL(tre, tre), V2_MUL(tim, tim));
        const bruun_v2 t2im = V2_ADD(V2_MUL(tre, tim), V2_MUL(tre, tim));
        const bruun_v2 t3re = V2_SUB(V2_MUL(t2re, tre), V2_MUL(t2im, tim));
        const bruun_v2 t3im = V2_ADD(V2_MUL(t2re, tim), V2_MUL(t2im, tre));

        bruun_v2 a, b;
        a = V2_LD(&c0[k].re); b = V2_LD(&c0[k + 1].re);
        const bruun_v2 c0re = V2_UNPLO(a, b), c0im = V2_UNPHI(a, b);
        a = V2_LD(&c1[k].re); b = V2_LD(&c1[k + 1].re);
        const bruun_v2 c1re = V2_UNPLO(a, b), c1im = V2_UNPHI(a, b);
        a = V2_LD(&c2[k].re); b = V2_LD(&c2[k + 1].re);
        const bruun_v2 c2re = V2_UNPLO(a, b), c2im = V2_UNPHI(a, b);
        a = V2_LD(&c3[k].re); b = V2_LD(&c3[k + 1].re);
        const bruun_v2 c3re = V2_UNPLO(a, b), c3im = V2_UNPHI(a, b);

        const bruun_v2 b0re = c0re, b0im = c0im;
        const bruun_v2 b1re = V2_SUB(V2_MUL(c1re, tre), V2_MUL(c1im, tim));
        const bruun_v2 b1im = V2_ADD(V2_MUL(c1re, tim), V2_MUL(c1im, tre));
        const bruun_v2 b2re = V2_SUB(V2_MUL(c2re, t2re), V2_MUL(c2im, t2im));
        const bruun_v2 b2im = V2_ADD(V2_MUL(c2re, t2im), V2_MUL(c2im, t2re));
        const bruun_v2 b3re = V2_SUB(V2_MUL(c3re, t3re), V2_MUL(c3im, t3im));
        const bruun_v2 b3im = V2_ADD(V2_MUL(c3re, t3im), V2_MUL(c3im, t3re));

        const bruun_v2 e0re = V2_ADD(b0re, b2re), e0im = V2_ADD(b0im, b2im);
        const bruun_v2 e1re = V2_SUB(b0re, b2re), e1im = V2_SUB(b0im, b2im);
        const bruun_v2 o0re = V2_ADD(b1re, b3re), o0im = V2_ADD(b1im, b3im);
        const bruun_v2 o1re = V2_SUB(b1re, b3re), o1im = V2_SUB(b1im, b3im);

        // Y[k] = e0 + o0
        const bruun_v2 yk_re = V2_ADD(e0re, o0re), yk_im = V2_ADD(e0im, o0im);
        // Y[k+M] = e1 - i*o1
        const bruun_v2 ykM_re = V2_ADD(e1re, o1im), ykM_im = V2_SUB(e1im, o1re);
        // Y[kp] = conj(e1 + i*o1)
        const bruun_v2 ykp_re = V2_SUB(e1re, o1im);
        const bruun_v2 ykp_im = V2_SUB(zero, V2_ADD(e1im, o1re));
        // Y[kp+M] = conj(e0 - o0)
        const bruun_v2 ykpM_re = V2_SUB(e0re, o0re);
        const bruun_v2 ykpM_im = V2_SUB(o0im, e0im);

        V2_ST(&out[k].re,         V2_UNPLO(yk_re, yk_im));
        V2_ST(&out[k + 1].re,     V2_UNPHI(yk_re, yk_im));
        V2_ST(&out[k + M].re,     V2_UNPLO(ykM_re, ykM_im));
        V2_ST(&out[k + 1 + M].re, V2_UNPHI(ykM_re, ykM_im));
        // partner: lane 0 -> M-1-k, lane 1 -> M-2-k
        V2_ST(&out[M - 1 - k].re,     V2_UNPLO(ykp_re, ykp_im));
        V2_ST(&out[M - 2 - k].re,     V2_UNPHI(ykp_re, ykp_im));
        V2_ST(&out[M - 1 - k + M].re, V2_UNPLO(ykpM_re, ykpM_im));
        V2_ST(&out[M - 2 - k + M].re, V2_UNPHI(ykpM_re, ykpM_im));
    }
    return k;
}
#endif

// ---------------------------------------------------------------------------
// half_shift_plan_t
//
// Reusable plan for the packed real -> half-bin complex transform of a single
// power-of-two size N (N >= 2). Forward maps N real samples to N/2 packed
// complex bins; inverse maps the N/2 packed bins back to the N real samples
// exactly (the packed transform is its own exact inverse, no 1/N scaling).
//
// CT is the complex element type (complex_t or complex_f32_t) and RT its scalar.
// ---------------------------------------------------------------------------

template <class CT, class RT>
class half_shift_plan_t {
public:
    explicit half_shift_plan_t(int n) : n_(n) {
        BRUUN_ASSERT(is_power2(n) && n >= 2);
        const double a4 = -M_PI / 4.0;            // N=4 leaf root: exp(-i*pi/4)
        root4_ = CT{static_cast<RT>(std::cos(a4)), static_cast<RT>(std::sin(a4))};

        // A radix-4 node of size s needs t_k = exp(-2*pi*i*(k+1/2)/s) for
        // k = 0 .. s/8 - 1.
        for (int s = 8; s <= n_; s <<= 1) {
            const int lg = ilog2_pow2(s);
            const int len = s / 8;
            heap_array<CT>& tab = tw_[lg];
            tab.resize(static_cast<std::size_t>(len));
            for (int k = 0; k < len; ++k) {
                const double ang = -2.0 * M_PI * (static_cast<double>(k) + 0.5) /
                                   static_cast<double>(s);
                tab[static_cast<std::size_t>(k)] =
                    CT{static_cast<RT>(std::cos(ang)), static_cast<RT>(std::sin(ang))};
            }
        }

        // Live scratch footprint is bounded by (N/2)*(4/3) < N complex.
        scratch_.resize(static_cast<std::size_t>(n_));
    }

    int size() const noexcept { return n_; }
    int bins() const noexcept { return n_ / 2; }

    // Forward: real input (length N, unit stride) -> packed complex (length N/2).
    void forward(const RT* input, CT* output) const {
        fwd(input, 1, n_, output, scratch_.data());
    }

    // Inverse: packed complex (length N/2) -> real output (length N, unit stride).
    void inverse(const CT* input, RT* output) const {
        inv(input, 1, n_, output, scratch_.data());
    }

private:
    void fwd(const RT* RESTRICT x, int stride, int N,
             CT* RESTRICT out, CT* RESTRICT scratch) const {
        if (N == 2) {
            out[0] = CT{x[0], -x[stride]};                  // H[0] = x0 - i*x1
            return;
        }
        if (N == 4) {
            const CT t = root4_;
            const CT t2 = cmul(t, t);
            const CT t3 = cmul(t2, t);
            const CT b0 = CT{x[0], static_cast<RT>(0)};
            const CT b1 = cmul_r(t, x[stride]);
            const CT b2 = cmul_r(t2, x[2 * stride]);
            const CT b3 = cmul_r(t3, x[3 * stride]);
            out[0] = cadd(cadd(b0, b1), cadd(b2, b3));       // b0+b1+b2+b3
            out[1] = cadd(csub(b0, b2), cmuli(csub(b3, b1)));// b0 - i*b1 - b2 + i*b3
            return;
        }

        const int M = N / 4;
        const int half = M / 2;                              // child packed length
        CT* RESTRICT c0 = scratch;
        CT* RESTRICT c1 = scratch + half;
        CT* RESTRICT c2 = scratch + 2 * half;
        CT* RESTRICT c3 = scratch + 3 * half;
        CT* RESTRICT sub = scratch + 4 * half;               // = scratch + N/2

        fwd(x + 0 * stride, stride * 4, M, c0, sub);
        fwd(x + 1 * stride, stride * 4, M, c1, sub);
        fwd(x + 2 * stride, stride * 4, M, c2, sub);
        fwd(x + 3 * stride, stride * 4, M, c3, sub);

        const CT* RESTRICT tab = tw_[ilog2_pow2(N)].data();

        int k = 0;
        if constexpr (sizeof(RT) == 8) {
#if BRUUN_LEVEL >= 1
            k = bdct_combine_fwd_simd_f64(
                reinterpret_cast<const complex_t*>(tab),
                reinterpret_cast<const complex_t*>(c0),
                reinterpret_cast<const complex_t*>(c1),
                reinterpret_cast<const complex_t*>(c2),
                reinterpret_cast<const complex_t*>(c3),
                reinterpret_cast<complex_t*>(out), M, half);
#endif
        }
        for (; k < half; ++k) {
            const CT t = tab[k];
            const CT t2 = cmul(t, t);
            const CT t3 = cmul(t2, t);

            const CT b0 = c0[k];
            const CT b1 = cmul(t, c1[k]);
            const CT b2 = cmul(t2, c2[k]);
            const CT b3 = cmul(t3, c3[k]);

            const CT e0 = cadd(b0, b2);
            const CT e1 = csub(b0, b2);
            const CT o0 = cadd(b1, b3);
            const CT o1 = csub(b1, b3);

            const int kp = M - 1 - k;
            const CT io1 = cmuli(o1);

            out[k]      = cadd(e0, o0);
            out[k + M]  = csub(e1, io1);
            out[kp]     = cconj(cadd(e1, io1));
            out[kp + M] = cconj(csub(e0, o0));
        }
    }

    // Exact algebraic inverse of fwd; recovers N real samples from N/2 bins.
    void inv(const CT* RESTRICT in, int stride, int N,
             RT* RESTRICT x, CT* RESTRICT scratch) const {
        if (N == 2) {
            x[0] = in[0].re;
            x[stride] = -in[0].im;
            return;
        }
        if (N == 4) {
            const CT t = root4_;
            const CT t2 = cmul(t, t);
            const CT t3 = cmul(t2, t);
            const CT Yk = in[0], YkM = in[1];           // self-paired leaf, k = kp = 0
            const CT e0 = cmul_r(cadd(Yk, cconj(YkM)), static_cast<RT>(0.5));
            const CT o0 = cmul_r(csub(Yk, cconj(YkM)), static_cast<RT>(0.5));
            const CT e1 = cmul_r(cadd(YkM, cconj(Yk)), static_cast<RT>(0.5));
            const CT o1 = cmul_r(cmuli(csub(YkM, cconj(Yk))), static_cast<RT>(0.5));
            const CT b0 = cmul_r(cadd(e0, e1), static_cast<RT>(0.5));
            const CT b2 = cmul_r(csub(e0, e1), static_cast<RT>(0.5));
            const CT b1 = cmul_r(cadd(o0, o1), static_cast<RT>(0.5));
            const CT b3 = cmul_r(csub(o0, o1), static_cast<RT>(0.5));
            x[0]          = b0.re;
            x[stride]     = cmul(cconj(t), b1).re;
            x[2 * stride] = cmul(cconj(t2), b2).re;
            x[3 * stride] = cmul(cconj(t3), b3).re;
            return;
        }

        const int M = N / 4;
        const int half = M / 2;
        CT* RESTRICT c0 = scratch;
        CT* RESTRICT c1 = scratch + half;
        CT* RESTRICT c2 = scratch + 2 * half;
        CT* RESTRICT c3 = scratch + 3 * half;
        CT* RESTRICT sub = scratch + 4 * half;

        const CT* RESTRICT tab = tw_[ilog2_pow2(N)].data();

        for (int k = 0; k < half; ++k) {
            const CT t = tab[k];
            const CT t2 = cmul(t, t);
            const CT t3 = cmul(t2, t);

            const int kp = M - 1 - k;
            const CT Yk   = in[k];
            const CT YkM  = in[k + M];
            const CT Ykp  = in[kp];
            const CT YkpM = in[kp + M];

            const CT e0 = cmul_r(cadd(Yk, cconj(YkpM)), static_cast<RT>(0.5));
            const CT o0 = cmul_r(csub(Yk, cconj(YkpM)), static_cast<RT>(0.5));
            const CT e1 = cmul_r(cadd(YkM, cconj(Ykp)), static_cast<RT>(0.5));
            const CT o1 = cmul_r(cmuli(csub(YkM, cconj(Ykp))), static_cast<RT>(0.5));

            const CT b0 = cmul_r(cadd(e0, e1), static_cast<RT>(0.5));
            const CT b2 = cmul_r(csub(e0, e1), static_cast<RT>(0.5));
            const CT b1 = cmul_r(cadd(o0, o1), static_cast<RT>(0.5));
            const CT b3 = cmul_r(csub(o0, o1), static_cast<RT>(0.5));

            c0[k] = b0;
            c1[k] = cmul(cconj(t), b1);
            c2[k] = cmul(cconj(t2), b2);
            c3[k] = cmul(cconj(t3), b3);
        }

        inv(c0, stride * 4, M, x + 0 * stride, sub);
        inv(c1, stride * 4, M, x + 1 * stride, sub);
        inv(c2, stride * 4, M, x + 2 * stride, sub);
        inv(c3, stride * 4, M, x + 3 * stride, sub);
    }

    int n_;
    CT root4_;
    heap_array<CT> tw_[32];
    mutable heap_array<CT> scratch_;
};

using half_shift_plan = half_shift_plan_t<complex_t, double>;
using half_shift_plan_f32 = half_shift_plan_t<complex_f32_t, float>;

// ---------------------------------------------------------------------------
// dctiv_plan_t
//
// Length-N DCT-IV:
//
//     C[k] = sum_{n=0}^{N-1} x[n] * cos( (pi/N) * (k+1/2) * (n+1/2) )
//
// computed as the real part of a rotated length-2N half-shift of the
// zero-padded input:
//
//     S[k] = sum_{n=0}^{N-1} x[n] * exp(-2*pi*i*(k+1/2)*n/(2N))    (k = 0..N-1)
//     C[k] = Re{ exp(-i*pi*(k+1/2)/(2N)) * S[k] }.
//
// S[0..N-1] is exactly the packed half-shift spectrum of the length-2N padded
// input, so one half-shift forward of size 2N plus N output rotations gives the
// orthogonal (unnormalized) DCT-IV. DCT-IV is its own inverse up to a 2/N scale.
// ---------------------------------------------------------------------------

template <class CT, class RT>
class dctiv_plan_t {
public:
    explicit dctiv_plan_t(int n) : n_(n), hs_(2 * n) {
        BRUUN_ASSERT(is_power2(n) && n >= 2);
        pad_.resize(static_cast<std::size_t>(2 * n));
        spec_.resize(static_cast<std::size_t>(n));      // 2N/2 = N packed bins
        rot_.resize(static_cast<std::size_t>(n));
        for (int k = 0; k < n; ++k) {
            const double ang = -M_PI * (static_cast<double>(k) + 0.5) /
                               (2.0 * static_cast<double>(n));
            rot_[static_cast<std::size_t>(k)] =
                CT{static_cast<RT>(std::cos(ang)), static_cast<RT>(std::sin(ang))};
        }
    }

    int size() const noexcept { return n_; }

    // Forward (unnormalized, orthogonal) DCT-IV: real -> real, both length N.
    void forward(const RT* input, RT* output) const {
        for (int n = 0; n < n_; ++n) pad_[static_cast<std::size_t>(n)] = input[n];
        for (int n = n_; n < 2 * n_; ++n) pad_[static_cast<std::size_t>(n)] = static_cast<RT>(0);

        hs_.forward(pad_.data(), spec_.data());

        for (int k = 0; k < n_; ++k) {
            const CT v = cmul(rot_[static_cast<std::size_t>(k)],
                              spec_[static_cast<std::size_t>(k)]);
            output[k] = v.re;
        }
    }

    // Inverse DCT-IV. DCT-IV is its own inverse up to a 2/N factor.
    void inverse(const RT* input, RT* output) const {
        forward(input, output);
        const RT s = static_cast<RT>(2) / static_cast<RT>(n_);
        for (int k = 0; k < n_; ++k) output[k] *= s;
    }

private:
    int n_;
    half_shift_plan_t<CT, RT> hs_;
    mutable heap_array<RT> pad_;
    mutable heap_array<CT> spec_;
    heap_array<CT> rot_;
};

using dctiv_plan = dctiv_plan_t<complex_t, double>;
using dctiv_plan_f32 = dctiv_plan_t<complex_f32_t, float>;

} // namespace bdct
