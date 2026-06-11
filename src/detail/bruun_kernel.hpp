#pragma once

// MIT joshuah.rainstar@gmail.com 2026
// Internal normalized-basis Bruun RFFT kernel.
//
// BFFT builds this kernel as part of the library. Public users select transform
// size and output layout through include/bfft/bfft.h or include/bfft/bfft.hpp;
// the Makefile chooses safe host compiler switches automatically. Heap-optimized
// native ordering and fused scatter are the default hot path. The library keeps a
// sequential two-phase standard-output pack available internally and uses it only
// when the standard FFT-order output is large enough to win on the active SIMD
// backend.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>
#include <utility>

#if !defined(_WIN32)
#include <dlfcn.h>
#endif

// ---------------------------------------------------------------------------
// SIMD backend resolution.
//   BRUUN_LEVEL 0 = scalar, 1 = 128-bit (SSE2/NEON), 2 = AVX2+FMA, 3 = AVX-512
// Wider levels reuse the narrower loops as tails where needed.
// ---------------------------------------------------------------------------

#if defined(__AVX512F__)
#  define BRUUN_LEVEL 3
#elif defined(__AVX2__) && defined(__FMA__)
#  define BRUUN_LEVEL 2
#elif defined(__SSE2__) || defined(_M_X64) || (defined(__aarch64__) && defined(__ARM_NEON))
#  define BRUUN_LEVEL 1
#else
#  define BRUUN_LEVEL 0
#endif

#if BRUUN_LEVEL >= 2 || (BRUUN_LEVEL >= 1 && (defined(__SSE2__) || defined(_M_X64)))
#  include <immintrin.h>
#  define BRUUN_X86_128 1
#endif

#if BRUUN_LEVEL >= 1 && defined(__aarch64__) && defined(__ARM_NEON)
#  include <arm_neon.h>
#  define BRUUN_NEON_128 1
#endif

#if BRUUN_LEVEL == 1 && !defined(BRUUN_X86_128) && !defined(BRUUN_NEON_128)
#  undef BRUUN_LEVEL
#  define BRUUN_LEVEL 0
#endif

// 2-lane double vector primitive shared by the SSE2 and NEON paths.
#if defined(BRUUN_X86_128)
typedef __m128d bruun_v2;
#  define V2_LD(p)        _mm_loadu_pd(p)
#  define V2_ST(p, a)     _mm_storeu_pd((p), (a))
#  define V2_ADD(a, b)    _mm_add_pd((a), (b))
#  define V2_SUB(a, b)    _mm_sub_pd((a), (b))
#  define V2_MUL(a, b)    _mm_mul_pd((a), (b))
#  define V2_SET1(x)      _mm_set1_pd(x)
#  define V2_SETLH(l, h)  _mm_set_pd((h), (l))
#  define V2_UNPLO(a, b)  _mm_unpacklo_pd((a), (b))
#  define V2_UNPHI(a, b)  _mm_unpackhi_pd((a), (b))
#  define V2_DUP0(a)      _mm_unpacklo_pd((a), (a))
#  define V2_DUP1(a)      _mm_unpackhi_pd((a), (a))
#  define V2_NEGHI(a)     _mm_xor_pd((a), _mm_set_pd(-0.0, 0.0))
#elif defined(BRUUN_NEON_128)
typedef float64x2_t bruun_v2;
#  define V2_LD(p)        vld1q_f64(p)
#  define V2_ST(p, a)     vst1q_f64((p), (a))
#  define V2_ADD(a, b)    vaddq_f64((a), (b))
#  define V2_SUB(a, b)    vsubq_f64((a), (b))
#  define V2_MUL(a, b)    vmulq_f64((a), (b))
#  define V2_SET1(x)      vdupq_n_f64(x)
#  define V2_SETLH(l, h)  vcombine_f64(vdup_n_f64(l), vdup_n_f64(h))
#  define V2_UNPLO(a, b)  vtrn1q_f64((a), (b))
#  define V2_UNPHI(a, b)  vtrn2q_f64((a), (b))
#  define V2_DUP0(a)      vdupq_laneq_f64((a), 0)
#  define V2_DUP1(a)      vdupq_laneq_f64((a), 1)
static inline float64x2_t bruun_neghi(float64x2_t a) {
    const uint64x2_t m = { 0ULL, 0x8000000000000000ULL };
    return vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(a), m));
}
#  define V2_NEGHI(a)     bruun_neghi(a)
#endif

// 4-lane float vector primitive for internal float32 helper loops.
#if defined(BRUUN_X86_128)
typedef __m128 bruun_v4f;
#  define V4F_LD(p)       _mm_loadu_ps(p)
#  define V4F_ST(p, a)    _mm_storeu_ps((p), (a))
#  define V4F_MUL(a, b)   _mm_mul_ps((a), (b))
#  define V4F_SET1(x)     _mm_set1_ps(x)
#  define V4F_ZERO()      _mm_setzero_ps()
#elif defined(BRUUN_NEON_128)
typedef float32x4_t bruun_v4f;
#  define V4F_LD(p)       vld1q_f32(p)
#  define V4F_ST(p, a)    vst1q_f32((p), (a))
#  define V4F_MUL(a, b)   vmulq_f32((a), (b))
#  define V4F_SET1(x)     vdupq_n_f32(x)
#  define V4F_ZERO()      vdupq_n_f32(0.0f)
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#define RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define RESTRICT __restrict__
#else
#define RESTRICT
#endif


// BFFT keeps heap-optimized native spectrum ordering enabled internally.
// Applications select public layouts through the API instead of compile flags.
#ifndef BRUUN_HEAPOPT_SPECTRUM_ORDER
#define BRUUN_HEAPOPT_SPECTRUM_ORDER 1
#endif

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884
#endif

namespace bruun {

struct complex_t {
    double re;
    double im;
};

struct complex_f32_t {
    float re;
    float im;
};

static inline void copy_real_to_complex_work_f32(const float* RESTRICT input,
                                                 float* RESTRICT re,
                                                 float* RESTRICT im,
                                                 int n) {
    int i = 0;

#if BRUUN_LEVEL >= 3
    const __m512 z16 = _mm512_setzero_ps();
    for (; i + 15 < n; i += 16) {
        _mm512_storeu_ps(re + i, _mm512_loadu_ps(input + i));
        _mm512_storeu_ps(im + i, z16);
    }
#endif
#if BRUUN_LEVEL >= 2
    const __m256 z8 = _mm256_setzero_ps();
    for (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(re + i, _mm256_loadu_ps(input + i));
        _mm256_storeu_ps(im + i, z8);
    }
#elif BRUUN_LEVEL == 1
    const bruun_v4f z4 = V4F_ZERO();
    for (; i + 3 < n; i += 4) {
        V4F_ST(re + i, V4F_LD(input + i));
        V4F_ST(im + i, z4);
    }
#endif

    for (; i < n; ++i) {
        re[i] = input[i];
        im[i] = 0.0f;
    }
}

static inline void pack_standard_spectrum_f32(const float* RESTRICT re,
                                              const float* RESTRICT im,
                                              complex_f32_t* RESTRICT X,
                                              int bins) {
    for (int k = 0; k < bins; ++k) {
        X[k].re = re[k];
        X[k].im = im[k];
    }
}

static inline void unpack_standard_spectrum_f32(const complex_f32_t* RESTRICT X,
                                                float* RESTRICT re,
                                                float* RESTRICT im,
                                                int n) {
    re[0] = X[0].re;
    im[0] = 0.0f;
    re[n / 2] = X[n / 2].re;
    im[n / 2] = 0.0f;
    for (int k = 1; k < n / 2; ++k) {
        re[k] = X[k].re;
        im[k] = X[k].im;
        re[n - k] = X[k].re;
        im[n - k] = -X[k].im;
    }
}

static inline void scale_complex_work_f32(float* RESTRICT re,
                                          float* RESTRICT im,
                                          int n,
                                          float scale) {
    int i = 0;

#if BRUUN_LEVEL >= 3
    const __m512 s16 = _mm512_set1_ps(scale);
    for (; i + 15 < n; i += 16) {
        _mm512_storeu_ps(re + i, _mm512_mul_ps(_mm512_loadu_ps(re + i), s16));
        _mm512_storeu_ps(im + i, _mm512_mul_ps(_mm512_loadu_ps(im + i), s16));
    }
#endif
#if BRUUN_LEVEL >= 2
    const __m256 s8 = _mm256_set1_ps(scale);
    for (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(re + i, _mm256_mul_ps(_mm256_loadu_ps(re + i), s8));
        _mm256_storeu_ps(im + i, _mm256_mul_ps(_mm256_loadu_ps(im + i), s8));
    }
#elif BRUUN_LEVEL == 1
    const bruun_v4f s4 = V4F_SET1(scale);
    for (; i + 3 < n; i += 4) {
        V4F_ST(re + i, V4F_MUL(V4F_LD(re + i), s4));
        V4F_ST(im + i, V4F_MUL(V4F_LD(im + i), s4));
    }
#endif

    for (; i < n; ++i) {
        re[i] *= scale;
        im[i] *= scale;
    }
}

static inline void copy_real_output_f32(const float* RESTRICT re,
                                        float* RESTRICT out,
                                        int n) {
    int i = 0;

#if BRUUN_LEVEL >= 3
    for (; i + 15 < n; i += 16) {
        _mm512_storeu_ps(out + i, _mm512_loadu_ps(re + i));
    }
#endif
#if BRUUN_LEVEL >= 2
    for (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(out + i, _mm256_loadu_ps(re + i));
    }
#elif BRUUN_LEVEL == 1
    for (; i + 3 < n; i += 4) {
        V4F_ST(out + i, V4F_LD(re + i));
    }
#endif

    for (; i < n; ++i) {
        out[i] = re[i];
    }
}

static inline const char* simd_backend_name() {
#if BRUUN_LEVEL == 3
    return "avx512-512";
#elif BRUUN_LEVEL == 2
    return "avx2-fma-256";
#elif defined(BRUUN_X86_128)
    return "sse2-128";
#elif defined(BRUUN_NEON_128)
    return "neon-128";
#else
    return "scalar";
#endif
}

static inline bool is_power2(int n) {
    return n > 0 && ((n & (n - 1)) == 0);
}

static inline int ilog2_pow2(int n) {
    int l = 0;
    while (n > 1) {
        n >>= 1;
        ++l;
    }
    return l;
}

static inline int graydecode_int(int g) {
    for (int s = 1; s < 32; s <<= 1) g ^= g >> s;
    return g;
}

static inline int bitrev_int(int r, int t) {
    int out = 0;
    for (int i = 0; i < t; ++i) {
        out = (out << 1) | (r & 1);
        r >>= 1;
    }
    return out;
}

static inline int bruun_idx_int(int m, int L) {
#if defined(__GNUC__) || defined(__clang__)
    const int t = 31 - __builtin_clz((unsigned)m);
#else
    int t = 0;
    for (int x = m; x > 1; x >>= 1) ++t;
#endif
    const int r = m ^ (1 << t);
    return (2 * graydecode_int(bitrev_int(r, t)) + 1) << ((L - 2) - t);
}

// ---------------------------------------------------------------------------
// Streaming kernels. Each has an optional 512 block, an optional 256 block,
// a 2-lane block for the 128-bit backend, and an exact scalar tail.
// ---------------------------------------------------------------------------

static inline void binomial_fwd(double* RESTRICT v, int h) {
    int i = 0;

#if BRUUN_LEVEL >= 3
    for (; i + 7 < h; i += 8) {
        const __m512d a = _mm512_loadu_pd(v + i);
        const __m512d b = _mm512_loadu_pd(v + h + i);
        _mm512_storeu_pd(v + i, _mm512_add_pd(a, b));
        _mm512_storeu_pd(v + h + i, _mm512_sub_pd(a, b));
    }
#endif
#if BRUUN_LEVEL >= 2
    for (; i + 3 < h; i += 4) {
        const __m256d a = _mm256_loadu_pd(v + i);
        const __m256d b = _mm256_loadu_pd(v + h + i);
        _mm256_storeu_pd(v + i, _mm256_add_pd(a, b));
        _mm256_storeu_pd(v + h + i, _mm256_sub_pd(a, b));
    }
#elif BRUUN_LEVEL == 1
    for (; i + 1 < h; i += 2) {
        const bruun_v2 a = V2_LD(v + i);
        const bruun_v2 b = V2_LD(v + h + i);
        V2_ST(v + i, V2_ADD(a, b));
        V2_ST(v + h + i, V2_SUB(a, b));
    }
#endif

    for (; i < h; ++i) {
        const double a = v[i];
        const double b = v[h + i];
        v[i] = a + b;
        v[h + i] = a - b;
    }
}

// Fused "copy input + first binomial split": v[i] = in[i] + in[h+i], v[h+i] = in[i] - in[h+i].
static inline void binomial_oop(const double* RESTRICT in, double* RESTRICT v, int h) {
    int i = 0;

#if BRUUN_LEVEL >= 3
    for (; i + 7 < h; i += 8) {
        const __m512d a = _mm512_loadu_pd(in + i);
        const __m512d b = _mm512_loadu_pd(in + h + i);
        _mm512_storeu_pd(v + i, _mm512_add_pd(a, b));
        _mm512_storeu_pd(v + h + i, _mm512_sub_pd(a, b));
    }
#endif
#if BRUUN_LEVEL >= 2
    for (; i + 3 < h; i += 4) {
        const __m256d a = _mm256_loadu_pd(in + i);
        const __m256d b = _mm256_loadu_pd(in + h + i);
        _mm256_storeu_pd(v + i, _mm256_add_pd(a, b));
        _mm256_storeu_pd(v + h + i, _mm256_sub_pd(a, b));
    }
#elif BRUUN_LEVEL == 1
    for (; i + 1 < h; i += 2) {
        const bruun_v2 a = V2_LD(in + i);
        const bruun_v2 b = V2_LD(in + h + i);
        V2_ST(v + i, V2_ADD(a, b));
        V2_ST(v + h + i, V2_SUB(a, b));
    }
#endif

    for (; i < h; ++i) {
        const double a = in[i];
        const double b = in[h + i];
        v[i] = a + b;
        v[h + i] = a - b;
    }
}

static inline void binomial_inv(double* RESTRICT v, int h) {
    int i = 0;

#if BRUUN_LEVEL >= 3
    {
        const __m512d half8 = _mm512_set1_pd(0.5);
        for (; i + 7 < h; i += 8) {
            const __m512d a = _mm512_loadu_pd(v + i);
            const __m512d b = _mm512_loadu_pd(v + h + i);
            _mm512_storeu_pd(v + i, _mm512_mul_pd(half8, _mm512_add_pd(a, b)));
            _mm512_storeu_pd(v + h + i, _mm512_mul_pd(half8, _mm512_sub_pd(a, b)));
        }
    }
#endif
#if BRUUN_LEVEL >= 2
    const __m256d half4 = _mm256_set1_pd(0.5);
    for (; i + 3 < h; i += 4) {
        const __m256d a = _mm256_loadu_pd(v + i);
        const __m256d b = _mm256_loadu_pd(v + h + i);
        _mm256_storeu_pd(v + i, _mm256_mul_pd(half4, _mm256_add_pd(a, b)));
        _mm256_storeu_pd(v + h + i, _mm256_mul_pd(half4, _mm256_sub_pd(a, b)));
    }
#elif BRUUN_LEVEL == 1
    const bruun_v2 half2 = V2_SET1(0.5);
    for (; i + 1 < h; i += 2) {
        const bruun_v2 a = V2_LD(v + i);
        const bruun_v2 b = V2_LD(v + h + i);
        V2_ST(v + i, V2_MUL(half2, V2_ADD(a, b)));
        V2_ST(v + h + i, V2_MUL(half2, V2_SUB(a, b)));
    }
#endif

    for (; i < h; ++i) {
        const double a = v[i];
        const double b = v[h + i];
        v[i] = 0.5 * (a + b);
        v[h + i] = 0.5 * (a - b);
    }
}

// One normalized-quadratic split: block [A0|B0|A1|B1] of quarters q.
static inline void norm_q_fwd(double* RESTRICT p, int q, double c_scalar, double s_scalar) {
    double* RESTRICT A0p = p;
    double* RESTRICT B0p = p + q;
    double* RESTRICT A1p = p + 2*q;
    double* RESTRICT B1p = p + 3*q;

    int n = 0;

#if BRUUN_LEVEL >= 3
    {
        const __m512d wc = _mm512_set1_pd(c_scalar);
        const __m512d ws = _mm512_set1_pd(s_scalar);

        for (; n + 7 < q; n += 8) {
            const __m512d A0 = _mm512_loadu_pd(A0p + n);
            const __m512d B0 = _mm512_loadu_pd(B0p + n);
            const __m512d A1 = _mm512_loadu_pd(A1p + n);
            const __m512d B1 = _mm512_loadu_pd(B1p + n);

            const __m512d R = _mm512_fmsub_pd(wc, B0, _mm512_mul_pd(ws, B1));
            const __m512d I = _mm512_fmadd_pd(ws, B0, _mm512_mul_pd(wc, B1));

            _mm512_storeu_pd(A0p + n, _mm512_add_pd(A0, R));
            _mm512_storeu_pd(B0p + n, _mm512_add_pd(A1, I));
            _mm512_storeu_pd(A1p + n, _mm512_sub_pd(A0, R));
            _mm512_storeu_pd(B1p + n, _mm512_sub_pd(I, A1));
        }
    }
#endif
#if BRUUN_LEVEL >= 2
    {
        const __m256d vc = _mm256_set1_pd(c_scalar);
        const __m256d vs = _mm256_set1_pd(s_scalar);

        for (; n + 3 < q; n += 4) {
            const __m256d A0 = _mm256_loadu_pd(A0p + n);
            const __m256d B0 = _mm256_loadu_pd(B0p + n);
            const __m256d A1 = _mm256_loadu_pd(A1p + n);
            const __m256d B1 = _mm256_loadu_pd(B1p + n);

            const __m256d R = _mm256_fmsub_pd(vc, B0, _mm256_mul_pd(vs, B1));
            const __m256d I = _mm256_fmadd_pd(vs, B0, _mm256_mul_pd(vc, B1));

            _mm256_storeu_pd(A0p + n, _mm256_add_pd(A0, R));
            _mm256_storeu_pd(B0p + n, _mm256_add_pd(A1, I));
            _mm256_storeu_pd(A1p + n, _mm256_sub_pd(A0, R));
            _mm256_storeu_pd(B1p + n, _mm256_sub_pd(I, A1));
        }
    }
#elif BRUUN_LEVEL == 1
    {
        const bruun_v2 vc = V2_SET1(c_scalar);
        const bruun_v2 vs = V2_SET1(s_scalar);

        for (; n + 1 < q; n += 2) {
            const bruun_v2 A0 = V2_LD(A0p + n);
            const bruun_v2 B0 = V2_LD(B0p + n);
            const bruun_v2 A1 = V2_LD(A1p + n);
            const bruun_v2 B1 = V2_LD(B1p + n);

            const bruun_v2 R = V2_SUB(V2_MUL(vc, B0), V2_MUL(vs, B1));
            const bruun_v2 I = V2_ADD(V2_MUL(vs, B0), V2_MUL(vc, B1));

            V2_ST(A0p + n, V2_ADD(A0, R));
            V2_ST(B0p + n, V2_ADD(A1, I));
            V2_ST(A1p + n, V2_SUB(A0, R));
            V2_ST(B1p + n, V2_SUB(I, A1));
        }
    }
#endif

    for (; n < q; ++n) {
        const double A0 = A0p[n];
        const double B0 = B0p[n];
        const double A1 = A1p[n];
        const double B1 = B1p[n];

        const double R = c_scalar * B0 - s_scalar * B1;
        const double I = s_scalar * B0 + c_scalar * B1;

        A0p[n] = A0 + R;
        B0p[n] = A1 + I;
        A1p[n] = A0 - R;
        B1p[n] = -A1 + I;
    }
}

// Two tree levels in one pass: parent rotation (c,s) plus both child rotations
// (c0,s0), (c1,s1) applied while the data is in registers. Halves the load/store
// traffic of the norm cascade. Caller guarantees q >= 16 so qh >= 8.
static inline void norm2_fused(double* RESTRICT p, int q,
                               double c, double s,
                               double c0, double s0,
                               double c1, double s1) {
    const int qh = q >> 1;
    double* RESTRICT A0 = p;
    double* RESTRICT B0 = p + q;
    double* RESTRICT A1 = p + 2*q;
    double* RESTRICT B1 = p + 3*q;

    int n = 0;

#if BRUUN_LEVEL >= 3
    {
        const __m512d vc  = _mm512_set1_pd(c),  vs  = _mm512_set1_pd(s);
        const __m512d vc0 = _mm512_set1_pd(c0), vs0 = _mm512_set1_pd(s0);
        const __m512d vc1 = _mm512_set1_pd(c1), vs1 = _mm512_set1_pd(s1);

        for (; n + 7 < qh; n += 8) {
            const __m512d a0n = _mm512_loadu_pd(A0 + n);
            const __m512d a0h = _mm512_loadu_pd(A0 + qh + n);
            const __m512d b0n = _mm512_loadu_pd(B0 + n);
            const __m512d b0h = _mm512_loadu_pd(B0 + qh + n);
            const __m512d a1n = _mm512_loadu_pd(A1 + n);
            const __m512d a1h = _mm512_loadu_pd(A1 + qh + n);
            const __m512d b1n = _mm512_loadu_pd(B1 + n);
            const __m512d b1h = _mm512_loadu_pd(B1 + qh + n);

            const __m512d Rn = _mm512_fmsub_pd(vc, b0n, _mm512_mul_pd(vs, b1n));
            const __m512d In = _mm512_fmadd_pd(vs, b0n, _mm512_mul_pd(vc, b1n));
            const __m512d Rh = _mm512_fmsub_pd(vc, b0h, _mm512_mul_pd(vs, b1h));
            const __m512d Ih = _mm512_fmadd_pd(vs, b0h, _mm512_mul_pd(vc, b1h));

            const __m512d u0 = _mm512_add_pd(a0n, Rn);
            const __m512d uh = _mm512_add_pd(a0h, Rh);
            const __m512d w0 = _mm512_add_pd(a1n, In);
            const __m512d wh = _mm512_add_pd(a1h, Ih);
            const __m512d v0 = _mm512_sub_pd(a0n, Rn);
            const __m512d vh = _mm512_sub_pd(a0h, Rh);
            const __m512d x0 = _mm512_sub_pd(In, a1n);
            const __m512d xh = _mm512_sub_pd(Ih, a1h);

            const __m512d R0 = _mm512_fmsub_pd(vc0, uh, _mm512_mul_pd(vs0, wh));
            const __m512d I0 = _mm512_fmadd_pd(vs0, uh, _mm512_mul_pd(vc0, wh));
            const __m512d R1 = _mm512_fmsub_pd(vc1, vh, _mm512_mul_pd(vs1, xh));
            const __m512d I1 = _mm512_fmadd_pd(vs1, vh, _mm512_mul_pd(vc1, xh));

            _mm512_storeu_pd(A0 + n,      _mm512_add_pd(u0, R0));
            _mm512_storeu_pd(A0 + qh + n, _mm512_add_pd(w0, I0));
            _mm512_storeu_pd(B0 + n,      _mm512_sub_pd(u0, R0));
            _mm512_storeu_pd(B0 + qh + n, _mm512_sub_pd(I0, w0));
            _mm512_storeu_pd(A1 + n,      _mm512_add_pd(v0, R1));
            _mm512_storeu_pd(A1 + qh + n, _mm512_add_pd(x0, I1));
            _mm512_storeu_pd(B1 + n,      _mm512_sub_pd(v0, R1));
            _mm512_storeu_pd(B1 + qh + n, _mm512_sub_pd(I1, x0));
        }
    }
#endif
#if BRUUN_LEVEL >= 2
    {
        const __m256d vc  = _mm256_set1_pd(c),  vs  = _mm256_set1_pd(s);
        const __m256d vc0 = _mm256_set1_pd(c0), vs0 = _mm256_set1_pd(s0);
        const __m256d vc1 = _mm256_set1_pd(c1), vs1 = _mm256_set1_pd(s1);

        for (; n + 3 < qh; n += 4) {
            const __m256d a0n = _mm256_loadu_pd(A0 + n);
            const __m256d a0h = _mm256_loadu_pd(A0 + qh + n);
            const __m256d b0n = _mm256_loadu_pd(B0 + n);
            const __m256d b0h = _mm256_loadu_pd(B0 + qh + n);
            const __m256d a1n = _mm256_loadu_pd(A1 + n);
            const __m256d a1h = _mm256_loadu_pd(A1 + qh + n);
            const __m256d b1n = _mm256_loadu_pd(B1 + n);
            const __m256d b1h = _mm256_loadu_pd(B1 + qh + n);

            const __m256d Rn = _mm256_fmsub_pd(vc, b0n, _mm256_mul_pd(vs, b1n));
            const __m256d In = _mm256_fmadd_pd(vs, b0n, _mm256_mul_pd(vc, b1n));
            const __m256d Rh = _mm256_fmsub_pd(vc, b0h, _mm256_mul_pd(vs, b1h));
            const __m256d Ih = _mm256_fmadd_pd(vs, b0h, _mm256_mul_pd(vc, b1h));

            const __m256d u0 = _mm256_add_pd(a0n, Rn);
            const __m256d uh = _mm256_add_pd(a0h, Rh);
            const __m256d w0 = _mm256_add_pd(a1n, In);
            const __m256d wh = _mm256_add_pd(a1h, Ih);
            const __m256d v0 = _mm256_sub_pd(a0n, Rn);
            const __m256d vh = _mm256_sub_pd(a0h, Rh);
            const __m256d x0 = _mm256_sub_pd(In, a1n);
            const __m256d xh = _mm256_sub_pd(Ih, a1h);

            const __m256d R0 = _mm256_fmsub_pd(vc0, uh, _mm256_mul_pd(vs0, wh));
            const __m256d I0 = _mm256_fmadd_pd(vs0, uh, _mm256_mul_pd(vc0, wh));
            const __m256d R1 = _mm256_fmsub_pd(vc1, vh, _mm256_mul_pd(vs1, xh));
            const __m256d I1 = _mm256_fmadd_pd(vs1, vh, _mm256_mul_pd(vc1, xh));

            _mm256_storeu_pd(A0 + n,      _mm256_add_pd(u0, R0));
            _mm256_storeu_pd(A0 + qh + n, _mm256_add_pd(w0, I0));
            _mm256_storeu_pd(B0 + n,      _mm256_sub_pd(u0, R0));
            _mm256_storeu_pd(B0 + qh + n, _mm256_sub_pd(I0, w0));
            _mm256_storeu_pd(A1 + n,      _mm256_add_pd(v0, R1));
            _mm256_storeu_pd(A1 + qh + n, _mm256_add_pd(x0, I1));
            _mm256_storeu_pd(B1 + n,      _mm256_sub_pd(v0, R1));
            _mm256_storeu_pd(B1 + qh + n, _mm256_sub_pd(I1, x0));
        }
    }
#elif BRUUN_LEVEL == 1
    {
        const bruun_v2 vc  = V2_SET1(c),  vs  = V2_SET1(s);
        const bruun_v2 vc0 = V2_SET1(c0), vs0 = V2_SET1(s0);
        const bruun_v2 vc1 = V2_SET1(c1), vs1 = V2_SET1(s1);

        for (; n + 1 < qh; n += 2) {
            const bruun_v2 a0n = V2_LD(A0 + n);
            const bruun_v2 a0h = V2_LD(A0 + qh + n);
            const bruun_v2 b0n = V2_LD(B0 + n);
            const bruun_v2 b0h = V2_LD(B0 + qh + n);
            const bruun_v2 a1n = V2_LD(A1 + n);
            const bruun_v2 a1h = V2_LD(A1 + qh + n);
            const bruun_v2 b1n = V2_LD(B1 + n);
            const bruun_v2 b1h = V2_LD(B1 + qh + n);

            const bruun_v2 Rn = V2_SUB(V2_MUL(vc, b0n), V2_MUL(vs, b1n));
            const bruun_v2 In = V2_ADD(V2_MUL(vs, b0n), V2_MUL(vc, b1n));
            const bruun_v2 Rh = V2_SUB(V2_MUL(vc, b0h), V2_MUL(vs, b1h));
            const bruun_v2 Ih = V2_ADD(V2_MUL(vs, b0h), V2_MUL(vc, b1h));

            const bruun_v2 u0 = V2_ADD(a0n, Rn);
            const bruun_v2 uh = V2_ADD(a0h, Rh);
            const bruun_v2 w0 = V2_ADD(a1n, In);
            const bruun_v2 wh = V2_ADD(a1h, Ih);
            const bruun_v2 v0 = V2_SUB(a0n, Rn);
            const bruun_v2 vh = V2_SUB(a0h, Rh);
            const bruun_v2 x0 = V2_SUB(In, a1n);
            const bruun_v2 xh = V2_SUB(Ih, a1h);

            const bruun_v2 R0 = V2_SUB(V2_MUL(vc0, uh), V2_MUL(vs0, wh));
            const bruun_v2 I0 = V2_ADD(V2_MUL(vs0, uh), V2_MUL(vc0, wh));
            const bruun_v2 R1 = V2_SUB(V2_MUL(vc1, vh), V2_MUL(vs1, xh));
            const bruun_v2 I1 = V2_ADD(V2_MUL(vs1, vh), V2_MUL(vc1, xh));

            V2_ST(A0 + n,      V2_ADD(u0, R0));
            V2_ST(A0 + qh + n, V2_ADD(w0, I0));
            V2_ST(B0 + n,      V2_SUB(u0, R0));
            V2_ST(B0 + qh + n, V2_SUB(I0, w0));
            V2_ST(A1 + n,      V2_ADD(v0, R1));
            V2_ST(A1 + qh + n, V2_ADD(x0, I1));
            V2_ST(B1 + n,      V2_SUB(v0, R1));
            V2_ST(B1 + qh + n, V2_SUB(I1, x0));
        }
    }
#endif

    for (; n < qh; ++n) {
        const double a0n = A0[n],      a0h = A0[qh + n];
        const double b0n = B0[n],      b0h = B0[qh + n];
        const double a1n = A1[n],      a1h = A1[qh + n];
        const double b1n = B1[n],      b1h = B1[qh + n];

        const double Rn = c * b0n - s * b1n;
        const double In = s * b0n + c * b1n;
        const double Rh = c * b0h - s * b1h;
        const double Ih = s * b0h + c * b1h;

        const double u0 = a0n + Rn, uh = a0h + Rh;
        const double w0 = a1n + In, wh = a1h + Ih;
        const double v0 = a0n - Rn, vh = a0h - Rh;
        const double x0 = In - a1n, xh = Ih - a1h;

        const double R0 = c0 * uh - s0 * wh;
        const double I0 = s0 * uh + c0 * wh;
        const double R1 = c1 * vh - s1 * xh;
        const double I1 = s1 * vh + c1 * xh;

        A0[n] = u0 + R0;
        A0[qh + n] = w0 + I0;
        B0[n] = u0 - R0;
        B0[qh + n] = I0 - w0;
        A1[n] = v0 + R1;
        A1[qh + n] = x0 + I1;
        B1[n] = v0 - R1;
        B1[qh + n] = I1 - x0;
    }
}

static inline void norm_q_inv(double* RESTRICT p, int q, double c_scalar, double s_scalar) {
    double* RESTRICT C0p = p;
    double* RESTRICT C1p = p + q;
    double* RESTRICT D0p = p + 2*q;
    double* RESTRICT D1p = p + 3*q;

    int n = 0;

#if BRUUN_LEVEL >= 3
    {
        const __m512d half = _mm512_set1_pd(0.5);
        const __m512d vc = _mm512_set1_pd(c_scalar);
        const __m512d vs = _mm512_set1_pd(s_scalar);

        for (; n + 7 < q; n += 8) {
            const __m512d C0v = _mm512_loadu_pd(C0p + n);
            const __m512d C1v = _mm512_loadu_pd(C1p + n);
            const __m512d D0v = _mm512_loadu_pd(D0p + n);
            const __m512d D1v = _mm512_loadu_pd(D1p + n);

            const __m512d A0 = _mm512_mul_pd(half, _mm512_add_pd(C0v, D0v));
            const __m512d R  = _mm512_mul_pd(half, _mm512_sub_pd(C0v, D0v));
            const __m512d I  = _mm512_mul_pd(half, _mm512_add_pd(C1v, D1v));
            const __m512d A1 = _mm512_mul_pd(half, _mm512_sub_pd(C1v, D1v));

            const __m512d B0 = _mm512_fmadd_pd(vc, R, _mm512_mul_pd(vs, I));
            const __m512d B1 = _mm512_fmsub_pd(vc, I, _mm512_mul_pd(vs, R));

            _mm512_storeu_pd(C0p + n, A0);
            _mm512_storeu_pd(C1p + n, B0);
            _mm512_storeu_pd(D0p + n, A1);
            _mm512_storeu_pd(D1p + n, B1);
        }
    }
#endif
#if BRUUN_LEVEL >= 2
    {
        const __m256d half = _mm256_set1_pd(0.5);
        const __m256d vc = _mm256_set1_pd(c_scalar);
        const __m256d vs = _mm256_set1_pd(s_scalar);

        for (; n + 3 < q; n += 4) {
            const __m256d C0v = _mm256_loadu_pd(C0p + n);
            const __m256d C1v = _mm256_loadu_pd(C1p + n);
            const __m256d D0v = _mm256_loadu_pd(D0p + n);
            const __m256d D1v = _mm256_loadu_pd(D1p + n);

            const __m256d A0 = _mm256_mul_pd(half, _mm256_add_pd(C0v, D0v));
            const __m256d R  = _mm256_mul_pd(half, _mm256_sub_pd(C0v, D0v));
            const __m256d I  = _mm256_mul_pd(half, _mm256_add_pd(C1v, D1v));
            const __m256d A1 = _mm256_mul_pd(half, _mm256_sub_pd(C1v, D1v));

            const __m256d B0 = _mm256_fmadd_pd(vc, R, _mm256_mul_pd(vs, I));
            const __m256d B1 = _mm256_fmsub_pd(vc, I, _mm256_mul_pd(vs, R));

            _mm256_storeu_pd(C0p + n, A0);
            _mm256_storeu_pd(C1p + n, B0);
            _mm256_storeu_pd(D0p + n, A1);
            _mm256_storeu_pd(D1p + n, B1);
        }
    }
#elif BRUUN_LEVEL == 1
    {
        const bruun_v2 half = V2_SET1(0.5);
        const bruun_v2 vc = V2_SET1(c_scalar);
        const bruun_v2 vs = V2_SET1(s_scalar);

        for (; n + 1 < q; n += 2) {
            const bruun_v2 C0v = V2_LD(C0p + n);
            const bruun_v2 C1v = V2_LD(C1p + n);
            const bruun_v2 D0v = V2_LD(D0p + n);
            const bruun_v2 D1v = V2_LD(D1p + n);

            const bruun_v2 A0 = V2_MUL(half, V2_ADD(C0v, D0v));
            const bruun_v2 R  = V2_MUL(half, V2_SUB(C0v, D0v));
            const bruun_v2 I  = V2_MUL(half, V2_ADD(C1v, D1v));
            const bruun_v2 A1 = V2_MUL(half, V2_SUB(C1v, D1v));

            const bruun_v2 B0 = V2_ADD(V2_MUL(vc, R), V2_MUL(vs, I));
            const bruun_v2 B1 = V2_SUB(V2_MUL(vc, I), V2_MUL(vs, R));

            V2_ST(C0p + n, A0);
            V2_ST(C1p + n, B0);
            V2_ST(D0p + n, A1);
            V2_ST(D1p + n, B1);
        }
    }
#endif

    for (; n < q; ++n) {
        const double C0v = C0p[n];
        const double C1v = C1p[n];
        const double D0v = D0p[n];
        const double D1v = D1p[n];

        const double A0 = 0.5 * (C0v + D0v);
        const double R  = 0.5 * (C0v - D0v);
        const double I  = 0.5 * (C1v + D1v);
        const double A1 = 0.5 * (C1v - D1v);

        C0p[n] = A0;
        C1p[n] = c_scalar * R + s_scalar * I;
        D0p[n] = A1;
        D1p[n] = c_scalar * I - s_scalar * R;
    }
}

// Exact inverse of norm2_fused: undo both child rotations and the parent
// rotation in one pass over the block. Caller guarantees q >= 16 so qh >= 8.
static inline void norm2_inv_fused(double* RESTRICT p, int q,
                                   double c, double s,
                                   double c0, double s0,
                                   double c1, double s1) {
    const int qh = q >> 1;
    double* RESTRICT A0 = p;
    double* RESTRICT B0 = p + q;
    double* RESTRICT A1 = p + 2*q;
    double* RESTRICT B1 = p + 3*q;

    int n = 0;

#if BRUUN_LEVEL >= 3
    {
        const __m512d hf  = _mm512_set1_pd(0.5);
        const __m512d vc  = _mm512_set1_pd(c),  vs  = _mm512_set1_pd(s);
        const __m512d vc0 = _mm512_set1_pd(c0), vs0 = _mm512_set1_pd(s0);
        const __m512d vc1 = _mm512_set1_pd(c1), vs1 = _mm512_set1_pd(s1);

        for (; n + 7 < qh; n += 8) {
            const __m512d A0n = _mm512_loadu_pd(A0 + n);
            const __m512d A0h = _mm512_loadu_pd(A0 + qh + n);
            const __m512d B0n = _mm512_loadu_pd(B0 + n);
            const __m512d B0h = _mm512_loadu_pd(B0 + qh + n);
            const __m512d A1n = _mm512_loadu_pd(A1 + n);
            const __m512d A1h = _mm512_loadu_pd(A1 + qh + n);
            const __m512d B1n = _mm512_loadu_pd(B1 + n);
            const __m512d B1h = _mm512_loadu_pd(B1 + qh + n);

            const __m512d u0 = _mm512_mul_pd(hf, _mm512_add_pd(A0n, B0n));
            const __m512d R0 = _mm512_mul_pd(hf, _mm512_sub_pd(A0n, B0n));
            const __m512d I0 = _mm512_mul_pd(hf, _mm512_add_pd(A0h, B0h));
            const __m512d w0 = _mm512_mul_pd(hf, _mm512_sub_pd(A0h, B0h));
            const __m512d uh = _mm512_fmadd_pd(vc0, R0, _mm512_mul_pd(vs0, I0));
            const __m512d wh = _mm512_fmsub_pd(vc0, I0, _mm512_mul_pd(vs0, R0));

            const __m512d v0 = _mm512_mul_pd(hf, _mm512_add_pd(A1n, B1n));
            const __m512d R1 = _mm512_mul_pd(hf, _mm512_sub_pd(A1n, B1n));
            const __m512d I1 = _mm512_mul_pd(hf, _mm512_add_pd(A1h, B1h));
            const __m512d x0 = _mm512_mul_pd(hf, _mm512_sub_pd(A1h, B1h));
            const __m512d vh = _mm512_fmadd_pd(vc1, R1, _mm512_mul_pd(vs1, I1));
            const __m512d xh = _mm512_fmsub_pd(vc1, I1, _mm512_mul_pd(vs1, R1));

            const __m512d a0n = _mm512_mul_pd(hf, _mm512_add_pd(u0, v0));
            const __m512d Rn  = _mm512_mul_pd(hf, _mm512_sub_pd(u0, v0));
            const __m512d In  = _mm512_mul_pd(hf, _mm512_add_pd(w0, x0));
            const __m512d a1n = _mm512_mul_pd(hf, _mm512_sub_pd(w0, x0));
            const __m512d a0h = _mm512_mul_pd(hf, _mm512_add_pd(uh, vh));
            const __m512d Rh  = _mm512_mul_pd(hf, _mm512_sub_pd(uh, vh));
            const __m512d Ih  = _mm512_mul_pd(hf, _mm512_add_pd(wh, xh));
            const __m512d a1h = _mm512_mul_pd(hf, _mm512_sub_pd(wh, xh));

            _mm512_storeu_pd(A0 + n,      a0n);
            _mm512_storeu_pd(A0 + qh + n, a0h);
            _mm512_storeu_pd(B0 + n,      _mm512_fmadd_pd(vc, Rn, _mm512_mul_pd(vs, In)));
            _mm512_storeu_pd(B0 + qh + n, _mm512_fmadd_pd(vc, Rh, _mm512_mul_pd(vs, Ih)));
            _mm512_storeu_pd(A1 + n,      a1n);
            _mm512_storeu_pd(A1 + qh + n, a1h);
            _mm512_storeu_pd(B1 + n,      _mm512_fmsub_pd(vc, In, _mm512_mul_pd(vs, Rn)));
            _mm512_storeu_pd(B1 + qh + n, _mm512_fmsub_pd(vc, Ih, _mm512_mul_pd(vs, Rh)));
        }
    }
#endif
#if BRUUN_LEVEL >= 2
    {
        const __m256d hf  = _mm256_set1_pd(0.5);
        const __m256d vc  = _mm256_set1_pd(c),  vs  = _mm256_set1_pd(s);
        const __m256d vc0 = _mm256_set1_pd(c0), vs0 = _mm256_set1_pd(s0);
        const __m256d vc1 = _mm256_set1_pd(c1), vs1 = _mm256_set1_pd(s1);

        for (; n + 3 < qh; n += 4) {
            const __m256d A0n = _mm256_loadu_pd(A0 + n);
            const __m256d A0h = _mm256_loadu_pd(A0 + qh + n);
            const __m256d B0n = _mm256_loadu_pd(B0 + n);
            const __m256d B0h = _mm256_loadu_pd(B0 + qh + n);
            const __m256d A1n = _mm256_loadu_pd(A1 + n);
            const __m256d A1h = _mm256_loadu_pd(A1 + qh + n);
            const __m256d B1n = _mm256_loadu_pd(B1 + n);
            const __m256d B1h = _mm256_loadu_pd(B1 + qh + n);

            const __m256d u0 = _mm256_mul_pd(hf, _mm256_add_pd(A0n, B0n));
            const __m256d R0 = _mm256_mul_pd(hf, _mm256_sub_pd(A0n, B0n));
            const __m256d I0 = _mm256_mul_pd(hf, _mm256_add_pd(A0h, B0h));
            const __m256d w0 = _mm256_mul_pd(hf, _mm256_sub_pd(A0h, B0h));
            const __m256d uh = _mm256_fmadd_pd(vc0, R0, _mm256_mul_pd(vs0, I0));
            const __m256d wh = _mm256_fmsub_pd(vc0, I0, _mm256_mul_pd(vs0, R0));

            const __m256d v0 = _mm256_mul_pd(hf, _mm256_add_pd(A1n, B1n));
            const __m256d R1 = _mm256_mul_pd(hf, _mm256_sub_pd(A1n, B1n));
            const __m256d I1 = _mm256_mul_pd(hf, _mm256_add_pd(A1h, B1h));
            const __m256d x0 = _mm256_mul_pd(hf, _mm256_sub_pd(A1h, B1h));
            const __m256d vh = _mm256_fmadd_pd(vc1, R1, _mm256_mul_pd(vs1, I1));
            const __m256d xh = _mm256_fmsub_pd(vc1, I1, _mm256_mul_pd(vs1, R1));

            const __m256d a0n = _mm256_mul_pd(hf, _mm256_add_pd(u0, v0));
            const __m256d Rn  = _mm256_mul_pd(hf, _mm256_sub_pd(u0, v0));
            const __m256d In  = _mm256_mul_pd(hf, _mm256_add_pd(w0, x0));
            const __m256d a1n = _mm256_mul_pd(hf, _mm256_sub_pd(w0, x0));
            const __m256d a0h = _mm256_mul_pd(hf, _mm256_add_pd(uh, vh));
            const __m256d Rh  = _mm256_mul_pd(hf, _mm256_sub_pd(uh, vh));
            const __m256d Ih  = _mm256_mul_pd(hf, _mm256_add_pd(wh, xh));
            const __m256d a1h = _mm256_mul_pd(hf, _mm256_sub_pd(wh, xh));

            _mm256_storeu_pd(A0 + n,      a0n);
            _mm256_storeu_pd(A0 + qh + n, a0h);
            _mm256_storeu_pd(B0 + n,      _mm256_fmadd_pd(vc, Rn, _mm256_mul_pd(vs, In)));
            _mm256_storeu_pd(B0 + qh + n, _mm256_fmadd_pd(vc, Rh, _mm256_mul_pd(vs, Ih)));
            _mm256_storeu_pd(A1 + n,      a1n);
            _mm256_storeu_pd(A1 + qh + n, a1h);
            _mm256_storeu_pd(B1 + n,      _mm256_fmsub_pd(vc, In, _mm256_mul_pd(vs, Rn)));
            _mm256_storeu_pd(B1 + qh + n, _mm256_fmsub_pd(vc, Ih, _mm256_mul_pd(vs, Rh)));
        }
    }
#elif BRUUN_LEVEL == 1
    {
        const bruun_v2 hf  = V2_SET1(0.5);
        const bruun_v2 vc  = V2_SET1(c),  vs  = V2_SET1(s);
        const bruun_v2 vc0 = V2_SET1(c0), vs0 = V2_SET1(s0);
        const bruun_v2 vc1 = V2_SET1(c1), vs1 = V2_SET1(s1);

        for (; n + 1 < qh; n += 2) {
            const bruun_v2 A0n = V2_LD(A0 + n);
            const bruun_v2 A0h = V2_LD(A0 + qh + n);
            const bruun_v2 B0n = V2_LD(B0 + n);
            const bruun_v2 B0h = V2_LD(B0 + qh + n);
            const bruun_v2 A1n = V2_LD(A1 + n);
            const bruun_v2 A1h = V2_LD(A1 + qh + n);
            const bruun_v2 B1n = V2_LD(B1 + n);
            const bruun_v2 B1h = V2_LD(B1 + qh + n);

            const bruun_v2 u0 = V2_MUL(hf, V2_ADD(A0n, B0n));
            const bruun_v2 R0 = V2_MUL(hf, V2_SUB(A0n, B0n));
            const bruun_v2 I0 = V2_MUL(hf, V2_ADD(A0h, B0h));
            const bruun_v2 w0 = V2_MUL(hf, V2_SUB(A0h, B0h));
            const bruun_v2 uh = V2_ADD(V2_MUL(vc0, R0), V2_MUL(vs0, I0));
            const bruun_v2 wh = V2_SUB(V2_MUL(vc0, I0), V2_MUL(vs0, R0));

            const bruun_v2 v0 = V2_MUL(hf, V2_ADD(A1n, B1n));
            const bruun_v2 R1 = V2_MUL(hf, V2_SUB(A1n, B1n));
            const bruun_v2 I1 = V2_MUL(hf, V2_ADD(A1h, B1h));
            const bruun_v2 x0 = V2_MUL(hf, V2_SUB(A1h, B1h));
            const bruun_v2 vh = V2_ADD(V2_MUL(vc1, R1), V2_MUL(vs1, I1));
            const bruun_v2 xh = V2_SUB(V2_MUL(vc1, I1), V2_MUL(vs1, R1));

            const bruun_v2 a0n = V2_MUL(hf, V2_ADD(u0, v0));
            const bruun_v2 Rn  = V2_MUL(hf, V2_SUB(u0, v0));
            const bruun_v2 In  = V2_MUL(hf, V2_ADD(w0, x0));
            const bruun_v2 a1n = V2_MUL(hf, V2_SUB(w0, x0));
            const bruun_v2 a0h = V2_MUL(hf, V2_ADD(uh, vh));
            const bruun_v2 Rh  = V2_MUL(hf, V2_SUB(uh, vh));
            const bruun_v2 Ih  = V2_MUL(hf, V2_ADD(wh, xh));
            const bruun_v2 a1h = V2_MUL(hf, V2_SUB(wh, xh));

            V2_ST(A0 + n,      a0n);
            V2_ST(A0 + qh + n, a0h);
            V2_ST(B0 + n,      V2_ADD(V2_MUL(vc, Rn), V2_MUL(vs, In)));
            V2_ST(B0 + qh + n, V2_ADD(V2_MUL(vc, Rh), V2_MUL(vs, Ih)));
            V2_ST(A1 + n,      a1n);
            V2_ST(A1 + qh + n, a1h);
            V2_ST(B1 + n,      V2_SUB(V2_MUL(vc, In), V2_MUL(vs, Rn)));
            V2_ST(B1 + qh + n, V2_SUB(V2_MUL(vc, Ih), V2_MUL(vs, Rh)));
        }
    }
#endif

    for (; n < qh; ++n) {
        const double A0n = A0[n], A0h = A0[qh + n];
        const double B0n = B0[n], B0h = B0[qh + n];
        const double A1n = A1[n], A1h = A1[qh + n];
        const double B1n = B1[n], B1h = B1[qh + n];

        const double u0 = 0.5 * (A0n + B0n);
        const double R0 = 0.5 * (A0n - B0n);
        const double I0 = 0.5 * (A0h + B0h);
        const double w0 = 0.5 * (A0h - B0h);
        const double uh = c0 * R0 + s0 * I0;
        const double wh = c0 * I0 - s0 * R0;

        const double v0 = 0.5 * (A1n + B1n);
        const double R1 = 0.5 * (A1n - B1n);
        const double I1 = 0.5 * (A1h + B1h);
        const double x0 = 0.5 * (A1h - B1h);
        const double vh = c1 * R1 + s1 * I1;
        const double xh = c1 * I1 - s1 * R1;

        const double a0n = 0.5 * (u0 + v0);
        const double Rn  = 0.5 * (u0 - v0);
        const double In  = 0.5 * (w0 + x0);
        const double a1n = 0.5 * (w0 - x0);
        const double a0h = 0.5 * (uh + vh);
        const double Rh  = 0.5 * (uh - vh);
        const double Ih  = 0.5 * (wh + xh);
        const double a1h = 0.5 * (wh - xh);

        A0[n] = a0n;
        A0[qh + n] = a0h;
        B0[n] = c * Rn + s * In;
        B0[qh + n] = c * Rh + s * Ih;
        A1[n] = a1n;
        A1[qh + n] = a1h;
        B1[n] = c * In - s * Rn;
        B1[qh + n] = c * Ih - s * Rh;
    }
}

class RFFT {
public:
    explicit RFFT(int n, bool fuse_tail = true)
        : N(n), L(ilog2_pow2(n)), NB(n / 2 + 1), fuse_tail(fuse_tail && n >= 32),
          IDX(n / 2), OUTIDX(n / 2), C(n / 2), S(n / 2)
    {
        if (!is_power2(N) || N < 4) throw std::invalid_argument("Bruun RFFT requires power-of-two N >= 4");

        IDX[0] = 0;
        C[0] = 0.0;
        S[0] = 0.0;

        // Build the Bruun angle table directly from the covering-map half-angle
        // recurrence. The old constructor built a full T[N] cosine table and then
        // sampled it:
        //     C[m] = cos(pi * IDX[m] / N), S[m] = sin(pi * IDX[m] / N)
        // That costs O(N) libm cos() calls and dominates huge-N setup. Here the
        // same values are generated by the Bruun tree:
        //     alpha(1) = pi/4
        //     alpha(2m)   = alpha(m)/2
        //     alpha(2m+1) = pi/2 - alpha(m)/2
        // using only sqrt/adds.
        for (int m = 1; m < N / 2; ++m) {
            IDX[m] = bruun_idx_int(m, L);
        }

        if (N >= 4) {
            const double r = std::sqrt(0.5);
            C[1] = r;
            S[1] = r;
        }

        for (int m = 1; 2*m < N / 2; ++m) {
            const double c = C[m];
            const double s = S[m];
            // ce = cos(alpha/2) is stable for alpha in (0, pi/2): 1 + c never cancels.
            // se = sin(alpha/2) via sqrt((1 - c)/2) cancels catastrophically as
            // c -> 1 (deep small-angle lineages), costing ~log(N) digits at large N.
            // sin(alpha/2) = sin(alpha) / (2 cos(alpha/2)) has no cancellation.
            const double ce = std::sqrt(0.5 * (1.0 + c));
            const double se = s / (2.0 * ce);

            C[2*m] = ce;
            S[2*m] = se;

            if (2*m + 1 < N / 2) {
                C[2*m + 1] = se;
                S[2*m + 1] = ce;
            }
        }

        // OUTIDX is the native complex-output slot for each Bruun leaf.
        // Default is ordinary FFTW frequency-bin order.
        OUTIDX[0] = 0;
        for (int m = 1; m < N / 2; ++m) OUTIDX[m] = IDX[m];

#if defined(BRUUN_HEAPOPT_SPECTRUM_ORDER)
        // Legal fast-layout constraint:
        //   every Bruun factor node must keep a contiguous interval of final leaves.
        // DFS preorder fragments factor subtrees. This order keeps all heap/factor
        // intervals contiguous but chooses sibling orientations inside each level
        // to reduce adjacent frequency travel.
        NATIVE_POS.assign(N / 2, 0);
        NATIVE_LEAF.assign(N / 2, 0);

        std::vector<int> inv_k(N / 2, 0);
        for (int m = 1; m < N / 2; ++m) inv_k[IDX[m]] = m;

        std::vector<int> k_order;
        k_order.reserve(N / 2);
        const int M = N / 2;

        auto cyclic_dist = [M](int a, int b) {
            int d = std::abs(a - b);
            return std::min(d, M - d);
        };

        std::vector<int> prev;
        prev.push_back(M / 2);
        k_order.push_back(M / 2);

        while (true) {
            std::vector<std::pair<int,int>> pairs;
            pairs.reserve(prev.size());
            for (int k : prev) {
                if ((k & 1) == 0) {
                    pairs.push_back(std::make_pair(k / 2, M - k / 2));
                }
            }
            if (pairs.empty()) break;

            // Same lexicographic orientation optimizer as before, but linear-time
            // and linear-memory. The previous prototype stored a full candidate
            // sequence inside each DP state, causing quadratic copying at huge N.
            // This keeps only costs plus backpointers, then reconstructs one level.
            const size_t P = pairs.size();

            std::vector<int> max0(P, 0), max1(P, 0);
            std::vector<long long> sum0(P, 0), sum1(P, 0);
            std::vector<unsigned char> back0(P, 0), back1(P, 0);

            auto start_of = [&pairs](size_t i, int o) {
                return o == 0 ? pairs[i].first : pairs[i].second;
            };
            auto end_of = [&pairs](size_t i, int o) {
                return o == 0 ? pairs[i].second : pairs[i].first;
            };
            auto better = [](int ma, long long sa, int mb, long long sb) {
                return ma < mb || (ma == mb && sa < sb);
            };

            for (size_t pi = 1; pi < P; ++pi) {
                for (int o = 0; o < 2; ++o) {
                    const int first = start_of(pi, o);

                    const int j0 = cyclic_dist(end_of(pi - 1, 0), first);
                    const int cand0_max = std::max(max0[pi - 1], j0);
                    const long long cand0_sum = sum0[pi - 1] + j0;

                    const int j1 = cyclic_dist(end_of(pi - 1, 1), first);
                    const int cand1_max = std::max(max1[pi - 1], j1);
                    const long long cand1_sum = sum1[pi - 1] + j1;

                    if (better(cand0_max, cand0_sum, cand1_max, cand1_sum)) {
                        if (o == 0) {
                            max0[pi] = cand0_max;
                            sum0[pi] = cand0_sum;
                            back0[pi] = 0;
                        } else {
                            max1[pi] = cand0_max;
                            sum1[pi] = cand0_sum;
                            back1[pi] = 0;
                        }
                    } else {
                        if (o == 0) {
                            max0[pi] = cand1_max;
                            sum0[pi] = cand1_sum;
                            back0[pi] = 1;
                        } else {
                            max1[pi] = cand1_max;
                            sum1[pi] = cand1_sum;
                            back1[pi] = 1;
                        }
                    }
                }
            }

            int choose =
                better(max0[P - 1], sum0[P - 1], max1[P - 1], sum1[P - 1]) ? 0 : 1;

            std::vector<unsigned char> orient(P);
            for (size_t rr = P; rr-- > 0;) {
                orient[rr] = static_cast<unsigned char>(choose);
                choose = (choose == 0) ? back0[rr] : back1[rr];
            }

            prev.clear();
            prev.reserve(2 * P);
            for (size_t pi = 0; pi < P; ++pi) {
                const int a = pairs[pi].first;
                const int b = pairs[pi].second;
                if (orient[pi] == 0) {
                    prev.push_back(a);
                    prev.push_back(b);
                } else {
                    prev.push_back(b);
                    prev.push_back(a);
                }
            }

            for (int k : prev) k_order.push_back(k);
        }

        int pos = 1;
        for (int k : k_order) {
            const int m = inv_k[k];
            if (m <= 0 || m >= N / 2) continue;
            NATIVE_POS[m] = pos;
            NATIVE_LEAF[pos] = m;
            ++pos;
        }

        for (int m = 1; m < N / 2; ++m) {
            if (NATIVE_POS[m] == 0) {
                NATIVE_POS[m] = pos;
                NATIVE_LEAF[pos] = m;
                ++pos;
            }
        }

        for (int m = 1; m < N / 2; ++m) OUTIDX[m] = NATIVE_POS[m];
#endif

        // Packed per-leaf metadata: one contiguous 144-byte entry per depth-3 leaf
        // block, read as a sequential stream during the transform instead of
        // heap-strided picks from C, S, and IDX.
        if (N >= 32) {
            TW.resize(N / 16);
            for (int m = 1; m < N / 16; ++m) {
                LeafTw& e = TW[m];
                for (int g = 0; g < 4; ++g) {
                    e.c4[g] = C[4*m + g];
                    e.s4[g] = S[4*m + g];
                }
                e.c2[0] = C[2*m];
                e.c2[1] = C[2*m + 1];
                e.s2[0] = S[2*m];
                e.s2[1] = S[2*m + 1];
                e.c1 = C[m];
                e.s1 = S[m];
                for (int j = 0; j < 8; ++j) e.idx[j] = OUTIDX[8*m + j];
            }
        }

        // Inverse bin permutation for sequential standard-order packing:
        // KINV[k] = m such that IDX[m] = k. IDX is a bijection [1, N/2) -> [1, N/2).
        KINV.assign(N / 2, 0);
        for (int m = 1; m < N / 2; ++m) KINV[IDX[m]] = m;
    }

    int size() const { return N; }
    int bins() const { return NB; }
    int work_size() const { return N; }
    int work_size_f32() const { return 2 * N; }
    int native_scratch_size() const { return NB; }

    bool standard_output_uses_two_phase() const {
#if BRUUN_LEVEL >= 2
        return N > 8192;
#elif BRUUN_LEVEL == 1
        return N > 1048576;
#else
        return false;
#endif
    }

    const char* standard_output_policy_name() const {
        if (standard_output_uses_two_phase()) {
            return "two-phase-standard-pack";
        }
        return "fused-scatter-plus-layout-convert";
    }

    // Fast native-output Bruun transform.
    // With BRUUN_HEAPOPT_SPECTRUM_ORDER, X is in heapopt Bruun-native order.
    // Without BRUUN_HEAPOPT_SPECTRUM_ORDER, native order is ordinary FFTW bin order.
    void forward_native(const double* RESTRICT input, complex_t* RESTRICT X, double* RESTRICT work) const {
        if (fuse_tail && N >= 64) {
            forward_recursive(input, work, X);
            return;
        }

        std::memcpy(work, input, sizeof(double) * N);

        if (fuse_tail) {
            forward_fused_tail(work, X);
        } else {
            forward_residues_inplace(work);
            residues_to_complex(work, X);
        }
    }

    void forward_standard_f32(const float* RESTRICT input,
                              complex_f32_t* RESTRICT X,
                              float* RESTRICT work,
                              complex_f32_t* RESTRICT native_tmp) const {
        (void)native_tmp;
        forward_standard_fft_f32(input, X, work);
    }

    void forward_native_f32(const float* RESTRICT input,
                            complex_f32_t* RESTRICT X,
                            float* RESTRICT work) const {
#if defined(BRUUN_HEAPOPT_SPECTRUM_ORDER)
        if (N >= 32) {
            std::vector<complex_f32_t> standard(NB);
            forward_standard_fft_f32(input, standard.data(), work);
            standard_to_native_complex_f32(standard.data(), X);
            return;
        }
#endif
        forward_standard_fft_f32(input, X, work);
    }

    // Standard FFTW-like real -> complex interface, using caller-provided native scratch.
    // X[k] is the ordinary k-th FFT bin on return.
    void forward_standard(const double* RESTRICT input,
                          complex_t* RESTRICT X,
                          double* RESTRICT work,
                          complex_t* RESTRICT native_tmp) const {
        if (standard_output_uses_two_phase()) {
            (void)native_tmp;
            forward_standard_two_phase(input, work, X);
            return;
        }
#if defined(BRUUN_HEAPOPT_SPECTRUM_ORDER)
        forward_native(input, native_tmp, work);
        native_to_standard_complex(native_tmp, X);
#else
        (void)native_tmp;
        forward_native(input, X, work);
#endif
    }

    // Convenience standard-output API. This allocates temporary native spectrum
    // storage when heapopt native order is enabled. Hot loops should prefer
    // forward_standard(..., native_tmp) to reuse that scratch.
    void forward(const double* RESTRICT input, complex_t* RESTRICT X, double* RESTRICT work) const {
#if defined(BRUUN_HEAPOPT_SPECTRUM_ORDER) && !defined(BRUUN_NATIVE_OUTPUT)
        std::vector<complex_t> native_tmp(NB);
        forward_standard(input, X, work, native_tmp.data());
#else
        forward_native(input, X, work);
#endif
    }

    // Standard FFTW-like complex -> real inverse interface.
    // This inverse is intentionally simple/unfused; the current competition target is r2c forward.
    void inverse(const complex_t* RESTRICT X, double* RESTRICT out) const {
        complex_to_residues(X, out);
        inverse_residues_inplace(out);
    }

    void inverse_f32(const complex_f32_t* RESTRICT X, float* RESTRICT out) const {
        inverse_standard_fft_f32(X, out);
    }

    void inverse_native_f32(const complex_f32_t* RESTRICT Xnative, float* RESTRICT out) const {
#if defined(BRUUN_HEAPOPT_SPECTRUM_ORDER)
        if (N >= 32) {
            std::vector<complex_f32_t> standard(NB);
            native_to_standard_complex_f32(Xnative, standard.data());
            inverse_standard_fft_f32(standard.data(), out);
            return;
        }
#endif
        inverse_standard_fft_f32(Xnative, out);
    }

    // Convert the transform's native complex spectrum layout to standard FFTW order.
    // With BRUUN_HEAPOPT_SPECTRUM_ORDER, native nontrivial bins are in a
    // block-contiguous, sibling-orientation-optimized Bruun covering order.
    // Without it, native order already is standard FFTW bin order.
    void native_to_standard_complex(const complex_t* RESTRICT nativeX,
                                    complex_t* RESTRICT standardX) const {
#if defined(BRUUN_HEAPOPT_SPECTRUM_ORDER)
        if (N < 32) {
            std::memcpy(standardX, nativeX, sizeof(complex_t) * NB);
            return;
        }
        standardX[0] = nativeX[0];
        standardX[N / 2] = nativeX[N / 2];
        for (int m = 1; m < N / 2; ++m) {
            standardX[IDX[m]] = nativeX[NATIVE_POS[m]];
        }
#else
        std::memcpy(standardX, nativeX, sizeof(complex_t) * NB);
#endif
    }

    // Convert standard FFTW-order bins into this plan's native complex layout.
    void standard_to_native_complex(const complex_t* RESTRICT standardX,
                                    complex_t* RESTRICT nativeX) const {
#if defined(BRUUN_HEAPOPT_SPECTRUM_ORDER)
        if (N < 32) {
            std::memcpy(nativeX, standardX, sizeof(complex_t) * NB);
            return;
        }
        nativeX[0] = standardX[0];
        nativeX[N / 2] = standardX[N / 2];
        for (int m = 1; m < N / 2; ++m) {
            nativeX[NATIVE_POS[m]] = standardX[IDX[m]];
        }
#else
        std::memcpy(nativeX, standardX, sizeof(complex_t) * NB);
#endif
    }

    void native_to_standard_complex_f32(const complex_f32_t* RESTRICT nativeX,
                                        complex_f32_t* RESTRICT standardX) const {
#if defined(BRUUN_HEAPOPT_SPECTRUM_ORDER)
        if (N < 32) {
            std::memcpy(standardX, nativeX, sizeof(complex_f32_t) * NB);
            return;
        }
        standardX[0] = nativeX[0];
        standardX[N / 2] = nativeX[N / 2];
        for (int m = 1; m < N / 2; ++m) {
            standardX[IDX[m]] = nativeX[NATIVE_POS[m]];
        }
#else
        std::memcpy(standardX, nativeX, sizeof(complex_f32_t) * NB);
#endif
    }

    void standard_to_native_complex_f32(const complex_f32_t* RESTRICT standardX,
                                        complex_f32_t* RESTRICT nativeX) const {
#if defined(BRUUN_HEAPOPT_SPECTRUM_ORDER)
        if (N < 32) {
            std::memcpy(nativeX, standardX, sizeof(complex_f32_t) * NB);
            return;
        }
        nativeX[0] = standardX[0];
        nativeX[N / 2] = standardX[N / 2];
        for (int m = 1; m < N / 2; ++m) {
            nativeX[NATIVE_POS[m]] = standardX[IDX[m]];
        }
#else
        std::memcpy(nativeX, standardX, sizeof(complex_f32_t) * NB);
#endif
    }

    // -----------------------------------------------------------------------
    // Coordinate systems and the call matrix.
    //
    // This plan exposes three representations of the spectrum of a length-N
    // real signal:
    //
    //   1. Residue domain ("residues"): N doubles. v[0], v[1] carry the DC and
    //      Nyquist content (X[0] = v[0] + v[1], X[N/2] = v[0] - v[1]); for each
    //      nontrivial Bruun leaf m in [1, N/2) the pair (v[2m], v[2m+1])
    //      represents the complex bin X[IDX[m]] = v[2m] - i * v[2m+1]. This is
    //      the transform's own CRT coordinate system. It costs no output
    //      permutation at all and is the fastest representation to produce,
    //      to consume, and to multiply filters in.
    //
    //   2. Native spectrum: NB complex bins in the plan's native slot order.
    //      With BRUUN_HEAPOPT_SPECTRUM_ORDER this is the heap-contiguous,
    //      sibling-orientation-optimized Bruun covering order; without it,
    //      native order is ordinary FFTW bin order.
    //
    //   3. Standard spectrum: NB complex bins in ordinary FFTW r2c order.
    //
    // Transform calls:
    //   forward_residues(in, res)        time -> residues        (fastest)
    //   inverse_residues(res)            residues -> time, in place
    //   forward_native(in, X, work)      time -> native spectrum
    //   inverse_native(X, out)           native spectrum -> time
    //   forward_standard(in, X, work, t) time -> standard bins (uses scratch t)
    //   forward(in, X, work)             convenience standard-bin forward
    //   inverse(X, out)                  standard bins -> time
    //   filter_signal(in, RF, out)       time -> time through one residue-domain
    //                                    filter, no bin conversion anywhere
    //
    // Layout converters:
    //   native_to_standard_complex / standard_to_native_complex
    //   residue_filter_from_standard / residue_filter_from_real
    // -----------------------------------------------------------------------

    // Time -> residues. Uses the fused-copy depth-first path when available.
    void forward_residues(const double* RESTRICT input, double* RESTRICT residues) const {
        if (fuse_tail && N >= 64) {
            forward_residues_recursive(input, residues);
            return;
        }
        std::memcpy(residues, input, sizeof(double) * N);
        forward_residues_inplace(residues);
    }

    // Residues -> time, in place.
    void inverse_residues(double* RESTRICT residues_signal) const {
        inverse_residues_inplace(residues_signal);
    }

    // Native spectrum -> time. With BRUUN_HEAPOPT_SPECTRUM_ORDER this reads the
    // heapopt native layout sequentially; otherwise identical to inverse().
    void inverse_native(const complex_t* RESTRICT Xnative, double* RESTRICT out) const {
#if defined(BRUUN_HEAPOPT_SPECTRUM_ORDER)
        if (N < 32) {
            inverse(Xnative, out);
            return;
        }
        out[0] = 0.5 * (Xnative[0].re + Xnative[N / 2].re);
        out[1] = 0.5 * (Xnative[0].re - Xnative[N / 2].re);
        for (int pos = 1; pos < N / 2; ++pos) {
            const int m = NATIVE_LEAF[pos];
            out[2*m] = Xnative[pos].re;
            out[2*m + 1] = -Xnative[pos].im;
        }
        inverse_residues_inplace(out);
#else
        inverse(Xnative, out);
#endif
    }

    // -----------------------------------------------------------------------
    // Residue-domain filters.
    //
    // A filter is an ordinary frequency response H[k] for k = 0..N/2 given in
    // standard FFTW r2c bin order. residue_filter_from_standard() converts it
    // once into a packed residue-domain coefficient array RF of filter_size()
    // doubles, aligned pair-for-pair with the residue vector. After that,
    // apply_residue_filter() is one sequential SIMD pass, and filter_signal()
    // runs time -> residues -> multiply -> time with no bin permutation at any
    // point. H[0] and H[N/2] must be real for a real-output filter; their
    // imaginary parts are ignored.
    //
    // RF layout: RF[0] = (H[0] + H[N/2])/2 and RF[1] = (H[0] - H[N/2])/2 form
    // the symmetric 2x2 acting on (v[0], v[1]); for m in [1, N/2),
    // RF[2m] = Re H[IDX[m]] and RF[2m+1] = Im H[IDX[m]], applied as the
    // conjugate-coordinate complex multiply
    //   y0 = hr*r0 + hi*r1,  y1 = hr*r1 - hi*r0.
    // -----------------------------------------------------------------------

    int filter_size() const { return N; }

    void residue_filter_from_standard(const complex_t* RESTRICT H, double* RESTRICT RF) const {
        RF[0] = 0.5 * (H[0].re + H[N / 2].re);
        RF[1] = 0.5 * (H[0].re - H[N / 2].re);
        for (int m = 1; m < N / 2; ++m) {
            const int k = IDX[m];
            RF[2*m] = H[k].re;
            RF[2*m + 1] = H[k].im;
        }
    }

    // Zero-phase (real) frequency response, Hmag[k] for k = 0..N/2.
    void residue_filter_from_real(const double* RESTRICT Hmag, double* RESTRICT RF) const {
        RF[0] = 0.5 * (Hmag[0] + Hmag[N / 2]);
        RF[1] = 0.5 * (Hmag[0] - Hmag[N / 2]);
        for (int m = 1; m < N / 2; ++m) {
            RF[2*m] = Hmag[IDX[m]];
            RF[2*m + 1] = 0.0;
        }
    }

    // One streaming pass: residues *= filter, entirely in residue coordinates.
    void apply_residue_filter(double* RESTRICT v, const double* RESTRICT RF) const {
        const double v0 = v[0];
        const double v1 = v[1];
        v[0] = RF[0] * v0 + RF[1] * v1;
        v[1] = RF[1] * v0 + RF[0] * v1;

        int j = 2;
        const int end = N;

#if BRUUN_LEVEL >= 3
        {
            const __m512d negodd = _mm512_set_pd(-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0);
            for (; j + 7 < end; j += 8) {
                const __m512d r = _mm512_loadu_pd(v + j);
                const __m512d h = _mm512_loadu_pd(RF + j);
                const __m512d hd = _mm512_movedup_pd(h);
                const __m512d ho = _mm512_permute_pd(h, 0xFF);
                const __m512d rs = _mm512_castsi512_pd(_mm512_xor_si512(
                    _mm512_castpd_si512(_mm512_permute_pd(r, 0x55)),
                    _mm512_castpd_si512(negodd)));
                _mm512_storeu_pd(v + j, _mm512_fmadd_pd(hd, r, _mm512_mul_pd(ho, rs)));
            }
        }
#endif
#if BRUUN_LEVEL >= 2
        {
            const __m256d negodd = _mm256_set_pd(-0.0, 0.0, -0.0, 0.0);
            for (; j + 3 < end; j += 4) {
                const __m256d r = _mm256_loadu_pd(v + j);
                const __m256d h = _mm256_loadu_pd(RF + j);
                const __m256d hd = _mm256_movedup_pd(h);
                const __m256d ho = _mm256_permute_pd(h, 0xF);
                const __m256d rs = _mm256_xor_pd(_mm256_permute_pd(r, 0x5), negodd);
                _mm256_storeu_pd(v + j, _mm256_fmadd_pd(hd, r, _mm256_mul_pd(ho, rs)));
            }
        }
#elif BRUUN_LEVEL == 1
        for (; j + 1 < end; j += 2) {
            const bruun_v2 r = V2_LD(v + j);
            const bruun_v2 h = V2_LD(RF + j);
            const bruun_v2 rs = V2_NEGHI(V2_UNPLO(V2_DUP1(r), r)); // [r1, -r0]
            V2_ST(v + j, V2_ADD(V2_MUL(V2_DUP0(h), r), V2_MUL(V2_DUP1(h), rs)));
        }
#endif

        for (; j < end; j += 2) {
            const double r0 = v[j];
            const double r1 = v[j + 1];
            const double hr = RF[j];
            const double hi = RF[j + 1];
            v[j] = hr * r0 + hi * r1;
            v[j + 1] = hr * r1 - hi * r0;
        }
    }

    // time -> residues -> filter -> time, with no bin-order conversion at any
    // point. out must hold N doubles; out may not alias in.
    void filter_signal(const double* RESTRICT in, const double* RESTRICT RF, double* RESTRICT out) const {
        forward_residues(in, out);
        apply_residue_filter(out, RF);
        inverse_residues_inplace(out);
    }

private:
    int N;
    int L;
    int NB;
    bool fuse_tail;
    std::vector<int> IDX;
    std::vector<int> OUTIDX;
#if defined(BRUUN_HEAPOPT_SPECTRUM_ORDER)
    std::vector<int> NATIVE_POS;
    std::vector<int> NATIVE_LEAF;
#endif
    std::vector<double> C;
    std::vector<double> S;

    struct LeafTw {
        double c4[4];
        double s4[4];
        double c2[2];
        double s2[2];
        double c1;
        double s1;
        int32_t idx[8];
    };
    std::vector<LeafTw> TW;

    std::vector<int> KINV;

    static void complex_fft_f32(float* RESTRICT re, float* RESTRICT im, int n, bool inverse) {
        int j = 0;
        for (int i = 1; i < n; ++i) {
            int bit = n >> 1;
            while (j & bit) {
                j ^= bit;
                bit >>= 1;
            }
            j ^= bit;
            if (i < j) {
                std::swap(re[i], re[j]);
                std::swap(im[i], im[j]);
            }
        }

        for (int len = 2; len <= n; len <<= 1) {
            float angle = static_cast<float>(-2.0 * M_PI / static_cast<double>(len));
            if (inverse) {
                angle = -angle;
            }
            const float wlen_re = std::cos(angle);
            const float wlen_im = std::sin(angle);
            const int half = len >> 1;

            for (int i = 0; i < n; i += len) {
                float w_re = 1.0f;
                float w_im = 0.0f;
                for (int k = 0; k < half; ++k) {
                    const int even = i + k;
                    const int odd = even + half;
                    const float u_re = re[even];
                    const float u_im = im[even];
                    const float v_re = re[odd] * w_re - im[odd] * w_im;
                    const float v_im = re[odd] * w_im + im[odd] * w_re;

                    re[even] = u_re + v_re;
                    im[even] = u_im + v_im;
                    re[odd] = u_re - v_re;
                    im[odd] = u_im - v_im;

                    const float next_re = w_re * wlen_re - w_im * wlen_im;
                    const float next_im = w_re * wlen_im + w_im * wlen_re;
                    w_re = next_re;
                    w_im = next_im;
                }
            }
        }

        if (inverse) {
            const float scale = 1.0f / static_cast<float>(n);
            scale_complex_work_f32(re, im, n, scale);
        }
    }

    void forward_standard_fft_f32(const float* RESTRICT input,
                                  complex_f32_t* RESTRICT X,
                                  float* RESTRICT work) const {
        float* RESTRICT re = work;
        float* RESTRICT im = work + N;
        copy_real_to_complex_work_f32(input, re, im, N);

        complex_fft_f32(re, im, N, false);
        pack_standard_spectrum_f32(re, im, X, NB);
    }

    void inverse_standard_fft_f32(const complex_f32_t* RESTRICT X, float* RESTRICT out) const {
        std::vector<float> work(static_cast<size_t>(2 * N));
        float* RESTRICT re = work.data();
        float* RESTRICT im = work.data() + N;

        unpack_standard_spectrum_f32(X, re, im, N);

        complex_fft_f32(re, im, N, true);
        copy_real_output_f32(re, out, N);
    }

    static inline void norm_q1_fwd(double* RESTRICT p, double c, double s) {
        const double A0 = p[0];
        const double B0 = p[1];
        const double A1 = p[2];
        const double B1 = p[3];

        const double R = c * B0 - s * B1;
        const double I = s * B0 + c * B1;

        p[0] = A0 + R;
        p[1] = A1 + I;
        p[2] = A0 - R;
        p[3] = -A1 + I;
    }

    static inline void norm_q2_fwd(double* RESTRICT p, double c, double s) {
        for (int n = 0; n < 2; ++n) {
            const double A0 = p[n];
            const double B0 = p[2 + n];
            const double A1 = p[4 + n];
            const double B1 = p[6 + n];

            const double R = c * B0 - s * B1;
            const double I = s * B0 + c * B1;

            p[n] = A0 + R;
            p[2 + n] = A1 + I;
            p[4 + n] = A0 - R;
            p[6 + n] = -A1 + I;
        }
    }

    inline void codelet_d1_pack(const double* RESTRICT p, int m, complex_t* RESTRICT X) const {
        const double t0 = C[m];
        const double t1 = S[m];
        const double t2 = t0 * p[1] - t1 * p[3];
        const double t3 = t1 * p[1] + t0 * p[3];
        const int k0 = OUTIDX[2*m];
        X[k0].re = p[0] + t2;
        X[k0].im = -(p[2] + t3);
        const int k1 = OUTIDX[2*m + 1];
        X[k1].re = p[0] - t2;
        X[k1].im = p[2] - t3;
    }

    inline void codelet_d2_pack(const double* RESTRICT p, int m, complex_t* RESTRICT X) const {
        const double t0 = C[m];
        const double t1 = S[m];
        const double t2 = t0 * p[2] - t1 * p[6];
        const double t3 = t1 * p[2] + t0 * p[6];
        const double t4 = t0 * p[3] - t1 * p[7];
        const double t5 = t1 * p[3] + t0 * p[7];
        const double t6 = C[2*m];
        const double t7 = S[2*m];
        const double t8 = t6 * (p[1] + t4) - t7 * (p[5] + t5);
        const double t9 = t7 * (p[1] + t4) + t6 * (p[5] + t5);
        const int k0 = OUTIDX[4*m];
        X[k0].re = (p[0] + t2) + t8;
        X[k0].im = -((p[4] + t3) + t9);
        const int k1 = OUTIDX[4*m + 1];
        X[k1].re = (p[0] + t2) - t8;
        X[k1].im = (p[4] + t3) - t9;
        const double t12 = C[2*m + 1];
        const double t13 = S[2*m + 1];
        const double t14 = t12 * (p[1] - t4) - t13 * (t5 - p[5]);
        const double t15 = t13 * (p[1] - t4) + t12 * (t5 - p[5]);
        const int k2 = OUTIDX[4*m + 2];
        X[k2].re = (p[0] - t2) + t14;
        X[k2].im = -((t3 - p[4]) + t15);
        const int k3 = OUTIDX[4*m + 3];
        X[k3].re = (p[0] - t2) - t14;
        X[k3].im = (t3 - p[4]) - t15;
    }

    // ----- depth-3 leaf codelets, all driven by the packed LeafTw stream -----

    // Scalar reference leaf codelet: 16-double block of node m -> 8 spectrum bins.
    inline void codelet_d3_tw_scalar(const double* RESTRICT p, const LeafTw& t, complex_t* RESTRICT X) const {
        double u[4], w[4], v[4], x[4];
        for (int j = 0; j < 4; ++j) {
            const double R = t.c1 * p[4 + j] - t.s1 * p[12 + j];
            const double I = t.s1 * p[4 + j] + t.c1 * p[12 + j];
            u[j] = p[j] + R;
            w[j] = p[8 + j] + I;
            v[j] = p[j] - R;
            x[j] = I - p[8 + j];
        }

        double g[4][4]; // four leaf blocks [A0, B0, A1, B1]
        for (int j = 0; j < 2; ++j) {
            const double R0 = t.c2[0] * u[2 + j] - t.s2[0] * w[2 + j];
            const double I0 = t.s2[0] * u[2 + j] + t.c2[0] * w[2 + j];
            g[0][2*0 + ((j == 0) ? 0 : 1)] = 0; // placeholder, overwritten below
            (void)R0; (void)I0;
        }
        // child 0 = [u | w], child 1 = [v | x]; each splits into two leaf blocks.
        {
            const double R0a = t.c2[0] * u[2] - t.s2[0] * w[2];
            const double I0a = t.s2[0] * u[2] + t.c2[0] * w[2];
            const double R0b = t.c2[0] * u[3] - t.s2[0] * w[3];
            const double I0b = t.s2[0] * u[3] + t.c2[0] * w[3];
            g[0][0] = u[0] + R0a; g[0][1] = u[1] + R0b; g[0][2] = w[0] + I0a; g[0][3] = w[1] + I0b;
            g[1][0] = u[0] - R0a; g[1][1] = u[1] - R0b; g[1][2] = I0a - w[0]; g[1][3] = I0b - w[1];

            const double R1a = t.c2[1] * v[2] - t.s2[1] * x[2];
            const double I1a = t.s2[1] * v[2] + t.c2[1] * x[2];
            const double R1b = t.c2[1] * v[3] - t.s2[1] * x[3];
            const double I1b = t.s2[1] * v[3] + t.c2[1] * x[3];
            g[2][0] = v[0] + R1a; g[2][1] = v[1] + R1b; g[2][2] = x[0] + I1a; g[2][3] = x[1] + I1b;
            g[3][0] = v[0] - R1a; g[3][1] = v[1] - R1b; g[3][2] = I1a - x[0]; g[3][3] = I1b - x[1];
        }

        for (int gi = 0; gi < 4; ++gi) {
            const double c = t.c4[gi];
            const double s = t.s4[gi];
            const double R = c * g[gi][1] - s * g[gi][3];
            const double I = s * g[gi][1] + c * g[gi][3];
            const int ke = t.idx[2*gi];
            const int ko = t.idx[2*gi + 1];
            X[ke].re = g[gi][0] + R;
            X[ke].im = -(g[gi][2] + I);
            X[ko].re = g[gi][0] - R;
            X[ko].im = g[gi][2] - I;
        }
    }

#if BRUUN_LEVEL == 1
    // 128-bit leaf codelet. Levels 1 and 2 are naturally 2-lane contiguous;
    // level 3 pairs (re, im) so each spectrum bin is one 128-bit store.
    inline void codelet_d3_tw_v2(const double* RESTRICT p, const LeafTw& t, complex_t* RESTRICT X) const {
        const bruun_v2 c1 = V2_SET1(t.c1);
        const bruun_v2 s1 = V2_SET1(t.s1);

        bruun_v2 u[2], w[2], v[2], x[2];
        for (int j = 0; j < 2; ++j) {
            const bruun_v2 b0 = V2_LD(p + 4 + 2*j);
            const bruun_v2 b1 = V2_LD(p + 12 + 2*j);
            const bruun_v2 R = V2_SUB(V2_MUL(c1, b0), V2_MUL(s1, b1));
            const bruun_v2 I = V2_ADD(V2_MUL(s1, b0), V2_MUL(c1, b1));
            const bruun_v2 a0 = V2_LD(p + 2*j);
            const bruun_v2 a1 = V2_LD(p + 8 + 2*j);
            u[j] = V2_ADD(a0, R);
            w[j] = V2_ADD(a1, I);
            v[j] = V2_SUB(a0, R);
            x[j] = V2_SUB(I, a1);
        }

        const bruun_v2 c20 = V2_SET1(t.c2[0]);
        const bruun_v2 s20 = V2_SET1(t.s2[0]);
        const bruun_v2 R0 = V2_SUB(V2_MUL(c20, u[1]), V2_MUL(s20, w[1]));
        const bruun_v2 I0 = V2_ADD(V2_MUL(s20, u[1]), V2_MUL(c20, w[1]));
        const bruun_v2 g0a = V2_ADD(u[0], R0);
        const bruun_v2 g0b = V2_ADD(w[0], I0);
        const bruun_v2 g1a = V2_SUB(u[0], R0);
        const bruun_v2 g1b = V2_SUB(I0, w[0]);

        const bruun_v2 c21 = V2_SET1(t.c2[1]);
        const bruun_v2 s21 = V2_SET1(t.s2[1]);
        const bruun_v2 R1 = V2_SUB(V2_MUL(c21, v[1]), V2_MUL(s21, x[1]));
        const bruun_v2 I1 = V2_ADD(V2_MUL(s21, v[1]), V2_MUL(c21, x[1]));
        const bruun_v2 g2a = V2_ADD(v[0], R1);
        const bruun_v2 g2b = V2_ADD(x[0], I1);
        const bruun_v2 g3a = V2_SUB(v[0], R1);
        const bruun_v2 g3b = V2_SUB(I1, x[0]);

        const bruun_v2 ga[4] = { g0a, g1a, g2a, g3a };
        const bruun_v2 gb[4] = { g0b, g1b, g2b, g3b };

        for (int gi = 0; gi < 4; ++gi) {
            const bruun_v2 a02 = V2_UNPLO(ga[gi], gb[gi]); // [x0, x2]
            const bruun_v2 b13 = V2_UNPHI(ga[gi], gb[gi]); // [x1, x3]
            const bruun_v2 csv = V2_SETLH(t.c4[gi], t.s4[gi]);   // [ c, s]
            const bruun_v2 cs2 = V2_SETLH(-t.s4[gi], t.c4[gi]);  // [-s, c]
            const bruun_v2 tv = V2_ADD(V2_MUL(csv, V2_DUP0(b13)), V2_MUL(cs2, V2_DUP1(b13))); // [R, I]
            const bruun_v2 ev = V2_NEGHI(V2_ADD(a02, tv)); // [x0+R, -(x2+I)]
            const bruun_v2 od = V2_SUB(a02, tv);           // [x0-R,   x2-I ]
            V2_ST(&X[t.idx[2*gi]].re, ev);
            V2_ST(&X[t.idx[2*gi + 1]].re, od);
        }
    }
#endif

#if BRUUN_LEVEL >= 2
    // 256-bit depth-3 leaf core operating on register inputs (the four quarters
    // of a 16-double leaf block). Lets callers that already hold the block in
    // registers (the fused depth-4 codelet) skip a 16-double store/reload.
    inline void d3_avx2_core(__m256d A0, __m256d B0, __m256d A1, __m256d B1,
                             const LeafTw& t, complex_t* RESTRICT X) const {
        const __m256d c1 = _mm256_set1_pd(t.c1);
        const __m256d s1 = _mm256_set1_pd(t.s1);
        const __m256d R1 = _mm256_fmsub_pd(c1, B0, _mm256_mul_pd(s1, B1));
        const __m256d I1 = _mm256_fmadd_pd(s1, B0, _mm256_mul_pd(c1, B1));

        const __m256d c0a = _mm256_add_pd(A0, R1);
        const __m256d c0b = _mm256_add_pd(A1, I1);
        const __m256d c1a = _mm256_sub_pd(A0, R1);
        const __m256d c1b = _mm256_sub_pd(I1, A1);

        const __m256d A0v = _mm256_permute2f128_pd(c0a, c1a, 0x20);
        const __m256d B0v = _mm256_permute2f128_pd(c0a, c1a, 0x31);
        const __m256d A1v = _mm256_permute2f128_pd(c0b, c1b, 0x20);
        const __m256d B1v = _mm256_permute2f128_pd(c0b, c1b, 0x31);

        const __m256d c2 = _mm256_permute4x64_pd(_mm256_castpd128_pd256(_mm_loadu_pd(t.c2)), 0x50);
        const __m256d s2 = _mm256_permute4x64_pd(_mm256_castpd128_pd256(_mm_loadu_pd(t.s2)), 0x50);
        const __m256d R2 = _mm256_fmsub_pd(c2, B0v, _mm256_mul_pd(s2, B1v));
        const __m256d I2 = _mm256_fmadd_pd(s2, B0v, _mm256_mul_pd(c2, B1v));

        const __m256d P = _mm256_add_pd(A0v, R2);
        const __m256d Q = _mm256_add_pd(A1v, I2);
        const __m256d M = _mm256_sub_pd(A0v, R2);
        const __m256d W = _mm256_sub_pd(I2, A1v);

        const __m256d A0w = _mm256_unpacklo_pd(P, M);
        const __m256d B0w = _mm256_unpackhi_pd(P, M);
        const __m256d A1w = _mm256_unpacklo_pd(Q, W);
        const __m256d B1w = _mm256_unpackhi_pd(Q, W);

        const __m256d c4 = _mm256_loadu_pd(t.c4);
        const __m256d s4 = _mm256_loadu_pd(t.s4);
        const __m256d R3 = _mm256_fmsub_pd(c4, B0w, _mm256_mul_pd(s4, B1w));
        const __m256d I3 = _mm256_fmadd_pd(s4, B0w, _mm256_mul_pd(c4, B1w));

        const __m256d sgn = _mm256_set1_pd(-0.0);
        const __m256d re_e = _mm256_add_pd(A0w, R3);
        const __m256d re_o = _mm256_sub_pd(A0w, R3);
        const __m256d im_e = _mm256_xor_pd(_mm256_add_pd(A1w, I3), sgn);
        const __m256d im_o = _mm256_sub_pd(A1w, I3);

        const __m256d pe = _mm256_unpacklo_pd(re_e, im_e); // leaves 8m,   8m+4
        const __m256d ph = _mm256_unpackhi_pd(re_e, im_e); // leaves 8m+2, 8m+6
        const __m256d qe = _mm256_unpacklo_pd(re_o, im_o); // leaves 8m+1, 8m+5
        const __m256d qh = _mm256_unpackhi_pd(re_o, im_o); // leaves 8m+3, 8m+7

        const int32_t* RESTRICT idx = t.idx;
        _mm_storeu_pd(&X[idx[0]].re, _mm256_castpd256_pd128(pe));
        _mm_storeu_pd(&X[idx[4]].re, _mm256_extractf128_pd(pe, 1));
        _mm_storeu_pd(&X[idx[2]].re, _mm256_castpd256_pd128(ph));
        _mm_storeu_pd(&X[idx[6]].re, _mm256_extractf128_pd(ph, 1));
        _mm_storeu_pd(&X[idx[1]].re, _mm256_castpd256_pd128(qe));
        _mm_storeu_pd(&X[idx[5]].re, _mm256_extractf128_pd(qe, 1));
        _mm_storeu_pd(&X[idx[3]].re, _mm256_castpd256_pd128(qh));
        _mm_storeu_pd(&X[idx[7]].re, _mm256_extractf128_pd(qh, 1));
    }

    inline void codelet_d3_tw_avx2(const double* RESTRICT p, const LeafTw& t, complex_t* RESTRICT X) const {
        d3_avx2_core(_mm256_loadu_pd(p), _mm256_loadu_pd(p + 4),
                     _mm256_loadu_pd(p + 8), _mm256_loadu_pd(p + 12), t, X);
    }

    // Fused depth-4 leaf group: the q=8 norm level plus both depth-3 leaf
    // codelets on one 32-double block, with the children kept in registers
    // (saves a 32-double store/reload per leaf group).
    inline void codelet_d4_avx2(const double* RESTRICT p, int m, complex_t* RESTRICT X) const {
        const __m256d a0l = _mm256_loadu_pd(p);
        const __m256d a0h = _mm256_loadu_pd(p + 4);
        const __m256d b0l = _mm256_loadu_pd(p + 8);
        const __m256d b0h = _mm256_loadu_pd(p + 12);
        const __m256d a1l = _mm256_loadu_pd(p + 16);
        const __m256d a1h = _mm256_loadu_pd(p + 20);
        const __m256d b1l = _mm256_loadu_pd(p + 24);
        const __m256d b1h = _mm256_loadu_pd(p + 28);

        const __m256d vc = _mm256_set1_pd(C[m]);
        const __m256d vs = _mm256_set1_pd(S[m]);

        const __m256d Rl = _mm256_fmsub_pd(vc, b0l, _mm256_mul_pd(vs, b1l));
        const __m256d Il = _mm256_fmadd_pd(vs, b0l, _mm256_mul_pd(vc, b1l));
        const __m256d Rh = _mm256_fmsub_pd(vc, b0h, _mm256_mul_pd(vs, b1h));
        const __m256d Ih = _mm256_fmadd_pd(vs, b0h, _mm256_mul_pd(vc, b1h));

        d3_avx2_core(_mm256_add_pd(a0l, Rl), _mm256_add_pd(a0h, Rh),
                     _mm256_add_pd(a1l, Il), _mm256_add_pd(a1h, Ih),
                     TW[2*m], X);
        d3_avx2_core(_mm256_sub_pd(a0l, Rl), _mm256_sub_pd(a0h, Rh),
                     _mm256_sub_pd(Il, a1l), _mm256_sub_pd(Ih, a1h),
                     TW[2*m + 1], X);
    }
#endif

#if BRUUN_LEVEL >= 3
    // 512-bit paired depth-3 core: lanes are [block(m2) | block(m2+1)] quarters.
    inline void d3x2_avx512_core(__m512d A0, __m512d B0, __m512d A1, __m512d B1,
                                 const LeafTw& t0, const LeafTw& t1,
                                 complex_t* RESTRICT X) const {
        const __m512d c1 = _mm512_insertf64x4(_mm512_castpd256_pd512(_mm256_set1_pd(t0.c1)), _mm256_set1_pd(t1.c1), 1);
        const __m512d s1 = _mm512_insertf64x4(_mm512_castpd256_pd512(_mm256_set1_pd(t0.s1)), _mm256_set1_pd(t1.s1), 1);

        const __m512d R1 = _mm512_fmsub_pd(c1, B0, _mm512_mul_pd(s1, B1));
        const __m512d I1 = _mm512_fmadd_pd(s1, B0, _mm512_mul_pd(c1, B1));

        const __m512d c0a = _mm512_add_pd(A0, R1);
        const __m512d c0b = _mm512_add_pd(A1, I1);
        const __m512d c1a = _mm512_sub_pd(A0, R1);
        const __m512d c1b = _mm512_sub_pd(I1, A1);

        const __m512i ixA = _mm512_set_epi64(13, 12, 5, 4, 9, 8, 1, 0);
        const __m512i ixB = _mm512_set_epi64(15, 14, 7, 6, 11, 10, 3, 2);
        const __m512d A0v = _mm512_permutex2var_pd(c0a, ixA, c1a);
        const __m512d B0v = _mm512_permutex2var_pd(c0a, ixB, c1a);
        const __m512d A1v = _mm512_permutex2var_pd(c0b, ixA, c1b);
        const __m512d B1v = _mm512_permutex2var_pd(c0b, ixB, c1b);

        const __m512i dup2 = _mm512_set_epi64(3, 3, 2, 2, 1, 1, 0, 0);
        const __m256d c2p = _mm256_insertf128_pd(_mm256_castpd128_pd256(_mm_loadu_pd(t0.c2)), _mm_loadu_pd(t1.c2), 1);
        const __m256d s2p = _mm256_insertf128_pd(_mm256_castpd128_pd256(_mm_loadu_pd(t0.s2)), _mm_loadu_pd(t1.s2), 1);
        const __m512d c2 = _mm512_permutexvar_pd(dup2, _mm512_castpd256_pd512(c2p));
        const __m512d s2 = _mm512_permutexvar_pd(dup2, _mm512_castpd256_pd512(s2p));

        const __m512d R2 = _mm512_fmsub_pd(c2, B0v, _mm512_mul_pd(s2, B1v));
        const __m512d I2 = _mm512_fmadd_pd(s2, B0v, _mm512_mul_pd(c2, B1v));

        const __m512d P = _mm512_add_pd(A0v, R2);
        const __m512d Q = _mm512_add_pd(A1v, I2);
        const __m512d M = _mm512_sub_pd(A0v, R2);
        const __m512d W = _mm512_sub_pd(I2, A1v);

        const __m512d A0w = _mm512_unpacklo_pd(P, M);
        const __m512d B0w = _mm512_unpackhi_pd(P, M);
        const __m512d A1w = _mm512_unpacklo_pd(Q, W);
        const __m512d B1w = _mm512_unpackhi_pd(Q, W);

        const __m512d c4 = _mm512_insertf64x4(_mm512_castpd256_pd512(_mm256_loadu_pd(t0.c4)), _mm256_loadu_pd(t1.c4), 1);
        const __m512d s4 = _mm512_insertf64x4(_mm512_castpd256_pd512(_mm256_loadu_pd(t0.s4)), _mm256_loadu_pd(t1.s4), 1);
        const __m512d R3 = _mm512_fmsub_pd(c4, B0w, _mm512_mul_pd(s4, B1w));
        const __m512d I3 = _mm512_fmadd_pd(s4, B0w, _mm512_mul_pd(c4, B1w));

        const __m512d sgn = _mm512_set1_pd(-0.0);
        const __m512d re_e = _mm512_add_pd(A0w, R3);
        const __m512d re_o = _mm512_sub_pd(A0w, R3);
        const __m512d im_e = _mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(_mm512_add_pd(A1w, I3)), _mm512_castpd_si512(sgn)));
        const __m512d im_o = _mm512_sub_pd(A1w, I3);

        const __m512d pe = _mm512_unpacklo_pd(re_e, im_e);
        const __m512d ph = _mm512_unpackhi_pd(re_e, im_e);
        const __m512d qe = _mm512_unpacklo_pd(re_o, im_o);
        const __m512d qh = _mm512_unpackhi_pd(re_o, im_o);

        _mm_storeu_pd(&X[t0.idx[0]].re, _mm512_castpd512_pd128(pe));
        _mm_storeu_pd(&X[t0.idx[4]].re, _mm512_extractf64x2_pd(pe, 1));
        _mm_storeu_pd(&X[t1.idx[0]].re, _mm512_extractf64x2_pd(pe, 2));
        _mm_storeu_pd(&X[t1.idx[4]].re, _mm512_extractf64x2_pd(pe, 3));
        _mm_storeu_pd(&X[t0.idx[2]].re, _mm512_castpd512_pd128(ph));
        _mm_storeu_pd(&X[t0.idx[6]].re, _mm512_extractf64x2_pd(ph, 1));
        _mm_storeu_pd(&X[t1.idx[2]].re, _mm512_extractf64x2_pd(ph, 2));
        _mm_storeu_pd(&X[t1.idx[6]].re, _mm512_extractf64x2_pd(ph, 3));
        _mm_storeu_pd(&X[t0.idx[1]].re, _mm512_castpd512_pd128(qe));
        _mm_storeu_pd(&X[t0.idx[5]].re, _mm512_extractf64x2_pd(qe, 1));
        _mm_storeu_pd(&X[t1.idx[1]].re, _mm512_extractf64x2_pd(qe, 2));
        _mm_storeu_pd(&X[t1.idx[5]].re, _mm512_extractf64x2_pd(qe, 3));
        _mm_storeu_pd(&X[t0.idx[3]].re, _mm512_castpd512_pd128(qh));
        _mm_storeu_pd(&X[t0.idx[7]].re, _mm512_extractf64x2_pd(qh, 1));
        _mm_storeu_pd(&X[t1.idx[3]].re, _mm512_extractf64x2_pd(qh, 2));
        _mm_storeu_pd(&X[t1.idx[7]].re, _mm512_extractf64x2_pd(qh, 3));
    }

    inline void codelet_d3x2_avx512(const double* RESTRICT p, int m2, complex_t* RESTRICT X) const {
        const __m512d A0 = _mm512_insertf64x4(_mm512_castpd256_pd512(_mm256_loadu_pd(p)),      _mm256_loadu_pd(p + 16), 1);
        const __m512d B0 = _mm512_insertf64x4(_mm512_castpd256_pd512(_mm256_loadu_pd(p + 4)),  _mm256_loadu_pd(p + 20), 1);
        const __m512d A1 = _mm512_insertf64x4(_mm512_castpd256_pd512(_mm256_loadu_pd(p + 8)),  _mm256_loadu_pd(p + 24), 1);
        const __m512d B1 = _mm512_insertf64x4(_mm512_castpd256_pd512(_mm256_loadu_pd(p + 12)), _mm256_loadu_pd(p + 28), 1);
        d3x2_avx512_core(A0, B0, A1, B1, TW[m2], TW[m2 + 1], X);
    }

    // Fused depth-4 leaf group at 512 bits: q=8 norm level plus both depth-3
    // codelets with children rearranged in registers instead of memory.
    inline void codelet_d4x2_avx512(const double* RESTRICT p, int m, complex_t* RESTRICT X) const {
        const __m512d za0 = _mm512_loadu_pd(p);
        const __m512d zb0 = _mm512_loadu_pd(p + 8);
        const __m512d za1 = _mm512_loadu_pd(p + 16);
        const __m512d zb1 = _mm512_loadu_pd(p + 24);

        const __m512d vc = _mm512_set1_pd(C[m]);
        const __m512d vs = _mm512_set1_pd(S[m]);

        const __m512d R = _mm512_fmsub_pd(vc, zb0, _mm512_mul_pd(vs, zb1));
        const __m512d I = _mm512_fmadd_pd(vs, zb0, _mm512_mul_pd(vc, zb1));

        const __m512d u  = _mm512_add_pd(za0, R); // child0 first half
        const __m512d w  = _mm512_add_pd(za1, I); // child0 second half
        const __m512d vv = _mm512_sub_pd(za0, R); // child1 first half
        const __m512d x  = _mm512_sub_pd(I, za1); // child1 second half

        const __m512d A0 = _mm512_insertf64x4(_mm512_castpd256_pd512(_mm512_castpd512_pd256(u)),  _mm512_castpd512_pd256(vv), 1);
        const __m512d B0 = _mm512_insertf64x4(_mm512_castpd256_pd512(_mm512_extractf64x4_pd(u, 1)),  _mm512_extractf64x4_pd(vv, 1), 1);
        const __m512d A1 = _mm512_insertf64x4(_mm512_castpd256_pd512(_mm512_castpd512_pd256(w)),  _mm512_castpd512_pd256(x), 1);
        const __m512d B1 = _mm512_insertf64x4(_mm512_castpd256_pd512(_mm512_extractf64x4_pd(w, 1)),  _mm512_extractf64x4_pd(x, 1), 1);

        d3x2_avx512_core(A0, B0, A1, B1, TW[2*m], TW[2*m + 1], X);
    }
#endif

    inline void d3_one(const double* RESTRICT p, int m, complex_t* RESTRICT X) const {
        const LeafTw& t = TW[m];
#if BRUUN_LEVEL >= 2
        codelet_d3_tw_avx2(p, t, X);
#elif BRUUN_LEVEL == 1
        codelet_d3_tw_v2(p, t, X);
#else
        codelet_d3_tw_scalar(p, t, X);
#endif
    }

    // Residue-writing leaf codelet: identical math to the packing codelets,
    // but the 16 results are written back into p in leaf-residue order. This is
    // the engine for the residue-domain interfaces (filtering, custom CRT layouts)
    // and for the optional two-phase pack.
    inline void codelet_d3_tw_res(double* RESTRICT p, const LeafTw& t) const {
#if BRUUN_LEVEL >= 2
        const __m256d A0 = _mm256_loadu_pd(p);
        const __m256d B0 = _mm256_loadu_pd(p + 4);
        const __m256d A1 = _mm256_loadu_pd(p + 8);
        const __m256d B1 = _mm256_loadu_pd(p + 12);

        const __m256d c1 = _mm256_set1_pd(t.c1);
        const __m256d s1 = _mm256_set1_pd(t.s1);
        const __m256d R1 = _mm256_fmsub_pd(c1, B0, _mm256_mul_pd(s1, B1));
        const __m256d I1 = _mm256_fmadd_pd(s1, B0, _mm256_mul_pd(c1, B1));

        const __m256d c0a = _mm256_add_pd(A0, R1);
        const __m256d c0b = _mm256_add_pd(A1, I1);
        const __m256d c1a = _mm256_sub_pd(A0, R1);
        const __m256d c1b = _mm256_sub_pd(I1, A1);

        const __m256d A0v = _mm256_permute2f128_pd(c0a, c1a, 0x20);
        const __m256d B0v = _mm256_permute2f128_pd(c0a, c1a, 0x31);
        const __m256d A1v = _mm256_permute2f128_pd(c0b, c1b, 0x20);
        const __m256d B1v = _mm256_permute2f128_pd(c0b, c1b, 0x31);

        const __m256d c2 = _mm256_permute4x64_pd(_mm256_castpd128_pd256(_mm_loadu_pd(t.c2)), 0x50);
        const __m256d s2 = _mm256_permute4x64_pd(_mm256_castpd128_pd256(_mm_loadu_pd(t.s2)), 0x50);
        const __m256d R2 = _mm256_fmsub_pd(c2, B0v, _mm256_mul_pd(s2, B1v));
        const __m256d I2 = _mm256_fmadd_pd(s2, B0v, _mm256_mul_pd(c2, B1v));

        const __m256d P = _mm256_add_pd(A0v, R2);
        const __m256d Q = _mm256_add_pd(A1v, I2);
        const __m256d M = _mm256_sub_pd(A0v, R2);
        const __m256d W = _mm256_sub_pd(I2, A1v);

        const __m256d A0w = _mm256_unpacklo_pd(P, M);
        const __m256d B0w = _mm256_unpackhi_pd(P, M);
        const __m256d A1w = _mm256_unpacklo_pd(Q, W);
        const __m256d B1w = _mm256_unpackhi_pd(Q, W);

        const __m256d c4 = _mm256_loadu_pd(t.c4);
        const __m256d s4 = _mm256_loadu_pd(t.s4);
        const __m256d R3 = _mm256_fmsub_pd(c4, B0w, _mm256_mul_pd(s4, B1w));
        const __m256d I3 = _mm256_fmadd_pd(s4, B0w, _mm256_mul_pd(c4, B1w));

        const __m256d E0 = _mm256_add_pd(A0w, R3); // leaf even residue r0
        const __m256d E1 = _mm256_add_pd(A1w, I3); // leaf even residue r1
        const __m256d O0 = _mm256_sub_pd(A0w, R3); // leaf odd residue r0
        const __m256d O1 = _mm256_sub_pd(I3, A1w); // leaf odd residue r1

        // 4x4 transpose to leaf-residue order: p[4g..4g+3] = [E0[g], E1[g], O0[g], O1[g]].
        const __m256d t0 = _mm256_unpacklo_pd(E0, E1);
        const __m256d t1 = _mm256_unpackhi_pd(E0, E1);
        const __m256d t2 = _mm256_unpacklo_pd(O0, O1);
        const __m256d t3 = _mm256_unpackhi_pd(O0, O1);

        _mm256_storeu_pd(p,      _mm256_permute2f128_pd(t0, t2, 0x20));
        _mm256_storeu_pd(p + 4,  _mm256_permute2f128_pd(t1, t3, 0x20));
        _mm256_storeu_pd(p + 8,  _mm256_permute2f128_pd(t0, t2, 0x31));
        _mm256_storeu_pd(p + 12, _mm256_permute2f128_pd(t1, t3, 0x31));
#else
        norm_q_fwd(p, 4, t.c1, t.s1);
        norm_q2_fwd(p, t.c2[0], t.s2[0]);
        norm_q2_fwd(p + 8, t.c2[1], t.s2[1]);
        norm_q1_fwd(p, t.c4[0], t.s4[0]);
        norm_q1_fwd(p + 4, t.c4[1], t.s4[1]);
        norm_q1_fwd(p + 8, t.c4[2], t.s4[2]);
        norm_q1_fwd(p + 12, t.c4[3], t.s4[3]);
#endif
    }

    void rec_fwd_res(double* RESTRICT v, int q, int m) const {
        if (q >= 16) {
            norm2_fused(v, q, C[m], S[m], C[2*m], S[2*m], C[2*m+1], S[2*m+1]);
            const int qq = q >> 2;
            rec_fwd_res(v,       qq, 4*m);
            rec_fwd_res(v + q,   qq, 4*m + 1);
            rec_fwd_res(v + 2*q, qq, 4*m + 2);
            rec_fwd_res(v + 3*q, qq, 4*m + 3);
            return;
        }
        if (q == 8) {
            norm_q_fwd(v, 8, C[m], S[m]);
            codelet_d3_tw_res(v, TW[2*m]);
            codelet_d3_tw_res(v + 16, TW[2*m + 1]);
            return;
        }
        codelet_d3_tw_res(v, TW[m]);
    }

    // Spine tail for residue-order transforms: the last spine blocks (16, 8, 4, 2)
    // resolved fully in place. Shared by the fast residue forward and the
    // optional two-phase pack.
    void residue_spine_tail_fwd(double* RESTRICT v) const {
        codelet_d3_tw_res(v + 16, TW[1]);
        binomial_fwd(v, 8);
        norm_q_fwd(v + 8, 2, C[1], S[1]);
        norm_q1_fwd(v + 8, C[2], S[2]);
        norm_q1_fwd(v + 12, C[3], S[3]);
        binomial_fwd(v, 4);
        norm_q1_fwd(v + 4, C[1], S[1]);
        binomial_fwd(v, 2);
    }

    // Fast depth-first residue forward with fused input copy. Requires N >= 64.
    void forward_residues_recursive(const double* RESTRICT input, double* RESTRICT v) const {
        binomial_oop(input, v, N / 2);

        for (int h = N / 2; h >= 32; h >>= 1) {
            rec_fwd_res(v + h, h >> 2, 1);
            binomial_fwd(v, h >> 1);
        }

        residue_spine_tail_fwd(v);
    }

    // Exact inverse of codelet_d3_tw_res: three inverse levels on one 16-double
    // leaf block, fused in registers on AVX2, generic norm_q_inv ladder otherwise.
    inline void codelet_d3_tw_res_inv(double* RESTRICT p, const LeafTw& t) const {
#if BRUUN_LEVEL >= 2
        const __m256d out0 = _mm256_loadu_pd(p);
        const __m256d out1 = _mm256_loadu_pd(p + 4);
        const __m256d out2 = _mm256_loadu_pd(p + 8);
        const __m256d out3 = _mm256_loadu_pd(p + 12);

        // Inverse of the forward 4x4 transpose.
        const __m256d t0 = _mm256_unpacklo_pd(out0, out1);
        const __m256d t1 = _mm256_unpackhi_pd(out0, out1);
        const __m256d t2 = _mm256_unpacklo_pd(out2, out3);
        const __m256d t3 = _mm256_unpackhi_pd(out2, out3);
        const __m256d E0 = _mm256_permute2f128_pd(t0, t2, 0x20);
        const __m256d O0 = _mm256_permute2f128_pd(t0, t2, 0x31);
        const __m256d E1 = _mm256_permute2f128_pd(t1, t3, 0x20);
        const __m256d O1 = _mm256_permute2f128_pd(t1, t3, 0x31);

        const __m256d hf = _mm256_set1_pd(0.5);

        // Level 3 inverse.
        const __m256d A0w = _mm256_mul_pd(hf, _mm256_add_pd(E0, O0));
        const __m256d R3  = _mm256_mul_pd(hf, _mm256_sub_pd(E0, O0));
        const __m256d I3  = _mm256_mul_pd(hf, _mm256_add_pd(E1, O1));
        const __m256d A1w = _mm256_mul_pd(hf, _mm256_sub_pd(E1, O1));
        const __m256d c4 = _mm256_loadu_pd(t.c4);
        const __m256d s4 = _mm256_loadu_pd(t.s4);
        const __m256d B0w = _mm256_fmadd_pd(c4, R3, _mm256_mul_pd(s4, I3));
        const __m256d B1w = _mm256_fmsub_pd(c4, I3, _mm256_mul_pd(s4, R3));

        // Level 2 inverse.
        const __m256d P = _mm256_unpacklo_pd(A0w, B0w);
        const __m256d M = _mm256_unpackhi_pd(A0w, B0w);
        const __m256d Q = _mm256_unpacklo_pd(A1w, B1w);
        const __m256d W = _mm256_unpackhi_pd(A1w, B1w);

        const __m256d A0v = _mm256_mul_pd(hf, _mm256_add_pd(P, M));
        const __m256d R2  = _mm256_mul_pd(hf, _mm256_sub_pd(P, M));
        const __m256d I2  = _mm256_mul_pd(hf, _mm256_add_pd(Q, W));
        const __m256d A1v = _mm256_mul_pd(hf, _mm256_sub_pd(Q, W));
        const __m256d c2 = _mm256_permute4x64_pd(_mm256_castpd128_pd256(_mm_loadu_pd(t.c2)), 0x50);
        const __m256d s2 = _mm256_permute4x64_pd(_mm256_castpd128_pd256(_mm_loadu_pd(t.s2)), 0x50);
        const __m256d B0v = _mm256_fmadd_pd(c2, R2, _mm256_mul_pd(s2, I2));
        const __m256d B1v = _mm256_fmsub_pd(c2, I2, _mm256_mul_pd(s2, R2));

        // Level 1 inverse.
        const __m256d c0a = _mm256_permute2f128_pd(A0v, B0v, 0x20);
        const __m256d c1a = _mm256_permute2f128_pd(A0v, B0v, 0x31);
        const __m256d c0b = _mm256_permute2f128_pd(A1v, B1v, 0x20);
        const __m256d c1b = _mm256_permute2f128_pd(A1v, B1v, 0x31);

        const __m256d A0 = _mm256_mul_pd(hf, _mm256_add_pd(c0a, c1a));
        const __m256d R1 = _mm256_mul_pd(hf, _mm256_sub_pd(c0a, c1a));
        const __m256d I1 = _mm256_mul_pd(hf, _mm256_add_pd(c0b, c1b));
        const __m256d A1 = _mm256_mul_pd(hf, _mm256_sub_pd(c0b, c1b));
        const __m256d c1v = _mm256_set1_pd(t.c1);
        const __m256d s1v = _mm256_set1_pd(t.s1);
        const __m256d B0 = _mm256_fmadd_pd(c1v, R1, _mm256_mul_pd(s1v, I1));
        const __m256d B1 = _mm256_fmsub_pd(c1v, I1, _mm256_mul_pd(s1v, R1));

        _mm256_storeu_pd(p,      A0);
        _mm256_storeu_pd(p + 4,  B0);
        _mm256_storeu_pd(p + 8,  A1);
        _mm256_storeu_pd(p + 12, B1);
#else
        norm_q_inv(p,      1, t.c4[0], t.s4[0]);
        norm_q_inv(p + 4,  1, t.c4[1], t.s4[1]);
        norm_q_inv(p + 8,  1, t.c4[2], t.s4[2]);
        norm_q_inv(p + 12, 1, t.c4[3], t.s4[3]);
        norm_q_inv(p,      2, t.c2[0], t.s2[0]);
        norm_q_inv(p + 8,  2, t.c2[1], t.s2[1]);
        norm_q_inv(p,      4, t.c1,    t.s1);
#endif
    }

    void rec_inv_res(double* RESTRICT v, int q, int m) const {
        if (q >= 16) {
            const int qq = q >> 2;
            rec_inv_res(v,       qq, 4*m);
            rec_inv_res(v + q,   qq, 4*m + 1);
            rec_inv_res(v + 2*q, qq, 4*m + 2);
            rec_inv_res(v + 3*q, qq, 4*m + 3);
            norm2_inv_fused(v, q, C[m], S[m], C[2*m], S[2*m], C[2*m+1], S[2*m+1]);
            return;
        }
        if (q == 8) {
            codelet_d3_tw_res_inv(v, TW[2*m]);
            codelet_d3_tw_res_inv(v + 16, TW[2*m + 1]);
            norm_q_inv(v, 8, C[m], S[m]);
            return;
        }
        codelet_d3_tw_res_inv(v, TW[m]);
    }

    // Exact reverse of residue_spine_tail_fwd.
    void residue_spine_tail_inv(double* RESTRICT v) const {
        binomial_inv(v, 2);
        norm_q_inv(v + 4, 1, C[1], S[1]);
        binomial_inv(v, 4);
        norm_q_inv(v + 8, 1, C[2], S[2]);
        norm_q_inv(v + 12, 1, C[3], S[3]);
        norm_q_inv(v + 8, 2, C[1], S[1]);
        binomial_inv(v, 8);
        codelet_d3_tw_res_inv(v + 16, TW[1]);
    }

    // Fast depth-first residue inverse: exact reverse of forward_residues_recursive.
    // Requires N >= 64.
    void inverse_residues_recursive(double* RESTRICT v) const {
        residue_spine_tail_inv(v);

        for (int h = 32; h <= N / 2; h <<= 1) {
            binomial_inv(v, h >> 1);
            rec_inv_res(v + h, h >> 2, 1);
        }

        binomial_inv(v, N / 2);
    }

    // Two-phase standard-output forward: residues land in block order in v, then
    // one pass writes ordinary FFT bins sequentially while reading residue pairs
    // through KINV. Policy uses this only for large standard-layout outputs.
    void forward_standard_two_phase(const double* RESTRICT input, double* RESTRICT v, complex_t* RESTRICT X) const {
        forward_residues_recursive(input, v);

        X[0].re = v[0] + v[1];
        X[0].im = 0.0;
        X[N / 2].re = v[0] - v[1];
        X[N / 2].im = 0.0;

        const int* RESTRICT kin = KINV.data();
        int k = 1;
#if BRUUN_LEVEL >= 1 && (defined(BRUUN_X86_128) || defined(BRUUN_NEON_128))
        for (; k < N / 2; ++k) {
            const bruun_v2 r = V2_LD(v + 2*kin[k]);
            V2_ST(&X[k].re, V2_NEGHI(r));
        }
#else
        for (; k < N / 2; ++k) {
            const int m = kin[k];
            X[k].re = v[2*m];
            X[k].im = -v[2*m + 1];
        }
#endif
    }


    // Depth-first traversal of the Bruun factor tree. Identical arithmetic to the
    // breadth-first stages, reordered so every sub-block becomes cache-resident
    // before the remaining log(s) passes touch it. Descends two levels per fused
    // pass; bottoms out in the fused depth-3 leaf codelets.
    void rec_fwd(double* RESTRICT v, int q, int m, complex_t* RESTRICT X) const {
        if (q >= 16) {
            norm2_fused(v, q, C[m], S[m], C[2*m], S[2*m], C[2*m+1], S[2*m+1]);
            const int qq = q >> 2;
            rec_fwd(v,       qq, 4*m,     X);
            rec_fwd(v + q,   qq, 4*m + 1, X);
            rec_fwd(v + 2*q, qq, 4*m + 2, X);
            rec_fwd(v + 3*q, qq, 4*m + 3, X);
            return;
        }
        if (q == 8) {
#if BRUUN_LEVEL >= 3
            codelet_d4x2_avx512(v, m, X);
#elif BRUUN_LEVEL >= 2
            codelet_d4_avx2(v, m, X);
#else
            norm_q_fwd(v, 8, C[m], S[m]);
            d3_one(v, 2*m, X);
            d3_one(v + 16, 2*m + 1, X);
#endif
            return;
        }
        d3_one(v, m, X);
    }

    // Fused copy + depth-first forward. Requires N >= 64.
    void forward_recursive(const double* RESTRICT input, double* RESTRICT v, complex_t* RESTRICT X) const {
        binomial_oop(input, v, N / 2);

        for (int h = N / 2; h >= 32; h >>= 1) {
            rec_fwd(v + h, h >> 2, 1, X);
            binomial_fwd(v, h >> 1);
        }

        d3_one(v + 16, 1, X);
        binomial_fwd(v, 8);
        codelet_d2_pack(v + 8, 1, X);
        binomial_fwd(v, 4);
        codelet_d1_pack(v + 4, 1, X);
        binomial_fwd(v, 2);
        pack_leaf_node(1, v[2], v[3], X);

        X[0].re = v[0] + v[1];
        X[0].im = 0.0;
        X[N / 2].re = v[0] - v[1];
        X[N / 2].im = 0.0;
    }

    inline void pack_leaf_node(int leaf, double r0, double r1, complex_t* RESTRICT X) const {
        const int k = OUTIDX[leaf];
        X[k].re = r0;
        X[k].im = -r1;
    }

    void forward_stage(double* RESTRICT v, int jj) const {
        const int s = N >> jj;
        const int h = s >> 1;
        const int q = s >> 2;
        const int m_end = 1 << jj;

        binomial_fwd(v, h);

        if (q == 1) {
            for (int m = 1; m < m_end; ++m) norm_q1_fwd(v + m*s, C[m], S[m]);
        } else if (q == 2) {
            for (int m = 1; m < m_end; ++m) norm_q2_fwd(v + m*s, C[m], S[m]);
        } else {
            for (int m = 1; m < m_end; ++m) norm_q_fwd(v + m*s, q, C[m], S[m]);
        }
    }

    void forward_residues_inplace(double* RESTRICT v) const {
        for (int jj = 0; jj < L - 1; ++jj) {
            forward_stage(v, jj);
        }
    }

    void forward_fused_tail(double* RESTRICT v, complex_t* RESTRICT X) const {
        for (int jj = 0; jj <= L - 5; ++jj) {
            forward_stage(v, jj);
        }

        binomial_fwd(v, 8);
        binomial_fwd(v, 4);
        binomial_fwd(v, 2);

        X[0].re = v[0] + v[1];
        X[0].im = 0.0;
        X[N / 2].re = v[0] - v[1];
        X[N / 2].im = 0.0;

        pack_leaf_node(1, v[2], v[3], X);
        codelet_d1_pack(v + 4, 1, X);
        codelet_d2_pack(v + 8, 1, X);

        for (int m = 1; m < N / 16; ++m) {
            d3_one(v + 16*m, m, X);
        }
    }

    void residues_to_complex(const double* RESTRICT v, complex_t* RESTRICT X) const {
        X[0].re = v[0] + v[1];
        X[0].im = 0.0;
        X[N / 2].re = v[0] - v[1];
        X[N / 2].im = 0.0;

        for (int m = 1; m < N / 2; ++m) {
            const int k = IDX[m];
            X[k].re = v[2*m];
            X[k].im = -v[2*m + 1];
        }
    }

    void complex_to_residues(const complex_t* RESTRICT X, double* RESTRICT v) const {
        v[0] = 0.5 * (X[0].re + X[N / 2].re);
        v[1] = 0.5 * (X[0].re - X[N / 2].re);

        for (int m = 1; m < N / 2; ++m) {
            const int k = IDX[m];
            v[2*m] = X[k].re;
            v[2*m + 1] = -X[k].im;
        }
    }

    void inverse_residues_inplace(double* RESTRICT v) const {
        if (fuse_tail && N >= 64) {
            inverse_residues_recursive(v);
            return;
        }

        for (int jj = L - 2; jj >= 0; --jj) {
            const int s = N >> jj;
            const int h = s >> 1;
            const int q = s >> 2;
            const int m_end = 1 << jj;

            for (int m = m_end - 1; m > 0; --m) {
                norm_q_inv(v + m*s, q, C[m], S[m]);
            }

            binomial_inv(v, h);
        }
    }
};

} // namespace bruun
