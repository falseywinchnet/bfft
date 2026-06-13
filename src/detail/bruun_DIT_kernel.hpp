#pragma once
// MIT joshuah.rainstar@gmail.com 2026
// bruun_DIT_kernel.hpp
//
// Decimation-in-time form of the BFFT four-star real transform.
//
// The shipped bruun_kernel.hpp is the DIF form: natural-order coefficients in,
// modular reduction down the CRT factor tree of z^N - 1, scrambled leaf
// residues out. This kernel is the time-decimated mirror: bit-reversed input
// gather, then CRT merges that lift half-size spectra through the covering map
// z -> z^2, emitting the spectrum in NATURAL bin order with no conversion pass
// and no pos -> m -> k address algebra.
//
// The merge butterfly is the four-star atom of the DIF kernel at q = 1. In the
// normalized leaf basis every spectrum slot is a local complex plane, so the
// merge of downstairs residues E, O at bin k into the upstairs leaf pair
// (theta, pi - theta), theta = pi*k/(2*nb), is
//
//     W              = e^{-i theta} * O          (one plane rotation)
//     X[k]           = E + W
//     X[2*nb - k]    = conj(E - W)
//
// the DC chain merges as binomial butterflies, and the two downstairs Nyquist
// reals assemble the upstairs z^2 + 1 leaf for free. Each stage is plane
// rotations plus sqrt(2)-scaled add/sub pairs, so the transform stays
// sqrt(N/2) times an orthogonal map: condition number 1, exact inverse with
// the per-level measure carried by the reconstruction (no global 1/N).
//
// Structural trade against the DIF kernel, stated honestly:
//   - DIF blocks share ONE rotation (c, s) broadcast per modulus block; here
//     the angle is a leaf property, so merges stream per-bin twiddle vectors,
//     the Cooley-Tukey condition. One master ladder still generates every
//     coefficient; coarser stages are exact even-index subsamples of the
//     finest stage, so the table is built with N/4 libm calls and copies.
//   - The permutation moves from the output (scattered writes / two-phase
//     pack) to the input (scattered reads), which overlap under the
//     prefetcher; this is the read-side realization of the standard-order
//     conversion the DIF kernel pays for at large N.
//
// In-place merges process bins k and nb - k together; the partner twiddle is
// derived in registers from T_k by swap-and-negate, since
//     T_{nb-k} = (s_k, -c_k) = -SWAP(T_k),  T_k = (cos theta_k, -sin theta_k).
// That quartet (E_k, O_k, E_{nb-k}, O_{nb-k}) is the four-star, reconstituted
// on this side of the decimation.
//
// Scope: double precision, forward + exact inverse, standard-bin and natural
// halfcomplex interfaces, pointwise spectral multiply for filtering. The f32
// mirror follows the same pattern as the DIF kernel's and is not duplicated
// here. Layout per merge state, as N doubles = N/2 complex slots:
//     slot 0       (re, im) = (DC, Nyquist) of the sub-spectrum
//     slot k       (re, im) = (Re X[k], Im X[k]),  1 <= k < nb
//
// SIMD backend resolution matches bruun_kernel.hpp:
// BRUUN_DIT_LEVEL 0 = scalar, 1 = 128-bit (SSE2/NEON), 2 = AVX2+FMA, 3 = AVX-512.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#if defined(BRUUN_DIT_FORCE_SCALAR)
# define BRUUN_DIT_LEVEL 0
#elif defined(__AVX512F__) && defined(__FMA__)
# define BRUUN_DIT_LEVEL 3
#elif defined(__AVX2__) && defined(__FMA__)
# define BRUUN_DIT_LEVEL 2
#elif defined(__SSE2__) || defined(_M_X64) || (defined(__aarch64__) && defined(__ARM_NEON))
# define BRUUN_DIT_LEVEL 1
#else
# define BRUUN_DIT_LEVEL 0
#endif

#if BRUUN_DIT_LEVEL >= 2 || (BRUUN_DIT_LEVEL == 1 && (defined(__SSE2__) || defined(_M_X64)))
# include <immintrin.h>
# define BRUUN_DIT_X86_128 1
#endif
#if BRUUN_DIT_LEVEL == 1 && defined(__aarch64__) && defined(__ARM_NEON)
# include <arm_neon.h>
# define BRUUN_DIT_NEON_128 1
#endif
#if BRUUN_DIT_LEVEL == 1 && !defined(BRUUN_DIT_X86_128) && !defined(BRUUN_DIT_NEON_128)
# undef BRUUN_DIT_LEVEL
# define BRUUN_DIT_LEVEL 0
#endif

// 2-lane double primitive: one complex slot per vector.
#if defined(BRUUN_DIT_X86_128) && BRUUN_DIT_LEVEL >= 1
typedef __m128d bdit_v2;
# define BV2_LD(p)        _mm_loadu_pd(p)
# define BV2_ST(p, a)     _mm_storeu_pd((p), (a))
# define BV2_ADD(a, b)    _mm_add_pd((a), (b))
# define BV2_SUB(a, b)    _mm_sub_pd((a), (b))
# define BV2_MUL(a, b)    _mm_mul_pd((a), (b))
# define BV2_SET1(x)      _mm_set1_pd(x)
# define BV2_DUP0(a)      _mm_unpacklo_pd((a), (a))
# define BV2_DUP1(a)      _mm_unpackhi_pd((a), (a))
# define BV2_SWAP(a)      _mm_shuffle_pd((a), (a), 0x1)
# define BV2_NEGHI(a)     _mm_xor_pd((a), _mm_set_pd(-0.0, 0.0))
# define BV2_NEGLO(a)     _mm_xor_pd((a), _mm_set_pd(0.0, -0.0))
# define BV2_NEG(a)       _mm_xor_pd((a), _mm_set1_pd(-0.0))
#elif defined(BRUUN_DIT_NEON_128)
typedef float64x2_t bdit_v2;
# define BV2_LD(p)        vld1q_f64(p)
# define BV2_ST(p, a)     vst1q_f64((p), (a))
# define BV2_ADD(a, b)    vaddq_f64((a), (b))
# define BV2_SUB(a, b)    vsubq_f64((a), (b))
# define BV2_MUL(a, b)    vmulq_f64((a), (b))
# define BV2_SET1(x)      vdupq_n_f64(x)
# define BV2_DUP0(a)      vdupq_laneq_f64((a), 0)
# define BV2_DUP1(a)      vdupq_laneq_f64((a), 1)
# define BV2_SWAP(a)      vextq_f64((a), (a), 1)
static inline float64x2_t bdit_neg_mask(float64x2_t a, uint64_t lo, uint64_t hi) {
    const uint64x2_t m = { lo, hi };
    return vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(a), m));
}
# define BV2_NEGHI(a)     bdit_neg_mask((a), 0ULL, 0x8000000000000000ULL)
# define BV2_NEGLO(a)     bdit_neg_mask((a), 0x8000000000000000ULL, 0ULL)
# define BV2_NEG(a)       bdit_neg_mask((a), 0x8000000000000000ULL, 0x8000000000000000ULL)
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#define BDIT_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define BDIT_RESTRICT __restrict__
#else
#define BDIT_RESTRICT
#endif

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884
#endif

namespace bruun_dit {

struct complex_t {
    double re;
    double im;
};

struct complex_f32_t {
    float re;
    float im;
};

static inline const char* simd_backend_name() {
#if BRUUN_DIT_LEVEL == 3
    return "avx512-512";
#elif BRUUN_DIT_LEVEL == 2
    return "avx2-fma-256";
#elif defined(BRUUN_DIT_X86_128)
    return "sse2-128";
#elif defined(BRUUN_DIT_NEON_128)
    return "neon-128";
#else
    return "scalar";
#endif
}

static inline bool is_power2(int n) { return n > 0 && ((n & (n - 1)) == 0); }

static inline int ilog2_pow2(int n) {
    int l = 0;
    while (n > 1) { n >>= 1; ++l; }
    return l;
}

static inline int bitrev_int(int r, int t) {
    int out = 0;
    for (int i = 0; i < t; ++i) { out = (out << 1) | (r & 1); r >>= 1; }
    return out;
}

// ---------------------------------------------------------------------------
// Complex-slot helpers. T stores the forward twiddle (cos t, -sin t), so the
// forward rotation W = e^{-i t} O and the inverse rotation O = e^{+i t} W are
//     fwd: W = cmul(O, T)            inv: O = cmul(W, conj(T))
// and conj(T) is one sign flip in registers, never a second table.
// ---------------------------------------------------------------------------

static inline void cmul_scalar(double or_, double oi, double tr, double ti,
                               double& wr, double& wi) {
    wr = or_ * tr - oi * ti;
    wi = or_ * ti + oi * tr;
}

static inline void cmul_scalar_f32(float or_, float oi, float tr, float ti,
                                   float& wr, float& wi) {
    wr = or_ * tr - oi * ti;
    wi = or_ * ti + oi * tr;
}

#if BRUUN_DIT_LEVEL >= 1
// (or,oi)*(tr,ti): SSE2/NEON-legal complex multiply, no addsub instruction.
static inline bdit_v2 bdit_cmul(bdit_v2 o, bdit_v2 t) {
    const bdit_v2 re = BV2_MUL(BV2_DUP0(o), t);                 // (or tr, or ti)
#if defined(BRUUN_DIT_NEON_128)
    return vfmaq_f64(re, BV2_DUP1(o), BV2_NEGLO(BV2_SWAP(t)));
#else
    const bdit_v2 im = BV2_MUL(BV2_DUP1(o), BV2_NEGLO(BV2_SWAP(t))); // (-oi ti, oi tr)
    return BV2_ADD(re, im);
#endif
}
#endif

#if BRUUN_DIT_LEVEL >= 2
static inline __m256d bdit_cmul4(__m256d o, __m256d t) {
    const __m256d tr = _mm256_movedup_pd(t);                    // (tr,tr,...)
    const __m256d ti = _mm256_permute_pd(t, 0xF);               // (ti,ti,...)
    const __m256d os = _mm256_permute_pd(o, 0x5);               // (oi,or,...)
    return _mm256_fmaddsub_pd(o, tr, _mm256_mul_pd(os, ti));    // (or tr - oi ti, oi tr + or ti)
}
static inline __m256d bdit_neghi4(__m256d a) {
    return _mm256_xor_pd(a, _mm256_set_pd(-0.0, 0.0, -0.0, 0.0));
}
static inline __m256d bdit_neg4(__m256d a) {
    return _mm256_xor_pd(a, _mm256_set1_pd(-0.0));
}
static inline __m256d bdit_swap4(__m256d a) {                   // per-slot (re,im)->(im,re)
    return _mm256_permute_pd(a, 0x5);
}
static inline __m256d bdit_rev_slots4(__m256d a) {              // (c0,c1)->(c1,c0)
    return _mm256_permute2f128_pd(a, a, 0x01);
}

static inline __m256 bdit_cmul8f(__m256 o, __m256 t) {
    const __m256 tr = _mm256_moveldup_ps(t);                    // (tr,tr,...)
    const __m256 ti = _mm256_movehdup_ps(t);                    // (ti,ti,...)
    const __m256 os = _mm256_permute_ps(o, 0xB1);               // (oi,or,...)
    return _mm256_fmaddsub_ps(o, tr, _mm256_mul_ps(os, ti));    // (or tr - oi ti, oi tr + or ti)
}
static inline __m256 bdit_neghi8f(__m256 a) {
    return _mm256_xor_ps(a, _mm256_set_ps(-0.0f, 0.0f, -0.0f, 0.0f,
                                          -0.0f, 0.0f, -0.0f, 0.0f));
}
static inline __m256 bdit_neg8f(__m256 a) {
    return _mm256_xor_ps(a, _mm256_set1_ps(-0.0f));
}
static inline __m256 bdit_swap8f(__m256 a) {
    return _mm256_permute_ps(a, 0xB1);
}
static inline __m256 bdit_rev_slots8f(__m256 a) {               // (c0,c1,c2,c3)->(c3,c2,c1,c0)
    const __m256i idx = _mm256_setr_epi32(6, 7, 4, 5, 2, 3, 0, 1);
    return _mm256_permutevar8x32_ps(a, idx);
}
#endif

#if BRUUN_DIT_LEVEL >= 3
static inline __m512d bdit_cmul8(__m512d o, __m512d t) {
    const __m512d tr = _mm512_movedup_pd(t);
    const __m512d ti = _mm512_permute_pd(t, 0xFF);
    const __m512d os = _mm512_permute_pd(o, 0x55);
    return _mm512_fmaddsub_pd(o, tr, _mm512_mul_pd(os, ti));
}
static inline __m512d bdit_xorpd8(__m512d a, __m512d m) {
    return _mm512_castsi512_pd(_mm512_xor_epi64(_mm512_castpd_si512(a),
                                                _mm512_castpd_si512(m)));
}
static inline __m512d bdit_neghi8(__m512d a) {
    return bdit_xorpd8(a, _mm512_set_pd(-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0));
}
static inline __m512d bdit_neg8(__m512d a) {
    return bdit_xorpd8(a, _mm512_set1_pd(-0.0));
}
static inline __m512d bdit_swap8(__m512d a) {
    return _mm512_permute_pd(a, 0x55);
}
static inline __m512d bdit_rev_slots8(__m512d a) {              // (c0,c1,c2,c3)->(c3,c2,c1,c0)
    const __m512i idx = _mm512_setr_epi64(6, 7, 4, 5, 2, 3, 0, 1);
    return _mm512_permutexvar_pd(idx, a);
}
#endif

// ---------------------------------------------------------------------------
// Streaming merge kernels.
//
// dit_merge_fwd: one in-place CRT merge of sibling half-spectra E | O, each
// nb complex slots, into a 2*nb-slot spectrum. p points at E; O = p + 2*nb
// doubles. Bins k and nb-k are processed as a quartet so every source is read
// before its slot is overwritten; the partner twiddle is -SWAP(T_k).
// tw holds (cos t_k, -sin t_k) for k = 1..nb-1, contiguous.
// ---------------------------------------------------------------------------

static inline void dit_merge_fwd(double* BDIT_RESTRICT p, int nb,
                                 const double* BDIT_RESTRICT tw) {
    double* BDIT_RESTRICT E = p;
    double* BDIT_RESTRICT O = p + 2 * nb;

    // DC chain + free Nyquist pairing (upstairs bin nb).
    {
        const double edc = E[0], eny = E[1];
        const double odc = O[0], ony = O[1];
        E[0] = edc + odc;          // new DC
        E[1] = edc - odc;          // new Nyquist
        O[0] = eny;                // slot nb: re = E nyquist
        O[1] = -ony;               // slot nb: im = -O nyquist
    }
    if (nb < 2) return;

    int k = 1;
    const int half = nb / 2;

#if BRUUN_DIT_LEVEL >= 3
    for (; k + 3 < half; k += 4) {
        const int kp = nb - k - 3;                       // partner run, ascending base
        const __m512d Ek = _mm512_loadu_pd(E + 2 * k);
        const __m512d Ok = _mm512_loadu_pd(O + 2 * k);
        const __m512d Ep = bdit_rev_slots8(_mm512_loadu_pd(E + 2 * kp));
        const __m512d Op = bdit_rev_slots8(_mm512_loadu_pd(O + 2 * kp));
        const __m512d Tk = _mm512_loadu_pd(tw + 2 * (k - 1));
        const __m512d Tp = bdit_neg8(bdit_swap8(Tk));    // T_{nb-k} = -SWAP(T_k)
        const __m512d Wk = bdit_cmul8(Ok, Tk);
        const __m512d Wp = bdit_cmul8(Op, Tp);
        const __m512d lo_k = _mm512_add_pd(Ek, Wk);
        const __m512d hi_k = bdit_neghi8(_mm512_sub_pd(Ek, Wk));
        const __m512d lo_p = _mm512_add_pd(Ep, Wp);
        const __m512d hi_p = bdit_neghi8(_mm512_sub_pd(Ep, Wp));
        _mm512_storeu_pd(E + 2 * k, lo_k);
        _mm512_storeu_pd(E + 2 * kp, bdit_rev_slots8(lo_p));
        _mm512_storeu_pd(O + 2 * kp, bdit_rev_slots8(hi_k)); // slot 2nb-k = O slot nb-k
        _mm512_storeu_pd(O + 2 * k, hi_p);                   // slot 2nb-(nb-k) = O slot k
    }
#endif
#if BRUUN_DIT_LEVEL >= 2
    for (; k + 1 < half; k += 2) {
        const int kp = nb - k - 1;
        const __m256d Ek = _mm256_loadu_pd(E + 2 * k);
        const __m256d Ok = _mm256_loadu_pd(O + 2 * k);
        const __m256d Ep = bdit_rev_slots4(_mm256_loadu_pd(E + 2 * kp));
        const __m256d Op = bdit_rev_slots4(_mm256_loadu_pd(O + 2 * kp));
        const __m256d Tk = _mm256_loadu_pd(tw + 2 * (k - 1));
        const __m256d Tp = bdit_neg4(bdit_swap4(Tk));
        const __m256d Wk = bdit_cmul4(Ok, Tk);
        const __m256d Wp = bdit_cmul4(Op, Tp);
        const __m256d lo_k = _mm256_add_pd(Ek, Wk);
        const __m256d hi_k = bdit_neghi4(_mm256_sub_pd(Ek, Wk));
        const __m256d lo_p = _mm256_add_pd(Ep, Wp);
        const __m256d hi_p = bdit_neghi4(_mm256_sub_pd(Ep, Wp));
        _mm256_storeu_pd(E + 2 * k, lo_k);
        _mm256_storeu_pd(E + 2 * kp, bdit_rev_slots4(lo_p));
        _mm256_storeu_pd(O + 2 * kp, bdit_rev_slots4(hi_k));
        _mm256_storeu_pd(O + 2 * k, hi_p);
    }
#elif BRUUN_DIT_LEVEL == 1
    for (; k < half; ++k) {
        const int kp = nb - k;
        const bdit_v2 Ek = BV2_LD(E + 2 * k);
        const bdit_v2 Ok = BV2_LD(O + 2 * k);
        const bdit_v2 Ep = BV2_LD(E + 2 * kp);
        const bdit_v2 Op = BV2_LD(O + 2 * kp);
        const bdit_v2 Tk = BV2_LD(tw + 2 * (k - 1));
        const bdit_v2 Tp = BV2_NEG(BV2_SWAP(Tk));
        const bdit_v2 Wk = bdit_cmul(Ok, Tk);
        const bdit_v2 Wp = bdit_cmul(Op, Tp);
        BV2_ST(E + 2 * k,  BV2_ADD(Ek, Wk));
        BV2_ST(O + 2 * kp, BV2_NEGHI(BV2_SUB(Ek, Wk)));
        BV2_ST(E + 2 * kp, BV2_ADD(Ep, Wp));
        BV2_ST(O + 2 * k,  BV2_NEGHI(BV2_SUB(Ep, Wp)));
    }
#endif
    for (; k < half; ++k) {
        const int kp = nb - k;
        const double tr = tw[2 * (k - 1)], ti = tw[2 * (k - 1) + 1];
        const double er = E[2 * k], ei = E[2 * k + 1];
        const double orr = O[2 * k], oi = O[2 * k + 1];
        const double pr = E[2 * kp], pi = E[2 * kp + 1];
        const double qr = O[2 * kp], qi = O[2 * kp + 1];
        double wr, wi, vr, vi;
        cmul_scalar(orr, oi, tr, ti, wr, wi);
        // Partner twiddle: T_{nb-k} = -SWAP(T_k) = (-ti, -tr), with T_k = (cos, -sin).
        cmul_scalar(qr, qi, -ti, -tr, vr, vi);
        E[2 * k]      = er + wr;  E[2 * k + 1]      = ei + wi;
        O[2 * kp]     = er - wr;  O[2 * kp + 1]     = -(ei - wi);
        E[2 * kp]     = pr + vr;  E[2 * kp + 1]     = pi + vi;
        O[2 * k]      = pr - vr;  O[2 * k + 1]      = -(pi - vi);
    }
    // Middle bin k = nb/2, theta = pi/4: self-paired, safe in place.
    {
        const int km = half;
        const double tr = tw[2 * (km - 1)], ti = tw[2 * (km - 1) + 1];
        const double er = E[2 * km], ei = E[2 * km + 1];
        const double orr = O[2 * km], oi = O[2 * km + 1];
        double wr, wi;
        cmul_scalar(orr, oi, tr, ti, wr, wi);
        E[2 * km]     = er + wr;  E[2 * km + 1]     = ei + wi;
        O[2 * km]     = er - wr;  O[2 * km + 1]     = -(ei - wi);
    }
}

// Exact inverse of dit_merge_fwd over the same block, same twiddle row.
//   E = 0.5 * (lo + conj(hi)),  W = 0.5 * (lo - conj(hi)),  O = cmul(W, conj(T))
static inline void dit_merge_inv(double* BDIT_RESTRICT p, int nb,
                                 const double* BDIT_RESTRICT tw) {
    double* BDIT_RESTRICT E = p;
    double* BDIT_RESTRICT O = p + 2 * nb;

    {
        const double dc = E[0], ny = E[1];
        const double br = O[0], bi = O[1];
        E[0] = 0.5 * (dc + ny);    // E dc
        O[0] = 0.5 * (dc - ny);    // O dc
        E[1] = br;                 // E nyquist
        O[1] = -bi;                // O nyquist
    }
    if (nb < 2) return;

    int k = 1;
    const int half = nb / 2;

#if BRUUN_DIT_LEVEL >= 3
    {
        const __m512d hf = _mm512_set1_pd(0.5);
        for (; k + 3 < half; k += 4) {
            const int kp = nb - k - 3;
            const __m512d lo_k = _mm512_loadu_pd(E + 2 * k);
            const __m512d hi_k = bdit_rev_slots8(_mm512_loadu_pd(O + 2 * kp));
            const __m512d lo_p = bdit_rev_slots8(_mm512_loadu_pd(E + 2 * kp));
            const __m512d hi_p = _mm512_loadu_pd(O + 2 * k);
            const __m512d Tk = _mm512_loadu_pd(tw + 2 * (k - 1));
            const __m512d Ck = bdit_neghi8(Tk);                  // conj(T_k)
            const __m512d Cp = bdit_neghi8(bdit_neg8(bdit_swap8(Tk)));
            const __m512d ch_k = bdit_neghi8(hi_k);
            const __m512d ch_p = bdit_neghi8(hi_p);
            const __m512d Ek = _mm512_mul_pd(hf, _mm512_add_pd(lo_k, ch_k));
            const __m512d Wk = _mm512_mul_pd(hf, _mm512_sub_pd(lo_k, ch_k));
            const __m512d Ep = _mm512_mul_pd(hf, _mm512_add_pd(lo_p, ch_p));
            const __m512d Wp = _mm512_mul_pd(hf, _mm512_sub_pd(lo_p, ch_p));
            _mm512_storeu_pd(E + 2 * k, Ek);
            _mm512_storeu_pd(O + 2 * k, bdit_cmul8(Wk, Ck));
            _mm512_storeu_pd(E + 2 * kp, bdit_rev_slots8(Ep));
            _mm512_storeu_pd(O + 2 * kp, bdit_rev_slots8(bdit_cmul8(Wp, Cp)));
        }
    }
#endif
#if BRUUN_DIT_LEVEL >= 2
    {
        const __m256d hf = _mm256_set1_pd(0.5);
        for (; k + 1 < half; k += 2) {
            const int kp = nb - k - 1;
            const __m256d lo_k = _mm256_loadu_pd(E + 2 * k);
            const __m256d hi_k = bdit_rev_slots4(_mm256_loadu_pd(O + 2 * kp));
            const __m256d lo_p = bdit_rev_slots4(_mm256_loadu_pd(E + 2 * kp));
            const __m256d hi_p = _mm256_loadu_pd(O + 2 * k);
            const __m256d Tk = _mm256_loadu_pd(tw + 2 * (k - 1));
            const __m256d Ck = bdit_neghi4(Tk);
            const __m256d Cp = bdit_neghi4(bdit_neg4(bdit_swap4(Tk)));
            const __m256d ch_k = bdit_neghi4(hi_k);
            const __m256d ch_p = bdit_neghi4(hi_p);
            const __m256d Ek = _mm256_mul_pd(hf, _mm256_add_pd(lo_k, ch_k));
            const __m256d Wk = _mm256_mul_pd(hf, _mm256_sub_pd(lo_k, ch_k));
            const __m256d Ep = _mm256_mul_pd(hf, _mm256_add_pd(lo_p, ch_p));
            const __m256d Wp = _mm256_mul_pd(hf, _mm256_sub_pd(lo_p, ch_p));
            _mm256_storeu_pd(E + 2 * k, Ek);
            _mm256_storeu_pd(O + 2 * k, bdit_cmul4(Wk, Ck));
            _mm256_storeu_pd(E + 2 * kp, bdit_rev_slots4(Ep));
            _mm256_storeu_pd(O + 2 * kp, bdit_rev_slots4(bdit_cmul4(Wp, Cp)));
        }
    }
#elif BRUUN_DIT_LEVEL == 1
    {
        const bdit_v2 hf = BV2_SET1(0.5);
        for (; k < half; ++k) {
            const int kp = nb - k;
            const bdit_v2 lo_k = BV2_LD(E + 2 * k);
            const bdit_v2 hi_k = BV2_LD(O + 2 * kp);
            const bdit_v2 lo_p = BV2_LD(E + 2 * kp);
            const bdit_v2 hi_p = BV2_LD(O + 2 * k);
            const bdit_v2 Tk = BV2_LD(tw + 2 * (k - 1));
            const bdit_v2 Ck = BV2_NEGHI(Tk);
            const bdit_v2 Cp = BV2_NEGHI(BV2_NEG(BV2_SWAP(Tk)));
            const bdit_v2 ch_k = BV2_NEGHI(hi_k);
            const bdit_v2 ch_p = BV2_NEGHI(hi_p);
            const bdit_v2 Ek = BV2_MUL(hf, BV2_ADD(lo_k, ch_k));
            const bdit_v2 Wk = BV2_MUL(hf, BV2_SUB(lo_k, ch_k));
            const bdit_v2 Ep = BV2_MUL(hf, BV2_ADD(lo_p, ch_p));
            const bdit_v2 Wp = BV2_MUL(hf, BV2_SUB(lo_p, ch_p));
            BV2_ST(E + 2 * k, Ek);
            BV2_ST(O + 2 * k, bdit_cmul(Wk, Ck));
            BV2_ST(E + 2 * kp, Ep);
            BV2_ST(O + 2 * kp, bdit_cmul(Wp, Cp));
        }
    }
#endif
    for (; k < half; ++k) {
        const int kp = nb - k;
        const double tr = tw[2 * (k - 1)], ti = tw[2 * (k - 1) + 1];
        const double lr = E[2 * k], li = E[2 * k + 1];
        const double hr = O[2 * kp], hi = O[2 * kp + 1];
        const double mr = E[2 * kp], mi = E[2 * kp + 1];
        const double nr = O[2 * k], ni = O[2 * k + 1];
        const double er = 0.5 * (lr + hr), ei = 0.5 * (li - hi);
        const double wr = 0.5 * (lr - hr), wi = 0.5 * (li + hi);
        const double pr = 0.5 * (mr + nr), pi = 0.5 * (mi - ni);
        const double vr = 0.5 * (mr - nr), vi = 0.5 * (mi + ni);
        double a, b, c, d;
        cmul_scalar(wr, wi, tr, -ti, a, b);              // conj(T_k)
        cmul_scalar(vr, vi, -ti, tr, c, d);              // conj(-SWAP(T_k)) = (-ti, tr)
        E[2 * k] = er;       E[2 * k + 1] = ei;
        O[2 * k] = a;        O[2 * k + 1] = b;
        E[2 * kp] = pr;      E[2 * kp + 1] = pi;
        O[2 * kp] = c;       O[2 * kp + 1] = d;
    }
    {
        const int km = half;
        const double tr = tw[2 * (km - 1)], ti = tw[2 * (km - 1) + 1];
        const double lr = E[2 * km], li = E[2 * km + 1];
        const double hr = O[2 * km], hi = O[2 * km + 1];
        const double er = 0.5 * (lr + hr), ei = 0.5 * (li - hi);
        const double wr = 0.5 * (lr - hr), wi = 0.5 * (li + hi);
        double a, b;
        cmul_scalar(wr, wi, tr, -ti, a, b);
        E[2 * km] = er;      E[2 * km + 1] = ei;
        O[2 * km] = a;       O[2 * km + 1] = b;
    }
}

// Single-precision forward merge. AVX2 processes four complex slots per vector;
// NEON processes two complex slots per vector. Other backends use the scalar
// reference path so the f32 experiment stays portable.
#if defined(BRUUN_DIT_NEON_128)
static inline float32x4_t bdit_f32_neg_mask(float32x4_t a,
                                            uint32_t a0, uint32_t a1,
                                            uint32_t a2, uint32_t a3) {
    const uint32x4_t m = { a0, a1, a2, a3 };
    return vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(a), m));
}

static inline float32x4_t bdit_f32_neg(float32x4_t a) {
    return bdit_f32_neg_mask(a, 0x80000000U, 0x80000000U, 0x80000000U, 0x80000000U);
}

static inline float32x4_t bdit_f32_neghi(float32x4_t a) {
    return bdit_f32_neg_mask(a, 0U, 0x80000000U, 0U, 0x80000000U);
}

static inline float32x4_t bdit_f32_negeven(float32x4_t a) {
    return bdit_f32_neg_mask(a, 0x80000000U, 0U, 0x80000000U, 0U);
}

static inline float32x4_t bdit_f32_swap(float32x4_t a) {
    return vrev64q_f32(a);
}

static inline float32x4_t bdit_f32_rev_slots(float32x4_t a) {
    return vextq_f32(a, a, 2);
}

static inline float32x4_t bdit_f32_cmul(float32x4_t o, float32x4_t t) {
    const float32x2_t tlo = vget_low_f32(t);
    const float32x2_t thi = vget_high_f32(t);
    const float32x4_t tr = vcombine_f32(vdup_lane_f32(tlo, 0), vdup_lane_f32(thi, 0));
    const float32x4_t ti = vcombine_f32(vdup_lane_f32(tlo, 1), vdup_lane_f32(thi, 1));
    const float32x4_t re = vmulq_f32(o, tr);
    return vfmaq_f32(re, bdit_f32_negeven(bdit_f32_swap(o)), ti);
}
#endif

static inline void dit_merge_fwd_f32(float* BDIT_RESTRICT p, int nb,
                                     const float* BDIT_RESTRICT tw) {
    float* BDIT_RESTRICT E = p;
    float* BDIT_RESTRICT O = p + 2 * nb;

    {
        const float edc = E[0], eny = E[1];
        const float odc = O[0], ony = O[1];
        E[0] = edc + odc;
        E[1] = edc - odc;
        O[0] = eny;
        O[1] = -ony;
    }
    if (nb < 2) return;

    int k = 1;
    const int half = nb / 2;

#if BRUUN_DIT_LEVEL >= 2
    for (; k + 3 < half; k += 4) {
        const int kp = nb - k - 3;
        const __m256 Ek = _mm256_loadu_ps(E + 2 * k);
        const __m256 Ok = _mm256_loadu_ps(O + 2 * k);
        const __m256 Ep = bdit_rev_slots8f(_mm256_loadu_ps(E + 2 * kp));
        const __m256 Op = bdit_rev_slots8f(_mm256_loadu_ps(O + 2 * kp));
        const __m256 Tk = _mm256_loadu_ps(tw + 2 * (k - 1));
        const __m256 Tp = bdit_neg8f(bdit_swap8f(Tk));
        const __m256 Wk = bdit_cmul8f(Ok, Tk);
        const __m256 Wp = bdit_cmul8f(Op, Tp);
        const __m256 lo_k = _mm256_add_ps(Ek, Wk);
        const __m256 hi_k = bdit_neghi8f(_mm256_sub_ps(Ek, Wk));
        const __m256 lo_p = _mm256_add_ps(Ep, Wp);
        const __m256 hi_p = bdit_neghi8f(_mm256_sub_ps(Ep, Wp));
        _mm256_storeu_ps(E + 2 * k, lo_k);
        _mm256_storeu_ps(E + 2 * kp, bdit_rev_slots8f(lo_p));
        _mm256_storeu_ps(O + 2 * kp, bdit_rev_slots8f(hi_k));
        _mm256_storeu_ps(O + 2 * k, hi_p);
    }
#elif defined(BRUUN_DIT_NEON_128)
    for (; k + 1 < half; k += 2) {
        const int kp = nb - k - 1;
        const float32x4_t Ek = vld1q_f32(E + 2 * k);
        const float32x4_t Ok = vld1q_f32(O + 2 * k);
        const float32x4_t Ep = bdit_f32_rev_slots(vld1q_f32(E + 2 * kp));
        const float32x4_t Op = bdit_f32_rev_slots(vld1q_f32(O + 2 * kp));
        const float32x4_t Tk = vld1q_f32(tw + 2 * (k - 1));
        const float32x4_t Tp = bdit_f32_neg(bdit_f32_swap(Tk));
        const float32x4_t Wk = bdit_f32_cmul(Ok, Tk);
        const float32x4_t Wp = bdit_f32_cmul(Op, Tp);
        const float32x4_t lo_k = vaddq_f32(Ek, Wk);
        const float32x4_t hi_k = bdit_f32_neghi(vsubq_f32(Ek, Wk));
        const float32x4_t lo_p = vaddq_f32(Ep, Wp);
        const float32x4_t hi_p = bdit_f32_neghi(vsubq_f32(Ep, Wp));
        vst1q_f32(E + 2 * k, lo_k);
        vst1q_f32(E + 2 * kp, bdit_f32_rev_slots(lo_p));
        vst1q_f32(O + 2 * kp, bdit_f32_rev_slots(hi_k));
        vst1q_f32(O + 2 * k, hi_p);
    }
#endif

    for (; k < half; ++k) {
        const int kp = nb - k;
        const float tr = tw[2 * (k - 1)], ti = tw[2 * (k - 1) + 1];
        const float er = E[2 * k], ei = E[2 * k + 1];
        const float orr = O[2 * k], oi = O[2 * k + 1];
        const float pr = E[2 * kp], pi = E[2 * kp + 1];
        const float qr = O[2 * kp], qi = O[2 * kp + 1];
        float wr, wi, vr, vi;
        cmul_scalar_f32(orr, oi, tr, ti, wr, wi);
        cmul_scalar_f32(qr, qi, -ti, -tr, vr, vi);
        E[2 * k]      = er + wr;  E[2 * k + 1]      = ei + wi;
        O[2 * kp]     = er - wr;  O[2 * kp + 1]     = -(ei - wi);
        E[2 * kp]     = pr + vr;  E[2 * kp + 1]     = pi + vi;
        O[2 * k]      = pr - vr;  O[2 * k + 1]      = -(pi - vi);
    }

    {
        const int km = half;
        const float tr = tw[2 * (km - 1)], ti = tw[2 * (km - 1) + 1];
        const float er = E[2 * km], ei = E[2 * km + 1];
        const float orr = O[2 * km], oi = O[2 * km + 1];
        float wr, wi;
        cmul_scalar_f32(orr, oi, tr, ti, wr, wi);
        E[2 * km]     = er + wr;  E[2 * km + 1]     = ei + wi;
        O[2 * km]     = er - wr;  O[2 * km + 1]     = -(ei - wi);
    }
}

// ---------------------------------------------------------------------------
// Plan
// ---------------------------------------------------------------------------

class RFFT_DIT {
public:
    explicit RFFT_DIT(int n)
        : N(n), L(ilog2_pow2(n)), NB(n / 2 + 1)
    {
        if (!is_power2(N) || N < 4)
            throw std::invalid_argument("Bruun DIT RFFT requires power-of-two N >= 4");

        const int M = N / 2;

        // Bit-reversal gather table over L-1 bits; it is an involution, so the
        // same table drives the inverse scatter as scattered reads.
        REV.resize(M);
        for (int b = 0; b < M; ++b) REV[b] = bitrev_int(b, L - 1);

        // One master twiddle ladder. The finest stage (nb = N/4) carries
        // theta_k = 2*pi*k/N for k = 1..N/4-1, stored interleaved as
        // (cos theta, -sin theta). Every coarser stage is the even-index
        // subsample of the next finer one, copied bit-exactly.
        TW_OFF.assign(L, 0);
        int total = 0;
        for (int nb = 1; nb <= N / 4; nb <<= 1) total += 2 * (nb - 1);
        TW.resize(total > 0 ? total : 1);

        {
            int off = total;
            int prev_off = -1;
            for (int nb = N / 4; nb >= 1; nb >>= 1) {
                off -= 2 * (nb - 1);
                TW_OFF[stage_of(nb)] = off;
                if (nb == N / 4) {
                    for (int kk = 1; kk < nb; ++kk) {
                        const double t = (M_PI * kk) / (2.0 * nb);
                        TW[off + 2 * (kk - 1)]     = std::cos(t);
                        TW[off + 2 * (kk - 1) + 1] = -std::sin(t);
                    }
                } else {
                    for (int kk = 1; kk < nb; ++kk) {
                        TW[off + 2 * (kk - 1)]     = TW[prev_off + 2 * (2 * kk - 1)];
                        TW[off + 2 * (kk - 1) + 1] = TW[prev_off + 2 * (2 * kk - 1) + 1];
                    }
                }
                prev_off = off;
            }
        }
    }

    int size() const { return N; }
    int bins() const { return NB; }
    int work_size() const { return N; }
    int halfcomplex_size() const { return N; }

    // time -> natural-order halfcomplex, hc[0..N): slot 0 = (DC, Nyquist),
    // slot k = (Re X[k], Im X[k]). This is the fast path: sequential output,
    // zero conversion, the natural analog of the DIF kernel's native order.
    void forward_halfcomplex(const double* BDIT_RESTRICT in,
                             double* BDIT_RESTRICT hc) const {
        gather_stage(in, hc);
        run_merges(hc);
    }

    // Experimental q=3 tiled bit-reversal gather. The transform result matches
    // forward_halfcomplex(), but the initial permutation moves cache-line
    // chunks through a small L1 tile instead of issuing one random read per
    // destination slot.
    void forward_halfcomplex_tiled(const double* BDIT_RESTRICT in,
                                   double* BDIT_RESTRICT hc) const {
        const int first_unmerged_stage = gather_tiled_stage(in, hc);
        run_merges_from(hc, first_unmerged_stage);
    }

    // Same transform when the caller already supplies samples in the DIT leaf
    // order expected by gather_stage:
    //     br[b]       = x[bit_reverse(b)]
    //     br[N/2 + b] = x[N/2 + bit_reverse(b)]
    // This is useful for pipelines that can produce the time stream in
    // scrambled order and want to avoid the input gather.
    void forward_halfcomplex_bitreversed(const double* BDIT_RESTRICT in,
                                         double* BDIT_RESTRICT hc) const {
        gather_bitreversed_stage(in, hc);
        run_merges(hc);
    }

    // Prepared DIT time layout: hc already contains the gather-stage slots
    //     slot b = (x[rev(b)] + x[rev(b)+N/2],
    //               x[rev(b)] - x[rev(b)+N/2]).
    // This is the zero-permutation pipeline entry point for framers that can
    // write directly into the decimated time domain.
    void forward_halfcomplex_pregathered_inplace(double* BDIT_RESTRICT hc) const {
        run_merges(hc);
    }

    void forward_halfcomplex_pregathered(const double* BDIT_RESTRICT prepared,
                                         double* BDIT_RESTRICT hc) const {
        std::memcpy(hc, prepared, sizeof(double) * static_cast<std::size_t>(N));
        forward_halfcomplex_pregathered_inplace(hc);
    }

    // time -> standard FFTW-order complex bins 0..N/2. The unpack is one
    // sequential streaming pass; there is no permutation here by construction.
    void forward(const double* BDIT_RESTRICT in, complex_t* BDIT_RESTRICT X,
                 double* BDIT_RESTRICT work) const {
        forward_halfcomplex(in, work);
        X[0].re = work[0];      X[0].im = 0.0;
        X[N / 2].re = work[1];  X[N / 2].im = 0.0;
        for (int k = 1; k < N / 2; ++k) {
            X[k].re = work[2 * k];
            X[k].im = work[2 * k + 1];
        }
    }

    void forward_tiled(const double* BDIT_RESTRICT in,
                       complex_t* BDIT_RESTRICT X,
                       double* BDIT_RESTRICT work) const {
        forward_halfcomplex_tiled(in, work);
        X[0].re = work[0];      X[0].im = 0.0;
        X[N / 2].re = work[1];  X[N / 2].im = 0.0;
        for (int k = 1; k < N / 2; ++k) {
            X[k].re = work[2 * k];
            X[k].im = work[2 * k + 1];
        }
    }

    void forward_bitreversed(const double* BDIT_RESTRICT in,
                             complex_t* BDIT_RESTRICT X,
                             double* BDIT_RESTRICT work) const {
        forward_halfcomplex_bitreversed(in, work);
        X[0].re = work[0];      X[0].im = 0.0;
        X[N / 2].re = work[1];  X[N / 2].im = 0.0;
        for (int k = 1; k < N / 2; ++k) {
            X[k].re = work[2 * k];
            X[k].im = work[2 * k + 1];
        }
    }

    void forward_pregathered(const double* BDIT_RESTRICT prepared,
                             complex_t* BDIT_RESTRICT X,
                             double* BDIT_RESTRICT work) const {
        forward_halfcomplex_pregathered(prepared, work);
        X[0].re = work[0];      X[0].im = 0.0;
        X[N / 2].re = work[1];  X[N / 2].im = 0.0;
        for (int k = 1; k < N / 2; ++k) {
            X[k].re = work[2 * k];
            X[k].im = work[2 * k + 1];
        }
    }

    // halfcomplex -> time, exact CRT un-merging; destroys hc.
    void inverse_halfcomplex(double* BDIT_RESTRICT hc,
                             double* BDIT_RESTRICT out) const {
        run_unmerges(hc);
        scatter_stage(hc, out);
    }

    void inverse(const complex_t* BDIT_RESTRICT X, double* BDIT_RESTRICT out,
                 double* BDIT_RESTRICT work) const {
        work[0] = X[0].re;
        work[1] = X[N / 2].re;
        for (int k = 1; k < N / 2; ++k) {
            work[2 * k]     = X[k].re;
            work[2 * k + 1] = X[k].im;
        }
        inverse_halfcomplex(work, out);
    }

    // Pointwise spectral multiply in the halfcomplex layout: the residue-domain
    // filter of the DIF kernel, restated on this side. Slot 0 multiplies DC and
    // Nyquist as the two real characters; every other slot is a complex multiply.
    static void pointwise_multiply(double* BDIT_RESTRICT a,
                                   const double* BDIT_RESTRICT h, int n) {
        a[0] *= h[0];
        a[1] *= h[1];
        int k = 1;
        const int M = n / 2;
#if BRUUN_DIT_LEVEL >= 3
        for (; k + 3 < M; k += 4) {
            const __m512d av = _mm512_loadu_pd(a + 2 * k);
            const __m512d hv = _mm512_loadu_pd(h + 2 * k);
            _mm512_storeu_pd(a + 2 * k, bdit_cmul8(av, hv));
        }
#endif
#if BRUUN_DIT_LEVEL >= 2
        for (; k + 1 < M; k += 2) {
            const __m256d av = _mm256_loadu_pd(a + 2 * k);
            const __m256d hv = _mm256_loadu_pd(h + 2 * k);
            _mm256_storeu_pd(a + 2 * k, bdit_cmul4(av, hv));
        }
#elif BRUUN_DIT_LEVEL == 1
        for (; k < M; ++k) {
            const bdit_v2 av = BV2_LD(a + 2 * k);
            const bdit_v2 hv = BV2_LD(h + 2 * k);
            BV2_ST(a + 2 * k, bdit_cmul(av, hv));
        }
#endif
        for (; k < M; ++k) {
            double wr, wi;
            cmul_scalar(a[2 * k], a[2 * k + 1], h[2 * k], h[2 * k + 1], wr, wi);
            a[2 * k] = wr; a[2 * k + 1] = wi;
        }
    }

private:
    int N, L, NB;
    std::vector<int> REV;
    std::vector<double> TW;
    std::vector<int> TW_OFF;

    int stage_of(int nb) const { return ilog2_pow2(nb); }

    // Fused bit-reversal gather + 2-point spectra: scattered reads, two of them
    // per sequential complex write. v[2b] = x[j] + x[j+N/2], v[2b+1] = x[j] -
    // x[j+N/2], j = REV[b].
    void gather_stage(const double* BDIT_RESTRICT in, double* BDIT_RESTRICT v) const {
        const int M = N / 2;
        const double* BDIT_RESTRICT hi = in + M;
        for (int b = 0; b < M; ++b) {
            const int j = REV[b];
            const double a = in[j];
            const double c = hi[j];
            v[2 * b]     = a + c;
            v[2 * b + 1] = a - c;
        }
    }

    int gather_tiled_stage(const double* BDIT_RESTRICT in,
                           double* BDIT_RESTRICT v) const {
        const int M = N / 2;
        const int bits = L - 1;
        const int tile_bits = 3;
        if (bits < 6) {
            gather_stage(in, v);
            return 0;
        }

        const int low_bits = bits - tile_bits;
        const int low_count = 1 << low_bits;
        const double* BDIT_RESTRICT hi_in = in + M;
        static const int rev3[8] = { 0, 4, 2, 6, 1, 5, 3, 7 };
        const double* BDIT_RESTRICT tw2 = TW.data() + TW_OFF[1];

        for (int lo0 = 0; lo0 < low_count; lo0 += 8) {
            double sums[8][8];
            double diffs[8][8];

            for (int t = 0; t < 8; ++t) {
                const int lo = lo0 + t;
                const int src = bitrev_int(lo, low_bits) << tile_bits;
                for (int h = 0; h < 8; ++h) {
                    const int j = src + rev3[h];
                    const double a = in[j];
                    const double c = hi_in[j];
                    sums[h][t] = a + c;
                    diffs[h][t] = a - c;
                }
            }

            for (int h = 0; h < 8; ++h) {
                double row[16];
                for (int t = 0; t < 8; ++t) {
                    row[2 * t] = sums[h][t];
                    row[2 * t + 1] = diffs[h][t];
                }

                dit_merge_fwd(row, 1, tw2);
                dit_merge_fwd(row + 4, 1, tw2);
                dit_merge_fwd(row + 8, 1, tw2);
                dit_merge_fwd(row + 12, 1, tw2);
                dit_merge_fwd(row, 2, tw2);
                dit_merge_fwd(row + 8, 2, tw2);

                double* BDIT_RESTRICT dst = v + 2 * (h * low_count + lo0);
                for (int t = 0; t < 16; ++t) {
                    dst[t] = row[t];
                }
            }
        }

        return 2;
    }

    void gather_bitreversed_stage(const double* BDIT_RESTRICT in,
                                  double* BDIT_RESTRICT v) const {
        const int M = N / 2;
        const double* BDIT_RESTRICT hi = in + M;
        for (int b = 0; b < M; ++b) {
            const double a = in[b];
            const double c = hi[b];
            v[2 * b]     = a + c;
            v[2 * b + 1] = a - c;
        }
    }

    // Inverse of gather_stage: scattered reads from v, two sequential write
    // streams into out. REV is an involution, so the same table serves.
    void scatter_stage(const double* BDIT_RESTRICT v, double* BDIT_RESTRICT out) const {
        const int M = N / 2;
        double* BDIT_RESTRICT hi = out + M;
        for (int j = 0; j < M; ++j) {
            const int b = REV[j];
            const double s = v[2 * b];
            const double d = v[2 * b + 1];
            out[j] = 0.5 * (s + d);
            hi[j]  = 0.5 * (s - d);
        }
    }

    void run_merges(double* BDIT_RESTRICT v) const {
        run_merges_from(v, 0);
    }

    void run_merges_from(double* BDIT_RESTRICT v, int first_stage) const {
        int stage = 0;
        for (int nb = 1; nb <= N / 4; nb <<= 1, ++stage) {
            if (stage < first_stage) {
                continue;
            }
            const double* tw = TW.data() + TW_OFF[stage];
            const int blk = 4 * nb;                      // doubles per merged block
            for (int off = 0; off < N; off += blk)
                dit_merge_fwd(v + off, nb, tw);
        }
    }

    void run_unmerges(double* BDIT_RESTRICT v) const {
        int stage = stage_of(N / 4);
        for (int nb = N / 4; nb >= 1; nb >>= 1, --stage) {
            const double* tw = TW.data() + TW_OFF[stage];
            const int blk = 4 * nb;
            for (int off = 0; off < N; off += blk)
                dit_merge_inv(v + off, nb, tw);
        }
    }
};

class RFFT_DIT_F32 {
public:
    explicit RFFT_DIT_F32(int n)
        : N(n), L(ilog2_pow2(n)), NB(n / 2 + 1)
    {
        if (!is_power2(N) || N < 4)
            throw std::invalid_argument("Bruun DIT f32 RFFT requires power-of-two N >= 4");

        const int M = N / 2;
        REV.resize(M);
        for (int b = 0; b < M; ++b) REV[b] = bitrev_int(b, L - 1);

        TW_OFF.assign(L, 0);
        int total = 0;
        for (int nb = 1; nb <= N / 4; nb <<= 1) total += 2 * (nb - 1);
        TW.resize(total > 0 ? total : 1);

        {
            int off = total;
            int prev_off = -1;
            for (int nb = N / 4; nb >= 1; nb >>= 1) {
                off -= 2 * (nb - 1);
                TW_OFF[stage_of(nb)] = off;
                if (nb == N / 4) {
                    for (int kk = 1; kk < nb; ++kk) {
                        const double t = (M_PI * kk) / (2.0 * nb);
                        TW[off + 2 * (kk - 1)]     = static_cast<float>(std::cos(t));
                        TW[off + 2 * (kk - 1) + 1] = static_cast<float>(-std::sin(t));
                    }
                } else {
                    for (int kk = 1; kk < nb; ++kk) {
                        TW[off + 2 * (kk - 1)]     = TW[prev_off + 2 * (2 * kk - 1)];
                        TW[off + 2 * (kk - 1) + 1] = TW[prev_off + 2 * (2 * kk - 1) + 1];
                    }
                }
                prev_off = off;
            }
        }
    }

    int size() const { return N; }
    int bins() const { return NB; }
    int work_size() const { return N; }
    int halfcomplex_size() const { return N; }

    void forward_halfcomplex(const float* BDIT_RESTRICT in,
                             float* BDIT_RESTRICT hc) const {
        gather_stage(in, hc);
        run_merges(hc);
    }

    void forward_halfcomplex_tiled(const float* BDIT_RESTRICT in,
                                   float* BDIT_RESTRICT hc) const {
        const int first_unmerged_stage = gather_tiled_stage(in, hc);
        run_merges_from(hc, first_unmerged_stage);
    }

    void forward_halfcomplex_bitreversed(const float* BDIT_RESTRICT in,
                                         float* BDIT_RESTRICT hc) const {
        gather_bitreversed_stage(in, hc);
        run_merges(hc);
    }

    void forward_halfcomplex_pregathered_inplace(float* BDIT_RESTRICT hc) const {
        run_merges(hc);
    }

    void forward_halfcomplex_pregathered(const float* BDIT_RESTRICT prepared,
                                         float* BDIT_RESTRICT hc) const {
        std::memcpy(hc, prepared, sizeof(float) * static_cast<std::size_t>(N));
        forward_halfcomplex_pregathered_inplace(hc);
    }

    void forward(const float* BDIT_RESTRICT in,
                 complex_f32_t* BDIT_RESTRICT X,
                 float* BDIT_RESTRICT work) const {
        forward_halfcomplex(in, work);
        unpack(work, X);
    }

    void forward_tiled(const float* BDIT_RESTRICT in,
                       complex_f32_t* BDIT_RESTRICT X,
                       float* BDIT_RESTRICT work) const {
        forward_halfcomplex_tiled(in, work);
        unpack(work, X);
    }

    void forward_bitreversed(const float* BDIT_RESTRICT in,
                             complex_f32_t* BDIT_RESTRICT X,
                             float* BDIT_RESTRICT work) const {
        forward_halfcomplex_bitreversed(in, work);
        unpack(work, X);
    }

    void forward_pregathered(const float* BDIT_RESTRICT prepared,
                             complex_f32_t* BDIT_RESTRICT X,
                             float* BDIT_RESTRICT work) const {
        forward_halfcomplex_pregathered(prepared, work);
        unpack(work, X);
    }

private:
    int N, L, NB;
    std::vector<int> REV;
    std::vector<float> TW;
    std::vector<int> TW_OFF;

    int stage_of(int nb) const { return ilog2_pow2(nb); }

    void unpack(const float* BDIT_RESTRICT hc,
                complex_f32_t* BDIT_RESTRICT X) const {
        X[0].re = hc[0];      X[0].im = 0.0f;
        X[N / 2].re = hc[1];  X[N / 2].im = 0.0f;
        for (int k = 1; k < N / 2; ++k) {
            X[k].re = hc[2 * k];
            X[k].im = hc[2 * k + 1];
        }
    }

    void gather_stage(const float* BDIT_RESTRICT in,
                      float* BDIT_RESTRICT v) const {
        const int M = N / 2;
        const float* BDIT_RESTRICT hi = in + M;
        for (int b = 0; b < M; ++b) {
            const int j = REV[b];
            const float a = in[j];
            const float c = hi[j];
            v[2 * b]     = a + c;
            v[2 * b + 1] = a - c;
        }
    }

    int gather_tiled_stage(const float* BDIT_RESTRICT in,
                           float* BDIT_RESTRICT v) const {
        const int M = N / 2;
        const int bits = L - 1;
        const int tile_bits = 3;
        if (bits < 6) {
            gather_stage(in, v);
            return 0;
        }

        const int low_bits = bits - tile_bits;
        const int low_count = 1 << low_bits;
        const float* BDIT_RESTRICT hi_in = in + M;
        static const int rev3[8] = { 0, 4, 2, 6, 1, 5, 3, 7 };
        const float* BDIT_RESTRICT tw2 = TW.data() + TW_OFF[1];

        for (int lo0 = 0; lo0 < low_count; lo0 += 8) {
            float sums[8][8];
            float diffs[8][8];

            for (int t = 0; t < 8; ++t) {
                const int lo = lo0 + t;
                const int src = bitrev_int(lo, low_bits) << tile_bits;
                for (int h = 0; h < 8; ++h) {
                    const int j = src + rev3[h];
                    const float a = in[j];
                    const float c = hi_in[j];
                    sums[h][t] = a + c;
                    diffs[h][t] = a - c;
                }
            }

            for (int h = 0; h < 8; ++h) {
                float row[16];
                for (int t = 0; t < 8; ++t) {
                    row[2 * t] = sums[h][t];
                    row[2 * t + 1] = diffs[h][t];
                }

                dit_merge_fwd_f32(row, 1, tw2);
                dit_merge_fwd_f32(row + 4, 1, tw2);
                dit_merge_fwd_f32(row + 8, 1, tw2);
                dit_merge_fwd_f32(row + 12, 1, tw2);
                dit_merge_fwd_f32(row, 2, tw2);
                dit_merge_fwd_f32(row + 8, 2, tw2);

                float* BDIT_RESTRICT dst = v + 2 * (h * low_count + lo0);
                for (int t = 0; t < 16; ++t) {
                    dst[t] = row[t];
                }
            }
        }

        return 2;
    }

    void gather_bitreversed_stage(const float* BDIT_RESTRICT in,
                                  float* BDIT_RESTRICT v) const {
        const int M = N / 2;
        const float* BDIT_RESTRICT hi = in + M;
        for (int b = 0; b < M; ++b) {
            const float a = in[b];
            const float c = hi[b];
            v[2 * b]     = a + c;
            v[2 * b + 1] = a - c;
        }
    }

    void run_merges(float* BDIT_RESTRICT v) const {
        run_merges_from(v, 0);
    }

    void run_merges_from(float* BDIT_RESTRICT v, int first_stage) const {
        int stage = 0;
        for (int nb = 1; nb <= N / 4; nb <<= 1, ++stage) {
            if (stage < first_stage) {
                continue;
            }
            const float* tw = TW.data() + TW_OFF[stage];
            const int blk = 4 * nb;
            for (int off = 0; off < N; off += blk)
                dit_merge_fwd_f32(v + off, nb, tw);
        }
    }
};

} // namespace bruun_dit
