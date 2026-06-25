#pragma once

// SIMD building blocks for generalized Bruun non-power-of-two pathways.
// These primitives live beside GenBruun and stay separate from the power-of-two
// Bruun kernel while following the same backend capability staging.
//
// Layout goal: keep non-power-of-two residue work in a structure-of-arrays form
// that mirrors the normalized Bruun kernel: contiguous [A0 | B0 | A1 | B1]
// lanes, coefficient streams beside the data, and multiply-add pairs expressed
// as FMA-friendly real rotations.
//
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

// BRUUN_NP2_LEVEL follows the main Bruun kernel staging:
//   0 = scalar, 1 = 128-bit SSE2/NEON, 2 = AVX2+FMA.
#if defined(__AVX2__) && defined(__FMA__)
#  include <immintrin.h>
#  define BRUUN_NP2_LEVEL 2
#elif defined(__SSE2__) || defined(_M_X64)
#  include <immintrin.h>
#  define BRUUN_NP2_LEVEL 1
#  define BRUUN_NP2_X86_128 1
#elif defined(__aarch64__) && defined(__ARM_NEON)
#  include <arm_neon.h>
#  define BRUUN_NP2_LEVEL 1
#  define BRUUN_NP2_NEON_128 1
#else
#  define BRUUN_NP2_LEVEL 0
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#define BRUUN_NP2_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define BRUUN_NP2_RESTRICT __restrict__
#else
#define BRUUN_NP2_RESTRICT
#endif

namespace bruun_nonpower2 {

#if defined(BRUUN_NP2_X86_128)
typedef __m128d np2_v2;
static inline np2_v2 v2_load(const double* p) { return _mm_loadu_pd(p); }
static inline void v2_store(double* p, np2_v2 a) { _mm_storeu_pd(p, a); }
static inline np2_v2 v2_add(np2_v2 a, np2_v2 b) { return _mm_add_pd(a, b); }
static inline np2_v2 v2_sub(np2_v2 a, np2_v2 b) { return _mm_sub_pd(a, b); }
static inline np2_v2 v2_mul(np2_v2 a, np2_v2 b) { return _mm_mul_pd(a, b); }
static inline np2_v2 v2_madd(np2_v2 a, np2_v2 b, np2_v2 c) { return v2_add(a, v2_mul(b, c)); }
static inline np2_v2 v2_msub(np2_v2 a, np2_v2 b, np2_v2 c) { return v2_sub(a, v2_mul(b, c)); }
static inline np2_v2 v2_set1(double x) { return _mm_set1_pd(x); }
#elif defined(BRUUN_NP2_NEON_128)
typedef float64x2_t np2_v2;
static inline np2_v2 v2_load(const double* p) { return vld1q_f64(p); }
static inline void v2_store(double* p, np2_v2 a) { vst1q_f64(p, a); }
static inline np2_v2 v2_add(np2_v2 a, np2_v2 b) { return vaddq_f64(a, b); }
static inline np2_v2 v2_sub(np2_v2 a, np2_v2 b) { return vsubq_f64(a, b); }
static inline np2_v2 v2_mul(np2_v2 a, np2_v2 b) { return vmulq_f64(a, b); }
static inline np2_v2 v2_madd(np2_v2 a, np2_v2 b, np2_v2 c) { return vfmaq_f64(a, b, c); }
static inline np2_v2 v2_msub(np2_v2 a, np2_v2 b, np2_v2 c) { return vfmsq_f64(a, b, c); }
static inline np2_v2 v2_set1(double x) { return vdupq_n_f64(x); }
#endif

struct QuadraticBlock {
    double* a0;
    double* b0;
    double* a1;
    double* b1;
    const double* c;
    const double* s;
    int length;
};

static inline void quadratic_fwd_variable(const QuadraticBlock& block);
static inline void quadratic_inv_variable(const QuadraticBlock& block);

struct TwoLevelBlock {
    double* a0;
    double* b0;
    double* a1;
    double* b1;
    const double* parent_c;
    const double* parent_s;
    const double* left_c;
    const double* left_s;
    const double* right_c;
    const double* right_s;
    int half_length;
};


// Descriptor for one non-power-of-two quadratic run inside shared SoA arenas.
struct QuadraticDescriptor {
    int offset;
    int coefficient_offset;
    int length;
};

// Small descriptor scheduler for batching variable-size quadratic runs.
class QuadraticScheduler {
public:
    // Remove all scheduled runs.
    void clear() {
        descriptors.clear();
        total_lane_count = 0;
    }

    // Add one run over shared data and coefficient arenas.
    void add_block(int offset, int coefficient_offset, int length) {
        if (offset < 0 || coefficient_offset < 0 || length < 0) {
            throw std::invalid_argument("negative non-power-of-two scheduler descriptor");
        }
        if (length == 0) {
            return;
        }
        QuadraticDescriptor descriptor;
        descriptor.offset = offset;
        descriptor.coefficient_offset = coefficient_offset;
        descriptor.length = length;
        descriptors.push_back(descriptor);
        total_lane_count += static_cast<std::size_t>(length);
    }

    // Stable-sort longer runs first so SIMD loops amortize descriptor overhead.
    void sort_for_simd() {
        std::stable_sort(descriptors.begin(), descriptors.end(), [](const QuadraticDescriptor& left,
                                                                     const QuadraticDescriptor& right) {
            if (left.length > right.length) {
                return true;
            }
            return false;
        });
    }

    std::size_t lane_count() const {
        return total_lane_count;
    }

    std::size_t block_count() const {
        return descriptors.size();
    }

    const std::vector<QuadraticDescriptor>& blocks() const {
        return descriptors;
    }

    // Execute all scheduled forward descriptors against shared SoA arenas.
    void execute(double* a0, double* b0, double* a1, double* b1,
                 const double* c, const double* s) const {
        if (total_lane_count == 0) {
            return;
        }
        if (!a0 || !b0 || !a1 || !b1 || !c || !s) {
            throw std::invalid_argument("null pointer in non-power-of-two scheduler execute");
        }
        for (const QuadraticDescriptor& descriptor : descriptors) {
            QuadraticBlock block;
            block.a0 = a0 + descriptor.offset;
            block.b0 = b0 + descriptor.offset;
            block.a1 = a1 + descriptor.offset;
            block.b1 = b1 + descriptor.offset;
            block.c = c + descriptor.coefficient_offset;
            block.s = s + descriptor.coefficient_offset;
            block.length = descriptor.length;
            quadratic_fwd_variable(block);
        }
    }

    // Execute all scheduled inverse descriptors against shared SoA arenas.
    void execute_inverse(double* a0, double* b0, double* a1, double* b1,
                         const double* c, const double* s) const {
        if (total_lane_count == 0) {
            return;
        }
        if (!a0 || !b0 || !a1 || !b1 || !c || !s) {
            throw std::invalid_argument("null pointer in non-power-of-two scheduler inverse execute");
        }
        for (const QuadraticDescriptor& descriptor : descriptors) {
            QuadraticBlock block;
            block.a0 = a0 + descriptor.offset;
            block.b0 = b0 + descriptor.offset;
            block.a1 = a1 + descriptor.offset;
            block.b1 = b1 + descriptor.offset;
            block.c = c + descriptor.coefficient_offset;
            block.s = s + descriptor.coefficient_offset;
            block.length = descriptor.length;
            quadratic_inv_variable(block);
        }
    }

private:
    std::vector<QuadraticDescriptor> descriptors;
    std::size_t total_lane_count = 0;
};

static inline const char* simd_backend_name() {
#if BRUUN_NP2_LEVEL >= 2
    return "avx2-fma-256";
#elif defined(BRUUN_NP2_NEON_128)
    return "neon-fma-128";
#elif defined(BRUUN_NP2_X86_128)
    return "sse2-128";
#else
    return "scalar";
#endif
}

static inline void validate_quadratic_block(const QuadraticBlock& block) {
    if (block.length < 0) throw std::invalid_argument("negative non-power-of-two block length");
    if (block.length == 0) return;
    if (!block.a0 || !block.b0 || !block.a1 || !block.b1 || !block.c || !block.s) {
        throw std::invalid_argument("null pointer in non-power-of-two quadratic block");
    }
}

// Apply one normalized real-quadratic CRT split over a variable-size block.
// The coefficient arrays are streams rather than scalars because mixed-radix and
// odd-size Bruun factors do not generally share one angle for every lane.
static inline void quadratic_fwd_variable(const QuadraticBlock& block) {
    validate_quadratic_block(block);

    double* BRUUN_NP2_RESTRICT a0p = block.a0;
    double* BRUUN_NP2_RESTRICT b0p = block.b0;
    double* BRUUN_NP2_RESTRICT a1p = block.a1;
    double* BRUUN_NP2_RESTRICT b1p = block.b1;
    const double* BRUUN_NP2_RESTRICT cp = block.c;
    const double* BRUUN_NP2_RESTRICT sp = block.s;

    int n = 0;
#if BRUUN_NP2_LEVEL >= 2
    for (; n + 3 < block.length; n += 4) {
        const __m256d A0 = _mm256_loadu_pd(a0p + n);
        const __m256d B0 = _mm256_loadu_pd(b0p + n);
        const __m256d A1 = _mm256_loadu_pd(a1p + n);
        const __m256d B1 = _mm256_loadu_pd(b1p + n);
        const __m256d C = _mm256_loadu_pd(cp + n);
        const __m256d S = _mm256_loadu_pd(sp + n);

        const __m256d R = _mm256_fmsub_pd(C, B0, _mm256_mul_pd(S, B1));
        const __m256d I = _mm256_fmadd_pd(S, B0, _mm256_mul_pd(C, B1));

        _mm256_storeu_pd(a0p + n, _mm256_add_pd(A0, R));
        _mm256_storeu_pd(b0p + n, _mm256_add_pd(A1, I));
        _mm256_storeu_pd(a1p + n, _mm256_sub_pd(A0, R));
        _mm256_storeu_pd(b1p + n, _mm256_sub_pd(I, A1));
    }
#elif BRUUN_NP2_LEVEL >= 1
    for (; n + 1 < block.length; n += 2) {
        const np2_v2 A0 = v2_load(a0p + n);
        const np2_v2 B0 = v2_load(b0p + n);
        const np2_v2 A1 = v2_load(a1p + n);
        const np2_v2 B1 = v2_load(b1p + n);
        const np2_v2 C = v2_load(cp + n);
        const np2_v2 S = v2_load(sp + n);

        const np2_v2 R = v2_msub(v2_mul(C, B0), S, B1);
        const np2_v2 I = v2_madd(v2_mul(S, B0), C, B1);

        v2_store(a0p + n, v2_add(A0, R));
        v2_store(b0p + n, v2_add(A1, I));
        v2_store(a1p + n, v2_sub(A0, R));
        v2_store(b1p + n, v2_sub(I, A1));
    }
#endif
    for (; n < block.length; ++n) {
        const double A0 = a0p[n];
        const double B0 = b0p[n];
        const double A1 = a1p[n];
        const double B1 = b1p[n];
        const double C = cp[n];
        const double S = sp[n];

        const double R = C * B0 - S * B1;
        const double I = S * B0 + C * B1;

        a0p[n] = A0 + R;
        b0p[n] = A1 + I;
        a1p[n] = A0 - R;
        b1p[n] = I - A1;
    }
}


// Invert one normalized real-quadratic CRT split over a variable-size block.
// The inverse recovers [A0|B0|A1|B1] from the forward output using the transpose
// of the unit rotation. This keeps the same coefficient streams and exact scalar
// tail behavior as the forward path.
static inline void quadratic_inv_variable(const QuadraticBlock& block) {
    validate_quadratic_block(block);

    double* BRUUN_NP2_RESTRICT a0p = block.a0;
    double* BRUUN_NP2_RESTRICT b0p = block.b0;
    double* BRUUN_NP2_RESTRICT a1p = block.a1;
    double* BRUUN_NP2_RESTRICT b1p = block.b1;
    const double* BRUUN_NP2_RESTRICT cp = block.c;
    const double* BRUUN_NP2_RESTRICT sp = block.s;

    int n = 0;
#if BRUUN_NP2_LEVEL >= 2
    const __m256d half = _mm256_set1_pd(0.5);
    for (; n + 3 < block.length; n += 4) {
        const __m256d U0 = _mm256_loadu_pd(a0p + n);
        const __m256d W0 = _mm256_loadu_pd(b0p + n);
        const __m256d V0 = _mm256_loadu_pd(a1p + n);
        const __m256d X0 = _mm256_loadu_pd(b1p + n);
        const __m256d C = _mm256_loadu_pd(cp + n);
        const __m256d S = _mm256_loadu_pd(sp + n);

        const __m256d A0 = _mm256_mul_pd(half, _mm256_add_pd(U0, V0));
        const __m256d R = _mm256_mul_pd(half, _mm256_sub_pd(U0, V0));
        const __m256d I = _mm256_mul_pd(half, _mm256_add_pd(W0, X0));
        const __m256d A1 = _mm256_mul_pd(half, _mm256_sub_pd(W0, X0));

        const __m256d B0 = _mm256_fmadd_pd(S, I, _mm256_mul_pd(C, R));
        const __m256d B1 = _mm256_fmsub_pd(C, I, _mm256_mul_pd(S, R));

        _mm256_storeu_pd(a0p + n, A0);
        _mm256_storeu_pd(b0p + n, B0);
        _mm256_storeu_pd(a1p + n, A1);
        _mm256_storeu_pd(b1p + n, B1);
    }
#elif BRUUN_NP2_LEVEL >= 1
    const np2_v2 half = v2_set1(0.5);
    for (; n + 1 < block.length; n += 2) {
        const np2_v2 U0 = v2_load(a0p + n);
        const np2_v2 W0 = v2_load(b0p + n);
        const np2_v2 V0 = v2_load(a1p + n);
        const np2_v2 X0 = v2_load(b1p + n);
        const np2_v2 C = v2_load(cp + n);
        const np2_v2 S = v2_load(sp + n);

        const np2_v2 A0 = v2_mul(half, v2_add(U0, V0));
        const np2_v2 R = v2_mul(half, v2_sub(U0, V0));
        const np2_v2 I = v2_mul(half, v2_add(W0, X0));
        const np2_v2 A1 = v2_mul(half, v2_sub(W0, X0));

        const np2_v2 B0 = v2_madd(v2_mul(C, R), S, I);
        const np2_v2 B1 = v2_msub(v2_mul(C, I), S, R);

        v2_store(a0p + n, A0);
        v2_store(b0p + n, B0);
        v2_store(a1p + n, A1);
        v2_store(b1p + n, B1);
    }
#endif
    for (; n < block.length; ++n) {
        const double U0 = a0p[n];
        const double W0 = b0p[n];
        const double V0 = a1p[n];
        const double X0 = b1p[n];
        const double C = cp[n];
        const double S = sp[n];

        const double A0 = 0.5 * (U0 + V0);
        const double R = 0.5 * (U0 - V0);
        const double I = 0.5 * (W0 + X0);
        const double A1 = 0.5 * (W0 - X0);
        const double B0 = C * R + S * I;
        const double B1 = C * I - S * R;

        a0p[n] = A0;
        b0p[n] = B0;
        a1p[n] = A1;
        b1p[n] = B1;
    }
}

static inline void validate_two_level_block(const TwoLevelBlock& block) {
    if (block.half_length < 0) throw std::invalid_argument("negative two-level block length");
    if (block.half_length == 0) return;
    if (!block.a0 || !block.b0 || !block.a1 || !block.b1 ||
        !block.parent_c || !block.parent_s || !block.left_c || !block.left_s ||
        !block.right_c || !block.right_s) {
        throw std::invalid_argument("null pointer in non-power-of-two fused block");
    }
}

// Fuse a parent split with two child splits while the intermediate values are in
// registers. This is the non-power-of-two analogue of the power-of-two norm2
// kernel, but child angles are streamed so uneven CRT branches can share the
// same FMA-optimal lowering without requiring power-of-two structure.
static inline void quadratic_two_level_fused_variable(const TwoLevelBlock& block) {
    validate_two_level_block(block);

    double* BRUUN_NP2_RESTRICT A0 = block.a0;
    double* BRUUN_NP2_RESTRICT B0 = block.b0;
    double* BRUUN_NP2_RESTRICT A1 = block.a1;
    double* BRUUN_NP2_RESTRICT B1 = block.b1;
    const int h = block.half_length;

    int n = 0;
#if BRUUN_NP2_LEVEL >= 2
    for (; n + 3 < h; n += 4) {
        const __m256d a0n = _mm256_loadu_pd(A0 + n);
        const __m256d a0h = _mm256_loadu_pd(A0 + h + n);
        const __m256d b0n = _mm256_loadu_pd(B0 + n);
        const __m256d b0h = _mm256_loadu_pd(B0 + h + n);
        const __m256d a1n = _mm256_loadu_pd(A1 + n);
        const __m256d a1h = _mm256_loadu_pd(A1 + h + n);
        const __m256d b1n = _mm256_loadu_pd(B1 + n);
        const __m256d b1h = _mm256_loadu_pd(B1 + h + n);

        const __m256d pc0 = _mm256_loadu_pd(block.parent_c + n);
        const __m256d ps0 = _mm256_loadu_pd(block.parent_s + n);
        const __m256d pc1 = _mm256_loadu_pd(block.parent_c + h + n);
        const __m256d ps1 = _mm256_loadu_pd(block.parent_s + h + n);

        const __m256d Rn = _mm256_fmsub_pd(pc0, b0n, _mm256_mul_pd(ps0, b1n));
        const __m256d In = _mm256_fmadd_pd(ps0, b0n, _mm256_mul_pd(pc0, b1n));
        const __m256d Rh = _mm256_fmsub_pd(pc1, b0h, _mm256_mul_pd(ps1, b1h));
        const __m256d Ih = _mm256_fmadd_pd(ps1, b0h, _mm256_mul_pd(pc1, b1h));

        const __m256d u0 = _mm256_add_pd(a0n, Rn);
        const __m256d uh = _mm256_add_pd(a0h, Rh);
        const __m256d w0 = _mm256_add_pd(a1n, In);
        const __m256d wh = _mm256_add_pd(a1h, Ih);
        const __m256d v0 = _mm256_sub_pd(a0n, Rn);
        const __m256d vh = _mm256_sub_pd(a0h, Rh);
        const __m256d x0 = _mm256_sub_pd(In, a1n);
        const __m256d xh = _mm256_sub_pd(Ih, a1h);

        const __m256d lc = _mm256_loadu_pd(block.left_c + n);
        const __m256d ls = _mm256_loadu_pd(block.left_s + n);
        const __m256d rc = _mm256_loadu_pd(block.right_c + n);
        const __m256d rs = _mm256_loadu_pd(block.right_s + n);

        const __m256d R0 = _mm256_fmsub_pd(lc, uh, _mm256_mul_pd(ls, wh));
        const __m256d I0 = _mm256_fmadd_pd(ls, uh, _mm256_mul_pd(lc, wh));
        const __m256d R1 = _mm256_fmsub_pd(rc, vh, _mm256_mul_pd(rs, xh));
        const __m256d I1 = _mm256_fmadd_pd(rs, vh, _mm256_mul_pd(rc, xh));

        _mm256_storeu_pd(A0 + n, _mm256_add_pd(u0, R0));
        _mm256_storeu_pd(A0 + h + n, _mm256_add_pd(w0, I0));
        _mm256_storeu_pd(B0 + n, _mm256_sub_pd(u0, R0));
        _mm256_storeu_pd(B0 + h + n, _mm256_sub_pd(I0, w0));
        _mm256_storeu_pd(A1 + n, _mm256_add_pd(v0, R1));
        _mm256_storeu_pd(A1 + h + n, _mm256_add_pd(x0, I1));
        _mm256_storeu_pd(B1 + n, _mm256_sub_pd(v0, R1));
        _mm256_storeu_pd(B1 + h + n, _mm256_sub_pd(I1, x0));
    }
#elif BRUUN_NP2_LEVEL >= 1
    for (; n + 1 < h; n += 2) {
        const np2_v2 a0n = v2_load(A0 + n);
        const np2_v2 a0h = v2_load(A0 + h + n);
        const np2_v2 b0n = v2_load(B0 + n);
        const np2_v2 b0h = v2_load(B0 + h + n);
        const np2_v2 a1n = v2_load(A1 + n);
        const np2_v2 a1h = v2_load(A1 + h + n);
        const np2_v2 b1n = v2_load(B1 + n);
        const np2_v2 b1h = v2_load(B1 + h + n);

        const np2_v2 pc0 = v2_load(block.parent_c + n);
        const np2_v2 ps0 = v2_load(block.parent_s + n);
        const np2_v2 pc1 = v2_load(block.parent_c + h + n);
        const np2_v2 ps1 = v2_load(block.parent_s + h + n);

        const np2_v2 Rn = v2_msub(v2_mul(pc0, b0n), ps0, b1n);
        const np2_v2 In = v2_madd(v2_mul(ps0, b0n), pc0, b1n);
        const np2_v2 Rh = v2_msub(v2_mul(pc1, b0h), ps1, b1h);
        const np2_v2 Ih = v2_madd(v2_mul(ps1, b0h), pc1, b1h);

        const np2_v2 u0 = v2_add(a0n, Rn);
        const np2_v2 uh = v2_add(a0h, Rh);
        const np2_v2 w0 = v2_add(a1n, In);
        const np2_v2 wh = v2_add(a1h, Ih);
        const np2_v2 v0 = v2_sub(a0n, Rn);
        const np2_v2 vh = v2_sub(a0h, Rh);
        const np2_v2 x0 = v2_sub(In, a1n);
        const np2_v2 xh = v2_sub(Ih, a1h);

        const np2_v2 lc = v2_load(block.left_c + n);
        const np2_v2 ls = v2_load(block.left_s + n);
        const np2_v2 rc = v2_load(block.right_c + n);
        const np2_v2 rs = v2_load(block.right_s + n);

        const np2_v2 R0 = v2_msub(v2_mul(lc, uh), ls, wh);
        const np2_v2 I0 = v2_madd(v2_mul(ls, uh), lc, wh);
        const np2_v2 R1 = v2_msub(v2_mul(rc, vh), rs, xh);
        const np2_v2 I1 = v2_madd(v2_mul(rs, vh), rc, xh);

        v2_store(A0 + n, v2_add(u0, R0));
        v2_store(A0 + h + n, v2_add(w0, I0));
        v2_store(B0 + n, v2_sub(u0, R0));
        v2_store(B0 + h + n, v2_sub(I0, w0));
        v2_store(A1 + n, v2_add(v0, R1));
        v2_store(A1 + h + n, v2_add(x0, I1));
        v2_store(B1 + n, v2_sub(v0, R1));
        v2_store(B1 + h + n, v2_sub(I1, x0));
    }
#endif
    for (; n < h; ++n) {
        const double a0n = A0[n];
        const double a0h = A0[h + n];
        const double b0n = B0[n];
        const double b0h = B0[h + n];
        const double a1n = A1[n];
        const double a1h = A1[h + n];
        const double b1n = B1[n];
        const double b1h = B1[h + n];

        const double Rn = block.parent_c[n] * b0n - block.parent_s[n] * b1n;
        const double In = block.parent_s[n] * b0n + block.parent_c[n] * b1n;
        const double Rh = block.parent_c[h + n] * b0h - block.parent_s[h + n] * b1h;
        const double Ih = block.parent_s[h + n] * b0h + block.parent_c[h + n] * b1h;

        const double u0 = a0n + Rn;
        const double uh = a0h + Rh;
        const double w0 = a1n + In;
        const double wh = a1h + Ih;
        const double v0 = a0n - Rn;
        const double vh = a0h - Rh;
        const double x0 = In - a1n;
        const double xh = Ih - a1h;

        const double R0 = block.left_c[n] * uh - block.left_s[n] * wh;
        const double I0 = block.left_s[n] * uh + block.left_c[n] * wh;
        const double R1 = block.right_c[n] * vh - block.right_s[n] * xh;
        const double I1 = block.right_s[n] * vh + block.right_c[n] * xh;

        A0[n] = u0 + R0;
        A0[h + n] = w0 + I0;
        B0[n] = u0 - R0;
        B0[h + n] = I0 - w0;
        A1[n] = v0 + R1;
        A1[h + n] = x0 + I1;
        B1[n] = v0 - R1;
        B1[h + n] = I1 - x0;
    }
}

}  // namespace bruun_nonpower2
