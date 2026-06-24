// bruun_nonpower2_simd.cpp
//
// SIMD building blocks for future non-power-of-two Bruun/CRT structures.
// This file is intentionally separate from the power-of-two Bruun kernels and
// does not include or modify fastplan or the power-of-two implementation.
//
// Layout goal: keep non-power-of-two residue work in a structure-of-arrays form
// that mirrors the normalized Bruun kernel: contiguous [A0 | B0 | A1 | B1]
// lanes, coefficient streams beside the data, and multiply-add pairs expressed
// as FMA-friendly real rotations.
//
// Build check:
//   g++ -O3 -std=c++17 -DBRUUN_NONPOWER2_SIMD_TEST bruun_nonpower2_simd.cpp -lm -o bruun_nonpower2_simd_test
// Optional AVX2/FMA:
//   g++ -O3 -std=c++17 -mavx2 -mfma -DBRUUN_NP2_SIMD_AVX2 -DBRUUN_NONPOWER2_SIMD_TEST bruun_nonpower2_simd.cpp -lm -o bruun_nonpower2_simd_test

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <vector>

#if defined(BRUUN_NP2_SIMD_AVX2)
#  if !defined(__AVX2__) || !defined(__FMA__)
#    error "BRUUN_NP2_SIMD_AVX2 requires AVX2 and FMA"
#  endif
#  include <immintrin.h>
#  define BRUUN_NP2_LEVEL 2
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

#ifdef BRUUN_NONPOWER2_SIMD_TEST
static double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
    double out = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        out = std::max(out, std::abs(a[i] - b[i]));
    }
    return out;
}

int main() {
    const int n = 19;
    std::vector<double> a0(n), b0(n), a1(n), b1(n), c(n), s(n);
    std::vector<double> ref_a0(n), ref_b0(n), ref_a1(n), ref_b1(n);
    std::vector<double> orig_a0(n), orig_b0(n), orig_a1(n), orig_b1(n);
    for (int i = 0; i < n; ++i) {
        a0[i] = 0.25 + 0.07 * i;
        b0[i] = -0.5 + 0.03 * i;
        a1[i] = 0.75 - 0.02 * i;
        b1[i] = -0.125 + 0.05 * i;
        const double angle = 0.11 * (i + 1);
        c[i] = std::cos(angle);
        s[i] = std::sin(angle);
    }
    ref_a0 = a0;
    ref_b0 = b0;
    ref_a1 = a1;
    ref_b1 = b1;
    orig_a0 = a0;
    orig_b0 = b0;
    orig_a1 = a1;
    orig_b1 = b1;

    bruun_nonpower2::QuadraticBlock block{a0.data(), b0.data(), a1.data(), b1.data(), c.data(), s.data(), n};
    bruun_nonpower2::quadratic_fwd_variable(block);

    for (int i = 0; i < n; ++i) {
        const double old_a0 = 0.25 + 0.07 * i;
        const double old_a1 = 0.75 - 0.02 * i;
        const double old_b0 = -0.5 + 0.03 * i;
        const double old_b1 = -0.125 + 0.05 * i;
        const double R = c[i] * old_b0 - s[i] * old_b1;
        const double I = s[i] * old_b0 + c[i] * old_b1;
        ref_a0[i] = old_a0 + R;
        ref_b0[i] = old_a1 + I;
        ref_a1[i] = old_a0 - R;
        ref_b1[i] = I - old_a1;
    }

    assert(max_abs_diff(a0, ref_a0) < 1e-12);
    assert(max_abs_diff(b0, ref_b0) < 1e-12);
    assert(max_abs_diff(a1, ref_a1) < 1e-12);
    assert(max_abs_diff(b1, ref_b1) < 1e-12);

    bruun_nonpower2::quadratic_inv_variable(block);
    assert(max_abs_diff(a0, orig_a0) < 1e-12);
    assert(max_abs_diff(b0, orig_b0) < 1e-12);
    assert(max_abs_diff(a1, orig_a1) < 1e-12);
    assert(max_abs_diff(b1, orig_b1) < 1e-12);

    const int block_count = 384;
    std::vector<int> lengths(block_count);
    int total_lanes = 0;
    for (int block_index = 0; block_index < block_count; ++block_index) {
        const int lane_mod = (block_index * 17 + 5) % 29;
        lengths[block_index] = 5 + lane_mod;
        total_lanes += lengths[block_index];
    }

    std::vector<double> bench_a0(total_lanes), bench_b0(total_lanes);
    std::vector<double> bench_a1(total_lanes), bench_b1(total_lanes);
    std::vector<double> bench_c(total_lanes), bench_s(total_lanes);
    for (int i = 0; i < total_lanes; ++i) {
        bench_a0[i] = 0.001 * (i % 97);
        bench_b0[i] = -0.002 * (i % 89);
        bench_a1[i] = 0.003 * (i % 83);
        bench_b1[i] = -0.004 * (i % 79);
        const double theta = 0.0007 * (i + 3);
        bench_c[i] = std::cos(theta);
        bench_s[i] = std::sin(theta);
    }

    bruun_nonpower2::QuadraticScheduler scheduler;
    int offset = 0;
    for (int length : lengths) {
        scheduler.add_block(offset, offset, length);
        offset += length;
    }
    scheduler.sort_for_simd();
    assert(scheduler.block_count() == static_cast<std::size_t>(block_count));
    assert(scheduler.lane_count() == static_cast<std::size_t>(total_lanes));

    std::vector<double> sched_orig_a0 = bench_a0;
    std::vector<double> sched_orig_b0 = bench_b0;
    std::vector<double> sched_orig_a1 = bench_a1;
    std::vector<double> sched_orig_b1 = bench_b1;
    scheduler.execute(bench_a0.data(), bench_b0.data(), bench_a1.data(), bench_b1.data(),
                      bench_c.data(), bench_s.data());
    scheduler.execute_inverse(bench_a0.data(), bench_b0.data(), bench_a1.data(), bench_b1.data(),
                              bench_c.data(), bench_s.data());
    assert(max_abs_diff(bench_a0, sched_orig_a0) < 1e-12);
    assert(max_abs_diff(bench_b0, sched_orig_b0) < 1e-12);
    assert(max_abs_diff(bench_a1, sched_orig_a1) < 1e-12);
    assert(max_abs_diff(bench_b1, sched_orig_b1) < 1e-12);

    const int repeats = 1200;
    const auto start = std::chrono::steady_clock::now();
    for (int repeat = 0; repeat < repeats; ++repeat) {
        scheduler.execute(bench_a0.data(), bench_b0.data(), bench_a1.data(), bench_b1.data(),
                          bench_c.data(), bench_s.data());
    }
    const auto stop = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = stop - start;
    const double lane_visits = static_cast<double>(total_lanes) * static_cast<double>(repeats);
    const double ns_per_lane = elapsed.count() * 1000000000.0 / lane_visits;

    std::printf("bruun_nonpower2_simd backend=%s ok blocks=%zu lanes=%zu ns_per_lane=%.3f\n",
                bruun_nonpower2::simd_backend_name(), scheduler.block_count(),
                scheduler.lane_count(), ns_per_lane);
    return 0;
}
#endif
