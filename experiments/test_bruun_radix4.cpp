// Experimental validation for the Bruun radix-4 kernel.
//
// Tests:
//   1. Breadth-first forward_residues matches BFFT oracle
//   2. Depth-first radix-4 forward_residues_radix4 matches BFFT oracle
//   3. norm2_fused matches sequential (parent + 2 child) norm_q splits
//   4. Chebyshev 14q split on synthetic polynomial data
//
// Build:
//   make && c++ -O2 -std=c++17 -Iinclude experiments/test_bruun_radix4.cpp \
//       build/libbfft.a -lm -o build/experiments/test_bruun_radix4
//
// Run:
//   ./build/experiments/test_bruun_radix4

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

#include "bfft/bfft.hpp"
#include "../src/detail/bruun_radix4_kernel.hpp"

static double max_abs_diff(const double* a, const double* b, int n) {
    double mx = 0.0;
    for (int i = 0; i < n; ++i) {
        double e = std::abs(a[i] - b[i]);
        if (e > mx) mx = e;
    }
    return mx;
}

static void fill_random(double* buf, int n, unsigned seed) {
    for (int i = 0; i < n; ++i) {
        seed = seed * 1103515245u + 12345u;
        buf[i] = (static_cast<double>(seed & 0x7fffffff) / 0x7fffffff) * 2.0 - 1.0;
    }
}

// -----------------------------------------------------------------------
// Test 1 & 2: End-to-end forward_residues against BFFT oracle
// -----------------------------------------------------------------------
static bool test_forward_residues(int N) {
    bfft::plan oracle(N);

    bruun_radix4::RFFT_Radix4 r4;
    if (!r4.init(N)) {
        std::printf("  N=%5d: RFFT_Radix4::init failed\n", N);
        return false;
    }

    double* input    = static_cast<double*>(std::calloc(N, sizeof(double)));
    double* ref_res  = static_cast<double*>(std::calloc(N, sizeof(double)));
    double* bf_res   = static_cast<double*>(std::calloc(N, sizeof(double)));
    double* r4_res   = static_cast<double*>(std::calloc(N, sizeof(double)));

    fill_random(input, N, 42u + N);

    oracle.forward_residues(input, ref_res);
    r4.forward_residues(input, bf_res);
    r4.forward_residues_radix4(input, r4_res);

    double err_bf = max_abs_diff(ref_res, bf_res, N);
    double err_r4 = max_abs_diff(ref_res, r4_res, N);

    bool ok_bf = err_bf < 1e-10;
    bool ok_r4 = err_r4 < 1e-10;

    std::printf("  N=%5d: breadth-first err=%.2e %s | radix4-df err=%.2e %s\n",
                N, err_bf, ok_bf ? "OK" : "FAIL",
                err_r4, ok_r4 ? "OK" : "FAIL");

    std::free(input);
    std::free(ref_res);
    std::free(bf_res);
    std::free(r4_res);

    return ok_bf && ok_r4;
}

// -----------------------------------------------------------------------
// Test 3: norm2_fused equivalence with sequential norm_q
// -----------------------------------------------------------------------
static bool test_norm2_fused_equivalence(int q) {
    const int total = 4 * q;
    double* block_fused = static_cast<double*>(std::malloc(sizeof(double) * total));
    double* block_seq   = static_cast<double*>(std::malloc(sizeof(double) * total));

    fill_random(block_fused, total, 7777u + q);
    std::memcpy(block_seq, block_fused, sizeof(double) * total);

    double c = std::cos(0.3), s = std::sin(0.3);
    double c0 = std::cos(0.15), s0 = std::sin(0.15);
    double c1 = std::cos(0.62), s1 = std::sin(0.62);

    bruun_radix4::norm2_fused_standalone(block_fused, q, c, s, c0, s0, c1, s1);

    // Sequential: parent norm_q, then 2 child norm_q on each half
    bruun_radix4::norm_q_standalone(block_seq, q, c, s);

    // After parent split, child 0 occupies [0..2q-1], child 1 occupies [2q..4q-1].
    // Child 0 has A0=[0..q/2-1], B0=[q/2..q-1], A1=[q..q+q/2-1], B1=[q+q/2..2q-1].
    // But that's the interleaved layout after the parent split — child 0's block is
    // the low halves of A0 and A1, and the low halves of B0 and B1 from the parent output.
    //
    // Actually, after the parent norm_q on 4q elements:
    //   child 0 = [A0_out(q), B0_out(q)]  at positions [0..q-1, q..2q-1]
    //   child 1 = [A1_out(q), B1_out(q)]  at positions [2q..3q-1, 3q..4q-1]
    //
    // Each child block is 2q elements. For norm_q with q/2:
    //   child 0 block: [A0_out[0..q/2-1], A0_out[q/2..q-1], B0_out[0..q/2-1], B0_out[q/2..q-1]]
    //   = 4 * (q/2) layout
    int cq = q / 2;
    bruun_radix4::norm_q_standalone(block_seq, cq, c0, s0);
    bruun_radix4::norm_q_standalone(block_seq + 2 * q, cq, c1, s1);

    double err = max_abs_diff(block_fused, block_seq, total);
    bool ok = err < 1e-12;
    std::printf("  q=%3d: fused vs sequential err=%.2e %s\n", q, err, ok ? "OK" : "FAIL");

    std::free(block_fused);
    std::free(block_seq);
    return ok;
}

// -----------------------------------------------------------------------
// Test 4: Chebyshev radix-4 split on a synthetic degree-3 polynomial
// -----------------------------------------------------------------------
static bool test_chebyshev_split() {
    const int q = 4;
    double a0[4] = {1.0, 0.5, -0.3, 0.7};
    double a1[4] = {0.2, -0.1, 0.4, 0.0};
    double a2[4] = {-0.5, 0.3, 0.1, -0.2};
    double a3[4] = {0.1, -0.4, 0.2, 0.6};

    double u = std::cos(0.3), u2 = u * u;
    double v = std::sin(0.3), v2 = v * v;

    double pu[4], mu[4], pv[4], mv[4];
    bruun_radix4::chebyshev_radix4_split(a0, a1, a2, a3, pu, mu, pv, mv,
                                          q, u, u2, v, v2);

    double max_err = 0.0;
    for (int n = 0; n < q; ++n) {
        double fu  = a0[n] + a1[n]*u  + a2[n]*u*u  + a3[n]*u*u*u;
        double fmu = a0[n] + a1[n]*(-u) + a2[n]*u*u + a3[n]*(-u)*u*u;
        double fv  = a0[n] + a1[n]*v  + a2[n]*v*v  + a3[n]*v*v*v;
        double fmv = a0[n] + a1[n]*(-v) + a2[n]*v*v + a3[n]*(-v)*v*v;

        max_err = std::max(max_err, std::abs(pu[n] - fu));
        max_err = std::max(max_err, std::abs(mu[n] - fmu));
        max_err = std::max(max_err, std::abs(pv[n] - fv));
        max_err = std::max(max_err, std::abs(mv[n] - fmv));
    }

    bool ok = max_err < 1e-14;
    std::printf("  Chebyshev split vs direct eval: err=%.2e %s\n", max_err, ok ? "OK" : "FAIL");
    return ok;
}

// -----------------------------------------------------------------------
// Test 5: Round-trip (forward_residues then inverse_residues) via BFFT
//         confirms residues are in the correct layout for reconstruction.
// -----------------------------------------------------------------------
static bool test_roundtrip(int N) {
    bfft::plan oracle(N);

    bruun_radix4::RFFT_Radix4 r4;
    r4.init(N);

    double* input   = static_cast<double*>(std::calloc(N, sizeof(double)));
    double* r4_res  = static_cast<double*>(std::calloc(N, sizeof(double)));

    fill_random(input, N, 12345u + N);

    r4.forward_residues(input, r4_res);
    oracle.inverse_residues(r4_res);

    double scale = 1.0 / N;
    double err = 0.0;
    for (int i = 0; i < N; ++i) {
        double e = std::abs(input[i] - r4_res[i] * scale);
        if (e > err) err = e;
    }

    bool ok = err < 1e-10;
    std::printf("  N=%5d: roundtrip err=%.2e %s\n", N, err, ok ? "OK" : "FAIL");

    std::free(input);
    std::free(r4_res);
    return ok;
}

// -----------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------
int main() {
    int pass = 0, fail = 0;

    std::printf("=== Test 1+2: forward_residues vs BFFT oracle ===\n");
    for (int N : {16, 32, 64, 128, 256, 512, 1024, 4096}) {
        if (test_forward_residues(N)) ++pass; else ++fail;
    }

    std::printf("\n=== Test 3: norm2_fused vs sequential norm_q ===\n");
    for (int q : {4, 8, 16, 32, 64}) {
        if (test_norm2_fused_equivalence(q)) ++pass; else ++fail;
    }

    std::printf("\n=== Test 4: Chebyshev radix-4 split correctness ===\n");
    if (test_chebyshev_split()) ++pass; else ++fail;

    std::printf("\n=== Test 5: Round-trip (radix4 fwd + BFFT inv) ===\n");
    for (int N : {16, 64, 256, 1024}) {
        if (test_roundtrip(N)) ++pass; else ++fail;
    }

    std::printf("\n%d passed, %d failed\n", pass, fail);
    return fail > 0 ? 1 : 0;
}
