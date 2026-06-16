// Validation and microbenchmark for the composed-Chebyshev radix-4 node.
//
// Tests:
//   1. composed_node == monomial_node == direct cubic evaluation (one node).
//   2. monomial <-> composed basis round-trip.
//   3. Full radix-4 recursion (monomial) leaves == direct tower evaluation,
//      and the leaf x-set == the roots of T_N.
//   4. Full recursion via composed_node == monomial recursion (in-tree proof).
//   5. Node throughput microbenchmark: normalized 20q vs monomial 14q vs
//      composed 12q, on flat arrays, to answer "is it faster".
//
// Build:
//   c++ -O2 -std=c++17 -Iinclude \
//       experiments/test_chebyshev_composed_radix4.cpp build/libbfft.a -lm \
//       -o build/experiments/test_chebyshev_composed_radix4

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "chebyshev_composed_radix4_kernel.hpp"
#include "../src/detail/bruun_radix4_kernel.hpp"  // norm2_fused_standalone (20q baseline)

namespace cc = cheb_composed_r4;

static double frand(unsigned& s) {
    s = s * 1103515245u + 12345u;
    return (static_cast<double>(s & 0x7fffffff) / 0x7fffffff) * 2.0 - 1.0;
}

static double max_abs_diff(const double* a, const double* b, int n) {
    double e = 0.0;
    for (int i = 0; i < n; ++i) e = std::max(e, std::abs(a[i] - b[i]));
    return e;
}

constexpr double pi = 3.141592653589793238462643383279502884;

// -----------------------------------------------------------------------
// Test 1: single node, three formulations agree.
// -----------------------------------------------------------------------
static bool test_node_equivalence(int q) {
    const int total = 4 * q;
    std::vector<double> mono(total), comp(total), direct(total);
    unsigned s = 99u + q;
    for (int i = 0; i < total; ++i) mono[i] = frand(s);
    comp = mono;

    const double a = 0.37;  // arbitrary node value, |a| < 1
    cc::NodeConst k = cc::node_const(a);

    // Direct cubic eval per lane: f(+u),f(-u),f(+v),f(-v).
    for (int n = 0; n < q; ++n) {
        const double a0 = mono[n], a1 = mono[q + n], a2 = mono[2 * q + n], a3 = mono[3 * q + n];
        auto f = [&](double y) { return a0 + a1 * y + a2 * y * y + a3 * y * y * y; };
        direct[n]         = f(k.u);
        direct[q + n]     = f(-k.u);
        direct[2 * q + n] = f(k.v);
        direct[3 * q + n] = f(-k.v);
    }

    std::vector<double> mfma = mono, mlow = mono;
    cc::monomial_node(mono.data(), q, k.u, k.v);
    cc::monomial_node_fma(mfma.data(), q, k.u, k.v);
    cc::lowmul_node(mlow.data(), q, k.u, k.v);

    // All composed variants share the same input layout and must agree.
    std::vector<double> c12 = comp, c13 = comp, c14 = comp, cfma = comp;
    cc::monomial_to_composed(c12.data(), q);
    cc::monomial_to_composed(c13.data(), q);
    cc::monomial_to_composed(c14.data(), q);
    cc::monomial_to_composed(cfma.data(), q);
    cc::composed_node(c12.data(), q, k.u, k.v, k.w);
    cc::composed_node_qdup(c13.data(), q, k.u, k.v, k.w);
    cc::composed_node_dup(c14.data(), q, k.u, k.v, k.w);
    cc::composed_node_fma(cfma.data(), q, k.u, k.v, k.w);

    double e = 0.0;
    e = std::max(e, max_abs_diff(mono.data(), direct.data(), total));
    e = std::max(e, max_abs_diff(mfma.data(), direct.data(), total));
    e = std::max(e, max_abs_diff(mlow.data(), direct.data(), total));
    e = std::max(e, max_abs_diff(c12.data(), direct.data(), total));
    e = std::max(e, max_abs_diff(c13.data(), direct.data(), total));
    e = std::max(e, max_abs_diff(c14.data(), direct.data(), total));
    e = std::max(e, max_abs_diff(cfma.data(), direct.data(), total));
    const bool ok = e < 1e-12;
    std::printf("  q=%4d: max err over {mono, mono_fma, lowmul, comp12/13/14, comp_fma} vs direct = %.1e  %s\n",
                q, e, ok ? "OK" : "FAIL");
    return ok;
}

// -----------------------------------------------------------------------
// Test 2: basis round-trip.
// -----------------------------------------------------------------------
static bool test_basis_roundtrip(int q) {
    const int total = 4 * q;
    std::vector<double> orig(total), x(total);
    unsigned s = 555u + q;
    for (int i = 0; i < total; ++i) orig[i] = frand(s);
    x = orig;
    cc::monomial_to_composed(x.data(), q);
    cc::composed_to_monomial(x.data(), q);
    const double e = max_abs_diff(x.data(), orig.data(), total);
    const bool ok = e < 1e-14;
    std::printf("  q=%4d: round-trip err=%.1e  %s\n", q, e, ok ? "OK" : "FAIL");
    return ok;
}

// -----------------------------------------------------------------------
// Test 3 & 4: full recursion correctness.
// -----------------------------------------------------------------------
static bool test_recursion(int N) {
    std::vector<double> C(N), leaves_m(N), leaves_c(N), xleaf(N);
    unsigned s = 31u + N;
    for (int i = 0; i < N; ++i) C[i] = frand(s);

    cc::leaf_x(0, N, 0.0, xleaf.data());

    leaves_m = C;
    cc::eval_monomial(leaves_m.data(), 0, N, 0.0);
    leaves_c = C;
    cc::eval_composed(leaves_c.data(), 0, N, 0.0);

    // Oracle: direct tower evaluation at each leaf x. The recursion and the
    // naive per-point tower evaluation are two different algorithms for the
    // same polynomial, so their gap reflects combined conditioning; report it
    // relative to the leaf magnitude.
    double e_direct = 0.0, max_leaf = 0.0;
    for (int i = 0; i < N; ++i) {
        const double d = cc::eval_direct(C.data(), 0, N, xleaf[i]);
        e_direct = std::max(e_direct, std::abs(leaves_m[i] - d));
        max_leaf = std::max(max_leaf, std::abs(leaves_m[i]));
    }
    const double rel_direct = e_direct / (max_leaf > 0.0 ? max_leaf : 1.0);
    const double e_comp_vs_mono = max_abs_diff(leaves_c.data(), leaves_m.data(), N);

    // Confirm the leaf x-set equals the roots of T_N.
    std::vector<double> got = xleaf, want(N);
    for (int k = 0; k < N; ++k) want[k] = std::cos((2.0 * k + 1.0) * pi / (2.0 * N));
    std::sort(got.begin(), got.end());
    std::sort(want.begin(), want.end());
    const double e_roots = max_abs_diff(got.data(), want.data(), N);

    // The naive tower oracle is O(N) deep, so its conditioning grows ~ N*eps.
    const double direct_tol = 1e-13 * N;
    const bool ok = rel_direct < direct_tol && e_comp_vs_mono < 1e-12 && e_roots < 1e-12;
    std::printf("  N=%5d: vs direct(rel)=%.1e  composed-vs-monomial=%.1e  roots=%.1e  %s\n",
                N, rel_direct, e_comp_vs_mono, e_roots, ok ? "OK" : "FAIL");
    return ok;
}

// -----------------------------------------------------------------------
// Test 5: node throughput microbenchmark.
//
// To expose the per-lane arithmetic difference (rather than memory bandwidth)
// the working set is kept L1-resident (4q * blocks * 8 bytes ~ 32 KiB) and
// hammered for many iterations; we report the best of several trials.
// -----------------------------------------------------------------------
static void microbench() {
    const int q = 256;                  // lanes per node
    const int blocks = 4;               // 4 * 4*256*8 B = 32 KiB working set
    const long total = static_cast<long>(blocks) * 4 * q;
    std::vector<double> base(total), buf(total);
    unsigned s = 7u;
    for (long i = 0; i < total; ++i) base[i] = frand(s);

    cc::NodeConst k = cc::node_const(0.37);
    // norm2_fused twiddles (arbitrary but valid unit-ish constants).
    const double c = std::cos(0.3), sn = std::sin(0.3);
    const double c0 = std::cos(0.15), s0 = std::sin(0.15);
    const double c1 = std::cos(0.62), s1 = std::sin(0.62);

    const int iters = 20000;
    const int trials = 7;
    auto run = [&](auto&& node) {
        double best = 1e300;
        for (int t = 0; t < trials; ++t) {
            std::memcpy(buf.data(), base.data(), sizeof(double) * total);
            const auto t0 = std::chrono::high_resolution_clock::now();
            for (int it = 0; it < iters; ++it)
                for (int b = 0; b < blocks; ++b) node(buf.data() + static_cast<long>(b) * 4 * q);
            const auto t1 = std::chrono::high_resolution_clock::now();
            const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
            best = std::min(best, ns / (static_cast<double>(iters) * blocks * q));
        }
        return best;  // best ns per lane
    };

    const double ns_fused = run([&](double* p) {
        bruun_radix4::norm2_fused_standalone(p, q, c, sn, c0, s0, c1, s1);
    });
    const double ns_mono = run([&](double* p) {
        cc::monomial_node(p, q, k.u, k.v);
    });
    const double ns_c12 = run([&](double* p) {
        cc::composed_node(p, q, k.u, k.v, k.w);
    });
    const double ns_c13 = run([&](double* p) {
        cc::composed_node_qdup(p, q, k.u, k.v, k.w);
    });
    const double ns_c14 = run([&](double* p) {
        cc::composed_node_dup(p, q, k.u, k.v, k.w);
    });
    const double ns_mfma = run([&](double* p) {
        cc::monomial_node_fma(p, q, k.u, k.v);
    });
    const double ns_cfma = run([&](double* p) {
        cc::composed_node_fma(p, q, k.u, k.v, k.w);
    });
    const double ns_low = run([&](double* p) {
        cc::lowmul_node(p, q, k.u, k.v);
    });

    std::printf("  normalized fused        (20q): %.4f ns/lane  (baseline)\n", ns_fused);
    std::printf("  monomial shared-su      (14q): %.4f ns/lane  %.3fx vs fused\n",
                ns_mono, ns_fused / ns_mono);
    std::printf("  composed full-shared    (12q): %.4f ns/lane  %.3fx vs mono\n",
                ns_c12, ns_mono / ns_c12);
    std::printf("  composed dup-Q-only     (13q): %.4f ns/lane  %.3fx vs mono\n",
                ns_c13, ns_mono / ns_c13);
    std::printf("  composed fully-dup      (14q): %.4f ns/lane  %.3fx vs mono\n",
                ns_c14, ns_mono / ns_c14);
    std::printf("  monomial all-FMA  (8 fma):     %.4f ns/lane  %.3fx vs mono  <== \n",
                ns_mfma, ns_mono / ns_mfma);
    std::printf("  composed all-FMA  (8 fma):     %.4f ns/lane  %.3fx vs mono\n",
                ns_cfma, ns_mono / ns_cfma);
    std::printf("  low-mul (4 mul+10 add):        %.4f ns/lane  %.3fx vs mono  (min-multiply)\n",
                ns_low, ns_mono / ns_low);
}

int main() {
    int pass = 0, fail = 0;

    std::printf("=== Test 1: composed == monomial == direct (single node) ===\n");
    for (int q : {1, 2, 4, 16, 64, 256}) { if (test_node_equivalence(q)) ++pass; else ++fail; }

    std::printf("\n=== Test 2: monomial <-> composed round-trip ===\n");
    for (int q : {1, 4, 16, 64}) { if (test_basis_roundtrip(q)) ++pass; else ++fail; }

    std::printf("\n=== Test 3+4: full recursion vs direct eval and T_N roots ===\n");
    for (int N : {4, 16, 64, 256, 1024, 4096, 16384}) {
        if (!cc::is_pow4(N)) continue;
        if (test_recursion(N)) ++pass; else ++fail;
    }

    std::printf("\n=== Test 5: node throughput microbenchmark ===\n");
    microbench();

    std::printf("\n%d passed, %d failed\n", pass, fail);
    return fail > 0 ? 1 : 0;
}
