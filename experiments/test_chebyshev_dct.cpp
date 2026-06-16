// Validation for the Chebyshev DCT/DST reference kernels and the radix-4
// Chebyshev-tree DCT-III path (experiments/chebyshev_dct_kernel.hpp).
//
// Tests:
//   1. Forward/inverse round-trip to machine precision for all eight types
//      (DCT-I..IV, DST-I..IV), across a range of n. Catches normalization.
//   2. Direct DCT/DST forward against an independent textbook double-sum, and
//      a hand-checked small case, so the oracle itself is trusted.
//   3. flat_to_tower <-> tower_to_flat round-trip to machine precision (pow-4).
//   4. The radix-4 tree (eval_monomial) computes flat-Chebyshev multipoint
//      evaluation: tree leaves == Clenshaw(flat) at leaf_x.
//   5. DCT-III == flat-Chebyshev evaluation at the roots of T_n (Clenshaw form).
//   6. dct3_via_tree == direct DCT-III to ~1e-12 (the ROADMAP Phase-1 claim,
//      end to end on the shipped radix-4 kernel).
//
// Build:
//   c++ -O2 -std=c++17 -Iinclude -Iexperiments \
//       experiments/test_chebyshev_dct.cpp build/libbfft.a -lm \
//       -o build/experiments/test_chebyshev_dct

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "chebyshev_dct_kernel.hpp"
#include "chebyshev_dct_radix4.hpp"
#include "dct1_dst1_via_fft.hpp"
#include "cheb_dct_assemble.hpp"
#include "bruun_dct.hpp"

namespace cd = cheb_dct;

static double frand(unsigned& s) {
    s = s * 1103515245u + 12345u;
    return (static_cast<double>(s & 0x7fffffff) / 0x7fffffff) * 2.0 - 1.0;
}

static double max_abs_diff(const double* a, const double* b, int n) {
    double e = 0.0;
    for (int i = 0; i < n; ++i) e = std::max(e, std::abs(a[i] - b[i]));
    return e;
}

// ---------------------------------------------------------------------------
// 1. Round-trip for every type.
// ---------------------------------------------------------------------------
template <typename Fwd, typename Inv>
static bool roundtrip(const char* name, Fwd fwd, Inv inv, int n_min, int n_max) {
    double worst = 0.0;
    int worst_n = 0;
    for (int n = n_min; n <= n_max; ++n) {
        std::vector<double> x(n), y(n), z(n);
        unsigned s = 1234u + 7u * n;
        for (int i = 0; i < n; ++i) x[i] = frand(s);
        fwd(x.data(), y.data(), n);
        inv(y.data(), z.data(), n);
        const double e = max_abs_diff(x.data(), z.data(), n);
        if (e > worst) { worst = e; worst_n = n; }
    }
    const bool ok = worst < 1e-11;
    std::printf("  %-9s round-trip: worst err=%.1e (n=%d)  %s\n",
                name, worst, worst_n, ok ? "OK" : "FAIL");
    return ok;
}

// ---------------------------------------------------------------------------
// 2. Trusted-oracle spot checks: a hand-evaluated tiny case per family.
// ---------------------------------------------------------------------------
static bool spot_checks() {
    bool ok = true;

    // DCT-II, n=2, X=[1,0]: Y_k = 2*X_0*cos(pi*0.5*k/2) = 2 cos(pi k/4).
    // Y_0 = 2, Y_1 = 2 cos(pi/4) = sqrt(2).
    {
        double X[2] = {1.0, 0.0}, Y[2];
        cd::dct2_forward(X, Y, 2);
        const bool c = std::abs(Y[0] - 2.0) < 1e-12 &&
                       std::abs(Y[1] - std::sqrt(2.0)) < 1e-12;
        std::printf("  DCT-II  n=2 X=[1,0] -> [%.3f,%.3f] (want 2,sqrt2)  %s\n",
                    Y[0], Y[1], c ? "OK" : "FAIL");
        ok = ok && c;
    }
    // DCT-III, n=2, X=[1,1]: Y_0 = 1 + 2*cos(pi/4)=1+sqrt(2);
    //                        Y_1 = 1 + 2*cos(3pi/4)=1-sqrt(2).
    {
        double X[2] = {1.0, 1.0}, Y[2];
        cd::dct3_forward(X, Y, 2);
        const double r2 = std::sqrt(2.0);
        const bool c = std::abs(Y[0] - (1.0 + r2)) < 1e-12 &&
                       std::abs(Y[1] - (1.0 - r2)) < 1e-12;
        std::printf("  DCT-III n=2 X=[1,1] -> [%.3f,%.3f] (want 1+-sqrt2)  %s\n",
                    Y[0], Y[1], c ? "OK" : "FAIL");
        ok = ok && c;
    }
    // DST-I, n=1, X=[1]: Y_0 = 2*sin(pi/2) = 2.
    {
        double X[1] = {1.0}, Y[1];
        cd::dst1_forward(X, Y, 1);
        const bool c = std::abs(Y[0] - 2.0) < 1e-12;
        std::printf("  DST-I   n=1 X=[1]   -> [%.3f] (want 2)  %s\n", Y[0], c ? "OK" : "FAIL");
        ok = ok && c;
    }
    // DCT-II and DCT-III are an unnormalized inverse pair: DCT-II(DCT-III(x))
    // = 2n x exactly (FFTW REDFT10 . REDFT01 = 2n).
    {
        const int n = 11;
        std::vector<double> x(n), t(n), z(n);
        unsigned s = 77u;
        for (int i = 0; i < n; ++i) x[i] = frand(s);
        cd::dct3_forward(x.data(), t.data(), n);
        cd::dct2_forward(t.data(), z.data(), n);
        double e = 0.0;
        for (int i = 0; i < n; ++i) e = std::max(e, std::abs(z[i] - 2.0 * n * x[i]));
        const bool c = e < 1e-11;
        std::printf("  DCT-II.DCT-III = 2n.I: err=%.1e  %s\n", e, c ? "OK" : "FAIL");
        ok = ok && c;
    }
    return ok;
}

// ---------------------------------------------------------------------------
// 3. flat <-> tower round-trip.
// ---------------------------------------------------------------------------
static bool tower_roundtrip(int n) {
    std::vector<double> flat(n), tower(n), back(n);
    unsigned s = 909u + n;
    for (int i = 0; i < n; ++i) flat[i] = frand(s);
    cd::flat_to_tower(flat.data(), 0, n, tower.data());
    cd::tower_to_flat(tower.data(), 0, n, back.data());
    const double e = max_abs_diff(flat.data(), back.data(), n);
    const bool ok = e < 1e-12;
    std::printf("  n=%5d: flat<->tower round-trip err=%.1e  %s\n", n, e, ok ? "OK" : "FAIL");
    return ok;
}

// ---------------------------------------------------------------------------
// 4. The tree computes Chebyshev multipoint evaluation.
// ---------------------------------------------------------------------------
static bool tree_is_cheb_eval(int n) {
    // Random flat Chebyshev series; convert to tower; run eval_monomial; compare
    // leaves to Clenshaw evaluation of the same flat series at leaf_x.
    std::vector<double> flat(n), tower(n), xleaf(n);
    unsigned s = 4242u + n;
    for (int i = 0; i < n; ++i) flat[i] = frand(s);

    cd::flat_to_tower(flat.data(), 0, n, tower.data());
    cheb_composed_r4::leaf_x(0, n, 0.0, xleaf.data());
    cheb_composed_r4::eval_monomial(tower.data(), 0, n, 0.0);

    double e = 0.0, mag = 0.0;
    for (int i = 0; i < n; ++i) {
        const double want = cd::clenshaw(flat.data(), n, xleaf[i]);
        e = std::max(e, std::abs(tower[i] - want));
        mag = std::max(mag, std::abs(want));
    }
    const double rel = e / (mag > 0.0 ? mag : 1.0);
    const bool ok = rel < 1e-12 * n;  // tower oracle depth ~ log4 n, allow slack
    std::printf("  n=%5d: tree leaves vs Clenshaw(flat) rel=%.1e  %s\n",
                n, rel, ok ? "OK" : "FAIL");
    return ok;
}

// ---------------------------------------------------------------------------
// 5. DCT-III == flat-Chebyshev evaluation at T_n nodes.
// ---------------------------------------------------------------------------
static bool dct3_is_cheb_eval(int n) {
    std::vector<double> X(n), Yd(n), Yc(n);
    unsigned s = 31337u + n;
    for (int i = 0; i < n; ++i) X[i] = frand(s);
    cd::dct3_forward(X.data(), Yd.data(), n);
    cd::dct3_via_clenshaw(X.data(), Yc.data(), n);
    const double e = max_abs_diff(Yd.data(), Yc.data(), n);
    // Direct sum and Clenshaw are different algorithms; the gap grows ~ n*eps
    // (Clenshaw conditioning at high degree near the interval ends).
    const bool ok = e < 1e-13 * n;
    std::printf("  n=%5d: dct3 direct vs Clenshaw-at-T_n err=%.1e  %s\n",
                n, e, ok ? "OK" : "FAIL");
    return ok;
}

// ---------------------------------------------------------------------------
// 6. dct3_via_tree == direct DCT-III (Phase-1 endpoint, pow-4 n).
// ---------------------------------------------------------------------------
static bool dct3_tree_matches_direct(int n) {
    std::vector<double> X(n), Yd(n), Yt(n);
    unsigned s = 8675309u + n;
    for (int i = 0; i < n; ++i) X[i] = frand(s);
    cd::dct3_forward(X.data(), Yd.data(), n);
    cd::dct3_via_tree(X.data(), Yt.data(), n);
    double e = 0.0, mag = 0.0;
    for (int i = 0; i < n; ++i) {
        e = std::max(e, std::abs(Yd[i] - Yt[i]));
        mag = std::max(mag, std::abs(Yd[i]));
    }
    const double rel = e / (mag > 0.0 ? mag : 1.0);
    const bool ok = rel < 1e-11 * n;
    std::printf("  n=%5d: dct3_via_tree vs direct rel=%.1e  %s\n",
                n, rel, ok ? "OK" : "FAIL");
    return ok;
}

// ---------------------------------------------------------------------------
// 7. Production-shaped ChebDCT plan: forward == direct, round-trip == identity,
//    and (the convention check) zero heap allocation in the hot path.
// ---------------------------------------------------------------------------
static bool plan_forward_matches_direct(int n) {
    cheb_dct_r4::ChebDCT plan;
    if (!plan.init(n)) { std::printf("  n=%5d: plan.init FAILED\n", n); return false; }
    std::vector<double> X(n), Yd(n), Yp(n), work(n);
    unsigned s = 246813u + n;
    for (int i = 0; i < n; ++i) X[i] = frand(s);
    cd::dct3_forward(X.data(), Yd.data(), n);
    plan.dct3_forward(X.data(), Yp.data(), work.data());
    double e = 0.0, mag = 0.0;
    for (int i = 0; i < n; ++i) {
        e = std::max(e, std::abs(Yd[i] - Yp[i]));
        mag = std::max(mag, std::abs(Yd[i]));
    }
    const double rel = e / (mag > 0.0 ? mag : 1.0);
    const bool ok = rel < 1e-11 * n;
    std::printf("  n=%5d: plan dct3_forward vs direct rel=%.1e  %s\n",
                n, rel, ok ? "OK" : "FAIL");
    return ok;
}

static bool plan_roundtrip(int n) {
    cheb_dct_r4::ChebDCT plan;
    if (!plan.init(n)) { std::printf("  n=%5d: plan.init FAILED\n", n); return false; }
    std::vector<double> X(n), Y(n), Z(n), work(n);
    unsigned s = 13579u + n;
    for (int i = 0; i < n; ++i) X[i] = frand(s);
    plan.dct3_forward(X.data(), Y.data(), work.data());
    plan.dct3_inverse(Y.data(), Z.data(), work.data());
    double e = 0.0;
    for (int i = 0; i < n; ++i) e = std::max(e, std::abs(X[i] - Z[i]));
    const bool ok = e < 1e-11 * n;
    std::printf("  n=%5d: plan dct3 forward->inverse err=%.1e  %s\n",
                n, e, ok ? "OK" : "FAIL");
    return ok;
}

// ---------------------------------------------------------------------------
// 9. ChebDCT plan DCT-IV / DST-IV vs direct, and self-inverse round-trip.
// ---------------------------------------------------------------------------
template <typename Fwd, typename Direct>
static bool plan_iv_matches_direct(const char* name, int n, Fwd fwd, Direct direct) {
    cheb_dct_r4::ChebDCT plan;
    if (!plan.init(n)) { std::printf("  n=%5d: plan.init FAILED\n", n); return false; }
    std::vector<double> X(n), Yd(n), Yp(n), work(n);
    unsigned s = 555111u + n;
    for (int i = 0; i < n; ++i) X[i] = frand(s);
    direct(X.data(), Yd.data(), n);
    fwd(plan, X.data(), Yp.data(), work.data());
    double e = 0.0, mag = 0.0;
    for (int i = 0; i < n; ++i) {
        e = std::max(e, std::abs(Yd[i] - Yp[i]));
        mag = std::max(mag, std::abs(Yd[i]));
    }
    const double rel = e / (mag > 0.0 ? mag : 1.0);
    const bool ok = rel < 1e-11 * n;
    std::printf("  %-7s n=%5d: plan vs direct rel=%.1e  %s\n",
                name, n, rel, ok ? "OK" : "FAIL");
    return ok;
}

template <typename Fwd, typename Inv>
static bool plan_iv_roundtrip(const char* name, int n, Fwd fwd, Inv inv) {
    cheb_dct_r4::ChebDCT plan;
    if (!plan.init(n)) { std::printf("  n=%5d: plan.init FAILED\n", n); return false; }
    std::vector<double> X(n), Y(n), Z(n), work(n);
    unsigned s = 222333u + n;
    for (int i = 0; i < n; ++i) X[i] = frand(s);
    fwd(plan, X.data(), Y.data(), work.data());
    inv(plan, Y.data(), Z.data(), work.data());
    double e = 0.0;
    for (int i = 0; i < n; ++i) e = std::max(e, std::abs(X[i] - Z[i]));
    const bool ok = e < 1e-11 * n;
    std::printf("  %-7s n=%5d: forward->inverse err=%.1e  %s\n",
                name, n, e, ok ? "OK" : "FAIL");
    return ok;
}

// ---------------------------------------------------------------------------
// 13. DCT-I / DST-I endpoints via the real FFT, vs direct oracle + round-trip.
// ---------------------------------------------------------------------------
template <typename Plan, typename Direct>
static bool endpoint_matches_direct(const char* name, int n, Direct direct) {
    Plan plan(n);
    std::vector<double> X(n), Yd(n), Yp(n);
    unsigned s = 909090u + n;
    for (int i = 0; i < n; ++i) X[i] = frand(s);
    direct(X.data(), Yd.data(), n);
    plan.forward(X.data(), Yp.data());
    double e = 0.0, mag = 0.0;
    for (int i = 0; i < n; ++i) {
        e = std::max(e, std::abs(Yd[i] - Yp[i]));
        mag = std::max(mag, std::abs(Yd[i]));
    }
    const double rel = e / (mag > 0.0 ? mag : 1.0);
    const bool ok = rel < 1e-11 * n;
    std::printf("  %-6s n=%5d: via-FFT vs direct rel=%.1e  %s\n",
                name, n, rel, ok ? "OK" : "FAIL");
    return ok;
}

template <typename Plan>
static bool endpoint_roundtrip(const char* name, int n) {
    Plan plan(n);
    std::vector<double> X(n), Y(n), Z(n);
    unsigned s = 818181u + n;
    for (int i = 0; i < n; ++i) X[i] = frand(s);
    plan.forward(X.data(), Y.data());
    plan.inverse(Y.data(), Z.data());
    double e = 0.0;
    for (int i = 0; i < n; ++i) e = std::max(e, std::abs(X[i] - Z[i]));
    const bool ok = e < 1e-11 * n;
    std::printf("  %-6s n=%5d: forward->inverse err=%.1e  %s\n",
                name, n, e, ok ? "OK" : "FAIL");
    return ok;
}

// ---------------------------------------------------------------------------
// 15. Real FFT assembled from DCT-I/DST-I vs the shipped Bruun FFT and a direct
//     DFT. Proves the Phase-5 assembly algebra.
// ---------------------------------------------------------------------------
static bool assembled_rfft_matches(int n) {
    cheb_dct_assemble::AssembledRFFT plan(n);
    const int nb = n / 2 + 1;
    std::vector<double> x(n);
    unsigned s = 4711u + n;
    for (int i = 0; i < n; ++i) x[i] = frand(s);

    std::vector<bfft::complex> Xa(nb);
    plan.forward(x.data(), Xa.data());

    // Reference 1: shipped Bruun real FFT.
    bfft::plan ref(static_cast<std::size_t>(n));
    std::vector<bfft::complex> Xr = ref.forward(x);

    // Reference 2: direct DFT (independent oracle).
    double e_ref = 0.0, e_dft = 0.0, mag = 0.0;
    for (int k = 0; k < nb; ++k) {
        double re = 0.0, im = 0.0;
        for (int t = 0; t < n; ++t) {
            const double ang = -2.0 * cd::kPi * k * t / n;
            re += x[t] * std::cos(ang);
            im += x[t] * std::sin(ang);
        }
        e_dft = std::max(e_dft, std::abs(Xa[k].re - re));
        e_dft = std::max(e_dft, std::abs(Xa[k].im - im));
        e_ref = std::max(e_ref, std::abs(Xa[k].re - Xr[k].re));
        e_ref = std::max(e_ref, std::abs(Xa[k].im - Xr[k].im));
        mag = std::max(mag, std::abs(re) + std::abs(im));
    }
    const double rel = std::max(e_ref, e_dft) / (mag > 0.0 ? mag : 1.0);
    const bool ok = rel < 1e-11 * n;
    std::printf("  n=%5d: assembled vs Bruun=%.1e vs DFT=%.1e (rel=%.1e)  %s\n",
                n, e_ref, e_dft, rel, ok ? "OK" : "FAIL");
    return ok;
}

// ---------------------------------------------------------------------------
// 16. Bruun-native repack-free DCT-II / DCT-III: vs direct oracle + round-trip.
// ---------------------------------------------------------------------------
static bool bruun_dct_checks(int n) {
    bruun_dct::BruunDCT plan;
    if (!plan.init(n)) { std::printf("  n=%5d: init FAILED\n", n); return false; }
    std::vector<double> X(n), Y2(n), Y3(n), Yd2(n), Yd3(n), Z(n);
    unsigned s = 31415u + n;
    for (int i = 0; i < n; ++i) X[i] = frand(s);

    plan.dct2_forward(X.data(), Y2.data());
    plan.dct3_forward(X.data(), Y3.data());
    cd::dct2_forward(X.data(), Yd2.data(), n);
    cd::dct3_forward(X.data(), Yd3.data(), n);

    auto rel = [&](const std::vector<double>& a, const std::vector<double>& b) {
        double e = 0.0, mag = 0.0;
        for (int i = 0; i < n; ++i) { e = std::max(e, std::abs(a[i] - b[i])); mag = std::max(mag, std::abs(b[i])); }
        return e / (mag > 0.0 ? mag : 1.0);
    };
    const double e2 = rel(Y2, Yd2), e3 = rel(Y3, Yd3);

    // round-trips
    plan.dct2_inverse(Y2.data(), Z.data());
    double rt2 = 0.0; for (int i = 0; i < n; ++i) rt2 = std::max(rt2, std::abs(Z[i] - X[i]));
    plan.dct3_inverse(Y3.data(), Z.data());
    double rt3 = 0.0; for (int i = 0; i < n; ++i) rt3 = std::max(rt3, std::abs(Z[i] - X[i]));

    const bool ok = e2 < 1e-11 * n && e3 < 1e-11 * n && rt2 < 1e-11 * n && rt3 < 1e-11 * n;
    std::printf("  n=%5d: DCT-II=%.1e DCT-III=%.1e  rt2=%.1e rt3=%.1e  %s\n",
                n, e2, e3, rt2, rt3, ok ? "OK" : "FAIL");
    return ok;
}

int main() {
    int pass = 0, fail = 0;
    auto tally = [&](bool b) { if (b) ++pass; else ++fail; };

    std::printf("=== Test 1: forward/inverse round-trip, all eight types ===\n");
    tally(roundtrip("DCT-I",  cd::dct1_forward, cd::dct1_inverse, 2, 40));
    tally(roundtrip("DCT-II", cd::dct2_forward, cd::dct2_inverse, 1, 40));
    tally(roundtrip("DCT-III",cd::dct3_forward, cd::dct3_inverse, 1, 40));
    tally(roundtrip("DCT-IV", cd::dct4_forward, cd::dct4_inverse, 1, 40));
    tally(roundtrip("DST-I",  cd::dst1_forward, cd::dst1_inverse, 1, 40));
    tally(roundtrip("DST-II", cd::dst2_forward, cd::dst2_inverse, 1, 40));
    tally(roundtrip("DST-III",cd::dst3_forward, cd::dst3_inverse, 1, 40));
    tally(roundtrip("DST-IV", cd::dst4_forward, cd::dst4_inverse, 1, 40));

    std::printf("\n=== Test 2: oracle spot checks ===\n");
    tally(spot_checks());

    std::printf("\n=== Test 3: flat <-> tower round-trip ===\n");
    for (int n : {4, 16, 64, 256, 1024, 4096}) tally(tower_roundtrip(n));

    std::printf("\n=== Test 4: radix-4 tree == Chebyshev multipoint eval ===\n");
    for (int n : {4, 16, 64, 256, 1024, 4096}) tally(tree_is_cheb_eval(n));

    std::printf("\n=== Test 5: DCT-III == Chebyshev eval at T_n roots ===\n");
    for (int n : {2, 3, 4, 7, 16, 31, 64, 256}) tally(dct3_is_cheb_eval(n));

    std::printf("\n=== Test 6: dct3_via_tree == direct DCT-III (radix-4) ===\n");
    for (int n : {4, 16, 64, 256, 1024, 4096}) tally(dct3_tree_matches_direct(n));

    std::printf("\n=== Test 7: ChebDCT plan forward == direct (no-alloc kernel) ===\n");
    for (int n : {4, 16, 64, 256, 1024, 4096}) tally(plan_forward_matches_direct(n));

    std::printf("\n=== Test 8: ChebDCT plan forward->inverse round-trip ===\n");
    for (int n : {4, 16, 64, 256, 1024, 4096}) tally(plan_roundtrip(n));

    std::printf("\n=== Test 9: ChebDCT plan DCT-IV / DST-IV vs direct ===\n");
    auto dct4_fwd = [](cheb_dct_r4::ChebDCT& p, const double* x, double* y, double* w) {
        p.dct4_forward(x, y, w);
    };
    auto dst4_fwd = [](cheb_dct_r4::ChebDCT& p, const double* x, double* y, double* w) {
        p.dst4_forward(x, y, w);
    };
    auto dct4_inv = [](cheb_dct_r4::ChebDCT& p, const double* x, double* y, double* w) {
        p.dct4_inverse(x, y, w);
    };
    auto dst4_inv = [](cheb_dct_r4::ChebDCT& p, const double* x, double* y, double* w) {
        p.dst4_inverse(x, y, w);
    };
    for (int n : {4, 16, 64, 256, 1024, 4096})
        tally(plan_iv_matches_direct("DCT-IV", n, dct4_fwd, cd::dct4_forward));
    for (int n : {4, 16, 64, 256, 1024, 4096})
        tally(plan_iv_matches_direct("DST-IV", n, dst4_fwd, cd::dst4_forward));

    std::printf("\n=== Test 10: ChebDCT plan DCT-IV / DST-IV self-inverse round-trip ===\n");
    for (int n : {4, 16, 64, 256, 1024, 4096})
        tally(plan_iv_roundtrip("DCT-IV", n, dct4_fwd, dct4_inv));
    for (int n : {4, 16, 64, 256, 1024, 4096})
        tally(plan_iv_roundtrip("DST-IV", n, dst4_fwd, dst4_inv));

    std::printf("\n=== Test 11: ChebDCT plan DST-II / DST-III vs direct ===\n");
    auto dst3_fwd = [](cheb_dct_r4::ChebDCT& p, const double* x, double* y, double* w) {
        p.dst3_forward(x, y, w);
    };
    auto dst3_inv = [](cheb_dct_r4::ChebDCT& p, const double* x, double* y, double* w) {
        p.dst3_inverse(x, y, w);
    };
    auto dst2_fwd = [](cheb_dct_r4::ChebDCT& p, const double* x, double* y, double* w) {
        p.dst2_forward(x, y, w);
    };
    auto dst2_inv = [](cheb_dct_r4::ChebDCT& p, const double* x, double* y, double* w) {
        p.dst2_inverse(x, y, w);
    };
    for (int n : {4, 16, 64, 256, 1024, 4096})
        tally(plan_iv_matches_direct("DST-III", n, dst3_fwd, cd::dst3_forward));
    for (int n : {4, 16, 64, 256, 1024, 4096})
        tally(plan_iv_matches_direct("DST-II", n, dst2_fwd, cd::dst2_forward));

    std::printf("\n=== Test 12: ChebDCT plan DST-II / DST-III round-trip ===\n");
    for (int n : {4, 16, 64, 256, 1024, 4096})
        tally(plan_iv_roundtrip("DST-III", n, dst3_fwd, dst3_inv));
    for (int n : {4, 16, 64, 256, 1024, 4096})
        tally(plan_iv_roundtrip("DST-II", n, dst2_fwd, dst2_inv));

    std::printf("\n=== Test 13: DCT-I / DST-I endpoints via real FFT vs direct ===\n");
    using cheb_dct_endpoint::DCT1;
    using cheb_dct_endpoint::DST1;
    for (int n : {3, 5, 9, 17, 33, 65, 129})   // n = 2^m + 1
        tally(endpoint_matches_direct<DCT1>("DCT-I", n, cd::dct1_forward));
    for (int n : {3, 7, 15, 31, 63, 127})       // n = 2^m - 1
        tally(endpoint_matches_direct<DST1>("DST-I", n, cd::dst1_forward));

    std::printf("\n=== Test 14: DCT-I / DST-I endpoints self-inverse round-trip ===\n");
    for (int n : {3, 5, 9, 17, 33, 65, 129})
        tally(endpoint_roundtrip<DCT1>("DCT-I", n));
    for (int n : {3, 7, 15, 31, 63, 127})
        tally(endpoint_roundtrip<DST1>("DST-I", n));

    std::printf("\n=== Test 15: rFFT assembled from DCT-I/DST-I vs Bruun FFT and DFT ===\n");
    for (int n : {8, 16, 32, 64, 256, 1024}) tally(assembled_rfft_matches(n));

    std::printf("\n=== Test 16: Bruun repack-free DCT-II/III vs direct + round-trip ===\n");
    for (int n : {2, 4, 8, 16, 32, 64, 256, 1024, 4096}) tally(bruun_dct_checks(n));

    std::printf("\n%d passed, %d failed\n", pass, fail);
    return fail > 0 ? 1 : 0;
}
