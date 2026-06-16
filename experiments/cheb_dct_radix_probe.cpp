// Radix 2 / 4 / 8 probe for the real (DCT) Chebyshev evaluation node.
//
// Context. The DCT/DST kernels in this tree are the Bruun real-residue tree with
// the imaginary (sine / Chebyshev-U) half pruned: the node is a *real* polynomial
// evaluation at the real Chebyshev nodes, not a complex rotation (see capstone.md,
// "Output endpoint"). The capstone settled radix-4 for the FFT empirically and
// proved the total multiply count is radix-INVARIANT (N log2 N either way). So the
// radix choice for the real node is purely an instruction-level question:
//
//   * lower radix  -> lighter node, shorter critical path, but MORE passes over
//                     memory (log2 N levels) => more load/store traffic;
//   * higher radix -> fewer passes (log8 N levels), but heavier node => more
//                     registers live, possible spills.
//
// The real node has a different instruction shape from the complex FFT node (no
// rotation), so its optimum may differ from the FFT's radix-4. This probe measures
// it rather than assuming. Three things per radix:
//   1. correctness of one node vs direct Horner evaluation at the r real points;
//   2. ns/output for the isolated node, L1-resident (raw node throughput);
//   3. ns/output for a full depth-r node tree over N (memory-traffic behavior).
//
// Build: see Makefile target $(DCT_RADIX_PROBE).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#define R __restrict__

static double frand(unsigned& s) {
    s = s * 1103515245u + 12345u;
    return (static_cast<double>(s & 0x7fffffff) / 0x7fffffff) * 2.0 - 1.0;
}

// ---------------------------------------------------------------------------
// Real Chebyshev evaluation nodes. Block layout [a0|a1|...|a_{r-1}] each q,
// in place -> [f(s0)|f(s1)|...|f(s_{r-1})] where s_* are the r real nodes.
// ---------------------------------------------------------------------------

// Radix-2: f(y)=a0+a1 y at y = +u,-u.  2 FMA / 2 outputs = 1 FMA/output.
static void node_r2(double* R p, int q, double u) {
    double* R a0 = p; double* R a1 = p + q;
    for (int n = 0; n < q; ++n) {
        const double A0 = a0[n], A1 = a1[n];
        a0[n] = A0 + u * A1;   // f(+u)
        a1[n] = A0 - u * A1;   // f(-u)
    }
}

// Radix-4: f(y)=a0+a1 y+a2 y^2+a3 y^3 at +u,-u,+v,-v via even/odd in y.
// 8 FMA / 4 outputs = 2 FMA/output (the capstone throughput optimum for the FFT).
static void node_r4(double* R p, int q, double u, double v) {
    const double u2 = u * u, v2 = v * v;
    double* R a0 = p; double* R a1 = p + q; double* R a2 = p + 2 * q; double* R a3 = p + 3 * q;
    for (int n = 0; n < q; ++n) {
        const double A0 = a0[n], A1 = a1[n], A2 = a2[n], A3 = a3[n];
        const double eu = A0 + A2 * u2, ou = A1 + A3 * u2;
        const double ev = A0 + A2 * v2, ov = A1 + A3 * v2;
        a0[n] = eu + u * ou; a1[n] = eu - u * ou;
        a2[n] = ev + v * ov; a3[n] = ev - v * ov;
    }
}

// Radix-8: f(y)=sum a_i y^i (i=0..7) at +-u1,+-v1,+-u2,+-v2 via even/odd in y.
// Even E(z)=a0+a2 z+a4 z^2+a6 z^3, Odd O(z)=a1+a3 z+a5 z^2+a7 z^3, z=y^2.
// Evaluate E,O (cubics in z) at the 4 squared nodes by Horner, then combine.
// ~32 FMA / 8 outputs = 4 FMA/output. Output order matches s8[] below.
static void node_r8(double* R p, int q,
                     double s0, double s1, double s2, double s3 /* the 4 + nodes */) {
    const double z0 = s0 * s0, z1 = s1 * s1, z2 = s2 * s2, z3 = s3 * s3;
    double* R a0 = p; double* R a1 = p + q; double* R a2 = p + 2 * q; double* R a3 = p + 3 * q;
    double* R a4 = p + 4 * q; double* R a5 = p + 5 * q; double* R a6 = p + 6 * q; double* R a7 = p + 7 * q;
    for (int n = 0; n < q; ++n) {
        const double e0 = a0[n], e1 = a2[n], e2 = a4[n], e3 = a6[n];
        const double o0 = a1[n], o1 = a3[n], o2 = a5[n], o3 = a7[n];
        // E(z_i), O(z_i) by Horner (3 FMA each).
        const double E0 = ((e3 * z0 + e2) * z0 + e1) * z0 + e0;
        const double E1 = ((e3 * z1 + e2) * z1 + e1) * z1 + e0;
        const double E2 = ((e3 * z2 + e2) * z2 + e1) * z2 + e0;
        const double E3 = ((e3 * z3 + e2) * z3 + e1) * z3 + e0;
        const double O0 = ((o3 * z0 + o2) * z0 + o1) * z0 + o0;
        const double O1 = ((o3 * z1 + o2) * z1 + o1) * z1 + o0;
        const double O2 = ((o3 * z2 + o2) * z2 + o1) * z2 + o0;
        const double O3 = ((o3 * z3 + o2) * z3 + o1) * z3 + o0;
        a0[n] = E0 + s0 * O0; a1[n] = E0 - s0 * O0;
        a2[n] = E1 + s1 * O1; a3[n] = E1 - s1 * O1;
        a4[n] = E2 + s2 * O2; a5[n] = E2 - s2 * O2;
        a6[n] = E3 + s3 * O3; a7[n] = E3 - s3 * O3;
    }
}

// ---------------------------------------------------------------------------
// Correctness: one node vs direct Horner at the r points.
// ---------------------------------------------------------------------------
static double horner(const double* a, int deg, double y) {
    double acc = a[deg];
    for (int i = deg - 1; i >= 0; --i) acc = acc * y + a[i];
    return acc;
}

static bool validate() {
    bool ok = true;
    // r2
    {
        double p[2] = {0.3, -0.7}; double a[2] = {p[0], p[1]};
        const double u = 0.9; node_r2(p, 1, u);
        ok &= std::abs(p[0] - horner(a, 1, u)) < 1e-13 && std::abs(p[1] - horner(a, 1, -u)) < 1e-13;
    }
    // r4
    {
        double p[4] = {0.3, -0.7, 0.2, 0.5}; double a[4] = {p[0], p[1], p[2], p[3]};
        const double u = 0.97, v = 0.21; node_r4(p, 1, u, v);
        ok &= std::abs(p[0] - horner(a, 3, u)) < 1e-13 && std::abs(p[1] - horner(a, 3, -u)) < 1e-13
           && std::abs(p[2] - horner(a, 3, v)) < 1e-13 && std::abs(p[3] - horner(a, 3, -v)) < 1e-13;
    }
    // r8
    {
        double a[8]; unsigned s = 5u; for (int i = 0; i < 8; ++i) a[i] = frand(s);
        double p[8]; std::memcpy(p, a, sizeof a);
        const double s0 = 0.99, s1 = 0.84, s2 = 0.57, s3 = 0.18;
        node_r8(p, 1, s0, s1, s2, s3);
        const double sgn[8] = {s0, -s0, s1, -s1, s2, -s2, s3, -s3};
        for (int i = 0; i < 8; ++i) ok &= std::abs(p[i] - horner(a, 7, sgn[i])) < 1e-12;
    }
    std::printf("  node correctness (r2,r4,r8 vs Horner): %s\n", ok ? "OK" : "FAIL");
    return ok;
}

// ---------------------------------------------------------------------------
// Node throughput: ns per output, L1-resident, reset once per trial.
// ---------------------------------------------------------------------------
template <typename F>
static double micro(F&& node, int r, int q, int blocks) {
    const long total = static_cast<long>(blocks) * r * q;
    std::vector<double> base(total), buf(total);
    unsigned s = 7u; for (long i = 0; i < total; ++i) base[i] = frand(s);
    const int iters = 20000, trials = 7;
    double best = 1e300;
    for (int t = 0; t < trials; ++t) {
        std::memcpy(buf.data(), base.data(), sizeof(double) * total);
        const auto t0 = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < iters; ++it)
            for (int b = 0; b < blocks; ++b) node(buf.data() + static_cast<long>(b) * r * q);
        const auto t1 = std::chrono::high_resolution_clock::now();
        const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
        best = std::min(best, ns / (static_cast<double>(iters) * blocks * q * r));  // per output
    }
    return best;
}

// ---------------------------------------------------------------------------
// Full depth-r node tree over N (captures passes/memory traffic). Arbitrary
// constants; this measures the schedule's locality, not a correct DCT.
// ---------------------------------------------------------------------------
static void tree2(double* p, int off, int m) {
    if (m == 1) return;
    const int q = m / 2;
    node_r2(p + off, q, 0.9);
    tree2(p, off, q); tree2(p, off + q, q);
}
static void tree4(double* p, int off, int m) {
    if (m == 1) return;
    const int q = m / 4;
    node_r4(p + off, q, 0.97, 0.21);
    tree4(p, off, q); tree4(p, off + q, q); tree4(p, off + 2 * q, q); tree4(p, off + 3 * q, q);
}
static void tree8(double* p, int off, int m) {
    if (m == 1) return;
    const int q = m / 8;
    node_r8(p + off, q, 0.99, 0.84, 0.57, 0.18);
    for (int c = 0; c < 8; ++c) tree8(p, off + c * q, q);
}

template <typename F>
static double tree_ns(F&& tree, int N) {
    std::vector<double> base(N), buf(N);
    unsigned s = 11u + N; for (int i = 0; i < N; ++i) base[i] = frand(s);
    const int iters = std::max(40, (1 << 22) / N), trials = 5;
    double best = 1e300;
    for (int t = 0; t < trials; ++t) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < iters; ++it) {
            std::memcpy(buf.data(), base.data(), sizeof(double) * N);
            tree(buf.data(), 0, N);
        }
        const auto t1 = std::chrono::high_resolution_clock::now();
        const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
        best = std::min(best, ns / (static_cast<double>(iters) * N));  // per output
    }
    return best;
}

int main() {
    std::printf("=== Radix 2/4/8 real-node probe ===\n");
    if (!validate()) return 1;

    std::printf("\nNode throughput (ns / output, L1-resident, 32 KiB working set):\n");
    const int q = 256;
    const double u = 0.9, u4 = 0.97, v4 = 0.21, s0 = 0.99, s1 = 0.84, s2 = 0.57, s3 = 0.18;
    const double r2ns = micro([&](double* x) { node_r2(x, q, u); }, 2, q, 8);
    const double r4ns = micro([&](double* x) { node_r4(x, q, u4, v4); }, 4, q, 4);
    const double r8ns = micro([&](double* x) { node_r8(x, q, s0, s1, s2, s3); }, 8, q, 2);
    std::printf("  radix-2 (1 FMA/out): %.4f ns/out\n", r2ns);
    std::printf("  radix-4 (2 FMA/out): %.4f ns/out\n", r4ns);
    std::printf("  radix-8 (4 FMA/out): %.4f ns/out\n", r8ns);

    std::printf("\nFull depth-r node tree (ns / output, incl. memory traffic):\n");
    std::printf("  %-9s %10s %10s %10s\n", "N", "radix-2", "radix-4", "radix-8");
    for (int N : {64, 4096, 262144}) {  // 8^2, 8^4, 8^6 (also powers of 2 and 4)
        const double t2 = tree_ns(tree2, N);
        const double t4 = tree_ns(tree4, N);
        const double t8 = tree_ns(tree8, N);
        std::printf("  %-9d %8.3f ns %8.3f ns %8.3f ns\n", N, t2, t4, t8);
    }
    return 0;
}
