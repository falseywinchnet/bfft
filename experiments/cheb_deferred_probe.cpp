// Deferred / cross-coupled Chebyshev representation — does it delete FMAs?
//
// The all-FMA radix-4 node materializes f(+u),f(-u),f(+v),f(-v) in 8 FMAs:
//   eu=fma(a2,u2,a0)  ou=fma(a3,u2,a1)  ev=fma(a2,v2,a0)  ov=fma(a3,v2,a1)
//   x0=fma(u,ou,eu)   x1=fma(-u,ou,eu)  x2=fma(v,ov,ev)   x3=fma(-v,ov,ev)
//
// The question (from the cross-coupling idea): can the tree carry the
// even/odd intermediate (eu,ou,ev,ov) — deferring the four +/- "output" FMAs —
// and pay materialization only at a boundary, spending < 8 FMA at internal
// nodes?
//
// Obstruction: child 0 (a'=+u) and child 1 (a'=-u) share (eu,ou), but their
// NEXT-level constants w0=sqrt((u+1)/2) and w1=sqrt((1-u)/2) differ, so the
// shared intermediate cannot be pushed down without replicating both subtrees.
// You can therefore only defer the +/- at the BOTTOM level (or one chosen
// boundary). This probe counts FMAs to show exactly that:
//   * materializing recursion:  2 N log4(N) FMA
//   * leaf-deferred recursion:  2 N log4(N) - N FMA   (leaf nodes drop 8 -> 4)
// and validates that the deferred leaves reconstruct the materialized values.
//
// Build:
//   c++ -O2 -std=c++17 -Iinclude experiments/cheb_deferred_probe.cpp \
//       build/libbfft.a -lm -o build/experiments/cheb_deferred_probe

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "chebyshev_composed_radix4_kernel.hpp"  // node_const, is_pow4

namespace cc = cheb_composed_r4;

// Operation-counting scalar: carries the value and bumps a global FMA counter
// on every fused multiply-add, so counts reflect the FMA-contracted form.
struct Ctr { static long fma; };
long Ctr::fma = 0;

struct CD {
    double v;
    CD() : v(0.0) {}
    CD(double x) : v(x) {}
};
static inline CD cfma(double a, CD b, CD c) { ++Ctr::fma; return CD(std::fma(a, b.v, c.v)); }

// All-FMA radix-4 recursion over CD. If defer_leaves, the bottom nodes (m==4)
// store (eu,ou,ev,ov) instead of the four signed values, skipping 4 FMA each.
static void eval(CD* p, int off, int m, double a, bool defer_leaves) {
    if (m == 1) return;
    const int q = m / 4;
    const cc::NodeConst k = cc::node_const(a);
    const double u = k.u, v = k.v, u2 = u * u, v2 = v * v;
    const bool leaf = (q == 1);  // m == 4: children are size 1
    for (int n = 0; n < q; ++n) {
        const CD A0 = p[off + n], A1 = p[off + q + n];
        const CD A2 = p[off + 2 * q + n], A3 = p[off + 3 * q + n];
        const CD eu = cfma(u2, A2, A0);
        const CD ou = cfma(u2, A3, A1);
        const CD ev = cfma(v2, A2, A0);
        const CD ov = cfma(v2, A3, A1);
        if (defer_leaves && leaf) {
            p[off + n] = eu; p[off + q + n] = ou;          // deferred even/odd
            p[off + 2 * q + n] = ev; p[off + 3 * q + n] = ov;
        } else {
            p[off + n]         = cfma(u, ou, eu);          // f(+u)
            p[off + q + n]     = cfma(-u, ou, eu);         // f(-u)
            p[off + 2 * q + n] = cfma(v, ov, ev);          // f(+v)
            p[off + 3 * q + n] = cfma(-v, ov, ev);         // f(-v)
        }
    }
    if (!leaf) {
        eval(p, off,           q,  u, defer_leaves);
        eval(p, off + q,       q, -u, defer_leaves);
        eval(p, off + 2 * q,   q,  v, defer_leaves);
        eval(p, off + 3 * q,   q, -v, defer_leaves);
    }
}

// Reconstruct the four signed leaf values from a leaf-deferred result, so we
// can confirm the deferral is loss-free. Walks to the m==4 nodes and applies
// the 4 output FMAs that eval() skipped.
static void materialize_leaves(CD* p, int off, int m, double a) {
    if (m == 1) return;
    const int q = m / 4;
    const cc::NodeConst k = cc::node_const(a);
    const double u = k.u, v = k.v;
    if (q == 1) {
        const CD eu = p[off], ou = p[off + 1], ev = p[off + 2], ov = p[off + 3];
        p[off]     = std::fma(u, ou.v, eu.v);
        p[off + 1] = std::fma(-u, ou.v, eu.v);
        p[off + 2] = std::fma(v, ov.v, ev.v);
        p[off + 3] = std::fma(-v, ov.v, ev.v);
        return;
    }
    materialize_leaves(p, off,         q,  u);
    materialize_leaves(p, off + q,     q, -u);
    materialize_leaves(p, off + 2 * q, q,  v);
    materialize_leaves(p, off + 3 * q, q, -v);
}

static long log4(int n) { long l = 0; while (n > 1) { n >>= 2; ++l; } return l; }

int main() {
    std::printf("Deferred-representation FMA accounting (all-FMA radix-4 nodes)\n\n");
    std::printf("%8s %12s %12s %10s %10s %12s  %s\n",
                "N", "materialize", "leaf-defer", "saved", "saved%%",
                "2N*log4N", "leaf-defer correct?");

    for (int N : {16, 64, 256, 1024, 4096, 16384, 65536}) {
        if (!cc::is_pow4(N)) continue;

        std::vector<CD> base(N);
        unsigned s = 17u + N;
        for (int i = 0; i < N; ++i) {
            s = s * 1103515245u + 12345u;
            base[i] = CD((static_cast<double>(s & 0x7fffffff) / 0x7fffffff) * 2.0 - 1.0);
        }

        std::vector<CD> mat = base;
        Ctr::fma = 0;
        eval(mat.data(), 0, N, 0.0, /*defer_leaves=*/false);
        const long fma_mat = Ctr::fma;

        std::vector<CD> def = base;
        Ctr::fma = 0;
        eval(def.data(), 0, N, 0.0, /*defer_leaves=*/true);
        const long fma_def = Ctr::fma;

        // Reconstruct and compare (the reconstruct FMAs are not counted: they
        // are the materialization the caller would do at the boundary).
        materialize_leaves(def.data(), 0, N, 0.0);
        double err = 0.0;
        for (int i = 0; i < N; ++i) err = std::max(err, std::abs(def[i].v - mat[i].v));

        const long saved = fma_mat - fma_def;
        const double pct = 100.0 * saved / fma_mat;
        std::printf("%8d %12ld %12ld %10ld %9.2f%% %12ld  err=%.1e %s\n",
                    N, fma_mat, fma_def, saved, pct, 2L * N * log4(N),
                    err, err < 1e-12 ? "OK" : "FAIL");
    }

    std::printf("\nReading:\n");
    std::printf(" * materialize = 2 N log4(N): the all-FMA floor; no internal\n");
    std::printf("   deferral is possible (sibling subtrees use different w).\n");
    std::printf(" * leaf-defer saves exactly N FMA (each of the N/4 bottom nodes\n");
    std::printf("   drops its 4 output FMAs), a constant-factor 1/(2 log4 N) win,\n");
    std::printf("   valid only if the boundary consumes the even/odd pair form\n");
    std::printf("   (eu,ou,ev,ov) rather than the four signed values -- e.g. a\n");
    std::printf("   scaled-native or DCT-projected output. Internal levels cannot\n");
    std::printf("   be deferred, so this is the whole prize of cross-coupling.\n");
    return 0;
}
