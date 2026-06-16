// Real-multiply count of the radix-4 Chebyshev node, with and without the
// u^2 + v^2 = 1 identity. Answers: how much arithmetic (multiply) labor can the
// Chebyshev representation save beyond the throughput-optimal 8-FMA node?
//
// Three node variants (per node-lane, data-dependent ops only):
//   shared-su  : eu,ou,ev,ov (4 mul) + su,sv (2 mul)            = 6 mul + 8 add
//   all-FMA    : duplicates su,sv into fma/fnma                 = 8 mul + 8 add
//   low-mul    : v^2 = 1-u^2 reuses u^2*a2, u^2*a3              = 4 mul + 10 add
//
// All three compute identical outputs (validated against a direct tower eval).
// The probe counts data-dependent multiplies/adds across the full radix-4
// recursion (N = 4^L) and prints the whole-transform totals and the leading
// constant in front of N*log2(N).
//
// Build:
//   c++ -O2 -std=c++17 -Iinclude experiments/cheb_multiply_probe.cpp \
//       build/libbfft.a -lm -o build/experiments/cheb_multiply_probe

#include <cmath>
#include <cstdio>
#include <vector>

#include "chebyshev_composed_radix4_kernel.hpp"

namespace cc = cheb_composed_r4;

namespace {

struct Counter { long mul = 0, add = 0; };

enum NodeKind { SHARED, ALLFMA, LOWMUL };

// One radix-4 node over q lanes, counting only data-dependent mul/add.
void node(double* p, int q, double u, double v, NodeKind kind, Counter& c) {
    const double u2 = u * u, v2 = v * v;  // node constants, not counted
    double* a0 = p; double* a1 = p + q; double* a2 = p + 2 * q; double* a3 = p + 3 * q;
    for (int n = 0; n < q; ++n) {
        const double A0 = a0[n], A1 = a1[n], A2 = a2[n], A3 = a3[n];
        double x0, x1, x2, x3;
        if (kind == LOWMUL) {
            const double m2 = u2 * A2;            // mul
            const double m3 = u2 * A3;            // mul
            const double eu = A0 + m2;            // add
            const double ev = (A0 + A2) - m2;     // 2 add
            const double ou = A1 + m3;            // add
            const double ov = (A1 + A3) - m3;     // 2 add
            const double su = u * ou;             // mul
            const double sv = v * ov;             // mul
            x0 = eu + su; x1 = eu - su; x2 = ev + sv; x3 = ev - sv;  // 4 add
            c.mul += 4; c.add += 10;
        } else {
            const double eu = A0 + A2 * u2;       // mul + add
            const double ou = A1 + A3 * u2;       // mul + add
            const double ev = A0 + A2 * v2;       // mul + add
            const double ov = A1 + A3 * v2;       // mul + add
            if (kind == SHARED) {
                const double su = u * ou, sv = v * ov;  // 2 mul
                x0 = eu + su; x1 = eu - su; x2 = ev + sv; x3 = ev - sv;  // 4 add
                c.mul += 6; c.add += 8;
            } else {  // ALLFMA: product duplicated into fma/fnma
                x0 = eu + u * ou; x1 = eu - u * ou;       // 2 mul (dup) + 2 add
                x2 = ev + v * ov; x3 = ev - v * ov;       // 2 mul (dup) + 2 add
                c.mul += 8; c.add += 8;
            }
        }
        a0[n] = x0; a1[n] = x1; a2[n] = x2; a3[n] = x3;
    }
}

void eval(double* p, int off, int m, double a, NodeKind kind, Counter& c) {
    if (m == 1) return;
    const int q = m / 4;
    const cc::NodeConst k = cc::node_const(a);
    node(p + off, q, k.u, k.v, kind, c);
    eval(p, off,           q,  k.u, kind, c);
    eval(p, off + q,       q, -k.u, kind, c);
    eval(p, off + 2 * q,   q,  k.v, kind, c);
    eval(p, off + 3 * q,   q, -k.v, kind, c);
}

double check_vs_direct(int N, NodeKind kind) {
    std::vector<double> C(N), leaves(N), xleaf(N);
    unsigned s = 71u + N;
    for (int i = 0; i < N; ++i) {
        s = s * 1103515245u + 12345u;
        C[i] = (static_cast<double>(s & 0x7fffffff) / 0x7fffffff) * 2.0 - 1.0;
    }
    cc::leaf_x(0, N, 0.0, xleaf.data());
    leaves = C;
    Counter c;
    eval(leaves.data(), 0, N, 0.0, kind, c);
    double e = 0.0, mx = 0.0;
    for (int i = 0; i < N; ++i) {
        const double d = cc::eval_direct(C.data(), 0, N, xleaf[i]);
        e = std::max(e, std::abs(leaves[i] - d));
        mx = std::max(mx, std::abs(leaves[i]));
    }
    return e / (mx > 0 ? mx : 1.0);
}

double log2i(int n) { double l = 0; while (n > 1) { n >>= 1; ++l; } return l; }

} // namespace

int main() {
    std::printf("Radix-4 Chebyshev node multiply count (data-dependent ops)\n\n");
    std::printf("%8s | %-26s | %-26s | %-26s\n", "N",
                "shared-su (6 mul/node)", "all-FMA (8 mul/node)", "low-mul (4 mul/node)");
    std::printf("%8s | %10s %10s rel | %10s %10s rel | %10s %10s rel\n",
                "", "mul", "add", "mul", "add", "mul", "add");

    for (int N : {16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576}) {
        if (!cc::is_pow4(N)) continue;

        std::vector<double> z(N, 0.0);
        Counter cs, cf, cl;
        eval(z.data(), 0, N, 0.0, SHARED, cs);
        std::fill(z.begin(), z.end(), 0.0);
        eval(z.data(), 0, N, 0.0, ALLFMA, cf);
        std::fill(z.begin(), z.end(), 0.0);
        eval(z.data(), 0, N, 0.0, LOWMUL, cl);

        const double nl = N * log2i(N);
        std::printf("%8d | %10ld %10ld %4.2f | %10ld %10ld %4.2f | %10ld %10ld %4.2f\n",
                    N, cs.mul, cs.add, cs.mul / nl,
                    cf.mul, cf.add, cf.mul / nl,
                    cl.mul, cl.add, cl.mul / nl);
    }

    std::printf("\nCorrectness (relative error vs direct tower eval):\n");
    for (int N : {16, 256, 4096, 65536}) {
        std::printf("  N=%6d  shared=%.1e  all-FMA=%.1e  low-mul=%.1e\n",
                    N, check_vs_direct(N, SHARED), check_vs_direct(N, ALLFMA),
                    check_vs_direct(N, LOWMUL));
    }

    std::printf("\nReading: 'rel' is multiplies / (N*log2 N). The u^2+v^2=1 identity\n");
    std::printf("takes the leading real-multiply constant from 0.75 (shared) and\n");
    std::printf("1.00 (all-FMA, throughput choice) down to 0.50 -- a 1/3 to 1/2 cut,\n");
    std::printf("at the cost of more adds and no FMA contraction. It is intrinsic to\n");
    std::printf("the Chebyshev basis: ordinary FFT twiddles admit no v^2 = 1 - u^2.\n");
    return 0;
}
