#ifndef BFFT_CHEBYSHEV_COMPOSED_RADIX4_KERNEL_HPP
#define BFFT_CHEBYSHEV_COMPOSED_RADIX4_KERNEL_HPP

// Composed-Chebyshev radix-4 Bruun node — arithmetic exploration.
//
// Goal
// ----
// The shipped fused radix-4 node (norm2_fused, experiments/bruun_radix4_kernel.hpp)
// costs 20q flops for q lanes. The monomial Chebyshev even/odd split
// (chebyshev_radix4_split) drops that to 14q. This header pushes to a 12q node
// by representing the local cubic in a *composed* Chebyshev basis so the inner
// T_2 layer is shared between the two even/odd halves.
//
// Three node tiers, for reference:
//
//   normalized fused (norm2_fused)        : 8 mul + 12 add = 20q
//   monomial Chebyshev (even/odd)         : 6 mul +  8 add = 14q
//   composed Chebyshev (this file)        : 4 mul +  8 add = 12q
//
// Leading local coefficient of the composed node:
//   12 / (4 * log2(4)) = 12 / 8 = 1.5
//
// Algebra
// -------
// A node reduces a cubic residue f(y) modulo T_4(y) - a, where the four real
// roots are y in {+u, -u, +v, -v}. Instead of the monomial cubic
//
//     f(y) = a0 + a1 y + a2 y^2 + a3 y^3,
//
// store it in the composed basis built on z = T_2(y) = 2 y^2 - 1:
//
//     f(y) = P(z) + y Q(z),   P(z) = p0 + p1 z,   Q(z) = q0 + q1 z.
//
// The roots of T_4(y) - a = T_2(z) - a are z = +/- w with w = sqrt((a + 1) / 2),
// and then T_2(y) = +w gives y = +/- u, T_2(y) = -w gives y = +/- v, with
// u = sqrt((1 + w) / 2), v = sqrt((1 - w) / 2). So:
//
//     Pp = P(+w) = p0 + w p1      Pm = P(-w) = p0 - w p1
//     Qp = Q(+w) = q0 + w q1      Qm = Q(-w) = q0 - w q1
//     f(+u) = Pp + u Qp           f(-u) = Pp - u Qp
//     f(+v) = Pm + v Qm           f(-v) = Pm - v Qm
//
// Per lane: w*p1, w*q1, u*Qp, v*Qm = 4 mul; Pp,Pm,Qp,Qm + 4 outputs = 8 add.
//
// Basis transition (monomial <-> composed), per lane:
//     p1 = a2 / 2     p0 = a0 + a2 / 2
//     q1 = a3 / 2     q0 = a1 + a3 / 2
// and back:
//     a2 = 2 p1       a0 = p0 - p1
//     a3 = 2 q1       a1 = q0 - q1
//
// The 12q win is only realized if the recursion *stays* in the composed basis
// so this conversion is not paid per node. The node outputs (the four
// evaluations) are scalar child coefficients in monomial form, so a full
// composed-basis transform must fuse the conversion into the child store
// (see the design note at the bottom of this file). This header provides the
// node primitives and a scalar reference recursion used to validate them.
//
// Empirical result (arm64, clang -O3, FMA; loop-body op counts)
// --------------------------------------------------------------
// Flops are the wrong cost model once FMA contraction is in play; what tracks
// runtime is the count of contracted ops in the loop body and the critical
// path. Measured loop bodies:
//
//   node                 flops  loose mul/add  fmla  body ops  ns/lane(rel)
//   composed 12q           12       4 + 8        0      12      ~0.88x mono14
//   composed 13q           13       3 + 6        1      10      ~0.95x mono14
//   monomial  14q (shared) 14       6 + 8        4      18(2x)  1.00x
//   monomial  all-FMA       -       0 + 0        8       8      ~1.05-1.08x
//   composed  all-FMA       -       0 + 0        8       8      ~1.05x mono14
//
// Why all-FMA wins, and why 8 is the floor:
//   * Every +/- output pair must NOT share its scale product. Sharing su = u*ou
//     between eu+su and eu-su forces mul + add + sub (3 ops) where fma + fnma
//     (2 ops) suffice. So duplicate the product; let each output contract.
//   * Each of the 4 outputs needs its own scale-and-combine -> 4 output FMAs are
//     unavoidable. The even/odd reduction needs eu,ou,ev,ov -> 4 more FMAs.
//     8 total, and no factorization of this 4x4 Vandermonde-on-{+/-u,+/-v} map
//     beats it: any shared product trades 2 FMAs for >=3 loose ops.
//   * Critical path is 2 FMA latencies (input -> ou -> out), with 4 independent
//     even/odd chains for ILP. The 12q form's path is 4 and its product sharing
//     kills contraction entirely (0 fmla, 12 ops) -- fewest flops, most ops.
//
// Conclusion: the optimum is not a lower flop count but the all-FMA even/odd
// node (monomial_node_fma / composed_node_fma). In that FMA-contracted limit
// the monomial and composed bases CONVERGE to the same 8-FMA body, so the basis
// choice no longer affects node cost -- it only decides whether a
// monomial<->composed conversion is owed at the tree boundary. The earlier
// 20q -> 14q -> 12q "flop ladder" was a red herring; 20q -> 8-FMA is the real
// jump, driven by instruction shape, not flop count.
//
// Can a deferred / cross-coupled representation go below 8 FMA per node?
// ---------------------------------------------------------------------
// Idea: carry the even/odd intermediate (eu,ou,ev,ov) through the tree and pay
// the four +/- "output" FMAs only at a boundary. The cross-coupling identities
// are real (u^2 + v^2 = 1, so eu+ev = 2a0+a2, eu-ev = w*a2, etc.), but they
// reorganize work rather than delete it: reconstructing the four signed
// children still costs the scale-and-combine.
//
// The hard obstruction is sibling asymmetry. Children 0 (a'=+u) and 1 (a'=-u)
// share (eu,ou), but their NEXT-level constants w0 = sqrt((u+1)/2) and
// w1 = sqrt((1-u)/2) differ (w0=0.98, w1=0.20 at the top node). A complex FFT's
// conjugate siblings run identical sub-DFTs and can share; here the real Bruun/
// Chebyshev factors are distinct, so a deferred intermediate cannot be pushed
// down without replicating both subtrees. Internal deferral is therefore
// impossible: the materializing recursion costs exactly 2 N log4(N) FMA.
//
// Only the BOTTOM level can be deferred: storing (eu,ou,ev,ov) at the N/4 leaf
// nodes skips their 4 output FMAs each, saving exactly N FMA -- a constant
// 1/(2 log4 N) fraction (25% at N=16, 6.25% at N=65536). It is loss-free but
// valid only when the consumer accepts the even/odd pair form rather than the
// four signed values (a scaled-native or DCT-projected boundary). That single
// deferrable level is the entire prize of the cross-coupled representation.
// See experiments/cheb_deferred_probe.cpp for the FMA accounting and proof.
//
// A SECOND, orthogonal axis: minimum-multiply via u^2 + v^2 = 1 (lowmul_node)
// ---------------------------------------------------------------------------
// Throughput and multiply-complexity are different optima. The 8-FMA node
// minimizes instruction count (8 ops) but issues 8 multiplies. The Chebyshev
// identity v^2 = 1 - u^2 instead minimizes MULTIPLIES: u^2*a2 serves both eu
// and ev, u^2*a3 serves both ou and ov, leaving 4 mul + 10 add (lowmul_node).
// Whole-transform real-multiply counts (experiments/cheb_multiply_probe.cpp):
//
//   form                 mul/node   total multiplies     wall-clock (FMA HW)
//   all-FMA (throughput)    8        1.00 * N log2 N      fastest (8 ops)
//   shared-su               6        0.75 * N log2 N      mid
//   low-mul (v^2=1-u^2)     4        0.50 * N log2 N      slowest (14 ops)
//
// So the representation can shed HALF its multiplies versus the throughput
// form -- a real cut in the leading multiply constant, intrinsic to the
// Chebyshev basis (ordinary FFT twiddles W^k admit no v^2 = 1 - u^2). It is a
// Pareto trade, not a free win: fewer multiplies cost more adds and forfeit FMA
// fusion, so low-mul is slower wall-clock on FMA hardware but preferable where
// multiplies dominate (no/!weak FMA, fixed-point, or counting omega). 4 mul is
// the floor for the nested even/odd factorization: each 2x2 even/odd block
// needs 1 essential mul and each +/- recombination needs 1; going lower needs a
// global Winograd bilinear module over the Chebyshev points (open, add-heavy).
// Higher radix (8, 16, ...) does NOT lower mul/node: the saving is per-node and
// the data multiplied differs at every level, so only memory traffic improves.
//
// Two further caveats on low-mul, both measured:
//   * Stability (experiments/cheb_stability_probe.cpp): the subtraction
//     ev = (a0+a2) - u^2 a2 adds rounding steps (3 vs the FMA's 1), but its
//     cancellation condition is identical to the stable form -- both lose
//     precision only when ev = a0 + v^2 a2 is itself ~0. Measured worst-case
//     relative error is ~2-3x the all-FMA form (1e-14 class, not catastrophic);
//     mean error is essentially equal. So low-mul is mildly, not dangerously,
//     less accurate.
//   * Port pressure: on cores with separate FP add/mul issue ports, 4 mul +
//     10 add is add-port-bound while 8 FMA balances both, so the all-FMA node
//     can win wall-clock by even more than its op-count edge suggests. The
//     multiply-count advantage of low-mul only converts to speed on genuinely
//     multiply-bound hardware (weak/no FMA, fixed-point) or when minimizing the
//     theoretical multiply constant omega rather than cycles.

#include <cmath>
#include <cstddef>

namespace cheb_composed_r4 {

// ---------------------------------------------------------------------------
// Cost model.
// ---------------------------------------------------------------------------
struct NodeCost {
    std::size_t multiplications;
    std::size_t additions;
    std::size_t flops;
};

inline NodeCost normalized_fused_cost(std::size_t q) {
    return {8 * q, 12 * q, 20 * q};
}
inline NodeCost monomial_cheb_cost(std::size_t q) {
    return {6 * q, 8 * q, 14 * q};
}
inline NodeCost composed_cheb_cost(std::size_t q) {
    return {4 * q, 8 * q, 12 * q};
}
inline NodeCost composed_qdup_cost(std::size_t q) {
    return {5 * q, 8 * q, 13 * q};
}

// ---------------------------------------------------------------------------
// Node geometry: derive (u, v, w) from a node's a-value.
// ---------------------------------------------------------------------------
struct NodeConst { double u, v, w; };

inline NodeConst node_const(double a) {
    const double w = std::sqrt((a + 1.0) * 0.5);
    const double u = std::sqrt((1.0 + w) * 0.5);
    const double v = std::sqrt((1.0 - w) * 0.5);
    return {u, v, w};
}

// ---------------------------------------------------------------------------
// Monomial Chebyshev node (14q). Block layout p = [a0 | a1 | a2 | a3], each q.
// In place -> [f(+u) | f(-u) | f(+v) | f(-v)].
// ---------------------------------------------------------------------------
inline void monomial_node(double* p, int q, double u, double v) {
    const double u2 = u * u, v2 = v * v;
    double* a0 = p; double* a1 = p + q; double* a2 = p + 2 * q; double* a3 = p + 3 * q;
    for (int n = 0; n < q; ++n) {
        const double eu = a0[n] + a2[n] * u2;
        const double ou = a1[n] + a3[n] * u2;
        const double ev = a0[n] + a2[n] * v2;
        const double ov = a1[n] + a3[n] * v2;
        const double su = u * ou, sv = v * ov;
        a0[n] = eu + su;   // f(+u)
        a1[n] = eu - su;   // f(-u)
        a2[n] = ev + sv;   // f(+v)
        a3[n] = ev - sv;   // f(-v)
    }
}

// ---------------------------------------------------------------------------
// Monomial Chebyshev node, all-FMA form. Same layout/result as monomial_node,
// but the scale-by-u/v product is *duplicated* into the +/- outputs so each
// output contracts to a single fma/fnma instead of being a shared mul feeding
// a loose add and a loose sub. Idealized op stream:
//
//   eu = fma(a2,u2,a0)   ou = fma(a3,u2,a1)      (2 fma)
//   ev = fma(a2,v2,a0)   ov = fma(a3,v2,a1)      (2 fma)
//   f(+u)=fma(u,ou,eu)   f(-u)=fnma(u,ou,eu)     (2 fma)
//   f(+v)=fma(v,ov,ev)   f(-v)=fnma(v,ov,ev)     (2 fma)
//
// 8 fused ops, zero loose add/mul, critical path 2 fma latencies, 4-wide ILP.
// This is the floor: each of the 4 outputs needs its own scale+combine, so 4
// output FMAs are unavoidable, and sharing u*ou would cost 3 ops (mul+add+sub)
// to produce 2 outputs versus 2 FMAs — strictly worse. Writing the product
// twice lets -ffp-contract fuse both; std::fma is avoided so non-FMA targets
// fall back to plain mul/add rather than a slow correctly-rounded libm call.
// ---------------------------------------------------------------------------
inline void monomial_node_fma(double* p, int q, double u, double v) {
    const double u2 = u * u, v2 = v * v;
    double* a0 = p; double* a1 = p + q; double* a2 = p + 2 * q; double* a3 = p + 3 * q;
    for (int n = 0; n < q; ++n) {
        const double A0 = a0[n], A1 = a1[n], A2 = a2[n], A3 = a3[n];
        const double eu = A0 + A2 * u2;
        const double ou = A1 + A3 * u2;
        const double ev = A0 + A2 * v2;
        const double ov = A1 + A3 * v2;
        a0[n] = eu + u * ou;   // f(+u)  -> fma
        a1[n] = eu - u * ou;   // f(-u)  -> fnma (product duplicated, not shared)
        a2[n] = ev + v * ov;   // f(+v)  -> fma
        a3[n] = ev - v * ov;   // f(-v)  -> fnma
    }
}

// ---------------------------------------------------------------------------
// Minimum-multiply radix-4 Chebyshev node (4 mul + 10 add), exploiting the
// Chebyshev identity v^2 = 1 - u^2 (since u^2 = (1+w)/2, v^2 = (1-w)/2). This
// trades FMA-friendliness for HALF the multiplies: the product u^2*a2 serves
// BOTH eu = a0 + u^2 a2 and ev = a0 + v^2 a2 = (a0+a2) - u^2 a2, and likewise
// u^2*a3 serves ou and ov. Only u*ou and v*ov remain as separate scale mults.
//
//   m2 = u^2 * a2                         (mul 1, shared by eu, ev)
//   m3 = u^2 * a3                         (mul 2, shared by ou, ov)
//   eu = a0 + m2     ev = (a0+a2) - m2
//   ou = a1 + m3     ov = (a1+a3) - m3
//   su = u * ou      sv = v * ov          (mul 3, mul 4)
//   f(+u)=eu+su  f(-u)=eu-su  f(+v)=ev+sv  f(-v)=ev-sv
//
// This is an arithmetic-complexity (multiply-count) optimum, NOT a throughput
// one: 4 mul + 10 add = 14 ops with no FMA contraction, so on FMA hardware the
// 8-FMA node is faster wall-clock. But the whole-transform multiply count drops
// from N*log2(N) to (N/2)*log2(N) real multiplies -- a 2x reduction in the
// leading multiply constant, intrinsic to the Chebyshev basis (ordinary FFT
// twiddles W^k satisfy no analogous real quadratic constraint). The identity
// v^2 = 1 - u^2 holds ONLY for these Chebyshev node constants, so this kernel
// must not be used with arbitrary radix-4 twiddles.
// ---------------------------------------------------------------------------
inline void lowmul_node(double* p, int q, double u, double v) {
    const double u2 = u * u;  // node constant; v2 = 1 - u2 is implicit
    double* a0 = p; double* a1 = p + q; double* a2 = p + 2 * q; double* a3 = p + 3 * q;
    for (int n = 0; n < q; ++n) {
        const double A0 = a0[n], A1 = a1[n], A2 = a2[n], A3 = a3[n];
        const double m2 = u2 * A2;          // shared product for eu, ev
        const double m3 = u2 * A3;          // shared product for ou, ov
        const double eu = A0 + m2;
        const double ev = (A0 + A2) - m2;
        const double ou = A1 + m3;
        const double ov = (A1 + A3) - m3;
        const double su = u * ou;
        const double sv = v * ov;
        a0[n] = eu + su;   // f(+u)
        a1[n] = eu - su;   // f(-u)
        a2[n] = ev + sv;   // f(+v)
        a3[n] = ev - sv;   // f(-v)
    }
}

// ---------------------------------------------------------------------------
// Composed Chebyshev node, all-FMA form. The composed basis also collapses to
// 8 FMAs when every +/- pair is written as fma/fnma rather than sharing the
// w-product or the u/v-product:
//
//   Pp=fma(w,p1,p0)  Pm=fnma(w,p1,p0)  Qp=fma(w,q1,q0)  Qm=fnma(w,q1,q0)
//   f(+u)=fma(u,Qp,Pp)  f(-u)=fnma(u,Qp,Pp)
//   f(+v)=fma(v,Qm,Pm)  f(-v)=fnma(v,Qm,Pm)
//
// Same 8-op / path-2 profile as monomial_node_fma; it just reaches it from the
// composed coefficients. So in the FMA-contracted limit monomial and composed
// converge — the basis choice no longer changes the op count, only whether a
// monomial_to_composed conversion is owed at the boundary.
// ---------------------------------------------------------------------------
inline void composed_node_fma(double* p, int q, double u, double v, double w) {
    double* P0 = p; double* P1 = p + q; double* Q0 = p + 2 * q; double* Q1 = p + 3 * q;
    for (int n = 0; n < q; ++n) {
        const double p0 = P0[n], p1 = P1[n], q0 = Q0[n], q1 = Q1[n];
        const double Pp = p0 + w * p1;
        const double Pm = p0 - w * p1;
        const double Qp = q0 + w * q1;
        const double Qm = q0 - w * q1;
        P0[n] = Pp + u * Qp;   // f(+u)
        P1[n] = Pp - u * Qp;   // f(-u)
        Q0[n] = Pm + v * Qm;   // f(+v)
        Q1[n] = Pm - v * Qm;   // f(-v)
    }
}

// ---------------------------------------------------------------------------
// Composed Chebyshev node (12q). Block layout p = [p0 | p1 | q0 | q1], each q.
// In place -> [f(+u) | f(-u) | f(+v) | f(-v)].
// ---------------------------------------------------------------------------
inline void composed_node(double* p, int q, double u, double v, double w) {
    double* P0 = p; double* P1 = p + q; double* Q0 = p + 2 * q; double* Q1 = p + 3 * q;
    for (int n = 0; n < q; ++n) {
        const double wp1 = w * P1[n];
        const double wq1 = w * Q1[n];
        const double Pp = P0[n] + wp1;
        const double Pm = P0[n] - wp1;
        const double Qp = Q0[n] + wq1;
        const double Qm = Q0[n] - wq1;
        const double su = u * Qp;
        const double sv = v * Qm;
        P0[n] = Pp + su;   // f(+u)
        P1[n] = Pp - su;   // f(-u)
        Q0[n] = Pm + sv;   // f(+v)
        Q1[n] = Pm - sv;   // f(-v)
    }
}

// ---------------------------------------------------------------------------
// Composed Chebyshev node, duplicate-Q-only (13q). Same layout/result as
// composed_node, but only the P side shares its w-multiply; the Q side is
// written as two independent FMAs (q0 +/- w*q1) so the critical path that
// feeds the u/v multiplies is FMA -> MUL -> ADD instead of MUL -> ADD -> MUL
// -> ADD. Cost: P side 1 mul + 2 add, Q side 2 mul + 2 add, u/v 2 mul,
// outputs 4 add = 5 mul + 8 add = 13q.
// ---------------------------------------------------------------------------
inline void composed_node_qdup(double* p, int q, double u, double v, double w) {
    double* P0 = p; double* P1 = p + q; double* Q0 = p + 2 * q; double* Q1 = p + 3 * q;
    for (int n = 0; n < q; ++n) {
        const double tP = w * P1[n];          // shared on the non-critical P side
        const double Pp = P0[n] + tP;
        const double Pm = P0[n] - tP;
        const double Qp = Q0[n] + w * Q1[n];  // independent FMA
        const double Qm = Q0[n] - w * Q1[n];  // independent FMA (fnma)
        const double su = u * Qp;
        const double sv = v * Qm;
        P0[n] = Pp + su;   // f(+u)
        P1[n] = Pp - su;   // f(-u)
        Q0[n] = Pm + sv;   // f(+v)
        Q1[n] = Pm - sv;   // f(-v)
    }
}

// ---------------------------------------------------------------------------
// Composed Chebyshev node, fully duplicated (14q). Both P and Q sides written
// as independent FMA pairs. Same flop count as the monomial node (6 mul +
// 8 add), so comparing the two isolates basis from instruction shape.
// ---------------------------------------------------------------------------
inline void composed_node_dup(double* p, int q, double u, double v, double w) {
    double* P0 = p; double* P1 = p + q; double* Q0 = p + 2 * q; double* Q1 = p + 3 * q;
    for (int n = 0; n < q; ++n) {
        const double Pp = P0[n] + w * P1[n];
        const double Pm = P0[n] - w * P1[n];
        const double Qp = Q0[n] + w * Q1[n];
        const double Qm = Q0[n] - w * Q1[n];
        const double su = u * Qp;
        const double sv = v * Qm;
        P0[n] = Pp + su;
        P1[n] = Pp - su;
        Q0[n] = Pm + sv;
        Q1[n] = Pm - sv;
    }
}

// ---------------------------------------------------------------------------
// Basis transitions over a block p = [b0 | b1 | b2 | b3], each q, in place.
// ---------------------------------------------------------------------------
inline void monomial_to_composed(double* p, int q) {
    double* a0 = p; double* a1 = p + q; double* a2 = p + 2 * q; double* a3 = p + 3 * q;
    for (int n = 0; n < q; ++n) {
        const double A0 = a0[n], A1 = a1[n], A2 = a2[n], A3 = a3[n];
        const double h2 = 0.5 * A2, h3 = 0.5 * A3;
        a0[n] = A0 + h2;   // p0  (block 0)
        a1[n] = h2;        // p1  (block 1)
        a2[n] = A1 + h3;   // q0  (block 2)
        a3[n] = h3;        // q1  (block 3)
    }
}

inline void composed_to_monomial(double* p, int q) {
    double* P0 = p; double* P1 = p + q; double* Q0 = p + 2 * q; double* Q1 = p + 3 * q;
    for (int n = 0; n < q; ++n) {
        const double p0 = P0[n], p1 = P1[n], q0 = Q0[n], q1 = Q1[n];
        P0[n] = p0 - p1;   // a0
        P1[n] = q0 - q1;   // a1
        Q0[n] = 2.0 * p1;  // a2
        Q1[n] = 2.0 * q1;  // a3
    }
}

// ---------------------------------------------------------------------------
// Scalar reference recursion (multipoint evaluation at the roots of T_N).
//
// The interior is stored in the nested cubic-in-y monomial tower:
//   F(x) = a0(x) + a1(x) y + a2(x) y^2 + a3(x) y^3,  y = T_{N/4}(x),
// and recursively each a_j is the same form on N/4 with y' = T_{N/16}(x).
// N must be a power of four. The two recursions below run the *same* routing;
// MONOMIAL uses monomial_node directly, COMPOSED converts each block to the
// composed basis and uses composed_node (proving composed_node in-tree).
// ---------------------------------------------------------------------------
inline bool is_pow4(int n) {
    if (n < 1) return false;
    while (n > 1) { if (n & 3) return false; n >>= 2; }
    return true;
}

inline void eval_monomial(double* p, int off, int m, double a) {
    if (m == 1) return;
    const int q = m / 4;
    const NodeConst k = node_const(a);
    monomial_node(p + off, q, k.u, k.v);
    eval_monomial(p, off,           q,  k.u);
    eval_monomial(p, off + q,       q, -k.u);
    eval_monomial(p, off + 2 * q,   q,  k.v);
    eval_monomial(p, off + 3 * q,   q, -k.v);
}

inline void eval_composed(double* p, int off, int m, double a) {
    if (m == 1) return;
    const int q = m / 4;
    const NodeConst k = node_const(a);
    monomial_to_composed(p + off, q);
    composed_node(p + off, q, k.u, k.v, k.w);
    eval_composed(p, off,           q,  k.u);
    eval_composed(p, off + q,       q, -k.u);
    eval_composed(p, off + 2 * q,   q,  k.v);
    eval_composed(p, off + 3 * q,   q, -k.v);
}

// Leaf x-coordinates in the order the recursion emits them.
inline void leaf_x(int off, int m, double a, double* xout) {
    if (m == 1) { xout[off] = a; return; }
    const int q = m / 4;
    const NodeConst k = node_const(a);
    leaf_x(off,           q,  k.u, xout);
    leaf_x(off + q,       q, -k.u, xout);
    leaf_x(off + 2 * q,   q,  k.v, xout);
    leaf_x(off + 3 * q,   q, -k.v, xout);
}

// Direct evaluation of the nested cubic-in-y tower at a point x (oracle).
inline double cheb_T(int n, double x) {
    if (n == 0) return 1.0;
    if (n == 1) return x;
    // For |x| <= 1 the trig form T_n(x) = cos(n acos x) avoids the recurrence's
    // error growth at large degree, keeping this a clean reference oracle.
    if (x >= -1.0 && x <= 1.0) return std::cos(n * std::acos(x));
    double t0 = 1.0, t1 = x;
    for (int i = 2; i <= n; ++i) { const double t2 = 2.0 * x * t1 - t0; t0 = t1; t1 = t2; }
    return t1;
}

inline double eval_direct(const double* C, int off, int m, double x) {
    if (m == 1) return C[off];
    const int q = m / 4;
    const double y = cheb_T(q, x);
    const double e0 = eval_direct(C, off,         q, x);
    const double e1 = eval_direct(C, off + q,     q, x);
    const double e2 = eval_direct(C, off + 2 * q, q, x);
    const double e3 = eval_direct(C, off + 3 * q, q, x);
    return e0 + y * e1 + y * y * e2 + y * y * y * e3;
}

// ---------------------------------------------------------------------------
// Design note: realizing 12q across the whole tree.
// ---------------------------------------------------------------------------
// composed_node reads composed coefficients but writes the four evaluations as
// monomial child sub-blocks (each child block of length q holds a0',a1',a2',a3'
// of length q/4). A full composed-basis transform must avoid the per-node
// monomial_to_composed conversion. That requires fusing the conversion into the
// child store: child p0'[k] = out0[k] + out0[k + q/2]/2 etc., i.e. processing
// lanes k and k + q/2 jointly (the same low/high pairing norm2_fused already
// uses). That fused composed node — emitting composed children directly — is
// the next step; this header proves the arithmetic and routing it depends on.

} // namespace cheb_composed_r4

#endif // BFFT_CHEBYSHEV_COMPOSED_RADIX4_KERNEL_HPP
