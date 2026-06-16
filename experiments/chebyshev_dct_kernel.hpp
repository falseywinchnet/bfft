#ifndef BFFT_CHEBYSHEV_DCT_KERNEL_HPP
#define BFFT_CHEBYSHEV_DCT_KERNEL_HPP

// Chebyshev DCT/DST endpoints — reference kernels (ROADMAP Phases 1-4).
//
// Why this file exists
// --------------------
// capstone.md established that the BFFT real-FFT bin is X[k] = C[k] - i*S[k]
// where C is a cosine projection (Chebyshev-T) and S is a sine projection
// (Chebyshev-U). The cosine tree alone is a DCT; the sine tree alone is a DST.
// Each is ~half an FFT because it never builds the half it does not need.
//
// This header is the *reference* layer the ROADMAP asks for first: a direct,
// obviously-correct scalar definition of every DCT/DST type (forward + inverse)
// matching FFTW's REDFT*/RODFT* conventions exactly, so it can serve as the
// oracle for the fused radix-4 kernels that follow. It then connects the cosine
// family to the project's *existing* radix-4 Chebyshev evaluation tree
// (experiments/chebyshev_composed_radix4_kernel.hpp, eval_monomial) by way of
// the flat-Chebyshev <-> nested-tower coefficient repack, and runs a genuine
// DCT-III through that tree (dct3_via_tree). That proves the ROADMAP's claim — "DCT-III
// is evaluation of a Chebyshev series at the roots of T_N" — end to end, on the
// kernel the FFT path already ships.
//
// Two distinct boundary costs appear here, and they are NOT the same:
//   * input coefficient repack (flat_to_tower): O(N log N) structured work, the
//     real-space analog of the FFT's bit-reversal-with-twiddles, needed because
//     the tree consumes the nested power-of-T_q tower, not a flat T_j series;
//   * output reorder (leaf order -> DCT bin order): the genuine O(N) permutation
//     the ROADMAP calls cheap. leaf_to_index below builds it by O(N^2) nearest-
//     node matching for the reference; a shipped kernel emits it analytically.
//
// Conventions (FFTW, unnormalized)
// --------------------------------
// For an input X[0..n-1] the eight transforms produce Y[0..n-1]:
//
//   DCT-I  (REDFT00): Y_k = X_0 + (-1)^k X_{n-1}
//                          + 2 sum_{j=1}^{n-2} X_j cos(pi j k/(n-1))
//   DCT-II (REDFT10): Y_k = 2 sum_{j=0}^{n-1} X_j cos(pi (j+1/2) k / n)
//   DCT-III(REDFT01): Y_k = X_0 + 2 sum_{j=1}^{n-1} X_j cos(pi j (k+1/2)/n)
//   DCT-IV (REDFT11): Y_k = 2 sum_{j=0}^{n-1} X_j cos(pi (j+1/2)(k+1/2)/n)
//   DST-I  (RODFT00): Y_k = 2 sum_{j=0}^{n-1} X_j sin(pi (j+1)(k+1)/(n+1))
//   DST-II (RODFT10): Y_k = 2 sum_{j=0}^{n-1} X_j sin(pi (j+1/2)(k+1)/n)
//   DST-III(RODFT01): Y_k = (-1)^k X_{n-1}
//                          + 2 sum_{j=0}^{n-2} X_j sin(pi (j+1)(k+1/2)/n)
//   DST-IV (RODFT11): Y_k = 2 sum_{j=0}^{n-1} X_j sin(pi (j+1/2)(k+1/2)/n)
//
// Inverse normalization (forward then inverse = identity):
//   DCT-I  : self-inverse / (2(n-1))
//   DCT-II : inverse is DCT-III / (2n)
//   DCT-III: inverse is DCT-II  / (2n)
//   DCT-IV : self-inverse / (2n)
//   DST-I  : self-inverse / (2(n+1))
//   DST-II : inverse is DST-III / (2n)
//   DST-III: inverse is DST-II  / (2n)
//   DST-IV : self-inverse / (2n)
//
// The DCT-III <-> Chebyshev identity
// ----------------------------------
// The nodes t_k = cos(pi (2k+1)/(2n)) are exactly the roots of T_n, and
// cos(pi j (k+1/2)/n) = cos(j*acos(t_k)) = T_j(t_k). So DCT-III(X) is the
// multipoint evaluation of the Chebyshev series with coefficients
//   c_0 = X_0,  c_j = 2 X_j  (j >= 1)
// at the n roots of T_n. The radix-4 tree eval_monomial computes exactly that
// multipoint evaluation, once the flat coefficients c are repacked into the
// tree's nested power-of-T_q tower layout (flat_to_tower below). That input
// repack is O(N log N) structured work (the real-space bit-reversal analog),
// distinct from the cheap O(N) output permutation back to DCT bin order.

#include <cmath>
#include <cstddef>
#include <vector>

#include "chebyshev_composed_radix4_kernel.hpp"  // eval_monomial, leaf_x, node_const, is_pow4

namespace cheb_dct {

constexpr double kPi = 3.141592653589793238462643383279502884;

// ===========================================================================
// 1. Direct (matrix) reference transforms — the oracle.
//    Each takes X[0..n-1] and writes Y[0..n-1]. O(n^2), correctness-first.
// ===========================================================================

inline void dct1_forward(const double* X, double* Y, int n) {
    // REDFT00, logical period 2(n-1). Requires n >= 2.
    const double L = static_cast<double>(n - 1);
    for (int k = 0; k < n; ++k) {
        double acc = X[0] + ((k & 1) ? -X[n - 1] : X[n - 1]);
        for (int j = 1; j < n - 1; ++j)
            acc += 2.0 * X[j] * std::cos(kPi * j * k / L);
        Y[k] = acc;
    }
}

inline void dct2_forward(const double* X, double* Y, int n) {
    // REDFT10, logical period 2n.
    const double N = static_cast<double>(n);
    for (int k = 0; k < n; ++k) {
        double acc = 0.0;
        for (int j = 0; j < n; ++j)
            acc += X[j] * std::cos(kPi * (j + 0.5) * k / N);
        Y[k] = 2.0 * acc;
    }
}

inline void dct3_forward(const double* X, double* Y, int n) {
    // REDFT01, logical period 2n. Inverse (up to 2n) of DCT-II.
    const double N = static_cast<double>(n);
    for (int k = 0; k < n; ++k) {
        double acc = X[0];
        for (int j = 1; j < n; ++j)
            acc += 2.0 * X[j] * std::cos(kPi * j * (k + 0.5) / N);
        Y[k] = acc;
    }
}

inline void dct4_forward(const double* X, double* Y, int n) {
    // REDFT11, logical period 2n. Self-inverse up to 2n.
    const double N = static_cast<double>(n);
    for (int k = 0; k < n; ++k) {
        double acc = 0.0;
        for (int j = 0; j < n; ++j)
            acc += X[j] * std::cos(kPi * (j + 0.5) * (k + 0.5) / N);
        Y[k] = 2.0 * acc;
    }
}

inline void dst1_forward(const double* X, double* Y, int n) {
    // RODFT00, logical period 2(n+1). Self-inverse up to 2(n+1).
    const double L = static_cast<double>(n + 1);
    for (int k = 0; k < n; ++k) {
        double acc = 0.0;
        for (int j = 0; j < n; ++j)
            acc += X[j] * std::sin(kPi * (j + 1) * (k + 1) / L);
        Y[k] = 2.0 * acc;
    }
}

inline void dst2_forward(const double* X, double* Y, int n) {
    // RODFT10, logical period 2n.
    const double N = static_cast<double>(n);
    for (int k = 0; k < n; ++k) {
        double acc = 0.0;
        for (int j = 0; j < n; ++j)
            acc += X[j] * std::sin(kPi * (j + 0.5) * (k + 1) / N);
        Y[k] = 2.0 * acc;
    }
}

inline void dst3_forward(const double* X, double* Y, int n) {
    // RODFT01, logical period 2n. Inverse (up to 2n) of DST-II.
    const double N = static_cast<double>(n);
    for (int k = 0; k < n; ++k) {
        double acc = (k & 1) ? -X[n - 1] : X[n - 1];
        for (int j = 0; j < n - 1; ++j)
            acc += 2.0 * X[j] * std::sin(kPi * (j + 1) * (k + 0.5) / N);
        Y[k] = acc;
    }
}

inline void dst4_forward(const double* X, double* Y, int n) {
    // RODFT11, logical period 2n. Self-inverse up to 2n.
    const double N = static_cast<double>(n);
    for (int k = 0; k < n; ++k) {
        double acc = 0.0;
        for (int j = 0; j < n; ++j)
            acc += X[j] * std::sin(kPi * (j + 0.5) * (k + 0.5) / N);
        Y[k] = 2.0 * acc;
    }
}

// ---------------------------------------------------------------------------
// Inverses. Each reuses the forward of the inverse type and rescales so that
// inverse(forward(x)) == x. Convenience std::vector wrappers below.
// ---------------------------------------------------------------------------
inline void scale(double* Y, int n, double s) {
    for (int i = 0; i < n; ++i) Y[i] *= s;
}

inline void dct1_inverse(const double* X, double* Y, int n) {
    dct1_forward(X, Y, n);
    scale(Y, n, 1.0 / (2.0 * (n - 1)));
}
inline void dct2_inverse(const double* X, double* Y, int n) {
    dct3_forward(X, Y, n);
    scale(Y, n, 1.0 / (2.0 * n));
}
inline void dct3_inverse(const double* X, double* Y, int n) {
    dct2_forward(X, Y, n);
    scale(Y, n, 1.0 / (2.0 * n));
}
inline void dct4_inverse(const double* X, double* Y, int n) {
    dct4_forward(X, Y, n);
    scale(Y, n, 1.0 / (2.0 * n));
}
inline void dst1_inverse(const double* X, double* Y, int n) {
    dst1_forward(X, Y, n);
    scale(Y, n, 1.0 / (2.0 * (n + 1)));
}
inline void dst2_inverse(const double* X, double* Y, int n) {
    dst3_forward(X, Y, n);
    scale(Y, n, 1.0 / (2.0 * n));
}
inline void dst3_inverse(const double* X, double* Y, int n) {
    dst2_forward(X, Y, n);
    scale(Y, n, 1.0 / (2.0 * n));
}
inline void dst4_inverse(const double* X, double* Y, int n) {
    dst4_forward(X, Y, n);
    scale(Y, n, 1.0 / (2.0 * n));
}

using Vec = std::vector<double>;

// ===========================================================================
// 2. Chebyshev series evaluation (Clenshaw) and the T_n nodes.
// ===========================================================================

// Evaluate sum_{j=0}^{m-1} c_j T_j(x) by Clenshaw recurrence.
inline double clenshaw(const double* c, int m, double x) {
    double b1 = 0.0, b2 = 0.0;
    for (int j = m - 1; j >= 1; --j) {
        const double b0 = 2.0 * x * b1 - b2 + c[j];
        b2 = b1;
        b1 = b0;
    }
    return x * b1 - b2 + c[0];
}

// The n roots of T_n, in increasing index order: t_k = cos(pi (2k+1)/(2n)).
inline Vec cheb_nodes(int n) {
    Vec t(n);
    for (int k = 0; k < n; ++k) t[k] = std::cos(kPi * (2.0 * k + 1.0) / (2.0 * n));
    return t;
}

// DCT-III as flat-Chebyshev multipoint evaluation. Builds c (c0=X0, cj=2Xj)
// and evaluates at the T_n nodes in index order. Equal to dct3_forward.
inline void dct3_via_clenshaw(const double* X, double* Y, int n) {
    Vec c(n);
    c[0] = X[0];
    for (int j = 1; j < n; ++j) c[j] = 2.0 * X[j];
    const Vec t = cheb_nodes(n);
    for (int k = 0; k < n; ++k) Y[k] = clenshaw(c.data(), n, t[k]);
}

// ===========================================================================
// 3. Flat-Chebyshev <-> nested-tower repack (the radix-4 boundary work).
//
//    The tree eval_monomial reads coefficients in the nested power-of-T_q
//    tower: f = g0 + T_q g1 + T_q^2 g2 + T_q^3 g3, q = m/4, each g_i the same
//    form on q. flat_to_tower converts a flat Chebyshev series (degrees 0..m-1)
//    into that layout; tower_to_flat is its exact inverse. m must be a power
//    of four.
// ===========================================================================

// Multiply a flat Chebyshev series g (length q) by T_q^p (p in {1,2,3}) and add
// the result, scaled by `s`, into the flat series f (length 4q). Uses
//   T_q     = T_q
//   T_q^2   = (T_{2q} + T_0) / 2
//   T_q^3   = (T_{3q} + 3 T_q) / 4
// and the product-to-sum rule T_a T_r = (T_{a+r} + T_{|a-r|}) / 2.
inline void add_T_single(double* f, const double* g, int q, int a, double s) {
    // f += s * T_a * g, with g of degree < q and a >= q so no aliasing past 0
    // for the dominant band; |a-r| handled by abs.
    for (int r = 0; r < q; ++r) {
        const double v = 0.5 * s * g[r];
        f[a + r] += v;
        f[std::abs(a - r)] += v;
    }
}

inline void add_Tpow(double* f, const double* g, int q, int p, double s) {
    if (p == 1) {
        add_T_single(f, g, q, q, s);
    } else if (p == 2) {
        add_T_single(f, g, q, 2 * q, 0.5 * s);  // (T_{2q})/2 part
        for (int r = 0; r < q; ++r) f[r] += 0.5 * s * g[r];  // (T_0)/2 part
    } else {  // p == 3
        add_T_single(f, g, q, 3 * q, 0.25 * s);  // (T_{3q})/4 part
        add_T_single(f, g, q, q, 0.75 * s);      // (3 T_q)/4 part
    }
}

// Synthesis: nested tower (length m) -> flat Chebyshev coefficients (length m).
inline void tower_to_flat(const double* tower, int off, int m, double* flat) {
    if (m == 1) { flat[0] = tower[off]; return; }
    const int q = m / 4;
    Vec g0(q), g1(q), g2(q), g3(q);
    tower_to_flat(tower, off,         q, g0.data());
    tower_to_flat(tower, off + q,     q, g1.data());
    tower_to_flat(tower, off + 2 * q, q, g2.data());
    tower_to_flat(tower, off + 3 * q, q, g3.data());
    for (int i = 0; i < m; ++i) flat[i] = 0.0;
    for (int r = 0; r < q; ++r) flat[r] += g0[r];  // g0 * T_q^0
    add_Tpow(flat, g1.data(), q, 1, 1.0);
    add_Tpow(flat, g2.data(), q, 2, 1.0);
    add_Tpow(flat, g3.data(), q, 3, 1.0);
}

// Analysis: flat Chebyshev coefficients (length m) -> nested tower (length m).
// Back-substitution from the top Chebyshev band. The top quarter [3q,4q) is
// produced only by T_q^3 g3, the band [2q,3q) only by T_q^2 g2 (after g3 is
// removed), [q,2q) only by T_q g1, and [0,q) is g0. The recovery factors come
// straight from the leading coefficients in add_Tpow:
//   c[3q]     = (1/4) b3_0          c[3q+r]   = (1/8) b3_r   (r>=1)
//   c'[2q]    = (1/2) b2_0          c'[2q+r]  = (1/4) b2_r   (r>=1)
//   c''[q]    =        b1_0         c''[q+r]  = (1/2) b1_r   (r>=1)
//   c'''[r]   =        b0_r
inline void flat_to_tower(const double* flat, int off, int m, double* tower) {
    if (m == 1) { tower[off] = flat[0]; return; }
    const int q = m / 4;
    Vec c(flat, flat + m);  // working copy

    Vec g3(q), g2(q), g1(q), g0(q);

    g3[0] = 4.0 * c[3 * q];
    for (int r = 1; r < q; ++r) g3[r] = 8.0 * c[3 * q + r];
    add_Tpow(c.data(), g3.data(), q, 3, -1.0);  // remove g3's full contribution

    g2[0] = 2.0 * c[2 * q];
    for (int r = 1; r < q; ++r) g2[r] = 4.0 * c[2 * q + r];
    add_Tpow(c.data(), g2.data(), q, 2, -1.0);

    g1[0] = c[q];
    for (int r = 1; r < q; ++r) g1[r] = 2.0 * c[q + r];
    add_Tpow(c.data(), g1.data(), q, 1, -1.0);

    for (int r = 0; r < q; ++r) g0[r] = c[r];

    // Recurse into each block, writing into the matching tower sub-range.
    flat_to_tower(g0.data(), 0, q, tower + off);
    flat_to_tower(g1.data(), 0, q, tower + off + q);
    flat_to_tower(g2.data(), 0, q, tower + off + 2 * q);
    flat_to_tower(g3.data(), 0, q, tower + off + 3 * q);
}

// ===========================================================================
// 4. DCT-III through the existing radix-4 Chebyshev tree (the hot path proof).
//
//    Pipeline: X -> flat Chebyshev coeffs c -> tower (flat_to_tower) ->
//    eval_monomial (the shipped radix-4 node recursion) -> leaves at leaf_x ->
//    permute leaves into k order. n must be a power of four.
// ===========================================================================

// Permutation P with leaf_x[i] == nodes[P[i]]; i.e. leaf index -> DCT-III index.
inline std::vector<int> leaf_to_index(int n) {
    Vec xleaf(n);
    cheb_composed_r4::leaf_x(0, n, 0.0, xleaf.data());
    const Vec nodes = cheb_nodes(n);  // nodes are strictly decreasing in k
    std::vector<int> perm(n);
    for (int i = 0; i < n; ++i) {
        // nodes[k] = cos(pi(2k+1)/2n): decreasing, so larger x -> smaller k.
        // Match by nearest value (nodes are well separated).
        int best = 0;
        double bestd = 1e300;
        for (int k = 0; k < n; ++k) {
            const double d = std::abs(xleaf[i] - nodes[k]);
            if (d < bestd) { bestd = d; best = k; }
        }
        perm[i] = best;
    }
    return perm;
}

inline void dct3_via_tree(const double* X, double* Y, int n) {
    Vec c(n);
    c[0] = X[0];
    for (int j = 1; j < n; ++j) c[j] = 2.0 * X[j];

    Vec tower(n);
    flat_to_tower(c.data(), 0, n, tower.data());

    // eval_monomial reduces in place, producing leaf evaluations in leaf order.
    cheb_composed_r4::eval_monomial(tower.data(), 0, n, 0.0);

    const std::vector<int> perm = leaf_to_index(n);
    for (int i = 0; i < n; ++i) Y[perm[i]] = tower[i];
}

}  // namespace cheb_dct

#endif  // BFFT_CHEBYSHEV_DCT_KERNEL_HPP
