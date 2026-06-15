#ifndef BFFT_BRUUN_RADIX4_KERNEL_HPP
#define BFFT_BRUUN_RADIX4_KERNEL_HPP

// Experimental self-contained Bruun radix-4 forward kernel.
//
// Two forward paths:
//   forward_residues()         – breadth-first, one norm_q per tree node
//   forward_residues_radix4()  – depth-first, fused 2-level radix-4 nodes (norm2_fused)
//
// Both produce the same N-element residue vector as BFFT's forward_residues.
// No SIMD – scalar reference only.

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace bruun_radix4 {

static inline bool is_pow2(int n) { return n > 0 && (n & (n - 1)) == 0; }
static inline int ilog2(int n) { int l = 0; while (n > 1) { n >>= 1; ++l; } return l; }

static inline int gray_decode(int g) {
    for (int s = 1; s < 32; s <<= 1) g ^= g >> s;
    return g;
}

static inline int bit_rev(int r, int t) {
    int out = 0;
    for (int i = 0; i < t; ++i) { out = (out << 1) | (r & 1); r >>= 1; }
    return out;
}

static inline int bruun_idx(int m, int L) {
    int t = 0;
    for (int x = m; x > 1; x >>= 1) ++t;
    int r = m ^ (1 << t);
    return (2 * gray_decode(bit_rev(r, t)) + 1) << ((L - 2) - t);
}

// ---------------------------------------------------------------------------
// Chebyshev radix-4 even/odd split: 14q flops for q coefficient lanes.
//
// f(y) = a0 + a1 y + a2 y^2 + a3 y^3
// E(y^2) = a0 + a2 y^2,  O(y^2) = a1 + a3 y^2
// f(±u) = E(u²) ± u O(u²),   f(±v) = E(v²) ± v O(v²)
// ---------------------------------------------------------------------------
struct Cheb4Const { double u, u2, v, v2; };

static inline void chebyshev_radix4_split(
    const double* a0, const double* a1,
    const double* a2, const double* a3,
    double* pu, double* mu, double* pv, double* mv,
    int q, double u, double u2, double v, double v2)
{
    for (int n = 0; n < q; ++n) {
        double eu  = a0[n] + a2[n] * u2;
        double ou  = a1[n] + a3[n] * u2;
        double sou = u * ou;
        double ev  = a0[n] + a2[n] * v2;
        double ov  = a1[n] + a3[n] * v2;
        double sov = v * ov;
        pu[n] = eu + sou;
        mu[n] = eu - sou;
        pv[n] = ev + sov;
        mv[n] = ev - sov;
    }
}

// ---------------------------------------------------------------------------
// RFFT_Radix4 – self-contained forward residue transform
// ---------------------------------------------------------------------------
class RFFT_Radix4 {
public:
    RFFT_Radix4() = default;
    ~RFFT_Radix4() { std::free(C_); }

    RFFT_Radix4(const RFFT_Radix4&) = delete;
    RFFT_Radix4& operator=(const RFFT_Radix4&) = delete;

    bool init(int n) {
        if (!is_pow2(n) || n < 16) return false;
        N_ = n;
        L_ = ilog2(n);
        C_ = static_cast<double*>(std::calloc(n / 2, sizeof(double)));
        if (!C_) return false;

        if (n >= 4) C_[1] = std::sqrt(0.5);
        for (int m = 1; 2 * m < n / 2; ++m) {
            double c = C_[m], s = stw(m);
            double ce = std::sqrt(0.5 * (1.0 + c));
            double se = s / (2.0 * ce);
            C_[2 * m] = ce;
            if (2 * m + 1 < n / 2) C_[2 * m + 1] = se;
        }
        return true;
    }

    int size() const { return N_; }

    // Breadth-first forward residue transform (single-level norm_q per node).
    void forward_residues(const double* input, double* v) const {
        std::memcpy(v, input, sizeof(double) * N_);
        for (int jj = 0; jj < L_ - 1; ++jj)
            fwd_stage(v, jj);
    }

    // Depth-first forward using fused 2-level radix-4 nodes where possible.
    void forward_residues_radix4(const double* input, double* v) const {
        if (N_ < 64) { forward_residues(input, v); return; }

        const int h = N_ / 2;
        for (int i = 0; i < h; ++i) {
            v[i]     = input[i] + input[h + i];
            v[h + i] = input[i] - input[h + i];
        }

        for (int sp = h; sp >= 32; sp >>= 1) {
            subtree_r4(v + sp, sp >> 2, 1);
            binomial(v, sp >> 1);
        }
        spine_tail(v);
    }

    // Accessor for twiddle constants (for per-node tests).
    double cosval(int m) const { return C_[m]; }
    double sinval(int m) const { return stw(m); }

    Cheb4Const cheb4_const(int m) const {
        Cheb4Const k;
        k.u  = C_[4 * m];
        k.u2 = 0.5 * (1.0 + C_[2 * m]);
        k.v  = C_[4 * m + 1];
        k.v2 = 0.5 * (1.0 - C_[2 * m]);
        return k;
    }

private:
    int N_ = 0, L_ = 0;
    double* C_ = nullptr;

    double stw(int m) const {
        return (m <= 1) ? (m == 1 ? C_[1] : 0.0) : C_[m ^ 1];
    }

    // ----- primitive operations -----

    static void binomial(double* v, int h) {
        for (int i = 0; i < h; ++i) {
            double a = v[i], b = v[h + i];
            v[i] = a + b;
            v[h + i] = a - b;
        }
    }

    static void norm_q(double* p, int q, double c, double s) {
        double* A0 = p, *B0 = p + q, *A1 = p + 2 * q, *B1 = p + 3 * q;
        for (int n = 0; n < q; ++n) {
            double a0 = A0[n], b0 = B0[n], a1 = A1[n], b1 = B1[n];
            double R = c * b0 - s * b1;
            double I = s * b0 + c * b1;
            A0[n] =  a0 + R;
            B0[n] =  a1 + I;
            A1[n] =  a0 - R;
            B1[n] = -a1 + I;
        }
    }

    static void norm2_fused(double* p, int q,
                            double c, double s,
                            double c0, double s0,
                            double c1, double s1)
    {
        const int qh = q >> 1;
        double* A0 = p, *B0 = p + q, *A1 = p + 2 * q, *B1 = p + 3 * q;
        for (int n = 0; n < qh; ++n) {
            double a0n = A0[n],      a0h = A0[qh + n];
            double b0n = B0[n],      b0h = B0[qh + n];
            double a1n = A1[n],      a1h = A1[qh + n];
            double b1n = B1[n],      b1h = B1[qh + n];

            double Rn = c * b0n - s * b1n;
            double In = s * b0n + c * b1n;
            double Rh = c * b0h - s * b1h;
            double Ih = s * b0h + c * b1h;

            double u0 = a0n + Rn, uh = a0h + Rh;
            double w0 = a1n + In, wh = a1h + Ih;
            double v0 = a0n - Rn, vh = a0h - Rh;
            double x0 = In - a1n, xh = Ih - a1h;

            double R0 = c0 * uh - s0 * wh;
            double I0 = s0 * uh + c0 * wh;
            double R1 = c1 * vh - s1 * xh;
            double I1 = s1 * vh + c1 * xh;

            A0[n]      = u0 + R0;
            A0[qh + n] = w0 + I0;
            B0[n]      = u0 - R0;
            B0[qh + n] = I0 - w0;
            A1[n]      = v0 + R1;
            A1[qh + n] = x0 + I1;
            B1[n]      = v0 - R1;
            B1[qh + n] = I1 - x0;
        }
    }

    // ----- breadth-first stages -----

    void fwd_stage(double* v, int jj) const {
        const int s = N_ >> jj;
        const int h = s >> 1;
        const int q = s >> 2;
        const int m_end = 1 << jj;

        binomial(v, h);

        for (int m = 1; m < m_end; ++m)
            norm_q(v + m * s, q, C_[m], stw(m));
    }

    // ----- depth-first radix-4 subtree -----

    void subtree_r4(double* v, int q, int m) const {
        if (q >= 16) {
            norm2_fused(v, q, C_[m], stw(m),
                        C_[2 * m], stw(2 * m),
                        C_[2 * m + 1], stw(2 * m + 1));
            int qq = q >> 2, cm = 4 * m;
            subtree_r4(v,           qq, cm);
            subtree_r4(v + q,       qq, cm + 1);
            subtree_r4(v + 2 * q,   qq, cm + 2);
            subtree_r4(v + 3 * q,   qq, cm + 3);
            return;
        }
        if (q == 8) {
            norm_q(v, 8, C_[m], stw(m));
            leaf_d3(v,      2 * m);
            leaf_d3(v + 16, 2 * m + 1);
            return;
        }
        if (q == 4) {
            leaf_d3(v, m);
            return;
        }
        if (q == 2) {
            norm_q(v, 2, C_[m], stw(m));
            norm_q(v,     1, C_[2 * m],     stw(2 * m));
            norm_q(v + 4, 1, C_[2 * m + 1], stw(2 * m + 1));
            return;
        }
    }

    void leaf_d3(double* p, int m) const {
        norm_q(p, 4, C_[m], stw(m));
        norm_q(p,     2, C_[2 * m],     stw(2 * m));
        norm_q(p + 8, 2, C_[2 * m + 1], stw(2 * m + 1));
        norm_q(p,      1, C_[4 * m],     stw(4 * m));
        norm_q(p + 4,  1, C_[4 * m + 1], stw(4 * m + 1));
        norm_q(p + 8,  1, C_[4 * m + 2], stw(4 * m + 2));
        norm_q(p + 12, 1, C_[4 * m + 3], stw(4 * m + 3));
    }

    void spine_tail(double* v) const {
        leaf_d3(v + 16, 1);
        binomial(v, 8);
        norm_q(v + 8, 2, C_[1], stw(1));
        norm_q(v + 8,  1, C_[2], stw(2));
        norm_q(v + 12, 1, C_[3], stw(3));
        binomial(v, 4);
        norm_q(v + 4, 1, C_[1], stw(1));
        binomial(v, 2);
    }
};

// ---------------------------------------------------------------------------
// Standalone norm_q and norm2_fused for use by external test code.
// ---------------------------------------------------------------------------
static inline void norm_q_standalone(double* p, int q, double c, double s) {
    double* A0 = p, *B0 = p + q, *A1 = p + 2 * q, *B1 = p + 3 * q;
    for (int n = 0; n < q; ++n) {
        double a0 = A0[n], b0 = B0[n], a1 = A1[n], b1 = B1[n];
        double R = c * b0 - s * b1;
        double I = s * b0 + c * b1;
        A0[n] =  a0 + R;
        B0[n] =  a1 + I;
        A1[n] =  a0 - R;
        B1[n] = -a1 + I;
    }
}

static inline void norm2_fused_standalone(double* p, int q,
                                          double c, double s,
                                          double c0, double s0,
                                          double c1, double s1)
{
    const int qh = q >> 1;
    double* A0 = p, *B0 = p + q, *A1 = p + 2 * q, *B1 = p + 3 * q;
    for (int n = 0; n < qh; ++n) {
        double a0n = A0[n],      a0h = A0[qh + n];
        double b0n = B0[n],      b0h = B0[qh + n];
        double a1n = A1[n],      a1h = A1[qh + n];
        double b1n = B1[n],      b1h = B1[qh + n];

        double Rn = c * b0n - s * b1n;
        double In = s * b0n + c * b1n;
        double Rh = c * b0h - s * b1h;
        double Ih = s * b0h + c * b1h;

        double u0 = a0n + Rn, uh = a0h + Rh;
        double w0 = a1n + In, wh = a1h + Ih;
        double v0 = a0n - Rn, vh = a0h - Rh;
        double x0 = In - a1n, xh = Ih - a1h;

        double R0 = c0 * uh - s0 * wh;
        double I0 = s0 * uh + c0 * wh;
        double R1 = c1 * vh - s1 * xh;
        double I1 = s1 * vh + c1 * xh;

        A0[n]      = u0 + R0;
        A0[qh + n] = w0 + I0;
        B0[n]      = u0 - R0;
        B0[qh + n] = I0 - w0;
        A1[n]      = v0 + R1;
        A1[qh + n] = x0 + I1;
        B1[n]      = v0 - R1;
        B1[qh + n] = I1 - x0;
    }
}

} // namespace bruun_radix4

#endif // BFFT_BRUUN_RADIX4_KERNEL_HPP
