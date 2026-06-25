#pragma once
// Generalized (arbitrary-N) Bruun real FFT: one real factorization of z^N-1,
// reusing the power-of-two normalized Bruun kernel as the 2-adic subtree plus
// condition-1 odd-radix codelets. No Bluestein/Rader/chirp. See
// notes/genbruun_port_handoff.md for the full design.
//
// A GenBruun owns its scratch (arena + pow2-subplan work). Like RFFT it is NOT
// thread-safe: use one per thread.

#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>
#include <map>

#include "bruun_kernel.hpp"

namespace bruun {

class GenBruun {
public:
    explicit GenBruun(int n) : N(n) { build(); }
    ~GenBruun() { for (auto* r : pow2_) delete r; }
    GenBruun(const GenBruun&) = delete;
    GenBruun& operator=(const GenBruun&) = delete;

    int size() const { return N; }
    int bins() const { return N / 2 + 1; }

    // input: N doubles. output: N/2+1 complex (standard rfft half-spectrum).
    void forward(const double* input, complex_t* X) {
        for (int i = 0; i < bins(); ++i) { X[i].re = 0; X[i].im = 0; }
        double* A = arena_.data();
        for (int i = 0; i < N; ++i) A[i] = input[i];
        for (const Op& o : ops_) exec_fwd(o, A, X);
    }

    // input: N/2+1 complex half-spectrum. output: N doubles.
    void inverse(const complex_t* Xhalf, double* output) {
        cx* Xf = xfull_.data();
        for (int k = 0; k <= N / 2; ++k) Xf[k] = cx{Xhalf[k].re, Xhalf[k].im};
        for (int k = 1; k < (N + 1) / 2; ++k) Xf[N - k] = cx{Xhalf[k].re, -Xhalf[k].im};
        double* A = arena_.data();
        for (int oi = (int)ops_.size() - 1; oi >= 0; --oi) exec_inv(ops_[oi], A, Xf);
        for (int i = 0; i < N; ++i) output[i] = A[i];
    }

    void forward_f32(const float* input, complex_f32_t* X) {
        for (int i = 0; i < bins(); ++i) { X[i].re = 0.0f; X[i].im = 0.0f; }
        float* A = arena_f32_.data();
        for (int i = 0; i < N; ++i) A[i] = input[i];
        for (const Op& o : ops_) exec_fwd_f32(o, A, X);
    }

    void inverse_f32(const complex_f32_t* Xhalf, float* output) {
        cxf* Xf = xfull_f32_.data();
        for (int k = 0; k <= N / 2; ++k) Xf[k] = cxf{Xhalf[k].re, Xhalf[k].im};
        for (int k = 1; k < (N + 1) / 2; ++k) Xf[N - k] = cxf{Xhalf[k].re, -Xhalf[k].im};
        float* A = arena_f32_.data();
        for (int oi = (int)ops_.size() - 1; oi >= 0; --oi) exec_inv_f32(ops_[oi], A, Xf);
        for (int i = 0; i < N; ++i) output[i] = A[i];
    }

private:
    struct cx { double re, im; };
    struct cxf { float re, im; };
    struct Frac { long long n, d; };
    static Frac mk(long long n, long long d) {
        if (d < 0) { n = -n; d = -d; }
        long long a = n < 0 ? -n : n, g = std::gcd(a, d); if (!g) g = 1; return {n / g, d / g};
    }
    static Frac fhalf(Frac f) { return mk(f.n, f.d * 2); }
    static Frac fcomp(Frac f) { return mk(f.d - f.n, 2 * f.d); }
    static Frac fdivp(Frac f, long long t, long long p) { return mk(f.n + t * f.d, f.d * p); }
    static long long fr(Frac f) { return ((f.n % f.d) + f.d) % f.d; }
    static void trig(Frac f, double& c, double& s) {
        long long m = fr(f), d = f.d; int quad = (int)((4 * m) / d); long long rem4 = (4 * m) % d;
        double a = 2.0 * std::acos(-1.0) * (double)rem4 / (4.0 * (double)d);
        double ca = std::cos(a), sa = std::sin(a);
        switch (quad & 3) { case 0: c = ca; s = sa; break; case 1: c = -sa; s = ca; break;
            case 2: c = -ca; s = -sa; break; default: c = sa; s = -ca; }
    }
    static double tcos(Frac f) { double c, s; trig(f, c, s); return c; }
    static double tsin(Frac f) { double c, s; trig(f, c, s); return s; }
    long long fbin(Frac f) const { long long k = std::llround((double)N * f.n / f.d); return ((k % N) + N) % N; }
    static bool is_pow2(int n) { return n >= 1 && (n & (n - 1)) == 0; }
    static int odd_prime(int n) { while (n % 2 == 0) n /= 2; if (n == 1) return 0;
        for (int d = 3; (long long)d * d <= n; d += 2) if (n % d == 0) return d; return n; }

    enum { MINUS_POW2, MINUS_SPLIT, BRUUN_POW2, BRUUN_ODD, LEAF };
    enum { CODELET_GENERIC, CODELET_3, CODELET_5, CODELET_7, CODELET_11, CODELET_13, CODELET_17, CODELET_19 };
    struct Op { int type, in, len = 0, p = 0, M = 0; long long sigma = 0; Frac f{0, 1};
        int sum = 0; std::vector<int> child, tw; int ctab = -1, perm = -1, pow2 = -1, codelet = CODELET_GENERIC; double s1 = 0.0; };

    int N;
    std::vector<Op> ops_;
    std::vector<double> twp_, ctp_;
    std::vector<int> permp_;
    std::vector<RFFT*> pow2_;
    std::map<int, int> pow2idx_;
    int top_ = 0, hi_ = 0, work_ = 0, work_f32_ = 0;
    heap_array<double> arena_, work_buf_;
    heap_array<float> arena_f32_, work_buf_f32_;
    heap_array<complex_t> bins_buf_;
    heap_array<complex_f32_t> bins_buf_f32_;
    heap_array<cx> xfull_;
    heap_array<cxf> xfull_f32_;

    int alloc(int n) { int o = top_; top_ += n; hi_ = std::max(hi_, top_); return o; }
    void freeto(int o) { top_ = o; }
    static int codelet_for_prime(int p) {
        switch (p) {
            case 3: return CODELET_3;
            case 5: return CODELET_5;
            case 7: return CODELET_7;
            case 11: return CODELET_11;
            case 13: return CODELET_13;
            case 17: return CODELET_17;
            case 19: return CODELET_19;
            default: return CODELET_GENERIC;
        }
    }
    int add_tw(int n, Frac base) { int off = (int)twp_.size(); twp_.resize(off + 2 * n);
        for (int i = 0; i < n; ++i) { double c, s; trig(mk(base.n * i, base.d), c, s);
            twp_[off + i] = c; twp_[off + n + i] = s; } return off; }
    int get_pow2(int M) { auto it = pow2idx_.find(M); if (it != pow2idx_.end()) return it->second;
        auto* r = new RFFT(); r->init(M); work_ = std::max(work_, M);
        work_f32_ = std::max(work_f32_, r->work_size_f32());
        int id = (int)pow2_.size(); pow2_.push_back(r); pow2idx_[M] = id; return id; }

    void build_rooted(int M, Frac root, int& ctab, int& perm, double& s1) {
        int D = 2 * M, half = M, Lg = 0; for (int t = D; t > 1; t >>= 1) ++Lg;
        ctab = (int)ctp_.size(); ctp_.resize(ctab + std::max(1, half));
        double* C = &ctp_[ctab];
        if (half > 1) C[1] = tcos(fhalf(root));
        s1 = tsin(fhalf(root));
        auto s_tw = [&](int m) { if (m == 1) return s1; int flip = (m > 1) ? 1 : 0; return C[m ^ flip]; };
        for (int m = 1; 2 * m < half; ++m) { double c = C[m], s = s_tw(m), ce = std::sqrt(0.5 * (1 + c));
            C[2 * m] = ce; if (2 * m + 1 < half) C[2 * m + 1] = s / (2 * ce); }
        perm = (int)permp_.size(); permp_.resize(perm + half); int depth = Lg - 1;
        for (int m = 0; m < half; ++m) { Frac f = root;
            for (int b = depth - 1; b >= 0; --b) f = ((m >> b) & 1) ? fcomp(f) : fhalf(f);
            permp_[perm + m] = (int)fbin(f); }
    }

    void plan_bruun(int in, int M, Frac f) {
        if (M == 1) { Op o; o.type = LEAF; o.in = in; o.f = f; ops_.push_back(o); return; }
        if (is_pow2(M)) { Op o; o.type = BRUUN_POW2; o.in = in; o.M = M; o.f = f;
            build_rooted(M, f, o.ctab, o.perm, o.s1); ops_.push_back(o); return; }
        int p = odd_prime(M), Mp = M / p;
        Op o; o.type = BRUUN_ODD; o.in = in; o.M = M; o.p = p; o.codelet = codelet_for_prime(p); o.f = f; int base = top_;
        for (int t = 0; t < p; ++t) { o.child.push_back(alloc(2 * Mp)); o.tw.push_back(add_tw(p, fdivp(f, t, p))); }
        ops_.push_back(o); Op saved = o;
        for (int t = 0; t < p; ++t) plan_bruun(saved.child[t], Mp, fdivp(f, t, p));
        freeto(base);
    }

    void plan_minus(int in, int D, long long sigma) {
        if (is_pow2(D)) { Op o; o.type = MINUS_POW2; o.in = in; o.len = D; o.sigma = sigma;
            o.pow2 = (D >= 4) ? get_pow2(D) : -1; ops_.push_back(o); return; }
        int p = odd_prime(D), M = D / p;
        Op o; o.type = MINUS_SPLIT; o.in = in; o.len = D; o.p = p; o.M = M; o.codelet = codelet_for_prime(p); o.sigma = sigma; int base = top_;
        o.sum = alloc(M);
        for (int j = 1; j <= (p - 1) / 2; ++j) { o.child.push_back(alloc(2 * M)); o.tw.push_back(add_tw(p, mk(j, p))); }
        ops_.push_back(o); Op saved = o;
        plan_minus(saved.sum, M, sigma * p);
        for (int j = 1; j <= (p - 1) / 2; ++j) { int br = saved.child[j - 1];
            if (M == 1) { Op l; l.type = LEAF; l.in = br; l.f = mk(j, p); ops_.push_back(l); }
            else plan_bruun(br, M, mk(j, p)); }
        freeto(base);
    }

    void build() {
        top_ = 0; hi_ = 0; int root = alloc(N); plan_minus(root, N, 1);
        arena_buf_resize();
    }
    void arena_buf_resize() {
        arena_.resize((size_t)hi_ + 4);
        arena_f32_.resize((size_t)hi_ + 4);
        work_buf_.resize((size_t)std::max(1, work_));
        work_buf_f32_.resize((size_t)std::max(1, work_f32_));
        bins_buf_.resize((size_t)std::max(1, work_ / 2 + 1));
        bins_buf_f32_.resize((size_t)std::max(1, work_ / 2 + 1));
        xfull_.resize((size_t)N);
        xfull_f32_.resize((size_t)N);
    }

    void place(complex_t* X, long long k, double re, double im) {
        k = ((k % N) + N) % N;
        if (k > N / 2) { k = N - k; im = -im; }   // fold to rfft half
        X[k].re = re; X[k].im = im;
    }

    void place_f32(complex_f32_t* X, long long k, float re, float im) {
        k = ((k % N) + N) % N;
        if (k > N / 2) { k = N - k; im = -im; }
        X[k].re = re; X[k].im = im;
    }

    static void zero_block(double* RESTRICT dst, int n) {
        int m = 0;
#if BRUUN_LEVEL >= 1
        const bruun_v2 z = V2_SET1(0.0);
        for (; m + 1 < n; m += 2) V2_ST(dst + m, z);
#endif
        for (; m < n; ++m) dst[m] = 0.0;
    }

    static void assign_scaled(double* RESTRICT dst, const double* RESTRICT src,
                              double scale, int n) {
        int m = 0;
#if BRUUN_LEVEL >= 1
        const bruun_v2 sc = V2_SET1(scale);
        for (; m + 1 < n; m += 2) V2_ST(dst + m, V2_MUL(V2_LD(src + m), sc));
#endif
        for (; m < n; ++m) dst[m] = src[m] * scale;
    }

    static void add_scaled(double* RESTRICT dst, const double* RESTRICT src,
                           double scale, int n) {
        int m = 0;
#if BRUUN_LEVEL >= 1
        const bruun_v2 sc = V2_SET1(scale);
        for (; m + 1 < n; m += 2) {
            V2_ST(dst + m, V2_MADD(V2_LD(dst + m), V2_LD(src + m), sc));
        }
#endif
        for (; m < n; ++m) dst[m] += src[m] * scale;
    }

    static void add_real_pair(double* RESTRICT dst, const double* RESTRICT lo,
                              const double* RESTRICT hi, double c, double s, int n) {
        int m = 0;
#if BRUUN_LEVEL >= 1
        const bruun_v2 cv = V2_SET1(c);
        const bruun_v2 sv = V2_SET1(s);
        for (; m + 1 < n; m += 2) {
            bruun_v2 acc = V2_LD(dst + m);
            acc = V2_MADD(acc, V2_LD(lo + m), cv);
            acc = V2_MADD(acc, V2_LD(hi + m), sv);
            V2_ST(dst + m, acc);
        }
#endif
        for (; m < n; ++m) dst[m] += lo[m] * c + hi[m] * s;
    }

    static void add_complex_projection(double* RESTRICT out_lo, double* RESTRICT out_hi,
                                       const double* RESTRICT in_lo, const double* RESTRICT in_hi,
                                       double c, double s, int n) {
        int m = 0;
#if BRUUN_LEVEL >= 1
        const bruun_v2 cv = V2_SET1(c);
        const bruun_v2 sv = V2_SET1(s);
        for (; m + 1 < n; m += 2) {
            const bruun_v2 av = V2_LD(in_lo + m);
            const bruun_v2 bv = V2_LD(in_hi + m);
            bruun_v2 acc_lo = V2_LD(out_lo + m);
            bruun_v2 acc_hi = V2_LD(out_hi + m);
            acc_lo = V2_MADD(acc_lo, av, cv);
            acc_lo = V2_MSUB(acc_lo, bv, sv);
            acc_hi = V2_MADD(acc_hi, bv, cv);
            acc_hi = V2_MADD(acc_hi, av, sv);
            V2_ST(out_lo + m, acc_lo);
            V2_ST(out_hi + m, acc_hi);
        }
#endif
        for (; m < n; ++m) {
            out_lo[m] += in_lo[m] * c - in_hi[m] * s;
            out_hi[m] += in_hi[m] * c + in_lo[m] * s;
        }
    }

    static void add_complex_adjoint(double* RESTRICT out_lo, double* RESTRICT out_hi,
                                    const double* RESTRICT child_lo, const double* RESTRICT child_hi,
                                    double c, double s, int n) {
        int m = 0;
#if BRUUN_LEVEL >= 1
        const bruun_v2 cv = V2_SET1(c);
        const bruun_v2 sv = V2_SET1(s);
        for (; m + 1 < n; m += 2) {
            const bruun_v2 clo = V2_LD(child_lo + m);
            const bruun_v2 chi = V2_LD(child_hi + m);
            bruun_v2 acc_lo = V2_LD(out_lo + m);
            bruun_v2 acc_hi = V2_LD(out_hi + m);
            acc_lo = V2_MADD(acc_lo, clo, cv);
            acc_lo = V2_MADD(acc_lo, chi, sv);
            acc_hi = V2_MADD(acc_hi, chi, cv);
            acc_hi = V2_MSUB(acc_hi, clo, sv);
            V2_ST(out_lo + m, acc_lo);
            V2_ST(out_hi + m, acc_hi);
        }
#endif
        for (; m < n; ++m) {
            out_lo[m] += child_lo[m] * c + child_hi[m] * s;
            out_hi[m] += child_hi[m] * c - child_lo[m] * s;
        }
    }

    static void minus_split3_forward(const double* RESTRICT in, double* RESTRICT sum,
                                     double* RESTRICT lo, double* RESTRICT hi,
                                     const double* RESTRICT C, const double* RESTRICT S, int M) {
        const double* RESTRICT r0 = in;
        const double* RESTRICT r1 = in + M;
        const double* RESTRICT r2 = in + 2 * M;
        int m = 0;
#if BRUUN_LEVEL >= 1
        const bruun_v2 c0 = V2_SET1(C[0]), c1 = V2_SET1(C[1]), c2 = V2_SET1(C[2]);
        const bruun_v2 s0 = V2_SET1(S[0]), s1 = V2_SET1(S[1]), s2 = V2_SET1(S[2]);
        for (; m + 1 < M; m += 2) {
            const bruun_v2 v0 = V2_LD(r0 + m);
            const bruun_v2 v1 = V2_LD(r1 + m);
            const bruun_v2 v2 = V2_LD(r2 + m);
            V2_ST(sum + m, V2_ADD(V2_ADD(v0, v1), v2));
            bruun_v2 l = V2_MADD(V2_MUL(v0, c0), v1, c1);
            l = V2_MADD(l, v2, c2);
            bruun_v2 h = V2_MADD(V2_MUL(v0, s0), v1, s1);
            h = V2_MADD(h, v2, s2);
            V2_ST(lo + m, l);
            V2_ST(hi + m, h);
        }
#endif
        for (; m < M; ++m) {
            const double v0 = r0[m], v1 = r1[m], v2 = r2[m];
            sum[m] = v0 + v1 + v2;
            lo[m] = v0 * C[0] + v1 * C[1] + v2 * C[2];
            hi[m] = v0 * S[0] + v1 * S[1] + v2 * S[2];
        }
    }

    static void bruun_odd3_forward_child(double* RESTRICT clo, double* RESTRICT chi,
                                         const double* RESTRICT lo, const double* RESTRICT hi,
                                         const double* RESTRICT C, const double* RESTRICT S, int Mp) {
        const double* RESTRICT a0 = lo;
        const double* RESTRICT a1 = lo + Mp;
        const double* RESTRICT a2 = lo + 2 * Mp;
        const double* RESTRICT b0 = hi;
        const double* RESTRICT b1 = hi + Mp;
        const double* RESTRICT b2 = hi + 2 * Mp;
        int m = 0;
#if BRUUN_LEVEL >= 1
        const bruun_v2 c0 = V2_SET1(C[0]), c1 = V2_SET1(C[1]), c2 = V2_SET1(C[2]);
        const bruun_v2 s0 = V2_SET1(S[0]), s1 = V2_SET1(S[1]), s2 = V2_SET1(S[2]);
        for (; m + 1 < Mp; m += 2) {
            const bruun_v2 av0 = V2_LD(a0 + m), av1 = V2_LD(a1 + m), av2 = V2_LD(a2 + m);
            const bruun_v2 bv0 = V2_LD(b0 + m), bv1 = V2_LD(b1 + m), bv2 = V2_LD(b2 + m);
            bruun_v2 out_l = V2_MSUB(V2_MUL(av0, c0), bv0, s0);
            out_l = V2_MADD(out_l, av1, c1);
            out_l = V2_MSUB(out_l, bv1, s1);
            out_l = V2_MADD(out_l, av2, c2);
            out_l = V2_MSUB(out_l, bv2, s2);
            bruun_v2 out_h = V2_MADD(V2_MUL(bv0, c0), av0, s0);
            out_h = V2_MADD(out_h, bv1, c1);
            out_h = V2_MADD(out_h, av1, s1);
            out_h = V2_MADD(out_h, bv2, c2);
            out_h = V2_MADD(out_h, av2, s2);
            V2_ST(clo + m, out_l);
            V2_ST(chi + m, out_h);
        }
#endif
        for (; m < Mp; ++m) {
            clo[m] = a0[m] * C[0] - b0[m] * S[0]
                   + a1[m] * C[1] - b1[m] * S[1]
                   + a2[m] * C[2] - b2[m] * S[2];
            chi[m] = b0[m] * C[0] + a0[m] * S[0]
                   + b1[m] * C[1] + a1[m] * S[1]
                   + b2[m] * C[2] + a2[m] * S[2];
        }
    }

    // FACTORED radix-3 forward, all 3 children at once, REAL two-plane (no complex).
    // = real (c,s) modulation by psi=theta/3 (table mod = cos/sin(i*psi)) followed by
    // the real radix-3 butterfly. 12 real mults/lane vs the dense 36. Matches the
    // per-child dense output (sum_i a_i cos(i*phi_t) - b_i sin(i*phi_t)) to rounding.
    static void bruun_odd3_forward_all(double* RESTRICT c0, double* RESTRICT c1, double* RESTRICT c2,
                                       const double* RESTRICT lo, const double* RESTRICT hi,
                                       const double* RESTRICT mod, int Mp) {
        const double* RESTRICT a0 = lo; const double* RESTRICT a1 = lo + Mp; const double* RESTRICT a2 = lo + 2 * Mp;
        const double* RESTRICT b0 = hi; const double* RESTRICT b1 = hi + Mp; const double* RESTRICT b2 = hi + 2 * Mp;
        double* RESTRICT c0l = c0; double* RESTRICT c0h = c0 + Mp;
        double* RESTRICT c1l = c1; double* RESTRICT c1h = c1 + Mp;
        double* RESTRICT c2l = c2; double* RESTRICT c2h = c2 + Mp;
        const double mc1 = mod[1], ms1 = mod[3 + 1], mc2 = mod[2], ms2 = mod[3 + 2];
        const double cr = -0.5, sr = 0.86602540378443864676;   // cos/sin(2*pi/3)
        int m = 0;
#if BRUUN_LEVEL >= 1
        const bruun_v2 vmc1 = V2_SET1(mc1), vms1 = V2_SET1(ms1), vmc2 = V2_SET1(mc2), vms2 = V2_SET1(ms2);
        const bruun_v2 vcr = V2_SET1(cr), vsr = V2_SET1(sr);
        for (; m + 1 < Mp; m += 2) {
            const bruun_v2 A0 = V2_LD(a0 + m), A1 = V2_LD(a1 + m), A2 = V2_LD(a2 + m);
            const bruun_v2 B0 = V2_LD(b0 + m), B1 = V2_LD(b1 + m), B2 = V2_LD(b2 + m);
            const bruun_v2 P1 = V2_MSUB(V2_MUL(A1, vmc1), B1, vms1), Q1 = V2_MADD(V2_MUL(A1, vms1), B1, vmc1);
            const bruun_v2 P2 = V2_MSUB(V2_MUL(A2, vmc2), B2, vms2), Q2 = V2_MADD(V2_MUL(A2, vms2), B2, vmc2);
            const bruun_v2 sp = V2_ADD(P1, P2), dp = V2_SUB(P1, P2), sq = V2_ADD(Q1, Q2), dq = V2_SUB(Q1, Q2);
            V2_ST(c0l + m, V2_ADD(A0, sp)); V2_ST(c0h + m, V2_ADD(B0, sq));
            const bruun_v2 m1 = V2_MADD(A0, vcr, sp), m2 = V2_MUL(vsr, dq);
            const bruun_v2 n1 = V2_MADD(B0, vcr, sq), n2 = V2_MUL(vsr, dp);
            V2_ST(c1l + m, V2_SUB(m1, m2)); V2_ST(c2l + m, V2_ADD(m1, m2));
            V2_ST(c1h + m, V2_ADD(n1, n2)); V2_ST(c2h + m, V2_SUB(n1, n2));
        }
#endif
        for (; m < Mp; ++m) {
            const double A0 = a0[m], A1 = a1[m], A2 = a2[m], B0 = b0[m], B1 = b1[m], B2 = b2[m];
            const double P1 = A1 * mc1 - B1 * ms1, Q1 = A1 * ms1 + B1 * mc1;
            const double P2 = A2 * mc2 - B2 * ms2, Q2 = A2 * ms2 + B2 * mc2;
            const double sp = P1 + P2, dp = P1 - P2, sq = Q1 + Q2, dq = Q1 - Q2;
            c0l[m] = A0 + sp; c0h[m] = B0 + sq;
            const double m1 = A0 + cr * sp, m2 = sr * dq, n1 = B0 + cr * sq, n2 = sr * dp;
            c1l[m] = m1 - m2; c2l[m] = m1 + m2; c1h[m] = n1 + n2; c2h[m] = n1 - n2;
        }
    }

    static void bruun_odd3_adjoint_child(double* RESTRICT lo, double* RESTRICT hi,
                                         const double* RESTRICT clo, const double* RESTRICT chi,
                                         const double* RESTRICT C, const double* RESTRICT S, int Mp) {
        double* RESTRICT a0 = lo;
        double* RESTRICT a1 = lo + Mp;
        double* RESTRICT a2 = lo + 2 * Mp;
        double* RESTRICT b0 = hi;
        double* RESTRICT b1 = hi + Mp;
        double* RESTRICT b2 = hi + 2 * Mp;
        int m = 0;
#if BRUUN_LEVEL >= 1
        const bruun_v2 third = V2_SET1(1.0 / 3.0);
        const bruun_v2 c0 = V2_SET1(C[0]), c1 = V2_SET1(C[1]), c2 = V2_SET1(C[2]);
        const bruun_v2 s0 = V2_SET1(S[0]), s1 = V2_SET1(S[1]), s2 = V2_SET1(S[2]);
        for (; m + 1 < Mp; m += 2) {
            const bruun_v2 cl = V2_MUL(V2_LD(clo + m), third);
            const bruun_v2 ch = V2_MUL(V2_LD(chi + m), third);
            bruun_v2 ta0 = V2_LD(a0 + m);
            bruun_v2 tb0 = V2_LD(b0 + m);
            bruun_v2 ta1 = V2_LD(a1 + m);
            bruun_v2 tb1 = V2_LD(b1 + m);
            bruun_v2 ta2 = V2_LD(a2 + m);
            bruun_v2 tb2 = V2_LD(b2 + m);
            ta0 = V2_MADD(ta0, cl, c0); ta0 = V2_MADD(ta0, ch, s0);
            tb0 = V2_MADD(tb0, ch, c0); tb0 = V2_MSUB(tb0, cl, s0);
            ta1 = V2_MADD(ta1, cl, c1); ta1 = V2_MADD(ta1, ch, s1);
            tb1 = V2_MADD(tb1, ch, c1); tb1 = V2_MSUB(tb1, cl, s1);
            ta2 = V2_MADD(ta2, cl, c2); ta2 = V2_MADD(ta2, ch, s2);
            tb2 = V2_MADD(tb2, ch, c2); tb2 = V2_MSUB(tb2, cl, s2);
            V2_ST(a0 + m, ta0); V2_ST(b0 + m, tb0);
            V2_ST(a1 + m, ta1); V2_ST(b1 + m, tb1);
            V2_ST(a2 + m, ta2); V2_ST(b2 + m, tb2);
        }
#endif
        for (; m < Mp; ++m) {
            const double cl = clo[m] / 3.0;
            const double ch = chi[m] / 3.0;
            a0[m] += cl * C[0] + ch * S[0];
            b0[m] += ch * C[0] - cl * S[0];
            a1[m] += cl * C[1] + ch * S[1];
            b1[m] += ch * C[1] - cl * S[1];
            a2[m] += cl * C[2] + ch * S[2];
            b2[m] += ch * C[2] - cl * S[2];
        }
    }

    // FACTORED radix-3 inverse adjoint, all children at once, REAL two-plane.
    // = (1/3) * M^T * B^T : inverse radix-3 butterfly (scaled 1/3) then inverse
    // (c,s) rotation by psi. Exact transpose of bruun_odd3_forward_all -> roundtrip
    // is identity. Writes the full parent (no accumulate).
    static void bruun_odd3_adjoint_all(double* RESTRICT lo, double* RESTRICT hi,
                                       const double* RESTRICT c0, const double* RESTRICT c1, const double* RESTRICT c2,
                                       const double* RESTRICT mod, int Mp) {
        double* RESTRICT a0 = lo; double* RESTRICT a1 = lo + Mp; double* RESTRICT a2 = lo + 2 * Mp;
        double* RESTRICT b0 = hi; double* RESTRICT b1 = hi + Mp; double* RESTRICT b2 = hi + 2 * Mp;
        const double* RESTRICT c0l = c0; const double* RESTRICT c0h = c0 + Mp;
        const double* RESTRICT c1l = c1; const double* RESTRICT c1h = c1 + Mp;
        const double* RESTRICT c2l = c2; const double* RESTRICT c2h = c2 + Mp;
        const double mc1 = mod[1], ms1 = mod[3 + 1], mc2 = mod[2], ms2 = mod[3 + 2];
        const double cr = -0.5, sr = 0.86602540378443864676, third = 1.0 / 3.0;
        int m = 0;
#if BRUUN_LEVEL >= 1
        const bruun_v2 vmc1 = V2_SET1(mc1), vms1 = V2_SET1(ms1), vmc2 = V2_SET1(mc2), vms2 = V2_SET1(ms2);
        const bruun_v2 vcr = V2_SET1(cr), vsr = V2_SET1(sr), v3 = V2_SET1(third);
        for (; m + 1 < Mp; m += 2) {
            const bruun_v2 g0 = V2_MUL(V2_LD(c0l + m), v3), g1 = V2_MUL(V2_LD(c1l + m), v3), g2 = V2_MUL(V2_LD(c2l + m), v3);
            const bruun_v2 h0 = V2_MUL(V2_LD(c0h + m), v3), h1 = V2_MUL(V2_LD(c1h + m), v3), h2 = V2_MUL(V2_LD(c2h + m), v3);
            const bruun_v2 sg = V2_ADD(g1, g2), dg = V2_SUB(g1, g2), sh = V2_ADD(h1, h2), dh = V2_SUB(h1, h2);
            const bruun_v2 P0 = V2_ADD(g0, sg), Q0 = V2_ADD(h0, sh);
            const bruun_v2 u = V2_MADD(g0, vcr, sg), w = V2_MUL(vsr, dh);
            const bruun_v2 x = V2_MADD(h0, vcr, sh), y = V2_MUL(vsr, dg);
            const bruun_v2 P1 = V2_ADD(u, w), P2 = V2_SUB(u, w), Q1 = V2_SUB(x, y), Q2 = V2_ADD(x, y);
            V2_ST(a0 + m, P0); V2_ST(b0 + m, Q0);
            V2_ST(a1 + m, V2_MADD(V2_MUL(P1, vmc1), Q1, vms1)); V2_ST(b1 + m, V2_MSUB(V2_MUL(Q1, vmc1), P1, vms1));
            V2_ST(a2 + m, V2_MADD(V2_MUL(P2, vmc2), Q2, vms2)); V2_ST(b2 + m, V2_MSUB(V2_MUL(Q2, vmc2), P2, vms2));
        }
#endif
        for (; m < Mp; ++m) {
            const double g0 = c0l[m] * third, g1 = c1l[m] * third, g2 = c2l[m] * third;
            const double h0 = c0h[m] * third, h1 = c1h[m] * third, h2 = c2h[m] * third;
            const double sg = g1 + g2, dg = g1 - g2, sh = h1 + h2, dh = h1 - h2;
            const double P0 = g0 + sg, Q0 = h0 + sh;
            const double u = g0 + cr * sg, w = sr * dh, x = h0 + cr * sh, y = sr * dg;
            const double P1 = u + w, P2 = u - w, Q1 = x - y, Q2 = x + y;
            a0[m] = P0; b0[m] = Q0;
            a1[m] = P1 * mc1 + Q1 * ms1; b1[m] = Q1 * mc1 - P1 * ms1;
            a2[m] = P2 * mc2 + Q2 * ms2; b2[m] = Q2 * mc2 - P2 * ms2;
        }
    }

    // FACTORED radix-5: real two-plane, modulate by psi=theta/5 then symmetric
    // real radix-5 butterfly. 32 mults/lane vs the dense 100. Constants = cos/sin
    // of 2pi/5 and 4pi/5.
    static constexpr double R5_C1 = 0.30901699437494742410, R5_C2 = -0.80901699437494742410;
    static constexpr double R5_S1 = 0.95105651629515357212, R5_S2 = 0.58778525229247312917;
    static void bruun_odd5_forward_all(double* RESTRICT c0, double* RESTRICT c1, double* RESTRICT c2,
                                       double* RESTRICT c3, double* RESTRICT c4,
                                       const double* RESTRICT lo, const double* RESTRICT hi,
                                       const double* RESTRICT mod, int Mp) {
        const double* RESTRICT a[5] = {lo, lo + Mp, lo + 2 * Mp, lo + 3 * Mp, lo + 4 * Mp};
        const double* RESTRICT b[5] = {hi, hi + Mp, hi + 2 * Mp, hi + 3 * Mp, hi + 4 * Mp};
        double* RESTRICT cl[5] = {c0, c1, c2, c3, c4};
        for (int m = 0; m < Mp; ++m) {
            double P[5], Q[5];
            P[0] = a[0][m]; Q[0] = b[0][m];
            for (int i = 1; i < 5; ++i) { double ai = a[i][m], bi = b[i][m], mc = mod[i], ms = mod[5 + i];
                P[i] = ai * mc - bi * ms; Q[i] = ai * ms + bi * mc; }
            const double pe1 = P[1] + P[4], pe2 = P[2] + P[3], po1 = P[1] - P[4], po2 = P[2] - P[3];
            const double qe1 = Q[1] + Q[4], qe2 = Q[2] + Q[3], qo1 = Q[1] - Q[4], qo2 = Q[2] - Q[3];
            cl[0][m] = P[0] + pe1 + pe2; cl[0][Mp + m] = Q[0] + qe1 + qe2;
            const double Ec1 = pe1 * R5_C1 + pe2 * R5_C2, Ec2 = pe1 * R5_C2 + pe2 * R5_C1;
            const double Os1 = qo1 * R5_S1 + qo2 * R5_S2, Os2 = qo1 * R5_S2 - qo2 * R5_S1;
            const double Fc1 = qe1 * R5_C1 + qe2 * R5_C2, Fc2 = qe1 * R5_C2 + qe2 * R5_C1;
            const double Ps1 = po1 * R5_S1 + po2 * R5_S2, Ps2 = po1 * R5_S2 - po2 * R5_S1;
            cl[1][m] = P[0] + Ec1 - Os1; cl[4][m] = P[0] + Ec1 + Os1;
            cl[2][m] = P[0] + Ec2 - Os2; cl[3][m] = P[0] + Ec2 + Os2;
            cl[1][Mp + m] = Q[0] + Fc1 + Ps1; cl[4][Mp + m] = Q[0] + Fc1 - Ps1;
            cl[2][Mp + m] = Q[0] + Fc2 + Ps2; cl[3][Mp + m] = Q[0] + Fc2 - Ps2;
        }
    }

    static void bruun_odd5_adjoint_all(double* RESTRICT lo, double* RESTRICT hi,
                                       const double* RESTRICT c0, const double* RESTRICT c1, const double* RESTRICT c2,
                                       const double* RESTRICT c3, const double* RESTRICT c4,
                                       const double* RESTRICT mod, int Mp) {
        double* RESTRICT a[5] = {lo, lo + Mp, lo + 2 * Mp, lo + 3 * Mp, lo + 4 * Mp};
        double* RESTRICT b[5] = {hi, hi + Mp, hi + 2 * Mp, hi + 3 * Mp, hi + 4 * Mp};
        const double* RESTRICT cl[5] = {c0, c1, c2, c3, c4};
        const double fifth = 0.2;
        for (int m = 0; m < Mp; ++m) {
            const double L0 = cl[0][m], L1 = cl[1][m], L2 = cl[2][m], L3 = cl[3][m], L4 = cl[4][m];
            const double H0 = cl[0][Mp + m], H1 = cl[1][Mp + m], H2 = cl[2][Mp + m], H3 = cl[3][Mp + m], H4 = cl[4][Mp + m];
            const double ce1 = L1 + L4, ce2 = L2 + L3, co1 = L1 - L4, co2 = L2 - L3;
            const double he1 = H1 + H4, he2 = H2 + H3, ho1 = H1 - H4, ho2 = H2 - H3;
            double P[5], Q[5];
            P[0] = (L0 + ce1 + ce2) * fifth; Q[0] = (H0 + he1 + he2) * fifth;
            const double EcA = ce1 * R5_C1 + ce2 * R5_C2, EcB = ce1 * R5_C2 + ce2 * R5_C1;
            const double OsA = ho1 * R5_S1 + ho2 * R5_S2, OsB = ho1 * R5_S2 - ho2 * R5_S1;
            const double FcA = he1 * R5_C1 + he2 * R5_C2, FcB = he1 * R5_C2 + he2 * R5_C1;
            const double PsA = co1 * R5_S1 + co2 * R5_S2, PsB = co1 * R5_S2 - co2 * R5_S1;
            P[1] = (L0 + EcA + OsA) * fifth; P[4] = (L0 + EcA - OsA) * fifth;
            P[2] = (L0 + EcB + OsB) * fifth; P[3] = (L0 + EcB - OsB) * fifth;
            Q[1] = (H0 + FcA - PsA) * fifth; Q[4] = (H0 + FcA + PsA) * fifth;
            Q[2] = (H0 + FcB - PsB) * fifth; Q[3] = (H0 + FcB + PsB) * fifth;
            a[0][m] = P[0]; b[0][m] = Q[0];
            for (int i = 1; i < 5; ++i) { double mc = mod[i], ms = mod[5 + i];
                a[i][m] = P[i] * mc + Q[i] * ms; b[i][m] = Q[i] * mc - P[i] * ms; }
        }
    }

    // FACTORED radix-7: real two-plane, modulate by psi=theta/7 then symmetric real
    // radix-7 butterfly. 60 mults/lane vs the dense 196. cos/sin of 2pi/7,4pi/7,6pi/7.
    static constexpr double R7_C1 = 0.62348980185873359, R7_C2 = -0.22252093395631440, R7_C3 = -0.90096886790241915;
    static constexpr double R7_S1 = 0.78183148246802980, R7_S2 = 0.97492791218182362, R7_S3 = 0.43388373911755812;
    static void bruun_odd7_forward_all(double* RESTRICT c0, double* RESTRICT c1, double* RESTRICT c2, double* RESTRICT c3,
                                       double* RESTRICT c4, double* RESTRICT c5, double* RESTRICT c6,
                                       const double* RESTRICT lo, const double* RESTRICT hi,
                                       const double* RESTRICT mod, int Mp) {
        const double* RESTRICT a[7] = {lo, lo+Mp, lo+2*Mp, lo+3*Mp, lo+4*Mp, lo+5*Mp, lo+6*Mp};
        const double* RESTRICT b[7] = {hi, hi+Mp, hi+2*Mp, hi+3*Mp, hi+4*Mp, hi+5*Mp, hi+6*Mp};
        double* RESTRICT cl[7] = {c0, c1, c2, c3, c4, c5, c6};
        for (int m = 0; m < Mp; ++m) {
            double P[7], Q[7]; P[0] = a[0][m]; Q[0] = b[0][m];
            for (int i = 1; i < 7; ++i) { double ai = a[i][m], bi = b[i][m], mc = mod[i], ms = mod[7 + i];
                P[i] = ai * mc - bi * ms; Q[i] = ai * ms + bi * mc; }
            const double pe1=P[1]+P[6],pe2=P[2]+P[5],pe3=P[3]+P[4],po1=P[1]-P[6],po2=P[2]-P[5],po3=P[3]-P[4];
            const double qe1=Q[1]+Q[6],qe2=Q[2]+Q[5],qe3=Q[3]+Q[4],qo1=Q[1]-Q[6],qo2=Q[2]-Q[5],qo3=Q[3]-Q[4];
            cl[0][m]=P[0]+pe1+pe2+pe3; cl[0][Mp+m]=Q[0]+qe1+qe2+qe3;
            const double Ec1=pe1*R7_C1+pe2*R7_C2+pe3*R7_C3,Ec2=pe1*R7_C2+pe2*R7_C3+pe3*R7_C1,Ec3=pe1*R7_C3+pe2*R7_C1+pe3*R7_C2;
            const double Os1=qo1*R7_S1+qo2*R7_S2+qo3*R7_S3,Os2=qo1*R7_S2-qo2*R7_S3-qo3*R7_S1,Os3=qo1*R7_S3-qo2*R7_S1+qo3*R7_S2;
            const double Fc1=qe1*R7_C1+qe2*R7_C2+qe3*R7_C3,Fc2=qe1*R7_C2+qe2*R7_C3+qe3*R7_C1,Fc3=qe1*R7_C3+qe2*R7_C1+qe3*R7_C2;
            const double Ps1=po1*R7_S1+po2*R7_S2+po3*R7_S3,Ps2=po1*R7_S2-po2*R7_S3-po3*R7_S1,Ps3=po1*R7_S3-po2*R7_S1+po3*R7_S2;
            cl[1][m]=P[0]+Ec1-Os1; cl[6][m]=P[0]+Ec1+Os1; cl[2][m]=P[0]+Ec2-Os2; cl[5][m]=P[0]+Ec2+Os2; cl[3][m]=P[0]+Ec3-Os3; cl[4][m]=P[0]+Ec3+Os3;
            cl[1][Mp+m]=Q[0]+Fc1+Ps1; cl[6][Mp+m]=Q[0]+Fc1-Ps1; cl[2][Mp+m]=Q[0]+Fc2+Ps2; cl[5][Mp+m]=Q[0]+Fc2-Ps2; cl[3][Mp+m]=Q[0]+Fc3+Ps3; cl[4][Mp+m]=Q[0]+Fc3-Ps3;
        }
    }

    static void bruun_odd7_adjoint_all(double* RESTRICT lo, double* RESTRICT hi,
                                       const double* RESTRICT c0, const double* RESTRICT c1, const double* RESTRICT c2, const double* RESTRICT c3,
                                       const double* RESTRICT c4, const double* RESTRICT c5, const double* RESTRICT c6,
                                       const double* RESTRICT mod, int Mp) {
        double* RESTRICT a[7] = {lo, lo+Mp, lo+2*Mp, lo+3*Mp, lo+4*Mp, lo+5*Mp, lo+6*Mp};
        double* RESTRICT b[7] = {hi, hi+Mp, hi+2*Mp, hi+3*Mp, hi+4*Mp, hi+5*Mp, hi+6*Mp};
        const double* RESTRICT cl[7] = {c0, c1, c2, c3, c4, c5, c6};
        const double sev = 1.0 / 7.0;
        for (int m = 0; m < Mp; ++m) {
            double L[7], H[7]; for (int t = 0; t < 7; ++t) { L[t] = cl[t][m]; H[t] = cl[t][Mp + m]; }
            const double ce1=L[1]+L[6],ce2=L[2]+L[5],ce3=L[3]+L[4],co1=L[1]-L[6],co2=L[2]-L[5],co3=L[3]-L[4];
            const double he1=H[1]+H[6],he2=H[2]+H[5],he3=H[3]+H[4],ho1=H[1]-H[6],ho2=H[2]-H[5],ho3=H[3]-H[4];
            double P[7], Q[7];
            P[0]=(L[0]+ce1+ce2+ce3)*sev; Q[0]=(H[0]+he1+he2+he3)*sev;
            const double EcA=ce1*R7_C1+ce2*R7_C2+ce3*R7_C3,EcB=ce1*R7_C2+ce2*R7_C3+ce3*R7_C1,EcC=ce1*R7_C3+ce2*R7_C1+ce3*R7_C2;
            const double OsA=ho1*R7_S1+ho2*R7_S2+ho3*R7_S3,OsB=ho1*R7_S2-ho2*R7_S3-ho3*R7_S1,OsC=ho1*R7_S3-ho2*R7_S1+ho3*R7_S2;
            const double FcA=he1*R7_C1+he2*R7_C2+he3*R7_C3,FcB=he1*R7_C2+he2*R7_C3+he3*R7_C1,FcC=he1*R7_C3+he2*R7_C1+he3*R7_C2;
            const double PsA=co1*R7_S1+co2*R7_S2+co3*R7_S3,PsB=co1*R7_S2-co2*R7_S3-co3*R7_S1,PsC=co1*R7_S3-co2*R7_S1+co3*R7_S2;
            P[1]=(L[0]+EcA+OsA)*sev;P[6]=(L[0]+EcA-OsA)*sev;P[2]=(L[0]+EcB+OsB)*sev;P[5]=(L[0]+EcB-OsB)*sev;P[3]=(L[0]+EcC+OsC)*sev;P[4]=(L[0]+EcC-OsC)*sev;
            Q[1]=(H[0]+FcA-PsA)*sev;Q[6]=(H[0]+FcA+PsA)*sev;Q[2]=(H[0]+FcB-PsB)*sev;Q[5]=(H[0]+FcB+PsB)*sev;Q[3]=(H[0]+FcC-PsC)*sev;Q[4]=(H[0]+FcC+PsC)*sev;
            a[0][m]=P[0]; b[0][m]=Q[0];
            for (int i = 1; i < 7; ++i) { double mc = mod[i], ms = mod[7 + i];
                a[i][m] = P[i] * mc + Q[i] * ms; b[i][m] = Q[i] * mc - P[i] * ms; }
        }
    }

    template <int P>
    static void fixed_odd_sum(const double* RESTRICT in, double* RESTRICT sum, int M) {
        int m = 0;
#if BRUUN_LEVEL >= 1
        for (; m + 1 < M; m += 2) {
            bruun_v2 acc = V2_LD(in + m);
            for (int i = 1; i < P; ++i) acc = V2_ADD(acc, V2_LD(in + i * M + m));
            V2_ST(sum + m, acc);
        }
#endif
        for (; m < M; ++m) {
            double acc = in[m];
            for (int i = 1; i < P; ++i) acc += in[i * M + m];
            sum[m] = acc;
        }
    }

    template <int P>
    static void fixed_real_projection(const double* RESTRICT in, double* RESTRICT lo,
                                      double* RESTRICT hi, const double* RESTRICT C,
                                      const double* RESTRICT S, int M) {
        int m = 0;
#if BRUUN_LEVEL >= 1
        bruun_v2 cv[P], sv[P];
        for (int i = 0; i < P; ++i) { cv[i] = V2_SET1(C[i]); sv[i] = V2_SET1(S[i]); }
        for (; m + 1 < M; m += 2) {
            bruun_v2 l = V2_MUL(V2_LD(in + m), cv[0]);
            bruun_v2 h = V2_MUL(V2_LD(in + m), sv[0]);
            for (int i = 1; i < P; ++i) {
                const bruun_v2 v = V2_LD(in + i * M + m);
                l = V2_MADD(l, v, cv[i]);
                h = V2_MADD(h, v, sv[i]);
            }
            V2_ST(lo + m, l);
            V2_ST(hi + m, h);
        }
#endif
        for (; m < M; ++m) {
            double l = in[m] * C[0], h = in[m] * S[0];
            for (int i = 1; i < P; ++i) {
                const double v = in[i * M + m];
                l += v * C[i];
                h += v * S[i];
            }
            lo[m] = l;
            hi[m] = h;
        }
    }

    static void minus_split3_inverse(double* RESTRICT r, const double* RESTRICT sum,
                                     const double* RESTRICT lo, const double* RESTRICT hi,
                                     const double* RESTRICT C, const double* RESTRICT S, int M) {
        double* RESTRICT r0 = r;
        double* RESTRICT r1 = r + M;
        double* RESTRICT r2 = r + 2 * M;
        int m = 0;
#if BRUUN_LEVEL >= 1
        const bruun_v2 dc = V2_SET1(1.0 / 3.0);
        const bruun_v2 two_thirds = V2_SET1(2.0 / 3.0);
        const bruun_v2 c0 = V2_SET1(C[0]), c1 = V2_SET1(C[1]), c2 = V2_SET1(C[2]);
        const bruun_v2 s0 = V2_SET1(S[0]), s1 = V2_SET1(S[1]), s2 = V2_SET1(S[2]);
        for (; m + 1 < M; m += 2) {
            const bruun_v2 base = V2_MUL(V2_LD(sum + m), dc);
            const bruun_v2 l = V2_LD(lo + m);
            const bruun_v2 h = V2_LD(hi + m);
            bruun_v2 a0 = V2_MADD(V2_MUL(l, c0), h, s0);
            bruun_v2 a1 = V2_MADD(V2_MUL(l, c1), h, s1);
            bruun_v2 a2 = V2_MADD(V2_MUL(l, c2), h, s2);
            V2_ST(r0 + m, V2_MADD(base, a0, two_thirds));
            V2_ST(r1 + m, V2_MADD(base, a1, two_thirds));
            V2_ST(r2 + m, V2_MADD(base, a2, two_thirds));
        }
#endif
        for (; m < M; ++m) {
            const double base = sum[m] / 3.0;
            r0[m] = base + (2.0 / 3.0) * (lo[m] * C[0] + hi[m] * S[0]);
            r1[m] = base + (2.0 / 3.0) * (lo[m] * C[1] + hi[m] * S[1]);
            r2[m] = base + (2.0 / 3.0) * (lo[m] * C[2] + hi[m] * S[2]);
        }
    }

    static void minus_split5_inverse_fused(double* RESTRICT r, const double* RESTRICT sum,
                                           const double* RESTRICT lo0, const double* RESTRICT hi0,
                                           const double* RESTRICT C0, const double* RESTRICT S0,
                                           const double* RESTRICT lo1, const double* RESTRICT hi1,
                                           const double* RESTRICT C1, const double* RESTRICT S1,
                                           int M) {
        constexpr int P = 5;
        const double dc_scale = 1.0 / 5.0;
        const double branch_scale = 2.0 / 5.0;
        int m = 0;
#if BRUUN_LEVEL >= 1
        bruun_v2 c0[P], s0[P], c1[P], s1[P];
        for (int i = 0; i < P; ++i) {
            c0[i] = V2_SET1(C0[i] * branch_scale);
            s0[i] = V2_SET1(S0[i] * branch_scale);
            c1[i] = V2_SET1(C1[i] * branch_scale);
            s1[i] = V2_SET1(S1[i] * branch_scale);
        }
        const bruun_v2 dc = V2_SET1(dc_scale);
        for (; m + 1 < M; m += 2) {
            const bruun_v2 base = V2_MUL(V2_LD(sum + m), dc);
            const bruun_v2 l0 = V2_LD(lo0 + m), h0 = V2_LD(hi0 + m);
            const bruun_v2 l1 = V2_LD(lo1 + m), h1 = V2_LD(hi1 + m);
            for (int i = 0; i < P; ++i) {
                bruun_v2 out = base;
                out = V2_MADD(out, l0, c0[i]);
                out = V2_MADD(out, h0, s0[i]);
                out = V2_MADD(out, l1, c1[i]);
                out = V2_MADD(out, h1, s1[i]);
                V2_ST(r + i * M + m, out);
            }
        }
#endif
        for (; m < M; ++m) {
            const double base = sum[m] * dc_scale;
            for (int i = 0; i < P; ++i) {
                r[i * M + m] = base
                             + lo0[m] * C0[i] * branch_scale
                             + hi0[m] * S0[i] * branch_scale
                             + lo1[m] * C1[i] * branch_scale
                             + hi1[m] * S1[i] * branch_scale;
            }
        }
    }

    static inline bruun_v2 madd_coeff(bruun_v2 acc, bruun_v2 x, double c) {
#if defined(BRUUN_NEON_128)
        return vfmaq_n_f64(acc, x, c);
#else
        return V2_MADD(acc, x, V2_SET1(c));
#endif
    }

    static void minus_split7_inverse_first(double* RESTRICT r,
                                           const double* RESTRICT sum,
                                           const double* RESTRICT lo,
                                           const double* RESTRICT hi,
                                           const double* RESTRICT C,
                                           const double* RESTRICT S, int M) {
        constexpr int P = 7;
        const double dc_scale = 1.0 / 7.0;
        const double branch_scale = 2.0 / 7.0;
        int m = 0;
#if BRUUN_LEVEL >= 1
        const bruun_v2 dc = V2_SET1(dc_scale);
        for (; m + 1 < M; m += 2) {
            const bruun_v2 base = V2_MUL(V2_LD(sum + m), dc);
            const bruun_v2 l = V2_LD(lo + m);
            const bruun_v2 h = V2_LD(hi + m);
            bruun_v2 out = base;
            out = madd_coeff(out, l, C[0] * branch_scale);
            out = madd_coeff(out, h, S[0] * branch_scale);
            V2_ST(r + m, out);
            out = base;
            out = madd_coeff(out, l, C[1] * branch_scale);
            out = madd_coeff(out, h, S[1] * branch_scale);
            V2_ST(r + M + m, out);
            out = base;
            out = madd_coeff(out, l, C[2] * branch_scale);
            out = madd_coeff(out, h, S[2] * branch_scale);
            V2_ST(r + 2 * M + m, out);
            out = base;
            out = madd_coeff(out, l, C[3] * branch_scale);
            out = madd_coeff(out, h, S[3] * branch_scale);
            V2_ST(r + 3 * M + m, out);
            out = base;
            out = madd_coeff(out, l, C[4] * branch_scale);
            out = madd_coeff(out, h, S[4] * branch_scale);
            V2_ST(r + 4 * M + m, out);
            out = base;
            out = madd_coeff(out, l, C[5] * branch_scale);
            out = madd_coeff(out, h, S[5] * branch_scale);
            V2_ST(r + 5 * M + m, out);
            out = base;
            out = madd_coeff(out, l, C[6] * branch_scale);
            out = madd_coeff(out, h, S[6] * branch_scale);
            V2_ST(r + 6 * M + m, out);
        }
#endif
        for (; m < M; ++m) {
            const double base = sum[m] * dc_scale;
            const double l = lo[m], h = hi[m];
            for (int i = 0; i < P; ++i) {
                r[i * M + m] = base + (l * C[i] + h * S[i]) * branch_scale;
            }
        }
    }

    static void minus_split7_inverse_add(double* RESTRICT r,
                                         const double* RESTRICT lo,
                                         const double* RESTRICT hi,
                                         const double* RESTRICT C,
                                         const double* RESTRICT S, int M) {
        constexpr int P = 7;
        const double branch_scale = 2.0 / 7.0;
        int m = 0;
#if BRUUN_LEVEL >= 1
        for (; m + 1 < M; m += 2) {
            const bruun_v2 l = V2_LD(lo + m);
            const bruun_v2 h = V2_LD(hi + m);
            bruun_v2 out = V2_LD(r + m);
            out = madd_coeff(out, l, C[0] * branch_scale);
            out = madd_coeff(out, h, S[0] * branch_scale);
            V2_ST(r + m, out);
            out = V2_LD(r + M + m);
            out = madd_coeff(out, l, C[1] * branch_scale);
            out = madd_coeff(out, h, S[1] * branch_scale);
            V2_ST(r + M + m, out);
            out = V2_LD(r + 2 * M + m);
            out = madd_coeff(out, l, C[2] * branch_scale);
            out = madd_coeff(out, h, S[2] * branch_scale);
            V2_ST(r + 2 * M + m, out);
            out = V2_LD(r + 3 * M + m);
            out = madd_coeff(out, l, C[3] * branch_scale);
            out = madd_coeff(out, h, S[3] * branch_scale);
            V2_ST(r + 3 * M + m, out);
            out = V2_LD(r + 4 * M + m);
            out = madd_coeff(out, l, C[4] * branch_scale);
            out = madd_coeff(out, h, S[4] * branch_scale);
            V2_ST(r + 4 * M + m, out);
            out = V2_LD(r + 5 * M + m);
            out = madd_coeff(out, l, C[5] * branch_scale);
            out = madd_coeff(out, h, S[5] * branch_scale);
            V2_ST(r + 5 * M + m, out);
            out = V2_LD(r + 6 * M + m);
            out = madd_coeff(out, l, C[6] * branch_scale);
            out = madd_coeff(out, h, S[6] * branch_scale);
            V2_ST(r + 6 * M + m, out);
        }
#endif
        for (; m < M; ++m) {
            const double l = lo[m], h = hi[m];
            for (int i = 0; i < P; ++i) r[i * M + m] += (l * C[i] + h * S[i]) * branch_scale;
        }
    }

    template <int P>
    static void fixed_minus_split_inverse_add(double* RESTRICT r,
                                              const double* RESTRICT lo,
                                              const double* RESTRICT hi,
                                              const double* RESTRICT C,
                                              const double* RESTRICT S, int M) {
        const double scale = 2.0 / static_cast<double>(P);
        int m = 0;
#if BRUUN_LEVEL >= 1
        bruun_v2 cv[P], sv[P];
        for (int i = 0; i < P; ++i) { cv[i] = V2_SET1(C[i] * scale); sv[i] = V2_SET1(S[i] * scale); }
        for (; m + 1 < M; m += 2) {
            const bruun_v2 l = V2_LD(lo + m);
            const bruun_v2 h = V2_LD(hi + m);
            for (int i = 0; i < P; ++i) {
                bruun_v2 acc = V2_LD(r + i * M + m);
                acc = V2_MADD(acc, l, cv[i]);
                acc = V2_MADD(acc, h, sv[i]);
                V2_ST(r + i * M + m, acc);
            }
        }
#endif
        for (; m < M; ++m) {
            const double l = lo[m], h = hi[m];
            for (int i = 0; i < P; ++i) r[i * M + m] += l * C[i] * scale + h * S[i] * scale;
        }
    }

    template <int P>
    static void fixed_minus_split_inverse_first(double* RESTRICT r,
                                                const double* RESTRICT sum,
                                                const double* RESTRICT lo,
                                                const double* RESTRICT hi,
                                                const double* RESTRICT C,
                                                const double* RESTRICT S, int M) {
        const double dc_scale = 1.0 / static_cast<double>(P);
        const double branch_scale = 2.0 / static_cast<double>(P);
        int m = 0;
#if BRUUN_LEVEL >= 1
        bruun_v2 cv[P], sv[P];
        for (int i = 0; i < P; ++i) { cv[i] = V2_SET1(C[i] * branch_scale); sv[i] = V2_SET1(S[i] * branch_scale); }
        const bruun_v2 dc = V2_SET1(dc_scale);
        for (; m + 1 < M; m += 2) {
            const bruun_v2 base = V2_MUL(V2_LD(sum + m), dc);
            const bruun_v2 l = V2_LD(lo + m);
            const bruun_v2 h = V2_LD(hi + m);
            for (int i = 0; i < P; ++i) {
                bruun_v2 out = base;
                out = V2_MADD(out, l, cv[i]);
                out = V2_MADD(out, h, sv[i]);
                V2_ST(r + i * M + m, out);
            }
        }
#endif
        for (; m < M; ++m) {
            const double base = sum[m] * dc_scale;
            const double l = lo[m], h = hi[m];
            for (int i = 0; i < P; ++i) r[i * M + m] = base + l * C[i] * branch_scale + h * S[i] * branch_scale;
        }
    }

    template <int P>
    static void fixed_bruun_odd_forward_child(double* RESTRICT clo, double* RESTRICT chi,
                                              const double* RESTRICT lo, const double* RESTRICT hi,
                                              const double* RESTRICT C, const double* RESTRICT S,
                                              int Mp) {
        int m = 0;
#if BRUUN_LEVEL >= 1
        bruun_v2 cv[P], sv[P];
        for (int i = 0; i < P; ++i) { cv[i] = V2_SET1(C[i]); sv[i] = V2_SET1(S[i]); }
        for (; m + 1 < Mp; m += 2) {
            const bruun_v2 a0 = V2_LD(lo + m);
            const bruun_v2 b0 = V2_LD(hi + m);
            bruun_v2 out_l = V2_MSUB(V2_MUL(a0, cv[0]), b0, sv[0]);
            bruun_v2 out_h = V2_MADD(V2_MUL(b0, cv[0]), a0, sv[0]);
            for (int i = 1; i < P; ++i) {
                const bruun_v2 a = V2_LD(lo + i * Mp + m);
                const bruun_v2 b = V2_LD(hi + i * Mp + m);
                out_l = V2_MADD(out_l, a, cv[i]);
                out_l = V2_MSUB(out_l, b, sv[i]);
                out_h = V2_MADD(out_h, b, cv[i]);
                out_h = V2_MADD(out_h, a, sv[i]);
            }
            V2_ST(clo + m, out_l);
            V2_ST(chi + m, out_h);
        }
#endif
        for (; m < Mp; ++m) {
            double out_l = lo[m] * C[0] - hi[m] * S[0];
            double out_h = hi[m] * C[0] + lo[m] * S[0];
            for (int i = 1; i < P; ++i) {
                out_l += lo[i * Mp + m] * C[i] - hi[i * Mp + m] * S[i];
                out_h += hi[i * Mp + m] * C[i] + lo[i * Mp + m] * S[i];
            }
            clo[m] = out_l;
            chi[m] = out_h;
        }
    }

    template <int P>
    static void fixed_bruun_odd_adjoint_child(double* RESTRICT lo, double* RESTRICT hi,
                                              const double* RESTRICT clo,
                                              const double* RESTRICT chi,
                                              const double* RESTRICT C,
                                              const double* RESTRICT S, int Mp) {
        const double scale = 1.0 / static_cast<double>(P);
        int m = 0;
#if BRUUN_LEVEL >= 1
        bruun_v2 cv[P], sv[P];
        for (int i = 0; i < P; ++i) { cv[i] = V2_SET1(C[i] * scale); sv[i] = V2_SET1(S[i] * scale); }
        for (; m + 1 < Mp; m += 2) {
            const bruun_v2 cl = V2_LD(clo + m);
            const bruun_v2 ch = V2_LD(chi + m);
            for (int i = 0; i < P; ++i) {
                bruun_v2 l = V2_LD(lo + i * Mp + m);
                bruun_v2 h = V2_LD(hi + i * Mp + m);
                l = V2_MADD(l, cl, cv[i]);
                l = V2_MADD(l, ch, sv[i]);
                h = V2_MADD(h, ch, cv[i]);
                h = V2_MSUB(h, cl, sv[i]);
                V2_ST(lo + i * Mp + m, l);
                V2_ST(hi + i * Mp + m, h);
            }
        }
#endif
        for (; m < Mp; ++m) {
            const double cl = clo[m], ch = chi[m];
            for (int i = 0; i < P; ++i) {
                lo[i * Mp + m] += cl * C[i] * scale + ch * S[i] * scale;
                hi[i * Mp + m] += ch * C[i] * scale - cl * S[i] * scale;
            }
        }
    }

    static void zero_block_f32(float* RESTRICT dst, int n) {
        int m = 0;
#if BRUUN_LEVEL >= 1
        const bruun_v4f z = V4F_ZERO();
        for (; m + 3 < n; m += 4) V4F_ST(dst + m, z);
#endif
        for (; m < n; ++m) dst[m] = 0.0f;
    }

    template <int P>
    static void fixed_odd_sum_f32(const float* RESTRICT in, float* RESTRICT sum, int M) {
        int m = 0;
#if BRUUN_LEVEL >= 1
        for (; m + 3 < M; m += 4) {
            bruun_v4f acc = V4F_LD(in + m);
            for (int i = 1; i < P; ++i) acc = V4F_ADD(acc, V4F_LD(in + i * M + m));
            V4F_ST(sum + m, acc);
        }
#endif
        for (; m < M; ++m) {
            float acc = in[m];
            for (int i = 1; i < P; ++i) acc += in[i * M + m];
            sum[m] = acc;
        }
    }

    template <int P>
    static void fixed_real_projection_f32(const float* RESTRICT in, float* RESTRICT lo,
                                          float* RESTRICT hi, const double* RESTRICT C,
                                          const double* RESTRICT S, int M) {
        int m = 0;
#if BRUUN_LEVEL >= 1
        bruun_v4f cv[P], sv[P];
        for (int i = 0; i < P; ++i) {
            cv[i] = V4F_SET1(static_cast<float>(C[i]));
            sv[i] = V4F_SET1(static_cast<float>(S[i]));
        }
        for (; m + 3 < M; m += 4) {
            const bruun_v4f v0 = V4F_LD(in + m);
            bruun_v4f l = V4F_MUL(v0, cv[0]);
            bruun_v4f h = V4F_MUL(v0, sv[0]);
            for (int i = 1; i < P; ++i) {
                const bruun_v4f v = V4F_LD(in + i * M + m);
                l = V4F_MADD(l, v, cv[i]);
                h = V4F_MADD(h, v, sv[i]);
            }
            V4F_ST(lo + m, l);
            V4F_ST(hi + m, h);
        }
#endif
        for (; m < M; ++m) {
            float l = in[m] * static_cast<float>(C[0]);
            float h = in[m] * static_cast<float>(S[0]);
            for (int i = 1; i < P; ++i) {
                const float v = in[i * M + m];
                l += v * static_cast<float>(C[i]);
                h += v * static_cast<float>(S[i]);
            }
            lo[m] = l;
            hi[m] = h;
        }
    }

    static void minus_split5_inverse_fused_f32(float* RESTRICT r, const float* RESTRICT sum,
                                               const float* RESTRICT lo0, const float* RESTRICT hi0,
                                               const double* RESTRICT C0, const double* RESTRICT S0,
                                               const float* RESTRICT lo1, const float* RESTRICT hi1,
                                               const double* RESTRICT C1, const double* RESTRICT S1,
                                               int M) {
        constexpr int P = 5;
        const float dc_scale = 1.0f / 5.0f;
        const float branch_scale = 2.0f / 5.0f;
        int m = 0;
#if BRUUN_LEVEL >= 1
        bruun_v4f c0[P], s0[P], c1[P], s1[P];
        for (int i = 0; i < P; ++i) {
            c0[i] = V4F_SET1(static_cast<float>(C0[i]) * branch_scale);
            s0[i] = V4F_SET1(static_cast<float>(S0[i]) * branch_scale);
            c1[i] = V4F_SET1(static_cast<float>(C1[i]) * branch_scale);
            s1[i] = V4F_SET1(static_cast<float>(S1[i]) * branch_scale);
        }
        const bruun_v4f dc = V4F_SET1(dc_scale);
        for (; m + 3 < M; m += 4) {
            const bruun_v4f base = V4F_MUL(V4F_LD(sum + m), dc);
            const bruun_v4f l0 = V4F_LD(lo0 + m), h0 = V4F_LD(hi0 + m);
            const bruun_v4f l1 = V4F_LD(lo1 + m), h1 = V4F_LD(hi1 + m);
            for (int i = 0; i < P; ++i) {
                bruun_v4f out = base;
                out = V4F_MADD(out, l0, c0[i]);
                out = V4F_MADD(out, h0, s0[i]);
                out = V4F_MADD(out, l1, c1[i]);
                out = V4F_MADD(out, h1, s1[i]);
                V4F_ST(r + i * M + m, out);
            }
        }
#endif
        for (; m < M; ++m) {
            const float base = sum[m] * dc_scale;
            for (int i = 0; i < P; ++i) {
                r[i * M + m] = base
                             + lo0[m] * static_cast<float>(C0[i]) * branch_scale
                             + hi0[m] * static_cast<float>(S0[i]) * branch_scale
                             + lo1[m] * static_cast<float>(C1[i]) * branch_scale
                             + hi1[m] * static_cast<float>(S1[i]) * branch_scale;
            }
        }
    }

    static inline bruun_v4f madd_coeff_f32(bruun_v4f acc, bruun_v4f x, float c) {
#if defined(BRUUN_NEON_128)
        return vfmaq_n_f32(acc, x, c);
#else
        return V4F_MADD(acc, x, V4F_SET1(c));
#endif
    }

    static void minus_split7_inverse_first_f32(float* RESTRICT r,
                                               const float* RESTRICT sum,
                                               const float* RESTRICT lo,
                                               const float* RESTRICT hi,
                                               const double* RESTRICT C,
                                               const double* RESTRICT S, int M) {
        constexpr int P = 7;
        const float dc_scale = 1.0f / 7.0f;
        const float branch_scale = 2.0f / 7.0f;
        int m = 0;
#if BRUUN_LEVEL >= 1
        const bruun_v4f dc = V4F_SET1(dc_scale);
        for (; m + 3 < M; m += 4) {
            const bruun_v4f base = V4F_MUL(V4F_LD(sum + m), dc);
            const bruun_v4f l = V4F_LD(lo + m);
            const bruun_v4f h = V4F_LD(hi + m);
            bruun_v4f out = base;
            out = madd_coeff_f32(out, l, static_cast<float>(C[0]) * branch_scale);
            out = madd_coeff_f32(out, h, static_cast<float>(S[0]) * branch_scale);
            V4F_ST(r + m, out);
            out = base;
            out = madd_coeff_f32(out, l, static_cast<float>(C[1]) * branch_scale);
            out = madd_coeff_f32(out, h, static_cast<float>(S[1]) * branch_scale);
            V4F_ST(r + M + m, out);
            out = base;
            out = madd_coeff_f32(out, l, static_cast<float>(C[2]) * branch_scale);
            out = madd_coeff_f32(out, h, static_cast<float>(S[2]) * branch_scale);
            V4F_ST(r + 2 * M + m, out);
            out = base;
            out = madd_coeff_f32(out, l, static_cast<float>(C[3]) * branch_scale);
            out = madd_coeff_f32(out, h, static_cast<float>(S[3]) * branch_scale);
            V4F_ST(r + 3 * M + m, out);
            out = base;
            out = madd_coeff_f32(out, l, static_cast<float>(C[4]) * branch_scale);
            out = madd_coeff_f32(out, h, static_cast<float>(S[4]) * branch_scale);
            V4F_ST(r + 4 * M + m, out);
            out = base;
            out = madd_coeff_f32(out, l, static_cast<float>(C[5]) * branch_scale);
            out = madd_coeff_f32(out, h, static_cast<float>(S[5]) * branch_scale);
            V4F_ST(r + 5 * M + m, out);
            out = base;
            out = madd_coeff_f32(out, l, static_cast<float>(C[6]) * branch_scale);
            out = madd_coeff_f32(out, h, static_cast<float>(S[6]) * branch_scale);
            V4F_ST(r + 6 * M + m, out);
        }
#endif
        for (; m < M; ++m) {
            const float base = sum[m] * dc_scale;
            const float l = lo[m], h = hi[m];
            for (int i = 0; i < P; ++i) {
                r[i * M + m] = base + (l * static_cast<float>(C[i]) + h * static_cast<float>(S[i])) * branch_scale;
            }
        }
    }

    static void minus_split7_inverse_add_f32(float* RESTRICT r,
                                             const float* RESTRICT lo,
                                             const float* RESTRICT hi,
                                             const double* RESTRICT C,
                                             const double* RESTRICT S, int M) {
        constexpr int P = 7;
        const float branch_scale = 2.0f / 7.0f;
        int m = 0;
#if BRUUN_LEVEL >= 1
        for (; m + 3 < M; m += 4) {
            const bruun_v4f l = V4F_LD(lo + m);
            const bruun_v4f h = V4F_LD(hi + m);
            bruun_v4f out = V4F_LD(r + m);
            out = madd_coeff_f32(out, l, static_cast<float>(C[0]) * branch_scale);
            out = madd_coeff_f32(out, h, static_cast<float>(S[0]) * branch_scale);
            V4F_ST(r + m, out);
            out = V4F_LD(r + M + m);
            out = madd_coeff_f32(out, l, static_cast<float>(C[1]) * branch_scale);
            out = madd_coeff_f32(out, h, static_cast<float>(S[1]) * branch_scale);
            V4F_ST(r + M + m, out);
            out = V4F_LD(r + 2 * M + m);
            out = madd_coeff_f32(out, l, static_cast<float>(C[2]) * branch_scale);
            out = madd_coeff_f32(out, h, static_cast<float>(S[2]) * branch_scale);
            V4F_ST(r + 2 * M + m, out);
            out = V4F_LD(r + 3 * M + m);
            out = madd_coeff_f32(out, l, static_cast<float>(C[3]) * branch_scale);
            out = madd_coeff_f32(out, h, static_cast<float>(S[3]) * branch_scale);
            V4F_ST(r + 3 * M + m, out);
            out = V4F_LD(r + 4 * M + m);
            out = madd_coeff_f32(out, l, static_cast<float>(C[4]) * branch_scale);
            out = madd_coeff_f32(out, h, static_cast<float>(S[4]) * branch_scale);
            V4F_ST(r + 4 * M + m, out);
            out = V4F_LD(r + 5 * M + m);
            out = madd_coeff_f32(out, l, static_cast<float>(C[5]) * branch_scale);
            out = madd_coeff_f32(out, h, static_cast<float>(S[5]) * branch_scale);
            V4F_ST(r + 5 * M + m, out);
            out = V4F_LD(r + 6 * M + m);
            out = madd_coeff_f32(out, l, static_cast<float>(C[6]) * branch_scale);
            out = madd_coeff_f32(out, h, static_cast<float>(S[6]) * branch_scale);
            V4F_ST(r + 6 * M + m, out);
        }
#endif
        for (; m < M; ++m) {
            const float l = lo[m], h = hi[m];
            for (int i = 0; i < P; ++i) {
                r[i * M + m] += (l * static_cast<float>(C[i]) + h * static_cast<float>(S[i])) * branch_scale;
            }
        }
    }

    template <int P>
    static void fixed_minus_split_inverse_add_f32(float* RESTRICT r,
                                                  const float* RESTRICT lo,
                                                  const float* RESTRICT hi,
                                                  const double* RESTRICT C,
                                                  const double* RESTRICT S, int M) {
        const float scale = 2.0f / static_cast<float>(P);
        int m = 0;
#if BRUUN_LEVEL >= 1
        bruun_v4f cv[P], sv[P];
        for (int i = 0; i < P; ++i) {
            cv[i] = V4F_SET1(static_cast<float>(C[i]) * scale);
            sv[i] = V4F_SET1(static_cast<float>(S[i]) * scale);
        }
        for (; m + 3 < M; m += 4) {
            const bruun_v4f l = V4F_LD(lo + m);
            const bruun_v4f h = V4F_LD(hi + m);
            for (int i = 0; i < P; ++i) {
                bruun_v4f acc = V4F_LD(r + i * M + m);
                acc = V4F_MADD(acc, l, cv[i]);
                acc = V4F_MADD(acc, h, sv[i]);
                V4F_ST(r + i * M + m, acc);
            }
        }
#endif
        for (; m < M; ++m) {
            const float l = lo[m], h = hi[m];
            for (int i = 0; i < P; ++i) {
                r[i * M + m] += l * static_cast<float>(C[i]) * scale
                              + h * static_cast<float>(S[i]) * scale;
            }
        }
    }

    template <int P>
    static void fixed_minus_split_inverse_first_f32(float* RESTRICT r,
                                                    const float* RESTRICT sum,
                                                    const float* RESTRICT lo,
                                                    const float* RESTRICT hi,
                                                    const double* RESTRICT C,
                                                    const double* RESTRICT S, int M) {
        const float dc_scale = 1.0f / static_cast<float>(P);
        const float branch_scale = 2.0f / static_cast<float>(P);
        int m = 0;
#if BRUUN_LEVEL >= 1
        bruun_v4f cv[P], sv[P];
        for (int i = 0; i < P; ++i) {
            cv[i] = V4F_SET1(static_cast<float>(C[i]) * branch_scale);
            sv[i] = V4F_SET1(static_cast<float>(S[i]) * branch_scale);
        }
        const bruun_v4f dc = V4F_SET1(dc_scale);
        for (; m + 3 < M; m += 4) {
            const bruun_v4f base = V4F_MUL(V4F_LD(sum + m), dc);
            const bruun_v4f l = V4F_LD(lo + m);
            const bruun_v4f h = V4F_LD(hi + m);
            for (int i = 0; i < P; ++i) {
                bruun_v4f out = base;
                out = V4F_MADD(out, l, cv[i]);
                out = V4F_MADD(out, h, sv[i]);
                V4F_ST(r + i * M + m, out);
            }
        }
#endif
        for (; m < M; ++m) {
            const float base = sum[m] * dc_scale;
            const float l = lo[m], h = hi[m];
            for (int i = 0; i < P; ++i) {
                r[i * M + m] = base
                             + l * static_cast<float>(C[i]) * branch_scale
                             + h * static_cast<float>(S[i]) * branch_scale;
            }
        }
    }

    template <int P>
    static void fixed_bruun_odd_forward_child_f32(float* RESTRICT clo, float* RESTRICT chi,
                                                  const float* RESTRICT lo,
                                                  const float* RESTRICT hi,
                                                  const double* RESTRICT C,
                                                  const double* RESTRICT S, int Mp) {
        int m = 0;
#if BRUUN_LEVEL >= 1
        bruun_v4f cv[P], sv[P];
        for (int i = 0; i < P; ++i) {
            cv[i] = V4F_SET1(static_cast<float>(C[i]));
            sv[i] = V4F_SET1(static_cast<float>(S[i]));
        }
        for (; m + 3 < Mp; m += 4) {
            const bruun_v4f a0 = V4F_LD(lo + m);
            const bruun_v4f b0 = V4F_LD(hi + m);
            bruun_v4f out_l = V4F_MSUB(V4F_MUL(a0, cv[0]), b0, sv[0]);
            bruun_v4f out_h = V4F_MADD(V4F_MUL(b0, cv[0]), a0, sv[0]);
            for (int i = 1; i < P; ++i) {
                const bruun_v4f a = V4F_LD(lo + i * Mp + m);
                const bruun_v4f b = V4F_LD(hi + i * Mp + m);
                out_l = V4F_MADD(out_l, a, cv[i]);
                out_l = V4F_MSUB(out_l, b, sv[i]);
                out_h = V4F_MADD(out_h, b, cv[i]);
                out_h = V4F_MADD(out_h, a, sv[i]);
            }
            V4F_ST(clo + m, out_l);
            V4F_ST(chi + m, out_h);
        }
#endif
        for (; m < Mp; ++m) {
            float out_l = lo[m] * static_cast<float>(C[0]) - hi[m] * static_cast<float>(S[0]);
            float out_h = hi[m] * static_cast<float>(C[0]) + lo[m] * static_cast<float>(S[0]);
            for (int i = 1; i < P; ++i) {
                out_l += lo[i * Mp + m] * static_cast<float>(C[i])
                       - hi[i * Mp + m] * static_cast<float>(S[i]);
                out_h += hi[i * Mp + m] * static_cast<float>(C[i])
                       + lo[i * Mp + m] * static_cast<float>(S[i]);
            }
            clo[m] = out_l;
            chi[m] = out_h;
        }
    }

    template <int P>
    static void fixed_bruun_odd_adjoint_child_f32(float* RESTRICT lo, float* RESTRICT hi,
                                                  const float* RESTRICT clo,
                                                  const float* RESTRICT chi,
                                                  const double* RESTRICT C,
                                                  const double* RESTRICT S, int Mp) {
        const float scale = 1.0f / static_cast<float>(P);
        int m = 0;
#if BRUUN_LEVEL >= 1
        bruun_v4f cv[P], sv[P];
        for (int i = 0; i < P; ++i) {
            cv[i] = V4F_SET1(static_cast<float>(C[i]) * scale);
            sv[i] = V4F_SET1(static_cast<float>(S[i]) * scale);
        }
        for (; m + 3 < Mp; m += 4) {
            const bruun_v4f cl = V4F_LD(clo + m);
            const bruun_v4f ch = V4F_LD(chi + m);
            for (int i = 0; i < P; ++i) {
                bruun_v4f l = V4F_LD(lo + i * Mp + m);
                bruun_v4f h = V4F_LD(hi + i * Mp + m);
                l = V4F_MADD(l, cl, cv[i]);
                l = V4F_MADD(l, ch, sv[i]);
                h = V4F_MADD(h, ch, cv[i]);
                h = V4F_MSUB(h, cl, sv[i]);
                V4F_ST(lo + i * Mp + m, l);
                V4F_ST(hi + i * Mp + m, h);
            }
        }
#endif
        for (; m < Mp; ++m) {
            const float cl = clo[m], ch = chi[m];
            for (int i = 0; i < P; ++i) {
                lo[i * Mp + m] += cl * static_cast<float>(C[i]) * scale
                                + ch * static_cast<float>(S[i]) * scale;
                hi[i * Mp + m] += ch * static_cast<float>(C[i]) * scale
                                - cl * static_cast<float>(S[i]) * scale;
            }
        }
    }

    bool dispatch_minus_split_forward(const Op& o, int j, const double* RESTRICT in,
                                      double* RESTRICT sum, double* RESTRICT lo,
                                      double* RESTRICT hi, const double* RESTRICT C,
                                      const double* RESTRICT S, int M) const {
        switch (o.codelet) {
            case CODELET_3:
                minus_split3_forward(in, sum, lo, hi, C, S, M);
                return true;
            case CODELET_5:
                if (j == 1) fixed_odd_sum<5>(in, sum, M);
                fixed_real_projection<5>(in, lo, hi, C, S, M);
                return true;
            case CODELET_7:
                if (j == 1) fixed_odd_sum<7>(in, sum, M);
                fixed_real_projection<7>(in, lo, hi, C, S, M);
                return true;
            case CODELET_11:
                if (j == 1) fixed_odd_sum<11>(in, sum, M);
                fixed_real_projection<11>(in, lo, hi, C, S, M);
                return true;
            case CODELET_13:
                if (j == 1) fixed_odd_sum<13>(in, sum, M);
                fixed_real_projection<13>(in, lo, hi, C, S, M);
                return true;
            case CODELET_17:
                if (j == 1) fixed_odd_sum<17>(in, sum, M);
                fixed_real_projection<17>(in, lo, hi, C, S, M);
                return true;
            case CODELET_19:
                if (j == 1) fixed_odd_sum<19>(in, sum, M);
                fixed_real_projection<19>(in, lo, hi, C, S, M);
                return true;
            default:
                return false;
        }
    }

    bool dispatch_bruun_odd_forward(const Op& o, double* RESTRICT clo,
                                    double* RESTRICT chi, const double* RESTRICT lo,
                                    const double* RESTRICT hi, const double* RESTRICT C,
                                    const double* RESTRICT S, int Mp) const {
        switch (o.codelet) {
            case CODELET_3:
                bruun_odd3_forward_child(clo, chi, lo, hi, C, S, Mp);
                return true;
            case CODELET_5:
                fixed_bruun_odd_forward_child<5>(clo, chi, lo, hi, C, S, Mp);
                return true;
            case CODELET_7:
                fixed_bruun_odd_forward_child<7>(clo, chi, lo, hi, C, S, Mp);
                return true;
            case CODELET_11:
                fixed_bruun_odd_forward_child<11>(clo, chi, lo, hi, C, S, Mp);
                return true;
            case CODELET_13:
                fixed_bruun_odd_forward_child<13>(clo, chi, lo, hi, C, S, Mp);
                return true;
            case CODELET_17:
                fixed_bruun_odd_forward_child<17>(clo, chi, lo, hi, C, S, Mp);
                return true;
            case CODELET_19:
                fixed_bruun_odd_forward_child<19>(clo, chi, lo, hi, C, S, Mp);
                return true;
            default:
                return false;
        }
    }

    bool dispatch_bruun_odd_adjoint(const Op& o, double* RESTRICT lo,
                                    double* RESTRICT hi, const double* RESTRICT clo,
                                    const double* RESTRICT chi, const double* RESTRICT C,
                                    const double* RESTRICT S, int Mp) const {
        switch (o.codelet) {
            case CODELET_3:
                bruun_odd3_adjoint_child(lo, hi, clo, chi, C, S, Mp);
                return true;
            case CODELET_5:
                fixed_bruun_odd_adjoint_child<5>(lo, hi, clo, chi, C, S, Mp);
                return true;
            case CODELET_7:
                fixed_bruun_odd_adjoint_child<7>(lo, hi, clo, chi, C, S, Mp);
                return true;
            case CODELET_11:
                fixed_bruun_odd_adjoint_child<11>(lo, hi, clo, chi, C, S, Mp);
                return true;
            case CODELET_13:
                fixed_bruun_odd_adjoint_child<13>(lo, hi, clo, chi, C, S, Mp);
                return true;
            case CODELET_17:
                fixed_bruun_odd_adjoint_child<17>(lo, hi, clo, chi, C, S, Mp);
                return true;
            case CODELET_19:
                fixed_bruun_odd_adjoint_child<19>(lo, hi, clo, chi, C, S, Mp);
                return true;
            default:
                return false;
        }
    }

    bool dispatch_minus_split_inverse(const Op& o, double* RESTRICT A) const {
        if (o.codelet == CODELET_3) {
            const int M = o.M; double* r = A + o.in; const double* sum = A + o.sum;
            const double* C = &twp_[o.tw[0]]; const double* S = C + 3;
            const double* lo = A + o.child[0]; const double* hi = lo + M;
            minus_split3_inverse(r, sum, lo, hi, C, S, M);
            return true;
        }
        if (o.codelet == CODELET_5) {
            const int M = o.M; double* r = A + o.in; const double* sum = A + o.sum;
            const double* C0 = &twp_[o.tw[0]]; const double* S0 = C0 + 5;
            const double* C1 = &twp_[o.tw[1]]; const double* S1 = C1 + 5;
            const double* lo0 = A + o.child[0]; const double* hi0 = lo0 + M;
            const double* lo1 = A + o.child[1]; const double* hi1 = lo1 + M;
            minus_split5_inverse_fused(r, sum, lo0, hi0, C0, S0, lo1, hi1, C1, S1, M);
            return true;
        }
        const int p = o.p, M = o.M;
        double* r = A + o.in;
        const double* sum = A + o.sum;
        for (int j = 1; j <= (p - 1) / 2; ++j) {
            const double* C = &twp_[o.tw[j - 1]];
            const double* S = C + p;
            const double* lo = A + o.child[j - 1];
            const double* hi = lo + M;
            switch (o.codelet) {
                case CODELET_7:
                    if (j == 1) minus_split7_inverse_first(r, sum, lo, hi, C, S, M);
                    else minus_split7_inverse_add(r, lo, hi, C, S, M);
                    break;
                case CODELET_11:
                    if (j == 1) fixed_minus_split_inverse_first<11>(r, sum, lo, hi, C, S, M);
                    else fixed_minus_split_inverse_add<11>(r, lo, hi, C, S, M);
                    break;
                case CODELET_13:
                    if (j == 1) fixed_minus_split_inverse_first<13>(r, sum, lo, hi, C, S, M);
                    else fixed_minus_split_inverse_add<13>(r, lo, hi, C, S, M);
                    break;
                case CODELET_17:
                    if (j == 1) fixed_minus_split_inverse_first<17>(r, sum, lo, hi, C, S, M);
                    else fixed_minus_split_inverse_add<17>(r, lo, hi, C, S, M);
                    break;
                case CODELET_19:
                    if (j == 1) fixed_minus_split_inverse_first<19>(r, sum, lo, hi, C, S, M);
                    else fixed_minus_split_inverse_add<19>(r, lo, hi, C, S, M);
                    break;
                default:
                    return false;
            }
        }
        return o.codelet != CODELET_GENERIC;
    }

    bool dispatch_minus_split_forward_f32(const Op& o, int j, const float* RESTRICT in,
                                          float* RESTRICT sum, float* RESTRICT lo,
                                          float* RESTRICT hi, const double* RESTRICT C,
                                          const double* RESTRICT S, int M) const {
        switch (o.codelet) {
            case CODELET_3:
                if (j == 1) fixed_odd_sum_f32<3>(in, sum, M);
                fixed_real_projection_f32<3>(in, lo, hi, C, S, M);
                return true;
            case CODELET_5:
                if (j == 1) fixed_odd_sum_f32<5>(in, sum, M);
                fixed_real_projection_f32<5>(in, lo, hi, C, S, M);
                return true;
            case CODELET_7:
                if (j == 1) fixed_odd_sum_f32<7>(in, sum, M);
                fixed_real_projection_f32<7>(in, lo, hi, C, S, M);
                return true;
            case CODELET_11:
                if (j == 1) fixed_odd_sum_f32<11>(in, sum, M);
                fixed_real_projection_f32<11>(in, lo, hi, C, S, M);
                return true;
            case CODELET_13:
                if (j == 1) fixed_odd_sum_f32<13>(in, sum, M);
                fixed_real_projection_f32<13>(in, lo, hi, C, S, M);
                return true;
            case CODELET_17:
                if (j == 1) fixed_odd_sum_f32<17>(in, sum, M);
                fixed_real_projection_f32<17>(in, lo, hi, C, S, M);
                return true;
            case CODELET_19:
                if (j == 1) fixed_odd_sum_f32<19>(in, sum, M);
                fixed_real_projection_f32<19>(in, lo, hi, C, S, M);
                return true;
            default:
                return false;
        }
    }

    bool dispatch_bruun_odd_forward_f32(const Op& o, float* RESTRICT clo,
                                        float* RESTRICT chi, const float* RESTRICT lo,
                                        const float* RESTRICT hi, const double* RESTRICT C,
                                        const double* RESTRICT S, int Mp) const {
        switch (o.codelet) {
            case CODELET_3:
                fixed_bruun_odd_forward_child_f32<3>(clo, chi, lo, hi, C, S, Mp);
                return true;
            case CODELET_5:
                fixed_bruun_odd_forward_child_f32<5>(clo, chi, lo, hi, C, S, Mp);
                return true;
            case CODELET_7:
                fixed_bruun_odd_forward_child_f32<7>(clo, chi, lo, hi, C, S, Mp);
                return true;
            case CODELET_11:
                fixed_bruun_odd_forward_child_f32<11>(clo, chi, lo, hi, C, S, Mp);
                return true;
            case CODELET_13:
                fixed_bruun_odd_forward_child_f32<13>(clo, chi, lo, hi, C, S, Mp);
                return true;
            case CODELET_17:
                fixed_bruun_odd_forward_child_f32<17>(clo, chi, lo, hi, C, S, Mp);
                return true;
            case CODELET_19:
                fixed_bruun_odd_forward_child_f32<19>(clo, chi, lo, hi, C, S, Mp);
                return true;
            default:
                return false;
        }
    }

    bool dispatch_bruun_odd_adjoint_f32(const Op& o, float* RESTRICT lo,
                                        float* RESTRICT hi, const float* RESTRICT clo,
                                        const float* RESTRICT chi, const double* RESTRICT C,
                                        const double* RESTRICT S, int Mp) const {
        switch (o.codelet) {
            case CODELET_3:
                fixed_bruun_odd_adjoint_child_f32<3>(lo, hi, clo, chi, C, S, Mp);
                return true;
            case CODELET_5:
                fixed_bruun_odd_adjoint_child_f32<5>(lo, hi, clo, chi, C, S, Mp);
                return true;
            case CODELET_7:
                fixed_bruun_odd_adjoint_child_f32<7>(lo, hi, clo, chi, C, S, Mp);
                return true;
            case CODELET_11:
                fixed_bruun_odd_adjoint_child_f32<11>(lo, hi, clo, chi, C, S, Mp);
                return true;
            case CODELET_13:
                fixed_bruun_odd_adjoint_child_f32<13>(lo, hi, clo, chi, C, S, Mp);
                return true;
            case CODELET_17:
                fixed_bruun_odd_adjoint_child_f32<17>(lo, hi, clo, chi, C, S, Mp);
                return true;
            case CODELET_19:
                fixed_bruun_odd_adjoint_child_f32<19>(lo, hi, clo, chi, C, S, Mp);
                return true;
            default:
                return false;
        }
    }

    bool dispatch_minus_split_inverse_f32(const Op& o, float* RESTRICT A) const {
        if (o.codelet == CODELET_3) {
            const int M = o.M; float* r = A + o.in; const float* sum = A + o.sum;
            const double* C = &twp_[o.tw[0]]; const double* S = C + 3;
            const float* lo = A + o.child[0]; const float* hi = lo + M;
            fixed_minus_split_inverse_first_f32<3>(r, sum, lo, hi, C, S, M);
            return true;
        }
        if (o.codelet == CODELET_5) {
            const int M = o.M; float* r = A + o.in; const float* sum = A + o.sum;
            const double* C0 = &twp_[o.tw[0]]; const double* S0 = C0 + 5;
            const double* C1 = &twp_[o.tw[1]]; const double* S1 = C1 + 5;
            const float* lo0 = A + o.child[0]; const float* hi0 = lo0 + M;
            const float* lo1 = A + o.child[1]; const float* hi1 = lo1 + M;
            minus_split5_inverse_fused_f32(r, sum, lo0, hi0, C0, S0, lo1, hi1, C1, S1, M);
            return true;
        }
        const int p = o.p, M = o.M;
        float* r = A + o.in;
        const float* sum = A + o.sum;
        for (int j = 1; j <= (p - 1) / 2; ++j) {
            const double* C = &twp_[o.tw[j - 1]];
            const double* S = C + p;
            const float* lo = A + o.child[j - 1];
            const float* hi = lo + M;
            switch (o.codelet) {
                case CODELET_7:
                    if (j == 1) minus_split7_inverse_first_f32(r, sum, lo, hi, C, S, M);
                    else minus_split7_inverse_add_f32(r, lo, hi, C, S, M);
                    break;
                case CODELET_11:
                    if (j == 1) fixed_minus_split_inverse_first_f32<11>(r, sum, lo, hi, C, S, M);
                    else fixed_minus_split_inverse_add_f32<11>(r, lo, hi, C, S, M);
                    break;
                case CODELET_13:
                    if (j == 1) fixed_minus_split_inverse_first_f32<13>(r, sum, lo, hi, C, S, M);
                    else fixed_minus_split_inverse_add_f32<13>(r, lo, hi, C, S, M);
                    break;
                case CODELET_17:
                    if (j == 1) fixed_minus_split_inverse_first_f32<17>(r, sum, lo, hi, C, S, M);
                    else fixed_minus_split_inverse_add_f32<17>(r, lo, hi, C, S, M);
                    break;
                case CODELET_19:
                    if (j == 1) fixed_minus_split_inverse_first_f32<19>(r, sum, lo, hi, C, S, M);
                    else fixed_minus_split_inverse_add_f32<19>(r, lo, hi, C, S, M);
                    break;
                default:
                    return false;
            }
        }
        return o.codelet != CODELET_GENERIC;
    }

    void exec_fwd(const Op& o, double* A, complex_t* X) {
        if (o.type == MINUS_POW2) {
            int D = o.len; const double* in = A + o.in;
            if (D == 1) { place(X, 0, in[0], 0); }
            else if (D == 2) { place(X, 0, in[0] + in[1], 0); place(X, o.sigma, in[0] - in[1], 0); }
            else { RFFT* r = pow2_[o.pow2]; complex_t* tb = bins_buf_.data();
                r->forward(in, tb, work_buf_.data());
                for (int k = 0; k <= D / 2; ++k) place(X, o.sigma * k, tb[k].re, tb[k].im); }
        } else if (o.type == MINUS_SPLIT) {
            int p = o.p, M = o.M; const double* in = A + o.in; double* sum = A + o.sum;
            for (int j = 1; j <= (p - 1) / 2; ++j) { const double* C = &twp_[o.tw[j - 1]]; const double* S = C + p;
                double* lo = A + o.child[j - 1]; double* hi = lo + M;
                if (!dispatch_minus_split_forward(o, j, in, sum, lo, hi, C, S, M)) {
                    if (j == 1) { zero_block(sum, M); for (int i = 0; i < p; ++i) add_scaled(sum, in + i * M, 1.0, M); }
                    zero_block(lo, M); zero_block(hi, M);
                    for (int i = 0; i < p; ++i) { add_scaled(lo, in + i * M, C[i], M); add_scaled(hi, in + i * M, S[i], M); }
                }
                if (M == 1) place(X, fbin(mk(j, p)), lo[0], -hi[0]); }
        } else if (o.type == BRUUN_ODD) {
            int p = o.p, Mp = o.M / p; double* v = A + o.in; const double* lo = v; const double* hi = v + o.M;
            if (o.codelet == CODELET_3) {                 // factored real all-children
                bruun_odd3_forward_all(A + o.child[0], A + o.child[1], A + o.child[2], lo, hi, &twp_[o.tw[0]], Mp);
            } else if (o.codelet == CODELET_5) {
                bruun_odd5_forward_all(A + o.child[0], A + o.child[1], A + o.child[2], A + o.child[3], A + o.child[4],
                                       lo, hi, &twp_[o.tw[0]], Mp);
            } else if (o.codelet == CODELET_7) {
                bruun_odd7_forward_all(A+o.child[0],A+o.child[1],A+o.child[2],A+o.child[3],A+o.child[4],A+o.child[5],A+o.child[6],
                                       lo, hi, &twp_[o.tw[0]], Mp);
            } else
            for (int t = 0; t < p; ++t) { const double* C = &twp_[o.tw[t]]; const double* S = C + p;
                double* clo = A + o.child[t]; double* chi = clo + Mp;
                if (!dispatch_bruun_odd_forward(o, clo, chi, lo, hi, C, S, Mp)) {
                    zero_block(clo, Mp); zero_block(chi, Mp);
                    for (int i = 0; i < p; ++i) add_complex_projection(clo, chi, lo + i * Mp, hi + i * Mp, C[i], S[i], Mp);
                } }
        } else if (o.type == BRUUN_POW2) {
            int M = o.M, D = 2 * M; double* v = A + o.in; const double* C = &ctp_[o.ctab];
            int Lg = 0; for (int t = D; t > 1; t >>= 1) ++Lg;
            auto s_tw = [&](int m) { if (m == 1) return o.s1; int flip = (m > 1) ? 1 : 0; return C[m ^ flip]; };
            for (int jj = 0; jj <= Lg - 2; ++jj) { int s = D >> jj, q = s >> 2, nb = 1 << jj;
                for (int m = 0; m < nb; ++m) { int node = nb + m; norm_q_fwd(v + m * s, q, C[node], s_tw(node)); } }
            const int* P = &permp_[o.perm];
            for (int m = 0; m < M; ++m) place(X, P[m], v[2 * m], -v[2 * m + 1]);
        } else { const double* v = A + o.in; place(X, fbin(o.f), v[0], -v[1]); }
    }

    void exec_inv(const Op& o, double* A, const cx* Xf) {
        if (o.type == LEAF) { cx v = Xf[fbin(o.f)]; A[o.in] = v.re; A[o.in + 1] = -v.im; }
        else if (o.type == BRUUN_POW2) {
            int M = o.M, D = 2 * M; double* v = A + o.in; const double* C = &ctp_[o.ctab]; const int* P = &permp_[o.perm];
            for (int m = 0; m < M; ++m) { cx val = Xf[P[m]]; v[2 * m] = val.re; v[2 * m + 1] = -val.im; }
            int Lg = 0; for (int t = D; t > 1; t >>= 1) ++Lg;
            auto s_tw = [&](int m) { if (m == 1) return o.s1; int flip = (m > 1) ? 1 : 0; return C[m ^ flip]; };
            for (int jj = Lg - 2; jj >= 0; --jj) { int s = D >> jj, q = s >> 2, nb = 1 << jj;
                for (int m = 0; m < nb; ++m) { int node = nb + m; norm_q_inv(v + m * s, q, C[node], s_tw(node)); } }
        } else if (o.type == BRUUN_ODD) {
            int p = o.p, Mp = o.M / p; double* lo = A + o.in; double* hi = lo + o.M;
            if (o.codelet == CODELET_3) {                 // factored real all-children adjoint
                bruun_odd3_adjoint_all(lo, hi, A + o.child[0], A + o.child[1], A + o.child[2], &twp_[o.tw[0]], Mp);
                return;
            }
            if (o.codelet == CODELET_5) {
                bruun_odd5_adjoint_all(lo, hi, A + o.child[0], A + o.child[1], A + o.child[2], A + o.child[3], A + o.child[4],
                                       &twp_[o.tw[0]], Mp);
                return;
            }
            if (o.codelet == CODELET_7) {
                bruun_odd7_adjoint_all(lo, hi, A+o.child[0],A+o.child[1],A+o.child[2],A+o.child[3],A+o.child[4],A+o.child[5],A+o.child[6],
                                       &twp_[o.tw[0]], Mp);
                return;
            }
            zero_block(lo, o.M); zero_block(hi, o.M);
            for (int t = 0; t < p; ++t) { const double* C = &twp_[o.tw[t]]; const double* S = C + p;
                const double* clo = A + o.child[t]; const double* chi = clo + Mp;
                if (!dispatch_bruun_odd_adjoint(o, lo, hi, clo, chi, C, S, Mp)) {
                    const double scale = 1.0 / p;
                    for (int i = 0; i < p; ++i) add_complex_adjoint(lo + i * Mp, hi + i * Mp, clo, chi, C[i] * scale, S[i] * scale, Mp);
                } }
        } else if (o.type == MINUS_SPLIT) {
            if (!dispatch_minus_split_inverse(o, A)) {
                int p = o.p, M = o.M; double* r = A + o.in; const double* sum = A + o.sum;
                for (int j = 1; j <= (p - 1) / 2; ++j) { const double* C = &twp_[o.tw[j - 1]]; const double* S = C + p;
                    const double* lo = A + o.child[j - 1]; const double* hi = lo + M;
                    if (j == 1) { const double dc_scale = 1.0 / p; for (int i = 0; i < p; ++i) assign_scaled(r + i * M, sum, dc_scale, M); }
                    const double scale = 2.0 / p;
                    for (int i = 0; i < p; ++i) add_real_pair(r + i * M, lo, hi, C[i] * scale, S[i] * scale, M);
                }
            }
        } else { // MINUS_POW2
            int D = o.len; double* r = A + o.in;
            if (D == 1) { r[0] = Xf[0].re; }
            else if (D == 2) { r[0] = 0.5 * (Xf[0].re + Xf[o.sigma].re); r[1] = 0.5 * (Xf[0].re - Xf[o.sigma].re); }
            else { RFFT* rp = pow2_[o.pow2]; complex_t* tb = bins_buf_.data();
                for (int k = 0; k <= D / 2; ++k) { cx v = Xf[(o.sigma * k) % N]; tb[k].re = v.re; tb[k].im = v.im; }
                rp->inverse(tb, r); }
        }
    }

    void exec_fwd_f32(const Op& o, float* A, complex_f32_t* X) {
        if (o.type == MINUS_POW2) {
            int D = o.len; const float* in = A + o.in;
            if (D == 1) { place_f32(X, 0, in[0], 0.0f); }
            else if (D == 2) { place_f32(X, 0, in[0] + in[1], 0.0f); place_f32(X, o.sigma, in[0] - in[1], 0.0f); }
            else { RFFT* r = pow2_[o.pow2]; complex_f32_t* tb = bins_buf_f32_.data();
                r->forward_standard_f32(in, tb, work_buf_f32_.data(), nullptr);
                for (int k = 0; k <= D / 2; ++k) place_f32(X, o.sigma * k, tb[k].re, tb[k].im); }
        } else if (o.type == MINUS_SPLIT) {
            int p = o.p, M = o.M; const float* in = A + o.in; float* sum = A + o.sum;
            for (int j = 1; j <= (p - 1) / 2; ++j) { const double* Cd = &twp_[o.tw[j - 1]]; const double* Sd = Cd + p;
                float* lo = A + o.child[j - 1]; float* hi = lo + M;
                if (!dispatch_minus_split_forward_f32(o, j, in, sum, lo, hi, Cd, Sd, M)) {
                    if (j == 1) {
                        for (int m = 0; m < M; ++m) { float acc = 0.0f;
                            for (int i = 0; i < p; ++i) acc += in[i * M + m];
                            sum[m] = acc; }
                    }
                    for (int m = 0; m < M; ++m) { float l = 0.0f, h = 0.0f;
                        for (int i = 0; i < p; ++i) { const float v = in[i * M + m];
                            l += v * static_cast<float>(Cd[i]); h += v * static_cast<float>(Sd[i]); }
                        lo[m] = l; hi[m] = h; }
                }
                if (M == 1) place_f32(X, fbin(mk(j, p)), lo[0], -hi[0]); }
        } else if (o.type == BRUUN_ODD) {
            int p = o.p, Mp = o.M / p; float* v = A + o.in; const float* lo = v; const float* hi = v + o.M;
            for (int t = 0; t < p; ++t) { const double* Cd = &twp_[o.tw[t]]; const double* Sd = Cd + p;
                float* clo = A + o.child[t]; float* chi = clo + Mp;
                if (!dispatch_bruun_odd_forward_f32(o, clo, chi, lo, hi, Cd, Sd, Mp)) {
                    for (int m = 0; m < Mp; ++m) { float out_l = 0.0f, out_h = 0.0f;
                        for (int i = 0; i < p; ++i) { const float c = static_cast<float>(Cd[i]), s = static_cast<float>(Sd[i]);
                            const float a = lo[i * Mp + m], b = hi[i * Mp + m];
                            out_l += a * c - b * s; out_h += b * c + a * s; }
                        clo[m] = out_l; chi[m] = out_h; }
                } }
        } else if (o.type == BRUUN_POW2) {
            int M = o.M, D = 2 * M; float* v = A + o.in; const double* C = &ctp_[o.ctab];
            int Lg = 0; for (int t = D; t > 1; t >>= 1) ++Lg;
            auto s_tw = [&](int m) { if (m == 1) return static_cast<float>(o.s1); int flip = (m > 1) ? 1 : 0; return static_cast<float>(C[m ^ flip]); };
            for (int jj = 0; jj <= Lg - 2; ++jj) { int s = D >> jj, q = s >> 2, nb = 1 << jj;
                for (int m = 0; m < nb; ++m) { int node = nb + m; norm_q_fwd_f32(v + m * s, q, static_cast<float>(C[node]), s_tw(node)); } }
            const int* P = &permp_[o.perm];
            for (int m = 0; m < M; ++m) place_f32(X, P[m], v[2 * m], -v[2 * m + 1]);
        } else { const float* v = A + o.in; place_f32(X, fbin(o.f), v[0], -v[1]); }
    }

    void exec_inv_f32(const Op& o, float* A, const cxf* Xf) {
        if (o.type == LEAF) { cxf v = Xf[fbin(o.f)]; A[o.in] = v.re; A[o.in + 1] = -v.im; }
        else if (o.type == BRUUN_POW2) {
            int M = o.M, D = 2 * M; float* v = A + o.in; const double* C = &ctp_[o.ctab]; const int* P = &permp_[o.perm];
            for (int m = 0; m < M; ++m) { cxf val = Xf[P[m]]; v[2 * m] = val.re; v[2 * m + 1] = -val.im; }
            int Lg = 0; for (int t = D; t > 1; t >>= 1) ++Lg;
            auto s_tw = [&](int m) { if (m == 1) return static_cast<float>(o.s1); int flip = (m > 1) ? 1 : 0; return static_cast<float>(C[m ^ flip]); };
            for (int jj = Lg - 2; jj >= 0; --jj) { int s = D >> jj, q = s >> 2, nb = 1 << jj;
                for (int m = 0; m < nb; ++m) { int node = nb + m; norm_q_inv_f32(v + m * s, q, static_cast<float>(C[node]), s_tw(node)); } }
        } else if (o.type == BRUUN_ODD) {
            int p = o.p, Mp = o.M / p; float* lo = A + o.in; float* hi = lo + o.M;
            zero_block_f32(lo, o.M); zero_block_f32(hi, o.M);
            const float scale = 1.0f / static_cast<float>(p);
            for (int t = 0; t < p; ++t) { const double* Cd = &twp_[o.tw[t]]; const double* Sd = Cd + p;
                const float* clo = A + o.child[t]; const float* chi = clo + Mp;
                if (!dispatch_bruun_odd_adjoint_f32(o, lo, hi, clo, chi, Cd, Sd, Mp)) {
                    for (int i = 0; i < p; ++i) { const float c = static_cast<float>(Cd[i]) * scale, s = static_cast<float>(Sd[i]) * scale;
                        for (int m = 0; m < Mp; ++m) {
                            lo[i * Mp + m] += clo[m] * c + chi[m] * s;
                            hi[i * Mp + m] += chi[m] * c - clo[m] * s; } }
                } }
        } else if (o.type == MINUS_SPLIT) {
            if (!dispatch_minus_split_inverse_f32(o, A)) {
                int p = o.p, M = o.M; float* r = A + o.in; const float* sum = A + o.sum;
                for (int j = 1; j <= (p - 1) / 2; ++j) { const double* Cd = &twp_[o.tw[j - 1]]; const double* Sd = Cd + p;
                    const float* lo = A + o.child[j - 1]; const float* hi = lo + M;
                    if (j == 1) { const float dc_scale = 1.0f / static_cast<float>(p);
                        for (int i = 0; i < p; ++i) for (int m = 0; m < M; ++m) r[i * M + m] = sum[m] * dc_scale; }
                    const float scale = 2.0f / static_cast<float>(p);
                    for (int i = 0; i < p; ++i) { const float c = static_cast<float>(Cd[i]) * scale, s = static_cast<float>(Sd[i]) * scale;
                        for (int m = 0; m < M; ++m) r[i * M + m] += lo[m] * c + hi[m] * s; }
                }
            }
        } else {
            int D = o.len; float* r = A + o.in;
            if (D == 1) { r[0] = Xf[0].re; }
            else if (D == 2) { r[0] = 0.5f * (Xf[0].re + Xf[o.sigma].re); r[1] = 0.5f * (Xf[0].re - Xf[o.sigma].re); }
            else { RFFT* rp = pow2_[o.pow2]; complex_f32_t* tb = bins_buf_f32_.data();
                for (int k = 0; k <= D / 2; ++k) { cxf v = Xf[(o.sigma * k) % N]; tb[k].re = v.re; tb[k].im = v.im; }
                rp->inverse_f32(tb, r); }
        }
    }
};

} // namespace bruun
