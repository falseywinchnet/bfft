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

private:
    struct cx { double re, im; };
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
    struct Op { int type, in, len = 0, p = 0, M = 0; long long sigma = 0; Frac f{0, 1};
        int sum = 0; std::vector<int> child, tw; int ctab = -1, perm = -1, pow2 = -1; };

    int N;
    std::vector<Op> ops_;
    std::vector<double> twp_, ctp_;
    std::vector<int> permp_;
    std::vector<RFFT*> pow2_;
    std::map<int, int> pow2idx_;
    int top_ = 0, hi_ = 0, work_ = 0;
    heap_array<double> arena_, work_buf_;
    heap_array<complex_t> bins_buf_;
    heap_array<cx> xfull_;

    int alloc(int n) { int o = top_; top_ += n; hi_ = std::max(hi_, top_); return o; }
    void freeto(int o) { top_ = o; }
    int add_tw(int n, Frac base) { int off = (int)twp_.size(); twp_.resize(off + 2 * n);
        for (int i = 0; i < n; ++i) { double c, s; trig(mk(base.n * i, base.d), c, s);
            twp_[off + i] = c; twp_[off + n + i] = s; } return off; }
    int get_pow2(int M) { auto it = pow2idx_.find(M); if (it != pow2idx_.end()) return it->second;
        auto* r = new RFFT(); r->init(M); work_ = std::max(work_, M);
        int id = (int)pow2_.size(); pow2_.push_back(r); pow2idx_[M] = id; return id; }

    void build_rooted(int M, Frac root, int& ctab, int& perm) {
        int D = 2 * M, half = M, Lg = 0; for (int t = D; t > 1; t >>= 1) ++Lg;
        ctab = (int)ctp_.size(); ctp_.resize(ctab + std::max(1, half));
        double* C = &ctp_[ctab];
        if (half > 1) C[1] = tcos(fhalf(root));
        double S1 = tsin(fhalf(root));
        auto s_tw = [&](int m) { if (m == 1) return S1; int flip = (m > 1) ? 1 : 0; return C[m ^ flip]; };
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
            build_rooted(M, f, o.ctab, o.perm); ops_.push_back(o); return; }
        int p = odd_prime(M), Mp = M / p;
        Op o; o.type = BRUUN_ODD; o.in = in; o.M = M; o.p = p; o.f = f; int base = top_;
        for (int t = 0; t < p; ++t) { o.child.push_back(alloc(2 * Mp)); o.tw.push_back(add_tw(p, fdivp(f, t, p))); }
        ops_.push_back(o); Op saved = o;
        for (int t = 0; t < p; ++t) plan_bruun(saved.child[t], Mp, fdivp(f, t, p));
        freeto(base);
    }

    void plan_minus(int in, int D, long long sigma) {
        if (is_pow2(D)) { Op o; o.type = MINUS_POW2; o.in = in; o.len = D; o.sigma = sigma;
            o.pow2 = (D >= 4) ? get_pow2(D) : -1; ops_.push_back(o); return; }
        int p = odd_prime(D), M = D / p;
        Op o; o.type = MINUS_SPLIT; o.in = in; o.len = D; o.p = p; o.M = M; o.sigma = sigma; int base = top_;
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
        work_buf_.resize((size_t)std::max(1, work_));
        bins_buf_.resize((size_t)std::max(1, work_ / 2 + 1));
        xfull_.resize((size_t)N);
    }

    void place(complex_t* X, long long k, double re, double im) {
        k = ((k % N) + N) % N;
        if (k > N / 2) { k = N - k; im = -im; }   // fold to rfft half
        X[k].re = re; X[k].im = im;
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
            for (int m = 0; m < M; ++m) { double s = 0; for (int i = 0; i < p; ++i) s += in[i * M + m]; sum[m] = s; }
            for (int j = 1; j <= (p - 1) / 2; ++j) { const double* C = &twp_[o.tw[j - 1]]; const double* S = C + p;
                double* lo = A + o.child[j - 1]; double* hi = lo + M;
                for (int m = 0; m < M; ++m) { double a = 0, b = 0;
                    for (int i = 0; i < p; ++i) { double r = in[i * M + m]; a += r * C[i]; b += r * S[i]; }
                    lo[m] = a; hi[m] = b; }
                if (M == 1) place(X, fbin(mk(j, p)), lo[0], -hi[0]); }
        } else if (o.type == BRUUN_ODD) {
            int p = o.p, Mp = o.M / p; double* v = A + o.in; const double* lo = v; const double* hi = v + o.M;
            for (int t = 0; t < p; ++t) { const double* C = &twp_[o.tw[t]]; const double* S = C + p;
                double* clo = A + o.child[t]; double* chi = clo + Mp;
                for (int m = 0; m < Mp; ++m) { double a = 0, b = 0;
                    for (int i = 0; i < p; ++i) { double Av = lo[i * Mp + m], Bv = hi[i * Mp + m];
                        a += Av * C[i] - Bv * S[i]; b += Bv * C[i] + Av * S[i]; }
                    clo[m] = a; chi[m] = b; } }
        } else if (o.type == BRUUN_POW2) {
            int M = o.M, D = 2 * M; double* v = A + o.in; const double* C = &ctp_[o.ctab];
            int Lg = 0; for (int t = D; t > 1; t >>= 1) ++Lg; double S1 = tsin(fhalf(o.f));
            auto s_tw = [&](int m) { if (m == 1) return S1; int flip = (m > 1) ? 1 : 0; return C[m ^ flip]; };
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
            int Lg = 0; for (int t = D; t > 1; t >>= 1) ++Lg; double S1 = tsin(fhalf(o.f));
            auto s_tw = [&](int m) { if (m == 1) return S1; int flip = (m > 1) ? 1 : 0; return C[m ^ flip]; };
            for (int jj = Lg - 2; jj >= 0; --jj) { int s = D >> jj, q = s >> 2, nb = 1 << jj;
                for (int m = 0; m < nb; ++m) { int node = nb + m; norm_q_inv(v + m * s, q, C[node], s_tw(node)); } }
        } else if (o.type == BRUUN_ODD) {
            int p = o.p, Mp = o.M / p; double* lo = A + o.in; double* hi = lo + o.M;
            for (int i = 0; i < o.M; ++i) { lo[i] = 0; hi[i] = 0; }
            for (int t = 0; t < p; ++t) { const double* C = &twp_[o.tw[t]]; const double* S = C + p;
                const double* clo = A + o.child[t]; const double* chi = clo + Mp;
                for (int i = 0; i < p; ++i) for (int m = 0; m < Mp; ++m) {
                    lo[i * Mp + m] += (clo[m] * C[i] + chi[m] * S[i]) / p;
                    hi[i * Mp + m] += (chi[m] * C[i] - clo[m] * S[i]) / p; } }
        } else if (o.type == MINUS_SPLIT) {
            int p = o.p, M = o.M; double* r = A + o.in; const double* sum = A + o.sum;
            for (int i = 0; i < p; ++i) for (int m = 0; m < M; ++m) r[i * M + m] = sum[m] / p;
            for (int j = 1; j <= (p - 1) / 2; ++j) { const double* C = &twp_[o.tw[j - 1]]; const double* S = C + p;
                const double* lo = A + o.child[j - 1]; const double* hi = lo + M;
                for (int i = 0; i < p; ++i) for (int m = 0; m < M; ++m)
                    r[i * M + m] += 2.0 * (lo[m] * C[i] + hi[m] * S[i]) / p; }
        } else { // MINUS_POW2
            int D = o.len; double* r = A + o.in;
            if (D == 1) { r[0] = Xf[0].re; }
            else if (D == 2) { r[0] = 0.5 * (Xf[0].re + Xf[o.sigma].re); r[1] = 0.5 * (Xf[0].re - Xf[o.sigma].re); }
            else { RFFT* rp = pow2_[o.pow2]; complex_t* tb = bins_buf_.data();
                for (int k = 0; k <= D / 2; ++k) { cx v = Xf[(o.sigma * k) % N]; tb[k].re = v.re; tb[k].im = v.im; }
                rp->inverse(tb, r); }
        }
    }
};

} // namespace bruun
