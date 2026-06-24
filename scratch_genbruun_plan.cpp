// Flattened plan-time schedule for the generalized (arbitrary-N) Bruun real FFT.
// Builds the factor-tree schedule, twiddle tables, rooted C-tables, and leaf->bin
// permutations ONCE at plan time into pooled arrays; execution walks a flat op
// list over a pre-sized scratch arena -- no allocation, no transcendentals.
//
// Build: g++ -O3 -march=native -std=c++17 -Isrc scratch_genbruun_plan.cpp -lm -o /tmp/gp && /tmp/gp

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <complex>
#include <random>
#include <numeric>
#include <map>
#include "detail/bruun_kernel.hpp"

using cd = std::complex<double>;
using std::vector;

struct Frac { long long n, d; };
static Frac mk(long long n, long long d) {
    if (d < 0) { n = -n; d = -d; }
    long long a = n < 0 ? -n : n, g = std::gcd(a, d); if (!g) g = 1; return {n / g, d / g};
}
static Frac fhalf(Frac f) { return mk(f.n, f.d * 2); }
static Frac fcomp(Frac f) { return mk(f.d - f.n, 2 * f.d); }
static Frac fdivp(Frac f, long long t, long long p) { return mk(f.n + t * f.d, f.d * p); }
static long long fr(Frac f) { return ((f.n % f.d) + f.d) % f.d; }
// High-precision-modeled twiddle: reduce to integer turn fraction, fold to [0,1/4]
// quadrant for best cos/sin accuracy, single libm call.
static void trig(Frac f, double& c, double& s) {
    long long m = fr(f), d = f.d;                 // angle = 2*pi*m/d, m in [0,d)
    // fold by quadrant on the rational to keep the libm argument in [0, pi/4]
    int quad = (int)((4 * m) / d); long long rem4 = (4 * m) % d;  // quadrant + remainder
    double a = 2.0 * acos(-1.0) * (double)rem4 / (4.0 * (double)d); // in [0, pi/2)
    double ca = cos(a), sa = sin(a);
    switch (quad & 3) {
        case 0: c = ca;  s = sa;  break;
        case 1: c = -sa; s = ca;  break;
        case 2: c = -ca; s = -sa; break;
        default: c = sa; s = -ca; break;
    }
}
static double tcos(Frac f) { double c, s; trig(f, c, s); return c; }
static double tsin(Frac f) { double c, s; trig(f, c, s); return s; }
static long long fbin(Frac f, long long N) { long long k = llround((double)N * f.n / f.d); return ((k % N) + N) % N; }
static bool is_pow2(int n) { return n >= 1 && (n & (n - 1)) == 0; }
static int odd_prime(int n) { while (n % 2 == 0) n /= 2; if (n == 1) return 0;
    for (int d = 3; (long long)d * d <= n; d += 2) if (n % d == 0) return d; return n; }

// ---- the plan ----
struct GenBruunPlan {
    int N = 0;
    enum { MINUS_POW2, MINUS_SPLIT, BRUUN_POW2, BRUUN_ODD, LEAF } ;
    struct Op {
        int type, in, len, p, M; long long sigma; Frac f;
        int sum;                 // MINUS_SPLIT: sum child region
        vector<int> child;       // child regions (branches / odd children)
        vector<int> tw;          // per-child twiddle-table offset (into tw pool)
        int ctab = -1, perm = -1, pow2 = -1; // BRUUN_POW2 tables / MINUS_POW2 plan
    };
    vector<Op> ops;
    vector<double> twp;          // twiddle pool: [cos_0..cos_{n-1}, sin_0..sin_{n-1}]
    vector<double> ctp;          // rooted C-table pool
    vector<int>    permp;        // leaf->bin permutation pool
    vector<bruun::RFFT*> pow2;   // pow2 minus-node plans
    std::map<int, int> pow2idx;
    int arena = 0, hi = 0, work = 0;

    int alloc(int n) { int o = arena; arena += n; hi = std::max(hi, arena); return o; }
    void freeto(int o) { arena = o; }

    int add_tw(int n, Frac base_mul) {           // table cos/sin(i*base_mul) i=0..n-1
        int off = (int)twp.size(); twp.resize(off + 2 * n);
        for (int i = 0; i < n; ++i) { double c, s; trig(mk(base_mul.n * i, base_mul.d), c, s);
            twp[off + i] = c; twp[off + n + i] = s; }
        return off;
    }
    int get_pow2(int M) {
        auto it = pow2idx.find(M); if (it != pow2idx.end()) return it->second;
        auto* r = new bruun::RFFT(); r->init(M); work = std::max(work, M);
        int id = (int)pow2.size(); pow2.push_back(r); pow2idx[M] = id; return id;
    }

    // rooted C-table + leaf->bin perm for a pow2 Bruun node of half-size M, root f
    void build_rooted(int M, Frac root, int& ctab, int& perm) {
        int D = 2 * M, half = M, Lg = 0; for (int t = D; t > 1; t >>= 1) ++Lg;
        ctab = (int)ctp.size(); ctp.resize(ctab + std::max(1, half));
        double* C = &ctp[ctab];
        C[1 % std::max(1, half)] = 0; // guard
        if (half > 1) C[1] = tcos(fhalf(root));
        double S1 = tsin(fhalf(root));
        auto s_tw = [&](int m) { if (m == 1) return S1; int flip = (m > 1) ? 1 : 0; return C[m ^ flip]; };
        for (int m = 1; 2 * m < half; ++m) { double c = C[m], s = s_tw(m), ce = std::sqrt(0.5 * (1 + c));
            C[2 * m] = ce; if (2 * m + 1 < half) C[2 * m + 1] = s / (2 * ce); }
        perm = (int)permp.size(); permp.resize(perm + half);
        int depth = Lg - 1;
        for (int m = 0; m < half; ++m) { Frac f = root;
            for (int b = depth - 1; b >= 0; --b) f = ((m >> b) & 1) ? fcomp(f) : fhalf(f);
            permp[perm + m] = (int)fbin(f, N); }
    }

    void plan_bruun(int in, int M, Frac f) {        // input region = cascade seed (len 2M)
        if (M == 1) { Op o; o.type = LEAF; o.in = in; o.f = f; ops.push_back(o); return; }
        if (is_pow2(M)) { Op o; o.type = BRUUN_POW2; o.in = in; o.M = M; o.f = f;
            build_rooted(M, f, o.ctab, o.perm); ops.push_back(o); return; }
        int p = odd_prime(M), Mp = M / p;
        Op o; o.type = BRUUN_ODD; o.in = in; o.M = M; o.p = p; o.f = f;
        int base = arena;
        for (int t = 0; t < p; ++t) { o.child.push_back(alloc(2 * Mp));
            o.tw.push_back(add_tw(p, fdivp(f, t, p))); }
        ops.push_back(o);
        Op saved = o;
        for (int t = 0; t < p; ++t) plan_bruun(saved.child[t], Mp, fdivp(f, t, p));
        freeto(base);
    }

    void plan_minus(int in, int D, long long sigma) {
        if (is_pow2(D)) { Op o; o.type = MINUS_POW2; o.in = in; o.len = D; o.sigma = sigma;
            o.pow2 = (D >= 4) ? get_pow2(D) : -1; ops.push_back(o); return; }
        int p = odd_prime(D), M = D / p;
        Op o; o.type = MINUS_SPLIT; o.in = in; o.len = D; o.p = p; o.M = M; o.sigma = sigma;
        int base = arena;
        o.sum = alloc(M);
        for (int j = 1; j <= (p - 1) / 2; ++j) {
            o.child.push_back(alloc(2 * M));            // branch seed (lo|hi)
            o.tw.push_back(add_tw(p, mk(j, p)));        // cos/sin(i*j/p)
        }
        ops.push_back(o);
        Op saved = o;
        plan_minus(saved.sum, M, sigma * p);
        for (int j = 1; j <= (p - 1) / 2; ++j) {
            int br = saved.child[j - 1];
            if (M == 1) { Op l; l.type = LEAF; l.in = br; l.f = mk(j, p); ops.push_back(l); }
            else plan_bruun(br, M, mk(j, p));
        }
        freeto(base);
    }

    void init(int n) {
        N = n; arena = 0; hi = 0;
        int root = alloc(N);            // input residue lives at [0,N)
        plan_minus(root, N, 1);
    }

    // ---- execution: allocation-free ----
    void place(cd* X, long long k, cd v) const {
        k = ((k % N) + N) % N; X[k] = v; if (k != 0 && (N - k) != k) X[N - k] = std::conj(v);
    }
    void forward(const double* x, cd* X, double* A, double* w) const {  // A: arena(hi), w: work(work)
        for (int i = 0; i < N; ++i) X[i] = 0;
        for (int i = 0; i < N; ++i) A[i] = x[i];
        for (const Op& o : ops) {
            if (o.type == MINUS_POW2) {
                int D = o.len; const double* in = A + o.in;
                if (D == 1) { place(X, 0, cd(in[0], 0)); }
                else if (D == 2) { place(X, 0, cd(in[0] + in[1], 0)); place(X, o.sigma, cd(in[0] - in[1], 0)); }
                else {
                    bruun::RFFT* r = pow2[o.pow2];
                    static thread_local vector<cd> tmp; tmp.assign(D / 2 + 1, cd(0, 0));
                    static thread_local vector<double> wk; wk.assign(D, 0.0);
                    r->forward(in, reinterpret_cast<bruun::complex_t*>(tmp.data()), wk.data());
                    for (int k = 0; k <= D / 2; ++k) place(X, o.sigma * k, tmp[k]);
                }
            } else if (o.type == MINUS_SPLIT) {
                int p = o.p, M = o.M; const double* in = A + o.in;
                double* sum = A + o.sum;
                for (int m = 0; m < M; ++m) { double s = 0; for (int i = 0; i < p; ++i) s += in[i * M + m]; sum[m] = s; }
                for (int j = 1; j <= (p - 1) / 2; ++j) {
                    const double* C = &twp[o.tw[j - 1]]; const double* S = C + p;
                    double* lo = A + o.child[j - 1]; double* hi = lo + M;
                    for (int m = 0; m < M; ++m) { double a = 0, b = 0;
                        for (int i = 0; i < p; ++i) { double r = in[i * M + m]; a += r * C[i]; b += r * S[i]; }
                        lo[m] = a; hi[m] = b; }
                    if (M == 1) place(X, fbin(o.f.n ? o.f : mk(j, p), N), cd(lo[0], -hi[0]));
                }
            } else if (o.type == BRUUN_ODD) {
                int p = o.p, twoM = 2 * o.M, Mp = o.M / p; double* v = A + o.in;
                const double* lo = v; const double* hi = v + o.M;
                for (int t = 0; t < p; ++t) {
                    const double* C = &twp[o.tw[t]]; const double* S = C + p;
                    double* clo = A + o.child[t]; double* chi = clo + Mp;
                    for (int m = 0; m < Mp; ++m) { double a = 0, b = 0;
                        for (int i = 0; i < p; ++i) { double Av = lo[i * Mp + m], Bv = hi[i * Mp + m];
                            a += Av * C[i] - Bv * S[i]; b += Bv * C[i] + Av * S[i]; }
                        clo[m] = a; chi[m] = b; }
                }
                (void)twoM;
            } else if (o.type == BRUUN_POW2) {
                int M = o.M, D = 2 * M; double* v = A + o.in;
                const double* C = &ctp[o.ctab]; int Lg = 0; for (int t = D; t > 1; t >>= 1) ++Lg;
                double S1 = tsin(fhalf(o.f));
                auto s_tw = [&](int m) { if (m == 1) return S1; int flip = (m > 1) ? 1 : 0; return C[m ^ flip]; };
                for (int jj = 0; jj <= Lg - 2; ++jj) { int s = D >> jj, q = s >> 2, nb = 1 << jj;
                    for (int m = 0; m < nb; ++m) { int node = nb + m;
                        bruun::norm_q_fwd(v + m * s, q, C[node], s_tw(node)); } }
                const int* P = &permp[o.perm];
                for (int m = 0; m < M; ++m) place(X, P[m], cd(v[2 * m], -v[2 * m + 1]));
            } else { // LEAF (bruun M==1, cascade frame): lo[0]-i hi[0]
                const double* v = A + o.in;
                place(X, fbin(o.f, N), cd(v[0], -v[1]));
            }
        }
    }

    // ---- inverse: reverse op-walk, each op fills its INPUT region from outputs ----
    void inverse(const cd* Xhalf, double* x, double* A) const {
        vector<cd> Xf(N);
        for (int k = 0; k <= N / 2; ++k) Xf[k] = Xhalf[k];
        for (int k = 1; k < (N + 1) / 2; ++k) Xf[N - k] = std::conj(Xhalf[k]);
        auto gat = [&](Frac f) { return Xf[(int)fbin(f, N)]; };
        for (int oi = (int)ops.size() - 1; oi >= 0; --oi) {
            const Op& o = ops[oi];
            if (o.type == LEAF) {
                cd v = gat(o.f); A[o.in] = v.real(); A[o.in + 1] = -v.imag();
            } else if (o.type == BRUUN_POW2) {
                int M = o.M, D = 2 * M; double* v = A + o.in;
                const double* C = &ctp[o.ctab]; const int* P = &permp[o.perm];
                for (int m = 0; m < M; ++m) { cd val = Xf[P[m]]; v[2 * m] = val.real(); v[2 * m + 1] = -val.imag(); }
                int Lg = 0; for (int t = D; t > 1; t >>= 1) ++Lg;
                double S1 = tsin(fhalf(o.f));
                auto s_tw = [&](int m) { if (m == 1) return S1; int flip = (m > 1) ? 1 : 0; return C[m ^ flip]; };
                for (int jj = Lg - 2; jj >= 0; --jj) { int s = D >> jj, q = s >> 2, nb = 1 << jj;
                    for (int m = 0; m < nb; ++m) { int node = nb + m;
                        bruun::norm_q_inv(v + m * s, q, C[node], s_tw(node)); } }
            } else if (o.type == BRUUN_ODD) {
                int p = o.p, Mp = o.M / p; double* lo = A + o.in; double* hi = lo + o.M;
                for (int i = 0; i < o.M; ++i) { lo[i] = 0; hi[i] = 0; }
                for (int t = 0; t < p; ++t) {
                    const double* C = &twp[o.tw[t]]; const double* S = C + p;
                    const double* clo = A + o.child[t]; const double* chi = clo + Mp;
                    for (int i = 0; i < p; ++i) for (int m = 0; m < Mp; ++m) {
                        lo[i * Mp + m] += (clo[m] * C[i] + chi[m] * S[i]) / p;
                        hi[i * Mp + m] += (chi[m] * C[i] - clo[m] * S[i]) / p; }
                }
            } else if (o.type == MINUS_SPLIT) {
                int p = o.p, M = o.M; double* r = A + o.in; const double* sum = A + o.sum;
                for (int i = 0; i < p; ++i) for (int m = 0; m < M; ++m) r[i * M + m] = sum[m] / p;
                for (int j = 1; j <= (p - 1) / 2; ++j) {
                    const double* C = &twp[o.tw[j - 1]]; const double* S = C + p;
                    const double* lo = A + o.child[j - 1]; const double* hi = lo + M;
                    for (int i = 0; i < p; ++i) for (int m = 0; m < M; ++m)
                        r[i * M + m] += 2.0 * (lo[m] * C[i] + hi[m] * S[i]) / p;
                }
            } else { // MINUS_POW2
                int D = o.len; double* r = A + o.in;
                if (D == 1) { r[0] = Xf[0].real(); }
                else if (D == 2) { r[0] = 0.5 * (Xf[0].real() + Xf[o.sigma].real()); r[1] = 0.5 * (Xf[0].real() - Xf[o.sigma].real()); }
                else {
                    bruun::RFFT* rp = pow2[o.pow2];
                    static thread_local vector<cd> tmp; tmp.assign(D / 2 + 1, cd(0, 0));
                    for (int k = 0; k <= D / 2; ++k) tmp[k] = Xf[(o.sigma * k) % N];
                    rp->inverse(reinterpret_cast<bruun::complex_t*>(tmp.data()), r);
                }
            }
        }
        for (int i = 0; i < N; ++i) x[i] = A[i];   // root residue lives at arena offset 0
    }
};

int main(int argc, char** argv) {
    vector<int> sizes = {3,5,7,9,15,27,45,75,127,225,257,509,511,521,1021,1024,1920,2187,3000,6075,10125};
    if (argc > 1) { sizes.clear(); for (int i = 1; i < argc; ++i) sizes.push_back(atoi(argv[i])); }
    printf("%7s %14s %8s\n", "N", "max_rel_err", "status");
    for (int N : sizes) {
        std::mt19937_64 rng(N); std::normal_distribution<double> nd; vector<double> x(N);
        for (auto& v : x) v = nd(rng);
        GenBruunPlan plan; plan.init(N);
        vector<cd> X(N); vector<double> A(plan.hi + 4);
        plan.forward(x.data(), X.data(), A.data(), nullptr);
        const double TWOPI = 2.0 * acos(-1.0); double worst = 0, refm = 1;
        for (int k = 0; k <= N / 2; ++k) { double re = 0, im = 0, cr = 0, ci = 0;
            for (int n = 0; n < N; ++n) { long long kn = ((long long)k * n) % N; double a = -TWOPI * (double)kn / N;
                double tr = x[n] * cos(a) - cr, sr = re + tr; cr = (sr - re) - tr; re = sr;
                double ti = x[n] * sin(a) - ci, si = im + ti; ci = (si - im) - ti; im = si; }
            cd ref(re, im); refm = std::max(refm, std::abs(ref)); worst = std::max(worst, std::abs(X[k] - ref)); }
        // inverse roundtrip
        vector<double> xr(N); vector<double> A2(plan.hi + 4);
        plan.inverse(X.data(), xr.data(), A2.data());
        double rt = 0, xm = 1;
        for (int i = 0; i < N; ++i) { xm = std::max(xm, std::fabs(x[i])); rt = std::max(rt, std::fabs(xr[i] - x[i])); }
        double rel = worst / refm, rrt = rt / xm;
        printf("%7d  fwd=%.3e  roundtrip=%.3e  %s\n", N, rel, rrt,
               (rel < 1e-12 && rrt < 1e-11) ? "OK" : "FAIL");
    }
    return 0;
}
