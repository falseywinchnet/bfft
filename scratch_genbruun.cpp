// Scalar C++ reference for the generalized (arbitrary-N) Bruun real FFT.
// Mirrors scratch_genbruun_exact.py. Not SIMD/fused yet -- correctness first.
//
// Build:
//   g++ -O2 -std=c++17 -Isrc scratch_genbruun.cpp -lm -o /tmp/genbruun && /tmp/genbruun
//
// Reuses bruun::RFFT (the existing normalized pow2 kernel) for pow2 minus-nodes;
// implements the rooted normalized cascade and the condition-1 odd splits anew.

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <complex>
#include <random>
#include <numeric>

#include "detail/bruun_kernel.hpp"

using cd = std::complex<double>;
using std::vector;

// ---- exact rational phase (turns), angle = 2*pi*f ----
struct Frac { long long n, d; };
static Frac mk(long long n, long long d) {
    if (d < 0) { n = -n; d = -d; }
    long long a = n < 0 ? -n : n;
    long long g = std::gcd(a, d); if (g == 0) g = 1;
    return {n / g, d / g};
}
static Frac fhalf(Frac f)      { return mk(f.n, f.d * 2); }          // f/2
static Frac fcomp_half(Frac f) { return mk(f.d - f.n, 2 * f.d); }    // 1/2 - f/2
static Frac fdivp(Frac f, long long t, long long p) {                // (f + t)/p
    return mk(f.n + t * f.d, f.d * p);
}
static long long freduce(Frac f) { return ((f.n % f.d) + f.d) % f.d; }
static double tcos(Frac f) {
    long double a = 2.0L * acosl(-1.0L) * (long double)freduce(f) / (long double)f.d;
    return (double)cosl(a);
}
static double tsin(Frac f) {
    long double a = 2.0L * acosl(-1.0L) * (long double)freduce(f) / (long double)f.d;
    return (double)sinl(a);
}
static long long fbin(Frac f, long long N) {
    long long k = (long long)llroundl((long double)N * (long double)f.n / (long double)f.d);
    return ((k % N) + N) % N;
}

static bool is_pow2(int n) { return n >= 1 && (n & (n - 1)) == 0; }
static int odd_prime(int n) {
    while (n % 2 == 0) n /= 2;
    if (n == 1) return 0;
    for (int d = 3; (long long)d * d <= n; d += 2) if (n % d == 0) return d;
    return n;
}

struct GenBruun {
    int N;
    vector<cd> X;
    bruun::RFFT pow2;   // reused for pow2 minus-nodes

    void place(long long k, cd v) {
        k = ((k % N) + N) % N;
        X[k] = v;
        if (k != 0 && (N - k) != k) X[N - k] = std::conj(v);
    }

    // Rooted normalized cascade using the SIMD butterfly norm_q_fwd, in place.
    // seed has length D=2M (M leaves). Reuses the kernel's angle-table recurrence
    // seeded at the root angle; root node's sine twiddle is handled explicitly.
    void nb_subtree(vector<double> v, Frac root) {
        int D = (int)v.size();
        if (D == 2) { place(fbin(root, N), cd(v[0], -v[1])); return; }
        int half = D / 2;
        int Lg = 0; for (int t = D; t > 1; t >>= 1) ++Lg;     // log2(D)
        vector<double> C(half, 0.0);
        C[1] = tcos(fhalf(root));
        double S1 = tsin(fhalf(root));
        auto s_tw = [&](int m) { if (m == 1) return S1; int flip = (m > 1) ? 1 : 0; return C[m ^ flip]; };
        for (int m = 1; 2 * m < half; ++m) {
            double c = C[m], s = s_tw(m), ce = std::sqrt(0.5 * (1.0 + c));
            C[2 * m] = ce;
            if (2 * m + 1 < half) C[2 * m + 1] = s / (2.0 * ce);
        }
        for (int jj = 0; jj <= Lg - 2; ++jj) {
            int s = D >> jj, q = s >> 2, nb = 1 << jj;
            for (int m = 0; m < nb; ++m) {
                int node = nb + m;
                bruun::norm_q_fwd(v.data() + m * s, q, C[node], s_tw(node));
            }
        }
        int depth = Lg - 1;                                   // bits per leaf slot
        for (int m = 0; m < half; ++m) {
            Frac f = root;
            for (int b = depth - 1; b >= 0; --b)
                f = ((m >> b) & 1) ? fcomp_half(f) : fhalf(f);
            place(fbin(f, N), cd(v[2 * m], -v[2 * m + 1]));
        }
    }

    // Bruun node already in the condition-1 cascade frame (value lo - i*hi).
    void reduce_bruun_cascade(const vector<double>& lo, const vector<double>& hi, int M, Frac f) {
        if (M == 1) { place(fbin(f, N), cd(lo[0], -hi[0])); return; }
        if (is_pow2(M)) {
            vector<double> seed(2 * M);
            for (int i = 0; i < M; ++i) { seed[i] = lo[i]; seed[M + i] = hi[i]; }
            nb_subtree(seed, f);
            return;
        }
        int p = odd_prime(M), Mp = M / p;
        for (int t = 0; t < p; ++t) {
            Frac phi = fdivp(f, t, p);
            vector<double> clo(Mp, 0.0), chi(Mp, 0.0);
            for (int i = 0; i < p; ++i) {
                double cc = tcos(fdivp(f, (long long)t + (long long)i * p, p)); // cos(i*phi) via exact i*phi
                // i*phi = i*(f+t)/p ; build as Frac directly:
                Frac iphi = mk((f.n + t * f.d) * i, f.d * p);
                cc = tcos(iphi); double ss = tsin(iphi);
                for (int m = 0; m < Mp; ++m) {
                    double A = lo[i * Mp + m], B = hi[i * Mp + m];
                    clo[m] += A * cc - B * ss;
                    chi[m] += B * cc + A * ss;
                }
            }
            reduce_bruun_cascade(clo, chi, Mp, phi);
        }
    }

    void reduce_minus(const vector<double>& r, int D, long long sigma) {
        if (is_pow2(D)) {
            if (D == 1) { place(sigma * 0, cd(r[0], 0.0)); return; }
            pow2.init(D);
            vector<cd> Xloc(D / 2 + 1);
            vector<double> work(D);
            pow2.forward(r.data(), reinterpret_cast<bruun::complex_t*>(Xloc.data()), work.data());
            for (int k = 0; k <= D / 2; ++k) place(sigma * k, Xloc[k]);
            return;
        }
        int p = odd_prime(D), M = D / p;
        vector<vector<double>> R(p, vector<double>(M));
        for (int i = 0; i < p; ++i)
            for (int m = 0; m < M; ++m) R[i][m] = r[i * M + m];
        vector<double> s(M, 0.0);
        for (int i = 0; i < p; ++i) for (int m = 0; m < M; ++m) s[m] += R[i][m];
        reduce_minus(s, M, sigma * p);
        for (int j = 1; j <= (p - 1) / 2; ++j) {
            vector<double> seed_lo(M, 0.0), seed_hi(M, 0.0);
            for (int i = 0; i < p; ++i) {
                Frac ij = mk((long long)i * j, p);
                double cc = tcos(ij), ss = tsin(ij);
                for (int m = 0; m < M; ++m) { seed_lo[m] += R[i][m] * cc; seed_hi[m] += R[i][m] * ss; }
            }
            Frac g = mk(j, p);
            if (M == 1) place(fbin(g, N), cd(seed_lo[0], -seed_hi[0]));
            else reduce_bruun_cascade(seed_lo, seed_hi, M, g);
        }
    }

    void run(const vector<double>& x) {
        N = (int)x.size();
        X.assign(N, cd(0, 0));
        reduce_minus(x, N, 1);
    }
};

int main(int argc, char** argv) {
    vector<int> sizes = {3,5,7,9,15,27,45,75,127,225,257,509,511,521,1021,1024,1920,2187,3000,6075,10125};
    if (argc > 1) { sizes.clear(); for (int i = 1; i < argc; ++i) sizes.push_back(atoi(argv[i])); }
    printf("%7s %14s %12s\n", "N", "max_rel_err", "status");
    for (int N : sizes) {
        std::mt19937_64 rng(N);
        std::normal_distribution<double> nd;
        vector<double> x(N);
        for (auto& v : x) v = nd(rng);
        GenBruun g; g.run(x);
        // direct DFT reference: integer-reduced phase (accurate cos/sin) + Kahan sum.
        const double TWOPI = 2.0 * acos(-1.0);
        double worst = 0, ref_max = 1;
        for (int k = 0; k <= N / 2; ++k) {
            double re = 0, im = 0, cre = 0, cim = 0;   // Kahan compensation
            for (int n = 0; n < N; ++n) {
                long long kn = ((long long)k * n) % N;
                double a = -TWOPI * (double)kn / (double)N;
                double tr = x[n] * cos(a) - cre, sr = re + tr; cre = (sr - re) - tr; re = sr;
                double ti = x[n] * sin(a) - cim, si = im + ti; cim = (si - im) - ti; im = si;
            }
            cd ref(re, im);
            ref_max = std::max(ref_max, std::abs(ref));
            worst = std::max(worst, std::abs(g.X[k] - ref));
        }
        double rel = worst / ref_max;
        printf("%7d %14.3e %12s\n", N, rel, rel < 1e-12 ? "OK" : "FAIL");
    }
    return 0;
}
