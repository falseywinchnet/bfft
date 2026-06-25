// Correctness test for the arbitrary-N (generalized Bruun) real FFT path.
// Covers odd, prime, prime-power, and odd-composite sizes plus a couple of
// power-of-two sizes (which take the native RFFT fast path) for parity.

#include <bfft/bfft.hpp>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

constexpr double pi = 3.141592653589793238462643383279502884;

// Reference rfft with integer-reduced phase (accurate cos/sin) and Kahan sums,
// so the reference itself is FFT-grade even for the high bins of large N.
std::vector<bfft::complex> ref_rfft(const std::vector<double>& x) {
    const long long n = (long long)x.size();
    std::vector<bfft::complex> out(n / 2 + 1);
    for (long long k = 0; k <= n / 2; ++k) {
        double re = 0, im = 0, cre = 0, cim = 0;
        for (long long t = 0; t < n; ++t) {
            const long long kt = (k * t) % n;
            const double a = -2.0 * pi * (double)kt / (double)n;
            const double tr = x[t] * std::cos(a) - cre, sr = re + tr; cre = (sr - re) - tr; re = sr;
            const double ti = x[t] * std::sin(a) - cim, si = im + ti; cim = (si - im) - ti; im = si;
        }
        out[k].re = re; out[k].im = im;
    }
    return out;
}

bool check(std::size_t n) {
    std::mt19937_64 rng(n);
    std::normal_distribution<double> nd;
    std::vector<double> x(n);
    for (auto& v : x) v = nd(rng);

    bfft::plan plan(n);
    const std::vector<bfft::complex> got = plan.forward(x);
    const std::vector<bfft::complex> ref = ref_rfft(x);

    double scale = 1.0;
    for (const auto& c : ref) scale = std::max(scale, std::hypot(c.re, c.im));
    double fwd_err = 0.0;
    for (std::size_t k = 0; k < ref.size(); ++k)
        fwd_err = std::max(fwd_err, std::hypot(got[k].re - ref[k].re, got[k].im - ref[k].im));
    fwd_err /= scale;

    std::vector<double> rt(n);
    plan.inverse(got.data(), rt.data());
    double xmax = 1.0, rt_err = 0.0;
    for (std::size_t i = 0; i < n; ++i) { xmax = std::max(xmax, std::fabs(x[i])); rt_err = std::max(rt_err, std::fabs(rt[i] - x[i])); }
    rt_err /= xmax;

    const double tol = 1e-11;
    if (fwd_err > tol || rt_err > tol) {
        std::fprintf(stderr, "n=%zu forward=%.3e roundtrip=%.3e exceeds %.3e\n", n, fwd_err, rt_err, tol);
        return false;
    }
    return true;
}

}  // namespace

int main() {
    const std::size_t sizes[] = {
        2, 3, 5, 6, 7, 9, 10, 12, 15, 21, 25, 27, 35, 45, 49, 63, 75, 81, 121, 125,
        127, 169, 225, 243, 257, 289, 343, 361, 405, 509, 511, 521, 625, 729,
        1021, 1331, 1920, 2187, 2197, 3000, 6075, 10125,
        8, 16, 1024,  // pow2 fast-path parity
    };
    for (std::size_t n : sizes) {
        if (!check(n)) {
            std::fprintf(stderr, "genbruun correctness FAILED at n=%zu\n", n);
            return 1;
        }
    }
    std::printf("genbruun correctness ok backend=%s\n", bfft_backend_name());
    return 0;
}
