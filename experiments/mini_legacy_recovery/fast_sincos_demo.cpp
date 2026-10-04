#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884
#endif

namespace {

constexpr double pi = 3.141592653589793238462643383279502884;
constexpr double tau = 2.0 * pi;
constexpr double target_neg_144_dbc = 6.309573444801929e-8; // 10^(-144/20)

struct acc_stats {
    double mse_vec = 0.0;
    double rms_vec = 0.0;
    double max_vec = 0.0;
    double max_sin_abs = 0.0;
    double max_cos_abs = 0.0;
};

struct bench_stats {
    double seconds = 0.0;
    double mphases = 0.0;
    double sink = 0.0;
};

std::size_t parse_size(const char* s, std::size_t fallback) {
    if (!s) return fallback;
    char* end = nullptr;
    unsigned long long v = std::strtoull(s, &end, 10);
    if (end == s || *end != '\0' || v == 0) {
        throw std::runtime_error("bad sample count");
    }
    return static_cast<std::size_t>(v);
}

double error_to_dbc(double e) {
    if (e == 0.0) return -INFINITY;
    return 20.0 * std::log10(e);
}

std::vector<double> make_phases(std::size_t n) {
    std::vector<double> phase(n);
    std::mt19937_64 rng(20260626);
    std::uniform_real_distribution<double> dist(0.0, tau);

    for (std::size_t i = 0; i < n; ++i) {
        phase[i] = dist(rng);
    }

    // Force boundary/pathological regions too.
    if (n >= 16) {
        phase[0] = 0.0;
        phase[1] = std::nextafter(0.0, 1.0);
        phase[2] = pi / 2.0;
        phase[3] = pi;
        phase[4] = 3.0 * pi / 2.0;
        phase[5] = std::nextafter(tau, 0.0);
        phase[6] = tau * 0.125;
        phase[7] = tau * 0.250;
        phase[8] = tau * 0.375;
        phase[9] = tau * 0.500;
        phase[10] = tau * 0.625;
        phase[11] = tau * 0.750;
        phase[12] = tau * 0.875;
    }

    return phase;
}

static inline void std_sincos_pair(double x, double* s, double* c) {
    *s = std::sin(x);
    *c = std::cos(x);
}

#ifndef HAVE_BUILTIN_SINCOS
#  define HAVE_BUILTIN_SINCOS 0
#endif

#if defined(__has_builtin)
#  if __has_builtin(__builtin_sincos)
#    undef HAVE_BUILTIN_SINCOS
#    define HAVE_BUILTIN_SINCOS 1
#  endif
#endif

#if defined(__GNUC__) && !defined(__clang__)
#  undef HAVE_BUILTIN_SINCOS
#  define HAVE_BUILTIN_SINCOS 1
#endif

#if HAVE_BUILTIN_SINCOS
static inline void builtin_sincos_pair(double x, double* s, double* c) {
    __builtin_sincos(x, s, c);
}
#endif

template<int K>
struct sincos_table {
    static_assert((K & (K - 1)) == 0, "K must be a power of two");

    std::array<double, K> s{};
    std::array<double, K> c{};

    sincos_table() {
        for (int i = 0; i < K; ++i) {
            const double a = tau * static_cast<double>(i) / static_cast<double>(K);
            s[static_cast<std::size_t>(i)] = std::sin(a);
            c[static_cast<std::size_t>(i)] = std::cos(a);
        }
    }

    static const sincos_table& get() {
        static const sincos_table t;
        return t;
    }
};

// phase must be in [0, 2*pi). That matches bfft mag/phase inverse storage.
template<int K, int POLY>
static inline void fast_table_sincos(double x, double* s_out, double* c_out) {
    static_assert(POLY == 3 || POLY == 5 || POLY == 7, "POLY must be 3, 5, or 7");

    constexpr double inv_step = static_cast<double>(K) / tau;
    constexpr double step = tau / static_cast<double>(K);

    // Nearest table ray. Since x >= 0, int truncation equals floor.
    const double qd = x * inv_step + 0.5;
    const auto qi = static_cast<std::int64_t>(qd);
    const auto idx = static_cast<std::size_t>(qi) & static_cast<std::size_t>(K - 1);

    const double r = x - static_cast<double>(qi) * step;
    const double r2 = r * r;

    double sr;
    double cr;

    if constexpr (POLY == 3) {
        // sin r ~= r - r^3/6
        // cos r ~= 1 - r^2/2
        sr = r * (1.0 - r2 * (1.0 / 6.0));
        cr = 1.0 - r2 * 0.5;
    } else if constexpr (POLY == 5) {
        // sin r ~= r - r^3/6 + r^5/120
        // cos r ~= 1 - r^2/2 + r^4/24
        sr = r * (1.0 + r2 * (-1.0 / 6.0 + r2 * (1.0 / 120.0)));
        cr = 1.0 + r2 * (-0.5 + r2 * (1.0 / 24.0));
    } else {
        // sin r ~= r - r^3/6 + r^5/120 - r^7/5040
        // cos r ~= 1 - r^2/2 + r^4/24 - r^6/720
        sr = r * (1.0 + r2 * (-1.0 / 6.0 + r2 * (1.0 / 120.0 - r2 * (1.0 / 5040.0))));
        cr = 1.0 + r2 * (-0.5 + r2 * (1.0 / 24.0 - r2 * (1.0 / 720.0)));
    }

    const auto& table = sincos_table<K>::get();
    const double s0 = table.s[idx];
    const double c0 = table.c[idx];

    // Rotate table ray by residual r.
    *c_out = c0 * cr - s0 * sr;
    *s_out = s0 * cr + c0 * sr;
}

template<class Fn>
acc_stats accuracy(const std::vector<double>& phases, Fn&& fn) {
    long double sum_vec2 = 0.0L;
    double max_vec = 0.0;
    double max_sin_abs = 0.0;
    double max_cos_abs = 0.0;

    for (double x : phases) {
        double sr;
        double cr;
        double sf;
        double cf;

        std_sincos_pair(x, &sr, &cr);
        fn(x, &sf, &cf);

        const double ds = sf - sr;
        const double dc = cf - cr;
        const double vec = std::sqrt(ds * ds + dc * dc);

        sum_vec2 += static_cast<long double>(vec) * vec;
        max_vec = std::max(max_vec, vec);
        max_sin_abs = std::max(max_sin_abs, std::abs(ds));
        max_cos_abs = std::max(max_cos_abs, std::abs(dc));
    }

    acc_stats out;
    out.mse_vec = static_cast<double>(sum_vec2 / static_cast<long double>(phases.size()));
    out.rms_vec = std::sqrt(out.mse_vec);
    out.max_vec = max_vec;
    out.max_sin_abs = max_sin_abs;
    out.max_cos_abs = max_cos_abs;
    return out;
}

template<class Fn>
bench_stats benchmark(const std::vector<double>& phases, Fn&& fn) {
    double sink = 0.0;

    const auto t0 = std::chrono::steady_clock::now();
    for (double x : phases) {
        double s;
        double c;
        fn(x, &s, &c);
        sink += 0.125 * s + 0.875 * c;
    }
    const auto t1 = std::chrono::steady_clock::now();

    bench_stats out;
    out.seconds = std::chrono::duration<double>(t1 - t0).count();
    out.mphases = static_cast<double>(phases.size()) / out.seconds / 1.0e6;
    out.sink = sink;
    return out;
}

void print_accuracy(const char* name, const acc_stats& a) {
    const bool pass = a.max_vec <= target_neg_144_dbc;
    std::printf("%-18s max_vec %.17g  dBc %.3f  rms %.17g  max_sin %.17g  max_cos %.17g  %s\n",
                name,
                a.max_vec,
                error_to_dbc(a.max_vec),
                a.rms_vec,
                a.max_sin_abs,
                a.max_cos_abs,
                pass ? "PASS(-144dBc)" : "FAIL(-144dBc)");
}

void print_bench(const char* name, const bench_stats& b) {
    std::printf("%-18s time %.6f s  %.3f Mphase/s  sink %.17g\n",
                name,
                b.seconds,
                b.mphases,
                b.sink);
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::size_t n = static_cast<std::size_t>(1) << 22;
        if (argc > 1) n = parse_size(argv[1], n);

        const auto phases = make_phases(n);

        // Warm tables before timing.
        {
            double s, c;
            fast_table_sincos<64, 5>(0.123, &s, &c);
            fast_table_sincos<128, 3>(0.123, &s, &c);
            fast_table_sincos<128, 5>(0.123, &s, &c);
            fast_table_sincos<256, 3>(0.123, &s, &c);
        }

        std::printf("fast sincos demo: phase -> sin/cos for polar inverse\n");
        std::printf("samples=%zu target(-144 dBc)=%.17g unit-vector error\n", n, target_neg_144_dbc);
        std::printf("\naccuracy vs std::sin/std::cos reference:\n");

        print_accuracy("table64_poly5", accuracy(phases, [](double x, double* s, double* c) {
            fast_table_sincos<64, 5>(x, s, c);
        }));
        print_accuracy("table128_poly3", accuracy(phases, [](double x, double* s, double* c) {
            fast_table_sincos<128, 3>(x, s, c);
        }));
        print_accuracy("table128_poly5", accuracy(phases, [](double x, double* s, double* c) {
            fast_table_sincos<128, 5>(x, s, c);
        }));
        print_accuracy("table256_poly3", accuracy(phases, [](double x, double* s, double* c) {
            fast_table_sincos<256, 3>(x, s, c);
        }));

        std::printf("\nspeed:\n");
        print_bench("std sin+cos", benchmark(phases, [](double x, double* s, double* c) {
            std_sincos_pair(x, s, c);
        }));

#if HAVE_BUILTIN_SINCOS
        print_bench("builtin_sincos", benchmark(phases, [](double x, double* s, double* c) {
            builtin_sincos_pair(x, s, c);
        }));
#else
        std::printf("%-18s not available on this compiler\n", "builtin_sincos");
#endif

        print_bench("table64_poly5", benchmark(phases, [](double x, double* s, double* c) {
            fast_table_sincos<64, 5>(x, s, c);
        }));
        print_bench("table128_poly3", benchmark(phases, [](double x, double* s, double* c) {
            fast_table_sincos<128, 3>(x, s, c);
        }));
        print_bench("table128_poly5", benchmark(phases, [](double x, double* s, double* c) {
            fast_table_sincos<128, 5>(x, s, c);
        }));
        print_bench("table256_poly3", benchmark(phases, [](double x, double* s, double* c) {
            fast_table_sincos<256, 3>(x, s, c);
        }));

        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        std::fprintf(stderr, "usage: fast_sincos_demo [samples=4194304]\n");
        return 2;
    }
}
