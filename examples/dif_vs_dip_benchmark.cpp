#include "../src/detail/bruun_dif_kernel.hpp"
#include "../src/detail/bruun_dip_kernel.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;

constexpr double pi = 3.141592653589793238462643383279502884;

struct timing {
    double best_ns = 0.0;
    double sink = 0.0;
};

struct result {
    std::size_t n = 0;
    int iters = 0;
    double dif_forward_ns = 0.0;
    double dip_forward_ns = 0.0;
    double dif_inverse_ns = 0.0;
    double dip_inverse_ns = 0.0;
    double dif_roundtrip_ns = 0.0;
    double dip_roundtrip_ns = 0.0;
    double forward_maxerr = 0.0;
    double dif_roundtrip_maxerr = 0.0;
    double dip_roundtrip_maxerr = 0.0;
    double sink = 0.0;
};

bool is_power2(std::size_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

std::size_t parse_size(const char* text) {
    char* end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 0);
    if (!end || *end != '\0' || value == 0) {
        throw std::invalid_argument("invalid size");
    }
    return static_cast<std::size_t>(value);
}

int parse_iters(const char* text) {
    char* end = nullptr;
    const long value = std::strtol(text, &end, 0);
    if (!end || *end != '\0' || value <= 0) {
        throw std::invalid_argument("invalid iteration count");
    }
    return static_cast<int>(value);
}

int default_iters(std::size_t n) {
    const std::size_t target_samples = 1u << 20;
    std::size_t iters = target_samples / n;
    if (iters < 3) iters = 3;
    if (iters > 4096) iters = 4096;
    return static_cast<int>(iters);
}

std::vector<double> make_signal(std::size_t n) {
    std::vector<double> input(n);
    std::mt19937_64 rng(0xD1F0D1F0ULL + static_cast<unsigned long long>(n));
    std::uniform_real_distribution<double> noise(-0.05, 0.05);
    for (std::size_t i = 0; i < n; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(n);
        input[i] = std::sin(2.0 * pi * 13.0 * t)
                 + 0.5 * std::cos(2.0 * pi * 37.0 * t)
                 + 0.25 * std::sin(2.0 * pi * 89.0 * t)
                 + noise(rng);
    }
    return input;
}

template <typename Func>
timing bench_ns(int iters, Func&& func) {
    timing best;
    best.best_ns = 1.0e300;
    for (int pass = 0; pass < 7; ++pass) {
        double sink = 0.0;
        const auto start = clock_type::now();
        for (int i = 0; i < iters; ++i) {
            sink += func(i, sink);
        }
        const auto stop = clock_type::now();
        const double ns = std::chrono::duration<double, std::nano>(stop - start).count()
                        / static_cast<double>(iters);
        if (ns < best.best_ns) {
            best.best_ns = ns;
            best.sink = sink;
        }
    }
    return best;
}

template <typename A, typename B>
double max_abs_complex(const std::vector<A>& a, const std::vector<B>& b) {
    double err = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        err = std::max(err, std::abs(a[i].re - b[i].re));
        err = std::max(err, std::abs(a[i].im - b[i].im));
    }
    return err;
}

double max_abs_real(const std::vector<double>& a, const std::vector<double>& b) {
    double err = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        err = std::max(err, std::abs(a[i] - b[i]));
    }
    return err;
}

result run_one(std::size_t n, int forced_iters) {
    if (n > static_cast<std::size_t>(2147483647)) {
        throw std::invalid_argument("N is too large for the experimental kernels");
    }

    bruun::DIF_RFFT_kernel dif;
    bruun::DIP_RFFT_kernel dip;
    if (!dif.init(static_cast<int>(n)) || !dip.init(static_cast<int>(n))) {
        throw std::runtime_error("kernel setup failed");
    }

    const std::size_t nb = n / 2 + 1;
    const int iters = forced_iters > 0 ? forced_iters : default_iters(n);
    const std::vector<double> original = make_signal(n);
    std::vector<double> input(original);

    std::vector<bruun::complex_t> dif_bins(nb);
    std::vector<bruun::complex_t> dip_bins(nb);
    std::vector<bruun::complex_t> dif_scratch(static_cast<std::size_t>(dif.native_scratch_size()));
    std::vector<double> dif_work(static_cast<std::size_t>(dif.work_size()));
    std::vector<double> dip_work(static_cast<std::size_t>(dip.work_size()));
    std::vector<double> dif_out(n);
    std::vector<double> dip_out(n);

    dif.forward_standard(original.data(), dif_bins.data(), dif_work.data(), dif_scratch.data());
    dip.forward_standard(original.data(), dip_bins.data(), dip_work.data());
    dif.inverse(dif_bins.data(), dif_out.data());
    dip.inverse_standard(dip_bins.data(), dip_out.data(), dip_work.data());

    result r;
    r.n = n;
    r.iters = iters;
    r.forward_maxerr = max_abs_complex(dif_bins, dip_bins);
    r.dif_roundtrip_maxerr = max_abs_real(original, dif_out);
    r.dip_roundtrip_maxerr = max_abs_real(original, dip_out);

    input = original;
    timing t = bench_ns(iters, [&](int i, double sink) {
        input[(static_cast<std::size_t>(i) * 131u + static_cast<std::size_t>(sink)) & (n - 1)] += 1e-12;
        dif.forward_standard(input.data(), dif_bins.data(), dif_work.data(), dif_scratch.data());
        return dif_bins[(static_cast<std::size_t>(i) * 17u) % nb].re;
    });
    r.dif_forward_ns = t.best_ns;
    r.sink += t.sink;

    input = original;
    t = bench_ns(iters, [&](int i, double sink) {
        input[(static_cast<std::size_t>(i) * 131u + static_cast<std::size_t>(sink)) & (n - 1)] += 1e-12;
        dip.forward_standard(input.data(), dip_bins.data(), dip_work.data());
        return dip_bins[(static_cast<std::size_t>(i) * 17u) % nb].re;
    });
    r.dip_forward_ns = t.best_ns;
    r.sink += t.sink;

    dif.forward_standard(original.data(), dif_bins.data(), dif_work.data(), dif_scratch.data());
    t = bench_ns(iters, [&](int i, double) {
        dif_bins[(static_cast<std::size_t>(i) * 17u) % nb].re += 1e-12;
        dif.inverse(dif_bins.data(), dif_out.data());
        return dif_out[(static_cast<std::size_t>(i) * 31u) & (n - 1)];
    });
    r.dif_inverse_ns = t.best_ns;
    r.sink += t.sink;

    dip.forward_standard(original.data(), dip_bins.data(), dip_work.data());
    t = bench_ns(iters, [&](int i, double) {
        dip_bins[(static_cast<std::size_t>(i) * 17u) % nb].re += 1e-12;
        dip.inverse_standard(dip_bins.data(), dip_out.data(), dip_work.data());
        return dip_out[(static_cast<std::size_t>(i) * 31u) & (n - 1)];
    });
    r.dip_inverse_ns = t.best_ns;
    r.sink += t.sink;

    input = original;
    t = bench_ns(iters, [&](int i, double sink) {
        input[(static_cast<std::size_t>(i) * 131u + static_cast<std::size_t>(sink)) & (n - 1)] += 1e-12;
        dif.forward_standard(input.data(), dif_bins.data(), dif_work.data(), dif_scratch.data());
        dif.inverse(dif_bins.data(), dif_out.data());
        return dif_out[(static_cast<std::size_t>(i) * 31u) & (n - 1)];
    });
    r.dif_roundtrip_ns = t.best_ns;
    r.sink += t.sink;

    input = original;
    t = bench_ns(iters, [&](int i, double sink) {
        input[(static_cast<std::size_t>(i) * 131u + static_cast<std::size_t>(sink)) & (n - 1)] += 1e-12;
        dip.forward_standard(input.data(), dip_bins.data(), dip_work.data());
        dip.inverse_standard(dip_bins.data(), dip_out.data(), dip_work.data());
        return dip_out[(static_cast<std::size_t>(i) * 31u) & (n - 1)];
    });
    r.dip_roundtrip_ns = t.best_ns;
    r.sink += t.sink;

    return r;
}

void print_header() {
    std::printf("%9s %8s %13s %13s %13s %13s %13s %13s %8s %8s %8s %12s\n",
                "N", "iters", "DIF_fwd_ns", "DIP_fwd_ns", "DIF_inv_ns",
                "DIP_inv_ns", "DIF_rt_ns", "DIP_rt_ns", "fwd_x",
                "inv_x", "rt_x", "checks");
    std::fflush(stdout);
}

void print_result(const result& r) {
    const double fwd_ratio = r.dip_forward_ns / r.dif_forward_ns;
    const double inv_ratio = r.dip_inverse_ns / r.dif_inverse_ns;
    const double rt_ratio = r.dip_roundtrip_ns / r.dif_roundtrip_ns;
    std::printf("%9zu %8d %13.2f %13.2f %13.2f %13.2f %13.2f %13.2f %8.3f %8.3f %8.3f "
                "ferr %.2e dif_rt %.2e dip_rt %.2e sink %.3e\n",
                r.n,
                r.iters,
                r.dif_forward_ns,
                r.dip_forward_ns,
                r.dif_inverse_ns,
                r.dip_inverse_ns,
                r.dif_roundtrip_ns,
                r.dip_roundtrip_ns,
                fwd_ratio,
                inv_ratio,
                rt_ratio,
                r.forward_maxerr,
                r.dif_roundtrip_maxerr,
                r.dip_roundtrip_maxerr,
                r.sink);
    std::fflush(stdout);
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc > 3) {
            std::fprintf(stderr, "usage: %s [N | max_pow] [iters]\n", argv[0]);
            return 2;
        }

        int forced_iters = 0;
        if (argc >= 3) {
            forced_iters = parse_iters(argv[2]);
        }

        print_header();
        if (argc >= 2) {
            const std::size_t arg = parse_size(argv[1]);
            if (is_power2(arg) && arg >= 4) {
                print_result(run_one(arg, forced_iters));
            } else if (arg >= 2 && arg <= 30) {
                for (int p = 2; p <= static_cast<int>(arg); ++p) {
                    print_result(run_one(static_cast<std::size_t>(1) << p, forced_iters));
                }
            } else {
                throw std::invalid_argument("argument must be a power-of-two N or max_pow in [2, 30]");
            }
            return 0;
        }

        for (int p = 4; p <= 20; ++p) {
            print_result(run_one(static_cast<std::size_t>(1) << p, forced_iters));
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "dif_vs_dip_benchmark failed: %s\n", e.what());
        return 1;
    }
}
