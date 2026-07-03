#include <bfft/bfft.hpp>

#include "../src/detail/bruun_dit_kernel.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
    double dit_forward_ns = 0.0;
    double dif_inverse_ns = 0.0;
    double dit_inverse_ns = 0.0;
    double dif_roundtrip_ns = 0.0;
    double dit_roundtrip_ns = 0.0;
    double forward_maxerr = 0.0;
    double dif_roundtrip_maxerr = 0.0;
    double dit_roundtrip_maxerr = 0.0;
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
    const std::size_t target_samples = 1u << 25;
    std::size_t iters = target_samples / n;
    if (iters < 8) {
        iters = 8;
    }
    if (iters > 32768) {
        iters = 32768;
    }
    return static_cast<int>(iters);
}

std::vector<double> make_signal(std::size_t n) {
    std::vector<double> input(n);
    std::mt19937_64 rng(0xD1F4D17ULL + static_cast<unsigned long long>(n));
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

template <typename ComplexA, typename ComplexB>
double max_abs_complex(const std::vector<ComplexA>& a,
                       const std::vector<ComplexB>& b) {
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
        throw std::invalid_argument("N is too large for the experimental kernel");
    }

    bfft::plan dif_plan(n);
    bruun::DIT_RFFT_kernel dit_plan;
    if (!dit_plan.reset(static_cast<int>(n))) {
        throw std::runtime_error("experimental DIT plan setup failed");
    }

    const std::size_t nb = dif_plan.bins();
    const int iters = forced_iters > 0 ? forced_iters : default_iters(n);
    const std::vector<double> original = make_signal(n);
    std::vector<double> input(original);

    std::vector<bfft::complex> dif_bins(nb);
    std::vector<bruun::complex_t> dit_bins(nb);
    std::vector<bfft::complex> dif_scratch(dif_plan.native_scratch_size());
    std::vector<double> dif_work(dif_plan.work_size());
    std::vector<double> dit_work(static_cast<std::size_t>(dit_plan.work_size()));
    std::vector<double> dif_out(n);
    std::vector<double> dit_out(n);

    dif_plan.forward(original.data(), dif_bins.data(), dif_work.data(), dif_scratch.data());
    dit_plan.forward_simd(original.data(), dit_bins.data(), dit_work.data());
    dif_plan.inverse(dif_bins.data(), dif_out.data());
    dit_plan.inverse_simd(dit_bins.data(), dit_out.data(), dit_work.data());

    result r;
    r.n = n;
    r.iters = iters;
    r.forward_maxerr = max_abs_complex(dif_bins, dit_bins);
    r.dif_roundtrip_maxerr = max_abs_real(original, dif_out);
    r.dit_roundtrip_maxerr = max_abs_real(original, dit_out);

    input = original;
    timing t = bench_ns(iters, [&](int i, double sink) {
        input[(static_cast<std::size_t>(i) * 131u + static_cast<std::size_t>(sink)) & (n - 1)] += 1e-12;
        dif_plan.forward(input.data(), dif_bins.data(), dif_work.data(), dif_scratch.data());
        return dif_bins[(static_cast<std::size_t>(i) * 17u) % nb].re;
    });
    r.dif_forward_ns = t.best_ns;
    r.sink += t.sink;

    input = original;
    t = bench_ns(iters, [&](int i, double sink) {
        input[(static_cast<std::size_t>(i) * 131u + static_cast<std::size_t>(sink)) & (n - 1)] += 1e-12;
        dit_plan.forward_simd(input.data(), dit_bins.data(), dit_work.data());
        return dit_bins[(static_cast<std::size_t>(i) * 17u) % nb].re;
    });
    r.dit_forward_ns = t.best_ns;
    r.sink += t.sink;

    dif_plan.forward(original.data(), dif_bins.data(), dif_work.data(), dif_scratch.data());
    t = bench_ns(iters, [&](int i, double) {
        dif_bins[(static_cast<std::size_t>(i) * 17u) % nb].re += 1e-12;
        dif_plan.inverse(dif_bins.data(), dif_out.data());
        return dif_out[(static_cast<std::size_t>(i) * 31u) & (n - 1)];
    });
    r.dif_inverse_ns = t.best_ns;
    r.sink += t.sink;

    dit_plan.forward_simd(original.data(), dit_bins.data(), dit_work.data());
    t = bench_ns(iters, [&](int i, double) {
        dit_bins[(static_cast<std::size_t>(i) * 17u) % nb].re += 1e-12;
        dit_plan.inverse_simd(dit_bins.data(), dit_out.data(), dit_work.data());
        return dit_out[(static_cast<std::size_t>(i) * 31u) & (n - 1)];
    });
    r.dit_inverse_ns = t.best_ns;
    r.sink += t.sink;

    input = original;
    t = bench_ns(iters, [&](int i, double sink) {
        input[(static_cast<std::size_t>(i) * 131u + static_cast<std::size_t>(sink)) & (n - 1)] += 1e-12;
        dif_plan.forward(input.data(), dif_bins.data(), dif_work.data(), dif_scratch.data());
        dif_plan.inverse(dif_bins.data(), dif_out.data());
        return dif_out[(static_cast<std::size_t>(i) * 31u) & (n - 1)];
    });
    r.dif_roundtrip_ns = t.best_ns;
    r.sink += t.sink;

    input = original;
    t = bench_ns(iters, [&](int i, double sink) {
        input[(static_cast<std::size_t>(i) * 131u + static_cast<std::size_t>(sink)) & (n - 1)] += 1e-12;
        dit_plan.forward_simd(input.data(), dit_bins.data(), dit_work.data());
        dit_plan.inverse_simd(dit_bins.data(), dit_out.data(), dit_work.data());
        return dit_out[(static_cast<std::size_t>(i) * 31u) & (n - 1)];
    });
    r.dit_roundtrip_ns = t.best_ns;
    r.sink += t.sink;

    return r;
}

void print_header() {
    std::printf("%9s %8s %13s %13s %13s %13s %13s %13s %8s %8s %8s %9s %9s %10s\n",
                "N", "iters", "DIF_fwd_ns", "DIT_fwd_ns", "DIF_inv_ns",
                "DIT_inv_ns", "DIF_rt_ns", "DIT_rt_ns", "fwd_x",
                "inv_x", "rt_x", "DIF_rt/s", "DIT_rt/s", "checks");
}

void print_result(const result& r) {
    const double fwd_ratio = r.dit_forward_ns / r.dif_forward_ns;
    const double inv_ratio = r.dit_inverse_ns / r.dif_inverse_ns;
    const double rt_ratio = r.dit_roundtrip_ns / r.dif_roundtrip_ns;
    const double dif_rt_sum_ratio = r.dif_roundtrip_ns / (r.dif_forward_ns + r.dif_inverse_ns);
    const double dit_rt_sum_ratio = r.dit_roundtrip_ns / (r.dit_forward_ns + r.dit_inverse_ns);
    std::printf("%9zu %8d %13.2f %13.2f %13.2f %13.2f %13.2f %13.2f %8.3f %8.3f %8.3f %9.3f %9.3f "
                "ferr %.2e dif_rt %.2e dit_rt %.2e sink %.3e\n",
                r.n,
                r.iters,
                r.dif_forward_ns,
                r.dit_forward_ns,
                r.dif_inverse_ns,
                r.dit_inverse_ns,
                r.dif_roundtrip_ns,
                r.dit_roundtrip_ns,
                fwd_ratio,
                inv_ratio,
                rt_ratio,
                dif_rt_sum_ratio,
                dit_rt_sum_ratio,
                r.forward_maxerr,
                r.dif_roundtrip_maxerr,
                r.dit_roundtrip_maxerr,
                r.sink);
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::size_t forced_n = 0;
        int forced_iters = 0;
        int positional = 0;

        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
                std::printf("usage: %s [N [iters]]\n", argv[0]);
                return 0;
            }
            if (positional == 0) {
                forced_n = parse_size(argv[i]);
            } else if (positional == 1) {
                forced_iters = parse_iters(argv[i]);
            } else {
                std::fprintf(stderr, "usage: %s [N [iters]]\n", argv[0]);
                return 2;
            }
            ++positional;
        }

        std::printf("DIF vs experimental DIT RFFT benchmark. backend: %s, DIT backend: %s\n",
                    bfft::backend_name().c_str(),
                    bruun::simd_backend_name());
        std::printf("DIT inverse column uses DIT_RFFT_kernel::inverse_simd.\n");
        print_header();

        if (forced_n > 0) {
            if (!is_power2(forced_n) || forced_n < 4) {
                std::fprintf(stderr, "N must be a power of two >= 4.\n");
                return 2;
            }
            print_result(run_one(forced_n, forced_iters));
            return 0;
        }

        const std::size_t sizes[] = {
            512,
            1024,
            2048,
            4096,
            8192,
            16384,
            32768,
            65536,
            131072,
            262144,
            524288,
            1048576
        };
        for (std::size_t n : sizes) {
            print_result(run_one(n, forced_iters));
        }
        return 0;
    } catch (const std::exception& exc) {
        std::fprintf(stderr, "dif_vs_dit_benchmark failed: %s\n", exc.what());
        return 1;
    }
}
