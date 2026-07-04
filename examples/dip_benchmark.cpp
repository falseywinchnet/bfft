#include <bfft/bfft.hpp>

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
    double baseline_forward_ns = 0.0;
    double dip_forward_ns = 0.0;
    double blocked_forward_ns = 0.0;
    double baseline_inverse_ns = 0.0;
    double dip_inverse_ns = 0.0;
    double blocked_inverse_ns = 0.0;
    double baseline_roundtrip_ns = 0.0;
    double dip_roundtrip_ns = 0.0;
    double blocked_roundtrip_ns = 0.0;
    double forward_maxerr = 0.0;
    double blocked_forward_maxerr = 0.0;
    double baseline_roundtrip_maxerr = 0.0;
    double dip_roundtrip_maxerr = 0.0;
    double blocked_roundtrip_maxerr = 0.0;
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

int parse_positive_int(const char* text, const char* name) {
    char* end = nullptr;
    const long value = std::strtol(text, &end, 0);
    if (!end || *end != '\0' || value <= 0 || value > 1024) {
        throw std::invalid_argument(name);
    }
    return static_cast<int>(value);
}

int default_iters(std::size_t n) {
    const std::size_t target_samples = 1u << 20;
    std::size_t iters = target_samples / n;
    if (iters < 3) {
        iters = 3;
    }
    if (iters > 4096) {
        iters = 4096;
    }
    return static_cast<int>(iters);
}

std::vector<double> make_signal(std::size_t n) {
    std::vector<double> input(n);
    std::mt19937_64 rng(0xD150D1A6ULL + static_cast<unsigned long long>(n));
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

result run_one(std::size_t n, int forced_iters, int block_depth, int block_cols) {
    if (n > static_cast<std::size_t>(2147483647)) {
        throw std::invalid_argument("N is too large for the experimental kernel");
    }

    bfft::plan baseline_plan(n);
    bruun::DIP_RFFT_kernel dip_plan;
    if (!dip_plan.reset(static_cast<int>(n))) {
        throw std::runtime_error("DIP plan setup failed");
    }

    const std::size_t nb = baseline_plan.bins();
    const int iters = forced_iters > 0 ? forced_iters : default_iters(n);
    const std::vector<double> original = make_signal(n);
    std::vector<double> input(original);

    std::vector<bfft::complex> baseline_bins(nb);
    std::vector<bruun::complex_t> dip_bins(nb);
    std::vector<bruun::complex_t> blocked_bins(nb);
    std::vector<bfft::complex> baseline_scratch(baseline_plan.native_scratch_size());
    std::vector<double> baseline_work(baseline_plan.work_size());
    std::vector<double> dip_work(static_cast<std::size_t>(dip_plan.work_size()));
    std::vector<double> blocked_work(static_cast<std::size_t>(dip_plan.blocked_work_size()));
    std::vector<double> baseline_out(n);
    std::vector<double> dip_out(n);
    std::vector<double> blocked_out(n);

    baseline_plan.forward(original.data(), baseline_bins.data(), baseline_work.data(), baseline_scratch.data());
    dip_plan.forward_standard(original.data(), dip_bins.data(), dip_work.data());
    dip_plan.forward_standard_blocked(original.data(), blocked_bins.data(), blocked_work.data(), block_depth, block_cols);
    baseline_plan.inverse(baseline_bins.data(), baseline_out.data());
    dip_plan.inverse_standard(dip_bins.data(), dip_out.data(), dip_work.data());
    dip_plan.inverse_standard_blocked(blocked_bins.data(), blocked_out.data(), blocked_work.data(), block_depth, block_cols);

    result r;
    r.n = n;
    r.iters = iters;
    r.forward_maxerr = max_abs_complex(baseline_bins, dip_bins);
    r.blocked_forward_maxerr = max_abs_complex(baseline_bins, blocked_bins);
    r.baseline_roundtrip_maxerr = max_abs_real(original, baseline_out);
    r.dip_roundtrip_maxerr = max_abs_real(original, dip_out);
    r.blocked_roundtrip_maxerr = max_abs_real(original, blocked_out);

    input = original;
    timing t = bench_ns(iters, [&](int i, double sink) {
        input[(static_cast<std::size_t>(i) * 131u + static_cast<std::size_t>(sink)) & (n - 1)] += 1e-12;
        baseline_plan.forward(input.data(), baseline_bins.data(), baseline_work.data(), baseline_scratch.data());
        return baseline_bins[(static_cast<std::size_t>(i) * 17u) % nb].re;
    });
    r.baseline_forward_ns = t.best_ns;
    r.sink += t.sink;

    input = original;
    t = bench_ns(iters, [&](int i, double sink) {
        input[(static_cast<std::size_t>(i) * 131u + static_cast<std::size_t>(sink)) & (n - 1)] += 1e-12;
        dip_plan.forward_standard(input.data(), dip_bins.data(), dip_work.data());
        return dip_bins[(static_cast<std::size_t>(i) * 17u) % nb].re;
    });
    r.dip_forward_ns = t.best_ns;
    r.sink += t.sink;

    input = original;
    t = bench_ns(iters, [&](int i, double sink) {
        input[(static_cast<std::size_t>(i) * 131u + static_cast<std::size_t>(sink)) & (n - 1)] += 1e-12;
        dip_plan.forward_standard_blocked(input.data(), blocked_bins.data(), blocked_work.data(), block_depth, block_cols);
        return blocked_bins[(static_cast<std::size_t>(i) * 17u) % nb].re;
    });
    r.blocked_forward_ns = t.best_ns;
    r.sink += t.sink;

    baseline_plan.forward(original.data(), baseline_bins.data(), baseline_work.data(), baseline_scratch.data());
    t = bench_ns(iters, [&](int i, double) {
        baseline_bins[(static_cast<std::size_t>(i) * 17u) % nb].re += 1e-12;
        baseline_plan.inverse(baseline_bins.data(), baseline_out.data());
        return baseline_out[(static_cast<std::size_t>(i) * 31u) & (n - 1)];
    });
    r.baseline_inverse_ns = t.best_ns;
    r.sink += t.sink;

    dip_plan.forward_standard(original.data(), dip_bins.data(), dip_work.data());
    t = bench_ns(iters, [&](int i, double) {
        dip_bins[(static_cast<std::size_t>(i) * 17u) % nb].re += 1e-12;
        dip_plan.inverse_standard(dip_bins.data(), dip_out.data(), dip_work.data());
        return dip_out[(static_cast<std::size_t>(i) * 31u) & (n - 1)];
    });
    r.dip_inverse_ns = t.best_ns;
    r.sink += t.sink;

    dip_plan.forward_standard_blocked(original.data(), blocked_bins.data(), blocked_work.data(), block_depth, block_cols);
    t = bench_ns(iters, [&](int i, double) {
        blocked_bins[(static_cast<std::size_t>(i) * 17u) % nb].re += 1e-12;
        dip_plan.inverse_standard_blocked(blocked_bins.data(), blocked_out.data(), blocked_work.data(), block_depth, block_cols);
        return blocked_out[(static_cast<std::size_t>(i) * 31u) & (n - 1)];
    });
    r.blocked_inverse_ns = t.best_ns;
    r.sink += t.sink;

    input = original;
    t = bench_ns(iters, [&](int i, double sink) {
        input[(static_cast<std::size_t>(i) * 131u + static_cast<std::size_t>(sink)) & (n - 1)] += 1e-12;
        baseline_plan.forward(input.data(), baseline_bins.data(), baseline_work.data(), baseline_scratch.data());
        baseline_plan.inverse(baseline_bins.data(), baseline_out.data());
        return baseline_out[(static_cast<std::size_t>(i) * 31u) & (n - 1)];
    });
    r.baseline_roundtrip_ns = t.best_ns;
    r.sink += t.sink;

    input = original;
    t = bench_ns(iters, [&](int i, double sink) {
        input[(static_cast<std::size_t>(i) * 131u + static_cast<std::size_t>(sink)) & (n - 1)] += 1e-12;
        dip_plan.forward_standard(input.data(), dip_bins.data(), dip_work.data());
        dip_plan.inverse_standard(dip_bins.data(), dip_out.data(), dip_work.data());
        return dip_out[(static_cast<std::size_t>(i) * 31u) & (n - 1)];
    });
    r.dip_roundtrip_ns = t.best_ns;
    r.sink += t.sink;

    input = original;
    t = bench_ns(iters, [&](int i, double sink) {
        input[(static_cast<std::size_t>(i) * 131u + static_cast<std::size_t>(sink)) & (n - 1)] += 1e-12;
        dip_plan.forward_standard_blocked(input.data(), blocked_bins.data(), blocked_work.data(), block_depth, block_cols);
        dip_plan.inverse_standard_blocked(blocked_bins.data(), blocked_out.data(), blocked_work.data(), block_depth, block_cols);
        return blocked_out[(static_cast<std::size_t>(i) * 31u) & (n - 1)];
    });
    r.blocked_roundtrip_ns = t.best_ns;
    r.sink += t.sink;

    return r;
}

void print_header() {
    std::printf("%9s %8s %11s %11s %11s %11s %11s %11s %11s %11s %11s %8s %8s %8s %12s\n",
                "N", "iters", "base_fwd", "dip_fwd", "blk_fwd", "base_inv",
                "dip_inv", "blk_inv", "base_rt", "dip_rt", "blk_rt",
                "blkf_x", "blki_x", "blkr_x", "checks");
    std::fflush(stdout);
}

void print_result(const result& r) {
    const double blocked_fwd_ratio = r.blocked_forward_ns / r.dip_forward_ns;
    const double blocked_inv_ratio = r.blocked_inverse_ns / r.dip_inverse_ns;
    const double blocked_rt_ratio = r.blocked_roundtrip_ns / r.dip_roundtrip_ns;
    std::printf("%9zu %8d %11.2f %11.2f %11.2f %11.2f %11.2f %11.2f %11.2f %11.2f %11.2f "
                "%8.3f %8.3f %8.3f ferr %.2e blk_ferr %.2e base_rt %.2e dip_rt %.2e blk_rt %.2e sink %.3e\n",
                r.n,
                r.iters,
                r.baseline_forward_ns,
                r.dip_forward_ns,
                r.blocked_forward_ns,
                r.baseline_inverse_ns,
                r.dip_inverse_ns,
                r.blocked_inverse_ns,
                r.baseline_roundtrip_ns,
                r.dip_roundtrip_ns,
                r.blocked_roundtrip_ns,
                blocked_fwd_ratio,
                blocked_inv_ratio,
                blocked_rt_ratio,
                r.forward_maxerr,
                r.blocked_forward_maxerr,
                r.baseline_roundtrip_maxerr,
                r.dip_roundtrip_maxerr,
                r.blocked_roundtrip_maxerr,
                r.sink);
    std::fflush(stdout);
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc > 5) {
            std::fprintf(stderr, "usage: %s [N | max_pow] [iters] [block_depth] [block_cols]\n", argv[0]);
            return 2;
        }

        int forced_iters = 0;
        if (argc >= 3) {
            forced_iters = parse_iters(argv[2]);
        }
        int block_depth = 4;
        int block_cols = 256;
        if (argc >= 4) {
            block_depth = parse_positive_int(argv[3], "invalid block depth");
        }
        if (argc >= 5) {
            block_cols = parse_positive_int(argv[4], "invalid block column count");
        }

        print_header();
        if (argc >= 2) {
            const std::size_t arg = parse_size(argv[1]);
            if (is_power2(arg) && arg >= 4) {
                print_result(run_one(arg, forced_iters, block_depth, block_cols));
            } else if (arg >= 2 && arg <= 30) {
                for (int p = 2; p <= static_cast<int>(arg); ++p) {
                    print_result(run_one(static_cast<std::size_t>(1) << p, forced_iters, block_depth, block_cols));
                }
            } else {
                throw std::invalid_argument("argument must be a power-of-two N or max_pow in [2, 30]");
            }
            return 0;
        }

        for (int p = 4; p <= 20; ++p) {
            print_result(run_one(static_cast<std::size_t>(1) << p, forced_iters, block_depth, block_cols));
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "dip_benchmark failed: %s\n", e.what());
        return 1;
    }
}
