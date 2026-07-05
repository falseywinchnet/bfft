// Three-way real-FFT microbenchmark: DIF vs DIT vs DIP, forward / inverse /
// roundtrip, f64. DIF is the ratio baseline (its residue kernel is the tuned
// reference the other two are chasing).
//
// Hardening (carried over from dif_vs_dip_benchmark, extended to 3 engines):
//   - INTERLEAVED passes: every pass times a chunk of DIF, then DIT, then DIP,
//     so all three ride the same thermal / frequency-scaling trajectory and
//     the "measure one fully then the next" ordering bias is cancelled.
//   - MEDIAN of many passes (not best-of): robust to transient spikes.
//   - CALIBRATED iteration counts: each timed chunk runs >= kTargetChunkNs, so
//     sub-microsecond transforms at small N get thousands of iterations and
//     become measurable instead of quantized to the clock.

#include "../src/detail/bruun_dif_kernel.hpp"
#include "../src/detail/bruun_dit_kernel.hpp"
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
constexpr double kTargetChunkNs = 2.0e6;
constexpr int kPasses = 11;

struct triple_timing {
    double a_ns = 0.0;    // DIF
    double b_ns = 0.0;    // DIT
    double c_ns = 0.0;    // DIP
    double sink = 0.0;
};

struct result {
    std::size_t n = 0;
    int iters = 0;
    double fwd[3] = {0, 0, 0};      // DIF, DIT, DIP
    double inv[3] = {0, 0, 0};
    double rt[3] = {0, 0, 0};
    double dit_fwd_err = 0.0;
    double dip_fwd_err = 0.0;
    double dit_rt_err = 0.0;
    double dip_rt_err = 0.0;
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

template <typename Func>
int calibrate_iters(Func&& func) {
    const auto start = clock_type::now();
    double sink = 0.0;
    for (int i = 0; i < 4; ++i) {
        sink += func(i, sink);
    }
    const auto stop = clock_type::now();
    const double per_op =
        std::chrono::duration<double, std::nano>(stop - start).count() / 4.0 + 1.0;
    double iters = kTargetChunkNs / per_op;
    if (iters < 3.0) iters = 3.0;
    if (iters > 262144.0) iters = 262144.0;
    return static_cast<int>(iters) + (sink == 0.12345 ? 1 : 0);
}

double median_of(double* v, int count) {
    std::sort(v, v + count);
    return (count & 1) ? v[count / 2] : 0.5 * (v[count / 2 - 1] + v[count / 2]);
}

// Interleaved three-engine measurement: every pass times a chunk of A, then B,
// then C, so all ride the same thermal trajectory. Reported value is the
// per-engine median of the per-pass times.
template <typename FA, typename FB, typename FC>
triple_timing bench_triple(int iters, FA&& fa, FB&& fb, FC&& fc) {
    double as[kPasses], bs[kPasses], cs[kPasses];
    triple_timing out;
    for (int pass = 0; pass < kPasses; ++pass) {
        double sink = 0.0;
        auto t0 = clock_type::now();
        for (int i = 0; i < iters; ++i) sink += fa(i, sink);
        auto t1 = clock_type::now();
        for (int i = 0; i < iters; ++i) sink += fb(i, sink);
        auto t2 = clock_type::now();
        for (int i = 0; i < iters; ++i) sink += fc(i, sink);
        auto t3 = clock_type::now();
        as[pass] = std::chrono::duration<double, std::nano>(t1 - t0).count() / iters;
        bs[pass] = std::chrono::duration<double, std::nano>(t2 - t1).count() / iters;
        cs[pass] = std::chrono::duration<double, std::nano>(t3 - t2).count() / iters;
        out.sink += sink;
    }
    out.a_ns = median_of(as, kPasses);
    out.b_ns = median_of(bs, kPasses);
    out.c_ns = median_of(cs, kPasses);
    return out;
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

std::vector<double> make_signal(std::size_t n) {
    std::vector<double> input(n);
    std::mt19937_64 rng(0xD1F0D17DULL + static_cast<unsigned long long>(n));
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

result run_one(std::size_t n, int forced_iters) {
    if (n > static_cast<std::size_t>(2147483647)) {
        throw std::invalid_argument("N is too large for the experimental kernels");
    }

    bruun::DIF_RFFT_kernel dif;
    bruun::DIT_RFFT_kernel dit;
    bruun::DIP_RFFT_kernel dip;
    if (!dif.init(static_cast<int>(n)) || !dit.init(static_cast<int>(n)) ||
        !dip.init(static_cast<int>(n))) {
        throw std::runtime_error("kernel setup failed");
    }

    const std::size_t nb = n / 2 + 1;
    const std::vector<double> original = make_signal(n);
    std::vector<double> input(original);

    std::vector<bruun::complex_t> dif_bins(nb), dit_bins(nb), dip_bins(nb);
    std::vector<bruun::complex_t> dif_scratch(static_cast<std::size_t>(dif.native_scratch_size()));
    std::vector<double> dif_work(static_cast<std::size_t>(dif.work_size()));
    std::vector<double> dit_work(static_cast<std::size_t>(dit.work_size()));
    std::vector<double> dip_work(static_cast<std::size_t>(dip.work_size()));
    std::vector<double> dif_out(n), dit_out(n), dip_out(n);

    // warm + correctness reference
    dif.forward_standard(original.data(), dif_bins.data(), dif_work.data(), dif_scratch.data());
    dit.forward_simd(original.data(), dit_bins.data(), dit_work.data());
    dip.forward_standard(original.data(), dip_bins.data(), dip_work.data());
    dif.inverse(dif_bins.data(), dif_out.data());
    dit.inverse_simd(dit_bins.data(), dit_out.data(), dit_work.data());
    dip.inverse_standard(dip_bins.data(), dip_out.data(), dip_work.data());

    result r;
    r.n = n;
    r.dit_fwd_err = max_abs_complex(dif_bins, dit_bins);
    r.dip_fwd_err = max_abs_complex(dif_bins, dip_bins);
    r.dit_rt_err = max_abs_real(original, dit_out);
    r.dip_rt_err = max_abs_real(original, dip_out);

    auto dif_fwd = [&](int i, double sink) {
        input[(static_cast<std::size_t>(i) * 131u + static_cast<std::size_t>(sink)) & (n - 1)] += 1e-12;
        dif.forward_standard(input.data(), dif_bins.data(), dif_work.data(), dif_scratch.data());
        return dif_bins[(static_cast<std::size_t>(i) * 17u) % nb].re;
    };
    auto dit_fwd = [&](int i, double sink) {
        input[(static_cast<std::size_t>(i) * 131u + static_cast<std::size_t>(sink)) & (n - 1)] += 1e-12;
        dit.forward_simd(input.data(), dit_bins.data(), dit_work.data());
        return dit_bins[(static_cast<std::size_t>(i) * 17u) % nb].re;
    };
    auto dip_fwd = [&](int i, double sink) {
        input[(static_cast<std::size_t>(i) * 131u + static_cast<std::size_t>(sink)) & (n - 1)] += 1e-12;
        dip.forward_standard(input.data(), dip_bins.data(), dip_work.data());
        return dip_bins[(static_cast<std::size_t>(i) * 17u) % nb].re;
    };
    auto dif_inv = [&](int i, double) {
        dif_bins[(static_cast<std::size_t>(i) * 17u) % nb].re += 1e-12;
        dif.inverse(dif_bins.data(), dif_out.data());
        return dif_out[(static_cast<std::size_t>(i) * 31u) & (n - 1)];
    };
    auto dit_inv = [&](int i, double) {
        dit_bins[(static_cast<std::size_t>(i) * 17u) % nb].re += 1e-12;
        dit.inverse_simd(dit_bins.data(), dit_out.data(), dit_work.data());
        return dit_out[(static_cast<std::size_t>(i) * 31u) & (n - 1)];
    };
    auto dip_inv = [&](int i, double) {
        dip_bins[(static_cast<std::size_t>(i) * 17u) % nb].re += 1e-12;
        dip.inverse_standard(dip_bins.data(), dip_out.data(), dip_work.data());
        return dip_out[(static_cast<std::size_t>(i) * 31u) & (n - 1)];
    };
    auto dif_rt = [&](int i, double sink) {
        input[(static_cast<std::size_t>(i) * 131u + static_cast<std::size_t>(sink)) & (n - 1)] += 1e-12;
        dif.forward_standard(input.data(), dif_bins.data(), dif_work.data(), dif_scratch.data());
        dif.inverse(dif_bins.data(), dif_out.data());
        return dif_out[(static_cast<std::size_t>(i) * 31u) & (n - 1)];
    };
    auto dit_rt = [&](int i, double sink) {
        input[(static_cast<std::size_t>(i) * 131u + static_cast<std::size_t>(sink)) & (n - 1)] += 1e-12;
        dit.forward_simd(input.data(), dit_bins.data(), dit_work.data());
        dit.inverse_simd(dit_bins.data(), dit_out.data(), dit_work.data());
        return dit_out[(static_cast<std::size_t>(i) * 31u) & (n - 1)];
    };
    auto dip_rt = [&](int i, double sink) {
        input[(static_cast<std::size_t>(i) * 131u + static_cast<std::size_t>(sink)) & (n - 1)] += 1e-12;
        dip.forward_standard(input.data(), dip_bins.data(), dip_work.data());
        dip.inverse_standard(dip_bins.data(), dip_out.data(), dip_work.data());
        return dip_out[(static_cast<std::size_t>(i) * 31u) & (n - 1)];
    };

    input = original;
    const int iters = forced_iters > 0 ? forced_iters : calibrate_iters(dif_fwd);
    r.iters = iters;

    input = original;
    triple_timing t = bench_triple(iters, dif_fwd, dit_fwd, dip_fwd);
    r.fwd[0] = t.a_ns; r.fwd[1] = t.b_ns; r.fwd[2] = t.c_ns;
    r.sink += t.sink;

    dif.forward_standard(original.data(), dif_bins.data(), dif_work.data(), dif_scratch.data());
    dit.forward_simd(original.data(), dit_bins.data(), dit_work.data());
    dip.forward_standard(original.data(), dip_bins.data(), dip_work.data());
    t = bench_triple(iters, dif_inv, dit_inv, dip_inv);
    r.inv[0] = t.a_ns; r.inv[1] = t.b_ns; r.inv[2] = t.c_ns;
    r.sink += t.sink;

    input = original;
    t = bench_triple(iters, dif_rt, dit_rt, dip_rt);
    r.rt[0] = t.a_ns; r.rt[1] = t.b_ns; r.rt[2] = t.c_ns;
    r.sink += t.sink;

    return r;
}

void print_header() {
    std::printf("%9s %8s %11s %11s %11s %11s %11s %11s %11s %11s %11s %8s %8s %8s %8s %8s %8s %s\n",
                "N", "iters",
                "DIF_fwd", "DIT_fwd", "DIP_fwd",
                "DIF_inv", "DIT_inv", "DIP_inv",
                "DIF_rt", "DIT_rt", "DIP_rt",
                "ditf_x", "dipf_x", "diti_x", "dipi_x", "ditr_x", "dipr_x",
                "checks");
    std::fflush(stdout);
}

void print_result(const result& r) {
    std::printf("%9zu %8d %11.2f %11.2f %11.2f %11.2f %11.2f %11.2f %11.2f %11.2f %11.2f "
                "%8.3f %8.3f %8.3f %8.3f %8.3f %8.3f "
                "dit_fe %.1e dip_fe %.1e dit_re %.1e dip_re %.1e\n",
                r.n, r.iters,
                r.fwd[0], r.fwd[1], r.fwd[2],
                r.inv[0], r.inv[1], r.inv[2],
                r.rt[0], r.rt[1], r.rt[2],
                r.fwd[1] / r.fwd[0], r.fwd[2] / r.fwd[0],
                r.inv[1] / r.inv[0], r.inv[2] / r.inv[0],
                r.rt[1] / r.rt[0], r.rt[2] / r.rt[0],
                r.dit_fwd_err, r.dip_fwd_err, r.dit_rt_err, r.dip_rt_err);
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
        std::fprintf(stderr, "dif_dit_dip_benchmark failed: %s\n", e.what());
        return 1;
    }
}
