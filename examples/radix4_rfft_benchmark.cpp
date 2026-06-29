#include <bfft/bfft.hpp>

#include "../src/detail/radix4_rfft_kernel.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <random>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;

constexpr double pi = 3.141592653589793238462643383279502884;

struct timing_result {
    double best_ns = 0.0;
    double checksum = 0.0;
};

std::vector<int> default_sizes() {
    std::vector<int> sizes;
    for (int n = 16; n <= 1 << 20; n <<= 2) {
        sizes.push_back(n);
    }
    return sizes;
}

std::vector<int> parse_sizes(int argc, char** argv) {
    std::vector<int> sizes;
    for (int i = 1; i < argc; ++i) {
        const int n = std::atoi(argv[i]);
        if (n > 0) {
            sizes.push_back(n);
        }
    }
    if (sizes.empty()) {
        sizes = default_sizes();
    }
    return sizes;
}

std::vector<double> make_signal(int n) {
    std::vector<double> input(static_cast<std::size_t>(n));
    std::mt19937_64 rng(0xBFF74ULL + static_cast<unsigned long long>(n));
    std::uniform_real_distribution<double> noise(-0.125, 0.125);
    for (int i = 0; i < n; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(n);
        const double a = std::sin(2.0 * pi * 13.0 * t);
        const double b = 0.5 * std::cos(2.0 * pi * 37.0 * t);
        input[static_cast<std::size_t>(i)] = a + b + noise(rng);
    }
    return input;
}

int repeat_count(int n) {
    const int target_samples = 1 << 24;
    int repeats = target_samples / n;
    if (repeats < 8) {
        repeats = 8;
    }
    if (repeats > 4096) {
        repeats = 4096;
    }
    return repeats;
}

template <typename Callable>
timing_result time_callable(int repeats, Callable&& callable) {
    timing_result result;
    result.best_ns = 1.0e300;
    for (int pass = 0; pass < 5; ++pass) {
        const auto start = clock_type::now();
        double checksum = 0.0;
        for (int r = 0; r < repeats; ++r) {
            checksum += callable();
        }
        const auto finish = clock_type::now();
        const double elapsed_ns = std::chrono::duration<double, std::nano>(finish - start).count();
        const double each_ns = elapsed_ns / static_cast<double>(repeats);
        if (each_ns < result.best_ns) {
            result.best_ns = each_ns;
            result.checksum = checksum;
        }
    }
    return result;
}

double max_error(const std::vector<double>& re,
                 const std::vector<double>& im,
                 const std::vector<bfft::complex>& ref) {
    double err = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        err = std::max(err, std::abs(re[i] - ref[i].re));
        err = std::max(err, std::abs(im[i] - ref[i].im));
    }
    return err;
}

void run_size(int n) {
    bruun::radix4_rfft_kernel radix4;
    if (!radix4.reset(n)) {
        std::printf("%d,unsupported,unsupported,unsupported,unsupported,0,0,0\n", n);
        return;
    }

    bfft::plan plan(static_cast<std::size_t>(n));
    const std::vector<double> input = make_signal(n);
    std::vector<double> work(static_cast<std::size_t>(radix4.work_size()));
    std::vector<double> re(static_cast<std::size_t>(radix4.bins()));
    std::vector<double> im(static_cast<std::size_t>(radix4.bins()));
    std::vector<bfft::complex> radix4_output(plan.bins());
    std::vector<bfft::complex> bfft_output(plan.bins());
    std::vector<bfft::complex> bfft_native(plan.bins());
    std::vector<double> bfft_work(plan.work_size());
    std::vector<bfft::complex> bfft_scratch(plan.native_scratch_size());
    bfft::workspace native_workspace = plan.create_workspace();

    plan.forward(input.data(), bfft_output.data(), bfft_work.data(), bfft_scratch.data());
    radix4.forward_complex_simd(input.data(), radix4_output.data(), work.data());
    for (std::size_t i = 0; i < radix4_output.size(); ++i) {
        re[i] = radix4_output[i].re;
        im[i] = radix4_output[i].im;
    }
    const double simd_error = max_error(re, im, bfft_output);
    radix4.forward_complex_scalar(input.data(), radix4_output.data(), work.data());
    for (std::size_t i = 0; i < radix4_output.size(); ++i) {
        re[i] = radix4_output[i].re;
        im[i] = radix4_output[i].im;
    }
    const double scalar_error = max_error(re, im, bfft_output);

    const int repeats = repeat_count(n);
    const timing_result bfft_standard = time_callable(repeats, [&]() {
        plan.forward(input.data(), bfft_output.data(), bfft_work.data(), bfft_scratch.data());
        return bfft_output[1].re;
    });
    const timing_result bfft_native_timing = time_callable(repeats, [&]() {
        plan.forward_native(input.data(), bfft_native.data(), native_workspace);
        return bfft_native[1].re;
    });
    const timing_result radix4_scalar = time_callable(repeats, [&]() {
        radix4.forward_complex_scalar(input.data(), radix4_output.data(), work.data());
        return radix4_output[1].re;
    });
    const timing_result radix4_simd = time_callable(repeats, [&]() {
        radix4.forward_complex_simd(input.data(), radix4_output.data(), work.data());
        return radix4_output[1].re;
    });

    std::printf("%d,%.2f,%.2f,%.2f,%.2f,%d,%.3e,%.3e\n",
                n,
                bfft_standard.best_ns,
                bfft_native_timing.best_ns,
                radix4_scalar.best_ns,
                radix4_simd.best_ns,
                repeats,
                scalar_error,
                simd_error);
}

} // namespace

int main(int argc, char** argv) {
    try {
        const std::vector<int> sizes = parse_sizes(argc, argv);
        std::printf("size,bfft_standard_ns,bfft_native_ns,radix4_scalar_ns,radix4_simd_ns,repeats,radix4_scalar_maxerr,radix4_simd_maxerr\n");
        for (int n : sizes) {
            run_size(n);
        }
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "radix4_rfft_benchmark: %s\n", ex.what());
        return 1;
    }
}
