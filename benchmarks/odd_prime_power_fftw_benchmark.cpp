// Odd, prime, and prime-power FFTW benchmark for arbitrary-size Bruun/QHSO work.
//
// Times FFTW (baseline) vs bfft for odd/prime/prime-power sizes. bfft arbitrary-N
// runs the generalized Bruun (z^N-1 real factorization) plan (bruun::GenBruun);
// power-of-two sizes use the native bruun::RFFT fast path.
//
// Build:
//   g++ -O3 -march=native -std=c++17 benchmarks/odd_prime_power_fftw_benchmark.cpp -ldl -lm -o odd_prime_power_fftw_bench
// Run:
//   ./odd_prime_power_fftw_bench [max_n] [iters]

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <dlfcn.h>
#endif

#include "detail/genbruun_kernel.hpp"  // brings bruun::RFFT, bruun::GenBruun, complex_t

namespace {

using fftw_plan = void*;
using fftwf_plan = void*;
using fftw_complex = double[2];
using fftwf_complex = float[2];

struct SizeCase {
    int n;
    const char* label;
};

struct BenchRow {
    int n;
    const char* label;
    const char* precision;
    int iterations;
    double fftw_rfft_ns;
    double fftw_irfft_ns;
    double fftw_roundtrip_rel_l2;
    const char* bfft_status;
    double bfft_rfft_ns;
    double bfft_irfft_ns;
    double bfft_vs_fftw_rel_l2;
};

static bool is_power2_int(int n) {
    return n > 0 && ((n & (n - 1)) == 0);
}

static bool is_prime_int(int n) {
    if (n < 2) return false;
    if (n == 2) return true;
    if ((n & 1) == 0) return false;
    const int limit = static_cast<int>(std::sqrt(static_cast<double>(n)));
    for (int candidate = 3; candidate <= limit; candidate += 2) {
        if (n % candidate == 0) return false;
    }
    return true;
}

static int previous_prime(int n) {
    for (int candidate = n; candidate >= 2; --candidate) {
        if (is_prime_int(candidate)) return candidate;
    }
    throw std::runtime_error("no previous prime exists");
}

static int next_prime(int n) {
    int candidate = std::max(2, n);
    while (true) {
        if (is_prime_int(candidate)) return candidate;
        ++candidate;
    }
}

static void add_case(std::vector<SizeCase>& cases, int n, const char* label, int max_n) {
    if (n > max_n) return;
    for (const SizeCase& existing : cases) {
        if (existing.n == n) return;
    }
    cases.push_back(SizeCase{n, label});
}

static std::vector<SizeCase> default_size_cases(int max_n) {
    std::vector<SizeCase> cases;

    int value = 9;
    for (int exponent = 2; exponent <= 10; ++exponent) {
        add_case(cases, value, "3^k", max_n);
        value *= 3;
    }

    value = 25;
    for (int exponent = 2; exponent <= 7; ++exponent) {
        add_case(cases, value, "5^k", max_n);
        value *= 5;
    }

    value = 49;
    for (int exponent = 2; exponent <= 6; ++exponent) {
        add_case(cases, value, "7^k", max_n);
        value *= 7;
    }

    const int odd_composites[] = {45, 75, 81, 135, 225, 243, 405, 675, 1215, 2025, 3645, 6075, 10125};
    for (int n : odd_composites) {
        add_case(cases, n, "odd composite", max_n);
    }

    for (int power = 5; power <= 20; ++power) {
        const int pivot = 1 << power;
        add_case(cases, previous_prime(pivot - 1), "prime below 2^k", max_n);
        add_case(cases, next_prime(pivot + 1), "prime above 2^k", max_n);
    }

    std::sort(cases.begin(), cases.end(), [](const SizeCase& a, const SizeCase& b) {
        return a.n < b.n;
    });
    return cases;
}

static int default_iterations(int n) {
    int iterations = 200000 / std::max(1, n);
    if (iterations < 8) iterations = 8;
    if (iterations > 1000) iterations = 1000;
    return iterations;
}

static double relative_l2_double(const double* a, const double* b, int n) {
    long double numerator = 0.0L;
    long double denominator = 0.0L;
    for (int i = 0; i < n; ++i) {
        const long double diff = static_cast<long double>(a[i]) - static_cast<long double>(b[i]);
        numerator += diff * diff;
        denominator += static_cast<long double>(b[i]) * static_cast<long double>(b[i]);
    }
    if (denominator < 1.0L) denominator = 1.0L;
    return static_cast<double>(std::sqrt(numerator / denominator));
}

static double relative_l2_float(const float* a, const float* b, int n) {
    long double numerator = 0.0L;
    long double denominator = 0.0L;
    for (int i = 0; i < n; ++i) {
        const long double diff = static_cast<long double>(a[i]) - static_cast<long double>(b[i]);
        numerator += diff * diff;
        denominator += static_cast<long double>(b[i]) * static_cast<long double>(b[i]);
    }
    if (denominator < 1.0L) denominator = 1.0L;
    return static_cast<double>(std::sqrt(numerator / denominator));
}

static double relative_l2_complex_double(const bruun::complex_t* a, const fftw_complex* b, int n) {
    long double numerator = 0.0L;
    long double denominator = 0.0L;
    for (int i = 0; i < n; ++i) {
        const long double dr = static_cast<long double>(a[i].re) - static_cast<long double>(b[i][0]);
        const long double di = static_cast<long double>(a[i].im) - static_cast<long double>(b[i][1]);
        numerator += dr * dr + di * di;
        denominator += static_cast<long double>(b[i][0]) * static_cast<long double>(b[i][0]);
        denominator += static_cast<long double>(b[i][1]) * static_cast<long double>(b[i][1]);
    }
    if (denominator < 1.0L) denominator = 1.0L;
    return static_cast<double>(std::sqrt(numerator / denominator));
}

struct FFTWDoubleApi {
    void* lib = nullptr;
    fftw_plan (*plan_r2c)(int, double*, fftw_complex*, unsigned) = nullptr;
    fftw_plan (*plan_c2r)(int, fftw_complex*, double*, unsigned) = nullptr;
    void (*execute)(const fftw_plan) = nullptr;
    void (*destroy_plan)(fftw_plan) = nullptr;
    void* (*malloc_fn)(size_t) = nullptr;
    void (*free_fn)(void*) = nullptr;

    bool load() {
#if defined(_WIN32)
        return false;
#else
        const char* names[] = {
            "libfftw3.dylib",
            "libfftw3.3.dylib",
            "/opt/homebrew/lib/libfftw3.dylib",
            "/usr/local/lib/libfftw3.dylib",
            "libfftw3.so.3",
            "libfftw3.so",
            nullptr
        };
        for (int i = 0; names[i] != nullptr; ++i) {
            lib = dlopen(names[i], RTLD_NOW);
            if (lib != nullptr) break;
        }
        if (lib == nullptr) return false;
        plan_r2c = reinterpret_cast<fftw_plan (*)(int, double*, fftw_complex*, unsigned)>(dlsym(lib, "fftw_plan_dft_r2c_1d"));
        plan_c2r = reinterpret_cast<fftw_plan (*)(int, fftw_complex*, double*, unsigned)>(dlsym(lib, "fftw_plan_dft_c2r_1d"));
        execute = reinterpret_cast<void (*)(const fftw_plan)>(dlsym(lib, "fftw_execute"));
        destroy_plan = reinterpret_cast<void (*)(fftw_plan)>(dlsym(lib, "fftw_destroy_plan"));
        malloc_fn = reinterpret_cast<void* (*)(size_t)>(dlsym(lib, "fftw_malloc"));
        free_fn = reinterpret_cast<void (*)(void*)>(dlsym(lib, "fftw_free"));
        return plan_r2c != nullptr && plan_c2r != nullptr && execute != nullptr && destroy_plan != nullptr && malloc_fn != nullptr && free_fn != nullptr;
#endif
    }

    ~FFTWDoubleApi() {
#if !defined(_WIN32)
        if (lib != nullptr) dlclose(lib);
#endif
    }
};

struct FFTWFloatApi {
    void* lib = nullptr;
    fftwf_plan (*plan_r2c)(int, float*, fftwf_complex*, unsigned) = nullptr;
    fftwf_plan (*plan_c2r)(int, fftwf_complex*, float*, unsigned) = nullptr;
    void (*execute)(const fftwf_plan) = nullptr;
    void (*destroy_plan)(fftwf_plan) = nullptr;
    void* (*malloc_fn)(size_t) = nullptr;
    void (*free_fn)(void*) = nullptr;

    bool load() {
#if defined(_WIN32)
        return false;
#else
        const char* names[] = {
            "libfftw3f.dylib",
            "libfftw3f.3.dylib",
            "/opt/homebrew/lib/libfftw3f.dylib",
            "/usr/local/lib/libfftw3f.dylib",
            "libfftw3f.so.3",
            "libfftw3f.so",
            nullptr
        };
        for (int i = 0; names[i] != nullptr; ++i) {
            lib = dlopen(names[i], RTLD_NOW);
            if (lib != nullptr) break;
        }
        if (lib == nullptr) return false;
        plan_r2c = reinterpret_cast<fftwf_plan (*)(int, float*, fftwf_complex*, unsigned)>(dlsym(lib, "fftwf_plan_dft_r2c_1d"));
        plan_c2r = reinterpret_cast<fftwf_plan (*)(int, fftwf_complex*, float*, unsigned)>(dlsym(lib, "fftwf_plan_dft_c2r_1d"));
        execute = reinterpret_cast<void (*)(const fftwf_plan)>(dlsym(lib, "fftwf_execute"));
        destroy_plan = reinterpret_cast<void (*)(fftwf_plan)>(dlsym(lib, "fftwf_destroy_plan"));
        malloc_fn = reinterpret_cast<void* (*)(size_t)>(dlsym(lib, "fftwf_malloc"));
        free_fn = reinterpret_cast<void (*)(void*)>(dlsym(lib, "fftwf_free"));
        return plan_r2c != nullptr && plan_c2r != nullptr && execute != nullptr && destroy_plan != nullptr && malloc_fn != nullptr && free_fn != nullptr;
#endif
    }

    ~FFTWFloatApi() {
#if !defined(_WIN32)
        if (lib != nullptr) dlclose(lib);
#endif
    }
};

class BfftRfftShim {
public:
    explicit BfftRfftShim(int n) : n_(n) {
        if (is_power2_int(n) && n >= 4) { plan_ = new bruun::RFFT(); plan_->init(n_); }
        else gen_ = new bruun::GenBruun(n_);              // arbitrary N (incl. odd/prime)
    }
    ~BfftRfftShim() { delete plan_; delete gen_; }

    bool forward_double(const double* input, bruun::complex_t* output, double* work, bruun::complex_t* native_tmp) const {
        if (plan_) { plan_->forward_standard(input, output, work, native_tmp); return true; }
        gen_->forward(input, output); return true;
    }
    bool inverse_double(const bruun::complex_t* input, double* output) const {
        if (plan_) { plan_->inverse(input, output); return true; }
        gen_->inverse(input, output); return true;
    }
    bool forward_float(const float*, fftwf_complex*, float*) const { return false; }
    bool inverse_float(const fftwf_complex*, float*, float*) const { return false; }
    const char* status_for_double() const { return "run"; }
    const char* status_for_float() const { return "skip-float-todo"; }

private:
    int n_;
    bruun::RFFT* plan_ = nullptr;        // pow2 fast path
    mutable bruun::GenBruun* gen_ = nullptr;  // arbitrary-N path
};

static BenchRow bench_double_case(const SizeCase& size_case, int forced_iterations, FFTWDoubleApi& fftw, std::mt19937_64& rng) {
    const int n = size_case.n;
    const int bins = n / 2 + 1;
    int iterations = default_iterations(n);
    if (forced_iterations > 0) {
        iterations = forced_iterations;
    }
    const unsigned fftw_measure = 64U;

    double* fftw_input = static_cast<double*>(fftw.malloc_fn(sizeof(double) * n));
    double* fftw_inverse = static_cast<double*>(fftw.malloc_fn(sizeof(double) * n));
    fftw_complex* fftw_spectrum = static_cast<fftw_complex*>(fftw.malloc_fn(sizeof(fftw_complex) * bins));

    fftw_plan forward_plan = fftw.plan_r2c(n, fftw_input, fftw_spectrum, fftw_measure);
    fftw_plan inverse_plan = fftw.plan_c2r(n, fftw_spectrum, fftw_inverse, fftw_measure);

    std::vector<double> original(n);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int i = 0; i < n; ++i) {
        const double value = dist(rng);
        original[i] = value;
        fftw_input[i] = value;
    }

    fftw.execute(forward_plan);
    fftw.execute(inverse_plan);
    for (int i = 0; i < n; ++i) {
        fftw_inverse[i] /= static_cast<double>(n);
    }
    const double fftw_roundtrip = relative_l2_double(fftw_inverse, original.data(), n);

    volatile double sink = 0.0;
    auto forward_start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        fftw_input[iter % n] += 1.0e-12;
        fftw.execute(forward_plan);
        sink += fftw_spectrum[(iter * 17) % bins][0];
    }
    auto forward_stop = std::chrono::high_resolution_clock::now();

    auto inverse_start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        fftw_spectrum[(iter * 13) % bins][0] += 1.0e-12;
        fftw.execute(inverse_plan);
        sink += fftw_inverse[(iter * 19) % n];
    }
    auto inverse_stop = std::chrono::high_resolution_clock::now();

    const double fftw_forward_ns = std::chrono::duration<double, std::nano>(forward_stop - forward_start).count() / static_cast<double>(iterations);
    const double fftw_inverse_ns = std::chrono::duration<double, std::nano>(inverse_stop - inverse_start).count() / static_cast<double>(iterations);

    BfftRfftShim bfft(n);
    std::vector<bruun::complex_t> bfft_output(bins);
    std::vector<bruun::complex_t> native_tmp(bins);
    std::vector<double> bfft_work(n);
    double bfft_forward_ns = 0.0;
    double bfft_inverse_ns = 0.0;
    double bfft_error = 0.0;
    std::memcpy(fftw_input, original.data(), sizeof(double) * n);
    fftw.execute(forward_plan);

    std::vector<double> bfft_input = original;
    const bool bfft_runs = bfft.forward_double(bfft_input.data(), bfft_output.data(), bfft_work.data(), native_tmp.data());
    if (bfft_runs) {
        bfft_error = relative_l2_complex_double(bfft_output.data(), fftw_spectrum, bins);
        auto bfft_start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < iterations; ++iter) {
            bfft_input[iter % n] += 1.0e-12;
            bfft.forward_double(bfft_input.data(), bfft_output.data(), bfft_work.data(), native_tmp.data());
            sink += bfft_output[(iter * 17) % bins].re;
        }
        auto bfft_stop = std::chrono::high_resolution_clock::now();
        bfft_forward_ns = std::chrono::duration<double, std::nano>(bfft_stop - bfft_start).count() / static_cast<double>(iterations);

        std::vector<double> bfft_inverse(n);
        auto bfft_inverse_start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < iterations; ++iter) {
            bfft_output[(iter * 13) % bins].re += 1.0e-12;
            bfft.inverse_double(bfft_output.data(), bfft_inverse.data());
            sink += bfft_inverse[(iter * 19) % n];
        }
        auto bfft_inverse_stop = std::chrono::high_resolution_clock::now();
        bfft_inverse_ns = std::chrono::duration<double, std::nano>(bfft_inverse_stop - bfft_inverse_start).count() / static_cast<double>(iterations);
    }

    if (sink == 123456789.0) std::printf("impossible sink %.9f\n", sink);

    fftw.destroy_plan(forward_plan);
    fftw.destroy_plan(inverse_plan);
    fftw.free_fn(fftw_input);
    fftw.free_fn(fftw_inverse);
    fftw.free_fn(fftw_spectrum);

    return BenchRow{n, size_case.label, "double", iterations, fftw_forward_ns, fftw_inverse_ns, fftw_roundtrip, bfft.status_for_double(), bfft_forward_ns, bfft_inverse_ns, bfft_error};
}

static BenchRow bench_float_case(const SizeCase& size_case, int forced_iterations, FFTWFloatApi& fftw, std::mt19937_64& rng) {
    const int n = size_case.n;
    const int bins = n / 2 + 1;
    int iterations = default_iterations(n);
    if (forced_iterations > 0) {
        iterations = forced_iterations;
    }
    const unsigned fftw_measure = 64U;

    float* fftw_input = static_cast<float*>(fftw.malloc_fn(sizeof(float) * n));
    float* fftw_inverse = static_cast<float*>(fftw.malloc_fn(sizeof(float) * n));
    fftwf_complex* fftw_spectrum = static_cast<fftwf_complex*>(fftw.malloc_fn(sizeof(fftwf_complex) * bins));

    fftwf_plan forward_plan = fftw.plan_r2c(n, fftw_input, fftw_spectrum, fftw_measure);
    fftwf_plan inverse_plan = fftw.plan_c2r(n, fftw_spectrum, fftw_inverse, fftw_measure);

    std::vector<float> original(n);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < n; ++i) {
        const float value = dist(rng);
        original[i] = value;
        fftw_input[i] = value;
    }

    fftw.execute(forward_plan);
    fftw.execute(inverse_plan);
    for (int i = 0; i < n; ++i) {
        fftw_inverse[i] /= static_cast<float>(n);
    }
    const double fftw_roundtrip = relative_l2_float(fftw_inverse, original.data(), n);

    volatile float sink = 0.0f;
    auto forward_start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        fftw_input[iter % n] += 1.0e-6f;
        fftw.execute(forward_plan);
        sink += fftw_spectrum[(iter * 17) % bins][0];
    }
    auto forward_stop = std::chrono::high_resolution_clock::now();

    auto inverse_start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        fftw_spectrum[(iter * 13) % bins][0] += 1.0e-6f;
        fftw.execute(inverse_plan);
        sink += fftw_inverse[(iter * 19) % n];
    }
    auto inverse_stop = std::chrono::high_resolution_clock::now();

    BfftRfftShim bfft(n);
    std::vector<std::array<float, 2>> bfft_output(bins);
    std::vector<float> bfft_work(n);
    const bool bfft_runs = bfft.forward_float(original.data(), reinterpret_cast<fftwf_complex*>(bfft_output.data()), bfft_work.data());
    double bfft_forward_ns = 0.0;
    double bfft_inverse_ns = 0.0;
    if (bfft_runs) {
        auto bfft_start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < iterations; ++iter) {
            original[iter % n] += 1.0e-6f;
            bfft.forward_float(original.data(), reinterpret_cast<fftwf_complex*>(bfft_output.data()), bfft_work.data());
            sink += bfft_output[(iter * 17) % bins][0];
        }
        auto bfft_stop = std::chrono::high_resolution_clock::now();
        bfft_forward_ns = std::chrono::duration<double, std::nano>(bfft_stop - bfft_start).count() / static_cast<double>(iterations);

        std::vector<float> bfft_inverse(n);
        auto bfft_inverse_start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < iterations; ++iter) {
            bfft_output[(iter * 13) % bins][0] += 1.0e-6f;
            bfft.inverse_float(reinterpret_cast<fftwf_complex*>(bfft_output.data()), bfft_inverse.data(), bfft_work.data());
            sink += bfft_inverse[(iter * 19) % n];
        }
        auto bfft_inverse_stop = std::chrono::high_resolution_clock::now();
        bfft_inverse_ns = std::chrono::duration<double, std::nano>(bfft_inverse_stop - bfft_inverse_start).count() / static_cast<double>(iterations);
    }

    if (sink == 123456.0f) std::printf("impossible sink %.9f\n", static_cast<double>(sink));

    fftw.destroy_plan(forward_plan);
    fftw.destroy_plan(inverse_plan);
    fftw.free_fn(fftw_input);
    fftw.free_fn(fftw_inverse);
    fftw.free_fn(fftw_spectrum);

    const double fftw_forward_ns = std::chrono::duration<double, std::nano>(forward_stop - forward_start).count() / static_cast<double>(iterations);
    const double fftw_inverse_ns = std::chrono::duration<double, std::nano>(inverse_stop - inverse_start).count() / static_cast<double>(iterations);

    return BenchRow{n, size_case.label, "float", iterations, fftw_forward_ns, fftw_inverse_ns, fftw_roundtrip, bfft.status_for_float(), bfft_forward_ns, bfft_inverse_ns, 0.0};
}

static void print_row(const BenchRow& row) {
    std::printf("%8d  %-16s  %-6s  %7d  %12.1f  %12.1f  %11.3e  %-15s",
                row.n, row.label, row.precision, row.iterations, row.fftw_rfft_ns, row.fftw_irfft_ns,
                row.fftw_roundtrip_rel_l2, row.bfft_status);
    if (row.bfft_rfft_ns > 0.0) {
        std::printf("  %12.1f  %12.1f  %11.3e\n", row.bfft_rfft_ns, row.bfft_irfft_ns, row.bfft_vs_fftw_rel_l2);
    } else {
        std::printf("  %12s  %12s  %11s\n", "n/a", "n/a", "n/a");
    }
}

} // namespace

int main(int argc, char** argv) {
    int max_n = 65536;
    int forced_iterations = 0;
    if (argc >= 2) max_n = std::atoi(argv[1]);
    if (argc >= 3) forced_iterations = std::atoi(argv[2]);

    FFTWDoubleApi fftw_double;
    FFTWFloatApi fftw_float;
    const bool have_double = fftw_double.load();
    const bool have_float = fftw_float.load();

    if (!have_double && !have_float) {
        std::fprintf(stderr, "FFTW double and float libraries were not found by dlopen; no benchmark can be timed.\n");
        return 1;
    }

    std::printf("Odd/prime/prime-power RFFT benchmark. FFTW is the timing baseline; bfft arbitrary-N uses the generalized Bruun (z^N-1 factorization) plan.\n");
    std::printf("%8s  %-16s  %-6s  %7s  %12s  %12s  %11s  %-15s  %12s  %12s  %11s\n",
                "N", "case", "prec", "iters", "FFTW_rfft", "FFTW_irfft", "FFTW_rt", "bfft_status", "bfft_rfft", "bfft_irfft", "bfft_err");

    std::mt19937_64 rng(42);
    const std::vector<SizeCase> cases = default_size_cases(max_n);
    for (const SizeCase& size_case : cases) {
        if (have_double) print_row(bench_double_case(size_case, forced_iterations, fftw_double, rng));
        if (have_float) print_row(bench_float_case(size_case, forced_iterations, fftw_float, rng));
    }

    return 0;
}
