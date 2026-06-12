#include <bfft/bfft.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <dlfcn.h>
#endif

#if defined(BFFT_COMPARE_WITH_IPP)
#include <ipps.h>
#endif

namespace {

using fftw_plan = void*;
using fftw_complex = double[2];

constexpr unsigned fftw_estimate_flag = 64U;
constexpr double pi = 3.141592653589793238462643383279502884;

struct ComplexValue {
    double re = 0.0;
    double im = 0.0;
};

struct FftwRuntime {
    void* lib = nullptr;
    fftw_plan (*plan_dft_r2c_1d)(int, double*, fftw_complex*, unsigned) = nullptr;
    void (*execute)(const fftw_plan) = nullptr;
    void (*destroy_plan)(fftw_plan) = nullptr;
    void* (*malloc_fn)(std::size_t) = nullptr;
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
            if (lib != nullptr) {
                break;
            }
        }
        if (lib == nullptr) {
            return false;
        }

        plan_dft_r2c_1d = reinterpret_cast<fftw_plan (*)(int, double*, fftw_complex*, unsigned)>(
            dlsym(lib, "fftw_plan_dft_r2c_1d"));
        execute = reinterpret_cast<void (*)(const fftw_plan)>(dlsym(lib, "fftw_execute"));
        destroy_plan = reinterpret_cast<void (*)(fftw_plan)>(dlsym(lib, "fftw_destroy_plan"));
        malloc_fn = reinterpret_cast<void* (*)(std::size_t)>(dlsym(lib, "fftw_malloc"));
        free_fn = reinterpret_cast<void (*)(void*)>(dlsym(lib, "fftw_free"));

        return plan_dft_r2c_1d != nullptr && execute != nullptr && destroy_plan != nullptr &&
               malloc_fn != nullptr && free_fn != nullptr;
#endif
    }

    ~FftwRuntime() {
#if !defined(_WIN32)
        if (lib != nullptr) {
            dlclose(lib);
        }
#endif
    }
};

std::size_t parse_size(const char* text, const char* name) {
    char* end = nullptr;
    unsigned long long value = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0') {
        throw std::runtime_error(std::string("bad ") + name);
    }
    return static_cast<std::size_t>(value);
}

bool is_power2(std::size_t n) {
    return n >= 4 && ((n & (n - 1)) == 0);
}

std::vector<double> make_input(std::size_t n) {
    std::vector<double> input(n);
    std::mt19937_64 rng(0xbff71042ULL ^ static_cast<unsigned long long>(n));
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (double& value : input) {
        value = dist(rng);
    }

    const std::size_t tone = std::max<std::size_t>(1, n / 16);
    for (std::size_t i = 0; i < n; ++i) {
        const double phase = 2.0 * pi * static_cast<double>(tone * i) / static_cast<double>(n);
        input[i] += 0.125 * std::cos(phase);
    }
    return input;
}

std::vector<ComplexValue> run_naive(const std::vector<double>& input) {
    const std::size_t n = input.size();
    std::vector<ComplexValue> output(n / 2 + 1);
    for (std::size_t k = 0; k <= n / 2; ++k) {
        double re = 0.0;
        double im = 0.0;
        for (std::size_t t = 0; t < n; ++t) {
            const double angle = -2.0 * pi * static_cast<double>(k * t) / static_cast<double>(n);
            re += input[t] * std::cos(angle);
            im += input[t] * std::sin(angle);
        }
        output[k].re = re;
        output[k].im = im;
    }
    return output;
}

std::vector<ComplexValue> run_bfft(const std::vector<double>& input) {
    bfft::plan plan(input.size());
    std::vector<bfft::complex> bfft_output(plan.bins());
    std::vector<bfft::complex> scratch(plan.native_scratch_size());
    std::vector<double> work(plan.work_size());
    plan.forward(input.data(), bfft_output.data(), work.data(), scratch.data());

    std::vector<ComplexValue> output(plan.bins());
    for (std::size_t i = 0; i < output.size(); ++i) {
        output[i].re = bfft_output[i].re;
        output[i].im = bfft_output[i].im;
    }
    return output;
}

bool run_fftw(FftwRuntime& fftw, const std::vector<double>& input, std::vector<ComplexValue>& output) {
    const std::size_t n = input.size();
    if (n > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return false;
    }

    double* fftw_input = static_cast<double*>(fftw.malloc_fn(sizeof(double) * n));
    fftw_complex* fftw_output = static_cast<fftw_complex*>(fftw.malloc_fn(sizeof(fftw_complex) * (n / 2 + 1)));
    if (fftw_input == nullptr || fftw_output == nullptr) {
        fftw.free_fn(fftw_input);
        fftw.free_fn(fftw_output);
        return false;
    }

    std::copy(input.begin(), input.end(), fftw_input);
    fftw_plan plan = fftw.plan_dft_r2c_1d(static_cast<int>(n), fftw_input, fftw_output, fftw_estimate_flag);
    if (plan == nullptr) {
        fftw.free_fn(fftw_input);
        fftw.free_fn(fftw_output);
        return false;
    }

    fftw.execute(plan);
    output.assign(n / 2 + 1, ComplexValue{});
    for (std::size_t i = 0; i < output.size(); ++i) {
        output[i].re = fftw_output[i][0];
        output[i].im = fftw_output[i][1];
    }

    fftw.destroy_plan(plan);
    fftw.free_fn(fftw_input);
    fftw.free_fn(fftw_output);
    return true;
}

#if defined(BFFT_COMPARE_WITH_IPP)
bool run_ipp(const std::vector<double>& input, std::vector<ComplexValue>& output) {
    const std::size_t n = input.size();
    if (n > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return false;
    }

    int spec_size = 0;
    int init_size = 0;
    int buffer_size = 0;
    IppStatus status = ippsDFTGetSize_C_64fc(static_cast<int>(n), IPP_FFT_NODIV_BY_ANY,
                                             ippAlgHintNone, &spec_size, &init_size, &buffer_size);
    if (status != ippStsNoErr) {
        return false;
    }

    std::vector<Ipp8u> spec_memory(static_cast<std::size_t>(spec_size));
    std::vector<Ipp8u> init_memory(static_cast<std::size_t>(init_size));
    std::vector<Ipp8u> buffer_memory(static_cast<std::size_t>(buffer_size));
    IppsDFTSpec_C_64fc* spec = reinterpret_cast<IppsDFTSpec_C_64fc*>(spec_memory.data());

    status = ippsDFTInit_C_64fc(static_cast<int>(n), IPP_FFT_NODIV_BY_ANY, ippAlgHintNone,
                                spec, init_memory.data());
    if (status != ippStsNoErr) {
        return false;
    }

    std::vector<Ipp64fc> ipp_input(n);
    std::vector<Ipp64fc> ipp_output(n);
    for (std::size_t i = 0; i < n; ++i) {
        ipp_input[i].re = input[i];
        ipp_input[i].im = 0.0;
    }

    status = ippsDFTFwd_CToC_64fc(ipp_input.data(), ipp_output.data(), spec, buffer_memory.data());
    if (status != ippStsNoErr) {
        return false;
    }

    output.assign(n / 2 + 1, ComplexValue{});
    for (std::size_t i = 0; i < output.size(); ++i) {
        output[i].re = ipp_output[i].re;
        output[i].im = ipp_output[i].im;
    }
    return true;
}
#endif

double max_relative_error(const std::vector<ComplexValue>& actual,
                          const std::vector<ComplexValue>& expected,
                          std::size_t n) {
    double max_abs = 0.0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const double dr = actual[i].re - expected[i].re;
        const double di = actual[i].im - expected[i].im;
        max_abs = std::max(max_abs, std::hypot(dr, di));
    }
    double scale = static_cast<double>(n);
    if (scale < 1.0) {
        scale = 1.0;
    }
    return max_abs / scale;
}

bool compare_one(const char* name,
                 const std::vector<ComplexValue>& actual,
                 const std::vector<ComplexValue>& expected,
                 std::size_t n,
                 double tolerance) {
    const double error = max_relative_error(actual, expected, n);
    std::printf("library=%s n=%zu max_rel=%.17g tolerance=%.17g\n", name, n, error, tolerance);
    if (error > tolerance) {
        std::fprintf(stderr, "%s n=%zu exceeded comparison tolerance\n", name, n);
        return false;
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::size_t max_log2 = 12;
        if (argc > 1) {
            max_log2 = parse_size(argv[1], "max_log2");
        }
        if (max_log2 < 2) {
            throw std::runtime_error("max_log2 must be at least 2");
        }

        FftwRuntime fftw;
        const bool fftw_available = fftw.load();
        if (fftw_available) {
            std::printf("library=fftw status=available\n");
        } else {
            std::printf("library=fftw status=missing\n");
        }

#if defined(BFFT_COMPARE_WITH_IPP)
        std::printf("library=ipp status=compiled\n");
#else
        std::printf("library=ipp status=not-compiled\n");
#endif

        bool ok = true;
        for (std::size_t log2_n = 2; log2_n <= max_log2; ++log2_n) {
            const std::size_t n = std::size_t{1} << log2_n;
            if (!is_power2(n)) {
                ok = false;
                continue;
            }

            const std::vector<double> input = make_input(n);
            const std::vector<ComplexValue> bfft_output = run_bfft(input);
            std::vector<ComplexValue> reference;

            if (fftw_available) {
                if (!run_fftw(fftw, input, reference)) {
                    std::fprintf(stderr, "fftw n=%zu failed at runtime; falling back to naive reference\n", n);
                    reference = run_naive(input);
                }
            } else {
                reference = run_naive(input);
            }

            if (!compare_one("bfft", bfft_output, reference, n, 1e-10)) {
                ok = false;
            }

#if defined(BFFT_COMPARE_WITH_IPP)
            std::vector<ComplexValue> ipp_output;
            if (run_ipp(input, ipp_output)) {
                if (!compare_one("ipp", ipp_output, reference, n, 1e-10)) {
                    ok = false;
                }
            } else {
                std::printf("library=ipp n=%zu status=runtime-unavailable\n", n);
            }
#endif
        }

        if (!ok) {
            return 1;
        }
        return 0;
    } catch (const std::exception& exc) {
        std::fprintf(stderr, "bfft_library_compare_probe failed: %s\n", exc.what());
        return 1;
    }
}
