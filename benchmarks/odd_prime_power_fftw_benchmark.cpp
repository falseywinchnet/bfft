// Odd and prime arbitrary-N RFFT benchmark.
//
// Output intentionally mirrors examples/benchmark.cpp so power-of-two and
// non-power-of-two runs can be compared column-for-column.
//
// Build:
//   g++ -O3 -std=c++17 -Iinclude benchmarks/odd_prime_power_fftw_benchmark.cpp build/libbfft.a -ldl -lm -o odd_prime_power_fftw_bench
// Run:
//   ./odd_prime_power_fftw_bench              # primes near powers of two
//   ./odd_prime_power_fftw_bench 4096         # odd N where 2048 < N < 4096
//   ./odd_prime_power_fftw_bench 3000 32      # one arbitrary non-power-of-two N

#include <bfft/bfft.hpp>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <random>
#include <stdexcept>
#include <vector>

#if !defined(_WIN32)
#include <dlfcn.h>
#endif

namespace {

using fftw_plan = void*;
using fftw_complex = double[2];
using fftwf_complex = float[2];

constexpr unsigned FFTW_ESTIMATE_FLAG = 64U;
constexpr double pi = 3.141592653589793238462643383279502884;

struct SizeCase {
  std::size_t n;
  const char* label;
};

struct FFTW {
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
    for (int i = 0; names[i]; ++i) {
      lib = dlopen(names[i], RTLD_NOW);
      if (lib) break;
    }
    if (!lib) return false;
    plan_dft_r2c_1d =
      reinterpret_cast<fftw_plan (*)(int, double*, fftw_complex*, unsigned)>(
        dlsym(lib, "fftw_plan_dft_r2c_1d"));
    execute = reinterpret_cast<void (*)(const fftw_plan)>(dlsym(lib, "fftw_execute"));
    destroy_plan = reinterpret_cast<void (*)(fftw_plan)>(dlsym(lib, "fftw_destroy_plan"));
    malloc_fn = reinterpret_cast<void* (*)(std::size_t)>(dlsym(lib, "fftw_malloc"));
    free_fn = reinterpret_cast<void (*)(void*)>(dlsym(lib, "fftw_free"));
    return plan_dft_r2c_1d && execute && destroy_plan && malloc_fn && free_fn;
#endif
  }

  ~FFTW() {
#if !defined(_WIN32)
    if (lib) dlclose(lib);
#endif
  }
};

struct FFTWf {
  void* lib = nullptr;
  fftw_plan (*plan_dft_r2c_1d)(int, float*, fftwf_complex*, unsigned) = nullptr;
  void (*execute)(const fftw_plan) = nullptr;
  void (*destroy_plan)(fftw_plan) = nullptr;
  void* (*malloc_fn)(std::size_t) = nullptr;
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
    for (int i = 0; names[i]; ++i) {
      lib = dlopen(names[i], RTLD_NOW);
      if (lib) break;
    }
    if (!lib) return false;
    plan_dft_r2c_1d =
      reinterpret_cast<fftw_plan (*)(int, float*, fftwf_complex*, unsigned)>(
        dlsym(lib, "fftwf_plan_dft_r2c_1d"));
    execute = reinterpret_cast<void (*)(const fftw_plan)>(dlsym(lib, "fftwf_execute"));
    destroy_plan = reinterpret_cast<void (*)(fftw_plan)>(dlsym(lib, "fftwf_destroy_plan"));
    malloc_fn = reinterpret_cast<void* (*)(std::size_t)>(dlsym(lib, "fftwf_malloc"));
    free_fn = reinterpret_cast<void (*)(void*)>(dlsym(lib, "fftwf_free"));
    return plan_dft_r2c_1d && execute && destroy_plan && malloc_fn && free_fn;
#endif
  }

  ~FFTWf() {
#if !defined(_WIN32)
    if (lib) dlclose(lib);
#endif
  }
};

bool is_power2(std::size_t n) {
  return n > 0 && ((n & (n - 1)) == 0);
}

bool is_prime(std::size_t n) {
  if (n < 2) return false;
  if (n == 2) return true;
  if ((n & 1U) == 0) return false;
  const std::size_t limit = static_cast<std::size_t>(std::sqrt(static_cast<double>(n)));
  for (std::size_t d = 3; d <= limit; d += 2) {
    if (n % d == 0) return false;
  }
  return true;
}

std::size_t previous_prime(std::size_t n) {
  while (n >= 2) {
    if (is_prime(n)) return n;
    --n;
  }
  throw std::runtime_error("no previous prime");
}

std::size_t next_prime(std::size_t n) {
  if (n < 2) n = 2;
  while (!is_prime(n)) ++n;
  return n;
}

int default_iters(std::size_t n) {
  const long long target = 200000LL;
  int it = static_cast<int>(target / static_cast<long long>(std::max<std::size_t>(1, n)));
  if (is_prime(n) && n > 257) it = std::min(it, 16);
  if (it < 2) it = 2;
  if (it > 2000) it = 2000;
  return it;
}

std::size_t parse_size(const char* text) {
  char* end = nullptr;
  const unsigned long long value = std::strtoull(text, &end, 10);
  if (end == text || *end != '\0' || value < 2) {
    throw std::runtime_error("bad size");
  }
  return static_cast<std::size_t>(value);
}

int parse_iters(const char* text) {
  char* end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (end == text || *end != '\0' || value < 0) {
    throw std::runtime_error("bad iteration count");
  }
  return static_cast<int>(value);
}

std::vector<double> make_input(std::size_t n, std::mt19937_64& rng) {
  std::vector<double> input(n);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (std::size_t i = 0; i < n; ++i) {
    const double t = static_cast<double>(i) / static_cast<double>(n);
    input[i] = 0.70 * std::sin(2.0 * pi * 3.0 * t)
             + 0.20 * std::cos(2.0 * pi * 11.0 * t)
             + 0.07 * std::sin(2.0 * pi * 29.0 * t)
             + 0.03 * dist(rng);
  }
  return input;
}

double max_abs_complex(const std::vector<bfft::complex>& a,
                       const std::vector<bfft::complex>& b) {
  double e = 0.0;
  const std::size_t n = std::min(a.size(), b.size());
  if (n == 0) return 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    e += std::abs(a[i].re - b[i].re) + std::abs(a[i].im - b[i].im);
  }
  return e / static_cast<double>(n);
}

double max_abs_complex_f32(const std::vector<bfft::complex_f32>& a,
                           const std::vector<bfft::complex_f32>& b) {
  double e = 0.0;
  const std::size_t n = std::min(a.size(), b.size());
  if (n == 0) return 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    e += std::abs(static_cast<double>(a[i].re - b[i].re))
       + std::abs(static_cast<double>(a[i].im - b[i].im));
  }
  return e / static_cast<double>(n);
}

double max_abs_real(const std::vector<double>& a, const std::vector<double>& b) {
  double e = 0.0;
  const std::size_t n = std::min(a.size(), b.size());
  for (std::size_t i = 0; i < n; ++i) {
    e = std::max(e, std::abs(a[i] - b[i]));
  }
  return e;
}

template <class Fn>
double bench_ns(int iters, Fn&& fn) {
  for (int i = 0; i < 3; ++i) fn();
  const auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) fn();
  const auto t1 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::nano>(t1 - t0).count()
       / static_cast<double>(iters);
}

void print_ns(double value) {
  if (std::isnan(value)) std::printf("%11s ", "n/a");
  else std::printf("%11.1f ", value);
}

struct Result {
  double fftw_ns = NAN;
  double fftwf_ns = NAN;
  double pffft_ns = NAN;
  double mkl_ns = NAN;
  double mklf_ns = NAN;
  double native_ns = NAN;
  double standard_ns = NAN;
  double native_f32_ns = NAN;
  double standard_f32_ns = NAN;
  double inverse_ns = NAN;
  double roundtrip_ns = NAN;
  double standard_over_fftw = NAN;
  double standard_f32_over_fftwf = NAN;
  double standard_over_pffft = NAN;
  double standard_f32_over_pffft = NAN;
  double standard_over_mkl = NAN;
  double standard_f32_over_mklf = NAN;
  double fftw_err = NAN;
  double fftwf_err = NAN;
  double mkl_err = NAN;
  double mklf_err = NAN;
  double roundtrip_err = NAN;
  double sink = 0.0;
};

Result bench_one(const SizeCase& size_case, int forced_iters, FFTW* fftw, FFTWf* fftwf, std::mt19937_64& rng) {
  const std::size_t n = size_case.n;
  const std::size_t nb = n / 2 + 1;
  const int iters = forced_iters > 0 ? forced_iters : default_iters(n);

  bfft::plan plan(n);
  std::vector<double> input = make_input(n, rng);
  std::vector<double> original = input;
  std::vector<double> work(plan.work_size());
  std::vector<float> work_f32(plan.work_size_f32());
  std::vector<bfft::complex> scratch(plan.native_scratch_size());
  std::vector<bfft::complex_f32> scratch_f32(plan.native_scratch_size());
  std::vector<bfft::complex> standard(plan.bins());
  std::vector<bfft::complex> native(plan.bins());
  std::vector<bfft::complex> reference(plan.bins());
  std::vector<bfft::complex_f32> standard_f32(plan.bins());
  std::vector<bfft::complex_f32> native_f32(plan.bins());
  std::vector<bfft::complex_f32> reference_f32(plan.bins());
  std::vector<double> inverse_out(n);
  Result result;

  fftw_plan fp = nullptr;
  double* fftw_in = nullptr;
  fftw_complex* fftw_out = nullptr;
  if (fftw && fftw->lib && n <= static_cast<std::size_t>(INT_MAX)) {
    fftw_in = static_cast<double*>(fftw->malloc_fn(sizeof(double) * n));
    fftw_out = static_cast<fftw_complex*>(fftw->malloc_fn(sizeof(fftw_complex) * nb));
    if (fftw_in && fftw_out) {
      std::copy(input.begin(), input.end(), fftw_in);
      fp = fftw->plan_dft_r2c_1d(static_cast<int>(n), fftw_in, fftw_out, FFTW_ESTIMATE_FLAG);
    }
  }

  if (fp) {
    result.fftw_ns = bench_ns(iters, [&] {
      input[static_cast<std::size_t>(result.sink) % n] += 1e-12;
      std::copy(input.begin(), input.end(), fftw_in);
      fftw->execute(fp);
      result.sink += fftw_out[(static_cast<std::size_t>(result.sink) * 17) % nb][0];
    });
    std::copy(original.begin(), original.end(), fftw_in);
    fftw->execute(fp);
    for (std::size_t i = 0; i < nb; ++i) {
      reference[i].re = fftw_out[i][0];
      reference[i].im = fftw_out[i][1];
    }
  }

  fftw_plan fpf = nullptr;
  float* fftwf_in = nullptr;
  fftwf_complex* fftwf_out = nullptr;
  std::vector<float> input_f32(n);
  for (std::size_t i = 0; i < n; ++i) input_f32[i] = static_cast<float>(original[i]);
  if (fftwf && fftwf->lib && n <= static_cast<std::size_t>(INT_MAX)) {
    fftwf_in = static_cast<float*>(fftwf->malloc_fn(sizeof(float) * n));
    fftwf_out = static_cast<fftwf_complex*>(fftwf->malloc_fn(sizeof(fftwf_complex) * nb));
    if (fftwf_in && fftwf_out) {
      std::copy(input_f32.begin(), input_f32.end(), fftwf_in);
      fpf = fftwf->plan_dft_r2c_1d(static_cast<int>(n), fftwf_in, fftwf_out, FFTW_ESTIMATE_FLAG);
    }
  }
  if (fpf) {
    result.fftwf_ns = bench_ns(iters, [&] {
      input_f32[static_cast<std::size_t>(result.sink) % n] += 1e-7f;
      std::copy(input_f32.begin(), input_f32.end(), fftwf_in);
      fftwf->execute(fpf);
      result.sink += fftwf_out[(static_cast<std::size_t>(result.sink) * 17) % nb][0];
    });
    for (std::size_t i = 0; i < n; ++i) input_f32[i] = static_cast<float>(original[i]);
    std::copy(input_f32.begin(), input_f32.end(), fftwf_in);
    fftwf->execute(fpf);
    for (std::size_t i = 0; i < nb; ++i) {
      reference_f32[i].re = fftwf_out[i][0];
      reference_f32[i].im = fftwf_out[i][1];
    }
  }

  input = original;
  result.native_ns = bench_ns(iters, [&] {
    input[static_cast<std::size_t>(result.sink) % n] += 1e-12;
    plan.forward_native(input.data(), native.data(), work.data());
    result.sink += native[(static_cast<std::size_t>(result.sink) * 17) % nb].re;
  });

  input = original;
  result.standard_ns = bench_ns(iters, [&] {
    input[static_cast<std::size_t>(result.sink) % n] += 1e-12;
    plan.forward(input.data(), standard.data(), work.data(), scratch.data());
    result.sink += standard[(static_cast<std::size_t>(result.sink) * 17) % nb].re;
  });

  for (std::size_t i = 0; i < n; ++i) input_f32[i] = static_cast<float>(original[i]);
  result.native_f32_ns = bench_ns(iters, [&] {
    input_f32[static_cast<std::size_t>(result.sink) % n] += 1e-7f;
    plan.forward_native_f32(input_f32.data(), native_f32.data(), work_f32.data());
    result.sink += native_f32[(static_cast<std::size_t>(result.sink) * 17) % nb].re;
  });

  for (std::size_t i = 0; i < n; ++i) input_f32[i] = static_cast<float>(original[i]);
  result.standard_f32_ns = bench_ns(iters, [&] {
    input_f32[static_cast<std::size_t>(result.sink) % n] += 1e-7f;
    plan.forward_f32(input_f32.data(), standard_f32.data(), work_f32.data(), scratch_f32.data());
    result.sink += standard_f32[(static_cast<std::size_t>(result.sink) * 17) % nb].re;
  });

  plan.forward(original.data(), standard.data(), work.data(), scratch.data());
  result.inverse_ns = bench_ns(iters, [&] {
    standard[(static_cast<std::size_t>(result.sink) * 17) % nb].re += 1e-12;
    plan.inverse(standard.data(), inverse_out.data());
    result.sink += inverse_out[(static_cast<std::size_t>(result.sink) * 17) % n];
  });

  input = original;
  result.roundtrip_ns = bench_ns(iters, [&] {
    input[static_cast<std::size_t>(result.sink) % n] += 1e-12;
    plan.forward(input.data(), standard.data(), work.data(), scratch.data());
    plan.inverse(standard.data(), inverse_out.data());
    result.sink += inverse_out[(static_cast<std::size_t>(result.sink) * 17) % n];
  });

  plan.forward(original.data(), standard.data(), work.data(), scratch.data());
  for (std::size_t i = 0; i < n; ++i) input_f32[i] = static_cast<float>(original[i]);
  plan.forward_f32(input_f32.data(), standard_f32.data(), work_f32.data(), scratch_f32.data());
  plan.inverse(standard.data(), inverse_out.data());
  if (fp) result.fftw_err = max_abs_complex(standard, reference);
  if (fpf) result.fftwf_err = max_abs_complex_f32(standard_f32, reference_f32);
  result.roundtrip_err = max_abs_real(original, inverse_out);
  if (fp && result.fftw_ns > 0.0) result.standard_over_fftw = result.standard_ns / result.fftw_ns;
  if (fpf && result.fftwf_ns > 0.0) result.standard_f32_over_fftwf = result.standard_f32_ns / result.fftwf_ns;

  if (fp) fftw->destroy_plan(fp);
  if (fpf) fftwf->destroy_plan(fpf);
  if (fftw && fftw_in) fftw->free_fn(fftw_in);
  if (fftw && fftw_out) fftw->free_fn(fftw_out);
  if (fftwf && fftwf_in) fftwf->free_fn(fftwf_in);
  if (fftwf && fftwf_out) fftwf->free_fn(fftwf_out);

  std::printf("%8zu %8d ", n, iters);
  print_ns(result.fftw_ns);
  print_ns(result.fftwf_ns);
  print_ns(result.pffft_ns);
  print_ns(result.mkl_ns);
  print_ns(result.mklf_ns);
  print_ns(result.native_ns);
  print_ns(result.standard_ns);
  print_ns(result.native_f32_ns);
  print_ns(result.standard_f32_ns);
  print_ns(result.inverse_ns);
  print_ns(result.roundtrip_ns);

  if (std::isnan(result.standard_over_fftw)) std::printf("%8s ", "n/a");
  else std::printf("%8.3f ", result.standard_over_fftw);
  if (std::isnan(result.standard_f32_over_fftwf)) std::printf("%8s ", "n/a");
  else std::printf("%8.3f ", result.standard_f32_over_fftwf);
  std::printf("%8s %8s %8s %8s ", "n/a", "n/a", "n/a", "n/a");

  if (std::isnan(result.fftw_err)) std::printf("err64 %9s ", "n/a");
  else std::printf("err64 %.1e ", result.fftw_err);
  if (std::isnan(result.fftwf_err)) std::printf("err32 %9s ", "n/a");
  else std::printf("err32 %.1e ", result.fftwf_err);
  std::printf("mkl64 %9s mkl32 %9s ", "n/a", "n/a");
  std::printf("rt %.1e sink %.8g  %s\n", result.roundtrip_err, result.sink, size_case.label);

  return result;
}

std::vector<SizeCase> default_prime_cases() {
  std::vector<SizeCase> cases;
  for (int power = 5; power <= 11; ++power) {
    const std::size_t pivot = std::size_t{1} << power;
    cases.push_back(SizeCase{previous_prime(pivot - 1), "prime-below-pow2"});
    cases.push_back(SizeCase{next_prime(pivot + 1), "prime-above-pow2"});
  }
  std::sort(cases.begin(), cases.end(), [](const SizeCase& a, const SizeCase& b) {
    return a.n < b.n;
  });
  cases.erase(std::unique(cases.begin(), cases.end(), [](const SizeCase& a, const SizeCase& b) {
    return a.n == b.n;
  }), cases.end());
  return cases;
}

std::vector<SizeCase> interval_cases(std::size_t upper_power2) {
  if (!is_power2(upper_power2) || upper_power2 < 8) {
    throw std::runtime_error("interval sweep N must be a power of two >= 8");
  }
  const std::size_t lower = upper_power2 >> 1;
  std::vector<SizeCase> cases;
  for (std::size_t n = lower + 1; n < upper_power2; n += 2) {
    cases.push_back(SizeCase{n, is_prime(n) ? "odd-prime-interval" : "odd-interval"});
  }
  return cases;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::size_t requested_n = 0;
    int forced_iters = 0;
    bool requested_mkl = false;
    int positional_count = 0;
    for (int i = 1; i < argc; ++i) {
      if (std::strcmp(argv[i], "--intel-mkl") == 0 || std::strcmp(argv[i], "--mkl") == 0) {
        requested_mkl = true;
        continue;
      } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
        std::printf("usage: %s [--intel-mkl] [N [iters]]\n", argv[0]);
        std::printf("  --intel-mkl accepted for CLI parity; MKL columns are n/a in this benchmark.\n");
        std::printf("  no N:       sweep primes adjacent to powers of two.\n");
        std::printf("  pow2 N:     sweep odd non-power-of-two sizes where N/2 < size < N.\n");
        std::printf("  nonpow2 N:  benchmark that single arbitrary-N size.\n");
        return 0;
      }
      if (positional_count == 0) {
        requested_n = parse_size(argv[i]);
        ++positional_count;
      } else if (positional_count == 1) {
        forced_iters = parse_iters(argv[i]);
        ++positional_count;
      } else {
        std::fprintf(stderr, "usage: %s [--intel-mkl] [N [iters]]\n", argv[0]);
        return 2;
      }
    }

    FFTW fftw;
    const bool have_fftw = fftw.load();
    FFTWf fftwf;
    const bool have_fftwf = fftwf.load();

    std::printf("BFFT non-power-of-two RFFT benchmark. backend: %s, version: %s\n",
                bfft::backend_name().c_str(),
                bfft::version_string().c_str());
    std::printf("standard, native-order, and f32 GenBruun-compatible public paths are measured.\n");
    std::printf("FFTW uses FFTW_ESTIMATE-style flag 64 for cheap large plans.\n");
    if (!have_fftw) std::fprintf(stderr, "FFTW not found by dlopen; FFTW column unavailable.\n");
    if (!have_fftwf) std::fprintf(stderr, "FFTWf not found by dlopen; FFTWf column unavailable.\n");
    std::printf("PFFFT disabled for this arbitrary-N benchmark.\n");
    if (requested_mkl) {
      std::printf("Intel oneMKL requested; MKL columns are n/a in this arbitrary-N benchmark.\n");
    } else {
      std::printf("Intel oneMKL disabled for this arbitrary-N benchmark.\n");
    }

    std::printf("%8s %8s %11s %11s %11s %11s %11s %11s %11s %11s %11s %11s %11s %8s %8s %8s %8s %8s %8s  %s\n",
                "N",
                "iters",
                "FFTW_ns",
                "FFTWf_ns",
                "PFFFT_ns",
                "MKL64_ns",
                "MKL32_ns",
                "Native_ns",
                "Std_ns",
                "F32Nat_ns",
                "F32Std_ns",
                "Inv_ns",
                "RT_ns",
                "S/F64",
                "F32/Ff",
                "S/P",
                "F32/P",
                "S/MKL",
                "F32/M",
                "checks");

    std::vector<SizeCase> cases;
    if (requested_n == 0) {
      cases = default_prime_cases();
    } else if (is_power2(requested_n)) {
      cases = interval_cases(requested_n);
    } else {
      cases.push_back(SizeCase{requested_n, is_prime(requested_n) ? "single-prime" : "single-nonpow2"});
    }

    FFTW* fftw_ptr = have_fftw ? &fftw : nullptr;
    FFTWf* fftwf_ptr = have_fftwf ? &fftwf : nullptr;
    std::mt19937_64 rng(42);
    for (const SizeCase& size_case : cases) {
      bench_one(size_case, forced_iters, fftw_ptr, fftwf_ptr, rng);
    }

    return 0;
  } catch (const std::exception& exc) {
    std::fprintf(stderr, "benchmark failed: %s\n", exc.what());
    return 1;
  }
}
