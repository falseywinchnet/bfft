#include <bfft/bfft.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
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

#if defined(BFFT_WITH_PFFFT)
#include <pffft.h>
#endif

namespace {

using fftw_plan = void*;
using fftw_complex = double[2];
using fftwf_complex = float[2];

constexpr unsigned FFTW_ESTIMATE_FLAG = 64U;
constexpr double pi = 3.141592653589793238462643383279502884;

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
      if (lib) {
        break;
      }
    }

    if (!lib) {
      return false;
    }

    plan_dft_r2c_1d =
      reinterpret_cast<fftw_plan (*)(int, double*, fftw_complex*, unsigned)>(
        dlsym(lib, "fftw_plan_dft_r2c_1d"));
    execute =
      reinterpret_cast<void (*)(const fftw_plan)>(
        dlsym(lib, "fftw_execute"));
    destroy_plan =
      reinterpret_cast<void (*)(fftw_plan)>(
        dlsym(lib, "fftw_destroy_plan"));
    malloc_fn =
      reinterpret_cast<void* (*)(std::size_t)>(
        dlsym(lib, "fftw_malloc"));
    free_fn =
      reinterpret_cast<void (*)(void*)>(
        dlsym(lib, "fftw_free"));

    return plan_dft_r2c_1d && execute && destroy_plan && malloc_fn && free_fn;
#endif
  }

  ~FFTW() {
#if !defined(_WIN32)
    if (lib) {
      dlclose(lib);
    }
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
      if (lib) {
        break;
      }
    }

    if (!lib) {
      return false;
    }

    plan_dft_r2c_1d =
      reinterpret_cast<fftw_plan (*)(int, float*, fftwf_complex*, unsigned)>(
        dlsym(lib, "fftwf_plan_dft_r2c_1d"));
    execute =
      reinterpret_cast<void (*)(const fftw_plan)>(
        dlsym(lib, "fftwf_execute"));
    destroy_plan =
      reinterpret_cast<void (*)(fftw_plan)>(
        dlsym(lib, "fftwf_destroy_plan"));
    malloc_fn =
      reinterpret_cast<void* (*)(std::size_t)>(
        dlsym(lib, "fftwf_malloc"));
    free_fn =
      reinterpret_cast<void (*)(void*)>(
        dlsym(lib, "fftwf_free"));

    return plan_dft_r2c_1d && execute && destroy_plan && malloc_fn && free_fn;
#endif
  }

  ~FFTWf() {
#if !defined(_WIN32)
    if (lib) {
      dlclose(lib);
    }
#endif
  }
};

bool is_power2(std::size_t n) {
  return n > 0 && ((n & (n - 1)) == 0);
}

int ilog2_pow2(std::size_t n) {
  int l = 0;
  while (n > 1) {
    n >>= 1;
    ++l;
  }
  return l;
}

int default_iters(std::size_t n) {
  const long long target = 180000000LL;
  const int l = std::max(1, ilog2_pow2(n));
  int it = static_cast<int>(target / (static_cast<long long>(n) * l));
  if (it < 16) {
    it = 16;
  }
  if (it > 200000) {
    it = 200000;
  }
  return it;
}

std::size_t parse_size(const char* text) {
  char* end = nullptr;
  unsigned long long value = std::strtoull(text, &end, 10);
  if (end == text || *end != '\0' || value == 0) {
    throw std::runtime_error("bad size");
  }
  return static_cast<std::size_t>(value);
}

int parse_iters(const char* text) {
  char* end = nullptr;
  long value = std::strtol(text, &end, 10);
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

  for (std::size_t i = 0; i < n; ++i) {
    const double dr = a[i].re - b[i].re;
    const double di = a[i].im - b[i].im;
    e = std::max(e, std::sqrt(dr * dr + di * di));
  }

  return e;
}


double max_abs_complex_f32(const std::vector<bfft::complex_f32>& a,
                           const std::vector<bfft::complex_f32>& b) {
  double e = 0.0;
  const std::size_t n = std::min(a.size(), b.size());

  for (std::size_t i = 0; i < n; ++i) {
    const double dr = static_cast<double>(a[i].re) - static_cast<double>(b[i].re);
    const double di = static_cast<double>(a[i].im) - static_cast<double>(b[i].im);
    e = std::max(e, std::sqrt(dr * dr + di * di));
  }

  return e;
}

double max_abs_real(const std::vector<double>& a,
                    const std::vector<double>& b) {
  double e = 0.0;
  const std::size_t n = std::min(a.size(), b.size());

  for (std::size_t i = 0; i < n; ++i) {
    e = std::max(e, std::abs(a[i] - b[i]));
  }

  return e;
}

template <class Fn>
double bench_ns(int iters, Fn&& fn) {
  const int warmups = 3;
  for (int i = 0; i < warmups; ++i) {
    fn();
  }

  const auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) {
    fn();
  }
  const auto t1 = std::chrono::high_resolution_clock::now();

  return std::chrono::duration<double, std::nano>(t1 - t0).count()
       / static_cast<double>(iters);
}

void print_ns(double value) {
  if (std::isnan(value)) {
    std::printf("%11s ", "n/a");
  } else {
    std::printf("%11.1f ", value);
  }
}

struct Result {
  double fftw_ns = NAN;
  double fftwf_ns = NAN;
  double pffft_ns = NAN;
  double native_ns = NAN;
  double standard_ns = NAN;
  double native_f32_ns = NAN;
  double standard_f32_ns = NAN;
  double roundtrip_ns = NAN;
  double standard_over_fftw = NAN;
  double standard_f32_over_fftwf = NAN;
  double standard_over_pffft = NAN;
  double standard_f32_over_pffft = NAN;
  double fftw_err = NAN;
  double fftwf_err = NAN;
  double roundtrip_err = NAN;
  double sink = 0.0;
};

Result bench_one(std::size_t n, int forced_iters, FFTW* fftw, FFTWf* fftwf, std::mt19937_64& rng) {
  const std::size_t nb = n / 2 + 1;
  const int iters = forced_iters > 0 ? forced_iters : default_iters(n);

  Result result;

  bfft::plan plan(n);

  std::vector<double> input = make_input(n, rng);
  std::vector<double> original = input;
  std::vector<float> input_f32(n);
  std::vector<float> original_f32(n);
  for (std::size_t i = 0; i < n; ++i) {
    original_f32[i] = static_cast<float>(original[i]);
  }
  std::vector<double> work(plan.work_size());
  std::vector<float> work_f32(plan.work_size_f32());
  std::vector<double> inverse_out(n);
  std::vector<bfft::complex> native(plan.bins());
  std::vector<bfft::complex> standard(plan.bins());
  std::vector<bfft::complex> scratch(plan.native_scratch_size());
  std::vector<bfft::complex> reference(plan.bins());
  std::vector<bfft::complex_f32> native_f32(plan.bins());
  std::vector<bfft::complex_f32> standard_f32(plan.bins());
  std::vector<bfft::complex_f32> scratch_f32(plan.native_scratch_size());
  std::vector<bfft::complex_f32> reference_f32(plan.bins());

  fftw_plan fp = nullptr;
  double* fftw_in = nullptr;
  fftw_complex* fftw_out = nullptr;
  fftw_plan fpf = nullptr;
  float* fftwf_in = nullptr;
  fftwf_complex* fftwf_out = nullptr;

  if (fftw && fftw->lib && n <= static_cast<std::size_t>(INT32_MAX)) {
    fftw_in = static_cast<double*>(fftw->malloc_fn(sizeof(double) * n));
    fftw_out = static_cast<fftw_complex*>(fftw->malloc_fn(sizeof(fftw_complex) * nb));

    if (fftw_in && fftw_out) {
      std::copy(input.begin(), input.end(), fftw_in);
      fp = fftw->plan_dft_r2c_1d(static_cast<int>(n), fftw_in, fftw_out, FFTW_ESTIMATE_FLAG);
    }
  }

  if (fp) {
    result.fftw_ns = bench_ns(iters, [&] {
      input[static_cast<std::size_t>(result.sink) & (n - 1)] += 1e-12;
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


  if (fftwf && fftwf->lib && n <= static_cast<std::size_t>(INT32_MAX)) {
    fftwf_in = static_cast<float*>(fftwf->malloc_fn(sizeof(float) * n));
    fftwf_out = static_cast<fftwf_complex*>(fftwf->malloc_fn(sizeof(fftwf_complex) * nb));

    if (fftwf_in && fftwf_out) {
      std::copy(original_f32.begin(), original_f32.end(), fftwf_in);
      fpf = fftwf->plan_dft_r2c_1d(static_cast<int>(n), fftwf_in, fftwf_out, FFTW_ESTIMATE_FLAG);
    }
  }

  if (fpf) {
    input_f32 = original_f32;
    result.fftwf_ns = bench_ns(iters, [&] {
      input_f32[static_cast<std::size_t>(result.sink) & (n - 1)] += 1e-7f;
      std::copy(input_f32.begin(), input_f32.end(), fftwf_in);
      fftwf->execute(fpf);
      result.sink += fftwf_out[(static_cast<std::size_t>(result.sink) * 17) % nb][0];
    });

    std::copy(original_f32.begin(), original_f32.end(), fftwf_in);
    fftwf->execute(fpf);

    for (std::size_t i = 0; i < nb; ++i) {
      reference_f32[i].re = fftwf_out[i][0];
      reference_f32[i].im = fftwf_out[i][1];
    }
  }

#if defined(BFFT_WITH_PFFFT)
  if (n >= 32 && n <= static_cast<std::size_t>(INT32_MAX)) {
    PFFFT_Setup* setup = pffft_new_setup(static_cast<int>(n), PFFFT_REAL);
    if (setup) {
      std::vector<float> p_in(n);
      std::vector<float> p_out(n);
      std::vector<float> p_work(n);

      for (std::size_t i = 0; i < n; ++i) {
        p_in[i] = static_cast<float>(original[i]);
      }

      result.pffft_ns = bench_ns(iters, [&] {
        p_in[static_cast<std::size_t>(result.sink) & (n - 1)] += 1e-7f;
        pffft_transform_ordered(setup, p_in.data(), p_out.data(), p_work.data(), PFFFT_FORWARD);
        result.sink += p_out[(static_cast<std::size_t>(result.sink) * 17) % n];
      });

      pffft_destroy_setup(setup);
    }
  }
#endif

  input = original;
  result.native_ns = bench_ns(iters, [&] {
    input[static_cast<std::size_t>(result.sink) & (n - 1)] += 1e-12;
    plan.forward_native(input.data(), native.data(), work.data());
    result.sink += native[(static_cast<std::size_t>(result.sink) * 17) % nb].re;
  });

  input = original;
  result.standard_ns = bench_ns(iters, [&] {
    input[static_cast<std::size_t>(result.sink) & (n - 1)] += 1e-12;
    plan.forward(input.data(), standard.data(), work.data(), scratch.data());
    result.sink += standard[(static_cast<std::size_t>(result.sink) * 17) % nb].re;
  });


  input_f32 = original_f32;
  result.native_f32_ns = bench_ns(iters, [&] {
    input_f32[static_cast<std::size_t>(result.sink) & (n - 1)] += 1e-7f;
    plan.forward_native_f32(input_f32.data(), native_f32.data(), work_f32.data());
    result.sink += native_f32[(static_cast<std::size_t>(result.sink) * 17) % nb].re;
  });

  input_f32 = original_f32;
  result.standard_f32_ns = bench_ns(iters, [&] {
    input_f32[static_cast<std::size_t>(result.sink) & (n - 1)] += 1e-7f;
    plan.forward_f32(input_f32.data(), standard_f32.data(), work_f32.data(), scratch_f32.data());
    result.sink += standard_f32[(static_cast<std::size_t>(result.sink) * 17) % nb].re;
  });

  input = original;
  result.roundtrip_ns = bench_ns(iters, [&] {
    input[static_cast<std::size_t>(result.sink) & (n - 1)] += 1e-12;
    plan.forward(input.data(), standard.data(), work.data(), scratch.data());
    plan.inverse(standard.data(), inverse_out.data());
    result.sink += inverse_out[(static_cast<std::size_t>(result.sink) * 17) % n];
  });

  plan.forward(original.data(), standard.data(), work.data(), scratch.data());
  plan.inverse(standard.data(), inverse_out.data());

  if (fp) {
    result.fftw_err = max_abs_complex(standard, reference);
  }

  plan.forward_f32(original_f32.data(), standard_f32.data(), work_f32.data(), scratch_f32.data());
  if (fpf) {
    result.fftwf_err = max_abs_complex_f32(standard_f32, reference_f32);
  }

  result.roundtrip_err = max_abs_real(original, inverse_out);

  if (fp && result.fftw_ns > 0.0) {
    result.standard_over_fftw = result.standard_ns / result.fftw_ns;
  }

  if (fpf && result.fftwf_ns > 0.0) {
    result.standard_f32_over_fftwf = result.standard_f32_ns / result.fftwf_ns;
  }

  if (!std::isnan(result.pffft_ns) && result.pffft_ns > 0.0) {
    result.standard_over_pffft = result.standard_ns / result.pffft_ns;
    result.standard_f32_over_pffft = result.standard_f32_ns / result.pffft_ns;
  }

  if (fp) {
    fftw->destroy_plan(fp);
  }
  if (fpf) {
    fftwf->destroy_plan(fpf);
  }
  if (fftw && fftw_in) {
    fftw->free_fn(fftw_in);
  }
  if (fftw && fftw_out) {
    fftw->free_fn(fftw_out);
  }
  if (fftwf && fftwf_in) {
    fftwf->free_fn(fftwf_in);
  }
  if (fftwf && fftwf_out) {
    fftwf->free_fn(fftwf_out);
  }

  std::printf("%8zu %8d ", n, iters);
  print_ns(result.fftw_ns);
  print_ns(result.fftwf_ns);
  print_ns(result.pffft_ns);
  print_ns(result.native_ns);
  print_ns(result.standard_ns);
  print_ns(result.native_f32_ns);
  print_ns(result.standard_f32_ns);
  print_ns(result.roundtrip_ns);

  if (std::isnan(result.standard_over_fftw)) {
    std::printf("%8s ", "n/a");
  } else {
    std::printf("%8.3f ", result.standard_over_fftw);
  }

  if (std::isnan(result.standard_f32_over_fftwf)) {
    std::printf("%8s ", "n/a");
  } else {
    std::printf("%8.3f ", result.standard_f32_over_fftwf);
  }

  if (std::isnan(result.standard_over_pffft)) {
    std::printf("%8s ", "n/a");
  } else {
    std::printf("%8.3f ", result.standard_over_pffft);
  }

  if (std::isnan(result.standard_f32_over_pffft)) {
    std::printf("%8s ", "n/a");
  } else {
    std::printf("%8.3f ", result.standard_f32_over_pffft);
  }

  if (std::isnan(result.fftw_err)) {
    std::printf("err64 %9s ", "n/a");
  } else {
    std::printf("err64 %.1e ", result.fftw_err);
  }

  if (std::isnan(result.fftwf_err)) {
    std::printf("err32 %9s ", "n/a");
  } else {
    std::printf("err32 %.1e ", result.fftwf_err);
  }

  std::printf("rt %.1e sink %.8g\n", result.roundtrip_err, result.sink);

  return result;
}

} // namespace

int main(int argc, char** argv) {
  try {
    std::size_t forced_n = 0;
    int forced_iters = 0;

    if (argc >= 2) {
      forced_n = parse_size(argv[1]);
    }
    if (argc >= 3) {
      forced_iters = parse_iters(argv[2]);
    }
    if (argc > 3) {
      std::fprintf(stderr, "usage: %s [N [iters]]\n", argv[0]);
      return 2;
    }

    FFTW fftw;
    const bool have_fftw = fftw.load();
    FFTWf fftwf;
    const bool have_fftwf = fftwf.load();

    std::printf("BFFT power-of-two RFFT benchmark. backend: %s, version: %s\n",
                bfft::backend_name().c_str(),
                bfft::version_string().c_str());
    std::printf("standard policy is plan-dependent; FFTW uses FFTW_ESTIMATE-style flag 64 for cheap large plans.\n");

    if (!have_fftw) {
      std::fprintf(stderr, "FFTW not found by dlopen; FFTW column unavailable.\n");
    }
    if (!have_fftwf) {
      std::fprintf(stderr, "FFTWf not found by dlopen; FFTWf/f32 ratio columns unavailable.\n");
    }

#if defined(BFFT_WITH_PFFFT)
    std::printf("PFFFT enabled at compile time.\n");
#else
    std::printf("PFFFT disabled; compile with -DBFFT_WITH_PFFFT and link pffft to enable.\n");
#endif

    std::printf("%8s %8s %11s %11s %11s %11s %11s %11s %11s %11s %8s %8s %8s %8s  %s\n",
                "N",
                "iters",
                "FFTW_ns",
                "FFTWf_ns",
                "PFFFT_ns",
                "Native_ns",
                "Std_ns",
                "F32Nat_ns",
                "F32Std_ns",
                "RT_ns",
                "S/F64",
                "F32/Ff",
                "S/P",
                "F32/P",
                "checks");

    std::mt19937_64 rng(42);

    if (forced_n > 0) {
      if (!is_power2(forced_n) || forced_n < 4) {
        std::fprintf(stderr, "N must be a power of two >= 4.\n");
        return 2;
      }
      bench_one(forced_n, forced_iters, have_fftw ? &fftw : nullptr, have_fftwf ? &fftwf : nullptr, rng);
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
      1048576,
      2097152,
      4194304,
      8388608,
      16777216,
      33554432,
      67108864,
      134217728
    };

    for (std::size_t n : sizes) {
      bench_one(n, forced_iters, have_fftw ? &fftw : nullptr, have_fftwf ? &fftwf : nullptr, rng);
    }

    return 0;
  } catch (const std::exception& exc) {
    std::fprintf(stderr, "benchmark failed: %s\n", exc.what());
    return 1;
  }
}
