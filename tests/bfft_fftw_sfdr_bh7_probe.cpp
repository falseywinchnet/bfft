#include <bfft/bfft.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
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

enum class WindowKind {
  rect,
  hann,
  bh7
};

enum class BfftMode {
  f64_standard,
  f64_native,
  f32_standard,
  f32_native
};

struct WindowSpec {
  WindowKind kind;
  const char* name;
  int main_radius;
};

struct BfftModeSpec {
  BfftMode mode;
  const char* name;
};

std::size_t parse_size(const char* text, const char* name) {
  char* end = nullptr;
  unsigned long long value = std::strtoull(text, &end, 10);
  if (end == text || *end != '\0') {
    throw std::runtime_error(std::string("bad ") + name);
  }
  return static_cast<std::size_t>(value);
}

WindowSpec parse_window(const char* text) {
  if (!text || std::strcmp(text, "rect") == 0 || std::strcmp(text, "rectangular") == 0) {
    return {WindowKind::rect, "rect", 0};
  }
  if (std::strcmp(text, "hann") == 0 || std::strcmp(text, "periodic-hann") == 0) {
    return {WindowKind::hann, "periodic-hann", 1};
  }
  if (std::strcmp(text, "bh7") == 0 ||
      std::strcmp(text, "blackman-harris-7") == 0 ||
      std::strcmp(text, "blackmanharris7") == 0) {
    return {WindowKind::bh7, "blackman-harris-7", 6};
  }
  throw std::runtime_error("bad window, expected rect, hann, or bh7");
}

BfftModeSpec parse_bfft_mode(const char* text) {
  if (!text || std::strcmp(text, "f64") == 0 || std::strcmp(text, "f64-standard") == 0 ||
      std::strcmp(text, "double") == 0 || std::strcmp(text, "standard") == 0) {
    return {BfftMode::f64_standard, "f64-standard"};
  }
  if (std::strcmp(text, "f64-native") == 0 || std::strcmp(text, "double-native") == 0 ||
      std::strcmp(text, "native") == 0) {
    return {BfftMode::f64_native, "f64-native"};
  }
  if (std::strcmp(text, "f32") == 0 || std::strcmp(text, "f32-standard") == 0 ||
      std::strcmp(text, "float") == 0 || std::strcmp(text, "float-standard") == 0) {
    return {BfftMode::f32_standard, "f32-standard"};
  }
  if (std::strcmp(text, "f32-native") == 0 || std::strcmp(text, "float-native") == 0 ||
      std::strcmp(text, "native-f32") == 0) {
    return {BfftMode::f32_native, "f32-native"};
  }
  throw std::runtime_error("bad bfft mode, expected f64-standard, f64-native, f32-standard, or f32-native");
}

bool is_native_mode(BfftMode mode) {
  return mode == BfftMode::f64_native || mode == BfftMode::f32_native;
}

bool is_f32_mode(BfftMode mode) {
  return mode == BfftMode::f32_standard || mode == BfftMode::f32_native;
}

double window_value(WindowKind kind, std::size_t i, std::size_t n) {
  const double theta = 2.0 * pi * static_cast<double>(i) / static_cast<double>(n);

  if (kind == WindowKind::rect) {
    return 1.0;
  }

  if (kind == WindowKind::hann) {
    return 0.5 - 0.5 * std::cos(theta);
  }

  // Seven-term Blackman-Harris cosine-sum window, periodic form.
  // w[n] = a0 - a1 cos(t) + a2 cos(2t) - ... + a6 cos(6t)
  // This gives coherent-bin main-lobe support k +/- 0..6 on the DFT grid.
  constexpr double a0 = 0.27105140069342;
  constexpr double a1 = 0.43329793923448;
  constexpr double a2 = 0.21812299954311;
  constexpr double a3 = 0.06592544638803;
  constexpr double a4 = 0.01081174209837;
  constexpr double a5 = 0.00077658482522;
  constexpr double a6 = 0.00001388721735;

  return a0
       - a1 * std::cos(theta)
       + a2 * std::cos(2.0 * theta)
       - a3 * std::cos(3.0 * theta)
       + a4 * std::cos(4.0 * theta)
       - a5 * std::cos(5.0 * theta)
       + a6 * std::cos(6.0 * theta);
}

std::vector<std::size_t> choose_bins(std::size_t n, std::size_t bins_per_size, int radius) {
  const std::size_t half = n / 2;
  if (half <= static_cast<std::size_t>(2 * radius + 1)) {
    return {};
  }

  const std::size_t first = static_cast<std::size_t>(radius + 1);
  const std::size_t last = half - static_cast<std::size_t>(radius + 1);
  if (last < first) {
    return {};
  }

  const std::size_t count = last - first + 1;
  std::vector<std::size_t> bins;

  if (bins_per_size == 0 || bins_per_size >= count) {
    bins.reserve(count);
    for (std::size_t k = first; k <= last; ++k) {
      bins.push_back(k);
    }
    return bins;
  }

  bins.reserve(bins_per_size);
  if (bins_per_size == 1) {
    bins.push_back((first + last) / 2);
    return bins;
  }

  for (std::size_t i = 0; i < bins_per_size; ++i) {
    const long double u =
      static_cast<long double>(i) * static_cast<long double>(count - 1) /
      static_cast<long double>(bins_per_size - 1);
    const std::size_t k = first + static_cast<std::size_t>(u + 0.5L);
    if (bins.empty() || bins.back() != k) {
      bins.push_back(k);
    }
  }

  return bins;
}

void make_tone(std::vector<double>& x, std::size_t k, bool sine, WindowKind window) {
  const std::size_t n = x.size();
  const double omega = 2.0 * pi * static_cast<double>(k) / static_cast<double>(n);

  for (std::size_t i = 0; i < n; ++i) {
    const double phase = omega * static_cast<double>(i);
    const double tone = sine ? std::sin(phase) : std::cos(phase);
    x[i] = tone * window_value(window, i, n);
  }
}

double hypot_complex(double re, double im) {
  return std::hypot(re, im);
}

double bin_abs(const bfft::complex& z) {
  return std::hypot(z.re, z.im);
}

double bin_abs(const bfft::complex_f32& z) {
  return std::hypot(static_cast<double>(z.re), static_cast<double>(z.im));
}

bool is_main_lobe(std::size_t bin, std::size_t k, int radius) {
  const std::size_t lo = k > static_cast<std::size_t>(radius) ? k - static_cast<std::size_t>(radius) : 0;
  const std::size_t hi = k + static_cast<std::size_t>(radius);
  return bin >= lo && bin <= hi;
}

struct SpectrumMetrics {
  double sfdr_db = std::numeric_limits<double>::infinity();
  double spur_dbc = -std::numeric_limits<double>::infinity();
  std::size_t spur_bin = 0;
  double spur_rel = 0.0;
  double rms_spur_rel = 0.0;
  double carrier = 0.0;
};

template <typename Complex>
SpectrumMetrics measure_bfft(const std::vector<Complex>& y, std::size_t k, int radius) {
  SpectrumMetrics m;
  m.carrier = bin_abs(y[k]);
  double max_spur = 0.0;
  long double spur_power = 0.0L;
  std::size_t spur_count = 0;

  for (std::size_t i = 0; i < y.size(); ++i) {
    if (is_main_lobe(i, k, radius)) {
      continue;
    }
    const double a = bin_abs(y[i]);
    if (a > max_spur) {
      max_spur = a;
      m.spur_bin = i;
    }
    spur_power += static_cast<long double>(a) * static_cast<long double>(a);
    ++spur_count;
  }

  if (m.carrier > 0.0) {
    m.spur_rel = max_spur / m.carrier;
    m.rms_spur_rel = spur_count ? std::sqrt(static_cast<double>(spur_power / static_cast<long double>(spur_count))) / m.carrier : 0.0;
    m.sfdr_db = max_spur > 0.0 ? 20.0 * std::log10(m.carrier / max_spur) : std::numeric_limits<double>::infinity();
    m.spur_dbc = max_spur > 0.0 ? 20.0 * std::log10(max_spur / m.carrier) : -std::numeric_limits<double>::infinity();
  }

  return m;
}

SpectrumMetrics measure_fftw(const fftw_complex* y, std::size_t bins, std::size_t k, int radius) {
  SpectrumMetrics m;
  m.carrier = hypot_complex(y[k][0], y[k][1]);
  double max_spur = 0.0;
  long double spur_power = 0.0L;
  std::size_t spur_count = 0;

  for (std::size_t i = 0; i < bins; ++i) {
    if (is_main_lobe(i, k, radius)) {
      continue;
    }
    const double a = hypot_complex(y[i][0], y[i][1]);
    if (a > max_spur) {
      max_spur = a;
      m.spur_bin = i;
    }
    spur_power += static_cast<long double>(a) * static_cast<long double>(a);
    ++spur_count;
  }

  if (m.carrier > 0.0) {
    m.spur_rel = max_spur / m.carrier;
    m.rms_spur_rel = spur_count ? std::sqrt(static_cast<double>(spur_power / static_cast<long double>(spur_count))) / m.carrier : 0.0;
    m.sfdr_db = max_spur > 0.0 ? 20.0 * std::log10(m.carrier / max_spur) : std::numeric_limits<double>::infinity();
    m.spur_dbc = max_spur > 0.0 ? 20.0 * std::log10(max_spur / m.carrier) : -std::numeric_limits<double>::infinity();
  }

  return m;
}


SpectrumMetrics measure_fftw(const fftwf_complex* y, std::size_t bins, std::size_t k, int radius) {
  SpectrumMetrics m;
  m.carrier = hypot_complex(y[k][0], y[k][1]);
  double max_spur = 0.0;
  long double spur_power = 0.0L;
  std::size_t spur_count = 0;

  for (std::size_t i = 0; i < bins; ++i) {
    if (is_main_lobe(i, k, radius)) {
      continue;
    }
    const double a = hypot_complex(y[i][0], y[i][1]);
    if (a > max_spur) {
      max_spur = a;
      m.spur_bin = i;
    }
    spur_power += static_cast<long double>(a) * static_cast<long double>(a);
    ++spur_count;
  }

  if (m.carrier > 0.0) {
    m.spur_rel = max_spur / m.carrier;
    m.rms_spur_rel = spur_count ? std::sqrt(static_cast<double>(spur_power / static_cast<long double>(spur_count))) / m.carrier : 0.0;
    m.sfdr_db = max_spur > 0.0 ? 20.0 * std::log10(m.carrier / max_spur) : std::numeric_limits<double>::infinity();
    m.spur_dbc = max_spur > 0.0 ? 20.0 * std::log10(max_spur / m.carrier) : -std::numeric_limits<double>::infinity();
  }

  return m;
}

template <typename Complex, typename RefComplex>
double max_bfft_vs_fftw_rel(const std::vector<Complex>& bfft_y,
                            const RefComplex* fftw_y,
                            std::size_t bins,
                            double denom) {
  if (denom <= 0.0) {
    denom = 1.0;
  }

  double worst = 0.0;
  for (std::size_t i = 0; i < bins; ++i) {
    const double dr = static_cast<double>(bfft_y[i].re) - fftw_y[i][0];
    const double di = static_cast<double>(bfft_y[i].im) - fftw_y[i][1];
    worst = std::max(worst, std::hypot(dr, di) / denom);
  }
  return worst;
}

const char* verdict(double delta_db, double mismatch_rel) {
  const double ad = std::fabs(delta_db);
  if (ad <= 1e-9 && mismatch_rel <= 1e-12) {
    return "same-as-fftw";
  }
  if (ad <= 1e-3 && mismatch_rel <= 1e-10) {
    return "excellent";
  }
  if (ad <= 0.1 && mismatch_rel <= 1e-8) {
    return "close";
  }
  return "different";
}

struct WorstRow {
  bool valid = false;
  std::size_t n = 0;
  std::string policy;
  std::string window;
  std::string bfft_mode;
  std::string fftw_precision;
  std::size_t transforms = 0;
  std::size_t k = 0;
  char wave = '?';
  SpectrumMetrics bfft;
  SpectrumMetrics fftw;
  double delta_db = 0.0;
  double mismatch_rel = 0.0;
};

WorstRow run_one_size(std::size_t n, std::size_t bins_per_size, WindowSpec window, BfftModeSpec bfft_mode, FFTW& fftw, FFTWf* fftwf) {
  const std::size_t bins = n / 2 + 1;
  const std::vector<std::size_t> ks = choose_bins(n, bins_per_size, window.main_radius);

  WorstRow worst;
  worst.n = n;
  worst.window = window.name;
  worst.bfft_mode = bfft_mode.name;
  const bool use_fftwf = is_f32_mode(bfft_mode.mode) && fftwf;
  worst.fftw_precision = use_fftwf ? "f32" : "f64";

  bfft::plan plan(n);
  if (is_native_mode(bfft_mode.mode)) {
    worst.policy = "native";
  } else {
    worst.policy = plan.standard_policy();
  }

  if (ks.empty()) {
    return worst;
  }

  std::vector<double> input(n);
  std::vector<double> work(plan.work_size());
  std::vector<float> input_f32(n);
  std::vector<float> work_f32(plan.work_size_f32());
  std::vector<bfft::complex> bfft_out(plan.bins());
  std::vector<bfft::complex> bfft_native(plan.bins());
  std::vector<bfft::complex_f32> bfft_out_f32(plan.bins());
  std::vector<bfft::complex_f32> bfft_native_f32(plan.bins());
  std::vector<bfft::complex> scratch(plan.native_scratch_size());
  std::vector<bfft::complex_f32> scratch_f32(plan.native_scratch_size());

  double* fftw_in = nullptr;
  fftw_complex* fftw_out = nullptr;
  float* fftwf_in = nullptr;
  fftwf_complex* fftwf_out = nullptr;
  fftw_plan fp = nullptr;

  if (use_fftwf) {
    fftwf_in = static_cast<float*>(fftwf->malloc_fn(sizeof(float) * n));
    fftwf_out = static_cast<fftwf_complex*>(fftwf->malloc_fn(sizeof(fftwf_complex) * bins));
    if (!fftwf_in || !fftwf_out) {
      if (fftwf_in) {
        fftwf->free_fn(fftwf_in);
      }
      if (fftwf_out) {
        fftwf->free_fn(fftwf_out);
      }
      throw std::runtime_error("fftwf allocation failed");
    }
    fp = fftwf->plan_dft_r2c_1d(static_cast<int>(n), fftwf_in, fftwf_out, FFTW_ESTIMATE_FLAG);
    if (!fp) {
      fftwf->free_fn(fftwf_in);
      fftwf->free_fn(fftwf_out);
      throw std::runtime_error("fftwf plan creation failed");
    }
  } else {
    fftw_in = static_cast<double*>(fftw.malloc_fn(sizeof(double) * n));
    fftw_out = static_cast<fftw_complex*>(fftw.malloc_fn(sizeof(fftw_complex) * bins));
    if (!fftw_in || !fftw_out) {
      if (fftw_in) {
        fftw.free_fn(fftw_in);
      }
      if (fftw_out) {
        fftw.free_fn(fftw_out);
      }
      throw std::runtime_error("fftw allocation failed");
    }
    fp = fftw.plan_dft_r2c_1d(static_cast<int>(n), fftw_in, fftw_out, FFTW_ESTIMATE_FLAG);
    if (!fp) {
      fftw.free_fn(fftw_in);
      fftw.free_fn(fftw_out);
      throw std::runtime_error("fftw plan creation failed");
    }
  }

  for (std::size_t k : ks) {
    for (int wave_i = 0; wave_i < 2; ++wave_i) {
      const bool sine = wave_i != 0;
      make_tone(input, k, sine, window.kind);

      if (use_fftwf) {
        for (std::size_t i = 0; i < n; ++i) {
          fftwf_in[i] = static_cast<float>(input[i]);
        }
        fftwf->execute(fp);
      } else {
        std::copy(input.begin(), input.end(), fftw_in);
        fftw.execute(fp);
      }

      SpectrumMetrics bm;
      double mismatch = 0.0;
      if (is_f32_mode(bfft_mode.mode)) {
        for (std::size_t i = 0; i < n; ++i) {
          input_f32[i] = static_cast<float>(input[i]);
        }
        if (bfft_mode.mode == BfftMode::f32_native) {
          plan.forward_native_f32(input_f32.data(), bfft_native_f32.data(), work_f32.data());
          plan.native_to_standard_f32(bfft_native_f32.data(), bfft_out_f32.data());
        } else {
          plan.forward_f32(input_f32.data(), bfft_out_f32.data(), work_f32.data(), scratch_f32.data());
        }
        bm = measure_bfft(bfft_out_f32, k, window.main_radius);
        const SpectrumMetrics fm_tmp = use_fftwf
          ? measure_fftw(fftwf_out, bins, k, window.main_radius)
          : measure_fftw(fftw_out, bins, k, window.main_radius);
        const double denom = std::max(bm.carrier, fm_tmp.carrier);
        mismatch = use_fftwf
          ? max_bfft_vs_fftw_rel(bfft_out_f32, fftwf_out, bins, denom)
          : max_bfft_vs_fftw_rel(bfft_out_f32, fftw_out, bins, denom);
      } else {
        if (bfft_mode.mode == BfftMode::f64_native) {
          plan.forward_native(input.data(), bfft_native.data(), work.data());
          plan.native_to_standard(bfft_native.data(), bfft_out.data());
        } else {
          plan.forward(input.data(), bfft_out.data(), work.data(), scratch.data());
        }
        bm = measure_bfft(bfft_out, k, window.main_radius);
        const SpectrumMetrics fm_tmp = measure_fftw(fftw_out, bins, k, window.main_radius);
        const double denom = std::max(bm.carrier, fm_tmp.carrier);
        mismatch = max_bfft_vs_fftw_rel(bfft_out, fftw_out, bins, denom);
      }

      const SpectrumMetrics fm = use_fftwf
        ? measure_fftw(fftwf_out, bins, k, window.main_radius)
        : measure_fftw(fftw_out, bins, k, window.main_radius);
      const double delta = bm.sfdr_db - fm.sfdr_db;

      ++worst.transforms;

      if (!worst.valid || bm.sfdr_db < worst.bfft.sfdr_db) {
        worst.valid = true;
        worst.k = k;
        worst.wave = sine ? 's' : 'c';
        worst.bfft = bm;
        worst.fftw = fm;
        worst.delta_db = delta;
        worst.mismatch_rel = mismatch;
      }
    }
  }

  if (use_fftwf) {
    fftwf->destroy_plan(fp);
    fftwf->free_fn(fftwf_in);
    fftwf->free_fn(fftwf_out);
  } else {
    fftw.destroy_plan(fp);
    fftw.free_fn(fftw_in);
    fftw.free_fn(fftw_out);
  }

  return worst;
}

void print_row(const WorstRow& r) {
  if (!r.valid) {
    std::printf("%zu,%s,%s,%s,%s,0,0,?,nan,nan,nan,nan,nan,0,0,nan,nan,nan,nan,nan,no-eligible-bins\n",
                r.n, r.policy.c_str(), r.window.c_str(), r.bfft_mode.c_str(), r.fftw_precision.c_str());
    return;
  }

  std::printf(
    "%zu,%s,%s,%s,%s,%zu,%zu,%c,%.8f,%.8f,%.8f,%.8f,%.8f,%zu,%zu,%.8e,%.8e,%.8e,%.8e,%.8e,%s\n",
    r.n,
    r.policy.c_str(),
    r.window.c_str(),
    r.bfft_mode.c_str(),
    r.fftw_precision.c_str(),
    r.transforms,
    r.k,
    r.wave,
    r.bfft.sfdr_db,
    r.fftw.sfdr_db,
    r.delta_db,
    r.bfft.spur_dbc,
    r.fftw.spur_dbc,
    r.bfft.spur_bin,
    r.fftw.spur_bin,
    r.bfft.spur_rel,
    r.fftw.spur_rel,
    r.bfft.rms_spur_rel,
    r.fftw.rms_spur_rel,
    r.mismatch_rel,
    verdict(r.delta_db, r.mismatch_rel));
}

} // namespace

int main(int argc, char** argv) {
  try {
    std::size_t max_pow = 22;
    std::size_t bins_per_size = 32;
    std::size_t min_pow = 2;
    WindowSpec window = parse_window("rect");
    BfftModeSpec bfft_mode = parse_bfft_mode("f64-standard");

    if (argc > 1) {
      max_pow = parse_size(argv[1], "max_pow");
    }
    if (argc > 2) {
      bins_per_size = parse_size(argv[2], "bins_per_size");
    }
    if (argc > 3) {
      min_pow = parse_size(argv[3], "min_pow");
    }
    if (argc > 4) {
      window = parse_window(argv[4]);
    }
    if (argc > 5) {
      bfft_mode = parse_bfft_mode(argv[5]);
    }

    if (min_pow < 2 || max_pow < min_pow || max_pow > 30) {
      throw std::runtime_error("expected 2 <= min_pow <= max_pow <= 30");
    }

    FFTW fftw;
    if (!fftw.load()) {
      std::fprintf(stderr, "could not load FFTW dynamically\n");
      return 2;
    }
    FFTWf fftwf;
    const bool have_fftwf = fftwf.load();
    if (is_f32_mode(bfft_mode.mode) && !have_fftwf) {
      std::fprintf(stderr, "could not load FFTWf dynamically; using double FFTW reference\n");
    }

    std::printf("# bfft vs fftw SFDR probe\n");
    std::printf("# bfft_version=%s backend=%s\n", bfft::version_string().c_str(), bfft::backend_name().c_str());
    std::printf("# args: max_pow=%zu bins_per_size=%zu min_pow=%zu window=%s bfft_mode=%s\n",
                max_pow, bins_per_size, min_pow, window.name, bfft_mode.name);
    std::printf("# rect excludes carrier only; hann excludes carrier +/-1; bh7 excludes carrier +/-6\n");
    std::printf("# bh7 is the periodic seven-term cosine-sum Blackman-Harris form\n");
    std::printf("# bins_per_size=0 means exhaustive over eligible k for every N\n");
    std::printf("# fftw_precision=%s\n", (is_f32_mode(bfft_mode.mode) && have_fftwf) ? "f32" : "f64");
    std::printf("N,policy,window,bfft_mode,fftw_precision,transforms,worst_k,wave,bfft_sfdr_db,fftw_sfdr_db,delta_db,bfft_spur_dbc,fftw_spur_dbc,bfft_spur_bin,fftw_spur_bin,bfft_spur_rel,fftw_spur_rel,bfft_rms_spur_rel,fftw_rms_spur_rel,bfft_vs_fftw_rel,verdict\n");

    for (std::size_t p = min_pow; p <= max_pow; ++p) {
      const std::size_t n = static_cast<std::size_t>(1) << p;
      if (n > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        break;
      }
      const WorstRow row = run_one_size(n, bins_per_size, window, bfft_mode, fftw, have_fftwf ? &fftwf : nullptr);
      print_row(row);
    }

    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
