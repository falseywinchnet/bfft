#include <bfft/bfft.hpp>

#include <algorithm>
#include <cmath>
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

constexpr unsigned FFTW_ESTIMATE_FLAG = 64U;
constexpr double pi = 3.141592653589793238462643383279502884;
constexpr double neg_inf_db = -9999.0;

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

    plan_dft_r2c_1d = reinterpret_cast<fftw_plan (*)(int, double*, fftw_complex*, unsigned)>(
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

enum class WindowKind {
  Rect,
  Hann
};

const char* window_name(WindowKind w) {
  return w == WindowKind::Hann ? "periodic-hann" : "rect";
}

std::size_t parse_size(const char* text, const char* name) {
  char* end = nullptr;
  unsigned long long value = std::strtoull(text, &end, 10);
  if (end == text || *end != '\0') {
    throw std::runtime_error(std::string("bad ") + name);
  }
  return static_cast<std::size_t>(value);
}

WindowKind parse_window(const char* text) {
  if (!text || std::strcmp(text, "rect") == 0 || std::strcmp(text, "none") == 0) {
    return WindowKind::Rect;
  }
  if (std::strcmp(text, "hann") == 0 || std::strcmp(text, "periodic-hann") == 0) {
    return WindowKind::Hann;
  }
  throw std::runtime_error("window must be rect or hann");
}

bool is_power2(std::size_t n) {
  return n >= 4 && ((n & (n - 1)) == 0);
}

std::vector<std::size_t> choose_bins(std::size_t n, std::size_t bins_per_size, WindowKind window) {
  const std::size_t first = window == WindowKind::Hann ? 2 : 1;
  const std::size_t last = window == WindowKind::Hann ? n / 2 - 2 : n / 2 - 1;
  std::vector<std::size_t> ks;
  if (last < first) return ks;

  const std::size_t count = last - first + 1;
  if (bins_per_size == 0 || bins_per_size >= count) {
    for (std::size_t k = first; k <= last; ++k) ks.push_back(k);
    return ks;
  }

  ks.push_back(first);
  if (last != first) ks.push_back(last);

  std::uint64_t state = 0x9e3779b97f4a7c15ULL ^ static_cast<std::uint64_t>(n);
  while (ks.size() < bins_per_size) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    const std::uint64_t r = state * 2685821657736338717ULL;
    const std::size_t k = first + static_cast<std::size_t>(r % count);
    if (std::find(ks.begin(), ks.end(), k) == ks.end()) ks.push_back(k);
  }

  std::sort(ks.begin(), ks.end());
  return ks;
}

void make_wave(std::vector<double>& x, std::size_t k, bool sine, WindowKind window) {
  const std::size_t n = x.size();
  const double scale = 2.0 * pi * static_cast<double>(k) / static_cast<double>(n);

  for (std::size_t t = 0; t < n; ++t) {
    const double a = scale * static_cast<double>(t);
    double v = sine ? std::sin(a) : std::cos(a);
    if (window == WindowKind::Hann) {
      const double w = 0.5 - 0.5 * std::cos(2.0 * pi * static_cast<double>(t) / static_cast<double>(n));
      v *= w;
    }
    x[t] = v;
  }
}

double mag(double re, double im) {
  return std::sqrt(re * re + im * im);
}

double db20(double x) {
  if (x <= 0.0) return neg_inf_db;
  return 20.0 * std::log10(x);
}

bool excluded_bin(std::size_t i, std::size_t k, WindowKind window) {
  if (i == k) return true;
  if (window == WindowKind::Hann) {
    if (k > 0 && i == k - 1) return true;
    if (i == k + 1) return true;
  }
  return false;
}

struct SfdrStats {
  double carrier = 0.0;
  double max_spur = 0.0;
  double spur_rel = 0.0;
  double sfdr_db = 0.0;
  double spur_dbc = 0.0;
  double spur_rms_rel = 0.0;
  std::size_t spur_bin = 0;
};

SfdrStats sfdr_bfft(const std::vector<bfft::complex>& out, std::size_t k, WindowKind window) {
  SfdrStats s;
  s.carrier = mag(out[k].re, out[k].im);

  double spur_power = 0.0;
  for (std::size_t i = 0; i < out.size(); ++i) {
    if (excluded_bin(i, k, window)) continue;
    const double m = mag(out[i].re, out[i].im);
    spur_power += m * m;
    if (m > s.max_spur) {
      s.max_spur = m;
      s.spur_bin = i;
    }
  }

  if (s.carrier > 0.0) {
    s.spur_rel = s.max_spur / s.carrier;
    s.spur_rms_rel = std::sqrt(spur_power) / s.carrier;
    s.spur_dbc = db20(s.spur_rel);
    s.sfdr_db = -s.spur_dbc;
  } else {
    s.spur_rel = std::numeric_limits<double>::infinity();
    s.spur_rms_rel = std::numeric_limits<double>::infinity();
    s.spur_dbc = std::numeric_limits<double>::infinity();
    s.sfdr_db = neg_inf_db;
  }

  return s;
}

SfdrStats sfdr_fftw(const fftw_complex* out, std::size_t bins, std::size_t k, WindowKind window) {
  SfdrStats s;
  s.carrier = mag(out[k][0], out[k][1]);

  double spur_power = 0.0;
  for (std::size_t i = 0; i < bins; ++i) {
    if (excluded_bin(i, k, window)) continue;
    const double m = mag(out[i][0], out[i][1]);
    spur_power += m * m;
    if (m > s.max_spur) {
      s.max_spur = m;
      s.spur_bin = i;
    }
  }

  if (s.carrier > 0.0) {
    s.spur_rel = s.max_spur / s.carrier;
    s.spur_rms_rel = std::sqrt(spur_power) / s.carrier;
    s.spur_dbc = db20(s.spur_rel);
    s.sfdr_db = -s.spur_dbc;
  } else {
    s.spur_rel = std::numeric_limits<double>::infinity();
    s.spur_rms_rel = std::numeric_limits<double>::infinity();
    s.spur_dbc = std::numeric_limits<double>::infinity();
    s.sfdr_db = neg_inf_db;
  }

  return s;
}

double bfft_vs_fftw_rel(const std::vector<bfft::complex>& b, const fftw_complex* f, std::size_t bins, double carrier) {
  if (carrier <= 0.0) return std::numeric_limits<double>::infinity();
  double max_abs = 0.0;
  for (std::size_t i = 0; i < bins; ++i) {
    const double dr = b[i].re - f[i][0];
    const double di = b[i].im - f[i][1];
    max_abs = std::max(max_abs, mag(dr, di));
  }
  return max_abs / carrier;
}

const char* verdict(double bfft_sfdr, double delta_db, double bfft_vs_fftw) {
  if (bfft_vs_fftw > 1e-10 && delta_db < -20.0) return "bfft-weaker";
  if (std::fabs(delta_db) < 0.01 && bfft_vs_fftw < 1e-12) return "same-as-fftw";
  if (bfft_sfdr > 180.0) return "excellent";
  if (bfft_sfdr > 140.0) return "good";
  if (bfft_sfdr > 100.0) return "moderate";
  return "bad";
}

struct Worst {
  std::size_t k = 0;
  char wave = '?';
  SfdrStats bfft;
  SfdrStats fftw;
  double delta_db = 0.0;
  double bfft_vs_fftw = 0.0;
};

Worst run_one_size(std::size_t n, std::size_t bins_per_size, WindowKind window, FFTW& fftw) {
  if (!is_power2(n)) {
    throw std::runtime_error("N must be a power of two and at least 4");
  }
  if (n > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("N exceeds FFTW int plan API limit");
  }

  const std::size_t bins = n / 2 + 1;
  bfft::plan plan(n);
  std::vector<double> input(n);
  std::vector<double> work(plan.work_size());
  std::vector<bfft::complex> bout(plan.bins());
  std::vector<bfft::complex> scratch(plan.native_scratch_size());

  double* fin = static_cast<double*>(fftw.malloc_fn(sizeof(double) * n));
  fftw_complex* fout = static_cast<fftw_complex*>(fftw.malloc_fn(sizeof(fftw_complex) * bins));
  if (!fin || !fout) {
    if (fin) fftw.free_fn(fin);
    if (fout) fftw.free_fn(fout);
    throw std::runtime_error("fftw allocation failed");
  }

  fftw_plan fp = fftw.plan_dft_r2c_1d(static_cast<int>(n), fin, fout, FFTW_ESTIMATE_FLAG);
  if (!fp) {
    fftw.free_fn(fin);
    fftw.free_fn(fout);
    throw std::runtime_error("fftw plan failed");
  }

  Worst worst;
  worst.bfft.sfdr_db = std::numeric_limits<double>::infinity();
  const auto ks = choose_bins(n, bins_per_size, window);

  for (std::size_t k : ks) {
    for (int s = 0; s < 2; ++s) {
      const bool sine = s != 0;
      make_wave(input, k, sine, window);

      plan.forward(input.data(), bout.data(), work.data(), scratch.data());
      std::copy(input.begin(), input.end(), fin);
      fftw.execute(fp);

      const SfdrStats bs = sfdr_bfft(bout, k, window);
      const SfdrStats fs = sfdr_fftw(fout, bins, k, window);
      const double cmp = bfft_vs_fftw_rel(bout, fout, bins, std::max(bs.carrier, fs.carrier));
      const double delta = bs.sfdr_db - fs.sfdr_db;

      const bool lower_sfdr = bs.sfdr_db < worst.bfft.sfdr_db;
      const bool same_sfdr_worse_cmp = std::fabs(bs.sfdr_db - worst.bfft.sfdr_db) < 1e-12 && cmp > worst.bfft_vs_fftw;
      if (lower_sfdr || same_sfdr_worse_cmp) {
        worst.k = k;
        worst.wave = sine ? 's' : 'c';
        worst.bfft = bs;
        worst.fftw = fs;
        worst.delta_db = delta;
        worst.bfft_vs_fftw = cmp;
      }
    }
  }

  fftw.destroy_plan(fp);
  fftw.free_fn(fin);
  fftw.free_fn(fout);

  return worst;
}

} // namespace

int main(int argc, char** argv) {
  try {
    const std::size_t max_pow = argc > 1 ? parse_size(argv[1], "max_pow") : 22;
    const std::size_t bins_per_size = argc > 2 ? parse_size(argv[2], "bins_per_size") : 32;
    const std::size_t min_pow = argc > 3 ? parse_size(argv[3], "min_pow") : 2;
    const WindowKind window = argc > 4 ? parse_window(argv[4]) : WindowKind::Rect;

    if (min_pow > max_pow || max_pow >= 63) {
      throw std::runtime_error("bad exponent range");
    }

    FFTW fftw;
    if (!fftw.load()) {
      std::fprintf(stderr, "could not load FFTW dynamic library\n");
      return 2;
    }

    std::printf("# bfft vs fftw SFDR probe\n");
    std::printf("# bfft_version=%s backend=%s\n", bfft::version_string().c_str(), bfft::backend_name().c_str());
    std::printf("# args: max_pow=%zu bins_per_size=%zu min_pow=%zu window=%s\n", max_pow, bins_per_size, min_pow, window_name(window));
    std::printf("# rectangular mode excludes only the carrier bin; periodic-hann mode excludes carrier and +/-1 main-lobe bins\n");
    std::printf("# bins_per_size=0 means exhaustive over eligible k for every N\n");
    std::printf("N,policy,window,transforms,worst_k,wave,bfft_sfdr_db,fftw_sfdr_db,delta_db,bfft_spur_dbc,fftw_spur_dbc,bfft_spur_bin,fftw_spur_bin,bfft_spur_rel,fftw_spur_rel,bfft_rms_spur_rel,fftw_rms_spur_rel,bfft_vs_fftw_rel,verdict\n");

    for (std::size_t p = min_pow; p <= max_pow; ++p) {
      const std::size_t n = std::size_t{1} << p;
      bfft::plan plan_for_policy(n);
      const auto ks = choose_bins(n, bins_per_size, window);
      if (ks.empty()) continue;
      const std::size_t transforms = ks.size() * 2;
      const Worst w = run_one_size(n, bins_per_size, window, fftw);
      std::printf(
        "%zu,%s,%s,%zu,%zu,%c,%.8f,%.8f,%.8f,%.8f,%.8f,%zu,%zu,%.8e,%.8e,%.8e,%.8e,%.8e,%s\n",
        n,
        plan_for_policy.standard_policy().c_str(),
        window_name(window),
        transforms,
        w.k,
        w.wave,
        w.bfft.sfdr_db,
        w.fftw.sfdr_db,
        w.delta_db,
        w.bfft.spur_dbc,
        w.fftw.spur_dbc,
        w.bfft.spur_bin,
        w.fftw.spur_bin,
        w.bfft.spur_rel,
        w.fftw.spur_rel,
        w.bfft.spur_rms_rel,
        w.fftw.spur_rms_rel,
        w.bfft_vs_fftw,
        verdict(w.bfft.sfdr_db, w.delta_db, w.bfft_vs_fftw));
    }

    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    std::fprintf(stderr, "usage: %s [max_pow=22] [bins_per_size=32] [min_pow=2] [rect|hann]\n", argv[0]);
    return 1;
  }
}
