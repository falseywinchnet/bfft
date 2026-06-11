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

std::size_t parse_size(const char* text, const char* name) {
  char* end = nullptr;
  unsigned long long value = std::strtoull(text, &end, 10);
  if (end == text || *end != '\0') {
    throw std::runtime_error(std::string("bad ") + name);
  }
  return static_cast<std::size_t>(value);
}

void make_wave(std::vector<double>& x, std::size_t k, bool sine) {
  const std::size_t n = x.size();
  const double scale = 2.0 * pi * static_cast<double>(k) / static_cast<double>(n);
  for (std::size_t t = 0; t < n; ++t) {
    const double a = scale * static_cast<double>(t);
    x[t] = sine ? std::sin(a) : std::cos(a);
  }
}

struct ComplexView {
  double re = 0.0;
  double im = 0.0;
};

struct SupportStats {
  double support_rel = 0.0;
  double leak_rel = 0.0;
  double onbin_rel = 0.0;
  double offbin_rel = 0.0;
};

SupportStats support_bfft(const std::vector<bfft::complex>& out, std::size_t k, bool sine, std::size_t n) {
  const double expected_mag = 0.5 * static_cast<double>(n);
  const double expected_re = sine ? 0.0 : expected_mag;
  const double expected_im = sine ? -expected_mag : 0.0;

  double max_off_mag = 0.0;
  double leak_power = 0.0;
  for (std::size_t i = 0; i < out.size(); ++i) {
    if (i == k) continue;
    const double mag2 = out[i].re * out[i].re + out[i].im * out[i].im;
    leak_power += mag2;
    max_off_mag = std::max(max_off_mag, std::sqrt(mag2));
  }

  const double dre = out[k].re - expected_re;
  const double dim = out[k].im - expected_im;
  const double onbin_abs = std::sqrt(dre * dre + dim * dim);
  const double leak_abs = std::sqrt(leak_power);

  SupportStats s;
  s.leak_rel = leak_abs / expected_mag;
  s.onbin_rel = onbin_abs / expected_mag;
  s.offbin_rel = max_off_mag / expected_mag;
  s.support_rel = std::sqrt(s.leak_rel * s.leak_rel + s.onbin_rel * s.onbin_rel);
  return s;
}

SupportStats support_fftw(const fftw_complex* out, std::size_t bins, std::size_t k, bool sine, std::size_t n) {
  const double expected_mag = 0.5 * static_cast<double>(n);
  const double expected_re = sine ? 0.0 : expected_mag;
  const double expected_im = sine ? -expected_mag : 0.0;

  double max_off_mag = 0.0;
  double leak_power = 0.0;
  for (std::size_t i = 0; i < bins; ++i) {
    if (i == k) continue;
    const double re = out[i][0];
    const double im = out[i][1];
    const double mag2 = re * re + im * im;
    leak_power += mag2;
    max_off_mag = std::max(max_off_mag, std::sqrt(mag2));
  }

  const double dre = out[k][0] - expected_re;
  const double dim = out[k][1] - expected_im;
  const double onbin_abs = std::sqrt(dre * dre + dim * dim);
  const double leak_abs = std::sqrt(leak_power);

  SupportStats s;
  s.leak_rel = leak_abs / expected_mag;
  s.onbin_rel = onbin_abs / expected_mag;
  s.offbin_rel = max_off_mag / expected_mag;
  s.support_rel = std::sqrt(s.leak_rel * s.leak_rel + s.onbin_rel * s.onbin_rel);
  return s;
}

double bfft_vs_fftw_rel(const std::vector<bfft::complex>& b, const fftw_complex* f, std::size_t bins, std::size_t n) {
  const double denom = 0.5 * static_cast<double>(n);
  double max_abs = 0.0;
  for (std::size_t i = 0; i < bins; ++i) {
    const double dr = b[i].re - f[i][0];
    const double di = b[i].im - f[i][1];
    max_abs = std::max(max_abs, std::sqrt(dr * dr + di * di));
  }
  return max_abs / denom;
}

std::vector<std::size_t> choose_bins(std::size_t n, std::size_t bins_per_size) {
  const std::size_t last = n / 2 - 1;
  std::vector<std::size_t> ks;
  if (last < 1) return ks;

  if (bins_per_size == 0 || bins_per_size >= last) {
    for (std::size_t k = 1; k <= last; ++k) ks.push_back(k);
    return ks;
  }

  ks.push_back(1);
  if (last > 1) ks.push_back(last);

  std::uint64_t state = 0x9e3779b97f4a7c15ULL ^ static_cast<std::uint64_t>(n);
  while (ks.size() < bins_per_size) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    const std::uint64_t r = state * 2685821657736338717ULL;
    const std::size_t k = 1 + static_cast<std::size_t>(r % last);
    if (std::find(ks.begin(), ks.end(), k) == ks.end()) ks.push_back(k);
  }

  std::sort(ks.begin(), ks.end());
  return ks;
}

const char* verdict(double bfft_rel, double fftw_rel, double ratio) {
  (void)fftw_rel;
  if (bfft_rel < 1e-11) return "roundoff";
  if (bfft_rel < 1e-9) return "small";
  if (bfft_rel < 1e-7) return ratio > 100.0 ? "weak-vs-ref" : "moderate";
  return "bad";
}

struct Worst {
  std::size_t k = 0;
  char wave = '?';
  SupportStats bfft;
  SupportStats fftw;
  double bfft_vs_fftw = 0.0;
  double ratio = 0.0;
};

Worst run_one_size(std::size_t n, std::size_t bins_per_size, FFTW& fftw) {
  if (n < 4 || (n & (n - 1)) != 0) {
    throw std::runtime_error("N must be a power of two and at least 4");
  }
  if (n > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("N exceeds FFTW int plan API limit");
  }

  bfft::plan bp(n);
  const std::size_t bins = bp.bins();
  std::vector<double> input(n);
  std::vector<double> work(bp.work_size());
  std::vector<bfft::complex> bfft_out(bins);
  std::vector<bfft::complex> scratch(bp.native_scratch_size());

  double* fftw_in = static_cast<double*>(fftw.malloc_fn(sizeof(double) * n));
  fftw_complex* fftw_out = static_cast<fftw_complex*>(fftw.malloc_fn(sizeof(fftw_complex) * bins));
  if (!fftw_in || !fftw_out) throw std::runtime_error("fftw allocation failed");

  fftw_plan fp = fftw.plan_dft_r2c_1d(static_cast<int>(n), fftw_in, fftw_out, FFTW_ESTIMATE_FLAG);
  if (!fp) throw std::runtime_error("fftw plan creation failed");

  Worst worst;
  const std::vector<std::size_t> ks = choose_bins(n, bins_per_size);
  for (std::size_t k : ks) {
    for (int s = 0; s < 2; ++s) {
      const bool sine = s != 0;
      make_wave(input, k, sine);
      std::copy(input.begin(), input.end(), fftw_in);

      bp.forward(input.data(), bfft_out.data(), work.data(), scratch.data());
      fftw.execute(fp);

      const SupportStats bs = support_bfft(bfft_out, k, sine, n);
      const SupportStats fs = support_fftw(fftw_out, bins, k, sine, n);
      const double cmp = bfft_vs_fftw_rel(bfft_out, fftw_out, bins, n);
      const double ratio = bs.support_rel / std::max(fs.support_rel, 1e-300);

      if (bs.support_rel > worst.bfft.support_rel) {
        worst.k = k;
        worst.wave = sine ? 's' : 'c';
        worst.bfft = bs;
        worst.fftw = fs;
        worst.bfft_vs_fftw = cmp;
        worst.ratio = ratio;
      }
    }
  }

  fftw.destroy_plan(fp);
  fftw.free_fn(fftw_in);
  fftw.free_fn(fftw_out);
  return worst;
}

} // namespace

int main(int argc, char** argv) {
  try {
    const std::size_t max_pow = argc > 1 ? parse_size(argv[1], "max_pow") : 22;
    const std::size_t bins_per_size = argc > 2 ? parse_size(argv[2], "bins_per_size") : 32;
    const std::size_t min_pow = argc > 3 ? parse_size(argv[3], "min_pow") : 2;

    FFTW fftw;
    if (!fftw.load()) {
      std::fprintf(stderr, "could not load FFTW3 dynamically; install fftw or adjust library paths\n");
      return 2;
    }

    std::printf("# bfft vs fftw invariant-support probe\n");
    std::printf("# bfft_version=%s backend=%s\n", bfft::version_string().c_str(), bfft::backend_name().c_str());
    std::printf("# args: max_pow=%zu bins_per_size=%zu min_pow=%zu\n", max_pow, bins_per_size, min_pow);
    std::printf("# bins_per_size=0 means exhaustive over k=1..N/2-1 for every N\n");
    std::printf("N,policy,transforms,worst_k,wave,bfft_support_rel,fftw_support_rel,bfft_over_fftw,bfft_vs_fftw_rel,bfft_leak_rel,fftw_leak_rel,bfft_onbin_rel,fftw_onbin_rel,bfft_offbin_rel,fftw_offbin_rel,verdict\n");

    for (std::size_t p = min_pow; p <= max_pow; ++p) {
      const std::size_t n = static_cast<std::size_t>(1) << p;
      bfft::plan plan(n);
      const std::size_t eligible = n / 2 > 1 ? n / 2 - 1 : 0;
      const std::size_t k_count = bins_per_size == 0 ? eligible : std::min(bins_per_size, eligible);
      const std::size_t transforms = 2 * k_count;
      Worst w = run_one_size(n, bins_per_size, fftw);

      std::printf("%zu,%s,%zu,%zu,%c,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%s\n",
        n,
        plan.standard_policy().c_str(),
        transforms,
        w.k,
        w.wave,
        w.bfft.support_rel,
        w.fftw.support_rel,
        w.ratio,
        w.bfft_vs_fftw,
        w.bfft.leak_rel,
        w.fftw.leak_rel,
        w.bfft.onbin_rel,
        w.fftw.onbin_rel,
        w.bfft.offbin_rel,
        w.fftw.offbin_rel,
        verdict(w.bfft.support_rel, w.fftw.support_rel, w.ratio));
    }

    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
