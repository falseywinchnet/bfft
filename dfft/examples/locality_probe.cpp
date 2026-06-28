#include <bfft/bfft.hpp>

#include <algorithm>
#include <chrono>
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

namespace {

constexpr double pi = 3.141592653589793238462643383279502884;

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

std::size_t parse_mib(const char* text) {
  const std::size_t value = parse_size(text);
  if (value > (std::numeric_limits<std::size_t>::max() >> 20)) {
    throw std::runtime_error("eviction size too large");
  }
  return value;
}

int default_iters(std::size_t n) {
  const long long target = 90000000LL;
  const int l = std::max(1, ilog2_pow2(n));
  int iters = static_cast<int>(target / (static_cast<long long>(n) * l));
  if (iters < 16) {
    iters = 16;
  }
  if (iters > 50000) {
    iters = 50000;
  }
  return iters;
}

std::vector<double> make_input(std::size_t n) {
  std::vector<double> input(n);
  std::mt19937_64 rng(42);
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

std::vector<float> make_input_f32(const std::vector<double>& input) {
  std::vector<float> out(input.size());
  for (std::size_t i = 0; i < input.size(); ++i) {
    out[i] = static_cast<float>(input[i]);
  }
  return out;
}

double mib(std::size_t bytes) {
  return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

template <class Fn>
double timed_ns(int iters, Fn&& fn) {
  const auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) {
    fn(i);
  }
  const auto t1 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::nano>(t1 - t0).count()
       / static_cast<double>(iters);
}

struct ProbeResult {
  double hot_ns = 0.0;
  double ring_ns = 0.0;
  double ring_over_hot = 0.0;
};

template <class Fn>
ProbeResult measure(int iters, std::size_t slots, Fn&& fn) {
  const int warmups = 4;
  for (int i = 0; i < warmups; ++i) {
    fn(i, 0);
  }

  ProbeResult result;
  result.hot_ns = timed_ns(iters, [&](int i) {
    fn(i, 0);
  });

  result.ring_ns = timed_ns(iters, [&](int i) {
    fn(i, static_cast<std::size_t>(i) % slots);
  });
  if (result.hot_ns > 0.0) {
    result.ring_over_hot = result.ring_ns / result.hot_ns;
  }

  return result;
}

std::size_t slot_count(std::size_t bytes, std::size_t ring_mib) {
  const std::size_t ring_bytes = std::max<std::size_t>(ring_mib << 20, bytes * 2);
  std::size_t slots = ring_bytes / std::max<std::size_t>(bytes, 1);
  if (slots < 2) {
    slots = 2;
  }
  if (slots > 64) {
    slots = 64;
  }
  return slots;
}

template <class T>
std::vector<std::vector<T>> make_slots(std::size_t count, std::size_t size) {
  return std::vector<std::vector<T>>(count, std::vector<T>(size));
}

template <class T>
std::vector<std::vector<T>> make_seeded_slots(const std::vector<T>& seed, std::size_t count) {
  std::vector<std::vector<T>> slots(count, seed);
  return slots;
}

void print_result(std::size_t n,
                  int iters,
                  const char* policy,
                  const char* mode,
                  std::size_t slots,
                  std::size_t bytes,
                  const ProbeResult& result,
                  double sink) {
  std::printf("%9zu %7d %-28s %-13s %5zu %10.3f %12.1f %12.1f %9.3f %12.4f %.8g\n",
              n,
              iters,
              policy,
              mode,
              slots,
              mib(bytes),
              result.hot_ns,
              result.ring_ns,
              result.ring_over_hot,
              result.hot_ns / static_cast<double>(n),
              sink);
}

void probe_one(std::size_t n, int forced_iters, std::size_t ring_mib) {
  const int iters = forced_iters > 0 ? forced_iters : default_iters(n);
  const std::size_t nb = n / 2 + 1;

  bfft::plan plan(n);
  const std::vector<double> seed = make_input(n);
  const std::vector<float> seed_f32 = make_input_f32(seed);
  double sink = 0.0;

  const std::string policy = plan.standard_policy();
  const std::size_t input_bytes = sizeof(double) * n;
  const std::size_t input_f32_bytes = sizeof(float) * n;
  const std::size_t work_bytes = sizeof(double) * plan.work_size();
  const std::size_t work_f32_bytes = sizeof(float) * plan.work_size_f32();
  const std::size_t complex_bytes = sizeof(bfft::complex) * nb;
  const std::size_t complex_f32_bytes = sizeof(bfft::complex_f32) * nb;
  const std::size_t scalar_bins_bytes = sizeof(double) * nb;
  const std::size_t scalar_bins_f32_bytes = sizeof(float) * nb;

  {
    const std::size_t bytes = input_bytes + work_bytes;
    const std::size_t slots = slot_count(bytes, ring_mib);
    std::vector<std::vector<double>> inputs = make_seeded_slots(seed, slots);
    std::vector<std::vector<double>> residues = make_slots<double>(slots, n);
    ProbeResult result = measure(iters, slots, [&](int i, std::size_t slot) {
      inputs[slot][static_cast<std::size_t>(i) & (n - 1)] += 1e-12;
      plan.forward_residues(inputs[slot].data(), residues[slot].data());
      sink += residues[slot][(static_cast<std::size_t>(i) * 17) & (n - 1)];
    });
    print_result(n, iters, policy.c_str(), "residue64", slots, bytes, result, sink);
  }

  {
    const std::size_t bytes = input_bytes + work_bytes + complex_bytes;
    const std::size_t slots = slot_count(bytes, ring_mib);
    std::vector<std::vector<double>> inputs = make_seeded_slots(seed, slots);
    std::vector<std::vector<double>> works = make_slots<double>(slots, plan.work_size());
    std::vector<std::vector<bfft::complex>> native = make_slots<bfft::complex>(slots, nb);
    ProbeResult result = measure(iters, slots, [&](int i, std::size_t slot) {
      inputs[slot][static_cast<std::size_t>(i) & (n - 1)] += 1e-12;
      plan.forward_native(inputs[slot].data(), native[slot].data(), works[slot].data());
      sink += native[slot][(static_cast<std::size_t>(i) * 17) % nb].re;
    });
    print_result(n, iters, policy.c_str(), "native64", slots, bytes, result, sink);
  }

  {
    const std::size_t bytes = input_bytes + work_bytes + 2 * complex_bytes;
    const std::size_t slots = slot_count(bytes, ring_mib);
    std::vector<std::vector<double>> inputs = make_seeded_slots(seed, slots);
    std::vector<std::vector<double>> works = make_slots<double>(slots, plan.work_size());
    std::vector<std::vector<bfft::complex>> standard = make_slots<bfft::complex>(slots, nb);
    std::vector<std::vector<bfft::complex>> scratch =
      make_slots<bfft::complex>(slots, plan.native_scratch_size());
    ProbeResult result = measure(iters, slots, [&](int i, std::size_t slot) {
      inputs[slot][static_cast<std::size_t>(i) & (n - 1)] += 1e-12;
      plan.forward(inputs[slot].data(), standard[slot].data(), works[slot].data(), scratch[slot].data());
      sink += standard[slot][(static_cast<std::size_t>(i) * 17) % nb].re;
    });
    print_result(n, iters, policy.c_str(), "standard64", slots, bytes, result, sink);
  }

  {
    const std::size_t bytes = 2 * complex_bytes;
    const std::size_t slots = slot_count(bytes, ring_mib);
    std::vector<std::vector<double>> works = make_slots<double>(slots, plan.work_size());
    std::vector<std::vector<bfft::complex>> native = make_slots<bfft::complex>(slots, nb);
    std::vector<std::vector<bfft::complex>> standard = make_slots<bfft::complex>(slots, nb);
    for (std::size_t slot = 0; slot < slots; ++slot) {
      plan.forward_native(seed.data(), native[slot].data(), works[slot].data());
    }
    ProbeResult result = measure(iters, slots, [&](int i, std::size_t slot) {
      native[slot][(static_cast<std::size_t>(i) * 13) % nb].re += 1e-12;
      plan.native_to_standard(native[slot].data(), standard[slot].data());
      sink += standard[slot][(static_cast<std::size_t>(i) * 17) % nb].re;
    });
    print_result(n, iters, policy.c_str(), "convert64", slots, bytes, result, sink);
  }

  {
    const std::size_t bytes = input_bytes + work_bytes + scalar_bins_bytes;
    const std::size_t slots = slot_count(bytes, ring_mib);
    std::vector<std::vector<double>> inputs = make_seeded_slots(seed, slots);
    std::vector<std::vector<double>> works = make_slots<double>(slots, plan.work_size());
    std::vector<std::vector<double>> magnitudes = make_slots<double>(slots, nb);
    ProbeResult result = measure(iters, slots, [&](int i, std::size_t slot) {
      inputs[slot][static_cast<std::size_t>(i) & (n - 1)] += 1e-12;
      plan.forward_magnitude(inputs[slot].data(), magnitudes[slot].data(), works[slot].data());
      sink += magnitudes[slot][(static_cast<std::size_t>(i) * 17) % nb];
    });
    print_result(n, iters, policy.c_str(), "magnitude64", slots, bytes, result, sink);
  }

  {
    const std::size_t bytes = input_f32_bytes + work_f32_bytes + complex_f32_bytes;
    const std::size_t slots = slot_count(bytes, ring_mib);
    std::vector<std::vector<float>> inputs = make_seeded_slots(seed_f32, slots);
    std::vector<std::vector<float>> works = make_slots<float>(slots, plan.work_size_f32());
    std::vector<std::vector<bfft::complex_f32>> native = make_slots<bfft::complex_f32>(slots, nb);
    ProbeResult result = measure(iters, slots, [&](int i, std::size_t slot) {
      inputs[slot][static_cast<std::size_t>(i) & (n - 1)] += 1e-7f;
      plan.forward_native_f32(inputs[slot].data(), native[slot].data(), works[slot].data());
      sink += native[slot][(static_cast<std::size_t>(i) * 17) % nb].re;
    });
    print_result(n, iters, policy.c_str(), "native32", slots, bytes, result, sink);
  }

  {
    const std::size_t bytes = input_f32_bytes + work_f32_bytes + complex_f32_bytes;
    const std::size_t slots = slot_count(bytes, ring_mib);
    std::vector<std::vector<float>> inputs = make_seeded_slots(seed_f32, slots);
    std::vector<std::vector<float>> works = make_slots<float>(slots, plan.work_size_f32());
    std::vector<std::vector<bfft::complex_f32>> standard = make_slots<bfft::complex_f32>(slots, nb);
    std::vector<std::vector<bfft::complex_f32>> scratch =
      make_slots<bfft::complex_f32>(slots, plan.native_scratch_size());
    ProbeResult result = measure(iters, slots, [&](int i, std::size_t slot) {
      inputs[slot][static_cast<std::size_t>(i) & (n - 1)] += 1e-7f;
      plan.forward_f32(inputs[slot].data(), standard[slot].data(), works[slot].data(), scratch[slot].data());
      sink += standard[slot][(static_cast<std::size_t>(i) * 17) % nb].re;
    });
    print_result(n, iters, policy.c_str(), "standard32", slots, bytes, result, sink);
  }

  {
    const std::size_t bytes = 2 * complex_f32_bytes;
    const std::size_t slots = slot_count(bytes, ring_mib);
    std::vector<std::vector<float>> works = make_slots<float>(slots, plan.work_size_f32());
    std::vector<std::vector<bfft::complex_f32>> native = make_slots<bfft::complex_f32>(slots, nb);
    std::vector<std::vector<bfft::complex_f32>> standard = make_slots<bfft::complex_f32>(slots, nb);
    for (std::size_t slot = 0; slot < slots; ++slot) {
      plan.forward_native_f32(seed_f32.data(), native[slot].data(), works[slot].data());
    }
    ProbeResult result = measure(iters, slots, [&](int i, std::size_t slot) {
      native[slot][(static_cast<std::size_t>(i) * 13) % nb].re += 1e-7f;
      plan.native_to_standard_f32(native[slot].data(), standard[slot].data());
      sink += standard[slot][(static_cast<std::size_t>(i) * 17) % nb].re;
    });
    print_result(n, iters, policy.c_str(), "convert32", slots, bytes, result, sink);
  }

  {
    const std::size_t bytes = input_f32_bytes + work_f32_bytes + scalar_bins_f32_bytes;
    const std::size_t slots = slot_count(bytes, ring_mib);
    std::vector<std::vector<float>> inputs = make_seeded_slots(seed_f32, slots);
    std::vector<std::vector<float>> works = make_slots<float>(slots, plan.work_size_f32());
    std::vector<std::vector<float>> magnitudes = make_slots<float>(slots, nb);
    ProbeResult result = measure(iters, slots, [&](int i, std::size_t slot) {
      inputs[slot][static_cast<std::size_t>(i) & (n - 1)] += 1e-7f;
      plan.forward_magnitude_f32(inputs[slot].data(), magnitudes[slot].data(), works[slot].data());
      sink += magnitudes[slot][(static_cast<std::size_t>(i) * 17) % nb];
    });
    print_result(n, iters, policy.c_str(), "magnitude32", slots, bytes, result, sink);
  }
}

} // namespace

int main(int argc, char** argv) {
  try {
    std::size_t forced_n = 0;
    int forced_iters = 0;
    std::size_t ring_mib = 64;

    int positional_count = 0;
    for (int i = 1; i < argc; ++i) {
      if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
        std::printf("usage: %s [N [iters [ring_mib]]]\n", argv[0]);
        std::printf("Measures hot-cache BFFT paths and rotating working-set paths sized by ring_mib.\n");
        return 0;
      }
      if (positional_count == 0) {
        forced_n = parse_size(argv[i]);
      } else if (positional_count == 1) {
        forced_iters = parse_iters(argv[i]);
      } else if (positional_count == 2) {
        ring_mib = parse_mib(argv[i]);
      } else {
        std::fprintf(stderr, "usage: %s [N [iters [ring_mib]]]\n", argv[0]);
        return 2;
      }
      ++positional_count;
    }

    std::printf("BFFT locality probe. backend: %s, version: %s, ring_mib: %zu\n",
                bfft::backend_name().c_str(),
                bfft::version_string().c_str(),
                ring_mib);
    std::printf("%9s %7s %-28s %-13s %5s %10s %12s %12s %9s %12s %s\n",
                "N",
                "iters",
                "policy",
                "mode",
                "slots",
                "bytes_MiB",
                "hot_ns",
                "ring_ns",
                "ring/hot",
                "hot_ns/N",
                "sink");

    if (forced_n > 0) {
      if (!is_power2(forced_n) || forced_n < 4) {
        std::fprintf(stderr, "N must be a power of two >= 4.\n");
        return 2;
      }
      probe_one(forced_n, forced_iters, ring_mib);
      return 0;
    }

    const std::size_t sizes[] = {
      1024,
      4096,
      16384,
      65536,
      262144,
      1048576
    };

    for (std::size_t n : sizes) {
      probe_one(n, forced_iters, ring_mib);
    }
    return 0;
  } catch (const std::exception& exc) {
    std::fprintf(stderr, "locality probe failed: %s\n", exc.what());
    return 1;
  }
}
