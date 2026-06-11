#include <bfft/bfft.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <string>
#include <vector>

namespace {

constexpr long double pi = 3.141592653589793238462643383279502884L;

struct ProbeStats {
    std::size_t transforms = 0;
    std::size_t worst_k = 0;
    char worst_wave = '?';
    long double max_support_rel = 0.0L;
    long double max_leak_rel = 0.0L;
    long double max_onbin_rel = 0.0L;
    long double max_offbin_rel = 0.0L;
    long double sum_support_rel_sq = 0.0L;
};

std::size_t parse_size_arg(const char* s, std::size_t fallback) {
    if (!s || !*s) return fallback;
    char* end = nullptr;
    const unsigned long long v = std::strtoull(s, &end, 10);
    if (!end || *end != '\0' || v == 0) return fallback;
    return static_cast<std::size_t>(v);
}

void add_bin(std::set<std::size_t>& bins, std::size_t n, std::size_t k) {
    const std::size_t last = n / 2 - 1;
    if (k >= 1 && k <= last) bins.insert(k);
}

std::vector<std::size_t> choose_bins(std::size_t n, std::size_t requested) {
    const std::size_t last = n / 2 - 1;
    std::vector<std::size_t> out;
    if (last == 0) return out;

    if (requested == 0 || requested >= last) {
        out.reserve(last);
        for (std::size_t k = 1; k <= last; ++k) out.push_back(k);
        return out;
    }

    std::set<std::size_t> bins;
    const std::size_t anchors[] = {
        1, 2, 3, 5, 7,
        n / 128, n / 64, n / 32, n / 16, n / 8, n / 6, n / 4, n / 3,
        last > 8 ? last - 8 : last,
        last > 4 ? last - 4 : last,
        last > 2 ? last - 2 : last,
        last > 1 ? last - 1 : last,
        last
    };
    for (std::size_t k : anchors) add_bin(bins, n, k);

    std::uint64_t state = 0x9e3779b97f4a7c15ULL ^ static_cast<std::uint64_t>(n);
    while (bins.size() < requested) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        const std::size_t k = 1 + static_cast<std::size_t>(state % last);
        add_bin(bins, n, k);
    }

    out.assign(bins.begin(), bins.end());
    if (out.size() > requested) out.resize(requested);
    return out;
}

void fill_wave(std::vector<double>& input, std::size_t k, bool sine) {
    const std::size_t n = input.size();
    for (std::size_t t = 0; t < n; ++t) {
        const long double angle = 2.0L * pi * static_cast<long double>(k) * static_cast<long double>(t) / static_cast<long double>(n);
        input[t] = static_cast<double>(sine ? std::sin(angle) : std::cos(angle));
    }
}

void probe_one_wave(const bfft::plan& plan,
                    std::size_t k,
                    bool sine,
                    std::vector<double>& input,
                    std::vector<double>& work,
                    std::vector<bfft::complex>& spectrum,
                    std::vector<bfft::complex>& scratch,
                    ProbeStats& stats) {
    const std::size_t n = plan.size();
    fill_wave(input, k, sine);
    plan.forward(input.data(), spectrum.data(), work.data(), scratch.data());

    const long double expected_mag = static_cast<long double>(n) / 2.0L;
    const long double expected_re = sine ? 0.0L : expected_mag;
    const long double expected_im = sine ? -expected_mag : 0.0L;

    long double off_energy = 0.0L;
    long double offbin_max = 0.0L;
    for (std::size_t i = 0; i < spectrum.size(); ++i) {
        const long double re = static_cast<long double>(spectrum[i].re);
        const long double im = static_cast<long double>(spectrum[i].im);
        if (i != k) {
            const long double mag = std::sqrt(re * re + im * im);
            off_energy += re * re + im * im;
            offbin_max = std::max(offbin_max, mag);
        }
    }

    const long double got_re = static_cast<long double>(spectrum[k].re);
    const long double got_im = static_cast<long double>(spectrum[k].im);
    const long double onbin_err = std::sqrt((got_re - expected_re) * (got_re - expected_re) +
                                           (got_im - expected_im) * (got_im - expected_im));

    const long double leak_rel = std::sqrt(off_energy) / expected_mag;
    const long double onbin_rel = onbin_err / expected_mag;
    const long double offbin_rel = offbin_max / expected_mag;
    const long double support_rel = std::sqrt(off_energy + onbin_err * onbin_err) / expected_mag;

    ++stats.transforms;
    stats.sum_support_rel_sq += support_rel * support_rel;
    stats.max_leak_rel = std::max(stats.max_leak_rel, leak_rel);
    stats.max_onbin_rel = std::max(stats.max_onbin_rel, onbin_rel);
    stats.max_offbin_rel = std::max(stats.max_offbin_rel, offbin_rel);
    if (support_rel > stats.max_support_rel) {
        stats.max_support_rel = support_rel;
        stats.worst_k = k;
        stats.worst_wave = sine ? 's' : 'c';
    }
}

ProbeStats probe_size(std::size_t n, std::size_t requested_bins) {
    bfft::plan plan(n);
    std::vector<double> input(plan.size());
    std::vector<double> work(plan.work_size());
    std::vector<bfft::complex> spectrum(plan.bins());
    std::vector<bfft::complex> scratch(plan.native_scratch_size());
    ProbeStats stats;

    const std::vector<std::size_t> bins = choose_bins(n, requested_bins);
    for (std::size_t k : bins) {
        probe_one_wave(plan, k, false, input, work, spectrum, scratch, stats);
        probe_one_wave(plan, k, true, input, work, spectrum, scratch, stats);
    }
    return stats;
}

const char* verdict(long double x) {
    if (x < 1e-10L) return "roundoff";
    if (x < 1e-7L) return "small";
    if (x < 1e-4L) return "visible";
    if (x < 1e-2L) return "large";
    return "unsafe";
}

} // namespace

int main(int argc, char** argv) {
    const std::size_t max_pow = parse_size_arg(argc > 1 ? argv[1] : nullptr, 20);
    const std::size_t bins_per_size = parse_size_arg(argc > 2 ? argv[2] : nullptr, 32);
    const std::size_t min_pow = parse_size_arg(argc > 3 ? argv[3] : nullptr, 2);

    std::cout << "# bfft invariant-support probe\n";
    std::cout << "# version=" << bfft::version_string() << " backend=" << bfft::backend_name() << "\n";
    std::cout << "# args: max_pow=" << max_pow << " bins_per_size=" << bins_per_size
              << " min_pow=" << min_pow << "\n";
    std::cout << "# bins_per_size=0 means exhaustive over k=1..N/2-1 for every N\n";
    std::cout << "N,policy,transforms,worst_k,wave,max_support_rel,rms_support_rel,max_leak_rel,max_onbin_rel,max_offbin_rel,verdict\n";

    for (std::size_t p = min_pow; p <= max_pow; ++p) {
        const std::size_t n = static_cast<std::size_t>(1) << p;
        try {
            bfft::plan plan(n);
            const std::string policy = plan.standard_policy();
            ProbeStats stats = probe_size(n, bins_per_size);
            const long double rms = stats.transforms
                ? std::sqrt(stats.sum_support_rel_sq / static_cast<long double>(stats.transforms))
                : 0.0L;

            std::cout << n << ',' << policy << ',' << stats.transforms << ','
                      << stats.worst_k << ',' << stats.worst_wave << ','
                      << std::scientific << std::setprecision(8)
                      << static_cast<double>(stats.max_support_rel) << ','
                      << static_cast<double>(rms) << ','
                      << static_cast<double>(stats.max_leak_rel) << ','
                      << static_cast<double>(stats.max_onbin_rel) << ','
                      << static_cast<double>(stats.max_offbin_rel) << ','
                      << verdict(stats.max_support_rel) << std::defaultfloat << '\n';
        } catch (const std::exception& e) {
            std::cerr << "N=" << n << " failed: " << e.what() << "\n";
            return 2;
        }
    }

    return 0;
}
