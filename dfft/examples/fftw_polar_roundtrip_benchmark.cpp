#include <bfft/bfft.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <limits>
#include <random>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <dlfcn.h>
#endif

namespace {
using clock_type = std::chrono::steady_clock;
using fftw_plan = void*;
using fftw_complex = double[2];
using fftwf_complex = float[2];

constexpr double pi = 3.141592653589793238462643383279502884;
constexpr unsigned FFTW_MEASURE_FLAG = 0U;
constexpr unsigned FFTW_DESTROY_INPUT_FLAG = 1U;
constexpr unsigned FFTW_UNALIGNED_FLAG = 2U;
constexpr unsigned FFTW_CONSERVE_MEMORY_FLAG = 4U;
constexpr unsigned FFTW_EXHAUSTIVE_FLAG = 8U;
constexpr unsigned FFTW_PRESERVE_INPUT_FLAG = 16U;
constexpr unsigned FFTW_PATIENT_FLAG = 32U;
constexpr unsigned FFTW_ESTIMATE_FLAG = 64U;

volatile double sink = 0.0;

double elapsed_ms(clock_type::time_point a, clock_type::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

double wrap_to_pi(double x) {
    while (x <= -pi) x += 2.0 * pi;
    while (x > pi) x -= 2.0 * pi;
    return x;
}

double safe_db(double ratio) {
    if (ratio <= 0.0) return -std::numeric_limits<double>::infinity();
    return 20.0 * std::log10(ratio);
}

std::string lower_string(const char* text) {
    if (!text) return std::string();
    std::string out(text);
    for (char& ch : out) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    return out;
}

const char* neon_status(const char* text) {
    std::string s = lower_string(text);
    if (s.empty()) return "unknown";
    if (s.find("neon") != std::string::npos) return "yes";
    if (s.find("simd") != std::string::npos || s.find("aarch64") != std::string::npos || s.find("arm") != std::string::npos) return "no";
    return "unknown";
}

struct FftwDouble {
    void* lib = nullptr;
    const char* path = nullptr;
    fftw_plan (*r2c)(int, double*, fftw_complex*, unsigned) = nullptr;
    fftw_plan (*c2r)(int, fftw_complex*, double*, unsigned) = nullptr;
    void (*execute)(const fftw_plan) = nullptr;
    void (*destroy)(fftw_plan) = nullptr;
    void* (*malloc_fn)(std::size_t) = nullptr;
    void (*free_fn)(void*) = nullptr;
    const char* version = nullptr;
    const char* cc = nullptr;
    const char* optim = nullptr;
    bool load() {
#if defined(_WIN32)
        return false;
#else
        const char* names[] = {"libfftw3.dylib", "libfftw3.3.dylib", "/opt/homebrew/lib/libfftw3.dylib", "/usr/local/lib/libfftw3.dylib", "libfftw3.so.3", "libfftw3.so", nullptr};
        for (int i = 0; names[i]; ++i) { lib = dlopen(names[i], RTLD_NOW); if (lib) { path = names[i]; break; } }
        if (!lib) return false;
        r2c = reinterpret_cast<decltype(r2c)>(dlsym(lib, "fftw_plan_dft_r2c_1d"));
        c2r = reinterpret_cast<decltype(c2r)>(dlsym(lib, "fftw_plan_dft_c2r_1d"));
        execute = reinterpret_cast<decltype(execute)>(dlsym(lib, "fftw_execute"));
        destroy = reinterpret_cast<decltype(destroy)>(dlsym(lib, "fftw_destroy_plan"));
        malloc_fn = reinterpret_cast<decltype(malloc_fn)>(dlsym(lib, "fftw_malloc"));
        free_fn = reinterpret_cast<decltype(free_fn)>(dlsym(lib, "fftw_free"));
        version = reinterpret_cast<const char*>(dlsym(lib, "fftw_version"));
        cc = reinterpret_cast<const char*>(dlsym(lib, "fftw_cc"));
        optim = reinterpret_cast<const char*>(dlsym(lib, "fftw_codelet_optim"));
        return r2c && c2r && execute && destroy && malloc_fn && free_fn;
#endif
    }
    ~FftwDouble() {
#if !defined(_WIN32)
        if (lib) dlclose(lib);
#endif
    }
};

struct FftwFloat {
    void* lib = nullptr; const char* path = nullptr;
    fftw_plan (*r2c)(int, float*, fftwf_complex*, unsigned) = nullptr;
    fftw_plan (*c2r)(int, fftwf_complex*, float*, unsigned) = nullptr;
    void (*execute)(const fftw_plan) = nullptr; void (*destroy)(fftw_plan) = nullptr;
    void* (*malloc_fn)(std::size_t) = nullptr; void (*free_fn)(void*) = nullptr;
    const char* version = nullptr; const char* cc = nullptr; const char* optim = nullptr;
    bool load() {
#if defined(_WIN32)
        return false;
#else
        const char* names[] = {"libfftw3f.dylib", "libfftw3f.3.dylib", "/opt/homebrew/lib/libfftw3f.dylib", "/usr/local/lib/libfftw3f.dylib", "libfftw3f.so.3", "libfftw3f.so", nullptr};
        for (int i = 0; names[i]; ++i) { lib = dlopen(names[i], RTLD_NOW); if (lib) { path = names[i]; break; } }
        if (!lib) return false;
        r2c = reinterpret_cast<decltype(r2c)>(dlsym(lib, "fftwf_plan_dft_r2c_1d"));
        c2r = reinterpret_cast<decltype(c2r)>(dlsym(lib, "fftwf_plan_dft_c2r_1d"));
        execute = reinterpret_cast<decltype(execute)>(dlsym(lib, "fftwf_execute"));
        destroy = reinterpret_cast<decltype(destroy)>(dlsym(lib, "fftwf_destroy_plan"));
        malloc_fn = reinterpret_cast<decltype(malloc_fn)>(dlsym(lib, "fftwf_malloc"));
        free_fn = reinterpret_cast<decltype(free_fn)>(dlsym(lib, "fftwf_free"));
        version = reinterpret_cast<const char*>(dlsym(lib, "fftwf_version"));
        cc = reinterpret_cast<const char*>(dlsym(lib, "fftwf_cc"));
        optim = reinterpret_cast<const char*>(dlsym(lib, "fftwf_codelet_optim"));
        return r2c && c2r && execute && destroy && malloc_fn && free_fn;
#endif
    }
    ~FftwFloat() {
#if !defined(_WIN32)
        if (lib) dlclose(lib);
#endif
    }
};

struct Metrics { double mse=0, snr=0, max_abs=0; };
struct Row { const char* name; double fwd_ms; double inv_ms; double total_ms; Metrics m; };
struct SfdrResult { double largest_spur_amp=0; std::size_t largest_spur_bin=0; double sfdr_db=-std::numeric_limits<double>::infinity(); };
struct Quality { const char* name; double mag_max=0, mag_rms=0, mag_spur=0, mag_spur_db=0, phase_max=0, phase_rms=0, phase_spur=0, phase_spur_db=0, complex_max_dbc=0, complex_rms_dbc=0; std::size_t used=0, skipped=0; };

Metrics error_metrics(const double* in, const double* out, std::size_t count) {
    Metrics m; double sig = 0.0;
    for (std::size_t i = 0; i < count; ++i) { double e = out[i] - in[i]; m.mse += e * e; sig += in[i] * in[i]; m.max_abs = std::max(m.max_abs, std::fabs(e)); }
    m.mse /= static_cast<double>(count); sig /= static_cast<double>(count);
    if (m.mse == 0.0) m.snr = std::numeric_limits<double>::infinity(); else m.snr = 10.0 * std::log10(sig / m.mse);
    return m;
}

SfdrResult sfdr_real(std::vector<double> v) {
    SfdrResult r; std::size_t n = v.size(); if (n == 0 || n > 8193) return r;
    double mean = 0.0; for (double x : v) mean += x; mean /= static_cast<double>(n); for (double& x : v) x -= mean;
    for (std::size_t k = 1; k < n; ++k) { double re = 0, im = 0; for (std::size_t t = 0; t < n; ++t) { double a = -2.0 * pi * static_cast<double>(k * t) / static_cast<double>(n); re += v[t] * std::cos(a); im += v[t] * std::sin(a); } double amp = std::hypot(re, im) / static_cast<double>(n); if (amp > r.largest_spur_amp) { r.largest_spur_amp = amp; r.largest_spur_bin = k; } }
    r.sfdr_db = safe_db(r.largest_spur_amp); return r;
}

template<class Polar, class Complex>
Quality quality(const char* name, const Polar* polar, const Complex* ref, std::size_t bins, double floor_amp, double phase_scale) {
    Quality q; q.name = name; std::vector<double> mag_err(bins), phase_err(bins); double max_ref = 0.0;
    for (std::size_t k = 0; k < bins; ++k) max_ref = std::max(max_ref, std::hypot(static_cast<double>(ref[k].re), static_cast<double>(ref[k].im)));
    double threshold = max_ref * phase_scale; double mag_sum = 0, phase_sum = 0, complex_sum = 0, complex_max = 0;
    for (std::size_t k = 0; k < bins; ++k) {
        double rr = ref[k].re; double ri = ref[k].im; double tm = polar[k].re; double tp = polar[k].im;
        double rm = std::hypot(rr, ri); double rp = std::atan2(ri, rr); if (rp < 0) rp += 2.0 * pi;
        double md = 20.0 * std::log10(std::max(tm, floor_amp)) - 20.0 * std::log10(std::max(rm, floor_amp));
        double pd = 0.0; if (rm > threshold) { pd = wrap_to_pi(tp - rp); q.used++; } else { q.skipped++; }
        mag_err[k] = md; phase_err[k] = pd; q.mag_max = std::max(q.mag_max, std::fabs(md)); q.phase_max = std::max(q.phase_max, std::fabs(pd)); mag_sum += md * md; phase_sum += pd * pd;
        double br = tm * std::cos(tp); double bi = tm * std::sin(tp); double ce = std::hypot(br - rr, bi - ri); complex_max = std::max(complex_max, ce); complex_sum += ce * ce;
    }
    q.mag_rms = std::sqrt(mag_sum / static_cast<double>(bins)); q.phase_rms = std::sqrt(phase_sum / static_cast<double>(bins));
    SfdrResult ms = sfdr_real(mag_err); SfdrResult ps = sfdr_real(phase_err); q.mag_spur = ms.largest_spur_amp; q.mag_spur_db = ms.sfdr_db; q.phase_spur = ps.largest_spur_amp; q.phase_spur_db = ps.sfdr_db;
    q.complex_max_dbc = safe_db(complex_max / std::max(max_ref, floor_amp)); q.complex_rms_dbc = safe_db(std::sqrt(complex_sum / static_cast<double>(bins)) / std::max(max_ref, floor_amp)); return q;
}

void print_row(const Row& r, std::size_t n, std::size_t frames) {
    double samples = static_cast<double>(n) * static_cast<double>(frames);
    std::printf("%-24s %9.3f %9.3f %9.3f %11.3f %10.3f %10.3e %9.3f %10.3e\n", r.name, r.fwd_ms, r.inv_ms, r.total_ms, samples / r.total_ms / 1000.0, static_cast<double>(frames) / r.total_ms, r.m.mse, r.m.snr, r.m.max_abs);
}

void print_quality(const Quality& q) {
    std::printf("%-20s %11.3e %11.3e %11.3e %11.3f %13.3e %13.3e %13.3e %13.3f %13.3f %13.3f used=%zu skipped=%zu\n", q.name, q.mag_max, q.mag_rms, q.mag_spur, q.mag_spur_db, q.phase_max, q.phase_rms, q.phase_spur, q.phase_spur_db, q.complex_max_dbc, q.complex_rms_dbc, q.used, q.skipped);
}

} // namespace

int main(int argc, char** argv) {
    std::size_t n = 2048, frames = 4096; unsigned fftw_flags = FFTW_MEASURE_FLAG; const char* mode = "MEASURE"; bool do_sfdr = true; int positional = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0) { std::printf("usage: fftw_polar_roundtrip_benchmark [N=2048] [frames=4096] [--fftw-measure|--fftw-estimate|--fftw-patient] [--sfdr|--no-sfdr]\n"); return 0; }
        if (std::strcmp(argv[i], "--fftw-estimate") == 0) { fftw_flags = FFTW_ESTIMATE_FLAG; mode = "ESTIMATE"; continue; }
        if (std::strcmp(argv[i], "--fftw-measure") == 0) { fftw_flags = FFTW_MEASURE_FLAG; mode = "MEASURE"; continue; }
        if (std::strcmp(argv[i], "--fftw-patient") == 0) { fftw_flags = FFTW_PATIENT_FLAG; mode = "PATIENT"; continue; }
        if (std::strcmp(argv[i], "--sfdr") == 0) { do_sfdr = true; continue; }
        if (std::strcmp(argv[i], "--no-sfdr") == 0) { do_sfdr = false; continue; }
        if (positional == 0) { n = static_cast<std::size_t>(std::strtoull(argv[i], nullptr, 10)); positional++; continue; }
        if (positional == 1) { frames = static_cast<std::size_t>(std::strtoull(argv[i], nullptr, 10)); positional++; continue; }
    }
    bfft::plan plan(n); std::size_t bins = plan.bins();
    std::vector<double> input(n * frames); std::vector<float> input_f(n * frames);
    std::mt19937_64 rng(20260627); std::uniform_real_distribution<double> noise(-0.0005, 0.0005);
    for (std::size_t f = 0; f < frames; ++f) for (std::size_t i = 0; i < n; ++i) { double t = static_cast<double>(i) / static_cast<double>(n); double x = 0.7*std::sin(2*pi*17*t)+0.2*std::sin(2*pi*113*t+0.3)+0.05*std::cos(2*pi*331*t)+noise(rng); input[f*n+i]=x; input_f[f*n+i]=static_cast<float>(x); }
    FftwDouble fftw; bool have_fftw = fftw.load(); FftwFloat fftwf; bool have_fftwf = fftwf.load();
    std::printf("BFFT/FFTW rectangular and polar roundtrip benchmark\nbackend: %s\nBFFT version: %s\nN=%zu frames=%zu total_samples=%zu\nFFTW plan mode: %s\n", bfft::backend_name().c_str(), bfft::version_string().c_str(), n, frames, n*frames, mode);
    std::printf("FFTW double: %s%s%s\nFFTW double version: %s\nFFTW double compiler: %s\nFFTW double codelet optim: %s\nFFTW double NEON: %s\n", have_fftw ? "found" : "not found", have_fftw ? " " : "", have_fftw ? fftw.path : "", fftw.version ? fftw.version : "n/a", fftw.cc ? fftw.cc : "n/a", fftw.optim ? fftw.optim : "n/a", neon_status(fftw.optim ? fftw.optim : nullptr));
    std::printf("FFTW float: %s%s%s\nFFTWf version: %s\nFFTWf compiler: %s\nFFTWf codelet optim: %s\nFFTWf NEON: %s\n", have_fftwf ? "found" : "not found", have_fftwf ? " " : "", have_fftwf ? fftwf.path : "", fftwf.version ? fftwf.version : "n/a", fftwf.cc ? fftwf.cc : "n/a", fftwf.optim ? fftwf.optim : "n/a", neon_status(fftwf.optim ? fftwf.optim : nullptr));
    if (!have_fftw) std::printf("FFTW not found by dlopen; FFTW rows skipped.\n");

    std::vector<Row> rows; std::vector<Quality> qualities;
    std::vector<double> out(n), work(plan.work_size()); std::vector<bfft::complex> spec(bins), polar(bins), scratch(plan.native_scratch_size()), ref64(bins);
    double bf=0, bi=0, bt=0; auto t0=clock_type::now(); for (std::size_t f=0; f<frames; ++f) { auto a=clock_type::now(); plan.forward(&input[f*n], spec.data(), work.data(), scratch.data()); auto b=clock_type::now(); plan.inverse(spec.data(), out.data()); auto c=clock_type::now(); bf += elapsed_ms(a,b); bi += elapsed_ms(b,c); sink += out[f%n]; } auto t1=clock_type::now(); plan.forward(input.data() + (frames - 1) * n, ref64.data(), work.data(), scratch.data()); rows.push_back({"bfft64 rectangular", bf, bi, elapsed_ms(t0,t1), error_metrics(input.data()+(frames-1)*n, out.data(), n)});
    bf=bi=0; t0=clock_type::now(); for (std::size_t f=0; f<frames; ++f) { auto a=clock_type::now(); plan.forward_mag_phase(&input[f*n], polar.data(), work.data()); auto b=clock_type::now(); plan.inverse_mag_phase(polar.data(), out.data()); auto c=clock_type::now(); bf += elapsed_ms(a,b); bi += elapsed_ms(b,c); sink += out[(f*7)%n]; } t1=clock_type::now(); rows.push_back({"bfft64 polar", bf, bi, elapsed_ms(t0,t1), error_metrics(input.data()+(frames-1)*n, out.data(), n)}); if (do_sfdr) qualities.push_back(quality("bfft64 polar", polar.data(), ref64.data(), bins, 1e-300, 1e-12));

    std::vector<float> outf(n), workf(plan.work_size_f32()); std::vector<bfft::complex_f32> specf(bins), polarf(bins), scratchf(plan.native_scratch_size()), ref32(bins);
    bf=bi=0; t0=clock_type::now(); for (std::size_t f=0; f<frames; ++f) { auto a=clock_type::now(); plan.forward_f32(&input_f[f*n], specf.data(), workf.data(), scratchf.data()); auto b=clock_type::now(); plan.inverse_f32(specf.data(), outf.data()); auto c=clock_type::now(); bf += elapsed_ms(a,b); bi += elapsed_ms(b,c); sink += outf[f%n]; } t1=clock_type::now(); for (std::size_t i=0;i<n;++i) out[i]=outf[i]; plan.forward_f32(input_f.data() + (frames - 1) * n, ref32.data(), workf.data(), scratchf.data()); rows.push_back({"bfft32 rectangular", bf, bi, elapsed_ms(t0,t1), error_metrics(input.data()+(frames-1)*n, out.data(), n)});
    bf=bi=0; t0=clock_type::now(); for (std::size_t f=0; f<frames; ++f) { auto a=clock_type::now(); plan.forward_mag_phase_f32(&input_f[f*n], polarf.data(), workf.data()); auto b=clock_type::now(); plan.inverse_mag_phase_f32(polarf.data(), outf.data()); auto c=clock_type::now(); bf += elapsed_ms(a,b); bi += elapsed_ms(b,c); sink += outf[(f*11)%n]; } t1=clock_type::now(); for (std::size_t i=0;i<n;++i) out[i]=outf[i]; rows.push_back({"bfft32 polar", bf, bi, elapsed_ms(t0,t1), error_metrics(input.data()+(frames-1)*n, out.data(), n)}); if (do_sfdr) qualities.push_back(quality("bfft32 polar", polarf.data(), ref32.data(), bins, 1e-30, 1e-5));

    if (have_fftw) {
        double* fin = static_cast<double*>(fftw.malloc_fn(sizeof(double)*n)); double* fout = static_cast<double*>(fftw.malloc_fn(sizeof(double)*n)); fftw_complex* freq = static_cast<fftw_complex*>(fftw.malloc_fn(sizeof(fftw_complex)*bins));
        auto a=clock_type::now(); fftw_plan pr = fftw.r2c(static_cast<int>(n), fin, freq, fftw_flags); auto b=clock_type::now(); fftw_plan pi2 = fftw.c2r(static_cast<int>(n), freq, fout, fftw_flags); auto c=clock_type::now(); std::printf("fftw64 plan r2c time %.3f ms\nfftw64 plan c2r time %.3f ms\n", elapsed_ms(a,b), elapsed_ms(b,c));
        bf=bi=0; t0=clock_type::now(); for (std::size_t f=0; f<frames; ++f) { std::memcpy(fin,&input[f*n],sizeof(double)*n); a=clock_type::now(); fftw.execute(pr); b=clock_type::now(); fftw.execute(pi2); for(std::size_t i=0;i<n;++i) out[i]=fout[i]/static_cast<double>(n); c=clock_type::now(); bf+=elapsed_ms(a,b); bi+=elapsed_ms(b,c); sink+=out[f%n]; } t1=clock_type::now(); rows.push_back({"fftw64 rectangular", bf, bi, elapsed_ms(t0,t1), error_metrics(input.data()+(frames-1)*n,out.data(),n)});
        std::vector<bfft::complex> fpolar(bins); bf=bi=0; t0=clock_type::now(); for (std::size_t f=0; f<frames; ++f) { std::memcpy(fin,&input[f*n],sizeof(double)*n); a=clock_type::now(); fftw.execute(pr); for(std::size_t k=0;k<bins;++k){ double ph=std::atan2(freq[k][1],freq[k][0]); if(ph<0) ph+=2*pi; fpolar[k]={std::hypot(freq[k][0],freq[k][1]),ph}; } b=clock_type::now(); for(std::size_t k=0;k<bins;++k){ freq[k][0]=fpolar[k].re*std::cos(fpolar[k].im); freq[k][1]=fpolar[k].re*std::sin(fpolar[k].im);} fftw.execute(pi2); for(std::size_t i=0;i<n;++i) out[i]=fout[i]/static_cast<double>(n); c=clock_type::now(); bf+=elapsed_ms(a,b); bi+=elapsed_ms(b,c); sink+=out[(f*13)%n]; } t1=clock_type::now(); rows.push_back({"fftw64 std-polar", bf, bi, elapsed_ms(t0,t1), error_metrics(input.data()+(frames-1)*n,out.data(),n)}); if (do_sfdr) qualities.push_back(quality("fftw64 std-polar", fpolar.data(), ref64.data(), bins, 1e-300, 1e-12));
        fftw.destroy(pr); fftw.destroy(pi2); fftw.free_fn(fin); fftw.free_fn(fout); fftw.free_fn(freq);
    }
    if (have_fftwf) {
        float* fin = static_cast<float*>(fftwf.malloc_fn(sizeof(float)*n)); float* fout = static_cast<float*>(fftwf.malloc_fn(sizeof(float)*n)); fftwf_complex* freq = static_cast<fftwf_complex*>(fftwf.malloc_fn(sizeof(fftwf_complex)*bins)); auto a=clock_type::now(); fftw_plan pr=fftwf.r2c(static_cast<int>(n),fin,freq,fftw_flags); auto b=clock_type::now(); fftw_plan pi2=fftwf.c2r(static_cast<int>(n),freq,fout,fftw_flags); auto c=clock_type::now(); std::printf("fftw32 plan r2c time %.3f ms\nfftw32 plan c2r time %.3f ms\n", elapsed_ms(a,b), elapsed_ms(b,c));
        bf=bi=0; t0=clock_type::now(); for(std::size_t f=0;f<frames;++f){ std::memcpy(fin,&input_f[f*n],sizeof(float)*n); a=clock_type::now(); fftwf.execute(pr); b=clock_type::now(); fftwf.execute(pi2); for(std::size_t i=0;i<n;++i) out[i]=static_cast<double>(fout[i]/static_cast<float>(n)); c=clock_type::now(); bf+=elapsed_ms(a,b); bi+=elapsed_ms(b,c); sink+=out[f%n]; } t1=clock_type::now(); rows.push_back({"fftw32 rectangular",bf,bi,elapsed_ms(t0,t1),error_metrics(input.data()+(frames-1)*n,out.data(),n)});
        std::vector<bfft::complex_f32> fpolar(bins); bf=bi=0; t0=clock_type::now(); for(std::size_t f=0;f<frames;++f){ std::memcpy(fin,&input_f[f*n],sizeof(float)*n); a=clock_type::now(); fftwf.execute(pr); for(std::size_t k=0;k<bins;++k){ float ph=std::atan2(freq[k][1],freq[k][0]); if(ph<0) ph+=static_cast<float>(2*pi); fpolar[k]={std::hypot(freq[k][0],freq[k][1]),ph}; } b=clock_type::now(); for(std::size_t k=0;k<bins;++k){ freq[k][0]=fpolar[k].re*std::cos(fpolar[k].im); freq[k][1]=fpolar[k].re*std::sin(fpolar[k].im);} fftwf.execute(pi2); for(std::size_t i=0;i<n;++i) out[i]=static_cast<double>(fout[i]/static_cast<float>(n)); c=clock_type::now(); bf+=elapsed_ms(a,b); bi+=elapsed_ms(b,c); sink+=out[(f*17)%n]; } t1=clock_type::now(); rows.push_back({"fftw32 std-polar",bf,bi,elapsed_ms(t0,t1),error_metrics(input.data()+(frames-1)*n,out.data(),n)}); if(do_sfdr) qualities.push_back(quality("fftw32 std-polar",fpolar.data(),ref32.data(),bins,1e-30,1e-5));
        fftwf.destroy(pr); fftwf.destroy(pi2); fftwf.free_fn(fin); fftwf.free_fn(fout); fftwf.free_fn(freq);
    }
    std::printf("\n%-24s %9s %9s %9s %11s %10s %10s %9s %10s\n", "name", "fwd_ms", "inv_ms", "total_ms", "Msamples/s", "kframes/s", "MSE", "SNR_dB", "max_abs"); for (const Row& r : rows) print_row(r,n,frames);
    if (do_sfdr) { std::printf("\n%-20s %11s %11s %11s %11s %13s %13s %13s %13s %13s %13s\n", "name", "mag_max_db", "mag_rms_db", "mag_spur", "mag_re_1db", "phase_max", "phase_rms", "phase_spur", "phase_dBrad", "complex_max", "complex_rms"); for (const Quality& q : qualities) print_quality(q); }
    std::printf("sink %.17g\n", sink); return 0;
}
