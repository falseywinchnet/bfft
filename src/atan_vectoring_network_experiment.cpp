// atan_vectoring_network_experiment.cpp
//
// Tests an FFT-like inverse-twiddle vectoring network for phase extraction.
//
// Instead of:
//   slope = minor / major -> table leaf
// or:
//   binary tree node lookup -> leaf lookup
//
// This does:
//   theta,c,s start at pi/8
//   for step = pi/16, pi/32, ...:
//       side = minor*c - major*s
//       sign = side > 0 ? +1 : -1
//       theta += sign*step
//       (c,s) = rotate((c,s), sign*step)
//
// Then:
//   h = cross / (mag + dot)
//   phase = theta + residual
//
// On AArch64, this file also includes explicit NEON f64 kernels for:
//   vec5+cubic and vec6/vec7+linear.
//
// Build scalar / native:
//   c++ -O3 -DNDEBUG -std=c++17 atan_vectoring_network_experiment.cpp -lm -o atan_vectoring_network_experiment
//
// Build on Apple Silicon:
//   clang++ -O3 -DNDEBUG -std=c++17 -mcpu=native atan_vectoring_network_experiment.cpp -lm -o /tmp/atan_vectoring_network_experiment
//
// Run:
//   ./atan_vectoring_network_experiment [samples=4194304]

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#define HAVE_NEON_F64 1
#else
#define HAVE_NEON_F64 0
#endif

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884
#endif

static constexpr double tau = 2.0 * M_PI;
static constexpr double pio2 = 0.5 * M_PI;
static constexpr double target_144 = 6.309573444801929e-8;

struct node_t { double c, s; };
struct leaf_t { double center, c, s; };
struct sample_t { double x, y, mag, ref; };

#if defined(__GNUC__) || defined(__clang__)
static inline double fma3(double a, double b, double c) { return __builtin_fma(a, b, c); }
#else
static inline double fma3(double a, double b, double c) { return a*b + c; }
#endif

static std::array<node_t, 31> make_nodes() {
    std::array<node_t, 31> nodes{};
    int p = 0;
    for (int d = 0; d < 5; ++d) {
        const int count = 1 << d;
        const double denom = 8.0 * static_cast<double>(count);
        for (int i = 0; i < count; ++i) {
            const double theta = static_cast<double>(2*i + 1) * M_PI / denom;
            nodes[static_cast<std::size_t>(p++)] = {std::cos(theta), std::sin(theta)};
        }
    }
    return nodes;
}

static const std::array<node_t, 31>& nodes32() {
    static const std::array<node_t, 31> n = make_nodes();
    return n;
}

template<int LEAVES>
struct leaf_table {
    std::array<leaf_t, LEAVES> leaf{};

    leaf_table() {
        const double width = (M_PI / 4.0) / static_cast<double>(LEAVES);
        for (int i = 0; i < LEAVES; ++i) {
            const double center = (static_cast<double>(i) + 0.5) * width;
            leaf[static_cast<std::size_t>(i)] = { center, std::cos(center), std::sin(center) };
        }
    }

    static const leaf_table& get() {
        static const leaf_table table;
        return table;
    }
};

template<int LEAVES, int Q>
struct slope_to_leaf_table {
    std::array<uint8_t, Q + 1> idx{};

    slope_to_leaf_table() {
        const double width = (M_PI / 4.0) / static_cast<double>(LEAVES);
        for (int q = 0; q <= Q; ++q) {
            const double t = (q == Q) ? 1.0 : (static_cast<double>(q) + 0.5) / static_cast<double>(Q);
            const double theta = std::atan(t);
            int leaf = static_cast<int>(theta / width);
            if (leaf < 0) leaf = 0;
            if (leaf >= LEAVES) leaf = LEAVES - 1;
            idx[static_cast<std::size_t>(q)] = static_cast<uint8_t>(leaf);
        }
    }

    static const slope_to_leaf_table& get() {
        static const slope_to_leaf_table table;
        return table;
    }
};

static inline double residual_linear(double h) {
    return h;
}

static inline double residual_cubic(double h) {
    const double s = h * h;
    return h * (1.0 - s * (1.0 / 3.0));
}

static inline double residual_degree7(double h) {
    const double s = h * h;
    double p = fma3(-1.0 / 7.0, s, 1.0 / 5.0);
    p = fma3(p, s, -1.0 / 3.0);
    p = fma3(p, s, 1.0);
    return h * p;
}

static inline int child_index(int idx, int base, double major, double minor) {
    const node_t node = nodes32()[static_cast<std::size_t>(base + idx)];
    const double side = fma3(minor, node.c, -(major * node.s));
    idx <<= 1;
    if (side > 0.0) idx |= 1;
    return idx;
}

static inline double first_octant_tree32_degree7(double major, double minor, double mag) {
    int idx = 0;
    idx = child_index(idx, 0, major, minor);
    idx = child_index(idx, 1, major, minor);
    idx = child_index(idx, 3, major, minor);
    idx = child_index(idx, 7, major, minor);
    idx = child_index(idx, 15, major, minor);

    const leaf_t leaf = leaf_table<32>::get().leaf[static_cast<std::size_t>(idx)];
    const double dot = fma3(major, leaf.c, minor * leaf.s);
    const double cross = fma3(minor, leaf.c, -(major * leaf.s));
    const double h = cross / (mag + dot);
    return leaf.center + 2.0 * residual_degree7(h);
}

template<int LEAVES, int Q, int RESIDUAL>
static inline double first_octant_slope(double major, double minor, double mag) {
    const double t = minor / major;
    int q = static_cast<int>(t * static_cast<double>(Q));
    if (q < 0) q = 0;
    if (q > Q) q = Q;

    const int idx = slope_to_leaf_table<LEAVES, Q>::get().idx[static_cast<std::size_t>(q)];
    const leaf_t leaf = leaf_table<LEAVES>::get().leaf[static_cast<std::size_t>(idx)];

    const double dot = fma3(major, leaf.c, minor * leaf.s);
    const double cross = fma3(minor, leaf.c, -(major * leaf.s));
    const double h = cross / (mag + dot);

    if constexpr (RESIDUAL == 1) {
        return leaf.center + 2.0 * residual_linear(h);
    } else if constexpr (RESIDUAL == 3) {
        return leaf.center + 2.0 * residual_cubic(h);
    } else {
        return leaf.center + 2.0 * residual_degree7(h);
    }
}

template<int STAGES, int RESIDUAL>
static inline double first_octant_vectoring(double major, double minor, double mag) {
    double theta = M_PI / 8.0;
    double c = std::cos(M_PI / 8.0);
    double s = std::sin(M_PI / 8.0);

    for (int stage = 0; stage < STAGES; ++stage) {
        const double step = M_PI / static_cast<double>(16u << stage);
        const double cstep = std::cos(step);
        const double sstep = std::sin(step);

        const double side = fma3(minor, c, -(major * s));
        const double sign = side > 0.0 ? 1.0 : -1.0;

        theta += sign * step;

        const double nc = fma3(c, cstep, -(sign * s * sstep));
        const double ns = fma3(sign * c, sstep, s * cstep);
        c = nc;
        s = ns;
    }

    const double dot = fma3(major, c, minor * s);
    const double cross = fma3(minor, c, -(major * s));
    const double h = cross / (mag + dot);

    if constexpr (RESIDUAL == 1) {
        return theta + 2.0 * residual_linear(h);
    } else if constexpr (RESIDUAL == 3) {
        return theta + 2.0 * residual_cubic(h);
    } else {
        return theta + 2.0 * residual_degree7(h);
    }
}

template<class FirstOctant>
static inline double atan2_core(double y, double x, double mag, FirstOctant&& first_octant) {
    const double ax = std::fabs(x);
    const double ay = std::fabs(y);

    if (ay == 0.0) {
        if (ax == 0.0) return 0.0;
        return x < 0.0 ? M_PI : 0.0;
    }
    if (ax == 0.0) {
        return y < 0.0 ? tau - pio2 : pio2;
    }

    double alpha = (ax >= ay)
        ? first_octant(ax, ay, mag)
        : pio2 - first_octant(ay, ax, mag);

    if (x < 0.0) alpha = M_PI - alpha;
    if (y < 0.0) alpha = -alpha;
    if (alpha < 0.0) alpha += tau;
    return alpha;
}

static inline double atan2_tree32(double y, double x, double mag) {
    return atan2_core(y, x, mag, [](double major, double minor, double m) {
        return first_octant_tree32_degree7(major, minor, m);
    });
}

template<int LEAVES, int Q, int RESIDUAL>
static inline double atan2_slope(double y, double x, double mag) {
    return atan2_core(y, x, mag, [](double major, double minor, double m) {
        return first_octant_slope<LEAVES, Q, RESIDUAL>(major, minor, m);
    });
}

template<int STAGES, int RESIDUAL>
static inline double atan2_vectoring(double y, double x, double mag) {
    return atan2_core(y, x, mag, [](double major, double minor, double m) {
        return first_octant_vectoring<STAGES, RESIDUAL>(major, minor, m);
    });
}

static inline double norm_atan2(double y, double x) {
    double a = std::atan2(y, x);
    if (a < 0.0) a += tau;
    return a;
}

static inline double phase_abs_err(double a, double b) {
    double d = std::fmod(a - b + M_PI, tau);
    if (d < 0.0) d += tau;
    d -= M_PI;
    return std::fabs(d);
}

struct acc_t { double max_abs = 0.0; double rms = 0.0; };

template<class Fn>
static acc_t accuracy_random(const char* name, const std::vector<sample_t>& samples, Fn&& fn) {
    long double sum2 = 0.0L;
    double max_abs = 0.0;

    for (const auto& v : samples) {
        const double a = fn(v.y, v.x, v.mag);
        const double e = phase_abs_err(a, v.ref);
        max_abs = std::max(max_abs, e);
        sum2 += static_cast<long double>(e) * e;
    }

    acc_t out;
    out.max_abs = max_abs;
    out.rms = std::sqrt(static_cast<double>(sum2 / static_cast<long double>(samples.size())));
    std::printf("%-22s random max %.17g rad  dBc %.3f  rms %.17g  %s\n",
                name,
                out.max_abs,
                out.max_abs > 0.0 ? 20.0 * std::log10(out.max_abs) : -INFINITY,
                out.rms,
                out.max_abs <= target_144 ? "PASS" : "FAIL");
    return out;
}

template<class Fn>
static acc_t accuracy_dense_first_quadrant(const char* name, Fn&& fn) {
    constexpr std::size_t n = 1u << 21;
    long double sum2 = 0.0L;
    double max_abs = 0.0;

    for (std::size_t i = 0; i <= n; ++i) {
        const double theta = (M_PI / 2.0) * static_cast<double>(i) / static_cast<double>(n);
        const double x = std::cos(theta);
        const double y = std::sin(theta);
        const double a = fn(y, x, 1.0);
        const double e = phase_abs_err(a, theta);
        max_abs = std::max(max_abs, e);
        sum2 += static_cast<long double>(e) * e;
    }

    acc_t out;
    out.max_abs = max_abs;
    out.rms = std::sqrt(static_cast<double>(sum2 / static_cast<long double>(n + 1)));
    std::printf("%-22s dense  max %.17g rad  dBc %.3f  rms %.17g  %s\n",
                name,
                out.max_abs,
                out.max_abs > 0.0 ? 20.0 * std::log10(out.max_abs) : -INFINITY,
                out.rms,
                out.max_abs <= target_144 ? "PASS" : "FAIL");
    return out;
}

template<class Fn>
static double speed_scalar(const char* name, const std::vector<sample_t>& samples, Fn&& fn) {
    volatile double sink = 0.0;
    const auto t0 = std::chrono::steady_clock::now();
    for (const auto& v : samples) {
        sink += fn(v.y, v.x, v.mag);
    }
    const auto t1 = std::chrono::steady_clock::now();

    const double sec = std::chrono::duration<double>(t1 - t0).count();
    const double mph = static_cast<double>(samples.size()) / sec / 1.0e6;
    std::printf("%-22s scalar time %.6f s  %.3f Mphase/s  sink %.17g\n", name, sec, mph, sink);
    return mph;
}

#if HAVE_NEON_F64
static inline float64x2_t vbslq_f64_mask(uint64x2_t mask, float64x2_t a, float64x2_t b) {
    return vreinterpretq_f64_u64(vbslq_u64(mask, vreinterpretq_u64_f64(a), vreinterpretq_u64_f64(b)));
}

template<int STAGES, int RESIDUAL>
static double speed_neon_vectoring(const char* name,
                                   const std::vector<double>& xs,
                                   const std::vector<double>& ys,
                                   const std::vector<double>& mags) {
    const std::size_t n = xs.size();
    const float64x2_t zero = vdupq_n_f64(0.0);
    const float64x2_t one = vdupq_n_f64(1.0);
    const float64x2_t negone = vdupq_n_f64(-1.0);
    const float64x2_t vpi = vdupq_n_f64(M_PI);
    const float64x2_t vpio2 = vdupq_n_f64(pio2);
    const float64x2_t vtau = vdupq_n_f64(tau);
    const float64x2_t two = vdupq_n_f64(2.0);
    const float64x2_t third = vdupq_n_f64(1.0 / 3.0);

    alignas(16) double tmp[2];
    volatile double sink = 0.0;

    const auto t0 = std::chrono::steady_clock::now();

    std::size_t i = 0;
    for (; i + 1 < n; i += 2) {
        const float64x2_t x = vld1q_f64(xs.data() + i);
        const float64x2_t y = vld1q_f64(ys.data() + i);
        const float64x2_t mag = vld1q_f64(mags.data() + i);

        const float64x2_t ax = vabsq_f64(x);
        const float64x2_t ay = vabsq_f64(y);
        const uint64x2_t swap = vcgtq_f64(ay, ax);

        const float64x2_t major = vbslq_f64_mask(swap, ay, ax);
        const float64x2_t minor = vbslq_f64_mask(swap, ax, ay);

        float64x2_t theta = vdupq_n_f64(M_PI / 8.0);
        float64x2_t c = vdupq_n_f64(std::cos(M_PI / 8.0));
        float64x2_t s = vdupq_n_f64(std::sin(M_PI / 8.0));

        #pragma unroll
        for (int stage = 0; stage < STAGES; ++stage) {
            const double step = M_PI / static_cast<double>(16u << stage);
            const float64x2_t vstep = vdupq_n_f64(step);
            const float64x2_t cstep = vdupq_n_f64(std::cos(step));
            const float64x2_t sstep = vdupq_n_f64(std::sin(step));

            const float64x2_t side = vfmsq_f64(vmulq_f64(minor, c), major, s);
            const uint64x2_t gt = vcgtq_f64(side, zero);
            const float64x2_t sign = vbslq_f64_mask(gt, one, negone);

            theta = vfmaq_f64(theta, sign, vstep);

            const float64x2_t signed_sstep = vmulq_f64(sign, sstep);
            const float64x2_t nc = vfmsq_f64(vmulq_f64(c, cstep), s, signed_sstep);
            const float64x2_t ns = vfmaq_f64(vmulq_f64(s, cstep), c, signed_sstep);
            c = nc;
            s = ns;
        }

        const float64x2_t dot = vfmaq_f64(vmulq_f64(minor, s), major, c);
        const float64x2_t cross = vfmsq_f64(vmulq_f64(minor, c), major, s);
        const float64x2_t h = vdivq_f64(cross, vaddq_f64(mag, dot));

        float64x2_t alpha;
        if constexpr (RESIDUAL == 1) {
            alpha = vfmaq_f64(theta, two, h);
        } else {
            const float64x2_t h2 = vmulq_f64(h, h);
            const float64x2_t poly = vsubq_f64(one, vmulq_f64(h2, third));
            const float64x2_t r = vmulq_f64(h, poly);
            alpha = vfmaq_f64(theta, two, r);
        }

        alpha = vbslq_f64_mask(swap, vsubq_f64(vpio2, alpha), alpha);
        alpha = vbslq_f64_mask(vcltq_f64(x, zero), vsubq_f64(vpi, alpha), alpha);
        alpha = vbslq_f64_mask(vcltq_f64(y, zero), vsubq_f64(zero, alpha), alpha);
        alpha = vbslq_f64_mask(vcltq_f64(alpha, zero), vaddq_f64(alpha, vtau), alpha);

        vst1q_f64(tmp, alpha);
        sink += tmp[0] + tmp[1];
    }

    for (; i < n; ++i) {
        sink += atan2_vectoring<STAGES, RESIDUAL>(ys[i], xs[i], mags[i]);
    }

    const auto t1 = std::chrono::steady_clock::now();

    const double sec = std::chrono::duration<double>(t1 - t0).count();
    const double mph = static_cast<double>(n) / sec / 1.0e6;
    std::printf("%-22s NEON   time %.6f s  %.3f Mphase/s  sink %.17g\n", name, sec, mph, sink);
    return mph;
}
#endif

int main(int argc, char** argv) {
    std::size_t n = 1u << 22;
    if (argc > 1) n = static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10));

    std::vector<sample_t> samples(n);
    std::vector<double> xs(n), ys(n), mags(n);

    std::mt19937_64 rng(20260627);
    std::uniform_real_distribution<double> phase_dist(0.0, tau);
    std::uniform_real_distribution<double> radius_log(-12.0, 12.0);

    for (std::size_t i = 0; i < n; ++i) {
        const double phase = phase_dist(rng);
        const double r = std::exp(radius_log(rng));
        const double x = r * std::cos(phase);
        const double y = r * std::sin(phase);
        samples[i] = { x, y, r, norm_atan2(y, x) };
        xs[i] = x;
        ys[i] = y;
        mags[i] = r;
    }

    (void)nodes32();
    (void)leaf_table<32>::get();
    (void)leaf_table<64>::get();
    (void)leaf_table<128>::get();
    (void)slope_to_leaf_table<32, 256>::get();
    (void)slope_to_leaf_table<64, 256>::get();
    (void)slope_to_leaf_table<128, 512>::get();

    std::printf("samples=%zu target=%.17g rad (-144 dBc)\n", n, target_144);
#if HAVE_NEON_F64
    std::printf("NEON f64 path: enabled\n\n");
#else
    std::printf("NEON f64 path: not available in this build\n\n");
#endif

    std::printf("accuracy vs std::atan2 reference:\n");
    accuracy_dense_first_quadrant("tree32 degree7", [](double y, double x, double m) {
        return atan2_tree32(y, x, m);
    });
    accuracy_dense_first_quadrant("slope32 cubic", [](double y, double x, double m) {
        return atan2_slope<32, 256, 3>(y, x, m);
    });
    accuracy_dense_first_quadrant("slope64 linear", [](double y, double x, double m) {
        return atan2_slope<64, 256, 1>(y, x, m);
    });
    accuracy_dense_first_quadrant("slope128 linear", [](double y, double x, double m) {
        return atan2_slope<128, 512, 1>(y, x, m);
    });
    accuracy_dense_first_quadrant("vec5 cubic", [](double y, double x, double m) {
        return atan2_vectoring<5, 3>(y, x, m);
    });
    accuracy_dense_first_quadrant("vec6 linear", [](double y, double x, double m) {
        return atan2_vectoring<6, 1>(y, x, m);
    });
    accuracy_dense_first_quadrant("vec7 linear", [](double y, double x, double m) {
        return atan2_vectoring<7, 1>(y, x, m);
    });

    std::printf("\nrandom accuracy spot check:\n");
    accuracy_random("vec5 cubic", samples, [](double y, double x, double m) {
        return atan2_vectoring<5, 3>(y, x, m);
    });
    accuracy_random("vec6 linear", samples, [](double y, double x, double m) {
        return atan2_vectoring<6, 1>(y, x, m);
    });
    accuracy_random("vec7 linear", samples, [](double y, double x, double m) {
        return atan2_vectoring<7, 1>(y, x, m);
    });

    std::printf("\nscalar speed:\n");
    speed_scalar("std::atan2", samples, [](double y, double x, double) { return norm_atan2(y, x); });
    speed_scalar("tree32 degree7", samples, [](double y, double x, double m) { return atan2_tree32(y, x, m); });
    speed_scalar("slope32 cubic", samples, [](double y, double x, double m) { return atan2_slope<32, 256, 3>(y, x, m); });
    speed_scalar("slope64 linear", samples, [](double y, double x, double m) { return atan2_slope<64, 256, 1>(y, x, m); });
    speed_scalar("slope128 linear", samples, [](double y, double x, double m) { return atan2_slope<128, 512, 1>(y, x, m); });
    speed_scalar("vec5 cubic", samples, [](double y, double x, double m) { return atan2_vectoring<5, 3>(y, x, m); });
    speed_scalar("vec6 linear", samples, [](double y, double x, double m) { return atan2_vectoring<6, 1>(y, x, m); });
    speed_scalar("vec7 linear", samples, [](double y, double x, double m) { return atan2_vectoring<7, 1>(y, x, m); });

#if HAVE_NEON_F64
    std::printf("\nexplicit NEON speed:\n");
    speed_neon_vectoring<5, 3>("vec5 cubic", xs, ys, mags);
    speed_neon_vectoring<6, 1>("vec6 linear", xs, ys, mags);
    speed_neon_vectoring<7, 1>("vec7 linear", xs, ys, mags);
#endif

    return 0;
}
