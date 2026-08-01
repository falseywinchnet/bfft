#include "../src/detail/meyer_kernel.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <utility>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;

inline std::size_t short_periodic_offset(std::size_t coordinate, int delta,
                                         std::size_t extent) {
    std::ptrdiff_t value = static_cast<std::ptrdiff_t>(coordinate) + delta;
    if (value < 0) value += static_cast<std::ptrdiff_t>(extent);
    if (value >= static_cast<std::ptrdiff_t>(extent))
        value -= static_cast<std::ptrdiff_t>(extent);
    return static_cast<std::size_t>(value);
}

template <std::size_t NA, std::size_t NB>
std::array<double, NA + NB - 1> convolve(
        const std::array<double, NA>& a,
        const std::array<double, NB>& b) {
    std::array<double, NA + NB - 1> out{};
    for (std::size_t i = 0; i < NA; ++i)
        for (std::size_t j = 0; j < NB; ++j)
            out[i + j] += a[i] * b[j];
    return out;
}

std::array<double, 7> cross_cube(double side) {
    const std::array<double, 3> kernel{
        side, 1.0 - 2.0 * side, side};
    return convolve(convolve(kernel, kernel), kernel);
}

template <int DY, int DX, int Origin, std::size_t N>
void directional_fir(meyer::engine& e, const double* source,
                     double* destination,
                     const std::array<double, N>& coefficients) {
    e.P.run([&](int tid) {
        for (std::size_t y = std::size_t(tid); y < e.H;
             y += std::size_t(e.P.lanes())) {
            std::array<std::size_t, N> rows{};
            for (std::size_t tap = 0; tap < N; ++tap) {
                const int offset = int(tap) - Origin;
                rows[tap] = short_periodic_offset(y, offset * DY, e.H);
            }
            double* out = destination + y * e.W;
            constexpr int first_offset = -Origin * DX;
            constexpr int last_offset = (int(N) - 1 - Origin) * DX;
            constexpr int minimum_offset =
                first_offset < last_offset ? first_offset : last_offset;
            constexpr int maximum_offset =
                first_offset > last_offset ? first_offset : last_offset;
            const std::size_t interior_begin =
                minimum_offset < 0 ? std::size_t(-minimum_offset) : 0;
            const std::size_t interior_end = maximum_offset > 0
                ? e.W - std::size_t(maximum_offset) : e.W;
            for (std::size_t x = 0; x < interior_begin; ++x) {
                double sum = 0.0;
                for (std::size_t tap = 0; tap < N; ++tap) {
                    const int offset = int(tap) - Origin;
                    const std::size_t xx =
                        short_periodic_offset(x, offset * DX, e.W);
                    sum += coefficients[tap] * source[rows[tap] * e.W + xx];
                }
                out[x] = sum;
            }
            for (std::size_t x = interior_begin; x < interior_end; ++x) {
                double sum = 0.0;
                for (std::size_t tap = 0; tap < N; ++tap) {
                    const int offset = int(tap) - Origin;
                    sum += coefficients[tap] *
                        source[rows[tap] * e.W + std::size_t(
                            std::ptrdiff_t(x) + offset * DX)];
                }
                out[x] = sum;
            }
            for (std::size_t x = interior_end; x < e.W; ++x) {
                double sum = 0.0;
                for (std::size_t tap = 0; tap < N; ++tap) {
                    const int offset = int(tap) - Origin;
                    const std::size_t xx =
                        short_periodic_offset(x, offset * DX, e.W);
                    sum += coefficients[tap] * source[rows[tap] * e.W + xx];
                }
                out[x] = sum;
            }
        }
    });
}

void accumulate_absolute(meyer::engine& e, const double* response) {
    const std::size_t count = e.H * e.W;
    e.P.run([&](int tid) {
        std::size_t lo, hi;
        e.split(tid, count, lo, hi);
        for (std::size_t i = lo; i < hi; ++i)
            e.condition_gate[i] += std::fabs(response[i]);
    });
}

template <int DY, int DX>
void accumulate_difference(meyer::engine& e, const double* value) {
    e.P.run([&](int tid) {
        for (std::size_t y = std::size_t(tid); y < e.H;
             y += std::size_t(e.P.lanes())) {
            const std::size_t yn =
                meyer::engine::periodic_unit_step<DY>(y, e.H);
            for (std::size_t x = 0; x < e.W; ++x) {
                const std::size_t xn =
                    meyer::engine::periodic_unit_step<DX>(x, e.W);
                const std::size_t i = y * e.W + x;
                e.condition_gate[i] +=
                    std::fabs(value[yn * e.W + xn] - value[i]);
            }
        }
    });
}

template <int LongDY, int LongDX, int CrossDY, int CrossDX,
          int DifferenceDY, int DifferenceDX, int Radius>
void accumulate_collapsed(meyer::engine& e, const double* image,
                          double side) {
    e.directional_box_three_pass<LongDY, LongDX, Radius>(image);
    const std::array<double, 7> cross3 = cross_cube(side);
    if constexpr (CrossDY == DifferenceDY && CrossDX == DifferenceDX) {
        const std::array<double, 2> difference{-1.0, 1.0};
        const std::array<double, 8> fused = convolve(cross3, difference);
        directional_fir<CrossDY, CrossDX, 3>(
            e, e.u.data(), e.w.data(), fused);
        accumulate_absolute(e, e.w.data());
    } else {
        directional_fir<CrossDY, CrossDX, 3>(
            e, e.u.data(), e.w.data(), cross3);
        accumulate_difference<DifferenceDY, DifferenceDX>(e, e.w.data());
    }
}

void gate_collapsed(meyer::engine& e, const double* image) {
    const std::size_t count = e.H * e.W;
    std::memset(e.condition_gate.data(), 0, count * sizeof(double));
    accumulate_collapsed<0, 1, 1, 0, 1, 0, 3>(e, image, 0.125);
    accumulate_collapsed<1, 0, 0, 1, 0, 1, 3>(e, image, 0.125);
    accumulate_collapsed<1, 1, 1, -1, 1, 1, 2>(e, image, 0.0625);
    accumulate_collapsed<1, -1, 1, 1, 1, -1, 2>(e, image, 0.0625);
    e.normalize_condition_gate();
}

void gate_staged(meyer::engine& e, const double* image) {
    const std::size_t count = e.H * e.W;
    std::memset(e.condition_gate.data(), 0, count * sizeof(double));
    e.accumulate_condition_direction<0, 1, 1, 0, 1, 0, 3>(
        image, 0.125);
    e.accumulate_condition_direction<1, 0, 0, 1, 0, 1, 3>(
        image, 0.125);
    e.accumulate_condition_direction<1, 1, 1, -1, 1, 1, 2>(
        image, 0.0625);
    e.accumulate_condition_direction<1, -1, 1, 1, 1, -1, 2>(
        image, 0.0625);
    e.normalize_condition_gate();
}

template <int Radius>
struct box_ring_three {
    static constexpr std::size_t width = 2 * Radius + 1;
    std::array<double, width> first{};
    std::array<double, width> second{};
    std::array<double, width> third{};
    std::size_t position = 0;
    double sum_first = 0.0;
    double sum_second = 0.0;
    double sum_third = 0.0;

    double push(double value) {
        const double old_first = first[position];
        first[position] = value;
        sum_first += value - old_first;
        const double out_first = sum_first / double(width);

        const double old_second = second[position];
        second[position] = out_first;
        sum_second += out_first - old_second;
        const double out_second = sum_second / double(width);

        const double old_third = third[position];
        third[position] = out_second;
        sum_third += out_second - old_third;
        const double out_third = sum_third / double(width);

        if (++position == width) position = 0;
        return out_third;
    }
};

struct cross_ring_three {
    std::array<double, 3> first{};
    std::array<double, 3> second{};
    std::array<double, 3> third{};
    std::size_t position = 0;
    double side;
    double center;

    explicit cross_ring_three(double s) : side(s), center(1.0 - 2.0 * s) {}

    double stage(std::array<double, 3>& values, double input) const {
        values[position] = input;
        const double newest = values[position];
        const double middle = values[(position + 2) % 3];
        const double oldest = values[(position + 1) % 3];
        return side * oldest + center * middle + side * newest;
    }

    double push(double value) {
        value = stage(first, value);
        value = stage(second, value);
        value = stage(third, value);
        if (++position == 3) position = 0;
        return value;
    }
};

template <int DY, int DX, class State, int Radius>
void directional_ring_cascade(meyer::engine& e, const double* source,
                              double* destination,
                              double cross_side = 0.0) {
    std::size_t cycles, length;
    if constexpr (DY == 0) {
        cycles = e.H;
        length = e.W;
    } else if constexpr (DX == 0) {
        cycles = e.W;
        length = e.H;
    } else {
        cycles = std::gcd(e.H, e.W);
        length = (e.H / cycles) * e.W;
    }
    constexpr std::size_t warmup = 6 * Radius;
    constexpr std::size_t latency = 3 * Radius;
    e.P.run([&](int tid) {
        for (std::size_t cycle = std::size_t(tid); cycle < cycles;
             cycle += std::size_t(e.P.lanes())) {
            std::size_t sy, sx;
            if constexpr (DY == 0) {
                sy = cycle; sx = 0;
            } else if constexpr (DX == 0) {
                sy = 0; sx = cycle;
            } else {
                sy = 0; sx = cycle;
            }
            std::size_t iy = sy, ix = sx;
            std::size_t oy = sy, ox = sx;
            for (std::size_t step = 0; step < warmup - latency; ++step)
                e.advance_direction<DY, DX>(oy, ox);

            State state = [&]() {
                if constexpr (std::is_same<State, cross_ring_three>::value)
                    return State(cross_side);
                else
                    return State();
            }();
            for (std::size_t time = 0; time < warmup + length; ++time) {
                const double output = state.push(source[iy * e.W + ix]);
                e.advance_direction<DY, DX>(iy, ix);
                if (time >= warmup) {
                    destination[oy * e.W + ox] = output;
                    e.advance_direction<DY, DX>(oy, ox);
                }
            }
        }
    });
}

template <int LongDY, int LongDX, int CrossDY, int CrossDX,
          int DifferenceDY, int DifferenceDX, int Radius>
void accumulate_ring(meyer::engine& e, const double* image,
                     double side) {
    directional_ring_cascade<LongDY, LongDX, box_ring_three<Radius>, Radius>(
        e, image, e.u.data());
    directional_ring_cascade<CrossDY, CrossDX, cross_ring_three, 1>(
        e, e.u.data(), e.w.data(), side);
    accumulate_difference<DifferenceDY, DifferenceDX>(e, e.w.data());
}

void gate_ring(meyer::engine& e, const double* image) {
    const std::size_t count = e.H * e.W;
    std::memset(e.condition_gate.data(), 0, count * sizeof(double));
    accumulate_ring<0, 1, 1, 0, 1, 0, 3>(e, image, 0.125);
    accumulate_ring<1, 0, 0, 1, 0, 1, 3>(e, image, 0.125);
    accumulate_ring<1, 1, 1, -1, 1, 1, 2>(e, image, 0.0625);
    accumulate_ring<1, -1, 1, 1, 1, -1, 2>(e, image, 0.0625);
    e.normalize_condition_gate();
}

double maximum_error(const std::vector<double>& a,
                     const std::vector<double>& b) {
    double result = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
        result = std::max(result, std::abs(a[i] - b[i]));
    return result;
}

template <class Function>
double benchmark(Function&& function, int repeats) {
    std::vector<double> samples;
    for (int batch = 0; batch < 5; ++batch) {
        const auto start = clock_type::now();
        for (int iteration = 0; iteration < repeats; ++iteration)
            function();
        const auto stop = clock_type::now();
        samples.push_back(
            std::chrono::duration<double, std::milli>(stop - start).count()
            / repeats);
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

void run_shape(std::size_t height, std::size_t width, int threads,
               int repeats) {
    meyer::engine e;
    if (e.init(height, width, 0.05, 40.0, 1, 1, 0.0, threads) != BFFT_OK ||
        !e.set_solver(1)) {
        std::printf("%zux%zu T%d: plan failure\n", height, width, threads);
        return;
    }
    e.ensure_jump_measure_storage();
    const std::size_t count = height * width;
    std::vector<double> image(count), baseline;
    for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
            const std::size_t i = y * width + x;
            image[i] = 105.0 + 43.0 * double(x) / double(width)
                + 27.0 * std::sin(0.071 * double(x) + 0.029 * double(y))
                + (x > width / 2 ? 51.0 : 0.0);
        }
    }

    gate_staged(e, image.data());
    baseline = e.condition_gate;
    gate_collapsed(e, image.data());
    const std::vector<double> prototype = e.condition_gate;
    e.build_condition_gate_facr(image.data());
    const double collapsed_error = maximum_error(baseline, e.condition_gate);
    const double production_error = maximum_error(prototype, e.condition_gate);
    gate_ring(e, image.data());
    const double ring_error = maximum_error(baseline, e.condition_gate);

    const double staged_ms = benchmark(
        [&] { gate_staged(e, image.data()); }, repeats);
    const double collapsed_ms = benchmark(
        [&] { e.build_condition_gate_facr(image.data()); }, repeats);
    const double ring_ms = benchmark(
        [&] { gate_ring(e, image.data()); }, repeats);
    std::printf(
        "%zux%zu T%d: staged %.3f ms | collapsed %.3f ms (%.2fx, err %.2e) "
        "prod %.1e | ring %.3f ms (%.2fx, err %.2e)\n",
        height, width, threads, staged_ms, collapsed_ms,
        staged_ms / collapsed_ms, collapsed_error, production_error, ring_ms,
        staged_ms / ring_ms, ring_error);
}

}  // namespace

int main() {
    for (int threads : {1, 4, 6}) {
        run_shape(288, 512, threads, 20);
        run_shape(512, 300, threads, 20);
        run_shape(1024, 1280, threads, 4);
        if (threads != 1)
            run_shape(1080, 2048, threads, 2);
    }
    return 0;
}
