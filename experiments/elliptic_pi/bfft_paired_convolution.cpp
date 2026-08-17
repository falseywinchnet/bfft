#include "../../src/detail/bruun_dif_kernel.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

using bruun::complex_t;

void multiply_spectra(const complex_t* a, const complex_t* b,
                      complex_t* out, std::size_t bins) {
  for (std::size_t k = 0; k < bins; ++k) {
    out[k] = {a[k].re * b[k].re - a[k].im * b[k].im,
              a[k].re * b[k].im + a[k].im * b[k].re};
  }
}

void shared_products(const complex_t* a, const complex_t* b,
                     complex_t* product, complex_t* difference_square,
                     std::size_t bins) {
  for (std::size_t k = 0; k < bins; ++k) {
    const double dr = a[k].re - b[k].re;
    const double di = a[k].im - b[k].im;
    product[k] = {a[k].re * b[k].re - a[k].im * b[k].im,
                  a[k].re * b[k].im + a[k].im * b[k].re};
    difference_square[k] = {dr * dr - di * di, 2.0 * dr * di};
  }
}

void minimal_products(const complex_t* a, const complex_t* b,
                      complex_t* product, complex_t* difference_square,
                      std::size_t bins) {
  // U=A+B, V=A-B.  Two rank-2 complex squarings recover both requested
  // quadratic forms:
  //   AB=(U^2-V^2)/4, (A-B)^2=V^2.
  // z^2 uses (zr-zi)(zr+zi) + i(2*zr*zi), two real multiplies.
  for (std::size_t k = 0; k < bins; ++k) {
    const double ur = a[k].re + b[k].re;
    const double ui = a[k].im + b[k].im;
    const double vr = a[k].re - b[k].re;
    const double vi = a[k].im - b[k].im;
    const double u2r = (ur - ui) * (ur + ui);
    const double u2i = 2.0 * ur * ui;
    const double v2r = (vr - vi) * (vr + vi);
    const double v2i = 2.0 * vr * vi;
    product[k] = {0.25 * (u2r - v2r), 0.25 * (u2i - v2i)};
    difference_square[k] = {v2r, v2i};
  }
}

void streaming_step(complex_t* a, const complex_t* b,
                    complex_t* product, complex_t* difference_square,
                    complex_t* correction_sum, double correction_weight,
                    std::size_t bins) {
  // Besides producing the two quadratic forms, retain
  // FFT((a+b)/2) for the next AGM step and accumulate the period correction
  // before its inverse transform.  Thus a recurrent step needs only the new
  // transform of b and the inverse transform of a*b.
  for (std::size_t k = 0; k < bins; ++k) {
    const double ur = a[k].re + b[k].re;
    const double ui = a[k].im + b[k].im;
    const double vr = a[k].re - b[k].re;
    const double vi = a[k].im - b[k].im;
    const double u2r = (ur - ui) * (ur + ui);
    const double u2i = 2.0 * ur * ui;
    const double v2r = (vr - vi) * (vr + vi);
    const double v2i = 2.0 * vr * vi;
    product[k] = {0.25 * (u2r - v2r), 0.25 * (u2i - v2i)};
    difference_square[k] = {v2r, v2i};
    correction_sum[k].re += 0.25 * correction_weight * v2r;
    correction_sum[k].im += 0.25 * correction_weight * v2i;
    a[k] = {0.5 * ur, 0.5 * ui};
  }
}

template <typename Function>
double median_seconds(Function&& function, unsigned repeats) {
  function();
  std::vector<double> samples;
  samples.reserve(repeats);
  for (unsigned repeat = 0; repeat < repeats; ++repeat) {
    const auto begin = std::chrono::steady_clock::now();
    function();
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double>(end - begin).count());
  }
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

template <typename Setup, typename Function>
double median_seconds_with_setup(Setup&& setup, Function&& function,
                                 unsigned repeats) {
  setup();
  function();
  std::vector<double> samples;
  samples.reserve(repeats);
  for (unsigned repeat = 0; repeat < repeats; ++repeat) {
    setup();
    const auto begin = std::chrono::steady_clock::now();
    function();
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double>(end - begin).count());
  }
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
  int n = 1 << 18;
  unsigned repeats = 9;
  unsigned stream_steps = 16;
  if (argc > 1) n = std::atoi(argv[1]);
  if (argc > 2) repeats = static_cast<unsigned>(std::atoi(argv[2]));
  if (argc > 3) stream_steps = static_cast<unsigned>(std::atoi(argv[3]));
  if (n < 4 || (n & (n - 1)) != 0 || repeats == 0 || stream_steps == 0)
    return 2;

  bruun::DIF_RFFT_kernel plan;
  if (!plan.init(n)) return 3;
  const std::size_t bins = static_cast<std::size_t>(plan.bins());
  std::vector<double> a(n), b(n), difference(n), work(n), product_out(n),
      square_out(n), shared_product_out(n), shared_square_out(n);
  std::vector<double> minimal_product_out(n), minimal_square_out(n);
  std::vector<double> eager_a(n), eager_correction(n), stream_product_out(n),
      stream_correction_out(n), stream_numerator_out(n),
      stream_reference_numerator(n);
  std::vector<complex_t> a_spectrum(bins), b_spectrum(bins),
      difference_spectrum(bins), product_spectrum(bins),
      square_spectrum(bins);
  std::vector<complex_t> initial_a_spectrum(bins), stream_a_spectrum(bins),
      correction_spectrum(bins), numerator_spectrum(bins);
  for (int i = 0; i < n; ++i) {
    const double x = static_cast<double>(i + 1);
    a[i] = std::sin(0.013 * x) + 0.25 * std::cos(0.031 * x);
    b[i] = std::cos(0.017 * x) - 0.125 * std::sin(0.029 * x);
    difference[i] = a[i] - b[i];
  }

  auto separate = [&] {
    plan.forward_native(a.data(), a_spectrum.data(), work.data());
    plan.forward_native(b.data(), b_spectrum.data(), work.data());
    plan.forward_native(difference.data(), difference_spectrum.data(),
                        work.data());
    multiply_spectra(a_spectrum.data(), b_spectrum.data(),
                     product_spectrum.data(), bins);
    multiply_spectra(difference_spectrum.data(), difference_spectrum.data(),
                     square_spectrum.data(), bins);
    plan.inverse_native(product_spectrum.data(), product_out.data());
    plan.inverse_native(square_spectrum.data(), square_out.data());
  };
  auto shared = [&] {
    plan.forward_native(a.data(), a_spectrum.data(), work.data());
    plan.forward_native(b.data(), b_spectrum.data(), work.data());
    shared_products(a_spectrum.data(), b_spectrum.data(),
                    product_spectrum.data(), square_spectrum.data(), bins);
    plan.inverse_native(product_spectrum.data(), shared_product_out.data());
    plan.inverse_native(square_spectrum.data(), shared_square_out.data());
  };
  auto minimal = [&] {
    plan.forward_native(a.data(), a_spectrum.data(), work.data());
    plan.forward_native(b.data(), b_spectrum.data(), work.data());
    minimal_products(a_spectrum.data(), b_spectrum.data(),
                     product_spectrum.data(), square_spectrum.data(), bins);
    plan.inverse_native(product_spectrum.data(), minimal_product_out.data());
    plan.inverse_native(square_spectrum.data(), minimal_square_out.data());
  };

  // This pair isolates the transform schedule across several AGM-like steps.
  // `eager` materializes both convolutions each time.  `streamed` retains the
  // mean spectrum and delays the sum of all correction inverses until the end.
  auto eager_setup = [&] {
    std::copy(a.begin(), a.end(), eager_a.begin());
    std::fill(eager_correction.begin(), eager_correction.end(), 0.0);
  };
  auto eager = [&] {
    double weight = 1.0;
    for (unsigned step = 0; step < stream_steps; ++step) {
      plan.forward_native(eager_a.data(), a_spectrum.data(), work.data());
      plan.forward_native(b.data(), b_spectrum.data(), work.data());
      minimal_products(a_spectrum.data(), b_spectrum.data(),
                       product_spectrum.data(), square_spectrum.data(), bins);
      plan.inverse_native(product_spectrum.data(), stream_product_out.data());
      plan.inverse_native(square_spectrum.data(), stream_correction_out.data());
      for (int i = 0; i < n; ++i) {
        eager_correction[i] += 0.25 * weight * stream_correction_out[i];
        eager_a[i] = 0.5 * (eager_a[i] + b[i]);
      }
      weight *= 2.0;
    }
  };
  plan.forward_native(a.data(), initial_a_spectrum.data(), work.data());
  auto stream_setup = [&] {
    std::copy(initial_a_spectrum.begin(), initial_a_spectrum.end(),
              stream_a_spectrum.begin());
    std::fill(correction_spectrum.begin(), correction_spectrum.end(),
              complex_t{0.0, 0.0});
  };
  auto streamed_exact = [&] {
    double weight = 1.0;
    for (unsigned step = 0; step < stream_steps; ++step) {
      plan.forward_native(b.data(), b_spectrum.data(), work.data());
      streaming_step(stream_a_spectrum.data(), b_spectrum.data(),
                     product_spectrum.data(), square_spectrum.data(),
                     correction_spectrum.data(), weight, bins);
      plan.inverse_native(product_spectrum.data(), stream_product_out.data());
      weight *= 2.0;
    }
    for (std::size_t k = 0; k < bins; ++k) {
      // ((a+b)/2)^2 = a*b + ((a-b)/2)^2 at the last step.
      numerator_spectrum[k] = {
          product_spectrum[k].re + 0.25 * square_spectrum[k].re,
          product_spectrum[k].im + 0.25 * square_spectrum[k].im};
    }
    plan.inverse_native(correction_spectrum.data(), stream_correction_out.data());
    plan.inverse_native(numerator_spectrum.data(), stream_numerator_out.data());
  };
  auto streamed_guarded = [&] {
    double weight = 1.0;
    for (unsigned step = 0; step < stream_steps; ++step) {
      plan.forward_native(b.data(), b_spectrum.data(), work.data());
      streaming_step(stream_a_spectrum.data(), b_spectrum.data(),
                     product_spectrum.data(), square_spectrum.data(),
                     correction_spectrum.data(), weight, bins);
      plan.inverse_native(product_spectrum.data(), stream_product_out.data());
      weight *= 2.0;
    }
    plan.inverse_native(correction_spectrum.data(), stream_correction_out.data());
    // At a guarded AGM stop, d^2 is below half an output ulp, so the already
    // materialized a*b differs from ((a+b)/2)^2 by an unobservable d^2.
  };

  separate();
  shared();
  minimal();
  double max_product_error = 0;
  double max_square_error = 0;
  double max_product_scale = 0;
  double max_square_scale = 0;
  double max_minimal_product_error = 0;
  double max_minimal_square_error = 0;
  for (int i = 0; i < n; ++i) {
    max_product_error = std::max(
        max_product_error, std::abs(product_out[i] - shared_product_out[i]));
    max_square_error = std::max(
        max_square_error, std::abs(square_out[i] - shared_square_out[i]));
    max_product_scale =
        std::max(max_product_scale, std::abs(product_out[i]));
    max_square_scale = std::max(max_square_scale, std::abs(square_out[i]));
    max_minimal_product_error =
        std::max(max_minimal_product_error,
                 std::abs(product_out[i] - minimal_product_out[i]));
    max_minimal_square_error =
        std::max(max_minimal_square_error,
                 std::abs(square_out[i] - minimal_square_out[i]));
  }
  const double separate_seconds = median_seconds(separate, repeats);
  const double shared_seconds = median_seconds(shared, repeats);
  const double minimal_seconds = median_seconds(minimal, repeats);
  const double eager_seconds =
      median_seconds_with_setup(eager_setup, eager, repeats);
  const double streamed_seconds =
      median_seconds_with_setup(stream_setup, streamed_exact, repeats);
  const double guarded_streamed_seconds =
      median_seconds_with_setup(stream_setup, streamed_guarded, repeats);

  eager_setup();
  eager();
  stream_setup();
  streamed_exact();
  plan.forward_native(eager_a.data(), a_spectrum.data(), work.data());
  multiply_spectra(a_spectrum.data(), a_spectrum.data(),
                   numerator_spectrum.data(), bins);
  plan.inverse_native(numerator_spectrum.data(),
                      stream_reference_numerator.data());
  double max_stream_correction_error = 0.0;
  double max_stream_correction_scale = 0.0;
  double max_stream_numerator_error = 0.0;
  double max_stream_numerator_scale = 0.0;
  for (int i = 0; i < n; ++i) {
    max_stream_correction_error =
        std::max(max_stream_correction_error,
                 std::abs(eager_correction[i] - stream_correction_out[i]));
    max_stream_correction_scale =
        std::max(max_stream_correction_scale, std::abs(eager_correction[i]));
    max_stream_numerator_error =
        std::max(max_stream_numerator_error,
                 std::abs(stream_reference_numerator[i] -
                          stream_numerator_out[i]));
    max_stream_numerator_scale =
        std::max(max_stream_numerator_scale,
                 std::abs(stream_reference_numerator[i]));
  }
  std::cout << std::setprecision(9) << "{\n  \"n\": " << n
            << ",\n  \"repeats\": " << repeats
            << ",\n  \"stream_steps\": " << stream_steps
            << ",\n  \"backend\": \"" << bruun::simd_backend_name() << "\""
            << ",\n  \"separate_seconds\": " << separate_seconds
            << ",\n  \"shared_seconds\": " << shared_seconds
            << ",\n  \"shared_to_separate\": "
            << shared_seconds / separate_seconds
            << ",\n  \"minimal_seconds\": " << minimal_seconds
            << ",\n  \"minimal_to_separate\": "
            << minimal_seconds / separate_seconds
            << ",\n  \"minimal_to_shared\": "
            << minimal_seconds / shared_seconds
            << ",\n  \"eager_stream_seconds\": " << eager_seconds
            << ",\n  \"deferred_stream_seconds\": " << streamed_seconds
            << ",\n  \"deferred_to_eager\": " << streamed_seconds / eager_seconds
            << ",\n  \"guarded_deferred_seconds\": "
            << guarded_streamed_seconds
            << ",\n  \"guarded_deferred_to_eager\": "
            << guarded_streamed_seconds / eager_seconds
            << ",\n  \"relative_stream_correction_error\": "
            << max_stream_correction_error /
                   std::max(1.0, max_stream_correction_scale)
            << ",\n  \"relative_stream_numerator_error\": "
            << max_stream_numerator_error /
                   std::max(1.0, max_stream_numerator_scale)
            << ",\n  \"max_product_error\": " << max_product_error
            << ",\n  \"max_square_error\": " << max_square_error
            << ",\n  \"relative_product_error\": "
            << max_product_error / std::max(1.0, max_product_scale)
            << ",\n  \"relative_square_error\": "
            << max_square_error / std::max(1.0, max_square_scale)
            << ",\n  \"relative_minimal_product_error\": "
            << max_minimal_product_error / std::max(1.0, max_product_scale)
            << ",\n  \"relative_minimal_square_error\": "
            << max_minimal_square_error / std::max(1.0, max_square_scale)
            << "\n}\n";
}
