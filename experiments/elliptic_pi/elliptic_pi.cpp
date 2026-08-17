#include <gmp.h>
#include <mpfr.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <iomanip>
#include <iostream>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

constexpr mpfr_rnd_t RND = MPFR_RNDN;

struct Real {
  mpfr_t value;
  explicit Real(mpfr_prec_t p) { mpfr_init2(value, p); }
  Real(const Real& other) {
    mpfr_init2(value, mpfr_get_prec(other.value));
    mpfr_set(value, other.value, RND);
  }
  Real& operator=(const Real& other) {
    if (this != &other) mpfr_set(value, other.value, RND);
    return *this;
  }
  ~Real() { mpfr_clear(value); }
  mpfr_prec_t precision() const { return mpfr_get_prec(value); }
};

void quartic_newton_step(Real& x, const Real& y) {
  const mpfr_prec_t p = x.precision();
  Real y4(p), one_plus(p), one_plus2(p), one_plus3(p), one_plus4(p), x2(p),
      polynomial(p), derivative(p), temp(p);
  mpfr_sqr(y4.value, y.value, RND);
  mpfr_sqr(y4.value, y4.value, RND);
  mpfr_add_ui(one_plus.value, x.value, 1, RND);
  mpfr_sqr(one_plus2.value, one_plus.value, RND);
  mpfr_mul(one_plus3.value, one_plus2.value, one_plus.value, RND);
  mpfr_sqr(one_plus4.value, one_plus2.value, RND);
  mpfr_sqr(x2.value, x.value, RND);

  // P = y^4(1+x)^4 - 8x(1+x^2).
  mpfr_mul(polynomial.value, y4.value, one_plus4.value, RND);
  mpfr_add_ui(temp.value, x2.value, 1, RND);
  mpfr_mul(temp.value, temp.value, x.value, RND);
  mpfr_mul_ui(temp.value, temp.value, 8, RND);
  mpfr_sub(polynomial.value, polynomial.value, temp.value, RND);

  // P' = 4y^4(1+x)^3 - 8(1+3x^2).
  mpfr_mul(derivative.value, y4.value, one_plus3.value, RND);
  mpfr_mul_ui(derivative.value, derivative.value, 4, RND);
  mpfr_mul_ui(temp.value, x2.value, 3, RND);
  mpfr_add_ui(temp.value, temp.value, 1, RND);
  mpfr_mul_ui(temp.value, temp.value, 8, RND);
  mpfr_sub(derivative.value, derivative.value, temp.value, RND);
  mpfr_div(temp.value, polynomial.value, derivative.value, RND);
  mpfr_sub(x.value, x.value, temp.value, RND);
}

void cubic_newton_step(Real& x, const Real& s) {
  const mpfr_prec_t p = x.precision();
  Real s3(p), one_plus(p), one_plus2(p), one_plus3(p), x2(p), polynomial(p),
      derivative(p), temp(p);
  mpfr_sqr(s3.value, s.value, RND);
  mpfr_mul(s3.value, s3.value, s.value, RND);
  mpfr_mul_ui(one_plus.value, x.value, 2, RND);
  mpfr_add_ui(one_plus.value, one_plus.value, 1, RND);
  mpfr_sqr(one_plus2.value, one_plus.value, RND);
  mpfr_mul(one_plus3.value, one_plus2.value, one_plus.value, RND);
  mpfr_sqr(x2.value, x.value, RND);

  // P = s^3(1+2x)^3 - 9x(1+x+x^2).
  mpfr_mul(polynomial.value, s3.value, one_plus3.value, RND);
  mpfr_add(temp.value, x.value, x2.value, RND);
  mpfr_add_ui(temp.value, temp.value, 1, RND);
  mpfr_mul(temp.value, temp.value, x.value, RND);
  mpfr_mul_ui(temp.value, temp.value, 9, RND);
  mpfr_sub(polynomial.value, polynomial.value, temp.value, RND);

  // P' = 6s^3(1+2x)^2 - 9(1+2x+3x^2).
  mpfr_mul(derivative.value, s3.value, one_plus2.value, RND);
  mpfr_mul_ui(derivative.value, derivative.value, 6, RND);
  mpfr_mul_ui(temp.value, x2.value, 3, RND);
  mpfr_mul_ui(one_plus3.value, x.value, 2, RND);
  mpfr_add(temp.value, temp.value, one_plus3.value, RND);
  mpfr_add_ui(temp.value, temp.value, 1, RND);
  mpfr_mul_ui(temp.value, temp.value, 9, RND);
  mpfr_sub(derivative.value, derivative.value, temp.value, RND);
  mpfr_div(temp.value, polynomial.value, derivative.value, RND);
  mpfr_sub(x.value, x.value, temp.value, RND);
}

Real solve_quartic_map_legacy(const Real& y, mpfr_prec_t target) {
  mpfr_prec_t p = std::min<mpfr_prec_t>(96, target);
  Real local_y(p), x(p), temp(p);
  mpfr_set(local_y.value, y.value, RND);
  mpfr_sqr(temp.value, local_y.value, RND);
  mpfr_sqr(temp.value, temp.value, RND);
  mpfr_div_ui(x.value, temp.value, 8, RND);
  for (int i = 0; i < 7; ++i) quartic_newton_step(x, local_y);
  while (p < target) {
    p = std::min<mpfr_prec_t>(target, 2 * p);
    mpfr_prec_round(x.value, p, RND);
    mpfr_prec_round(local_y.value, p, RND);
    mpfr_set(local_y.value, y.value, RND);
    quartic_newton_step(x, local_y);
  }
  return x;
}

// Multiplication-only Newton on the reciprocal fourth-root auxiliary.
// If c=y^4 and r^4=1-c, then the small quartic-map root is recovered without
// cancellation from
//
//   x = c / ((1+r)^2(1+r^2)).
//
// Solve (1-c)s^4=1 for s=1/r via s <- s(5-(1-c)s^4)/4, then r=(1-c)s^3.
// This is still a fixed polynomial root selection, but it replaces the
// division in every staged x-Newton step with multiplications and uses only
// one full division for the final rational reconstruction.
void inverse_fourth_root_newton_step(Real& reciprocal_root, const Real& b,
                                     Real& square, Real& fourth,
                                     Real& temp) {
  mpfr_sqr(square.value, reciprocal_root.value, RND);
  mpfr_sqr(fourth.value, square.value, RND);
  mpfr_mul(temp.value, b.value, fourth.value, RND);
  mpfr_ui_sub(temp.value, 5, temp.value, RND);
  mpfr_mul(reciprocal_root.value, reciprocal_root.value, temp.value, RND);
  mpfr_div_2ui(reciprocal_root.value, reciprocal_root.value, 2, RND);
}

Real solve_quartic_map_from_fourth(const Real& coefficient,
                                   mpfr_prec_t target) {
  Real b(target), result(target);
  if (mpfr_zero_p(coefficient.value)) {
    mpfr_set_zero(result.value, 0);
    return result;
  }
  mpfr_ui_sub(b.value, 1, coefficient.value, RND);

  // The final multiplication by c converts a relative error of 2^-m in r
  // into an absolute error of about 2^(exp(c)-m) in x.
  const mpfr_exp_t coefficient_exponent = mpfr_get_exp(coefficient.value);
  const mpfr_exp_t exponent = coefficient_exponent - 3;
  const mpfr_prec_t seed_bits = std::max<mpfr_prec_t>(
      2, static_cast<mpfr_prec_t>(-coefficient_exponent));
  if (2 * seed_bits >= target + 32) {
    // x=c/8+O(c^2); the correction is already below all guard bits.
    mpfr_div_2ui(result.value, coefficient.value, 3, RND);
    return result;
  }
  const mpfr_prec_t required = std::max<mpfr_prec_t>(
      64, std::min<mpfr_prec_t>(target, target + exponent + 16));
  mpfr_prec_t p = std::min<mpfr_prec_t>(
      required, std::max<mpfr_prec_t>(96, seed_bits + 16));
  Real local_b(p), local_coefficient(p), reciprocal_root(p), square(p),
      fourth(p), temp(p), r(p), denominator(p);
  mpfr_set(local_b.value, b.value, RND);
  mpfr_set_ui(reciprocal_root.value, 1, RND);
  int initial_steps = 0;
  mpfr_prec_t estimated_bits = seed_bits;
  while (estimated_bits < p) {
    ++initial_steps;
    estimated_bits = std::min<mpfr_prec_t>(p, 2 * estimated_bits);
  }
  initial_steps = std::max(1, initial_steps);
  for (int i = 0; i < initial_steps; ++i) {
    inverse_fourth_root_newton_step(reciprocal_root, local_b, square, fourth,
                                    temp);
  }
  while (p < required) {
    p = std::min<mpfr_prec_t>(required, 2 * p);
    mpfr_prec_round(local_b.value, p, RND);
    mpfr_prec_round(local_coefficient.value, p, RND);
    mpfr_prec_round(reciprocal_root.value, p, RND);
    mpfr_prec_round(square.value, p, RND);
    mpfr_prec_round(fourth.value, p, RND);
    mpfr_prec_round(temp.value, p, RND);
    mpfr_prec_round(r.value, p, RND);
    mpfr_prec_round(denominator.value, p, RND);
    mpfr_set(local_b.value, b.value, RND);
    inverse_fourth_root_newton_step(reciprocal_root, local_b, square, fourth,
                                    temp);
  }

  mpfr_sqr(square.value, reciprocal_root.value, RND);
  mpfr_mul(r.value, square.value, reciprocal_root.value, RND);
  mpfr_mul(r.value, r.value, local_b.value, RND);
  mpfr_sqr(square.value, r.value, RND);
  mpfr_add_ui(fourth.value, r.value, 1, RND);
  mpfr_sqr(denominator.value, fourth.value, RND);
  mpfr_add_ui(temp.value, square.value, 1, RND);
  mpfr_mul(denominator.value, denominator.value, temp.value, RND);
  mpfr_set(local_coefficient.value, coefficient.value, RND);
  mpfr_div(r.value, local_coefficient.value, denominator.value, RND);
  mpfr_set(result.value, r.value, RND);
  return result;
}

Real solve_cubic_map(const Real& s, mpfr_prec_t target) {
  mpfr_prec_t p = std::min<mpfr_prec_t>(96, target);
  Real local_s(p), x(p), temp(p);
  mpfr_set(local_s.value, s.value, RND);
  mpfr_sqr(temp.value, local_s.value, RND);
  mpfr_mul(temp.value, temp.value, local_s.value, RND);
  mpfr_div_ui(x.value, temp.value, 9, RND);
  for (int i = 0; i < 7; ++i) cubic_newton_step(x, local_s);
  while (p < target) {
    p = std::min<mpfr_prec_t>(target, 2 * p);
    mpfr_prec_round(x.value, p, RND);
    mpfr_prec_round(local_s.value, p, RND);
    mpfr_set(local_s.value, s.value, RND);
    cubic_newton_step(x, local_s);
  }
  return x;
}

void lambda3_normalized_newton_step(Real& ratio, const Real& u) {
  const mpfr_prec_t p = ratio.precision();
  Real u2(p), u3(p), u6(p), u8(p), c1(p), c2(p), c3(p), c4(p),
      temp(p), factor(p), polynomial(p), derivative(p);
  mpfr_sqr(u2.value, u.value, RND);
  mpfr_mul(u3.value, u2.value, u.value, RND);
  mpfr_sqr(u6.value, u3.value, RND);
  mpfr_sqr(u8.value, u2.value, RND);
  mpfr_sqr(u8.value, u8.value, RND);

  // c1 = -(33u^2-96u+64)/2^6.
  mpfr_mul_ui(c1.value, u2.value, 33, RND);
  mpfr_mul_ui(temp.value, u.value, 96, RND);
  mpfr_sub(c1.value, c1.value, temp.value, RND);
  mpfr_add_ui(c1.value, c1.value, 64, RND);
  mpfr_neg(c1.value, c1.value, RND);
  mpfr_div_2ui(c1.value, c1.value, 6, RND);

  // c2 = 3u^3(64u^2-127u+64)/2^15.
  mpfr_mul_ui(factor.value, u2.value, 64, RND);
  mpfr_mul_ui(temp.value, u.value, 127, RND);
  mpfr_sub(factor.value, factor.value, temp.value, RND);
  mpfr_add_ui(factor.value, factor.value, 64, RND);
  mpfr_mul(c2.value, u3.value, factor.value, RND);
  mpfr_mul_ui(c2.value, c2.value, 3, RND);
  mpfr_div_2ui(c2.value, c2.value, 15, RND);

  // c3 = -u^6(64u^2-96u+33)/2^22; c4=u^8/2^32.
  mpfr_mul_ui(factor.value, u2.value, 64, RND);
  mpfr_mul_ui(temp.value, u.value, 96, RND);
  mpfr_sub(factor.value, factor.value, temp.value, RND);
  mpfr_add_ui(factor.value, factor.value, 33, RND);
  mpfr_mul(c3.value, u6.value, factor.value, RND);
  mpfr_neg(c3.value, c3.value, RND);
  mpfr_div_2ui(c3.value, c3.value, 22, RND);
  mpfr_div_2ui(c4.value, u8.value, 32, RND);

  // Q(r)=1+c1*r+c2*r^2+c3*r^3+c4*r^4.
  mpfr_mul(polynomial.value, c4.value, ratio.value, RND);
  mpfr_add(polynomial.value, polynomial.value, c3.value, RND);
  mpfr_mul(polynomial.value, polynomial.value, ratio.value, RND);
  mpfr_add(polynomial.value, polynomial.value, c2.value, RND);
  mpfr_mul(polynomial.value, polynomial.value, ratio.value, RND);
  mpfr_add(polynomial.value, polynomial.value, c1.value, RND);
  mpfr_mul(polynomial.value, polynomial.value, ratio.value, RND);
  mpfr_add_ui(polynomial.value, polynomial.value, 1, RND);

  mpfr_mul_ui(derivative.value, c4.value, 4, RND);
  mpfr_mul(derivative.value, derivative.value, ratio.value, RND);
  mpfr_mul_ui(temp.value, c3.value, 3, RND);
  mpfr_add(derivative.value, derivative.value, temp.value, RND);
  mpfr_mul(derivative.value, derivative.value, ratio.value, RND);
  mpfr_mul_ui(temp.value, c2.value, 2, RND);
  mpfr_add(derivative.value, derivative.value, temp.value, RND);
  mpfr_mul(derivative.value, derivative.value, ratio.value, RND);
  mpfr_add(derivative.value, derivative.value, c1.value, RND);

  mpfr_div(temp.value, polynomial.value, derivative.value, RND);
  mpfr_sub(ratio.value, ratio.value, temp.value, RND);
}

Real solve_lambda3_normalized(const Real& u, mpfr_prec_t target,
                              Real* ratio_output = nullptr) {
  mpfr_prec_t p = std::min<mpfr_prec_t>(96, target);
  Real local_u(p), ratio(p), u3(p), result(p);
  mpfr_set(local_u.value, u.value, RND);
  mpfr_set_ui(ratio.value, 1, RND);
  for (int i = 0; i < 7; ++i)
    lambda3_normalized_newton_step(ratio, local_u);
  while (p < target) {
    p = std::min<mpfr_prec_t>(target, 2 * p);
    mpfr_prec_round(local_u.value, p, RND);
    mpfr_prec_round(ratio.value, p, RND);
    mpfr_prec_round(u3.value, p, RND);
    mpfr_prec_round(result.value, p, RND);
    mpfr_set(local_u.value, u.value, RND);
    lambda3_normalized_newton_step(ratio, local_u);
  }
  mpfr_sqr(u3.value, local_u.value, RND);
  mpfr_mul(u3.value, u3.value, local_u.value, RND);
  mpfr_mul(result.value, u3.value, ratio.value, RND);
  mpfr_div_2ui(result.value, result.value, 8, RND);
  if (ratio_output != nullptr) mpfr_set(ratio_output->value, ratio.value, RND);
  return result;
}

Real solve_initial_root(mpfr_prec_t target, bool cubic) {
  mpfr_prec_t p = std::min<mpfr_prec_t>(96, target);
  Real x(p), polynomial(p), derivative(p), temp(p);
  if (cubic)
    mpfr_set_d(x.value, 1.0 / 3.0, RND);
  else
    mpfr_set_d(x.value, 0.4, RND);
  bool initial_stage = true;
  while (true) {
    const int iterations = initial_stage ? 7 : 1;
    for (int i = 0; i < iterations; ++i) {
      mpfr_sqr(polynomial.value, x.value, RND);
      mpfr_mul_ui(polynomial.value, polynomial.value, cubic ? 2 : 1, RND);
      mpfr_mul_ui(derivative.value, x.value, 2, RND);
      mpfr_add(polynomial.value, polynomial.value, derivative.value, RND);
      mpfr_sub_ui(polynomial.value, polynomial.value, 1, RND);
      mpfr_mul_ui(derivative.value, x.value, cubic ? 4 : 2, RND);
      mpfr_add_ui(derivative.value, derivative.value, 2, RND);
      mpfr_div(temp.value, polynomial.value, derivative.value, RND);
      mpfr_sub(x.value, x.value, temp.value, RND);
    }
    if (p == target) break;
    p = std::min<mpfr_prec_t>(target, 2 * p);
    mpfr_prec_round(x.value, p, RND);
    mpfr_prec_round(polynomial.value, p, RND);
    mpfr_prec_round(derivative.value, p, RND);
    mpfr_prec_round(temp.value, p, RND);
    initial_stage = false;
  }
  return x;
}

Real solve_quartic_initial_root(mpfr_prec_t target) {
  mpfr_prec_t p = std::min<mpfr_prec_t>(96, target);
  Real reciprocal_sqrt(p), square(p), temp(p), result(target);
  mpfr_set_d(reciprocal_sqrt.value, 0.7, RND);
  bool initial_stage = true;
  while (true) {
    const int iterations = initial_stage ? 5 : 1;
    for (int i = 0; i < iterations; ++i) {
      // Newton for 2s^2=1: s <- s(3-2s^2)/2.
      mpfr_sqr(square.value, reciprocal_sqrt.value, RND);
      mpfr_mul_2ui(temp.value, square.value, 1, RND);
      mpfr_ui_sub(temp.value, 3, temp.value, RND);
      mpfr_mul(reciprocal_sqrt.value, reciprocal_sqrt.value, temp.value, RND);
      mpfr_div_2ui(reciprocal_sqrt.value, reciprocal_sqrt.value, 1, RND);
    }
    if (p == target) break;
    p = std::min<mpfr_prec_t>(target, 2 * p);
    mpfr_prec_round(reciprocal_sqrt.value, p, RND);
    mpfr_prec_round(square.value, p, RND);
    mpfr_prec_round(temp.value, p, RND);
    initial_stage = false;
  }
  mpfr_mul_2ui(result.value, reciprocal_sqrt.value, 1, RND);
  mpfr_sub_ui(result.value, result.value, 1, RND);
  return result;
}

unsigned quartic_iterations(unsigned long bits) {
  return std::max(1, static_cast<int>(std::ceil(std::log(bits / 8.0) /
                                                std::log(4.0))) +
                         1);
}

unsigned cubic_iterations(unsigned long bits) {
  return std::max(1, static_cast<int>(std::ceil(std::log(bits / 5.0) /
                                                std::log(3.0))) +
                         1);
}

Real pi_quartic_implicit_impl(unsigned long bits, bool optimized) {
  const mpfr_prec_t p = bits + (optimized ? 32 : 64);
  Real y = optimized ? solve_quartic_initial_root(p)
                     : solve_initial_root(p, false);
  Real z(p), one_plus(p), one_plus2(p), one_plus4(p), y2(p), y4(p), factor(p),
      correction(p), result(p);
  mpfr_sqr(y2.value, y.value, RND);
  mpfr_mul_ui(z.value, y2.value, 2, RND);
  mpfr_sqr(y4.value, y2.value, RND);
  const unsigned count = quartic_iterations(bits);
  for (unsigned n = 0; n < count; ++n) {
    y = optimized ? solve_quartic_map_from_fourth(y4, p)
                  : solve_quartic_map_legacy(y, p);
    mpfr_add_ui(one_plus.value, y.value, 1, RND);
    mpfr_sqr(one_plus2.value, one_plus.value, RND);
    mpfr_sqr(one_plus4.value, one_plus2.value, RND);
    mpfr_mul(z.value, z.value, one_plus4.value, RND);
    mpfr_sqr(y2.value, y.value, RND);
    mpfr_add(factor.value, one_plus.value, y2.value, RND);
    mpfr_mul(correction.value, y.value, factor.value, RND);
    mpfr_mul_2ui(correction.value, correction.value, 2 * n + 3, RND);
    mpfr_sub(z.value, z.value, correction.value, RND);
    if (optimized && n + 1 < count) mpfr_sqr(y4.value, y2.value, RND);
  }
  mpfr_ui_div(result.value, 1, z.value, RND);
  return result;
}

Real pi_quartic_implicit(unsigned long bits) {
  return pi_quartic_implicit_impl(bits, true);
}

Real pi_quartic_implicit_legacy(unsigned long bits) {
  return pi_quartic_implicit_impl(bits, false);
}

enum class QuarticRootKernel { RootN, TwoSqrt };

Real pi_quartic_radical_impl(unsigned long bits, QuarticRootKernel kernel,
                             bool stable_reconstruction) {
  const mpfr_prec_t p = bits + 64;
  Real y(p), z(p), coefficient(p), root(p), one_minus(p), one_plus(p),
      one_plus2(p), one_plus4(p), root2(p), denominator(p), y2(p), factor(p),
      correction(p), result(p);
  mpfr_set_ui(y.value, 2, RND);
  mpfr_sqrt(y.value, y.value, RND);
  mpfr_sub_ui(y.value, y.value, 1, RND);
  mpfr_sqr(z.value, y.value, RND);
  mpfr_mul_ui(z.value, z.value, 2, RND);
  const unsigned count = quartic_iterations(bits);
  for (unsigned n = 0; n < count; ++n) {
    mpfr_sqr(y2.value, y.value, RND);
    mpfr_sqr(coefficient.value, y2.value, RND);
    mpfr_ui_sub(root.value, 1, coefficient.value, RND);
    if (kernel == QuarticRootKernel::RootN) {
      mpfr_rootn_ui(root.value, root.value, 4, RND);
    } else {
      mpfr_sqrt(root.value, root.value, RND);
      mpfr_sqrt(root.value, root.value, RND);
    }
    mpfr_add_ui(one_plus.value, root.value, 1, RND);
    if (stable_reconstruction) {
      // 1-r = c/((1+r)(1+r^2)) for r^4=1-c, hence
      // y=(1-r)/(1+r)=c/((1+r)^2(1+r^2)).
      mpfr_sqr(one_plus2.value, one_plus.value, RND);
      mpfr_sqr(root2.value, root.value, RND);
      mpfr_add_ui(root2.value, root2.value, 1, RND);
      mpfr_mul(denominator.value, one_plus2.value, root2.value, RND);
      mpfr_div(y.value, coefficient.value, denominator.value, RND);
    } else {
      mpfr_ui_sub(one_minus.value, 1, root.value, RND);
      mpfr_div(y.value, one_minus.value, one_plus.value, RND);
    }
    mpfr_add_ui(one_plus.value, y.value, 1, RND);
    mpfr_sqr(one_plus2.value, one_plus.value, RND);
    mpfr_sqr(one_plus4.value, one_plus2.value, RND);
    mpfr_mul(z.value, z.value, one_plus4.value, RND);
    mpfr_sqr(y2.value, y.value, RND);
    mpfr_add(factor.value, one_plus.value, y2.value, RND);
    mpfr_mul(correction.value, y.value, factor.value, RND);
    mpfr_mul_2ui(correction.value, correction.value, 2 * n + 3, RND);
    mpfr_sub(z.value, z.value, correction.value, RND);
  }
  mpfr_ui_div(result.value, 1, z.value, RND);
  return result;
}

Real pi_quartic_radical(unsigned long bits) {
  return pi_quartic_radical_impl(bits, QuarticRootKernel::RootN, false);
}

Real pi_quartic_rootn_stable(unsigned long bits) {
  return pi_quartic_radical_impl(bits, QuarticRootKernel::RootN, true);
}

Real pi_quartic_two_sqrt(unsigned long bits) {
  return pi_quartic_radical_impl(bits, QuarticRootKernel::TwoSqrt, false);
}

Real pi_quartic_two_sqrt_stable(unsigned long bits) {
  return pi_quartic_radical_impl(bits, QuarticRootKernel::TwoSqrt, true);
}

Real pi_cubic_implicit(unsigned long bits) {
  const mpfr_prec_t p = bits + 64;
  Real s = solve_initial_root(p, true);
  Real a(p), power(p), r(p), r2(p), correction(p), result(p);
  mpfr_set_ui(a.value, 1, RND);
  mpfr_div_ui(a.value, a.value, 3, RND);
  mpfr_set_ui(power.value, 1, RND);
  const unsigned count = cubic_iterations(bits);
  for (unsigned n = 0; n < count; ++n) {
    s = solve_cubic_map(s, p);
    mpfr_mul_ui(r.value, s.value, 2, RND);
    mpfr_add_ui(r.value, r.value, 1, RND);
    mpfr_sqr(r2.value, r.value, RND);
    mpfr_sub_ui(correction.value, r2.value, 1, RND);
    mpfr_mul(correction.value, correction.value, power.value, RND);
    mpfr_mul(a.value, a.value, r2.value, RND);
    mpfr_sub(a.value, a.value, correction.value, RND);
    mpfr_mul_ui(power.value, power.value, 3, RND);
  }
  mpfr_ui_div(result.value, 1, a.value, RND);
  return result;
}

Real pi_lambda3_terminal_nome(unsigned long bits) {
  const mpfr_prec_t p = bits + 64;
  Real u(p), degree_power(p), result(p);
  mpfr_set_ui(u.value, 1, RND);
  mpfr_div_ui(u.value, u.value, 2, RND);
  mpfr_set_ui(degree_power.value, 1, RND);
  unsigned long threshold = bits + 128;
  unsigned long power = 1;
  // pi/log(2)>4, so 4*3^N target bits is a conservative cusp bound.
  while (power <= threshold / 4 + 1) {
    u = solve_lambda3_normalized(u, p);
    mpfr_mul_ui(degree_power.value, degree_power.value, 3, RND);
    if (power > (~0UL) / 3) break;
    power *= 3;
  }
  mpfr_div_2ui(result.value, u.value, 4, RND);
  mpfr_log(result.value, result.value, RND);
  mpfr_neg(result.value, result.value, RND);
  mpfr_div(result.value, result.value, degree_power.value, RND);
  return result;
}

[[maybe_unused]] Real pi_lambda3_incremental_nome(unsigned long bits) {
  const mpfr_prec_t p = bits + 64;
  Real u(p), ratio(p), delta(p), degree_power(p), correction(p), result(p);
  mpfr_set_ui(u.value, 1, RND);
  mpfr_div_ui(u.value, u.value, 2, RND);
  mpfr_set_ui(degree_power.value, 1, RND);
  mpfr_const_log2(result.value, RND);
  mpfr_mul_ui(result.value, result.value, 5, RND);  // log(32)

  unsigned long threshold = bits + 128;
  unsigned long power = 1;
  while (power <= threshold / 4 + 1) {
    u = solve_lambda3_normalized(u, p, &ratio);
    mpfr_mul_ui(degree_power.value, degree_power.value, 3, RND);
    mpfr_sub_ui(delta.value, ratio.value, 1, RND);
    if (!mpfr_zero_p(delta.value)) {
      // If delta has exponent e and the weight 3^(j+1) has exponent w,
      // m=bits+e-w relative bits suffice for n-bit absolute accuracy.  Keep
      // 48 guard bits for rounding and accumulation.  Storing delta rather
      // than ratio is essential when delta is much smaller than 2^-m.
      const mpfr_exp_t delta_exponent = mpfr_get_exp(delta.value);
      const mpfr_exp_t weight_exponent = mpfr_get_exp(degree_power.value);
      const mpfr_exp_t requested = static_cast<mpfr_exp_t>(bits) + 48
                                   + delta_exponent - weight_exponent;
      const mpfr_prec_t local_precision = std::max<mpfr_prec_t>(
          64, std::min<mpfr_prec_t>(
                  p, static_cast<mpfr_prec_t>(std::max<mpfr_exp_t>(64,
                                                                  requested))));
      Real local_delta(local_precision), local_weight(local_precision),
          local_log(local_precision);
      mpfr_set(local_delta.value, delta.value, RND);
      mpfr_set(local_weight.value, degree_power.value, RND);
      mpfr_log1p(local_log.value, local_delta.value, RND);
      mpfr_div(local_log.value, local_log.value, local_weight.value, RND);
      mpfr_set(correction.value, local_log.value, RND);
      mpfr_sub(result.value, result.value, correction.value, RND);
    }
    if (power > (~0UL) / 3) break;
    power *= 3;
  }
  return result;
}

Real pi_agm_legacy(unsigned long bits) {
  const mpfr_prec_t p = bits + 32;
  Real a(p), b(p), t(p), power(p), next_a(p), next_b(p), delta(p), temp(p),
      result(p);
  mpfr_set_ui(a.value, 1, RND);
  mpfr_set_ui(b.value, 2, RND);
  mpfr_sqrt(b.value, b.value, RND);
  mpfr_ui_div(b.value, 1, b.value, RND);
  mpfr_set_ui(t.value, 1, RND);
  mpfr_div_ui(t.value, t.value, 4, RND);
  mpfr_set_ui(power.value, 1, RND);
  const unsigned count = static_cast<unsigned>(std::ceil(std::log2(bits))) + 1;
  for (unsigned i = 0; i < count; ++i) {
    mpfr_add(next_a.value, a.value, b.value, RND);
    mpfr_div_ui(next_a.value, next_a.value, 2, RND);
    mpfr_mul(next_b.value, a.value, b.value, RND);
    mpfr_sqrt(next_b.value, next_b.value, RND);
    mpfr_sub(delta.value, a.value, next_a.value, RND);
    mpfr_sqr(delta.value, delta.value, RND);
    mpfr_mul(delta.value, delta.value, power.value, RND);
    mpfr_sub(t.value, t.value, delta.value, RND);
    mpfr_mul_ui(power.value, power.value, 2, RND);
    a = next_a;
    b = next_b;
  }
  mpfr_add(temp.value, a.value, b.value, RND);
  mpfr_sqr(temp.value, temp.value, RND);
  mpfr_mul_ui(t.value, t.value, 4, RND);
  mpfr_div(result.value, temp.value, t.value, RND);
  return result;
}

enum class AgmRootKernel {
  DirectSqrt,
  ReciprocalSqrt,
  WarmReciprocalNewton,
  ReducedDefect,
  HybridDefect
};

mpfr_prec_t reduced_defect_precision(const Real& half_difference,
                                     mpfr_prec_t target) {
  const mpfr_exp_t defect_exponent =
      2 * mpfr_get_exp(half_difference.value) - 1;
  return std::min<mpfr_prec_t>(
      target, std::max<mpfr_prec_t>(64, target + defect_exponent + 8));
}

void reduced_defect_geometric_mean(Real& geometric_mean, const Real& mean,
                                   const Real& half_difference,
                                   const Real& difference_squared,
                                   mpfr_prec_t target) {
  // For m=(a+b)/2 and d=(a-b)/2, this cancellation-free defect formula
  // permits the nonlinear work to lose precision as d collapses:
  //   sqrt(ab) = m - d^2/(m + sqrt(m^2-d^2)).
  const mpfr_prec_t local_precision =
      reduced_defect_precision(half_difference, target);
  Real local_mean(local_precision), local_difference_squared(local_precision),
      radicand(local_precision), root(local_precision),
      denominator(local_precision), defect(local_precision);
  mpfr_set(local_mean.value, mean.value, RND);
  mpfr_set(local_difference_squared.value, difference_squared.value, RND);
  mpfr_sqr(radicand.value, local_mean.value, RND);
  mpfr_sub(radicand.value, radicand.value, local_difference_squared.value,
           RND);
  mpfr_sqrt(root.value, radicand.value, RND);
  mpfr_add(denominator.value, local_mean.value, root.value, RND);
  mpfr_div(defect.value, local_difference_squared.value, denominator.value,
           RND);
  mpfr_sub(geometric_mean.value, mean.value, defect.value, RND);
}

void reciprocal_sqrt_newton_step(Real& inverse_root, const Real& argument,
                                 Real& square, Real& temp) {
  mpfr_sqr(square.value, inverse_root.value, RND);
  mpfr_mul(temp.value, argument.value, square.value, RND);
  mpfr_ui_sub(temp.value, 3, temp.value, RND);
  mpfr_mul(inverse_root.value, inverse_root.value, temp.value, RND);
  mpfr_div_2ui(inverse_root.value, inverse_root.value, 1, RND);
}

void warm_reciprocal_sqrt(Real& inverse_root, const Real& argument,
                          mpfr_prec_t target, mpfr_prec_t inherited_bits) {
  mpfr_prec_t precision = std::min<mpfr_prec_t>(
      target, std::max<mpfr_prec_t>(64, inherited_bits + 16));
  Real local_inverse(precision), local_argument(precision), square(precision),
      temp(precision);
  mpfr_set(local_inverse.value, inverse_root.value, RND);
  mpfr_set(local_argument.value, argument.value, RND);
  mpfr_prec_t estimated = std::max<mpfr_prec_t>(2, inherited_bits);
  while (estimated < precision) {
    reciprocal_sqrt_newton_step(local_inverse, local_argument, square, temp);
    estimated = std::min<mpfr_prec_t>(precision, 2 * estimated);
  }
  while (precision < target) {
    precision = std::min<mpfr_prec_t>(target, 2 * precision);
    mpfr_prec_round(local_inverse.value, precision, RND);
    mpfr_prec_round(local_argument.value, precision, RND);
    mpfr_prec_round(square.value, precision, RND);
    mpfr_prec_round(temp.value, precision, RND);
    mpfr_set(local_argument.value, argument.value, RND);
    reciprocal_sqrt_newton_step(local_inverse, local_argument, square, temp);
  }
  mpfr_set(inverse_root.value, local_inverse.value, RND);
}

Real pi_agm_impl(unsigned long bits, AgmRootKernel root_kernel,
                 unsigned* iteration_output = nullptr,
                 mpfr_prec_t precision_override = 0,
                 unsigned defect_threshold_percent = 0) {
  const mpfr_prec_t p =
      precision_override == 0 ? bits + 32 : precision_override;
  Real a(p), b(p), t_value(p), next_a(p), next_b(p), delta(p), correction(p),
      difference_squared(p), product(p), inverse_root(p), numerator(p),
      result(p);
  mpfr_set_ui(a.value, 1, RND);
  if (root_kernel == AgmRootKernel::DirectSqrt ||
      root_kernel == AgmRootKernel::ReducedDefect ||
      root_kernel == AgmRootKernel::HybridDefect) {
    mpfr_set_ui(b.value, 1, RND);
    mpfr_div_2ui(b.value, b.value, 1, RND);
    mpfr_sqrt(b.value, b.value, RND);
  } else {
    mpfr_set_ui(b.value, 2, RND);
    mpfr_rec_sqrt(b.value, b.value, RND);
  }
  mpfr_set_ui(t_value.value, 1, RND);
  mpfr_div_2ui(t_value.value, t_value.value, 2, RND);

  // The correction multiplier is exactly 2^i.  Keeping it as an MPFR value
  // turns a free exponent shift into a full multiplication at every step.
  unsigned iterations = 0;
  for (unsigned i = 0;; ++i) {
    ++iterations;
    mpfr_add(next_a.value, a.value, b.value, RND);
    mpfr_div_2ui(next_a.value, next_a.value, 1, RND);
    mpfr_sub(delta.value, a.value, next_a.value, RND);
    mpfr_sqr(difference_squared.value, delta.value, RND);
    mpfr_set(correction.value, difference_squared.value, RND);
    mpfr_mul_2ui(correction.value, correction.value, i, RND);
    mpfr_sub(t_value.value, t_value.value, correction.value, RND);

    // Once the current t correction is below every requested and guard bit,
    // all later corrections and the next arithmetic/geometric discrepancy
    // are smaller.  The arithmetic mean already suffices for a^2/t, so avoid
    // the final full-precision multiplication and square root entirely.
    if (mpfr_zero_p(correction.value) ||
        mpfr_get_exp(correction.value) <
            -static_cast<mpfr_exp_t>(bits + 24)) {
      mpfr_swap(a.value, next_a.value);
      break;
    }
    const bool use_defect =
        root_kernel == AgmRootKernel::ReducedDefect ||
        (root_kernel == AgmRootKernel::HybridDefect &&
         100 * reduced_defect_precision(delta, p) <=
             defect_threshold_percent * p);
    if (use_defect) {
      reduced_defect_geometric_mean(next_b, next_a, delta,
                                    difference_squared, p);
    } else {
      mpfr_mul(product.value, a.value, b.value, RND);
    }
    if (root_kernel == AgmRootKernel::DirectSqrt ||
        (root_kernel == AgmRootKernel::HybridDefect && !use_defect)) {
      mpfr_sqrt(next_b.value, product.value, RND);
    } else if (root_kernel == AgmRootKernel::ReciprocalSqrt) {
      mpfr_rec_sqrt(inverse_root.value, product.value, RND);
      mpfr_mul(next_b.value, product.value, inverse_root.value, RND);
    } else if (root_kernel == AgmRootKernel::WarmReciprocalNewton) {
      if (i == 0) {
        mpfr_rec_sqrt(inverse_root.value, product.value, RND);
      } else {
        const mpfr_exp_t exponent = mpfr_get_exp(delta.value);
        const mpfr_prec_t inherited = static_cast<mpfr_prec_t>(
            std::max<mpfr_exp_t>(2, -exponent));
        warm_reciprocal_sqrt(inverse_root, product, p, inherited);
      }
      mpfr_mul(next_b.value, product.value, inverse_root.value, RND);
    }
    mpfr_swap(a.value, next_a.value);
    mpfr_swap(b.value, next_b.value);
  }
  mpfr_sqr(numerator.value, a.value, RND);
  mpfr_div(result.value, numerator.value, t_value.value, RND);
  if (iteration_output != nullptr) *iteration_output = iterations;
  return result;
}

Real pi_agm(unsigned long bits, unsigned* iteration_output = nullptr) {
  return pi_agm_impl(bits, AgmRootKernel::DirectSqrt, iteration_output);
}

Real pi_agm_aligned(unsigned long bits,
                    unsigned* iteration_output = nullptr) {
  const mpfr_prec_t requested = bits + 32;
  const mpfr_prec_t aligned = static_cast<mpfr_prec_t>(
      ((requested + GMP_NUMB_BITS - 1) / GMP_NUMB_BITS) * GMP_NUMB_BITS);
  return pi_agm_impl(bits, AgmRootKernel::DirectSqrt, iteration_output,
                     aligned);
}

Real pi_agm_reciprocal(unsigned long bits,
                       unsigned* iteration_output = nullptr) {
  return pi_agm_impl(bits, AgmRootKernel::ReciprocalSqrt, iteration_output);
}

Real pi_agm_warm_reciprocal(unsigned long bits,
                            unsigned* iteration_output = nullptr) {
  return pi_agm_impl(bits, AgmRootKernel::WarmReciprocalNewton,
                     iteration_output);
}

Real pi_agm_reduced_defect(unsigned long bits,
                           unsigned* iteration_output = nullptr) {
  return pi_agm_impl(bits, AgmRootKernel::ReducedDefect, iteration_output);
}

Real pi_agm_hybrid_defect(unsigned long bits, unsigned threshold_percent,
                          unsigned* iteration_output = nullptr) {
  return pi_agm_impl(bits, AgmRootKernel::HybridDefect, iteration_output, 0,
                     threshold_percent);
}

Real pi_agm_hybrid(unsigned long bits,
                   unsigned* iteration_output = nullptr) {
  // Below a few thousand bits the temporary setup costs as much as the root
  // saved.  Above that range 65% is the best stable M4/GMP crossover: use the
  // defect only when its absolute-accuracy precision has fallen far enough.
  if (bits < 4096) return pi_agm(bits, iteration_output);
  return pi_agm_hybrid_defect(bits, 65, iteration_output);
}

Real pi_agm_parallel_hybrid(unsigned long bits,
                            unsigned* iteration_output = nullptr) {
  // Coarse two-lane version of the useful sinusoidal bifurcation.  Once m and
  // d are known, the period correction d^2 and geometric root sqrt(a*b) are
  // independent.  Keep the main thread on the correction while one worker
  // computes the product-root.  The final near-square step remains on the
  // reduced-defect path because it consumes d^2 directly.
  if (bits < 100000) return pi_agm_hybrid(bits, iteration_output);
  const mpfr_prec_t p = bits + 32;
  Real a(p), b(p), t_value(p), next_a(p), next_b(p), delta(p), correction(p),
      difference_squared(p), product(p), numerator(p), result(p);
  mpfr_set_ui(a.value, 1, RND);
  mpfr_set_ui(b.value, 1, RND);
  mpfr_div_2ui(b.value, b.value, 1, RND);
  mpfr_sqrt(b.value, b.value, RND);
  mpfr_set_ui(t_value.value, 1, RND);
  mpfr_div_2ui(t_value.value, t_value.value, 2, RND);

  std::mutex worker_mutex;
  std::condition_variable worker_start;
  std::condition_variable worker_done;
  bool job_pending = false;
  bool job_finished = false;
  bool worker_shutdown = false;
  std::thread geometric_worker([&] {
    std::unique_lock<std::mutex> lock(worker_mutex);
    for (;;) {
      worker_start.wait(lock, [&] { return job_pending || worker_shutdown; });
      if (worker_shutdown) break;
      job_pending = false;
      lock.unlock();
      mpfr_mul(product.value, a.value, b.value, RND);
      mpfr_sqrt(next_b.value, product.value, RND);
      lock.lock();
      job_finished = true;
      worker_done.notify_one();
    }
  });

  unsigned iterations = 0;
  for (unsigned i = 0;; ++i) {
    ++iterations;
    mpfr_add(next_a.value, a.value, b.value, RND);
    mpfr_div_2ui(next_a.value, next_a.value, 1, RND);
    mpfr_sub(delta.value, a.value, next_a.value, RND);

    // Quadratic convergence overshoots the stopping threshold substantially.
    // This conservative exponent test avoids launching a root worker for the
    // terminal iteration whose correction is already irrelevant.
    const mpfr_exp_t predicted_correction_exponent =
        2 * mpfr_get_exp(delta.value) + static_cast<mpfr_exp_t>(i);
    const bool predicted_stop =
        predicted_correction_exponent <
        -static_cast<mpfr_exp_t>(bits + 26);
    const bool use_defect =
        !predicted_stop &&
        100 * reduced_defect_precision(delta, p) <= 65 * p;

    bool launched_geometric_job = false;
    if (!predicted_stop && !use_defect) {
      {
        std::lock_guard<std::mutex> lock(worker_mutex);
        job_finished = false;
        job_pending = true;
      }
      worker_start.notify_one();
      launched_geometric_job = true;
    }

    mpfr_sqr(difference_squared.value, delta.value, RND);
    mpfr_set(correction.value, difference_squared.value, RND);
    mpfr_mul_2ui(correction.value, correction.value, i, RND);
    mpfr_sub(t_value.value, t_value.value, correction.value, RND);
    const bool stop =
        mpfr_zero_p(correction.value) ||
        mpfr_get_exp(correction.value) <
            -static_cast<mpfr_exp_t>(bits + 24);
    if (launched_geometric_job) {
      std::unique_lock<std::mutex> lock(worker_mutex);
      worker_done.wait(lock, [&] { return job_finished; });
    }
    if (stop) {
      mpfr_swap(a.value, next_a.value);
      break;
    }
    if (use_defect) {
      reduced_defect_geometric_mean(next_b, next_a, delta,
                                    difference_squared, p);
    } else if (!launched_geometric_job) {
      // Only reachable if the conservative prediction missed the exact
      // stopping boundary.  Preserve correctness without depending on it.
      mpfr_mul(product.value, a.value, b.value, RND);
      mpfr_sqrt(next_b.value, product.value, RND);
    }
    mpfr_swap(a.value, next_a.value);
    mpfr_swap(b.value, next_b.value);
  }
  {
    std::lock_guard<std::mutex> lock(worker_mutex);
    worker_shutdown = true;
  }
  worker_start.notify_one();
  geometric_worker.join();
  mpfr_sqr(numerator.value, a.value, RND);
  mpfr_div(result.value, numerator.value, t_value.value, RND);
  if (iteration_output != nullptr) *iteration_output = iterations;
  return result;
}

Real pi_agm_exact_mpn(unsigned long bits,
                      unsigned* iteration_output = nullptr) {
  // Experimental GMP-specific path.  After the first AGM step both positive
  // operands have MPFR exponent zero.  At limb-aligned precision their
  // mantissas A and B therefore satisfy
  //
  //   sqrt(a*b) = sqrt(A*B) * 2^(-p).
  //
  // Write floor(sqrt(A*B)) directly into the destination mantissa, avoiding
  // an intermediate rounded MPFR product.  The enclosing algorithm has 32+
  // guard bits, so directed truncation here remains harmless to the requested
  // result while exposing the actual cost of exact limb-level fusion.
  const mpfr_prec_t requested = bits + 32;
  const mpfr_prec_t p = static_cast<mpfr_prec_t>(
      ((requested + GMP_NUMB_BITS - 1) / GMP_NUMB_BITS) * GMP_NUMB_BITS);
  const mp_size_t limbs = static_cast<mp_size_t>(p / GMP_NUMB_BITS);
  Real a(p), b(p), t_value(p), next_a(p), next_b(p), delta(p), correction(p),
      product(p), numerator(p), result(p);
  std::vector<mp_limb_t> exact_product(2 * limbs);
  std::vector<mp_limb_t> integer_root(limbs);
  mpfr_set_ui(a.value, 1, RND);
  mpfr_set_ui(b.value, 1, RND);
  mpfr_div_2ui(b.value, b.value, 1, RND);
  mpfr_sqrt(b.value, b.value, RND);
  mpfr_set_ui(t_value.value, 1, RND);
  mpfr_div_2ui(t_value.value, t_value.value, 2, RND);
  unsigned iterations = 0;
  for (unsigned i = 0;; ++i) {
    ++iterations;
    mpfr_add(next_a.value, a.value, b.value, RND);
    mpfr_div_2ui(next_a.value, next_a.value, 1, RND);
    mpfr_sub(delta.value, a.value, next_a.value, RND);
    mpfr_sqr(correction.value, delta.value, RND);
    mpfr_mul_2ui(correction.value, correction.value, i, RND);
    mpfr_sub(t_value.value, t_value.value, correction.value, RND);
    if (mpfr_zero_p(correction.value) ||
        mpfr_get_exp(correction.value) <
            -static_cast<mpfr_exp_t>(bits + 24)) {
      mpfr_swap(a.value, next_a.value);
      break;
    }
    if (i == 0) {
      mpfr_mul(product.value, a.value, b.value, RND);
      mpfr_sqrt(next_b.value, product.value, RND);
    } else {
      if (a.value[0]._mpfr_exp != 0 || b.value[0]._mpfr_exp != 0) {
        std::cerr << "unexpected AGM exponent in exact mpn kernel\n";
        std::abort();
      }
      mpn_mul_n(exact_product.data(), a.value[0]._mpfr_d,
                b.value[0]._mpfr_d, limbs);
      mpn_sqrtrem(integer_root.data(), nullptr, exact_product.data(),
                  2 * limbs);
      mpn_copyi(next_b.value[0]._mpfr_d, integer_root.data(), limbs);
      next_b.value[0]._mpfr_sign = 1;
      next_b.value[0]._mpfr_exp = 0;
    }
    mpfr_swap(a.value, next_a.value);
    mpfr_swap(b.value, next_b.value);
  }
  mpfr_sqr(numerator.value, a.value, RND);
  mpfr_div(result.value, numerator.value, t_value.value, RND);
  if (iteration_output != nullptr) *iteration_output = iterations;
  return result;
}

struct AgmProfile {
  Real result;
  double total = 0;
  double initial_root = 0;
  double mean = 0;
  double period = 0;
  double multiply = 0;
  double root = 0;
  double final_division = 0;
  unsigned iterations = 0;

  explicit AgmProfile(mpfr_prec_t p) : result(p) {}
};

AgmProfile profile_agm(unsigned long bits) {
  const mpfr_prec_t p = bits + 32;
  AgmProfile profile(p);
  Real a(p), b(p), t_value(p), next_a(p), next_b(p), delta(p), correction(p),
      product(p), numerator(p);
  const auto total_begin = std::chrono::steady_clock::now();
  mpfr_set_ui(a.value, 1, RND);
  mpfr_set_ui(b.value, 1, RND);
  mpfr_div_2ui(b.value, b.value, 1, RND);
  auto begin = std::chrono::steady_clock::now();
  mpfr_sqrt(b.value, b.value, RND);
  auto end = std::chrono::steady_clock::now();
  profile.initial_root += std::chrono::duration<double>(end - begin).count();
  mpfr_set_ui(t_value.value, 1, RND);
  mpfr_div_2ui(t_value.value, t_value.value, 2, RND);
  for (unsigned i = 0;; ++i) {
    ++profile.iterations;
    begin = std::chrono::steady_clock::now();
    mpfr_add(next_a.value, a.value, b.value, RND);
    mpfr_div_2ui(next_a.value, next_a.value, 1, RND);
    mpfr_sub(delta.value, a.value, next_a.value, RND);
    end = std::chrono::steady_clock::now();
    profile.mean += std::chrono::duration<double>(end - begin).count();

    begin = std::chrono::steady_clock::now();
    mpfr_sqr(correction.value, delta.value, RND);
    mpfr_mul_2ui(correction.value, correction.value, i, RND);
    mpfr_sub(t_value.value, t_value.value, correction.value, RND);
    end = std::chrono::steady_clock::now();
    profile.period += std::chrono::duration<double>(end - begin).count();
    if (mpfr_zero_p(correction.value) ||
        mpfr_get_exp(correction.value) <
            -static_cast<mpfr_exp_t>(bits + 24)) {
      mpfr_swap(a.value, next_a.value);
      break;
    }

    begin = std::chrono::steady_clock::now();
    mpfr_mul(product.value, a.value, b.value, RND);
    end = std::chrono::steady_clock::now();
    profile.multiply += std::chrono::duration<double>(end - begin).count();
    begin = std::chrono::steady_clock::now();
    mpfr_sqrt(next_b.value, product.value, RND);
    end = std::chrono::steady_clock::now();
    profile.root += std::chrono::duration<double>(end - begin).count();
    mpfr_swap(a.value, next_a.value);
    mpfr_swap(b.value, next_b.value);
  }
  begin = std::chrono::steady_clock::now();
  mpfr_sqr(numerator.value, a.value, RND);
  mpfr_div(profile.result.value, numerator.value, t_value.value, RND);
  end = std::chrono::steady_clock::now();
  profile.final_division +=
      std::chrono::duration<double>(end - begin).count();
  profile.total = std::chrono::duration<double>(
                      std::chrono::steady_clock::now() - total_begin)
                      .count();
  return profile;
}

struct Split {
  mpz_t P, Q, T;
  Split() { mpz_inits(P, Q, T, nullptr); }
  ~Split() { mpz_clears(P, Q, T, nullptr); }
};

void ramanujan_split(unsigned long a, unsigned long b, unsigned long linear_a,
                     unsigned long linear_b, unsigned long denominator_base,
                     bool alternating, Split& out) {
  if (b - a == 1) {
    if (a == 0) {
      mpz_set_ui(out.P, 1);
      mpz_set_ui(out.Q, 1);
      mpz_set_ui(out.T, linear_a);
    } else {
      mpz_set_ui(out.P, 6 * a - 5);
      mpz_mul_ui(out.P, out.P, 2 * a - 1);
      mpz_mul_ui(out.P, out.P, 6 * a - 1);
      mpz_set_ui(out.Q, a);
      mpz_pow_ui(out.Q, out.Q, 3);
      mpz_mul_ui(out.Q, out.Q, denominator_base);
      mpz_set_ui(out.T, linear_b);
      mpz_mul_ui(out.T, out.T, a);
      mpz_add_ui(out.T, out.T, linear_a);
      mpz_mul(out.T, out.T, out.P);
      if (alternating && (a & 1UL)) mpz_neg(out.T, out.T);
    }
    return;
  }
  const unsigned long middle = (a + b) / 2;
  Split left, right;
  ramanujan_split(a, middle, linear_a, linear_b, denominator_base,
                  alternating, left);
  ramanujan_split(middle, b, linear_a, linear_b, denominator_base,
                  alternating, right);
  mpz_mul(out.P, left.P, right.P);
  mpz_mul(out.Q, left.Q, right.Q);
  mpz_mul(out.T, left.T, right.Q);
  mpz_addmul(out.T, left.P, right.T);
}

void combine_splits(const Split& left, const Split& right, Split& out) {
  mpz_mul(out.P, left.P, right.P);
  mpz_mul(out.Q, left.Q, right.Q);
  mpz_mul(out.T, left.T, right.Q);
  mpz_addmul(out.T, left.P, right.T);
}

Real pi_chudnovsky_impl(unsigned long bits, bool parallel) {
  const mpfr_prec_t p = bits + 32;
  const unsigned long terms =
      static_cast<unsigned long>(std::ceil(bits / 47.110413138215842)) + 1;
  Split split;
  if (parallel && terms >= 4) {
    const unsigned long middle = terms / 2;
    Split left, right;
    std::thread left_worker([&] {
      ramanujan_split(0, middle, 13591409UL, 545140134UL,
                      10939058860032000UL, true, left);
    });
    ramanujan_split(middle, terms, 13591409UL, 545140134UL,
                    10939058860032000UL, true, right);
    left_worker.join();
    combine_splits(left, right, split);
  } else {
    ramanujan_split(0, terms, 13591409UL, 545140134UL,
                    10939058860032000UL, true, split);
  }
  Real constant(p), numerator(p), denominator(p), result(p);
  mpfr_set_ui(constant.value, 10005, RND);
  mpfr_sqrt(constant.value, constant.value, RND);
  mpfr_mul_ui(constant.value, constant.value, 426880, RND);
  mpfr_set_z(numerator.value, split.Q, RND);
  mpfr_mul(numerator.value, numerator.value, constant.value, RND);
  mpfr_set_z(denominator.value, split.T, RND);
  mpfr_div(result.value, numerator.value, denominator.value, RND);
  return result;
}

Real pi_chudnovsky(unsigned long bits) {
  return pi_chudnovsky_impl(bits, false);
}

Real pi_chudnovsky_parallel(unsigned long bits) {
  return pi_chudnovsky_impl(bits, true);
}

Real pi_cm67(unsigned long bits) {
  // The D=-67 class-number-one series, with j=-5280^3:
  //
  //   1760 sqrt(330)/pi = sum (10177+261702n)
  //       (1/6)_n(1/2)_n(5/6)_n/(n!)^3 (-1/440^3)^n.
  //
  // Its binary-splitting recurrence differs from Chudnovsky only in the
  // linear coefficient and denominator base 5280^3/24.
  const mpfr_prec_t p = bits + 32;
  const unsigned long terms =
      static_cast<unsigned long>(std::ceil(bits / 26.342910802943852)) + 1;
  Split split;
  ramanujan_split(0, terms, 10177UL, 261702UL, 6133248000UL, true, split);
  Real constant(p), numerator(p), denominator(p), result(p);
  mpfr_set_ui(constant.value, 330, RND);
  mpfr_sqrt(constant.value, constant.value, RND);
  mpfr_mul_ui(constant.value, constant.value, 1760, RND);
  mpfr_set_z(numerator.value, split.Q, RND);
  mpfr_mul(numerator.value, numerator.value, constant.value, RND);
  mpfr_set_z(denominator.value, split.T, RND);
  mpfr_div(result.value, numerator.value, denominator.value, RND);
  return result;
}

unsigned long lemniscatic_terms(unsigned long bits) {
  const double bits_per_term = std::log2(287496.0 / 1728.0);
  return static_cast<unsigned long>(std::ceil(bits / bits_per_term)) + 2;
}

Real pi_lemniscatic_bsplit(unsigned long bits) {
  // The conductor-two Q(i) point tau=2i has j=66^3 and gives
  //
  //   1/pi = 4/(11 sqrt(33)) sum_n (6n)!/((3n)!(n!)^3)
  //                                 (63n+5)/66^(3n).
  //
  // The quotient of consecutive hypergeometric terms tends to
  // 1728/66^3, or about 7.378 useful bits per term.
  const mpfr_prec_t p = bits + 32;
  const unsigned long terms = lemniscatic_terms(bits);
  Split split;
  ramanujan_split(0, terms, 5UL, 63UL, 11979UL, false, split);
  Real constant(p), numerator(p), denominator(p), result(p);
  mpfr_set_ui(constant.value, 33, RND);
  mpfr_sqrt(constant.value, constant.value, RND);
  mpfr_mul_ui(constant.value, constant.value, 11, RND);
  mpfr_set_z(numerator.value, split.Q, RND);
  mpfr_mul(numerator.value, numerator.value, constant.value, RND);
  mpfr_set_z(denominator.value, split.T, RND);
  mpfr_mul_2ui(denominator.value, denominator.value, 2, RND);
  mpfr_div(result.value, numerator.value, denominator.value, RND);
  return result;
}

double correct_bits(const Real& value) {
  const mpfr_prec_t p = value.precision() + 64;
  Real extended(p), reference(p), error(p), logarithm(p);
  mpfr_set(extended.value, value.value, RND);
  mpfr_const_pi(reference.value, RND);
  mpfr_sub(error.value, extended.value, reference.value, RND);
  mpfr_abs(error.value, error.value, RND);
  mpfr_log2(logarithm.value, error.value, RND);
  return -mpfr_get_d(logarithm.value, RND);
}

struct Timing {
  Real result;
  double median;
  double minimum;
  double maximum;
};

struct LimbKernelProfile {
  double mpfr_multiply = 0;
  double mpfr_root = 0;
  double mpfr_combined = 0;
  double mpn_multiply = 0;
  double mpn_root = 0;
  double mpn_combined = 0;
  mp_size_t limbs = 0;
};

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

LimbKernelProfile profile_limb_kernel(unsigned long bits, unsigned repeats) {
  // Limb-align the precision so the mpn root has no partial output limb.  The
  // operands are advanced several AGM steps to make their mantissas dense.
  const mpfr_prec_t p = static_cast<mpfr_prec_t>(
      ((bits + GMP_NUMB_BITS - 1) / GMP_NUMB_BITS) * GMP_NUMB_BITS);
  const mp_size_t limbs = static_cast<mp_size_t>(p / GMP_NUMB_BITS);
  Real a(p), b(p), mean(p), product(p), root(p);
  mpfr_set_ui(a.value, 1, RND);
  mpfr_set_ui(b.value, 1, RND);
  mpfr_div_2ui(b.value, b.value, 1, RND);
  mpfr_sqrt(b.value, b.value, RND);
  for (unsigned step = 0; step < 4; ++step) {
    mpfr_add(mean.value, a.value, b.value, RND);
    mpfr_div_2ui(mean.value, mean.value, 1, RND);
    mpfr_mul(product.value, a.value, b.value, RND);
    mpfr_sqrt(root.value, product.value, RND);
    mpfr_swap(a.value, mean.value);
    mpfr_swap(b.value, root.value);
  }

  std::vector<mp_limb_t> exact_product(2 * limbs);
  std::vector<mp_limb_t> integer_root(limbs);
  mpn_mul_n(exact_product.data(), a.value[0]._mpfr_d,
            b.value[0]._mpfr_d, limbs);
  LimbKernelProfile result;
  result.limbs = limbs;
  result.mpfr_multiply = median_seconds(
      [&] { mpfr_mul(product.value, a.value, b.value, RND); }, repeats);
  result.mpfr_root = median_seconds(
      [&] { mpfr_sqrt(root.value, product.value, RND); }, repeats);
  result.mpfr_combined = median_seconds(
      [&] {
        mpfr_mul(product.value, a.value, b.value, RND);
        mpfr_sqrt(root.value, product.value, RND);
      },
      repeats);
  result.mpn_multiply = median_seconds(
      [&] {
        mpn_mul_n(exact_product.data(), a.value[0]._mpfr_d,
                  b.value[0]._mpfr_d, limbs);
      },
      repeats);
  result.mpn_root = median_seconds(
      [&] {
        mpn_sqrtrem(integer_root.data(), nullptr, exact_product.data(),
                    2 * limbs);
      },
      repeats);
  result.mpn_combined = median_seconds(
      [&] {
        mpn_mul_n(exact_product.data(), a.value[0]._mpfr_d,
                  b.value[0]._mpfr_d, limbs);
        mpn_sqrtrem(integer_root.data(), nullptr, exact_product.data(),
                    2 * limbs);
      },
      repeats);
  return result;
}

template <typename Function>
Timing timed_median(Function&& function, unsigned repeats) {
  Real result = function();  // Untimed warm-up.
  std::vector<double> samples;
  samples.reserve(repeats);
  for (unsigned i = 0; i < repeats; ++i) {
    const auto begin = std::chrono::steady_clock::now();
    Real sample = function();
    const auto end = std::chrono::steady_clock::now();
    if (i == 0) result = sample;
    samples.push_back(std::chrono::duration<double>(end - begin).count());
  }
  std::sort(samples.begin(), samples.end());
  return {result, samples[samples.size() / 2], samples.front(), samples.back()};
}

}  // namespace

int main(int argc, char** argv) {
  unsigned long bits = 10000;
  unsigned repeats = 3;
  std::string suite = "all";
  for (int i = 1; i < argc; i += 2) {
    if (i + 1 >= argc) {
      std::cerr << "usage: " << argv[0]
                << " [--bits N] [--repeats N] "
                   "[--suite all|core|profile|limb|fusion|parallel]\n";
      return 2;
    }
    const std::string option(argv[i]);
    if (option == "--bits") bits = std::stoul(argv[i + 1]);
    else if (option == "--repeats") repeats = std::stoul(argv[i + 1]);
    else if (option == "--suite") suite = argv[i + 1];
    else {
      std::cerr << "unknown option: " << option << "\n";
      return 2;
    }
  }
  if (bits < 64 || repeats == 0) return 2;

  auto row = [](const char* name, const auto& result, bool comma) {
    std::cout << "  \"" << name << "\": {\"seconds\": " << result.median
              << ", \"min_seconds\": " << result.minimum
              << ", \"max_seconds\": " << result.maximum
              << ", \"correct_bits\": " << correct_bits(result.result) << "}"
              << (comma ? "," : "") << "\n";
  };
  if (suite == "limb") {
    const LimbKernelProfile profile = profile_limb_kernel(bits, repeats);
    std::cout << std::setprecision(9) << "{\n  \"bits\": " << bits
              << ",\n  \"limbs\": " << profile.limbs
              << ",\n  \"repeats\": " << repeats
              << ",\n  \"mpfr\": {\"multiply\": " << profile.mpfr_multiply
              << ", \"root\": " << profile.mpfr_root
              << ", \"combined\": " << profile.mpfr_combined
              << "},\n  \"mpn_exact\": {\"multiply\": "
              << profile.mpn_multiply << ", \"root\": " << profile.mpn_root
              << ", \"combined\": " << profile.mpn_combined
              << "},\n  \"mpn_to_mpfr_ratio\": "
              << profile.mpn_combined / profile.mpfr_combined << "\n}\n";
    return 0;
  }
  if (suite == "profile") {
    AgmProfile aggregate(bits + 32);
    for (unsigned repeat = 0; repeat <= repeats; ++repeat) {
      AgmProfile sample = profile_agm(bits);
      if (repeat == 0) continue;
      aggregate.result = sample.result;
      aggregate.total += sample.total;
      aggregate.initial_root += sample.initial_root;
      aggregate.mean += sample.mean;
      aggregate.period += sample.period;
      aggregate.multiply += sample.multiply;
      aggregate.root += sample.root;
      aggregate.final_division += sample.final_division;
      aggregate.iterations = sample.iterations;
    }
    const double scale = 1.0 / repeats;
    const double fused_ceiling =
        aggregate.total /
        (aggregate.total - aggregate.multiply - aggregate.root);
    std::cout << std::setprecision(9) << "{\n  \"bits\": " << bits
              << ",\n  \"repeats\": " << repeats
              << ",\n  \"iterations\": " << aggregate.iterations
              << ",\n  \"seconds\": {"
              << "\"total\": " << aggregate.total * scale
              << ", \"initial_root\": " << aggregate.initial_root * scale
              << ", \"mean\": " << aggregate.mean * scale
              << ", \"period\": " << aggregate.period * scale
              << ", \"multiply\": " << aggregate.multiply * scale
              << ", \"root\": " << aggregate.root * scale
              << ", \"final_division\": "
              << aggregate.final_division * scale
              << "},\n  \"multiply_root_fraction\": "
              << (aggregate.multiply + aggregate.root) / aggregate.total
              << ",\n  \"perfect_fusion_speedup_ceiling\": " << fused_ceiling
              << ",\n  \"correct_bits\": " << correct_bits(aggregate.result)
              << "\n}\n";
    return 0;
  }
  if (suite == "fusion" || suite == "parallel") {
    struct FusionCase {
      const char* name;
      std::function<Real()> function;
      std::vector<double> samples;
      std::unique_ptr<Real> result;
    };
    std::vector<FusionCase> cases;
    if (suite == "fusion") {
      cases.push_back(
          {"hybrid_50", [&] { return pi_agm_hybrid_defect(bits, 50); }, {},
           nullptr});
      cases.push_back(
          {"hybrid_65", [&] { return pi_agm_hybrid_defect(bits, 65); }, {},
           nullptr});
      cases.push_back(
          {"hybrid_75", [&] { return pi_agm_hybrid_defect(bits, 75); }, {},
           nullptr});
    } else {
      cases.push_back(
          {"hybrid_65", [&] { return pi_agm_hybrid_defect(bits, 65); }, {},
           nullptr});
    }
    cases.push_back({"parallel_hybrid",
                     [&] { return pi_agm_parallel_hybrid(bits); }, {},
                     nullptr});
    cases.push_back({"parallel_chudnovsky",
                     [&] { return pi_chudnovsky_parallel(bits); }, {},
                     nullptr});
    cases.push_back({"chudnovsky", [&] { return pi_chudnovsky(bits); }, {},
                     nullptr});
    cases.push_back({"direct", [&] { return pi_agm(bits); }, {}, nullptr});
    for (auto& item : cases) {
      item.result = std::make_unique<Real>(item.function());
      item.samples.reserve(repeats);
    }
    for (unsigned round = 0; round < repeats; ++round) {
      for (std::size_t offset = 0; offset < cases.size(); ++offset) {
        auto& item = cases[(round + offset) % cases.size()];
        const auto begin = std::chrono::steady_clock::now();
        Real sample = item.function();
        const auto end = std::chrono::steady_clock::now();
        item.samples.push_back(
            std::chrono::duration<double>(end - begin).count());
      }
    }
    const auto& direct_samples = cases.back().samples;
    std::cout << std::setprecision(9) << "{\n  \"bits\": " << bits
              << ",\n  \"repeats\": " << repeats << ",\n";
    for (std::size_t index = 0; index < cases.size(); ++index) {
      auto& item = cases[index];
      std::vector<double> ratios;
      ratios.reserve(repeats);
      for (unsigned round = 0; round < repeats; ++round)
        ratios.push_back(item.samples[round] / direct_samples[round]);
      std::sort(ratios.begin(), ratios.end());
      std::sort(item.samples.begin(), item.samples.end());
      std::cout << "  \"" << item.name << "\": {\"seconds\": "
                << item.samples[item.samples.size() / 2]
                << ", \"min_seconds\": " << item.samples.front()
                << ", \"max_seconds\": " << item.samples.back()
                << ", \"median_ratio_to_direct\": "
                << ratios[ratios.size() / 2]
                << ", \"correct_bits\": " << correct_bits(*item.result)
                << "}" << (index + 1 < cases.size() ? "," : "") << "\n";
    }
    std::cout << "}\n";
    return 0;
  }
  if (suite == "core") {
    unsigned agm_steps = 0;
    struct CoreCase {
      const char* name;
      std::function<Real()> function;
      std::vector<double> samples;
      std::unique_ptr<Real> result;
    };
    std::vector<CoreCase> cases;
    cases.push_back({"quartic_two_sqrt",
                     [&] { return pi_quartic_two_sqrt(bits); }, {}, nullptr});
    cases.push_back({"lemniscatic_bsplit",
                     [&] { return pi_lemniscatic_bsplit(bits); }, {}, nullptr});
    cases.push_back(
        {"chudnovsky", [&] { return pi_chudnovsky(bits); }, {}, nullptr});
    cases.push_back({"chudnovsky_parallel",
                     [&] { return pi_chudnovsky_parallel(bits); }, {},
                     nullptr});
    cases.push_back({"agm_reciprocal",
                     [&] { return pi_agm_reciprocal(bits); }, {}, nullptr});
    cases.push_back({"agm_warm_reciprocal",
                     [&] { return pi_agm_warm_reciprocal(bits); }, {},
                     nullptr});
    cases.push_back({"agm_reduced_defect",
                     [&] { return pi_agm_reduced_defect(bits); }, {},
                     nullptr});
    cases.push_back({"agm_hybrid_defect_50",
                     [&] { return pi_agm_hybrid_defect(bits, 50); }, {},
                     nullptr});
    cases.push_back({"agm_hybrid_defect_75",
                     [&] { return pi_agm_hybrid_defect(bits, 75); }, {},
                     nullptr});
    cases.push_back({"agm_hybrid_defect_90",
                     [&] { return pi_agm_hybrid_defect(bits, 90); }, {},
                     nullptr});
    cases.push_back({"agm_hybrid", [&] { return pi_agm_hybrid(bits); }, {},
                     nullptr});
    cases.push_back({"agm_parallel_hybrid",
                     [&] { return pi_agm_parallel_hybrid(bits); }, {},
                     nullptr});
    cases.push_back({"agm_exact_mpn", [&] { return pi_agm_exact_mpn(bits); },
                     {}, nullptr});
    cases.push_back({"agm_aligned", [&] { return pi_agm_aligned(bits); }, {},
                     nullptr});
    cases.push_back({"agm", [&] { return pi_agm(bits, &agm_steps); }, {},
                     nullptr});
    for (auto& item : cases) {
      item.result = std::make_unique<Real>(item.function());
      item.samples.reserve(repeats);
    }
    // Rotate the execution order each round so frequency and thermal drift do
    // not consistently favor the same algorithm.
    for (unsigned round = 0; round < repeats; ++round) {
      for (std::size_t offset = 0; offset < cases.size(); ++offset) {
        auto& item = cases[(round + offset) % cases.size()];
        const auto begin = std::chrono::steady_clock::now();
        Real sample = item.function();
        const auto end = std::chrono::steady_clock::now();
        item.samples.push_back(
            std::chrono::duration<double>(end - begin).count());
      }
    }
    std::cout << std::setprecision(9) << "{\n  \"bits\": " << bits
              << ",\n  \"repeats\": " << repeats
              << ",\n  \"agm_iterations\": " << agm_steps
              << ",\n  \"quartic_iterations\": " << quartic_iterations(bits)
              << ",\n  \"lemniscatic_terms\": " << lemniscatic_terms(bits)
              << ",\n";
    std::vector<std::vector<double>> ratios(cases.size());
    const auto& agm_samples = cases.back().samples;
    for (std::size_t index = 0; index < cases.size(); ++index) {
      auto& item = cases[index];
      ratios[index].reserve(repeats);
      for (unsigned round = 0; round < repeats; ++round)
        ratios[index].push_back(item.samples[round] / agm_samples[round]);
      std::sort(ratios[index].begin(), ratios[index].end());
      std::sort(item.samples.begin(), item.samples.end());
      std::cout << "  \"" << item.name << "\": {\"seconds\": "
                << item.samples[item.samples.size() / 2]
                << ", \"min_seconds\": " << item.samples.front()
                << ", \"max_seconds\": " << item.samples.back()
                << ", \"median_ratio_to_agm\": "
                << ratios[index][ratios[index].size() / 2]
                << ", \"correct_bits\": " << correct_bits(*item.result) << "}"
                << (index + 1 < cases.size() ? "," : "") << "\n";
    }
    std::cout << "}\n";
    return 0;
  }
  if (suite != "all") {
    std::cerr <<
        "suite must be 'all', 'core', 'profile', 'limb', 'fusion', or "
        "'parallel'\n";
    return 2;
  }

  auto implicit4 = timed_median([&] { return pi_quartic_implicit(bits); }, repeats);
  auto implicit4_legacy =
      timed_median([&] { return pi_quartic_implicit_legacy(bits); }, repeats);
  auto radical4 = timed_median([&] { return pi_quartic_radical(bits); }, repeats);
  auto rootn_stable =
      timed_median([&] { return pi_quartic_rootn_stable(bits); }, repeats);
  auto two_sqrt =
      timed_median([&] { return pi_quartic_two_sqrt(bits); }, repeats);
  auto two_sqrt_stable =
      timed_median([&] { return pi_quartic_two_sqrt_stable(bits); }, repeats);
  auto implicit3 = timed_median([&] { return pi_cubic_implicit(bits); }, repeats);
  auto nome3 =
      timed_median([&] { return pi_lambda3_terminal_nome(bits); }, repeats);
  auto chudnovsky = timed_median([&] { return pi_chudnovsky(bits); }, repeats);
  auto cm67 = timed_median([&] { return pi_cm67(bits); }, repeats);
  auto lemniscatic_bsplit =
      timed_median([&] { return pi_lemniscatic_bsplit(bits); }, repeats);
  unsigned agm_steps = 0;
  auto agm =
      timed_median([&] { return pi_agm(bits, &agm_steps); }, repeats);
  auto agm_legacy = timed_median([&] { return pi_agm_legacy(bits); }, repeats);
  std::cout << std::setprecision(9) << "{\n  \"bits\": " << bits
            << ",\n  \"repeats\": " << repeats
            << ",\n  \"agm_iterations\": " << agm_steps
            << ",\n  \"quartic_iterations\": " << quartic_iterations(bits)
            << ",\n  \"lemniscatic_terms\": " << lemniscatic_terms(bits)
            << ",\n";
  row("quartic_implicit", implicit4, true);
  row("quartic_implicit_legacy", implicit4_legacy, true);
  row("quartic_radical", radical4, true);
  row("quartic_rootn_stable", rootn_stable, true);
  row("quartic_two_sqrt", two_sqrt, true);
  row("quartic_two_sqrt_stable", two_sqrt_stable, true);
  row("cubic_implicit", implicit3, true);
  row("lambda3_nome", nome3, true);
  row("chudnovsky", chudnovsky, true);
  row("cm67_bsplit", cm67, true);
  row("lemniscatic_bsplit", lemniscatic_bsplit, true);
  row("agm", agm, true);
  row("agm_legacy", agm_legacy, false);
  std::cout << "}\n";
  return 0;
}
