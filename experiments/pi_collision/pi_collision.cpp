#include <gmp.h>
#include <mpfr.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

constexpr mpfr_rnd_t RND = MPFR_RNDN;

struct Real {
  mpfr_t value;

  explicit Real(mpfr_prec_t precision) { mpfr_init2(value, precision); }
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

void complex_multiply(Real& real_out, Real& imag_out, const Real& a,
                      const Real& b, const Real& c, const Real& d) {
  const mpfr_prec_t p = real_out.precision();
  Real ac(p), bd(p), ad(p), bc(p);
  mpfr_mul(ac.value, a.value, c.value, RND);
  mpfr_mul(bd.value, b.value, d.value, RND);
  mpfr_mul(ad.value, a.value, d.value, RND);
  mpfr_mul(bc.value, b.value, c.value, RND);
  mpfr_sub(real_out.value, ac.value, bd.value, RND);
  mpfr_add(imag_out.value, ad.value, bc.value, RND);
}

Real tangent_multiple(const Real& x, unsigned long m) {
  const mpfr_prec_t p = x.precision();
  Real result(p), x2(p), numerator(p), denominator(p), temp(p);
  if (m == 1) {
    mpfr_set(result.value, x.value, RND);
    return result;
  }
  if (m == 2) {
    mpfr_sqr(x2.value, x.value, RND);
    mpfr_ui_sub(denominator.value, 1, x2.value, RND);
    mpfr_mul_ui(numerator.value, x.value, 2, RND);
    mpfr_div(result.value, numerator.value, denominator.value, RND);
    return result;
  }
  if (m == 3) {
    mpfr_sqr(x2.value, x.value, RND);
    mpfr_ui_sub(temp.value, 3, x2.value, RND);
    mpfr_mul(numerator.value, x.value, temp.value, RND);
    mpfr_mul_ui(temp.value, x2.value, 3, RND);
    mpfr_ui_sub(denominator.value, 1, temp.value, RND);
    mpfr_div(result.value, numerator.value, denominator.value, RND);
    return result;
  }

  Real out_real(p), out_imag(p), base_real(p), base_imag(p);
  mpfr_set_ui(out_real.value, 1, RND);
  mpfr_set_zero(out_imag.value, 0);
  mpfr_set_ui(base_real.value, 1, RND);
  mpfr_set(base_imag.value, x.value, RND);

  unsigned long exponent = m;
  while (exponent != 0) {
    if (exponent & 1UL) {
      Real next_real(p), next_imag(p);
      complex_multiply(next_real, next_imag, out_real, out_imag, base_real,
                       base_imag);
      out_real = next_real;
      out_imag = next_imag;
    }
    exponent >>= 1;
    if (exponent != 0) {
      Real next_real(p), next_imag(p);
      complex_multiply(next_real, next_imag, base_real, base_imag, base_real,
                       base_imag);
      base_real = next_real;
      base_imag = next_imag;
    }
  }
  mpfr_div(result.value, out_imag.value, out_real.value, RND);
  return result;
}

void tangent_double_in_place(Real& x, Real& square, Real& denominator) {
  mpfr_sqr(square.value, x.value, RND);
  mpfr_ui_sub(denominator.value, 1, square.value, RND);
  mpfr_mul_ui(x.value, x.value, 2, RND);
  mpfr_div(x.value, x.value, denominator.value, RND);
}

bool is_power_of_two(unsigned long value) {
  return value != 0 && (value & (value - 1)) == 0;
}

void tangent_transport_in_place(Real& x, unsigned long m, Real& square,
                                Real& denominator, Real& numerator,
                                Real& temp) {
  if (is_power_of_two(m)) {
    for (unsigned long radix = m; radix > 1; radix >>= 1)
      tangent_double_in_place(x, square, denominator);
    return;
  }
  if (m == 3) {
    mpfr_sqr(square.value, x.value, RND);
    mpfr_ui_sub(temp.value, 3, square.value, RND);
    mpfr_mul(numerator.value, x.value, temp.value, RND);
    mpfr_mul_ui(temp.value, square.value, 3, RND);
    mpfr_ui_sub(denominator.value, 1, temp.value, RND);
    mpfr_div(x.value, numerator.value, denominator.value, RND);
    return;
  }
  x = tangent_multiple(x, m);
}

Real tangent_tower(const Real& x, unsigned long m, unsigned long depth) {
  Real y(x);
  Real square(x.precision()), denominator(x.precision()), numerator(x.precision()),
      temp(x.precision());
  for (unsigned long i = 0; i < depth; ++i)
    tangent_transport_in_place(y, m, square, denominator, numerator, temp);
  return y;
}

void set_power(Real& output, unsigned long m, unsigned long depth) {
  mpz_t integer_power;
  mpz_init(integer_power);
  mpz_ui_pow_ui(integer_power, m, depth);
  mpfr_set_z(output.value, integer_power, RND);
  mpz_clear(integer_power);
}

void root_newton_step(Real& x, unsigned long m, unsigned long depth) {
  const mpfr_prec_t p = x.precision();
  Real M(p), y(p), x2(p), y2(p), numerator(p), denominator(p), correction(p);
  set_power(M, m, depth);
  y = tangent_tower(x, m, depth);

  // x <- x - (D_M(x)-1) / D'_M(x), with
  // D'_M(x) = M(1+D_M(x)^2)/(1+x^2).
  mpfr_sqr(x2.value, x.value, RND);
  mpfr_add_ui(x2.value, x2.value, 1, RND);
  mpfr_sqr(y2.value, y.value, RND);
  mpfr_add_ui(y2.value, y2.value, 1, RND);
  mpfr_sub_ui(numerator.value, y.value, 1, RND);
  mpfr_mul(numerator.value, numerator.value, x2.value, RND);
  mpfr_mul(denominator.value, M.value, y2.value, RND);
  mpfr_div(correction.value, numerator.value, denominator.value, RND);
  mpfr_sub(x.value, x.value, correction.value, RND);
}

Real smallest_root(unsigned long m, unsigned long depth,
                   mpfr_prec_t target_precision,
                   bool leave_final_newton_for_fusion = false) {
  mpfr_prec_t p = std::min<mpfr_prec_t>(96, target_precision);
  Real x(p), M(p);
  set_power(M, m, depth);
  mpfr_set_ui(x.value, 7, RND);
  mpfr_div(x.value, x.value, M.value, RND);
  mpfr_div_ui(x.value, x.value, 8, RND);
  for (int i = 0; i < 7; ++i) root_newton_step(x, m, depth);

  while (p < target_precision) {
    p = std::min<mpfr_prec_t>(target_precision, 2 * p);
    mpfr_prec_round(x.value, p, RND);
    root_newton_step(x, m, depth);
    if (p < target_precision || !leave_final_newton_for_fusion)
      root_newton_step(x, m, depth);
  }
  return x;
}

unsigned long recommended_depth(unsigned long bits, unsigned long m) {
  return static_cast<unsigned long>(
             std::ceil(std::sqrt(bits / std::log2(static_cast<double>(m))))) +
         3;
}

Real pi_collision(unsigned long bits, unsigned long m, unsigned long depth,
                  double* root_seconds, double* fusion_seconds) {
  const mpfr_prec_t p = static_cast<mpfr_prec_t>(bits + 96);
  const auto root_begin = std::chrono::steady_clock::now();
  // The last Newton evaluation is left for the fused pass below.  Its tangent
  // tower is exactly the tower needed by the geometric extrapolation.
  Real x = smallest_root(m, depth, p, true);
  const auto root_end = std::chrono::steady_clock::now();
  if (root_seconds)
    *root_seconds = std::chrono::duration<double>(root_end - root_begin).count();
  const auto fusion_begin = root_end;
  Real M(p), r(p), qpoch(p), rpower(p), one_minus(p), weight(p), scale(p),
      head_power(p), tail_power(p), total(p), term(p), factor(p), temp(p),
      tangent_square(p), tangent_denominator(p), tangent_numerator(p),
      tangent_temp(p), x_square(p), correction(p), radix_squared(p);
  set_power(M, m, depth);
  mpfr_set_ui(r.value, m, RND);
  mpfr_sqr(r.value, r.value, RND);
  mpfr_set(radix_squared.value, r.value, RND);
  mpfr_ui_div(r.value, 1, r.value, RND);

  // (r;r)_N and lambda_N = 1/(r;r)_N.
  mpfr_set_ui(qpoch.value, 1, RND);
  mpfr_set(rpower.value, r.value, RND);
  for (unsigned long k = 0; k < depth; ++k) {
    mpfr_ui_sub(one_minus.value, 1, rpower.value, RND);
    mpfr_mul(qpoch.value, qpoch.value, one_minus.value, RND);
    mpfr_mul(rpower.value, rpower.value, r.value, RND);
  }
  mpfr_ui_div(weight.value, 1, qpoch.value, RND);
  mpfr_mul_ui(scale.value, M.value, 4, RND);
  mpfr_set(head_power.value, r.value, RND);
  mpfr_pow_ui(tail_power.value, r.value, depth, RND);
  mpfr_set_zero(total.value, 0);

  for (unsigned long s = 0; s <= depth; ++s) {
    mpfr_mul(term.value, weight.value, scale.value, RND);
    mpfr_mul(term.value, term.value, x.value, RND);
    mpfr_add(total.value, total.value, term.value, RND);
    if (s == depth) break;

    // The weight update and D_m transport are deliberately streamed together.
    tangent_transport_in_place(x, m, tangent_square, tangent_denominator,
                               tangent_numerator, tangent_temp);
    mpfr_ui_sub(factor.value, 1, tail_power.value, RND);
    mpfr_mul(factor.value, factor.value, head_power.value, RND);
    mpfr_neg(factor.value, factor.value, RND);
    mpfr_ui_sub(temp.value, 1, head_power.value, RND);
    mpfr_div(factor.value, factor.value, temp.value, RND);
    mpfr_mul(weight.value, weight.value, factor.value, RND);
    mpfr_div_ui(scale.value, scale.value, m, RND);
    mpfr_mul(head_power.value, head_power.value, r.value, RND);
    mpfr_mul(tail_power.value, tail_power.value, radix_squared.value, RND);
  }

  // If c is the Newton correction to the root, F(x-c) = F(x)-cF'(x)
  // up to a quadratic residual.  The factors M and (1+x^2) cancel, leaving
  // 4(D_M(x)-1)H/(1+D_M(x)^2), where
  // H=sum_s w_s(1+x_s^2).  The same geometric interpolation makes H an
  // n-bit approximation to sec^2(0)=1, so replacing it by 1 changes only
  // terms far below the requested precision.
  mpfr_sub_ui(correction.value, x.value, 1, RND);
  mpfr_mul_ui(correction.value, correction.value, 4, RND);
  mpfr_sqr(x_square.value, x.value, RND);
  mpfr_add_ui(x_square.value, x_square.value, 1, RND);
  mpfr_div(correction.value, correction.value, x_square.value, RND);
  mpfr_sub(total.value, total.value, correction.value, RND);
  const auto fusion_end = std::chrono::steady_clock::now();
  if (fusion_seconds)
    *fusion_seconds =
        std::chrono::duration<double>(fusion_end - fusion_begin).count();
  return total;
}

Real pi_agm(unsigned long bits) {
  const mpfr_prec_t p = static_cast<mpfr_prec_t>(bits + 32);
  Real a(p), b(p), t(p), power(p), next_a(p), next_b(p), delta(p), temp(p),
      result(p);
  mpfr_set_ui(a.value, 1, RND);
  mpfr_set_ui(b.value, 2, RND);
  mpfr_sqrt(b.value, b.value, RND);
  mpfr_ui_div(b.value, 1, b.value, RND);
  mpfr_set_ui(t.value, 1, RND);
  mpfr_div_ui(t.value, t.value, 4, RND);
  mpfr_set_ui(power.value, 1, RND);

  const unsigned iterations =
      static_cast<unsigned>(std::ceil(std::log2(bits))) + 1;
  for (unsigned i = 0; i < iterations; ++i) {
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

struct Split {
  mpz_t P, Q, T;
  Split() {
    mpz_inits(P, Q, T, nullptr);
  }
  ~Split() { mpz_clears(P, Q, T, nullptr); }
};

void chudnovsky_split(unsigned long a, unsigned long b, Split& out) {
  constexpr unsigned long A = 13591409;
  constexpr unsigned long B = 545140134;
  constexpr unsigned long C3_OVER_24 = 10939058860032000UL;
  if (b - a == 1) {
    if (a == 0) {
      mpz_set_ui(out.P, 1);
      mpz_set_ui(out.Q, 1);
      mpz_set_ui(out.T, A);
      return;
    }
    mpz_set_ui(out.P, 6 * a - 5);
    mpz_mul_ui(out.P, out.P, 2 * a - 1);
    mpz_mul_ui(out.P, out.P, 6 * a - 1);
    mpz_set_ui(out.Q, a);
    mpz_pow_ui(out.Q, out.Q, 3);
    mpz_mul_ui(out.Q, out.Q, C3_OVER_24);
    mpz_set_ui(out.T, B);
    mpz_mul_ui(out.T, out.T, a);
    mpz_add_ui(out.T, out.T, A);
    mpz_mul(out.T, out.T, out.P);
    if (a & 1UL) mpz_neg(out.T, out.T);
    return;
  }

  const unsigned long middle = (a + b) / 2;
  Split left, right;
  chudnovsky_split(a, middle, left);
  chudnovsky_split(middle, b, right);
  mpz_mul(out.P, left.P, right.P);
  mpz_mul(out.Q, left.Q, right.Q);
  mpz_mul(out.T, left.T, right.Q);
  mpz_addmul(out.T, left.P, right.T);
}

Real pi_chudnovsky(unsigned long bits) {
  const mpfr_prec_t p = static_cast<mpfr_prec_t>(bits + 32);
  const unsigned long terms =
      static_cast<unsigned long>(std::ceil(bits / 47.110413138215842)) + 1;
  Split split;
  chudnovsky_split(0, terms, split);
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

double correct_bits(const Real& value) {
  const mpfr_prec_t p = value.precision() + 64;
  Real extended(p), reference(p), error(p), logarithm(p);
  mpfr_set(extended.value, value.value, RND);
  mpfr_const_pi(reference.value, RND);
  mpfr_sub(error.value, extended.value, reference.value, RND);
  mpfr_abs(error.value, error.value, RND);
  if (mpfr_zero_p(error.value)) return INFINITY;
  mpfr_log2(logarithm.value, error.value, RND);
  return -mpfr_get_d(logarithm.value, RND);
}

template <typename Function>
std::pair<Real, double> timed(Function&& function) {
  const auto begin = std::chrono::steady_clock::now();
  Real result = function();
  const auto end = std::chrono::steady_clock::now();
  return {result, std::chrono::duration<double>(end - begin).count()};
}

}  // namespace

int main(int argc, char** argv) {
  unsigned long bits = 10000;
  unsigned long m = 2;
  unsigned long depth = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--bits" && i + 1 < argc) bits = std::stoul(argv[++i]);
    else if (arg == "--m" && i + 1 < argc) m = std::stoul(argv[++i]);
    else if (arg == "--depth" && i + 1 < argc) depth = std::stoul(argv[++i]);
    else {
      std::cerr << "usage: " << argv[0]
                << " [--bits N] [--m N] [--depth N]\n";
      return 2;
    }
  }
  if (bits < 64 || m < 2) {
    std::cerr << "require --bits >= 64 and --m >= 2\n";
    return 2;
  }
  if (depth == 0) depth = recommended_depth(bits, m);

  double root_seconds = 0.0, fusion_seconds = 0.0;
  auto collision = timed([&] {
    return pi_collision(bits, m, depth, &root_seconds, &fusion_seconds);
  });
  auto chudnovsky = timed([&] { return pi_chudnovsky(bits); });
  auto agm = timed([&] { return pi_agm(bits); });

  std::cout << std::setprecision(9)
            << "{\n"
            << "  \"bits\": " << bits << ",\n"
            << "  \"m\": " << m << ",\n"
            << "  \"depth\": " << depth << ",\n"
            << "  \"collision\": {\"seconds\": " << collision.second
            << ", \"root_seconds\": " << root_seconds
            << ", \"fusion_seconds\": " << fusion_seconds
            << ", \"correct_bits\": " << correct_bits(collision.first) << "},\n"
            << "  \"chudnovsky\": {\"seconds\": " << chudnovsky.second
            << ", \"correct_bits\": " << correct_bits(chudnovsky.first) << "},\n"
            << "  \"agm\": {\"seconds\": " << agm.second
            << ", \"correct_bits\": " << correct_bits(agm.first) << "}\n"
            << "}\n";
  return 0;
}
