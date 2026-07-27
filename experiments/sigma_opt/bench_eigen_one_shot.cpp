// Isolated benchmark for an exact one-shot solve of Sigma's SPD normal matrix.
//
// Build:
//   c++ -O3 -DNDEBUG -std=c++17 -I/opt/homebrew/include/eigen3 \
//       experiments/sigma_opt/bench_eigen_one_shot.cpp \
//       -o /tmp/bench_eigen_one_shot
//
// Input is a symmetric Matrix Market file written by the companion study.

#include <Eigen/Core>
#include <Eigen/SparseCholesky>
#include <unsupported/Eigen/SparseExtra>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
using Sparse = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;
using Dense = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

static double milliseconds(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

static double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const std::size_t n = values.size();
  return n % 2 ? values[n / 2]
               : 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: bench_eigen_one_shot matrix.mtx\n";
    return 2;
  }
  Sparse a;
  if (!Eigen::loadMarket(a, std::string(argv[1]))) {
    std::cerr << "could not load " << argv[1] << "\n";
    return 2;
  }
  a.makeCompressed();
  Dense rhs(a.rows(), 3);
  for (Eigen::Index i = 0; i < rhs.rows(); ++i) {
    rhs(i, 0) = std::sin(0.013 * static_cast<double>(i));
    rhs(i, 1) = std::cos(0.017 * static_cast<double>(i));
    rhs(i, 2) = std::sin(0.019 * static_cast<double>(i) + 0.3);
  }

  std::vector<double> factor_ms;
  std::vector<double> solve_ms;
  double residual = 0.0;
  Eigen::Index factor_nnz = 0;
  for (int repeat = 0; repeat < 9; ++repeat) {
    Eigen::SimplicialLDLT<Sparse, Eigen::Lower, Eigen::AMDOrdering<int>> ldlt;
    auto t0 = Clock::now();
    ldlt.compute(a);
    factor_ms.push_back(milliseconds(t0));
    if (ldlt.info() != Eigen::Success) {
      std::cerr << "LDLT factorization failed\n";
      return 1;
    }
    t0 = Clock::now();
    Dense x = ldlt.solve(rhs);
    solve_ms.push_back(milliseconds(t0));
    if (ldlt.info() != Eigen::Success) {
      std::cerr << "LDLT solve failed\n";
      return 1;
    }
    residual = (a * x - rhs).norm() / rhs.norm();
    factor_nnz = ldlt.matrixL().nestedExpression().nonZeros();
  }

  std::cout << "n " << a.rows() << "  nnz(A) " << a.nonZeros()
            << "  nnz(L) " << factor_nnz << "\n";
  std::cout << "factor best "
            << *std::min_element(factor_ms.begin(), factor_ms.end())
            << " ms  median " << median(factor_ms) << " ms\n";
  std::cout << "solve3 best "
            << *std::min_element(solve_ms.begin(), solve_ms.end())
            << " ms  median " << median(solve_ms) << " ms\n";
  std::cout << "relative residual " << residual << "\n";
  return 0;
}
