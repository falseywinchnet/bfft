#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

using kernel = void (*)(const double*, const double*, const double*, double*,
                        double*, std::size_t, double);

void incumbent(const double* x, const double* y, const double*, double* px,
               double* py, std::size_t n, double tau) {
    for (std::size_t i = 0; i < n; ++i) {
        const double r = std::sqrt(x[i] * x[i] + y[i] * y[i]);
        const double shrink =
            std::fmax(r - tau, 0.0) / std::fmax(r, 1e-12);
        const double project = 1.0 - shrink;
        px[i] = project * x[i];
        py[i] = project * y[i];
    }
}

void squared_branch(const double* x, const double* y, const double*,
                    double* px, double* py, std::size_t n, double tau) {
    const double tau2 = tau * tau;
    for (std::size_t i = 0; i < n; ++i) {
        const double r2 = x[i] * x[i] + y[i] * y[i];
        const double project =
            r2 <= tau2 ? 1.0 : tau / std::sqrt(r2);
        px[i] = project * x[i];
        py[i] = project * y[i];
    }
}

template <int Steps>
void float_seed_newton(const double* x, const double* y, const double*,
                       double* px, double* py, std::size_t n, double tau) {
    const double tau2 = tau * tau;
    for (std::size_t i = 0; i < n; ++i) {
        const double r2 = x[i] * x[i] + y[i] * y[i];
        double project = 1.0;
        if (r2 > tau2) {
            double q = 1.0 / std::sqrt(static_cast<float>(r2));
            q *= 1.5 - 0.5 * r2 * q * q;
            if constexpr (Steps == 2)
                q *= 1.5 - 0.5 * r2 * q * q;
            project = tau * q;
        }
        px[i] = project * x[i];
        py[i] = project * y[i];
    }
}

void known_radius(const double* x, const double* y, const double* radius,
                  double* px, double* py, std::size_t n, double tau) {
    for (std::size_t i = 0; i < n; ++i) {
        const double project = std::fmin(1.0, tau / radius[i]);
        px[i] = project * x[i];
        py[i] = project * y[i];
    }
}

void copy_floor(const double* x, const double* y, const double*, double* px,
                double* py, std::size_t n, double) {
    for (std::size_t i = 0; i < n; ++i) {
        px[i] = x[i];
        py[i] = y[i];
    }
}

double run(kernel fn, const std::vector<double>& x,
           const std::vector<double>& y, const std::vector<double>& radius,
           std::vector<double>& px, std::vector<double>& py, double tau,
           int repeats) {
    fn(x.data(), y.data(), radius.data(), px.data(), py.data(), x.size(),
       tau);
    const auto start = std::chrono::steady_clock::now();
    for (int repeat = 0; repeat < repeats; ++repeat)
        fn(x.data(), y.data(), radius.data(), px.data(), py.data(), x.size(),
           tau);
    const auto stop = std::chrono::steady_clock::now();
    volatile double checksum = px[x.size() / 3] + py[x.size() * 2 / 3];
    (void)checksum;
    return std::chrono::duration<double, std::nano>(stop - start).count() /
        (double(repeats) * double(x.size()));
}

double max_error(const std::vector<double>& ax,
                 const std::vector<double>& ay,
                 const std::vector<double>& bx,
                 const std::vector<double>& by) {
    double error = 0.0;
    for (std::size_t i = 0; i < ax.size(); ++i) {
        error = std::max(error, std::abs(ax[i] - bx[i]));
        error = std::max(error, std::abs(ay[i] - by[i]));
    }
    return error;
}

}  // namespace

int main() {
    constexpr std::size_t n = std::size_t(1) << 20;
    constexpr double tau = 4.0;
    constexpr int repeats = 80;
    std::vector<double> x(n), y(n), radius(n), refx(n), refy(n), px(n), py(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double angle = 0.017 * double(i);
        // Deliberately mix interior, boundary-near, and exterior points.
        const double r = (i % 3 == 0) ? 0.75 * tau :
            (i % 3 == 1 ? (0.999 + 0.002 * double(i % 17) / 16.0) * tau :
                           (1.25 + double(i % 29) / 29.0) * tau);
        x[i] = r * std::cos(angle);
        y[i] = r * std::sin(angle);
        radius[i] = r;
    }
    incumbent(x.data(), y.data(), radius.data(), refx.data(), refy.data(), n,
              tau);

    const struct {
        const char* name;
        kernel fn;
    } cases[] = {
        {"copy/memory floor", copy_floor},
        {"incumbent exact", incumbent},
        {"squared-radius branch", squared_branch},
        {"float seed + 1 Newton", float_seed_newton<1>},
        {"float seed + 2 Newton", float_seed_newton<2>},
        {"precomputed radius", known_radius},
    };

    for (const auto& item : cases) {
        std::vector<double> samples;
        for (int trial = 0; trial < 7; ++trial)
            samples.push_back(
                run(item.fn, x, y, radius, px, py, tau, repeats));
        std::sort(samples.begin(), samples.end());
        item.fn(x.data(), y.data(), radius.data(), px.data(), py.data(), n,
                tau);
        std::printf("%-24s %6.3f ns/pixel  max error %.3e\n", item.name,
                    samples[samples.size() / 2],
                    max_error(refx, refy, px, py));
    }
}
