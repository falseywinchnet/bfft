#ifndef BFFT_BDCT_HPP
#define BFFT_BDCT_HPP

#include <bfft/bdct.h>
#include <bfft/bfft.hpp>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace bfft {

/* SIMD backend selected by the build target. */
inline std::string bdct_backend() {
    return bdct_backend_name();
}

/* Reusable native half-bin-shifted real transform. Forward maps N real samples
   to N/2 packed complex bins; inverse recovers the N real samples exactly. */
class half_shift {
public:
    explicit half_shift(std::size_t n) {
        bdct_plan* raw = nullptr;
        status result = bdct_plan_create(n, &raw);
        if (result != BFFT_OK) throw error(result);
        impl_.reset(raw);
    }

    std::size_t size() const noexcept { return bdct_plan_size(impl_.get()); }
    std::size_t bins() const noexcept { return bdct_plan_bins(impl_.get()); }

    void forward(const double* input, complex* output) const {
        check(bdct_forward(impl_.get(), input, output));
    }
    void inverse(const complex* input, double* output) const {
        check(bdct_inverse(impl_.get(), input, output));
    }
    void forward_f32(const float* input, complex_f32* output) const {
        check(bdct_forward_f32(impl_.get(), input, output));
    }
    void inverse_f32(const complex_f32* input, float* output) const {
        check(bdct_inverse_f32(impl_.get(), input, output));
    }

    /* Convenience overloads that allocate the result vector. */
    std::vector<complex> forward(const std::vector<double>& input) const {
        if (input.size() != size()) throw error(BFFT_ERROR_INVALID_ARGUMENT);
        std::vector<complex> output(bins());
        forward(input.data(), output.data());
        return output;
    }
    std::vector<double> inverse(const std::vector<complex>& input) const {
        if (input.size() != bins()) throw error(BFFT_ERROR_INVALID_ARGUMENT);
        std::vector<double> output(size());
        inverse(input.data(), output.data());
        return output;
    }

private:
    struct deleter {
        void operator()(bdct_plan* p) const noexcept { bdct_plan_destroy(p); }
    };
    static void check(status result) {
        if (result != BFFT_OK) throw error(result);
    }
    std::unique_ptr<bdct_plan, deleter> impl_;
};

/* Reusable orthogonal DCT-IV. Forward and inverse are real-to-real of length N;
   inverse applies the 2/N scale. */
class dctiv {
public:
    explicit dctiv(std::size_t n) {
        bdct_dctiv_plan* raw = nullptr;
        status result = bdct_dctiv_plan_create(n, &raw);
        if (result != BFFT_OK) throw error(result);
        impl_.reset(raw);
    }

    std::size_t size() const noexcept { return bdct_dctiv_plan_size(impl_.get()); }

    void forward(const double* input, double* output) const {
        check(bdct_dctiv_forward(impl_.get(), input, output));
    }
    void inverse(const double* input, double* output) const {
        check(bdct_dctiv_inverse(impl_.get(), input, output));
    }
    void forward_f32(const float* input, float* output) const {
        check(bdct_dctiv_forward_f32(impl_.get(), input, output));
    }
    void inverse_f32(const float* input, float* output) const {
        check(bdct_dctiv_inverse_f32(impl_.get(), input, output));
    }

    std::vector<double> forward(const std::vector<double>& input) const {
        if (input.size() != size()) throw error(BFFT_ERROR_INVALID_ARGUMENT);
        std::vector<double> output(size());
        forward(input.data(), output.data());
        return output;
    }
    std::vector<double> inverse(const std::vector<double>& input) const {
        if (input.size() != size()) throw error(BFFT_ERROR_INVALID_ARGUMENT);
        std::vector<double> output(size());
        inverse(input.data(), output.data());
        return output;
    }

private:
    struct deleter {
        void operator()(bdct_dctiv_plan* p) const noexcept { bdct_dctiv_plan_destroy(p); }
    };
    static void check(status result) {
        if (result != BFFT_OK) throw error(result);
    }
    std::unique_ptr<bdct_dctiv_plan, deleter> impl_;
};

} // namespace bfft

#endif
