#ifndef BFFT_BFFT_HPP
#define BFFT_BFFT_HPP

#include <bfft/bfft.h>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace bfft {

using complex = bfft_complex;
using layout = bfft_layout;
using status = bfft_status;

inline std::string version_string() {
    return bfft_version_string();
}

inline std::string backend_name() {
    return bfft_backend_name();
}

class error : public std::runtime_error {
public:
    explicit error(status code)
        : std::runtime_error(bfft_status_string(code)), code_(code) {}

    status code() const noexcept {
        return code_;
    }

private:
    status code_;
};

class plan {
public:
    explicit plan(std::size_t n) {
        bfft_plan* raw = nullptr;
        status result = bfft_plan_create(n, &raw);
        if (result != BFFT_OK) {
            throw error(result);
        }
        impl_.reset(raw);
    }

    std::size_t size() const noexcept {
        return bfft_plan_size(impl_.get());
    }

    std::size_t bins() const noexcept {
        return bfft_plan_bins(impl_.get());
    }

    std::size_t work_size() const noexcept {
        return bfft_plan_work_size(impl_.get());
    }

    std::size_t native_scratch_size() const noexcept {
        return bfft_plan_native_scratch_size(impl_.get());
    }

    std::string standard_policy() const {
        return bfft_plan_standard_policy(impl_.get());
    }

    void forward(const double* input,
                 complex* output,
                 double* work,
                 complex* native_scratch) const {
        check(bfft_forward(impl_.get(), input, output, work, native_scratch));
    }

    std::vector<complex> forward(const std::vector<double>& input) const {
        if (input.size() != size()) {
            throw error(BFFT_ERROR_INVALID_ARGUMENT);
        }
        std::vector<complex> output(bins());
        std::vector<double> work(work_size());
        std::vector<complex> scratch(native_scratch_size());
        forward(input.data(), output.data(), work.data(), scratch.data());
        return output;
    }

    void forward_native(const double* input, complex* output, double* work) const {
        check(bfft_forward_native(impl_.get(), input, output, work));
    }

    void inverse(const complex* input, double* output) const {
        check(bfft_inverse(impl_.get(), input, output));
    }

    std::vector<double> inverse(const std::vector<complex>& input) const {
        if (input.size() != bins()) {
            throw error(BFFT_ERROR_INVALID_ARGUMENT);
        }
        std::vector<double> output(size());
        inverse(input.data(), output.data());
        return output;
    }

    void inverse_native(const complex* input, double* output) const {
        check(bfft_inverse_native(impl_.get(), input, output));
    }

    void forward_residues(const double* input, double* residues) const {
        check(bfft_forward_residues(impl_.get(), input, residues));
    }

    void inverse_residues(double* residues_signal) const {
        check(bfft_inverse_residues(impl_.get(), residues_signal));
    }

    void native_to_standard(const complex* native_input, complex* standard_output) const {
        check(bfft_native_to_standard(impl_.get(), native_input, standard_output));
    }

    void standard_to_native(const complex* standard_input, complex* native_output) const {
        check(bfft_standard_to_native(impl_.get(), standard_input, native_output));
    }

    std::size_t filter_size() const noexcept {
        return bfft_filter_size(impl_.get());
    }

    void residue_filter_from_standard(const complex* response, double* residue_filter) const {
        check(bfft_residue_filter_from_standard(impl_.get(), response, residue_filter));
    }

    void residue_filter_from_real(const double* response, double* residue_filter) const {
        check(bfft_residue_filter_from_real(impl_.get(), response, residue_filter));
    }

    void apply_residue_filter(double* residues, const double* residue_filter) const {
        check(bfft_apply_residue_filter(impl_.get(), residues, residue_filter));
    }

    void filter_signal(const double* input, const double* residue_filter, double* output) const {
        check(bfft_filter_signal(impl_.get(), input, residue_filter, output));
    }

private:
    struct deleter {
        void operator()(bfft_plan* plan_ptr) const noexcept {
            bfft_plan_destroy(plan_ptr);
        }
    };

    static void check(status result) {
        if (result != BFFT_OK) {
            throw error(result);
        }
    }

    std::unique_ptr<bfft_plan, deleter> impl_;
};

} // namespace bfft

#endif
