#include <bfft/bodft.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <thread>
#include <vector>

namespace {
void check(const bool condition, const char* const message) {
    if (condition) return;
    std::fprintf(stderr, "%s\n", message);
    std::abort();
}
class Storage final {
public:
    Storage(const std::size_t bytes, const std::size_t alignment)
        : bytes_(bytes), alignment_(alignment), data_(::operator new(bytes, std::align_val_t(alignment))) {
        std::memset(data_, 0xa5, bytes);
    }
    ~Storage() { ::operator delete(data_, std::align_val_t(alignment_)); }
    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;
    void* data() noexcept { return data_; }
    std::size_t size() const noexcept { return bytes_; }
    bool untouched() const noexcept {
        const unsigned char* const bytes = static_cast<const unsigned char*>(data_);
        for (std::size_t index = 0; index < bytes_; ++index) {
            if (bytes[index] != 0xa5) return false;
        }
        return true;
    }
private:
    const std::size_t bytes_;
    const std::size_t alignment_;
    void* const data_;
};

struct Worker final {
    const bodft_prepared_plan* plan{nullptr};
    bodft_workspace* workspace{nullptr};
    std::vector<double> input;
    std::vector<double> output;
    std::vector<bfft_complex> spectrum;
};
void round_trip(Worker& worker) {
    for (unsigned int repeat = 0; repeat < 16; ++repeat) {
        check(bodft_forward_prepared(worker.plan, worker.workspace,
            worker.input.data(), worker.input.size(), worker.spectrum.data(), worker.spectrum.size()) == BFFT_OK,
            "prepared forward");
        check(bodft_inverse_prepared(worker.plan, worker.workspace,
            worker.spectrum.data(), worker.spectrum.size(), worker.output.data(), worker.output.size()) == BFFT_OK,
            "prepared inverse");
        for (std::size_t index = 0; index < worker.input.size(); ++index)
            check(std::abs(worker.input[index] - worker.output[index]) < 1e-11, "prepared round trip");
    }
}

void test_double(const std::size_t size) {
    bodft_storage_requirements requirements{};
    check(bodft_query_storage(size, BODFT_PRECISION_F64, &requirements) == BFFT_OK, "query double");
    Storage plan_bytes(requirements.plan_bytes, requirements.plan_alignment);
    Storage first_bytes(requirements.workspace_bytes, requirements.workspace_alignment);
    Storage second_bytes(requirements.workspace_bytes, requirements.workspace_alignment);
    const bodft_prepared_plan* plan = nullptr;
    check(bodft_prepare(size, BODFT_PRECISION_F64, plan_bytes.data(), plan_bytes.size() - 1, &plan)
        == BFFT_ERROR_INVALID_ARGUMENT && plan == nullptr && plan_bytes.untouched(), "short plan preserved");
    unsigned char* const misaligned = static_cast<unsigned char*>(plan_bytes.data()) + 1;
    check(bodft_prepare(size, BODFT_PRECISION_F64, misaligned, plan_bytes.size(), &plan)
        == BFFT_ERROR_INVALID_ARGUMENT && plan_bytes.untouched(), "plan alignment checked");
    check(bodft_prepare(size, BODFT_PRECISION_F64, plan_bytes.data(),
        std::numeric_limits<std::size_t>::max(), &plan) == BFFT_ERROR_INVALID_ARGUMENT
        && plan == nullptr && plan_bytes.untouched(), "overflowing supplied extent rejected");
    check(bodft_prepare(size, BODFT_PRECISION_F64, plan_bytes.data(), plan_bytes.size(), &plan)
        == BFFT_OK, "prepare exact plan storage");
    bodft_workspace* first = nullptr;
    bodft_workspace* second = nullptr;
    check(bodft_prepare_workspace(plan, first_bytes.data(), first_bytes.size() - 1, &first)
        == BFFT_ERROR_INVALID_ARGUMENT && first == nullptr && first_bytes.untouched(), "short workspace preserved");
    check(bodft_prepare_workspace(plan, plan_bytes.data(), plan_bytes.size(), &first)
        == BFFT_ERROR_INVALID_ARGUMENT && first == nullptr, "plan/workspace overlap rejected");
    check(bodft_prepare_workspace(plan, first_bytes.data(), first_bytes.size(), &first) == BFFT_OK, "first workspace");
    check(bodft_prepare_workspace(plan, second_bytes.data(), second_bytes.size(), &second) == BFFT_OK, "second workspace");
    const unsigned char* const start = static_cast<const unsigned char*>(plan_bytes.data());
    const std::vector<unsigned char> plan_snapshot(start, start + plan_bytes.size());
    Worker a{plan, first, std::vector<double>(size), std::vector<double>(size), std::vector<bfft_complex>(size / 2)};
    Worker b{plan, second, std::vector<double>(size), std::vector<double>(size), std::vector<bfft_complex>(size / 2)};
    for (std::size_t index = 0; index < size; ++index) {
        a.input[index] = std::sin(static_cast<double>(index) * 0.13);
        b.input[index] = std::cos(static_cast<double>(index) * 0.23);
    }
    const std::vector<bfft_complex> before = a.spectrum;
    const unsigned char* const scratch_start = static_cast<const unsigned char*>(first_bytes.data());
    const std::vector<unsigned char> scratch_snapshot(scratch_start, scratch_start + first_bytes.size());
    check(bodft_forward_prepared(plan, first, a.input.data(), size - 1, a.spectrum.data(), size / 2)
        == BFFT_ERROR_INVALID_ARGUMENT, "short active input rejected");
    check(std::memcmp(before.data(), a.spectrum.data(), before.size() * sizeof(bfft_complex)) == 0,
        "rejected output untouched");
    check(bodft_forward_prepared(plan, first, a.input.data(), size,
        reinterpret_cast<bfft_complex*>(a.input.data()), size / 2) == BFFT_ERROR_INVALID_ARGUMENT,
        "input/output overlap rejected");
    check(bodft_forward_prepared(plan, first, a.input.data(), size,
        static_cast<bfft_complex*>(first_bytes.data()), size / 2) == BFFT_ERROR_INVALID_ARGUMENT,
        "output/workspace overlap rejected");
    check(bodft_forward_prepared(plan, first, a.input.data(), size, a.spectrum.data(), size / 2 - 1)
        == BFFT_ERROR_INVALID_ARGUMENT, "short output rejected");
    const unsigned char* const input_bytes = reinterpret_cast<const unsigned char*>(a.input.data());
    check(bodft_forward_prepared(plan, first, reinterpret_cast<const double*>(input_bytes + 1), size,
        a.spectrum.data(), size / 2) == BFFT_ERROR_INVALID_ARGUMENT, "input alignment checked");
    check(std::memcmp(scratch_snapshot.data(), first_bytes.data(), scratch_snapshot.size()) == 0,
        "validation failures preserve workspace");
    std::thread first_thread(round_trip, std::ref(a));
    std::thread second_thread(round_trip, std::ref(b));
    first_thread.join();
    second_thread.join();
    check(std::memcmp(plan_snapshot.data(), plan_bytes.data(), plan_snapshot.size()) == 0,
        "shared plan stays byte immutable");
    bodft_plan* legacy = nullptr;
    check(bodft_plan_create(size, &legacy) == BFFT_OK, "legacy create");
    std::vector<bfft_complex> reference(size / 2);
    check(bodft_forward(legacy, a.input.data(), reference.data()) == BFFT_OK, "legacy forward");
    for (std::size_t index = 0; index < reference.size(); ++index) {
        check(std::abs(reference[index].re - a.spectrum[index].re) < 1e-11, "legacy real agreement");
        check(std::abs(reference[index].im - a.spectrum[index].im) < 1e-11, "legacy imaginary agreement");
    }
    bodft_plan_destroy(legacy);
    std::printf("%zu,%zu,%zu\n", size, requirements.plan_bytes, requirements.workspace_bytes);
}

void test_workspace_shape() {
    bodft_storage_requirements small{};
    bodft_storage_requirements large{};
    check(bodft_query_storage(8, BODFT_PRECISION_F64, &small) == BFFT_OK, "small query");
    check(bodft_query_storage(16, BODFT_PRECISION_F64, &large) == BFFT_OK, "large query");
    Storage first_bytes(small.plan_bytes, small.plan_alignment);
    Storage second_bytes(large.plan_bytes, large.plan_alignment);
    Storage work_bytes(small.workspace_bytes, small.workspace_alignment);
    const bodft_prepared_plan* first = nullptr;
    const bodft_prepared_plan* second = nullptr;
    bodft_workspace* work = nullptr;
    check(bodft_prepare(8, BODFT_PRECISION_F64, first_bytes.data(), first_bytes.size(), &first) == BFFT_OK,
        "small plan");
    check(bodft_prepare(16, BODFT_PRECISION_F64, second_bytes.data(), second_bytes.size(), &second) == BFFT_OK,
        "large plan");
    check(bodft_prepare_workspace(first, work_bytes.data(), work_bytes.size(), &work) == BFFT_OK, "small work");
    double input[16]{};
    bfft_complex output[8]{};
    check(bodft_forward_prepared(second, work, input, 16, output, 8) == BFFT_ERROR_INVALID_ARGUMENT,
        "workspace shape mismatch");
    // Reuse the larger allocation for a new same-shape immutable plan.
    check(bodft_prepare(8, BODFT_PRECISION_F64, second_bytes.data(), second_bytes.size(), &second) == BFFT_OK,
        "replacement plan");
    check(bodft_forward_prepared(second, work, input, 8, output, 4) == BFFT_OK,
        "workspace independent of plan identity");
}

void test_float(const std::size_t size) {
    bodft_storage_requirements requirements{};
    check(bodft_query_storage(size, BODFT_PRECISION_F32, &requirements) == BFFT_OK, "query float");
    Storage plan_bytes(requirements.plan_bytes, requirements.plan_alignment);
    Storage work_bytes(requirements.workspace_bytes, requirements.workspace_alignment);
    const bodft_prepared_plan* plan = nullptr;
    bodft_workspace* workspace = nullptr;
    check(bodft_prepare(size, BODFT_PRECISION_F32, plan_bytes.data(), plan_bytes.size(), &plan) == BFFT_OK, "float plan");
    check(bodft_prepare_workspace(plan, work_bytes.data(), work_bytes.size(), &workspace) == BFFT_OK, "float workspace");
    std::vector<float> input(size, 0.0F);
    std::vector<float> output(size, 0.0F);
    std::vector<bfft_complex_f32> spectrum(size / 2);
    for (std::size_t index = 0; index < size; ++index) input[index] = std::sin(static_cast<float>(index) * 0.17F);
    check(bodft_forward_prepared_f32(plan, workspace, input.data(), size, spectrum.data(), size / 2) == BFFT_OK,
        "float forward");
    check(bodft_inverse_prepared_f32(plan, workspace, spectrum.data(), size / 2, output.data(), size) == BFFT_OK,
        "float inverse");
    for (std::size_t index = 0; index < size; ++index)
        check(std::abs(input[index] - output[index]) < 2e-5F, "float round trip");
    std::vector<double> wrong_input(size, 0.0);
    std::vector<bfft_complex> wrong_output(size / 2);
    check(bodft_forward_prepared(plan, workspace, wrong_input.data(), size, wrong_output.data(), size / 2)
        == BFFT_ERROR_INVALID_ARGUMENT, "precision mismatch rejected");
}
} // namespace

int main() {
    bodft_storage_requirements preserved{1, 2, 3, 4};
    const std::size_t invalid_sizes[]{0, 1, 3, std::numeric_limits<std::size_t>::max()};
    for (const std::size_t size : invalid_sizes)
        check(bodft_query_storage(size, BODFT_PRECISION_F64, &preserved) == BFFT_ERROR_INVALID_ARGUMENT
            && preserved.plan_bytes == 1 && preserved.workspace_bytes == 3, "invalid query preserves result");
    std::puts("size,plan_bytes,workspace_bytes");
    test_workspace_shape();
    for (std::size_t size = 2; size <= 8192; size *= 2) {
        test_double(size);
        test_float(size);
    }
    return 0;
}
