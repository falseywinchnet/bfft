// Frozen pre-optimization implementation for behavioral equivalence tests.
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace rvfx::entropy_v1 {
constexpr int side = 33, bins = 64;
using Color = std::array<float,3>;
using Matrix = std::array<std::array<double,3>,3>;
struct Config {
    float decorrelation = .65f;
    float allocation = .8f;
    float noise_floor = .015f;
    float max_gain = 6.f;
    float adaptation_seconds = 1.5f;
    float max_change_per_second = .25f;
};
struct Diagnostics {
    std::size_t samples = 0;
    float mean_entropy = 0;
    Matrix covariance{}, transform{};
    std::array<std::array<float,bins+1>,3> curves{};
};
// RGBA float atlas: width side*side, height side; x = blue*side+red, y=green.
class Profile {
public:
    Profile();
    void reset();
    bool analyze(const std::uint8_t* rgba, int width, int height,
                 std::size_t stride, const Config& config);
    bool advance(double seconds, const Config& config);
    Color apply(Color rgb) const;
    const std::vector<float>& atlas() const { return current_; }
    const Diagnostics& diagnostics() const { return diagnostic_; }
private:
    std::vector<float> current_, target_;
    Diagnostics diagnostic_{};
};
Matrix whitening(const Matrix& covariance, double floor, double max_gain);
} // namespace rvfx::entropy_v1
