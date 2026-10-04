#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace rvfx::entropy {
constexpr int side = 33, bins = 64;
using Color = std::array<float,3>;
using Matrix = std::array<std::array<double,3>,3>;
enum class ChainOrder { ColorContrastBrightness, ColorBrightnessContrast,
    ContrastColorBrightness, ContrastBrightnessColor,
    BrightnessColorContrast, BrightnessContrastColor };
struct Config {
    float decorrelation = .65f;
    float allocation = .8f;
    float noise_floor = .015f;
    float max_gain = 6.f;
    float adaptation_seconds = 1.5f;
    float max_change_per_second = .25f;
    bool chains = false;
    float rgb_amount = 1.f;
    float brightness_amount = 0.f;
    float contrast_amount = 0.f;
    float brightness_reference = .5f;
    ChainOrder chain_order = ChainOrder::ColorContrastBrightness;
};
struct Diagnostics {
    std::size_t samples = 0;
    float mean_entropy = 0;
    Matrix covariance{}, transform{};
    std::array<std::array<float,bins+1>,3> curves{};
    // Independent source-luma statistics; never measured after an adjustment.
    std::array<float,bins+1> contrast_cdf{};
    float midtone = .5f, brightness_gain = 1.f;
    bool tone_active = false;
};
// RGBA float atlas: width side*side, height side; x = blue*side+red, y=green.
class Profile {
public:
    Profile();
    void reset();
    bool analyze(const std::uint8_t* rgba, int width, int height,
                 std::size_t stride, const Config& config);
    bool advance(double seconds, const Config& config);
    // Same float update as advance; pack the result while it is in cache.
    // Returns true only when the GPU-visible UNORM16 table changes.
    bool advance_and_pack(double seconds, const Config& config,
                          std::vector<std::uint16_t>& packed);
    Color apply(Color rgb) const;
    // Direct target composition, before LUT approximation or temporal smoothing.
    // Requires a successful analysis with matching RGB/reference settings;
    // stage amounts and order may differ without another observation.
    Color target_color(Color rgb, const Config& config) const;
    const std::vector<float>& atlas() const { return current_; }
    const Diagnostics& diagnostics() const { return diagnostic_; }
private:
    struct Sample { Color rgb{}; float h=0; std::uint16_t family=0; bool valid=false; };
    std::vector<Sample> samples_;
    std::vector<float> current_, target_;
    bool advance_impl(double seconds, const Config& config, std::vector<std::uint16_t>* packed);
    float distance_=0;
    bool distance_dirty_=false;
    Diagnostics diagnostic_{};
    Color mean_{};
};
Matrix whitening(const Matrix& covariance, double floor, double max_gain);
} // namespace rvfx::entropy
