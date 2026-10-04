#pragma once
// Meyer cartoon structure for the mosaic: luma on a power-of-two lattice with
// symmetric reflection padding (BFFT Cartoon's layout), resampled back to the
// analysis grid in [0,1].  Resampling maps are cached per size.
#include <cstddef>
#include <cstdint>
#include <vector>

struct bfft_meyer_plan;

namespace mosaic {

class Cartoon {
public:
    Cartoon() = default;
    Cartoon(const Cartoon&) = delete;
    Cartoon& operator=(const Cartoon&) = delete;
    ~Cartoon();
    // rgba: straight-alpha RGBA8, stride in bytes.  lattice: 256..2048.
    // out: analysis-size structure, clamped to [0,1].
    bool run(const std::uint8_t* rgba, int width, int height, std::size_t stride, int lattice,
             int threads, std::vector<float>& out);

private:
    struct Tap { int i0, i1; float t; };
    bfft_meyer_plan* plan_ = nullptr;
    int cw_ = 0, ch_ = 0, threads_ = -1, w_ = 0, h_ = 0;
    std::vector<float> luma_;
    std::vector<double> image_, cartoon_, texture_;
    std::vector<Tap> in_x_, in_y_, out_x_, out_y_;
};

} // namespace mosaic
