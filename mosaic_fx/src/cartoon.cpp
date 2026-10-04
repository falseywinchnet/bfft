#include "mosaic/cartoon.hpp"

#include <bfft/meyer.h>

#include <algorithm>
#include <cmath>

namespace mosaic {
namespace {

int reflect(int v, int n)
{
    while (v < 0 || v >= n) v = v < 0 ? -v - 1 : 2 * n - v - 1;
    return v;
}

} // namespace

Cartoon::~Cartoon() { bfft_meyer_plan_destroy(plan_); }

bool Cartoon::run(const std::uint8_t* rgba, int w, int h, std::size_t stride, int lattice, int threads,
                  std::vector<float>& out)
{
    lattice = std::clamp(lattice, 256, 2048);
    // Wide frames use a half-height lattice, as BFFT Cartoon does.
    const int cw = lattice, ch = double(w) / h >= 1.3 ? lattice / 2 : lattice;
    if (!plan_ || cw != cw_ || ch != ch_ || threads != threads_) {
        bfft_meyer_plan_destroy(plan_);
        plan_ = nullptr;
        if (bfft_meyer_plan_create(std::size_t(ch), std::size_t(cw), 0.05, 40.0, 1, 1, 0.0, threads, &plan_) != BFFT_OK)
            return false;
        cw_ = cw; ch_ = ch; threads_ = threads;
        w_ = 0; // force map rebuild
    }
    const double scale = std::min(double(cw) / w, double(ch) / h);
    if (w != w_ || h != h_) {
        // Bilinear taps, built once per size: lattice <- analysis (with
        // reflection of the fitted content) and analysis <- lattice.
        const int iw = std::max(1, int(std::lround(w * scale))), ih = std::max(1, int(std::lround(h * scale)));
        auto tap = [](double s, int n) {
            s = std::clamp(s, 0.0, n - 1.0);
            const int i0 = int(s);
            return Tap{i0, std::min(n - 1, i0 + 1), float(s - i0)};
        };
        in_x_.resize(cw);
        in_y_.resize(ch);
        for (int x = 0; x < cw; ++x) in_x_[x] = tap((reflect(x, iw) + 0.5) / scale - 0.5, w);
        for (int y = 0; y < ch; ++y) in_y_[y] = tap((reflect(y, ih) + 0.5) / scale - 0.5, h);
        out_x_.resize(w);
        out_y_.resize(h);
        for (int x = 0; x < w; ++x) out_x_[x] = tap((x + 0.5) * scale - 0.5, cw);
        for (int y = 0; y < h; ++y) out_y_[y] = tap((y + 0.5) * scale - 0.5, ch);
        w_ = w;
        h_ = h;
    }
    luma_.resize(std::size_t(w) * h);
    for (int y = 0; y < h; ++y) {
        const std::uint8_t* p = rgba + std::size_t(y) * stride;
        float* l = &luma_[std::size_t(y) * w];
        for (int x = 0; x < w; ++x, p += 4) l[x] = 0.2126f * p[0] + 0.7152f * p[1] + 0.0722f * p[2];
    }
    image_.resize(std::size_t(cw) * ch);
    cartoon_.resize(image_.size());
    texture_.resize(image_.size());
    for (int y = 0; y < ch; ++y) {
        const Tap ty = in_y_[y];
        const float* r0 = &luma_[std::size_t(ty.i0) * w];
        const float* r1 = &luma_[std::size_t(ty.i1) * w];
        double* o = &image_[std::size_t(y) * cw];
        for (int x = 0; x < cw; ++x) {
            const Tap tx = in_x_[x];
            const float a = r0[tx.i0] + (r0[tx.i1] - r0[tx.i0]) * tx.t;
            const float b = r1[tx.i0] + (r1[tx.i1] - r1[tx.i0]) * tx.t;
            o[x] = a + (b - a) * ty.t;
        }
    }
    if (bfft_meyer_split(plan_, image_.data(), cartoon_.data(), texture_.data()) != BFFT_OK) return false;
    out.resize(std::size_t(w) * h);
    constexpr double inv = 1.0 / 255.0;
    for (int y = 0; y < h; ++y) {
        const Tap ty = out_y_[y];
        const double* r0 = &cartoon_[std::size_t(ty.i0) * cw];
        const double* r1 = &cartoon_[std::size_t(ty.i1) * cw];
        float* o = &out[std::size_t(y) * w];
        for (int x = 0; x < w; ++x) {
            const Tap tx = out_x_[x];
            const double a = r0[tx.i0] + (r0[tx.i1] - r0[tx.i0]) * tx.t;
            const double b = r1[tx.i0] + (r1[tx.i1] - r1[tx.i0]) * tx.t;
            o[x] = float(std::clamp((a + (b - a) * ty.t) * inv, 0.0, 1.0));
        }
    }
    return true;
}

} // namespace mosaic
