#include "mosaic/mosaic.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>

namespace mosaic {
namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

struct Tables {
    float linear[256];
    Tables()
    {
        for (int i = 0; i < 256; ++i) {
            const float s = i / 255.f;
            linear[i] = s <= 0.04045f ? s / 12.92f : std::pow((s + 0.055f) / 1.055f, 2.4f);
        }
    }
};
const Tables& tables()
{
    static const Tables t;
    return t;
}

inline std::uint32_t hash32(std::uint32_t x)
{
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}
inline float unit(std::uint32_t h) { return float(h >> 8) * (1.f / 16777216.f); }

// Bit-seeded cube root refined by one Halley step (cubic convergence, one
// divide): relative error ~1e-5 on [0,1], several times cheaper than cbrt.
inline float fast_cbrt(float x)
{
    if (x <= 0) return 0;
    std::uint32_t i;
    std::memcpy(&i, &x, 4);
    i = i / 3 + 709921077u;
    float y;
    std::memcpy(&y, &i, 4);
    const float y3 = y * y * y;
    return y * (y3 + 2 * x) / (2 * y3 + x);
}

// c*log(c) for block-entropy counts (at most 64 samples per 8x8 block).
struct EntropyTable {
    float xlogx[65];
    EntropyTable()
    {
        xlogx[0] = 0;
        for (int c = 1; c <= 64; ++c) xlogx[c] = float(c * std::log(double(c)));
    }
};
const EntropyTable& entropy_table()
{
    static const EntropyTable t;
    return t;
}

constexpr int kEntropyBlock = 8;
constexpr int kBlock = 16;   // dirty-region block
constexpr int kSig = 8;      // prefilter block
constexpr float kInf = std::numeric_limits<float>::infinity();

Lab operator+(Lab a, Lab b) { return {a.l + b.l, a.a + b.a, a.b + b.b}; }
Lab operator*(Lab a, float s) { return {a.l * s, a.a * s, a.b * s}; }
float dist2(Lab a, Lab b)
{
    const float dl = a.l - b.l, da = a.a - b.a, db = a.b - b.b;
    return dl * dl + da * da + db * db;
}

template <class T, class Get, class Set>
void box3(int w, int h, Get get, Set set, std::vector<T>& tmp)
{
    // [1 2 1]/4 separable, clamped borders.
    tmp.resize(std::size_t(w) * h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const int l = std::max(0, x - 1), r = std::min(w - 1, x + 1);
            tmp[std::size_t(y) * w + x] = (get(y, l) + get(y, x) * 2.f + get(y, r)) * 0.25f;
        }
    for (int y = 0; y < h; ++y) {
        const int u = std::max(0, y - 1), d = std::min(h - 1, y + 1);
        for (int x = 0; x < w; ++x)
            set(y, x, (tmp[std::size_t(u) * w + x] + tmp[std::size_t(y) * w + x] * 2.f +
                       tmp[std::size_t(d) * w + x]) * 0.25f);
    }
}

} // namespace

namespace {
Lab linear_to_oklab(float r, float g, float b)
{
    const float l = fast_cbrt(0.4122214708f * r + 0.5363325363f * g + 0.0514459929f * b);
    const float m = fast_cbrt(0.2119034982f * r + 0.6806995451f * g + 0.1073969566f * b);
    const float s = fast_cbrt(0.0883024619f * r + 0.2817188376f * g + 0.6299787005f * b);
    return {0.2104542553f * l + 0.7936177850f * m - 0.0040720468f * s,
            1.9779984951f * l - 2.4285922050f * m + 0.4505937099f * s,
            0.0259040371f * l + 0.7827717662f * m - 0.8086757660f * s};
}
} // namespace

Lab srgb8_to_oklab(std::uint8_t r8, std::uint8_t g8, std::uint8_t b8)
{
    const auto& t = tables();
    return linear_to_oklab(t.linear[r8], t.linear[g8], t.linear[b8]);
}

void oklab_to_srgb(const Lab& c, float rgb[3])
{
    float l = c.l + 0.3963377774f * c.a + 0.2158037573f * c.b;
    float m = c.l - 0.1055613458f * c.a - 0.0638541728f * c.b;
    float s = c.l - 0.0894841775f * c.a - 1.2914855480f * c.b;
    l = l * l * l; m = m * m * m; s = s * s * s;
    const float lin[3] = {4.0767416621f * l - 3.3077115913f * m + 0.2309699292f * s,
                          -1.2684380046f * l + 2.6097574011f * m - 0.3413193965f * s,
                          -0.0041960863f * l - 0.7034186147f * m + 1.7076147010f * s};
    for (int i = 0; i < 3; ++i) {
        const float v = std::clamp(lin[i], 0.f, 1.f);
        rgb[i] = v <= 0.0031308f ? 12.92f * v : 1.055f * std::pow(v, 1.f / 2.4f) - 0.055f;
    }
}


void Mosaic::reset()
{
    tiles_.clear();
    grid_ = {};
    stats_ = {};
    reference_.clear();
    size_px_.clear();
    signature_.clear();
    now_signature_.clear();
    wake_.clear();
    prefiltered_ = false;
    prefilter_frame_ = nullptr;
    woken_ = 0;
    full_measure_ = false;
    dirty_.clear();
    runs_.clear();
    color_palette_.clear();
    color_outline_ = color_lj_ = color_cj_ = -1;
    max_cell_ = 0;
    f_ = {};
}

namespace {
template <class T> void fit(std::vector<T>& v, std::size_t n, T value = T())
{
    if (v.size() != n) v.assign(n, value);
}
} // namespace

// ---------------------------------------------------------------- prefilter

void Mosaic::prefilter(const std::uint8_t* rgba, std::size_t stride)
{
    const int w = f_.width, h = f_.height;
    const int sbw = (w + kSig - 1) / kSig, sbh = (h + kSig - 1) / kSig;
    const std::size_t blocks = std::size_t(sbw) * sbh;
    now_signature_.assign(blocks * 3, 0);
    // One pass: luma and two chroma differences summed per 8x8 block.
    for (int y = 0; y < h; ++y) {
        const std::uint8_t* p = rgba + std::size_t(y) * stride;
        std::int32_t* acc = &now_signature_[std::size_t(y / kSig) * sbw * 3];
        for (int bx = 0; bx < sbw; ++bx) {
            const int x1 = std::min(w, (bx + 1) * kSig);
            std::int32_t l = 0, rg = 0, bg = 0;
            for (int x = bx * kSig; x < x1; ++x, p += 4) {
                l += 54 * p[0] + 183 * p[1] + 19 * p[2];
                rg += (int(p[0]) - p[1]) << 8;
                bg += (int(p[2]) - p[1]) << 8;
            }
            acc[3 * bx] += l;
            acc[3 * bx + 1] += rg;
            acc[3 * bx + 2] += bg;
        }
    }
    // Mean in 1/16 code units.
    for (int by = 0; by < sbh; ++by) {
        const int ch = std::min(kSig, h - by * kSig);
        for (int bx = 0; bx < sbw; ++bx) {
            const int count = ch * std::min(kSig, w - bx * kSig) * 16;
            std::int32_t* v = &now_signature_[(std::size_t(by) * sbw + bx) * 3];
            for (int k = 0; k < 3; ++k) v[k] /= count;
        }
    }
    wake_.assign(blocks, 0);
    full_measure_ = false;
    if (signature_.size() != now_signature_.size()) {
        std::fill(wake_.begin(), wake_.end(), 1);
        woken_ = std::uint32_t(blocks);
        full_measure_ = true;
    } else {
        // Remove the global (median) shift per channel: exposure and white
        // balance changes recolor tiles instead of waking geometry.
        const int limit = int(config_.prefilter_luma * 16);
        int median[3];
        delta_.resize(blocks);
        for (int k = 0; k < 3; ++k) {
            for (std::size_t i = 0; i < blocks; ++i) delta_[i] = now_signature_[3 * i + k] - signature_[3 * i + k];
            std::nth_element(delta_.begin(), delta_.begin() + blocks / 2, delta_.end());
            median[k] = delta_[blocks / 2];
            if (std::abs(median[k]) > limit / 2) full_measure_ = true;
        }
        woken_ = 0;
        for (std::size_t i = 0; i < blocks; ++i) {
            bool on = false;
            for (int k = 0; k < 3; ++k)
                on |= std::abs(now_signature_[3 * i + k] - signature_[3 * i + k] - median[k]) > limit;
            wake_[i] = on;
            woken_ += on;
        }
    }
    prefiltered_ = true;
}

bool Mosaic::wants_structure(const std::uint8_t* rgba, int width, int height, std::size_t stride)
{
    if (tiles_.empty() || width != f_.width || height != f_.height) {
        prefiltered_ = false;
        return true;
    }
    prefilter(rgba, stride);
    prefilter_frame_ = rgba;
    return woken_ > 0;
}

// ---------------------------------------------------------------- regions

void Mosaic::build_runs(bool full)
{
    const int w = f_.width, h = f_.height;
    const int dbw = (w + kBlock - 1) / kBlock, dbh = (h + kBlock - 1) / kBlock;
    runs_.clear();
    for (int by = 0; by < dbh; ++by) {
        const int y0 = by * kBlock, y1 = std::min(h, y0 + kBlock);
        if (full) {
            runs_.push_back({0, y0, w, y1});
            continue;
        }
        for (int bx = 0; bx < dbw;) {
            if (!dirty_[std::size_t(by) * dbw + bx]) { ++bx; continue; }
            int e = bx;
            while (e < dbw && dirty_[std::size_t(by) * dbw + e]) ++e;
            runs_.push_back({bx * kBlock, y0, std::min(w, e * kBlock), y1});
            bx = e;
        }
    }
}

namespace {
struct Rect { int x0, y0, x1, y1; };
inline Rect grow(const Run& r, int g, int w, int h)
{
    return {std::max(0, r.x0 - g), std::max(0, r.y0 - g), std::min(w, r.x1 + g), std::min(h, r.y1 + g)};
}
} // namespace

// ---------------------------------------------------------------- geometry

namespace {
constexpr int kAngleCodes = 126; // edge normal doubled-angle codes 1..126; bit 7 marks visited
struct AngleTable {
    float c[kAngleCodes + 1], s[kAngleCodes + 1];
    AngleTable()
    {
        c[0] = 1; s[0] = 0;
        for (int k = 0; k < kAngleCodes; ++k) {
            const double a = -3.14159265358979 + (k + 0.5) * (2 * 3.14159265358979 / kAngleCodes);
            c[k + 1] = float(std::cos(a));
            s[k + 1] = float(std::sin(a));
        }
    }
};
const AngleTable& angle_table()
{
    static const AngleTable t;
    return t;
}
inline std::uint8_t angle_code(float c2, float s2)
{
    int k = int((std::atan2(s2, c2) + 3.14159265f) * (kAngleCodes / 6.2831853f));
    return std::uint8_t(1 + std::clamp(k, 0, kAngleCodes - 1));
}
} // namespace

void Mosaic::analyze_geometry(const std::uint8_t* rgba, std::size_t stride)
{
    const Config& c = config_;
    const int w = f_.width, h = f_.height;
    const int dbw = (w + kBlock - 1) / kBlock;
    auto tick = Clock::now();
    auto lap = [&](int k) {
        const auto now = Clock::now();
        stats_.stage_ms[k] = std::chrono::duration<double, std::milli>(now - tick).count();
        tick = now;
    };
    const bool full = runs_.size() == std::size_t((h + kBlock - 1) / kBlock) && !runs_.empty() &&
                      runs_[0].x0 == 0 && runs_[0].x1 == w;
    auto in_dirty = [&](int x, int y) { return full || dirty_[std::size_t(y / kBlock) * dbw + x / kBlock] != 0; };

    // Local color entropy on 8x8 blocks (5-bit RGB ids, variance gated): the
    // entropy stretch's concentration measure.  Raw values persist per block;
    // only blocks inside the runs are recomputed, then the small grid is
    // smoothed and the size class written on the runs.
    const int bw = (w + kEntropyBlock - 1) / kEntropyBlock;
    const int bh = (h + kEntropyBlock - 1) / kEntropyBlock;
    std::array<int, kEntropyBlock * kEntropyBlock> ids{};
    histogram_.resize(32768, 0);
    const auto& xl = entropy_table().xlogx;
    for (const Run& run : runs_)
        for (int by = run.y0 / kEntropyBlock; by * kEntropyBlock < run.y1; ++by)
            for (int bx = run.x0 / kEntropyBlock; bx * kEntropyBlock < run.x1; ++bx) {
                int count = 0;
                int sum[3] = {0, 0, 0}, sq[3] = {0, 0, 0};
                for (int y = by * kEntropyBlock; y < std::min(h, (by + 1) * kEntropyBlock); ++y) {
                    const std::uint8_t* p = rgba + std::size_t(y) * stride + 4 * bx * kEntropyBlock;
                    for (int x = bx * kEntropyBlock; x < std::min(w, (bx + 1) * kEntropyBlock); ++x, p += 4) {
                        ids[count++] = ((p[0] >> 3) << 10) | ((p[1] >> 3) << 5) | (p[2] >> 3);
                        for (int k = 0; k < 3; ++k) { sum[k] += p[k]; sq[k] += p[k] * p[k]; }
                    }
                }
                float value = 0;
                if (count >= 4) {
                    // H / log n = 1 - sum c log c / (n log n): histogram, no sort.
                    float sum_xlogx = 0;
                    for (int i = 0; i < count; ++i) ++histogram_[ids[i]];
                    for (int i = 0; i < count; ++i) {
                        std::uint8_t& cnt = histogram_[ids[i]];
                        if (cnt) { sum_xlogx += xl[cnt]; cnt = 0; }
                    }
                    const double e = 1.0 - sum_xlogx / xl[count];
                    double variance = 0;
                    for (int k = 0; k < 3; ++k) {
                        const double m = double(sum[k]) / count;
                        variance += std::max(0.0, double(sq[k]) / count - m * m) / 3;
                    }
                    variance /= 255.0 * 255.0;
                    const double floor = 0.015;
                    value = float(e * variance / (variance + floor * floor));
                }
                block_raw_[std::size_t(by) * bw + bx] = value;
            }
    block_tmp_ = block_raw_;
    {
        std::vector<float> tmp;
        for (int pass = 0; pass < 2; ++pass)
            box3<float>(bw, bh, [&](int y, int x) { return block_tmp_[std::size_t(y) * bw + x]; },
                        [&](int y, int x, float v) { block_tmp_[std::size_t(y) * bw + x] = v; }, tmp);
    }
    const int classes = std::clamp(c.size_classes, 1, 4);
    float thresholds[3];
    for (int k = 0; k < classes - 1; ++k)
        thresholds[k] = classes == 2 ? 0.5f * (c.entropy_low + c.entropy_high)
                                     : c.entropy_low + (c.entropy_high - c.entropy_low) * k / (classes - 2);
    for (const Run& run : runs_)
        for (int y = run.y0; y < run.y1; ++y) {
            const float fy = std::clamp((y + 0.5f) / kEntropyBlock - 0.5f, 0.f, float(bh - 1));
            const int y0 = int(fy), y1 = std::min(bh - 1, y0 + 1);
            const float ty = fy - y0;
            const float* r0 = &block_tmp_[std::size_t(y0) * bw];
            const float* r1 = &block_tmp_[std::size_t(y1) * bw];
            std::uint8_t* lv = &f_.level[std::size_t(y) * w];
            for (int x = run.x0; x < run.x1; ++x) {
                const float fx = std::clamp((x + 0.5f) / kEntropyBlock - 0.5f, 0.f, float(bw - 1));
                const int x0 = int(fx), x1 = std::min(bw - 1, x0 + 1);
                const float tx = fx - x0;
                const float e = (r0[x0] * (1 - tx) + r0[x1] * tx) * (1 - ty) + (r1[x0] * (1 - tx) + r1[x1] * tx) * ty;
                int level = 0;
                for (int k = 0; k < classes - 1; ++k) level += e > thresholds[k];
                lv[x] = std::uint8_t(level);
            }
        }
    lap(0);

    // Edges, run by run in run-local scratch: Oklab a,b on run + 3 px, a
    // separable [1 2 1]^2 blur on run + 2, gradients on run + 1 (frame border
    // excluded), NMS on the run.  Only the edge code (doubled-angle normal)
    // and outline weight persist.
    const float wc = c.chroma_edge_weight;
    const float threshold2 = c.edge_threshold * c.edge_threshold;
    const float inv_threshold = 1.f / std::max(c.edge_threshold, 1e-6f);
    const float inv_span = 1.f / std::max(c.outline_contrast - 1, 1e-3f);
    const auto& lin = tables().linear;
    for (const Run& run : runs_) {
        const Rect r3 = grow(run, 3, w, h), r2 = grow(run, 2, w, h), r1 = grow(run, 1, w, h);
        const int w3 = r3.x1 - r3.x0, w2 = r2.x1 - r2.x0, w1 = r1.x1 - r1.x0;
        ab_raw_.resize(std::size_t(w3) * (r3.y1 - r3.y0) * 2);
        for (int y = r3.y0; y < r3.y1; ++y) {
            const std::uint8_t* p = rgba + std::size_t(y) * stride + 4 * r3.x0;
            float* o = &ab_raw_[std::size_t(y - r3.y0) * w3 * 2];
            for (int x = 0; x < w3; ++x, p += 4) {
                const float rl = lin[p[0]], gl = lin[p[1]], bl = lin[p[2]];
                const float l = fast_cbrt(0.4122214708f * rl + 0.5363325363f * gl + 0.0514459929f * bl);
                const float m = fast_cbrt(0.2119034982f * rl + 0.6806995451f * gl + 0.1073969566f * bl);
                const float s3 = fast_cbrt(0.0883024619f * rl + 0.2817188376f * gl + 0.6299787005f * bl);
                o[2 * x] = 1.9779984951f * l - 2.4285922050f * m + 0.4505937099f * s3;
                o[2 * x + 1] = 0.0259040371f * l + 0.7827717662f * m - 0.8086757660f * s3;
            }
        }
        auto raw = [&](int x, int y) { return &ab_raw_[(std::size_t(y - r3.y0) * w3 + (x - r3.x0)) * 2]; };
        // Horizontal on rows r2 +- 1 (clamped), vertical to r2.
        const int hy0 = std::max(0, r2.y0 - 1), hy1 = std::min(h, r2.y1 + 1);
        ab_h_.resize(std::size_t(w2) * (hy1 - hy0) * 2);
        for (int y = hy0; y < hy1; ++y) {
            float* o = &ab_h_[std::size_t(y - hy0) * w2 * 2];
            for (int x = r2.x0; x < r2.x1; ++x) {
                const float* l = raw(std::max(0, x - 1), y);
                const float* m = raw(x, y);
                const float* rr = raw(std::min(w - 1, x + 1), y);
                o[2 * (x - r2.x0)] = l[0] + 2 * m[0] + rr[0];
                o[2 * (x - r2.x0) + 1] = l[1] + 2 * m[1] + rr[1];
            }
        }
        ab_blur_.resize(std::size_t(w2) * (r2.y1 - r2.y0) * 2);
        for (int y = r2.y0; y < r2.y1; ++y) {
            const float* up = &ab_h_[std::size_t(std::max(hy0, y - 1) - hy0) * w2 * 2];
            const float* mid = &ab_h_[std::size_t(y - hy0) * w2 * 2];
            const float* dn = &ab_h_[std::size_t(std::min(hy1 - 1, y + 1) - hy0) * w2 * 2];
            float* o = &ab_blur_[std::size_t(y - r2.y0) * w2 * 2];
            for (int i = 0; i < w2 * 2; ++i) o[i] = (up[i] + 2 * mid[i] + dn[i]) * (1.f / 16.f);
        }
        auto blur = [&](int x, int y) { return &ab_blur_[(std::size_t(y - r2.y0) * w2 + (x - r2.x0)) * 2]; };
        grad_.assign(std::size_t(w1) * (r1.y1 - r1.y0) * 3, 0.f); // mag^2, gx, gy
        for (int y = std::max(1, r1.y0); y < std::min(h - 1, r1.y1); ++y)
            for (int x = std::max(1, r1.x0); x < std::min(w - 1, r1.x1); ++x) {
                const std::size_t i = std::size_t(y) * w + x;
                const float sx = 0.5f * (f_.structure[i + 1] - f_.structure[i - 1]);
                const float sy = 0.5f * (f_.structure[i + w] - f_.structure[i - w]);
                const float* xl0 = blur(x - 1, y);
                const float* xr0 = blur(x + 1, y);
                const float* yu0 = blur(x, y - 1);
                const float* yd0 = blur(x, y + 1);
                const float ax = 0.5f * (xr0[0] - xl0[0]) * wc, ay = 0.5f * (yd0[0] - yu0[0]) * wc;
                const float cx = 0.5f * (xr0[1] - xl0[1]) * wc, cy = 0.5f * (yd0[1] - yu0[1]) * wc;
                const float es = sx * sx + sy * sy, ea = ax * ax + ay * ay, eb = cx * cx + cy * cy;
                float* g = &grad_[(std::size_t(y - r1.y0) * w1 + (x - r1.x0)) * 3];
                g[0] = es + ea + eb; // squared: NMS is order-preserving
                if (es >= ea && es >= eb) { g[1] = sx; g[2] = sy; }
                else if (ea >= eb) { g[1] = ax; g[2] = ay; }
                else { g[1] = cx; g[2] = cy; }
            }
        const std::ptrdiff_t gw = std::ptrdiff_t(w1) * 3;
        for (int y = run.y0; y < run.y1; ++y)
            for (int x = run.x0; x < run.x1; ++x) {
                const std::size_t i = std::size_t(y) * w + x;
                std::uint8_t code = 0;
                if (y > 0 && y + 1 < h && x > 0 && x + 1 < w) {
                    const float* g = &grad_[(std::size_t(y - r1.y0) * w1 + (x - r1.x0)) * 3];
                    if (g[0] > threshold2) {
                        const float ux = std::fabs(g[1]), uy = std::fabs(g[2]);
                        std::ptrdiff_t step;
                        if (ux > 2.414f * uy) step = 3;
                        else if (uy > 2.414f * ux) step = gw;
                        else step = (g[1] * g[2] > 0) ? gw + 3 : gw - 3;
                        if (g[0] > g[-step] && g[0] >= g[step]) {
                            code = angle_code(g[1] * g[1] - g[2] * g[2], 2 * g[1] * g[2]);
                            f_.outline[i] = std::uint8_t(
                                255.f * std::clamp((std::sqrt(g[0]) * inv_threshold - 1) * inv_span, 0.f, 1.f));
                        }
                    }
                }
                f_.edge[i] = code;
            }
    }
    lap(1);
    lap(2);

    // Short-fragment removal by flood fill over the runs (bit 7 marks
    // visited).  A component continuing into an unexamined region is kept:
    // its length outside is unknown.
    if (c.edge_min_length > 1) {
        for (const Run& run : runs_)
            for (int y = run.y0; y < run.y1; ++y)
                for (int x = run.x0; x < run.x1; ++x) {
                    const int start = y * w + x;
                    if (!f_.edge[start] || (f_.edge[start] & 0x80)) continue;
                    component_.clear();
                    stack_.assign(1, start);
                    f_.edge[start] |= 0x80;
                    bool anchored = false;
                    while (!stack_.empty()) {
                        const int i = stack_.back();
                        stack_.pop_back();
                        component_.push_back(i);
                        const int px = i % w, py = i / w;
                        for (int dy = -1; dy <= 1; ++dy)
                            for (int dx = -1; dx <= 1; ++dx) {
                                const int xx = px + dx, yy = py + dy;
                                if ((!dx && !dy) || xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
                                const int j = yy * w + xx;
                                if (!f_.edge[j]) continue;
                                if (!in_dirty(xx, yy)) { anchored = true; continue; }
                                if (f_.edge[j] & 0x80) continue;
                                f_.edge[j] |= 0x80;
                                stack_.push_back(j);
                            }
                    }
                    if (!anchored && int(component_.size()) < c.edge_min_length)
                        for (int i : component_) f_.edge[i] = 0x80; // removed, still marked visited
                }
        for (const Run& run : runs_)
            for (int y = run.y0; y < run.y1; ++y) {
                std::uint8_t* e = &f_.edge[std::size_t(y) * w];
                for (int x = run.x0; x < run.x1; ++x) e[x] &= 0x7f;
            }
    }
    stats_.edge_pixels = 0;
    for (const Run& run : runs_)
        for (int y = run.y0; y < run.y1; ++y)
            for (int x = run.x0; x < run.x1; ++x) stats_.edge_pixels += f_.edge[std::size_t(y) * w + x] != 0;
    lap(3);

    // Exact distance/feature transform (Felzenszwalb-Huttenlocher).  Column
    // sites come from two row-major sweeps and are kept only for the rows of
    // interest (runs +- 2); the envelope runs only on those rows.
    std::fill(rows_.begin(), rows_.end(), 0);
    for (const Run& run : runs_)
        for (int y = std::max(0, run.y0 - 2); y < std::min(h, run.y1 + 2); ++y) rows_[y] = 1;
    last_.assign(std::size_t(w), -1);
    for (int y = 0; y < h; ++y) {
        const std::uint8_t* e = &f_.edge[std::size_t(y) * w];
        for (int x = 0; x < w; ++x)
            if (e[x]) last_[x] = std::int16_t(y);
        if (rows_[y]) std::copy_n(last_.data(), w, &col_site_[std::size_t(y) * w]);
    }
    last_.assign(std::size_t(w), -1);
    for (int y = h - 1; y >= 0; --y) {
        const std::uint8_t* e = &f_.edge[std::size_t(y) * w];
        for (int x = 0; x < w; ++x)
            if (e[x]) last_[x] = std::int16_t(y);
        if (!rows_[y]) continue;
        std::int16_t* cs = &col_site_[std::size_t(y) * w];
        for (int x = 0; x < w; ++x)
            if (last_[x] >= 0 && (cs[x] < 0 || last_[x] - y < y - cs[x])) cs[x] = last_[x];
    }
    lap(4);

    // Per row of interest: envelope -> raw doubled-angle normal (from squared
    // offsets, no sqrt/divide except at the frame test) and distance, then the
    // two separable [1 2 1]^2 blurs through 5-row and 3-row rings.  Output
    // rows need +-2 rows of context, so contiguous row segments are streamed.
    const auto& at = angle_table();
    ring_h_.resize(std::size_t(w) * 2 * 5);
    ring_p_.resize(std::size_t(w) * 2 * 3);
    row_c_.resize(std::size_t(w));
    row_s_.resize(std::size_t(w));
    auto hslot = [&](int k) { return &ring_h_[std::size_t(k % 5) * w * 2]; };
    auto pslot = [&](int k) { return &ring_p_[std::size_t(k % 3) * w * 2]; };
    int xa = 0, xb = w; // x-span of the current row segment (runs +- 2)
    auto raw_row = [&](int y) {
        const std::int16_t* col = &col_site_[std::size_t(y) * w];
        int k = -1;
        auto fq = [&](int q) { const double d = y - col[q]; return d * d + double(q) * q; };
        for (int q = 0; q < w; ++q) {
            if (col[q] < 0) continue;
            const double vq = fq(q);
            double sv = 0;
            while (k >= 0) {
                const int p = v_[k];
                sv = (vq - fq(p)) / (2.0 * (q - p));
                if (sv <= z_[k]) --k; else break;
            }
            ++k;
            v_[k] = q;
            z_[k] = k == 0 ? -1e300 : sv;
            z_[k + 1] = 1e300;
        }
        float* dist = &f_.distance[std::size_t(y) * w];
        std::uint8_t* frame_row = &f_.from_frame[std::size_t(y) * w];
        const float fy = std::min(y + 0.5f, h - y - 0.5f);
        int j = 0;
        for (int x = xa; x < xb; ++x) {
            float ex = 0, ey = 0, ed2 = kInf;
            int site = -1;
            if (k >= 0) {
                while (z_[j + 1] < x) ++j;
                const int q = v_[j];
                ex = float(x - q);
                ey = float(y - col[q]);
                ed2 = ex * ex + ey * ey;
                site = col[q] * w + q;
            }
            float c2v, s2v;
            std::uint8_t frame = 0;
            float distance = ed2 < kInf ? std::sqrt(ed2) : kInf;
            if (c.frame_edges) {
                const float fx = std::min(x + 0.5f, w - x - 0.5f), fd = std::min(fx, fy);
                if (fd * fd < ed2) {
                    frame = 1;
                    distance = fd;
                    // Axis-aligned frame normals double to (+1,0) for left/
                    // right and (-1,0) for top/bottom (left/right wins ties).
                    c2v = fy < fx ? -1.f : 1.f;
                    s2v = 0;
                }
            }
            if (!frame) {
                if (site < 0) { c2v = -1; s2v = 0; }
                else if (ed2 == 0) {
                    const std::uint8_t code = f_.edge[std::size_t(site)];
                    c2v = at.c[code]; s2v = at.s[code];
                } else {
                    const float inv = 1.f / ed2;
                    c2v = (ex * ex - ey * ey) * inv;
                    s2v = 2 * ex * ey * inv;
                }
            }
            dist[x] = distance;
            frame_row[x] = frame;
            row_c_[x] = c2v;
            row_s_[x] = s2v;
        }
        // Horizontal blur; the span ends clamp like the frame (they lie at
        // least 2 px outside every run, so run pixels see exact context).
        float* o = hslot(y);
        for (int x = xa; x < xb; ++x) {
            const int l = std::max(xa, x - 1), r = std::min(xb - 1, x + 1);
            o[2 * x] = row_c_[l] + 2 * row_c_[x] + row_c_[r];
            o[2 * x + 1] = row_s_[l] + 2 * row_s_[x] + row_s_[r];
        }
    };
    auto pass1 = [&](int y, int ys, int ye) { // vertical of H rows, then horizontal, into P(y)
        const float* up = hslot(std::max(ys, y - 1));
        const float* mid = hslot(y);
        const float* dn = hslot(std::min(ye - 1, y + 1));
        float* o = pslot(y);
        // Vertical into o, then horizontal in place through row_c_/row_s_.
        for (int x = xa; x < xb; ++x) {
            row_c_[x] = (up[2 * x] + 2 * mid[2 * x] + dn[2 * x]) * (1.f / 16.f);
            row_s_[x] = (up[2 * x + 1] + 2 * mid[2 * x + 1] + dn[2 * x + 1]) * (1.f / 16.f);
        }
        for (int x = xa; x < xb; ++x) {
            const int l = std::max(xa, x - 1), r = std::min(xb - 1, x + 1);
            o[2 * x] = row_c_[l] + 2 * row_c_[x] + row_c_[r];
            o[2 * x + 1] = row_s_[l] + 2 * row_s_[x] + row_s_[r];
        }
    };
    auto output = [&](int y, int ys, int ye) {
        const float* up = pslot(std::max(ys, y - 1));
        const float* mid = pslot(y);
        const float* dn = pslot(std::min(ye - 1, y + 1));
        std::int16_t* oc = &f_.c2[std::size_t(y) * w];
        std::int16_t* os = &f_.s2[std::size_t(y) * w];
        // Output reads raw x +- 3: only the span interior is exact.
        for (int x = xa == 0 ? 0 : xa + 3; x < (xb == w ? w : xb - 3); ++x) {
            oc[x] = std::int16_t(std::lround(std::clamp((up[2 * x] + 2 * mid[2 * x] + dn[2 * x]) * (1.f / 16.f), -1.f, 1.f) * 32767.f));
            os[x] = std::int16_t(std::lround(std::clamp((up[2 * x + 1] + 2 * mid[2 * x + 1] + dn[2 * x + 1]) * (1.f / 16.f), -1.f, 1.f) * 32767.f));
        }
    };
    for (int ys = 0; ys < h;) {
        if (!rows_[ys]) { ++ys; continue; }
        int ye = ys;
        while (ye < h && rows_[ye]) ++ye;
        xa = w; xb = 0;
        for (const Run& run : runs_)
            if (run.y1 + 2 > ys && run.y0 - 2 < ye) { xa = std::min(xa, run.x0); xb = std::max(xb, run.x1); }
        xa = std::max(0, xa - 4);
        xb = std::min(w, xb + 4);
        // Context is exact for rows >= 2 from a segment end that is not the
        // frame edge; only those rows (which include every run row) are written.
        const int lo = ys == 0 ? 0 : ys + 2, hi = ye == h ? h : ye - 2;
        for (int k = ys; k < ye; ++k) {
            raw_row(k);
            if (k - 1 >= ys) pass1(k - 1, ys, ye);
            if (k - 2 >= ys && k - 2 >= lo && k - 2 < hi) output(k - 2, ys, ye);
        }
        pass1(ye - 1, ys, ye);
        for (int y = std::max(lo, ye - 2); y < std::min(hi, ye); ++y) output(y, ys, ye);
        ys = ye;
    }
    lap(5);
}

Tile Mosaic::make_tile(float x, float y, int level, int row, float outline, bool regular) const
{
    const Config& c = config_;
    const int w = f_.width;
    const std::size_t i = std::size_t(std::lround(y)) * w + std::size_t(std::lround(x));
    Tile t;
    t.x = x; t.y = y;
    t.size_class = std::uint8_t(level);
    t.row = std::uint16_t(row);
    t.outline = std::uint8_t(std::lround(255 * std::clamp(outline, 0.f, 1.f)));
    t.key = hash32(std::uint32_t(std::lround(x * 4)) * 73856093u ^
                   std::uint32_t(std::lround(y * 4)) * 19349663u ^ std::uint32_t(level) * 83492791u);
    const float u0 = unit(hash32(t.key ^ 0x9e3779b9u)), u1 = unit(hash32(t.key ^ 0x85ebca6bu));
    const float u2 = unit(hash32(t.key ^ 0xc2b2ae35u)), u3 = unit(hash32(t.key ^ 0x27d4eb2fu));
    const float u4 = unit(hash32(t.key ^ 0x165667b1u)), u5 = unit(hash32(t.key ^ 0xd3a2646cu));
    // Row tangent is perpendicular to the normal away from the edge; nx/ny
    // hold the smoothed doubled-angle normal, so halve its angle here.
    float angle = 0.f;
    if (!regular) {
        const float c2 = f_.c2[i], s2 = f_.s2[i];
        angle = (c2 != 0 || s2 != 0 ? 0.5f * std::atan2(s2, c2) : 0.f) + 1.5707963f;
    }
    angle += c.angle_jitter * (2 * u0 - 1);
    t.c = std::cos(angle);
    t.s = std::sin(angle);
    const float size = size_px_[level];
    t.half_n = 0.53f * size * (1 + c.shape_jitter * (2 * u1 - 1));
    t.half_t = 0.53f * size * c.elongation * (1 + c.shape_jitter * (2 * u2 - 1));
    const float r = std::sqrt(u3), a = 6.2831853f * u4;
    t.tilt_x = r * std::cos(a);
    t.tilt_y = r * std::sin(a);
    t.phase = u5;
    t.measured.l = -1; // measure on first color pass
    return t;
}

void Mosaic::nucleate(bool full)
{
    const Config& c = config_;
    const int w = f_.width, h = f_.height;
    const std::size_t n = std::size_t(w) * h;
    const int classes = std::clamp(c.size_classes, 1, 4);
    const float spacing = std::clamp(c.spacing, 0.5f, 1.5f);
    const int dbw = (w + kBlock - 1) / kBlock;

    if (full) {
        // One scale for all classes so the expected count meets the budget.
        double rel[4], inv[4];
        for (int k = 0; k < classes; ++k) { rel[k] = std::pow(double(c.size_ratio), -k); inv[k] = 1.0 / (rel[k] * rel[k]); }
        std::size_t population[4] = {0, 0, 0, 0};
        for (std::size_t i = 0; i < n; ++i) ++population[f_.level[i]];
        double inverse_area = 0;
        for (int k = 0; k < classes; ++k) inverse_area += population[k] * inv[k];
        // 0.93: measured greedy packing is slightly looser than one tile per size^2.
        double scale = 0.93 * std::sqrt(inverse_area / std::max<std::uint32_t>(c.target_tiles, 1));
        scale = std::max(scale, 3.0 / rel[classes - 1]);
        size_px_.resize(classes);
        for (int k = 0; k < classes; ++k) size_px_[k] = float(scale * rel[k]);
        stats_.removed = std::uint32_t(tiles_.size());
        tiles_.clear();
    } else {
        std::size_t kept = 0;
        for (const Tile& t : tiles_) {
            const int x = std::clamp(int(t.x), 0, w - 1), y = std::clamp(int(t.y), 0, h - 1);
            if (!dirty_[std::size_t(y / kBlock) * dbw + x / kBlock]) tiles_[kept++] = t;
        }
        stats_.removed = std::uint32_t(tiles_.size() - kept);
        tiles_.resize(kept);
    }

    const float cell = size_px_[0];
    const int cols = int(w / cell) + 1, rows = int(h / cell) + 1;
    head_.assign(std::size_t(cols) * rows, -1);
    next_.clear();
    next_.reserve(tiles_.size() + 8192);
    auto insert = [&](int index) {
        const Tile& t = tiles_[index];
        const int cx = std::clamp(int(t.x / cell), 0, cols - 1), cy = std::clamp(int(t.y / cell), 0, rows - 1);
        next_.push_back(head_[std::size_t(cy) * cols + cx]);
        head_[std::size_t(cy) * cols + cx] = index;
    };
    for (int k = 0; k < int(tiles_.size()); ++k) insert(k);
    auto free_at = [&](float x, float y, float size) {
        const int cx = int(x / cell), cy = int(y / cell);
        for (int yy = std::max(0, cy - 1); yy <= std::min(rows - 1, cy + 1); ++yy)
            for (int xx = std::max(0, cx - 1); xx <= std::min(cols - 1, cx + 1); ++xx)
                for (int k = head_[std::size_t(yy) * cols + xx]; k >= 0; k = next_[k]) {
                    const Tile& t = tiles_[k];
                    const float lim = spacing * 0.5f * (size + size_px_[t.size_class]);
                    const float dx = x - t.x, dy = y - t.y;
                    if (dx * dx + dy * dy < lim * lim) return false;
                }
        return true;
    };
    const std::size_t before = tiles_.size();

    // Bounding box of the runs; the exclusion mask is cleared and painted only there.
    int bx0 = w, by0 = h, bx1 = 0, by1 = 0;
    for (const Run& r : runs_) {
        bx0 = std::min(bx0, r.x0); by0 = std::min(by0, r.y0);
        bx1 = std::max(bx1, r.x1); by1 = std::max(by1, r.y1);
    }
    if (bx0 >= bx1) return;
    const int bbw = bx1 - bx0;
    blocked_.assign(std::size_t(bbw) * (by1 - by0), 0); // bbox-local exclusion mask
    auto blocked_at = [&](int x, int y) -> std::uint8_t& { return blocked_[std::size_t(y - by0) * bbw + (x - bx0)]; };
    // Pixels certainly excluded by an accepted seed (radius uses the smallest
    // possible candidate size), so the gap fill tests only the remainder.
    const float smallest = size_px_.back();
    auto block_disk = [&](const Tile& t) {
        const float r = spacing * 0.5f * (smallest + size_px_[t.size_class]) - 0.75f;
        if (r <= 0) return;
        const int y0 = std::max(by0, int(std::ceil(t.y - r))), y1 = std::min(by1 - 1, int(std::floor(t.y + r)));
        for (int y = y0; y <= y1; ++y) {
            const float dy = y - t.y, half = std::sqrt(std::max(0.f, r * r - dy * dy));
            const int x0 = std::max(bx0, int(std::ceil(t.x - half))), x1 = std::min(bx1 - 1, int(std::floor(t.x + half)));
            if (x0 <= x1) std::memset(&blocked_at(x0, y), 1, std::size_t(x1 - x0 + 1));
        }
    };
    // Raster order within each block row across its runs (a full rebuild is
    // plain raster order).
    auto for_each_pixel = [&](auto&& fn) {
        for (std::size_t a = 0; a < runs_.size();) {
            std::size_t b = a;
            while (b < runs_.size() && runs_[b].y0 == runs_[a].y0) ++b;
            for (int y = runs_[a].y0; y < runs_[a].y1; ++y)
                for (std::size_t k = a; k < b; ++k)
                    for (int x = runs_[k].x0; x < runs_[k].x1; ++x) fn(x, y);
            a = b;
        }
    };

    // Contour rows: centrelines at distance (k + offset) * size.
    const float offset = c.outline > 0 ? 0.f : 0.5f;
    float inv_size[4] = {0, 0, 0, 0};
    for (int k = 0; k < classes; ++k) inv_size[k] = 1.f / size_px_[k];
    candidates_.clear();
    int max_row = 0;
    for_each_pixel([&](int x, int y) {
        const std::size_t i = std::size_t(y) * w + x;
        const int level = f_.level[i];
        const float d = f_.distance[i];
        const float rowf = d * inv_size[level] - offset;
        if (rowf < -0.5f || !(rowf < 65000.f)) return;
        const int k = std::max(0, int(rowf + 0.5f));
        if (c.contour_rows > 0 && k >= c.contour_rows) return;
        if (std::fabs(d - (k + offset) * size_px_[level]) < 0.55f) {
            candidates_.emplace_back(k, int(i));
            max_row = std::max(max_row, k);
        }
    });
    // Stable counting sort by row index (raster order within a row).
    {
        row_count_.assign(std::size_t(max_row) + 2, 0);
        for (const auto& cand : candidates_) ++row_count_[std::size_t(cand.first) + 1];
        for (std::size_t k = 1; k < row_count_.size(); ++k) row_count_[k] += row_count_[k - 1];
        sorted_.resize(candidates_.size());
        for (const auto& cand : candidates_) sorted_[std::size_t(row_count_[std::size_t(cand.first)]++)] = cand;
        candidates_.swap(sorted_);
    }
    for (const auto& [k, i] : candidates_) {
        const float x = float(i % w), y = float(i / w);
        const int level = f_.level[i];
        if (!free_at(x, y, size_px_[level])) continue;
        // Outline darkness follows edge contrast: weak edges stay faint.
        const float outline = k == 0 && offset == 0 && !f_.from_frame[i] ? f_.outline[i] * (1.f / 255.f) : 0.f;
        tiles_.push_back(make_tile(x, y, level, std::min(k, 0xfffe), outline, false));
        insert(int(tiles_.size()) - 1);
    }
    const float reach = spacing * size_px_[0];
    for (const Tile& t : tiles_)
        if (t.x > bx0 - reach && t.x < bx1 + reach && t.y > by0 - reach && t.y < by1 + reach) block_disk(t);
    // Gap fill: raster order, oriented by the field inside the contour
    // region and on a regular grid beyond it.
    for_each_pixel([&](int x, int y) {
        if (blocked_at(x, y)) return;
        const std::size_t i = std::size_t(y) * w + x;
        const int level = f_.level[i];
        const float size = size_px_[level];
        if (!free_at(float(x), float(y), size)) return;
        const bool regular = c.contour_rows > 0 && f_.distance[i] / size - offset >= c.contour_rows - 0.5f;
        tiles_.push_back(make_tile(float(x), float(y), level, 0xffff, 0.f, regular));
        insert(int(tiles_.size()) - 1);
        block_disk(tiles_.back());
    });
    stats_.added = std::uint32_t(tiles_.size() - before);
}

// Tiles are re-measured only when fresh, when one of the 8x8 prefilter
// blocks they cover woke, or after a global shift; display colors are
// re-quantized only for re-measured tiles or when the palette/color settings
// changed.
void Mosaic::measure_and_color(const std::uint8_t* rgba, std::size_t stride, const std::vector<Lab>& palette)
{
    const Config& c = config_;
    const int w = f_.width, h = f_.height;
    const int sbw = (w + kSig - 1) / kSig;
    const auto& lin = tables().linear;
    bool same_palette = palette.size() == color_palette_.size() && c.outline == color_outline_ &&
                        c.lightness_jitter == color_lj_ && c.chroma_jitter == color_cj_;
    for (std::size_t i = 0; same_palette && i < palette.size(); ++i)
        same_palette = palette[i].l == color_palette_[i].l && palette[i].a == color_palette_[i].a &&
                       palette[i].b == color_palette_[i].b;
    if (!same_palette) {
        color_palette_ = palette;
        color_outline_ = c.outline;
        color_lj_ = c.lightness_jitter;
        color_cj_ = c.chroma_jitter;
    }
    const bool gate = prefiltered_ && !full_measure_ && wake_.size() == std::size_t(sbw) * ((h + kSig - 1) / kSig);
    auto sample = [&](float x, float y, float acc[3]) {
        const int x0 = std::clamp(int(x - 0.5f), 0, w - 2);
        const int y0 = std::clamp(int(y - 0.5f), 0, h - 2);
        const std::uint8_t* p0 = rgba + std::size_t(y0) * stride + 4 * x0;
        const std::uint8_t* p1 = p0 + stride;
        for (int k = 0; k < 3; ++k) acc[k] += lin[p0[k]] + lin[p0[4 + k]] + lin[p1[k]] + lin[p1[4 + k]];
    };
    const float threshold2 = c.recolor_threshold * c.recolor_threshold;
    stats_.recolored = 0;
    for (Tile& t : tiles_) {
        const bool fresh = t.measured.l < 0;
        bool measure = fresh || !gate;
        if (!measure) {
            const float r = std::max(t.half_t, t.half_n);
            const int x0 = std::max(0, int(t.x - r)) / kSig, x1 = std::min(w - 1, int(t.x + r)) / kSig;
            const int y0 = std::max(0, int(t.y - r)) / kSig, y1 = std::min(h - 1, int(t.y + r)) / kSig;
            for (int by = y0; by <= y1 && !measure; ++by)
                for (int bx = x0; bx <= x1; ++bx)
                    if (wake_[std::size_t(by) * sbw + bx]) { measure = true; break; }
        }
        bool display = !same_palette || fresh;
        if (measure) {
            const float at = 0.4f * t.half_t, an = 0.4f * t.half_n;
            float acc[3] = {0, 0, 0};
            sample(t.x, t.y, acc);
            sample(t.x, t.y, acc);
            sample(t.x + at * t.c, t.y + at * t.s, acc);
            sample(t.x - at * t.c, t.y - at * t.s, acc);
            sample(t.x - an * t.s, t.y + an * t.c, acc);
            sample(t.x + an * t.s, t.y - an * t.c, acc);
            const Lab m = linear_to_oklab(acc[0] * (1.f / 24), acc[1] * (1.f / 24), acc[2] * (1.f / 24));
            if (fresh || dist2(m, t.measured) > threshold2) {
                t.measured = m;
                display = true;
            }
        }
        if (!display) continue;
        // Display color is a pure function of the committed measurement and
        // the palette, so it only moves when one of those does.
        Lab d = t.measured;
        if (!palette.empty()) {
            float best = std::numeric_limits<float>::max();
            for (const Lab& p : palette) {
                const float e = dist2(t.measured, p);
                if (e < best) { best = e; d = p; }
            }
        }
        if (t.outline && c.outline > 0) {
            const float o = c.outline * t.outline / 255.f;
            d.l *= 1 - 0.75f * o;
            d.a *= 1 - 0.3f * o;
            d.b *= 1 - 0.3f * o;
        }
        d.l += c.lightness_jitter * (2 * unit(hash32(t.key ^ 0x51ed270bu)) - 1);
        d.a += c.chroma_jitter * (2 * unit(hash32(t.key ^ 0x2c1b3c6du)) - 1);
        d.b += c.chroma_jitter * (2 * unit(hash32(t.key ^ 0x297a2d39u)) - 1);
        t.color = d;
        oklab_to_srgb(d, t.rgb);
        if (!fresh) ++stats_.recolored; // display changes of existing tiles
    }
}

void Mosaic::build_grid()
{
    grid_.cell = size_px_.empty() ? 1.f : size_px_[0];
    grid_.cols = int(f_.width / grid_.cell) + 1;
    grid_.rows = int(f_.height / grid_.cell) + 1;
    const std::size_t cells = std::size_t(grid_.cols) * grid_.rows;
    grid_.start.assign(cells + 1, 0);
    next_.resize(tiles_.size()); // reused as the per-tile cell index
    for (std::size_t k = 0; k < tiles_.size(); ++k) {
        const int cx = std::clamp(int(tiles_[k].x / grid_.cell), 0, grid_.cols - 1);
        const int cy = std::clamp(int(tiles_[k].y / grid_.cell), 0, grid_.rows - 1);
        next_[k] = cy * grid_.cols + cx;
        ++grid_.start[std::size_t(next_[k]) + 1];
    }
    max_cell_ = 0;
    for (std::size_t i = 0; i < cells; ++i) {
        max_cell_ = std::max(max_cell_, grid_.start[i + 1]);
        grid_.start[i + 1] += grid_.start[i];
    }
    grid_.items.resize(tiles_.size());
    head_.assign(grid_.start.begin(), grid_.start.end() - 1); // reused as fill cursors
    for (std::size_t k = 0; k < tiles_.size(); ++k) grid_.items[std::size_t(head_[next_[k]]++)] = std::uint32_t(k);
}

const Stats& Mosaic::update_colors(const std::uint8_t* rgba, int width, int height, std::size_t stride,
                                   const std::vector<Lab>& palette, bool keep_pending)
{
    const auto t0 = Clock::now();
    stats_ = {};
    if (tiles_.empty() || width != f_.width || height != f_.height) return stats_;
    if (!prefiltered_ || prefilter_frame_ != rgba) prefilter(rgba, stride);
    stats_.woken_blocks = woken_;
    measure_and_color(rgba, stride, palette);
    // Everything was re-measured after a global shift: that frame is the new
    // reference.  Otherwise unwoken blocks keep accumulating drift.
    if (!keep_pending) {
        if (full_measure_) signature_ = now_signature_;
        else
            for (std::size_t i = 0; i < wake_.size(); ++i)
                if (wake_[i]) std::copy_n(&now_signature_[3 * i], 3, &signature_[3 * i]);
    }
    prefiltered_ = false;
    stats_.tiles = std::uint32_t(tiles_.size());
    stats_.max_cell_tiles = max_cell_;
    stats_.tile_scale = size_px_.empty() ? 0.f : size_px_[0];
    stats_.color_ms = ms_since(t0);
    return stats_;
}

namespace {
void allocate(Fields& f, int w, int h)
{
    const std::size_t n = std::size_t(w) * h;
    f.width = w;
    f.height = h;
    f.structure.assign(n, 0.f);
    f.edge.assign(n, 0);
    f.outline.assign(n, 0);
    f.level.assign(n, 0);
    f.distance.assign(n, 0.f);
    f.c2.assign(n, 0);
    f.s2.assign(n, 0);
    f.from_frame.assign(n, 0);
}
} // namespace

const Stats& Mosaic::process(const std::uint8_t* rgba, int width, int height, std::size_t stride,
                             const float* structure, const std::vector<Lab>& palette)
{
    if (width != f_.width || height != f_.height) {
        reset();
        allocate(f_, width, height);
    }
    const std::size_t n = std::size_t(width) * height;
    if (structure) {
        for (std::size_t i = 0; i < n; ++i) f_.structure[i] = std::clamp(structure[i], 0.f, 1.f);
    } else {
        // Without a cartoon, Oklab lightness is the structure.
        for (int y = 0; y < height; ++y) {
            const std::uint8_t* p = rgba + std::size_t(y) * stride;
            float* o = &f_.structure[std::size_t(y) * width];
            for (int x = 0; x < width; ++x, p += 4) o[x] = std::clamp(srgb8_to_oklab(p[0], p[1], p[2]).l, 0.f, 1.f);
        }
    }
    return process_impl(rgba, stride, palette);
}

const Stats& Mosaic::process_swap(const std::uint8_t* rgba, int width, int height, std::size_t stride,
                                  std::vector<float>& structure, const std::vector<Lab>& palette)
{
    if (width != f_.width || height != f_.height) {
        reset();
        allocate(f_, width, height);
    }
    if (structure.size() != std::size_t(width) * height) return process(rgba, width, height, stride, nullptr, palette);
    f_.structure.swap(structure);
    return process_impl(rgba, stride, palette);
}

const Stats& Mosaic::process_impl(const std::uint8_t* rgba, std::size_t stride, const std::vector<Lab>& palette)
{
    const int w = f_.width, h = f_.height;
    const std::size_t n = std::size_t(w) * h;
    stats_ = {};
    // Reusable scratch: only the int16 column sites are frame-sized.
    fit(col_site_, n, std::int16_t(-1));
    fit(rows_, std::size_t(h), std::uint8_t(0));
    fit(v_, std::size_t(w), 0);
    fit(z_, std::size_t(w) + 1, 0.0);
    const int bw8 = (w + kEntropyBlock - 1) / kEntropyBlock, bh8 = (h + kEntropyBlock - 1) / kEntropyBlock;
    fit(block_raw_, std::size_t(bw8) * bh8, 0.f);

    if (!prefiltered_ || prefilter_frame_ != rgba) prefilter(rgba, stride);
    stats_.woken_blocks = woken_;
    const int dbw = (w + kBlock - 1) / kBlock, dbh = (h + kBlock - 1) / kBlock;
    const std::size_t blocks = std::size_t(dbw) * dbh;
    stats_.blocks = std::uint32_t(blocks);
    dirty_.assign(blocks, 0);
    bool rebuild = tiles_.empty() || reference_.size() != n;
    if (!rebuild && woken_) {
        // Only blocks the prefilter woke (with a one-block halo) are tested:
        // drift against what the tiles were laid on, with each block's mean
        // change removed so brightness shifts recolor instead.
        const int sbw = (w + kSig - 1) / kSig, sbh = (h + kSig - 1) / kSig;
        std::vector<std::uint8_t>& candidate = candidate_;
        candidate.assign(blocks, 0);
        for (int sy = 0; sy < sbh; ++sy)
            for (int sx = 0; sx < sbw; ++sx) {
                if (!wake_[std::size_t(sy) * sbw + sx]) continue;
                const int bx = sx * kSig / kBlock, by = sy * kSig / kBlock;
                for (int yy = std::max(0, by - 1); yy <= std::min(dbh - 1, by + 1); ++yy)
                    for (int xx = std::max(0, bx - 1); xx <= std::min(dbw - 1, bx + 1); ++xx)
                        candidate[std::size_t(yy) * dbw + xx] = 1;
            }
        std::uint32_t marked = 0;
        for (int by = 0; by < dbh; ++by)
            for (int bx = 0; bx < dbw; ++bx) {
                if (!candidate[std::size_t(by) * dbw + bx]) continue;
                const int x0 = bx * kBlock, x1 = std::min(w, x0 + kBlock), y0 = by * kBlock, y1 = std::min(h, y0 + kBlock);
                double mean = 0;
                for (int y = y0; y < y1; ++y)
                    for (int x = x0; x < x1; ++x) mean += f_.structure[std::size_t(y) * w + x] - reference_[std::size_t(y) * w + x];
                const int count = (x1 - x0) * (y1 - y0);
                mean /= count;
                double change = 0;
                for (int y = y0; y < y1; ++y)
                    for (int x = x0; x < x1; ++x)
                        change += std::fabs(f_.structure[std::size_t(y) * w + x] - reference_[std::size_t(y) * w + x] - mean);
                if (change / count > config_.restructure_threshold) { dirty_[std::size_t(by) * dbw + bx] = 1; ++marked; }
            }
        if (marked > config_.scene_cut_fraction * blocks) rebuild = true;
        else if (marked) {
            // One-block halo so rows re-form smoothly into the kept tiling.
            std::vector<std::uint8_t> grown(dirty_);
            for (int by = 0; by < dbh; ++by)
                for (int bx = 0; bx < dbw; ++bx) {
                    if (!dirty_[std::size_t(by) * dbw + bx]) continue;
                    for (int yy = std::max(0, by - 1); yy <= std::min(dbh - 1, by + 1); ++yy)
                        for (int xx = std::max(0, bx - 1); xx <= std::min(dbw - 1, bx + 1); ++xx)
                            grown[std::size_t(yy) * dbw + xx] = 1;
                }
            dirty_.swap(grown);
        }
    }
    if (rebuild) std::fill(dirty_.begin(), dirty_.end(), 1);
    for (auto d : dirty_) stats_.dirty_blocks += d;

    if (rebuild || stats_.dirty_blocks) {
        build_runs(rebuild);
        const auto ta = Clock::now();
        analyze_geometry(rgba, stride);
        stats_.analysis_ms = ms_since(ta);
        const auto tn = Clock::now();
        nucleate(rebuild);
        stats_.nucleation_ms = ms_since(tn);
        if (rebuild) {
            reference_ = f_.structure;
            stats_.rebuilt = true;
        } else {
            for (const Run& r : runs_)
                for (int y = r.y0; y < r.y1; ++y)
                    std::copy_n(&f_.structure[std::size_t(y) * w + r.x0], r.x1 - r.x0, &reference_[std::size_t(y) * w + r.x0]);
        }
        stats_.geometry = true;
    }
    const auto tc = Clock::now();
    measure_and_color(rgba, stride, palette);
    if (stats_.geometry) build_grid();
    // Woken blocks were examined: they become the new prefilter reference.
    if (rebuild || full_measure_ || signature_.size() != now_signature_.size()) signature_ = now_signature_;
    else
        for (std::size_t i = 0; i < wake_.size(); ++i)
            if (wake_[i]) std::copy_n(&now_signature_[3 * i], 3, &signature_[3 * i]);
    prefiltered_ = false;
    stats_.color_ms = ms_since(tc);
    stats_.tiles = std::uint32_t(tiles_.size());
    stats_.tile_scale = size_px_.empty() ? 0.f : size_px_[0];
    stats_.max_cell_tiles = max_cell_;
    return stats_;
}

void light_vectors(const RenderParams& p, float L[3], float H[3])
{
    const float az = p.light_azimuth + p.sweep * p.time;
    const float ce = std::cos(p.light_elevation);
    L[0] = ce * std::cos(az);
    L[1] = ce * std::sin(az);
    L[2] = std::sin(p.light_elevation);
    H[0] = L[0]; H[1] = L[1]; H[2] = L[2] + 1;
    const float hl = std::sqrt(H[0] * H[0] + H[1] * H[1] + H[2] * H[2]);
    for (int k = 0; k < 3; ++k) H[k] /= hl;
}

void render(const std::vector<Tile>& tiles, const Grid& g, const RenderParams& p, std::uint8_t* out,
            int width, int height, std::size_t stride)
{
    float L[3], H[3];
    light_vectors(p, L, H);
    const float* grout = p.grout;
    const float inv = 1.f / p.scale;
    const float half_grout = 0.5f * p.grout_px;

    for (int oy = 0; oy < height; ++oy) {
        std::uint8_t* row = out + std::size_t(oy) * stride;
        const float qy = (oy + 0.5f) * inv;
        const int cy = int(qy / g.cell);
        for (int ox = 0; ox < width; ++ox) {
            const float qx = (ox + 0.5f) * inv;
            const int cx = int(qx / g.cell);
            float best = kInf, second = kInf;
            int owner = -1, other = -1;
            float bu = 0, bv = 0;
            for (int yy = std::max(0, cy - 1); yy <= std::min(g.rows - 1, cy + 1); ++yy)
                for (int xx = std::max(0, cx - 1); xx <= std::min(g.cols - 1, cx + 1); ++xx) {
                    const std::size_t cell = std::size_t(yy) * g.cols + xx;
                    for (std::uint32_t it = g.start[cell]; it < g.start[cell + 1]; ++it) {
                        const Tile& t = tiles[g.items[it]];
                        const float dx = qx - t.x, dy = qy - t.y;
                        const float u = dx * t.c + dy * t.s, v = -dx * t.s + dy * t.c;
                        const float rho = std::max(std::fabs(u) / t.half_t, std::fabs(v) / t.half_n);
                        if (rho < best) {
                            second = best; other = owner;
                            best = rho; owner = int(g.items[it]);
                            bu = u; bv = v;
                        } else if (rho < second) {
                            second = rho; other = int(g.items[it]);
                        }
                    }
                }
            float color[3] = {grout[0], grout[1], grout[2]};
            if (owner >= 0 && best < 1.f) {
                const Tile& t = tiles[owner];
                const float mt = t.half_t - std::fabs(bu), mn = t.half_n - std::fabs(bv);
                float edge = std::min(mt, mn) * p.scale;
                if (other >= 0) {
                    const Tile& o = tiles[other];
                    const float rate = 1.f / std::min(t.half_t, t.half_n) + 1.f / std::min(o.half_t, o.half_n);
                    edge = std::min(edge, (second - best) / rate * p.scale);
                }
                const float cover = std::clamp(edge - half_grout + 0.5f, 0.f, 1.f);
                // Facet normal: random tilt plus a bevel toward the nearer side.
                float nx = p.facet * t.tilt_x, ny = p.facet * t.tilt_y;
                const float bevel = p.bevel * (1 - std::clamp((edge - half_grout) / p.bevel_px, 0.f, 1.f));
                if (mt < mn) {
                    const float sgn = bu < 0 ? -1.f : 1.f;
                    nx += bevel * sgn * t.c; ny += bevel * sgn * t.s;
                } else {
                    const float sgn = bv < 0 ? -1.f : 1.f;
                    nx += -bevel * sgn * t.s; ny += bevel * sgn * t.c;
                }
                const float nl = 1.f / std::sqrt(nx * nx + ny * ny + 1);
                const float N[3] = {nx * nl, ny * nl, nl};
                const float lambert = N[0] * L[0] + N[1] * L[1] + N[2] * L[2];
                const float shade = 1 + p.diffuse * (lambert - L[2]);
                const float nh = std::max(0.f, N[0] * H[0] + N[1] * H[1] + N[2] * H[2]);
                float spec = p.glint * std::pow(nh, p.shininess);
                if (p.twinkle > 0)
                    spec *= 1 - p.twinkle * (0.5f + 0.5f * std::sin(6.2831853f * (t.phase + 0.37f * p.time)));
                for (int k = 0; k < 3; ++k) {
                    const float tint = 1 - p.metal + p.metal * t.rgb[k];
                    const float v = t.rgb[k] * shade + spec * tint;
                    color[k] = grout[k] + (v - grout[k]) * cover;
                }
            }
            for (int k = 0; k < 3; ++k)
                row[4 * ox + k] = std::uint8_t(std::lround(std::clamp(color[k], 0.f, 1.f) * 255.f));
            row[4 * ox + 3] = 255;
        }
    }
}

namespace {
inline std::uint16_t q16(float v) { return std::uint16_t(std::lround(std::clamp(v, 0.f, 1.f) * 65535.f)); }
inline float d16(std::uint16_t v) { return v * (1.f / 65535.f); }
} // namespace

void pack(const Mosaic& m, GpuPack& out, bool geometry)
{
    const auto& tiles = m.tiles();
    const Grid& g = m.grid();
    const std::uint32_t count = std::uint32_t(std::min<std::size_t>(tiles.size(), GpuPack::kMaxTiles));
    constexpr int T = GpuPack::kTilesPerRow;
    if (geometry || out.order.size() != count) {
        const auto& f = m.fields();
        out.count = count;
        out.analysis_w = float(std::max(1, f.width));
        out.analysis_h = float(std::max(1, f.height));
        out.cell = g.cell;
        out.cells_w = std::max(1, g.cols);
        out.cells_h = std::max(1, g.rows);
        float half = 1e-3f;
        for (const Tile& t : tiles) half = std::max(half, std::max(t.half_t, t.half_n));
        out.half_scale = half;
        out.rows = std::max(1, int((count + T - 1) / T));
        out.geometry.assign(std::size_t(2 * T) * out.rows * 4, 0);
        out.cells.assign(std::size_t(out.cells_w) * out.cells_h * 4, 0);
        out.order.resize(count);
        out.max_cell = 0;
        const float iw = 1.f / out.analysis_w, ih = 1.f / out.analysis_h, ihalf = 1.f / half;
        std::uint32_t slot = 0;
        for (int cell = 0; cell < g.cols * g.rows; ++cell) {
            const std::uint32_t start = slot;
            for (std::uint32_t it = g.start[cell]; it < g.start[cell + 1] && slot < count; ++it, ++slot) {
                const Tile& t = tiles[g.items[it]];
                out.order[slot] = g.items[it];
                std::uint16_t* p = &out.geometry[(std::size_t(slot / T) * 2 * T + std::size_t(slot % T) * 2) * 4];
                p[0] = q16(t.x * iw); p[1] = q16(t.y * ih);
                p[2] = q16(t.c * 0.5f + 0.5f); p[3] = q16(t.s * 0.5f + 0.5f);
                p[4] = q16(t.half_t * ihalf); p[5] = q16(t.half_n * ihalf);
                p[6] = q16(t.tilt_x * 0.5f + 0.5f); p[7] = q16(t.tilt_y * 0.5f + 0.5f);
            }
            const std::uint32_t n = slot - start;
            out.max_cell = std::max(out.max_cell, n);
            std::uint16_t* c = &out.cells[std::size_t(cell) * 4];
            c[0] = std::uint16_t(start & 0xffff);
            c[1] = std::uint16_t(start >> 16);
            c[2] = std::uint16_t(n);
        }
    }
    out.colors.resize(std::size_t(T) * out.rows * 4);
    for (std::uint32_t slot = 0; slot < out.count; ++slot) {
        const Tile& t = tiles[out.order[slot]];
        std::uint16_t* p = &out.colors[std::size_t(slot) * 4];
        p[0] = q16(t.rgb[0]); p[1] = q16(t.rgb[1]); p[2] = q16(t.rgb[2]); p[3] = q16(t.phase);
    }
}

void unpack(const GpuPack& in, std::vector<Tile>& tiles, Grid& g)
{
    constexpr int T = GpuPack::kTilesPerRow;
    tiles.assign(in.count, Tile{});
    for (std::uint32_t i = 0; i < in.count; ++i) {
        const std::uint16_t* p = &in.geometry[(std::size_t(i / T) * 2 * T + std::size_t(i % T) * 2) * 4];
        const std::uint16_t* q = &in.colors[std::size_t(i) * 4];
        Tile& t = tiles[i];
        t.x = d16(p[0]) * in.analysis_w;
        t.y = d16(p[1]) * in.analysis_h;
        t.c = d16(p[2]) * 2 - 1;
        t.s = d16(p[3]) * 2 - 1;
        t.half_t = d16(p[4]) * in.half_scale;
        t.half_n = d16(p[5]) * in.half_scale;
        t.tilt_x = d16(p[6]) * 2 - 1;
        t.tilt_y = d16(p[7]) * 2 - 1;
        for (int k = 0; k < 3; ++k) t.rgb[k] = d16(q[k]);
        t.phase = d16(q[3]);
    }
    g.cell = in.cell;
    g.cols = in.cells_w;
    g.rows = in.cells_h;
    g.start.assign(std::size_t(g.cols) * g.rows + 1, 0);
    g.items.resize(in.count);
    for (int cell = 0; cell < g.cols * g.rows; ++cell) {
        const std::uint16_t* c = &in.cells[std::size_t(cell) * 4];
        g.start[cell] = std::uint32_t(c[0]) | (std::uint32_t(c[1]) << 16);
        g.start[cell + 1] = g.start[cell] + c[2];
    }
    for (std::uint32_t i = 0; i < in.count; ++i) g.items[i] = i;
}

} // namespace mosaic
