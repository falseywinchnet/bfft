#include "mosaic/mosaic.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

namespace {
int failures = 0;
#define CHECK(cond, ...)                                                          \
    do {                                                                          \
        if (!(cond)) {                                                            \
            ++failures;                                                           \
            std::printf("FAIL %s:%d %s: ", __FILE__, __LINE__, #cond);            \
            std::printf(__VA_ARGS__);                                             \
            std::printf("\n");                                                    \
        }                                                                         \
    } while (0)

constexpr int W = 480, H = 270;

// Smooth background, a disc, a striped (high entropy) rectangle.
std::vector<std::uint8_t> scene(float disc_x, unsigned noise_seed = 0, int noise = 0)
{
    std::vector<std::uint8_t> p(std::size_t(W) * H * 4, 255);
    unsigned state = noise_seed * 2654435761u + 1;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            float r = 40 + 120.f * x / W, g = 60 + 80.f * y / H, b = 150;
            if ((x - disc_x) * (x - disc_x) + (y - 135.f) * (y - 135.f) < 60.f * 60.f) { r = 230; g = 190; b = 40; }
            if (x > 300 && x < 440 && y > 40 && y < 140) {
                // Many-colored fine texture: high local color entropy.
                const unsigned hsh = (unsigned(x) * 73856093u) ^ (unsigned(y) * 19349663u);
                r = float(40 + (hsh >> 3) % 200); g = float(40 + (hsh >> 11) % 200); b = float(40 + (hsh >> 19) % 200);
            }
            float v[3] = {r, g, b};
            for (int k = 0; k < 3; ++k) {
                if (noise) { state = state * 1664525u + 1013904223u; v[k] += float(int(state >> 24) % (2 * noise + 1) - noise); }
                p[(std::size_t(y) * W + x) * 4 + k] = std::uint8_t(std::fmin(255.f, std::fmax(0.f, v[k])));
            }
        }
    return p;
}

bool same_tiles(const std::vector<mosaic::Tile>& a, const std::vector<mosaic::Tile>& b)
{
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (a[i].x != b[i].x || a[i].y != b[i].y || a[i].key != b[i].key || a[i].c != b[i].c ||
            a[i].color.l != b[i].color.l || a[i].color.a != b[i].color.a)
            return false;
    return true;
}

void test_oklab_roundtrip()
{
    float worst = 0;
    for (int r = 0; r < 256; r += 17)
        for (int g = 0; g < 256; g += 17)
            for (int b = 0; b < 256; b += 17) {
                float out[3];
                mosaic::oklab_to_srgb(mosaic::srgb8_to_oklab(std::uint8_t(r), std::uint8_t(g), std::uint8_t(b)), out);
                worst = std::fmax(worst, std::fabs(out[0] * 255 - r));
                worst = std::fmax(worst, std::fabs(out[1] * 255 - g));
                worst = std::fmax(worst, std::fabs(out[2] * 255 - b));
            }
    CHECK(worst < 0.6f, "worst %.3f", worst);
}

void test_deterministic_and_budget()
{
    mosaic::Config c;
    c.target_tiles = 3000;
    const auto img = scene(150);
    mosaic::Mosaic a, b;
    a.set_config(c);
    b.set_config(c);
    a.process(img.data(), W, H, W * 4, nullptr, {});
    b.process(img.data(), W, H, W * 4, nullptr, {});
    CHECK(same_tiles(a.tiles(), b.tiles()), "independent instances disagree");
    const auto n = a.tiles().size();
    CHECK(n > 2000 && n < 4500, "tiles %zu for budget 3000", n);
    // More tiles land in the high-entropy stripes than an equal-area smooth patch.
    int stripes = 0, smooth = 0;
    for (const auto& t : a.tiles()) {
        if (t.x > 300 && t.x < 440 && t.y > 40 && t.y < 140) ++stripes;
        if (t.x > 20 && t.x < 160 && t.y > 160 && t.y < 260) ++smooth;
    }
    CHECK(stripes > smooth * 3 / 2, "stripes %d smooth %d", stripes, smooth);
    int outline = 0;
    for (const auto& t : a.tiles()) outline += t.outline > 0;
    CHECK(outline > 50, "outline tiles %d", outline);
    // Reprocessing a reset instance reproduces the same nucleation.
    a.reset();
    a.process(img.data(), W, H, W * 4, nullptr, {});
    CHECK(same_tiles(a.tiles(), b.tiles()), "reset changed nucleation");
}

void test_render_coverage()
{
    mosaic::Config c;
    c.target_tiles = 3000;
    const auto img = scene(150);
    mosaic::Mosaic m;
    m.set_config(c);
    m.process(img.data(), W, H, W * 4, nullptr, {});
    mosaic::RenderParams p;
    p.scale = 2;
    std::vector<std::uint8_t> out(std::size_t(W) * 2 * H * 2 * 4);
    mosaic::render(m, p, out.data(), W * 2, H * 2, std::size_t(W) * 2 * 4);
    const float* grout = p.grout;
    std::size_t grout_px = 0;
    for (std::size_t i = 0; i < out.size() / 4; ++i) {
        int d = 0;
        for (int k = 0; k < 3; ++k) d += std::abs(int(out[i * 4 + k]) - int(std::lround(grout[k] * 255)));
        grout_px += d < 6;
    }
    const double fraction = double(grout_px) / (out.size() / 4);
    CHECK(fraction > 0.04 && fraction < 0.35, "grout fraction %.3f", fraction);
}

void test_temporal_persistence()
{
    mosaic::Config c;
    c.target_tiles = 3000;
    mosaic::Mosaic m;
    m.set_config(c);
    m.process(scene(150).data(), W, H, W * 4, nullptr, {});
    const auto first = m.tiles();

    // Sensor noise alone: nothing re-nucleates or recolors.
    const auto& s1 = m.process(scene(150, 7, 3).data(), W, H, W * 4, nullptr, {});
    CHECK(s1.added == 0 && s1.removed == 0 && !s1.rebuilt, "noise restructured +%u -%u", s1.added, s1.removed);
    CHECK(s1.recolored == 0, "noise recolored %u", s1.recolored);
    CHECK(same_tiles(first, m.tiles()), "noise changed tiles");

    // Moving the disc only re-nucleates near it; far tiles are untouched.
    const auto& s2 = m.process(scene(170).data(), W, H, W * 4, nullptr, {});
    CHECK(!s2.rebuilt && s2.dirty_blocks > 0 && s2.dirty_blocks < s2.blocks / 2,
          "dirty %u of %u rebuilt %d", s2.dirty_blocks, s2.blocks, s2.rebuilt);
    std::size_t kept_far = 0, far = 0;
    for (const auto& t : first) {
        if (t.x < 300) continue; // disc spans x < 260 after dilation
        ++far;
        for (const auto& u : m.tiles())
            if (u.key == t.key && u.x == t.x && u.y == t.y) { ++kept_far; break; }
    }
    CHECK(far > 0 && kept_far == far, "far tiles kept %zu of %zu", kept_far, far);

    // A different scene is a cut: full deterministic rebuild.
    std::vector<std::uint8_t> other(std::size_t(W) * H * 4, 255);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const bool on = ((x / 12) + (y / 12)) % 2;
            std::uint8_t* q = &other[(std::size_t(y) * W + x) * 4];
            q[0] = on ? 230 : 30; q[1] = on ? 200 : 60; q[2] = on ? 40 : 160;
        }
    const auto& s3 = m.process(other.data(), W, H, W * 4, nullptr, {});
    CHECK(s3.rebuilt, "scene cut not rebuilt");
    mosaic::Mosaic fresh;
    fresh.set_config(c);
    fresh.process(other.data(), W, H, W * 4, nullptr, {});
    CHECK(same_tiles(fresh.tiles(), m.tiles()), "rebuild depends on history");
}

void test_palette_quantization()
{
    mosaic::Config c;
    c.target_tiles = 2000;
    c.lightness_jitter = 0;
    c.chroma_jitter = 0;
    c.outline = 0;
    std::vector<mosaic::Lab> palette = {{0.3f, 0, 0}, {0.6f, 0.05f, 0.1f}, {0.8f, -0.05f, 0.02f}};
    mosaic::Mosaic m;
    m.set_config(c);
    m.process(scene(150).data(), W, H, W * 4, nullptr, palette);
    bool all = true;
    for (const auto& t : m.tiles()) {
        bool hit = false;
        for (const auto& p : palette) hit |= t.color.l == p.l && t.color.a == p.a && t.color.b == p.b;
        all &= hit;
    }
    CHECK(all, "unjittered tile colors must be palette nodes");
}
void test_realtime_paths()
{
    mosaic::Config c;
    c.target_tiles = 3000;
    mosaic::Mosaic m;
    m.set_config(c);
    const auto base = scene(150);
    CHECK(m.wants_structure(base.data(), W, H, W * 4), "empty mosaic must want structure");
    m.process(base.data(), W, H, W * 4, nullptr, {});
    const auto laid = m.tiles();
    CHECK(m.stats().max_cell_tiles > 0 && m.stats().max_cell_tiles <= 24,
          "max cell tiles %u exceeds shader loop bound", m.stats().max_cell_tiles);

    // Prefilter: sensor noise sleeps, a moving object wakes geometry.
    const auto noisy = scene(150, 3, 3);
    CHECK(!m.wants_structure(noisy.data(), W, H, W * 4), "noise woke the prefilter");
    const auto moved = scene(160);
    CHECK(m.wants_structure(moved.data(), W, H, W * 4), "moving disc did not wake the prefilter");

    // Color-only updates never touch geometry.
    m.update_colors(noisy.data(), W, H, W * 4, {});
    bool same_geometry = m.tiles().size() == laid.size();
    for (std::size_t i = 0; same_geometry && i < laid.size(); ++i)
        same_geometry = laid[i].x == m.tiles()[i].x && laid[i].y == m.tiles()[i].y && laid[i].c == m.tiles()[i].c;
    CHECK(same_geometry, "update_colors moved tiles");
    CHECK(m.stats().recolored == 0, "noise recolored %u in color path", m.stats().recolored);

    // A global brightness shift recolors but does not re-lay tiles.
    auto bright = base;
    for (std::size_t i = 0; i < bright.size(); i += 4)
        for (int k = 0; k < 3; ++k) bright[i + k] = std::uint8_t(std::min(255, bright[i + k] + 30));
    const auto& st = m.process(bright.data(), W, H, W * 4, nullptr, {});
    CHECK(!st.rebuilt && st.added == 0 && st.removed == 0, "brightness re-laid tiles: rebuilt %d +%u -%u dirty %u",
          st.rebuilt, st.added, st.removed, st.dirty_blocks);
    CHECK(st.recolored > laid.size() / 2, "brightness recolored only %u", st.recolored);
}

void test_pack_roundtrip()
{
    mosaic::Config c;
    c.target_tiles = 3000;
    mosaic::Mosaic m;
    m.set_config(c);
    m.process(scene(150).data(), W, H, W * 4, nullptr, {});
    mosaic::GpuPack pk;
    mosaic::pack(m, pk);
    CHECK(pk.count == m.tiles().size(), "packed %u of %zu", pk.count, m.tiles().size());
    CHECK(pk.max_cell == m.stats().max_cell_tiles, "pack cell max %u vs %u", pk.max_cell, m.stats().max_cell_tiles);
    std::vector<mosaic::Tile> tiles;
    mosaic::Grid grid;
    mosaic::unpack(pk, tiles, grid);
    mosaic::RenderParams p;
    p.scale = 2;
    const int ow = W * 2, oh = H * 2;
    std::vector<std::uint8_t> a(std::size_t(ow) * oh * 4), b(a.size());
    mosaic::render(m, p, a.data(), ow, oh, std::size_t(ow) * 4);
    mosaic::render(tiles, grid, p, b.data(), ow, oh, std::size_t(ow) * 4);
    double sum = 0;
    int worst = 0;
    std::size_t big = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const int e = std::abs(int(a[i]) - int(b[i]));
        sum += e;
        worst = std::max(worst, e);
        big += e > 8;
    }
    // 16-bit quantization only moves sub-pixel grout boundaries.
    CHECK(sum / a.size() < 0.05 && big < a.size() / 2000, "pack render mean %.4f worst %d big %zu",
          sum / a.size(), worst, big);
}
void test_region_matches_full_analysis()
{
    // Incremental geometry runs only on dirty blocks; inside them the fields
    // must agree with a fresh full-frame analysis of the same frame.
    mosaic::Config c;
    c.target_tiles = 3000;
    mosaic::Mosaic inc, full;
    inc.set_config(c);
    full.set_config(c);
    inc.process(scene(150).data(), W, H, W * 4, nullptr, {});
    const auto moved = scene(170);
    const auto& st = inc.process(moved.data(), W, H, W * 4, nullptr, {});
    CHECK(!st.rebuilt && st.dirty_blocks > 0, "expected an incremental update");
    full.process(moved.data(), W, H, W * 4, nullptr, {});
    const auto& a = inc.fields();
    const auto& b = full.fields();
    std::size_t n = 0, edge_diff = 0, level_diff = 0;
    for (int y = 90; y < 180; ++y)
        for (int x = 130; x < 220; ++x) { // inside the dirty region around the disc edge
            const std::size_t i = std::size_t(y) * W + x;
            ++n;
            edge_diff += a.edge[i] != b.edge[i];
            level_diff += a.level[i] != b.level[i];
        }
    CHECK(edge_diff * 100 < n && level_diff * 100 < n, "region fields differ: edges %zu levels %zu of %zu",
          edge_diff, level_diff, n);
    // Distances are exact against the instance's own (partly retained) edge
    // map plus the frame: brute force over the window.
    std::vector<std::pair<int, int>> edges;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            if (a.edge[std::size_t(y) * W + x]) edges.emplace_back(x, y);
    double worst = 0;
    std::size_t checked = 0;
    const int dbw = (W + 15) / 16;
    for (int y = 0; y < H; y += 3)
        for (int x = 0; x < W; x += 3) {
            if (!inc.dirty_blocks()[std::size_t(y / 16) * dbw + x / 16]) continue;
            ++checked;
            double best = std::min(std::min(x + 0.5, W - x - 0.5), std::min(y + 0.5, H - y - 0.5));
            for (const auto& [ex, ey] : edges) best = std::min(best, std::hypot(double(x - ex), double(y - ey)));
            worst = std::max(worst, std::fabs(best - a.distance[std::size_t(y) * W + x]));
        }
    CHECK(checked > 500 && worst < 1e-3, "incremental distance transform error %.5f over %zu", worst, checked);
}
} // namespace

int main()
{
    test_oklab_roundtrip();
    test_deterministic_and_budget();
    test_render_coverage();
    test_temporal_persistence();
    test_palette_quantization();
    test_realtime_paths();
    test_pack_roundtrip();
    test_region_matches_full_analysis();
    std::printf(failures ? "%d failure(s)\n" : "all mosaic tests passed\n", failures);
    return failures ? 1 : 0;
}
