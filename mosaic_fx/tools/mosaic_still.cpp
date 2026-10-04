// Still-image driver: PPM in, mosaic PPM out, plus a nucleation diagnostic.
//   mosaic_still in.ppm out.ppm [--tiles N] [--classes C] [--palette P]
//       [--analysis-width W] [--lattice 512|1024|2048] [--no-meyer]
//       [--outline X] [--contour-rows N] [--time T] [--diag diag.ppm]
//       [--grout PX] [--glint G] [--facet F] [--frames K --shift DX]
#include "mosaic/mosaic.hpp"
#include "rvfx/engine.hpp"

#include "mosaic/cartoon.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct Image { int w = 0, h = 0; std::vector<std::uint8_t> rgb; };

bool read_ppm(const char* path, Image& im)
{
    FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    char magic[3] = {0};
    int maxv = 0;
    auto skip = [&] {
        int ch;
        while ((ch = std::fgetc(f)) != EOF) {
            if (ch == '#') { while ((ch = std::fgetc(f)) != EOF && ch != '\n') {} }
            else if (!std::isspace(ch)) { std::ungetc(ch, f); break; }
        }
    };
    bool ok = std::fscanf(f, "%2s", magic) == 1 && std::strcmp(magic, "P6") == 0;
    skip(); ok = ok && std::fscanf(f, "%d", &im.w) == 1;
    skip(); ok = ok && std::fscanf(f, "%d", &im.h) == 1;
    skip(); ok = ok && std::fscanf(f, "%d", &maxv) == 1 && maxv == 255;
    std::fgetc(f);
    if (ok) {
        im.rgb.resize(std::size_t(im.w) * im.h * 3);
        ok = std::fread(im.rgb.data(), 1, im.rgb.size(), f) == im.rgb.size();
    }
    std::fclose(f);
    return ok;
}

bool write_ppm(const char* path, int w, int h, const std::vector<std::uint8_t>& rgba)
{
    FILE* f = std::fopen(path, "wb");
    if (!f) return false;
    std::fprintf(f, "P6\n%d %d\n255\n", w, h);
    std::vector<std::uint8_t> rgb(std::size_t(w) * h * 3);
    for (std::size_t i = 0; i < std::size_t(w) * h; ++i)
        for (int k = 0; k < 3; ++k) rgb[i * 3 + k] = rgba[i * 4 + k];
    const bool ok = std::fwrite(rgb.data(), 1, rgb.size(), f) == rgb.size();
    std::fclose(f);
    return ok;
}

// Area-average downsample (or bilinear upsample) of RGB into RGBA.
std::vector<std::uint8_t> resample(const Image& im, int x0, int w, int h)
{
    std::vector<std::uint8_t> out(std::size_t(w) * h * 4, 255);
    const double sx = double(im.w) / w, sy = double(im.h) / h;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const double ax = x * sx, bx = (x + 1) * sx, ay = y * sy, by = (y + 1) * sy;
            double acc[3] = {0, 0, 0}, wsum = 0;
            for (int yy = int(ay); yy < std::min(im.h, int(std::ceil(by))); ++yy)
                for (int xx = int(ax); xx < std::min(im.w, int(std::ceil(bx))); ++xx) {
                    const double wx = std::min(bx, xx + 1.0) - std::max(ax, double(xx));
                    const double wy = std::min(by, yy + 1.0) - std::max(ay, double(yy));
                    const double wgt = std::max(0.0, wx) * std::max(0.0, wy);
                    for (int k = 0; k < 3; ++k) acc[k] += wgt * im.rgb[(std::size_t(yy) * im.w + xx) * 3 + k];
                    wsum += wgt;
                }
            for (int k = 0; k < 3; ++k)
                out[(std::size_t(y) * w + x) * 4 + k] = std::uint8_t(std::lround(acc[k] / std::max(wsum, 1e-9)));
        }
    (void)x0;
    return out;
}

std::vector<mosaic::Lab> palette_of(rvfx::Engine& engine, const std::vector<std::uint8_t>& rgba, int w, int h)
{
    rvfx::FrameView frame;
    frame.data = rgba.data();
    frame.width = std::uint32_t(w);
    frame.height = std::uint32_t(h);
    frame.stride = w * 4;
    frame.format = rvfx::PixelFormat::RGBA;
    engine.process(frame);
    std::vector<mosaic::Lab> out;
    for (const auto& p : engine.palette()) out.push_back({p.l, p.a, p.b});
    return out;
}

void diagnostic(const mosaic::Mosaic& m, std::vector<std::uint8_t>& out)
{
    const auto& f = m.fields();
    out.assign(std::size_t(f.width) * f.height * 4, 255);
    static const std::uint8_t tint[4][3] = {{60, 60, 90}, {50, 85, 60}, {95, 70, 45}, {90, 45, 80}};
    for (std::size_t i = 0; i < std::size_t(f.width) * f.height; ++i) {
        const float s = 0.35f + 0.4f * f.structure[i];
        for (int k = 0; k < 3; ++k) out[i * 4 + k] = std::uint8_t(tint[f.level[i]][k] * s * 2);
        if (f.edge[i]) { out[i * 4] = 255; out[i * 4 + 1] = 60; out[i * 4 + 2] = 40; }
    }
    for (const auto& t : m.tiles()) {
        const int x = int(std::lround(t.x)), y = int(std::lround(t.y));
        std::uint8_t c[3] = {80, 220, 255};
        if (t.outline) { c[0] = 255; c[1] = 230; c[2] = 40; }
        else if (t.row == 0xffff) { c[0] = 255; c[1] = 60; c[2] = 220; }
        for (int dy = 0; dy <= 0; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                const float px = t.x + dx * t.c, py = t.y + dx * t.s;
                const int xi = int(std::lround(px)), yi = int(std::lround(py)) + dy;
                if (xi < 0 || yi < 0 || xi >= f.width || yi >= f.height) continue;
                for (int k = 0; k < 3; ++k) out[(std::size_t(yi) * f.width + xi) * 4 + k] = c[k];
            }
        (void)x; (void)y;
    }
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s in.ppm out.ppm [options]\n", argv[0]);
        return 64;
    }
    mosaic::Config config;
    mosaic::RenderParams params;
    int analysis_width = 960, lattice = 1024, palette_colors = 24, frames = 1;
    float shift = 0, family = 1.0f;
    int warmup = 4, bench = 0, palette_width = 320;
    bool meyer = true;
    const char* diag = nullptr;
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&] { return i + 1 < argc ? argv[++i] : "0"; };
        if (a == "--tiles") config.target_tiles = std::uint32_t(std::atoi(next()));
        else if (a == "--classes") config.size_classes = std::atoi(next());
        else if (a == "--ratio") config.size_ratio = float(std::atof(next()));
        else if (a == "--palette") palette_colors = std::atoi(next());
        else if (a == "--analysis-width") analysis_width = std::atoi(next());
        else if (a == "--lattice") lattice = std::atoi(next());
        else if (a == "--no-meyer") meyer = false;
        else if (a == "--outline") config.outline = float(std::atof(next()));
        else if (a == "--contour-rows") config.contour_rows = std::atoi(next());
        else if (a == "--edge") config.edge_threshold = float(std::atof(next()));
        else if (a == "--entropy") { config.entropy_low = float(std::atof(next())); config.entropy_high = float(std::atof(next())); }
        else if (a == "--time") params.time = float(std::atof(next()));
        else if (a == "--grout") params.grout_px = float(std::atof(next()));
        else if (a == "--glint") params.glint = float(std::atof(next()));
        else if (a == "--facet") params.facet = float(std::atof(next()));
        else if (a == "--diag") diag = next();
        else if (a == "--family") family = float(std::atof(next()));
        else if (a == "--warmup") warmup = std::atoi(next());
        else if (a == "--shininess") params.shininess = float(std::atof(next()));
        else if (a == "--frames") frames = std::atoi(next());
        else if (a == "--bench") bench = std::atoi(next());
        else if (a == "--palette-width") palette_width = std::atoi(next());
        else if (a == "--shift") shift = float(std::atof(next()));
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 64; }
    }
    Image im;
    if (!read_ppm(argv[1], im)) { std::fprintf(stderr, "cannot read %s\n", argv[1]); return 66; }
    const int aw = std::min(analysis_width, im.w);
    const int ah = std::max(1, int(std::lround(double(im.h) * aw / im.w)));

    mosaic::Mosaic m;
    m.set_config(config);
    mosaic::Cartoon cartoon;
    rvfx::Config pc;
    pc.posterize_only = true;
    pc.glyph_layer = false;
    pc.palette_colors = std::uint32_t(std::max(2, palette_colors));
    pc.trace_width = std::uint32_t(std::min(aw, palette_width));
    pc.family_priority = family;
    rvfx::Engine engine(pc);
    params.scale = float(im.w) / aw;
    std::vector<std::uint8_t> out(std::size_t(im.w) * im.h * 4);

    for (int frame = 0; frame < frames; ++frame) {
        Image moved = im;
        if (frame > 0 && shift != 0) {
            // Synthetic pan of the right half: the left half stays still.
            const int dx = int(std::lround(shift * frame));
            for (int y = 0; y < im.h; ++y)
                for (int x = im.w / 2; x < im.w; ++x) {
                    const int sx = std::clamp(x - dx, im.w / 2, im.w - 1);
                    for (int k = 0; k < 3; ++k)
                        moved.rgb[(std::size_t(y) * im.w + x) * 3 + k] = im.rgb[(std::size_t(y) * im.w + sx) * 3 + k];
                }
        }
        const auto t0 = std::chrono::steady_clock::now();
        const auto rgba = resample(moved, 0, aw, ah);
        const auto t1 = std::chrono::steady_clock::now();
        std::vector<float> structure;
        if (meyer && !cartoon.run(rgba.data(), aw, ah, std::size_t(aw) * 4, lattice, 0, structure)) {
            std::fprintf(stderr, "Meyer split failed\n");
            return 70;
        }
        const auto t2 = std::chrono::steady_clock::now();
        std::vector<mosaic::Lab> palette;
        if (palette_colors > 0)
            for (int k = 0; k < (frame ? 1 : std::max(1, warmup)); ++k) palette = palette_of(engine, rgba, aw, ah);
        if (frame == 0 && std::getenv("MOSAIC_PRINT_PALETTE"))
            for (const auto& p : engine.palette())
                std::printf("palette L=%.3f a=%.3f b=%.3f rgb=%d,%d,%d\n", p.l, p.a, p.b, p.r, p.g, p.blue);
        const auto t3 = std::chrono::steady_clock::now();
        const auto& st = m.process(rgba.data(), aw, ah, std::size_t(aw) * 4,
                                   meyer ? structure.data() : nullptr, palette);
        const auto t4 = std::chrono::steady_clock::now();
        params.time = params.time + (frame ? 1.f / 30.f : 0.f);
        mosaic::render(m, params, out.data(), im.w, im.h, std::size_t(im.w) * 4);
        const auto t5 = std::chrono::steady_clock::now();
        auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        std::printf("{\"frame\":%d,\"analysis\":[%d,%d],\"output\":[%d,%d],\"tiles\":%u,\"added\":%u,"
                    "\"removed\":%u,\"recolored\":%u,\"dirty_blocks\":%u,\"blocks\":%u,\"rebuilt\":%s,"
                    "\"edge_pixels\":%u,\"largest_tile_px\":%.2f,\"ms\":{\"resample\":%.2f,\"meyer\":%.2f,"
                    "\"palette\":%.2f,\"geometry\":%.2f,\"nucleation\":%.2f,\"color\":%.2f,\"process\":%.2f,"
                    "\"render_cpu\":%.2f}}\n",
                    frame, aw, ah, im.w, im.h, st.tiles, st.added, st.removed, st.recolored, st.dirty_blocks,
                    st.blocks, st.rebuilt ? "true" : "false", st.edge_pixels, st.tile_scale, ms(t0, t1),
                    ms(t1, t2), ms(t2, t3), st.analysis_ms, st.nucleation_ms, st.color_ms, ms(t3, t4),
                    ms(t4, t5));
        if (frame == 0) {
            std::vector<int> classes(4, 0);
            for (auto l : m.fields().level) ++classes[l];
            std::printf("{\"class_pixels\":[%d,%d,%d,%d]}\n", classes[0], classes[1], classes[2], classes[3]);
        }
        if (frames > 1) {
            char path[1024];
            std::snprintf(path, sizeof path, "%s.%03d.ppm", argv[2], frame);
            write_ppm(path, im.w, im.h, out);
        }
    }
    if (bench > 0) {
        // Medians of the realtime stages on the last frame's input.
        const auto rgba = resample(im, 0, aw, ah);
        std::vector<float> structure;
        auto median = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
        auto time = [&](auto&& fn) {
            std::vector<double> v;
            for (int r = 0; r < bench; ++r) {
                const auto a = std::chrono::steady_clock::now();
                fn();
                v.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count());
            }
            return median(v);
        };
        const double meyer_ms = time([&] { cartoon.run(rgba.data(), aw, ah, std::size_t(aw) * 4, lattice, 0, structure); });
        std::vector<mosaic::Lab> palette;
        const double palette_ms = time([&] { palette = palette_of(engine, rgba, aw, ah); });
        mosaic::Mosaic b;
        b.set_config(config);
        const double rebuild_ms = time([&] { b.reset(); b.process(rgba.data(), aw, ah, std::size_t(aw) * 4, structure.data(), palette); });
        const double steady_ms = time([&] { b.process(rgba.data(), aw, ah, std::size_t(aw) * 4, structure.data(), palette); });
        bool wants = false;
        const double prefilter_ms = time([&] { wants = b.wants_structure(rgba.data(), aw, ah, std::size_t(aw) * 4); });
        const double colors_ms = time([&] { b.update_colors(rgba.data(), aw, ah, std::size_t(aw) * 4, palette); });
        mosaic::GpuPack pk;
        const double pack_ms = time([&] { mosaic::pack(b, pk); });
        std::printf("{\"bench\":{\"repeats\":%d,\"tiles\":%zu,\"max_cell\":%u,\"meyer_ms\":%.2f,\"palette_ms\":%.2f,"
                    "\"rebuild_ms\":%.2f,\"structure_unchanged_ms\":%.2f,\"prefilter_ms\":%.3f,\"prefilter_wakes\":%s,"
                    "\"update_colors_ms\":%.3f,\"pack_ms\":%.3f,\"rebuild_geometry_ms\":%.2f,\"rebuild_nucleation_ms\":%.2f}}\n",
                    bench, b.tiles().size(), pk.max_cell, meyer_ms, palette_ms, rebuild_ms, steady_ms, prefilter_ms,
                    wants ? "true" : "false", colors_ms, pack_ms, 0.0, 0.0);
        b.reset();
        const auto& st = b.process(rgba.data(), aw, ah, std::size_t(aw) * 4, structure.data(), palette);
        std::printf("{\"rebuild_split_ms\":{\"geometry\":%.2f,\"nucleation\":%.2f,\"color_grid\":%.2f,"
                    "\"entropy\":%.2f,\"edges\":%.2f,\"unused\":%.2f,\"fragments\":%.2f,\"column_sites\":%.2f,\"distance_orientation\":%.2f}}\n",
                    st.analysis_ms, st.nucleation_ms, st.color_ms, st.stage_ms[0], st.stage_ms[1], st.stage_ms[2],
                    st.stage_ms[3], st.stage_ms[4], st.stage_ms[5]);
    }
    if (!write_ppm(argv[2], im.w, im.h, out)) return 73;
    if (diag) {
        std::vector<std::uint8_t> d;
        diagnostic(m, d);
        write_ppm(diag, m.fields().width, m.fields().height, d);
    }
    return 0;
}
