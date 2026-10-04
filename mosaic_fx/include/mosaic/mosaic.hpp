#pragma once
// Deterministic tessellated-mosaic core.
//
// Pipeline (all direct, single-pass constructions; no relaxation or solver
// iterations):
//   1. Oklab source, structure luma (normally the Meyer cartoon), local
//      color entropy on 8x8 blocks.
//   2. Edges: gradient of structure + chroma, non-maximum suppressed, short
//      fragments removed.
//   3. Exact Euclidean distance/feature transform to those edges (and the
//      frame).  Level sets of distance/size are the contour rows
//      ("andamento"); the feature direction orients each tile.
//   4. Entropy picks one of a few tile sizes; one global scale is chosen so
//      the expected tile count meets the budget.
//   5. Nucleation: row-centreline pixels in row order (outline row first),
//      then a raster gap fill, accepted greedily against a spacing rule.
//   6. Backfill (render): each pixel belongs to the oriented-square nearest
//      tile; grout where no square covers it or near a cell boundary.
//
// Temporal and cost model: tiles persist.  wants_structure() is an 8x8
// block-luma prefilter with the global (median) shift removed, so exposure
// changes only recolor.  Only woken blocks are tested structurally, and only
// dirty blocks are re-analysed and re-nucleated; every field persists
// full-frame, and all geometry stages run on the dirty runs alone.  A full
// rebuild is a pure function of the frame.

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace mosaic {

struct Lab { float l = 0, a = 0, b = 0; };

struct Config {
    std::uint32_t target_tiles = 15000;
    int size_classes = 3;            // distinct tile sizes, 1..4
    float size_ratio = 1.5f;         // linear size ratio between classes
    float entropy_low = 0.30f;       // class thresholds span [low, high]
    float entropy_high = 0.65f;
    float edge_threshold = 0.035f;   // structure/chroma gradient per pixel
    float chroma_edge_weight = 1.0f;
    int edge_min_length = 8;         // drop edge fragments shorter than this
    bool frame_edges = true;         // rows also follow the frame border
    int contour_rows = 0;            // 0: rows everywhere; N: regular grid beyond N rows
    float outline = 0.6f;            // 0: no outline row; 1: darkest outline tiles
    float outline_contrast = 4.0f;   // edge strength (x threshold) for full outline
    float spacing = 0.92f;           // minimum seed separation / tile size
    float elongation = 1.08f;        // tile length along the row / width
    float shape_jitter = 0.08f;
    float angle_jitter = 0.06f;      // radians
    float lightness_jitter = 0.035f;
    float chroma_jitter = 0.008f;
    float recolor_threshold = 0.035f;     // Oklab distance before a tile recolors
    // Block mean |structure change - block mean change| that re-nucleates.
    float restructure_threshold = 0.035f;
    float prefilter_luma = 5.0f;          // 8x8 block luma change (8-bit, median shift removed)
    float scene_cut_fraction = 0.55f;     // dirty fraction that forces a full rebuild
};

struct Tile {
    float x = 0, y = 0;          // centre, analysis pixels
    float c = 1, s = 0;          // row (tangent) direction cos/sin
    float half_t = 1, half_n = 1;
    float tilt_x = 0, tilt_y = 0; // facet normal perturbation (glint)
    float phase = 0;             // twinkle phase in [0,1)
    std::uint8_t size_class = 0; // 0 = largest
    std::uint8_t outline = 0;    // edge strength of an outline-row tile, 0..255
    std::uint16_t row = 0;       // contour row index, 0xffff for gap fill
    std::uint32_t key = 0;       // position hash: jitter is history independent
    Lab measured;                // color the tile was last committed to
    Lab color;                   // displayed color
    float rgb[3] = {0, 0, 0};    // displayed color, gamma-encoded sRGB
};

struct Stats {
    std::uint32_t tiles = 0, added = 0, removed = 0, recolored = 0;
    std::uint32_t dirty_blocks = 0, blocks = 0, woken_blocks = 0;
    std::uint32_t edge_pixels = 0;
    bool rebuilt = false;
    bool geometry = false;      // tile placement changed this call
    float tile_scale = 0;       // pixel size of the largest class
    std::uint32_t max_cell_tiles = 0;
    double analysis_ms = 0, nucleation_ms = 0, color_ms = 0;
    // Geometry split: lab, entropy, gradient+NMS, fragments, distance, orientation.
    double stage_ms[6] = {0, 0, 0, 0, 0, 0};
};

// Persistent per-pixel state: 22 bytes/pixel.  Everything else the
// geometry needs is per-run scratch or a few-row ring.
struct Fields {
    int width = 0, height = 0;
    std::vector<float> structure;     // [0,1]
    std::vector<std::uint8_t> edge;   // 0, or 1..126: edge with its doubled-angle normal code
    std::vector<std::uint8_t> outline;// outline weight at edge pixels, 0..255
    std::vector<std::uint8_t> level;  // size class per pixel (0 = largest)
    std::vector<float> distance;      // to nearest edge/frame
    std::vector<std::int16_t> c2, s2; // smoothed doubled-angle normal x 32767, unnormalized
    std::vector<std::uint8_t> from_frame;
};

// Uniform bucket grid over tile centres (CSR).  A pixel's owner is always in
// its own or an adjacent cell because cell >= largest tile size.
struct Grid {
    float cell = 1;
    int cols = 0, rows = 0;
    std::vector<std::uint32_t> start, items;
};

// Horizontal run of dirty 16-pixel blocks: [x0,x1) x [y0,y1) in pixels.
struct Run { int x0, y0, x1, y1; };

class Mosaic {
public:
    void set_config(const Config& config) { config_ = config; }
    const Config& config() const { return config_; }
    void reset();
    // rgba: straight-alpha RGBA8 at analysis resolution.  structure: optional
    // luma in [0,1] at the same resolution (Meyer cartoon); nullptr uses the
    // Oklab lightness.  palette: optional Oklab nodes for tile colors.
    const Stats& process(const std::uint8_t* rgba, int width, int height,
                         std::size_t stride, const float* structure,
                         const std::vector<Lab>& palette);
    // Same, taking the structure buffer by swap (values already in [0,1]);
    // on return `structure` holds the previous buffer for reuse.
    const Stats& process_swap(const std::uint8_t* rgba, int width, int height,
                              std::size_t stride, std::vector<float>& structure,
                              const std::vector<Lab>& palette);
    // Prefilter: true when tiling is absent, the size changed, or some 8x8
    // block's mean luma moved (after removing the median shift) by more than
    // prefilter_luma since the frame its block was last examined.  The result
    // is cached for the next process()/update_colors() call.
    bool wants_structure(const std::uint8_t* rgba, int width, int height, std::size_t stride);
    // Colors only: tiles, grid and geometry stay fixed.  Requires a previous
    // process() at the same size.  keep_pending=true (geometry deferred by a
    // rate limit) leaves woken blocks pending for the next process().
    const Stats& update_colors(const std::uint8_t* rgba, int width, int height,
                               std::size_t stride, const std::vector<Lab>& palette,
                               bool keep_pending = false);
    const std::vector<Tile>& tiles() const { return tiles_; }
    const Grid& grid() const { return grid_; }
    const Fields& fields() const { return f_; }
    const Stats& stats() const { return stats_; }
    // 16x16 block mask re-analysed by the last process() call (row-major).
    const std::vector<std::uint8_t>& dirty_blocks() const { return dirty_; }

private:
    const Stats& process_impl(const std::uint8_t* rgba, std::size_t stride, const std::vector<Lab>& palette);
    void prefilter(const std::uint8_t* rgba, std::size_t stride);
    void build_runs(bool full);
    void analyze_geometry(const std::uint8_t* rgba, std::size_t stride);
    void nucleate(bool full);
    void measure_and_color(const std::uint8_t* rgba, std::size_t stride, const std::vector<Lab>& palette);
    void build_grid();
    Tile make_tile(float x, float y, int level, int row, float outline, bool regular) const;

    Config config_;
    Fields f_;
    std::vector<Tile> tiles_;
    Grid grid_;
    Stats stats_;
    std::vector<float> reference_;   // structure the current tiles were laid on
    std::vector<float> size_px_;     // tile size per class
    // Prefilter: per 8x8 block, mean luma and two chroma differences (1/16
    // codes, three ints per block) when the block was last examined; the
    // current frame's values and wake mask.
    std::vector<std::int32_t> signature_, now_signature_, delta_;
    std::vector<std::uint8_t> wake_;
    bool prefiltered_ = false;
    std::uint32_t woken_ = 0;
    // Region bookkeeping and reusable scratch (sized by the dirty runs, not
    // the frame, except the int16 column sites).
    std::vector<std::uint8_t> dirty_, candidate_;
    std::vector<Run> runs_;
    std::vector<float> ab_raw_, ab_h_, ab_blur_, grad_;
    std::vector<float> block_raw_, block_tmp_;
    std::vector<std::int16_t> col_site_, last_;
    std::vector<int> v_, stack_, component_;
    std::vector<double> z_;
    std::vector<float> ring_h_, ring_p_, row_c_, row_s_;
    std::vector<std::uint8_t> rows_, blocked_, histogram_;
    std::vector<std::pair<int, int>> candidates_, sorted_;
    std::vector<int> row_count_, head_, next_;
    const std::uint8_t* prefilter_frame_ = nullptr;
    bool full_measure_ = false;   // global (exposure) shift: measure every tile
    std::uint32_t max_cell_ = 0;
    // Color cache: palette and color settings the display colors were built from.
    std::vector<Lab> color_palette_;
    float color_outline_ = -1, color_lj_ = -1, color_cj_ = -1;
};

struct RenderParams {
    float scale = 1;             // output pixels per analysis pixel
    float grout_px = 1.3f;       // grout width in output pixels
    float grout[3] = {0.22f, 0.205f, 0.18f}; // gamma-encoded sRGB
    float bevel_px = 1.8f;
    float bevel = 0.30f;         // edge normal tilt
    float facet = 0.24f;         // random facet tilt: sets how many tiles glint
    float diffuse = 0.35f;
    float glint = 1.1f;
    float shininess = 90.f;
    float metal = 0.55f;         // glint tint by tile color
    float light_azimuth = 2.3f;  // radians, image coordinates (y down)
    float light_elevation = 0.95f;
    float sweep = 0.25f;         // azimuth radians per second
    float twinkle = 0.35f;       // per-tile flicker depth
    float time = 0;
};

// Light and half vectors for a render time; shared by CPU and GPU paths.
void light_vectors(const RenderParams& params, float light[3], float half[3]);

// CPU reference renderer (the OBS shader mirrors it).  out is RGBA8.
void render(const std::vector<Tile>& tiles, const Grid& grid, const RenderParams& params,
            std::uint8_t* out, int width, int height, std::size_t stride);
inline void render(const Mosaic& m, const RenderParams& p, std::uint8_t* out, int w, int h, std::size_t stride)
{
    render(m.tiles(), m.grid(), p, out, w, h, stride);
}

// GPU upload layout, RGBA16 UNORM (OBS 32.2.1 Metal mishandles 128-bit
// uploads).  Tiles are stored in grid-cell order.  Geometry (uploaded only
// when placement changes), two texels per tile:
//   g0 = (x / analysis_w, y / analysis_h, c/2 + 1/2, s/2 + 1/2)
//   g1 = (half_t / half_scale, half_n / half_scale, tilt_x/2 + 1/2, tilt_y/2 + 1/2)
// Colors, one texel per tile: (r, g, b, twinkle phase).
// Cell texel = (start & 0xffff, start >> 16, count, 0) in raw 16-bit units.
struct GpuPack {
    static constexpr int kTilesPerRow = 1024;
    static constexpr std::uint32_t kMaxTiles = 65534; // index 65535 marks grout in the GPU owner cache
    int rows = 1;                // texture rows of kTilesPerRow tiles
    int cells_w = 1, cells_h = 1;
    float analysis_w = 1, analysis_h = 1, cell = 1, half_scale = 1;
    std::uint32_t count = 0, max_cell = 0;
    std::vector<std::uint16_t> geometry, colors, cells;
    std::vector<std::uint32_t> order; // tile index for each packed slot
};
// geometry=false repacks colors only, in the existing slot order.
void pack(const Mosaic& mosaic, GpuPack& out, bool geometry = true);
// Decode exactly what the shader sees (for CPU/GPU equivalence checks).
void unpack(const GpuPack& in, std::vector<Tile>& tiles, Grid& grid);

Lab srgb8_to_oklab(std::uint8_t r, std::uint8_t g, std::uint8_t b);
void oklab_to_srgb(const Lab& lab, float rgb[3]); // gamma-encoded, clipped [0,1]

} // namespace mosaic
