#include "rvfx/engine.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>

namespace rvfx {
namespace {

using Clock = std::chrono::steady_clock;

double millis(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

float clamp01(float x) { return std::max(0.0f, std::min(1.0f, x)); }

std::uint64_t mix64(std::uint64_t x) {
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

struct Lab {
    float l, a, b, alpha;
};

struct EdgeState {
    bool active = false;
    std::uint32_t age = 0;
    std::uint16_t c0 = 0, c1 = 0;
    float phase = 0.0f;
};

struct GlyphParticle {
    bool alive = false;
    bool arcing = false;
    float x = 0.0f, y = 0.0f, previous_x = 0.0f, previous_y = 0.0f;
    float vx = 0.0f, vy = 0.0f, angular_velocity = 0.0f;
    float life = 0.0f, maximum_life = 1.0f;
    std::uint8_t glyph = 0;
};

struct Yuv8 { std::uint8_t y, u, v; };

std::uint8_t byte(float x) {
    return static_cast<std::uint8_t>(std::max(0, std::min(255,
        static_cast<int>(std::lround(x)))));
}

} // namespace

struct Engine::Impl {
    Config cfg;
    std::array<float, 256> linear{};
    std::vector<PaletteColor> palette;
    std::vector<Lab> sample_scratch;
    std::vector<float> sample_detail;
    std::vector<float> sample_importance;
    std::vector<std::uint16_t> sample_owner;
    std::vector<Lab> palette_centroids;
    std::vector<Lab> palette_parents;
    std::vector<Lab> grid_lab;
    std::vector<std::uint64_t> grid_token;
    std::vector<std::uint16_t> labels;
    std::vector<EdgeState> vertical, horizontal;
    std::vector<TraceSegment> segments;
    std::vector<DrawCommand> commands;
    std::vector<GlyphParticle> particles;
    std::vector<std::uint32_t> visit_order;
    FrameStats stats;
    std::uint32_t gw = 0, gh = 0, source_w = 0, source_h = 0;
    std::uint64_t frame_number = 0;
    std::uint64_t topology_signature = 0;
    std::uint64_t scheduled_signature = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t visit_cycle = 0;
    std::size_t visit_cursor = 0;
    bool initialized = false;

    explicit Impl(Config c) : cfg(sanitize(c)) {
        for (std::size_t i = 0; i < linear.size(); ++i) {
            const float s = static_cast<float>(i) / 255.0f;
            linear[i] = s <= 0.04045f ? s / 12.92f
                                      : std::pow((s + 0.055f) / 1.055f, 2.4f);
        }
        reserve_outputs();
    }

    static Config sanitize(Config c) {
        c.trace_width = std::max(32u, std::min(1920u, c.trace_width));
        c.palette_colors = std::max(2u, std::min(64u, c.palette_colors));
        c.palette_samples = std::max(c.palette_colors * 8u,
                                     std::min(65536u, c.palette_samples));
        c.segments_per_frame = std::max(1u, std::min(65536u, c.segments_per_frame));
        c.detail_priority = std::max(0.0f, std::min(8.0f, c.detail_priority));
        c.population_exponent = std::max(0.1f, std::min(1.0f, c.population_exponent));
        c.lightness_weight = std::max(0.0f, std::min(4.0f, c.lightness_weight));
        c.chroma_weight = std::max(0.0f, std::min(4.0f, c.chroma_weight));
        c.hue_weight = std::max(0.0f, std::min(4.0f, c.hue_weight));
        c.alpha_weight = std::max(0.0f, std::min(4.0f, c.alpha_weight));
        c.node_separation = std::max(0.0f, std::min(2.5f, c.node_separation));
        c.minimum_leaf = std::max(1u, std::min(256u, c.minimum_leaf));
        c.bifurcation_refinement = std::min(12u, c.bifurcation_refinement);
        c.prior_learning_rate = std::max(0.001f, std::min(1.0f, c.prior_learning_rate));
        c.trace_speed = std::max(0.001f, std::min(1.0f, c.trace_speed));
        c.trace_persistence = std::max(0.0f, std::min(0.98f, c.trace_persistence));
        c.glow = clamp01(c.glow);
        c.glyph_particles = std::min(8192u, c.glyph_particles);
        c.glyph_spawn_per_frame = std::min(512u, c.glyph_spawn_per_frame);
        c.frame_budget_ms = std::max(1.0f, c.frame_budget_ms);
        return c;
    }

    void reserve_outputs() {
        commands.reserve(static_cast<std::size_t>(cfg.segments_per_frame) * 3u+
                         static_cast<std::size_t>(cfg.glyph_particles) * 2u);
        segments.reserve(static_cast<std::size_t>(cfg.trace_width) * 4u);
        visit_order.reserve(static_cast<std::size_t>(cfg.trace_width) * 4u);
        if (particles.size() != cfg.glyph_particles)
            particles.resize(cfg.glyph_particles);
    }

    Lab read_lab(const FrameView& f, std::uint32_t x, std::uint32_t y) const {
        std::uint8_t r8, g8, b8, alpha8 = 255;
        if (f.format == PixelFormat::RGBA || f.format == PixelFormat::BGRA) {
            const auto* p = f.data + static_cast<std::ptrdiff_t>(y) * f.stride + 4u * x;
            r8 = f.format == PixelFormat::RGBA ? p[0] : p[2];
            g8 = p[1]; b8 = f.format == PixelFormat::RGBA ? p[2] : p[0]; alpha8 = p[3];
        } else {
            const auto y8 = f.data[static_cast<std::ptrdiff_t>(y)*f.stride+x];
            std::uint8_t u8, v8;
            if (f.format == PixelFormat::NV12) {
                const auto* uv = f.plane1 + static_cast<std::ptrdiff_t>(y/2)*f.stride1 + 2u*(x/2);
                u8=uv[0]; v8=uv[1];
            } else {
                u8=f.plane1[static_cast<std::ptrdiff_t>(y/2)*f.stride1+x/2];
                v8=f.plane2[static_cast<std::ptrdiff_t>(y/2)*f.stride2+x/2];
            }
            const float yy=clamp01(f.full_range ? y8/255.0f : (y8-16.0f)/219.0f);
            const float scale=f.full_range?255.0f:224.0f;
            const float u=(u8-128.0f)/scale, v=(v8-128.0f)/scale;
            r8=byte(255.0f*clamp01(yy+1.5748f*v));
            g8=byte(255.0f*clamp01(yy-0.187324f*u-0.468124f*v));
            b8=byte(255.0f*clamp01(yy+1.8556f*u));
        }
        const float r = linear[r8], g = linear[g8], b = linear[b8];
        const float ll = 0.4122214708f*r + 0.5363325363f*g + 0.0514459929f*b;
        const float mm = 0.2119034982f*r + 0.6806995451f*g + 0.1073969566f*b;
        const float ss = 0.0883024619f*r + 0.2817188376f*g + 0.6299787005f*b;
        const float l = std::cbrt(std::max(0.0f, ll));
        const float m = std::cbrt(std::max(0.0f, mm));
        const float s = std::cbrt(std::max(0.0f, ss));
        return {0.2104542553f*l + 0.7936177850f*m - 0.0040720468f*s,
                1.9779984951f*l - 2.4285922050f*m + 0.4505937099f*s,
                0.0259040371f*l + 0.7827717662f*m - 0.8086757660f*s,
                static_cast<float>(alpha8) / 255.0f};
    }

    static Yuv8 rgb_to_yuv(std::uint8_t r8, std::uint8_t g8, std::uint8_t b8,
                           bool full_range) {
        const float r=r8/255.0f, g=g8/255.0f, b=b8/255.0f;
        const float y=0.2126f*r+0.7152f*g+0.0722f*b;
        const float u=(b-y)/1.8556f, v=(r-y)/1.5748f;
        const float chroma=full_range?255.0f:224.0f;
        return {byte(full_range?255.0f*y:16.0f+219.0f*y),
                byte(128.0f+chroma*u), byte(128.0f+chroma*v)};
    }

    float distance2(const Lab& x, const PaletteColor& p) const {
        const float sample_c=std::hypot(x.a,x.b),center_c=std::hypot(p.a,p.b);
        const float dl=cfg.lightness_weight*(x.l-p.l);
        const float dc=cfg.chroma_weight*(sample_c-center_c);
        float hue_term=0.0f;
        if(sample_c>1e-8f&&center_c>1e-8f){
            const float cosine=std::clamp((x.a*p.a+x.b*p.b)/(sample_c*center_c),-1.0f,1.0f);
            hue_term=2.0f*cfg.hue_weight*cfg.hue_weight*sample_c*center_c*(1.0f-cosine);
        }
        const float da=cfg.alpha_weight*(x.alpha-p.alpha);
        return dl*dl+dc*dc+hue_term+da*da;
    }

    static PaletteColor make_color(const Lab& c) {
        Lab mapped=c;mapped.l=clamp01(mapped.l);mapped.alpha=clamp01(mapped.alpha);
        const auto linear_rgb=[](const Lab& value){
            const float l_=value.l+0.3963377774f*value.a+0.2158037573f*value.b;
            const float m_=value.l-0.1055613458f*value.a-0.0638541728f*value.b;
            const float s_=value.l-0.0894841775f*value.a-1.2914855480f*value.b;
            const float l=l_*l_*l_,m=m_*m_*m_,s=s_*s_*s_;
            return std::array<float,3>{4.0767416621f*l-3.3077115913f*m+0.2309699292f*s,
                -1.2684380046f*l+2.6097574011f*m-0.3413193965f*s,
                -0.0041960863f*l-0.7034186147f*m+1.7076147010f*s};
        };
        const auto in_gamut=[](const std::array<float,3>& rgb){
            return rgb[0]>=0.0f&&rgb[0]<=1.0f&&rgb[1]>=0.0f&&rgb[1]<=1.0f&&rgb[2]>=0.0f&&rgb[2]<=1.0f;
        };
        auto rgb=linear_rgb(mapped);
        const float chroma=std::hypot(mapped.a,mapped.b);
        if(!in_gamut(rgb)&&chroma>1e-8f){
            const float ua=mapped.a/chroma,ub=mapped.b/chroma;float low=0.0f,high=chroma;
            for(int iteration=0;iteration<14;++iteration){
                const float middle=.5f*(low+high);Lab candidate=mapped;
                candidate.a=ua*middle;candidate.b=ub*middle;
                if(in_gamut(linear_rgb(candidate)))low=middle;else high=middle;
            }
            mapped.a=ua*low;mapped.b=ub*low;rgb=linear_rgb(mapped);
        }
        PaletteColor out;
        out.l=mapped.l;out.a=mapped.a;out.b=mapped.b;out.alpha=mapped.alpha;
        const auto gamma = [](float v) {
            v = clamp01(v);
            return v <= 0.0031308f ? 12.92f*v
                                  : 1.055f*std::pow(v, 1.0f/2.4f) - 0.055f;
        };
        out.r=byte(255.0f*gamma(rgb[0]));out.g=byte(255.0f*gamma(rgb[1]));
        out.blue=byte(255.0f*gamma(rgb[2]));out.opacity=byte(255.0f*out.alpha);
        return out;
    }

    void ensure_shape(const FrameView& f) {
        const auto new_gw = std::min(f.width, cfg.trace_width);
        const auto new_gh = std::max(1u, static_cast<std::uint32_t>(
            std::lround(static_cast<double>(f.height) * new_gw / f.width)));
        if (new_gw == gw && new_gh == gh && source_w == f.width && source_h == f.height)
            return;
        gw = new_gw; gh = new_gh; source_w = f.width; source_h = f.height;
        grid_lab.resize(static_cast<std::size_t>(gw) * gh);
        grid_token.assign(grid_lab.size(),std::numeric_limits<std::uint64_t>::max());
        labels.resize(grid_lab.size());
        vertical.assign(gw > 1 ? static_cast<std::size_t>(gw - 1) * gh : 0, {});
        horizontal.assign(gh > 1 ? static_cast<std::size_t>(gw) * (gh - 1) : 0, {});
        segments.clear();
    }

    void fill_samples(const FrameView& f) {
        const auto count = std::min<std::size_t>(cfg.palette_samples,
                                                 static_cast<std::size_t>(f.width) * f.height);
        sample_scratch.resize(count);
        sample_detail.resize(count);
        sample_importance.resize(count);
        sample_owner.resize(count);
        const std::uint64_t total = static_cast<std::uint64_t>(f.width) * f.height;
        const std::uint64_t offset = mix64(frame_number + 0x9e3779b97f4a7c15ULL) % total;
        for (std::size_t i = 0; i < count; ++i) {
            const auto q = (offset + (static_cast<std::uint64_t>(i) * total) / count) % total;
            const auto x = static_cast<std::uint32_t>(q % f.width);
            const auto y = static_cast<std::uint32_t>(q / f.width);
            const Lab c = read_lab(f, x, y);
            const Lab dx = read_lab(f, std::min(x+1, f.width-1), y);
            const Lab dy = read_lab(f, x, std::min(y+1, f.height-1));
            sample_scratch[i] = c;
            sample_detail[i] = std::abs(c.l-dx.l)+std::abs(c.l-dy.l) +
                0.5f*(std::abs(c.a-dx.a)+std::abs(c.b-dx.b)+
                      std::abs(c.a-dy.a)+std::abs(c.b-dy.b));
        }
        std::array<std::uint32_t,16*12*24> occupied{};
        std::vector<std::uint16_t> bins(count);
        for(std::size_t i=0;i<count;++i){
            const auto& c=sample_scratch[i];
            const auto light=static_cast<std::uint32_t>(std::clamp(c.l*15.999f,0.0f,15.0f));
            const float chroma=std::hypot(c.a,c.b);
            const auto chroma_bin=static_cast<std::uint32_t>(std::clamp(chroma/.4f*11.999f,0.0f,11.0f));
            std::uint32_t hue_bin=0;
            if(chroma>=.015f){
                constexpr float pi=3.14159265358979323846f;
                hue_bin=static_cast<std::uint32_t>(std::clamp((std::atan2(c.b,c.a)+pi)/(2*pi)*23.999f,0.0f,23.0f));
            }
            bins[i]=static_cast<std::uint16_t>((light*12u+chroma_bin)*24u+hue_bin);
            ++occupied[bins[i]];
        }
        auto sorted_detail=sample_detail;
        const auto percentile=sorted_detail.empty()?0u:9u*(sorted_detail.size()-1u)/10u;
        if(!sorted_detail.empty())std::nth_element(sorted_detail.begin(),sorted_detail.begin()+percentile,sorted_detail.end());
        const float scale=sorted_detail.empty()?1.0f:std::max(1e-6f,sorted_detail[percentile]);
        double total_weight=0.0;
        for(std::size_t i=0;i<count;++i){
            const float detail=std::clamp(sample_detail[i]/scale,0.0f,4.0f);
            const float rarity=std::pow(static_cast<float>(std::max(1u,occupied[bins[i]])),
                                        cfg.population_exponent-1.0f);
            sample_importance[i]=(1.0f+cfg.detail_priority*detail)*rarity;
            total_weight+=sample_importance[i];
        }
        const float normalizer=count?static_cast<float>(count/std::max(total_weight,1e-12)):1.0f;
        for(auto& weight:sample_importance)weight=std::clamp(weight*normalizer,.03f,30.0f);
    }

    Lab weighted_center(const std::vector<std::uint32_t>& indices) const {
        double mass=0.0,l=0.0,a=0.0,b=0.0,alpha=0.0;
        for(const auto index:indices){const double w=sample_importance[index];const auto& c=sample_scratch[index];
            mass+=w;l+=w*c.l;a+=w*c.a;b+=w*c.b;alpha+=w*c.alpha;}
        const double safe=std::max(mass,1e-15);return {static_cast<float>(l/safe),static_cast<float>(a/safe),
            static_cast<float>(b/safe),static_cast<float>(alpha/safe)};
    }

    struct SplitProposal {
        double gain=0.0;
        std::vector<std::uint32_t> left,right;
    };

    SplitProposal propose_split(const std::vector<std::uint32_t>& indices) const {
        SplitProposal best;if(indices.size()<2u*cfg.minimum_leaf)return best;
        using Coordinate=std::array<double,4>;
        std::vector<Coordinate> coordinates(indices.size());double mass=0.0,hx=0.0,hy=0.0,mean_chroma=0.0;
        for(const auto index:indices){const double w=sample_importance[index];const auto& c=sample_scratch[index];
            const double chroma=std::hypot(c.a,c.b);mass+=w;hx+=w*c.a;hy+=w*c.b;mean_chroma+=w*chroma;}
        const double center_hue=std::atan2(hy,hx);mean_chroma/=std::max(mass,1e-15);
        Coordinate center{};
        for(std::size_t i=0;i<indices.size();++i){const auto& c=sample_scratch[indices[i]];
            const double chroma=std::hypot(c.a,c.b);double hue=std::atan2(c.b,c.a)-center_hue;
            hue=std::atan2(std::sin(hue),std::cos(hue));
            coordinates[i]={cfg.lightness_weight*c.l,cfg.chroma_weight*chroma,
                cfg.hue_weight*std::sqrt(std::max(chroma*mean_chroma,1e-8))*hue,cfg.alpha_weight*c.alpha};
            const double w=sample_importance[indices[i]];for(int axis=0;axis<4;++axis)center[axis]+=w*coordinates[i][axis];
        }
        for(auto& value:center)value/=std::max(mass,1e-15);
        double old_sse=0.0;std::array<std::array<double,4>,4> covariance{};
        for(std::size_t i=0;i<indices.size();++i){const double w=sample_importance[indices[i]];Coordinate delta{};
            for(int axis=0;axis<4;++axis)delta[axis]=coordinates[i][axis]-center[axis];
            for(int row=0;row<4;++row)for(int column=0;column<4;++column)covariance[row][column]+=w*delta[row]*delta[column];
            for(const auto value:delta)old_sse+=w*value*value;}
        if(old_sse<=1e-14)return best;
        Coordinate principal{1.0,0.0,0.0,0.0};
        for(int iteration=0;iteration<12;++iteration){Coordinate next{};double norm=0.0;
            for(int row=0;row<4;++row)for(int column=0;column<4;++column)next[row]+=covariance[row][column]*principal[column];
            for(const auto value:next)norm+=value*value;norm=std::sqrt(norm);if(norm<=1e-15)break;
            for(int axis=0;axis<4;++axis)principal[axis]=next[axis]/norm;}
        std::array<Coordinate,5> directions{};directions[0]=principal;
        for(int axis=0;axis<4;++axis)directions[axis+1][axis]=1.0;
        for(const auto& direction:directions){
            std::vector<std::size_t> order(indices.size());std::iota(order.begin(),order.end(),0u);
            std::stable_sort(order.begin(),order.end(),[&](std::size_t left,std::size_t right){
                double lp=0.0,rp=0.0;for(int axis=0;axis<4;++axis){lp+=coordinates[left][axis]*direction[axis];rp+=coordinates[right][axis]*direction[axis];}
                return lp<rp;});
            double left_mass=0.0,left_norm=0.0,total_norm=0.0;Coordinate left_sum{},total_sum{};
            for(std::size_t i=0;i<indices.size();++i){const double w=sample_importance[indices[i]];double norm=0.0;
                for(int axis=0;axis<4;++axis){total_sum[axis]+=w*coordinates[i][axis];norm+=coordinates[i][axis]*coordinates[i][axis];}
                total_norm+=w*norm;}
            double candidate_sse=std::numeric_limits<double>::max();std::size_t candidate_cut=0;
            for(std::size_t position=0;position+cfg.minimum_leaf<order.size();++position){const auto i=order[position];
                const double w=sample_importance[indices[i]];double norm=0.0;left_mass+=w;
                for(int axis=0;axis<4;++axis){left_sum[axis]+=w*coordinates[i][axis];norm+=coordinates[i][axis]*coordinates[i][axis];}
                left_norm+=w*norm;const std::size_t left_count=position+1,right_count=order.size()-left_count;
                if(left_count<cfg.minimum_leaf||right_count<cfg.minimum_leaf)continue;
                const double right_mass=mass-left_mass;if(left_mass<=1e-15||right_mass<=1e-15)continue;
                double left_center_norm=0.0,right_center_norm=0.0;
                for(int axis=0;axis<4;++axis){left_center_norm+=left_sum[axis]*left_sum[axis];
                    const double right=total_sum[axis]-left_sum[axis];right_center_norm+=right*right;}
                const double sse=left_norm-left_center_norm/left_mass+(total_norm-left_norm)-right_center_norm/right_mass;
                if(sse<candidate_sse){candidate_sse=sse;candidate_cut=left_count;}
            }
            if(candidate_sse==std::numeric_limits<double>::max())continue;
            std::vector<bool> side(indices.size(),true);for(std::size_t i=0;i<candidate_cut;++i)side[order[i]]=false;
            for(std::uint32_t iteration=0;iteration<cfg.bifurcation_refinement;++iteration){
                std::array<Coordinate,2> centers{};std::array<double,2> weights{};std::array<std::size_t,2> counts{};
                for(std::size_t i=0;i<indices.size();++i){const auto group=side[i]?1u:0u;const double w=sample_importance[indices[i]];
                    weights[group]+=w;++counts[group];for(int axis=0;axis<4;++axis)centers[group][axis]+=w*coordinates[i][axis];}
                for(int group=0;group<2;++group)for(auto& value:centers[group])value/=std::max(weights[group],1e-15);
                auto updated=side;std::array<std::size_t,2> next_counts{};
                for(std::size_t i=0;i<indices.size();++i){double d0=0.0,d1=0.0;for(int axis=0;axis<4;++axis){
                    const double a=coordinates[i][axis]-centers[0][axis],b=coordinates[i][axis]-centers[1][axis];d0+=a*a;d1+=b*b;}
                    updated[i]=d1<d0;++next_counts[updated[i]?1u:0u];}
                if(next_counts[0]<cfg.minimum_leaf||next_counts[1]<cfg.minimum_leaf||updated==side)break;side.swap(updated);
            }
            std::array<Coordinate,2> centers{};std::array<double,2> weights{};
            for(std::size_t i=0;i<indices.size();++i){const auto group=side[i]?1u:0u;const double w=sample_importance[indices[i]];
                weights[group]+=w;for(int axis=0;axis<4;++axis)centers[group][axis]+=w*coordinates[i][axis];}
            for(int group=0;group<2;++group)for(auto& value:centers[group])value/=std::max(weights[group],1e-15);
            double new_sse=0.0;for(std::size_t i=0;i<indices.size();++i){const auto group=side[i]?1u:0u;const double w=sample_importance[indices[i]];
                for(int axis=0;axis<4;++axis){const double delta=coordinates[i][axis]-centers[group][axis];new_sse+=w*delta*delta;}}
            const double gain=old_sse-new_sse;if(gain<=best.gain+1e-14)continue;
            best={};best.gain=gain;best.left.reserve(indices.size());best.right.reserve(indices.size());
            for(std::size_t i=0;i<indices.size();++i)(side[i]?best.right:best.left).push_back(indices[i]);
        }
        return best;
    }

    void refresh_palette() {
        palette.resize(palette_centroids.size());
        for(std::size_t i=0;i<palette.size();++i){const auto& center=palette_centroids[i];const auto& parent=palette_parents[i];
            const Lab display{parent.l+cfg.node_separation*(center.l-parent.l),
                parent.a+cfg.node_separation*(center.a-parent.a),parent.b+cfg.node_separation*(center.b-parent.b),
                parent.alpha+cfg.node_separation*(center.alpha-parent.alpha)};
            palette[i]=make_color(display);}
    }

    void seed_palette(const std::vector<Lab>& s) {
        struct Leaf { std::vector<std::uint32_t> indices;Lab center,parent;SplitProposal proposal; };
        palette.clear();palette_centroids.clear();palette_parents.clear();if(s.empty())return;
        // Exact split searches sort several candidate projections per leaf.
        // Bound only that cold-start tree construction; the immediately
        // following centroid update still consumes the full sample budget.
        const std::size_t seed_count=std::min<std::size_t>(s.size(),1536u);
        Leaf root;root.indices.resize(seed_count);
        for(std::size_t i=0;i<seed_count;++i)root.indices[i]=static_cast<std::uint32_t>(i*s.size()/seed_count);
        root.center=weighted_center(root.indices);root.parent=root.center;root.proposal=propose_split(root.indices);
        std::vector<Leaf> leaves;leaves.push_back(std::move(root));
        while(leaves.size()<cfg.palette_colors){std::size_t choice=leaves.size();double gain=0.0;
            for(std::size_t i=0;i<leaves.size();++i)if(leaves[i].proposal.gain>gain){gain=leaves[i].proposal.gain;choice=i;}
            if(choice==leaves.size())break;Leaf parent=std::move(leaves[choice]);leaves.erase(leaves.begin()+choice);
            Leaf left,right;left.indices=std::move(parent.proposal.left);right.indices=std::move(parent.proposal.right);
            left.center=weighted_center(left.indices);right.center=weighted_center(right.indices);
            left.parent=right.parent=parent.center;left.proposal=propose_split(left.indices);right.proposal=propose_split(right.indices);
            leaves.push_back(std::move(left));leaves.push_back(std::move(right));
        }
        for(const auto& leaf:leaves){palette_centroids.push_back(leaf.center);palette_parents.push_back(leaf.parent);}
        while(palette_centroids.size()<cfg.palette_colors){const auto source=palette_centroids.size()%leaves.size();
            palette_centroids.push_back(palette_centroids[source]);palette_parents.push_back(palette_parents[source]);}
        refresh_palette();initialized=true;
    }

    void update_palette(const FrameView& f) {
        fill_samples(f);
        const auto& s = sample_scratch;
        if (!initialized || palette.size() != cfg.palette_colors) seed_palette(s);
        const std::size_t k = palette.size();
        std::array<double,64> weight{}, sl{}, sa{}, sb{}, salpha{};
        for (std::size_t i = 0; i < s.size(); ++i) {
            float best = std::numeric_limits<float>::max(); std::size_t bi = 0;
            for (std::size_t j = 0; j < k; ++j) {
                const float d = distance2(s[i], palette[j]);
                if (d < best) { best = d; bi = j; }
            }
            sample_owner[i] = static_cast<std::uint16_t>(bi);
        }
        for (std::size_t i = 0; i < s.size(); ++i) {
            const auto j = sample_owner[i];
            const double w = sample_importance[i];
            weight[j] += w; sl[j] += w*s[i].l; sa[j] += w*s[i].a;
            sb[j] += w*s[i].b; salpha[j] += w*s[i].alpha;
        }
        const float lr = initialized ? cfg.prior_learning_rate : 1.0f;
        for (std::size_t j = 0; j < k; ++j) if (weight[j] > 0.0) {
            Lab target{static_cast<float>(sl[j]/weight[j]), static_cast<float>(sa[j]/weight[j]),
                       static_cast<float>(sb[j]/weight[j]), static_cast<float>(salpha[j]/weight[j])};
            auto& center=palette_centroids[j];auto& parent=palette_parents[j];
            const Lab delta{lr*(target.l-center.l),lr*(target.a-center.a),lr*(target.b-center.b),
                            lr*(target.alpha-center.alpha)};
            center={center.l+delta.l,center.a+delta.a,center.b+delta.b,center.alpha+delta.alpha};
            parent={parent.l+delta.l,parent.a+delta.a,parent.b+delta.b,parent.alpha+delta.alpha};
        }
        refresh_palette();
    }

    void assign_lattice(const FrameView& f) {
        stats.changed_cells=0;stats.reused_cells=0;
        for (std::uint32_t y = 0; y < gh; ++y) {
            const auto sy = std::min(f.height - 1, (2*y + 1)*f.height/(2*gh));
            for (std::uint32_t x = 0; x < gw; ++x) {
                const auto sx = std::min(f.width - 1, (2*x + 1)*f.width/(2*gw));
                const std::size_t q = static_cast<std::size_t>(y)*gw+x;
                std::uint64_t token=static_cast<std::uint64_t>(f.format)<<32;
                if (f.format==PixelFormat::RGBA || f.format==PixelFormat::BGRA) {
                    std::uint32_t packed;
                    std::memcpy(&packed,f.data+static_cast<std::ptrdiff_t>(sy)*f.stride+4u*sx,4);
                    token|=packed;
                } else {
                    const auto yy=f.data[static_cast<std::ptrdiff_t>(sy)*f.stride+sx];
                    std::uint8_t u,v;
                    if(f.format==PixelFormat::NV12){const auto* uv=f.plane1+
                        static_cast<std::ptrdiff_t>(sy/2)*f.stride1+2u*(sx/2);u=uv[0];v=uv[1];}
                    else{u=f.plane1[static_cast<std::ptrdiff_t>(sy/2)*f.stride1+sx/2];
                        v=f.plane2[static_cast<std::ptrdiff_t>(sy/2)*f.stride2+sx/2];}
                    token|=yy|(static_cast<std::uint64_t>(u)<<8)|(static_cast<std::uint64_t>(v)<<16);
                    token|=static_cast<std::uint64_t>(f.full_range)<<40;
                }
                Lab lab;
                if(grid_token[q]==token){lab=grid_lab[q];++stats.reused_cells;}
                else{lab=read_lab(f,sx,sy);grid_lab[q]=lab;grid_token[q]=token;++stats.changed_cells;}
                float best = std::numeric_limits<float>::max(); std::uint16_t bi = 0;
                for (std::uint16_t j = 0; j < palette.size(); ++j) {
                    const float d = distance2(lab, palette[j]);
                    if (d < best) { best = d; bi = j; }
                }
                labels[q] = bi;
            }
        }
    }

    void update_edges() {
        segments.clear();
        const float sx = static_cast<float>(source_w) / gw;
        const float sy = static_cast<float>(source_h) / gh;
        for (std::uint32_t y = 0; y < gh; ++y) for (std::uint32_t x = 0; x+1 < gw; ++x) {
            const std::size_t qi = static_cast<std::size_t>(y)*(gw-1)+x;
            auto& e = vertical[qi];
            const auto c0 = labels[static_cast<std::size_t>(y)*gw+x];
            const auto c1 = labels[static_cast<std::size_t>(y)*gw+x+1];
            e.active = c0 != c1;
            if (!e.active) { e.age = 0; continue; }
            e.age++; e.c0 = c0; e.c1 = c1;
            if (e.age == 1) e.phase = static_cast<float>((mix64(qi) & 0xffffu) / 65536.0);
        }
        const std::uint64_t base = 1ULL << 63;
        for (std::uint32_t y = 0; y+1 < gh; ++y) for (std::uint32_t x = 0; x < gw; ++x) {
            const std::size_t qi = static_cast<std::size_t>(y)*gw+x;
            auto& e = horizontal[qi];
            const auto c0 = labels[static_cast<std::size_t>(y)*gw+x];
            const auto c1 = labels[static_cast<std::size_t>(y+1)*gw+x];
            e.active = c0 != c1;
            if (!e.active) { e.age = 0; continue; }
            e.age++; e.c0 = c0; e.c1 = c1;
            if (e.age == 1) e.phase = static_cast<float>((mix64(base|qi) & 0xffffu) / 65536.0);
        }

        // Compile maximal collinear runs only after every dense edge slot has
        // been updated. Slots retain temporal identity; runs are the compact
        // draw/SVG representation consumed by the reveal scheduler.
        for (std::uint32_t x=0; x+1<gw; ++x) {
            std::uint32_t y=0;
            while (y<gh) {
                const std::size_t first=static_cast<std::size_t>(y)*(gw-1)+x;
                const auto& start=vertical[first];
                if (!start.active) { ++y; continue; }
                const auto c0=start.c0, c1=start.c1; std::uint32_t age=start.age;
                const std::uint32_t y0=y++;
                while (y<gh) {
                    const auto& next=vertical[static_cast<std::size_t>(y)*(gw-1)+x];
                    if (!next.active || next.c0!=c0 || next.c1!=c1) break;
                    age=std::min(age,next.age); ++y;
                }
                segments.push_back({first,(x+1)*sx,y0*sy,(x+1)*sx,y*sy,c0,c1,age});
            }
        }
        for (std::uint32_t y=0; y+1<gh; ++y) {
            std::uint32_t x=0;
            while (x<gw) {
                const std::size_t first=static_cast<std::size_t>(y)*gw+x;
                const auto& start=horizontal[first];
                if (!start.active) { ++x; continue; }
                const auto c0=start.c0, c1=start.c1; std::uint32_t age=start.age;
                const std::uint32_t x0=x++;
                while (x<gw) {
                    const auto& next=horizontal[static_cast<std::size_t>(y)*gw+x];
                    if (!next.active || next.c0!=c0 || next.c1!=c1) break;
                    age=std::min(age,next.age); ++x;
                }
                segments.push_back({base|first,x0*sx,(y+1)*sy,x*sx,(y+1)*sy,c0,c1,age});
            }
        }
        topology_signature=mix64(segments.size());
        for(const auto& s:segments)
            topology_signature=mix64(topology_signature^s.id^
                (static_cast<std::uint64_t>(s.left_color)<<16)^s.right_color);
    }

    static DrawCommand partial(const TraceSegment& s, std::uint64_t random) {
        const float length=std::sqrt((s.x2-s.x1)*(s.x2-s.x1)+(s.y2-s.y1)*(s.y2-s.y1));
        const float unit0=(random&0xffffu)/65535.0f;
        const float unit1=((random>>16)&0xffffu)/65535.0f;
        const bool flicker=((random>>32)&7u)==0;
        const float wanted=flicker ? 3.0f+17.0f*unit0 : 18.0f+102.0f*unit0;
        const float fraction=length>0.001f?std::min(1.0f,wanted/length):1.0f;
        const float tail=(1.0f-fraction)*unit1;
        const float head=tail+fraction;
        DrawCommand c;
        c.source_id=s.id;
        c.x1 = s.x1 + (s.x2-s.x1)*tail; c.y1 = s.y1 + (s.y2-s.y1)*tail;
        c.x2 = s.x1 + (s.x2-s.x1)*head; c.y2 = s.y1 + (s.y2-s.y1)*head;
        return c;
    }

    void shuffle_visits(std::uint64_t seed) {
        for(std::size_t i=visit_order.size();i>1;--i){
            seed=mix64(seed+i);
            std::swap(visit_order[i-1],visit_order[seed%i]);
        }
    }

    void synchronize_visits() {
        if(scheduled_signature==topology_signature&&visit_order.size()==segments.size())return;
        visit_order.resize(segments.size());
        std::iota(visit_order.begin(),visit_order.end(),0u);
        visit_cursor=0;visit_cycle=0;scheduled_signature=topology_signature;
        shuffle_visits(topology_signature);
    }

    std::uint32_t next_visit() {
        if(visit_cursor==visit_order.size()){
            visit_cursor=0;++visit_cycle;
            shuffle_visits(topology_signature^mix64(visit_cycle));
        }
        return visit_order[visit_cursor++];
    }

    void emit_glyph_layer() {
        constexpr float dt=1.0f/30.0f;
        static constexpr char glyphs[]="01ZXKMSV";
        if (!cfg.glyph_layer) {
            for (auto& p:particles) p.alive=false;
            stats.live_glyphs=0; return;
        }
        std::uint32_t live=0;
        for (auto& p:particles) {
            if (!p.alive) continue;
            p.previous_x=p.x; p.previous_y=p.y;
            if (p.arcing) {
                const float angle=p.angular_velocity*dt;
                const float cs=std::cos(angle), sn=std::sin(angle);
                const float vx=cs*p.vx-sn*p.vy;
                p.vy=sn*p.vx+cs*p.vy+4.0f*dt; p.vx=vx;
            } else {
                p.vy+=18.0f*dt;
            }
            p.x+=p.vx*dt; p.y+=p.vy*dt; p.life-=dt;
            if (p.life<=0.0f || p.x<-16.0f || p.x>source_w+16.0f ||
                p.y<-16.0f || p.y>source_h+16.0f) { p.alive=false; continue; }
            ++live;
            const float fade=clamp01(p.life/std::max(0.001f,p.maximum_life));
            DrawCommand trail; trail.kind=CommandKind::GlyphTrail;
            trail.x1=p.previous_x; trail.y1=p.previous_y; trail.x2=p.x; trail.y2=p.y;
            trail.width=1.0f; trail.opacity=0.40f*fade; trail.glow=0.45f*cfg.glow;
            trail.r=130; trail.g=179; trail.b=97; commands.push_back(trail);
            DrawCommand glyph=trail; glyph.kind=CommandKind::Glyph;
            glyph.x1=glyph.x2=p.x; glyph.y1=glyph.y2=p.y; glyph.glyph=p.glyph;
            glyph.opacity=0.86f*fade; glyph.glow=cfg.glow; commands.push_back(glyph);
        }
        if (segments.empty()) {
            stats.live_glyphs=live; return;
        }
        std::uint32_t spawned=0;
        for (std::size_t slot=0; slot<particles.size() && spawned<cfg.glyph_spawn_per_frame; ++slot) {
            auto& p=particles[slot]; if (p.alive) continue;
            const std::uint64_t random=mix64(frame_number*0x9e3779b97f4a7c15ULL+slot*37u+spawned);
            const auto& s=segments[random%segments.size()];
            const float t=((random>>16)&0xffffu)/65535.0f;
            p.x=s.x1+(s.x2-s.x1)*t; p.y=s.y1+(s.y2-s.y1)*t;
            p.previous_x=p.x; p.previous_y=p.y;
            const bool arc=cfg.glyph_motion==GlyphMotion::Arcing ||
                (cfg.glyph_motion==GlyphMotion::Mixed && ((random>>33)&1u));
            p.arcing=arc;
            if (arc) {
                const float dx=s.x2-s.x1, dy=s.y2-s.y1;
                const float norm=std::max(0.001f,std::sqrt(dx*dx+dy*dy));
                const float speed=45.0f+55.0f*((random>>40)&255u)/255.0f;
                const float direction=((random>>48)&1u)?1.0f:-1.0f;
                p.vx=direction*speed*dx/norm; p.vy=direction*speed*dy/norm;
                p.angular_velocity=((random>>49)&1u?1.0f:-1.0f)*
                    (0.55f+0.9f*((random>>50)&127u)/127.0f);
            } else {
                p.vx=-14.0f+28.0f*((random>>40)&255u)/255.0f;
                p.vy=45.0f+75.0f*((random>>48)&255u)/255.0f;
                p.angular_velocity=0.0f;
            }
            p.maximum_life=p.life=0.65f+1.35f*((random>>24)&255u)/255.0f;
            p.glyph=glyphs[(random>>8)&7u]; p.alive=true; ++spawned; ++live;
            DrawCommand glyph; glyph.kind=CommandKind::Glyph;
            glyph.x1=glyph.x2=p.x; glyph.y1=glyph.y2=p.y; glyph.glyph=p.glyph;
            glyph.r=130; glyph.g=179; glyph.b=97; glyph.opacity=0.86f;
            glyph.glow=cfg.glow; commands.push_back(glyph);
        }
        stats.live_glyphs=live;
    }

    void emit_effects() {
        commands.clear();
        const std::uint32_t n = static_cast<std::uint32_t>(segments.size());
        if (n) {
          synchronize_visits();
          const std::uint32_t take = std::min(cfg.segments_per_frame, n);
          for (std::uint32_t i = 0; i < take; ++i) {
            const std::uint32_t index=next_visit();
            const auto& s = segments[index];
            const auto visit_random=mix64(s.id^mix64(visit_cycle+1)^frame_number);
            auto c = partial(s,visit_random);
            const auto noise = static_cast<int>((visit_random >> 56) & 31u) - 15;
            if (cfg.effect == EffectMode::Phosphor) {
                c.kind = CommandKind::Trace; c.width = 1.0f + 0.65f*cfg.glow;
                c.r = byte(130+noise*0.45f); c.g = byte(179+noise*0.8f);
                c.b = byte(97+noise*0.35f); c.opacity = 0.82f; c.glow = cfg.glow;
                commands.push_back(c);
            } else if (cfg.effect == EffectMode::LiquidMetal) {
                c.kind = CommandKind::Trace; c.width = 2.2f; c.opacity = 0.88f;
                c.r = byte(205+noise); c.g = byte(212+noise); c.b = byte(218+noise);
                c.glow = 0.35f + 0.4f*cfg.glow; commands.push_back(c);
            } else {
                auto shadow = c; shadow.kind = CommandKind::EmbossShadow;
                shadow.x1 += 1.0f; shadow.y1 += 1.0f; shadow.x2 += 1.0f; shadow.y2 += 1.0f;
                shadow.r = 12; shadow.g = 16; shadow.b = 14; shadow.width = 2.0f;
                shadow.opacity = 0.72f; commands.push_back(shadow);
                c.kind = CommandKind::Sheen; c.width = 1.25f; c.glow = cfg.glow;
                const auto& pc = palette[s.left_color % palette.size()];
                c.r = pc.r; c.g = pc.g; c.b = pc.blue; c.opacity = 0.92f;
                commands.push_back(c);
            }
          }
        }
        emit_glyph_layer();
    }

    FrameStats process(const FrameView& f) {
        stats = {};
        const bool packed=f.format==PixelFormat::RGBA || f.format==PixelFormat::BGRA;
        const bool valid_chroma=f.format==PixelFormat::NV12
            ? f.plane1 && f.stride1>=static_cast<std::ptrdiff_t>(2*((f.width+1)/2))
            : f.format!=PixelFormat::I420 ||
              (f.plane1&&f.plane2&&f.stride1>=static_cast<std::ptrdiff_t>((f.width+1)/2)&&
               f.stride2>=static_cast<std::ptrdiff_t>((f.width+1)/2));
        if (!f.data || f.width == 0 || f.height == 0 ||
            f.stride < static_cast<std::ptrdiff_t>(packed?4*f.width:f.width) || !valid_chroma)
            return stats;
        const auto t0 = Clock::now(); ensure_shape(f);
        update_palette(f); const auto t1 = Clock::now();
        assign_lattice(f); const auto t2 = Clock::now();
        if(cfg.posterize_only){segments.clear();commands.clear();stats.live_glyphs=0;}
        else update_edges();
        const auto t3 = Clock::now();
        if(!cfg.posterize_only)emit_effects();
        const auto t4 = Clock::now();
        stats.palette_ms = millis(t0,t1); stats.posterize_ms = millis(t1,t2);
        stats.trace_ms = millis(t2,t3); stats.effects_ms = millis(t3,t4);
        stats.total_ms = millis(t0,t4); stats.active_segments = static_cast<std::uint32_t>(segments.size());
        stats.emitted_commands = static_cast<std::uint32_t>(commands.size());
        stats.within_budget = stats.total_ms <= cfg.frame_budget_ms;
        ++frame_number;
        return stats;
    }

    static void put_pixel(const MutableFrameView& d, int x, int y,
                          std::uint8_t r, std::uint8_t g, std::uint8_t b, float alpha) {
        if (x < 0 || y < 0 || x >= static_cast<int>(d.width) || y >= static_cast<int>(d.height)) return;
        if (d.format==PixelFormat::RGBA || d.format==PixelFormat::BGRA) {
            auto* p = d.data + static_cast<std::ptrdiff_t>(y)*d.stride + 4*x;
            const int ri = d.format == PixelFormat::RGBA ? 0 : 2;
            const int bi = d.format == PixelFormat::RGBA ? 2 : 0;
            p[ri] = byte(p[ri]*(1-alpha)+r*alpha); p[1] = byte(p[1]*(1-alpha)+g*alpha);
            p[bi] = byte(p[bi]*(1-alpha)+b*alpha); p[3] = 255;
            return;
        }
        const auto color=rgb_to_yuv(r,g,b,d.full_range);
        auto& yy=d.data[static_cast<std::ptrdiff_t>(y)*d.stride+x];
        yy=byte(yy*(1-alpha)+color.y*alpha);
        if (d.format==PixelFormat::NV12) {
            auto* uv=d.plane1+static_cast<std::ptrdiff_t>(y/2)*d.stride1+2*(x/2);
            uv[0]=byte(uv[0]*(1-alpha)+color.u*alpha);
            uv[1]=byte(uv[1]*(1-alpha)+color.v*alpha);
        } else {
            auto& u=d.plane1[static_cast<std::ptrdiff_t>(y/2)*d.stride1+x/2];
            auto& v=d.plane2[static_cast<std::ptrdiff_t>(y/2)*d.stride2+x/2];
            u=byte(u*(1-alpha)+color.u*alpha); v=byte(v*(1-alpha)+color.v*alpha);
        }
    }

    static void line(const MutableFrameView& d, const DrawCommand& c, int ox=0, int oy=0, float opacity_scale=1.0f) {
        int x0 = static_cast<int>(std::lround(c.x1))+ox, y0 = static_cast<int>(std::lround(c.y1))+oy;
        const int x1 = static_cast<int>(std::lround(c.x2))+ox, y1 = static_cast<int>(std::lround(c.y2))+oy;
        const int dx = std::abs(x1-x0), sx = x0<x1 ? 1 : -1;
        const int dy = -std::abs(y1-y0), sy = y0<y1 ? 1 : -1;
        int err = dx+dy;
        const int radius = std::max(0, static_cast<int>(std::floor(0.5f*c.width)));
        for (;;) {
            for (int by=-radius; by<=radius; ++by)
                for (int bx=-radius; bx<=radius; ++bx)
                    put_pixel(d,x0+bx,y0+by,c.r,c.g,c.b,c.opacity*opacity_scale);
            if (x0==x1 && y0==y1) break;
            const int e2=2*err; if (e2>=dy) { err+=dy; x0+=sx; }
            if (e2<=dx) { err+=dx; y0+=sy; }
        }
    }

    static std::array<std::uint8_t,7> glyph_bitmap(std::uint8_t glyph) {
        switch (glyph) {
        case '0': return {14,17,19,21,25,17,14};
        case '1': return {4,12,4,4,4,4,14};
        case 'Z': return {31,1,2,4,8,16,31};
        case 'X': return {17,17,10,4,10,17,17};
        case 'K': return {17,18,20,24,20,18,17};
        case 'M': return {17,27,21,21,17,17,17};
        case 'S': return {15,16,16,14,1,1,30};
        case 'V': return {17,17,17,17,10,10,4};
        default: return {4,4,4,31,4,4,4};
        }
    }

    static void draw_glyph(const MutableFrameView& d,const DrawCommand& c) {
        const auto rows=glyph_bitmap(c.glyph);
        const int origin_x=static_cast<int>(std::lround(c.x1))-2;
        const int origin_y=static_cast<int>(std::lround(c.y1))-3;
        for (int y=0;y<7;++y) for (int x=0;x<5;++x) if (rows[y]&(1u<<(4-x))) {
            if(c.glow>0.01f){
                put_pixel(d,origin_x+x-1,origin_y+y,c.r,c.g,c.b,.16f*c.glow*c.opacity);
                put_pixel(d,origin_x+x+1,origin_y+y,c.r,c.g,c.b,.16f*c.glow*c.opacity);
                put_pixel(d,origin_x+x,origin_y+y-1,c.r,c.g,c.b,.16f*c.glow*c.opacity);
                put_pixel(d,origin_x+x,origin_y+y+1,c.r,c.g,c.b,.16f*c.glow*c.opacity);
            }
            put_pixel(d,origin_x+x,origin_y+y,c.r,c.g,c.b,c.opacity);
        }
    }

    void render(const FrameView& source, const MutableFrameView& d) const {
        if (!d.data || d.width != source.width || d.height != source.height || labels.empty()) return;
        if (d.format==PixelFormat::RGBA || d.format==PixelFormat::BGRA) {
            for (std::uint32_t y=0; y<d.height; ++y) {
                const auto gy = std::min(gh-1, y*gh/d.height);
                for (std::uint32_t x=0; x<d.width; ++x) {
                    const auto gx = std::min(gw-1, x*gw/d.width);
                    const auto& c = palette[labels[static_cast<std::size_t>(gy)*gw+gx]];
                    auto* p = d.data + static_cast<std::ptrdiff_t>(y)*d.stride + 4*x;
                    if (d.format == PixelFormat::RGBA) { p[0]=c.r; p[1]=c.g; p[2]=c.blue; }
                    else { p[0]=c.blue; p[1]=c.g; p[2]=c.r; }
                    p[3]=c.opacity;
                }
            }
        } else {
            if (!d.plane1 || (d.format==PixelFormat::I420&&!d.plane2)) return;
            std::array<Yuv8,64> yuv{};
            for (std::size_t i=0;i<palette.size();++i)
                yuv[i]=rgb_to_yuv(palette[i].r,palette[i].g,palette[i].blue,d.full_range);
            for (std::uint32_t y=0; y<d.height; ++y) {
                const auto gy=std::min(gh-1,y*gh/d.height);
                auto* row=d.data+static_cast<std::ptrdiff_t>(y)*d.stride;
                for (std::uint32_t x=0;x<d.width;++x) {
                    const auto gx=std::min(gw-1,x*gw/d.width);
                    row[x]=yuv[labels[static_cast<std::size_t>(gy)*gw+gx]].y;
                }
            }
            for (std::uint32_t cy=0;cy<(d.height+1)/2;++cy) {
                const auto y=std::min(d.height-1,2*cy+1);
                const auto gy=std::min(gh-1,y*gh/d.height);
                for (std::uint32_t cx=0;cx<(d.width+1)/2;++cx) {
                    const auto x=std::min(d.width-1,2*cx+1);
                    const auto gx=std::min(gw-1,x*gw/d.width);
                    const auto color=yuv[labels[static_cast<std::size_t>(gy)*gw+gx]];
                    if (d.format==PixelFormat::NV12) {
                        auto* uv=d.plane1+static_cast<std::ptrdiff_t>(cy)*d.stride1+2*cx;
                        uv[0]=color.u; uv[1]=color.v;
                    } else {
                        d.plane1[static_cast<std::ptrdiff_t>(cy)*d.stride1+cx]=color.u;
                        d.plane2[static_cast<std::ptrdiff_t>(cy)*d.stride2+cx]=color.v;
                    }
                }
            }
        }
        for (const auto& c : commands) {
            if (c.kind!=CommandKind::Glyph && c.glow > 0.01f) {
                line(d,c,-1,0,0.20f*c.glow); line(d,c,1,0,0.20f*c.glow);
                line(d,c,0,-1,0.20f*c.glow); line(d,c,0,1,0.20f*c.glow);
            }
            if (c.kind == CommandKind::Glyph) {
                draw_glyph(d,c);
            } else line(d,c);
        }
    }

    std::string svg_snapshot() const {
        std::ostringstream out;out<<std::fixed<<std::setprecision(2);
        out<<"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\""<<source_w
           <<"\" height=\""<<source_h<<"\" viewBox=\"0 0 "<<source_w<<' '<<source_h
           <<"\" shape-rendering=\"geometricPrecision\">\n"
           <<"<rect width=\"100%\" height=\"100%\" fill=\"#050907\"/>\n"
           <<"<path d=\"";
        for(const auto& s:segments)out<<'M'<<s.x1<<' '<<s.y1<<'L'<<s.x2<<' '<<s.y2;
        out<<"\" fill=\"none\" stroke=\"#82b361\" stroke-opacity=\".32\" stroke-width=\"1\"/>\n";
        const auto hex=[](std::uint8_t value){const char* digits="0123456789abcdef";
            std::string s(2,'0');s[0]=digits[value>>4];s[1]=digits[value&15];return s;};
        for(const auto& c:commands){
            const std::string color="#"+hex(c.r)+hex(c.g)+hex(c.b);
            if(c.kind==CommandKind::Glyph){
                out<<"<text x=\""<<c.x1<<"\" y=\""<<c.y1
                   <<"\" fill=\""<<color<<"\" fill-opacity=\""<<c.opacity
                   <<"\" font-family=\"monospace\" font-size=\"7\" text-anchor=\"middle\">"
                   <<static_cast<char>(c.glyph)<<"</text>\n";
            }else{
                out<<"<path d=\"M"<<c.x1<<' '<<c.y1<<'L'<<c.x2<<' '<<c.y2
                   <<"\" fill=\"none\" stroke=\""<<color<<"\" stroke-opacity=\""
                   <<c.opacity<<"\" stroke-width=\""<<c.width<<"\" stroke-linecap=\"round\"/>\n";
            }
        }
        out<<"</svg>\n";return out.str();
    }
};

Engine::Engine(Config config) : impl_(std::make_unique<Impl>(config)) {}
Engine::~Engine() = default;
Engine::Engine(Engine&&) noexcept = default;
Engine& Engine::operator=(Engine&&) noexcept = default;
void Engine::set_config(const Config& c) {
    const auto next=Impl::sanitize(c);const auto& old=impl_->cfg;
    const bool reseed=next.palette_colors!=old.palette_colors||next.palette_samples!=old.palette_samples||
        next.detail_priority!=old.detail_priority||next.population_exponent!=old.population_exponent||
        next.lightness_weight!=old.lightness_weight||next.chroma_weight!=old.chroma_weight||
        next.hue_weight!=old.hue_weight||next.alpha_weight!=old.alpha_weight||
        next.minimum_leaf!=old.minimum_leaf||next.bifurcation_refinement!=old.bifurcation_refinement;
    const bool reseparate=next.node_separation!=old.node_separation;
    impl_->cfg=next;if(reseed)impl_->initialized=false;else if(reseparate&&!impl_->palette_centroids.empty())impl_->refresh_palette();
    impl_->reserve_outputs();
}
const Config& Engine::config() const noexcept { return impl_->cfg; }
void Engine::reset() { const Config c=impl_->cfg; impl_=std::make_unique<Impl>(c); }
const FrameStats& Engine::process(const FrameView& f) { impl_->process(f); return impl_->stats; }
void Engine::render(const FrameView& s, const MutableFrameView& d) const { impl_->render(s,d); }
const std::vector<PaletteColor>& Engine::palette() const noexcept { return impl_->palette; }
const std::vector<std::uint16_t>& Engine::labels() const noexcept { return impl_->labels; }
const std::vector<TraceSegment>& Engine::active_segments() const noexcept { return impl_->segments; }
const std::vector<DrawCommand>& Engine::commands() const noexcept { return impl_->commands; }
std::string Engine::svg_snapshot() const { return impl_->svg_snapshot(); }
std::uint32_t Engine::lattice_width() const noexcept { return impl_->gw; }
std::uint32_t Engine::lattice_height() const noexcept { return impl_->gh; }

} // namespace rvfx
