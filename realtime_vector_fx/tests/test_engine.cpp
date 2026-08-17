#include "rvfx/engine.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

std::vector<std::uint8_t> checker(std::uint32_t w, std::uint32_t h, int shift=0) {
    std::vector<std::uint8_t> image(static_cast<std::size_t>(w)*h*4);
    for (std::uint32_t y=0; y<h; ++y) for (std::uint32_t x=0; x<w; ++x) {
        const bool a = (((static_cast<int>(x)+shift)/16)+(y/16))%2 == 0;
        auto* p=&image[(static_cast<std::size_t>(y)*w+x)*4];
        p[0]=a?220:25; p[1]=a?55:85; p[2]=a?35:220; p[3]=255;
    }
    return image;
}

rvfx::FrameView view(const std::vector<std::uint8_t>& p, std::uint32_t w, std::uint32_t h) {
    return {p.data(),w,h,static_cast<std::ptrdiff_t>(4*w),rvfx::PixelFormat::RGBA};
}

std::uint8_t qbyte(float x){return static_cast<std::uint8_t>(std::max(0.0f,std::min(255.0f,std::round(x))));}

struct Nv12 { std::vector<std::uint8_t> y,uv; };
Nv12 to_nv12(const std::vector<std::uint8_t>& rgba,std::uint32_t w,std::uint32_t h) {
    Nv12 out{{},{} }; out.y.resize(static_cast<std::size_t>(w)*h);
    out.uv.resize(static_cast<std::size_t>(w)*((h+1)/2),128);
    const auto color=[&](std::uint32_t x,std::uint32_t y){
        const auto* p=&rgba[(static_cast<std::size_t>(y)*w+x)*4];
        const float r=p[0]/255.0f,g=p[1]/255.0f,b=p[2]/255.0f;
        const float yy=.2126f*r+.7152f*g+.0722f*b;
        return std::array<std::uint8_t,3>{qbyte(255*yy),qbyte(128+255*(b-yy)/1.8556f),
                                         qbyte(128+255*(r-yy)/1.5748f)};
    };
    for(std::uint32_t y=0;y<h;++y)for(std::uint32_t x=0;x<w;++x)
        out.y[static_cast<std::size_t>(y)*w+x]=color(x,y)[0];
    for(std::uint32_t y=0;y<h;y+=2)for(std::uint32_t x=0;x<w;x+=2){
        const auto c=color(x,y);const auto q=static_cast<std::size_t>(y/2)*w+x;
        out.uv[q]=c[1];out.uv[q+1]=c[2];
    }
    return out;
}

} // namespace

int main() {
    constexpr std::uint32_t w=320,h=180;
    rvfx::Config cfg; cfg.trace_width=160; cfg.palette_colors=4;
    cfg.palette_samples=1024; cfg.segments_per_frame=300;
    cfg.glyph_particles=32; cfg.glyph_spawn_per_frame=4;
    rvfx::Engine engine(cfg);
    auto a=checker(w,h);
    const auto& first=engine.process(view(a,w,h));
    assert(engine.palette().size()==4);
    assert(engine.lattice_width()==160 && engine.lattice_height()==90);
    assert(first.changed_cells==160*90 && first.reused_cells==0);
    assert(!engine.active_segments().empty());
    assert(first.live_glyphs==4);
    assert(std::any_of(engine.active_segments().begin(),engine.active_segments().end(),[](const auto& s){
        return std::abs(s.x2-s.x1)>4.0f || std::abs(s.y2-s.y1)>4.0f;
    })); // unit lattice edges were compiled into continuous runs
    const auto stable_id=engine.active_segments().front().id;
    const auto second=engine.process(view(a,w,h));
    assert(second.changed_cells==0 && second.reused_cells==160*90);
    assert(std::any_of(engine.active_segments().begin(),engine.active_segments().end(),
                       [stable_id](const auto& s){return s.id==stable_id && s.age>=2;}));
    assert(std::any_of(engine.commands().begin(),engine.commands().end(),[](const auto& c){
        return c.kind==rvfx::CommandKind::GlyphTrail &&
               (std::abs(c.x2-c.x1)>0.01f || std::abs(c.y2-c.y1)>0.01f);
    }));

    // The recovered SVG Oscilloscope Renderer contract requires a complete
    // shuffled cycle: no stable trace run repeats before every run is visited.
    rvfx::Config visit_cfg=cfg;visit_cfg.palette_colors=2;visit_cfg.segments_per_frame=1;
    visit_cfg.glyph_layer=false;visit_cfg.glyph_particles=0;
    rvfx::Engine visitor(visit_cfg);visitor.process(view(a,w,h));
    const auto visit_count=visitor.active_segments().size();assert(visit_count>1);
    std::vector<std::uint64_t> visited{visitor.commands().front().source_id};
    for(std::size_t i=1;i<visit_count;++i){
        visitor.process(view(a,w,h));assert(visitor.active_segments().size()==visit_count);
        assert(visitor.commands().size()==1);visited.push_back(visitor.commands().front().source_id);
    }
    std::sort(visited.begin(),visited.end());
    assert(std::adjacent_find(visited.begin(),visited.end())==visited.end());

    // A visit emits a bounded local slice rather than necessarily redrawing
    // the entire source line.
    auto split=a;
    for(std::uint32_t y=0;y<h;++y)for(std::uint32_t x=0;x<w;++x){
        auto* p=&split[(static_cast<std::size_t>(y)*w+x)*4];
        p[0]=x<w/2?230:20;p[1]=x<w/2?40:80;p[2]=x<w/2?25:225;p[3]=255;
    }
    rvfx::Engine slicer(visit_cfg);slicer.process(view(split,w,h));
    assert(slicer.active_segments().size()==1&&slicer.commands().size()==1);
    const auto& slice=slicer.commands().front();
    const float slice_length=std::hypot(slice.x2-slice.x1,slice.y2-slice.y1);
    const auto& whole=slicer.active_segments().front();
    const float whole_length=std::hypot(whole.x2-whole.x1,whole.y2-whole.y1);
    assert(slice_length<whole_length);

    cfg.glyph_particles=1;cfg.glyph_spawn_per_frame=1;cfg.glyph_motion=rvfx::GlyphMotion::Falling;
    rvfx::Engine falling(cfg);falling.process(view(a,w,h));falling.process(view(a,w,h));
    const auto falling_trail=std::find_if(falling.commands().begin(),falling.commands().end(),[](const auto& c){
        return c.kind==rvfx::CommandKind::GlyphTrail;
    });
    assert(falling_trail!=falling.commands().end()&&falling_trail->y2>falling_trail->y1);

    cfg.glyph_motion=rvfx::GlyphMotion::Arcing;rvfx::Engine arcing(cfg);
    arcing.process(view(a,w,h));arcing.process(view(a,w,h));
    const auto arc1_it=std::find_if(arcing.commands().begin(),arcing.commands().end(),[](const auto& c){
        return c.kind==rvfx::CommandKind::GlyphTrail;
    });
    assert(arc1_it!=arcing.commands().end());const auto arc1=*arc1_it;
    arcing.process(view(a,w,h));
    const auto arc2_it=std::find_if(arcing.commands().begin(),arcing.commands().end(),[](const auto& c){
        return c.kind==rvfx::CommandKind::GlyphTrail;
    });
    assert(arc2_it!=arcing.commands().end());const auto arc2=*arc2_it;
    const float cross=(arc1.x2-arc1.x1)*(arc2.y2-arc2.y1)-
                      (arc1.y2-arc1.y1)*(arc2.x2-arc2.x1);
    assert(std::abs(cross)>1e-4f); // velocity rotated between frames

    auto before=engine.palette(); auto shifted=checker(w,h,1);
    const auto shifted_stats=engine.process(view(shifted,w,h));
    assert(shifted_stats.changed_cells>0 && shifted_stats.changed_cells<160*90);
    float movement=0.0f;
    for (std::size_t i=0;i<before.size();++i)
        movement+=std::abs(before[i].l-engine.palette()[i].l);
    assert(movement<0.25f); // temporal priors prevent palette identity jumps

    std::vector<std::uint8_t> rendered(a.size());
    rvfx::MutableFrameView out{rendered.data(),w,h,static_cast<std::ptrdiff_t>(4*w),rvfx::PixelFormat::RGBA};
    engine.render(view(a,w,h),out);
    assert(std::any_of(rendered.begin(),rendered.end(),[](std::uint8_t q){return q!=0;}));
    const auto svg=engine.svg_snapshot();
    assert(svg.find("<svg")!=std::string::npos&&svg.find("<path")!=std::string::npos);
    assert(svg.find("#82b361")!=std::string::npos&&svg.find("<text")!=std::string::npos);

    cfg.glyph_layer=false; cfg.effect=rvfx::EffectMode::LiquidMetal; engine.set_config(cfg);
    engine.process(view(a,w,h));
    assert(std::all_of(engine.commands().begin(),engine.commands().end(),[](const auto& c){
        return c.kind==rvfx::CommandKind::Trace;
    }));
    cfg.effect=rvfx::EffectMode::EmbossSheen; engine.set_config(cfg);
    engine.process(view(a,w,h));
    assert(engine.commands().size()%2==0 && !engine.commands().empty());
    assert(std::all_of(engine.commands().begin(),engine.commands().end(),[](const auto& c){
        return c.kind==rvfx::CommandKind::EmbossShadow || c.kind==rvfx::CommandKind::Sheen;
    }));

    // Native 4:2:0 paths sample and composite without an RGB staging frame.
    auto nv=to_nv12(a,w,h); const auto original_y=nv.y;
    rvfx::FrameView nv_in{nv.y.data(),w,h,static_cast<std::ptrdiff_t>(w),rvfx::PixelFormat::NV12};
    nv_in.plane1=nv.uv.data();nv_in.stride1=w;
    rvfx::MutableFrameView nv_out{nv.y.data(),w,h,static_cast<std::ptrdiff_t>(w),rvfx::PixelFormat::NV12};
    nv_out.plane1=nv.uv.data();nv_out.stride1=w;
    rvfx::Engine nv_engine(cfg);const auto nv_first=nv_engine.process(nv_in);nv_engine.render(nv_in,nv_out);
    assert(nv_engine.palette().size()==cfg.palette_colors);
    assert(nv.y!=original_y);
    assert(nv_first.changed_cells==160*90);

    std::vector<std::uint8_t> u(static_cast<std::size_t>(w/2)*(h/2));
    std::vector<std::uint8_t> v(u.size());
    for(std::uint32_t y=0;y<h/2;++y)for(std::uint32_t x=0;x<w/2;++x){
        u[static_cast<std::size_t>(y)*(w/2)+x]=nv.uv[static_cast<std::size_t>(y)*w+2*x];
        v[static_cast<std::size_t>(y)*(w/2)+x]=nv.uv[static_cast<std::size_t>(y)*w+2*x+1];
    }
    rvfx::FrameView i420{nv.y.data(),w,h,static_cast<std::ptrdiff_t>(w),rvfx::PixelFormat::I420};
    i420.plane1=u.data();i420.plane2=v.data();i420.stride1=i420.stride2=w/2;
    rvfx::MutableFrameView i420_out{nv.y.data(),w,h,static_cast<std::ptrdiff_t>(w),rvfx::PixelFormat::I420};
    i420_out.plane1=u.data();i420_out.plane2=v.data();i420_out.stride1=i420_out.stride2=w/2;
    rvfx::Engine i420_engine(cfg);i420_engine.process(i420);i420_engine.render(i420,i420_out);
    assert(!i420_engine.active_segments().empty());
    std::cout << "rvfx core tests passed\n";
}
