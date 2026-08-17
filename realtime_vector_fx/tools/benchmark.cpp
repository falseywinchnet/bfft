#include "rvfx/engine.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

int main(int argc,char** argv) {
    const std::uint32_t w=argc>1?static_cast<std::uint32_t>(std::stoul(argv[1])):1280;
    const std::uint32_t h=argc>2?static_cast<std::uint32_t>(std::stoul(argv[2])):720;
    const int frames=argc>3?std::stoi(argv[3]):180;
    bool nv12=false,static_scene=false;
    for(int i=4;i<argc;++i){const std::string option=argv[i];
        nv12|=option=="nv12";static_scene|=option=="static";}
    std::vector<std::uint8_t> pixels(nv12?static_cast<std::size_t>(w)*h:
                                          static_cast<std::size_t>(w)*h*4);
    std::vector<std::uint8_t> uv(nv12?static_cast<std::size_t>(w)*((h+1)/2):0);
    rvfx::Config cfg; cfg.trace_width=480; cfg.palette_colors=8;
    cfg.palette_samples=4096; cfg.segments_per_frame=2048;
    rvfx::Engine engine(cfg);
    std::vector<double> total, core;
    std::uint64_t reused_cells=0,changed_cells=0;
    for(int f=0;f<frames+12;++f) {
        const int scene_frame=static_scene?0:f;
        const auto rgb=[&](std::uint32_t x,std::uint32_t y){
            const float wave=0.5f+0.5f*std::sin(0.025f*x+0.019f*y+0.08f*scene_frame);
            const bool cell=(((x+3*scene_frame)/80)+(y/60))&1u;
            return std::array<std::uint8_t,3>{static_cast<std::uint8_t>(30+190*wave),
                static_cast<std::uint8_t>(35+150*(cell?wave:1-wave)),
                static_cast<std::uint8_t>(35+185*(cell?1-wave:wave))};
        };
        for(std::uint32_t y=0;y<h;++y) for(std::uint32_t x=0;x<w;++x) {
            const auto c=rgb(x,y);
            if(!nv12){auto* p=&pixels[(static_cast<std::size_t>(y)*w+x)*4];
                p[0]=c[0];p[1]=c[1];p[2]=c[2];p[3]=255;
            }else{const float r=c[0]/255.0f,g=c[1]/255.0f,b=c[2]/255.0f;
                pixels[static_cast<std::size_t>(y)*w+x]=static_cast<std::uint8_t>(255*(.2126f*r+.7152f*g+.0722f*b));}
        }
        if(nv12)for(std::uint32_t y=0;y<h;y+=2)for(std::uint32_t x=0;x<w;x+=2){
            const auto c=rgb(x,y);const float r=c[0]/255.0f,g=c[1]/255.0f,b=c[2]/255.0f;
            const float yy=.2126f*r+.7152f*g+.0722f*b;const auto q=static_cast<std::size_t>(y/2)*w+x;
            uv[q]=static_cast<std::uint8_t>(std::clamp(128+255*(b-yy)/1.8556f,0.0f,255.0f));
            uv[q+1]=static_cast<std::uint8_t>(std::clamp(128+255*(r-yy)/1.5748f,0.0f,255.0f));
        }
        rvfx::FrameView in{pixels.data(),w,h,static_cast<std::ptrdiff_t>(nv12?w:4*w),
                           nv12?rvfx::PixelFormat::NV12:rvfx::PixelFormat::RGBA};
        if(nv12){in.plane1=uv.data();in.stride1=w;}
        const auto start=std::chrono::steady_clock::now();
        const auto stats=engine.process(in);
        rvfx::MutableFrameView out{pixels.data(),w,h,static_cast<std::ptrdiff_t>(nv12?w:4*w),
                                  nv12?rvfx::PixelFormat::NV12:rvfx::PixelFormat::RGBA};
        if(nv12){out.plane1=uv.data();out.stride1=w;}
        engine.render(in,out);
        const auto stop=std::chrono::steady_clock::now();
        if(f>=12) {
            core.push_back(stats.total_ms);
            reused_cells+=stats.reused_cells;changed_cells+=stats.changed_cells;
            total.push_back(std::chrono::duration<double,std::milli>(stop-start).count());
        }
    }
    std::sort(total.begin(),total.end()); std::sort(core.begin(),core.end());
    const auto pct=[](const std::vector<double>& v,double p){return v[std::min(v.size()-1,static_cast<std::size_t>(p*v.size()))];};
    const double mean=std::accumulate(total.begin(),total.end(),0.0)/total.size();
    std::cout<<std::fixed<<std::setprecision(3)
             <<"{\"resolution\":\""<<w<<"x"<<h<<"\",\"format\":\""<<(nv12?"nv12":"rgba")
             <<"\",\"scene\":\""<<(static_scene?"static":"changing")<<"\",\"frames\":"<<frames
             <<",\"core_p50_ms\":"<<pct(core,.50)<<",\"core_p95_ms\":"<<pct(core,.95)
             <<",\"mean_reused_cells\":"<<(reused_cells/frames)
             <<",\"mean_changed_cells\":"<<(changed_cells/frames)
             <<",\"composited_mean_ms\":"<<mean<<",\"composited_p95_ms\":"<<pct(total,.95)
             <<",\"budget_ms\":"<<cfg.frame_budget_ms<<",\"passes_30fps\":"
             <<(pct(total,.95)<=cfg.frame_budget_ms?"true":"false")<<"}\n";
    return pct(total,.95)<=cfg.frame_budget_ms?0:2;
}
