#include "rvfx/engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc,char** argv) {
    const std::filesystem::path output=argc>1?argv[1]:"/tmp/rvfx_demo_frames";
    const int frames=argc>2?std::stoi(argv[2]):90;
    constexpr std::uint32_t w=640,h=360;
    std::filesystem::create_directories(output);
    std::vector<std::uint8_t> pixels(static_cast<std::size_t>(w)*h*4);
    rvfx::Config cfg;cfg.trace_width=320;cfg.palette_colors=10;
    cfg.palette_samples=3072;cfg.segments_per_frame=1200;
    cfg.glyph_particles=320;cfg.glyph_spawn_per_frame=10;
    cfg.glyph_motion=rvfx::GlyphMotion::Mixed;cfg.glow=.72f;
    rvfx::Engine engine(cfg);double total_ms=0.0;
    for(int frame=0;frame<frames;++frame){
        if(frame==frames/3)cfg.effect=rvfx::EffectMode::LiquidMetal;
        if(frame==2*frames/3)cfg.effect=rvfx::EffectMode::EmbossSheen;
        engine.set_config(cfg);
        const float time=frame/30.0f;
        for(std::uint32_t y=0;y<h;++y)for(std::uint32_t x=0;x<w;++x){
            const float nx=(x-w*.5f)/h,ny=(y-h*.5f)/h;
            const float wave=.5f+.5f*std::sin(18*nx+4*std::sin(5*ny+time));
            const float cx=nx-.38f*std::sin(.9f*time),cy=ny-.22f*std::cos(1.2f*time);
            const float disc=std::max(0.0f,1.0f-5.5f*std::sqrt(cx*cx+cy*cy));
            const float ring=std::max(0.0f,1.0f-18.0f*std::abs(std::sqrt((nx+.38f)*(nx+.38f)+
                (ny-.08f)*(ny-.08f))-(.24f+.025f*std::sin(time))));
            const bool blocks=(((x+static_cast<int>(45*time))/64)+(y/48))%2==0;
            auto* p=&pixels[(static_cast<std::size_t>(y)*w+x)*4];
            p[0]=static_cast<std::uint8_t>(std::clamp(18+92*wave+130*disc,0.0f,255.0f));
            p[1]=static_cast<std::uint8_t>(std::clamp(20+65*(blocks?wave:1-wave)+170*ring,0.0f,255.0f));
            p[2]=static_cast<std::uint8_t>(std::clamp(30+115*(1-wave)+90*disc,0.0f,255.0f));p[3]=255;
        }
        rvfx::FrameView in{pixels.data(),w,h,static_cast<std::ptrdiff_t>(4*w),rvfx::PixelFormat::RGBA};
        const auto started=std::chrono::steady_clock::now();engine.process(in);
        rvfx::MutableFrameView out{pixels.data(),w,h,static_cast<std::ptrdiff_t>(4*w),rvfx::PixelFormat::RGBA};
        engine.render(in,out);total_ms+=std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now()-started).count();
        std::ostringstream name;name<<"frame_"<<std::setw(4)<<std::setfill('0')<<frame<<".ppm";
        std::ofstream file(output/name.str(),std::ios::binary);file<<"P6\n"<<w<<" "<<h<<"\n255\n";
        for(std::size_t i=0;i<pixels.size();i+=4)file.write(reinterpret_cast<const char*>(&pixels[i]),3);
    }
    std::ofstream svg(output/"snapshot.svg");svg<<engine.svg_snapshot();
    std::cout<<"rendered "<<frames<<" frames to "<<output<<" at "
             <<std::fixed<<std::setprecision(3)<<total_ms/frames<<" ms/frame\n";
}
