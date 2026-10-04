#include "rvfx/entropy_stretch.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
// P6 RGB reference renderer. Analysis uses a sparse point lattice, rendering
// uses the exact same trilinear profile interpolation as the GPU shader.
int main(int argc,char** argv){
    if(argc!=3){std::fprintf(stderr,"usage: entropy_demo input.ppm output.ppm\n");return 2;}
    auto* in=std::fopen(argv[1],"rb");if(!in)return 2;int w,h,max;char magic[3];
    if(std::fscanf(in,"%2s %d %d %d",magic,&w,&h,&max)!=4||magic[0]!='P'||magic[1]!='6'||w<1||h<1||max!=255)return 3;
    std::fgetc(in);std::vector<unsigned char> rgb(std::size_t(w)*h*3);if(std::fread(rgb.data(),1,rgb.size(),in)!=rgb.size())return 4;std::fclose(in);
    int aw=std::min(128,w),ah=std::clamp(int(double(aw)*h/w+.5),1,256);std::vector<unsigned char> sample(aw*ah*4,255);
    for(int y=0;y<ah;++y)for(int x=0;x<aw;++x){int sx=std::min(w-1,int((x+.25)*w/aw)),sy=std::min(h-1,int((y+.25)*h/ah));
        for(int c=0;c<3;++c)sample[(y*aw+x)*4+c]=rgb[(sy*w+sx)*3+c];}
    rvfx::entropy::Profile p;rvfx::entropy::Config cfg;p.analyze(sample.data(),aw,ah,aw*4,cfg);
    for(int i=0;i<1500;++i)p.advance(1./60,cfg);
    auto* out=std::fopen(argv[2],"wb");if(!out)return 5;std::fprintf(out,"P6\n%d %d\n255\n",w,h);
    for(std::size_t i=0;i<rgb.size();i+=3){rvfx::entropy::Color c;for(int k=0;k<3;++k)c[k]=rgb[i+k]/255.f;auto q=p.apply(c);
        for(int k=0;k<3;++k)rgb[i+k]=static_cast<unsigned char>(std::clamp(int(q[k]*255+.5),0,255));}
    std::fwrite(rgb.data(),1,rgb.size(),out);std::fclose(out);
    std::printf("%dx%d, %zu lattice samples, normalized mean entropy %.5f\n",w,h,p.diagnostics().samples,p.diagnostics().mean_entropy);
}
