#include "rvfx/entropy_stretch.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace rvfx::entropy;
void check(bool p,const char* message){if(!p){std::fprintf(stderr,"FAIL: %s\n",message);std::exit(1);}}
std::vector<uint8_t> scene(int w,int h){std::vector<uint8_t> a(w*h*4,255);
    for(int y=0;y<h;++y)for(int x=0;x<w;++x){auto i=4*(y*w+x);
        // Left: low-entropy colors, right: varied high-color-entropy material.
        if(x<w/2){a[i]=uint8_t(30+15*x/(w/2));a[i+1]=a[i];a[i+2]=a[i];}
        else {a[i]=uint8_t(140+(x*13+y*7)%50);a[i+1]=uint8_t(130+(x*7+y*17)%50);a[i+2]=uint8_t(120+(x*19+y*3)%50);}}
    return a;}
void converge(Profile& p,const Config& cfg){for(int i=0;i<900;++i)p.advance(1./60,cfg);}
int main(){
    Profile p;Config cfg;Color c{.13f,.5f,.97f};auto q=p.apply(c);
    for(int i=0;i<3;++i)check(std::abs(c[i]-q[i])<2e-6,"identity LUT and trilinear interpolation");
    Matrix cov{{{{.04,.012,.004}},{{.012,.02,.006}},{{.004,.006,.01}}}};
    auto m=whitening(cov,1e-6,100);double variance=(.04+.02+.01)/3;
    for(int i=0;i<3;++i)for(int j=0;j<3;++j){double x=0;for(int a=0;a<3;++a)for(int b=0;b<3;++b)x+=m[i][a]*cov[a][b]*m[j][b];
        check(std::abs(x-(i==j?variance:0))<1e-10,"whitening covariance identity");}
    auto a=scene(128,72);check(p.analyze(a.data(),128,72,128*4,cfg),"analysis");
    check(p.diagnostics().samples==128*72,"sample count");
    for(auto& curve:p.diagnostics().curves){check(curve[0]==0&&curve[bins]==1,"CDF endpoints");
        for(int i=1;i<=bins;++i)check(curve[i]>curve[i-1],"strict CDF monotonicity");
        check((curve[48]-curve[32])>.25f,"high-entropy interval expands beyond identity");
        check((curve[16]-curve[0])<.25f,"low-entropy interval condenses below identity");}
    auto old=p.atlas();p.advance(1./60,cfg);float largest=0;
    for(std::size_t i=0;i<old.size();++i)largest=std::max(largest,std::abs(p.atlas()[i]-old[i]));
    check(largest<=cfg.max_change_per_second/60+1e-7,"profile slew bound");
    Profile p30,p60;check(p30.analyze(a.data(),128,72,512,cfg)&&p60.analyze(a.data(),128,72,512,cfg),"rate inputs");
    Config smooth=cfg;smooth.max_change_per_second=100;
    for(int i=0;i<60;++i)p60.advance(1./60,smooth);for(int i=0;i<30;++i)p30.advance(1./30,smooth);
    for(std::size_t i=0;i<p30.atlas().size();++i)check(std::abs(p30.atlas()[i]-p60.atlas()[i])<2e-6,"time-based exponential rate invariance");
    converge(p,cfg);for(float x:p.atlas())check(std::isfinite(x)&&x>=0&&x<=1,"bounded finite gamut");
    auto before=p.atlas();check(!p.analyze(nullptr,128,72,512,cfg),"null rejection");
    check(!p.analyze(a.data(),128,72,8,cfg),"stride rejection");
    for(std::size_t i=3;i<a.size();i+=4)a[i]=0;
    check(!p.analyze(a.data(),128,72,512,cfg),"transparent samples ignored");
    check(p.atlas()==before,"invalid input retains profile");
    std::vector<uint8_t> flat(32*32*4,90);for(std::size_t i=3;i<flat.size();i+=4)flat[i]=255;
    Profile constant;check(constant.analyze(flat.data(),32,32,128,cfg),"flat analysis");converge(constant,cfg);
    check(constant.diagnostics().mean_entropy==0,"flat entropy zero");
    q=constant.apply({90.f/255,90.f/255,90.f/255});for(float x:q)check(std::abs(x-90.f/255)<1e-5,"flat scene stable");
    for(int y=0;y<32;++y)for(int x=0;x<32;++x)for(int c=0;c<3;++c)flat[(y*32+x)*4+c]=uint8_t(x*8);
    check(constant.analyze(flat.data(),32,32,128,cfg),"rank-one analysis");converge(constant,cfg);
    for(int i=0;i<256;++i){q=constant.apply({i/255.f,i/255.f,i/255.f});check(std::abs(q[0]-q[1])<1e-5&&std::abs(q[1]-q[2])<1e-5,"no false chroma on grayscale");}
    // Analysis honors row padding and premultiplied-alpha normalization.
    auto source=scene(128,72);std::vector<uint8_t> padded(72*528,13);
    for(int y=0;y<72;++y)std::copy_n(source.data()+512*y,512,padded.data()+528*y);
    Profile padding;check(padding.analyze(padded.data(),128,72,528,cfg),"padded stride");
    check(std::abs(padding.diagnostics().mean_entropy-p30.diagnostics().mean_entropy)<1e-6,"padding excluded");
    auto start=std::chrono::steady_clock::now();for(int i=0;i<100;++i)padding.analyze(source.data(),128,72,512,cfg);
    double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/100;
    std::printf("entropy core invariants passed; 128x72 analysis + 33^3 LUT %.3f ms/update\n",ms);
    return 0;
}
