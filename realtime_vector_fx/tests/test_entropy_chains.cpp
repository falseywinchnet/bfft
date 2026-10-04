#include "rvfx/entropy_stretch.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
using namespace rvfx::entropy;
void check(bool b,const char* why){if(!b){std::fprintf(stderr,"FAIL: %s\n",why);std::exit(1);}}
float y(Color c){return .2126f*c[0]+.7152f*c[1]+.0722f*c[2];}
float error(Color a,Color b){float e=0;for(int c=0;c<3;++c)e=std::max(e,std::abs(a[c]-b[c]));return e;}
void settle(Profile& p,Config c){c.adaptation_seconds=.05;c.max_change_per_second=100;for(int i=0;i<120;++i)p.advance(.1,c);}
std::vector<uint8_t> source(int seed){std::mt19937 rng(seed);std::vector<uint8_t> a(128*72*4,255);
    for(int row=0;row<72;++row)for(int x=0;x<128;++x)for(int c=0;c<3;++c)
        a[4*(row*128+x)+c]=x<64?uint8_t(20+x/3):uint8_t(70+rng()%100);
    return a;}
int main(){
    Config legacy;legacy.decorrelation=.09;legacy.allocation=.11;
    Config cfg=legacy;cfg.chains=true;
    // All orderings with tone stages disabled must preserve every evolving float
    // and GPU upload, including on successive changing source frames.
    for(int order=0;order<6;++order){Profile old,chain;std::vector<uint16_t> a,b;cfg.chain_order=ChainOrder(order);
        for(int frame=0;frame<90;++frame){if(frame%15==0){auto image=source(frame);check(old.analyze(image.data(),128,72,512,legacy),"legacy analysis");
            check(chain.analyze(image.data(),128,72,512,cfg),"chain analysis");}
            check(old.advance_and_pack(1./60,legacy,a)==chain.advance_and_pack(1./60,cfg,b),"same upload decision");
            check(old.atlas()==chain.atlas()&&a==b,"RGB-only chain bit-exact legacy preservation");}}
    auto image=source(71);Profile p;cfg.rgb_amount=0;cfg.brightness_amount=0;cfg.contrast_amount=0;
    check(p.analyze(image.data(),128,72,512,cfg),"identity chain");settle(p,cfg);Profile identity;
    check(p.atlas()==identity.atlas(),"all amounts zero exact identity");
    auto stats=p.diagnostics();
    for(int order=0;order<6;++order){cfg.chain_order=ChainOrder(order);cfg.rgb_amount=.71;cfg.brightness_amount=.63;cfg.contrast_amount=.82;
        cfg.decorrelation=.8;cfg.allocation=.9;Profile varied;check(varied.analyze(image.data(),128,72,512,cfg),"varied chain");auto d=varied.diagnostics();
        check(d.contrast_cdf==stats.contrast_cdf&&d.midtone==stats.midtone&&d.brightness_gain==stats.brightness_gain,"tone analysis independent of RGB settings, stage amounts and order");
        check(d.covariance==stats.covariance,"source covariance independent of chain");
        for(int i=1;i<=bins;++i)check(d.contrast_cdf[i]>d.contrast_cdf[i-1],"strictly positive contrast density");}
    cfg.rgb_amount=0;cfg.brightness_amount=0;cfg.contrast_amount=1;cfg.chain_order=ChainOrder::ColorContrastBrightness;
    p.analyze(image.data(),128,72,512,cfg);float pivot=p.diagnostics().midtone;
    check(error(p.target_color({pivot,pivot,pivot},cfg),{pivot,pivot,pivot})<2e-6,"contrast preserves source midtone");
    float previous=-1;for(int i=0;i<=4096;++i){float v=float(i)/4096;auto c=p.target_color({v,v,v},cfg);
        check(std::abs(c[0]-c[1])<2e-6&&std::abs(c[1]-c[2])<2e-6,"tone stages preserve neutral axis");
        check(c[0]>=previous,"contrast grayscale monotonicity");previous=c[0];}
    check(error(p.target_color({0,0,0},cfg),{0,0,0})==0&&error(p.target_color({1,1,1},cfg),{1,1,1})==0,"contrast endpoints");
    auto at=[&](float v){return p.target_color({v,v,v},cfg)[0];};
    check(at(.5f)-at(.3f)>.2f,"entropy-supported brightness interval expands");
    cfg.brightness_amount=1;cfg.contrast_amount=0;auto d=p.diagnostics();float gain=d.brightness_gain;
    check(error(p.target_color({pivot,pivot,pivot},cfg),{gain*pivot/(1+(gain-1)*pivot),gain*pivot/(1+(gain-1)*pivot),gain*pivot/(1+(gain-1)*pivot)})<2e-6,"brightness bounded odds mapping");
    previous=-1;for(int i=0;i<=4096;++i){float v=float(i)/4096;auto c=p.target_color({v,v,v},cfg);check(c[0]>=previous,"brightness monotonicity");previous=c[0];}
    // Independent hand composition: order and per-stage interpolation semantics.
    cfg.rgb_amount=.3;cfg.contrast_amount=.6;cfg.brightness_amount=.4;
    p.analyze(image.data(),128,72,512,cfg);Color input{.18,.34,.6};
    Config only=cfg;only.brightness_amount=only.contrast_amount=0;auto c=p.target_color(input,only);
    only=cfg;only.rgb_amount=only.brightness_amount=0;c=p.target_color(c,only);
    only=cfg;only.rgb_amount=only.contrast_amount=0;c=p.target_color(c,only);
    check(error(c,p.target_color(input,cfg))<1e-7,"ordered composition of independently sampled stages");
    only=cfg;only.rgb_amount=only.contrast_amount=0;only.brightness_amount=1;auto full=p.target_color(input,only);only.brightness_amount=.4;
    Color partial;for(int k=0;k<3;++k)partial[k]=input[k]+.4f*(full[k]-input[k]);
    check(error(partial,p.target_color(input,only))<1e-7,"stage amount interpolates with its own input");
    Config reversed=cfg;reversed.chain_order=ChainOrder::BrightnessContrastColor;
    check(error(p.target_color(input,cfg),p.target_color(input,reversed))>1e-4,"order actually affects composition");
    // Verify luma placement and common chroma contraction, including saturated colors.
    for(Color in:{Color{1,0,0},Color{0,1,0},Color{0,0,1},Color{.2,.4,.6}}){
        only.brightness_amount=1;auto out=p.target_color(in,only);float before=y(in),after=y(out),expected=gain*before/(1+(gain-1)*before);
        check(std::abs(after-expected)<2e-6,"gamut mapping preserves requested luma");
        float a=in[0]-before,b=in[1]-before,u=out[0]-after,v=out[1]-after;
        check(std::abs(a*v-b*u)<2e-6,"gamut mapping preserves chroma direction");}
    // Dense off-grid evaluation against the analytic target, at subtle and full settings.
    std::mt19937 rng(19);std::uniform_real_distribution<float> random(0,1);
    for(int strong=0;strong<2;++strong){double sum=0;float worst=0;int n=0;
        cfg=legacy;cfg.chains=true;cfg.brightness_amount=cfg.contrast_amount=strong?1.f:.15f;
        for(int order=0;order<6;++order){cfg.chain_order=ChainOrder(order);p.analyze(image.data(),128,72,512,cfg);settle(p,cfg);
            for(float value:p.atlas())check(std::isfinite(value)&&value>=0&&value<=1,"finite in-gamut composed LUT");
            for(int i=0;i<10000;++i){Color in{random(rng),random(rng),random(rng)};float e=error(p.apply(in),p.target_color(in,cfg));sum+=e;worst=std::max(worst,e);++n;}}
        std::printf("chain LUT vs direct composition (%s): mean max-channel error %.6f, maximum %.6f\n",strong?"full tone amounts":"0.15 tone amounts",sum/n,worst);
        check(sum/n<.005&&worst<.1,"LUT approximation error envelope");}
    cfg.rgb_amount=0;cfg.brightness_amount=cfg.contrast_amount=1;
    for(uint8_t value:{uint8_t(0),uint8_t(90),uint8_t(255)}){std::vector<uint8_t> flat(32*32*4,value);for(std::size_t i=3;i<flat.size();i+=4)flat[i]=255;
        p.analyze(flat.data(),32,32,128,cfg);check(!p.diagnostics().tone_active,"flat scene disables tone stretch");settle(p,cfg);for(std::size_t i=0;i<p.atlas().size();++i)check(std::abs(p.atlas()[i]-identity.atlas()[i])<2e-7,"flat tone profile settles to identity");
        check(p.target_color({.2,.4,.6},cfg)==Color({.2,.4,.6}),"flat tone target exact identity");}
    cfg=legacy;cfg.chains=true;cfg.brightness_amount=cfg.contrast_amount=1;
    Profile moving;moving.analyze(image.data(),128,72,512,cfg);auto previous_lut=moving.atlas();moving.advance(1./60,cfg);
    for(std::size_t i=0;i<previous_lut.size();++i)check(std::abs(previous_lut[i]-moving.atlas()[i])<=cfg.max_change_per_second/60+1e-7,"complete chain respects temporal slew bound");
    std::vector<uint8_t> hidden(32*32*4,0);auto retained=p.atlas();check(!p.analyze(hidden.data(),32,32,128,cfg),"transparent rejection");check(p.atlas()==retained,"transparent input retains mapping");
    for(bool enabled:{false,true}){cfg=legacy;cfg.chains=enabled;cfg.brightness_amount=cfg.contrast_amount=.15;
        auto start=std::chrono::steady_clock::now();for(int i=0;i<100;++i)p.analyze(image.data(),128,72,512,cfg);
        std::printf("128x72 %s analysis + LUT %.3f ms/update\n",enabled?"chain":"legacy",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/100);}
    std::puts("entropy chain composition, preservation, independence, tone and gamut checks passed");
}
