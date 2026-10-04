#include "rvfx/entropy_stretch.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#if defined(__aarch64__) && !defined(RVFX_ENTROPY_SCALAR)
#include <arm_neon.h>
#endif

namespace rvfx::entropy {
namespace {
// A block contains at most 64 samples. Keep the original division/log/product
// and accumulation order, but evaluate these repeated terms once per process.
struct EntropyTerms {
    double count_log[65]{};
    double term[65][65]{};
    EntropyTerms() {
        for(int n=1;n<=64;++n) {
            count_log[n]=std::log2(double(n));
            for(int k=1;k<=n;++k) { double p=double(k)/n;term[n][k]=p*std::log2(p); }
        }
    }
};
float unit(float x) { return std::clamp(x,0.f,1.f); }
std::size_t index(int r,int g,int b) { return 4u*(g*side*side+b*side+r); }
int color_bin(Color c) { return std::min(7,int(c[0]*8))*64+std::min(7,int(c[1]*8))*8+std::min(7,int(c[2]*8)); }
Color transform(Color c, const Color& mean,const Matrix& m) {
    Color out=mean;
    for(int i=0;i<3;++i)for(int j=0;j<3;++j)out[i]+=float(m[i][j])*(c[j]-mean[j]);
    return out;
}
float curve(const std::array<float,bins+1>& f,float x) {
    float u=unit(x)*bins;int k=std::min(bins-1,int(u));return f[k]+(u-k)*(f[k+1]-f[k]);
}
float luma(Color c) { return .2126f*c[0]+.7152f*c[1]+.0722f*c[2]; }
Color mix_stage(Color in, Color out, float amount) {
    amount=unit(amount);
    if(amount==0)return in;
    if(amount==1)return out;
    for(int c=0;c<3;++c)out[c]=in[c]+amount*(out[c]-in[c]);
    return out;
}
// Move along the neutral axis; contract chroma only as much as the SDR cube
// requires. Neutral inputs stay neutral and no channel clips independently.
Color tone(Color in,float y) {
    float old=luma(in),scale=1;y=unit(y);
    for(int c=0;c<3;++c){float d=in[c]-old;
        if(d>0)scale=std::min(scale,(1-y)/d);
        else if(d<0)scale=std::min(scale,-y/d);}
    Color out;for(int c=0;c<3;++c)out[c]=unit(y+scale*(in[c]-old));
    return out;
}
float quantile(const std::array<double,bins>& hist,double count,double fraction) {
    double sum=0,wanted=count*fraction;
    for(int b=0;b<bins;++b){
        if(hist[b]>0&&sum+hist[b]>=wanted)
            return float((b+(wanted-sum)/hist[b])/bins);
        sum+=hist[b];
    }
    return 1;
}
}
Matrix whitening(const Matrix& covariance,double floor,double max_gain) {
    Matrix a=covariance,v{};for(int i=0;i<3;++i)v[i][i]=1;
    // Fixed bounded 3x3 Jacobi diagonalization, no external solver dependency.
    for(int sweep=0;sweep<12;++sweep)for(int p=0;p<3;++p)for(int q=p+1;q<3;++q){
        if(std::abs(a[p][q])<1e-15)continue;
        double angle=.5*std::atan2(2*a[p][q],a[q][q]-a[p][p]);
        double c=std::cos(angle),s=std::sin(angle);
        for(int k=0;k<3;++k){double x=a[k][p],y=a[k][q];a[k][p]=c*x-s*y;a[k][q]=s*x+c*y;}
        for(int k=0;k<3;++k){double x=a[p][k],y=a[q][k];a[p][k]=c*x-s*y;a[q][k]=s*x+c*y;}
        for(int k=0;k<3;++k){double x=v[k][p],y=v[k][q];v[k][p]=c*x-s*y;v[k][q]=s*x+c*y;}
    }
    double sigma=std::sqrt(std::max(0.,(covariance[0][0]+covariance[1][1]+covariance[2][2])/3));
    Matrix out{};
    for(int k=0;k<3;++k){double gain=std::clamp(sigma/std::sqrt(std::max(a[k][k],floor*floor)),.25,max_gain);
        for(int i=0;i<3;++i)for(int j=0;j<3;++j)out[i][j]+=v[i][k]*gain*v[j][k];}
    return out;
}
Profile::Profile(){reset();}
void Profile::reset(){
    current_.resize(side*side*side*4);
    for(int g=0;g<side;++g)for(int b=0;b<side;++b)for(int r=0;r<side;++r){auto i=index(r,g,b);
        current_[i]=float(r)/(side-1);current_[i+1]=float(g)/(side-1);current_[i+2]=float(b)/(side-1);current_[i+3]=1;}
    target_=current_;distance_=0;distance_dirty_=false;diagnostic_={};mean_={};
}
bool Profile::analyze(const std::uint8_t* rgba,int width,int height,std::size_t stride,const Config& cfg){
    if(!rgba||width<1||height<1||width>512||height>512||stride<std::size_t(width)*4)return false;
    samples_.resize(std::size_t(width)*height);auto& samples=samples_;
    static const EntropyTerms terms;
    diagnostic_={};
    for(int y=0;y<height;++y)for(int x=0;x<width;++x){auto p=rgba+y*stride+4*x;auto& s=samples[y*width+x];
        // OBS render targets are premultiplied; omit weak/hidden alpha from statistics.
        if(p[3]<128){s={};continue;}s.valid=true;s.h=0;++diagnostic_.samples;
        for(int c=0;c<3;++c)s.rgb[c]=unit(float(p[c])/p[3]);}
    if(diagnostic_.samples<16)return false;
    for(int ty=0;ty<height;ty+=8)for(int tx=0;tx<width;tx+=8){
        std::array<int,64> ids{};int n=0;Color sum{},sq{};
        for(int y=ty;y<std::min(ty+8,height);++y)for(int x=tx;x<std::min(tx+8,width);++x){auto& s=samples[y*width+x];if(!s.valid)continue;
            int id=0;for(int c=0;c<3;++c){id=id*32+std::min(31,int(s.rgb[c]*32));sum[c]+=s.rgb[c];sq[c]+=s.rgb[c]*s.rgb[c];}ids[n++]=id;}
        if(n<4)continue;std::sort(ids.begin(),ids.begin()+n);double h=0;
        for(int i=0;i<n;){int j=i+1;while(j<n&&ids[j]==ids[i])++j;h-=terms.term[n][j-i];i=j;}
        double variance=0;for(int c=0;c<3;++c)variance+=std::max(0.f,sq[c]/n-(sum[c]/n)*(sum[c]/n))/3;
        const double floor=std::max(.001f,cfg.noise_floor);
        h=h/terms.count_log[n]*variance/(variance+floor*floor);
        for(int y=ty;y<std::min(ty+8,height);++y)for(int x=tx;x<std::min(tx+8,width);++x)samples[y*width+x].h=float(h);
    }
    std::array<double,512> population{},entropy_sum{};
    for(auto& s:samples)if(s.valid){int b=color_bin(s.rgb);s.family=std::uint16_t(b);population[b]++;entropy_sum[b]+=s.h;diagnostic_.mean_entropy+=s.h;}
    diagnostic_.mean_entropy/=diagnostic_.samples;
    Color mean{};std::array<double,3> mean_sum{};double total=0;Matrix moments{};
    for(auto& s:samples)if(s.valid){double w=.05+s.h*s.h;total+=w;
        for(int i=0;i<3;++i){mean_sum[i]+=w*s.rgb[i];for(int j=0;j<3;++j)moments[i][j]+=w*s.rgb[i]*s.rgb[j];}}
    for(int i=0;i<3;++i)mean[i]=float(mean_sum[i]/total);
    for(int i=0;i<3;++i)for(int j=0;j<3;++j)diagnostic_.covariance[i][j]=moments[i][j]/total-(mean_sum[i]/total)*(mean_sum[j]/total);
    auto m=whitening(diagnostic_.covariance,std::max(.001f,cfg.noise_floor),std::clamp(cfg.max_gain,1.f,16.f));
    for(int i=0;i<3;++i)for(int j=0;j<3;++j)m[i][j]=unit(cfg.decorrelation)*m[i][j]+(1-unit(cfg.decorrelation))*(i==j);
    diagnostic_.transform=m;
    // One weight per occupied family instead of one pow/division per sample.
    std::array<double,512> family_weight{};
    for(int b=0;b<512;++b)if(population[b]>0) {
        double n=population[b],e=entropy_sum[b]/(n+4);
        family_weight[b]=e*e/std::pow(n,.75);
    }
    std::array<std::array<double,bins>,3> hist{};
    for(auto& s:samples)if(s.valid){double weight=family_weight[s.family];
        auto c=transform(s.rgb,mean,m);
        for(int k=0;k<3;++k)hist[k][std::min(bins-1,int(unit(c[k])*bins))]+=weight;}
    for(int c=0;c<3;++c){auto h=hist[c];for(int pass=0;pass<2;++pass){auto old=h;
        for(int b=0;b<bins;++b)h[b]=(old[std::max(0,b-1)]+2*old[b]+old[std::min(bins-1,b+1)])/4;}
        double sum=std::accumulate(h.begin(),h.end(),0.);auto& f=diagnostic_.curves[c];f[0]=0;
        if(sum<1e-10){for(int b=1;b<=bins;++b)f[b]=float(b)/bins;continue;}
        // Bounded positive range density. Renormalization retains monotonicity.
        double norm=0;for(auto& x:h){x=.2+.8*std::min(8.,x*bins/sum);norm+=x;}
        double acc=0;for(int b=0;b<bins;++b){acc+=h[b];f[b+1]=float(acc/norm);}f[bins]=1;
    }
    mean_=mean;
    if(cfg.chains){
        std::array<double,bins> population_y{},entropy_y{},density{};
        float low=1,high=0;
        for(const auto& s:samples)if(s.valid){float y=luma(s.rgb);int b=std::min(bins-1,int(unit(y)*bins));
            population_y[b]++;entropy_y[b]+=s.h;low=std::min(low,y);high=std::max(high,y);}
        diagnostic_.midtone=std::clamp(quantile(population_y,double(diagnostic_.samples),.5),.01f,.99f);
        // A flat or near-flat source supplies no reliable stretch evidence.
        diagnostic_.tone_active=high-low>std::max(.001f,cfg.noise_floor);
        float p=diagnostic_.midtone,reference=std::clamp(cfg.brightness_reference,.1f,.9f);
        diagnostic_.brightness_gain=diagnostic_.tone_active?
            std::clamp(reference*(1-p)/(p*(1-reference)),.25f,4.f):1.f;
        for(int b=0;b<bins;++b)if(population_y[b]>0){double n=population_y[b],e=entropy_y[b]/(n+4);
            density[b]=e*e*std::pow(n,.25);}
        for(int pass=0;pass<2;++pass){auto old=density;for(int b=0;b<bins;++b)
            density[b]=(old[std::max(0,b-1)]+2*old[b]+old[std::min(bins-1,b+1)])/4;}
        double total_density=std::accumulate(density.begin(),density.end(),0.);
        auto& f=diagnostic_.contrast_cdf;f[0]=0;
        if(total_density<1e-10||!diagnostic_.tone_active){for(int b=1;b<=bins;++b)f[b]=float(b)/bins;}
        else{
            double norm=0;for(auto& d:density){d=.2+.8*std::min(8.,d*bins/total_density);norm+=d;}
            double acc=0;for(int b=0;b<bins;++b){acc+=density[b];f[b+1]=float(acc/norm);}f[bins]=1;
        }
    }
    const bool legacy_target=!cfg.chains||(cfg.rgb_amount>=1&&cfg.brightness_amount<=0&&cfg.contrast_amount<=0);
    if(!legacy_target){
        for(int g=0;g<side;++g)for(int b=0;b<side;++b)for(int r=0;r<side;++r){
            Color in{float(r)/(side-1),float(g)/(side-1),float(b)/(side-1)};
            auto out=target_color(in,cfg);auto i=index(r,g,b);
            for(int c=0;c<3;++c)target_[i+c]=out[c];target_[i+3]=1;
        }
    }else{
        // Keep the legacy hot loop separate: optional composition must not
        // inhibit its vectorization or change its floating-point evaluation.
        for(int g=0;g<side;++g)for(int b=0;b<side;++b)for(int r=0;r<side;++r){
            Color in{float(r)/(side-1),float(g)/(side-1),float(b)/(side-1)};
            auto out=transform(in,mean,m);auto i=index(r,g,b);
            for(int c=0;c<3;++c){float x=unit(out[c]);target_[i+c]=x+unit(cfg.allocation)*(curve(diagnostic_.curves[c],x)-x);}
            target_[i+3]=1;
        }
    }
    distance_dirty_=true;
    return true;
}
Color Profile::target_color(Color rgb,const Config& cfg)const{
    if(diagnostic_.samples<16)return rgb;
    auto color=[&](Color in){
        auto out=transform(in,mean_,diagnostic_.transform);
        for(int c=0;c<3;++c){float x=unit(out[c]);out[c]=x+unit(cfg.allocation)*(curve(diagnostic_.curves[c],x)-x);}
        return cfg.chains?mix_stage(in,out,cfg.rgb_amount):out;
    };
    if(!cfg.chains)return color(rgb);
    auto contrast=[&](Color in){
        if(cfg.contrast_amount<=0||!diagnostic_.tone_active)return in;
        float p=diagnostic_.midtone,y=luma(in),fp=curve(diagnostic_.contrast_cdf,p),fy=curve(diagnostic_.contrast_cdf,y);
        float out=y<=p?p*fy/fp:p+(1-p)*(fy-fp)/(1-fp);
        return mix_stage(in,tone(in,out),cfg.contrast_amount);
    };
    auto brightness=[&](Color in){
        if(cfg.brightness_amount<=0||diagnostic_.brightness_gain==1)return in;
        float y=luma(in),gain=diagnostic_.brightness_gain;
        return mix_stage(in,tone(in,gain*y/(1+(gain-1)*y)),cfg.brightness_amount);
    };
    // Zero RGB amount is a true bypass, including the legacy RGB clipping.
    auto rgb_stage=[&](Color in){return cfg.rgb_amount<=0?in:color(in);};
    switch(cfg.chain_order){
        case ChainOrder::ColorBrightnessContrast:return contrast(brightness(rgb_stage(rgb)));
        case ChainOrder::ContrastColorBrightness:return brightness(rgb_stage(contrast(rgb)));
        case ChainOrder::ContrastBrightnessColor:return rgb_stage(brightness(contrast(rgb)));
        case ChainOrder::BrightnessColorContrast:return contrast(rgb_stage(brightness(rgb)));
        case ChainOrder::BrightnessContrastColor:return rgb_stage(contrast(brightness(rgb)));
        default:return brightness(contrast(rgb_stage(rgb)));
    }
}
bool Profile::advance(double seconds,const Config& cfg){
    return advance_impl(seconds,cfg,nullptr);
}
bool Profile::advance_and_pack(double seconds,const Config& cfg,std::vector<std::uint16_t>& packed){
    return advance_impl(seconds,cfg,&packed);
}
bool Profile::advance_impl(double seconds,const Config& cfg,std::vector<std::uint16_t>* packed){
    if(!std::isfinite(seconds)||seconds<=0)return false;
    seconds=std::min(seconds,.1);
    double alpha=-std::expm1(-seconds/std::max(.05f,cfg.adaptation_seconds));
    const std::size_t n=current_.size();
    if(distance_dirty_){
    double largest=0;
#if defined(__aarch64__) && !defined(RVFX_ENTROPY_SCALAR)
    // max is order-independent here: all LUT entries are finite [0,1].
    float32x4_t maximum=vdupq_n_f32(0);
    for(std::size_t i=0;i<n;i+=4)
        maximum=vmaxq_f32(maximum,vabsq_f32(vsubq_f32(vld1q_f32(target_.data()+i),vld1q_f32(current_.data()+i))));
    largest=vmaxvq_f32(maximum);
#else
    for(std::size_t i=0;i<n;++i)largest=std::max(largest,double(std::abs(target_[i]-current_[i])));
#endif
    distance_=float(largest);distance_dirty_=false;
    }
    const double largest=distance_;
    const bool advance=largest>=1e-7;
    if(advance)alpha=std::min(alpha,std::max(.001f,cfg.max_change_per_second)*seconds/largest);
    else if(!packed)return false;
    bool changed=false;
    if(packed&&packed->size()!=n){packed->resize(n);changed=true;}
#if defined(__aarch64__) && !defined(RVFX_ENTROPY_SCALAR)
    const float32x4_t rate=vdupq_n_f32(float(alpha));
    uint16x4_t changes=vdup_n_u16(0);
    float32x4_t next_distance=vdupq_n_f32(0);
    for(std::size_t i=0;i<n;i+=4){
        auto value=vld1q_f32(current_.data()+i);
        if(advance){
            // Clang's original ARM64 scalar expression uses the same FMA.
            value=vfmaq_f32(value,rate,vsubq_f32(vld1q_f32(target_.data()+i),value));
            vst1q_f32(current_.data()+i,value);
            next_distance=vmaxq_f32(next_distance,vabsq_f32(vsubq_f32(vld1q_f32(target_.data()+i),value)));
        }
        if(packed){
            // The original pack uses a fused multiply-add too; narrowing is
            // exact because the input is bounded to 0..65535 after truncation.
            auto rounded=vfmaq_n_f32(vdupq_n_f32(.5f),value,65535.f);
            auto codes=vmovn_u32(vcvtq_u32_f32(rounded));
            changes=vorr_u16(changes,veor_u16(codes,vld1_u16(packed->data()+i)));
            vst1_u16(packed->data()+i,codes);
        }
    }
    if(advance)distance_=vmaxvq_f32(next_distance);
    if(packed)changed=changed||vmaxv_u16(changes)!=0;
#else
    float next_distance=0;
    for(std::size_t i=0;i<n;++i){
        if(advance){current_[i]+=float(alpha)*(target_[i]-current_[i]);next_distance=std::max(next_distance,std::abs(target_[i]-current_[i]));}
        if(packed){auto code=std::uint16_t(current_[i]*65535.f+.5f);changed=changed||code!=(*packed)[i];(*packed)[i]=code;}
    }
    if(advance)distance_=next_distance;
#endif
    // Do not stop the float recurrence when quantization hides its motion.
    // Accumulated sub-code updates still become visible at the original time.
    return packed?changed:advance;
}
Color Profile::apply(Color rgb)const{
    int lo[3];float t[3];for(int c=0;c<3;++c){float x=unit(rgb[c])*(side-1);lo[c]=std::min(side-2,int(x));t[c]=x-lo[c];}
    Color out{};for(int g=0;g<2;++g)for(int b=0;b<2;++b)for(int r=0;r<2;++r){float w=(r?t[0]:1-t[0])*(g?t[1]:1-t[1])*(b?t[2]:1-t[2]);auto i=index(lo[0]+r,lo[1]+g,lo[2]+b);for(int c=0;c<3;++c)out[c]+=w*current_[i+c];}return out;
}
} // namespace rvfx::entropy
