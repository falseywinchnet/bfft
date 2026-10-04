// Frozen pre-optimization implementation for behavioral equivalence tests.
#include "entropy_stretch_v1.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace rvfx::entropy_v1 {
namespace {
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
    target_=current_;diagnostic_={};
}
bool Profile::analyze(const std::uint8_t* rgba,int width,int height,std::size_t stride,const Config& cfg){
    if(!rgba||width<1||height<1||width>512||height>512||stride<std::size_t(width)*4)return false;
    struct Sample {Color rgb;float h=0;bool valid=false;};
    std::vector<Sample> samples(std::size_t(width)*height);
    diagnostic_={};
    for(int y=0;y<height;++y)for(int x=0;x<width;++x){auto p=rgba+y*stride+4*x;auto& s=samples[y*width+x];
        // OBS render targets are premultiplied; omit weak/hidden alpha from statistics.
        if(p[3]<128)continue;s.valid=true;++diagnostic_.samples;
        for(int c=0;c<3;++c)s.rgb[c]=unit(float(p[c])/p[3]);}
    if(diagnostic_.samples<16)return false;
    for(int ty=0;ty<height;ty+=8)for(int tx=0;tx<width;tx+=8){
        std::array<int,64> ids{};int n=0;Color sum{},sq{};
        for(int y=ty;y<std::min(ty+8,height);++y)for(int x=tx;x<std::min(tx+8,width);++x){auto& s=samples[y*width+x];if(!s.valid)continue;
            int id=0;for(int c=0;c<3;++c){id=id*32+std::min(31,int(s.rgb[c]*32));sum[c]+=s.rgb[c];sq[c]+=s.rgb[c]*s.rgb[c];}ids[n++]=id;}
        if(n<4)continue;std::sort(ids.begin(),ids.begin()+n);double h=0;
        for(int i=0;i<n;){int j=i+1;while(j<n&&ids[j]==ids[i])++j;double p=double(j-i)/n;h-=p*std::log2(p);i=j;}
        double variance=0;for(int c=0;c<3;++c)variance+=std::max(0.f,sq[c]/n-(sum[c]/n)*(sum[c]/n))/3;
        const double floor=std::max(.001f,cfg.noise_floor);
        h=h/std::log2(double(n))*variance/(variance+floor*floor);
        for(int y=ty;y<std::min(ty+8,height);++y)for(int x=tx;x<std::min(tx+8,width);++x)samples[y*width+x].h=float(h);
    }
    std::array<double,512> population{},entropy_sum{};
    for(auto& s:samples)if(s.valid){int b=color_bin(s.rgb);population[b]++;entropy_sum[b]+=s.h;diagnostic_.mean_entropy+=s.h;}
    diagnostic_.mean_entropy/=diagnostic_.samples;
    Color mean{};std::array<double,3> mean_sum{};double total=0;Matrix moments{};
    for(auto& s:samples)if(s.valid){double w=.05+s.h*s.h;total+=w;
        for(int i=0;i<3;++i){mean_sum[i]+=w*s.rgb[i];for(int j=0;j<3;++j)moments[i][j]+=w*s.rgb[i]*s.rgb[j];}}
    for(int i=0;i<3;++i)mean[i]=float(mean_sum[i]/total);
    for(int i=0;i<3;++i)for(int j=0;j<3;++j)diagnostic_.covariance[i][j]=moments[i][j]/total-(mean_sum[i]/total)*(mean_sum[j]/total);
    auto m=whitening(diagnostic_.covariance,std::max(.001f,cfg.noise_floor),std::clamp(cfg.max_gain,1.f,16.f));
    for(int i=0;i<3;++i)for(int j=0;j<3;++j)m[i][j]=unit(cfg.decorrelation)*m[i][j]+(1-unit(cfg.decorrelation))*(i==j);
    diagnostic_.transform=m;
    std::array<std::array<double,bins>,3> hist{};
    for(auto& s:samples)if(s.valid){int b=color_bin(s.rgb);double n=population[b];
        // Conditional entropy, with a four-sample confidence prior. Total mass
        // per RGB family grows as n^.25, so a flat background cannot win by area.
        double e=entropy_sum[b]/(n+4);double weight=e*e/std::pow(n,.75);
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
    for(int g=0;g<side;++g)for(int b=0;b<side;++b)for(int r=0;r<side;++r){
        Color in{float(r)/(side-1),float(g)/(side-1),float(b)/(side-1)};
        auto out=transform(in,mean,m);auto i=index(r,g,b);
        for(int c=0;c<3;++c){float x=unit(out[c]);target_[i+c]=x+unit(cfg.allocation)*(curve(diagnostic_.curves[c],x)-x);}
        target_[i+3]=1;
    }
    return true;
}
bool Profile::advance(double seconds,const Config& cfg){
    if(!std::isfinite(seconds)||seconds<=0)return false;
    // A stalled render may not cause a large catch-up jump.
    seconds=std::min(seconds,.1);double alpha=-std::expm1(-seconds/std::max(.05f,cfg.adaptation_seconds));
    double largest=0;for(std::size_t i=0;i<current_.size();++i)largest=std::max(largest,double(std::abs(target_[i]-current_[i])));
    if(largest<1e-7)return false;
    alpha=std::min(alpha,std::max(.001f,cfg.max_change_per_second)*seconds/largest);
    for(std::size_t i=0;i<current_.size();++i)current_[i]+=float(alpha)*(target_[i]-current_[i]);return true;
}
Color Profile::apply(Color rgb)const{
    int lo[3];float t[3];for(int c=0;c<3;++c){float x=unit(rgb[c])*(side-1);lo[c]=std::min(side-2,int(x));t[c]=x-lo[c];}
    Color out{};for(int g=0;g<2;++g)for(int b=0;b<2;++b)for(int r=0;r<2;++r){float w=(r?t[0]:1-t[0])*(g?t[1]:1-t[1])*(b?t[2]:1-t[2]);auto i=index(lo[0]+r,lo[1]+g,lo[2]+b);for(int c=0;c<3;++c)out[c]+=w*current_[i+c];}return out;
}
} // namespace rvfx::entropy_v1
