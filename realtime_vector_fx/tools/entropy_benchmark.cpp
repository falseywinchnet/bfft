#include "rvfx/entropy_stretch.hpp"
#include "../tests/reference/entropy_stretch_v1.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
using Clock=std::chrono::steady_clock;
using namespace rvfx;
void require(bool okay,const char* what){if(!okay){std::fprintf(stderr,"FAIL: %s\n",what);std::exit(1);}}
template<class T> bool same_bits(const T& a,const T& b){return std::memcmp(&a,&b,sizeof(T))==0;}
bool same_lut(const std::vector<float>& a,const std::vector<float>& b){return a.size()==b.size()&&std::memcmp(a.data(),b.data(),a.size()*sizeof(float))==0;}
std::vector<uint8_t> input(int w,int h,int seed,int kind){std::mt19937 rng(seed);std::vector<uint8_t> v(h*(w*4+16));
    for(int y=0;y<h;++y)for(int x=0;x<w;++x){auto* p=v.data()+y*(w*4+16)+4*x;
        p[3]=kind==3?uint8_t(rng()%256):255;
        for(int c=0;c<3;++c){int q=kind==0?35+(x+y)%9:kind==1?(x*3+y*5+c*7)%256:kind==2?int(rng()%256):int(rng()%(p[3]+1));p[c]=uint8_t(q);}}
    return v;
}
struct Summary{double median,min,max;};
Summary summarize(std::vector<double> a){std::sort(a.begin(),a.end());return {a[a.size()/2],a.front(),a.back()};}
template<class Fn> double time_us(Fn fn,int repeats){auto start=Clock::now();for(int i=0;i<repeats;++i)fn(i);return std::chrono::duration<double,std::micro>(Clock::now()-start).count()/repeats;}
int main(int argc,char** argv){
    std::size_t compared=0;
    for(auto dims: {std::pair<int,int>{32,19},{128,72},{256,144},{13,7}})for(int kind=0;kind<4;++kind){
        entropy::Profile p;entropy_v1::Profile ref;entropy::Config cfg;entropy_v1::Config old;
        for(int frame=0;frame<36;++frame){
            if(frame%3==0){auto v=input(dims.first,dims.second,frame+kind*701,kind);
                float d=frame%2?.04f:.65f,a=frame%2?.07f:.8f;cfg.decorrelation=old.decorrelation=d;cfg.allocation=old.allocation=a;
                bool one=p.analyze(v.data(),dims.first,dims.second,dims.first*4+16,cfg);
                bool two=ref.analyze(v.data(),dims.first,dims.second,dims.first*4+16,old);require(one==two,"analysis acceptance");
                const auto& x=p.diagnostics();const auto& y=ref.diagnostics();
                require(x.samples==y.samples&&same_bits(x.mean_entropy,y.mean_entropy)&&same_bits(x.covariance,y.covariance)&&same_bits(x.transform,y.transform)&&same_bits(x.curves,y.curves),"bit-exact analysis diagnostics");
            }
            double dt=frame%7==0?.1:1./(frame%2?30:60);require(p.advance(dt,cfg)==ref.advance(dt,old),"advance acceptance");
            require(same_lut(p.atlas(),ref.atlas()),"bit-exact temporal LUT");compared+=p.atlas().size();
        }
    }
    std::size_t skipped=0;
    {
        entropy::Profile p;entropy_v1::Profile ref;entropy::Config cfg;entropy_v1::Config old;
        std::vector<uint16_t> packed(p.atlas().size()),expected(p.atlas().size()),previous;
        for(std::size_t i=0;i<packed.size();++i)packed[i]=expected[i]=uint16_t(p.atlas()[i]*65535.f+.5f);
        for(int frame=0;frame<1200;++frame){
            if(frame%180==0){auto v=input(128,72,frame+911,frame/180%4);
                cfg.decorrelation=old.decorrelation=.09f;cfg.allocation=old.allocation=.11f;cfg.max_gain=old.max_gain=16;
                cfg.adaptation_seconds=old.adaptation_seconds=.05f;
                cfg.max_change_per_second=old.max_change_per_second=2;
                p.analyze(v.data(),128,72,528,cfg);ref.analyze(v.data(),128,72,528,old);
            }
            double dt=frame%41==0?0.:1./60;
            previous=expected;bool upload=p.advance_and_pack(dt,cfg,packed);ref.advance(dt,old);
            for(std::size_t i=0;i<expected.size();++i)expected[i]=uint16_t(ref.atlas()[i]*65535.f+.5f);
            require(same_lut(p.atlas(),ref.atlas()),"packed path retains bit-exact float recurrence");
            require(packed==expected,"bit-exact UNORM16 upload");
            require(upload==(previous!=expected),"upload skipped iff GPU contents are identical");
            skipped+=!upload;compared+=p.atlas().size();
        }
        require(skipped>100,"steady profiles avoid redundant uploads");
    }
    std::printf("{\"equivalence_float_values\":%zu,\"bit_exact\":true,\"cases\":[\n",compared);
    if(argc>1&&std::strcmp(argv[1],"--verify-only")==0){std::puts("]}");return 0;}
    bool first=true;
    for(auto dims:{std::pair<int,int>{128,72},{256,144}}){
        auto v=input(dims.first,dims.second,17,2);entropy::Profile p;entropy_v1::Profile ref;entropy::Config cfg;entropy_v1::Config old;
        cfg.decorrelation=old.decorrelation=.09f;cfg.allocation=old.allocation=.11f;cfg.max_gain=old.max_gain=16;
        std::vector<double> a,b,c,d;for(int run=0;run<31;++run){
            auto opt=[&](int){p.analyze(v.data(),dims.first,dims.second,dims.first*4+16,cfg);};
            auto base=[&](int){ref.analyze(v.data(),dims.first,dims.second,dims.first*4+16,old);};
            if(run%2){a.push_back(time_us(opt,50));b.push_back(time_us(base,50));}else{b.push_back(time_us(base,50));a.push_back(time_us(opt,50));}
            p.reset();ref.reset();opt(0);base(0);
            auto step=[&](int){p.advance(1./60,cfg);};auto orig=[&](int){ref.advance(1./60,old);};
            if(run%2){c.push_back(time_us(step,40));d.push_back(time_us(orig,40));}else{d.push_back(time_us(orig,40));c.push_back(time_us(step,40));}
        }
        auto sa=summarize(a),sb=summarize(b),sc=summarize(c),sd=summarize(d);
        std::printf("%s{\"width\":%d,\"height\":%d,\"analysis_us\":{\"before\":%.3f,\"after\":%.3f},\"advance_us\":{\"before\":%.3f,\"after\":%.3f}}",first?"":",\n",dims.first,dims.second,sb.median,sa.median,sd.median,sc.median);first=false;
    }
    std::puts("\n]}");
}
