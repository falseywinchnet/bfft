#include "generated_variants.hpp"
#include <algorithm>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>
using P=bruun::DIP_Experiment<0>;
void check(bool b,const char* message){if(!b)throw std::runtime_error(message);}
int main(){try{
    std::mt19937_64 rng(5471);std::uniform_real_distribution<double>u(-1,1);
    int cases=0;
    for(int q:{1,2,3,4,7,8,16,64,257}) for(double phi:{0.,1e-12,.01,.2,.5,.7853981633974483}) {
        std::vector<double>a(8*q+2,12345),b,x;
        for(int i=1;i<=8*q;++i)a[i]=u(rng);b=a;x=a;
        double c0=cos(2*phi),s0=sin(2*phi),c1=cos(phi),s1=sin(phi);
        P::cell4_fwd_ip(a.data()+1,q,c0,s0,c1,s1,s1,c1);
        P::cell4_gauge(b.data()+1,q,c1,s1,c0,s0,c0*c1-s0*s1,s0*c1+c0*s1);
        long double energy=0,out_energy=0;
        for(int i=1;i<=8*q;++i){check(std::abs(a[i]-b[i])<3e-14,"cell equivalence");energy+=(long double)x[i]*x[i];out_energy+=(long double)b[i]*b[i];}
        check(std::abs(out_energy-4*energy)<1e-12L*(1+energy),"scaled isometry");
        check(b.front()==12345&&b.back()==12345,"sentinel");
        P::cell4_inv_ip(b.data()+1,q,.5*c0,.5*s0,.5*c1,.5*s1,.5*s1,.5*c1);
        for(int i=1;i<=8*q;++i)check(std::abs(b[i]-x[i])<2e-14,"inverse cell");
        ++cases;
    }
    P p;check(!p.init(0)&&!p.init(3)&&!p.init(12),"invalid sizes");check(p.init(2048),"init");
    for(int w:{8,16,32}) for(int d=1;d<2048/w/2;++d){
        std::vector<double>a(2*w),b;for(auto&v:a)v=u(rng);b=a;
        p.fwd_span_reference(a.data(),w,d,2048/w);
        if(w==8)p.gauge8(b.data(),d);if(w==16)p.gauge16(b.data(),d);if(w==32)p.gauge32(b.data(),d);
        for(int i=0;i<2*w;++i)check(std::abs(a[i]-b[i])<2e-12,"terminal equivalence");++cases;
    }
    std::printf("PASS %d packet/cell cases: equivalence, inverse, energy, tails, sentinels, invalid sizes\n",cases);
}catch(const std::exception&e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
