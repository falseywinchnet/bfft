#include "generated_variants.hpp"
#include "../../src/detail/bruun_dif_kernel.hpp"
#include "../../src/detail/bruun_dit_kernel.hpp"
#include <algorithm>
#include <chrono>
#include <complex>
#include <cstdio>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef __APPLE__
#include <pthread.h>
#endif

using Clock=std::chrono::steady_clock;
using C=bruun::complex_t;
volatile double sink=0;
void require(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
double error(const std::vector<C>& a,const std::vector<C>& b) {
    double err=0,scale=1;
    for(size_t i=0;i<a.size();++i) {
        require(std::isfinite(a[i].re)&&std::isfinite(a[i].im),"nonfinite spectrum");
        err=std::max(err,std::hypot(a[i].re-b[i].re,a[i].im-b[i].im));
        scale=std::max(scale,std::hypot(b[i].re,b[i].im));
    }
    return err/scale;
}
struct Case {std::string name;std::function<void()> run;std::vector<double> times;double err=0,rt=0;};
template<class P> void add(std::vector<Case>& cases,const char* name,P& p,std::vector<double>& x,std::vector<C>& y,std::vector<double>& w) {
    cases.push_back({name,[&]{p.forward_standard(x.data(),y.data(),w.data());},{}});
}
// A no-mutation numerical oracle for arbitrary complex packet coefficients.
void packet_test() {
    bruun::DIP_Experiment<0> p; require(p.init(2048),"packet plan");
    std::mt19937_64 rng(42);std::uniform_real_distribution<double> u(-1,1);
    for(int w:{8,16,32}) for(int d=1;d<2048/w/2;++d) {
        std::vector<double> a(2*w),b;for(auto&v:a)v=u(rng);b=a;
        p.fwd_span_reference(a.data(),w,d,2048/w);
        if(w==8)p.gauge8(b.data(),d);if(w==16)p.gauge16(b.data(),d);if(w==32)p.gauge32(b.data(),d);
        for(int j=0;j<2*w;++j)require(std::abs(a[j]-b[j])<2e-12,"packet gauge identity");
    }
}
int main(int argc,char**argv) {try {
#ifdef __APPLE__
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE,0);
#endif
    packet_test();
    const int maxlog=argc>1?std::atoi(argv[1]):20;
    const int rounds=argc>2?std::atoi(argv[2]):11;
    for(int logn=argc>3?std::atoi(argv[3]):2;logn<=maxlog;++logn) {
        const int n=1<<logn;
        bruun::DIP_RFFT_kernel dip; bruun::DIF_RFFT_kernel dif; bruun::DIT_RFFT_kernel dit;
        bruun::DIP_Experiment<0> audit; bruun::DIP_Experiment<1> fused;
        bruun::DIP_Experiment<2> g8;bruun::DIP_Experiment<3> g16;bruun::DIP_Experiment<4> g32; bruun::DIP_Experiment<5> core; bruun::DIP_Experiment<6> hybrid; bruun::DIP_Experiment<7> unfolded; bruun::DIP_Experiment<8> selective;
        require(dip.init(n)&&dif.init(n)&&dit.init(n)&&audit.init(n)&&fused.init(n)&&g8.init(n)&&g16.init(n)&&g32.init(n)&&core.init(n)&&hybrid.init(n)&&unfolded.init(n)&&selective.init(n),"init");
        std::vector<double>x(n),w(n),back(n);std::vector<C>y(n/2+1),ref(y.size()),tmp(y.size());
        std::mt19937_64 rng(123+n);std::uniform_real_distribution<double>u(-1,1);
        std::vector<Case>cases;
        add(cases,"dip",dip,x,y,w);add(cases,"dip_clone",audit,x,y,w);add(cases,"fused8",fused,x,y,w);
        add(cases,"gauge8",g8,x,y,w);add(cases,"gauge16",g16,x,y,w);add(cases,"gauge32",g32,x,y,w);
        add(cases,"gauge_core",core,x,y,w);add(cases,"gauge_hybrid",hybrid,x,y,w);
        add(cases,"unfolded_packet",unfolded,x,y,w);
        add(cases,"gauge_selective",selective,x,y,w);
        cases.push_back({"dif",[&]{dif.forward_standard(x.data(),y.data(),w.data(),tmp.data());},{}});
        cases.push_back({"dit",[&]{dit.forward_simd(x.data(),y.data(),w.data());},{}});
        for(int pattern=0;pattern<8;++pattern) {
            for(int j=0;j<n;++j) {
                if(pattern==0)x[j]=j==n/3?1:0;
                else if(pattern==1)x[j]=1;
                else if(pattern==2)x[j]=(j&1)?-1:1;
                else if(pattern==3)x[j]=std::sin(6.283185307179586*7*j/n)+.3*std::cos(6.283185307179586*13*j/n);
                else x[j]=u(rng)*(pattern==5?1e-100:pattern==6?1e100:1);
            }
            dit.forward_simd(x.data(),ref.data(),w.data());
            if(pattern==7) {
                // Direct long-double DFT: all small bins, sampled large bins.
                for(int k=0;k<=n/2;k+=(n<=256?1:std::max(1,n/32))) {
                    std::complex<long double> sum=0;
                    for(int j=0;j<n;++j) {
                        long double phase=-2*acosl(-1.L)*k*j/n;
                        sum+=static_cast<long double>(x[j])*std::complex<long double>(cosl(phase),sinl(phase));
                    }
                    require(std::abs(sum-std::complex<long double>(ref[k].re,ref[k].im))<2e-11L*n,"direct DFT");
                }
            }
            for(auto& c:cases) {
                c.run();c.err=std::max(c.err,error(y,ref));require(c.err<2e-12,"forward mismatch");
                dip.inverse_standard(y.data(),back.data(),w.data());double err=0,scale=1e-300;
                for(int j=0;j<n;++j) {require(std::isfinite(back[j]),"nonfinite inverse");err=std::max(err,std::abs(back[j]-x[j]));scale=std::max(scale,std::abs(x[j]));}
                c.rt=std::max(c.rt,err/scale);require(c.rt<2e-12,"roundtrip mismatch");
            }
        }
        // Identical finite input on each call; no recurrent FFT overflow.
        for(auto&v:x)v=u(rng);
        int iters=std::max(10,std::min(100000,12000000/n));
        for(auto& c:cases)for(int i=0;i<4;++i)c.run();
        for(int r=0;r<rounds;++r) for(size_t j=0;j<cases.size();++j) {
            auto& c=cases[(j+r)%cases.size()];auto begin=Clock::now();
            for(int it=0;it<iters;++it){c.run();sink=y[(it% (n/2+1))].re;}
            c.times.push_back(std::chrono::duration<double,std::nano>(Clock::now()-begin).count()/iters);
        }
        for(auto&c:cases) {
            auto raw_times=c.times;
            std::sort(c.times.begin(),c.times.end());
            std::printf("{\"n\":%d,\"method\":\"%s\",\"ns\":%.6f,\"min_ns\":%.6f,\"max_ns\":%.6f,\"relative_error\":%.6g,\"roundtrip_error\":%.6g,\"iters\":%d,\"rounds\":%d,\"samples_ns\":[",n,c.name.c_str(),c.times[rounds/2],c.times.front(),c.times.back(),c.err,c.rt,iters,rounds);
            for(size_t i=0;i<raw_times.size();++i)std::printf("%s%.6f",i?",":"",raw_times[i]);
            std::printf("]}\n");
        }
        std::fflush(stdout);
    }
    return 0;
}catch(const std::exception&e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}}
