#include "normalized_quartic.hpp"
#include "../../src/detail/bruun_dif_kernel.hpp"
#include <cstdio>
#include <random>
#include <limits>

static void require(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
int main() {
    using namespace quartic_walk;
    std::mt19937_64 rng(9506);std::uniform_real_distribution<double> uniform(-1,1);
    int checks=0;double worst=0,peak=0,oracle_peak=0;
    for(int n=4;n<=65536;n*=2) {
        bruun::DIF_RFFT_kernel dif;require(dif.init(n),"DIF init");
        std::vector<double> input(n),work(n+4),dw(n);
        std::vector<bruun::complex_t> out(n/2+1),reference(n/2+1),scratch(n/2+1);
        for(auto policy:{Policy::sibling,Policy::separated,Policy::random}) {
            int seeds=policy==Policy::random && n<=64?32:1;
            for(int seed=1;seed<=seeds;++seed) {
                for(bool fused:{false,true}) {
                Plan plan(n,policy,seed,fused);
                for(int signal=0;signal<(n<=64?n+5:5);++signal) {
                    for(int i=0;i<n;++i) {
                        if(signal==0) input[i]=uniform(rng);
                        else if(signal==1) input[i]=1;
                        else if(signal==2) input[i]=i%2?1:-1;
                        else if(signal==3) input[i]=std::sin(2*pi*3*i/n);
                        else if(signal==4) input[i]=0;
                        else input[i]=i==signal-5?1:0;
                    }
                    for(int i=0;i<4;++i) work[n+i]=1234567+i;
                    plan.forward(input.data(),out.data(),work.data());
                    dif.forward_standard(input.data(),reference.data(),dw.data(),scratch.data());
                    double err2=0,ref2=0;
                    for(int k=0;k<=n/2;++k) {
                        double dr=out[k].re-reference[k].re,di=out[k].im-reference[k].im;
                        require(std::isfinite(dr)&&std::isfinite(di),"nonfinite");
                        peak=std::max(peak,std::max(std::abs(dr),std::abs(di)));
                        err2+=dr*dr+di*di;ref2+=reference[k].re*reference[k].re+reference[k].im*reference[k].im;
                        if(n<=64) {
                            long double re=0,im=0;
                            for(int j=0;j<n;++j) {long double angle=2*acosl(-1.L)*j*k/n;re+=input[j]*cosl(angle);im-=input[j]*sinl(angle);}
                            double e=std::max(std::abs(double(re)-out[k].re),std::abs(double(im)-out[k].im));
                            oracle_peak=std::max(oracle_peak,e);require(e<2e-12*n,"direct DFT mismatch");
                        }
                    }
                    double relative=std::sqrt(err2/std::max(ref2,1e-300));
                    worst=std::max(worst,relative);require(relative<2e-12,"DIF mismatch");
                    for(int i=0;i<4;++i) require(work[n+i]==1234567+i,"work overflow");
                    ++checks;
                }
                }
            }
        }
    }
    bool rejected=false;try {Plan bad(12);}catch(const std::invalid_argument&) {rejected=true;}
    require(rejected,"invalid size admitted");
    std::printf("{\"checks\":%d,\"max_N\":65536,\"worst_relative_l2_vs_dif\":%.17g,\"max_abs_vs_dif\":%.17g,\"max_abs_vs_long_double_dft\":%.17g,\"simd_level\":%d}\n",checks,worst,peak,oracle_peak,BRUUN_LEVEL);
}
