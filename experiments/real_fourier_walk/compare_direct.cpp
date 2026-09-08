// Reuse the repository's DIF/DIP paired benchmark utilities without its main.
#define main original_dif_dip_benchmark_main
#include "../../examples/dif_vs_dip_benchmark.cpp"
#undef main
#include "normalized_quartic.hpp"

int main() {
    for(int n:{64,128,4096}) {
        bruun::DIF_RFFT_kernel dif;if(!dif.init(n))return 1;
        quartic_walk::Plan walk(n);
        auto input=make_signal(n);
        std::vector<double> dw(n),ww(n);
        std::vector<bruun::complex_t> d(n/2+1),w(n/2+1),scratch(n/2+1);
        for(int run=0;run<5;++run) {
            auto fd=[&](int i,double) {input[(i*131u)&(n-1)]+=1e-12;dif.forward_standard(input.data(),d.data(),dw.data(),scratch.data());return d[(i*17u)%(n/2+1)].re;};
            auto fw=[&](int i,double) {input[(i*131u)&(n-1)]+=1e-12;walk.forward(input.data(),w.data(),ww.data());return w[(i*17u)%(n/2+1)].re;};
            int iters=std::max(2000,8000000/n);
            auto ab=bench_pair(iters,fd,fw),ba=bench_pair(iters,fw,fd);
            double dns=(ab.a_ns+ba.b_ns)/2,wns=(ab.b_ns+ba.a_ns)/2;
            std::printf("{\"N\":%d,\"run\":%d,\"direct_dif_ns\":%.6f,\"walk_ns\":%.6f,\"ratio\":%.6f,\"sink\":%.6f}\n",n,run,dns,wns,wns/dns,ab.sink+ba.sink);
        }
    }
}
