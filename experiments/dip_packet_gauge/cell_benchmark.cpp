#include "generated_variants.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>
#ifdef __APPLE__
#include <pthread.h>
#endif
volatile double cell_sink=0;
int main(){
#ifdef __APPLE__
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE,0);
#endif
    using P=bruun::DIP_Experiment<0>;using Clock=std::chrono::steady_clock;
    const double phi=.23,c1=cos(phi),s1=sin(phi),c2=cos(2*phi),s2=sin(2*phi),c3=cos(3*phi),s3=sin(3*phi);
    for(int q:{1,2,4,8,16,32,64,128,256,512,1024,4096,16384}) {
        std::vector<double>x(8*q),v(8*q);for(size_t i=0;i<x.size();++i)x[i]=sin(double(i));
        const int iters=std::max(100,1000000/q);std::vector<double>old,newer;
        for(int round=0;round<15;++round)for(int t=0;t<2;++t){
            const bool gauge=(t+round)%2;const auto start=Clock::now();
            for(int i=0;i<iters;++i){
                std::memcpy(v.data(),x.data(),x.size()*sizeof(double));
                if(gauge)P::cell4_gauge(v.data(),q,c1,s1,c2,s2,c3,s3);
                else P::cell4_fwd_ip(v.data(),q,c2,s2,c1,s1,s1,c1);
                cell_sink=v[i%v.size()];
            }
            (gauge?newer:old).push_back(std::chrono::duration<double,std::nano>(Clock::now()-start).count()/iters);
        }
        std::sort(old.begin(),old.end());std::sort(newer.begin(),newer.end());
        printf("{\"q\":%d,\"old_ns\":%.6f,\"gauge_ns\":%.6f,\"speedup\":%.6f}\n",q,old[7],newer[7],old[7]/newer[7]);fflush(stdout);
    }
}
