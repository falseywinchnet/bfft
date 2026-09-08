#pragma once
// Experimental normalized quartic packet plan. No production dispatch changes.
#include "../../src/detail/bruun_simd_backend.hpp"
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace quartic_walk {
#if defined(__GNUC__) || defined(__clang__)
#define QW_REGISTER_INLINE inline __attribute__((always_inline))
#else
#define QW_REGISTER_INLINE inline
#endif
constexpr double pi = 3.141592653589793238462643383279502884;
enum class Policy { sibling, separated, random };
struct Op {
    int base, w, choice, ridge; // ridge = source pair index, or -1
    double c[2], s[2];
    int bins[4]; // source-child order, zero means the DC/Nyquist ridge
};
class Plan {
    int n_;
    std::uint64_t rng_;
    Policy policy_;
    bool fused_;
    std::vector<Op> ops_;
    static constexpr int routes_[3][4] = {{0,1,2,3},{0,2,1,3},{0,3,1,2}};
    void build(int base, int h, double a, double b) {
        if (h == 1) return;
        Op op{}; op.base=base; op.w=h/2; op.ridge=-1;
        const double angles[2]={a,b}; double child[4];
        for(int p=0;p<2;++p) {
            if(angles[p]<0) {
                op.ridge=p; child[2*p]=-1; child[2*p+1]=pi/2;
                op.c[p]=1; op.s[p]=0;
            } else {
                child[2*p]=angles[p]/2; child[2*p+1]=pi-angles[p]/2;
                op.c[p]=std::cos(child[2*p]); op.s[p]=std::sin(child[2*p]);
            }
        }
        op.choice=0;
        if(policy_==Policy::random) {
            rng_^=rng_<<13; rng_^=rng_>>7; rng_^=rng_<<17;
            op.choice=int(rng_%3);
        } else if(policy_==Policy::separated && op.ridge<0) {
            double best=-1;
            for(int route=0;route<3;++route) {
                auto r=routes_[route];
                double score=std::min(std::abs(std::cos(child[r[0]])-std::cos(child[r[1]])),
                                      std::abs(std::cos(child[r[2]])-std::cos(child[r[3]])));
                if(score>best) {best=score;op.choice=route;}
            }
        }
        for(int j=0;j<4;++j) op.bins[j]=child[j]<0?0:int(std::llround(child[j]*n_/(2*pi)));
        ops_.push_back(op);
        auto r=routes_[op.choice];
        build(base,h/2,child[r[0]],child[r[1]]);
        build(base+2*h,h/2,child[r[2]],child[r[3]]);
    }
    template<int R> static inline void stream(double* v,const Op& o) {
        const int w=o.w; int i=0;
        // Lanes are adjacent real columns, as in BFFT; no lane reductions.
#if BRUUN_LEVEL >= 1
        const auto c0=V2_SET1(o.c[0]),s0=V2_SET1(o.s[0]);
        const auto c1=V2_SET1(o.c[1]),s1=V2_SET1(o.s[1]);
        for(;i+1<w;i+=2) {
            bruun_v2 y[8];
            const auto a=V2_LD(v+i),b=V2_LD(v+w+i),c=V2_LD(v+2*w+i),d=V2_LD(v+3*w+i);
            const auto e=V2_LD(v+4*w+i),f=V2_LD(v+5*w+i),g=V2_LD(v+6*w+i),h=V2_LD(v+7*w+i);
            if(o.ridge==0) {
                y[0]=V2_ADD(a,c);y[1]=V2_ADD(b,d);y[2]=V2_SUB(a,c);y[3]=V2_SUB(b,d);
            } else {
                auto re=V2_MSUB(V2_MUL(c0,b),s0,d),im=V2_MADD(V2_MUL(s0,b),c0,d);
                y[0]=V2_ADD(a,re);y[1]=V2_ADD(c,im);y[2]=V2_SUB(a,re);y[3]=V2_SUB(im,c);
            }
            if(o.ridge==1) {
                y[4]=V2_ADD(e,g);y[5]=V2_ADD(f,h);y[6]=V2_SUB(e,g);y[7]=V2_SUB(f,h);
            } else {
                auto re=V2_MSUB(V2_MUL(c1,f),s1,h),im=V2_MADD(V2_MUL(s1,f),c1,h);
                y[4]=V2_ADD(e,re);y[5]=V2_ADD(g,im);y[6]=V2_SUB(e,re);y[7]=V2_SUB(im,g);
            }
            // Compile-time routing: all eight inputs loaded before any store.
            for(int j=0;j<4;++j) {
                V2_ST(v+(2*j)*w+i,y[2*routes_[R][j]]);
                V2_ST(v+(2*j+1)*w+i,y[2*routes_[R][j]+1]);
            }
        }
#endif
        for(;i<w;++i) {
            double y[8];
            for(int p=0;p<2;++p) {
                auto x=v+4*p*w+i;
                if(o.ridge==p) {y[4*p]=x[0]+x[2*w];y[4*p+1]=x[w]+x[3*w];y[4*p+2]=x[0]-x[2*w];y[4*p+3]=x[w]-x[3*w];}
                else {double re=o.c[p]*x[w]-o.s[p]*x[3*w],im=o.s[p]*x[w]+o.c[p]*x[3*w];
                    y[4*p]=x[0]+re;y[4*p+1]=x[2*w]+im;y[4*p+2]=x[0]-re;y[4*p+3]=im-x[2*w];}
            }
            for(int j=0;j<4;++j) {v[2*j*w+i]=y[2*routes_[R][j]];v[(2*j+1)*w+i]=y[2*routes_[R][j]+1];}
        }
    }
    template<class Complex> inline void leaf(const double* v,const Op& o,Complex* out) const {
        double y[8];
#if BRUUN_LEVEL >= 1
        if(o.ridge<0) {
            auto ab=V2_LD(v),cd=V2_LD(v+2),ef=V2_LD(v+4),gh=V2_LD(v+6);
            auto a=V2_UNPLO(ab,ef),b=V2_UNPHI(ab,ef),c=V2_UNPLO(cd,gh),d=V2_UNPHI(cd,gh);
            auto C=V2_LD(o.c),S=V2_LD(o.s);
            auto re=V2_MSUB(V2_MUL(C,b),S,d),im=V2_MADD(V2_MUL(S,b),C,d);
            auto lr=V2_ADD(a,re),li=V2_ADD(c,im),hr=V2_SUB(a,re),hi=V2_SUB(im,c);
            V2_ST(y,V2_UNPLO(lr,li));V2_ST(y+4,V2_UNPHI(lr,li));
            V2_ST(y+2,V2_UNPLO(hr,hi));V2_ST(y+6,V2_UNPHI(hr,hi));
        } else
#endif
        {
            for(int p=0;p<2;++p) {
                auto x=v+4*p;
                if(o.ridge==p) {y[4*p]=x[0]+x[2];y[4*p+1]=x[1]+x[3];y[4*p+2]=x[0]-x[2];y[4*p+3]=x[1]-x[3];}
                else {double re=o.c[p]*x[1]-o.s[p]*x[3],im=o.s[p]*x[1]+o.c[p]*x[3];
                    y[4*p]=x[0]+re;y[4*p+1]=x[2]+im;y[4*p+2]=x[0]-re;y[4*p+3]=im-x[2];}
            }
        }
        for(int j=0;j<4;++j) {
            int k=o.bins[j];
            if(!k) {out[0]={y[2*j]+y[2*j+1],0};out[n_/2]={y[2*j]-y[2*j+1],0};}
            else out[k]={y[2*j],-y[2*j+1]};
        }
    }
#if BRUUN_LEVEL >= 1
    template<class Complex> QW_REGISTER_INLINE void leaf_registers(bruun_v2 ab,bruun_v2 cd,bruun_v2 ef,bruun_v2 gh,const Op& o,Complex* out) const {
        if(o.ridge>=0) {
            double v[8];V2_ST(v,ab);V2_ST(v+2,cd);V2_ST(v+4,ef);V2_ST(v+6,gh);
            leaf(v,o,out);return;
        }
        auto a=V2_UNPLO(ab,ef),b=V2_UNPHI(ab,ef),c=V2_UNPLO(cd,gh),d=V2_UNPHI(cd,gh);
        auto C=V2_LD(o.c),S=V2_LD(o.s);
        auto re=V2_MSUB(V2_MUL(C,b),S,d),im=V2_MADD(V2_MUL(S,b),C,d);
        auto lr=V2_ADD(a,re),li=V2_ADD(c,im),hr=V2_SUB(a,re),hi=V2_SUB(im,c);
        double y[8];
        V2_ST(y,V2_UNPLO(lr,li));V2_ST(y+4,V2_UNPHI(lr,li));
        V2_ST(y+2,V2_UNPLO(hr,hi));V2_ST(y+6,V2_UNPHI(hr,hi));
        for(int j=0;j<4;++j) out[o.bins[j]]={y[2*j],-y[2*j+1]};
    }
    template<int R,class Complex> inline void codelet16(const double* v,const Op& o,const Op& left,const Op& right,Complex* out) const {
        bruun_v2 y[8];
        const auto a=V2_LD(v),b=V2_LD(v+2),c=V2_LD(v+4),d=V2_LD(v+6);
        const auto e=V2_LD(v+8),f=V2_LD(v+10),g=V2_LD(v+12),h=V2_LD(v+14);
        const auto c0=V2_SET1(o.c[0]),s0=V2_SET1(o.s[0]),c1=V2_SET1(o.c[1]),s1=V2_SET1(o.s[1]);
        if(o.ridge==0) {y[0]=V2_ADD(a,c);y[1]=V2_ADD(b,d);y[2]=V2_SUB(a,c);y[3]=V2_SUB(b,d);}
        else {auto re=V2_MSUB(V2_MUL(c0,b),s0,d),im=V2_MADD(V2_MUL(s0,b),c0,d);
            y[0]=V2_ADD(a,re);y[1]=V2_ADD(c,im);y[2]=V2_SUB(a,re);y[3]=V2_SUB(im,c);}
        if(o.ridge==1) {y[4]=V2_ADD(e,g);y[5]=V2_ADD(f,h);y[6]=V2_SUB(e,g);y[7]=V2_SUB(f,h);}
        else {auto re=V2_MSUB(V2_MUL(c1,f),s1,h),im=V2_MADD(V2_MUL(s1,f),c1,h);
            y[4]=V2_ADD(e,re);y[5]=V2_ADD(g,im);y[6]=V2_SUB(e,re);y[7]=V2_SUB(im,g);}
        leaf_registers(y[2*routes_[R][0]],y[2*routes_[R][0]+1],y[2*routes_[R][1]],y[2*routes_[R][1]+1],left,out);
        leaf_registers(y[2*routes_[R][2]],y[2*routes_[R][2]+1],y[2*routes_[R][3]],y[2*routes_[R][3]+1],right,out);
    }
#endif
public:
    explicit Plan(int n, Policy policy=Policy::separated,std::uint64_t seed=7043,bool fused=true):n_(n),rng_(seed?seed:1),policy_(policy),fused_(fused) {
        if(n<4 || (n&(n-1))) throw std::invalid_argument("quartic size must be a power of two >=4");
        ops_.reserve(n/4);build(0,n/4,-1,pi/2);
    }
    std::size_t plan_bytes() const {return ops_.size()*sizeof(Op);}
    std::size_t crossings() const {std::size_t k=0;for(auto& o:ops_) k+=o.choice!=0;return k;}
    std::size_t executed_crossings() const {std::size_t k=0;for(auto& o:ops_) k+=o.choice!=0 && o.w>1;return k;}
    template<class Complex> void forward(const double* input,Complex* output,double* work) const {
        // Natural samples -> initial (ridge, quarter-turn) normalized packet.
        int h=n_/2,i=0;
#if BRUUN_LEVEL >= 1
        for(;i+1<h;i+=2) {auto a=V2_LD(input+i),b=V2_LD(input+h+i);V2_ST(work+i,V2_ADD(a,b));V2_ST(work+h+i,V2_SUB(a,b));}
#endif
        for(;i<h;++i) {work[i]=input[i]+input[h+i];work[h+i]=input[i]-input[h+i];}
        if(n_==4) {output[0]={work[0]+work[1],0};output[1]={work[2],-work[3]};output[2]={work[0]-work[1],0};return;}
        for(std::size_t k=0;k<ops_.size();++k) {
            const auto& o=ops_[k];
            auto v=work+o.base;
#if BRUUN_LEVEL >= 1
            if(fused_ && o.w==2) {
                if(o.choice==0) codelet16<0>(v,o,ops_[k+1],ops_[k+2],output);
                else if(o.choice==1) codelet16<1>(v,o,ops_[k+1],ops_[k+2],output);
                else codelet16<2>(v,o,ops_[k+1],ops_[k+2],output);
                k+=2;continue;
            }
#endif
            if(o.w==1) leaf(v,o,output);
            else if(o.choice==0) stream<0>(v,o);
            else if(o.choice==1) stream<1>(v,o);
            else stream<2>(v,o);
        }
    }
};
#undef QW_REGISTER_INLINE
}
