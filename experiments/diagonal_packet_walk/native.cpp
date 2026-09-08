// Scalar scheduled reference: all methods use the same complex operations.
// Counts are source-level complex transfers, NOT DRAM traffic or CPU counters.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <type_traits>
#include <vector>
#ifdef PACKET_NEON
#include <arm_neon.h>
// One natural-order complex double occupies one complete 128-bit register.
// No AoS/SoA conversion or padding occurs at the transform boundaries.
struct Z {
    float64x2_t v;
    Z(double r=0,double i=0):v{r,i}{}
    explicit Z(float64x2_t value):v(value){}
    double real()const{return vgetq_lane_f64(v,0);}
    double imag()const{return vgetq_lane_f64(v,1);}
};
Z operator+(Z a,Z b){return Z(vaddq_f64(a.v,b.v));}
Z operator-(Z a,Z b){return Z(vsubq_f64(a.v,b.v));}
Z product(Z a,Z b){
    auto swapped=vextq_f64(a.v,a.v,1);
    const uint64x2_t mask={UINT64_C(0x8000000000000000),0};
    auto signed_swap=vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(swapped),mask));
    return Z(vfmaq_laneq_f64(vmulq_laneq_f64(a.v,b.v,0),signed_swap,b.v,1));
}
Z conjugate(Z a){
    const uint64x2_t mask={0,UINT64_C(0x8000000000000000)};
    return Z(vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(a.v),mask)));
}
Z quarter(Z a,bool inverse){
    const uint64x2_t mask=inverse?uint64x2_t{UINT64_C(0x8000000000000000),0}:uint64x2_t{0,UINT64_C(0x8000000000000000)};
    auto swapped=vextq_f64(a.v,a.v,1);
    return Z(vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(swapped),mask)));
}
Z scaled(Z a,double value){return Z(vmulq_n_f64(a.v,value));}
#else
struct Z {
    double r=0,i=0;
    double real()const{return r;}
    double imag()const{return i;}
};
Z operator+(Z a,Z b){return {a.r+b.r,a.i+b.i};}
Z operator-(Z a,Z b){return {a.r-b.r,a.i-b.i};}
Z product(Z a,Z b){return {a.r*b.r-a.i*b.i,a.r*b.i+a.i*b.r};}
Z conjugate(Z a){return {a.r,-a.i};}
Z quarter(Z a,bool inverse){return inverse?Z{-a.i,a.r}:Z{a.i,-a.r};}
Z scaled(Z a,double value){return {a.r*value,a.i*value};}
#endif

static_assert(sizeof(Z)==16,"Complex storage must be exactly two doubles");
double mag(Z a){return std::hypot(a.real(),a.imag());}
struct Counts {
    uint64_t reads=0,writes=0,swaps=0,butterflies=0,multiplies=0,quarter_turns=0,scales=0;
};
struct EmptyCounts {};
template<bool Count> struct Ops {
    std::conditional_t<Count,Counts,EmptyCounts> c;
    Z read(const Z* p){if constexpr(Count) ++c.reads;return *p;}
    void write(Z* p,Z v){if constexpr(Count) ++c.writes;*p=v;}
    Z multiply(Z a,Z b){if constexpr(Count) ++c.multiplies;return product(a,b);}
};
struct FFTPlan {
    int n;
    std::vector<Z> roots;
    std::vector<int> reverse;
    explicit FFTPlan(int size):n(size),roots(size),reverse(size){
        for(int k=0;k<n;++k){double t=-2*std::acos(-1.)*k/n;roots[k]={std::cos(t),std::sin(t)};}
        for(int k=0;k<n;++k){int x=k,y=0;for(int l=n;l>1;l/=2){y=2*y+(x&1);x>>=1;}reverse[k]=y;}
    }
    template<bool C> Z rotate(Z x,int k,bool inverse,Ops<C>& op) const {
        if(!k)return x;
        if(4*k==n){if constexpr(C)++op.c.quarter_turns;return quarter(x,inverse);}
        return op.multiply(x,inverse?conjugate(roots[k]):roots[k]);
    }
    template<bool C> void reorder(Z* x,int stride,Ops<C>& op) const {
        for(int k=0;k<n;++k)if(k<reverse[k]){
            Z a=op.read(x+k*stride),b=op.read(x+reverse[k]*stride);
            op.write(x+k*stride,b);op.write(x+reverse[k]*stride,a);
            if constexpr(C)++op.c.swaps;
        }
    }
    template<bool C> void dit(Z* x,int stride,bool inverse,Ops<C>& op,bool ordered=false) const {
        if(!ordered)reorder(x,stride,op);
        for(int len=2;len<=n;len*=2)for(int base=0;base<n;base+=len)for(int j=0;j<len/2;++j){
            Z a=op.read(x+(base+j)*stride);
            Z b=rotate(op.read(x+(base+j+len/2)*stride),j*(n/len),inverse,op);
            op.write(x+(base+j)*stride,a+b);op.write(x+(base+j+len/2)*stride,a-b);
            if constexpr(C)++op.c.butterflies;
        }
    }
    template<bool C> void dif(Z* x,Ops<C>& op) const {
        for(int len=n;len>=2;len/=2)for(int base=0;base<n;base+=len)for(int j=0;j<len/2;++j){
            Z a=op.read(x+base+j),b=op.read(x+base+j+len/2);
            op.write(x+base+j,a+b);
            op.write(x+base+j+len/2,rotate(a-b,j*(n/len),false,op));
            if constexpr(C)++op.c.butterflies;
        }
        reorder(x,1,op);
    }
    template<bool C> void dif_unordered(Z* x,int stride,bool inverse,Ops<C>& op,const Z* prefactor=nullptr)const{
        for(int len=n;len>=2;len/=2)for(int base=0;base<n;base+=len)for(int j=0;j<len/2;++j){
            int ia=(base+j)*stride,ib=(base+j+len/2)*stride;
            Z a=op.read(x+ia),b=op.read(x+ib);
            if(prefactor&&len==n){a=op.multiply(a,prefactor[ia]);b=op.multiply(b,prefactor[ib]);}
            op.write(x+ia,a+b);op.write(x+ib,rotate(a-b,j*(n/len),inverse,op));
            if constexpr(C)++op.c.butterflies;
        }
    }
    template<bool C> void stockham(const Z* x,Z* out,Z* scratch,Ops<C>& op) const {
        int stages=0;for(int v=n;v>1;v/=2)++stages;
        const Z* src=x;
        Z* dst=(stages&1)?out:scratch;
        for(int p=1;p<n;p*=2){
            int q=n/(2*p);
            for(int j=0;j<p;++j)for(int k=0;k<q;++k){
                Z a=op.read(src+k+2*q*j);
                Z b=rotate(op.read(src+k+q*(2*j+1)),j*q,false,op);
                op.write(dst+k+q*j,a+b);op.write(dst+k+q*(j+p),a-b);
                if constexpr(C)++op.c.butterflies;
            }
            src=dst;dst=(dst==out)?scratch:out;
        }
    }
};

struct Diagonal {
    int a,b,m,n;
    FFTPlan fa,fb;
    std::vector<int> pack,source,cancel_source;
    std::vector<Z> incoming,edge,diagonal,cancel_incoming;
    explicit Diagonal(int r):a(1<<(r/2)),b(1<<((r+1)/2)),m(a*b),n(m*m),fa(a),fb(b),
        pack(n),source(n),cancel_source(n),incoming(n),edge(n),diagonal(n),cancel_incoming(n){
        auto cis=[](double t){return Z{std::cos(t),std::sin(t)};};
        double twopi=2*std::acos(-1.);
        for(int q=0;q<b;++q)for(int p=0;p<a;++p)for(int j=0;j<b;++j)for(int l=0;l<a;++l){
            int idx=((q*a+p)*b+j)*a+l;
            pack[idx]=p+a*j+m*((q+a*j+b*l)%m);
        }
        for(int p=0;p<a;++p)for(int beta=0;beta<a;++beta)for(int q=0;q<b;++q)for(int alpha=0;alpha<b;++alpha){
            int block=((p*a+beta)*b+q)*b+alpha;
            int pp=(a-beta)%a,qp=(b-alpha-pp%b+b)%b;
            int ap=(p+q)%b,bp=p;
            source[block]=((qp*a+pp)*b+ap)*a+bp;
            int tau=(a-beta)%a;
            edge[block]=cis(-twopi*tau*q/m);
            Z phase=cis(-twopi*(p*qp+q*pp)/m);
            incoming[block]=product(phase,conjugate(edge[block]));
            // Here q denotes h on the reduced route.
            cancel_source[block]=((qp*a+pp)*b+q)*a+p;
            cancel_incoming[block]=cis(-twopi*p*(double(qp)/m+double(q)/b));
            // In the internal dual layout q is h and alpha is j.
            diagonal[block]=cis(-twopi*double(tau+a*q)*(p+a*alpha)/n);
        }
    }
    template<bool C> void run(const Z* x,Z* out,Z* work,Z* blocks,Ops<C>& op,bool cancelled=false)const{
        for(int i=0;i<n;++i)op.write(work+i,op.read(x+pack[i]));
        // Q in coset-major layout (q,p,j,l).
        for(int base=0;base<n;base+=m){
            for(int j=0;j<b;++j)fa.dit(work+base+j*a,1,false,op);
            if(!cancelled)for(int l=0;l<a;++l)fb.dit(work+base+l,a,false,op);
        }
        // Torus monomial, carry grouping, and input edge phase fused in one pass.
        for(int i=0;i<n;++i)op.write(blocks+i,op.multiply(op.read(work+(cancelled?cancel_source[i]:source[i])),cancelled?cancel_incoming[i]:incoming[i]));
        for(int base=0;base<n;base+=b*b){
            for(int q=0;q<b;++q)fb.dit(blocks+base+q*b,1,true,op);
            if(!cancelled)for(int j=0;j<b;++j)fb.dit(blocks+base+j,b,true,op);
            for(int i=base;i<base+b*b;++i)op.write(blocks+i,op.multiply(op.read(blocks+i),diagonal[i]));
            for(int j=0;j<b;++j)fb.dit(blocks+base+j,b,false,op);
            if(!cancelled)for(int q=0;q<b;++q)fb.dit(blocks+base+q*b,1,false,op);
        }
        // Output edge and return to coset-major layout are fused.
        for(int p=0;p<a;++p)for(int beta=0;beta<a;++beta)for(int q=0;q<b;++q)for(int alpha=0;alpha<b;++alpha){
            int block=((p*a+beta)*b+q)*b+alpha;
            int idx=((q*a+p)*b+alpha)*a+beta;
            op.write(work+idx,op.multiply(op.read(blocks+block),edge[block]));
        }
        for(int base=0;base<n;base+=m){
            if(!cancelled)for(int l=0;l<a;++l)fb.dit(work+base+l,a,true,op);
            for(int j=0;j<b;++j)fa.dit(work+base+j*a,1,true,op);
        }
        double scale=1./(b*b);
        for(int i=0;i<n;++i){Z z=op.read(work+i);if(cancelled)op.write(out+pack[i],z);else {op.write(out+pack[i],scaled(z,scale));if constexpr(C)op.c.scales+=2;}}
    }
    template<bool C> void fused(const Z* x,Z* out,Z* work,Z* blocks,Ops<C>& op)const{
        // Fold each local reversal into an existing gather/scatter.
        for(int i=0;i<n;++i){int l=i%a;int source_i=i-l+fa.reverse[l];op.write(work+i,op.read(x+pack[source_i]));}
        for(int base=0;base<n;base+=a)fa.dit(work+base,1,false,op,true);
        for(int i=0;i<n;++i){int alpha=i%b;int k=i-alpha+fb.reverse[alpha];op.write(blocks+i,op.multiply(op.read(work+cancel_source[k]),cancel_incoming[k]));}
        for(int base=0;base<n;base+=b*b){
            for(int h=0;h<b;++h)fb.dit(blocks+base+h*b,1,true,op,true);
            // Fuse the diagonal pass with the first forward butterfly reads.
            for(int j=0;j<b;++j)fb.dif_unordered(blocks+base+j,b,false,op,diagonal.data()+base+j);
        }
        for(int p=0;p<a;++p)for(int beta=0;beta<a;++beta)for(int qpos=0;qpos<b;++qpos)for(int j=0;j<b;++j){
            int q=fb.reverse[qpos];
            int src=((p*a+beta)*b+qpos)*b+j;
            int phase_i=((p*a+beta)*b+q)*b+j;
            int dst=((q*a+p)*b+j)*a+beta;
            op.write(work+dst,op.multiply(op.read(blocks+src),edge[phase_i]));
        }
        for(int base=0;base<n;base+=a)fa.dif_unordered(work+base,1,true,op);
        for(int i=0;i<n;++i){int lpos=i%a;int k=i-lpos+fa.reverse[lpos];op.write(out+pack[k],op.read(work+i));}
    }
};

template<bool C> std::conditional_t<C,Counts,void> run(const std::string& method,const std::vector<Z>& x,std::vector<Z>& out,
    std::vector<Z>& w,std::vector<Z>& v,const FFTPlan& f,const Diagonal& d){
    Ops<C> op;
    if(method=="fused")d.fused(x.data(),out.data(),w.data(),v.data(),op);
    else if(method=="packet"||method=="cancelled")d.run(x.data(),out.data(),w.data(),v.data(),op,method=="cancelled");
    else if(method=="stockham")f.stockham(x.data(),out.data(),w.data(),op);
    else {
        for(int i=0;i<f.n;++i)op.write(out.data()+i,op.read(x.data()+i));
        if(method=="dit")f.dit(out.data(),1,false,op);else f.dif(out.data(),op);
    }
    if constexpr(C)return op.c;
}

int main(int argc,char** argv){
    int maxr=argc>1?std::stoi(argv[1]):8;
    std::ofstream dump;
    if(argc>2){dump.open(argv[2],std::ios::binary);if(!dump)return 3;}
    std::mt19937_64 rng(1729);std::normal_distribution<double> normal;
    std::cout<<std::setprecision(12)<<"{\"schema\":2,\"backend\":\""
#ifdef PACKET_NEON
      <<"neon_complex_pair"
#else
      <<"scalar"
#endif
      <<"\",\"count_unit\":\"source-level complex element access\",\"results\":[\n";
    bool first=true;
    volatile double sink=0;
    for(int r=1;r<=maxr;++r){
        Diagonal d(r);FFTPlan f(d.n);
        std::vector<Z>x(d.n),out(d.n),w(d.n),v(d.n),oracle(d.n);
        for(auto& z:x)z={normal(rng),normal(rng)};
        if(dump.is_open()){
            int32_t n=d.n;dump.write(reinterpret_cast<const char*>(&n),sizeof(n));
            for(Z z:x){double pair[2]={z.real(),z.imag()};dump.write(reinterpret_cast<const char*>(pair),sizeof(pair));}
        }
        run<false>("dit",x,oracle,w,v,f,d);
        for(const std::string method:{"dit","dif","stockham","packet","cancelled","fused"}){
            Counts c=run<true>(method,x,out,w,v,f,d);
            double maxerr=0;
            for(int i=0;i<d.n;++i)maxerr=std::max(maxerr,mag(out[i]-oracle[i]));
            // Small N independent direct DFT, all inputs complex.
            double directerr=0;
            if(d.n<=256)for(int k=0;k<d.n;++k){Z sum;for(int j=0;j<d.n;++j)sum=sum+product(x[j],f.roots[(k*j)%d.n]);directerr=std::max(directerr,mag(sum-out[k]));}
            if(maxerr>2e-10||directerr>2e-10){std::cerr<<"correctness failure "<<r<<" "<<method<<" "<<maxerr<<" "<<directerr<<"\n";return 2;}
            if(dump.is_open())for(Z z:out){double pair[2]={z.real(),z.imag()};dump.write(reinterpret_cast<const char*>(pair),sizeof(pair));}
            int repeats=std::max(2,200000/d.n);
            std::vector<double> timings;
            for(int trial=0;trial<9;++trial){
                auto begin=std::chrono::steady_clock::now();
                for(int k=0;k<repeats;++k){run<false>(method,x,out,w,v,f,d);sink+=out[k%d.n].real();}
                auto end=std::chrono::steady_clock::now();
                timings.push_back(std::chrono::duration<double,std::micro>(end-begin).count()/repeats);
            }
            std::sort(timings.begin(),timings.end());
            if(!first)std::cout<<",\n";first=false;
            std::cout<<"{\"r\":"<<r<<",\"N\":"<<d.n<<",\"method\":\""<<method<<"\",\"max_error_vs_dit\":"<<maxerr
                <<",\"direct_error_small_N\":"<<directerr<<",\"median_us\":"<<timings[4]<<",\"min_us\":"<<timings.front()
                <<",\"max_us\":"<<timings.back()<<",\"reads\":"<<c.reads<<",\"writes\":"<<c.writes
                <<",\"bit_reversal_swaps\":"<<c.swaps<<",\"butterflies\":"<<c.butterflies<<",\"complex_multiplies\":"<<c.multiplies
                <<",\"quarter_turns\":"<<c.quarter_turns<<",\"real_scalings\":"<<c.scales
#ifdef PACKET_NEON
                <<",\"intrinsic_lane_swaps\":"<<(c.multiplies+c.quarter_turns)
                <<",\"coefficient_vector_reads\":"<<c.multiplies
                <<",\"coefficient_lane_operands\":"<<(2*c.multiplies)
                <<",\"boundary_layout_conversions\":0"
#endif
                <<"}"<<std::flush;
        }
    }
    std::cout<<"\n]}\n";
    return sink==123456789?1:0;
}
