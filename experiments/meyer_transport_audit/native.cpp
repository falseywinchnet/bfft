// Experimental runner only: production engine and ABI are unchanged.
#include "../../src/detail/meyer_kernel.hpp"

extern "C" {
void* audit_create(int n, int threads, double lam, double mu) {
    auto* e = new meyer::engine;
    if(e->init(n,n,lam,mu,64,1,0.,threads)!=BFFT_OK) {delete e; return nullptr;}
    e->ensure_flow_storage();
    return e;
}
void audit_destroy(void* p) {delete static_cast<meyer::engine*>(p);}
void audit_split(void* p, const double* image, double* texture, int mode,
                 int passes, double alpha, int horizon, int jumps) {
    auto& e=*static_cast<meyer::engine*>(p);
    const auto n=e.H*e.W;
    if(mode==1) {
        std::vector<double> cartoon(n);
        e.split_flow_jump(image,cartoon.data(),texture,4,horizon,2,jumps);
        return;
    }
    e.passes=passes;
    if(mode==0) {e.run_split_reduced_spectral(image,texture);return;}
    // Same four-pass prefix as the published schedules.
    e.passes=4;
    e.run_split_reduced_spectral(image,nullptr);
    const double cu=e.lam, eu=2*e.lam, cw=1/e.mu, ew=10/e.mu;
    for(int k=4;k<passes;++k) {
        if(mode==4 && k>=passes-2) {
            e.flow_ordinary_step(cu,eu,cw,ew);
            continue;
        }
        if(mode==3 || mode==4) {
            // Fuse the displacement into the live state. Keep both exact
            // NEW primals intact until all new gradients have been read.
            e.fwd2d_reflection(e.bux,e.buy,eu,e.d_spec);
            e.fwd2d_reflection(e.bvx,e.bvy,ew,e.q_spec);
            e.solve_meyer_triangle_to(e.u_spec,e.w_spec,e.d_spec,e.q_spec,
                cu,eu,cw,ew,e.flow_q0.us,e.flow_q0.ws);
            auto& un=e.flow_q0.field[0];auto& wn=e.flow_q0.field[1];
            e.inv2d(e.flow_q0.us,un.data());e.inv2d(e.flow_q0.ws,wn.data());
            e.P.run([&](int tid) {
                for(std::size_t y=tid;y<e.H;y+=e.P.lanes())
                for(std::size_t x=0;x<e.W;++x) {
                    const auto i=y*e.W+x,ix=y*e.W+(x+1==e.W?0:x+1),iy=(y+1==e.H?0:y+1)*e.W+x;
                    const double su=std::fmin(1.,(1/eu)/std::fmax(std::sqrt(e.bux[i]*e.bux[i]+e.buy[i]*e.buy[i]),1e-30));
                    const double sw=std::fmin(1.,(1/ew)/std::fmax(std::sqrt(e.bvx[i]*e.bvx[i]+e.bvy[i]*e.bvy[i]),1e-30));
                    e.bux[i]+=alpha*(un[ix]-un[i]+(su-1)*e.bux[i]);
                    e.buy[i]+=alpha*(un[iy]-un[i]+(su-1)*e.buy[i]);
                    e.bvx[i]+=alpha*(wn[ix]-wn[i]+(sw-1)*e.bvx[i]);
                    e.bvy[i]+=alpha*(wn[iy]-wn[i]+(sw-1)*e.bvy[i]);
                    e.u[i]+=alpha*(un[i]-e.u[i]);e.w[i]+=alpha*(wn[i]-e.w[i]);
                }
            });
            e.P.run([&](int tid) {
                std::size_t lo,hi;e.split(tid,e.n2(),lo,hi);
                for(auto i=lo;i<hi;++i) {
                    e.u_spec.a[i]+=alpha*(e.flow_q0.us.a[i]-e.u_spec.a[i]);
                    e.u_spec.b[i]+=alpha*(e.flow_q0.us.b[i]-e.u_spec.b[i]);
                    e.w_spec.a[i]+=alpha*(e.flow_q0.ws.a[i]-e.w_spec.a[i]);
                    e.w_spec.b[i]+=alpha*(e.flow_q0.ws.b[i]-e.w_spec.b[i]);
                }
            });
            continue;
        }
        e.build_flow_residual(cu,eu,cw,ew);
        std::array<std::vector<double>*,6> z={&e.u,&e.w,&e.bux,&e.buy,&e.bvx,&e.bvy};
        e.P.run([&](int tid) {
            std::size_t lo,hi;e.split(tid,n,lo,hi);
            for(int f=0;f<6;++f) for(auto i=lo;i<hi;++i)
                (*z[f])[i]+=alpha*e.flow_q0.field[f][i];
        });
        e.P.run([&](int tid) {
            std::size_t lo,hi;e.split(tid,e.n2(),lo,hi);
            for(auto i=lo;i<hi;++i) {
                e.u_spec.a[i]+=alpha*e.flow_q0.us.a[i];
                e.u_spec.b[i]+=alpha*e.flow_q0.us.b[i];
                e.w_spec.a[i]+=alpha*e.flow_q0.ws.a[i];
                e.w_spec.b[i]+=alpha*e.flow_q0.ws.b[i];
            }
        });
    }
    e.finish_split_texture(image,texture);
}
void audit_state(void* p,double* out) {
    auto& e=*static_cast<meyer::engine*>(p);
    std::array<std::vector<double>*,6> z={&e.u,&e.w,&e.bux,&e.buy,&e.bvx,&e.bvy};
    for(int f=0;f<6;++f) std::copy(z[f]->begin(),z[f]->end(),out+f*e.H*e.W);
}
}
