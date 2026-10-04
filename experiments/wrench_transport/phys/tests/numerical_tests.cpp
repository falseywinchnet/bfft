// Independent numerical checks: dense elimination and exhaustive SAT axes.
#include "../block_ldl.hpp"
#include "../collide.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>
using namespace zc::phys;
namespace {
unsigned seed = 90210;
double random_value() { seed = seed * 1664525u + 1013904223u; return double(seed) / 4294967296.0; }
double signed_value() { return 2 * random_value() - 1; }
double support(const WorldHull& h, Vec3 n, bool maximum) {
    double result = maximum ? -1e300 : 1e300;
    for (Vec3 p : h.vertices) result = maximum ? std::max(result, dot(p,n)) : std::min(result, dot(p,n));
    return result;
}
double oracle(const WorldHull& a, const WorldHull& b) {
    double result = -1e300;
    auto axis = [&](Vec3 n) {
        const double size = length(n);
        if (size < 1e-12) return;
        n = n * (1 / size);
        result = std::max(result, support(b,n,false) - support(a,n,true));
        result = std::max(result, support(a,n,false) - support(b,n,true));
    };
    for (Vec3 n : a.normals) axis(n);
    for (Vec3 n : b.normals) axis(n);
    for (Vec3 ea : a.edge_directions) for (Vec3 eb : b.edge_directions) axis(cross(ea,eb));
    return result;
}
}
int main() {
    double worst_residual = 0, worst_solution = 0;
    for (int trial = 0; trial < 80; ++trial) {
        const int nodes = 2 + trial % 15, n = nodes * 6;
        std::vector<double> matrix(n*n,0), rhs(n), actual(n), expected(n);
        std::vector<std::int32_t> pairs;
        // Graph-Laplacian rank-one relations plus positive mass: independently SPD.
        for (int i = 0; i < n; ++i) { matrix[i*n+i] = 0.5 + random_value(); rhs[i] = signed_value(); }
        for (int a = 0; a < nodes; ++a) for (int b = a+1; b < nodes; ++b) {
            if (b != a+1 && random_value() > .25) continue;
            pairs.push_back(a); pairs.push_back(b);
            double v[12]; for (double& x : v) x = 4 * signed_value();
            for (int i = 0; i < 12; ++i) for (int j = 0; j < 12; ++j) {
                int r = (i < 6 ? a*6+i : b*6+i-6), c = (j < 6 ? a*6+j : b*6+j-6);
                matrix[r*n+c] += v[i]*v[j];
            }
        }
        BlockFactor factor; factor.analyze(nodes,pairs); factor.clear_blocks();
        for (int a = 0; a < nodes; ++a) for (int i = 0; i < 6; ++i) for (int j = 0; j < 6; ++j)
            factor.diagonal_block(a)[i*6+j] = matrix[(a*6+i)*n+a*6+j];
        for (size_t k = 0; k < pairs.size(); k += 2) {
            int a=pairs[k], b=pairs[k+1]; bool transposed; double* block=factor.coupling_block(a,b,transposed);
            for (int i=0;i<6;++i) for(int j=0;j<6;++j) block[transposed ? j*6+i : i*6+j]=matrix[(a*6+i)*n+b*6+j];
        }
        if (!factor.factorize()) return 1;
        factor.solve(rhs,actual);
        // Dense Gaussian elimination with partial pivoting, not block Cholesky.
        auto dense = matrix; expected = rhs;
        for(int k=0;k<n;++k) {
            int pivot=k; for(int i=k+1;i<n;++i) if(std::abs(dense[i*n+k])>std::abs(dense[pivot*n+k])) pivot=i;
            for(int j=0;j<n;++j) std::swap(dense[k*n+j],dense[pivot*n+j]); std::swap(expected[k],expected[pivot]);
            for(int i=k+1;i<n;++i) { double m=dense[i*n+k]/dense[k*n+k]; for(int j=k+1;j<n;++j) dense[i*n+j]-=m*dense[k*n+j]; expected[i]-=m*expected[k]; }
        }
        for(int i=n-1;i>=0;--i) { for(int j=i+1;j<n;++j) expected[i]-=dense[i*n+j]*expected[j]; expected[i]/=dense[i*n+i]; }
        for(int i=0;i<n;++i) { double residual=-rhs[i]; for(int j=0;j<n;++j) residual+=matrix[i*n+j]*actual[j]; worst_residual=std::max(worst_residual,std::abs(residual)); worst_solution=std::max(worst_solution,std::abs(actual[i]-expected[i])); }
    }
    std::printf("dense oracle: 80 coupled SPD systems; residual %.3e, solution error %.3e\n",worst_residual,worst_solution);
    if(worst_residual>1e-10 || worst_solution>1e-10) return 1;
    std::vector<Shape> shapes;
    for (int kind=0;kind<12;++kind) {
        ShapeDesc desc; HullDesc hull;
        if(kind%3==0) for(int i=0;i<8;++i) hull.points.push_back({(i&1?.06:-.06),(i&2?.05:-.05),(i&4?.02:-.02)});
        else for(int i=0;i<18+kind;++i) { Vec3 p=normalized({signed_value(),signed_value(),signed_value()}); double r=.045+.02*random_value(); hull.points.push_back({p.x*r,p.y*r,p.z*r*(kind%3==1?.3:1)}); }
        desc.hulls.push_back(hull); shapes.push_back(cook(desc));
    }
    int missing=0, false_contact=0, invalid=0; double worst_outside=0;
    for (int trial=0;trial<6000;++trial) {
        WorldHull a,b; auto& sa=shapes[trial%12]; auto& sb=shapes[(trial/12+5)%12];
        Quat qa=from_axis_angle({signed_value(),signed_value(),signed_value()},6*random_value());
        Quat qb=trial%7==0?qa:from_axis_angle({signed_value(),signed_value(),signed_value()},6*random_value());
        Vec3 direction=normalized({signed_value(),signed_value(),signed_value()});
        transform_hull(sa.hulls[0],{},qa,a);
        double lo=0, hi=.3, target=-.004+.012*random_value();
        // Place near a face separation; final truth additionally checks ALL edge axes.
        for(int k=0;k<25;++k) {
            double mid=(lo+hi)*.5; transform_hull(sb.hulls[0],direction*mid,qb,b);
            double gap=-1e300;
            for(size_t f=0;f<a.normals.size();++f) gap=std::max(gap,support(b,a.normals[f],false)-a.offsets[f]);
            for(size_t f=0;f<b.normals.size();++f) gap=std::max(gap,support(a,b.normals[f],false)-b.offsets[f]);
            if(gap>target) hi=mid; else lo=mid;
        }
        transform_hull(sb.hulls[0],direction*hi,qb,b);
        double truth=oracle(a,b), separated; CollideScratch scratch; std::vector<ManifoldPoint> points;
        collide_hulls(a,b,.004,scratch,points,separated);
        if(truth < -1e-7 && points.empty()) ++missing;
        if(truth > .004+1e-7 && !points.empty()) ++false_contact;
        for(auto& point: points) {
            if(!std::isfinite(point.separation) || std::abs(length(point.normal)-1)>1e-8) ++invalid;
            for(size_t f=0;f<a.normals.size();++f) worst_outside=std::max(worst_outside,dot(a.normals[f],point.point_a)-a.offsets[f]);
            for(size_t f=0;f<b.normals.size();++f) worst_outside=std::max(worst_outside,dot(b.normals[f],point.point_b)-b.offsets[f]);
        }
    }
    std::printf("SAT oracle: 6000 pairs; missed overlaps %d, false contacts %d, invalid %d; outside %.3e m\n",missing,false_contact,invalid,worst_outside);
    // Existing witness manifolds allow up to 5 mm lateral reach.
    return missing || false_contact || invalid || worst_outside > .0051;
}
