#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using RGB = std::array<double, 3>;
constexpr double pi = std::numbers::pi_v<double>;

struct Vec3 {
    double x = 0.0, y = 0.0, z = 0.0;
};

Vec3 operator+(Vec3 a, Vec3 b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
Vec3 operator-(Vec3 a, Vec3 b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
Vec3 operator*(Vec3 a, double s) { return {a.x*s, a.y*s, a.z*s}; }
Vec3 operator*(double s, Vec3 a) { return a*s; }
Vec3 operator/(Vec3 a, double s) { return {a.x/s, a.y/s, a.z/s}; }
double dot(Vec3 a, Vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x};
}
double norm(Vec3 a) { return std::sqrt(dot(a,a)); }
Vec3 unit(Vec3 a) { const double n=norm(a); return n>0.0 ? a/n : Vec3{}; }

RGB &operator+=(RGB &a, const RGB &b) {
    for (int c=0;c<3;++c) a[c]+=b[c];
    return a;
}
RGB operator*(const RGB &a, double s) {
    return {a[0]*s,a[1]*s,a[2]*s};
}
double energy(const RGB &a) { return a[0]+a[1]+a[2]; }

enum Group : int {
    Floor, Back, LeftWall, RightWall, Ceiling, AreaLight,
    RedSphere, BlueSphere, GreenSphere, GroupCount
};

struct Patch {
    Vec3 center, normal;
    double area = 0.0;
    double radius = 0.0;
    RGB albedo{}, emission{};
    int object_id = -1;
    Group group = Floor;
};

struct Sphere {
    Vec3 center;
    double radius = 0.0;
    RGB albedo{};
    int object_id = -1;
    Group group = RedSphere;
    std::vector<int> patches;
};

struct PlaneGrid {
    Group group = Floor;
    Vec3 origin, u, v, normal;
    int nu = 0, nv = 0, start = 0;
};

struct Edge { int dst = 0; float weight = 0.0f; };

struct Scene {
    std::vector<Patch> patches;
    std::array<Sphere,3> spheres;
    std::vector<PlaneGrid> planes;
    std::vector<std::vector<Edge>> adjacency;
    std::vector<double> row_sum;
    double transport_scale = 1.0;
};

struct Timings {
    double subdivision_ms=0, geometry_ms=0, transport_ms=0;
    double lookup_ms=0, raster_ms=0, write_ms=0, total_ms=0;
};

double subdivision_bound(double separation, double hp, double hq,
                         double eta_p, double eta_q) {
    const double R=hp+hq, d=separation-R;
    if (d<=0.0) return std::numeric_limits<double>::infinity();
    return ((eta_p+eta_q+4.0*R/d)/(d*d)+2.0*R/(d*d*d))/pi;
}

double lambert_kernel(Vec3 x, Vec3 nx, Vec3 y, Vec3 ny) {
    const Vec3 r=y-x; const double d=norm(r);
    if (d<=0.0) return 0.0;
    const Vec3 w=r/d;
    return std::max(dot(nx,w),0.0)*std::max(dot(ny,-1.0*w),0.0)/(pi*d*d);
}

int adaptive_sphere_samples(int width, double radius, Vec3 center,
                            Vec3 camera, double fov_degrees) {
    const double focal=0.5*width/std::tan(0.5*fov_degrees*pi/180.0);
    const double projected_radius=focal*radius/norm(center-camera);
    int count=48;
    // A cell's projected diameter falls like 1/sqrt(N). The twenty-pixel
    // ceiling is followed by continuous radiance interpolation; silhouettes
    // remain analytic and therefore do not inherit this cell scale.
    while (2.0*projected_radius/std::sqrt(static_cast<double>(count))>10.0
           && count<3072) count*=4;
    return count;
}

void add_plane(Scene &scene, Group group, Vec3 origin, Vec3 u, Vec3 v,
               int nu, int nv, Vec3 normal, RGB albedo, RGB emission_density={}) {
    PlaneGrid grid{group,origin,u,v,normal,nu,nv,
                   static_cast<int>(scene.patches.size())};
    scene.planes.push_back(grid);
    const Vec3 du=u/static_cast<double>(nu), dv=v/static_cast<double>(nv);
    const double area=norm(cross(du,dv));
    const double radius=0.5*std::sqrt(dot(du,du)+dot(dv,dv));
    for (int i=0;i<nu;++i) for (int j=0;j<nv;++j) {
        const Vec3 c=origin+du*(i+0.5)+dv*(j+0.5);
        scene.patches.push_back({c,normal,area,radius,albedo,
                                 emission_density*area,-1,group});
    }
}

void add_sphere(Scene &scene, Sphere &sphere, int count) {
    const double area=4.0*pi*sphere.radius*sphere.radius/count;
    const double cell_radius=1.15*std::sqrt(area/pi);
    const double golden=pi*(3.0-std::sqrt(5.0));
    sphere.patches.reserve(count);
    for (int i=0;i<count;++i) {
        const double y=1.0-2.0*(i+0.5)/count;
        const double r=std::sqrt(std::max(1.0-y*y,0.0));
        const double a=golden*i;
        const Vec3 n{r*std::cos(a),y,r*std::sin(a)};
        sphere.patches.push_back(static_cast<int>(scene.patches.size()));
        scene.patches.push_back({sphere.center+sphere.radius*n,n,area,
            cell_radius,sphere.albedo,{},sphere.object_id,sphere.group});
    }
}

Scene build_scene(int width, int height, int &sphere_samples) {
    (void)height;
    Scene s;
    const RGB white{0.72,0.74,0.76};
    // Smooth planar fields need far fewer cells than the analytic sphere
    // silhouettes. These grids are approximately four times the linear
    // density of the 800x600 preview.
    add_plane(s,Floor,{-3,0,-5},{6,0,0},{0,0,6},12,12,{0,1,0},white);
    add_plane(s,Back,{-3,0,-5},{6,0,0},{0,4,0},12,8,{0,0,1},white);
    add_plane(s,LeftWall,{-3,0,-5},{0,0,6},{0,4,0},10,8,{1,0,0},
              {0.58,0.16,0.12});
    add_plane(s,RightWall,{3,0,1},{0,0,-6},{0,4,0},10,8,{-1,0,0},
              {0.12,0.22,0.62});
    add_plane(s,Ceiling,{-3,4,1},{6,0,0},{0,0,-6},12,12,{0,-1,0},white);
    add_plane(s,AreaLight,{-1.15,3.92,-3.0},{2.3,0,0},{0,0,1.7},6,4,
              {0,-1,0},{0.02,0.02,0.02},{34,31,27});
    s.spheres={Sphere{{-1.35,1.05,-2.65},0.72,{0.78,0.08,0.055},10,RedSphere,{}},
               Sphere{{1.22,0.92,-3.15},0.62,{0.055,0.20,0.82},11,BlueSphere,{}},
               Sphere{{0.15,0.77,-1.20},0.48,{0.08,0.72,0.20},12,GreenSphere,{}}};
    const Vec3 camera{0,2.15,6.8};
    sphere_samples=0;
    for (auto &sphere:s.spheres) {
        const int count=adaptive_sphere_samples(width,sphere.radius,sphere.center,
                                                camera,48.0);
        sphere_samples=std::max(sphere_samples,count);
        add_sphere(s,sphere,count);
    }
    return s;
}

bool visible(const Patch &a, const Patch &b,
             const std::array<Sphere,3> &spheres) {
    const Vec3 d=b.center-a.center; const double d2=dot(d,d);
    if (d2<=0.0) return false;
    for (const auto &sphere:spheres) {
        const bool source=sphere.object_id==a.object_id;
        const bool receiver=sphere.object_id==b.object_id;
        if (source!=receiver) continue;
        const double t=dot(sphere.center-a.center,d)/d2;
        if (t<=1e-10 || t>=1.0-1e-10) continue;
        const Vec3 q=a.center+d*t-sphere.center;
        if (dot(q,q)<=sphere.radius*sphere.radius*(1.0-1e-10)) return false;
    }
    return true;
}

void compile_geometry(Scene &scene) {
    const int n=static_cast<int>(scene.patches.size());
    scene.adjacency.assign(n,{}); scene.row_sum.assign(n,0.0);
    for (int i=0;i<n;++i) scene.adjacency[i].reserve(n/5);
    for (int i=0;i<n;++i) for (int j=i+1;j<n;++j) {
        const Patch &a=scene.patches[i], &b=scene.patches[j];
        const Vec3 delta=b.center-a.center; const double distance=norm(delta);
        if (distance<=0.0) continue;
        const Vec3 w=delta/distance;
        const double ca=dot(a.normal,w), cb=dot(b.normal,-1.0*w);
        if (ca<=0.0 || cb<=0.0 || !visible(a,b,scene.spheres)) continue;
        const double denominator=pi*distance*distance+2.0*(a.area+b.area);
        const double common=ca*cb/denominator;
        const double fij=b.area*common, fji=a.area*common;
        if (fij>1e-12) scene.adjacency[i].push_back({j,static_cast<float>(fij)});
        if (fji>1e-12) scene.adjacency[j].push_back({i,static_cast<float>(fji)});
    }
    double maximum=0.0;
    for (int i=0;i<n;++i) {
        for (const auto &e:scene.adjacency[i]) scene.row_sum[i]+=e.weight;
        maximum=std::max(maximum,scene.row_sum[i]);
    }
    scene.transport_scale=std::min(0.80,0.98/std::max(maximum,1e-30));
    for (int i=0;i<n;++i) {
        scene.row_sum[i]=0.0;
        for (auto &e:scene.adjacency[i]) {
            e.weight=static_cast<float>(e.weight*scene.transport_scale);
            scene.row_sum[i]+=e.weight;
        }
    }
}

struct TransportResult {
    std::vector<RGB> outgoing;
    int depths=0;
    std::size_t active_edges=0;
    double requested_error=0, certified_error=0;
    std::vector<std::array<double,4>> depth_work;
};

TransportResult solve_transport(const Scene &scene, double error_fraction) {
    const int n=static_cast<int>(scene.patches.size());
    std::vector<RGB> frontier(n), outgoing(n), incoming(n);
    double source_energy=0.0, rho_max=0.0;
    for (int i=0;i<n;++i) {
        frontier[i]=outgoing[i]=scene.patches[i].emission;
        source_energy+=energy(frontier[i]);
        for (double x:scene.patches[i].albedo) rho_max=std::max(rho_max,x);
    }
    const double requested=error_fraction*source_energy;
    const double tail_budget=0.5*requested;
    const double transport_budget=requested-tail_budget;
    const double incident_budget=transport_budget*(1.0-rho_max)/rho_max;
    double remaining=incident_budget, discarded=0.0, tail_bound=0.0;
    TransportResult result; result.requested_error=requested;
    for (int depth=0;depth<100;++depth) {
        double front_energy=0.0;
        for (const auto &q:frontier) front_energy+=energy(q);
        if (front_energy<=std::max(1e-14,tail_budget*(1.0-rho_max))) {
            tail_bound=front_energy/(1.0-rho_max); break;
        }
        std::fill(incoming.begin(),incoming.end(),RGB{});
        const double call_budget=0.5*remaining;
        double call_discard=0.0;
        std::size_t edges=0, active_sources=0;
        for (int i=0;i<n;++i) {
            const double e=energy(frontier[i]);
            if (e<=0.0) continue;
            const double bound=e*scene.row_sum[i];
            if (call_discard+bound<=call_budget) {
                call_discard+=bound; continue;
            }
            ++active_sources;
            edges+=scene.adjacency[i].size();
            for (const auto &edge:scene.adjacency[i]) {
                for (int c=0;c<3;++c)
                    incoming[edge.dst][c]+=edge.weight*frontier[i][c];
            }
        }
        discarded+=call_discard; remaining=std::max(incident_budget-discarded,0.0);
        double next_energy=0.0;
        for (int i=0;i<n;++i) for (int c=0;c<3;++c) {
            frontier[i][c]=scene.patches[i].albedo[c]*incoming[i][c];
            outgoing[i][c]+=frontier[i][c]; next_energy+=frontier[i][c];
        }
        result.active_edges+=edges;
        result.depth_work.push_back({front_energy,next_energy,
            static_cast<double>(active_sources),static_cast<double>(edges)});
        ++result.depths;
        if (next_energy==0.0) break;
    }
    result.certified_error=rho_max/(1.0-rho_max)*discarded+tail_bound;
    result.outgoing=std::move(outgoing);
    return result;
}

struct SphereLookup {
    int width=512,height=256;
    std::vector<RGB> data;
};

SphereLookup build_lookup(const Scene &scene, const Sphere &sphere,
                          const std::vector<RGB> &radiance) {
    SphereLookup lut; lut.data.resize(lut.width*lut.height);
    for (int y=0;y<lut.height;++y) for (int x=0;x<lut.width;++x) {
        const double latitude=pi*(0.5-(y+0.5)/lut.height);
        const double longitude=2.0*pi*((x+0.5)/lut.width-0.5);
        const Vec3 n{std::cos(latitude)*std::sin(longitude),std::sin(latitude),
                     std::cos(latitude)*std::cos(longitude)};
        std::array<std::pair<double,int>,4> best{{{-2,-1},{-2,-1},{-2,-1},{-2,-1}}};
        for (int index:sphere.patches) {
            const double score=dot(n,scene.patches[index].normal);
            if (score<=best[3].first) continue;
            best[3]={score,index};
            for (int k=3;k>0 && best[k].first>best[k-1].first;--k)
                std::swap(best[k],best[k-1]);
        }
        RGB value{}; double total=0.0;
        for (auto [score,index]:best) {
            const double w=1.0/std::max(1.0-score,1e-5);
            value+=radiance[index]*w; total+=w;
        }
        lut.data[y*lut.width+x]=value*(1.0/total);
    }
    return lut;
}

RGB sample_lookup(const SphereLookup &lut, Vec3 n) {
    const double longitude=std::atan2(n.x,n.z);
    const double latitude=std::asin(std::clamp(n.y,-1.0,1.0));
    double fx=(longitude/(2*pi)+0.5)*lut.width-0.5;
    double fy=(0.5-latitude/pi)*lut.height-0.5;
    int x0=static_cast<int>(std::floor(fx)), y0=static_cast<int>(std::floor(fy));
    const double tx=fx-x0, ty=fy-y0;
    auto at=[&](int x,int y)->const RGB& {
        x=(x%lut.width+lut.width)%lut.width;
        y=std::clamp(y,0,lut.height-1);
        return lut.data[y*lut.width+x];
    };
    RGB out{};
    for (int c=0;c<3;++c) out[c]=(1-ty)*((1-tx)*at(x0,y0)[c]+tx*at(x0+1,y0)[c])
        +ty*((1-tx)*at(x0,y0+1)[c]+tx*at(x0+1,y0+1)[c]);
    return out;
}

bool plane_hit(const PlaneGrid &g, Vec3 origin, Vec3 ray, double &t, Vec3 &p) {
    const double denominator=dot(g.normal,ray);
    if (std::abs(denominator)<1e-12) return false;
    t=dot(g.origin-origin,g.normal)/denominator;
    if (t<=1e-6) return false;
    p=origin+ray*t;
    const Vec3 delta=p-g.origin;
    const double fu=dot(delta,g.u)/dot(g.u,g.u);
    const double fv=dot(delta,g.v)/dot(g.v,g.v);
    return fu>=0.0 && fu<=1.0 && fv>=0.0 && fv<=1.0;
}

RGB sample_plane(const Scene &scene, const PlaneGrid &g, Vec3 p,
                 const std::vector<RGB> &radiance) {
    const Vec3 delta=p-g.origin;
    const double u=std::clamp(dot(delta,g.u)/dot(g.u,g.u)*g.nu-0.5,0.0,g.nu-1.0);
    const double v=std::clamp(dot(delta,g.v)/dot(g.v,g.v)*g.nv-0.5,0.0,g.nv-1.0);
    const int i0=static_cast<int>(std::floor(u)), j0=static_cast<int>(std::floor(v));
    const int i1=std::min(i0+1,g.nu-1), j1=std::min(j0+1,g.nv-1);
    const double tu=u-i0,tv=v-j0;
    auto at=[&](int i,int j)->const RGB& { return radiance[g.start+i*g.nv+j]; };
    RGB out{};
    for(int c=0;c<3;++c) out[c]=(1-tv)*((1-tu)*at(i0,j0)[c]+tu*at(i1,j0)[c])
        +tv*((1-tu)*at(i0,j1)[c]+tu*at(i1,j1)[c]);
    (void)scene; return out;
}

std::vector<std::uint8_t> render(const Scene &scene,
    const std::vector<RGB> &radiance,const std::array<SphereLookup,3> &lookups,
    int width,int height) {
    std::vector<std::uint8_t> image(static_cast<std::size_t>(width)*height*3,0);
    const Vec3 camera{0,2.15,6.8}, target{0,1.55,-2.15};
    const Vec3 forward=unit(target-camera);
    const Vec3 right=unit(cross(forward,{0,1,0}));
    const Vec3 up=cross(right,forward);
    const double scale=std::tan(24.0*pi/180.0), aspect=double(width)/height;
    std::atomic<int> next_row{0};
    const unsigned workers=std::max(1u,std::thread::hardware_concurrency());
    std::vector<std::thread> threads;
    for (unsigned worker=0;worker<workers;++worker) threads.emplace_back([&] {
        for (;;) {
            const int y=next_row.fetch_add(1); if (y>=height) break;
            for (int x=0;x<width;++x) {
                const double px=(2.0*(x+0.5)/width-1.0)*aspect*scale;
                const double py=(1.0-2.0*(y+0.5)/height)*scale;
                const Vec3 ray=unit(forward+right*px+up*py);
                double nearest=std::numeric_limits<double>::infinity();
                int plane_index=-1,sphere_index=-1; Vec3 hit{},normal{};
                for (int k=0;k<static_cast<int>(scene.planes.size());++k) {
                    double t; Vec3 p;
                    if (plane_hit(scene.planes[k],camera,ray,t,p) && t<nearest) {
                        nearest=t;plane_index=k;sphere_index=-1;hit=p;
                    }
                }
                for (int k=0;k<3;++k) {
                    const auto &s=scene.spheres[k]; const Vec3 rel=camera-s.center;
                    const double b=dot(ray,rel), c=dot(rel,rel)-s.radius*s.radius;
                    const double disc=b*b-c; if (disc<0.0) continue;
                    const double root=std::sqrt(disc);
                    double t=-b-root; if (t<=1e-6) t=-b+root;
                    if (t>1e-6 && t<nearest) {
                        nearest=t;sphere_index=k;plane_index=-1;hit=camera+ray*t;
                        normal=unit(hit-s.center);
                    }
                }
                RGB linear{};
                if (sphere_index>=0) linear=sample_lookup(lookups[sphere_index],normal);
                else if (plane_index>=0)
                    linear=sample_plane(scene,scene.planes[plane_index],hit,radiance);
                const std::size_t offset=(static_cast<std::size_t>(y)*width+x)*3;
                for (int c=0;c<3;++c) {
                    const double mapped=std::pow(std::clamp(1.0-std::exp(-0.72*std::max(linear[c],0.0)),0.0,1.0),1.0/2.2);
                    image[offset+c]=static_cast<std::uint8_t>(std::lround(255.0*mapped));
                }
            }
        }
    });
    for (auto &thread:threads) thread.join();
    return image;
}

RGB geometry_color(Group group) {
    switch (group) {
    case Floor: return {0.23,0.25,0.27};
    case Back: return {0.18,0.20,0.23};
    case LeftWall: return {0.34,0.09,0.07};
    case RightWall: return {0.06,0.12,0.34};
    case Ceiling: return {0.15,0.16,0.19};
    case AreaLight: return {0.72,0.67,0.28};
    case RedSphere: return {0.48,0.06,0.04};
    case BlueSphere: return {0.04,0.12,0.52};
    case GreenSphere: return {0.04,0.42,0.10};
    default: return {0.15,0.15,0.15};
    }
}

std::vector<std::uint8_t> render_geometry(const Scene &scene,int width,int height) {
    std::vector<std::uint8_t> image(static_cast<std::size_t>(width)*height*3,0);
    const Vec3 camera{0,2.15,6.8}, target{0,1.55,-2.15};
    const Vec3 forward=unit(target-camera);
    const Vec3 right=unit(cross(forward,{0,1,0}));
    const Vec3 up=cross(right,forward);
    const double scale=std::tan(24.0*pi/180.0), aspect=double(width)/height;
    std::atomic<int> next_row{0};
    const unsigned workers=std::max(1u,std::thread::hardware_concurrency());
    std::vector<std::thread> threads;
    for (unsigned worker=0;worker<workers;++worker) threads.emplace_back([&] {
        for (;;) {
            const int y=next_row.fetch_add(1); if (y>=height) break;
            for (int x=0;x<width;++x) {
                const double px=(2.0*(x+0.5)/width-1.0)*aspect*scale;
                const double py=(1.0-2.0*(y+0.5)/height)*scale;
                const Vec3 ray=unit(forward+right*px+up*py);
                double nearest=std::numeric_limits<double>::infinity();
                int plane_index=-1,sphere_index=-1; Vec3 hit{},normal{};
                for (int k=0;k<static_cast<int>(scene.planes.size());++k) {
                    double t; Vec3 p;
                    if (plane_hit(scene.planes[k],camera,ray,t,p) && t<nearest) {
                        nearest=t;plane_index=k;sphere_index=-1;hit=p;
                    }
                }
                for (int k=0;k<3;++k) {
                    const auto &s=scene.spheres[k]; const Vec3 rel=camera-s.center;
                    const double b=dot(ray,rel), c=dot(rel,rel)-s.radius*s.radius;
                    const double disc=b*b-c; if (disc<0.0) continue;
                    const double root=std::sqrt(disc);
                    double t=-b-root; if (t<=1e-6) t=-b+root;
                    if (t>1e-6 && t<nearest) {
                        nearest=t;sphere_index=k;plane_index=-1;hit=camera+ray*t;
                        normal=unit(hit-s.center);
                    }
                }
                RGB color{}; bool boundary=false;
                if (sphere_index>=0) {
                    color=geometry_color(scene.spheres[sphere_index].group);
                    boundary=std::abs(dot(normal,-1.0*ray))<0.10;
                } else if (plane_index>=0) {
                    const auto &g=scene.planes[plane_index]; color=geometry_color(g.group);
                    const Vec3 delta=hit-g.origin;
                    const double gu=dot(delta,g.u)/dot(g.u,g.u)*g.nu;
                    const double gv=dot(delta,g.v)/dot(g.v,g.v)*g.nv;
                    const double du=std::min(gu-std::floor(gu),std::ceil(gu)-gu);
                    const double dv=std::min(gv-std::floor(gv),std::ceil(gv)-gv);
                    boundary=du<0.018 || dv<0.018;
                }
                if (boundary) color={0.86,0.89,0.94};
                const std::size_t offset=(static_cast<std::size_t>(y)*width+x)*3;
                for (int c=0;c<3;++c)
                    image[offset+c]=static_cast<std::uint8_t>(std::lround(255.0*std::clamp(color[c],0.0,1.0)));
            }
        }
    });
    for (auto &thread:threads) thread.join();

    // The spheres have no arbitrary triangle mesh: their transport cells are
    // Fibonacci-Voronoi samples. Plot every front-facing cell centroid so the
    // diagnostic shows the representation the solver actually transports.
    for (const auto &sphere:scene.spheres) for (int index:sphere.patches) {
        const Patch &patch=scene.patches[index];
        if (dot(patch.normal,camera-patch.center)<=0.0) continue;
        const Vec3 rel=patch.center-camera; const double depth=dot(rel,forward);
        if (depth<=0.0) continue;
        const double nx=dot(rel,right)/(depth*aspect*scale);
        const double ny=dot(rel,up)/(depth*scale);
        const int cx=static_cast<int>(std::lround((nx+1.0)*0.5*width));
        const int cy=static_cast<int>(std::lround((1.0-ny)*0.5*height));
        for (int oy=-1;oy<=1;++oy) for (int ox=-1;ox<=1;++ox) {
            const int x=cx+ox,y=cy+oy;
            if (x<0 || x>=width || y<0 || y>=height) continue;
            const std::size_t offset=(static_cast<std::size_t>(y)*width+x)*3;
            image[offset]=232; image[offset+1]=238; image[offset+2]=246;
        }
    }
    return image;
}

void write_ppm(const std::string &path,const std::vector<std::uint8_t> &image,
               int width,int height) {
    std::ofstream out(path,std::ios::binary);
    if (!out) throw std::runtime_error("cannot open output image");
    out<<"P6\n"<<width<<" "<<height<<"\n255\n";
    out.write(reinterpret_cast<const char*>(image.data()),
              static_cast<std::streamsize>(image.size()));
}

double milliseconds(Clock::time_point a,Clock::time_point b) {
    return std::chrono::duration<double,std::milli>(b-a).count();
}

bool self_test() {
    std::mt19937_64 rng(9102026); std::uniform_real_distribution<double> u(-1,1);
    const Vec3 cp{0,0,0},cq{4,0.3,-0.2},np{1,0,0},nq{-1,0,0};
    const double hp=.24,hq=.31,ep=.08,eq=.11;
    const double reference=lambert_kernel(cp,np,cq,nq);
    const double bound=subdivision_bound(norm(cq-cp),hp,hq,ep,eq);
    for (int i=0;i<10000;++i) {
        Vec3 a=unit({u(rng),u(rng),u(rng)})*(hp*std::abs(u(rng)));
        Vec3 b=unit({u(rng),u(rng),u(rng)})*(hq*std::abs(u(rng)));
        Vec3 dn1{0,ep*u(rng),ep*u(rng)},dn2{0,eq*u(rng),eq*u(rng)};
        if (std::abs(lambert_kernel(cp+a,unit(np+dn1),cq+b,unit(nq+dn2))-reference)>bound)
            return false;
    }
    std::cout<<"native subdivision bound: ok\n"; return true;
}

} // namespace

int main(int argc,char **argv) try {
    int width=3200,height=2400; double error_fraction=0.002;
    bool geometry_only=false;
    std::string out="/tmp/photonic_standard_16x.ppm";
    for (int i=1;i<argc;++i) {
        const std::string arg=argv[i];
        if (arg=="--self-test") return self_test()?0:1;
        if (arg=="--geometry-only") { geometry_only=true; continue; }
        if (i+1>=argc) throw std::runtime_error("missing argument value");
        if (arg=="--width") width=std::stoi(argv[++i]);
        else if (arg=="--height") height=std::stoi(argv[++i]);
        else if (arg=="--error-fraction") error_fraction=std::stod(argv[++i]);
        else if (arg=="--out") out=argv[++i];
        else throw std::runtime_error("unknown argument: "+arg);
    }
    const auto total_start=Clock::now(); Timings timing; int sphere_samples=0;
    auto started=Clock::now(); Scene scene=build_scene(width,height,sphere_samples);
    auto finished=Clock::now(); timing.subdivision_ms=milliseconds(started,finished);
    if (geometry_only) {
        started=Clock::now(); auto image=render_geometry(scene,width,height);
        finished=Clock::now(); timing.raster_ms=milliseconds(started,finished);
        started=Clock::now(); write_ppm(out,image,width,height); finished=Clock::now();
        timing.write_ms=milliseconds(started,finished);
        timing.total_ms=milliseconds(total_start,finished);
        std::cout<<std::fixed<<std::setprecision(3)
            <<"{\n  \"geometry_only\": true,\n  \"width\": "<<width
            <<",\n  \"height\": "<<height
            <<",\n  \"surface_patches\": "<<scene.patches.size()
            <<",\n  \"sphere_samples_each\": "<<sphere_samples
            <<",\n  \"raster_ms\": "<<timing.raster_ms
            <<",\n  \"write_ms\": "<<timing.write_ms
            <<",\n  \"total_ms\": "<<timing.total_ms
            <<",\n  \"output\": \""<<out<<"\"\n}\n";
        return 0;
    }
    started=Clock::now(); compile_geometry(scene); finished=Clock::now();
    timing.geometry_ms=milliseconds(started,finished);
    started=Clock::now(); const auto transport=solve_transport(scene,error_fraction);
    finished=Clock::now(); timing.transport_ms=milliseconds(started,finished);
    std::vector<RGB> radiance(scene.patches.size());
    for (std::size_t i=0;i<radiance.size();++i)
        radiance[i]=transport.outgoing[i]*(1.0/(pi*scene.patches[i].area));
    started=Clock::now(); std::array<SphereLookup,3> lookups;
    std::array<std::thread,3> lookup_threads;
    for (int k=0;k<3;++k) lookup_threads[k]=std::thread([&,k]{
        lookups[k]=build_lookup(scene,scene.spheres[k],radiance);
    });
    for (auto &thread:lookup_threads) thread.join();
    finished=Clock::now(); timing.lookup_ms=milliseconds(started,finished);
    started=Clock::now(); auto image=render(scene,radiance,lookups,width,height);
    finished=Clock::now(); timing.raster_ms=milliseconds(started,finished);
    started=Clock::now(); write_ppm(out,image,width,height); finished=Clock::now();
    timing.write_ms=milliseconds(started,finished);
    timing.total_ms=milliseconds(total_start,finished);
    std::size_t edge_count=0; for (const auto &row:scene.adjacency) edge_count+=row.size();
    std::cout<<std::fixed<<std::setprecision(3)
        <<"{\n  \"width\": "<<width<<",\n  \"height\": "<<height
        <<",\n  \"pixels\": "<<static_cast<std::uint64_t>(width)*height
        <<",\n  \"surface_patches\": "<<scene.patches.size()
        <<",\n  \"sphere_samples_each\": "<<sphere_samples
        <<",\n  \"cached_directed_edges\": "<<edge_count
        <<",\n  \"active_edge_applications\": "<<transport.active_edges
        <<",\n  \"transport_depths\": "<<transport.depths
        <<",\n  \"transport_scale\": "<<scene.transport_scale
        <<",\n  \"requested_error\": "<<transport.requested_error
        <<",\n  \"certified_error\": "<<transport.certified_error
        <<",\n  \"subdivision_ms\": "<<timing.subdivision_ms
        <<",\n  \"geometry_ms\": "<<timing.geometry_ms
        <<",\n  \"transport_ms\": "<<timing.transport_ms
        <<",\n  \"lookup_ms\": "<<timing.lookup_ms
        <<",\n  \"raster_ms\": "<<timing.raster_ms
        <<",\n  \"write_ms\": "<<timing.write_ms
        <<",\n  \"total_ms\": "<<timing.total_ms
        <<",\n  \"output\": \""<<out<<"\"\n}\n";
    return 0;
} catch (const std::exception &error) {
    std::cerr<<"error: "<<error.what()<<"\n"; return 2;
}
